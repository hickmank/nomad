from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import Any

from .model_env import ModelEnvironment, PythonEnvironment

logger = logging.getLogger(__name__)
_CLOSE_LOCK_GRACE_SECONDS = 0.25
_PROCESS_EXIT_GRACE_SECONDS = 2.0


class ModelRPCError(RuntimeError):
    """Raised when an isolated model worker returns a JSON-RPC error."""


@dataclass(frozen=True, slots=True)
class RemoteModel:
    """Host-side metadata and loading instructions for an isolated model."""

    environment: PythonEnvironment
    model_class: str
    source: str
    name: str
    description: str
    batch_size: int
    args_schema: dict[str, Any]
    output_schema: dict[str, Any]

    def load_params(self, device: str) -> dict[str, Any]:
        return {
            "model_class": self.model_class,
            "source": self.source,
            "tool_name": self.name,
            "batch_size": self.batch_size,
            "device": device,
        }


class ModelProcess:
    """A synchronous JSON-RPC client for one cached model environment."""

    def __init__(
        self,
        environment: PythonEnvironment,
        *,
        timeout_seconds: float | None = 30.0,
    ) -> None:
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than 0 or None")
        self.environment = environment
        self.timeout_seconds = timeout_seconds
        python = environment.ensure()
        child_env = dict(os.environ)
        if isinstance(environment, ModelEnvironment):
            child_env.pop("PYTHONPATH", None)
            child_env.pop("PYTHONHOME", None)
            venv = python.parent.parent
            child_env["VIRTUAL_ENV"] = str(venv)
            child_env["PATH"] = os.pathsep.join(
                part for part in (str(python.parent), child_env.get("PATH")) if part
            )
        self._process = subprocess.Popen(
            [str(python), "-m", "nomad.model_worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env=child_env,
            cwd=environment.base_dir
            if isinstance(environment, ModelEnvironment)
            else None,
            start_new_session=os.name != "nt",
        )
        self._request_id = 0
        self._lock = threading.Lock()
        self._responses: Queue[str | None] = Queue()
        self._stderr: deque[str] = deque(maxlen=100)
        self._stdout_thread = threading.Thread(
            target=self._drain_stdout,
            daemon=True,
            name=f"nomad-model-stdout-{self._process.pid}",
        )
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            daemon=True,
            name=f"nomad-model-stderr-{self._process.pid}",
        )
        self._stdout_thread.start()
        self._stderr_thread.start()

    @property
    def pid(self) -> int:
        return self._process.pid

    @property
    def alive(self) -> bool:
        return self._process.poll() is None

    def request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        timeout_seconds = self.timeout_seconds
        deadline = (
            None if timeout_seconds is None else time.monotonic() + timeout_seconds
        )
        if timeout_seconds is None:
            acquired = self._lock.acquire()
        else:
            acquired = self._lock.acquire(timeout=timeout_seconds)
        if not acquired:
            raise self._timeout_error(method)

        try:
            if not self.alive:
                raise ModelRPCError(self._exited_message())
            self._request_id += 1
            request_id = self._request_id
            payload = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params or {},
            }
            self._write_request(payload, method=method, deadline=deadline)

            response_timeout = (
                None if deadline is None else max(0.0, deadline - time.monotonic())
            )
            try:
                response_line = self._responses.get(timeout=response_timeout)
            except Empty as exc:
                assert timeout_seconds is not None
                self._terminate_process()
                raise self._timeout_error(method) from exc
            if response_line is None:
                raise ModelRPCError(self._exited_message())
            try:
                response = json.loads(response_line)
            except json.JSONDecodeError as exc:
                self._terminate_process()
                raise ModelRPCError(
                    f"Model worker returned invalid JSON: {response_line.rstrip()}"
                ) from exc
            if not isinstance(response, dict):
                self._terminate_process()
                raise ModelRPCError("Model worker returned a non-object response")
            if response.get("jsonrpc") != "2.0" or response.get("id") != request_id:
                self._terminate_process()
                raise ModelRPCError(
                    "Model worker returned a mismatched JSON-RPC response"
                )
            if error := response.get("error"):
                message = error.get("message", "model worker error")
                raise ModelRPCError(str(message))
            return response.get("result")
        finally:
            self._lock.release()

    def _write_request(
        self,
        payload: dict[str, Any],
        *,
        method: str,
        deadline: float | None,
    ) -> None:
        """Write one request without allowing a blocked pipe to evade the deadline."""
        assert self._process.stdin is not None
        serialized = json.dumps(payload) + "\n"
        errors: list[BaseException] = []
        completed = threading.Event()

        def write() -> None:
            try:
                assert self._process.stdin is not None
                self._process.stdin.write(serialized)
                self._process.stdin.flush()
            except BaseException as exc:  # transported back to the request thread
                errors.append(exc)
            finally:
                completed.set()

        if deadline is None:
            write()
        else:
            writer = threading.Thread(
                target=write,
                daemon=True,
                name=f"nomad-model-stdin-{self._process.pid}",
            )
            writer.start()
            if not completed.wait(max(0.0, deadline - time.monotonic())):
                self._terminate_process()
                raise self._timeout_error(method)

        if errors:
            raise ModelRPCError(self._exited_message()) from errors[0]

    def _timeout_error(self, method: str) -> ModelRPCError:
        assert self.timeout_seconds is not None
        return ModelRPCError(
            f"Model worker RPC '{method}' timed out after "
            f"{self.timeout_seconds:g} seconds"
        )

    def load(self, remote: RemoteModel, device: str) -> dict[str, Any]:
        result = self.request("load", remote.load_params(device))
        if not isinstance(result, dict):
            raise ModelRPCError("Model worker returned invalid model metadata")
        return result

    def run_batch(
        self,
        inputs: list[dict[str, Any]],
        *,
        batch_size: int,
    ) -> list[Any]:
        result = self.request(
            "run_batch",
            {"inputs": inputs, "batch_size": batch_size},
        )
        if not isinstance(result, list):
            raise ModelRPCError("Model worker returned a non-list batch result")
        return result

    def close(self) -> None:
        acquired = self._lock.acquire(timeout=_CLOSE_LOCK_GRACE_SECONDS)
        if not acquired:
            # A request may be blocked in readline(). Terminating the process group
            # closes the protocol pipe and allows that request to release the lock.
            self._terminate_process()
            acquired = self._lock.acquire(timeout=_PROCESS_EXIT_GRACE_SECONDS)
        if not acquired:
            logger.warning("Model worker %s did not release its RPC lock", self.pid)
            return

        try:
            if self.alive:
                try:
                    assert self._process.stdin is not None
                    self._request_id += 1
                    self._process.stdin.write(
                        json.dumps(
                            {
                                "jsonrpc": "2.0",
                                "id": self._request_id,
                                "method": "shutdown",
                                "params": {},
                            }
                        )
                        + "\n"
                    )
                    self._process.stdin.flush()
                    self._process.wait(timeout=_PROCESS_EXIT_GRACE_SECONDS)
                except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                    self._terminate_process()
            if os.name != "nt":
                self._terminate_remaining_process_group()

            for stream in (
                self._process.stdin,
                self._process.stdout,
                self._process.stderr,
            ):
                if stream is not None:
                    stream.close()
        finally:
            self._lock.release()

    def _terminate_process(self) -> None:
        if os.name == "nt":
            if not self.alive:
                return
            try:
                self._process.terminate()
                self._process.wait(timeout=_PROCESS_EXIT_GRACE_SECONDS)
            except (OSError, subprocess.TimeoutExpired):
                if self.alive:
                    self._process.kill()
                    self._process.wait()
            return

        if not self.alive and not self._process_group_exists():
            return
        try:
            os.killpg(self._process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        if self.alive:
            try:
                self._process.wait(timeout=_PROCESS_EXIT_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                pass
        self._terminate_remaining_process_group()
        if self.alive:
            self._process.wait()

    def _terminate_remaining_process_group(self) -> None:
        if os.name == "nt" or not self._process_group_exists():
            return
        try:
            os.killpg(self._process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + _PROCESS_EXIT_GRACE_SECONDS
        while self._process_group_exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        if self._process_group_exists():
            try:
                os.killpg(self._process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def _process_group_exists(self) -> bool:
        try:
            os.killpg(self._process.pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:  # pragma: no cover - different-user descendants
            return True
        return True

    def _drain_stderr(self) -> None:
        assert self._process.stderr is not None
        for line in self._process.stderr:
            line = line.rstrip()
            self._stderr.append(line)
            logger.debug("model worker %s: %s", self.pid, line)

    def _drain_stdout(self) -> None:
        assert self._process.stdout is not None
        try:
            for line in self._process.stdout:
                self._responses.put(line)
        finally:
            self._responses.put(None)

    def _exited_message(self) -> str:
        message = f"Model worker exited with status {self._process.poll()}"
        if self._stderr:
            message += f": {self._stderr[-1]}"
        return message

    def __enter__(self) -> ModelProcess:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def inspect_remote_model(
    *,
    environment: PythonEnvironment,
    model_class: str,
    source: str | Path,
    tool_name: str | None,
    batch_size: int | None,
    timeout_seconds: float | None = 30.0,
) -> RemoteModel:
    """Load an isolated model once to retrieve its public tool metadata."""
    with ModelProcess(environment, timeout_seconds=timeout_seconds) as process:
        result = process.request(
            "load",
            {
                "model_class": model_class,
                "source": str(source),
                "tool_name": tool_name,
                "batch_size": batch_size,
                "device": "cpu",
            },
        )
    if not isinstance(result, dict):
        raise ModelRPCError("Model worker returned invalid model metadata")
    return RemoteModel(
        environment=environment,
        model_class=model_class,
        source=str(source),
        name=str(result["name"]),
        description=str(result["description"]),
        batch_size=max(1, int(result["batch_size"])),
        args_schema=dict(result["args_schema"]),
        output_schema=dict(result["output_schema"]),
    )


__all__ = [
    "ModelProcess",
    "ModelRPCError",
    "RemoteModel",
    "inspect_remote_model",
]
