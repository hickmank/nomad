from __future__ import annotations

import logging
import os
import signal
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .jsonrpc import JSONRPCClient, JSONRPCError, ProtocolError
from .model_env import ExistingEnvironment, ModelEnvironment, PythonEnvironment

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
        python = environment.ensure()
        child_env = dict(os.environ)
        isolated = isinstance(environment, ModelEnvironment | ExistingEnvironment)
        if isolated:
            child_env.pop("PYTHONPATH", None)
            child_env.pop("PYTHONHOME", None)
            child_env.pop("VIRTUAL_ENV", None)
            if isinstance(environment, ModelEnvironment) or environment.is_venv:
                child_env["VIRTUAL_ENV"] = str(python.parent.parent)
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
            cwd=environment.base_dir if isolated else None,
            start_new_session=os.name != "nt",
        )
        assert self._process.stdin is not None
        assert self._process.stdout is not None
        # A worker that timed out or sent garbage may be stuck or out of sync,
        # so the client terminates it before reporting the failure.
        self._rpc = JSONRPCClient(
            self._process.stdin,
            self._process.stdout,
            timeout_seconds=timeout_seconds,
            on_failure=self._terminate_process,
            name=f"nomad-model-{self._process.pid}",
        )
        self._stderr: deque[str] = deque(maxlen=100)
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            daemon=True,
            name=f"nomad-model-stderr-{self._process.pid}",
        )
        self._stderr_thread.start()

    @property
    def pid(self) -> int:
        return self._process.pid

    @property
    def alive(self) -> bool:
        return self._process.poll() is None

    @property
    def timeout_seconds(self) -> float | None:
        return self._rpc.timeout_seconds

    @timeout_seconds.setter
    def timeout_seconds(self, value: float | None) -> None:
        self._rpc.timeout_seconds = value

    def request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        """Call ``method`` on the worker, translating failures to ``ModelRPCError``."""
        if not self.alive:
            raise ModelRPCError(self._exited_message())
        try:
            return self._rpc.request(method, params)
        except TimeoutError as exc:
            raise ModelRPCError(
                f"Model worker RPC '{method}' timed out after "
                f"{self.timeout_seconds:g} seconds"
            ) from exc
        except ConnectionError as exc:
            raise ModelRPCError(self._exited_message()) from exc
        except ProtocolError as exc:
            raise ModelRPCError(
                f"Model worker sent an invalid JSON-RPC response: {exc}"
            ) from exc
        except JSONRPCError as exc:
            raise ModelRPCError(exc.message) from exc

    def load(self, remote: RemoteModel, device: str) -> dict[str, Any]:
        result = self.request("load", remote.load_params(device))
        if not isinstance(result, dict):
            raise ModelRPCError("Model worker returned invalid model metadata")
        return result

    def run_batch(
        self,
        remote: RemoteModel,
        device: str,
        inputs: list[dict[str, Any]],
        *,
        batch_size: int,
    ) -> list[Any]:
        # Sending the load parameters lets the worker (re)load the model itself
        # if it was unloaded or offloaded since the last request.
        result = self.request(
            "run_batch",
            {
                "model": remote.load_params(device),
                "inputs": inputs,
                "batch_size": batch_size,
            },
        )
        if not isinstance(result, list):
            raise ModelRPCError("Model worker returned a non-list batch result")
        return result

    def close(self) -> None:
        acquired = self._rpc.acquire(timeout=_CLOSE_LOCK_GRACE_SECONDS)
        if not acquired:
            # A request may be waiting for a reply. Terminating the process group
            # closes the protocol pipe and allows that request to release the lock.
            self._terminate_process()
            acquired = self._rpc.acquire(timeout=_PROCESS_EXIT_GRACE_SECONDS)
        if not acquired:
            logger.warning("Model worker %s did not release its RPC lock", self.pid)
            return

        try:
            if self.alive:
                try:
                    # Ask the worker to exit; its reply is not needed.
                    self._rpc.send("shutdown")
                    self._process.wait(timeout=_PROCESS_EXIT_GRACE_SECONDS)
                except (ConnectionError, subprocess.TimeoutExpired):
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
            self._rpc.release()

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
