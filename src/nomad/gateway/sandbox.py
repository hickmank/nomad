from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import tempfile
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any

from .. import metrics as nomad_metrics
from ..jsonrpc import Dispatcher, encode_frame, encode_response, read_frame
from ..otel import get_tracer, set_span_error, set_span_ok
from .config import GatewayRuntimeOptions
from .middleware import (
    MiddlewareChain,
    ToolCallContext,
)
from .runtime.runner import _default_serializer
from .tool_index import ToolIndex
from .upstream import UpstreamProxy
from .wrappers import WrapperGenerator

logger = logging.getLogger(__name__)
tracer = get_tracer("nomad.gateway")


def _jsonable(value: Any) -> Any:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            value = model_dump(mode="python")
        except TypeError:
            value = model_dump()
    try:
        return json.loads(json.dumps(value, default=str))
    except TypeError:
        return str(value)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _bound_result(result: Any, limit: int) -> Any:
    if result is None:
        return None
    try:
        serialized = json.dumps(result, default=str)
    except TypeError:
        result = str(result)
        serialized = result
    if len(serialized) <= limit:
        return result
    return {
        "error": (
            "RESULT is too large to return. "
            f"Keep RESULT under about {_format_recommended_result_size(limit)} "
            "and write larger outputs to files instead."
        )
    }


def _format_recommended_result_size(limit: int) -> str:
    recommended = max(1, round(limit / 4))
    for unit, scale in (
        ("GiB", 1024**3),
        ("MiB", 1024**2),
        ("KiB", 1024),
    ):
        if recommended >= scale:
            value = round(recommended / scale)
            return f"{max(1, value)} {unit}"
    return f"{recommended} B"


@dataclass(slots=True)
class SandboxResult:
    """Result captured from a sandboxed Python execution."""

    stdout: str
    stderr: str
    result: Any
    returncode: int | None
    duration_seconds: float
    tool_calls: list[dict[str, Any]]


def _encode_bridge_frame(payload: Mapping[str, Any]) -> bytes:
    return encode_frame(payload, default=_default_serializer)


def _log_bridge_error(exc: Exception) -> None:
    logger.error("Bridge request failed", exc_info=exc)


class SandboxBridge:
    """JSON-RPC bridge between sandbox runtime and gateway."""

    def __init__(
        self,
        run_id: str,
        upstream: UpstreamProxy,
        tool_index: ToolIndex,
        wrappers: WrapperGenerator,
        base_middleware: MiddlewareChain,
    ):
        self._run_id = run_id
        self._upstream = upstream
        self._tool_index = tool_index
        self._wrappers = wrappers
        self._base_middleware = base_middleware

        self._server: asyncio.AbstractServer | None = None
        self._host = "127.0.0.1"
        self._port = 0
        self._started_at_ns = time.perf_counter_ns()
        self.tool_calls: list[dict[str, Any]] = []

    async def __aenter__(self) -> SandboxBridge:
        self._server = await asyncio.start_server(self._handle_client, self._host, 0)
        sock = self._server.sockets[0]
        sockname = sock.getsockname()
        if isinstance(sockname, tuple):
            self._host = sockname[0]
            self._port = sockname[1]
        else:  # pragma: no cover
            self._host = "127.0.0.1"
            self._port = int(sockname)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    @property
    def address(self) -> tuple[str, int]:
        return self._host, self._port

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ):
        dispatcher = Dispatcher(
            {
                "call_tool": self._handle_call_tool,
                "ensure_wrapper": self._handle_ensure_wrapper,
            },
            on_error=_log_bridge_error,
        )
        try:
            while True:
                message = await read_frame(reader)
                if message is None:
                    break
                response = await dispatcher.dispatch_async(message)
                if response is not None:
                    writer.write(encode_response(response, _encode_bridge_frame))
                    await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    async def _handle_call_tool(self, params: Mapping[str, Any]) -> Any:
        server = params["server"]
        tool = params["tool"]
        arguments = params.get("arguments", {})
        tool_call_id = f"{self._run_id}:{len(self.tool_calls) + 1}"
        context = ToolCallContext(
            server=server,
            tool=tool,
            arguments=arguments,
            run_id=self._run_id,
            metadata={},
        )
        chain = self._base_middleware
        start_ns = time.perf_counter_ns()
        input_payload = _jsonable(arguments)
        try:
            async with chain.around_tool(context) as ctx:
                raw = await self._upstream.call_tool(server, tool, arguments)
                result = await chain.after_tool(ctx, raw)
            end_ns = time.perf_counter_ns()
            output_payload = _jsonable(result)
            self.tool_calls.append(
                {
                    "tool_call_id": tool_call_id,
                    "server": server,
                    "tool": tool,
                    "start_offset_ns": start_ns - self._started_at_ns,
                    "end_offset_ns": end_ns - self._started_at_ns,
                    "duration_ms": (end_ns - start_ns) / 1e6,
                    "input": input_payload,
                    "output": output_payload,
                    "status": "ok",
                }
            )
            return output_payload
        except Exception as exc:  # noqa: BLE001
            end_ns = time.perf_counter_ns()
            self.tool_calls.append(
                {
                    "tool_call_id": tool_call_id,
                    "server": server,
                    "tool": tool,
                    "start_offset_ns": start_ns - self._started_at_ns,
                    "end_offset_ns": end_ns - self._started_at_ns,
                    "duration_ms": (end_ns - start_ns) / 1e6,
                    "input": input_payload,
                    "output": {
                        "status": "error",
                        "error_type": exc.__class__.__name__,
                        "message": str(exc),
                    },
                    "status": "error",
                    "error": str(exc),
                }
            )
            raise

    async def _handle_ensure_wrapper(self, params: Mapping[str, Any]) -> Any:
        server = params["server"]
        tools = await self._tool_index.list_server_tools(server)
        return self._wrappers.build_module_spec(server, tools)


class SandboxExecutor:
    """Run sandboxed python executions."""

    def __init__(
        self,
        upstream: UpstreamProxy,
        tool_index: ToolIndex,
        base_middleware: MiddlewareChain,
        options: GatewayRuntimeOptions,
    ):
        self._upstream = upstream
        self._tool_index = tool_index
        self._wrappers_root = options.wrappers_root
        self._closed = False
        try:
            self._wrappers = WrapperGenerator(
                self._wrappers_root,
                options.wrappers_package,
            )
        except Exception:
            shutil.rmtree(self._wrappers_root, ignore_errors=True)
            raise
        self._base_middleware = base_middleware

    @property
    def wrappers_root(self) -> Path:
        return self._wrappers_root

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        await asyncio.to_thread(shutil.rmtree, self._wrappers_root, True)

    @staticmethod
    def _schedule_cleanup(path: Path) -> None:
        async def _cleanup() -> None:
            await asyncio.to_thread(shutil.rmtree, path, ignore_errors=True)

        asyncio.get_running_loop().create_task(_cleanup())

    def _create_workspace(
        self, options: GatewayRuntimeOptions
    ) -> tuple[str, Path, Path | None]:
        run_id = uuid.uuid4().hex
        if options.workspace_root is not None:
            options.workspace_root.mkdir(parents=True, exist_ok=True)
            return run_id, options.workspace_root, None
        workspace = Path(tempfile.mkdtemp(prefix="nomad_gateway_workspace_"))
        return run_id, workspace, workspace

    async def _run_script_path(
        self,
        script_path: Path,
        options: GatewayRuntimeOptions,
        env_overrides: Mapping[str, str] | None = None,
        *,
        script_args: list[str] = (),
        capture_stdio: bool = True,
    ) -> SandboxResult:
        run_id, workspace, cleanup_path = self._create_workspace(options)
        script_path = Path(script_path)
        run_dir = Path(tempfile.mkdtemp(prefix="nomad_run_"))
        result_path = run_dir / "result.json"
        result_path.unlink(missing_ok=True)
        start_time = time.perf_counter()

        with tracer.start_as_current_span(
            "nomad.gateway.sandbox.run",
            attributes={
                "nomad.gateway.run_id": run_id,
                "nomad.gateway.script_path": str(script_path),
                "nomad.gateway.capture_stdio": capture_stdio,
                "nomad.gateway.timeout_seconds": options.timeout_seconds,
            },
        ) as span:
            async with SandboxBridge(
                run_id=run_id,
                upstream=self._upstream,
                tool_index=self._tool_index,
                wrappers=self._wrappers,
                base_middleware=self._base_middleware,
            ) as bridge:
                host, port = bridge.address
                env = os.environ.copy()
                env.update(
                    {
                        "NOMAD_MCP_GATEWAY_BRIDGE": f"{host}:{port}",
                        "NOMAD_MCP_SCRIPT_PATH": str(script_path),
                        "NOMAD_MCP_RESULT_PATH": str(result_path),
                        "NOMAD_MCP_WRAPPERS_ROOT": str(options.wrappers_root),
                        "NOMAD_MCP_WRAPPERS_PACKAGE": options.wrappers_package,
                        "PYTHONUNBUFFERED": "1",
                    }
                )
                if env_overrides:
                    env.update(env_overrides)
                env["PYTHONPATH"] = str(options.wrappers_root)
                runner_args = tuple(script_args)

                runner_ref = files(f"{__package__}.runtime").joinpath("runner.py")
                with as_file(runner_ref) as runner_path:
                    if capture_stdio:
                        process = await asyncio.create_subprocess_exec(
                            str(options.python_executable),
                            str(runner_path),
                            *runner_args,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                            cwd=str(workspace),
                            env=env,
                        )

                        try:
                            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                                process.communicate(),
                                timeout=options.timeout_seconds,
                            )
                        except TimeoutError:
                            process.kill()
                            await process.wait()
                            nomad_metrics.record_gateway_sandbox_duration(
                                time.perf_counter() - start_time,
                                status="timeout",
                                returncode=process.returncode,
                            )
                            raise TimeoutError("Sandbox execution exceeded timeout")

                        stdout = _truncate(
                            stdout_bytes.decode("utf-8", errors="replace"),
                            options.stdout_limit,
                        )
                        stderr = _truncate(
                            stderr_bytes.decode("utf-8", errors="replace"),
                            options.stderr_limit,
                        )
                    else:
                        process = await asyncio.create_subprocess_exec(
                            str(options.python_executable),
                            str(runner_path),
                            *runner_args,
                            cwd=str(workspace),
                            env=env,
                        )

                        try:
                            await asyncio.wait_for(
                                process.wait(),
                                timeout=options.timeout_seconds,
                            )
                        except TimeoutError:
                            process.kill()
                            await process.wait()
                            nomad_metrics.record_gateway_sandbox_duration(
                                time.perf_counter() - start_time,
                                status="timeout",
                                returncode=process.returncode,
                            )
                            raise TimeoutError("Sandbox execution exceeded timeout")
                        stdout = ""
                        stderr = ""
            try:
                result_payload: Any = None
                if result_path.exists():
                    with result_path.open("r", encoding="utf-8") as fh:
                        try:
                            payload = json.load(fh)
                            if isinstance(payload, dict):
                                result_payload = payload.get("result")
                        except json.JSONDecodeError:
                            result_payload = None

                if result_payload is None and process.returncode != 0:
                    result_payload = {
                        "error": (
                            "Sandbox execution returned non-zero exit code: "
                            f"{process.returncode}"
                        )
                    }

                normalized_result = _bound_result(result_payload, options.result_limit)
                duration_seconds = time.perf_counter() - start_time
                sandbox_status = "error" if process.returncode else "ok"
                span.set_attribute("nomad.gateway.returncode", process.returncode)
                span.set_attribute("nomad.gateway.duration_seconds", duration_seconds)
                span.set_attribute(
                    "nomad.gateway.tool_call_count", len(bridge.tool_calls)
                )
                nomad_metrics.record_gateway_sandbox_duration(
                    duration_seconds,
                    status=sandbox_status,
                    returncode=process.returncode,
                )
                if process.returncode:
                    set_span_error(span, f"Sandbox returned {process.returncode}")
                else:
                    set_span_ok(span)

                return SandboxResult(
                    stdout=stdout,
                    stderr=stderr,
                    result=normalized_result,
                    returncode=process.returncode,
                    duration_seconds=duration_seconds,
                    tool_calls=bridge.tool_calls,
                )
            finally:
                self._schedule_cleanup(run_dir)
                if cleanup_path is not None:
                    self._schedule_cleanup(cleanup_path)

    async def run_script(
        self,
        script_path: Path,
        options: GatewayRuntimeOptions,
        env_overrides: Mapping[str, str] | None = None,
        *,
        script_args: list[str] = (),
        capture_stdio: bool = True,
    ) -> SandboxResult:
        script_path = Path(script_path).expanduser()
        if not script_path.is_absolute():
            script_path = (options.workspace_root or Path.cwd()) / script_path

        script_path = script_path.absolute()
        if not script_path.is_file():
            raise FileNotFoundError(f"Script file not found: {script_path}")

        return await self._run_script_path(
            script_path=script_path,
            options=options,
            env_overrides=env_overrides,
            script_args=script_args,
            capture_stdio=capture_stdio,
        )

    async def run_code(
        self,
        code: str,
        options: GatewayRuntimeOptions,
        env_overrides: Mapping[str, str] | None = None,
        *,
        capture_stdio: bool = True,
    ) -> SandboxResult:
        run_dir = Path(tempfile.mkdtemp(prefix="nomad_code_"))
        script_path = run_dir / "code.py"
        script_path.write_text(code, encoding="utf-8")
        try:
            return await self._run_script_path(
                script_path=script_path,
                options=options,
                env_overrides=env_overrides,
                capture_stdio=capture_stdio,
            )
        finally:
            self._schedule_cleanup(run_dir)
