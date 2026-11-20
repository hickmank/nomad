from __future__ import annotations

import asyncio
import logging
import tempfile
import time
from collections.abc import Sequence
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

from fastmcp import Context, FastMCP

from .. import metrics as nomad_metrics
from ..logging_utils import (
    LoggerThresholdFilter,
    attach_handler,
    more_verbose_log_level,
    parse_log_level,
)
from ..otel import (
    configure_otel,
    get_tracer,
    set_span_error,
    set_span_ok,
    shutdown_otel,
)
from .config import (
    AllowlistMiddlewareConfig,
    GatewayConfig,
    GatewayRuntimeOptions,
    LoggingMiddlewareConfig,
    MiddlewareEntry,
    RedactionMiddlewareConfig,
)
from .middleware import (
    AllowlistMiddleware,
    LoggingMiddleware,
    Middleware,
    MiddlewareChain,
    RedactionMiddleware,
    TelemetryMiddleware,
)
from .sandbox import SandboxExecutor, SandboxResult
from .tool_index import ToolIndex
from .upstream import UpstreamProxy

logger = logging.getLogger(__name__)
tracer = get_tracer("nomad.gateway")

try:
    _PACKAGE_VERSION = version("nomad-scifm")
except PackageNotFoundError:  # pragma: no cover - local checkout
    _PACKAGE_VERSION = "0.0.0"


_USE_DEFAULT_TIMEOUT = object()


def _build_middleware(entries: Sequence[MiddlewareEntry]) -> MiddlewareChain:
    middleware: list[Middleware] = [TelemetryMiddleware()]
    for entry in entries:
        if isinstance(entry, AllowlistMiddlewareConfig):
            middleware.append(
                AllowlistMiddleware(
                    allow=entry.options.get("allow", []),
                    deny=entry.options.get("deny", []),
                )
            )
        elif isinstance(entry, LoggingMiddlewareConfig):
            middleware.append(
                LoggingMiddleware(level=entry.options.get("level", "info"))
            )
        elif isinstance(entry, RedactionMiddlewareConfig):
            middleware.append(
                RedactionMiddleware(mode=entry.options.get("mode", "basic"))
            )
        else:
            logger.warning("Unknown middleware '%s'", entry.kind)
    return MiddlewareChain(middleware)


class CodeModeGateway:
    """Expose a minimal async MCP proxy for sandboxed execution."""

    def __init__(self, config: GatewayConfig):
        self.config = config
        self._middleware = _build_middleware(config.middleware)
        self._upstream = UpstreamProxy(config.servers)
        self._tool_index = ToolIndex(
            self._upstream,
            wrappers_package=config.defaults.wrappers_package,
        )
        wrappers_root = Path(tempfile.mkdtemp(prefix="nomad_gateway_wrappers_"))
        self._base_options = GatewayRuntimeOptions.from_defaults_and_overrides(
            config.defaults,
            wrappers_root=wrappers_root,
        )
        self._sandbox = SandboxExecutor(
            upstream=self._upstream,
            tool_index=self._tool_index,
            base_middleware=self._middleware,
            options=self._base_options,
        )
        self._mcp = FastMCP(
            "MCP Gateway",
            version=_PACKAGE_VERSION,
            on_duplicate="error",
        )
        self._register_tools()

    async def __aenter__(self) -> CodeModeGateway:
        try:
            await self._upstream.start()
        except BaseException:
            try:
                await self._upstream.stop()
            except BaseException:
                logger.exception("Failed to clean up upstreams after startup failure")
            try:
                await self._sandbox.aclose()
            except BaseException:
                logger.exception("Failed to clean up sandbox after startup failure")
            raise
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        try:
            await self._upstream.stop()
        finally:
            await self._sandbox.aclose()

    def _register_tools(self) -> None:
        self._mcp.tool(self.search_code_tools)
        self._mcp.tool(self.execute_mcp_code)
        self._mcp.tool(self.execute_mcp_script)

    @property
    def fastmcp(self) -> FastMCP:
        return self._mcp

    async def serve(
        self,
        transport: str | None = None,
        **transport_kwargs: Any,
    ) -> None:
        async with self:
            await self._mcp.run_async(
                transport=transport,
                show_banner=False,
                **transport_kwargs,
            )

    async def search_code_tools(
        self,
        query: str | None,
        server_filter: str | None = None,
        detail_level: Literal["brief", "full"] = "brief",
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Discover MCP-backed helpers available inside the code-mode sandbox.

        :param query: Optional keyword search across tool names,
            descriptions, and argument metadata. Returns results
            for all tools, limited to ``limit``, if ``None``.
        :param server_filter: Restrict results to a specific upstream server ID.
        :param detail_level: ``"brief"`` (default) returns
            ``python_import_path``, ``description`` and ``signature``.
            ``"full"`` returns ``python_import_path`` plus the direct
            ``inputSchema``/``outputSchema`` payload and description.
        :param limit: Maximum number of matching tools to return.
            Pass ``None`` to return all matches.
        :returns: A list of dicts describing each matching tool.

        Notes:
            Typical flow:

            1. Call this tool to locate the wrapper import path and inspect the
               target helper.
            2. Import the generated wrapper in your snippet and call
               ``some_tool(...)`` or ``await some_tool_async(...)`` while
               running within ``execute_mcp_code`` or ``execute_mcp_script``.

               Example import:

               ::

                   from mcp_tools.some_server import some_tool, some_tool_async
        """
        level = "full" if detail_level == "full" else "brief"
        return await self._tool_index.search_code_payload(
            query=query,
            server_filter=server_filter,
            detail_level=level,
            limit=limit,
        )

    async def execute_mcp_script(
        self,
        script_path: Path,
        ctx: Context,
        env: dict[str, str] | None = None,
    ):
        """Like `execute_mcp_code` but takes in a path to a python script to evaluate

        See `execute_mcp_code` for more details. Keep ``RESULT`` under about
            16 KiB and write larger outputs to files instead.
        """
        start_time = time.perf_counter()
        nomad_metrics.record_gateway_request("execute_mcp_script")
        with tracer.start_as_current_span(
            "nomad.gateway.execute_mcp_script",
            attributes={
                "nomad.gateway.request_id": str(ctx.request_id),
                "nomad.gateway.script_path": str(script_path),
            },
        ) as span:
            try:
                await ctx.info(
                    "execute_mcp_script started",
                    extra={"request_id": ctx.request_id, "phase": "start"},
                )
                result = await self.run_script(script_path, env)
                span.set_attribute(
                    "nomad.gateway.tool_call_count", len(result.tool_calls)
                )
                set_span_ok(span)
                await ctx.info(
                    "execute_mcp_script completed",
                    extra={
                        "request_id": ctx.request_id,
                        "phase": "end",
                    },
                )
            except TimeoutError as exc:
                set_span_error(span, exc)
                nomad_metrics.record_gateway_request_duration(
                    "execute_mcp_script",
                    time.perf_counter() - start_time,
                    status="timeout",
                )
                return self._timeout_payload(
                    exc,
                    duration_seconds=time.perf_counter() - start_time,
                )
            except Exception as exc:
                set_span_error(span, exc)
                nomad_metrics.record_gateway_request_duration(
                    "execute_mcp_script",
                    time.perf_counter() - start_time,
                    status="error",
                )
                raise

        nomad_metrics.record_gateway_request_duration(
            "execute_mcp_script",
            time.perf_counter() - start_time,
            status="ok",
        )
        return self._format_gateway_response(result)

    async def execute_mcp_code(
        self,
        code: str,
        ctx: Context,
        env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Run Python inside the Nomad sandbox and optionally call MCP tools.

        :param code: Python source to execute. Printed output is captured in
            ``stdout`` and ``stderr``. To return a value, assign it to
            ``RESULT`` (for example, ``RESULT = {"answer": 42}``). If
            ``RESULT`` is assigned multiple times, only the final value is
            kept. With no assignment, ``result`` is ``None``. Keep ``RESULT``
            under about 16 KiB and write larger outputs to files.
        :param env: Extra environment variables for the run.
        :returns: A payload that includes ``stdout``, ``stderr``, the value
            assigned to ``RESULT`` (if any), ``duration_seconds``, and error
            details when present.

        Notes:
            - Tool wrappers are auto-generated under ``mcp_tools.<server>`` and
              expose both synchronous and asynchronous helpers::

                  from mcp_tools.some_server import some_tool, some_tool_async

                  RESULT = some_tool(param="value")  # synchronous call

                  import asyncio

                  async def main():
                      return await some_tool_async(param="value")

                  RESULT = asyncio.run(main())  # asynchronous call

            - These helpers normalize the MCP payload into plain Python objects.
            - Prefer using the async helpers over synchronous helpers.
            - Sandbox rules for filesystem/network access and approval policy
              still apply to any code or tool use inside the sandbox
        """

        start_time = time.perf_counter()
        nomad_metrics.record_gateway_request("execute_mcp_code")
        with tracer.start_as_current_span(
            "nomad.gateway.execute_mcp_code",
            attributes={
                "nomad.gateway.request_id": str(ctx.request_id),
                "nomad.gateway.code_size": len(code),
            },
        ) as span:
            try:
                await ctx.info(
                    "execute_mcp_code started",
                    extra={"request_id": ctx.request_id, "phase": "start"},
                )
                result = await self.run_code(code, env)
                span.set_attribute(
                    "nomad.gateway.tool_call_count", len(result.tool_calls)
                )
                set_span_ok(span)
                await ctx.info(
                    "execute_mcp_code completed",
                    extra={
                        "request_id": ctx.request_id,
                        "phase": "end",
                    },
                )
            except TimeoutError as exc:
                set_span_error(span, exc)
                nomad_metrics.record_gateway_request_duration(
                    "execute_mcp_code",
                    time.perf_counter() - start_time,
                    status="timeout",
                )
                return self._timeout_payload(
                    exc,
                    duration_seconds=time.perf_counter() - start_time,
                )
            except Exception as exc:
                set_span_error(span, exc)
                nomad_metrics.record_gateway_request_duration(
                    "execute_mcp_code",
                    time.perf_counter() - start_time,
                    status="error",
                )
                raise
        nomad_metrics.record_gateway_request_duration(
            "execute_mcp_code",
            time.perf_counter() - start_time,
            status="ok",
        )
        return self._format_gateway_response(result)

    async def run_script(
        self,
        script_path: Path,
        env: dict[str, str] | None = None,
        *,
        script_args: Sequence[str] = (),
        timeout_seconds: float | None | object = _USE_DEFAULT_TIMEOUT,
        capture_stdio: bool = True,
    ) -> SandboxResult:
        script_path = Path(script_path)
        overrides: dict[str, Any] | None = None
        if timeout_seconds is not _USE_DEFAULT_TIMEOUT:
            overrides = {"timeout_seconds": timeout_seconds}
        options = GatewayRuntimeOptions.from_defaults_and_overrides(
            self.config.defaults,
            overrides=overrides,
            wrappers_root=self._sandbox.wrappers_root,
        )
        return await self._sandbox.run_script(
            script_path=script_path,
            options=options,
            env_overrides=env,
            script_args=script_args,
            capture_stdio=capture_stdio,
        )

    async def run_code(
        self,
        code: str,
        env: dict[str, str] | None = None,
        *,
        timeout_seconds: float | None | object = _USE_DEFAULT_TIMEOUT,
        capture_stdio: bool = True,
    ) -> SandboxResult:
        overrides: dict[str, Any] | None = None
        if timeout_seconds is not _USE_DEFAULT_TIMEOUT:
            overrides = {"timeout_seconds": timeout_seconds}
        options = GatewayRuntimeOptions.from_defaults_and_overrides(
            self.config.defaults,
            overrides=overrides,
            wrappers_root=self._sandbox.wrappers_root,
        )
        return await self._sandbox.run_code(
            code=code,
            options=options,
            env_overrides=env,
            capture_stdio=capture_stdio,
        )

    @staticmethod
    def _format_gateway_response(result: SandboxResult) -> dict[str, Any]:
        payload = asdict(result)
        payload.pop("tool_calls", None)
        return payload

    @staticmethod
    def _timeout_payload(
        exc: TimeoutError,
        *,
        duration_seconds: float,
    ) -> dict[str, Any]:
        message = str(exc) or "Sandbox execution exceeded timeout"
        return {
            "stdout": "",
            "stderr": message,
            "result": {"error": "timeout"},
            "tool_calls": [],
            "duration_seconds": duration_seconds,
        }


def run_gateway(
    config: GatewayConfig,
    transport: str | None = None,
    *,
    log_level: str = "INFO",
    log_file_handler: logging.Handler | None = None,
    **kwargs: Any,
) -> None:
    """Run a code-mode gateway from an already-loaded config."""
    effective_transport = transport or "stdio"
    if (
        effective_transport == "stdio"
        and "workspace_root" not in config.defaults.model_fields_set
    ):
        config.defaults.workspace_root = Path.cwd().resolve()

    configure_otel(
        service_name=config.telemetry.service_name,
        service_version=_PACKAGE_VERSION,
        enabled=config.telemetry.enabled,
        otlp_endpoint=config.telemetry.otlp_endpoint,
    )
    gateway = CodeModeGateway(config)
    fastmcp_logger = logging.getLogger("fastmcp")
    added_filters: list[tuple[logging.Handler, logging.Filter]] = []
    if log_file_handler is not None:
        numeric_level = parse_log_level(log_level)
        fastmcp_logger.setLevel(more_verbose_log_level(numeric_level))
        for handler in fastmcp_logger.handlers:
            filter_ = LoggerThresholdFilter(
                default_level=numeric_level,
                overrides={
                    "fastmcp.server.context.to_client": logging.WARNING,
                },
            )
            handler.addFilter(filter_)
            added_filters.append((handler, filter_))
        attach_handler(fastmcp_logger, log_file_handler)
    try:
        asyncio.run(gateway.serve(transport=transport, **kwargs))
    finally:
        for handler, filter_ in added_filters:
            handler.removeFilter(filter_)
        if log_file_handler is not None and log_file_handler in fastmcp_logger.handlers:
            fastmcp_logger.removeHandler(log_file_handler)
        shutdown_otel()
