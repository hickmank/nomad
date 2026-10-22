from __future__ import annotations

import asyncio
import importlib
import importlib.abc
import importlib.util
import os
import sys
import threading
from typing import Any

from nomad.jsonrpc import (
    JSONRPCError,
    ProtocolError,
    decode,
    encode_frame,
    make_request,
    parse_response,
    read_frame,
)

from .constants import DEFAULT_WRAPPERS_PACKAGE
from .runner import _default_serializer
from .wrapper_factory import build_server_module

BRIDGE_ENV = "NOMAD_MCP_GATEWAY_BRIDGE"
WRAPPERS_PACKAGE_ENV = "NOMAD_MCP_WRAPPERS_PACKAGE"
WRAPPERS_PACKAGE = os.environ.get(WRAPPERS_PACKAGE_ENV, DEFAULT_WRAPPERS_PACKAGE)


class BridgeError(RuntimeError):
    """Raised when the gateway bridge returns an error."""


class ToolCallError(RuntimeError):
    """Raised when an upstream MCP tool returns an error result."""


class BridgeClient:
    """JSON-RPC client running on a dedicated asyncio loop."""

    def __init__(self, host: str, port: int):
        self._host = host
        self._port = port
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="mcp-gateway-bridge"
        )
        self._thread.start()
        self._id_lock = threading.Lock()
        self._counter = 0

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _next_id(self) -> int:
        with self._id_lock:
            self._counter += 1
            return self._counter

    async def _request_async(self, method: str, params: dict[str, Any]) -> Any:
        reader, writer = await asyncio.open_connection(self._host, self._port)
        try:
            request_id = self._next_id()
            payload = make_request(method, params, request_id=request_id)
            try:
                writer.write(encode_frame(payload, default=_default_serializer))
            except ValueError as exc:
                raise BridgeError(str(exc)) from exc
            await writer.drain()
            try:
                message = await read_frame(reader)
                if message is None:
                    raise BridgeError("Bridge closed connection unexpectedly")
                return parse_response(decode(message), expected_id=request_id)
            except (ConnectionError, ProtocolError) as exc:
                raise BridgeError(str(exc)) from exc
            except JSONRPCError as exc:
                raise BridgeError(exc.message) from exc
        finally:
            writer.close()
            await writer.wait_closed()

    def request(self, method: str, params: dict[str, Any]) -> Any:
        future = asyncio.run_coroutine_threadsafe(
            self._request_async(method, params), self._loop
        )
        return future.result()

    async def request_async(self, method: str, params: dict[str, Any]) -> Any:
        return await self._request_async(method, params)


_bridge: BridgeClient | None = None
_importer_installed = False


def _get_bridge() -> BridgeClient:
    global _bridge
    if _bridge is None:
        config = os.environ.get(BRIDGE_ENV)
        if not config:
            raise BridgeError(f"Bridge environment variable '{BRIDGE_ENV}' not set")
        host, port_str = config.split(":")
        _bridge = BridgeClient(host, int(port_str))
    return _bridge


def _tool_error_message(payload: Any) -> str | None:
    if not isinstance(payload, dict) or not payload.get("isError"):
        return None

    structured = payload.get("structuredContent")
    if isinstance(structured, dict):
        for key in ("error", "message"):
            value = structured.get(key)
            if isinstance(value, str) and value:
                return value

    content = payload.get("content")
    if isinstance(content, list):
        for entry in content:
            if not isinstance(entry, dict):
                continue
            text = entry.get("text")
            if isinstance(text, str) and text:
                return text
            data = entry.get("data")
            if isinstance(data, dict):
                for key in ("error", "message"):
                    value = data.get(key)
                    if isinstance(value, str) and value:
                        return value
            if data is not None:
                return str(data)

    if structured is not None:
        return str(structured)
    return "Upstream tool call failed"


def _raise_for_tool_error(payload: Any) -> Any:
    message = _tool_error_message(payload)
    if message is not None:
        raise ToolCallError(message)
    return payload


async def call_tool_async(
    server: str,
    tool: str,
    arguments: dict[str, Any] | None = None,
) -> Any:
    """Call an upstream MCP tool through the active sandbox bridge."""
    arguments = arguments or {}
    bridge = _get_bridge()
    payload = await bridge.request_async(
        "call_tool", {"server": server, "tool": tool, "arguments": arguments}
    )
    return _raise_for_tool_error(payload)


def call_tool_sync(
    server: str,
    tool: str,
    arguments: dict[str, Any] | None = None,
) -> Any:
    """Synchronously call an upstream MCP tool through the sandbox bridge."""
    arguments = arguments or {}
    bridge = _get_bridge()
    payload = bridge.request(
        "call_tool", {"server": server, "tool": tool, "arguments": arguments}
    )
    return _raise_for_tool_error(payload)


def fetch_server_module_spec(server: str) -> dict[str, Any]:
    """Fetch generated-wrapper metadata for an upstream server."""
    bridge = _get_bridge()
    result = bridge.request("ensure_wrapper", {"server": server})
    if not isinstance(result, dict):
        raise BridgeError("Gateway returned malformed wrapper metadata")
    return result


class _ServerModuleLoader(importlib.abc.Loader):
    def __init__(self, finder: WrapperFinder, server: str):
        self._finder = finder
        self._server = server

    def create_module(self, spec):
        return None  # use default module creation

    def exec_module(self, module) -> None:
        spec = self._finder.get_server_spec(self._server)
        build_server_module(module, spec)


class WrapperFinder(importlib.abc.MetaPathFinder):
    """Meta path finder that asks the gateway to generate wrapper modules."""

    def __init__(self):
        self._spec_cache: dict[str, dict[str, Any]] = {}
        self._cache_lock = threading.Lock()

    def get_server_spec(self, server: str) -> dict[str, Any]:
        with self._cache_lock:
            spec = self._spec_cache.get(server)
        if spec is not None:
            return spec
        spec = fetch_server_module_spec(server)
        with self._cache_lock:
            self._spec_cache[server] = spec
        return spec

    def find_spec(self, fullname: str, path, target=None):
        if not fullname.startswith(WRAPPERS_PACKAGE):
            return None

        parts = fullname.split(".")
        if len(parts) == 1:
            return None  # Root package handled by filesystem

        if len(parts) == 2:
            server = parts[1]
            loader = _ServerModuleLoader(self, server)
            return importlib.util.spec_from_loader(fullname, loader)

        return None


def install_wrapper_importer() -> None:
    """Install the import hook that exposes generated MCP wrapper modules."""
    global _importer_installed
    if _importer_installed:
        return
    importlib.invalidate_caches()
    importer = WrapperFinder()
    if importer not in sys.meta_path:
        sys.meta_path.insert(0, importer)
    _importer_installed = True


__all__ = [
    "BridgeError",
    "ToolCallError",
    "call_tool_async",
    "call_tool_sync",
    "install_wrapper_importer",
]
