"""JSON-RPC 2.0 messages, dispatch, and framing for Nomad's internal channels.

Nomad speaks JSON-RPC 2.0 (https://www.jsonrpc.org/specification) on two
channels: between the server and its model workers (newline-delimited over
stdio) and between the code-mode gateway and its sandbox runtime
(length-prefixed over TCP). This module holds what both channels share; each
channel owns its transport, except that ``JSONRPCServer`` provides the
newline-delimited stdio server loop used by model workers.

Methods take by-name params, and batch requests are not supported. The module
depends only on the standard library because model workers and the sandbox
runtime import it from interpreters other than the Nomad host's.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import os
import sys
import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, TextIO

JSONRPC_VERSION = "2.0"

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
SERVER_ERROR = -32000
"""Implementation-defined server error, used for exceptions raised by handlers."""

FRAME_HEADER_SIZE = 4
DEFAULT_MAX_FRAME_SIZE = 16 * 1024 * 1024

Handler = Callable[[dict[str, Any]], Any]

_RPC_METHOD_ATTRIBUTE = "__jsonrpc_method__"


class JSONRPCError(RuntimeError):
    """A JSON-RPC error object, either received from or sent to a peer."""

    def __init__(self, message: str, *, code: int = SERVER_ERROR, data: Any = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.data = data

    def to_dict(self) -> dict[str, Any]:
        error: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.data is not None:
            error["data"] = self.data
        return error


class ProtocolError(ValueError):
    """Raised when a peer sends a message that is not valid JSON-RPC 2.0."""


@dataclass(frozen=True, slots=True)
class Request:
    method: str
    params: dict[str, Any] | list[Any]
    id: int | str | None
    is_notification: bool


def make_request(
    method: str,
    params: Mapping[str, Any] | None = None,
    *,
    request_id: int | str,
) -> dict[str, Any]:
    return {
        "jsonrpc": JSONRPC_VERSION,
        "id": request_id,
        "method": method,
        "params": dict(params or {}),
    }


def make_result(request_id: int | str | None, result: Any) -> dict[str, Any]:
    return {"jsonrpc": JSONRPC_VERSION, "id": request_id, "result": result}


def make_error(request_id: int | str | None, error: JSONRPCError) -> dict[str, Any]:
    return {"jsonrpc": JSONRPC_VERSION, "id": request_id, "error": error.to_dict()}


def decode(message: str | bytes) -> Any:
    """Decode one serialized message, raising ``ProtocolError`` for invalid JSON."""
    try:
        return json.loads(message)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"invalid JSON: {exc}") from exc


def _is_valid_id(value: Any) -> bool:
    return value is None or (
        isinstance(value, int | str) and not isinstance(value, bool)
    )


def parse_request(payload: Any) -> Request:
    """Validate a decoded request, raising ``JSONRPCError`` with a spec error code."""
    if not isinstance(payload, dict) or payload.get("jsonrpc") != JSONRPC_VERSION:
        raise JSONRPCError("Invalid Request", code=INVALID_REQUEST)
    method = payload.get("method")
    request_id = payload.get("id")
    if not isinstance(method, str) or not _is_valid_id(request_id):
        raise JSONRPCError("Invalid Request", code=INVALID_REQUEST)
    params = payload.get("params", {})
    if not isinstance(params, dict | list):
        raise JSONRPCError("Invalid params", code=INVALID_PARAMS)
    return Request(
        method=method,
        params=params,
        id=request_id,
        is_notification="id" not in payload,
    )


def parse_response(payload: Any, *, expected_id: int | str) -> Any:
    """Return a decoded response's result or raise the peer's ``JSONRPCError``.

    Raises ``ProtocolError`` if the response is malformed or answers another
    request.
    """
    if not isinstance(payload, dict) or payload.get("jsonrpc") != JSONRPC_VERSION:
        raise ProtocolError("response is not a JSON-RPC 2.0 object")
    if ("result" in payload) == ("error" in payload):
        raise ProtocolError("response must contain exactly one of result or error")

    response_id = payload.get("id")
    if "error" in payload:
        error = payload["error"]
        if (
            not isinstance(error, dict)
            or not isinstance(error.get("code"), int)
            or not isinstance(error.get("message"), str)
        ):
            raise ProtocolError("response error object is malformed")
        # A null id means the peer could not read the request's id.
        if response_id is not None and response_id != expected_id:
            raise ProtocolError("response id does not match the request")
        raise JSONRPCError(error["message"], code=error["code"], data=error.get("data"))

    if response_id != expected_id:
        raise ProtocolError("response id does not match the request")
    return payload["result"]


class Dispatcher:
    """Route serialized requests to handlers and build spec-compliant responses.

    Handlers receive the request's params. A handler may raise ``JSONRPCError``
    to choose the error code; any other exception becomes a ``SERVER_ERROR``
    response and is passed to ``on_error``.
    """

    def __init__(
        self,
        handlers: Mapping[str, Handler],
        *,
        on_error: Callable[[Exception], None] | None = None,
    ):
        self._handlers = dict(handlers)
        self._on_error = on_error

    def dispatch(self, message: str | bytes) -> dict[str, Any] | None:
        """Handle one request; return its response, or ``None`` for a notification."""
        request, response = self._parse(message)
        if request is not None:
            try:
                result = self._call(request)
                if inspect.isawaitable(result):
                    if inspect.iscoroutine(result):
                        result.close()
                    raise TypeError(f"Handler for '{request.method}' is async")
                response = make_result(request.id, result)
            except Exception as exc:  # noqa: BLE001
                response = self._error_response(request.id, exc)
        return self._reply(request, response)

    async def dispatch_async(self, message: str | bytes) -> dict[str, Any] | None:
        """Like ``dispatch``, awaiting handlers that return awaitables."""
        request, response = self._parse(message)
        if request is not None:
            try:
                result = self._call(request)
                if inspect.isawaitable(result):
                    result = await result
                response = make_result(request.id, result)
            except Exception as exc:  # noqa: BLE001
                response = self._error_response(request.id, exc)
        return self._reply(request, response)

    @staticmethod
    def _parse(
        message: str | bytes,
    ) -> tuple[Request | None, dict[str, Any] | None]:
        try:
            payload = decode(message)
        except ProtocolError:
            return None, make_error(None, JSONRPCError("Parse error", code=PARSE_ERROR))
        try:
            return parse_request(payload), None
        except JSONRPCError as error:
            return None, make_error(None, error)

    def _call(self, request: Request) -> Any:
        handler = self._handlers.get(request.method)
        if handler is None:
            raise JSONRPCError(
                f"Method not found: '{request.method}'", code=METHOD_NOT_FOUND
            )
        if not isinstance(request.params, dict):
            raise JSONRPCError(
                f"Method '{request.method}' takes by-name params",
                code=INVALID_PARAMS,
            )
        return handler(request.params)

    def _error_response(
        self, request_id: int | str | None, exc: Exception
    ) -> dict[str, Any]:
        if isinstance(exc, JSONRPCError):
            return make_error(request_id, exc)
        if self._on_error is not None:
            self._on_error(exc)
        return make_error(request_id, JSONRPCError(str(exc), code=SERVER_ERROR))

    @staticmethod
    def _reply(
        request: Request | None, response: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        if request is not None and request.is_notification:
            return None
        return response


def rpc_method[F: Callable[..., Any]](func: F) -> F:
    """Expose a ``JSONRPCServer`` method as the RPC method of the same name.

    The method is called with the request's params mapping. Apply it beneath
    ``@staticmethod`` when combining the two.
    """
    setattr(func, _RPC_METHOD_ATTRIBUTE, func.__name__)
    return func


class JSONRPCServer:
    """Serve a subclass's ``@rpc_method`` methods over newline-delimited stdio.

    Subclasses tag their RPC methods with ``rpc_method`` and run ``start``; a
    handler calls ``stop`` to end the loop once its response is written.
    """

    def __init__(self) -> None:
        self._stopping = False

    def handlers(self) -> dict[str, Handler]:
        """Return the bound ``@rpc_method`` methods keyed by RPC method name."""
        handlers: dict[str, Handler] = {}
        for attribute in dir(type(self)):
            # Inspect the class statically so that discovering methods never
            # evaluates properties or other descriptors.
            value = inspect.getattr_static(self, attribute)
            func = getattr(value, "__func__", value)
            method_name = getattr(func, _RPC_METHOD_ATTRIBUTE, None)
            if method_name is not None:
                handlers[method_name] = getattr(self, attribute)
        return handlers

    def stop(self) -> None:
        """Make ``start`` return after the current request is answered."""
        self._stopping = True

    def on_error(self, exc: Exception) -> None:
        """Report an exception raised by a handler; prints to stderr by default."""
        traceback.print_exception(exc, file=sys.stderr)

    def start(self, stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout) -> int:
        """Answer requests read from ``stdin`` until it closes or ``stop`` is called.

        Args:
            stdin: Stream of newline-delimited requests.
            stdout: Stream that receives newline-delimited responses.

        Returns:
            The process exit code, ``0``.
        """
        protocol_stdout = stdout
        close_protocol_stdout = False
        if stdout is sys.stdout and hasattr(stdout, "fileno"):
            # Keep a private copy of the real stdout for protocol messages and
            # point file descriptor 1 at stderr, so output written by handler
            # code (including native extensions) cannot corrupt the protocol.
            protocol_fd = os.dup(stdout.fileno())
            os.dup2(sys.stderr.fileno(), stdout.fileno())
            protocol_stdout = os.fdopen(
                protocol_fd,
                "w",
                encoding=stdout.encoding or "utf-8",
                buffering=1,
            )
            close_protocol_stdout = True

        try:
            with contextlib.redirect_stdout(sys.stderr):
                dispatcher = Dispatcher(self.handlers(), on_error=self.on_error)
                for line in stdin:
                    response = dispatcher.dispatch(line)
                    if response is not None:
                        protocol_stdout.write(encode_response(response, encode_line))
                        protocol_stdout.flush()
                    if self._stopping:
                        break
        finally:
            if close_protocol_stdout:
                protocol_stdout.close()
        return 0


def encode_response[T: (str, bytes)](
    response: Mapping[str, Any],
    encode: Callable[[Mapping[str, Any]], T],
) -> T:
    """Encode a response, replacing it with an internal error if encoding fails."""
    try:
        return encode(response)
    except (TypeError, ValueError) as exc:
        error = JSONRPCError(str(exc), code=INTERNAL_ERROR)
        return encode(make_error(response.get("id"), error))


def encode_line(
    payload: Mapping[str, Any],
    *,
    default: Callable[[Any], Any] | None = None,
) -> str:
    """Encode a message as one newline-terminated line of JSON."""
    # json.dumps escapes newlines inside strings, so the message stays on one line.
    return json.dumps(payload, default=default) + "\n"


def encode_frame(
    payload: Mapping[str, Any],
    *,
    default: Callable[[Any], Any] | None = None,
    max_size: int = DEFAULT_MAX_FRAME_SIZE,
) -> bytes:
    """Encode a message as a big-endian length-prefixed frame."""
    body = json.dumps(payload, default=default).encode("utf-8")
    if len(body) > max_size:
        raise ValueError(f"JSON-RPC message exceeds max size ({max_size} bytes)")
    return len(body).to_bytes(FRAME_HEADER_SIZE, byteorder="big") + body


async def read_frame(
    reader: asyncio.StreamReader,
    *,
    max_size: int = DEFAULT_MAX_FRAME_SIZE,
) -> bytes | None:
    """Read one length-prefixed frame, or return ``None`` at a clean end of stream."""
    try:
        header = await reader.readexactly(FRAME_HEADER_SIZE)
    except asyncio.IncompleteReadError as exc:
        if not exc.partial:
            return None
        raise ConnectionError("Peer closed connection mid-frame") from exc
    size = int.from_bytes(header, byteorder="big")
    if size > max_size:
        raise ProtocolError(f"JSON-RPC message exceeds max size ({max_size} bytes)")
    try:
        return await reader.readexactly(size)
    except asyncio.IncompleteReadError as exc:
        raise ConnectionError("Peer closed connection mid-frame") from exc


__all__ = [
    "DEFAULT_MAX_FRAME_SIZE",
    "INTERNAL_ERROR",
    "INVALID_PARAMS",
    "INVALID_REQUEST",
    "JSONRPCError",
    "METHOD_NOT_FOUND",
    "PARSE_ERROR",
    "SERVER_ERROR",
    "Dispatcher",
    "JSONRPCServer",
    "ProtocolError",
    "Request",
    "decode",
    "encode_frame",
    "encode_line",
    "encode_response",
    "make_error",
    "make_request",
    "make_result",
    "parse_request",
    "parse_response",
    "read_frame",
    "rpc_method",
]
