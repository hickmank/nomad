"""JSON-RPC 2.0 messages, dispatch, and framing for Nomad's internal channels.

Nomad speaks JSON-RPC 2.0 (https://www.jsonrpc.org/specification) on two
channels: between the server and its model workers (newline-delimited over
stdio) and between the code-mode gateway and its sandbox runtime
(length-prefixed over TCP). This module holds what both channels share; each
channel owns its transport, except that ``JSONRPCServer`` and ``JSONRPCClient``
provide the two ends of the newline-delimited stdio channel used by model
workers.

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
import threading
import time
import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from queue import Empty, Queue
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


class JSONRPCClient:
    """Send requests to a newline-delimited JSON-RPC peer and wait for replies.

    One request is in flight at a time; concurrent callers queue on a lock. A
    background thread reads the peer's responses so that a request can give up
    at its deadline instead of blocking in ``readline``.
    """

    def __init__(
        self,
        writer: TextIO,
        reader: TextIO,
        *,
        timeout_seconds: float | None = 30.0,
        on_failure: Callable[[], None] | None = None,
        name: str = "jsonrpc",
    ) -> None:
        """Start reading responses from ``reader``.

        Args:
            writer: Stream that requests are written to.
            reader: Stream of newline-delimited responses.
            timeout_seconds: Deadline for each request, including waiting for
                the lock and writing the request. ``None`` waits forever.
            on_failure: Called, with the lock held, when a request times out or
                the peer sends an invalid response, before the error is raised.
                Use it to stop a peer that can no longer be trusted.
            name: Prefix for the names of the client's threads.
        """
        self.timeout_seconds = timeout_seconds
        self._writer = writer
        self._on_failure = on_failure
        self._name = name
        self._request_id = 0
        # Reentrant so a caller holding the lock (see ``acquire``) can still
        # call ``send`` or ``request``.
        self._lock = threading.RLock()
        self._responses: Queue[str | None] = Queue()
        self._reader_thread = threading.Thread(
            target=self._read_responses,
            args=(reader,),
            daemon=True,
            name=f"{name}-stdout",
        )
        self._reader_thread.start()

    def acquire(self, timeout: float | None = None) -> bool:
        """Hold the request lock so no request starts until ``release``."""
        if timeout is None:
            return self._lock.acquire()
        return self._lock.acquire(timeout=timeout)

    def release(self) -> None:
        """Release a lock taken with ``acquire``."""
        self._lock.release()

    def request(self, method: str, params: Mapping[str, Any] | None = None) -> Any:
        """Send one request and return its result.

        Raises:
            TimeoutError: If the request does not complete within
                ``timeout_seconds``.
            ConnectionError: If the request cannot be written or the peer
                closes its output before replying.
            ProtocolError: If the peer's response is not valid JSON-RPC.
            JSONRPCError: If the peer returns an error response.
        """
        timeout_seconds = self.timeout_seconds
        deadline = (
            None if timeout_seconds is None else time.monotonic() + timeout_seconds
        )
        if not self.acquire(timeout_seconds):
            raise self._timeout_error(method)
        try:
            self._request_id += 1
            request_id = self._request_id
            self._write(
                encode_line(make_request(method, params, request_id=request_id)),
                method=method,
                deadline=deadline,
            )
            response_timeout = (
                None if deadline is None else max(0.0, deadline - time.monotonic())
            )
            try:
                response_line = self._responses.get(timeout=response_timeout)
            except Empty as exc:
                self._fail()
                raise self._timeout_error(method) from exc
            if response_line is None:
                raise ConnectionError("Peer closed its output")
            try:
                return parse_response(decode(response_line), expected_id=request_id)
            except ProtocolError:
                self._fail()
                raise
        finally:
            self.release()

    def send(self, method: str, params: Mapping[str, Any] | None = None) -> None:
        """Write a request without waiting for its response.

        Raises:
            ConnectionError: If the request cannot be written.
        """
        with self._lock:
            self._request_id += 1
            payload = make_request(method, params, request_id=self._request_id)
            try:
                self._writer.write(encode_line(payload))
                self._writer.flush()
            except (OSError, ValueError) as exc:
                raise ConnectionError("Cannot write to peer") from exc

    def _write(self, line: str, *, method: str, deadline: float | None) -> None:
        """Write one line without letting a blocked pipe evade the deadline."""
        errors: list[BaseException] = []
        completed = threading.Event()

        def write() -> None:
            try:
                self._writer.write(line)
                self._writer.flush()
            except BaseException as exc:  # transported back to the request thread
                errors.append(exc)
            finally:
                completed.set()

        if deadline is None:
            write()
        else:
            writer = threading.Thread(
                target=write, daemon=True, name=f"{self._name}-stdin"
            )
            writer.start()
            if not completed.wait(max(0.0, deadline - time.monotonic())):
                self._fail()
                raise self._timeout_error(method)

        if errors:
            raise ConnectionError("Cannot write to peer") from errors[0]

    def _fail(self) -> None:
        if self._on_failure is not None:
            self._on_failure()

    def _timeout_error(self, method: str) -> TimeoutError:
        assert self.timeout_seconds is not None
        return TimeoutError(
            f"JSON-RPC request '{method}' timed out after "
            f"{self.timeout_seconds:g} seconds"
        )

    def _read_responses(self, reader: TextIO) -> None:
        try:
            for line in reader:
                self._responses.put(line)
        finally:
            # Wake a waiting request when the peer's output ends.
            self._responses.put(None)


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
    "JSONRPCClient",
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
