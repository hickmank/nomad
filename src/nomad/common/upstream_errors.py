from __future__ import annotations

import ssl
from collections.abc import Iterable
from typing import Any

from fastmcp.mcp_config import RemoteMCPServer, StdioMCPServer


class UpstreamConnectionError(RuntimeError):
    """Raised when the gateway cannot initialize an upstream MCP session."""

    def __init__(self, server: str, params: Any, cause: BaseException):
        self.server = server
        self.params = params
        self.cause = cause
        super().__init__(_format_upstream_connection_error(server, params, cause))


def _format_upstream_connection_error(
    server: str,
    params: Any,
    cause: BaseException,
) -> str:
    target = _describe_server_params(params)
    classification = _classify_connection_failure(cause)
    detail = _first_meaningful_exception_message(cause)

    lines = [f"Failed to connect to upstream MCP server '{server}' ({target})."]
    if classification:
        lines.append(classification)
    if detail:
        lines.append(f"Underlying error: {detail}")
    lines.append(
        "Check the server URL/command, credentials, network access, and TLS "
        "trust configuration."
    )
    return " ".join(lines)


def _describe_server_params(params: Any) -> str:
    if isinstance(params, RemoteMCPServer):
        return f"http {params.url}"
    if isinstance(params, StdioMCPServer):
        args = " ".join(params.args or [])
        command = f"{params.command} {args}".strip()
        return f"stdio command '{command}'"
    transport = getattr(params, "transport", None)
    url = getattr(params, "url", None)
    if url:
        return f"{transport or 'http'} {url}"
    command = getattr(params, "command", None)
    if command:
        return f"stdio command '{command}'"
    return type(params).__name__


def _classify_connection_failure(cause: BaseException) -> str | None:
    for exc in _walk_exceptions(cause):
        status_code = _http_status_code(exc)
        if status_code in {401, 403}:
            return (
                f"Authentication failed with HTTP {status_code}. "
                "Verify the upstream MCP server token or authorization headers."
            )
        if isinstance(exc, ssl.SSLError) or _looks_like_tls_error(exc):
            return (
                "TLS certificate verification failed. Install the issuing CA, "
                "fix the certificate chain, or point the client at the correct "
                "trust store."
            )
    return None


def _http_status_code(exc: BaseException) -> int | None:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if isinstance(status_code, int):
        return status_code

    text = f"{type(exc).__name__}: {exc}"
    for status_code in (401, 403):
        if str(status_code) in text:
            return status_code
    return None


def _looks_like_tls_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(
        marker in text
        for marker in (
            "certificate_verify_failed",
            "certificate verify failed",
            "ssl:",
            "tls:",
        )
    )


def _first_meaningful_exception_message(cause: BaseException) -> str:
    for exc in _walk_exceptions(cause):
        if isinstance(exc, BaseExceptionGroup):
            continue
        message = str(exc).strip()
        if message and not message.startswith("unhandled errors in a TaskGroup"):
            return f"{type(exc).__name__}: {message}"
    return type(cause).__name__


def _walk_exceptions(exc: BaseException) -> Iterable[BaseException]:
    pending = [exc]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        identity = id(current)
        if identity in seen:
            continue
        seen.add(identity)
        yield current

        if current.__context__ is not None:
            pending.append(current.__context__)
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if isinstance(current, BaseExceptionGroup):
            pending.extend(reversed(current.exceptions))
