from __future__ import annotations

import logging
import ssl
from pathlib import Path
from typing import Any

import pytest
from fastmcp.mcp_config import RemoteMCPServer
from typer.testing import CliRunner

from nomad.common.config_errors import ConfigError
from nomad.common.upstream_errors import UpstreamConnectionError
from nomad.gateway import cli, server
from nomad.gateway.config import GatewayConfig, GatewayDefaults

runner = CliRunner()


class _FakeCodeModeGateway:
    created_with: GatewayConfig | None = None

    def __init__(self, config: GatewayConfig):
        type(self).created_with = config

    async def serve(self, transport=None, **kwargs):
        return None


@pytest.mark.parametrize("transport", [None, "stdio"])
def test_stdio_defaults_workspace_to_cwd(
    monkeypatch,
    tmp_path: Path,
    transport: str | None,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(server, "CodeModeGateway", _FakeCodeModeGateway)
    config = GatewayConfig(servers={})

    server.run_gateway(config, transport=transport)

    created = _FakeCodeModeGateway.created_with
    assert created is not None
    assert created.defaults.workspace_root == tmp_path.resolve()


def test_stdio_preserves_configured_workspace(monkeypatch, tmp_path: Path):
    launch_dir = tmp_path / "launch"
    launch_dir.mkdir()
    configured_workspace = tmp_path / "configured"
    monkeypatch.chdir(launch_dir)
    monkeypatch.setattr(server, "CodeModeGateway", _FakeCodeModeGateway)
    config = GatewayConfig(
        servers={},
        defaults=GatewayDefaults(workspace_root=configured_workspace),
    )

    server.run_gateway(config, transport="stdio")

    created = _FakeCodeModeGateway.created_with
    assert created is not None
    assert created.defaults.workspace_root == configured_workspace


def test_http_keeps_temporary_workspace_default(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(server, "CodeModeGateway", _FakeCodeModeGateway)
    config = GatewayConfig(servers={})
    temporary_workspace = config.defaults.workspace_root

    server.run_gateway(config, transport="http")

    created = _FakeCodeModeGateway.created_with
    assert created is not None
    assert created.defaults.workspace_root == temporary_workspace
    assert created.defaults.workspace_root != tmp_path.resolve()


@pytest.fixture
def dummy_config(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text("servers: {}\n", encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "transport",
    ["http", "streamable-http", "streamable_http"],
)
def test_http_transport_normalizes(
    dummy_config: Path,
    monkeypatch,
    transport: str,
):
    captured: dict[str, Any] = {}

    monkeypatch.setattr(cli, "_configure_logging", lambda level, log_file=None: None)

    def fake_run_gateway(
        config, *, transport=None, log_level="INFO", log_file_handler=None, **kwargs
    ):
        captured["config"] = config
        captured["transport"] = transport
        captured["log_level"] = log_level
        captured["log_file_handler"] = log_file_handler
        captured["kwargs"] = kwargs

    monkeypatch.setattr(cli, "run_gateway", fake_run_gateway)

    result = runner.invoke(
        cli.app,
        [
            "--config",
            str(dummy_config),
            "--transport",
            transport,
        ],
    )
    assert result.exit_code == 0
    assert isinstance(captured["config"], GatewayConfig)
    assert captured["transport"] == "http"
    assert captured["log_level"] == "INFO"
    assert captured["log_file_handler"] is None
    assert captured["kwargs"] == {}


def test_invalid_transport(dummy_config: Path, monkeypatch):
    monkeypatch.setattr(cli, "_configure_logging", lambda level, log_file=None: None)

    result = runner.invoke(
        cli.app,
        [
            "--config",
            str(dummy_config),
            "--transport",
            "not_a_transport",
        ],
    )

    assert result.exit_code == 2


def test_sse_transport_cli_rejected(dummy_config: Path, monkeypatch):
    monkeypatch.setattr(cli, "_configure_logging", lambda level, log_file=None: None)

    result = runner.invoke(
        cli.app,
        [
            "--config",
            str(dummy_config),
            "--transport",
            "sse",
        ],
    )

    assert result.exit_code == 2


def test_sse_server_config_rejected(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "servers:\n  bad:\n    transport: sse\n    url: http://localhost:9000/sse\n",
        encoding="utf-8",
    )

    with pytest.raises(Exception, match="Unsupported MCP transport 'sse'"):
        GatewayConfig.from_file(config_path)


def test_missing_gateway_config_has_specific_error(tmp_path: Path):
    config_path = tmp_path / "missing.yaml"

    with pytest.raises(ConfigError, match="Config file not found"):
        GatewayConfig.from_file(config_path)


def test_malformed_gateway_config_cli_has_parse_location(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(cli, "_configure_logging", lambda level, log_file=None: None)
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("servers:\n  broken: [\n", encoding="utf-8")

    result = runner.invoke(cli.app, ["--config", str(config_path)])

    assert result.exit_code == 2
    assert "YAML syntax error" in result.output
    assert "line 3" in result.output
    assert "Traceback" not in result.output


def test_gateway_config_schema_error_lists_field(tmp_path: Path):
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(
        "servers: {}\nunknown: value\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError) as exc_info:
        GatewayConfig.from_file(config_path)

    message = str(exc_info.value)
    assert "Config schema validation failed. Fix the following issue(s):" in message
    assert "Unknown field `unknown`" in message


def test_gateway_config_schema_error_describes_bad_servers_section(tmp_path: Path):
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("servers: []\n", encoding="utf-8")

    with pytest.raises(ConfigError) as exc_info:
        GatewayConfig.from_file(config_path)

    assert "'servers' must be a mapping" in str(exc_info.value)


def test_gateway_config_schema_error_names_invalid_server(tmp_path: Path):
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(
        "servers:\n  secure:\n    transport: http\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError) as exc_info:
        GatewayConfig.from_file(config_path)

    message = str(exc_info.value)
    assert "Server 'secure' config is invalid" in message
    assert "Missing required field `url`" in message


@pytest.mark.parametrize("transport", ["http", "streamable-http", "streamable_http"])
def test_http_server_config_transport_aliases(tmp_path: Path, transport: str):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"servers:\n  secure:\n    transport: {transport}\n    url: https://example.invalid/mcp\n",
        encoding="utf-8",
    )

    config = GatewayConfig.from_file(config_path)

    assert isinstance(config.servers["secure"], RemoteMCPServer)
    assert config.servers["secure"].transport == "http"


def test_upstream_connection_failure_cli_is_concise(
    dummy_config: Path,
    monkeypatch,
):
    monkeypatch.setattr(cli, "_configure_logging", lambda level, log_file=None: None)

    params = RemoteMCPServer(url="https://example.invalid/mcp")

    def fake_run_gateway(*args, **kwargs):
        raise UpstreamConnectionError(
            "secure",
            params,
            RuntimeError("401 Unauthorized"),
        )

    monkeypatch.setattr(cli, "run_gateway", fake_run_gateway)

    result = runner.invoke(cli.app, ["--config", str(dummy_config)])

    assert result.exit_code == 1
    assert "Failed to connect to upstream MCP server 'secure'" in result.output
    assert "Authentication failed with HTTP 401" in result.output
    assert "Traceback" not in result.output


def test_upstream_ssl_failure_message_names_tls():
    params = RemoteMCPServer(url="https://example.invalid/mcp")

    error = UpstreamConnectionError(
        "secure",
        params,
        ssl.SSLError("CERTIFICATE_VERIFY_FAILED"),
    )

    message = str(error)
    assert "TLS certificate verification failed" in message
    assert "https://example.invalid/mcp" in message


def test_upstream_connection_failure_handles_cyclic_exception_chain():
    params = RemoteMCPServer(url="https://example.invalid/mcp")
    first = RuntimeError("first")
    second = RuntimeError("second")
    first.__cause__ = second
    second.__context__ = first

    error = UpstreamConnectionError("cyclic", params, first)

    assert "Failed to connect to upstream MCP server 'cyclic'" in str(error)
    assert "Underlying error: RuntimeError: first" in str(error)


def test_log_file_forwarded(dummy_config: Path, monkeypatch, tmp_path: Path):
    captured: dict[str, Any] = {}
    log_file = tmp_path / "gateway.jsonl"
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")

    monkeypatch.setattr(
        cli,
        "_configure_logging",
        lambda level, log_file=None: file_handler,
    )

    def fake_run_gateway(
        config, *, transport=None, log_level="INFO", log_file_handler=None, **kwargs
    ):
        captured["transport"] = transport
        captured["log_level"] = log_level
        captured["log_file_handler"] = log_file_handler

    monkeypatch.setattr(cli, "run_gateway", fake_run_gateway)

    result = runner.invoke(
        cli.app,
        [
            "--config",
            str(dummy_config),
            "--log-file",
            str(log_file),
        ],
    )

    assert result.exit_code == 0
    assert captured["transport"] is None
    assert captured["log_level"] == "INFO"
    assert captured["log_file_handler"] is file_handler
