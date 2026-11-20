from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from fastmcp import FastMCP
from fastmcp.mcp_config import StdioMCPServer
from pydantic import BaseModel
from typer.testing import CliRunner

from nomad import cli as nomad_cli
from nomad import export as nomad_export
from nomad.config import SearchToolConfig
from nomad.export import ExportTarget
from nomad.gateway import cli as gateway_cli
from nomad.gateway.server import SandboxResult
from nomad.tool_search import register_search_tool

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text)


def test_nomad_cli_exposes_code_mode(monkeypatch, tmp_path: Path):
    called: dict[str, Any] = {}

    def fake_launch_code_mode(config, *, log_level, log_file, transport, host, port):
        called["config"] = config
        called["log_level"] = log_level
        called["log_file"] = log_file
        called["transport"] = transport
        called["host"] = host
        called["port"] = port

    monkeypatch.setattr(gateway_cli, "launch_code_mode", fake_launch_code_mode)

    config_path = tmp_path / "gateway.yaml"
    config_path.write_text("servers: {}\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        nomad_cli.app,
        [
            "code-mode",
            "--config",
            str(config_path),
            "--log-level",
            "DEBUG",
            "--log-file",
            str(tmp_path / "gateway.jsonl"),
            "--transport",
            "stdio",
            "--host",
            "127.0.0.1",
            "--port",
            "9001",
        ],
    )

    assert result.exit_code == 0
    assert called["config"] == config_path
    assert called["log_level"] == "DEBUG"
    assert called["log_file"] == tmp_path / "gateway.jsonl"
    assert called["transport"] == "stdio"
    assert called["host"] == "127.0.0.1"
    assert called["port"] == 9001


def test_nomad_cli_code_mode_exec_runs_script(monkeypatch, tmp_path: Path):
    called: dict[str, Any] = {}

    def fake_run_code_mode_script(**kwargs):
        called.update(kwargs)
        return {
            "result": {"value": 42},
            "tool_calls": [],
        }

    monkeypatch.setattr(
        nomad_cli, "run_code_mode_script", fake_run_code_mode_script, raising=True
    )

    config_path = tmp_path / "gateway.yaml"
    config_path.write_text("servers: {}\n", encoding="utf-8")
    script_path = tmp_path / "script.py"
    script_path.write_text("print('hi')\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        nomad_cli.app,
        [
            "code-mode-exec",
            str(script_path),
            "--config",
            str(config_path),
            "--gateway-log-level",
            "DEBUG",
        ],
    )

    assert result.exit_code == 0
    assert called["config_path"] == config_path
    assert called["script_path"] == script_path
    assert called["gateway_log_level"] == "DEBUG"
    assert called["script_args"] == []
    assert result.output == ""


def test_nomad_cli_cmx_forwards_args_after_separator(monkeypatch, tmp_path: Path):
    called: dict[str, Any] = {}

    def fake_run_code_mode_script(**kwargs):
        called.update(kwargs)
        return {
            "result": None,
            "tool_calls": [],
        }

    monkeypatch.setattr(
        nomad_cli, "run_code_mode_script", fake_run_code_mode_script, raising=True
    )

    config_path = tmp_path / "gateway.yaml"
    config_path.write_text("servers: {}\n", encoding="utf-8")
    script_path = tmp_path / "script.py"
    script_path.write_text("print('hi')\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        nomad_cli.app,
        [
            "cmx",
            str(script_path),
            "--config",
            str(config_path),
            "--",
            "--sample-flag",
            "value",
        ],
    )

    assert result.exit_code == 0
    assert called["config_path"] == config_path
    assert called["script_path"] == script_path
    assert called["script_args"] == ["--sample-flag", "value"]
    assert result.output == ""


def test_nomad_cli_cmx_accepts_local_nomad_config(monkeypatch, tmp_path: Path):
    called: dict[str, Any] = {}

    def fake_run_code_mode_script(**kwargs):
        called.update(kwargs)
        return {"result": None, "tool_calls": []}

    monkeypatch.setattr(
        nomad_cli, "run_code_mode_script", fake_run_code_mode_script, raising=True
    )

    nomad_config_path = tmp_path / "nomad.yaml"
    nomad_config_path.write_text("tools: []\n", encoding="utf-8")
    script_path = tmp_path / "script.py"
    script_path.write_text("print('hi')\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        nomad_cli.app,
        [
            "cmx",
            "--nomad-config",
            str(nomad_config_path),
            str(script_path),
            "--",
            "--sample-flag",
            "value",
        ],
    )

    assert result.exit_code == 0
    assert called["config_path"] is None
    assert called["nomad_config_path"] == nomad_config_path
    assert called["script_path"] == script_path
    assert called["script_args"] == ["--sample-flag", "value"]


@pytest.mark.parametrize(
    ("config_options", "expected_error"),
    [
        ([], "One of --config or --nomad-config is required"),
        (
            ["--config", "{gateway}", "--nomad-config", "{nomad}"],
            "--config and --nomad-config are mutually exclusive",
        ),
    ],
)
def test_nomad_cli_cmx_requires_exactly_one_config(
    monkeypatch,
    tmp_path: Path,
    config_options: list[str],
    expected_error: str,
):
    def fail_run_code_mode_script(**kwargs):
        raise AssertionError("run_code_mode_script should not be called")

    monkeypatch.setattr(
        nomad_cli, "run_code_mode_script", fail_run_code_mode_script, raising=True
    )

    gateway_config = tmp_path / "gateway.yaml"
    gateway_config.write_text("servers: {}\n", encoding="utf-8")
    nomad_config = tmp_path / "nomad.yaml"
    nomad_config.write_text("tools: []\n", encoding="utf-8")
    script_path = tmp_path / "script.py"
    script_path.write_text("print('hi')\n", encoding="utf-8")
    options = [
        option.format(gateway=gateway_config, nomad=nomad_config)
        for option in config_options
    ]

    result = CliRunner().invoke(
        nomad_cli.app,
        ["cmx", *options, str(script_path)],
    )

    assert result.exit_code != 0
    assert expected_error in strip_ansi(result.output)


def test_local_nomad_gateway_config_uses_owned_stdio_server(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setenv("NOMAD_CMX_TEST_SENTINEL", "inherited")
    nomad_config = tmp_path / "nomad.yaml"
    workspace = tmp_path / "workspace"

    gateway_config = nomad_cli._local_nomad_gateway_config(nomad_config, workspace)

    server = gateway_config.servers["nomad"]
    assert gateway_config.defaults.workspace_root == workspace
    assert isinstance(server, StdioMCPServer)
    assert server.command == sys.executable
    assert server.args == [
        "-m",
        "nomad",
        "serve",
        str(nomad_config),
        "--transport",
        "stdio",
    ]
    assert server.env["NOMAD_CMX_TEST_SENTINEL"] == "inherited"
    assert server.keep_alive is False


def test_nomad_cli_cmx_runs_script_against_local_nomad_server(tmp_path: Path):
    tool_module = tmp_path / "cmx_test_tool.py"
    tool_module.write_text(
        "import os\n\ndef server_pid() -> int:\n    return os.getpid()\n",
        encoding="utf-8",
    )
    nomad_config = tmp_path / "nomad.yaml"
    nomad_config.write_text(
        "\n".join(
            [
                "tool_manager: {enabled: false}",
                "tools:",
                "  - tool: nomad.common.name_sanitize.sanitize_mcp_name",
                "    name: sanitize_mcp_name",
                "  - tool: cmx_test_tool.server_pid",
                "    name: server_pid",
                "",
            ]
        ),
        encoding="utf-8",
    )
    script_path = tmp_path / "script.py"
    script_path.write_text(
        "from mcp_tools.nomad import sanitize_mcp_name, server_pid\n"
        "RESULT = {\n"
        "    'value': sanitize_mcp_name(raw='hello world'),\n"
        "    'server_pid': server_pid(),\n"
        "}\n",
        encoding="utf-8",
    )
    existing_pythonpath = os.environ.get("PYTHONPATH")
    subprocess_env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            part for part in (str(tmp_path), existing_pythonpath) if part
        ),
    }

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nomad",
            "cmx",
            "--nomad-config",
            str(nomad_config),
            "--output",
            "-",
            str(script_path),
        ],
        capture_output=True,
        env=subprocess_env,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["result"]["value"] == "hello_world"
    with pytest.raises(ProcessLookupError):
        os.kill(payload["result"]["server_pid"], 0)


def test_nomad_cli_cmx_validates_local_config_before_startup(tmp_path: Path):
    nomad_config = tmp_path / "invalid-nomad.yaml"
    nomad_config.write_text("tools:\n  - [\n", encoding="utf-8")
    script_path = tmp_path / "script.py"
    script_path.write_text("RESULT = 1\n", encoding="utf-8")

    result = CliRunner().invoke(
        nomad_cli.app,
        [
            "cmx",
            "--nomad-config",
            str(nomad_config),
            str(script_path),
        ],
    )

    assert result.exit_code != 0
    output = strip_ansi(result.output)
    assert "Failed to load Nomad config" in output
    assert "Failed to connect to upstream MCP server" not in output
    assert "Traceback" not in output


def test_nomad_cli_code_mode_exec_forwards_args_after_separator(
    monkeypatch, tmp_path: Path
):
    called: dict[str, Any] = {}

    def fake_run_code_mode_script(**kwargs):
        called.update(kwargs)
        return {
            "result": None,
            "tool_calls": [],
        }

    monkeypatch.setattr(
        nomad_cli, "run_code_mode_script", fake_run_code_mode_script, raising=True
    )

    config_path = tmp_path / "gateway.yaml"
    config_path.write_text("servers: {}\n", encoding="utf-8")
    script_path = tmp_path / "script.py"
    script_path.write_text("print('hi')\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        nomad_cli.app,
        [
            "code-mode-exec",
            str(script_path),
            "--config",
            str(config_path),
            "--",
            "--sample-flag",
            "value",
        ],
    )

    assert result.exit_code == 0
    assert called["config_path"] == config_path
    assert called["script_path"] == script_path
    assert called["script_args"] == ["--sample-flag", "value"]
    assert result.output == ""


def test_nomad_cli_code_mode_exec_rejects_unknown_nomad_option(
    monkeypatch, tmp_path: Path
):
    def fail_run_code_mode_script(**kwargs):
        raise AssertionError("run_code_mode_script should not be called")

    monkeypatch.setattr(
        nomad_cli, "run_code_mode_script", fail_run_code_mode_script, raising=True
    )

    config_path = tmp_path / "gateway.yaml"
    config_path.write_text("servers: {}\n", encoding="utf-8")
    script_path = tmp_path / "script.py"
    script_path.write_text("print('hi')\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        nomad_cli.app,
        [
            "code-mode-exec",
            str(script_path),
            "--config",
            str(config_path),
            "--ouptut",
            "results.json",
        ],
        env={"FORCE_COLOR": "1"},
    )

    assert result.exit_code != 0
    output = strip_ansi(result.output)
    assert "--ouptut" in output
    assert "--output" in output


def test_nomad_cli_code_mode_exec_writes_output(monkeypatch, tmp_path: Path):
    def fake_run_code_mode_script(**kwargs):
        return {
            "result": None,
            "tool_calls": [{"tool": "add"}],
        }

    monkeypatch.setattr(
        nomad_cli, "run_code_mode_script", fake_run_code_mode_script, raising=True
    )

    config_path = tmp_path / "gateway.yaml"
    config_path.write_text("servers: {}\n", encoding="utf-8")
    script_path = tmp_path / "script.py"
    script_path.write_text("print('hi')\n", encoding="utf-8")
    output_path = tmp_path / "results.json"

    runner = CliRunner()
    result = runner.invoke(
        nomad_cli.app,
        [
            "code-mode-exec",
            str(script_path),
            "--config",
            str(config_path),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    assert result.output == ""

    written = output_path.read_text(encoding="utf-8")
    payload = json.loads(written)
    assert payload["tool_calls"] == [{"tool": "add"}]


def test_nomad_cli_code_mode_exec_writes_stdout_when_output_is_dash(
    monkeypatch, tmp_path: Path
):
    def fake_run_code_mode_script(**kwargs):
        return {
            "result": {"value": 42},
            "tool_calls": [],
        }

    monkeypatch.setattr(
        nomad_cli, "run_code_mode_script", fake_run_code_mode_script, raising=True
    )

    config_path = tmp_path / "gateway.yaml"
    config_path.write_text("servers: {}\n", encoding="utf-8")
    script_path = tmp_path / "script.py"
    script_path.write_text("print('hi')\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        nomad_cli.app,
        [
            "code-mode-exec",
            str(script_path),
            "--config",
            str(config_path),
            "--output",
            "-",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["result"]["value"] == 42


@pytest.mark.parametrize(
    "extra_args, expected_target, expected_oras_registry",
    [
        ([], ExportTarget.DISK, None),
        (["--to", "https"], ExportTarget.HTTPS, None),
        (
            ["--to", "oras", "--oras-registry", "registry.example.com/scifm"],
            ExportTarget.ORAS,
            "registry.example.com/scifm",
        ),
    ],
)
def test_nomad_cli_export_accepts_targets(
    monkeypatch,
    tmp_path: Path,
    extra_args: list[str],
    expected_target: ExportTarget,
    expected_oras_registry: str | None,
):
    config_path = tmp_path / "nomad.yml"
    config_path.write_text(
        "tool_manager: {enabled: true}\ntools: []\nfmod_models: []\n"
    )
    output_dir = tmp_path / "bundle"

    called: dict[str, Any] = {}

    def fake_export_models_config(
        config_path_arg, output_dir_arg, *, to, oras_registry, pin
    ):
        called["config_path"] = config_path_arg
        called["output_dir"] = output_dir_arg
        called["to"] = to
        called["oras_registry"] = oras_registry
        called["pin"] = pin
        return output_dir_arg / "nomad.yml"

    monkeypatch.setattr(nomad_export, "export_models_config", fake_export_models_config)

    runner = CliRunner()
    result = runner.invoke(
        nomad_cli.app,
        ["export", str(config_path), str(output_dir), *extra_args],
    )

    assert result.exit_code == 0
    assert result.output == ""
    assert called == {
        "config_path": config_path,
        "output_dir": output_dir,
        "to": expected_target,
        "oras_registry": expected_oras_registry,
        "pin": True,
    }


def test_nomad_cli_export_report(monkeypatch, tmp_path: Path):
    config_path = tmp_path / "nomad.yml"
    config_path.write_text(
        "tool_manager: {enabled: true}\ntools: []\nfmod_models: []\n"
    )
    output_dir = tmp_path / "report"
    called = []

    monkeypatch.setattr(
        nomad_export,
        "export_model_report",
        lambda config, output: called.append((config, output)),
    )
    monkeypatch.setattr(
        nomad_export,
        "export_models_config",
        lambda *args, **kwargs: pytest.fail("deployment export should not run"),
    )

    result = CliRunner().invoke(
        nomad_cli.app,
        ["export", "--report", str(config_path), str(output_dir)],
    )

    assert result.exit_code == 0
    assert called == [(config_path, output_dir)]


def test_nomad_cli_serve_registers_resolved_model_card_source(
    monkeypatch, tmp_path: Path
):
    resolved_model_dir = tmp_path / "resolved-model"
    resolved_model_dir.mkdir()

    class DummyInput(BaseModel):
        value: int = 0

    class DummyOutput(BaseModel):
        value: int = 0

    class DummyTool:
        name = "dummy-tool"
        description = "Dummy tool"
        args_schema = DummyInput
        output_schema = DummyOutput

        def __call__(self, input: DummyInput) -> DummyOutput:
            return DummyOutput(value=input.value)

    class DummyModelConfig:
        tool_name = None
        name_or_path = "models/demo-model"

        def __init__(self):
            self.resolve_calls: list[Path] = []

        def resolve_source(self, *, base_dir: Path | None = None):
            assert base_dir is not None
            self.resolve_calls.append(base_dir)
            return resolved_model_dir

    class DummyLocator:
        def __init__(self):
            self.register_calls: list[tuple[str, Path]] = []

        def register(self, tool_name, source):
            self.register_calls.append((tool_name, source))

    class DummyServer:
        def __init__(self, *args, **kwargs):
            self.registered_tools: list[str] = []
            self.transport: str | None = None

        def add_tool(self, tool):
            self.registered_tools.append(tool.name)
            return tool

        def tool(self, **kwargs):
            def decorator(fn):
                return fn

            return decorator

        def run(self, *, transport, **kwargs):
            self.transport = transport

    dummy_fm = DummyModelConfig()
    dummy_locator = DummyLocator()
    dummy_config = types.SimpleNamespace(
        tools=[],
        tool_manager=types.SimpleNamespace(enabled=False),
        fmod_models=[dummy_fm],
        context_dir=tmp_path / "config-dir",
        build_tool=lambda fm: DummyTool(),
        search_tool=types.SimpleNamespace(expose=False),
    )

    monkeypatch.setattr(
        nomad_cli.ServerConfig,
        "from_file",
        classmethod(lambda cls, path: dummy_config),
    )
    monkeypatch.setattr(nomad_cli, "FastMCP", DummyServer)
    monkeypatch.setattr(nomad_cli, "ModelCardLocator", lambda: dummy_locator)
    monkeypatch.setattr(
        nomad_cli, "register_model_card_tool", lambda server, locator: None
    )

    runner = CliRunner()
    result = runner.invoke(
        nomad_cli.app,
        ["serve", str(tmp_path / "nomad.yml"), "--transport", "stdio"],
    )

    assert result.exit_code == 0
    assert dummy_fm.resolve_calls == [dummy_config.context_dir]
    assert dummy_locator.register_calls == [("dummy-tool", resolved_model_dir)]


def test_nomad_cli_serve_continues_after_model_load_failure(
    monkeypatch, tmp_path: Path
):
    class DummyTool:
        def __init__(self, name: str):
            self.name = name

    class DummyModelConfig:
        def __init__(self, name: str):
            self.tool_name = name
            self.name_or_path = f"models/{name}"

        def resolve_source(self, *, base_dir: Path | None = None):
            return tmp_path / self.name_or_path

    class DummyManager:
        devices: list[str] = []

        def __init__(self):
            self.registered: list[tuple[str, Path]] = []
            self.added_to_fastmcp = False

        def register_tool(self, name, tool, *, source):
            self.registered.append((name, source))

        def add_to_fastmcp(self, server):
            self.added_to_fastmcp = True

    class DummyLocator:
        def __init__(self):
            self.registered: list[tuple[str, Path]] = []

        def register(self, tool_name, source):
            self.registered.append((tool_name, source))

    class DummyServer:
        ran = False

        def __init__(self, *args, **kwargs):
            pass

        def tool(self, **kwargs):
            def decorator(fn):
                return fn

            return decorator

        def run(self, *, transport, **kwargs):
            DummyServer.ran = True

    manager = DummyManager()
    locator = DummyLocator()
    bad_model = DummyModelConfig("bad")
    good_model = DummyModelConfig("good")

    def build_tool(fm):
        if fm is bad_model:
            raise RuntimeError("load failed")
        return DummyTool(fm.tool_name)

    dummy_config = types.SimpleNamespace(
        tools=[],
        tool_manager=types.SimpleNamespace(
            enabled=True,
            instantiate=lambda: manager,
        ),
        fmod_models=[bad_model, good_model],
        context_dir=tmp_path,
        build_tool=build_tool,
        search_tool=types.SimpleNamespace(expose=False),
    )

    monkeypatch.setattr(
        nomad_cli.ServerConfig,
        "from_file",
        classmethod(lambda cls, path: dummy_config),
    )
    monkeypatch.setattr(nomad_cli, "FastMCP", DummyServer)
    monkeypatch.setattr(nomad_cli, "ModelCardLocator", lambda: locator)
    monkeypatch.setattr(
        nomad_cli, "register_model_card_tool", lambda server, locator: None
    )

    result = CliRunner().invoke(
        nomad_cli.app,
        [
            "serve",
            str(tmp_path / "nomad.yml"),
            "--transport",
            "stdio",
            "--no-strict",
        ],
    )

    assert result.exit_code == 0
    assert DummyServer.ran is True
    assert manager.added_to_fastmcp is True
    assert manager.registered == [("good", tmp_path / "models/good")]
    assert locator.registered == [("good", tmp_path / "models/good")]
    assert "Failed to load torch model: `bad`" in result.stderr


def test_nomad_cli_serve_strict_model_load_failure_exits(monkeypatch, tmp_path: Path):
    class DummyModelConfig:
        tool_name = "bad"
        name_or_path = "models/bad"

    class DummyManager:
        devices: list[str] = []

        def add_to_fastmcp(self, server):
            raise AssertionError(
                "manager should not be registered after strict failure"
            )

    class DummyServer:
        ran = False

        def __init__(self, *args, **kwargs):
            pass

        def tool(self, **kwargs):
            def decorator(fn):
                return fn

            return decorator

        def run(self, *, transport, **kwargs):
            DummyServer.ran = True

    dummy_config = types.SimpleNamespace(
        tools=[],
        tool_manager=types.SimpleNamespace(
            enabled=True,
            instantiate=lambda: DummyManager(),
        ),
        fmod_models=[DummyModelConfig()],
        context_dir=tmp_path,
        build_tool=lambda fm: (_ for _ in ()).throw(RuntimeError("load failed")),
        search_tool=types.SimpleNamespace(expose=False),
    )

    monkeypatch.setattr(
        nomad_cli.ServerConfig,
        "from_file",
        classmethod(lambda cls, path: dummy_config),
    )
    monkeypatch.setattr(nomad_cli, "FastMCP", DummyServer)
    monkeypatch.setattr(nomad_cli, "ModelCardLocator", lambda: object())
    monkeypatch.setattr(
        nomad_cli, "register_model_card_tool", lambda server, locator: None
    )

    result = CliRunner().invoke(
        nomad_cli.app,
        ["serve", str(tmp_path / "nomad.yml"), "--transport", "stdio"],
    )

    assert result.exit_code == 1
    assert isinstance(result.exception, RuntimeError)
    assert DummyServer.ran is False
    assert "Failed to load torch model: `bad`" in result.stderr


def test_nomad_cli_serve_continues_after_model_registration_failure(
    monkeypatch, tmp_path: Path
):
    class DummyTool:
        def __init__(self, name: str):
            self.name = name

    class DummyModelConfig:
        def __init__(self, name: str):
            self.tool_name = name
            self.name_or_path = f"models/{name}"

        def resolve_source(self, *, base_dir: Path | None = None):
            return tmp_path / self.name_or_path

    class DummyManager:
        devices: list[str] = []

        def __init__(self):
            self.registered: list[tuple[str, Path]] = []
            self.added_to_fastmcp = False

        def register_tool(self, name, tool, *, source):
            if name == "bad":
                raise RuntimeError("registration failed")
            self.registered.append((name, source))

        def add_to_fastmcp(self, server):
            self.added_to_fastmcp = True

    class DummyLocator:
        def __init__(self):
            self.registered: list[tuple[str, Path]] = []

        def register(self, tool_name, source):
            self.registered.append((tool_name, source))

    class DummyServer:
        ran = False

        def __init__(self, *args, **kwargs):
            pass

        def tool(self, **kwargs):
            def decorator(fn):
                return fn

            return decorator

        def run(self, *, transport, **kwargs):
            DummyServer.ran = True

    manager = DummyManager()
    locator = DummyLocator()
    models = [DummyModelConfig("bad"), DummyModelConfig("good")]
    dummy_config = types.SimpleNamespace(
        tools=[],
        tool_manager=types.SimpleNamespace(
            enabled=True,
            instantiate=lambda: manager,
        ),
        fmod_models=models,
        context_dir=tmp_path,
        build_tool=lambda fm: DummyTool(fm.tool_name),
        search_tool=types.SimpleNamespace(expose=False),
    )

    monkeypatch.setattr(
        nomad_cli.ServerConfig,
        "from_file",
        classmethod(lambda cls, path: dummy_config),
    )
    monkeypatch.setattr(nomad_cli, "FastMCP", DummyServer)
    monkeypatch.setattr(nomad_cli, "ModelCardLocator", lambda: locator)
    monkeypatch.setattr(
        nomad_cli, "register_model_card_tool", lambda server, locator: None
    )

    result = CliRunner().invoke(
        nomad_cli.app,
        [
            "serve",
            str(tmp_path / "nomad.yml"),
            "--transport",
            "stdio",
            "--no-strict",
        ],
    )

    assert result.exit_code == 0
    assert DummyServer.ran is True
    assert manager.added_to_fastmcp is True
    assert manager.registered == [("good", tmp_path / "models/good")]
    assert locator.registered == [("good", tmp_path / "models/good")]
    assert "Failed to load torch model: `bad`" in result.stderr


def test_nomad_cli_serve_registers_search_tools_when_enabled(
    monkeypatch, tmp_path: Path
):
    class DummyServer:
        last_instance = None

        def __init__(self, *args, **kwargs):
            self.registered_tools: list[str] = []
            self.transport: str | None = None
            DummyServer.last_instance = self

        def add_tool(self, tool, **kwargs):
            self.registered_tools.append(
                kwargs.get("name")
                or getattr(tool, "name", None)
                or getattr(tool, "__name__", None)
            )

        def tool(self, **kwargs):
            def decorator(fn):
                self.add_tool(fn, **kwargs)
                return fn

            return decorator

        def run(self, *, transport, **kwargs):
            self.transport = transport

    dummy_config = types.SimpleNamespace(
        tools=[],
        tool_manager=types.SimpleNamespace(enabled=False),
        fmod_models=[],
        search_tool=types.SimpleNamespace(expose=True),
    )

    monkeypatch.setattr(
        nomad_cli.ServerConfig,
        "from_file",
        classmethod(lambda cls, path: dummy_config),
    )
    monkeypatch.setattr(nomad_cli, "FastMCP", DummyServer)
    monkeypatch.setattr(nomad_cli, "ModelCardLocator", lambda: object())
    monkeypatch.setattr(
        nomad_cli, "register_model_card_tool", lambda server, locator: None
    )

    runner = CliRunner()
    result = runner.invoke(
        nomad_cli.app,
        ["serve", str(tmp_path / "nomad.yml"), "--transport", "stdio"],
    )

    assert result.exit_code == 0
    assert DummyServer.last_instance is not None
    assert DummyServer.last_instance.registered_tools == ["search_tools"]


@pytest.mark.parametrize(
    "transport",
    ["http", "streamable-http", "streamable_http"],
)
def test_nomad_cli_serve_http_transport_aliases(
    monkeypatch,
    tmp_path: Path,
    transport: str,
):
    class DummyServer:
        last_instance = None

        def __init__(self, *args, **kwargs):
            self.transport: str | None = None
            self.kwargs: dict[str, Any] = {}
            DummyServer.last_instance = self

        def tool(self, **kwargs):
            def decorator(fn):
                return fn

            return decorator

        def run(self, *, transport, **kwargs):
            self.transport = transport
            self.kwargs = kwargs

    dummy_config = types.SimpleNamespace(
        tools=[],
        tool_manager=types.SimpleNamespace(enabled=False),
        fmod_models=[],
        search_tool=types.SimpleNamespace(expose=False),
    )

    monkeypatch.setattr(
        nomad_cli.ServerConfig,
        "from_file",
        classmethod(lambda cls, path: dummy_config),
    )
    monkeypatch.setattr(nomad_cli, "FastMCP", DummyServer)
    monkeypatch.setattr(nomad_cli, "ModelCardLocator", lambda: object())
    monkeypatch.setattr(
        nomad_cli, "register_model_card_tool", lambda server, locator: None
    )

    result = CliRunner().invoke(
        nomad_cli.app,
        [
            "serve",
            str(tmp_path / "nomad.yml"),
            "--transport",
            transport,
            "--host",
            "127.0.0.1",
            "--port",
            "9000",
        ],
    )

    assert result.exit_code == 0
    assert DummyServer.last_instance is not None
    assert DummyServer.last_instance.transport == "http"
    assert DummyServer.last_instance.kwargs["host"] == "127.0.0.1"
    assert DummyServer.last_instance.kwargs["port"] == 9000
    assert DummyServer.last_instance.kwargs["stateless_http"] is True


def test_nomad_cli_serve_reports_malformed_config_without_traceback(tmp_path: Path):
    config_path = tmp_path / "nomad.yml"
    config_path.write_text("fmod_models:\n  - [\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        nomad_cli.app,
        ["serve", str(config_path), "--transport", "stdio"],
    )

    assert result.exit_code == 1
    assert "Failed to load config" in result.output
    assert "YAML syntax error" in result.output
    assert "Traceback" not in result.output


def test_public_search_tools_returns_local_tool_matches():
    server = FastMCP("test")

    def add_numbers(a: int, b: int) -> int:
        """Add two numbers together."""
        return a + b

    server.add_tool(add_numbers)
    register_search_tool(
        server,
        search_config=SearchToolConfig(expose=True),
    )

    search_tool = asyncio.run(server.get_tool("search_tools"))
    assert search_tool is not None

    payload = asyncio.run(
        search_tool.fn(
            query="add",
            server_filter="nomad",
        )
    )

    assert isinstance(payload, list)
    assert payload
    assert payload[0]["name"] == "add_numbers"
    assert payload[0]["description"] == "Add two numbers together."
    assert "python_import_path" not in payload[0]
    assert "signature" not in payload[0]


def test_run_code_mode_script_rejects_both_config_types(tmp_path: Path):
    gateway_config = tmp_path / "gateway.yaml"
    gateway_config.write_text("servers: {}\n", encoding="utf-8")
    nomad_config = tmp_path / "nomad.yaml"
    nomad_config.write_text("tools: []\n", encoding="utf-8")
    script_path = tmp_path / "script.py"
    script_path.write_text("RESULT = None\n", encoding="utf-8")

    with pytest.raises(ValueError, match="mutually exclusive"):
        nomad_cli.run_code_mode_script(
            config_path=gateway_config,
            nomad_config_path=nomad_config,
            script_path=script_path,
            gateway_log_level="CRITICAL",
        )


def test_run_code_mode_script_disables_timeout(monkeypatch, tmp_path: Path):
    called: dict[str, Any] = {}

    config_path = tmp_path / "gateway.yaml"
    config_path.write_text("servers: {}\n", encoding="utf-8")
    script_path = tmp_path / "script.py"
    script_path.write_text("print('hi')\n", encoding="utf-8")

    class DummyGateway:
        def __init__(self, config):
            called["config"] = config

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def run_script(
            self,
            script_path: Path,
            env: dict[str, str] | None = None,
            *,
            script_args=(),
            timeout_seconds,
            capture_stdio,
        ) -> SandboxResult:
            called["script_path"] = script_path
            called["script_args"] = list(script_args)
            called["timeout_seconds"] = timeout_seconds
            called["capture_stdio"] = capture_stdio
            return SandboxResult(
                stdout="ignored",
                stderr="ignored",
                result=None,
                tool_calls=[],
                returncode=0,
                duration_seconds=0.25,
            )

    monkeypatch.setattr(nomad_cli, "CodeModeGateway", DummyGateway)

    result = nomad_cli.run_code_mode_script(
        config_path=config_path,
        script_path=script_path,
        gateway_log_level="INFO",
        script_args=["--alpha", "beta"],
    )

    assert called["script_args"] == ["--alpha", "beta"]
    assert called["timeout_seconds"] is None
    assert called["capture_stdio"] is False
    assert result == {
        "result": None,
        "tool_calls": [],
        "returncode": 0,
        "duration_seconds": 0.25,
    }


def test_run_code_mode_script_streams_stdio_to_terminal(tmp_path: Path, capfd):
    config_path = tmp_path / "gateway.yaml"
    config_path.write_text("servers: {}\n", encoding="utf-8")
    script_path = tmp_path / "script.py"
    script_path.write_text(
        "\n".join(
            [
                "import sys",
                "print('CODE_MODE_STDOUT_MARKER')",
                "print('CODE_MODE_STDERR_MARKER', file=sys.stderr)",
                "RESULT = {'ok': True}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    capfd.readouterr()
    result = nomad_cli.run_code_mode_script(
        config_path=config_path,
        script_path=script_path,
        gateway_log_level="CRITICAL",
    )
    captured = capfd.readouterr()

    assert "CODE_MODE_STDOUT_MARKER" in captured.out
    assert "CODE_MODE_STDERR_MARKER" in captured.err
    assert "stdout" not in result
    assert "stderr" not in result
    assert result["result"] == {"ok": True}


def test_nomad_cli_serve_appends_jsonl_logs(monkeypatch, tmp_path: Path):
    class DummyServer:
        def __init__(self, *args, **kwargs):
            self.transport: str | None = None

        def run(self, *, transport, **kwargs):
            self.transport = transport
            logging.getLogger("nomad.test").info("serve marker")

    dummy_config = types.SimpleNamespace(
        tools=[],
        tool_manager=types.SimpleNamespace(enabled=False),
        fmod_models=[],
        search_tool=types.SimpleNamespace(expose=False),
    )

    monkeypatch.setattr(
        nomad_cli.ServerConfig,
        "from_file",
        classmethod(lambda cls, path: dummy_config),
    )
    monkeypatch.setattr(nomad_cli, "FastMCP", DummyServer)
    monkeypatch.setattr(nomad_cli, "ModelCardLocator", lambda: object())
    monkeypatch.setattr(
        nomad_cli, "register_model_card_tool", lambda server, locator: None
    )

    log_file = tmp_path / "serve.jsonl"
    runner = CliRunner()
    for _ in range(2):
        result = runner.invoke(
            nomad_cli.app,
            [
                "serve",
                str(tmp_path / "nomad.yml"),
                "--transport",
                "stdio",
                "--no-tool-manager",
                "--log-file",
                str(log_file),
            ],
        )
        assert result.exit_code == 0

    payloads = [
        json.loads(line)
        for line in log_file.read_text(encoding="utf-8").strip().splitlines()
    ]
    payloads = [payload for payload in payloads if payload["logger"] == "nomad.test"]
    assert len(payloads) == 2

    first = payloads[0]
    second = payloads[1]
    assert first["logger"] == "nomad.test"
    assert second["logger"] == "nomad.test"
    assert first["message"] == "serve marker"
    assert second["message"] == "serve marker"


def test_nomad_cli_serve_uses_default_stderr_log_format(monkeypatch, tmp_path: Path):
    class DummyServer:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, *, transport, **kwargs):
            logging.getLogger("nomad.test").info("serve marker")

    dummy_config = types.SimpleNamespace(
        tools=[],
        tool_manager=types.SimpleNamespace(enabled=False),
        fmod_models=[],
        search_tool=types.SimpleNamespace(expose=False),
    )

    monkeypatch.setattr(
        nomad_cli.ServerConfig,
        "from_file",
        classmethod(lambda cls, path: dummy_config),
    )
    monkeypatch.setattr(nomad_cli, "FastMCP", DummyServer)
    monkeypatch.setattr(nomad_cli, "ModelCardLocator", lambda: object())
    monkeypatch.setattr(
        nomad_cli, "register_model_card_tool", lambda server, locator: None
    )

    result = CliRunner().invoke(
        nomad_cli.app,
        [
            "serve",
            str(tmp_path / "nomad.yml"),
            "--transport",
            "stdio",
        ],
    )

    assert result.exit_code == 0
    assert re.search(
        r"\d{4}-\d{2}-\d{2} .* \|\s+INFO\s+\| nomad\.test:\d+ - serve marker",
        result.stderr,
    )


def test_nomad_cli_serve_logs_visible_devices(monkeypatch, tmp_path: Path):
    class DummyServer:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, *, transport, **kwargs):
            pass

    class DummyManager:
        accelerator_info = [
            types.SimpleNamespace(device="cuda:0", name="NVIDIA TestGPU"),
            types.SimpleNamespace(device="cuda:1", name="NVIDIA TestGPU"),
        ]

        def add_to_fastmcp(self, server):
            pass

    dummy_config = types.SimpleNamespace(
        tools=[],
        tool_manager=types.SimpleNamespace(
            enabled=True,
            instantiate=lambda: DummyManager(),
        ),
        fmod_models=[],
        search_tool=types.SimpleNamespace(expose=False),
    )

    monkeypatch.setattr(
        nomad_cli.ServerConfig,
        "from_file",
        classmethod(lambda cls, path: dummy_config),
    )
    monkeypatch.setattr(nomad_cli, "FastMCP", DummyServer)
    monkeypatch.setattr(nomad_cli, "ModelCardLocator", lambda: object())
    monkeypatch.setattr(
        nomad_cli, "register_model_card_tool", lambda server, locator: None
    )

    log_file = tmp_path / "serve.jsonl"
    result = CliRunner().invoke(
        nomad_cli.app,
        [
            "serve",
            str(tmp_path / "nomad.yml"),
            "--transport",
            "stdio",
            "--log-file",
            str(log_file),
        ],
    )

    assert result.exit_code == 0
    payloads = [
        json.loads(line)
        for line in log_file.read_text(encoding="utf-8").strip().splitlines()
    ]
    messages = [
        payload["message"] for payload in payloads if payload["logger"] == "nomad.cli"
    ]
    assert (
        "Visible devices: cuda:0 (NVIDIA TestGPU), cuda:1 (NVIDIA TestGPU)" in messages
    )
