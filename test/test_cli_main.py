from __future__ import annotations

import asyncio
import json
import logging
import re
import types
from pathlib import Path
from typing import Any

import pytest
from fastmcp import FastMCP
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


@pytest.mark.asyncio
async def test_manager_lifespan_closes_manager():
    class DummyManager:
        closed = False

        async def aclose(self):
            self.closed = True

    manager = DummyManager()

    async with nomad_cli._manager_lifespan(manager)(object()):
        assert manager.closed is False

    assert manager.closed is True


def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text)


def empty_tool_manager_config():
    manager = types.SimpleNamespace(
        accelerator_info=[],
        add_to_fastmcp=lambda server: None,
    )
    return types.SimpleNamespace(instantiate=lambda: manager)


def test_nomad_cli_serve_has_no_tool_manager_toggle():
    result = CliRunner().invoke(nomad_cli.app, ["serve", "--help"])

    assert result.exit_code == 0
    assert "--tool-manager" not in strip_ansi(result.output)
    assert "--no-tool-manager" not in strip_ansi(result.output)


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

    def fake_run_code_mode_script(
        *, config_path, script_path, gateway_log_level, directory=None, script_args=()
    ):
        called["config_path"] = config_path
        called["script_path"] = script_path
        called["gateway_log_level"] = gateway_log_level
        called["script_args"] = list(script_args)
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

    def fake_run_code_mode_script(
        *, config_path, script_path, gateway_log_level, directory=None, script_args=()
    ):
        called["config_path"] = config_path
        called["script_path"] = script_path
        called["gateway_log_level"] = gateway_log_level
        called["script_args"] = list(script_args)
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


def test_nomad_cli_code_mode_exec_forwards_args_after_separator(
    monkeypatch, tmp_path: Path
):
    called: dict[str, Any] = {}

    def fake_run_code_mode_script(
        *, config_path, script_path, gateway_log_level, directory=None, script_args=()
    ):
        called["config_path"] = config_path
        called["script_path"] = script_path
        called["gateway_log_level"] = gateway_log_level
        called["script_args"] = list(script_args)
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
    def fake_run_code_mode_script(
        *, config_path, script_path, gateway_log_level, directory=None, script_args=()
    ):
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
    def fake_run_code_mode_script(
        *, config_path, script_path, gateway_log_level, directory=None, script_args=()
    ):
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
    "extra_args, expected_target, expected_oras_registry, expected_build_venvs",
    [
        ([], ExportTarget.DISK, None, True),
        (["--no-venv"], ExportTarget.DISK, None, False),
        (["--to", "https"], ExportTarget.HTTPS, None, True),
        (
            ["--to", "oras", "--oras-registry", "registry.example.com/scifm"],
            ExportTarget.ORAS,
            "registry.example.com/scifm",
            True,
        ),
    ],
)
def test_nomad_cli_export_accepts_targets(
    monkeypatch,
    tmp_path: Path,
    extra_args: list[str],
    expected_target: ExportTarget,
    expected_oras_registry: str | None,
    expected_build_venvs: bool,
):
    config_path = tmp_path / "nomad.yml"
    config_path.write_text("tool_manager: {}\ntools: []\nfmod_models: []\n")
    output_dir = tmp_path / "bundle"

    called: dict[str, Any] = {}

    def fake_export_models_config(
        config_path_arg,
        output_dir_arg,
        *,
        to,
        oras_registry,
        pin,
        build_venvs,
    ):
        called["config_path"] = config_path_arg
        called["output_dir"] = output_dir_arg
        called["to"] = to
        called["oras_registry"] = oras_registry
        called["pin"] = pin
        called["build_venvs"] = build_venvs
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
        "build_venvs": expected_build_venvs,
    }


def test_nomad_cli_export_reports_environment_build_failure(
    monkeypatch, tmp_path: Path
):
    config_path = tmp_path / "nomad.yml"
    config_path.write_text("fmod_models: []\n", encoding="utf-8")

    def fail_export(*args, **kwargs):
        raise RuntimeError("model environment installation failed")

    monkeypatch.setattr(nomad_export, "export_models_config", fail_export)

    result = CliRunner().invoke(
        nomad_cli.app,
        ["export", str(config_path), str(tmp_path / "bundle")],
    )

    assert result.exit_code == 1
    assert "model environment installation failed" in result.output
    assert "Traceback" not in result.output


def test_nomad_cli_export_report(monkeypatch, tmp_path: Path):
    config_path = tmp_path / "nomad.yml"
    config_path.write_text("tool_manager: {}\ntools: []\nfmod_models: []\n")
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
        batch_size = 1
        args_schema = DummyInput
        output_schema = DummyOutput

        def __call__(self, input: DummyInput) -> DummyOutput:
            return DummyOutput(value=input.value)

    class DummyManager:
        accelerator_info: list[Any] = []

        def register_remote_tool(self, remote):
            self.remote = remote

        def add_to_fastmcp(self, server):
            pass

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
    manager = DummyManager()
    dummy_config = types.SimpleNamespace(
        tools=[],
        tool_manager=types.SimpleNamespace(
            instantiate=lambda: manager,
        ),
        fmod_models=[dummy_fm],
        context_dir=tmp_path / "config-dir",
        build_remote=lambda fm: types.SimpleNamespace(
            name=DummyTool.name,
            description=DummyTool.description,
            batch_size=DummyTool.batch_size,
            source=str(fm.resolve_source(base_dir=dummy_config.context_dir)),
        ),
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
        accelerator_info: list[Any] = []

        def __init__(self):
            self.registered: list[tuple[str, Path]] = []
            self.added_to_fastmcp = False

        def register_remote_tool(self, remote):
            self.registered.append((remote.name, Path(remote.source)))

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

    def build_remote(fm):
        if fm is bad_model:
            raise RuntimeError("load failed")
        return types.SimpleNamespace(
            name=fm.tool_name,
            source=str(fm.resolve_source(base_dir=tmp_path)),
            batch_size=1,
        )

    dummy_config = types.SimpleNamespace(
        tools=[],
        tool_manager=types.SimpleNamespace(
            instantiate=lambda: manager,
        ),
        fmod_models=[bad_model, good_model],
        context_dir=tmp_path,
        build_remote=build_remote,
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
        accelerator_info: list[Any] = []

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
            instantiate=lambda: DummyManager(),
        ),
        fmod_models=[DummyModelConfig()],
        context_dir=tmp_path,
        build_remote=lambda fm: (_ for _ in ()).throw(RuntimeError("load failed")),
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
        accelerator_info: list[Any] = []

        def __init__(self):
            self.registered: list[tuple[str, Path]] = []
            self.added_to_fastmcp = False

        def register_remote_tool(self, remote):
            if remote.name == "bad":
                raise RuntimeError("registration failed")
            self.registered.append((remote.name, Path(remote.source)))

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
            instantiate=lambda: manager,
        ),
        fmod_models=models,
        context_dir=tmp_path,
        build_remote=lambda fm: types.SimpleNamespace(
            name=fm.tool_name,
            source=str(fm.resolve_source(base_dir=tmp_path)),
            batch_size=1,
        ),
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
        tool_manager=empty_tool_manager_config(),
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
        tool_manager=empty_tool_manager_config(),
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
        tool_manager=empty_tool_manager_config(),
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
        tool_manager=empty_tool_manager_config(),
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
