from __future__ import annotations

import os
import sys
import textwrap
import venv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from fastmcp import Client
from fastmcp.mcp_config import StdioMCPServer

from nomad.gateway.config import GatewayConfig, GatewayDefaults
from nomad.gateway.server import CodeModeGateway

DUMMY_SERVER = Path(__file__).parent / "tools" / "dummy_mcp_server.py"
DUMMY_CONFLICT_SERVER = Path(__file__).parent / "tools" / "dummy_conflict_server.py"


@dataclass(slots=True)
class GatewayHarness:
    client: Client
    gateway: CodeModeGateway


@dataclass(frozen=True, slots=True)
class ImportCase:
    module_name: str
    expected_result: str | None
    expected_error: str | None
    script_root: Path


def _is_subpath(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _build_config(tmp_path: Path) -> GatewayConfig:
    servers = {
        "dummy": StdioMCPServer(
            command=sys.executable,
            args=[str(DUMMY_SERVER.resolve())],
            env=os.environ.copy(),
        )
    }
    defaults = GatewayDefaults(
        timeout_seconds=10,
        stdout_limit=64,
        stderr_limit=64,
        result_limit=256,
        workspace_root=tmp_path / "runs",
    )
    return GatewayConfig(servers=servers, defaults=defaults, middleware=[])


def _build_conflict_config(tmp_path: Path) -> GatewayConfig:
    servers = {
        "conflict": StdioMCPServer(
            command=sys.executable,
            args=[str(DUMMY_CONFLICT_SERVER.resolve())],
            env=os.environ.copy(),
        )
    }
    defaults = GatewayDefaults(
        timeout_seconds=10,
        stdout_limit=64,
        stderr_limit=64,
        result_limit=256,
        workspace_root=tmp_path / "runs",
    )
    return GatewayConfig(servers=servers, defaults=defaults, middleware=[])


@pytest.mark.asyncio
async def test_gateway_startup_failure_cleans_up_resources(monkeypatch, tmp_path: Path):
    gateway = CodeModeGateway(_build_config(tmp_path))
    wrappers_root = gateway._sandbox.wrappers_root
    stop_called = False

    async def fail_start() -> None:
        raise RuntimeError("startup failed")

    async def track_stop() -> None:
        nonlocal stop_called
        stop_called = True

    monkeypatch.setattr(gateway._upstream, "start", fail_start)
    monkeypatch.setattr(gateway._upstream, "stop", track_stop)

    with pytest.raises(RuntimeError, match="startup failed"):
        await gateway.__aenter__()

    assert stop_called is True
    assert not wrappers_root.exists()


def _workspace_root(tmp_path: Path) -> Path:
    return tmp_path / "runs"


def _site_packages_dir(venv_root: Path) -> Path:
    if os.name == "nt":
        return venv_root / "Lib" / "site-packages"

    candidates = sorted((venv_root / "lib").glob("python*/site-packages"))
    assert candidates, f"Could not find site-packages under {venv_root}"
    return candidates[0]


def _site_packages_stub_dir(root: Path) -> Path:
    if os.name == "nt":
        return root / "Lib" / "site-packages"
    return (
        root
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )


def _create_workspace_virtualenv(workspace_root: Path) -> Path:
    venv_root = workspace_root / ".venv"
    venv.EnvBuilder(with_pip=False, symlinks=os.name != "nt").create(venv_root)
    site_packages = _site_packages_dir(venv_root)
    site_packages.mkdir(parents=True, exist_ok=True)
    return site_packages


def _import_probe_code(module_name: str) -> str:
    return textwrap.dedent(
        f"""
        try:
            import {module_name} as _module
        except ModuleNotFoundError as exc:
            RESULT = {{"error": str(exc)}}
        else:
            RESULT = {{"value": _module.VALUE}}
        """
    )


@pytest_asyncio.fixture
async def gateway(tmp_path: Path):
    config = _build_config(tmp_path)
    async with CodeModeGateway(config) as gateway_obj:
        async with Client(gateway_obj.fastmcp) as client:
            await client.initialize()
            yield GatewayHarness(client=client, gateway=gateway_obj)


@pytest_asyncio.fixture
async def gateway_with_workspace_venv(tmp_path: Path):
    workspace_root = _workspace_root(tmp_path)
    workspace_root.mkdir(parents=True, exist_ok=True)
    _create_workspace_virtualenv(workspace_root)

    config = _build_config(tmp_path)
    async with CodeModeGateway(config) as gateway_obj:
        async with Client(gateway_obj.fastmcp) as client:
            await client.initialize()
            yield GatewayHarness(client=client, gateway=gateway_obj)


@pytest.fixture(
    params=["execute_mcp_code", "execute_mcp_script"],
    ids=["execute_mcp_code", "execute_mcp_script"],
)
def sandbox_execution_mode(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(
    params=["workspace_module", "external_module"],
    ids=["workspace-module", "external-module"],
)
def workspace_import_case(
    request: pytest.FixtureRequest,
    gateway: GatewayHarness,
    tmp_path: Path,
) -> ImportCase:
    workspace_root = gateway.gateway.config.defaults.workspace_root
    assert workspace_root is not None

    script_root = tmp_path / request.param
    script_root.mkdir(parents=True, exist_ok=True)

    if request.param == "workspace_module":
        module_name = "workspace_only_module"
        (workspace_root / f"{module_name}.py").write_text(
            "VALUE = 'workspace'\n",
            encoding="utf-8",
        )
        return ImportCase(
            module_name=module_name,
            expected_result="workspace",
            expected_error=None,
            script_root=script_root,
        )

    module_name = "external_only_module"
    (script_root / f"{module_name}.py").write_text(
        "VALUE = 'outside-workspace'\n",
        encoding="utf-8",
    )
    return ImportCase(
        module_name=module_name,
        expected_result=None,
        expected_error=f"No module named '{module_name}'",
        script_root=script_root,
    )


@pytest.fixture(
    params=["workspace_venv_package", "external_venv_package"],
    ids=["workspace-.venv-package", "external-.venv-package"],
)
def workspace_venv_import_case(
    request: pytest.FixtureRequest,
    gateway_with_workspace_venv: GatewayHarness,
    tmp_path: Path,
) -> ImportCase:
    workspace_root = gateway_with_workspace_venv.gateway.config.defaults.workspace_root
    assert workspace_root is not None

    script_root = tmp_path / request.param
    script_root.mkdir(parents=True, exist_ok=True)

    if request.param == "workspace_venv_package":
        module_name = "workspace_venv_pkg"
        package_root = _site_packages_dir(workspace_root / ".venv") / module_name
        package_root.mkdir(parents=True, exist_ok=True)
        (package_root / "__init__.py").write_text(
            "VALUE = 'workspace-venv'\n",
            encoding="utf-8",
        )
        return ImportCase(
            module_name=module_name,
            expected_result="workspace-venv",
            expected_error=None,
            script_root=script_root,
        )

    module_name = "external_venv_pkg"
    package_root = _site_packages_stub_dir(tmp_path / "external_venv") / module_name
    package_root.mkdir(parents=True, exist_ok=True)
    (package_root / "__init__.py").write_text(
        "VALUE = 'outside-workspace-venv'\n",
        encoding="utf-8",
    )
    return ImportCase(
        module_name=module_name,
        expected_result=None,
        expected_error=f"No module named '{module_name}'",
        script_root=script_root,
    )


async def _execute_source(
    gateway: GatewayHarness,
    execution_mode: str,
    source: str,
    script_root: Path,
) -> dict[str, Any]:
    result = await _run_source(gateway, execution_mode, source, script_root)
    return {"result": result.result}


async def _run_source(
    gateway: GatewayHarness,
    execution_mode: str,
    source: str,
    script_root: Path,
    *,
    timeout_seconds: float | None = None,
):
    timeout_override = (
        {} if timeout_seconds is None else {"timeout_seconds": timeout_seconds}
    )
    if execution_mode == "execute_mcp_code":
        return await gateway.gateway.run_code(source, **timeout_override)

    script_path = script_root / "script.py"
    script_path.write_text(source, encoding="utf-8")
    return await gateway.gateway.run_script(script_path, **timeout_override)


@pytest.mark.asyncio
async def test_search_tools_returns_expected_results(gateway: GatewayHarness):
    response = await gateway.client.call_tool("search_code_tools", {"query": "add"})
    items: list[dict[str, Any]] = response.structured_content["result"]
    assert items
    assert any(item["python_import_path"] == "mcp_tools.dummy.add" for item in items)
    assert all(
        set(item) == {"python_import_path", "description", "signature"}
        for item in items
    )


@pytest.mark.asyncio
async def test_search_tools_full_returns_all_fields(gateway: GatewayHarness):
    response = await gateway.client.call_tool(
        "search_code_tools",
        {"query": "add", "detail_level": "full"},
    )
    items: list[dict[str, Any]] = response.structured_content["result"]
    assert items
    add_tool = next(
        item for item in items if item["python_import_path"] == "mcp_tools.dummy.add"
    )
    assert add_tool["python_import_path"] == "mcp_tools.dummy.add"
    assert add_tool["description"] == "Add two numbers"
    assert isinstance(add_tool["inputSchema"], dict)
    assert isinstance(add_tool["outputSchema"], dict)


@pytest.mark.asyncio
async def test_search_tools_matches_argument_keywords(gateway: GatewayHarness):
    response = await gateway.client.call_tool(
        "search_code_tools",
        {"query": "excited"},
    )
    items: list[dict[str, Any]] = response.structured_content["result"]
    assert [item["python_import_path"] for item in items] == ["mcp_tools.dummy.greet"]
    assert items[0]["signature"] == "(*, name: str, excited: bool | None = None) -> str"


@pytest.mark.asyncio
async def test_search_tools_limit_caps_results(gateway: GatewayHarness):
    response = await gateway.client.call_tool(
        "search_code_tools",
        {"query": "", "limit": 1},
    )
    items: list[dict[str, Any]] = response.structured_content["result"]
    assert len(items) == 1


@pytest.mark.asyncio
async def test_search_tools_limit_none_returns_all_results(gateway: GatewayHarness):
    response = await gateway.client.call_tool(
        "search_code_tools",
        {"query": "", "limit": None},
    )
    items: list[dict[str, Any]] = response.structured_content["result"]
    assert {item["python_import_path"] for item in items} == {
        "mcp_tools.dummy.add",
        "mcp_tools.dummy.boom",
        "mcp_tools.dummy.echo_object",
        "mcp_tools.dummy.echo_payload",
        "mcp_tools.dummy.make_well_format",
        "mcp_tools.dummy.multiply",
        "mcp_tools.dummy.greet",
    }


@pytest.mark.asyncio
async def test_execute_code_invokes_upstream_tool(gateway: GatewayHarness):
    code = """
from mcp_tools.dummy import add

first = add(a=1, b=2)
second = add(a=3, b=4)
RESULT = {"first": first, "second": second}
"""
    payload = await gateway.client.call_tool("execute_mcp_code", {"code": code})
    data = payload.data
    assert data["result"]["first"] == 3
    assert data["result"]["second"] == 7
    assert data["duration_seconds"] >= 0
    assert "tool_calls" not in data

    wrappers_root = gateway.gateway._sandbox.wrappers_root
    assert (wrappers_root / "nomad" / "gateway" / "runtime" / "__init__.py").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code",
    [
        (
            "from nomad.gateway.runtime import BridgeError, ToolCallError\n"
            "from mcp_tools.dummy import boom\n"
            "\n"
            "try:\n"
            "    boom()\n"
            "except Exception as exc:\n"
            "    RESULT = {\n"
            "        'type': type(exc).__name__,\n"
            "        'is_tool_call_error': isinstance(exc, ToolCallError),\n"
            "        'is_bridge_error': isinstance(exc, BridgeError),\n"
            "        'message': str(exc),\n"
            "    }\n"
        ),
        (
            "import asyncio\n"
            "from nomad.gateway.runtime import BridgeError, ToolCallError\n"
            "from mcp_tools.dummy import boom_async\n"
            "\n"
            "async def main():\n"
            "    try:\n"
            "        await boom_async()\n"
            "    except Exception as exc:\n"
            "        return {\n"
            "            'type': type(exc).__name__,\n"
            "            'is_tool_call_error': isinstance(exc, ToolCallError),\n"
            "            'is_bridge_error': isinstance(exc, BridgeError),\n"
            "            'message': str(exc),\n"
            "        }\n"
            "\n"
            "RESULT = asyncio.run(main())\n"
        ),
    ],
    ids=["sync-wrapper", "async-wrapper"],
)
async def test_tool_wrappers_raise_on_upstream_error(
    gateway: GatewayHarness,
    code: str,
):
    result = await gateway.gateway.run_code(code)

    assert result.returncode == 0
    assert result.result["type"] == "ToolCallError"
    assert result.result["is_tool_call_error"] is True
    assert result.result["is_bridge_error"] is False
    assert "Error calling tool 'boom'" in result.result["message"]
    assert "boom from tool" in result.result["message"]


@pytest.mark.asyncio
async def test_execute_code_response_includes_captured_stdio(gateway: GatewayHarness):
    payload = await gateway.client.call_tool(
        "execute_mcp_code",
        {
            "code": (
                "import sys\n"
                "print('hello from stdout')\n"
                "print('hello from stderr', file=sys.stderr)\n"
                "RESULT = {'status': 'ok'}\n"
            )
        },
    )
    data = payload.data
    assert "hello from stdout" in data["stdout"]
    assert "hello from stderr" in data["stderr"]
    assert data["result"] == {"status": "ok"}
    assert data["duration_seconds"] >= 0


@pytest.mark.asyncio
async def test_execute_code_handles_large_bridge_payloads(gateway: GatewayHarness):
    payload = await gateway.client.call_tool(
        "execute_mcp_code",
        {
            "code": (
                "from mcp_tools.dummy import greet\n"
                "name = 'x' * 70000\n"
                "message = greet(name=name)\n"
                "RESULT = {'message_length': len(message)}\n"
            )
        },
    )
    data = payload.data
    assert data["result"] == {"message_length": 70008}
    assert data["duration_seconds"] >= 0


@pytest.mark.asyncio
async def test_execute_code_large_result_returns_explicit_error(
    gateway: GatewayHarness,
):
    payload = await gateway.client.call_tool(
        "execute_mcp_code",
        {"code": "RESULT = 'x' * 2048\n"},
    )
    data = payload.data
    assert data["result"] == {
        "error": (
            "RESULT is too large to return. "
            "Keep RESULT under about 64 B and write larger outputs to files "
            "instead."
        )
    }
    assert data["duration_seconds"] >= 0


@pytest.mark.asyncio
async def test_persistent_workspace_reused(gateway: GatewayHarness):
    sandbox = gateway.gateway._sandbox
    workspace_root = gateway.gateway.config.defaults.workspace_root
    assert workspace_root is not None

    payload = await gateway.client.call_tool(
        "execute_mcp_code",
        {"code": "import os\nRESULT = os.getcwd()"},
    )
    first_cwd = Path(payload.data["result"]).resolve()
    assert first_cwd == workspace_root.resolve()

    payload = await gateway.client.call_tool(
        "execute_mcp_code",
        {"code": "import os\nRESULT = os.getcwd()"},
    )
    second_cwd = Path(payload.data["result"]).resolve()
    assert second_cwd == workspace_root.resolve()

    wrappers_root = sandbox.wrappers_root
    assert wrappers_root.exists()
    assert not _is_subpath(wrappers_root, workspace_root)


@pytest.mark.asyncio
async def test_gateways_use_unique_wrapper_roots(tmp_path: Path):
    config = _build_config(tmp_path)
    gateway_one = CodeModeGateway(config)
    gateway_two = CodeModeGateway(config)

    try:
        root_one = gateway_one._sandbox.wrappers_root
        root_two = gateway_two._sandbox.wrappers_root

        assert root_one != root_two
        assert (root_one / "nomad" / "gateway" / "runtime" / "__init__.py").exists()
        assert (root_two / "nomad" / "gateway" / "runtime" / "__init__.py").exists()
    finally:
        await gateway_one._sandbox.aclose()
        await gateway_two._sandbox.aclose()


@pytest.mark.asyncio
@pytest.mark.skipif(os.name == "nt", reason="POSIX-only interpreter stub")
async def test_uses_workspace_virtualenv_for_runtime(tmp_path: Path):
    workspace_root = tmp_path / "runs"
    workspace_root.mkdir(parents=True, exist_ok=True)

    venv_root = workspace_root / ".venv"
    python_path = venv_root / "bin" / "python"
    exec_path = str(Path(sys.executable))
    marker_value = str(python_path)
    python_path.parent.mkdir(parents=True, exist_ok=True)
    python_path.write_text(
        textwrap.dedent(
            f"""#!{exec_path}
import os as _os
import sys as _sys
_os.environ["NOMAD_GATEWAY_STUB"] = {marker_value!r}
_os.execv({exec_path!r}, [{exec_path!r}, *_sys.argv[1:]])
"""
        ),
        encoding="utf-8",
    )
    python_path.chmod(0o755)
    (venv_root / "pyvenv.cfg").write_text("home = stub\n", encoding="utf-8")

    config = _build_config(tmp_path)

    async with CodeModeGateway(config) as gateway_obj:
        async with Client(gateway_obj.fastmcp) as client:
            await client.initialize()
            payload = await client.call_tool(
                "execute_mcp_code",
                {
                    "code": (
                        "import os, sys\n"
                        "RESULT = {\n"
                        "    'exe': sys.executable,\n"
                        "    'marker': os.environ.get('NOMAD_GATEWAY_STUB'),\n"
                        "}\n"
                    )
                },
                raise_on_error=False,
            )
            assert not payload.is_error, getattr(payload, "text", payload.content)
            result = payload.data["result"]
            assert result["marker"] == marker_value
            resolved_exec = Path(result["exe"]).resolve()
            assert resolved_exec == Path(sys.executable).resolve()


@pytest.mark.asyncio
async def test_execute_script_reads_and_runs_file(
    gateway: GatewayHarness, tmp_path: Path
):
    script_path = tmp_path / "script.py"
    script_path.write_text(
        "\n".join(
            [
                "from mcp_tools.dummy import add",
                "first = add(a=1, b=2)",
                "second = add(a=3, b=4)",
                "RESULT = {'first': first, 'second': second}",
            ]
        ),
        encoding="utf-8",
    )

    payload = await gateway.client.call_tool(
        "execute_mcp_script", {"script_path": str(script_path)}
    )
    data = payload.data
    assert data["result"]["first"] == 3
    assert data["result"]["second"] == 7
    assert "tool_calls" not in data


@pytest.mark.asyncio
async def test_run_script_resolves_relative_path_from_workspace(
    gateway: GatewayHarness,
):
    workspace_root = gateway.gateway.config.defaults.workspace_root
    assert workspace_root is not None
    script_path = workspace_root / "relative.py"
    script_path.write_text("RESULT = __file__\n", encoding="utf-8")

    result = await gateway.gateway.run_script(Path("relative.py"))

    assert Path(result.result).absolute() == script_path.absolute()


@pytest.mark.asyncio
async def test_run_script_expands_user_path(
    gateway: GatewayHarness,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    script_path = tmp_path / "home_script.py"
    script_path.write_text("RESULT = __file__\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    result = await gateway.gateway.run_script(Path("~/home_script.py"))

    assert Path(result.result).absolute() == script_path.absolute()


@pytest.mark.asyncio
async def test_run_script_resolves_relative_path_from_cwd_without_workspace(
    gateway: GatewayHarness,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    script_path = tmp_path / "relative.py"
    script_path.write_text("RESULT = __file__\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    gateway.gateway.config.defaults.workspace_root = None

    result = await gateway.gateway.run_script(Path("relative.py"))

    assert Path(result.result).absolute() == script_path.absolute()


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_name", ["missing.py", "directory"])
async def test_run_script_rejects_paths_that_are_not_files(
    gateway: GatewayHarness,
    invalid_name: str,
):
    workspace_root = gateway.gateway.config.defaults.workspace_root
    assert workspace_root is not None
    (workspace_root / "directory").mkdir(exist_ok=True)

    with pytest.raises(FileNotFoundError, match="Script file not found"):
        await gateway.gateway.run_script(Path(invalid_name))


@pytest.mark.asyncio
async def test_execute_script_reports_missing_file_to_client(
    gateway: GatewayHarness,
):
    payload = await gateway.client.call_tool(
        "execute_mcp_script",
        {"script_path": "missing.py"},
        raise_on_error=False,
    )

    assert payload.is_error
    assert payload.content
    assert "Script file not found" in payload.content[0].text


@pytest.mark.asyncio
async def test_execute_script_preserves_file_context_for_package_scripts(
    gateway: GatewayHarness, tmp_path: Path
):
    package_root = tmp_path / "script_pkg"
    package_root.mkdir()
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    (package_root / "helper.py").write_text("VALUE = 42\n", encoding="utf-8")
    script_path = package_root / "main.py"
    script_path.write_text(
        textwrap.dedent(
            """
            try:
                from .helper import VALUE
            except ImportError as exc:
                RESULT = {
                    "error": str(exc),
                    "package": globals().get("__package__"),
                    "file": globals().get("__file__"),
                }
            else:
                RESULT = {"value": VALUE}
            """
        ),
        encoding="utf-8",
    )

    payload = await gateway.client.call_tool(
        "execute_mcp_script", {"script_path": str(script_path)}
    )
    data = payload.data
    assert (
        data["result"]["error"]
        == "attempted relative import with no known parent package"
    )
    assert data["result"]["package"] == ""
    assert Path(data["result"]["file"]).resolve() == script_path.resolve()


@pytest.mark.asyncio
async def test_pydantic_model_arguments_are_serialized_for_object_inputs(
    gateway: GatewayHarness,
    sandbox_execution_mode: str,
    tmp_path: Path,
):
    source = textwrap.dedent(
        """
        from pydantic import BaseModel
        from mcp_tools.dummy import echo_object

        class Sample(BaseModel):
            value: int

        RESULT = echo_object(a=Sample(value=42))
        """
    )

    script_root = tmp_path / sandbox_execution_mode
    script_root.mkdir(parents=True, exist_ok=True)
    result = await _run_source(
        gateway,
        sandbox_execution_mode,
        source,
        script_root,
    )

    assert result.returncode == 0
    assert result.result == {"value": 42}
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["tool"] == "echo_object"
    assert result.tool_calls[0]["input"] == {"a": {"value": 42}}


@pytest.mark.asyncio
async def test_pydantic_model_arguments_use_json_encoded_fields(
    gateway: GatewayHarness,
    sandbox_execution_mode: str,
    tmp_path: Path,
):
    source = textwrap.dedent(
        """
        from datetime import date, timedelta
        from enum import Enum

        from pydantic import BaseModel
        from mcp_tools.dummy import echo_payload

        class Color(str, Enum):
            RED = "red"

        class Sample(BaseModel):
            color: Color
            day: date
            delta: timedelta

        RESULT = echo_payload(
            a=Sample(
                color=Color.RED,
                day=date(2024, 1, 2),
                delta=timedelta(seconds=90),
            )
        )
        """
    )

    script_root = tmp_path / f"{sandbox_execution_mode}_json_fields"
    script_root.mkdir(parents=True, exist_ok=True)
    result = await _run_source(
        gateway,
        sandbox_execution_mode,
        source,
        script_root,
    )

    assert result.returncode == 0
    assert result.result == {
        "color": "red",
        "day": "2024-01-02",
        "delta": "PT1M30S",
    }
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["tool"] == "echo_payload"
    assert result.tool_calls[0]["input"] == {
        "a": {
            "color": "red",
            "day": "2024-01-02",
            "delta": "PT1M30S",
        }
    }


@pytest.mark.asyncio
async def test_code_mode_deserializes_well_format_tensor_outputs(
    gateway: GatewayHarness,
    sandbox_execution_mode: str,
    tmp_path: Path,
):
    source = textwrap.dedent(
        """
        import torch

        from mcp_tools.dummy import make_well_format

        out = make_well_format()
        field = out["t0_fields"]["temperature"]
        mask = out["boundary_conditions"]["wall"]["mask"]
        values = out["boundary_conditions"]["wall"]["values"]

        RESULT = {
            "field_is_tensor": isinstance(field, torch.Tensor),
            "mask_is_tensor": isinstance(mask, torch.Tensor),
            "values_is_tensor": isinstance(values, torch.Tensor),
            "field": field.tolist(),
            "mask": mask.tolist(),
            "values": values.tolist(),
        }
        """
    )

    script_root = tmp_path / f"{sandbox_execution_mode}_well_format"
    script_root.mkdir(parents=True, exist_ok=True)
    result = await _run_source(
        gateway,
        sandbox_execution_mode,
        source,
        script_root,
        timeout_seconds=45,
    )

    assert result.returncode == 0
    assert result.result == {
        "field_is_tensor": True,
        "mask_is_tensor": True,
        "values_is_tensor": True,
        "field": [1.0, 2.0],
        "mask": [True, False],
        "values": [1.0],
    }


@pytest.mark.asyncio
async def test_workspace_imports_are_scoped_to_workspace(
    gateway: GatewayHarness,
    sandbox_execution_mode: str,
    workspace_import_case: ImportCase,
):
    source = _import_probe_code(workspace_import_case.module_name)
    payload = await _execute_source(
        gateway,
        sandbox_execution_mode,
        source,
        workspace_import_case.script_root,
    )
    if workspace_import_case.expected_error is None:
        assert payload["result"] == {"value": workspace_import_case.expected_result}
    else:
        assert payload["result"] == {"error": workspace_import_case.expected_error}


@pytest.mark.asyncio
async def test_workspace_virtualenv_imports_are_scoped_to_workspace(
    gateway_with_workspace_venv: GatewayHarness,
    sandbox_execution_mode: str,
    workspace_venv_import_case: ImportCase,
):
    source = _import_probe_code(workspace_venv_import_case.module_name)
    payload = await _execute_source(
        gateway_with_workspace_venv,
        sandbox_execution_mode,
        source,
        workspace_venv_import_case.script_root,
    )
    if workspace_venv_import_case.expected_error is None:
        assert payload["result"] == {
            "value": workspace_venv_import_case.expected_result
        }
    else:
        assert payload["result"] == {"error": workspace_venv_import_case.expected_error}


@pytest.mark.asyncio
async def test_search_tools_raises_on_sanitized_name_conflict(tmp_path: Path):
    async with CodeModeGateway(_build_conflict_config(tmp_path)) as gateway_obj:
        with pytest.raises(
            ValueError,
            match=r"both sanitize to 'foo_bar'",
        ):
            await gateway_obj.search_code_tools(
                query="",
                server_filter="conflict",
            )


@pytest.mark.asyncio
async def test_wrapper_generation_raises_on_sanitized_name_conflict(tmp_path: Path):
    async with CodeModeGateway(_build_conflict_config(tmp_path)) as gateway_obj:
        result = await gateway_obj.run_code(
            "import mcp_tools.conflict\nRESULT = 'unreachable'\n"
        )

        assert result.returncode != 0
        assert "both sanitize to 'foo_bar'" in result.result["error"]
