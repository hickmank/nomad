from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import sysconfig
import textwrap
from collections.abc import Sequence
from datetime import date, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from nomad.gateway.runtime import runner
from nomad.gateway.runtime.runner import _default_serializer


class _SampleModel(BaseModel):
    value: int


class _Color(str, Enum):
    RED = "red"


class _JsonEncodedModel(BaseModel):
    color: _Color
    day: date
    delta: timedelta


class _HasModelDump:
    def model_dump(self) -> dict[str, Any]:
        return {"foo": "bar"}


class _BrokenModelDump:
    def model_dump(self) -> dict[str, Any]:
        raise RuntimeError("boom")


def _load_tensor_test_deps():
    torch = pytest.importorskip("torch")
    pytest.importorskip("zstandard")
    from nomad.well_format import Tensor

    return torch, Tensor


def test_default_serializer_handles_pydantic_model():
    sample = _SampleModel(value=42)
    assert _default_serializer(sample) == {"value": 42}


def test_default_serializer_handles_model_dump_duck_type():
    obj = _HasModelDump()
    assert _default_serializer(obj) == {"foo": "bar"}


def test_default_serializer_uses_pydantic_json_mode_for_fields():
    payload = _JsonEncodedModel(
        color=_Color.RED,
        day=date(2024, 1, 2),
        delta=timedelta(seconds=90),
    )
    assert _default_serializer(payload) == {
        "color": "red",
        "day": "2024-01-02",
        "delta": "PT1M30S",
    }


def test_default_serializer_handles_nested_lists_with_models_and_scalars():
    payload = [1, "two", None, _SampleModel(value=3), {"sample": _SampleModel(value=4)}]
    assert _default_serializer(payload) == [
        1,
        "two",
        None,
        {"value": 3},
        {"sample": {"value": 4}},
    ]


def test_default_serializer_converts_sets_to_lists():
    output = _default_serializer({1, 2, 3})
    assert isinstance(output, list)
    assert set(output) == {1, 2, 3}


def test_default_serializer_fallback_is_string():
    class _NoSpecial:
        pass

    result = _default_serializer(_NoSpecial())
    assert isinstance(result, str)


def test_default_serializer_falls_back_when_model_dump_raises():
    result = _default_serializer(_BrokenModelDump())
    assert isinstance(result, str)


def test_default_serializer_uses_nomad_tensor_wire_format():
    torch, Tensor = _load_tensor_test_deps()

    class TensorEnvelope(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        payload: Tensor

    tensor = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    serialized = _default_serializer(tensor)

    assert isinstance(serialized, str)
    decoded = TensorEnvelope.model_validate({"payload": serialized})
    assert torch.equal(decoded.payload, tensor)


def test_default_serializer_preserves_model_shape_when_json_mode_fails():
    torch, Tensor = _load_tensor_test_deps()

    class TensorPayload(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        payload: Any

    class TensorEnvelope(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        payload: Tensor

    tensor = torch.tensor([1.0, 2.0], dtype=torch.float32)
    serialized = _default_serializer(TensorPayload(payload=tensor))

    assert isinstance(serialized, dict)
    assert isinstance(serialized["payload"], str)
    decoded = TensorEnvelope.model_validate(serialized)
    assert torch.equal(decoded.payload, tensor)


def _prepare_env(monkeypatch: pytest.MonkeyPatch, script_path, result_path, wrappers):
    monkeypatch.setenv("NOMAD_MCP_SCRIPT_PATH", str(script_path))
    monkeypatch.setenv("NOMAD_MCP_RESULT_PATH", str(result_path))
    monkeypatch.setenv("NOMAD_MCP_WRAPPERS_ROOT", str(wrappers))


def _write_import_trace_helper(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            """
            import atexit
            import json
            import os
            import sys
            from pathlib import Path

            _BASELINE = set(sys.modules)
            _TRACE_PATH = Path(os.environ["NOMAD_IMPORT_TRACE_PATH"])


            def _raw_module_origin(module):
                spec = getattr(module, "__spec__", None)
                origin = getattr(spec, "origin", None)
                if origin is None:
                    origin = getattr(module, "__file__", None)
                return origin


            def _module_origin(module):
                origin = _raw_module_origin(module)
                if origin is not None:
                    return origin

                # Some runtime-visible modules are synthetic aliases that do not
                # carry their own spec or __file__ (for example typing.io on
                # Python 3.12). Fall back to the nearest loaded ancestor so we
                # classify them by the package that exposed them.
                name = getattr(module, "__name__", "")
                while "." in name:
                    name = name.rpartition(".")[0]
                    parent = sys.modules.get(name)
                    if parent is None:
                        continue
                    origin = _raw_module_origin(parent)
                    if origin is not None:
                        return origin

                return None


            def _dump_imports():
                payload = []
                for name in sorted(set(sys.modules) - _BASELINE):
                    module = sys.modules.get(name)
                    if module is None:
                        continue
                    payload.append({"name": name, "origin": _module_origin(module)})
                _TRACE_PATH.write_text(json.dumps(payload), encoding="utf-8")


            atexit.register(_dump_imports)
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )


def _write_wrapper_root_package(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    path.joinpath("__init__.py").write_text(
        textwrap.dedent(
            '''
            """Auto-generated MCP tool wrappers."""

            from nomad.gateway.runtime import install_wrapper_importer as _install_importer

            _install_importer()

            __all__ = []
            '''
        ).strip()
        + "\n",
        encoding="utf-8",
    )


def _is_path_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _is_stdlib_origin(origin: str | None, stdlib_roots: tuple[Path, ...]) -> bool:
    if origin in {"built-in", "frozen"}:
        return True
    if not isinstance(origin, str) or origin.startswith("<"):
        return False

    path = Path(origin).resolve()
    return any(_is_path_under(path, root) for root in stdlib_roots)


def _stdlib_roots() -> tuple[Path, ...]:
    return tuple(
        Path(path).resolve()
        for path in {
            sysconfig.get_path("stdlib"),
            sysconfig.get_path("platstdlib"),
        }
        if path
    )


def _find_forbidden_imports(
    imported_modules: list[dict[str, str | None]],
    *,
    allowed_roots: Sequence[Path] = (),
    allowed_files: Sequence[Path] = (),
    allowed_prefixes: Sequence[str] = (),
) -> list[dict[str, str | None]]:
    stdlib_roots = _stdlib_roots()
    resolved_roots = tuple(path.resolve() for path in allowed_roots)
    resolved_files = {path.resolve() for path in allowed_files}
    forbidden: list[dict[str, str | None]] = []

    for entry in imported_modules:
        name = entry["name"]
        origin = entry["origin"]

        if any(
            name == prefix or name.startswith(f"{prefix}.")
            for prefix in allowed_prefixes
        ):
            continue
        if _is_stdlib_origin(origin, stdlib_roots):
            continue
        if not isinstance(origin, str) or origin.startswith("<"):
            forbidden.append(entry)
            continue

        path = Path(origin).resolve()
        if path in resolved_files:
            continue
        if any(_is_path_under(path, root) for root in resolved_roots):
            continue

        forbidden.append(entry)

    return forbidden


def _prepare_wrappers_root(tmp_path: Path) -> tuple[Path, Path, Path]:
    source_nomad_root = Path(runner.__file__).resolve().parents[2]
    wrappers_root = tmp_path / "wrappers"
    shutil.copytree(source_nomad_root, wrappers_root / "nomad")

    helper_root = tmp_path / "helper"
    helper_root.mkdir()
    _write_import_trace_helper(helper_root / "sitecustomize.py")

    wrappers_package_root = wrappers_root / "mcp_tools"
    _write_wrapper_root_package(wrappers_package_root)
    return wrappers_root, helper_root, wrappers_package_root


def _run_traced_python_script(
    *,
    tmp_path: Path,
    wrappers_root: Path,
    helper_root: Path,
    source: str,
    extra_env: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    script_path = tmp_path / "script.py"
    trace_path = tmp_path / "import-trace.json"
    workspace_root = tmp_path / "workspace"

    workspace_root.mkdir()
    script_path.write_text(source, encoding="utf-8")

    env = os.environ.copy()
    env.update(
        {
            "NOMAD_IMPORT_TRACE_PATH": str(trace_path),
            "PYTHONPATH": os.pathsep.join((str(helper_root), str(wrappers_root))),
        }
    )
    if extra_env:
        env.update(extra_env)

    completed = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=workspace_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed, trace_path


def _wrapper_tool_call_source() -> str:
    return (
        textwrap.dedent(
            """
        import json
        import os
        import socketserver
        import threading

        TOOL_SPEC = {
            "package": "mcp_tools",
            "server": "dummy",
            "module": "dummy",
            "exports": ["add"],
            "tools": [
                {
                    "identifier": "add",
                    "tool_name": "add",
                    "description": "Add two integers.",
                    "schema": {
                        "name": "add",
                        "description": "Add two integers.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "a": {"type": "integer"},
                                "b": {"type": "integer"},
                            },
                            "required": ["a", "b"],
                        },
                        "outputSchema": {
                            "type": "object",
                            "properties": {
                                "structuredContent": {
                                    "type": "object",
                                    "properties": {
                                        "result": {"type": "integer"},
                                    },
                                },
                            },
                        },
                    },
                },
            ],
        }


        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                header = self.rfile.read(4)
                if not header:
                    return
                message_size = int.from_bytes(header, byteorder="big")
                request = json.loads(self.rfile.read(message_size).decode("utf-8"))
                method = request["method"]
                if method == "ensure_wrapper":
                    response = {
                        "jsonrpc": "2.0",
                        "id": request["id"],
                        "result": TOOL_SPEC,
                    }
                elif method == "call_tool":
                    args = request["params"]["arguments"]
                    response = {
                        "jsonrpc": "2.0",
                        "id": request["id"],
                        "result": {"structuredContent": {"result": args["a"] + args["b"]}},
                    }
                else:
                    response = {
                        "jsonrpc": "2.0",
                        "id": request["id"],
                        "error": {"code": -32601, "message": f"unknown method: {method}"},
                    }
                body = json.dumps(response).encode("utf-8")
                self.wfile.write(len(body).to_bytes(4, byteorder="big") + body)


        server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            host, port = server.server_address
            os.environ["NOMAD_MCP_GATEWAY_BRIDGE"] = f"{host}:{port}"
            from mcp_tools.dummy import add

            RESULT = add(a=1, b=2)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=1)
        """
        ).strip()
        + "\n"
    )


def _wrapper_tensor_tool_call_source() -> str:
    return (
        textwrap.dedent(
            """
        import json
        import os
        import socketserver
        import threading

        import torch
        from pydantic import BaseModel, ConfigDict

        from nomad.tensor_codac import serialize_tensor
        from nomad.well_format import Tensor

        class StructuredResult(BaseModel):
            model_config = ConfigDict(arbitrary_types_allowed=True)
            result: Tensor


        TOOL_SPEC = {
            "package": "mcp_tools",
            "server": "dummy",
            "module": "dummy",
            "exports": ["make_tensor"],
            "tools": [
                {
                    "identifier": "make_tensor",
                    "tool_name": "make_tensor",
                    "description": "Return a tensor.",
                    "schema": {
                        "name": "make_tensor",
                        "description": "Return a tensor.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {},
                        },
                        "outputSchema": {
                            "type": "object",
                            "properties": {
                                "structuredContent": StructuredResult.model_json_schema(),
                            },
                        },
                    },
                },
            ],
        }


        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                header = self.rfile.read(4)
                if not header:
                    return
                message_size = int.from_bytes(header, byteorder="big")
                request = json.loads(self.rfile.read(message_size).decode("utf-8"))
                method = request["method"]
                if method == "ensure_wrapper":
                    response = {
                        "jsonrpc": "2.0",
                        "id": request["id"],
                        "result": TOOL_SPEC,
                    }
                elif method == "call_tool":
                    response = {
                        "jsonrpc": "2.0",
                        "id": request["id"],
                        "result": {
                            "structuredContent": {
                                "result": serialize_tensor(
                                    torch.tensor([1.0, 2.0], dtype=torch.float32)
                                )
                            }
                        },
                    }
                else:
                    response = {
                        "jsonrpc": "2.0",
                        "id": request["id"],
                        "error": {"code": -32601, "message": f"unknown method: {method}"},
                    }
                body = json.dumps(response).encode("utf-8")
                self.wfile.write(len(body).to_bytes(4, byteorder="big") + body)


        server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            host, port = server.server_address
            os.environ["NOMAD_MCP_GATEWAY_BRIDGE"] = f"{host}:{port}"
            from mcp_tools.dummy import make_tensor

            RESULT = {"payload": make_tensor()}
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=1)
        """
        ).strip()
        + "\n"
    )


def test_runner_main_writes_result_when_script_succeeds(tmp_path, monkeypatch):
    script_path = tmp_path / "script.py"
    result_path = tmp_path / "result.json"
    wrappers_root = tmp_path / "wrappers"
    wrappers_root.mkdir()
    script_path.write_text("RESULT = {'value': 123}\n", encoding="utf-8")

    _prepare_env(monkeypatch, script_path, result_path, wrappers_root)
    monkeypatch.setattr(sys, "path", list(sys.path))

    exit_code = runner.main()
    assert exit_code == 0

    payload = json.loads(result_path.read_text("utf-8"))
    assert payload == {"result": {"value": 123}}


def test_runner_main_forwards_argv_to_script(tmp_path, monkeypatch):
    script_path = tmp_path / "script.py"
    result_path = tmp_path / "result.json"
    wrappers_root = tmp_path / "wrappers"
    wrappers_root.mkdir()
    script_path.write_text("import sys\nRESULT = sys.argv\n", encoding="utf-8")

    _prepare_env(monkeypatch, script_path, result_path, wrappers_root)
    monkeypatch.setattr(sys, "path", list(sys.path))

    original_argv = list(sys.argv)
    exit_code = runner.main(["--alpha", "beta"])
    assert exit_code == 0
    assert sys.argv == original_argv

    payload = json.loads(result_path.read_text("utf-8"))
    assert payload == {"result": [str(script_path), "--alpha", "beta"]}


def test_runner_main_records_error_payload(tmp_path, monkeypatch):
    script_path = tmp_path / "script.py"
    result_path = tmp_path / "result.json"
    wrappers_root = tmp_path / "wrappers"
    wrappers_root.mkdir()
    script_path.write_text("raise RuntimeError('boom')\n", encoding="utf-8")

    _prepare_env(monkeypatch, script_path, result_path, wrappers_root)
    monkeypatch.setattr(sys, "path", list(sys.path))

    with pytest.raises(RuntimeError):
        runner.main()

    payload = json.loads(result_path.read_text("utf-8"))
    assert payload == {"result": {"error": "boom"}}


def test_runner_main_writes_tensor_results_for_nested_containers(tmp_path, monkeypatch):
    torch, Tensor = _load_tensor_test_deps()

    class NestedTensorResult(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        payload: dict[str, list[Tensor]]

    script_path = tmp_path / "script.py"
    result_path = tmp_path / "result.json"
    wrappers_root = tmp_path / "wrappers"
    wrappers_root.mkdir()
    script_path.write_text(
        textwrap.dedent(
            """
            import torch

            RESULT = {"payload": {"tensors": [torch.tensor([1.0, 2.0], dtype=torch.float32)]}}
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    _prepare_env(monkeypatch, script_path, result_path, wrappers_root)
    monkeypatch.setattr(sys, "path", list(sys.path))

    exit_code = runner.main()
    assert exit_code == 0

    payload = json.loads(result_path.read_text("utf-8"))
    decoded = NestedTensorResult.model_validate(payload["result"])
    assert isinstance(payload["result"]["payload"]["tensors"][0], str)
    assert torch.equal(
        decoded.payload["tensors"][0],
        torch.tensor([1.0, 2.0], dtype=torch.float32),
    )


def test_runner_main_reencodes_tensor_results_from_wrappers(tmp_path, monkeypatch):
    torch, Tensor = _load_tensor_test_deps()

    class TensorResult(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        payload: Tensor

    script_path = tmp_path / "script.py"
    result_path = tmp_path / "result.json"
    wrappers_root, _, _ = _prepare_wrappers_root(tmp_path)
    script_path.write_text(_wrapper_tensor_tool_call_source(), encoding="utf-8")

    _prepare_env(monkeypatch, script_path, result_path, wrappers_root)
    monkeypatch.setattr(sys, "path", [*sys.path, str(wrappers_root)])

    exit_code = runner.main()
    assert exit_code == 0

    payload = json.loads(result_path.read_text("utf-8"))
    decoded = TensorResult.model_validate(payload["result"])
    assert isinstance(payload["result"]["payload"], str)
    assert torch.equal(decoded.payload, torch.tensor([1.0, 2.0], dtype=torch.float32))


@pytest.mark.parametrize(
    ("case_name", "source"),
    [
        ("runner-script", "RESULT = {'ok': True}\n"),
        ("wrapper-root-import", "import mcp_tools\nRESULT = 'ok'\n"),
        ("wrapper-tool-call", _wrapper_tool_call_source()),
    ],
    ids=["runner-script", "wrapper-root-import", "wrapper-tool-call"],
)
def test_runtime_entrypoints_import_only_runtime_and_stdlib(
    tmp_path: Path,
    case_name: str,
    source: str,
):
    wrappers_root, helper_root, wrappers_package_root = _prepare_wrappers_root(tmp_path)
    runtime_root = wrappers_root / "nomad" / "gateway" / "runtime"

    if case_name == "runner-script":
        runner_path = runtime_root / "runner.py"
        script_path = tmp_path / "script.py"
        result_path = tmp_path / "result.json"
        trace_path = tmp_path / "import-trace.json"
        workspace_root = tmp_path / "workspace"
        workspace_root.mkdir()
        script_path.write_text(source, encoding="utf-8")

        env = os.environ.copy()
        env.update(
            {
                "NOMAD_IMPORT_TRACE_PATH": str(trace_path),
                "NOMAD_MCP_SCRIPT_PATH": str(script_path),
                "NOMAD_MCP_RESULT_PATH": str(result_path),
                "NOMAD_MCP_WRAPPERS_ROOT": str(wrappers_root),
                "PYTHONPATH": os.pathsep.join((str(helper_root), str(wrappers_root))),
            }
        )

        completed = subprocess.run(
            [sys.executable, str(runner_path)],
            cwd=workspace_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        assert completed.returncode == 0, completed.stderr
        assert json.loads(result_path.read_text(encoding="utf-8")) == {
            "result": {"ok": True}
        }
        imported_modules = json.loads(trace_path.read_text(encoding="utf-8"))
        forbidden = _find_forbidden_imports(
            imported_modules,
            allowed_roots=(runtime_root, helper_root),
        )
        assert not forbidden, (
            "runner.py imported modules outside nomad.gateway.runtime or the stdlib: "
            f"{forbidden}"
        )
        return

    completed, trace_path = _run_traced_python_script(
        tmp_path=tmp_path,
        wrappers_root=wrappers_root,
        helper_root=helper_root,
        source=source,
    )
    assert completed.returncode == 0, completed.stderr

    imported_modules = json.loads(trace_path.read_text(encoding="utf-8"))
    forbidden = _find_forbidden_imports(
        imported_modules,
        allowed_roots=(runtime_root, helper_root),
        allowed_files=(
            wrappers_root / "nomad" / "__init__.py",
            wrappers_root / "nomad" / "gateway" / "__init__.py",
            # Shared JSON-RPC 2.0 support; its own imports are traced too.
            wrappers_root / "nomad" / "jsonrpc.py",
            wrappers_package_root / "__init__.py",
        ),
        allowed_prefixes=("mcp_tools",),
    )
    assert not forbidden, (
        f"wrapper import path loaded modules outside the runtime boundary: {forbidden}"
    )
