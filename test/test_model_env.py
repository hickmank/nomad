from __future__ import annotations

import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor
from hashlib import blake2b
from pathlib import Path
from threading import Barrier, Event, Lock

import pytest

from nomad import model_env
from nomad.model_rpc import ModelProcess


def _expected_checksum(
    *,
    contents: bytes,
    nomad_requirement: str,
    nomad_project: bytes,
) -> str:
    digest = blake2b()
    digest.update(b"nomad-venv-requirements-")
    digest.update(contents)
    digest.update(b"\0nomad-requirement\0")
    digest.update(nomad_requirement.encode())
    digest.update(b"\0nomad-project\0")
    digest.update(nomad_project)
    digest.update(b"\0python-runtime\0")
    digest.update(model_env._python_runtime_identity())
    return digest.hexdigest()


def test_environment_checksum_uses_typed_contents_and_nomad(
    monkeypatch,
    tmp_path: Path,
):
    nomad_requirement = "nomad-scifm==1.2.3"
    nomad_project = b"nomad project metadata"
    monkeypatch.setattr(
        model_env,
        "_nomad_requirement",
        lambda: (nomad_requirement, nomad_project),
    )
    environment = model_env.resolve_model_environment(
        "demo-package==1.2", base_dir=tmp_path
    )

    assert environment is not None
    assert environment.requirements == ("demo-package==1.2",)
    assert environment.contents == b"demo-package==1.2"
    assert environment.checksum == _expected_checksum(
        contents=b"demo-package==1.2",
        nomad_requirement=nomad_requirement,
        nomad_project=nomad_project,
    )


def test_single_requirement_is_one_item_requirement_list(tmp_path: Path):
    single = model_env.resolve_model_environment("demo-package>=1", base_dir=tmp_path)
    listed = model_env.resolve_model_environment(["demo-package>=1"], base_dir=tmp_path)

    assert single is not None
    assert listed is not None
    assert single.requirements == ("demo-package>=1",)
    assert single.checksum == listed.checksum


def test_single_url_requirement_ending_in_reserved_filename_is_not_a_path(
    tmp_path: Path,
):
    requirement = "demo @ https://example.com/requirements.txt"

    environment = model_env.resolve_model_environment(requirement, base_dir=tmp_path)

    assert environment.requirements == (requirement,)


def test_omitted_environment_uses_host_python(tmp_path: Path):
    environment = model_env.resolve_model_environment(None, base_dir=tmp_path)

    assert isinstance(environment, model_env.HostEnvironment)
    assert environment.ensure() == Path(model_env.sys.executable).absolute()


def test_environment_checksum_includes_python_runtime(monkeypatch, tmp_path: Path):
    environment = model_env.resolve_model_environment([], base_dir=tmp_path)
    monkeypatch.setattr(model_env, "_python_runtime_identity", lambda: b"runtime-a")
    first = environment.checksum
    monkeypatch.setattr(model_env, "_python_runtime_identity", lambda: b"runtime-b")

    assert environment.checksum != first


def test_environment_creation_uses_one_checksum_snapshot(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.delenv("NOMAD_CACHE", raising=False)
    snapshots = iter(
        [
            ("nomad-scifm==1", b"first project"),
            ("nomad-scifm==2", b"second project"),
        ]
    )
    calls = 0

    def nomad_requirement() -> tuple[str, bytes]:
        nonlocal calls
        calls += 1
        return next(snapshots)

    captured: dict[str, str] = {}

    def fake_create(
        self: model_env.ModelEnvironment,
        target: Path,
        *,
        checksum: str,
        nomad_requirement: str,
    ) -> None:
        captured["checksum"] = checksum
        captured["nomad_requirement"] = nomad_requirement
        (target / "bin").mkdir(parents=True)
        (target / "bin" / "python").touch()

    monkeypatch.setattr(model_env, "_nomad_requirement", nomad_requirement)
    monkeypatch.setattr(model_env.ModelEnvironment, "_create", fake_create)
    environment = model_env.resolve_model_environment([], base_dir=tmp_path)

    python = environment.ensure()

    assert calls == 1
    assert captured["nomad_requirement"] == "nomad-scifm==1"
    assert python.parent.parent.name == captured["checksum"]


def test_environment_can_be_built_into_export_cache(monkeypatch, tmp_path: Path):
    environment = model_env.resolve_model_environment(
        ["demo-package==1"],
        base_dir=tmp_path,
    )

    def fake_create(
        self: model_env.ModelEnvironment,
        target: Path,
        **kwargs,
    ) -> None:
        (target / "bin").mkdir(parents=True)
        (target / "bin" / "python").touch()

    monkeypatch.setattr(model_env.ModelEnvironment, "_create", fake_create)
    cache_root = tmp_path / "export"

    python = environment.ensure(cache_root=cache_root)

    assert python == cache_root / "venv" / environment.checksum / "bin" / "python"


def test_environment_cache_is_reused(monkeypatch, tmp_path: Path):
    commands: list[list[str]] = []
    installed_requirements: list[str] = []
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.delenv("NOMAD_CACHE", raising=False)
    monkeypatch.setattr(model_env.shutil, "which", lambda command: "/bin/uv")

    def fake_run(command: list[str], *, cwd: Path) -> None:
        commands.append(command)
        if command[1] == "venv":
            target = Path(command[-1])
            (target / "bin").mkdir(parents=True)
            (target / "bin" / "python").touch()
        elif command[1:3] == ["pip", "install"]:
            installed_requirements.append(Path(command[-1]).read_text())

    monkeypatch.setattr(model_env, "_run", fake_run)
    environment = model_env.resolve_model_environment(
        ["demo-package"], base_dir=tmp_path
    )
    assert environment is not None

    first = environment.ensure()
    second = environment.ensure()

    assert first == second
    assert (
        first
        == tmp_path
        / "cache"
        / "nomad"
        / "venv"
        / environment.checksum
        / "bin"
        / "python"
    )
    assert len(commands) == 2
    assert commands[0][1:3] == ["venv", "--python"]
    assert "--system-site-packages" not in commands[0]
    assert commands[1][1:3] == ["pip", "install"]
    assert not (environment.cache_dir / ".nomad-requirements.txt").exists()
    injected = installed_requirements[0]
    assert injected.splitlines()[0] == "demo-package"
    assert injected.splitlines()[1].startswith("nomad-scifm @ file://")


def test_completed_environment_reuse_does_not_require_lock_access(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setenv("NOMAD_CACHE", str(tmp_path / "cache"))
    environment = model_env.resolve_model_environment(
        ["demo-package"], base_dir=tmp_path
    )

    def fake_create(
        self: model_env.ModelEnvironment,
        target: Path,
        **kwargs,
    ) -> None:
        (target / "bin").mkdir(parents=True)
        (target / "bin" / "python").touch()

    monkeypatch.setattr(model_env.ModelEnvironment, "_create", fake_create)
    first = environment.ensure()
    monkeypatch.setattr(
        model_env,
        "FileLock",
        lambda *args, **kwargs: pytest.fail("completed cache should not take its lock"),
    )

    assert environment.ensure() == first


def test_incomplete_environment_with_python_is_rebuilt(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("NOMAD_CACHE", str(tmp_path / "cache"))
    environment = model_env.resolve_model_environment(
        ["demo-package"], base_dir=tmp_path
    )
    stale_python = environment.python
    stale_python.parent.mkdir(parents=True)
    stale_python.touch()
    create_count = 0

    def fake_create(
        self: model_env.ModelEnvironment,
        target: Path,
        **kwargs,
    ) -> None:
        nonlocal create_count
        create_count += 1
        (target / "bin").mkdir(parents=True)
        (target / "bin" / "python").touch()

    monkeypatch.setattr(model_env.ModelEnvironment, "_create", fake_create)

    assert environment.ensure() == stale_python
    assert create_count == 1


def test_same_checksum_environment_creation_is_serialized(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.delenv("NOMAD_CACHE", raising=False)
    environment = model_env.resolve_model_environment(
        ["demo-package"], base_dir=tmp_path
    )
    started = Barrier(3)
    create_entered = Event()
    second_create_entered = Event()
    release_create = Event()
    state_lock = Lock()
    create_count = 0

    def fake_create(
        self: model_env.ModelEnvironment,
        target: Path,
        **kwargs,
    ) -> None:
        nonlocal create_count
        with state_lock:
            create_count += 1
            if create_count > 1:
                second_create_entered.set()
        create_entered.set()
        assert release_create.wait(timeout=2)
        (target / "bin").mkdir(parents=True)
        (target / "bin" / "python").touch()

    def ensure() -> Path:
        started.wait()
        return environment.ensure()

    monkeypatch.setattr(model_env.ModelEnvironment, "_create", fake_create)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(ensure) for _ in range(2)]
        started.wait()
        assert create_entered.wait(timeout=2)
        assert not second_create_entered.wait(timeout=0.1)
        release_create.set()
        results = [future.result(timeout=2) for future in futures]

    assert create_count == 1
    assert results[0] == results[1]


def test_different_checksum_environments_build_concurrently(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.delenv("NOMAD_CACHE", raising=False)
    environments = [
        model_env.resolve_model_environment([requirement], base_dir=tmp_path)
        for requirement in ("first-package", "second-package")
    ]
    started = Barrier(3)
    both_creates_entered = Event()
    state_lock = Lock()
    active_creates = 0

    def fake_create(
        self: model_env.ModelEnvironment,
        target: Path,
        **kwargs,
    ) -> None:
        nonlocal active_creates
        with state_lock:
            active_creates += 1
            if active_creates == 2:
                both_creates_entered.set()
        assert both_creates_entered.wait(timeout=2)
        (target / "bin").mkdir(parents=True)
        (target / "bin" / "python").touch()

    def ensure(environment: model_env.ModelEnvironment) -> Path:
        started.wait()
        return environment.ensure()

    monkeypatch.setattr(model_env.ModelEnvironment, "_create", fake_create)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(ensure, environment) for environment in environments]
        started.wait()
        assert both_creates_entered.wait(timeout=2)
        results = [future.result(timeout=2) for future in futures]

    assert active_creates == 2
    assert results[0] != results[1]


def test_requirements_with_different_base_directories_share_cache(
    monkeypatch, tmp_path: Path
):
    commands: list[list[str]] = []
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.delenv("NOMAD_CACHE", raising=False)
    monkeypatch.setattr(model_env.shutil, "which", lambda command: "/bin/uv")

    def fake_run(command: list[str], *, cwd: Path) -> None:
        commands.append(command)
        if command[1] == "venv":
            target = Path(command[-1])
            (target / "bin").mkdir(parents=True)
            (target / "bin" / "python").touch()

    monkeypatch.setattr(model_env, "_run", fake_run)
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()

    first = model_env.resolve_model_environment(["demo-package"], base_dir=first_dir)
    second = model_env.resolve_model_environment(["demo-package"], base_dir=second_dir)
    assert first.checksum == second.checksum

    first.ensure()
    second.ensure()

    assert len(commands) == 2
    assert first.cache_dir.is_dir()
    assert first.cache_dir == second.cache_dir


def test_environment_creation_falls_back_to_stdlib_venv(monkeypatch, tmp_path: Path):
    commands: list[list[str]] = []
    installed_requirements: list[str] = []
    monkeypatch.setenv("NOMAD_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(model_env.shutil, "which", lambda command: None)
    monkeypatch.setattr(
        model_env,
        "_nomad_requirement",
        lambda: ("nomad-scifm==1.2.3", b"nomad project metadata"),
    )

    def fake_run(command: list[str], *, cwd: Path) -> None:
        commands.append(command)
        if command[1:3] == ["-m", "venv"]:
            target = Path(command[-1])
            (target / "bin").mkdir(parents=True)
            (target / "bin" / "python").touch()
        elif command[1:4] == ["-m", "pip", "install"]:
            installed_requirements.append(Path(command[-1]).read_text())

    monkeypatch.setattr(model_env, "_run", fake_run)
    environment = model_env.resolve_model_environment([], base_dir=tmp_path)

    environment.ensure()

    assert commands[0][1:3] == ["-m", "venv"]
    assert commands[1][1:4] == ["-m", "pip", "install"]
    assert installed_requirements == ["nomad-scifm==1.2.3\n"]
    assert not (environment.cache_dir / ".nomad-requirements.txt").exists()


@pytest.mark.parametrize("builder", ["uv", "venv"])
def test_real_cached_environment_runs_worker(
    builder: str,
    monkeypatch,
    tmp_path: Path,
):
    uv = shutil.which("uv")
    if builder == "uv" and uv is None:
        pytest.skip("uv is not installed")

    dependency_wheel = tmp_path / "fixture_dependency-0.0.0-py3-none-any.whl"
    _write_wheel(
        dependency_wheel,
        name="fixture-dependency",
        files={"fixture_dependency/__init__.py": "VALUE = 42\n"},
    )
    nomad_wheel = tmp_path / "nomad_scifm-0.0.0-py3-none-any.whl"
    _write_wheel(
        nomad_wheel,
        name="nomad-scifm",
        files={
            "nomad/__init__.py": "",
            "nomad/model_worker.py": """
import json
import os
import subprocess
import sys
import fixture_dependency

for line in sys.stdin:
    request = json.loads(line)
    method = request["method"]
    result = (
        {
            "dependency": fixture_dependency.VALUE,
            "executable": sys.executable,
            "virtual_env": os.environ.get("VIRTUAL_ENV"),
            "spawned_executable": subprocess.check_output(
                ["python", "-c", "import sys; print(sys.executable)"],
                text=True,
            ).strip(),
        }
        if method == "probe"
        else None
    )
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}) + "\\n")
    sys.stdout.flush()
    if method == "shutdown":
        break
""",
        },
    )
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / f"cache-{builder}"))
    monkeypatch.delenv("NOMAD_CACHE", raising=False)
    monkeypatch.setattr(
        model_env.shutil,
        "which",
        lambda command: uv if builder == "uv" and command == "uv" else None,
    )
    monkeypatch.setattr(
        model_env,
        "_nomad_requirement",
        lambda: (
            f"nomad-scifm @ {nomad_wheel.as_uri()}",
            nomad_wheel.read_bytes(),
        ),
    )
    environment = model_env.resolve_model_environment(
        [f"fixture-dependency @ {dependency_wheel.as_uri()}"],
        base_dir=tmp_path,
    )

    with ModelProcess(environment) as process:
        result = process.request("probe")

    assert environment.python.is_file()
    assert result["dependency"] == 42
    assert Path(result["executable"]).resolve() == environment.python.resolve()
    assert Path(result["spawned_executable"]).resolve() == environment.python.resolve()
    assert (
        Path(result["virtual_env"]).resolve()
        == environment.python.parent.parent.resolve()
    )


def _write_wheel(path: Path, *, name: str, files: dict[str, str]) -> None:
    version = "0.0.0"
    distribution = name.replace("-", "_")
    dist_info = f"{distribution}-{version}.dist-info"
    wheel_files = {
        **files,
        f"{dist_info}/METADATA": (
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
        ),
        f"{dist_info}/WHEEL": (
            "Wheel-Version: 1.0\n"
            "Generator: nomad-test\n"
            "Root-Is-Purelib: true\n"
            "Tag: py3-none-any\n"
        ),
    }
    record = "".join(f"{wheel_path},,\n" for wheel_path in wheel_files)
    record_path = f"{dist_info}/RECORD"
    with zipfile.ZipFile(path, "w") as wheel:
        for wheel_path, contents in wheel_files.items():
            wheel.writestr(wheel_path, contents)
        wheel.writestr(record_path, record + f"{record_path},,\n")
