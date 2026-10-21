from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import sysconfig
from dataclasses import dataclass, field
from hashlib import blake2b
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from uuid import uuid4

from filelock import FileLock

from .hub import get_cache_root

logger = logging.getLogger(__name__)

_COMPLETE_MARKER = ".nomad-complete"
_CACHE_FORMAT = "3"
_HASH_PREFIX = "nomad-venv-"


@dataclass(frozen=True, slots=True)
class ModelEnvironment:
    """A normalized model environment and its content-addressed cache key."""

    requirements: tuple[str, ...]
    base_dir: Path = field(default_factory=Path.cwd)

    @property
    def contents(self) -> bytes:
        return "\n".join(self.requirements).encode()

    @property
    def checksum(self) -> str:
        nomad_requirement, nomad_project = _nomad_requirement()
        return self._checksum(
            nomad_requirement=nomad_requirement,
            nomad_project=nomad_project,
        )

    def _checksum(self, *, nomad_requirement: str, nomad_project: bytes) -> str:
        digest = blake2b()
        digest.update(f"{_HASH_PREFIX}requirements-".encode())
        digest.update(self.contents)
        digest.update(b"\0nomad-requirement\0")
        digest.update(nomad_requirement.encode())
        digest.update(b"\0nomad-project\0")
        digest.update(nomad_project)
        digest.update(b"\0python-runtime\0")
        digest.update(_python_runtime_identity())
        return digest.hexdigest()

    @property
    def cache_dir(self) -> Path:
        return get_cache_root() / "venv" / self.checksum

    @property
    def python(self) -> Path:
        return _venv_python(self.cache_dir)

    def ensure(self, *, cache_root: Path | None = None) -> Path:
        """Create the cached environment if necessary and return its Python."""
        root = (cache_root or get_cache_root()) / "venv"
        root.mkdir(parents=True, exist_ok=True)
        nomad_requirement, nomad_project = _nomad_requirement()
        checksum = self._checksum(
            nomad_requirement=nomad_requirement,
            nomad_project=nomad_project,
        )
        target = root / checksum
        expected_marker = f"{_CACHE_FORMAT}:{checksum}\n"
        if python := _completed_environment(target, expected_marker):
            logger.debug("Reusing model environment %s", target)
            return python

        lock = FileLock(str(root / f".{checksum}.lock"))
        with lock:
            if python := _completed_environment(target, expected_marker):
                logger.debug("Reusing model environment %s", target)
                return python

            if target.exists():
                shutil.rmtree(target)

            temporary = root / f".{checksum}.tmp-{uuid4().hex}"
            try:
                self._create(
                    temporary,
                    checksum=checksum,
                    nomad_requirement=nomad_requirement,
                )
                (temporary / _COMPLETE_MARKER).write_text(
                    expected_marker, encoding="utf-8"
                )
                temporary.replace(target)
            except BaseException:
                shutil.rmtree(temporary, ignore_errors=True)
                raise

        return _venv_python(target)

    def _create(
        self,
        target: Path,
        *,
        checksum: str,
        nomad_requirement: str,
    ) -> None:
        uv = shutil.which("uv")
        if uv:
            logger.info("Creating model environment %s with uv", checksum)
            _run(
                [
                    uv,
                    "venv",
                    "--python",
                    sys.executable,
                    str(target),
                ],
                cwd=self.base_dir,
            )
            installer = [uv, "pip", "install", "--python", str(_venv_python(target))]
        else:
            logger.info(
                "uv is unavailable; creating model environment %s with venv",
                checksum,
            )
            _run(
                [sys.executable, "-m", "venv", str(target)],
                cwd=self.base_dir,
            )
            installer = [str(_venv_python(target)), "-m", "pip", "install"]

        requirements = target / ".nomad-requirements.txt"
        requirements.touch(mode=0o600)
        try:
            requirements.write_text(
                "\n".join([*self.requirements, nomad_requirement]) + "\n",
                encoding="utf-8",
            )
            _run([*installer, "-r", str(requirements)], cwd=self.base_dir)
        finally:
            requirements.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class HostEnvironment:
    """The Python environment already running the Nomad host process."""

    python: Path
    prefix: Path

    @property
    def checksum(self) -> str:
        return blake2b(f"{_HASH_PREFIX}host-{self.prefix}".encode()).hexdigest()

    def ensure(self) -> Path:
        return self.python


PythonEnvironment = ModelEnvironment | HostEnvironment


def resolve_model_environment(
    value: str | list[str] | None,
    *,
    base_dir: Path | None = None,
) -> PythonEnvironment:
    """Normalize a model's PEP 508 requirements for subprocess execution."""
    if value is None:
        return HostEnvironment(
            python=Path(sys.executable).absolute(),
            prefix=Path(sys.prefix).resolve(),
        )

    resolved_base = (base_dir or Path.cwd()).expanduser().resolve()
    if isinstance(value, str):
        requirements = (value,)
    else:
        requirements = tuple(value)

    return ModelEnvironment(
        requirements=requirements,
        base_dir=resolved_base,
    )


def _venv_python(venv: Path) -> Path:
    posix_python = venv / "bin" / "python"
    if posix_python.exists() or not (venv / "Scripts" / "python.exe").exists():
        return posix_python
    return venv / "Scripts" / "python.exe"


def _completed_environment(target: Path, expected_marker: str) -> Path | None:
    python = _venv_python(target)
    marker = target / _COMPLETE_MARKER
    try:
        if python.is_file() and marker.read_text(encoding="utf-8") == expected_marker:
            return python
    except OSError:
        pass
    return None


def _python_runtime_identity() -> bytes:
    """Return a portable identity for venv compatibility, not an executable path."""
    implementation = sys.implementation
    return "\0".join(
        (
            implementation.name,
            implementation.cache_tag or "",
            f"{sys.version_info.major}.{sys.version_info.minor}",
            sysconfig.get_platform(),
        )
    ).encode()


def _nomad_requirement() -> tuple[str, bytes]:
    """Return the running Nomad requirement and raw project checksum material."""
    source_root = Path(__file__).resolve().parent.parent.parent
    pyproject = source_root / "pyproject.toml"
    if pyproject.is_file():
        return (
            f"nomad-scifm @ {source_root.as_uri()}",
            pyproject.read_bytes(),
        )

    try:
        installed_version = version("nomad-scifm")
    except PackageNotFoundError as exc:  # pragma: no cover - broken installation
        raise RuntimeError(
            "Cannot construct a model environment because Nomad is not installed"
        ) from exc
    requirement = f"nomad-scifm=={installed_version}"
    return requirement, b""


def _run(command: list[str], *, cwd: Path) -> None:
    try:
        subprocess.run(command, cwd=cwd, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Failed to construct model environment with {Path(command[0]).name}"
        ) from exc


__all__ = [
    "HostEnvironment",
    "ModelEnvironment",
    "PythonEnvironment",
    "resolve_model_environment",
]
