from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import sysconfig
from dataclasses import dataclass, field
from hashlib import blake2b
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import url2pathname
from uuid import uuid4

from filelock import FileLock, Timeout
from packaging.requirements import InvalidRequirement, Requirement

from .hub import get_cache_root

logger = logging.getLogger(__name__)

_COMPLETE_MARKER = ".nomad-complete"
_CACHE_FORMAT = "3"
_HASH_PREFIX = "nomad-venv-"
# How long to wait for another process that is building the same environment.
# Building can download and install large packages such as torch, so this is
# generous; it exists to turn a hung builder into an error, not to bound builds.
_BUILD_LOCK_TIMEOUT_SECONDS = 3600.0
# Directories skipped when hashing a local package: VCS metadata, caches,
# virtual environments, and build output. They are large, change without the
# package changing, and are not what an installer builds from.
_SKIPPED_SOURCE_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".venv",
        "venv",
        "build",
        "dist",
        "node_modules",
        ".tox",
        ".nox",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
    }
)


@dataclass(frozen=True, slots=True)
class ModelEnvironment:
    """A normalized model environment and its content-addressed cache key."""

    requirements: tuple[str, ...]
    base_dir: Path = field(default_factory=Path.cwd)
    # Whether ``ensure`` may build a missing environment. Offline deployments set
    # this to False so a missing prebuilt venv fails fast instead of trying to
    # download packages. It is a runtime policy, not part of the environment's
    # identity, so it is excluded from equality and from the checksum.
    allow_build: bool = field(default=True, compare=False)
    # Digest of the local ``file:`` packages in ``requirements``, taken once at
    # construction: the checksum is read on every model load, and re-hashing a
    # source tree each time would be slow.
    local_sources: bytes = field(init=False, repr=False)

    def __post_init__(self) -> None:
        digest = blake2b()
        for _name, path in _local_requirement_paths(self.requirements):
            digest.update(_local_source_digest(path))
        object.__setattr__(self, "local_sources", digest.digest())

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
        # Environments are built from the host interpreter (``sys.executable``),
        # and their compiled wheels and bytecode only work with that Python's
        # version, implementation, and platform. Keying on these keeps a shared
        # or exported cache from handing a 3.12 venv to a 3.13 host.
        digest.update(b"\0python-runtime\0")
        digest.update(_python_runtime_identity())
        # Local ``file:`` packages are identified by their contents, not just
        # their path, so editing one gives a new environment instead of
        # silently reusing a stale build.
        digest.update(b"\0local-sources\0")
        digest.update(self.local_sources)
        return digest.hexdigest()

    @property
    def cache_dir(self) -> Path:
        return get_cache_root() / "venv" / self.checksum

    @property
    def python(self) -> Path:
        return _venv_python(self.cache_dir)

    def ensure(self, *, cache_root: Path | None = None) -> Path:
        """Create the cached environment if necessary and return its Python.

        Raises:
            RuntimeError: If the environment is missing and ``allow_build`` is
                False.
        """
        root = (cache_root or get_cache_root()) / "venv"
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

        # Check before touching the cache so an offline server never creates
        # directories or lock files for an environment it may not build.
        if not self.allow_build:
            raise RuntimeError(
                "Model environment is not prebuilt and building is disabled "
                f"(tool_manager.build_envs is false). Expected {target}. Run "
                "`nomad export` with the same Python and Nomad installation that "
                "serves the bundle, and set NOMAD_CACHE to the export directory."
            )

        root.mkdir(parents=True, exist_ok=True)
        lock_path = root / f".{checksum}.lock"
        # Another process may be building the same environment; wait for it,
        # but not forever. Only acquiring the lock raises ``Timeout``.
        try:
            with FileLock(str(lock_path), timeout=_BUILD_LOCK_TIMEOUT_SECONDS):
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
                except BaseException as exc:
                    shutil.rmtree(temporary, ignore_errors=True)
                    if isinstance(exc, RuntimeError):
                        # The installer's own output is already on stderr; add what
                        # was being built and the usual cause for exported bundles.
                        raise RuntimeError(
                            f"{exc} while building model environment {target} for "
                            f"requirements {list(self.requirements)!r}. If this "
                            "server runs from a `nomad export` bundle, check that "
                            "NOMAD_CACHE points at the export directory and that it "
                            "runs with the same Python and Nomad as the export."
                        ) from exc
                    raise
        except Timeout as exc:
            raise RuntimeError(
                f"Timed out after {_BUILD_LOCK_TIMEOUT_SECONDS:g} seconds waiting "
                f"for another process to build model environment {target} "
                f"(lock {lock_path})"
            ) from exc

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
            # uv reuses wheels it built from local directories unless their
            # build metadata changes, so force a fresh build of each local
            # package, including Nomad itself when it runs from a checkout;
            # the checksum already says their contents changed.
            local = _local_requirement_paths((*self.requirements, nomad_requirement))
            for name, _path in local:
                installer += ["--refresh-package", name]
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


@dataclass(frozen=True, slots=True)
class ExistingEnvironment:
    """A pre-built Python interpreter (venv or conda) managed outside Nomad.

    .. warning::
       For local development purposes only. ExistingEnvironments cannot be
       bundled and deployed reliably.
    """

    python: Path
    base_dir: Path = field(default_factory=Path.cwd)

    def __post_init__(self) -> None:
        # Nomad neither builds nor bundles this interpreter, so anything that
        # relies on it is not deployable as is; say so wherever one is created.
        logger.warning(
            "Using external interpreter %s; python: is development-only and a "
            "server that relies on it is not deployable by itself (use env for "
            "deployment)",
            self.python,
        )

    @property
    def checksum(self) -> str:
        digest = blake2b()
        digest.update(f"{_HASH_PREFIX}existing-".encode())
        digest.update(str(self.python).encode())
        if self.is_venv:
            digest.update(b"\0pyvenv-cfg\0")
            digest.update((self.prefix / "pyvenv.cfg").read_bytes())
        return digest.hexdigest()

    @property
    def prefix(self) -> Path:
        """Return the environment prefix, i.e. the parent of its ``bin`` directory."""
        return self.python.parent.parent

    @property
    def is_venv(self) -> bool:
        return (self.prefix / "pyvenv.cfg").is_file()

    def ensure(self) -> Path:
        if not self.python.is_file():
            raise FileNotFoundError(
                f"Model Python interpreter not found: {self.python}"
            )
        if not os.access(self.python, os.X_OK):
            raise PermissionError(
                f"Model Python interpreter is not executable: {self.python}"
            )
        return self.python


PythonEnvironment = ModelEnvironment | HostEnvironment | ExistingEnvironment


def resolve_model_environment(
    value: str | list[str] | None,
    *,
    base_dir: Path | None = None,
    python: str | Path | None = None,
    allow_build: bool = True,
) -> PythonEnvironment:
    """Normalize a model's PEP 508 requirements or interpreter for subprocesses.

    ``allow_build`` only applies to ``env`` requirements; host and existing
    interpreters are never built by Nomad.
    """
    if python is not None:
        if value is not None:
            raise ValueError("A model environment cannot set both env and python")
        resolved_base = (base_dir or Path.cwd()).expanduser().resolve()
        interpreter = Path(python).expanduser()
        if not interpreter.is_absolute():
            interpreter = resolved_base / interpreter
        # Keep symlinks: a venv's bin/python usually links outside the venv.
        return ExistingEnvironment(
            python=Path(os.path.normpath(interpreter)),
            base_dir=resolved_base,
        )

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
        allow_build=allow_build,
    )


def _local_requirement_paths(
    requirements: tuple[str, ...],
) -> list[tuple[str, Path]]:
    """Return ``(name, path)`` for each requirement that is a local ``file:`` URL.

    Only plain ``file:`` URLs are included. VCS URLs such as ``git+file:``
    install a commit rather than the working tree, so their contents do not
    affect the build in the same way.
    """
    local: list[tuple[str, Path]] = []
    for requirement in requirements:
        try:
            parsed = Requirement(requirement)
        except InvalidRequirement:
            continue
        if parsed.url is None:
            continue
        url = urlsplit(parsed.url)
        if url.scheme != "file":
            continue
        local.append((parsed.name, Path(url2pathname(url.path))))
    return local


def _local_source_digest(path: Path) -> bytes:
    """Return a digest of a local package file or source tree.

    Directories are walked in sorted order, hashing each file's relative path
    and contents and skipping ``_SKIPPED_SOURCE_DIRS``, ``*.egg-info``, and any
    virtual environment (a directory containing ``pyvenv.cfg``).
    Symlinked directories are not followed. A missing path gets a fixed digest
    so the installer reports the real error.

    Args:
        path: The file or directory a ``file:`` requirement points at.

    Returns:
        The digest bytes.
    """
    digest = blake2b()
    digest.update(str(path).encode())
    if path.is_file():
        digest.update(b"\0file\0")
        digest.update(path.read_bytes())
        return digest.digest()
    if not path.is_dir():
        digest.update(b"\0missing\0")
        return digest.digest()

    for root, dirs, files in os.walk(path):
        # Prune in place so os.walk does not descend into skipped directories,
        # and sort so the digest does not depend on directory listing order.
        dirs[:] = sorted(
            name for name in dirs if not _is_skipped_source_dir(Path(root), name)
        )
        for name in sorted(files):
            file_path = Path(root) / name
            if not file_path.is_file():
                continue
            digest.update(b"\0path\0")
            digest.update(file_path.relative_to(path).as_posix().encode())
            digest.update(b"\0contents\0")
            digest.update(file_path.read_bytes())
    return digest.digest()


def copy_local_source(source: Path, destination: Path) -> None:
    """Copy a local package file or source tree, skipping what hashing skips.

    Args:
        source: The file or directory a ``file:`` requirement points at.
        destination: Where to put the copy; must not exist yet.
    """
    if source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return

    def ignore(root: str, names: list[str]) -> list[str]:
        return [name for name in names if _is_skipped_source_dir(Path(root), name)]

    shutil.copytree(source, destination, ignore=ignore)


def _is_skipped_source_dir(parent: Path, name: str) -> bool:
    """Return whether ``parent / name`` is a directory local packages skip.

    Skipped are ``_SKIPPED_SOURCE_DIRS``, ``*.egg-info``, and any virtual
    environment (a directory containing ``pyvenv.cfg``).
    """
    path = parent / name
    if not path.is_dir():
        return False
    return (
        name in _SKIPPED_SOURCE_DIRS
        or name.endswith(".egg-info")
        or (path / "pyvenv.cfg").is_file()
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
            _nomad_checkout_material(source_root),
        )

    try:
        installed_version = version("nomad-scifm")
    except PackageNotFoundError as exc:  # pragma: no cover - broken installation
        raise RuntimeError(
            "Cannot construct a model environment because Nomad is not installed"
        ) from exc
    requirement = f"nomad-scifm=={installed_version}"
    return requirement, b""


def _nomad_checkout_material(source_root: Path) -> bytes:
    """Return checksum material for a Nomad source checkout.

    Covers ``pyproject.toml`` and the ``src`` tree that gets installed, so any
    edit to Nomad's code or dependencies gives model environments a new
    identity, while edits to tests or docs do not.

    Args:
        source_root: The checkout's root directory.

    Returns:
        The checksum material.
    """
    return b"\0".join(
        (
            (source_root / "pyproject.toml").read_bytes(),
            _local_source_digest(source_root / "src"),
        )
    )


def _run(command: list[str], *, cwd: Path) -> None:
    try:
        subprocess.run(command, cwd=cwd, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"{Path(command[0]).name} exited with status {exc.returncode}"
        ) from exc


__all__ = [
    "ExistingEnvironment",
    "HostEnvironment",
    "ModelEnvironment",
    "PythonEnvironment",
    "copy_local_source",
    "resolve_model_environment",
]
