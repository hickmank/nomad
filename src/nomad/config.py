from __future__ import annotations

import logging
import pkgutil
from dataclasses import replace
from importlib import import_module
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastmcp import FastMCP
from fastmcp.tools import Tool as FastMCPTool
from packaging.requirements import InvalidRequirement, Requirement
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from nomad.hub import RepoSpec

from ._torch_module_compat import (
    add_torch_module_tool_to_fastmcp,
    is_torch_module_tool_instance,
)
from .common.config_errors import load_config_mapping, validate_config_data
from .common.name_sanitize import sanitize_mcp_name
from .fm_base_tool import TorchModuleTool
from .model_env import PythonEnvironment, resolve_model_environment
from .model_rpc import RemoteModel, inspect_remote_model

logger = logging.getLogger(__name__)


def resolve_symbol(dotted_path: str) -> Any:
    """Resolve a dotted path to a Python object."""
    logger.debug("Resolving symbol '%s'", dotted_path)
    module_path, _, attr_name = dotted_path.rpartition(".")
    if not module_path:
        raise ValueError(f"Invalid dotted path '{dotted_path}'")

    module = import_module(module_path)
    if hasattr(module, attr_name):
        return getattr(module, attr_name)

    if not hasattr(module, "__path__"):
        raise AttributeError(
            f"Module '{module_path}' does not expose attribute '{attr_name}'",
        )

    resolved = _search_in_package(module, attr_name, seen=set())
    if resolved is None:
        raise AttributeError(
            f"Unable to locate '{attr_name}' within package '{module_path}'",
        )
    return resolved


def _search_in_package(module, attr_name: str, seen: set[str]) -> Any | None:
    module_name = getattr(module, "__name__", repr(module))
    if module_name in seen:
        return None
    seen.add(module_name)

    if hasattr(module, attr_name):
        return getattr(module, attr_name)

    if not hasattr(module, "__path__"):
        return None

    for _, child_name, _ in pkgutil.iter_modules(module.__path__):
        if child_name.startswith("_"):
            continue
        full_name = f"{module.__name__}.{child_name}"
        try:
            child_module = import_module(full_name)
        except Exception:  # pragma: no cover - defensive fallback
            logger.debug(
                "Failed to import %s while resolving symbol",
                full_name,
                exc_info=True,
            )
            continue

        found = _search_in_package(child_module, attr_name, seen)
        if found is not None:
            return found

    return None


class TorchModuleConfig(BaseModel):
    """Configuration entry for a :class:`nomad.fm_base_tool.TorchModuleTool`."""

    model_class: str
    """Dotted import path for a class with a ``from_pretrained`` constructor."""

    name_or_path: str
    """Model weights URI or legacy source resolved by :class:`nomad.hub.RepoSpec`."""

    # Overrides for TorchModuleTool
    tool_name: str | None = None
    """Optional public MCP tool name override."""

    batch_size: int | None = None
    """Optional maximum batch size override."""

    env: str | list[str] | None = None
    """One PEP 508 requirement or a list for an isolated model environment."""

    model_config = ConfigDict(extra="allow")

    @field_validator("env")
    @classmethod
    def _validate_environment_requirements(
        cls,
        value: str | list[str] | None,
    ) -> str | list[str] | None:
        requirements = [value] if isinstance(value, str) else value or []
        for requirement in requirements:
            if any(character in requirement for character in ("\r", "\n", "\0")):
                raise ValueError(
                    "PEP 508 requirements cannot contain line breaks or NUL bytes"
                )
            if requirement.strip().casefold() in {
                "requirements.txt",
                "pyproject.toml",
            }:
                raise ValueError(
                    "Environment file paths are not supported; use a PEP 508 "
                    "requirement"
                )
            try:
                parsed = Requirement(requirement)
            except InvalidRequirement as exc:
                raise ValueError(
                    f"Invalid PEP 508 requirement: {requirement!r}"
                ) from exc
            if parsed.url is not None:
                url = urlsplit(parsed.url)
                file_scheme = url.scheme == "file" or url.scheme.endswith("+file")
                windows_drive = len(url.scheme) == 1 and parsed.url[1:2] == ":"
                invalid_file_url = file_scheme and not url.path.startswith("/")
                if (
                    not url.scheme
                    or windows_drive
                    or invalid_file_url
                    or (not file_scheme and not url.netloc)
                ):
                    raise ValueError(
                        "PEP 508 direct references must use an absolute URL: "
                        f"{requirement!r}"
                    )
        return value

    def resolve_spec(self, *, base_dir: Path | None = None) -> RepoSpec:
        """Resolve ``name_or_path`` to a normalized model source specification."""
        return RepoSpec.parse(self.name_or_path, base_dir=base_dir)

    def resolve_source(self, *, base_dir: Path | None = None) -> str | Path:
        """Resolve ``name_or_path`` to a loadable local path."""
        return self.resolve_spec(base_dir=base_dir).pull()

    def resolve_environment(self, *, base_dir: Path | None = None) -> PythonEnvironment:
        """Normalize ``env`` for an isolated model subprocess."""
        return resolve_model_environment(self.env, base_dir=base_dir)

    def build_remote(
        self,
        *,
        base_dir: Path | None = None,
        rpc_timeout_seconds: float | None = 30.0,
    ) -> RemoteModel:
        """Inspect this model inside its configured subprocess environment."""
        environment = self.resolve_environment(base_dir=base_dir)
        spec = self.resolve_spec(base_dir=base_dir)
        remote = inspect_remote_model(
            environment=environment,
            model_class=self.model_class,
            source=spec.pull(),
            tool_name=self.tool_name,
            batch_size=self.batch_size,
            timeout_seconds=rpc_timeout_seconds,
        )
        name = remote.name
        if self.tool_name:
            name = self.tool_name
        elif getattr(spec, "scheme", None) == "hf":
            name = spec.location
        elif not name:
            name = spec.location
        return replace(remote, name=sanitize_mcp_name(name))

    def _instantiate_from_spec(self, spec: RepoSpec) -> TorchModuleTool:
        """Instantiate the configured model class from an already parsed source."""
        model_cls = resolve_symbol(self.model_class)
        if not hasattr(model_cls, "from_pretrained"):
            raise TypeError(
                f"{self.model_class} does not expose a 'from_pretrained' constructor",
            )

        source = spec.pull()
        model = model_cls.from_pretrained(str(source))
        if not is_torch_module_tool_instance(model):
            raise TypeError(
                "Instantiated object must be a TorchModuleTool "
                f"(received {type(model)!r})",
            )
        return model

    def instantiate(
        self, *, base_dir: Path | None = None, **overrides
    ) -> TorchModuleTool:
        """Instantiate the configured model class from its configured source."""
        logger.info(
            "Loading foundation model '%s' from '%s'",
            self.model_class,
            self.name_or_path,
        )

        return self._instantiate_from_spec(self.resolve_spec(base_dir=base_dir))

    def build_tool(
        self, *, base_dir: Path | None = None, **overrides
    ) -> TorchModuleTool:
        """Instantiate the tool and apply name and batching overrides."""
        spec = self.resolve_spec(base_dir=base_dir)
        tool = self._instantiate_from_spec(spec)

        # Apply overrides
        if self.tool_name:
            logger.debug("Overriding tool name to '%s'", self.tool_name)
            tool.name = self.tool_name
        elif getattr(spec, "scheme", None) == "hf":
            tool.name = spec.location
        elif not tool.name:
            tool.name = spec.location
        assert tool.name is not None
        tool.name = sanitize_mcp_name(tool.name)
        if self.batch_size is not None:
            tool.batch_size = self.batch_size

        return tool

    def add_to_fastmcp(self, server: FastMCP, *, base_dir: Path | None = None):
        """Instantiate and register this model tool on a FastMCP server."""
        logger.info(
            "Registering foundation model '%s'",
            self.tool_name or self.name_or_path,
        )
        model = self.build_tool(base_dir=base_dir)
        return add_torch_module_tool_to_fastmcp(server, model)


class ToolConfig(BaseModel):
    """Configuration entry for a regular Python callable exposed as a tool."""

    tool: str
    """Dotted import path for the callable or callable factory."""

    name: str | None = None
    """Optional public MCP tool name."""

    tool_kwargs: dict | None = Field(default=None)
    """Optional keyword arguments used to instantiate callable factories."""

    model_config = ConfigDict(extra="allow")

    @property
    def fn(self):
        """Resolve and optionally instantiate the configured callable."""
        fn = resolve_symbol(self.tool)
        if self.tool_kwargs:
            fn = fn(**self.tool_kwargs)
        return fn

    @model_validator(mode="after")
    def set_name(self):
        if self.name is None:
            sanitized = sanitize_mcp_name(self.tool)
            logger.debug("Sanitizing tool name for '%s' -> '%s'", self.tool, sanitized)
            self.name = sanitized
        return self

    @model_validator(mode="before")
    @classmethod
    def simple_tool(cls, tool: str | dict) -> dict:
        if isinstance(tool, str):
            return {
                "tool": tool,
                "name": sanitize_mcp_name(tool.rsplit(".", 1)[-1]),
            }
        return tool

    def add_to_fastmcp(self, server: FastMCP):
        """Register this callable on a FastMCP server."""
        logger.info("Registering tool '%s'", self.tool)
        server.add_tool(FastMCPTool.from_function(self.fn, name=self.name))


class ToolManagerConfig(BaseModel):
    """Configuration for Torch model scheduling and subprocess lifecycle."""

    idle_seconds: float | None = Field(default=300.0, ge=0)
    """Idle seconds before reducing a tool's device allocation by one slot. ``None`` disables device-slot idle eviction."""

    max_pending_per_tool: int | None = Field(default=None, ge=1)
    """Maximum queued requests per tool. ``None`` disables the queue limit."""

    max_devices_per_tool: int | None = Field(default=None, ge=1)
    """Maximum device slots one tool may occupy. ``None`` uses all managed slots."""

    gc_idle_seconds: float | None = Field(default=300.0, ge=0)
    """Server idle seconds before clearing unused Python and accelerator caches. ``None`` disables cache clearing."""

    disk_idle_seconds: float | None = Field(default=600.0, ge=0)
    """Tool idle seconds after full offload before dropping the resident instance. ``None`` keeps resident instances loaded."""

    venv_idle_seconds: float | None = Field(default=1200.0, ge=0)
    """Idle seconds before terminating an isolated model subprocess. ``None`` disables subprocess idle eviction."""

    rpc_timeout_seconds: float | None = Field(default=30.0, gt=0)
    """Maximum seconds for a model subprocess RPC. ``None`` disables the deadline."""

    model_config = ConfigDict(extra="forbid")

    def instantiate(self, *, device_provider=None, **overrides):
        """Create a :class:`nomad.torch_tool_manager.TorchModelToolManager`."""
        from .torch_tool_manager import TorchModelToolManager

        config = self
        if overrides:
            data = self.model_dump()
            data.update(overrides)
            config = type(self).model_validate(data)
        logger.debug("Initializing TorchModelToolManager")
        logger.debug(
            "TorchModelToolManager options: %s",
            config.model_dump(),
        )
        return TorchModelToolManager(config, device_provider=device_provider)


class TelemetryConfig(BaseModel):
    """OpenTelemetry exporter configuration for ``nomad serve``."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool | None = None
    """Enable OTel exporter setup. ``None`` enables it when OTel env vars exist."""

    service_name: str = "nomad"
    """OpenTelemetry service.name for server spans and metrics."""

    otlp_endpoint: str | None = None
    """Optional OTLP/HTTP endpoint. Environment variables are used when unset."""


class SearchWeights(BaseModel):
    """Weights for tool search ranking."""

    model_config = ConfigDict(extra="forbid")

    tantivy_score: float = 1.0
    name_field: float = 3.0
    import_path_field: float = 2.0
    title_field: float = 1.5
    description_field: float = 1.0
    arguments_field: float = 1.25
    exact_name: float = 12.0
    prefix_name: float = 8.0
    suffix_name: float = 4.0
    substring_name: float = 2.0


class SearchToolConfig(BaseModel):
    """Search behavior and ranking configuration."""

    model_config = ConfigDict(extra="forbid")

    expose: bool = False
    """Expose a ``search_tools`` tool when launched with ``nomad serve``."""

    candidate_limit: int = Field(default=50, ge=1)
    """Upper limit on Tantivy candidates retrieved during ranked search."""

    weights: SearchWeights = Field(default_factory=SearchWeights)
    """Factors controlling search result rankings."""


class ServerConfig(BaseModel):
    """Top-level configuration consumed by ``nomad serve``."""

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)
    context_dir: Path = Field(default_factory=Path.cwd, exclude=True, frozen=True)
    """Directory used to resolve paths relative to the config file."""

    fmod_models: list[TorchModuleConfig] = Field(default_factory=list)
    """Torch model tools to load and expose."""

    tools: list[ToolConfig] = Field(default_factory=list)
    """Regular Python tools to import and expose."""

    tool_manager: ToolManagerConfig = Field(default_factory=ToolManagerConfig)
    """Torch model scheduling and device management options."""

    search_tool: SearchToolConfig = Field(default_factory=SearchToolConfig)
    """Optional MCP tool discovery settings."""

    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    """OpenTelemetry exporter settings for server metrics and tracing."""

    @model_validator(mode="before")
    @classmethod
    def _reject_legacy_search_tool_flag(
        cls, values: dict[str, Any] | Any
    ) -> dict[str, Any] | Any:
        if isinstance(values, dict) and "expose_search_tool" in values:
            raise ValueError(
                "Server config no longer supports 'expose_search_tool'. "
                "Use 'search_tool.expose' instead."
            )
        return values

    def instantiate_model(
        self, fmod: TorchModuleConfig, **overrides
    ) -> TorchModuleTool:
        """Instantiate one model entry relative to this config file."""
        return fmod.instantiate(base_dir=self.context_dir, **overrides)

    def build_tool(self, fmod: TorchModuleConfig, **overrides) -> TorchModuleTool:
        """Instantiate one model entry and apply configured tool overrides."""
        return fmod.build_tool(base_dir=self.context_dir, **overrides)

    def build_remote(self, fmod: TorchModuleConfig) -> RemoteModel:
        """Inspect one subprocess-backed model entry."""
        return fmod.build_remote(
            base_dir=self.context_dir,
            rpc_timeout_seconds=self.tool_manager.rpc_timeout_seconds,
        )

    @classmethod
    def from_file(cls, path: str | Path) -> ServerConfig:
        """Load a YAML or JSON server config from disk."""
        path = Path(path)
        logger.info("Loading server configuration from %s", path)
        path, data = load_config_mapping(
            path,
            supported_suffixes=(".yaml", ".yml", ".json"),
            empty_hint="a mapping with server settings",
        )

        def _validate(data: dict[str, Any]) -> ServerConfig:
            return cls.model_validate(
                {
                    **data,
                    "context_dir": path.resolve().parent,
                }
            )

        return validate_config_data(data, validate=_validate, source=path)
