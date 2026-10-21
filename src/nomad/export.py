from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property
from pathlib import Path
from typing import Annotated, Any, Literal
from uuid import uuid4

import click
import typer
import yaml
from fastmcp.tools import Tool as FastMCPTool

from nomad.common.config_errors import ConfigError
from nomad.common.name_sanitize import sanitize_export_name, sanitize_mcp_name
from nomad.config import ServerConfig, TorchModuleConfig
from nomad.hub import RepoSpec
from nomad.logging_utils import configure_root_logging, parse_log_level
from nomad.model_cards import ModelCardLocator
from nomad.model_env import ModelEnvironment

LogLevelName = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
LOGGER = logging.getLogger(__name__)
ORAS_TAG_MAX_CHARS = 128
ORAS_TAG_SUFFIX_MIN_CHARS = 8
ORAS_TAG_SUFFIX_MAX_CHARS = 16


class ExportTarget(StrEnum):
    """Supported destinations for `nomad export` model sources."""

    DISK = "disk"
    HTTPS = "https"
    ORAS = "oras"

    def pushable(self) -> bool:
        return self in {ExportTarget.DISK, ExportTarget.ORAS}


@dataclass
class ModelSourceExport:
    source: RepoSpec
    destination: RepoSpec | None = None

    @cached_property
    def resolved(self) -> RepoSpec:
        return self.source.resolved()

    def export(
        self,
        *,
        to: ExportTarget,
        output_dir: Path,
        oras_repository: str | None,
        pin: bool,
        used_names: set[str],
    ) -> None:
        dst = None
        if to == ExportTarget.DISK:
            uniq_name = _choose_unique_name(self.source.name(), used_names=used_names)
            dst = RepoSpec.parse((output_dir / "models" / uniq_name).as_uri())
        elif to == ExportTarget.HTTPS:
            if not self.source.is_remote:
                uniq_name = _choose_unique_name(
                    self.source.name(), used_names=used_names
                )
                dst = RepoSpec.parse((output_dir / "models" / uniq_name).as_uri())
            elif self.source.scheme == "git+ssh":
                dst = self.source.as_https()
            else:
                dst = self.source
        elif to == ExportTarget.ORAS:
            uniq_name = _choose_unique_name(self.source.name(), used_names=used_names)
            tag = _oras_artifact_tag(
                artifact_name=uniq_name,
                source_tag=_oras_tag(self.resolved),
            )
            dst = RepoSpec.parse(f"oras://{oras_repository}:{tag}")
        else:
            raise ValueError(f"Unsupported export target: {to}")

        should_push = to.pushable() or (
            to == ExportTarget.HTTPS and not self.source.is_remote
        )
        dst_resolved = None
        if should_push and self.resolved != dst:
            src = self.resolved.pull()
            if dst.is_remote and to != ExportTarget.HTTPS:
                LOGGER.info("Pushing model (%s) to %s", src.name, dst.uri())
            dst_resolved = dst.push(src)

        if pin:
            self.destination = dst_resolved or dst.resolved()
        else:
            self.destination = dst


@dataclass
class ModelEntryExport:
    raw_entry: dict[str, Any]
    export_id: str

    def comments(
        self, exports: dict[str, ModelSourceExport], *, rewritten_source: str
    ) -> list[str]:
        export = exports[self.export_id]
        assert export.destination is not None
        out = []
        if self.raw_entry["name_or_path"] != rewritten_source:
            out.append(f"source: {self.raw_entry['name_or_path']}")
        if export.resolved.uri() != rewritten_source:
            out.append(f"uri: {export.resolved.uri()}")

        return out


def _render_exported_yaml(data: dict[str, Any], model_comments: list[list[str]]) -> str:
    rendered: list[str] = []

    for key, value in data.items():
        if key != "fmod_models":
            rendered.append(
                yaml.safe_dump(
                    {key: value}, sort_keys=False, allow_unicode=True
                ).rstrip()
            )
            continue

        if not value:
            rendered.append("fmod_models: []")
            continue

        rendered.append("fmod_models:")
        for index, entry in enumerate(value):
            for comment in model_comments[index]:
                rendered.append(f"  # {comment}")
            item_lines = (
                yaml.safe_dump([entry], sort_keys=False, allow_unicode=True)
                .rstrip()
                .splitlines()
            )
            rendered.extend(f"  {line}" for line in item_lines)

    return "\n".join(rendered) + "\n"


def _collect_model_exports(
    raw_models: list[Any],
    *,
    base_dir: Path,
) -> tuple[list[ModelEntryExport], dict[str, ModelSourceExport]]:
    entry_exports: list[ModelEntryExport] = []
    source_exports: dict[str, ModelSourceExport] = {}

    for raw_entry in raw_models:
        if not isinstance(raw_entry, dict):
            continue

        original_source = raw_entry.get("name_or_path")
        if not isinstance(original_source, str):
            raise ValueError("Model configuration requires 'name_or_path'")

        spec = RepoSpec.parse(original_source, base_dir=base_dir)
        source_export = ModelSourceExport(spec)
        export_id = source_export.resolved.uri()
        source_exports[export_id] = source_export

        entry_exports.append(
            ModelEntryExport(
                raw_entry=raw_entry,
                export_id=export_id,
            )
        )

    return entry_exports, source_exports


def _oras_tag(spec: RepoSpec) -> str:
    try:
        tag_source = spec.resolved().cache_digest()
    except ValueError:
        tag_source = uuid4().hex
    return sanitize_export_name(tag_source)[:ORAS_TAG_MAX_CHARS]


def _oras_artifact_tag(*, artifact_name: str, source_tag: str) -> str:
    suffix_chars = min(
        ORAS_TAG_SUFFIX_MAX_CHARS,
        ORAS_TAG_MAX_CHARS - len(artifact_name) - 1,
    )
    if suffix_chars >= ORAS_TAG_SUFFIX_MIN_CHARS:
        return f"{artifact_name}-{source_tag[:suffix_chars]}"

    LOGGER.warning(
        "ORAS artifact name '%s' is too long to append a source tag suffix; "
        "using source tag '%s' as the artifact tag",
        artifact_name,
        source_tag,
    )
    return source_tag


def _choose_unique_name(raw_name: str, *, used_names: set[str]) -> str:
    base_name = sanitize_export_name(raw_name)
    candidate = base_name
    suffix_number = 2
    while candidate in used_names:
        candidate = f"{base_name}-{suffix_number}"
        suffix_number += 1
    used_names.add(candidate)
    return candidate


def _oras_repository(oras_registry: str | None) -> str:
    if oras_registry is None:
        raise ValueError("--oras-registry is required when exporting to ORAS")

    repository = oras_registry.removeprefix("oras://").strip("/")
    if not repository:
        raise ValueError("--oras-registry cannot be empty")
    return repository


def _validate_unique_tool_names(raw_models: list[Any]) -> None:
    used_tool_names: dict[str, tuple[int, str]] = {}
    for index, raw_entry in enumerate(raw_models):
        if not isinstance(raw_entry, dict):
            continue
        tool_name = raw_entry.get("tool_name")
        if not isinstance(tool_name, str):
            continue

        mcp_name = sanitize_mcp_name(tool_name)
        previous = used_tool_names.get(mcp_name)
        if previous is not None:
            previous_index, previous_tool_name = previous
            raise ValueError(
                f"Exported model entries {previous_index + 1} and {index + 1} "
                f"use duplicate MCP tool name '{mcp_name}' from tool_name values "
                f"'{previous_tool_name}' and '{tool_name}'. Set explicit unique "
                "tool_name values before exporting."
            )
        used_tool_names[mcp_name] = (index, tool_name)


def _build_exported_model_environments(
    models: list[TorchModuleConfig],
    *,
    base_dir: Path,
    cache_root: Path,
) -> None:
    """Materialize explicit model environments inside an export bundle."""
    for model in models:
        if model.env is None:
            continue
        environment = model.resolve_environment(base_dir=base_dir)
        assert isinstance(environment, ModelEnvironment)
        environment.ensure(cache_root=cache_root)


def export_models_config(
    config_path: Path,
    output_dir: Path,
    *,
    to: ExportTarget = ExportTarget.DISK,
    oras_registry: str | None = None,
    pin: bool = True,
    build_venvs: bool = True,
) -> Path:
    """Export local assets and rewrite ``nomad.yml`` for the requested target."""
    config_path = config_path.expanduser()
    config = ServerConfig.from_file(config_path)
    exported_data = config.model_dump(mode="python", exclude_none=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    used_names: set[str] = set()
    oras_repository = (
        _oras_repository(oras_registry) if to == ExportTarget.ORAS else None
    )

    raw_models = exported_data.get("fmod_models", []) or []
    if build_venvs:
        _build_exported_model_environments(
            config.fmod_models,
            base_dir=config.context_dir,
            cache_root=output_dir,
        )
    entry_exports, source_exports = _collect_model_exports(
        raw_models,
        base_dir=config.context_dir,
    )

    if to == ExportTarget.ORAS and oras_repository is None:  # pragma: no cover
        raise ValueError("--oras-registry is required when exporting to ORAS")

    for source_export in source_exports.values():
        source_export.export(
            to=to,
            output_dir=output_dir,
            oras_repository=oras_repository,
            pin=pin,
            used_names=used_names,
        )

    model_comments: list[list[str]] = []
    for entry_export in entry_exports:
        source_export = source_exports[entry_export.export_id]
        if source_export.destination is None:  # pragma: no cover
            raise ValueError("Model source was not exported")

        rewritten_source = source_export.destination.uri(base_dir=output_dir)
        model_comments.append(
            entry_export.comments(source_exports, rewritten_source=rewritten_source)
        )
        entry_export.raw_entry["name_or_path"] = rewritten_source
        entry_export.raw_entry["tool_name"] = entry_export.raw_entry.get(
            "tool_name", source_export.source.name()
        )

    _validate_unique_tool_names(raw_models)

    config_output = output_dir / "nomad.yml"
    payload = _render_exported_yaml(exported_data, model_comments)
    config_output.write_text(payload, encoding="utf-8")
    return config_output


def export_model_report(config_path: Path, output_dir: Path) -> Path:
    """Export model cards and a categorized index of configured tools."""
    config_path = config_path.expanduser()
    config = ServerConfig.from_file(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    locator = ModelCardLocator()
    model_entries: list[tuple[str, str, Path]] = []
    used_names: set[str] = set()

    for fm_config in config.fmod_models:
        tool = config.build_remote(fm_config)
        assert tool.name is not None
        if tool.name in used_names:
            raise ValueError(f"Duplicate model tool name '{tool.name}' in report")
        used_names.add(tool.name)

        source = Path(tool.source)
        locator.register(tool.name, source)
        card_path = output_dir / f"{tool.name}.md"
        card_path.write_text(locator.read_model_card(tool.name), encoding="utf-8")
        model_entries.append((tool.name, tool.description, card_path))

    regular_entries = []
    for tool_config in config.tools:
        tool = FastMCPTool.from_function(tool_config.fn, name=tool_config.name)
        regular_entries.append((tool.name, tool.description or ""))

    lines = ["# Nomad Tool Report", "", "## SciFM Tools"]
    for tool_name, description, card_path in model_entries:
        lines.extend(
            [
                "",
                f"### [{tool_name}]({card_path.name})",
                "",
                description,
            ]
        )

    lines.extend(["", "## Other Tools"])
    for tool_name, description in regular_entries:
        lines.extend(["", f"### {tool_name}", "", description])

    readme = output_dir / "README.md"
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return readme


def export(
    config: Annotated[
        Path,
        typer.Argument(
            help="Path to the Nomad server config file to export.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Argument(
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help=(
                "Directory where Nomad should write bundled models and the "
                "rewritten `nomad.yml`."
            ),
        ),
    ],
    log_level: Annotated[
        LogLevelName,
        typer.Option(
            "--log-level",
            help="Minimum log level for export progress messages.",
        ),
    ] = "INFO",
    to: Annotated[
        ExportTarget,
        typer.Option(
            "--to",
            help=(
                "Where exported model sources should point: `disk` copies all "
                "models into the bundle, while `https` keeps remote URI "
                "sources remote and rewrites git+ssh repos to git+https, and "
                "`oras` pushes models to an ORAS registry."
            ),
        ),
    ] = ExportTarget.DISK,
    oras_registry: Annotated[
        str | None,
        typer.Option(
            "--oras-registry",
            help=(
                "Existing ORAS artifact repository for `--to oras`, such as "
                "`registry.example.org/scifm`."
            ),
        ),
    ] = None,
    pin: Annotated[
        bool,
        typer.Option(
            "--pin/--no-pin", help="Pin exported sources to a resolved reference"
        ),
    ] = True,
    report: Annotated[
        bool,
        typer.Option(
            "--report",
            help="Export model cards and a linked README instead of a deployment bundle.",
        ),
    ] = False,
    venv: Annotated[
        bool,
        typer.Option(
            "--venv/--no-venv",
            help="Build explicit model environments under `<output>/venv`.",
        ),
    ] = True,
):
    """Create a deployment bundle from a Nomad config.

    Nomad copies or downloads every configured model into `<output>/models` and
    builds explicit model environments under `<output>/venv`, then writes a
    rewritten `<output>/nomad.yml`. Use `--no-venv` to skip environment
    construction. Use `--to https` to keep remote models remote while rewriting
    git+ssh sources to git+https transport. Use `--to oras --oras-registry
    REGISTRY/REPOSITORY` to push models to ORAS and rewrite the config to pinned
    ORAS artifact URIs.
    """
    numeric_level = parse_log_level(log_level)
    configure_root_logging(stderr_level=numeric_level)
    LOGGER.setLevel(numeric_level)
    try:
        if report:
            export_model_report(config, output)
        else:
            export_models_config(
                config,
                output,
                to=to,
                oras_registry=oras_registry,
                pin=pin,
                build_venvs=venv,
            )
    except ConfigError as exc:
        raise click.ClickException(f"Failed to load config: {exc}") from exc
    except (RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
