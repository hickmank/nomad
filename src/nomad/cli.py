import asyncio
import json
import logging
import os
import sys
from collections.abc import Sequence
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Annotated, Any, Literal

import click
import typer
from fastmcp import FastMCP
from fastmcp.mcp_config import StdioMCPServer
from typer.core import TyperCommand

from ._torch_module_compat import add_torch_module_tool_to_fastmcp
from .common.config_errors import ConfigError
from .common.upstream_errors import UpstreamConnectionError
from .config import ServerConfig
from .export import export as export_command
from .gateway import cli as gateway_cli
from .gateway.config import GatewayConfig, GatewayDefaults
from .gateway.server import CodeModeGateway
from .logging_utils import configure_root_logging, parse_log_level
from .model_cards import ModelCardLocator, register_model_card_tool
from .otel import configure_otel, shutdown_otel
from .tool_search import register_search_tool

LogLevelName = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
ServeTransport = Literal["stdio", "http", "streamable-http", "streamable_http"]

app = typer.Typer(
    no_args_is_help=True,
    help=(
        "Serve SciFM-backed MCP tools, export self-contained deployment bundles, "
        "and run scripts through Nomad's code-mode gateway."
    ),
    rich_markup_mode="markdown",
)
app.command(
    "export",
    short_help="Bundle models and rewrite a deployable config.",
)(export_command)

LOGGER = logging.getLogger(__name__)
try:
    _PACKAGE_VERSION = version("nomad-scifm")
except PackageNotFoundError:  # pragma: no cover - local checkout
    _PACKAGE_VERSION = "0.0.0"
_CODE_MODE_EXEC_CONTEXT_SETTINGS = {
    "allow_extra_args": True,
}


def _normalize_serve_transport(transport: str) -> str:
    normalized = transport.replace("_", "-")
    if normalized == "streamable-http":
        return "http"
    return normalized


def _format_visible_devices(manager: Any) -> str:
    accelerator_info = getattr(manager, "accelerator_info", None)
    if accelerator_info:
        formatted: list[str] = []
        for info in accelerator_info:
            device = getattr(info, "device", info)
            name = getattr(info, "name", None)
            if name and name != str(device):
                formatted.append(f"{device} ({name})")
            else:
                formatted.append(str(device))
        return ", ".join(formatted)

    devices = getattr(manager, "devices", None)
    if devices:
        return ", ".join(str(device) for device in devices)
    return "none"


def _local_nomad_gateway_config(
    config_path: Path,
    workspace_root: Path,
) -> GatewayConfig:
    return GatewayConfig(
        servers={
            "nomad": StdioMCPServer(
                command=sys.executable,
                args=[
                    "-m",
                    "nomad",
                    "serve",
                    str(config_path),
                    "--transport",
                    "stdio",
                ],
                env=os.environ.copy(),
                keep_alive=False,
            )
        },
        defaults=GatewayDefaults(workspace_root=workspace_root),
    )


def run_code_mode_script(
    config_path: Path | None,
    script_path: Path,
    gateway_log_level: str,
    directory: Path = Path.cwd(),
    script_args: Sequence[str] = (),
    nomad_config_path: Path | None = None,
) -> dict[str, Any]:
    """Execute a Python script inside the code-mode gateway sandbox."""

    gateway_cli._configure_logging(gateway_log_level)
    if config_path is not None and nomad_config_path is not None:
        raise ValueError("Gateway and Nomad server configs are mutually exclusive")
    if config_path is not None:
        try:
            gateway_config = GatewayConfig.from_file(config_path)
        except ConfigError as exc:
            raise typer.BadParameter(
                f"Failed to load gateway config '{config_path}': {exc}"
            ) from exc
        gateway_config.defaults.workspace_root = directory
    elif nomad_config_path is not None:
        try:
            ServerConfig.from_file(nomad_config_path)
        except ConfigError as exc:
            raise typer.BadParameter(
                f"Failed to load Nomad config '{nomad_config_path}': {exc}"
            ) from exc
        gateway_config = _local_nomad_gateway_config(nomad_config_path, directory)
    else:  # pragma: no cover - guarded by the CLI
        raise ValueError("A gateway or Nomad server config is required")
    gateway_telemetry = getattr(gateway_config, "telemetry", None)
    configure_otel(
        service_name=getattr(gateway_telemetry, "service_name", "nomad-gateway"),
        service_version=_PACKAGE_VERSION,
        enabled=getattr(gateway_telemetry, "enabled", None),
        otlp_endpoint=getattr(gateway_telemetry, "otlp_endpoint", None),
    )

    async def _execute() -> dict[str, Any]:
        async with CodeModeGateway(gateway_config) as gateway:
            result = await gateway.run_script(
                script_path,
                script_args=script_args,
                timeout_seconds=None,
                capture_stdio=False,
            )
            payload = asdict(result)
            payload.pop("stdout", None)
            payload.pop("stderr", None)
            return payload

    try:
        return asyncio.run(_execute())
    except UpstreamConnectionError as exc:
        raise click.ClickException(str(exc)) from exc
    except TimeoutError:
        return {
            "result": {"error": "timeout"},
            "tool_calls": [],
        }
    finally:
        shutdown_otel()


@app.command(
    short_help="Serve Nomad tools over stdio or HTTP.",
)
def serve(
    config: Annotated[
        Path,
        typer.Argument(
            help="Path to the Nomad server config file, usually `nomad.yml`.",
            envvar="NOMAD_SERVE_CONFIG",
        ),
    ],
    log_level: Annotated[
        LogLevelName,
        typer.Option(
            "--log-level",
            help="Minimum log level written to stderr.",
            envvar="NOMAD_SERVE_LOG_LEVEL",
        ),
    ] = "INFO",
    transport: Annotated[
        ServeTransport,
        typer.Option(
            "--transport",
            "-t",
            case_sensitive=False,
            envvar="NOMAD_SERVE_TRANSPORT",
            help=(
                "Server transport. Use `stdio` for local MCP clients or "
                "`http` for network clients."
            ),
        ),
    ] = "stdio",
    use_tool_manager: Annotated[
        bool,
        typer.Option(
            "--tool-manager/--no-tool-manager",
            help=(
                "Enable Nomad's PyTorch tool manager for batching and shared "
                "accelerator scheduling."
            ),
        ),
    ] = True,
    host: Annotated[
        str,
        typer.Option(
            "--host",
            envvar="NOMAD_SERVE_HOST",
            help="Host interface for `http`. Ignored for `stdio`.",
        ),
    ] = "localhost",
    port: Annotated[
        int,
        typer.Option(
            "--port",
            help="Port for `http`. Ignored for `stdio`.",
            envvar="NOMAD_SERVE_PORT",
        ),
    ] = 8000,
    log_file: Annotated[
        Path | None,
        typer.Option(
            "--log-file",
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
            help="Optional JSONL file for structured append-only logs.",
            envvar="NOMAD_SERVE_LOG_FILE",
        ),
    ] = None,
    strict: Annotated[
        bool,
        typer.Option(
            "--strict/--no-strict",
            help="Fail startup if any model fails to load or register.",
            envvar="NOMAD_SERVE_STRICT",
        ),
    ] = True,
):
    """Start a Nomad MCP server from a config file.

    The config determines which MCP tools, model-backed tools, model cards, and
    optional search endpoints are registered. Use `stdio` for local clients and
    `http` when exposing Nomad over the network.
    """

    transport = _normalize_serve_transport(transport)
    numeric_level = parse_log_level(log_level)
    configure_root_logging(stderr_level=numeric_level, log_file=log_file)
    LOGGER.setLevel(numeric_level)
    try:
        config: ServerConfig = ServerConfig.from_file(config)
    except ConfigError as exc:
        raise click.ClickException(f"Failed to load config: {exc}") from exc
    telemetry = getattr(config, "telemetry", None)
    configure_otel(
        service_name=getattr(telemetry, "service_name", "nomad"),
        service_version=_PACKAGE_VERSION,
        enabled=getattr(telemetry, "enabled", None),
        otlp_endpoint=getattr(telemetry, "otlp_endpoint", None),
    )

    server = FastMCP("nomad", on_duplicate="warn")
    card_locator = ModelCardLocator()
    register_model_card_tool(server, card_locator)

    for tool in config.tools:
        tool.add_to_fastmcp(server)

    manager_cfg = config.tool_manager
    use_manager = use_tool_manager and manager_cfg.enabled

    if not manager_cfg.enabled and use_tool_manager:
        LOGGER.info("Tool manager disabled by configuration")

    manager = manager_cfg.instantiate() if use_manager else None
    LOGGER.info("Visible devices: %s", _format_visible_devices(manager))
    for fm_config in config.fmod_models:
        fm_name = fm_config.tool_name or fm_config.name_or_path
        try:
            tool = config.build_tool(fm_config)

            source = fm_config.resolve_source(base_dir=config.context_dir)

            LOGGER.info("Registering torch model '%s'", fm_name)

            if manager:
                manager.register_tool(
                    tool.name,
                    tool,
                    source=source,
                )
            else:
                add_torch_module_tool_to_fastmcp(server, tool)

            card_locator.register(
                tool.name or fm_config.name_or_path,
                source,
            )
        except Exception:
            LOGGER.exception("Failed to load torch model: `%s`", fm_name)
            if strict:
                raise

    if manager:
        manager.add_to_fastmcp(server)

    if config.search_tool.expose:
        register_search_tool(
            server,
            search_config=config.search_tool,
        )

    try:
        run_kwargs: dict[str, Any] = {
            "log_level": log_level,
            "show_banner": False,
        }
        if transport != "stdio":
            run_kwargs.update(
                {
                    "host": host,
                    "port": port,
                    "stateless_http": True,
                }
            )

        server.run(transport=transport, **run_kwargs)
    finally:
        shutdown_otel()


class CmxCommand(TyperCommand):
    def collect_usage_pieces(self, ctx: click.Context) -> list[str]:
        return [*super().collect_usage_pieces(ctx), "[-- SCRIPT_ARGS...]"]


@app.command(
    "code-mode-exec",
    cls=CmxCommand,
    context_settings=_CODE_MODE_EXEC_CONTEXT_SETTINGS,
    short_help="Run a script within the code-mode sandbox. Alias: cmx",
)
@app.command(
    "cmx",
    cls=CmxCommand,
    hidden=True,
    context_settings=_CODE_MODE_EXEC_CONTEXT_SETTINGS,
)
def code_mode_exec(
    ctx: typer.Context,
    script: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Path to the Python script to execute inside the sandbox.",
        ),
    ],
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help=(
                "Path to the code-mode gateway config file. Mutually exclusive "
                "with --nomad-config."
            ),
        ),
    ] = None,
    nomad_config: Annotated[
        Path | None,
        typer.Option(
            "--nomad-config",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help=(
                "Start a local Nomad stdio server from this server config. "
                "Mutually exclusive with --config."
            ),
        ),
    ] = None,
    directory: Annotated[
        Path,
        typer.Option(
            "--directory",
            dir_okay=True,
            file_okay=False,
            resolve_path=True,
            help="Workspace directory exposed to the sandbox.",
            show_default="current directory",
        ),
    ] = Path.cwd(),
    output: Annotated[
        str | None,
        typer.Option(
            "--output",
            "-o",
            help=(
                "Optional destination for the JSON result payload. "
                "Omit to suppress output; use '-' for stdout."
            ),
        ),
    ] = None,
    gateway_log_level: Annotated[
        str,
        typer.Option(
            "--gateway-log-level",
            help="Minimum log level emitted by the temporary gateway process.",
        ),
    ] = logging.getLevelName(logging.WARNING),
) -> None:
    """Run a script within Nomad's code-mode sandbox.

    This is useful for validating a gateway config, testing tool access from a
    script, or capturing a structured JSON result without starting a long-lived
    gateway process. ``--nomad-config`` starts an owned local Nomad server and
    exposes it as ``mcp_tools.nomad``. Arguments after ``--`` are forwarded to
    the `SCRIPT`.
    """

    if config is not None and nomad_config is not None:
        raise click.UsageError("--config and --nomad-config are mutually exclusive.")
    if config is None and nomad_config is None:
        raise click.UsageError("One of --config or --nomad-config is required.")

    result = run_code_mode_script(
        config_path=config,
        nomad_config_path=nomad_config,
        script_path=script,
        gateway_log_level=gateway_log_level,
        directory=directory,
        script_args=list(ctx.args),
    )
    payload = json.dumps(result, indent=2)

    if output is not None:
        if output == "-":
            typer.echo(payload)
        else:
            output_path = Path(output).expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(payload + "\n", encoding="utf-8")


app.command(
    "code-mode",
    short_help="Run the standalone code-mode gateway.",
)(gateway_cli.run)
