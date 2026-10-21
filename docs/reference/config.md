# Configuration Guide

Use this page for hand-written configuration examples and operational notes.
For exact model fields, see {doc}`api-config`.

Nomad expands environment placeholders in string config values before schema
validation. Use `${VAR}` to read an environment variable and
`${VAR:DEFAULT}` to supply a fallback value when it is unset. This is the same
mechanism used by examples such as `Bearer ${NOMAD_API_KEY}`.
Implementation: {repo_file}`src/nomad/common/env.py <src/nomad/common/env.py>`.

## Nomad server config

`nomad serve` reads a YAML or JSON file into
{py:class}`nomad.config.ServerConfig`.

```yaml
tool_manager:
  idle_seconds: 120
  gc_idle_seconds: 300
  disk_idle_seconds: 600
  venv_idle_seconds: 1200
  rpc_timeout_seconds: 30
  max_pending_per_tool: 50000
search_tool:
  expose: true
telemetry:
  enabled: true
  service_name: nomad
  otlp_endpoint: http://127.0.0.1:4318
tools:
  - my_package.tools.health_check
fmod_models:
  - model_class: my_package.models.MyTorchTool
    name_or_path: my-org/my-model
    env: my-model-package>=1.2
    tool_name: my-model
    batch_size: 16
```

Use `tools` for regular Python callables and `fmod_models` for
{py:class}`nomad.fm_base_tool.TorchModuleTool` implementations. `name_or_path`
can point at several model source types:

Set a model's `env` to one PEP 508 requirement string or a list of PEP 508
requirements. `requirements.txt` and `pyproject.toml` paths are not supported.
Direct references must use absolute URLs, such as
`my-package @ file:///opt/packages/my-package`.

Models run in per-device Python subprocesses. When `env` is omitted, the
subprocess uses the same virtual environment and Python executable as the Nomad
host. For explicit environments, Nomad prefers `uv` to create the virtual
environment and falls back to the standard-library `venv` module. Explicit
environments are content-addressed under
`$XDG_CACHE_HOME/nomad/venv` (or `~/.cache/nomad/venv`) and reused across
models with the same requirements. Nomad injects the currently running Nomad
package into each explicit environment's requirements. A slot keeps its
subprocess when switching between models with the same environment;
`tool_manager.venv_idle_seconds` controls when an idle subprocess is terminated
and defaults to 1200 seconds. `tool_manager.rpc_timeout_seconds` limits each
subprocess request to 30 seconds by default; set it to `null` to disable the
deadline. The tool manager is always enabled and controls batching, device
assignment, and model subprocess lifecycles.

`nomad export` leaves requirement strings and lists unchanged and builds their
explicit environments under `<output>/venv` by default, without loading the
models. Set `NOMAD_CACHE=<output>` when serving the exported config so those
environments are reused. Build the bundle at its final absolute deployment path
(for example, `nomad export nomad.yml /opt/nomad`): virtual environments are not
relocatable and must run with a Python runtime compatible with the one used for
the export. Pass `--no-venv` to omit them from an export.

| Name | Example |
| --- | --- |
| Local directory URI | `file:models/my-model` |
| Hugging Face URI | `hf://my-org/my-model` |
| ORAS artifact | `oras://registry.example.org/my-org/my-model:v1` |
| Git/Git LFS HTTPS URI | `git+https://example.org/my-org/models.git@main` |
| Git/Git LFS SSH URI | `git+ssh://git@example.org/my-org/models.git@main` |
| Local path[^path-or-hf] | `models/my-model` |
| Hugging Face repo ID[^path-or-hf] | `my-org/my-model` |
| Plain HTTPS Git URL[^compat-uri] | `https://example.org/my-org/models.git@main` |
| SCP-style Git URL[^compat-uri] | `git@example.org:my-org/models.git@main` |

Add `#path/to/dir` to any source when the model files live in a subdirectory,
for example `hf://my-org/my-model#weights` or
`git+https://example.org/my-org/models.git@main#weights`.

[^compat-uri]: Accepted for compatibility and resolved to URI forms. Plain
    `http://` Git URLs are handled the same way as `https://`.

[^path-or-hf]: Checked as local paths first. If no local path exists, Nomad
    treats them as Hugging Face repo IDs.

`nomad serve` emits OpenTelemetry metrics for model-serving operations when OTel
export is enabled. See {doc}`otel` for telemetry setup and metric reference
tables.

## Code-mode gateway config

`nomad code-mode` and `nomad code-mode-exec` read YAML, TOML, or JSON into
{py:class}`nomad.gateway.config.GatewayConfig`.

```yaml
servers:
  nomad:
    transport: http
    url: http://127.0.0.1:8181/mcp
defaults:
  timeout_seconds: 60
  stdout_limit: 65536
  stderr_limit: 65536
  result_limit: 65536
telemetry:
  enabled: true
  service_name: nomad-gateway
  otlp_endpoint: http://127.0.0.1:4318
```

`transport` is `stdio` or `http` in gateway config files. Legacy
`streamable-http` and `streamable_http` values are still accepted as aliases
for `http`.
The `workspace_root` default controls where sandboxed scripts run. When omitted,
a gateway using the `stdio` transport defaults it to the gateway process's
current working directory. Other transports use a temporary workspace. If the
workspace contains `.venv/bin/python` or `.venv/Scripts/python.exe`, the gateway
uses it for sandbox execution.

The gateway emits OpenTelemetry metrics and spans for MCP tool entrypoints,
sandbox runs, and upstream MCP tool calls. See {doc}`otel` for the full metric
reference.

Nomad configures OTLP/HTTP trace and metric exporters when `telemetry.enabled`
is `true`, `NOMAD_OTEL_ENABLED=true`, or standard OTel environment variables
such as `OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`, or
`OTEL_EXPORTER_OTLP_METRICS_ENDPOINT`, `OTEL_TRACES_EXPORTER`, or
`OTEL_METRICS_EXPORTER` are set. Set `OTEL_SDK_DISABLED=true` to disable SDK
setup. Install `nomad-scifm[otel]` to include the optional OTel SDK and OTLP exporter
dependencies. See {doc}`otel` for links to OTel exporter documentation.

## Middleware

Gateway middleware is configured with the `middleware` list on
{py:class}`nomad.gateway.config.GatewayConfig`.

```yaml
middleware:
  - kind: allowlist
    allow:
      - nomad:*
    deny:
      - nomad:dangerous_*
  - kind: redaction
    mode: basic
  - kind: logging
    level: info
```

Allowlist patterns match `server:tool`. Redaction currently supports `basic`
and `none`.
