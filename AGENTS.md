# Nomad Contributor and Operator Reference

## Purpose

Nomad (`nomad-scifm`) connects AI agents to scientific foundation models
(SciFMs) using the Model Context Protocol (MCP). It has two primary modes:

- `nomad serve` loads regular Python tools and PyTorch-backed model tools from
  a server configuration and serves them over MCP.
- `nomad code-mode` proxies one or more MCP servers into a Python sandbox.
  Sandboxed code imports generated wrappers from `mcp_tools.<server>`.

Nomad is a Python 3.12+ project. The distribution is `nomad-scifm`; the Python
package and CLI are both named `nomad`.

## Local Environment

Nomad is installed in this checkout's `.venv`. Use its interpreter directly
when running Python or the CLI:

```shell
./.venv/bin/python -m nomad --help
./.venv/bin/python -m pytest
```

On Chicoma, create or refresh the environment with:

```shell
module load python
uv sync --all-groups
source .venv/bin/activate
```

`uv run ...` is also supported. The repository's `Justfile` is the source for
standard lint, test, image, and documentation commands.

### Chicoma MCP Inspector

MCP Inspector is a Node.js tool and is not installed with Nomad. On Chicoma,
the Node runtime and npm/npx are supplied by separate modules:

```shell
module load node-js/24.13.0
module load npm/8.19.3
node --version
npm --version
npx --version
```

When working through SSH or tmux, use Inspector's terminal UI instead of its
default web UI, which attempts to open a browser on the remote host:

```shell
npx @modelcontextprotocol/inspector --tui http://localhost:8181/mcp
```

In the TUI, press `C` to connect, use `Tab` or left/right arrows to choose the
`Tools` tab, then use `Enter` to inspect or invoke a selected tool. Press `?`
for context-specific keybindings. `--cli` is intended for scripted,
machine-readable requests; use `--tui` for interactive learning.

## Repository Map

| Path | Role |
| --- | --- |
| `src/nomad/cli.py` | Top-level CLI: `serve`, `export`, `code-mode`, and `code-mode-exec` (`cmx`). |
| `src/nomad/config.py` | Server configuration schemas and tool/model loading. |
| `src/nomad/fm_base_tool.py` | `TorchModuleTool`, the adapter base class for PyTorch models. |
| `src/nomad/torch_tool_manager.py` | GPU scheduling, queuing, batching, and idle offload for model tools. |
| `src/nomad/hub.py` | Resolves and caches local, Hugging Face, Git/LFS, and ORAS model sources. |
| `src/nomad/gateway/` | Code-mode gateway, sandbox, generated wrappers, and middleware. |
| `src/nomad/well_format.py` | Typed tensor and Well-format scientific-state exchange. |
| `container/demo/` | Runnable public demo server, container files, and model adapters. |
| `docs/` | MyST/Sphinx user, deployment, API, and developer documentation. |
| `test/` | Pytest suite; gateway tests live in `test/gateway/`. |
| `skills/connect-scifm-to-nomad/` | Agent workflow for adapting an external SciFM. |

## CLI Workflows

Inspect the installed CLI before relying on exact flags:

```shell
./.venv/bin/python -m nomad --help
./.venv/bin/python -m nomad serve --help
./.venv/bin/python -m nomad code-mode --help
./.venv/bin/python -m nomad code-mode-exec --help
```

Start the demo as an HTTP MCP server from the repository root:

```shell
uv run --directory container/demo nomad serve \
  --transport http --host localhost --port 8181 nomad.yml
```

The MCP endpoint is `http://localhost:8181/mcp`. For a local graphical
environment, inspect it with:

```shell
npx @modelcontextprotocol/inspector http://localhost:8181/mcp
```

For a remote Chicoma session, use the terminal workflow in
[Chicoma MCP Inspector](#chicoma-mcp-inspector) instead.

For a local stdio client, omit the HTTP flags; `serve` defaults to `stdio`.
`--strict` is the default and fails startup when any model cannot load. Use
`--no-strict` only when intentionally allowing other configured tools to run.

Run a script against one or more MCP servers through the gateway:

```shell
./.venv/bin/python -m nomad code-mode-exec \
  --config gateway.yml --output - script.py -- script-arguments
```

`code-mode-exec` executes the script once and writes the final global `RESULT`
as JSON when `--output -` is supplied. `cmx` is an alias. Sandboxed code can
import synchronous and asynchronous wrappers such as:

```python
from mcp_tools.nomad import search_pubchem

RESULT = search_pubchem(molecule="caffeine")
```

Wrapper return values deserialize Nomad tensor media to `torch.Tensor`, but
outer Pydantic objects remain dictionaries. Rebuild them explicitly, for
example with `WellFormat.model_validate(result)`.

## Server Configuration

`nomad serve` reads YAML or JSON into `ServerConfig`; relative model paths are
resolved relative to the configuration file. Environment placeholders in string
values use `${VAR}` or `${VAR:DEFAULT}`.

```yaml
tool_manager:
  enabled: true
  idle_seconds: 120
  max_pending_per_tool: 50000
search_tool:
  expose: true
tools:
  - my_package.tools.health_check
fmod_models:
  - model_class: my_package.models.MyTorchTool
    name_or_path: hf://my-org/my-model
    tool_name: my-model
    batch_size: 16
```

- Use `tools` for CPU-only regular Python callables, referenced by dotted path.
  A mapping may supply `tool`, optional public `name`, and `tool_kwargs` for a
  callable factory.
- Use `fmod_models` only for `TorchModuleTool` implementations. The supported
  entries are `model_class`, `name_or_path`, optional `tool_name`, and optional
  `batch_size`.
- `name_or_path` supports local paths or `file:`, `hf://`, `oras://`,
  `git+https://`, and `git+ssh://` sources; append `#subdirectory` when needed.
  Remote artifacts are downloaded to a cache. Prefer pinned external sources
  for deployments.
- The tool manager batches requests, schedules accelerator slots, offloads idle
  models, and can drop resident CPU instances after `disk_idle_seconds`. Disable
  it only for a concrete reason with `tool_manager.enabled: false` or
  `--no-tool-manager`.
- `search_tool.expose: true` adds the `search_tools` MCP tool. Do not use the
  removed `expose_search_tool` setting.
- Server telemetry can be enabled with `telemetry.enabled: true`,
  `NOMAD_OTEL_ENABLED=true`, or standard OTel exporter environment variables.

## Gateway Configuration

`nomad code-mode` and `code-mode-exec` accept YAML, JSON, or TOML gateway
configuration. Each upstream server uses `stdio` or `http` transport.

```yaml
servers:
  nomad:
    transport: http
    url: http://127.0.0.1:8181/mcp
defaults:
  timeout_seconds: 60
  workspace_root: .
middleware:
  - kind: allowlist
    allow: ["nomad:*"]
  - kind: redaction
    mode: basic
```

The sandbox uses `workspace_root` as its working directory and chooses that
directory's `.venv/bin/python` when present. For a stdio gateway without an
explicit workspace, it defaults to the gateway process's current directory;
other transports use a temporary workspace. Middleware supports `allowlist`,
`redaction`, and `logging`.

## Building a Model Tool

Follow `docs/guides/model-builder.md` and use the bundled
`connect-scifm-to-nomad` skill for external model integrations. A credible
model integration requires:

1. An importable, pip-installable model/server package.
2. A reachable model artifact directory referenced by `name_or_path`.
3. A runtime `README.md` model card based on `container/model-card.md`.
4. A `fmod_models` entry, plus any CPU helper tools under `tools`.

Implement `TorchModuleTool.from_pretrained(name_or_path)` to return a fresh,
fully initialized adapter. Implement typed Pydantic `args_schema` and
`output_schema`, batch-aware `preprocess`, and `postprocess` that yields one
output per input; override `_forward` only when needed. Load models in
`from_pretrained`, not a custom `__init__`. Move tensors to `self.device` in
`preprocess`; one tool instance must use one accelerator only.

Design agent-facing schemas around one scientific query per call. Prefer small,
JSON-friendly inputs and outputs. Use `Tensor`, `TensorField`, `WellFormat`, or
`AutoRegressiveInput` only when scientific tensor or gridded-state semantics
require them. Do not make users manually construct serialized tensor payloads;
validate those tools through a FastMCP client or code-mode script.

Model and tool code must not write to stdout or persistent files. Use logging;
use invocation-scoped `tempfile` files only when unavoidable. Keep
model-specific dependencies out of this root project; add them to the separate
model/deployment package instead.

## Development and Validation

```shell
# Lint formatting and static checks
uv run --group dev just lint

# Full test suite; CI additionally uses -n auto and test-compat
uv run --group dev just test
uv run --group dev just test-compat

# One focused test file or test
./.venv/bin/python -m pytest -q test/test_config.py

# Build HTML documentation (also syncs the demo environment)
uv run --no-dev --group docs just docs

# Preview docs with rebuilds
uv run --no-dev --group docs just docs --live
```

Tests use pytest with automatic asyncio support; GPU-specific tests carry the
`gpu` marker. Linting is driven by `prek` and Ruff. Documentation examples are
tested as config schemas, so update the associated docs when changing config
behavior. Generated documentation output is ignored.

For an end-to-end check, start the demo server, list and invoke a simple tool
with MCP Inspector, then run a `code-mode-exec` script that imports and calls a
wrapper. For model tools, a successful request is only a transport check;
also verify scientifically meaningful outputs, units, preprocessing, and edge
cases.

## Deployment Notes

- Build the demo image with `uv run --group dev just build-image`.
- The demo container cache path is `/var/cache/nomad`; mount it to reuse remote
  model artifacts. Shared writable caches require reliable filesystem locking.
- Container secrets live under `/run/secrets`. Git credentials may use a
  credential-store file, Kubernetes-style host directories, or
  `GIT_CREDENTIAL_<HOST>_{USERNAME,PASSWORD}` variables. ORAS credentials use a
  Docker-compatible auth file selected with `NOMAD_AUTH_FILE`.
- Nomad uses the operating-system trust store. Install organization CAs there;
  container deployments can mount a complete replacement bundle at
  `/etc/ssl/certs/ca-certificates.crt`.
- Enable OTel export with server/gateway telemetry settings or OTel environment
  variables. The demo Compose stack includes a collector, Jaeger, Prometheus,
  and Grafana.

## Documentation Sources

Use local documentation as the source of truth, in this order for common work:

1. `docs/guides/getting-started.md` for serving, connecting, and code-mode use.
2. `docs/guides/model-builder.md` for a new SciFM adapter.
3. `docs/reference/config.md` and `src/nomad/config.py` for configuration.
4. `docs/reference/api-gateway.md` for wrapper, sandbox, and `RESULT` behavior.
5. `docs/deployments/deployment-spec.md` for non-negotiable deployment rules.
6. `docs/guides/developer.md`, `Justfile`, and CI workflows for contributor
   commands.
