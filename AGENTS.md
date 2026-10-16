# Nomad Contributor and Operator Reference

## Purpose

Nomad (`nomad-scifm`) connects AI agents to scientific foundation models
(SciFMs) using the Model Context Protocol (MCP). It has two primary modes:

- `nomad serve` loads regular Python tools and PyTorch-backed model tools from
  a server configuration and serves them over MCP.
- `nomad code-mode` proxies one or more MCP servers into a Python sandbox and
  is itself an MCP server. Sandboxed code imports generated wrappers from
  `mcp_tools.<server>`. `nomad code-mode-exec` (`cmx`) runs one script through
  the same sandbox without starting a long-lived gateway.

Nomad is a Python 3.12+ project (`.python-version` pins 3.13; the container
uses 3.14). The distribution is `nomad-scifm`; the Python package and CLI are
both named `nomad`.

## Key Facts and Gotchas

Read these before changing or running anything:

- Every `nomad serve` server registers `get_model_card`. `search_tools` is
  added only when `search_tool.expose: true`. Model cards are the `README.md`
  in the model artifact directory, or the Hugging Face repo README.
- `serve` defaults to `--transport stdio`, `--port 8000`, `--strict`, and
  `--tool-manager`. Examples use port 8181; the container uses 38217.
- `--strict` fails startup if any configured model fails to load.
- `code-mode-exec` differs from `code-mode`:
  - It ignores `defaults.workspace_root`; `--directory` (default: current
    directory) always replaces it (`src/nomad/cli.py:96`).
  - It runs with no timeout, so `defaults.timeout_seconds` only applies to
    `code-mode` (`src/nomad/cli.py:110`).
  - It prints nothing unless `--output -` (stdout) or `--output FILE` is given.
  - Its JSON payload has `result`, `returncode`, `duration_seconds`, and
    `tool_calls`; sandbox stdout/stderr are not captured.
- `code-mode` takes `--config/-c`, not a positional config path. It exposes
  three MCP tools: `search_code_tools`, `execute_mcp_code`, and
  `execute_mcp_script`.
- Each wrapped tool has a sync function `<name>` and an async `<name>_async`.
  Wrappers return `structuredContent` (unwrapping a lone `result` key) and
  decode Nomad tensor media to `torch.Tensor`, but outer Pydantic objects stay
  dictionaries. Rebuild them, e.g. `WellFormat.model_validate(result)`.
- The sandbox interpreter is `<workspace_root>/.venv/bin/python` when present,
  otherwise the gateway's own interpreter. It must be able to `import nomad`.
- A `name_or_path` with no scheme is resolved against the config directory. If
  no such local path exists, it silently becomes `hf://<value>`. The demo
  `nomad.yml` relies on this and is unpinned; prefer explicit, pinned sources
  for deployments.
- Config typos are often silent. Unknown top-level server keys and gateway
  `defaults` keys are ignored, and `fmod_models`/`tools` entries accept extra
  keys (`extra="allow"`). Only `tool_manager`, `search_tool`, `telemetry`, and
  top-level gateway keys reject unknown fields. Check spelling when a setting
  seems to have no effect.
- `${VAR}` without a default raises a config error if `VAR` is unset. The
  docstring at `src/nomad/common/env.py:27`, which says it becomes empty, is wrong.
- Hugging Face downloads go to the HF cache (`HF_HOME`, default
  `~/.cache/huggingface`). Other remote sources use `NOMAD_CACHE`, then
  `$XDG_CACHE_HOME/nomad`, then `~/.cache/nomad`.
- Demo adapters call `from_pretrained(..., trust_remote_code=True)`, so they
  execute Python code from the referenced Hugging Face repositories.
- `fmod_models[].model_class` and `tools[]` are dotted import paths. The
  serving interpreter must be able to import them. `python -m nomad` adds the
  current directory to `sys.path`; the `nomad` console script does not.
- Code-mode wrapper names are Python identifiers derived from MCP tool names,
  so they can differ from the served name. For example, the demo MCP tool
  `mist_models---mist_26p9M_kkgx0omx_qm9` imports as
  `mist_models_mist_26p9M_kkgx0omx_qm9`. Use `search_code_tools` to get the
  exact import path.

## Local Environment

Nomad is installed in this checkout's `.venv`. Use its interpreter directly
when running Python or the CLI:

```shell
./.venv/bin/python -m nomad --help
./.venv/bin/python -m pytest
```

Create or refresh the environment with `uv sync --all-groups`. `uv run ...` is
also supported. The repository's `Justfile` is the source for standard lint,
test, image, and documentation commands.

The demo server in `container/demo/` is a separate uv project with its own
`.venv` and model dependencies (transformers, openbabel, pubchempy, and so on).
Its first `uv run` creates that environment. The first `nomad serve nomad.yml`
then downloads about 590 MB of Hugging Face weights (five MIST models of about
109 MB each, plus Janus at about 43 MB). `nomad-smoke.yml` loads PubChem search
plus one MIST model and is the faster first check.

### Platform Notes

**macOS (Apple Silicon or Intel).** No GPU is required. torch and openbabel
have macOS wheels, and CUDA-only dependencies are restricted to Linux in the
lockfiles. Device detection prefers CUDA, then MPS, then CPU
(`src/nomad/torch_tool_manager.py:142`); on Apple Silicon models load on `mps`.
MCP Inspector runs through `npx` and opens a browser:

```shell
npx @modelcontextprotocol/inspector http://localhost:8181/mcp
```

In the web UI, choose the **Streamable HTTP** transport, confirm the URL, and
click **Connect**. The Docker image (`container/demo/Dockerfile`) is
Linux/CUDA-specific; for local learning, run the demo with `uv run` instead.

**Chicoma (LANL HPC, SSH/tmux).** Load modules before creating environments or
running Inspector:

```shell
module load python
module load node-js/24.13.0
module load npm/8.19.3
```

The default Inspector web UI tries to open a browser on the remote host, so
use its terminal UI:

```shell
npx @modelcontextprotocol/inspector --tui http://localhost:8181/mcp
```

In the TUI, press `C` to connect, use `Tab` or left/right arrows to choose the
`Tools` tab, then use `Enter` to inspect or invoke a selected tool. Press `?`
for context-specific keybindings. `--cli` is intended for scripted,
machine-readable requests.

## Repository Map

| Path | Role |
| --- | --- |
| `src/nomad/cli.py` | Top-level CLI: `serve`, `export`, `code-mode`, and `code-mode-exec` (`cmx`). |
| `src/nomad/config.py` | Server configuration schemas and tool/model loading. |
| `src/nomad/fm_base_tool.py` | `TorchModuleTool`, the adapter base class for PyTorch models. |
| `src/nomad/torch_tool_manager.py` | Device scheduling, queuing, batching, and idle offload for model tools. |
| `src/nomad/hub.py` | Resolves and caches local, Hugging Face, Git/LFS, and ORAS model sources. |
| `src/nomad/model_cards.py` | The built-in `get_model_card` tool. |
| `src/nomad/tool_search.py` | The optional `search_tools` tool. |
| `src/nomad/export.py` | `nomad export`: deployment bundles, pinning, and ORAS pushes. |
| `src/nomad/well_format.py` | Typed tensor and Well-format scientific-state exchange. |
| `src/nomad/gateway/` | Code-mode gateway: `server.py`, `config.py`, `sandbox.py`, `wrappers.py`, `middleware/`. |
| `src/nomad/gateway/runtime/` | Code that runs inside the sandbox: import hook, bridge client, wrapper factory, deserialization. |
| `container/demo/` | Runnable public demo server, container files, and model adapters. |
| `container/model-card.md` | Template for a model's runtime `README.md`. |
| `docs/` | MyST/Sphinx user, deployment, API, and developer documentation. |
| `test/` | Pytest suite; gateway tests live in `test/gateway/`. |
| `skills/connect-scifm-to-nomad/` | Agent workflow for adapting an external SciFM. |
| `learning/` | Ordered, hands-on lessons for new contributors. |

### Code Reading Path

To understand the system quickly, read in this order:

1. `src/nomad/cli.py:131` (`serve`): startup order is `get_model_card`,
   regular `tools`, model tools through the tool manager, then `search_tools`.
2. `src/nomad/config.py:308` (`ServerConfig`) and `TorchModuleConfig.build_tool`
   (`:143`), which resolves sources and applies `tool_name`/`batch_size`.
3. `src/nomad/fm_base_tool.py:34` (`TorchModuleTool`): the
   `preprocess` -> `_forward` -> `postprocess` pipeline.
4. `container/demo/src/nomad_demo/mist.py`: a complete real adapter.
   `pubchem.py` is a complete regular tool.
5. `src/nomad/torch_tool_manager.py:195` (`TorchModelToolManager`): request
   queues, batch assembly, device slots, and idle eviction.
6. `src/nomad/gateway/server.py` (`CodeModeGateway`), then
   `gateway/sandbox.py`, `gateway/runtime/client.py` (import hook), and
   `gateway/runtime/wrapper_factory.py`.

## CLI Workflows

Inspect the installed CLI before relying on exact flags:

```shell
./.venv/bin/python -m nomad --help
./.venv/bin/python -m nomad serve --help
./.venv/bin/python -m nomad code-mode --help
./.venv/bin/python -m nomad code-mode-exec --help
./.venv/bin/python -m nomad export --help
```

Start the demo as an HTTP MCP server from the repository root:

```shell
uv run --directory container/demo nomad serve \
  --transport http --host localhost --port 8181 nomad.yml
```

The MCP endpoint is `http://localhost:8181/mcp`. See
[Platform Notes](#platform-notes) for the Inspector command. For a local stdio
client, omit the HTTP flags. Use `--no-strict` only when you intend to let
other configured tools run while some models fail to load.

Run a script against one or more MCP servers through the gateway:

```shell
./.venv/bin/python -m nomad code-mode-exec \
  --config gateway.yml --output - script.py -- script-arguments
```

The sandbox returns the final global `RESULT`:

```python
from mcp_tools.nomad import search_pubchem

RESULT = search_pubchem(molecule="caffeine")
```

Serve the gateway itself to an MCP client (stdio by default, or with
`--transport http --port N`):

```shell
./.venv/bin/python -m nomad code-mode --config gateway.yml
```

`nomad export CONFIG OUTPUT` copies or downloads every model into
`OUTPUT/models` and writes a rewritten `OUTPUT/nomad.yml`. Pinning is on by
default. Options: `--to https|oras`, `--oras-registry`, and `--report` (model
cards only).

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
  callable factory. A bare string is named after its last path segment.
- Use `fmod_models` only for `TorchModuleTool` implementations. The meaningful
  keys are `model_class`, `name_or_path`, optional `tool_name`, and optional
  `batch_size`. Without `tool_name`, an `hf` source is named after its repo id,
  and other sources use the adapter's `name`. Names are sanitized for MCP.
- `name_or_path` supports local paths or `file:`, `hf://`, `oras://`,
  `git+https://`, and `git+ssh://` sources; append `#subdirectory` when needed.
  Plain `https://` becomes `git+https`.
- `tool_manager` fields and defaults:

  | Field | Default |
  | --- | --- |
  | `enabled` | `true` |
  | `idle_seconds` | 300 (offload an idle model from its accelerator) |
  | `gc_idle_seconds` | 300 |
  | `disk_idle_seconds` | 600 (drop resident CPU instances) |
  | `max_pending_per_tool` | unlimited |
  | `max_devices_per_tool` | unlimited |

  Disable the manager only for a concrete reason, with
  `tool_manager.enabled: false` or `--no-tool-manager`.
- `search_tool` fields: `expose` (default `false`), `candidate_limit`, and
  `weights`. Do not use the removed `expose_search_tool` setting.
- `telemetry` fields: `enabled`, `service_name`, and `otlp_endpoint`. Telemetry
  can also be enabled with `NOMAD_OTEL_ENABLED=true` or standard
  `OTEL_*` exporter environment variables.

## Gateway Configuration

`nomad code-mode` and `code-mode-exec` accept YAML, JSON, or TOML gateway
configuration. Top-level keys are `servers`, `defaults`, `middleware`, and
`telemetry`. Each upstream server uses `stdio` (`command`, `args`, ...) or
`http` (`url`, `headers`) transport.

```yaml
servers:
  nomad:
    transport: http
    url: http://localhost:8181/mcp
defaults:
  timeout_seconds: 60
  workspace_root: .
middleware:
  - kind: allowlist
    allow: ["nomad:*"]
  - kind: redaction
    mode: basic
```

- `defaults` includes `timeout_seconds` (default 30), `workspace_root`,
  `stdout_limit`/`stderr_limit`/`result_limit` (64 KiB each), and
  `wrappers_package` (`mcp_tools`). A relative `workspace_root` resolves against
  the config file.
- For `code-mode` under stdio without an explicit `workspace_root`, the
  workspace is the current directory. Otherwise it is a fresh temporary
  directory. `code-mode-exec` always uses `--directory`.
- Middleware kinds are `allowlist` (`allow`/`deny` globs), `redaction`
  (`mode: basic|none`), and `logging` (`level`). Telemetry middleware is always
  added.

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
output per input; override `_forward` only when needed (the default calls
`self.fm(inputs).to("cpu")`). Load models in `from_pretrained`, not a custom
`__init__`. Move tensors to `self.device` in `preprocess`; one tool instance
must use one accelerator only.

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
# Lint formatting and static checks (prek + Ruff)
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
`gpu` marker. YAML blocks in `docs/guides/getting-started.md`,
`docs/reference/config.md`, and `docs/guides/model-builder.md` are validated
against the config schemas by `test/test_docs.py`, so update those docs when
changing config behavior. Generated documentation output is ignored.

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
7. `learning/` for a guided, hands-on introduction.
