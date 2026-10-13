# Learning Nomad

This guide records a hands-on introduction to Nomad from this repository.
It uses the public demo server and has been written for a Chicoma SSH/tmux
session. Run commands from the repository root unless stated otherwise.

## What You Will Learn

1. Nomad's two operating modes: serving MCP tools and calling them from code.
2. How to start, connect to, inspect, and invoke the demo MCP server.
3. How code-mode turns an MCP tool into a Python function.

## Prerequisites

Nomad is installed in the checkout virtual environment:

```shell
./.venv/bin/python -m nomad --help
```

The demo server uses its own project environment, managed by `uv`. It may
download model artifacts on first startup, so network access and a little time
are required.

MCP Inspector is independent of Nomad. On Chicoma, load its runtime and package
manager modules in the pane that will run Inspector:

```shell
module load node-js/24.13.0
module load npm/8.19.3
node --version
npm --version
npx --version
```

## Step 1: Understand The Architecture

Nomad has two complementary modes:

- `nomad serve` loads tools from a server configuration and exposes them over
  MCP. It can serve regular Python callables and PyTorch-backed SciFMs.
- `nomad code-mode-exec` connects to MCP server(s), creates Python wrappers for
  their tools, and runs a script in a sandbox.

Confirm the available commands:

```shell
./.venv/bin/python -m nomad --help
./.venv/bin/python -m nomad serve --help
./.venv/bin/python -m nomad code-mode-exec --help
```

The demo configuration is `container/demo/nomad.yml`. It enables the CPU-only
`search_pubchem` helper, several MIST molecular-property models, a DiffUnet2
model, and the `search_tools` discovery tool. Every Nomad server also registers
`get_model_card`.

## Step 2: Run And Inspect The Demo Server

In one tmux pane, start the demo as an HTTP server:

```shell
uv run --directory container/demo nomad serve \
  --transport http \
  --host localhost \
  --port 8181 \
  nomad.yml
```

Keep this process running. Its MCP endpoint is:

```text
http://localhost:8181/mcp
```

In a second pane, use MCP Inspector's terminal UI. The default web UI tries to
open a browser on the remote host, so it is unsuitable for an SSH/tmux session:

```shell
npx @modelcontextprotocol/inspector --tui http://localhost:8181/mcp
```

Press `C` to connect. Use `Tab` or left/right arrows to select **Tools**. Use
up/down arrows to choose a tool and `Enter` to inspect or invoke it. Press `?`
at any point to see the relevant keybindings.

Completed exercises:

1. Call `get_model_card` for `diffunet2_heat_pli_f2l`.
2. Call `search_pubchem` with `caffeine`.

These calls demonstrate direct MCP use: clients provide JSON-shaped arguments
and receive structured results. The TUI is only a client; the tools execute in
the separate Nomad server process.

## Step 3: Call The Server From Python

Leave the demo server from Step 2 running. Create `gateway.yml` in the
repository root:

```yaml
servers:
  nomad:
    transport: http
    url: http://127.0.0.1:8181/mcp
defaults:
  timeout_seconds: 60
  workspace_root: .
```

Create `lesson_01.py` in the repository root:

```python
from mcp_tools.nomad import search_pubchem

matches = search_pubchem(molecule="caffeine") or []

RESULT = {
    "match_count": len(matches),
    "first_match": matches[0] if matches else None,
}
```

Run the script through the gateway:

```shell
./.venv/bin/python -m nomad code-mode-exec \
  --config gateway.yml \
  --output - \
  lesson_01.py
```

The result is JSON containing:

- `result`, the final global value assigned to `RESULT` by the script.
- `tool_calls`, metadata describing the upstream MCP requests.
- `returncode`, which is `0` for a successful script run.
- `duration_seconds`, the sandbox execution time.

The Python script imports and calls `search_pubchem` like an ordinary function.
Nomad's gateway generated that wrapper, connected it to the configured MCP
server, and forwarded the function call as an MCP tool request.

## Next Lessons

1. Inspect `search_tools` and use it to discover tools from a natural-language
   query.
2. Call a model tool from code mode, using the SMILES returned by
   `search_pubchem`.
3. Create and serve a minimal CPU-only Python tool through `nomad serve`.
4. Learn the server configuration and tool-manager settings.
5. Build and validate a batch-aware `TorchModuleTool` adapter.
