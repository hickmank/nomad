# 01: Architecture and MCP

## MCP Basics

The [Model Context Protocol](https://modelcontextprotocol.io) is a JSON-RPC
protocol that lets an AI client (Claude, opencode, URSA, and others) discover
and call **tools** provided by a **server**.

- A **tool** has a name, a description, an **input schema** (JSON Schema),
  and optionally an **output schema**. The client calls `tools/list` to
  discover tools and `tools/call` with JSON arguments to call one.
- A result contains `content` (text blocks for the LLM) and, when there is an
  output schema, `structuredContent` (JSON that matches that schema).
- **Transports** carry the messages:
  - **stdio**: the client launches the server as a subprocess and talks over
    stdin and stdout. This works only on one machine.
  - **Streamable HTTP**: the server listens on a URL such as
    `http://localhost:8181/mcp`. Clients can be anywhere on the network.

Because stdio uses stdout for protocol messages, **server code must never
print to stdout.** That is why Nomad's model-builder rules require using
logging instead.

## Nomad's Two Modes

```text
             ┌────────────── nomad serve ──────────────┐
 MCP client ─┤ get_model_card   search_tools (optional)│
 (Inspector, │ regular tools  (e.g. search_pubchem)    │
  opencode)  │ model tools ── TorchModelToolManager ── GPU/MPS/CPU
             └─────────────────────────────────────────┘
                       ▲
                       │ MCP (http or stdio)
             ┌─────────┴──── nomad code-mode ───────────┐
 MCP client ─┤ search_code_tools                         │
 (agent)     │ execute_mcp_code / execute_mcp_script ──► sandboxed Python:
             └───────────────────────────────────────────┘   from mcp_tools.nomad import ...
```

1. **`nomad serve`** reads a config file (`nomad.yml`) and turns two kinds of
   entries into MCP tools:
   - `tools`: plain Python functions, for example a PubChem lookup.
   - `fmod_models`: PyTorch **SciFM** (scientific foundation model) adapters,
     which subclass `TorchModuleTool`. The **tool manager** handles batching,
     device placement, and offloading idle models.

   Every server also gets `get_model_card`, and gets `search_tools` if it is
   enabled.

2. **Code mode** connects to one or more MCP servers and generates a Python
   module per server (`mcp_tools.<server>`), with one function per tool. It
   runs user code in a sandboxed subprocess. There are two ways to use it:
   - `nomad code-mode` is itself an MCP server, with three tools. An agent
     writes Python instead of making one tool call at a time. This lets it
     loop, filter, and chain calls without passing every intermediate result
     back through the LLM's context.
   - `nomad code-mode-exec` (alias `cmx`) runs one script and exits. Use it
     for testing and for re-running scripts written by an agent.

## Check the CLI

```shell
./.venv/bin/python -m nomad --help
./.venv/bin/python -m nomad serve --help
./.venv/bin/python -m nomad code-mode --help
./.venv/bin/python -m nomad code-mode-exec --help
./.venv/bin/python -m nomad export --help
```

Notice:

- `serve` takes the config as a positional argument and defaults to stdio on
  port 8000.
- `code-mode` and `code-mode-exec` take `--config/-c`.

## Read the Demo Config

Open `container/demo/nomad.yml` and `container/demo/nomad-smoke.yml`. Identify:

- the regular tool (`nomad_demo.pubchem.search_pubchem`);
- the five MIST models (molecular property predictors) and the DiffUnet2
  model (a PDE surrogate);
- that `name_or_path: mist-models/...` has no `hf://` prefix. It is treated
  as a Hugging Face repo only because no local directory of that name exists.

## Check Your Understanding

1. Which transport would you use for an agent on your laptop calling a Nomad
   server on an HPC node?
2. Why might an agent prefer `execute_mcp_code` over calling 50 model tools
   directly?
3. What is the difference between a `tools` entry and an `fmod_models` entry?

Next: [02 Serve and inspect the demo](02-serve-and-inspect.md).
