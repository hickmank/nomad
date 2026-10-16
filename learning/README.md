# Learning Nomad

A hands-on path from "what is MCP?" to "I can build, test, and ship a Nomad
model tool." Work through the lessons in order; each one builds on the last.

The lessons were written and checked on macOS (Apple Silicon) with `uv`,
`node`, and `npx` installed. Where Chicoma (LANL HPC over SSH/tmux) differs,
a **Chicoma** note says so.

| # | Lesson | You will | Needs demo models? |
| --- | --- | --- | --- |
| 00 | [Setup](00-setup.md) | Check tools and create a scratch area | No |
| 01 | [Architecture and MCP](01-architecture.md) | Learn the vocabulary and the two modes | No |
| 02 | [Serve and inspect the demo](02-serve-and-inspect.md) | Run `nomad serve` and use MCP Inspector | Yes (~590 MB, first run) |
| 03 | [Code mode from a script](03-code-mode-exec.md) | Call MCP tools as Python with `cmx` | Yes |
| 04 | [Discovery and model calls](04-discovery-and-models.md) | Chain `search_tools`, PubChem, and MIST | Yes |
| 05 | [The gateway as an MCP server](05-gateway-server.md) | See what an agent sees in code mode | Yes |
| 06 | [Connect an agent](06-connect-an-agent.md) | Give opencode access to Nomad | Yes |
| 07 | [Reading the code](07-code-walkthrough.md) | Trace `serve`, an adapter, and the tool manager | No |
| 08 | [Your first CPU tool](08-first-cpu-tool.md) | Serve your own Python function | No |
| 09 | [Tests and lint](09-tests-and-lint.md) | Run the contributor checks | No |
| 10 | [Your first model tool](10-first-model-tool.md) | Build a batch-aware `TorchModuleTool` | No |
| 11 | [Tensors, WellFormat, and export](11-tensors-and-export.md) | Exchange tensors and build deployment bundles | Optional |

Lessons 07 to 10 do not need the demo models, so you can do them first if
you want to delay the download.

## Conventions

- Run commands from the repository root unless a lesson says otherwise.
- `./.venv/bin/python -m nomad ...` uses the Nomad checkout's environment.
  `uv run --directory container/demo nomad ...` uses the demo's environment,
  which also has the demo model dependencies.
- Put files you create in `scratch/` at the repository root. It is listed in
  `.gitignore`, so you will not commit them by accident.
- Most lessons keep a server running in one terminal tab while you work in a
  second tab.

## Reference Material

- `AGENTS.md`: a dense list of facts and gotchas for contributors and agents.
- `docs/guides/getting-started.md`, `docs/guides/model-builder.md`, and
  `docs/reference/`: the official user documentation.
