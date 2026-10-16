# 00: Setup

## Check Your Tools

```shell
uv --version
node --version
npx --version
./.venv/bin/python --version
./.venv/bin/python -m nomad --help
```

If `.venv` is missing or out of date, recreate it with:

```shell
uv sync --all-groups
```

> **Chicoma:** run `module load python`, `module load node-js/24.13.0`, and
> `module load npm/8.19.3` first, in every shell that needs them.

## What You Will Download, and When

| Item | Size | When |
| --- | --- | --- |
| Demo environment (`container/demo/.venv`: torch, transformers, openbabel, ...) | a few hundred MB | First `uv run --directory container/demo ...` |
| Demo model weights from Hugging Face | about 590 MB | First `nomad serve nomad.yml` |
| MCP Inspector (npm) | small | First `npx @modelcontextprotocol/inspector` |

Model weights go to the Hugging Face cache (`~/.cache/huggingface` unless
`HF_HOME` is set), so they are only downloaded once. To reclaim the space
later, use `./.venv/bin/hf cache scan` and `./.venv/bin/hf cache delete`.

On macOS, everything runs without a GPU. On Apple Silicon, models run on the
`mps` device.

## Create a Scratch Area

```shell
mkdir -p scratch
```

All lesson files go here. `scratch/` is in `.gitignore`.

> **Naming pitfall:** do not give scratch scripts the same name as a Python
> module, such as `attr.py`, `json.py`, `mcp.py`, or `torch.py`.
> `python -m nomad` puts the current directory on `sys.path`, so a file like
> that shadows the real library and causes confusing import errors. Prefixes
> like `lesson_03_*.py` are safe.

## Optional: Prepare the Demo Environment Now

This builds the demo environment without downloading any model weights:

```shell
uv sync --directory container/demo
```

Next: [01 Architecture and MCP](01-architecture.md).
