# 03: Code Mode From a Script

Leave the demo server from lesson 02 running in tab A.

## Gateway Config

Create `scratch/gateway.yml`:

```yaml
servers:
  nomad:
    transport: http
    url: http://localhost:8181/mcp
```

The key under `servers` (here `nomad`) becomes the module name:
`mcp_tools.nomad`.

## First Script

Create `scratch/lesson_03.py`:

```python
from mcp_tools.nomad import search_pubchem

matches = search_pubchem(molecule="caffeine") or []

RESULT = {
    "match_count": len(matches),
    "first_match": matches[0] if matches else None,
}
```

Run it:

```shell
./.venv/bin/python -m nomad code-mode-exec \
  --config scratch/gateway.yml \
  --output - \
  scratch/lesson_03.py
```

The output JSON contains:

| Key | Meaning |
| --- | --- |
| `result` | The final value of the global `RESULT` (JSON-serialized). |
| `returncode` | `0` when the script succeeded. |
| `duration_seconds` | How long the sandbox took to run. |
| `tool_calls` | One entry per upstream MCP call, with inputs, outputs, and timing. |

## How `cmx` Differs From `code-mode`

These rules about `code-mode-exec` often catch people out:

- **Without `--output`, it prints nothing.** Use `--output -` for stdout or
  `--output result.json` for a file.
- **`print()` output is not captured.** Sandbox stdout and stderr go directly
  to your terminal and are not included in the JSON. Put what you need into
  `RESULT`.
- **There is no timeout,** even if the config sets
  `defaults.timeout_seconds`.
- **The workspace is `--directory`** (default: the current directory), even
  if the config sets `defaults.workspace_root`. The script's working
  directory is this folder.
- **Script arguments** go after `--`:
  `... scratch/lesson_03.py -- --molecule aspirin`. The script reads them
  from `sys.argv`.

## What Happened Behind the Scenes

1. The gateway connected to `http://localhost:8181/mcp` and listed its tools.
2. It started a sandbox Python process. Because `./.venv/bin/python` exists
   in the workspace (the repository root), it used that interpreter, which
   can `import nomad`. If you use `--directory` somewhere without a `.venv`,
   the gateway's own interpreter is used instead.
3. When the script ran `import mcp_tools.nomad`, an import hook asked the
   gateway for the tool list and built a function for each tool, plus an
   `_async` version.
4. Each call to a function sends a request to the gateway, which forwards it
   to the MCP server.

## Exercises

1. **Results are dicts, not objects.** `matches[0]` is a `dict`, so use
   `matches[0]["smi"]`, not `matches[0].smi`. Change the script to return
   only the SMILES strings.
2. **Script arguments.** Read the molecule name from `sys.argv[1]` and run
   the script with `-- aspirin`.
3. **Errors.** Add `raise ValueError("boom")` and look at `returncode` and
   `result`.
4. **Write to a file.** Use `--output scratch/out.json` and open the file.

Next: [04 Discovery and model calls](04-discovery-and-models.md).
