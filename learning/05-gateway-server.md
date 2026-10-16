# 05: The Gateway as an MCP Server

In lesson 03, `cmx` ran a script once and exited. In real use, an agent
connects to `nomad code-mode`, which is a long-running MCP server with only
three tools:

| Tool | Purpose |
| --- | --- |
| `search_code_tools` | Find wrappers and return their `python_import_path` and signature (or the full schemas with `detail_level: full`). |
| `execute_mcp_code` | Run a code string in the sandbox. Returns `stdout`, `stderr`, `result`, `returncode`, and `duration_seconds`. |
| `execute_mcp_script` | The same as `execute_mcp_code`, but takes a path to a script file. |

## Launch Through Inspector (stdio)

Inspector can start the gateway itself as a stdio subprocess:

```shell
npx @modelcontextprotocol/inspector -- \
  ./.venv/bin/python -m nomad code-mode --config scratch/gateway.yml
```

In the web UI, the transport is **STDIO** and the command is already filled
in. Click **Connect**, then **List Tools**.

Under stdio, if `defaults.workspace_root` is not set, the sandbox's working
directory is the directory the gateway was started from (the repository
root). For HTTP, it is a fresh temporary directory.

## Exercises

1. Call `search_code_tools` with `query = qm9`. Copy the
   `python_import_path`.
2. Call it again with `detail_level = full` and compare the output with the
   MIST schema you saw in lesson 02.
3. Call `execute_mcp_code` with:

   ```python
   from mcp_tools.nomad import search_pubchem
   print("looking up caffeine")
   RESULT = search_pubchem(molecule="caffeine")[0]["smi"]
   ```

   Notice that `stdout` **is** captured here, unlike with `cmx`.
4. Call `execute_mcp_script` with
   `script_path = scratch/lesson_04_chain.py`.
5. **Timeouts.** `code-mode` applies `defaults.timeout_seconds` (default
   30). Add a `defaults` block with `timeout_seconds: 5` to
   `scratch/gateway.yml`, reconnect, and run `import time; time.sleep(10)`.

## Middleware

The gateway can filter or log calls. Add this to `scratch/gateway.yml`:

```yaml
middleware:
  - kind: allowlist
    allow: ["nomad:search_*", "nomad:get_model_card"]
  - kind: logging
    level: INFO
```

Reconnect and try calling a MIST tool from `execute_mcp_code`. The other
middleware kind is `redaction` (`mode: basic` or `mode: none`).

## Gateway Over HTTP

To run the gateway as a network service:

```shell
./.venv/bin/python -m nomad code-mode --config scratch/gateway.yml \
  --transport http --port 8282
```

Then connect Inspector to `http://localhost:8282/mcp` using **Streamable
HTTP**.

Next: [06 Connect an agent](06-connect-an-agent.md).
