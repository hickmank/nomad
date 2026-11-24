# Getting Started

This guide gets you to a working connection in three common cases:

- Launching a Nomad [MCP][] server to host SciFMs
- Creating a code-mode gateway for an existing [MCP][] server
- Running Python scripts within Nomad's code-mode sandbox

## Starting a Nomad Server

From the repository root, run `cd container/demo`, then start the server
with [`nomad serve`](cli-nomad-serve):

```shell
uv run nomad serve \
  --transport http \
  --port 8181 \
  nomad.yml
```

This launches the Nomad server defined in
{repo_file}`nomad.yml <container/demo/nomad.yml>`, including multiple
SciFMs and standard MCP tools. After startup, you should see:

```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://localhost:8181
```

You can now launch the [MCP Inspector][] to inspect and interact with the server:

```shell
npx @modelcontextprotocol/inspector http://localhost:8181/mcp
```

If you use [URSA][], add the following to your URSA config (for example,
`ursa.yml`):

```yaml
mcp_servers:
  nomad:
    transport: http
    url: http://localhost:8181/mcp
```

Then start URSA with `ursa --config ursa.yml`.

> URSA must be installed for the command above to work. See [URSA][] for
> installation instructions.

## Connect to a Hosted Nomad Server

If you have access to a hosted Nomad server, add it to your [URSA][] configuration
with its HTTP endpoint:

```yaml
mcp_servers:
  nomad:
    transport: http
    url: https://nomad.example.org/mcp
    # Authentication headers are optional
    headers:
      authorization: Bearer ${NOMAD_API_KEY} # Nomad will pull this from your env
```

Then launch URSA with the following command:

```bash
export NOMAD_API_KEY="<your-api-key>"
ursa --config ursa.yml
```

Enter `execute what tools do you have?` in the [URSA][] terminal to confirm that
[URSA][] has connected to the MCP server.


## Code-mode Gateway

Nomad's code-mode gateway proxies existing MCP servers (including a [`nomad serve`](cli-nomad-serve)) into three tools:

- {py:meth}`search_code_tools <nomad.gateway.server.CodeModeGateway.search_code_tools>`:
  Surface schemas and documentation for the proxied MCP tools.
- {py:meth}`execute_mcp_code <nomad.gateway.server.CodeModeGateway.execute_mcp_code>`:
  Execute a Python code snippet in a sandbox containing the proxied MCP tools.
- {py:meth}`execute_mcp_script <nomad.gateway.server.CodeModeGateway.execute_mcp_script>`:
  Execute a Python script file in a sandbox containing the proxied MCP tools.

API spec: {doc}`/reference/api-gateway`

This lets agents interact with proxied MCP tools as Python functions and run
code like:

```python
from mcp_tools.nomad import get_model_card

RESULT = "library_name: transformers" in get_model_card("mist_models---mist_26p9M_kkgx0omx_qm9")
```

In practice, agents can call tools directly and manipulate their results in
ordinary Python. For more detail on code mode, see
[Anthropic's post on Code Execution with MCP](https://www.anthropic.com/engineering/code-execution-with-mcp).

### Starting a Code-Mode Gateway

First, create `gateway.yml`:

```yaml
servers:
  # Or any existing MCP server
  nomad:
    transport: http
    url: http://127.0.0.1:8017/mcp
defaults:
  timeout_seconds: 60
```

Then use [MCP Inspector][] to interact with the gateway launched with
[`nomad code-mode`](cli-nomad-code-mode):

```shell
npx @modelcontextprotocol/inspector -- \
  uv run nomad code-mode --config gateway.yml
```

This opens a web dashboard where you can interact with the gateway. To begin:

- Click on "Tools" along the top bar, then "List Tools". This shows the list
  of tools served by the gateway.
- Select {py:meth}`execute_mcp_code <nomad.gateway.server.CodeModeGateway.execute_mcp_code>`
- Type `RESULT = 5 * 3` into the text area labeled `code` that appears to the
  right.
- Scroll down and click "Run Tool".

You should then see the following appear just below the "Run Tool" button:

```json
{
  "stdout": "",
  "stderr": "",
  "result": 15
}
```

While you can try more complex code snippets from the [MCP Inspector][],
[`nomad code-mode-exec`](cli-nomad-code-mode-exec) provides a nicer interface
for running code within the code-mode sandbox.

### Running Scripts within the Code-Mode Sandbox

Use [`nomad code-mode-exec`](cli-nomad-code-mode-exec) to run a Python script
within Nomad's code-mode sandbox. This is useful for rerunning scripts created
by agents for Nomad's
{py:meth}`execute_mcp_script <nomad.gateway.server.CodeModeGateway.execute_mcp_script>` tool,
or to interact with Nomad-hosted SciFMs (or any other [MCP][] server)
programmatically.

Create `test.py` with the following content:

```python
from mcp_tools.nomad import search_pubchem, mist_models_mist_26p9M_kkgx0omx_qm9

mol = search_pubchem(molecule="caffeine")[0]
RESULT = mist_models_mist_26p9M_kkgx0omx_qm9(smi=mol.smi)
```

Then create a gateway config for the local Nomad server (`gateway.yaml`):

```yaml
servers:
  nomad:
    transport: http
    url: http://127.0.0.1:8181/mcp
```

> Remember to start the local Nomad server first.

Finally, run the script in the sandbox:

```shell
uv run nomad code-mode-exec \
  --config gateway.yaml \
  --output - \
  test.py
```

For a local server, CMX can start and stop Nomad itself instead of using a
gateway config and a separately managed server:

```shell
uv run nomad cmx \
  --nomad-config nomad.yml \
  --output - \
  test.py
```

The local server is available to the script as `mcp_tools.nomad` and is stopped
when the script finishes. `--nomad-config` and `--config` are mutually
exclusive.

This produces output like:
```jsonc
{
  "result": {
    "molecule": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "properties": {
      "mu": {
        "value": 3.6489930152893066,
        "units": null,
        "description": "Dipole Moment"
      },
      // Additional MIST predicted properties
    }
  },
  "tool_calls": [
    // Information on called tools
  ]
}
```

See [`nomad code-mode-exec`](cli-nomad-code-mode-exec) for full invocation details
including passing script arguments. The {doc}`Code-Mode Gateway Reference
</reference/api-gateway>` describes wrapper return-value deserialization,
`RESULT` serialization, and the JSON payload returned by `code-mode-exec`.

[MCP]: https://modelcontextprotocol.io
[MCP Inspector]: https://modelcontextprotocol.io/docs/tools/inspector
[URSA]: https://lanl.github.io/ursa/
