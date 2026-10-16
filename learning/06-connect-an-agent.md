# 06: Connect an Agent

Nomad exists so that agents can use SciFMs. This lesson connects
[opencode](https://opencode.ai/docs/mcp-servers/) to Nomad in two ways:
directly, and through the code-mode gateway. The same ideas apply to other
MCP clients, such as URSA or Claude Desktop.

Keep the demo server from lesson 02 running.

## A Lesson-Only opencode Config

To avoid changing your global or project config, put the config in
`scratch/` and point to it with `OPENCODE_CONFIG`.

Create `scratch/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "nomad": {
      "type": "remote",
      "url": "http://localhost:8181/mcp",
      "oauth": false,
      "enabled": true
    },
    "nomad_code": {
      "type": "local",
      "command": [
        "./.venv/bin/python", "-m", "nomad", "code-mode",
        "--config", "scratch/gateway.yml", "--log-level", "WARNING"
      ],
      "timeout": 30000,
      "enabled": false
    }
  }
}
```

The two entries are:

- **`nomad`** connects directly to the server. The agent sees every tool and
  every schema, which uses up context as you add more models.
- **`nomad_code`** connects through the gateway. The agent sees only three
  tools and writes Python to use the rest. It is disabled for now.
  `timeout` is raised because the gateway takes a few seconds to start.

## Direct Connection

From the repository root:

```shell
OPENCODE_CONFIG=scratch/opencode.json opencode
```

Try:

- `What nomad tools do you have? Use the nomad MCP server.`
- `Use nomad to look up caffeine on PubChem and predict its QM9 properties.
  Report the dipole moment with units.`

Watch tab A's log to see the calls arrive. Run `opencode mcp list` to check
the connection status.

## Code-Mode Connection

In `scratch/opencode.json`, set `nomad.enabled` to `false` and
`nomad_code.enabled` to `true`. Restart opencode, then ask:

- `Using nomad_code, find the boiling-point model and predict boiling points
  for methanol, ethanol, propanol and butanol. Return a table.`

Compare what happens with the direct connection:

- How many tool calls did the agent make?
- Did it call `search_code_tools` first?
- Did it handle the `---` naming difference by itself?

These are the points of friction to improve when developing Nomad's
agent-facing descriptions and schemas.

## Clean Up

The scratch config does not change anything outside this session. Remove
`OPENCODE_CONFIG` from your command to go back to your normal setup.

Next: [07 Reading the code](07-code-walkthrough.md).
