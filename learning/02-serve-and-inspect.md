# 02: Serve and Inspect the Demo

## Start the Server

In terminal tab **A**:

```shell
uv run --directory container/demo nomad serve \
  --transport http \
  --host localhost \
  --port 8181 \
  nomad.yml
```

The first run creates `container/demo/.venv` and downloads about 590 MB of
weights. When the server is ready, you will see:

```text
INFO:     Uvicorn running on http://localhost:8181 (Press CTRL+C to quit)
```

Things to notice in the log:

- `Visible devices: mps` on Apple Silicon (`cuda:N` on a GPU node, or `cpu`).
- One `Registering foundation model ...` line per model.
- `--strict` is on by default, so the server refuses to start if any model
  fails to load. `--no-strict` is for when you deliberately want the other
  tools to run anyway.

> **Faster first run:** use `nomad-smoke.yml` instead of `nomad.yml`. It
> loads PubChem search plus one MIST model. The DiffUnet2 exercise below
> needs the full `nomad.yml`.

Keep this server running in tab A for lessons 02 to 06.

## Connect MCP Inspector

In terminal tab **B**:

```shell
npx @modelcontextprotocol/inspector http://localhost:8181/mcp
```

A browser tab opens. In the left panel:

1. Set **Transport Type** to **Streamable HTTP**.
2. Check that the URL is `http://localhost:8181/mcp`.
3. Click **Connect**.
4. Open the **Tools** tab and click **List Tools**.

> **Chicoma:** the web UI tries to open a browser on the remote host. Use
> `npx @modelcontextprotocol/inspector --tui http://localhost:8181/mcp`
> instead. Press `C` to connect, `Tab` to move between panes, `Enter` to
> select or invoke, and `?` for help.

## Exercises

1. **Look at the tool list.** Find `get_model_card`, `search_tools`,
   `search_pubchem`, five `mist_models---...` tools, and
   `diffunet2_heat_pli_f2l`. The MIST tool names come from the Hugging Face
   repo id, adjusted to be valid MCP names (`/` becomes `---`, and `.`
   becomes `p`).
2. **Read a schema.** Select a MIST tool and look at its input schema (`smi`,
   a SMILES string) and its output schema (`molecule` plus `properties`).
3. **Call `get_model_card`** with `tool_name =
   mist_models---mist_26p9M_kkgx0omx_qm9`. Then call it for
   `diffunet2_heat_pli_f2l`. Model cards are the model repo's `README.md`.
4. **Call `search_pubchem`** with `molecule = caffeine`. Copy the `smi` value
   from the result.
5. **Call `mist_models---mist_26p9M_kkgx0omx_qm9`** with that SMILES. Check
   that each property has a value, units, and a description.
6. **Try `search_tools`** with `query = boiling point` and then
   `query = dipole`.
7. **Cause an error on purpose.** Call a MIST tool with `smi = not-a-smiles`
   and look at how the error is reported to the client.

## What This Shows

The Inspector is just an MCP client. The tools run in the server process in
tab A. Watch tab A's log while you make calls. The first call to a model can
be slow because the tool manager is loading it onto the device. After
`idle_seconds` (120 in the demo config) without calls, the model is moved off
the device again.

Next: [03 Code mode from a script](03-code-mode-exec.md).
