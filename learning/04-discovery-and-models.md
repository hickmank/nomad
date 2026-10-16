# 04: Discovery and Model Calls

Leave the demo server running. This lesson chains tools in the way an agent
would: find a tool, read its model card, look up a molecule, then predict its
properties.

## Python Names vs MCP Names

MCP tool names may contain characters that are not valid in Python names, so
code mode converts them. For example:

| MCP tool name | Python import |
| --- | --- |
| `mist_models---mist_26p9M_kkgx0omx_qm9` | `mist_models_mist_26p9M_kkgx0omx_qm9` |
| `search_pubchem` | `search_pubchem` |

Lesson 05 shows how to get these names automatically with
`search_code_tools`.

## Discover Tools From Code

Create `scratch/lesson_04_discover.py`:

```python
from mcp_tools.nomad import search_tools

RESULT = {
    query: [hit["name"] for hit in search_tools(query=query, limit=3)]
    for query in ["boiling point", "odour", "heat", "pubchem"]
}
```

Run it with `cmx` as in lesson 03. `search_tools` ranks tools by name,
description, and argument text. It does not query the models.

## Chain PubChem and MIST

Create `scratch/lesson_04_chain.py`:

```python
import sys

from mcp_tools.nomad import (
    get_model_card,
    mist_models_mist_26p9M_kkgx0omx_qm9 as qm9,
    search_pubchem,
)

names = sys.argv[1:] or ["caffeine", "aspirin", "ethanol"]
card = get_model_card(tool_name="mist_models---mist_26p9M_kkgx0omx_qm9")

rows = []
for name in names:
    hits = search_pubchem(molecule=name)
    if not hits:
        rows.append({"name": name, "error": "not found"})
        continue
    pred = qm9(smi=hits[0]["smi"])
    rows.append(
        {
            "name": name,
            "smiles": pred["molecule"],
            "mu": pred["properties"]["mu"],
            "n_properties": len(pred["properties"]),
        }
    )

RESULT = {"card_first_line": card.splitlines()[0], "rows": rows}
```

```shell
./.venv/bin/python -m nomad cmx -c scratch/gateway.yml -o - \
  scratch/lesson_04_chain.py -- caffeine benzene water
```

Note that `get_model_card` takes the **MCP** tool name, not the Python name.

## Concurrency and Batching

Every wrapper also has an `_async` version. When requests arrive at the same
time, the tool manager can group them into one model batch (up to
`batch_size`, which is 16 for the MIST models in the demo).

```python
import asyncio

from mcp_tools.nomad import mist_models_mist_26p9M_kkgx0omx_qm9_async as qm9

SMILES = ["CCO", "CCCO", "CCCCO", "c1ccccc1", "O", "CC(=O)O"]


async def main():
    return await asyncio.gather(*(qm9(smi=s) for s in SMILES))


RESULT = [
    (r["molecule"], r["properties"]["mu"]["value"]) for r in asyncio.run(main())
]
```

Compare `duration_seconds` with the sequential version.

## Check That the Science Makes Sense

A successful call shows that the plumbing works, not that the answer is
right. For model tools, always ask:

- Are the units and descriptions what you expect? (Check the model card.)
- Do the trends make sense? For example, does the dipole moment of benzene
  come out near 0?
- What happens with invalid, very large, or charged inputs?

Next: [05 The gateway as an MCP server](05-gateway-server.md).
