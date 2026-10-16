# 11: Tensors, WellFormat, and Export

Two topics to finish with: passing tensor data between agents and models, and
packaging a server for deployment.

## Tensors and WellFormat

Most tools should take small JSON inputs, such as a SMILES string. Some
models, like PDE surrogates, need whole fields as input. Nomad provides
Pydantic types for these, in `src/nomad/well_format.py`:

| Type | Use |
| --- | --- |
| `Tensor` | Any `torch.Tensor` field. Sent as a base64, zstd-compressed torch serialization with media type `application/vnd.nomad.tensor`. |
| `TensorField(min_rank=..., shape_str=...)` | A floating-point tensor with a minimum rank and a documented shape. |
| `WellFormat` | A scientific state on a grid, following [The Well](https://polymathic-ai.org/the_well/): fields, dimensions, and metadata. |
| `AutoRegressiveInput` | An `initial_state: WellFormat` plus a `duration`, for rollout models. |

How a tensor field is handled at each step:

1. On the **server**, Pydantic encodes the tensor field as a base64 string.
2. The **code-mode wrapper** reads the output schema, sees the media type,
   and turns the field back into a `torch.Tensor`.
3. The **outer object stays a `dict`**. Call
   `WellFormat.model_validate(result)` to get a `WellFormat` object back.
4. When `RESULT` is returned, tensors are serialized again, so return small
   summaries (shapes and statistics) rather than whole fields.

### Exercise: DiffUnet2 (Full Demo Server)

Start the full demo (`nomad.yml`) as in lesson 02. Then create
`scratch/lesson_11.py`:

```python
from pathlib import Path

from mcp_tools.nomad import diffunet2_heat_pli_f2l
from nomad.well_format import AutoRegressiveInput, WellFormat

state = WellFormat.from_file(Path("docs/guides/angled_half_res.h5"))
payload = AutoRegressiveInput(duration=100, initial_state=state).model_dump()
pred = WellFormat.model_validate(diffunet2_heat_pli_f2l(**payload))

field = pred.t0_fields["av_density"]
RESULT = {
    "input_shape": list(state.t0_fields["av_density"].shape),
    "output_shape": list(field.shape),
    "min": float(field.min()),
    "max": float(field.max()),
}
```

Then work through `docs/guides/nomad_inference.ipynb`. It runs the same call
over a plain FastMCP client and plots the input and output fields:

```shell
uv run --group docs jupyter lab docs/guides/nomad_inference.ipynb
```

Read `container/demo/src/nomad_demo/diffunet2.py` to see a `WellFormat`
adapter.

## Export: Deployment Bundles

`nomad export CONFIG OUTPUT` resolves every model source. By default
(`--to disk`), it copies or downloads every model into `OUTPUT/models/` and
writes a rewritten `OUTPUT/nomad.yml` with pinned sources and fixed tool
names.

Try it on your lesson server (this downloads nothing, because the model is
local):

```shell
cd scratch
../.venv/bin/python -m nomad export lesson_server.yml export_out
cat export_out/nomad.yml
```

Things to notice:

- `name_or_path` becomes `file:models/toy_model`, and comments record the
  original source.
- `tool_name` is now set explicitly. Before, it came from the adapter's name;
  after export, it comes from the source directory name. Check whether your
  tool's name changed.
- Every default value is now written out.

Other modes:

- `--to https`: keep remote sources remote, and rewrite `git+ssh` to
  `git+https`.
- `--to oras --oras-registry REGISTRY/REPO`: push the models to an OCI
  registry.
- `--report`: write only the model cards and an index, for review.

Read `docs/deployments/guide.md` and `docs/deployments/deployment-spec.md`
for container, cache, secret, and telemetry rules. Note that the demo
`Dockerfile` is Linux/CUDA-only.

## Where to Go Next

- Pick a small issue. The incorrect docstring at `src/nomad/common/env.py:27`
  and the silent acceptance of unknown config keys are both real, small
  improvements.
- Adapt a real SciFM with `skills/connect-scifm-to-nomad/`.
- Enable telemetry (`docs/reference/otel.md`) and use the demo Compose stack
  to look at traces of the batching you studied in lessons 07 and 10.
