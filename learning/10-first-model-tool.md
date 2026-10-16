# 10: Your First Model Tool

You will build a tiny but complete `TorchModuleTool`, with a model artifact
directory, a model card, and a batch-aware adapter, then serve it through
the tool manager. The model computes `y = 2x + 1`, so the correct answer is
easy to check.

Read `docs/guides/model-builder.md` alongside this lesson. For real external
models, use the `skills/connect-scifm-to-nomad/` workflow.

## 1. Create the Model Artifact

A model is a **directory** referenced by `name_or_path`. It holds the
weights and a `README.md`, which `get_model_card` returns.

Create `scratch/make_toy_weights.py`:

```python
from pathlib import Path

import torch

out = Path(__file__).parent / "toy_model"
out.mkdir(exist_ok=True)
model = torch.nn.Linear(1, 1)
with torch.no_grad():
    model.weight.fill_(2.0)
    model.bias.fill_(1.0)
torch.save(model.state_dict(), out / "weights.pt")
(out / "README.md").write_text("# Toy Linear Model\n\nComputes y = 2x + 1.\n")
```

```shell
./.venv/bin/python scratch/make_toy_weights.py
```

For a real model, base the README on `container/model-card.md`.

## 2. Write the Adapter

Create `scratch/toy_adapter.py`:

```python
from collections.abc import Iterable, Sequence
from pathlib import Path

import torch
from pydantic import BaseModel, Field

from nomad.fm_base_tool import TorchModuleTool


class Point(BaseModel):
    x: float = Field(description="Input value")


class Prediction(BaseModel):
    x: float
    y: float = Field(description="Model output, y = 2x + 1")


class ToyLinearTool(TorchModuleTool[Point, Prediction, torch.Tensor, torch.Tensor]):
    args_schema: type[Point] = Point
    output_schema: type[Prediction] = Prediction

    @classmethod
    def from_pretrained(cls, name_or_path: str, **kwargs) -> "ToyLinearTool":
        model = torch.nn.Linear(1, 1)
        model.load_state_dict(torch.load(Path(name_or_path) / "weights.pt"))
        return cls(
            name="toy_linear",
            description="Computes y = 2x + 1 for one input value.",
            fm=model.eval(),
        )

    def preprocess(self, inputs: Sequence[Point]) -> torch.Tensor:
        # One row per request; move to this instance's device.
        return torch.tensor([[p.x] for p in inputs], device=self.device)

    def _forward(self, model_inputs: torch.Tensor) -> torch.Tensor:
        # Keep x alongside y so postprocess can pair them; return on CPU.
        return torch.cat([model_inputs, self.fm(model_inputs)], dim=1).cpu()

    def postprocess(self, model_output: torch.Tensor) -> Iterable[Prediction]:
        for x, y in model_output.tolist():  # exactly one output per input
            yield Prediction(x=x, y=y)
```

The rules this follows, from `AGENTS.md` and the model-builder guide:

- Load the model in `from_pretrained`, not in a custom `__init__`.
- `preprocess` receives a **list** of inputs (a batch) and moves tensors to
  `self.device`.
- `postprocess` yields one output per input, in order.
- Do not print or write files. Use logging.
- Each instance uses one device. The tool manager makes copies for other
  devices.

## 3. Serve It

Add the model to `scratch/lesson_server.yml` from lesson 08:

```yaml
search_tool:
  expose: true
tools:
  - lesson_tools.celsius_to_kelvin
fmod_models:
  - model_class: toy_adapter.ToyLinearTool
    name_or_path: toy_model
    batch_size: 4
```

`toy_model` is resolved relative to the config file, so it points to
`scratch/toy_model/`. Because that directory exists, it is **not** treated
as a Hugging Face repo.

```shell
cd scratch
../.venv/bin/python -m nomad serve --transport http --port 8282 lesson_server.yml
```

Check for `Visible devices: mps` (or `cpu` or `cuda`) in the log.

## 4. Call It

Create `scratch/lesson_10.py`:

```python
import asyncio

from mcp_tools.mine import get_model_card, toy_linear, toy_linear_async


async def many():
    return await asyncio.gather(*(toy_linear_async(x=float(i)) for i in range(10)))


RESULT = {
    "one": toy_linear(x=3.0),
    "many": [r["y"] for r in asyncio.run(many())],
    "card": get_model_card(tool_name="toy_linear"),
}
```

From the repository root:

```shell
./.venv/bin/python -m nomad cmx -c scratch/lesson_08_gateway.yml -o - scratch/lesson_10.py
```

You should get `"one": {"x": 3.0, "y": 7.0}` and `"many": [1.0, 3.0, ..., 19.0]`.

## 5. Test the Adapter Without a Server

A quick unit test makes development faster:

```python
# scratch/test_toy.py
from pathlib import Path

from toy_adapter import Point, ToyLinearTool


def test_toy_linear():
    tool = ToyLinearTool.from_pretrained(str(Path(__file__).parent / "toy_model"))
    assert tool(Point(x=1.0)).y == 3.0
    outputs = tool.batch([Point(x=i) for i in range(5)])
    assert [p.y for p in outputs] == [1, 3, 5, 7, 9]
```

```shell
cd scratch && ../.venv/bin/python -m pytest -q test_toy.py
```

## Exercises

1. **Naming.** Remove `name="toy_linear"` from `from_pretrained` and add
   `tool_name: toy-v1` in the YAML. What does the tool end up being called?
2. **Batching.** Add `logging.getLogger(__name__).info("batch=%d", len(inputs))`
   in `preprocess`. Call the tool 10 times at once with `batch_size: 4`, and
   then with `batch_size: 1`, and compare the logs.
3. **Strict mode.** Point `name_or_path` at a directory that does not exist
   and start the server with and without `--no-strict`. Where does the
   silent `hf://` fallback appear in the error?
4. **Idle offload.** Set `tool_manager: {idle_seconds: 5}`, call the tool,
   wait, and watch the log.
5. **Packaging.** Real adapters live in an installable package with their
   own dependencies (see `container/demo/pyproject.toml`). How would you
   move `toy_adapter.py` into a package so that the `nomad` console script
   can import it without `PYTHONPATH`?

Next: [11 Tensors, WellFormat, and export](11-tensors-and-export.md).
