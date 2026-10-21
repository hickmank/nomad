# Hosting a SciFM with Nomad

This guide walks through the shortest credible path from a scientific model to
a running Nomad tool. The goal is to package the model code, describe it in
`nomad.yml`, and verify that Nomad can start the server and expose the tool
through MCP.

By the end of the guide, `nomad serve ...` should start cleanly, your tool
should appear in [MCP Inspector][], and a test invocation should return output.

:::::{dropdown} Agent Skill
:color: muted

The [Connect SciFM to Nomad skill](../deployments/agent-skills.md#connect-scifm-to-nomad)
guides your preferred agent through connecting a SciFM to Nomad:

1. Install [Codex][codex],
   [Claude Code][claude-code],
   [URSA][ursa], or your coding agent of choice.
2. Start the agent from your model's code folder.
3. Ask the agent to install the skill with this prompt:

   ```text
   Download the skill from
   https://github.com/lanl/nomad/tree/main/skills/connect-scifm-to-nomad,
   resolve any symlinked references, and set it up in my skills directory. Then
   show me how to use the skill.
   ```
4. Ask the agent to connect your model.

These recordings show [Codex][codex] and [URSA][ursa] connecting
[`mist-models/mist-mixtures-zffffbex`](https://huggingface.co/mist-models/mist-mixtures-zffffbex)
to Nomad, starting from clean checkouts of [MIST](https://github.com/BattModels/mist)
and finishing with validated MCP tools.

::::{tab-set}
:::{tab-item} Codex

```{asciinema-player} codex-mist.cast
:cols: 100
:rows: 24
:speed: 13.476
:idle-time-limit: 1
:poster: npt:0:03
:theme: asciinema
:autoplay:
:loop:
```
:::

:::{tab-item} URSA
```{asciinema-player} ursa-mist.cast
:cols: 100
:rows: 30
:speed: 1.9
:idle-time-limit: 1
:poster: npt:0:03
:theme: asciinema
:autoplay:
:loop:
```
:::
::::

:::::

## What you are building

Nomad serves tools from configuration rather than from an ad hoc Python entry
point. In practice, connecting a SciFM means supplying four pieces:

- Importable Python code for the tool class or function in a
  [pip-installable](https://packaging.python.org/en/latest/overview/) package.
- A reachable model directory for the weights and configuration (`name_or_path`).
- A model card based on {repo_file}`container/model-card.md <container/model-card.md>`.
- A config entry (`fmod_models` or `tools`) in {repo_file}`nomad.yml <container/demo/nomad.yml>`.

This page focuses on getting that integration working end to end. Specific
deployment rules and packaging constraints live in the
{doc}`Deployment Specification </deployments/deployment-spec>`.

> This guide uses [uv][] for package management.
> [uv][] is not required to deploy a model with Nomad, but it makes the setup
> steps easier.

## Create an Importable Package

`nomad serve` loads your class or function by dotted import path, so that exact
symbol must be importable in the model's configured `env`. When `env` is
omitted, the model subprocess uses the same Python environment as the server.

Minimal repository shape:

```text
my_pkg/
  pyproject.toml
  src/my_pkg/mcp.py
```

> You can create this quickly with `uv init --package ./my_pkg`.
> Replace `./my_pkg` with your desired package name.

To quickly check, from the top-level `./my_pkg` directory, run:

```bash
uv run python -c "from my_pkg.mcp import MyModelTool; print(MyModelTool)"
```

For local development, either install your package into the environment used
to run `nomad serve`, or set the model's `env` to a PEP 508 direct reference
such as `my-pkg @ file:///absolute/path/to/my_pkg`. Nomad installs `env` into a
cached model-specific environment. The field accepts one PEP 508 requirement
string or a list of them; requirements and project file paths are not
supported, and direct references must use absolute URLs.

For demo deployments without an explicit model `env`, the host environment is
defined by
{repo_file}`container/demo/pyproject.toml <container/demo/pyproject.toml>`,
so your package must be added there as a pinned dependency. If the package
comes from another repository or a local checkout, also add a matching
`[tool.uv.sources]` entry so the demo build can resolve it. Do not
add model-specific dependencies to the root project unless they are required
outside the deployment package.

## Writing a `TorchModuleTool`

A {py:class}`nomad.fm_base_tool.TorchModuleTool` is the adapter between
Nomad's MCP-facing tool interface and your underlying PyTorch model. In
practice, you are not exposing your raw training interface. You are defining an
agent-friendly scientific query and then wiring that query into the model.

Before you write code, answer three questions:

- What is the scientific query you want the tool to answer?
- What should one tool call take as input?
- What should one tool call return?

For agent workflows, prefer JSON-friendly inputs and outputs over raw tensors
or large binary payloads. A good pattern is to make one tool call represent one
scientific query and return one structured answer. That keeps the interface
easy for an agent to supply, inspect, and reuse.

For rollout, PDE, or grid-based models, start by looking at
{repo_file}`src/nomad/well_format.py <src/nomad/well_format.py>`. In
particular, {py:class}`nomad.well_format.AutoRegressiveInput` and
{py:class}`nomad.well_format.WellFormat` provide a ready-made contract for
stateful scientific model inputs and outputs. PDE tools should also document
the model `dt` in their tool description, prefer `t_start=0.0`, and ensure
`t_final >= duration` when using rollout-style inputs.

### The four methods you usually implement

Most model integrations only need to define:

- {py:meth}`TorchModuleTool.from_pretrained <nomad.fm_base_tool.TorchModuleTool.from_pretrained>`:
  load the model and any supporting artifacts such as tokenizers, then return a
  configured tool instance.
- {py:meth}`TorchModuleTool.preprocess <nomad.fm_base_tool.TorchModuleTool.preprocess>`:
  convert a batch of typed tool inputs into the batched structure your model
  actually expects.
- {py:meth}`TorchModuleTool._forward <nomad.fm_base_tool.TorchModuleTool._forward>`:
  optionally override the default forward path when you need custom model
  invocation or need to preserve metadata through inference.
- {py:meth}`TorchModuleTool.postprocess <nomad.fm_base_tool.TorchModuleTool.postprocess>`:
  convert batched model outputs back into one typed result per input item.


### Minimal skeleton

```python
from collections.abc import Sequence

import torch
from pydantic import BaseModel, Field

from nomad.fm_base_tool import TorchModuleTool

# # For PDE Models
# from nomad.well_format import AutoRegressiveInput, WellFormat

# # For SciFMs that need tensor inputs or outputs
# from nomad.well_format import Tensor


class MyModelInput(BaseModel):
    molecule: str = Field(description="Input query for one prediction")


class MyModelOutput(BaseModel):
    score: float = Field(description="Predicted value for the input query")


class MyModelTool(
    TorchModuleTool[
        MyModelInput,
        MyModelOutput,
        dict[str, torch.Tensor],
        torch.Tensor,
    ]
):
    args_schema = MyModelInput
    output_schema = MyModelOutput

    @classmethod
    def from_pretrained(cls, name_or_path: str, **kwargs):
        model = load_model_somehow(name_or_path, **kwargs)
        return cls(
            name="my-model",
            description="Predicts a score for one molecule.",
            fm=model,
            batch_size=32,
            device=model.device,
        )

    def preprocess(self, inputs: Sequence[MyModelInput]) -> dict[str, torch.Tensor]:
        encoded = encode_batch([item.molecule for item in inputs])
        return {key: value.to(self.device) for key, value in encoded.items()}

    def postprocess(self, model_output: torch.Tensor):
        for row in model_output:
            yield MyModelOutput(score=float(row.item()))
```

### Implementation notes

- Define `args_schema` and `output_schema` as Pydantic models that describe the
  MCP tool interface, not your raw model tensors.
- Put heavyweight loading logic in
  {py:meth}`TorchModuleTool.from_pretrained <nomad.fm_base_tool.TorchModuleTool.from_pretrained>`,
  not `__init__`.
- In {py:meth}`TorchModuleTool.preprocess <nomad.fm_base_tool.TorchModuleTool.preprocess>`,
  assume you are receiving a batch. Even if each call looks scalar at the CLI
  or in MCP Inspector, Nomad may batch several requests together.
- Move tensors to `self.device` in
  {py:meth}`TorchModuleTool.preprocess <nomad.fm_base_tool.TorchModuleTool.preprocess>`.
  `TorchModuleTool` execution should use only that device. Multi-GPU models are not supported.
- In {py:meth}`TorchModuleTool.postprocess <nomad.fm_base_tool.TorchModuleTool.postprocess>`,
  yield one output object at a time rather than returning the whole batch as one
  result.
- Use the tool description and field descriptions carefully. Agents and human
  operators both rely on that text to decide when and how to call the tool.
  Keep descriptions concise and put detailed model information in the model
  card.
- If your model produces very large outputs, rethink the tool interface before
  exposing it directly. Large payloads are awkward for both MCP clients and
  language-model-driven agents.
- Do not write to stdout from model or tool code. Use Python logging instead.
- Tools should not write persistent files. If a temporary file is unavoidable,
  create it per invocation with the {py:mod}`tempfile` module.

> Overriding `__init__` is *highly* discouraged because
> {py:class}`~nomad.fm_base_tool.TorchModuleTool` is a
> {py:class}`pydantic.BaseModel`. Custom `__init__` behavior is possible, but
> [considerably more complex](https://pydantic.dev/docs/validation/latest/concepts/models/#defining-a-custom-__init__)
> than loading the model in
> {py:meth}`TorchModuleTool.from_pretrained <nomad.fm_base_tool.TorchModuleTool.from_pretrained>`.

## Choosing `fmod_models` vs `tools`

- Use `fmod_models` when your class is a
  {py:class}`nomad.fm_base_tool.TorchModuleTool`.
- Use `tools` for regular callables, such as CPU-only Python functions.

If you are serving one SciFM, start with a complete `nomad.yml` like this:

```yaml
fmod_models:
  - model_class: my_pkg.mcp.MyModelTool
    name_or_path: my-org/my-model-v2
    tool_name: my-model-v2
    batch_size: 32
```

Add entries under `tools` only when you also want regular Python callables
registered alongside the model.

## Deciding on Input and Output Schemas

The input and output schemas of your
{py:class}`~nomad.fm_base_tool.TorchModuleTool` define how agents interact with
your model. Designing this interface is key; the rest of the
{py:class}`~nomad.fm_base_tool.TorchModuleTool` implementation is plumbing.
Keep these guidelines in mind:

1. Prefer inputs that map directly to a single model invocation.
2. Consider multiple tools instead of one more complex tool. Nomad handles
   de-duplicating model weights, so the runtime penalty for more tools is
   minimal.
3. Prefer built-in formats where possible, but build an interface that works
   for your model first.

| Model Interface | Suggested Schema |
|:---:|:---|
| Single fixed-shape array | {py:func}`nomad.well_format.TensorField` with a {py:class}`pydantic.AfterValidator` |
| Arbitrary tensor | {py:data}`nomad.well_format.Tensor` |
| Coordinates, BCs and named physical fields | {py:class}`~nomad.well_format.WellFormat` |
| Stateful evolution or rollout | {py:class}`~nomad.well_format.WellFormat` and {py:class}`~nomad.well_format.AutoRegressiveInput` |
| Small scalar or list data | Ordinary JSON-friendly {py:class}`pydantic.BaseModel` fields |

### Enforcing Exact Tensor Shapes

To enforce an exact shape, add a {py:class}`~pydantic.AfterValidator`:

```python
from typing import Annotated

import torch
from pydantic import AfterValidator

from nomad.well_format import TensorField


def require_image_shape(value: torch.Tensor) -> torch.Tensor:
    if value.shape != (3, 224, 224):
        raise ValueError("Expected shape (3, 224, 224)")
    return value


ImageTensor = Annotated[
    TensorField(min_rank=3, shape_str="channels height width"),
    AfterValidator(require_image_shape),
]
```

## Hosting Your Model Weights

The weights and configuration for your model must be stored in a
[Hugging Face-style model directory](https://huggingface.co/docs/hub/models).
Prefer hosting that directory as an external model source. The path to this
directory is passed to
{py:meth}`TorchModuleTool.from_pretrained <nomad.fm_base_tool.TorchModuleTool.from_pretrained>`
to load your model. This directory is referenced by the `name_or_path` field
for the model under `fmod_models` in the config.

`TorchModuleTool.from_pretrained(name_or_path)` should accept the resolved
`name_or_path` without requiring extra keyword arguments and should return a
fresh, independent tool instance each time it is called. The tool manager may
call it repeatedly while serving a model; see {doc}`../reference/api-tool-manager`
for lifecycle details.

Recommended `name_or_path` values:

| Source | Example |
| --- | --- |
| [Hugging Face](https://huggingface.co/docs/hub/models) | `hf://my-org/my-model-v2` |
| [ORAS artifact](https://oras.land/docs/) | `oras://registry.example.org/my-org/my-model:v1#models/my-model-v1` |
| [Git/Git LFS](https://git-lfs.com/) | `git+ssh://git@example.org/my-org/models.git@main#models/my-model-v1` |

> You may use a local path for `name_or_path` during testing, but this is not
> supported for demo deployments. ORAS artifacts are preferred.

If you use a local model directory for development, choose a directory name
that satisfies [SEP-986][] tool naming constraints,
use a distinct directory name for each weight revision, and include a model
card matching the {repo_file}`template <container/model-card.md>`.

### Upload Weights to Hugging Face

Use Hugging Face when you want to publish the model directory as a Hub model
repository. Follow Hugging Face's
[uploading models guide](https://huggingface.co/docs/hub/en/models-uploading#uploading-models)
to create the repository and upload the files.

Once the model is uploaded, reference it with an `hf://` URI:

```text
name_or_path: hf://my-org/my-model-v2
```

If the model files are inside a subdirectory of the repository, add that
subdirectory as a fragment:

```text
name_or_path: hf://my-org/my-model-v2#models/my-model-v2
```

### Push Weights to ORAS

Use [ORAS](https://oras.land/) when you want model weights in an
OCI-compatible registry, separate from your source repository. Start with a
[Hugging Face-style model directory](https://huggingface.co/docs/hub/models):

```text
models/my-model-v1/
  config.json
  model.safetensors
  tokenizer.json
```

[Install ORAS](https://oras.land/docs/installation/), log in to your registry,
then push the model directory from the project root:

```bash
oras login registry.example.org
oras push \
  registry.example.org/my-org/my-model:v1 \
  models/my-model-v1/
```

Reference the pushed artifact in `nomad.yml` with the model directory as a
subpath:

```text
name_or_path: oras://registry.example.org/my-org/my-model:v1#models/my-model-v1
```

For reproducible deployments, resolve the tag to a digest and use the pinned
URI:

```bash
oras resolve registry.example.org/my-org/my-model:v1
```

```text
name_or_path: oras://registry.example.org/my-org/my-model@sha256:...#models/my-model-v1
```

If the registry is private, make sure the environment that runs Nomad can
authenticate to it before startup.

### Store Weights with Git LFS

Weights hosted within your project's repository MUST be stored with
[Git LFS](https://git-lfs.com/). Install Git LFS before continuing.
The model directory should still match the
[Hugging Face model repository format](https://huggingface.co/docs/hub/models).
Then, within your project's repository:

```shell
git lfs install
git lfs track '*.safetensors' '*.pth' # Plus any other large files used by your model
git add .gitattributes
git add path/to/model_directory
git commit -m "Adding model"
git push
```


## Validate

Start the nomad MCP server by running:

```bash
nomad serve --transport http --host localhost --port 8000 path/to/nomad.yml
```

Expected output (truncated):

```text
INFO:     Application startup complete.
INFO:     Uvicorn running on http://localhost:8000
```

Choose a validation surface based on what you need to test:

| Validation task | Useful surface |
| --- | --- |
| Inspect schemas or make one small JSON-friendly call | [MCP Inspector][] |
| Construct typed inputs, deserialize tensor-bearing outputs, or keep a reusable client test | A [FastMCP client](https://gofastmcp.com/clients/client) script or Nomad's code-mode gateway |
| Make many calls, compose tools, aggregate results, or create plots and other artifacts | A FastMCP client script or Nomad's code-mode gateway |

Inspector can confirm that a simple tool is reachable, but it is not always a
useful scientific-validation workflow. For example, mapping a scalar model over
hundreds of latitude, longitude, and depth coordinates is better handled in a
script even though each individual request is JSON-friendly.

### Programmatic Validation

If your {py:class}`~nomad.fm_base_tool.TorchModuleTool` accepts or returns
{py:class}`~nomad.well_format.WellFormat`,
{py:class}`~nomad.well_format.AutoRegressiveInput`, or
{py:data}`nomad.well_format.Tensor` fields, use a FastMCP client or code-mode
script to construct inputs and reconstruct outputs. Tensor values cross MCP as
compressed base64-encoded torch serializations; do not copy those serialized
values into Inspector by hand.

Nomad's code-mode gateway is especially useful when you want to explore the
model iteratively, make many calls, or turn results into plots. The following
example uses [`nomad code-mode-exec`](cli-nomad-code-mode-exec) to run a
reviewable script.

Create `gateway.yml`:

```yaml
servers:
  nomad:
    transport: http
    url: http://localhost:8000/mcp
```

Then create `test_rollout.py`. The generated Python wrapper name is the
sanitized form of your MCP tool name, so `my-model-v2` becomes `my_model_v2`:

```python
from mcp_tools.nomad import my_model_v2
from nomad.well_format import WellFormat

initial_state = WellFormat.from_file("path/to/input.h5")

# Nomad decodes tensor fields returned by wrappers, but does not
# reconstruct the outer Pydantic object automatically.
rollout = my_model_v2(
    duration=10,
    initial_state=initial_state,
)

# Reconstruct WellFormat from the returned plain-dict output
rollout = WellFormat.model_validate(rollout)
rollout.to_file("path/to/output.h5")
```

Run the script in the code-mode sandbox:

```bash
nomad code-mode-exec --config gateway.yml test_rollout.py
```

### Connect an Agent to the Code-Mode Gateway

An agent connected to the gateway can use
{py:meth}`execute_mcp_script <nomad.gateway.server.CodeModeGateway.execute_mcp_script>`
for the same reviewable, script-oriented workflow. Configure your agent to
launch the gateway over stdio, replacing `/absolute/path/to/gateway.yml` with
the absolute path to the file above.

Once connected, ask the agent to:

- Create a plot of my model's predictions.
- Explore the predicted result of sweeping these parameters: ...
- Compare the model's predictions for these inputs: ...

::::{tab-set}
:::{tab-item} Codex

Add this to `~/.codex/config.toml` or a trusted project's
`.codex/config.toml`:

```toml
[mcp_servers.nomad-code-mode]
command = "uv"
args = [
  "run",
  "nomad",
  "code-mode",
  "--config",
  "/absolute/path/to/gateway.yml",
]
```

See the [Codex MCP documentation](https://developers.openai.com/codex/mcp)
for other configuration methods and options.

Restart Codex or use `/mcp` to confirm that `nomad-code-mode` is connected.
:::

:::{tab-item} Claude Code

Add this project-scoped server to `.mcp.json`:

```json
{
  "mcpServers": {
    "nomad-code-mode": {
      "type": "stdio",
      "command": "uv",
      "args": [
        "run",
        "nomad",
        "code-mode",
        "--config",
        "/absolute/path/to/gateway.yml"
      ]
    }
  }
}
```

See the [Claude Code MCP documentation](https://code.claude.com/docs/en/mcp)
for configuration scopes, approvals, and other setup methods.

Start Claude Code, approve the project server when prompted, and use `/mcp`
to confirm the connection.
:::

:::{tab-item} URSA
:selected:

Add the gateway to the `mcp_servers` section of your URSA configuration:

```yaml
mcp_servers:
  nomad-code-mode:
    transport: stdio
    command: uv
    args:
      - run
      - nomad
      - code-mode
      - --config
      - /absolute/path/to/gateway.yml
```

See the [URSA configuration documentation](https://lanl.github.io/ursa/configuration/files-and-env/)
for reusable configuration files and environment handling.

Start URSA with `ursa --config ursa.yml`.
:::
::::

For a fuller worked example using `WellFormat` inputs and a PDE surrogate, see
the {doc}`Nomad Inference notebook </guides/nomad_inference>`.


## Demo Deployment Additions

If you are wiring a new model into the repository's demo deployment scaffold,
update these files together:

- {repo_file}`container/demo/pyproject.toml <container/demo/pyproject.toml>`:
  add your model package to `dependencies`, and add or update the matching
  `[tool.uv.sources]` entry if the package comes from Git or a local checkout.
- {repo_file}`container/demo/nomad.yml <container/demo/nomad.yml>`:
  add your `fmod_models` entry, plus any helper `tools` entries and
  `tool_manager` settings needed for batching or queue depth.
- {repo_file}`container/model-card.md <container/model-card.md>`:
  use this template when you are shipping a local model directory that needs a
  model card in the exported bundle.

Paths in
{repo_file}`container/demo/nomad.yml <container/demo/nomad.yml>`
are resolved relative to that file. During image builds,
{ref}`the demo image build process <building-the-image>`
runs `nomad export` before building the container, so local model assets
referenced by `name_or_path` are copied into the build context automatically.

For the full demo deployment workflow, see the
{doc}`Deployment Guide </deployments/guide>`.

[uv]: https://docs.astral.sh/uv/
[MCP Inspector]: https://modelcontextprotocol.io/docs/tools/inspector
[codex]: https://chatgpt.com/codex/
[claude-code]: https://code.claude.com/docs/en/overview
[ursa]: https://lanl.github.io/ursa/
