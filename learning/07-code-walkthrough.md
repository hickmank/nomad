# 07: Reading the Code

Now that you have used Nomad from the outside, trace a request through the
code. Open each file and answer the questions before moving on. This lesson
does not need the demo server.

## 1. Startup: `src/nomad/cli.py`

Read `serve` (around line 131) and `_serve` below it.

- Where is the config loaded, and how does `context_dir` get set? (See
  `ServerConfig.from_file` in `src/nomad/config.py`.)
- In what order are tools registered? (Answer: `get_model_card`, then
  `tools`, then `fmod_models`, then `search_tools`.) Why must
  `search_tools` come last?
- What happens to a model that fails to load, with `--strict` and with
  `--no-strict`?
- When is `TorchModelToolManager` used, and when is a model added directly to
  FastMCP?

## 2. Config Schema: `src/nomad/config.py`

- `ServerConfig` (around line 308): list its fields.
- `TorchModuleConfig.build_tool` (around line 143): how is the tool name
  chosen when `tool_name` is not set? Compare with the demo names you saw in
  lesson 02.
- `resolve_symbol`: how are dotted paths like `nomad_demo.mist.FinetunedMistModel`
  imported?
- `src/nomad/hub.py`, around lines 450 to 485: trace how
  `mist-models/mist-26.9M-...` becomes an `hf://` source.

## 3. The Adapter Contract: `src/nomad/fm_base_tool.py`

`TorchModuleTool` is a Pydantic model with four type parameters:
`[Input, Output, ModelInput, ModelOutput]`.

- `preprocess(list[Input]) -> ModelInput`
- `_forward(ModelInput) -> ModelOutput` (default: `self.fm(x).to("cpu")`)
- `postprocess(ModelOutput) -> Iterable[Output]`, which must yield exactly
  one output per input
- `from_pretrained(name_or_path)` builds a fresh instance. Do not override
  `__init__`.
- `model_post_init` moves `fm` to `self.device`.

## 4. A Real Adapter: `container/demo/src/nomad_demo/mist.py`

- What are `args_schema` and `output_schema`? Find them in the Inspector
  schema from lesson 02.
- How does `preprocess` pad a batch of SMILES of different lengths?
- Why does `_forward` pass `molecules` through to `postprocess`?
- Where do the property names, units, and descriptions come from?

Also read `pubchem.py`, a regular tool: a typed function with a docstring.
FastMCP builds the schema from the type hints.

## 5. Scheduling: `src/nomad/torch_tool_manager.py`

Start at `TorchModelToolManager` (around line 195), then read `call_tool`,
`_dispatch_loop`, `_next_batch_size`, `_next_available_requests`,
`_execute_request`, `_evict_idle_tools`, and `_offload_slot`.

- How are requests that arrive at the same time grouped into one batch?
- What is a "slot", and how does `max_devices_per_tool` limit them?
- What happens on an out-of-memory error (`_is_out_of_memory`)?
- Which settings control offloading (`idle_seconds`), cache clearing
  (`gc_idle_seconds`), and dropping a model from memory
  (`disk_idle_seconds`)?

## 6. Code Mode: `src/nomad/gateway/`

1. `server.py`, `CodeModeGateway`: the three MCP tools and `run_script`.
2. `upstream.py`: connections to the upstream MCP servers.
3. `sandbox.py`: how the subprocess is launched, its `PYTHONPATH`, its
   working directory, and the output limits.
4. `wrappers.py`: copies `nomad` into a temporary folder and writes
   `mcp_tools/__init__.py`.
5. `runtime/client.py`: `WrapperFinder`, the import hook, and the bridge
   back to the gateway.
6. `runtime/wrapper_factory.py`: builds `foo` and `foo_async` from a tool
   schema.
7. `runtime/deserialization.py`: turns tensor fields back into
   `torch.Tensor`.
8. `runtime/runner.py`: runs the script and collects `RESULT`.

Question: why is the wrapper module built inside the sandbox from metadata,
instead of being generated as source files?

## 7. Tests as Documentation

Read `test/test_config.py`, `test/test_torch_tool_manager_basic.py`, and
`test/gateway/test_gateway.py`. Tests are often the clearest description of
how something is meant to behave.

Next: [08 Your first CPU tool](08-first-cpu-tool.md).
