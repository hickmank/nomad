# Tool Manager Reference

The tool manager is the runtime that makes multiple
{py:class}`nomad.fm_base_tool.TorchModuleTool` models practical to serve from a
single Nomad process. It keeps track of available devices, loads tools onto an
accelerator when they need to run, and can offload idle tools back to CPU so
GPU memory can be shared across a larger set of models.

A device slot is the manager's internal record for one schedulable device. It
tracks the model currently assigned to that device and, for isolated models,
the Python subprocess serving it. There is no separate "slot manager" type.

It also queues requests, batches calls for tools that support batching, and can
register the managed tools with FastMCP.

For managed Torch tools, the manager keeps one resident CPU instance when the
tool is loaded from configuration and may create additional tool instances from
the configured source when a busy tool is assigned to more than one device slot.
Idle tools scale down one device allocation per
{py:attr}`~nomad.config.ToolManagerConfig.idle_seconds` interval. After a tool
is fully removed from device slots,
{py:attr}`~nomad.config.ToolManagerConfig.disk_idle_seconds` controls when the
resident CPU instance may be dropped; a later request reloads it with
`TorchModuleTool.from_pretrained(source)`. Separately,
{py:attr}`~nomad.config.ToolManagerConfig.gc_idle_seconds` controls when unused
Python and accelerator caches are cleared while the server is idle.

## Model environments

Models run in per-device Python subprocesses. When `env` is omitted, the
subprocess uses the same virtual environment and Python executable as the Nomad
host. For explicit environments, Nomad prefers `uv` to create the virtual
environment and falls back to the standard-library `venv` module. Explicit
environments are content-addressed under
`$XDG_CACHE_HOME/nomad/venv` (or `~/.cache/nomad/venv`) and reused across
models with the same requirements. Nomad injects the currently running Nomad
package into each explicit environment's requirements. A slot keeps its
subprocess when switching between models with the same environment;
`tool_manager.venv_idle_seconds` controls when an idle subprocess is terminated
and defaults to 1200 seconds. `tool_manager.rpc_timeout_seconds` limits each
subprocess request to 30 seconds by default; set it to `null` to disable the
deadline. `tool_manager.build_envs` (default `true`) lets Nomad build a missing
`env` environment on demand; when it is `false`, a model whose environment is
not already built fails to load instead of downloading packages. The tool
manager is always enabled and controls batching, device assignment, and model
subprocess lifecycles.

```{eval-rst}
.. automodule:: nomad.torch_tool_manager
```
