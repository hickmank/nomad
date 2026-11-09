from __future__ import annotations

import gc
import importlib
import sys
from collections.abc import Mapping
from typing import Any, TextIO

import torch

from nomad import accelerator
from nomad._torch_module_compat import is_torch_module_tool_instance
from nomad.jsonrpc import JSONRPCServer, rpc_method


def _resolve_symbol(dotted_path: str) -> Any:
    module_path, separator, attribute = dotted_path.rpartition(".")
    if not separator:
        raise ValueError(f"Invalid dotted path '{dotted_path}'")
    return getattr(importlib.import_module(module_path), attribute)


class ModelWorker(JSONRPCServer):
    def __init__(self) -> None:
        super().__init__()
        self.tool: Any = None
        # Identifies which model ``self.tool`` was loaded from, so ``run_batch``
        # can tell whether the requested model is already resident.
        self._loaded_key: tuple[str, str, str | None] | None = None

    @rpc_method
    def load(self, params: Mapping[str, Any]) -> dict[str, Any]:
        self._discard_tool()
        model_class = _resolve_symbol(str(params["model_class"]))
        if not hasattr(model_class, "from_pretrained"):
            raise TypeError(
                f"{params['model_class']} does not expose a 'from_pretrained' constructor"
            )
        tool = model_class.from_pretrained(str(params["source"]))
        if not is_torch_module_tool_instance(tool):
            raise TypeError("Instantiated object is not a TorchModuleTool")
        if params.get("tool_name"):
            tool.name = str(params["tool_name"])
        if params.get("batch_size") is not None:
            tool.batch_size = max(1, int(params["batch_size"]))
        tool.fm = tool.fm.to(str(params.get("device", "cpu")))
        tool.device = torch.device(str(params.get("device", "cpu")))
        self.tool = tool
        self._loaded_key = _model_key(params)
        return {
            "name": tool.name or model_class.__name__,
            "description": tool.description,
            "batch_size": max(1, int(tool.batch_size or 1)),
            "args_schema": tool.args_schema.model_json_schema(),
            "output_schema": tool.output_schema.model_json_schema(),
        }

    @rpc_method
    def run_batch(self, params: Mapping[str, Any]) -> list[Any]:
        # The request carries the model's load parameters, so a worker whose
        # model was unloaded or offloaded can always serve it; there is no
        # state in which run_batch fails for lack of a loaded model.
        tool = self._ensure_tool(params["model"])
        inputs = [tool.args_schema.model_validate(item) for item in params["inputs"]]
        outputs = tool.batch_as_completed(
            inputs,
            max_concurency=max(1, int(params["batch_size"])),
        )
        return [
            tool.output_schema.model_validate(output).model_dump(mode="json")
            for output in outputs
        ]

    @rpc_method
    def set_batch_size(self, params: Mapping[str, Any]) -> None:
        # With no model loaded there is nothing to resize; run_batch passes its
        # batch size on every call anyway.
        if self.tool is not None:
            self.tool.batch_size = max(1, int(params["batch_size"]))

    @rpc_method
    def shutdown(self, _params: Mapping[str, Any]) -> None:
        self.stop()

    @rpc_method
    def offload(self, _params: Mapping[str, Any] | None = None) -> None:
        if self.tool is not None:
            self._move_tool(torch.device("cpu"))

    @rpc_method
    def unload(self, _params: Mapping[str, Any]) -> None:
        self._discard_tool()

    @staticmethod
    @rpc_method
    def clear_cache(_params: Mapping[str, Any]) -> None:
        accelerator.empty_accelerator_cache()

    @staticmethod
    @rpc_method
    def device_memory(params: Mapping[str, Any]) -> dict[str, int]:
        return accelerator.device_memory(torch.device(str(params["device"])))

    def _discard_tool(self) -> None:
        if self.tool is None:
            return
        self.offload()
        self.tool = None
        self._loaded_key = None
        gc.collect()

    def _ensure_tool(self, model_params: Mapping[str, Any]) -> Any:
        """Return the requested model, loading or moving it onto its device.

        Args:
            model_params: The same mapping ``load`` accepts.

        Returns:
            The loaded tool, placed on ``model_params["device"]``.
        """
        if self.tool is None or self._loaded_key != _model_key(model_params):
            self.load(model_params)
        device = torch.device(str(model_params.get("device", "cpu")))
        if self.tool.device != device:
            # The model was offloaded (or loaded elsewhere); move it back.
            self._move_tool(device)
        return self.tool

    def _move_tool(self, device: torch.device) -> None:
        self.tool.fm = self.tool.fm.to(device)
        self.tool.device = device


def _model_key(params: Mapping[str, Any]) -> tuple[str, str, str | None]:
    """Return the fields that identify which model ``params`` loads."""
    tool_name = params.get("tool_name")
    return (
        str(params["model_class"]),
        str(params["source"]),
        str(tool_name) if tool_name else None,
    )


def main(stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout) -> int:
    return ModelWorker().start(stdin, stdout)


if __name__ == "__main__":
    raise SystemExit(main())
