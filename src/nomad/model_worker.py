from __future__ import annotations

import contextlib
import gc
import importlib
import os
import sys
import traceback
from collections.abc import Mapping
from typing import Any, TextIO

from nomad.jsonrpc import Dispatcher, Handler, encode_line, encode_response


def _resolve_symbol(dotted_path: str) -> Any:
    module_path, separator, attribute = dotted_path.rpartition(".")
    if not separator:
        raise ValueError(f"Invalid dotted path '{dotted_path}'")
    return getattr(importlib.import_module(module_path), attribute)


class ModelWorker:
    def __init__(self) -> None:
        self.tool: Any = None
        self.should_stop = False

    def handlers(self) -> dict[str, Handler]:
        return {
            "load": self.load,
            "run_batch": self.run_batch,
            "offload": lambda _params: self.offload(),
            "unload": lambda _params: self._discard_tool(),
            "clear_cache": lambda _params: self.clear_cache(),
            "device_memory": self.device_memory,
            "set_batch_size": self.set_batch_size,
            "shutdown": self.shutdown,
        }

    def load(self, params: Mapping[str, Any]) -> dict[str, Any]:
        self._discard_tool()
        model_class = _resolve_symbol(str(params["model_class"]))
        if not hasattr(model_class, "from_pretrained"):
            raise TypeError(
                f"{params['model_class']} does not expose a 'from_pretrained' constructor"
            )
        tool = model_class.from_pretrained(str(params["source"]))
        required = ("fm", "args_schema", "output_schema", "batch_as_completed")
        if any(not hasattr(tool, attribute) for attribute in required):
            raise TypeError("Instantiated object is not a TorchModuleTool")
        if params.get("tool_name"):
            tool.name = str(params["tool_name"])
        if params.get("batch_size") is not None:
            tool.batch_size = max(1, int(params["batch_size"]))
        tool.fm = tool.fm.to(str(params.get("device", "cpu")))
        try:
            import torch

            tool.device = torch.device(str(params.get("device", "cpu")))
        except ImportError:  # pragma: no cover - TorchModuleTool already needs torch
            pass
        self.tool = tool
        return {
            "name": tool.name or model_class.__name__,
            "description": tool.description,
            "batch_size": max(1, int(tool.batch_size or 1)),
            "args_schema": tool.args_schema.model_json_schema(),
            "output_schema": tool.output_schema.model_json_schema(),
        }

    def run_batch(self, params: Mapping[str, Any]) -> list[Any]:
        tool = self._require_tool()
        inputs = [tool.args_schema.model_validate(item) for item in params["inputs"]]
        outputs = tool.batch_as_completed(
            inputs,
            max_concurency=max(1, int(params["batch_size"])),
        )
        return [
            tool.output_schema.model_validate(output).model_dump(mode="json")
            for output in outputs
        ]

    def set_batch_size(self, params: Mapping[str, Any]) -> None:
        self._require_tool().batch_size = max(1, int(params["batch_size"]))

    def shutdown(self, _params: Mapping[str, Any]) -> None:
        self.should_stop = True

    def offload(self) -> None:
        tool = self._require_tool()
        tool.fm = tool.fm.to("cpu")
        try:
            import torch

            tool.device = torch.device("cpu")
        except ImportError:  # pragma: no cover
            pass

    @staticmethod
    def clear_cache() -> None:
        gc.collect()
        try:
            import torch
        except ImportError:  # pragma: no cover
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        mps = getattr(torch, "mps", None)
        if mps is not None and hasattr(mps, "empty_cache"):
            mps.empty_cache()

    @staticmethod
    def device_memory(params: Mapping[str, Any]) -> dict[str, int]:
        try:
            import torch
        except ImportError:  # pragma: no cover
            return {}

        device = torch.device(str(params["device"]))
        if device.type == "cuda":
            index = (
                device.index
                if device.index is not None
                else torch.cuda.current_device()
            )
            return {
                "allocated": int(torch.cuda.memory_allocated(index)),
                "reserved": int(torch.cuda.memory_reserved(index)),
            }
        if device.type == "mps":
            result = {}
            if hasattr(torch.mps, "current_allocated_memory"):
                result["allocated"] = int(torch.mps.current_allocated_memory())
            if hasattr(torch.mps, "driver_allocated_memory"):
                result["reserved"] = int(torch.mps.driver_allocated_memory())
            return result
        return {}

    def _discard_tool(self) -> None:
        if self.tool is None:
            return
        self.offload()
        self.tool = None
        gc.collect()

    def _require_tool(self) -> Any:
        if self.tool is None:
            raise RuntimeError("No model is loaded")
        return self.tool


def _print_exception(exc: Exception) -> None:
    traceback.print_exception(exc, file=sys.stderr)


def main(stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout) -> int:
    worker = ModelWorker()
    protocol_stdout = stdout
    close_protocol_stdout = False
    if stdout is sys.stdout and hasattr(stdout, "fileno"):
        protocol_fd = os.dup(stdout.fileno())
        os.dup2(sys.stderr.fileno(), stdout.fileno())
        protocol_stdout = os.fdopen(
            protocol_fd,
            "w",
            encoding=stdout.encoding or "utf-8",
            buffering=1,
        )
        close_protocol_stdout = True

    try:
        with contextlib.redirect_stdout(sys.stderr):
            dispatcher = Dispatcher(worker.handlers(), on_error=_print_exception)
            for line in stdin:
                response = dispatcher.dispatch(line)
                if response is not None:
                    protocol_stdout.write(encode_response(response, encode_line))
                    protocol_stdout.flush()
                if worker.should_stop:
                    break
    finally:
        if close_protocol_stdout:
            protocol_stdout.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
