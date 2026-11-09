"""Accelerator memory helpers shared by the Nomad host and model workers."""

from __future__ import annotations

import gc

import torch


def empty_accelerator_cache() -> None:
    """Release unused Python and accelerator memory back to the runtime."""
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        # torch.accelerator.memory asserts on MPS, which lacks a DeviceAllocator.
        torch.mps.empty_cache()
    else:
        accelerator = getattr(torch, "accelerator", None)
        if accelerator is not None and accelerator.is_available():
            accelerator.memory.empty_cache()


def device_memory(device: torch.device) -> dict[str, int]:
    """Return best-effort memory usage for ``device`` in this process.

    Args:
        device: The device to measure.

    Returns:
        A mapping with ``"allocated"`` and/or ``"reserved"`` byte counts. Keys
        are omitted when the backend cannot report them, so CPU devices yield an
        empty mapping.
    """
    if device.type == "cuda":
        index = (
            device.index if device.index is not None else torch.cuda.current_device()
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
