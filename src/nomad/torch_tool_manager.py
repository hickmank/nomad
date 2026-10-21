import asyncio
import contextlib
import gc
import logging
import threading
import time
import weakref
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import torch
from fastmcp import FastMCP
from fastmcp.tools import FunctionTool as FastMCPTool
from pydantic import BaseModel

from . import metrics as nomad_metrics
from ._torch_module_compat import build_torch_module_fastmcp_tool
from .config import ToolManagerConfig
from .fm_base_tool import TorchModuleTool
from .metrics import Observation
from .model_rpc import ModelProcess, RemoteModel

logger = logging.getLogger(__name__)
T = TypeVar("T")
_DEVICE_MEMORY_SNAPSHOT_SECONDS = 1.0


def empty_accelerator_cache() -> None:
    """Release unused Python and accelerator memory back to the runtime."""
    gc.collect()

    accelerator = getattr(torch, "accelerator", None)
    if accelerator is not None and accelerator.is_available():
        accelerator.memory.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


@dataclass(frozen=True, slots=True)
class AcceleratorInfo:
    """Metadata describing an available accelerator device."""

    device: torch.device
    kind: str
    index: int | None
    name: str | None


@dataclass(slots=True)
class DeviceSlot:
    """Tracks which tool currently occupies a device."""

    device: torch.device
    current_tool: str | None = None
    tool: TorchModuleTool | None = None
    worker: ModelProcess | None = None
    worker_tool: str | None = None
    busy: bool = False
    last_used: float = field(default_factory=time.monotonic)


@dataclass(slots=True)
class ToolRequest:
    """Represents a pending tool invocation."""

    input: BaseModel | dict[str, Any]
    future: asyncio.Future
    enqueued_at: float = field(default_factory=time.monotonic)
    metrics_recorded: bool = False


@dataclass(slots=True)
class ToolState:
    """Queue and metadata for a registered tool."""

    name: str
    registered_name: str
    tool: TorchModuleTool | None
    cls: type[TorchModuleTool] | None
    source: str | Path
    args_schema: type[BaseModel] | dict[str, Any]
    output_schema: type[BaseModel] | dict[str, Any]
    description: str
    remote: RemoteModel | None = None
    queue: deque[ToolRequest] = field(default_factory=deque)
    enqueued: bool = False
    batch_size: int = 1
    assigned_slots: set[int] = field(default_factory=set)
    loaded_slots: set[int] = field(default_factory=set)
    last_used: float = field(default_factory=time.monotonic)
    resident_slot: int | None = None

    def reserve_slot(self, device_index: int) -> None:
        self.assigned_slots.add(device_index)

    def release_slot(self, device_index: int) -> None:
        self.loaded_slots.discard(device_index)
        self.assigned_slots.discard(device_index)
        if self.resident_slot == device_index:
            self.resident_slot = None

    def mark_slot_loaded(
        self,
        device_index: int,
        *,
        now: float,
        is_resident: bool,
    ) -> None:
        self.loaded_slots.add(device_index)
        if is_resident:
            self.resident_slot = device_index
        self.last_used = now

    def mark_slot_free(
        self,
        device_index: int,
        *,
        now: float,
        remains_loaded: bool,
    ) -> None:
        if remains_loaded:
            self.last_used = now
        else:
            self.release_slot(device_index)

    def load_new_tool(self, *, is_resident_load: bool) -> TorchModuleTool:
        if self.cls is None:
            raise RuntimeError("Isolated tools are loaded by their model subprocess")
        start = time.monotonic()
        status = "ok"
        try:
            replica = self.cls.from_pretrained(str(self.source))
            replica.name = self.name
            replica.batch_size = self.batch_size
            return replica
        except Exception:
            status = "error"
            raise
        finally:
            nomad_metrics.record_tool_disk_load(
                self.registered_name,
                time.monotonic() - start,
                load_kind="resident" if is_resident_load else "replica",
                status=status,
            )


def _detect_accelerators() -> list[AcceleratorInfo]:
    """Discover accelerators on the current host.

    Preference order is CUDA, Apple MPS, then CPU fallback.
    """
    accelerators: list[AcceleratorInfo] = []

    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            device = torch.device(f"cuda:{idx}")
            try:
                name = torch.cuda.get_device_name(idx)
            except Exception:  # pragma: no cover - defensive
                name = None
            accelerators.append(
                AcceleratorInfo(
                    device=device,
                    kind="cuda",
                    index=idx,
                    name=name,
                )
            )

    else:
        mps_backend = getattr(torch.backends, "mps", None)
        if (
            mps_backend is not None
            and getattr(mps_backend, "is_available", lambda: False)()
        ):
            device = torch.device("mps")
            accelerators.append(
                AcceleratorInfo(
                    device=device,
                    kind="mps",
                    index=0,
                    name="mps",
                )
            )

    if not accelerators:
        cpu_device = torch.device("cpu")
        accelerators.append(
            AcceleratorInfo(
                device=cpu_device,
                kind="cpu",
                index=None,
                name="cpu",
            )
        )

    return accelerators


class TorchModelToolManager:
    """Coordinate execution of multiple ``TorchModuleTool`` instances across devices."""

    def __init__(
        self,
        config: ToolManagerConfig | None = None,
        *,
        device_provider: Callable[[], list[torch.device]] | None = None,
    ) -> None:
        """
        Args:
            config: Tool manager configuration. Defaults to
                :class:`nomad.config.ToolManagerConfig`.
            device_provider: Optional function returning the devices to manage.
        """
        config = config or ToolManagerConfig()
        idle_seconds = config.idle_seconds
        gc_idle_seconds = config.gc_idle_seconds
        disk_idle_seconds = config.disk_idle_seconds
        venv_idle_seconds = config.venv_idle_seconds
        rpc_timeout_seconds = config.rpc_timeout_seconds
        self.idle_seconds = idle_seconds
        self.gc_idle_seconds = gc_idle_seconds
        self.disk_idle_seconds = disk_idle_seconds
        self.venv_idle_seconds = venv_idle_seconds
        self.rpc_timeout_seconds = rpc_timeout_seconds
        max_pending_per_tool = config.max_pending_per_tool
        if max_pending_per_tool is not None and max_pending_per_tool < 1:
            raise ValueError("max_pending_per_tool must be >= 1 or None")
        self._max_pending_per_tool = max_pending_per_tool
        threshold = float(idle_seconds) if idle_seconds is not None else None
        self._idle_threshold = None if threshold is None else max(threshold, 0.0)
        disk_threshold = (
            float(disk_idle_seconds) if disk_idle_seconds is not None else None
        )
        self._disk_idle_threshold = (
            None if disk_threshold is None else max(disk_threshold, 0.0)
        )
        venv_threshold = (
            float(venv_idle_seconds) if venv_idle_seconds is not None else None
        )
        self._venv_idle_threshold = (
            None if venv_threshold is None else max(venv_threshold, 0.0)
        )
        janitor_intervals = [
            self._janitor_interval_for_threshold(configured_threshold)
            for configured_threshold in (
                self._idle_threshold,
                self._disk_idle_threshold,
                self._venv_idle_threshold,
            )
            if configured_threshold is not None
        ]
        self._janitor_interval = min(janitor_intervals) if janitor_intervals else None
        gc_threshold = float(gc_idle_seconds) if gc_idle_seconds is not None else None
        self._gc_idle_threshold = (
            None if gc_threshold is None else max(gc_threshold, 0.0)
        )

        if device_provider is not None:
            provided_devices = list(device_provider())
            if not provided_devices:
                raise ValueError("device_provider must return at least one device")
            provided_devices = self._normalize_devices(provided_devices)
            self._accelerators = [
                AcceleratorInfo(
                    device=torch.device(dev),
                    kind=torch.device(dev).type,
                    index=torch.device(dev).index,
                    name=str(dev),
                )
                for dev in provided_devices
            ]
        else:
            self._accelerators = _detect_accelerators()

        self._device_slots: list[DeviceSlot] = [
            DeviceSlot(device=info.device) for info in self._accelerators
        ]
        if not self._device_slots:
            raise RuntimeError("No devices detected for TorchModelToolManager")

        max_devices_per_tool = config.max_devices_per_tool
        if max_devices_per_tool is None:
            max_devices_per_tool = len(self._device_slots)
        if max_devices_per_tool < 1:
            raise ValueError("max_devices_per_tool must be >= 1")
        self._max_devices_per_tool = max_devices_per_tool

        self._tools: dict[str, ToolState] = {}
        self._inflight_by_tool: dict[str, int] = {}
        self._worker_memory_cache_lock = threading.Lock()
        self._worker_memory_snapshots: weakref.WeakKeyDictionary[
            ModelProcess,
            dict[str, tuple[float, dict[str, int] | None]],
        ] = weakref.WeakKeyDictionary()
        self._worker_memory_locks: weakref.WeakKeyDictionary[
            ModelProcess,
            dict[str, threading.Lock],
        ] = weakref.WeakKeyDictionary()

        self._condition = asyncio.Condition()
        self._request_futures: set[asyncio.Future] = set()
        self._pending_tools: deque[str] = deque()
        self._rr_index: int = 0
        self._last_server_activity = time.monotonic()

        self._started = False
        self._closed = False
        self._tasks: set[asyncio.Task] = set()
        nomad_metrics.register_tool_manager(self)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def devices(self) -> list[torch.device]:
        """Devices managed by this instance."""
        return [info.device for info in self._accelerators]

    @property
    def accelerator_info(self) -> list[AcceleratorInfo]:
        """Detailed accelerator metadata."""
        return list(self._accelerators)

    @staticmethod
    def _normalize_devices(devices: list[torch.device]) -> list[torch.device]:
        normalized: list[torch.device] = []
        seen: set[tuple[str, int | None]] = set()
        cpu_seen = False
        for raw_device in devices:
            device = torch.device(raw_device)
            if device.type == "cpu":
                if cpu_seen:
                    continue
                cpu_seen = True
                device = torch.device("cpu")
            key = (device.type, device.index)
            if key in seen:
                continue
            seen.add(key)
            normalized.append(device)
        return normalized

    @staticmethod
    def _janitor_interval_for_threshold(threshold: float) -> float:
        return 0.1 if threshold == 0 else max(0.1, threshold / 2)

    def register_tool(
        self,
        name: str,
        tool: TorchModuleTool,
        *,
        source: str | Path,
    ) -> None:
        """Register a tool for managed execution."""
        if name in self._tools:
            raise ValueError(f"Tool '{name}' is already registered")

        effective_batch = max(1, tool.batch_size or 1)
        tool.batch_size = effective_batch
        self._tools[name] = ToolState(
            name=tool.name or name,
            registered_name=name,
            tool=tool,
            cls=tool.__class__,
            source=source,
            args_schema=tool.args_schema,
            output_schema=tool.output_schema,
            description=tool.description,
            batch_size=effective_batch,
        )
        self._inflight_by_tool[name] = 0
        self._offload_tool(tool)

    def register_remote_tool(self, remote: RemoteModel) -> None:
        """Register a model whose implementation lives in a subprocess."""
        name = remote.name
        if name in self._tools:
            raise ValueError(f"Tool '{name}' is already registered")

        self._tools[name] = ToolState(
            name=name,
            registered_name=name,
            tool=None,
            cls=None,
            source=remote.source,
            args_schema=remote.args_schema,
            output_schema=remote.output_schema,
            description=remote.description,
            remote=remote,
            batch_size=max(1, remote.batch_size),
        )
        self._inflight_by_tool[name] = 0

    def add_to_fastmcp(self, server: FastMCP) -> dict[str, FastMCPTool]:
        """Register all managed tools with a FastMCP server."""
        fast_tools: dict[str, FastMCPTool] = {}
        for name, state in self._tools.items():
            fast_tool = self._build_fastmcp_tool(name, state)
            fast_tools[name] = server.add_tool(fast_tool)

        return fast_tools

    async def call_tool(
        self,
        name: str,
        input: BaseModel | dict[str, Any] | None = None,
        /,
        **kwargs: Any,
    ) -> BaseModel | dict[str, Any]:
        """Queue a call to ``name`` and await its result."""
        if self._closed:
            nomad_metrics.record_tool_request_rejection(name, "manager_closed")
            raise RuntimeError("TorchModelToolManager is closed")

        if name not in self._tools:
            nomad_metrics.record_tool_request_rejection(name, "unknown_tool")
            raise KeyError(f"Unknown tool '{name}'")

        tool_state = self._tools[name]
        request_input = self._normalize_input(tool_state.args_schema, input, **kwargs)

        loop = asyncio.get_running_loop()
        self._ensure_started(loop)
        future: asyncio.Future = loop.create_future()
        self._request_futures.add(future)
        request = ToolRequest(input=request_input, future=future)

        try:
            try:
                await self._enqueue_request(name, request)
            except RuntimeError:
                nomad_metrics.record_tool_request_rejection(name, "queue_full")
                raise

            nomad_metrics.record_tool_request(name)

            try:
                return await asyncio.shield(future)
            except asyncio.CancelledError:
                await self._cancel_request(name, request)
                future.cancel()
                nomad_metrics.record_tool_request_cancellation(name)
                self._record_request_duration(name, request, "cancelled")
                raise
        finally:
            self._request_futures.discard(future)

    async def aclose(self) -> None:
        """Cancel background tasks and drain queues."""
        if self._closed:
            return

        self._closed = True
        async with self._condition:
            for future in self._request_futures:
                if not future.done():
                    future.set_exception(
                        RuntimeError("TorchModelToolManager is closed")
                    )
            for state in self._tools.values():
                state.queue.clear()
                state.enqueued = False
            self._pending_tools.clear()
            self._condition.notify_all()

        for task in list(self._tasks):
            task.cancel()

        # Closing workers first unblocks RPC requests that may be waiting for a
        # response from an unresponsive child.
        await asyncio.to_thread(self.close_workers)

        for task in list(self._tasks):
            with contextlib.suppress(asyncio.CancelledError):
                await task

        self._tasks.clear()

    def close_workers(self) -> None:
        """Synchronously terminate all model subprocesses owned by slots."""
        for slot in self._device_slots:
            if slot.worker is not None:
                slot.worker.close()
                slot.worker = None
                slot.worker_tool = None

    async def __aenter__(self) -> "TorchModelToolManager":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_started(self, loop: asyncio.AbstractEventLoop) -> None:
        if self._started:
            return

        dispatch_task = loop.create_task(self._dispatch_loop())
        self._track_task(dispatch_task)
        if self._janitor_interval is not None:
            janitor_task = loop.create_task(self._janitor_loop())
            self._track_task(janitor_task)
        if self._gc_idle_threshold is not None:
            gc_task = loop.create_task(self._gc_janitor_loop())
            self._track_task(gc_task)
        self._started = True

    def _track_task(self, task: asyncio.Task) -> None:
        self._tasks.add(task)

        def _discard(_):
            self._tasks.discard(task)

        task.add_done_callback(_discard)

    def _normalize_input(
        self,
        args_schema: type[BaseModel] | dict[str, Any],
        input: BaseModel | dict[str, Any] | None,
        **kwargs: Any,
    ) -> BaseModel | dict[str, Any]:
        if isinstance(args_schema, dict):
            if isinstance(input, BaseModel):
                payload = input.model_dump(mode="json")
            elif isinstance(input, dict):
                payload = dict(input)
            elif input is None:
                payload = {}
            else:
                raise TypeError("Isolated model input must be an object")
            payload.update(kwargs)
            if not payload:
                raise ValueError("No input provided for tool execution")
            return payload
        if input is not None:
            if isinstance(input, args_schema):
                return input
            if isinstance(input, BaseModel):
                return input
            if isinstance(input, dict):
                return args_schema(**input)
        if kwargs:
            return args_schema(**kwargs)
        raise ValueError("No input provided for tool execution")

    async def _enqueue_request(self, tool_name: str, request: ToolRequest) -> None:
        async with self._condition:
            state = self._tools[tool_name]

            if self._max_pending_per_tool is not None:
                active_pending = sum(
                    1 for pending in state.queue if not pending.future.cancelled()
                )
                if active_pending >= self._max_pending_per_tool:
                    raise RuntimeError(
                        f"Too many pending requests for tool '{tool_name}'"
                    )

            self._mark_server_active_locked()
            state.queue.append(request)
            if not state.enqueued:
                state.enqueued = True
                self._pending_tools.append(tool_name)
            self._condition.notify_all()

    async def _cancel_request(self, tool_name: str, request: ToolRequest) -> None:
        async with self._condition:
            state = self._tools[tool_name]
            try:
                state.queue.remove(request)
            except ValueError:
                return

            if not state.queue:
                state.enqueued = False
            if self._server_is_idle_locked():
                self._mark_server_active_locked()
            self._condition.notify_all()

    async def _dispatch_loop(self) -> None:
        try:
            while True:
                async with self._condition:
                    while True:
                        if self._closed:
                            return

                        assignment = self._next_assignment_locked()

                        if assignment is not None:
                            break

                        await self._condition.wait()

                task = asyncio.create_task(self._execute_request(*assignment))
                self._track_task(task)
        except asyncio.CancelledError:
            raise

    async def _janitor_loop(self) -> None:
        if self._janitor_interval is None:
            return

        try:
            while not self._closed:
                await asyncio.sleep(self._janitor_interval)
                await self._evict_idle_tools()
                await self._evict_disk_idle_tools()
                await self._evict_idle_workers()
        except asyncio.CancelledError:
            raise

    async def _gc_janitor_loop(self) -> None:
        threshold = self._gc_idle_threshold
        if threshold is None:
            return

        try:
            while not self._closed:
                async with self._condition:
                    deadline = self._last_server_activity + threshold
                delay = max(0.0, deadline - time.monotonic())
                if delay == 0.0 and threshold == 0:
                    delay = 0.1
                await asyncio.sleep(delay)
                await self._clear_cache_if_server_idle()
        except asyncio.CancelledError:
            raise

    async def _clear_cache_if_server_idle(self, *, now: float | None = None) -> None:
        threshold = self._gc_idle_threshold
        if threshold is None:
            return

        now = time.monotonic() if now is None else now
        should_clear = False
        activity_at_clear: float | None = None
        async with self._condition:
            if (
                self._server_is_idle_locked()
                and now - self._last_server_activity >= threshold
            ):
                should_clear = True
                activity_at_clear = self._last_server_activity
                workers = [
                    slot.worker
                    for slot in self._device_slots
                    if slot.worker is not None and slot.worker.alive
                ]

        if should_clear:
            start = time.monotonic()
            status = "ok"
            try:
                await self._run_blocking(self._clear_accelerator_caches, workers)
            except Exception:
                status = "error"
                raise
            finally:
                nomad_metrics.record_device_cache_clear(
                    time.monotonic() - start,
                    status=status,
                )
                if status == "ok":
                    async with self._condition:
                        if (
                            self._server_is_idle_locked()
                            and self._last_server_activity == activity_at_clear
                        ):
                            self._mark_gc_activity_locked()

    @staticmethod
    def _clear_accelerator_caches(workers: list[ModelProcess]) -> None:
        empty_accelerator_cache()
        for worker in workers:
            try:
                worker.request("clear_cache")
            except Exception:
                logger.warning(
                    "Failed to clear accelerator cache in model worker %s",
                    worker.pid,
                    exc_info=True,
                )
                worker.close()

    def _mark_server_active_locked(self) -> None:
        self._mark_gc_activity_locked()

    def _mark_gc_activity_locked(self, *, now: float | None = None) -> None:
        self._last_server_activity = time.monotonic() if now is None else now

    def _server_is_idle_locked(self) -> bool:
        if any(slot.busy for slot in self._device_slots):
            return False

        for state in self._tools.values():
            if any(not request.future.cancelled() for request in state.queue):
                return False
        return True

    def _next_available_device_index(self) -> int | None:
        total = len(self._device_slots)
        for offset in range(total):
            idx = (self._rr_index + offset) % total
            slot = self._device_slots[idx]
            if not slot.busy:
                return idx
        return None

    def _advance_round_robin(self, last_index: int) -> None:
        self._rr_index = (last_index + 1) % len(self._device_slots)

    def _next_assignment_locked(
        self,
    ) -> tuple[str, int, list[ToolRequest], int] | None:
        for _ in range(len(self._pending_tools)):
            tool_name = self._pending_tools.popleft()
            state = self._tools[tool_name]
            if not state.queue:
                state.enqueued = False
                continue

            device_index = self._next_available_device_index_for_tool(tool_name)
            if device_index is None:
                self._pending_tools.append(tool_name)
                continue

            batch_size = self._next_batch_size(state)
            state.enqueued = False
            requests, _ = self._next_available_requests(tool_name, batch_size)
            if not requests:
                state.enqueued = False
                continue

            slot = self._device_slots[device_index]
            slot.busy = True
            state.reserve_slot(device_index)
            self._advance_round_robin(device_index)
            return tool_name, device_index, requests, batch_size
        return None

    def _next_available_device_index_for_tool(self, tool_name: str) -> int | None:
        total = len(self._device_slots)
        for offset in range(total):
            idx = (self._rr_index + offset) % total
            slot = self._device_slots[idx]
            if not slot.busy and slot.current_tool == tool_name:
                return idx

        state = self._tools[tool_name]
        if len(state.assigned_slots) >= min(self._max_devices_per_tool, total):
            return None

        if not self._can_allocate_new_slot_fairly(tool_name):
            return None

        return self._next_available_device_index()

    def _can_allocate_new_slot_fairly(self, tool_name: str) -> bool:
        free_slots = sum(1 for slot in self._device_slots if not slot.busy)
        if free_slots <= 1:
            return True

        waiting_tools = {
            pending_tool
            for pending_tool in self._pending_tools
            if pending_tool != tool_name and self._tools[pending_tool].queue
        }
        reservations_needed = 0
        for waiting_tool in waiting_tools:
            state = self._tools[waiting_tool]
            if state.assigned_slots:
                continue
            if len(state.assigned_slots) >= min(
                self._max_devices_per_tool,
                len(self._device_slots),
            ):
                continue
            reservations_needed += 1

        return free_slots - 1 >= reservations_needed

    async def _evict_idle_tools(self, *, now: float | None = None) -> None:
        threshold = self._idle_threshold
        if threshold is None:
            return

        now = time.monotonic() if now is None else now
        evictions: list[tuple[str, int, float]] = []
        async with self._condition:
            for tool_name, state in self._tools.items():
                if not state.loaded_slots:
                    continue

                slot_entries = [
                    (idx, self._device_slots[idx]) for idx in state.loaded_slots
                ]
                if any(slot.busy for _, slot in slot_entries):
                    continue

                idle_time = now - state.last_used
                if idle_time >= threshold:
                    slot_index, slot = min(
                        slot_entries,
                        key=lambda item: item[1].last_used,
                    )
                    evictions.append((tool_name, slot_index, idle_time))

        for tool_name, slot_index, idle_time in evictions:
            await self._offload_slot(
                slot_index,
                expected_tool_name=tool_name,
                idle_time=idle_time,
                record_idle_eviction=True,
            )

    async def _evict_disk_idle_tools(self, *, now: float | None = None) -> None:
        threshold = self._disk_idle_threshold
        if threshold is None:
            return

        now = time.monotonic() if now is None else now
        remote_evictions: list[tuple[int, str, float]] = []
        async with self._condition:
            for tool_name, state in self._tools.items():
                if state.loaded_slots or state.assigned_slots:
                    continue
                if self._inflight_by_tool.get(tool_name, 0):
                    continue
                if any(not request.future.cancelled() for request in state.queue):
                    continue

                idle_time = now - state.last_used
                if idle_time < threshold:
                    continue

                if state.remote is not None:
                    for index, slot in enumerate(self._device_slots):
                        if (
                            slot.worker is not None
                            and slot.worker_tool == tool_name
                            and not slot.busy
                        ):
                            slot.busy = True
                            remote_evictions.append((index, tool_name, idle_time))
                    continue
                if state.tool is None:
                    continue
                state.tool = None
                nomad_metrics.record_tool_disk_unload(tool_name)
                state.last_used = now
                logger.debug(
                    "Unloaded tool '%s' from CPU after %.2fs fully offloaded",
                    tool_name,
                    idle_time,
                )

        for index, tool_name, idle_time in remote_evictions:
            await self._unload_slot_worker_model(
                index,
                tool_name=tool_name,
                idle_time=idle_time,
                now=now,
            )

    async def _unload_slot_worker_model(
        self,
        device_index: int,
        *,
        tool_name: str,
        idle_time: float,
        now: float,
    ) -> None:
        slot = self._device_slots[device_index]
        worker = slot.worker
        worker_failed = False
        try:
            if worker is not None and worker.alive:
                await self._run_blocking(worker.request, "unload")
            elif worker is not None:
                worker_failed = True
                await self._run_blocking(worker.close)
        except Exception:
            worker_failed = True
            logger.warning(
                "Failed to unload tool '%s' from model worker",
                tool_name,
                exc_info=True,
            )
            if worker is not None:
                await self._run_blocking(worker.close)
        finally:
            async with self._condition:
                slot = self._device_slots[device_index]
                if slot.worker_tool == tool_name:
                    slot.worker_tool = None
                if worker_failed:
                    slot.worker = None
                slot.busy = False
                slot.last_used = now
                state = self._tools[tool_name]
                state.last_used = now
                nomad_metrics.record_tool_disk_unload(tool_name)
                logger.debug(
                    "Unloaded subprocess tool '%s' after %.2fs fully offloaded",
                    tool_name,
                    idle_time,
                )
                self._condition.notify_all()

    async def _evict_idle_workers(self, *, now: float | None = None) -> None:
        threshold = self._venv_idle_threshold
        if threshold is None:
            return

        now = time.monotonic() if now is None else now
        candidates: list[int] = []
        async with self._condition:
            for index, slot in enumerate(self._device_slots):
                if (
                    slot.worker is not None
                    and not slot.busy
                    and now - slot.last_used >= threshold
                ):
                    slot.busy = True
                    candidates.append(index)

        for index in candidates:
            await self._terminate_slot_worker(index, now=now)

    async def _terminate_slot_worker(
        self,
        device_index: int,
        *,
        now: float | None = None,
    ) -> None:
        slot = self._device_slots[device_index]
        worker = slot.worker
        tool_name = slot.current_tool
        try:
            if worker is not None:
                await self._run_blocking(worker.close)
        finally:
            async with self._condition:
                slot = self._device_slots[device_index]
                if tool_name is not None and slot.current_tool == tool_name:
                    state = self._tools[tool_name]
                    state.release_slot(device_index)
                    state.last_used = time.monotonic() if now is None else now
                    nomad_metrics.record_tool_disk_unload(tool_name)
                    slot.current_tool = None
                    slot.tool = None
                slot.worker = None
                slot.worker_tool = None
                slot.busy = False
                slot.last_used = time.monotonic() if now is None else now
                self._condition.notify_all()

    def _next_batch_size(self, state: ToolState) -> int:
        batch_size = state.batch_size
        if self._max_pending_per_tool is not None:
            batch_size = min(batch_size, self._max_pending_per_tool)
        return batch_size

    def _next_available_requests(
        self,
        tool_name: str,
        batch_size: int,
    ) -> tuple[list[ToolRequest], bool]:
        state = self._tools[tool_name]
        requests: list[ToolRequest] = []
        has_more = False

        while state.queue and len(requests) < batch_size:
            request = state.queue.popleft()
            if request.future.cancelled():
                continue
            requests.append(request)

        if state.queue:
            has_more = True
            if not state.enqueued:
                state.enqueued = True
                self._pending_tools.append(tool_name)

        if not requests:
            state.enqueued = False
        return requests, has_more

    def _requeue_requests_front(
        self,
        tool_name: str,
        requests: list[ToolRequest],
    ) -> None:
        state = self._tools[tool_name]
        while requests:
            request = requests.pop()
            if request.future.cancelled():
                continue
            state.queue.appendleft(request)
        if state.queue and not state.enqueued:
            state.enqueued = True
            self._pending_tools.appendleft(tool_name)

    def _record_request_duration(
        self,
        tool_name: str,
        request: ToolRequest,
        status: str,
    ) -> None:
        if request.metrics_recorded:
            return
        request.metrics_recorded = True
        nomad_metrics.record_tool_request_duration(
            tool_name,
            time.monotonic() - request.enqueued_at,
            status=status,
        )

    async def _execute_request(
        self,
        tool_name: str,
        device_index: int,
        requests: list[ToolRequest],
        batch_size: int,
    ) -> None:
        if tool_name is None or device_index is None:
            return

        slot = self._device_slots[device_index]

        try:
            while True:
                if not requests:
                    break

                tool = await self._load_tool_on_slot(tool_name, device_index)
                batch_started = time.monotonic()
                for request in requests:
                    nomad_metrics.record_tool_queue_wait(
                        tool_name,
                        batch_started - request.enqueued_at,
                    )
                inflight_count = len(requests)
                self._inflight_by_tool[tool_name] += inflight_count
                run_requests = requests
                try:
                    results = await self._run_blocking(
                        self._run_batch,
                        tool,
                        self._tools[tool_name],
                        requests,
                        batch_size,
                    )
                except Exception as exc:
                    nomad_metrics.record_tool_batch(
                        tool_name,
                        slot.device,
                        batch_size=len(run_requests),
                        duration_seconds=time.monotonic() - batch_started,
                        status="error",
                    )
                    if self._is_out_of_memory(exc) and len(run_requests) > 1:
                        new_batch = max(1, len(run_requests) // 2)
                        state = self._tools[tool_name]
                        state.batch_size = new_batch
                        if state.tool is not None:
                            state.tool.batch_size = new_batch
                        if isinstance(tool, ModelProcess):
                            await self._run_blocking(
                                tool.request,
                                "set_batch_size",
                                {"batch_size": new_batch},
                            )
                        else:
                            tool.batch_size = new_batch
                        nomad_metrics.record_tool_batch_reduction(
                            tool_name,
                            slot.device,
                            old_batch_size=len(run_requests),
                            new_batch_size=new_batch,
                        )
                        logger.warning(
                            "Reduced batch size for '%s' to %s after OOM",
                            tool_name,
                            new_batch,
                        )
                        async with self._condition:
                            self._requeue_requests_front(tool_name, requests)
                            self._condition.notify_all()
                        requests = []
                        continue
                    raise
                finally:
                    self._inflight_by_tool[tool_name] -= inflight_count
                    assert self._inflight_by_tool[tool_name] >= 0

                nomad_metrics.record_tool_batch(
                    tool_name,
                    slot.device,
                    batch_size=len(run_requests),
                    duration_seconds=time.monotonic() - batch_started,
                    status="ok",
                )

                for request, output in zip(requests, results):
                    if not request.future.done():
                        request.future.set_result(output)
                        self._record_request_duration(tool_name, request, "ok")
                break
        except Exception as exc:
            for request in requests:
                if not request.future.done():
                    request.future.set_exception(exc)
                    self._record_request_duration(tool_name, request, "error")
            logger.exception(
                "Error while executing tool '%s' on %s",
                tool_name,
                slot.device,
                exc_info=exc,
            )
        finally:
            await self._mark_slot_free(device_index, tool_name=tool_name)

    async def _run_blocking(
        self,
        fn: Callable[..., T],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Run blocking work in a worker thread without breaking slot semantics."""
        task = asyncio.create_task(asyncio.to_thread(fn, *args, **kwargs))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # The underlying torch operation is not cancellable; wait for it to finish
            # before releasing the device slot in the caller's ``finally`` path.
            with contextlib.suppress(Exception):
                await asyncio.shield(task)
            raise

    async def _finish_non_cancellable(
        self,
        fn: Callable[..., T],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[T, bool]:
        """Run blocking work and report cancellation after the work completes."""
        task = asyncio.create_task(asyncio.to_thread(fn, *args, **kwargs))
        was_cancelled = False
        while True:
            try:
                return await asyncio.shield(task), was_cancelled
            except asyncio.CancelledError:
                was_cancelled = True
                if task.done():
                    return task.result(), was_cancelled

    def _run_batch(
        self,
        tool: TorchModuleTool | ModelProcess,
        state: ToolState,
        requests: list[ToolRequest],
        batch_size: int,
    ) -> list[BaseModel | dict[str, Any] | Any]:
        inputs = [request.input for request in requests]
        if isinstance(tool, ModelProcess):
            serialized_inputs = [
                item.model_dump(mode="json")
                if isinstance(item, BaseModel)
                else dict(item)
                for item in inputs
            ]
            outputs = tool.run_batch(serialized_inputs, batch_size=batch_size)
        else:
            outputs = list(
                tool.batch_as_completed(
                    inputs,
                    max_concurency=batch_size,
                )
            )
        if len(outputs) != len(requests):
            raise RuntimeError(
                f"Tool '{state.name}' returned {len(outputs)} outputs for "
                f"{len(requests)} requests"
            )
        return outputs

    def _build_fastmcp_tool(
        self,
        name: str,
        state: ToolState,
    ) -> FastMCPTool:
        return build_torch_module_fastmcp_tool(
            state,
            invoke=lambda args: self.call_tool(name, args),
        )

    def _is_out_of_memory(self, exc: Exception) -> bool:
        cuda_oom = getattr(torch.cuda, "OutOfMemoryError", ())
        if cuda_oom and isinstance(exc, cuda_oom):
            return True
        if isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower():
            return True
        return False

    async def _mark_slot_free(
        self,
        device_index: int,
        *,
        tool_name: str | None = None,
    ) -> None:
        async with self._condition:
            slot = self._device_slots[device_index]
            slot.busy = False
            slot.last_used = time.monotonic()
            if slot.current_tool is not None:
                state = self._tools[slot.current_tool]
                state.mark_slot_free(
                    device_index,
                    now=slot.last_used,
                    remains_loaded=True,
                )
            elif tool_name is not None:
                state = self._tools[tool_name]
                state.mark_slot_free(
                    device_index,
                    now=slot.last_used,
                    remains_loaded=False,
                )
            if self._server_is_idle_locked():
                self._mark_server_active_locked()
            self._condition.notify_all()

    async def _offload_slot(
        self,
        device_index: int,
        *,
        expected_tool_name: str | None = None,
        keep_slot_busy: bool = False,
        idle_time: float | None = None,
        record_idle_eviction: bool = False,
    ) -> bool:
        async with self._condition:
            slot = self._device_slots[device_index]
            if slot.busy and not keep_slot_busy:
                return False
            if slot.current_tool is None:
                return False
            if (
                expected_tool_name is not None
                and slot.current_tool != expected_tool_name
            ):
                return False

            tool_name = slot.current_tool
            tool = slot.tool
            state = self._tools[tool_name]
            worker = slot.worker if state.remote is not None else None
            device = slot.device
            slot.busy = True

        completed = False
        was_cancelled = False
        worker_failed = False
        try:
            if worker is not None:
                if worker.alive:
                    try:
                        _, was_cancelled = await self._finish_non_cancellable(
                            worker.request,
                            "offload",
                        )
                    except Exception:
                        logger.warning(
                            "Model worker failed while offloading '%s'; terminating it",
                            tool_name,
                            exc_info=True,
                        )
                        worker_failed = True
                        await self._finish_non_cancellable(worker.close)
                else:
                    worker_failed = True
                    await self._finish_non_cancellable(worker.close)
            elif tool is not None:
                _, was_cancelled = await self._finish_non_cancellable(
                    self._offload_tool,
                    tool,
                )
            completed = True
        finally:
            async with self._condition:
                slot = self._device_slots[device_index]
                if completed:
                    state = self._tools[tool_name]
                    update_time = time.monotonic()
                    if record_idle_eviction:
                        nomad_metrics.record_tool_idle_eviction(tool_name, device)
                    state.release_slot(device_index)
                    state.last_used = update_time
                    slot.current_tool = None
                    slot.tool = None
                    if worker_failed:
                        slot.worker = None
                        slot.worker_tool = None
                    slot.last_used = update_time
                    self._mark_gc_activity_locked(now=update_time)
                    if idle_time is not None:
                        logger.debug(
                            "Reduced allocations for tool '%s' after %.2fs idle on %s",
                            tool_name,
                            idle_time,
                            device,
                        )
                if not keep_slot_busy:
                    slot.busy = False
                self._condition.notify_all()

        if was_cancelled:
            raise asyncio.CancelledError
        return completed

    async def _load_tool_on_slot(
        self,
        tool_name: str,
        device_index: int,
    ) -> TorchModuleTool | ModelProcess:
        slot = self._device_slots[device_index]
        if slot.worker is not None and not slot.worker.alive:
            dead_worker = slot.worker
            dead_tool_name = slot.current_tool
            await self._finish_non_cancellable(dead_worker.close)
            slot.worker = None
            slot.worker_tool = None
            slot.current_tool = None
            slot.tool = None
            if dead_tool_name is not None:
                dead_state = self._tools[dead_tool_name]
                dead_state.loaded_slots.discard(device_index)
                if dead_tool_name != tool_name:
                    dead_state.release_slot(device_index)

        if slot.current_tool == tool_name:
            state = self._tools[tool_name]
            if state.remote is not None:
                if slot.worker is not None and slot.worker.alive:
                    return slot.worker
            else:
                assert slot.tool is not None
                return slot.tool

        if slot.current_tool is not None:
            await self._offload_slot(
                device_index,
                expected_tool_name=slot.current_tool,
                keep_slot_busy=True,
            )

        state = self._tools[tool_name]
        if state.remote is not None:
            remote = state.remote
            worker = slot.worker
            if worker is not None and (
                not worker.alive
                or worker.environment.checksum != remote.environment.checksum
            ):
                await self._finish_non_cancellable(worker.close)
                slot.worker = None
                slot.worker_tool = None
                worker = None
            if worker is None:
                worker, was_cancelled = await self._finish_non_cancellable(
                    ModelProcess,
                    remote.environment,
                    timeout_seconds=self.rpc_timeout_seconds,
                )
                slot.worker = worker
                if was_cancelled:
                    worker.close()
                    slot.worker = None
                    raise asyncio.CancelledError

            logger.info(
                "Loading subprocess tool '%s' onto %s in process %s",
                tool_name,
                slot.device,
                worker.pid,
            )
            start = time.monotonic()
            status = "ok"
            slot.worker_tool = None
            try:
                _, was_cancelled = await self._finish_non_cancellable(
                    worker.load,
                    remote,
                    str(slot.device),
                )
                if state.batch_size != remote.batch_size:
                    await self._finish_non_cancellable(
                        worker.request,
                        "set_batch_size",
                        {"batch_size": state.batch_size},
                    )
            except Exception:
                status = "error"
                raise
            finally:
                nomad_metrics.record_tool_disk_load(
                    tool_name,
                    time.monotonic() - start,
                    load_kind="resident",
                    status=status,
                )

            slot.current_tool = tool_name
            slot.tool = None
            slot.worker_tool = tool_name
            slot.last_used = time.monotonic()
            state.mark_slot_loaded(
                device_index,
                now=slot.last_used,
                is_resident=False,
            )
            if was_cancelled:
                raise asyncio.CancelledError
            return worker

        if slot.worker is not None:
            await self._finish_non_cancellable(slot.worker.close)
            slot.worker = None
            slot.worker_tool = None

        is_resident_tool = False
        if state.tool is not None and state.resident_slot is None:
            state.resident_slot = device_index
            tool = state.tool
            is_resident_tool = True
        else:
            is_resident_load = state.tool is None and state.resident_slot is None
            if is_resident_load:
                state.resident_slot = device_index
            try:
                tool, was_cancelled = await self._finish_non_cancellable(
                    state.load_new_tool,
                    is_resident_load=is_resident_load,
                )
            except Exception:
                if is_resident_load:
                    state.resident_slot = None
                raise
            if is_resident_load:
                state.tool = tool
                if was_cancelled:
                    state.resident_slot = None
                else:
                    is_resident_tool = True
            if was_cancelled:
                raise asyncio.CancelledError
        tool.batch_size = state.batch_size
        logger.info("Loading tool '%s' onto %s", tool_name, slot.device)
        start = time.monotonic()
        _, was_cancelled = await self._finish_non_cancellable(
            self._load_tool,
            tool,
            slot.device,
        )
        nomad_metrics.record_tool_device_load(
            tool_name,
            slot.device,
            time.monotonic() - start,
        )
        slot.current_tool = tool_name
        slot.tool = tool
        slot.last_used = time.monotonic()
        state.mark_slot_loaded(
            device_index,
            now=slot.last_used,
            is_resident=is_resident_tool,
        )
        if was_cancelled:
            raise asyncio.CancelledError
        return tool

    def _load_tool(self, tool: TorchModuleTool, device: torch.device) -> None:
        try:
            tool.fm = tool.fm.to(device)
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to move tool '%s' to %s", tool.name, device)
            raise
        tool.device = device

    def _offload_tool(self, tool: TorchModuleTool) -> None:
        tool.fm = tool.fm.to(torch.device("cpu"))
        tool.device = torch.device("cpu")
        if tool.name is not None:
            nomad_metrics.record_tool_offload(tool.name)
        logger.info("Offloaded tool '%s' to CPU", tool.name)

    def queue_length_observations(self):
        for tool_name, state in self._tools.items():
            active_pending = sum(
                1 for pending in state.queue if not pending.future.cancelled()
            )
            yield Observation(active_pending, {"tool": tool_name})

    def inflight_observations(self):
        for tool_name, count in self._inflight_by_tool.items():
            yield Observation(count, {"tool": tool_name})

    def configured_batch_size_observations(self):
        for tool_name, state in self._tools.items():
            yield Observation(state.batch_size, {"tool": tool_name})

    def pending_tools_observations(self):
        yield Observation(len(self._pending_tools), {})

    def device_busy_observations(self):
        for slot in self._device_slots:
            yield Observation(
                1 if slot.busy else 0,
                nomad_metrics.tool_device_attributes(
                    slot.current_tool or "",
                    slot.device,
                ),
            )

    def loaded_tool_observations(self):
        for slot in self._device_slots:
            if slot.current_tool is None:
                continue
            yield Observation(
                1,
                nomad_metrics.tool_device_attributes(slot.current_tool, slot.device),
            )

    def resident_tool_observations(self):
        for tool_name, state in self._tools.items():
            if state.tool is None:
                continue
            yield Observation(1, {"tool": tool_name})

    def device_memory_allocated_observations(self):
        yield from self._device_memory_observations("allocated")

    def device_memory_reserved_observations(self):
        yield from self._device_memory_observations("reserved")

    def device_utilization_observations(self):
        for info in self._accelerators:
            value = self._device_utilization_value(info.device)
            if value is None:
                continue
            yield Observation(value, self._accelerator_attributes(info))

    def _accelerator_attributes(self, info: AcceleratorInfo) -> dict[str, Any]:
        return {
            "device": str(info.device),
            "device.type": info.kind,
            "device.index": info.index if info.index is not None else -1,
            "device.name": info.name or "",
        }

    def _device_memory_observations(self, kind: str):
        for info in self._accelerators:
            value = self._device_memory_value(info.device, kind)
            if value is None:
                continue
            yield Observation(value, self._accelerator_attributes(info))

    def _device_memory_value(self, device: torch.device, kind: str) -> int | None:
        host_value = self._host_device_memory_value(device, kind)
        total = host_value or 0
        measured = host_value is not None
        seen_workers: set[int] = set()
        for slot in self._device_slots:
            worker = slot.worker
            if slot.device != device or worker is None or not worker.alive:
                continue
            worker_identity = id(worker)
            if worker_identity in seen_workers:
                continue
            seen_workers.add(worker_identity)
            values = self._worker_device_memory_values(worker, device)
            if values is None:
                continue
            worker_value = values.get(kind)
            if isinstance(worker_value, int) and not isinstance(worker_value, bool):
                total += worker_value
                measured = True
        return total if measured else None

    def _worker_device_memory_values(
        self,
        worker: ModelProcess,
        device: torch.device,
    ) -> dict[str, int] | None:
        device_key = str(device)
        with self._worker_memory_cache_lock:
            worker_locks = self._worker_memory_locks.setdefault(worker, {})
            snapshot_lock = worker_locks.setdefault(device_key, threading.Lock())

        with snapshot_lock:
            now = time.monotonic()
            with self._worker_memory_cache_lock:
                cached = self._worker_memory_snapshots.get(worker, {}).get(device_key)
            if cached is not None and now - cached[0] < _DEVICE_MEMORY_SNAPSHOT_SECONDS:
                return cached[1]

            try:
                raw_values = worker.request("device_memory", {"device": str(device)})
            except Exception:
                logger.debug(
                    "Failed to read device memory from model worker %s",
                    worker.pid,
                    exc_info=True,
                )
                values = None
            else:
                values = (
                    {
                        key: value
                        for key, value in raw_values.items()
                        if key in {"allocated", "reserved"}
                        and isinstance(value, int)
                        and not isinstance(value, bool)
                    }
                    if isinstance(raw_values, dict)
                    else None
                )
            with self._worker_memory_cache_lock:
                snapshots = self._worker_memory_snapshots.setdefault(worker, {})
                snapshots[device_key] = (time.monotonic(), values)
            return values

    @staticmethod
    def _host_device_memory_value(
        device: torch.device,
        kind: str,
    ) -> int | None:
        if device.type == "cuda":
            index = (
                device.index
                if device.index is not None
                else torch.cuda.current_device()
            )
            if kind == "allocated":
                return int(torch.cuda.memory_allocated(index))
            if kind == "reserved":
                return int(torch.cuda.memory_reserved(index))
        if device.type == "mps":
            if kind == "allocated" and hasattr(torch.mps, "current_allocated_memory"):
                return int(torch.mps.current_allocated_memory())
            if kind == "reserved" and hasattr(torch.mps, "driver_allocated_memory"):
                return int(torch.mps.driver_allocated_memory())
        return None

    def _device_utilization_value(self, device: torch.device) -> float | None:
        if device.type != "cuda" or not hasattr(torch.cuda, "utilization"):
            return None

        index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        try:
            return float(torch.cuda.utilization(index)) / 100.0
        except Exception:  # pragma: no cover - backend/NVML availability varies
            return None
