import asyncio
import contextlib
import threading
import time
from collections.abc import Sequence
from typing import ClassVar

import pytest
import torch
from fastmcp import FastMCP
from pydantic import BaseModel

from nomad import metrics as nomad_metrics
from nomad.config import ToolManagerConfig
from nomad.fm_base_tool import TorchModuleTool, default_device
from nomad.torch_tool_manager import ToolRequest, TorchModelToolManager


class DummyModule(torch.nn.Module):
    all_forward_start_times: ClassVar[list[float]] = []

    def __init__(
        self,
        increment: int = 1,
        delay: float = 0.0,
        to_delay: float = 0.0,
        label: str | None = None,
        oom_threshold: int | None = None,
        forward_started: threading.Event | None = None,
        forward_release: threading.Event | None = None,
    ) -> None:
        super().__init__()
        self.increment = increment
        self.delay = delay
        self.to_delay = to_delay
        self.label = label or "module"
        self.last_device = torch.device("cpu")
        self.forward_batch_sizes: list[int] = []
        self.execution_devices: list[torch.device] = []
        self.forward_start_times: list[float] = []
        self.oom_threshold = oom_threshold
        self.forward_started = forward_started
        self.forward_release = forward_release

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        self.forward_batch_sizes.append(batch_size)
        if self.oom_threshold is not None and batch_size > self.oom_threshold:
            raise RuntimeError("CUDA out of memory")
        self.execution_devices.append(self.last_device)
        self.forward_start_times.append(time.perf_counter())
        self.all_forward_start_times.append(self.forward_start_times[-1])
        if self.forward_started is not None:
            self.forward_started.set()
        if self.forward_release is not None:
            assert self.forward_release.wait(timeout=1.0)
        if self.delay:
            time.sleep(self.delay)
        return x + self.increment

    def to(self, device, **kwargs):
        if self.to_delay:
            time.sleep(self.to_delay)
        self.last_device = torch.device(device)
        return self


class DummyInput(BaseModel):
    value: int


class DummyOutput(BaseModel):
    value: int


class DummyTool(TorchModuleTool[DummyInput, DummyOutput, torch.Tensor, torch.Tensor]):
    args_schema: type[DummyInput] = DummyInput
    output_schema: type[DummyOutput] = DummyOutput
    description: str = "Add one to the provided value."
    clone_specs: ClassVar[dict[str, dict[str, int | float | str | None]]] = {}
    clone_sources: ClassVar[dict[str, str]] = {}
    load_counts: ClassVar[dict[str, int]] = {}

    @classmethod
    def from_pretrained(cls, name_or_path: str, **kwargs):
        assert kwargs == {}
        source = str(name_or_path)
        cls.load_counts[source] = cls.load_counts.get(source, 0) + 1
        spec = cls.clone_specs[source]
        module = DummyModule(
            increment=int(spec["increment"]),
            delay=float(spec["delay"]),
            to_delay=float(spec["to_delay"]),
            label=str(spec["label"]),
            oom_threshold=spec["oom_threshold"],
        )
        return cls(
            description=f"{spec['label']} tool",
            fm=module,
            batch_size=int(spec["batch_size"]),
            device=torch.device("cpu"),
        )

    def preprocess(self, input: Sequence[DummyInput]) -> torch.Tensor:
        values = [item.value for item in input]
        return torch.tensor(values, dtype=torch.float32).unsqueeze(-1)

    def postprocess(self, model_output: torch.Tensor):
        for value in model_output.squeeze(-1):
            yield DummyOutput(value=int(value.item()))


@pytest.fixture()
def dummy_tool_factory():
    def _factory(
        *,
        increment: int = 1,
        delay: float = 0.0,
        to_delay: float = 0.0,
        label: str = "tool",
        batch_size: int = 4,
        oom_threshold: int | None = None,
        forward_started: threading.Event | None = None,
        forward_release: threading.Event | None = None,
    ) -> DummyTool:
        module = DummyModule(
            increment=increment,
            delay=delay,
            to_delay=to_delay,
            label=label,
            oom_threshold=oom_threshold,
            forward_started=forward_started,
            forward_release=forward_release,
        )
        source = f"dummy://{label}-{len(DummyTool.clone_specs)}"
        DummyTool.clone_specs[source] = {
            "increment": increment,
            "delay": delay,
            "to_delay": to_delay,
            "label": label,
            "batch_size": batch_size,
            "oom_threshold": oom_threshold,
        }
        DummyTool.clone_sources[label] = source
        return DummyTool(
            description=f"{label} tool",
            fm=module,
            batch_size=batch_size,
            device=torch.device("cpu"),
        )

    return _factory


def make_accelerator_device(index: int = 0) -> torch.device:
    base = default_device()
    dtype = base.type
    if dtype in {"cuda", "xpu"}:
        return torch.device(f"{dtype}:{index}")
    if dtype == "mps":
        return torch.device("mps")
    return torch.device("cpu")


def make_accelerator_devices(count: int) -> list[torch.device]:
    return [torch.device(f"cuda:{idx}") for idx in range(count)]


@pytest.fixture()
def dummy_tool(dummy_tool_factory) -> DummyTool:
    return dummy_tool_factory(label="dummy")


def test_default_device_detection():
    manager = TorchModelToolManager()
    devices = manager.devices
    assert devices, "At least one device should be detected"
    for device in devices:
        assert isinstance(device, torch.device)


def test_cpu_device_provider_is_deduplicated():
    manager = TorchModelToolManager(
        device_provider=lambda: [torch.device("cpu"), torch.device("cpu")]
    )
    assert manager.devices == [torch.device("cpu")]
    assert manager._max_devices_per_tool == 1


def test_max_devices_per_tool_defaults_to_device_count():
    devices = make_accelerator_devices(3)
    manager = TorchModelToolManager(device_provider=lambda: devices)
    assert manager.devices == devices
    assert manager._max_devices_per_tool == 3


@pytest.mark.asyncio()
async def test_finish_non_cancellable_returns_result_without_cancellation():
    manager = TorchModelToolManager(device_provider=lambda: [torch.device("cpu")])

    result, was_cancelled = await manager._finish_non_cancellable(lambda: "done")

    assert result == "done"
    assert was_cancelled is False


@pytest.mark.asyncio()
async def test_finish_non_cancellable_reports_cancellation_after_completion():
    manager = TorchModelToolManager(device_provider=lambda: [torch.device("cpu")])
    started = threading.Event()
    release = threading.Event()

    def wait_for_release():
        started.set()
        assert release.wait(timeout=1.0)
        return "done"

    task = asyncio.create_task(manager._finish_non_cancellable(wait_for_release))
    assert await asyncio.to_thread(started.wait, 1.0)

    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()

    release.set()
    result, was_cancelled = await asyncio.wait_for(task, timeout=1.0)

    assert result == "done"
    assert was_cancelled is True


@pytest.mark.asyncio()
async def test_finish_non_cancellable_propagates_worker_exception_after_cancellation():
    manager = TorchModelToolManager(device_provider=lambda: [torch.device("cpu")])
    started = threading.Event()
    release = threading.Event()

    def wait_then_fail():
        started.set()
        assert release.wait(timeout=1.0)
        raise RuntimeError("worker failed")

    task = asyncio.create_task(manager._finish_non_cancellable(wait_then_fail))
    assert await asyncio.to_thread(started.wait, 1.0)

    task.cancel()
    await asyncio.sleep(0)
    release.set()

    with pytest.raises(RuntimeError, match="worker failed"):
        await asyncio.wait_for(task, timeout=1.0)


@pytest.mark.asyncio()
@pytest.mark.gpu
async def test_device_provider_injection(dummy_tool: DummyTool):
    fake_devices = make_accelerator_devices(2)
    manager = TorchModelToolManager(device_provider=lambda: fake_devices)
    assert manager.devices == fake_devices

    manager.register_tool("dummy", dummy_tool, source="dummy://dummy")

    result = await manager.call_tool("dummy", {"value": 1})
    assert isinstance(result, DummyOutput)
    assert result.value == 2

    fm = dummy_tool.fm
    assert isinstance(fm, DummyModule)
    assert fm.last_device == fake_devices[0]
    assert fm.execution_devices == [fake_devices[0]]
    await manager.aclose()


@pytest.mark.asyncio()
@pytest.mark.gpu
async def test_round_robin_assignment_and_eviction(dummy_tool_factory):
    fake_devices = make_accelerator_devices(2)
    manager = TorchModelToolManager(device_provider=lambda: fake_devices)

    tool_a = dummy_tool_factory(label="A", increment=1)
    tool_b = dummy_tool_factory(label="B", increment=2)
    tool_c = dummy_tool_factory(label="C", increment=3)

    manager.register_tool("A", tool_a, source=DummyTool.clone_sources["A"])
    manager.register_tool("B", tool_b, source=DummyTool.clone_sources["B"])
    manager.register_tool("C", tool_c, source=DummyTool.clone_sources["C"])

    out_a = await manager.call_tool("A", {"value": 1})
    out_b = await manager.call_tool("B", {"value": 2})
    out_c = await manager.call_tool("C", {"value": 3})

    assert [out_a.value, out_b.value, out_c.value] == [2, 4, 6]

    assert tool_a.fm.execution_devices == [fake_devices[0]]
    assert tool_b.fm.execution_devices == [fake_devices[1]]
    assert tool_c.fm.execution_devices == [fake_devices[0]]

    assert tool_a.fm.last_device == torch.device("cpu")
    assert tool_b.fm.last_device == fake_devices[1]
    assert tool_c.fm.last_device == fake_devices[0]
    await manager.aclose()


@pytest.mark.asyncio()
@pytest.mark.gpu
async def test_queue_single_device_serializes_calls(dummy_tool_factory):
    device = make_accelerator_device(0)
    manager = TorchModelToolManager(device_provider=lambda: [device])

    slow_tool = dummy_tool_factory(label="slow", delay=0.05)
    fast_tool = dummy_tool_factory(label="fast")

    manager.register_tool("slow", slow_tool, source=DummyTool.clone_sources["slow"])
    manager.register_tool("fast", fast_tool, source=DummyTool.clone_sources["fast"])

    slow_task = asyncio.create_task(manager.call_tool("slow", {"value": 0}))
    await asyncio.sleep(0)
    fast_task = asyncio.create_task(manager.call_tool("fast", {"value": 1}))

    slow_result, fast_result = await asyncio.gather(slow_task, fast_task)
    assert slow_result.value == 1
    assert fast_result.value == 2

    assert slow_tool.fm.execution_devices == [device]
    assert fast_tool.fm.execution_devices == [device]

    slow_start = slow_tool.fm.forward_start_times[0]
    fast_start = fast_tool.fm.forward_start_times[0]
    assert fast_start >= slow_start
    assert fast_start - slow_start >= slow_tool.fm.delay * 0.8

    assert slow_tool.fm.last_device == torch.device("cpu")
    assert fast_tool.fm.last_device == device
    await manager.aclose()


@pytest.mark.asyncio()
@pytest.mark.gpu
async def test_batching_respects_tool_batch_size(dummy_tool_factory):
    device = make_accelerator_device(0)
    manager = TorchModelToolManager(device_provider=lambda: [device])
    tool = dummy_tool_factory(label="batch", batch_size=4)
    manager.register_tool("batch", tool, source=DummyTool.clone_sources["batch"])

    tasks = [
        asyncio.create_task(manager.call_tool("batch", {"value": idx}))
        for idx in range(10)
    ]

    results = await asyncio.gather(*tasks)
    assert [result.value for result in results] == [idx + 1 for idx in range(10)]

    assert tool.fm.forward_batch_sizes == [4, 4, 2]
    assert len(tool.fm.execution_devices) == 3
    await manager.aclose()


@pytest.mark.asyncio()
@pytest.mark.gpu
async def test_clone_failure_releases_device_slot(monkeypatch, dummy_tool_factory):
    fake_devices = make_accelerator_devices(2)
    manager = TorchModelToolManager(device_provider=lambda: fake_devices)
    tool = dummy_tool_factory(label="clone-fails", delay=0.05, batch_size=1)
    manager.register_tool(
        "clone-fails",
        tool,
        source=DummyTool.clone_sources["clone-fails"],
    )

    def fail_from_pretrained(cls, name_or_path: str, **kwargs):
        raise RuntimeError(f"clone failed for {name_or_path}")

    monkeypatch.setattr(
        DummyTool,
        "from_pretrained",
        classmethod(fail_from_pretrained),
    )

    tasks = [
        asyncio.create_task(manager.call_tool("clone-fails", {"value": idx}))
        for idx in range(2)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    assert sum(isinstance(result, DummyOutput) for result in results) == 1
    errors = [result for result in results if isinstance(result, RuntimeError)]
    assert len(errors) == 1
    assert "clone failed" in str(errors[0])
    assert [slot.busy for slot in manager._device_slots] == [False, False]
    assert len(manager._tools["clone-fails"].loaded_slots) == 1

    result = await manager.call_tool("clone-fails", {"value": 10})
    assert result.value == 11
    await manager.aclose()


@pytest.mark.asyncio()
@pytest.mark.gpu
async def test_scheduler_reserves_capacity_for_other_queued_tools(
    dummy_tool_factory,
):
    fake_devices = make_accelerator_devices(2)
    manager = TorchModelToolManager(device_provider=lambda: fake_devices)

    hot_tool = dummy_tool_factory(label="hot", delay=0.05, batch_size=1)
    other_tool = dummy_tool_factory(label="other", delay=0.05, batch_size=1)
    manager.register_tool("hot", hot_tool, source=DummyTool.clone_sources["hot"])
    manager.register_tool("other", other_tool, source=DummyTool.clone_sources["other"])

    loop = asyncio.get_running_loop()
    await manager._enqueue_request(
        "hot",
        ToolRequest(input=DummyInput(value=1), future=loop.create_future()),
    )
    await manager._enqueue_request(
        "other",
        ToolRequest(input=DummyInput(value=3), future=loop.create_future()),
    )
    await manager._enqueue_request(
        "hot",
        ToolRequest(input=DummyInput(value=2), future=loop.create_future()),
    )

    first = manager._next_assignment_locked()
    second = manager._next_assignment_locked()

    assert first is not None
    assert second is not None
    assert {first[0], second[0]} == {"hot", "other"}

    assert [slot.busy for slot in manager._device_slots] == [True, True]
    assert manager._tools["hot"].loaded_slots == set()
    assert manager._tools["other"].loaded_slots == set()
    await manager.aclose()


@pytest.mark.asyncio()
@pytest.mark.gpu
async def test_tool_execution_does_not_block_event_loop(dummy_tool_factory):
    device = make_accelerator_device(0)
    manager = TorchModelToolManager(device_provider=lambda: [device])
    forward_started = threading.Event()
    forward_release = threading.Event()
    tool = dummy_tool_factory(
        label="blocking",
        forward_started=forward_started,
        forward_release=forward_release,
    )
    manager.register_tool("blocking", tool, source=DummyTool.clone_sources["blocking"])

    event_loop_progressed = asyncio.Event()

    async def mark_event_loop_progress() -> None:
        await asyncio.sleep(0)
        event_loop_progressed.set()

    call_task = asyncio.create_task(manager.call_tool("blocking", {"value": 1}))
    try:
        assert await asyncio.to_thread(forward_started.wait, 1.0)
        progress_task = asyncio.create_task(mark_event_loop_progress())
        await asyncio.wait_for(event_loop_progressed.wait(), timeout=1.0)
        await progress_task

        assert not call_task.done()
        forward_release.set()
        result = await asyncio.wait_for(call_task, timeout=2.0)
        assert result.value == 2
    finally:
        forward_release.set()
        if not call_task.done():
            call_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await call_task
        await manager.aclose()


@pytest.mark.asyncio()
async def test_manager_close_releases_inflight_and_queued_callers(dummy_tool_factory):
    manager = TorchModelToolManager(device_provider=lambda: [torch.device("cpu")])
    forward_started = threading.Event()
    forward_release = threading.Event()
    tool = dummy_tool_factory(
        label="close-pending",
        forward_started=forward_started,
        forward_release=forward_release,
    )
    manager.register_tool(
        "close-pending",
        tool,
        source=DummyTool.clone_sources["close-pending"],
    )

    inflight = asyncio.create_task(manager.call_tool("close-pending", {"value": 1}))
    assert await asyncio.to_thread(forward_started.wait, 1.0)
    queued = asyncio.create_task(manager.call_tool("close-pending", {"value": 2}))
    await asyncio.sleep(0)

    close = asyncio.create_task(manager.aclose())
    for call in (inflight, queued):
        with pytest.raises(RuntimeError, match="TorchModelToolManager is closed"):
            await asyncio.wait_for(call, timeout=1.0)

    forward_release.set()
    await asyncio.wait_for(close, timeout=1.0)


@pytest.mark.asyncio()
async def test_tool_device_load_does_not_block_event_loop(dummy_tool_factory):
    device = make_accelerator_device(0)
    manager = TorchModelToolManager(device_provider=lambda: [device])
    tool = dummy_tool_factory(label="device-load-blocking", to_delay=0.25)
    manager.register_tool(
        "device-load-blocking",
        tool,
        source=DummyTool.clone_sources["device-load-blocking"],
    )

    heartbeat_gaps: list[float] = []
    stop_heartbeat = asyncio.Event()

    async def heartbeat() -> None:
        last_tick = time.perf_counter()
        while not stop_heartbeat.is_set():
            await asyncio.sleep(0.01)
            now = time.perf_counter()
            heartbeat_gaps.append(now - last_tick)
            last_tick = now

    heartbeat_task = asyncio.create_task(heartbeat())
    try:
        await asyncio.sleep(0.03)
        result = await asyncio.wait_for(
            manager.call_tool("device-load-blocking", {"value": 1}),
            timeout=2.0,
        )
        assert result.value == 2
    finally:
        stop_heartbeat.set()
        await asyncio.wait_for(heartbeat_task, timeout=1.0)
        await manager.aclose()

    assert heartbeat_gaps, "Heartbeat should record scheduler intervals"
    assert max(heartbeat_gaps) < 0.18


@pytest.mark.asyncio()
async def test_idle_offloading(monkeypatch, dummy_tool_factory):
    device = make_accelerator_device(0)
    manager = TorchModelToolManager(
        ToolManagerConfig(idle_seconds=0.2),
        device_provider=lambda: [device],
    )
    tool = dummy_tool_factory(label="idle")
    manager.register_tool("idle", tool, source=DummyTool.clone_sources["idle"])

    clock = {"value": 1_000.0}

    def fake_monotonic():
        return clock["value"]

    monkeypatch.setattr("nomad.torch_tool_manager.time.monotonic", fake_monotonic)

    await manager.call_tool("idle", {"value": 10})
    assert tool.fm.last_device == device

    # Advance time but not enough to trigger eviction
    clock["value"] += 0.05
    await manager._evict_idle_tools(now=clock["value"])
    assert tool.fm.last_device == device

    # Advance past idle threshold and evict
    clock["value"] += 0.3
    idle_evictions: list[tuple[str, torch.device]] = []
    monkeypatch.setattr(
        nomad_metrics,
        "record_tool_idle_eviction",
        lambda tool_name, device: idle_evictions.append((tool_name, device)),
    )

    await manager._evict_idle_tools(now=clock["value"])
    assert tool.fm.last_device == torch.device("cpu")
    assert idle_evictions == [("idle", device)]
    await manager.aclose()


@pytest.mark.asyncio()
async def test_idle_offloading_reduces_one_tool_allocation_at_a_time(
    monkeypatch,
    dummy_tool_factory,
):
    fake_devices = make_accelerator_devices(2)
    manager = TorchModelToolManager(
        ToolManagerConfig(idle_seconds=0.2, max_devices_per_tool=2),
        device_provider=lambda: fake_devices,
    )
    tool = dummy_tool_factory(label="idle-replica", batch_size=1)
    manager.register_tool(
        "idle-replica",
        tool,
        source=DummyTool.clone_sources["idle-replica"],
    )

    clock = {"value": 1_000.0}

    def fake_monotonic():
        return clock["value"]

    monkeypatch.setattr("nomad.torch_tool_manager.time.monotonic", fake_monotonic)

    tasks = [
        asyncio.create_task(manager.call_tool("idle-replica", {"value": idx}))
        for idx in range(2)
    ]
    await asyncio.gather(*tasks)

    state = manager._tools["idle-replica"]
    assert len(state.loaded_slots) == 2

    clock["value"] += 0.3
    await manager._evict_idle_tools(now=clock["value"])
    assert len(state.loaded_slots) == 1

    clock["value"] += 0.05
    await manager._evict_idle_tools(now=clock["value"])
    assert len(state.loaded_slots) == 1

    clock["value"] += 0.3
    await manager._evict_idle_tools(now=clock["value"])
    assert len(state.loaded_slots) == 0
    await manager.aclose()


@pytest.mark.asyncio()
async def test_gc_idle_keeps_clearing_accelerator_cache_while_idle(
    monkeypatch,
    dummy_tool_factory,
):
    clock = {"value": 1_000.0}
    cache_clears: list[float] = []

    def fake_monotonic():
        return clock["value"]

    def fake_empty_accelerator_cache():
        cache_clears.append(clock["value"])

    cache_clear_metrics: list[str] = []

    monkeypatch.setattr("nomad.torch_tool_manager.time.monotonic", fake_monotonic)
    monkeypatch.setattr(
        "nomad.torch_tool_manager.empty_accelerator_cache",
        fake_empty_accelerator_cache,
    )
    monkeypatch.setattr(
        nomad_metrics,
        "record_device_cache_clear",
        lambda _duration, *, status: cache_clear_metrics.append(status),
    )

    manager = TorchModelToolManager(
        ToolManagerConfig(gc_idle_seconds=None, idle_seconds=None),
        device_provider=lambda: [torch.device("cpu")],
    )
    tool = dummy_tool_factory(label="gc")
    manager.register_tool("gc", tool, source=DummyTool.clone_sources["gc"])

    await manager.call_tool("gc", {"value": 1})
    manager._gc_idle_threshold = 0.2

    clock["value"] += 0.1
    await manager._clear_cache_if_server_idle(now=clock["value"])
    assert cache_clears == []

    clock["value"] += 0.2
    await manager._clear_cache_if_server_idle(now=clock["value"])
    assert cache_clears == [clock["value"]]
    assert cache_clear_metrics == ["ok"]

    clock["value"] += 1.0
    await manager._clear_cache_if_server_idle(now=clock["value"])
    assert len(cache_clears) == 2

    await manager.call_tool("gc", {"value": 2})
    clock["value"] += 0.3
    await manager._clear_cache_if_server_idle(now=clock["value"])
    assert len(cache_clears) == 3
    await manager.aclose()


@pytest.mark.asyncio()
async def test_gc_janitor_clears_cache_repeatedly_while_idle(
    monkeypatch,
    dummy_tool_factory,
):
    cache_clears: list[float] = []

    def fake_empty_accelerator_cache():
        cache_clears.append(time.monotonic())

    monkeypatch.setattr(
        "nomad.torch_tool_manager.empty_accelerator_cache",
        fake_empty_accelerator_cache,
    )

    manager = TorchModelToolManager(
        ToolManagerConfig(
            gc_idle_seconds=0.05,
            idle_seconds=None,
            disk_idle_seconds=None,
        ),
        device_provider=lambda: [torch.device("cpu")],
    )
    tool = dummy_tool_factory(label="gc-loop")
    manager.register_tool("gc-loop", tool, source=DummyTool.clone_sources["gc-loop"])

    await manager.call_tool("gc-loop", {"value": 1})

    try:
        deadline = time.monotonic() + 1.0
        while len(cache_clears) < 2 and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        assert len(cache_clears) >= 2
    finally:
        await manager.aclose()


@pytest.mark.asyncio()
async def test_tool_eviction_resets_gc_idle_timer(monkeypatch, dummy_tool_factory):
    clock = {"value": 1_000.0}
    cache_clears: list[float] = []

    def fake_monotonic():
        return clock["value"]

    def fake_empty_accelerator_cache():
        cache_clears.append(clock["value"])

    monkeypatch.setattr("nomad.torch_tool_manager.time.monotonic", fake_monotonic)
    monkeypatch.setattr(
        "nomad.torch_tool_manager.empty_accelerator_cache",
        fake_empty_accelerator_cache,
    )

    manager = TorchModelToolManager(
        ToolManagerConfig(gc_idle_seconds=None, idle_seconds=0.2),
        device_provider=lambda: [torch.device("cpu")],
    )
    tool = dummy_tool_factory(label="gc-after-evict")
    manager.register_tool(
        "gc-after-evict", tool, source=DummyTool.clone_sources["gc-after-evict"]
    )

    await manager.call_tool("gc-after-evict", {"value": 1})
    manager._gc_idle_threshold = 0.2

    clock["value"] += 0.25
    await manager._evict_idle_tools(now=clock["value"])
    await manager._clear_cache_if_server_idle(now=clock["value"])
    assert cache_clears == []

    clock["value"] += 0.25
    await manager._clear_cache_if_server_idle(now=clock["value"])
    assert cache_clears == [clock["value"]]
    await manager.aclose()


@pytest.mark.asyncio()
async def test_disk_idle_unloads_cpu_tool_and_reloads_from_source(
    monkeypatch,
    dummy_tool_factory,
):
    clock = {"value": 1_000.0}
    disk_loads: list[tuple[str, float, str, str]] = []
    disk_unloads: list[str] = []

    def fake_monotonic():
        return clock["value"]

    monkeypatch.setattr("nomad.torch_tool_manager.time.monotonic", fake_monotonic)
    monkeypatch.setattr(
        nomad_metrics,
        "record_tool_disk_load",
        lambda tool_name, duration, *, load_kind, status: disk_loads.append(
            (tool_name, duration, load_kind, status)
        ),
    )
    monkeypatch.setattr(
        nomad_metrics,
        "record_tool_disk_unload",
        lambda tool_name: disk_unloads.append(tool_name),
    )

    device = make_accelerator_device(0)
    manager = TorchModelToolManager(
        ToolManagerConfig(idle_seconds=0.2, disk_idle_seconds=0.5),
        device_provider=lambda: [device],
    )
    tool = dummy_tool_factory(label="disk-idle")
    source = DummyTool.clone_sources["disk-idle"]
    manager.register_tool("disk-idle", tool, source=source)

    result = await manager.call_tool("disk-idle", {"value": 1})
    assert result.value == 2

    state = manager._tools["disk-idle"]
    assert state.tool is tool

    clock["value"] += 0.25
    await manager._evict_idle_tools(now=clock["value"])
    assert state.loaded_slots == set()
    assert state.tool is tool

    clock["value"] += 0.45
    await manager._evict_disk_idle_tools(now=clock["value"])
    assert state.tool is tool

    server = FastMCP()
    fast_tools = manager.add_to_fastmcp(server)
    assert "disk-idle" in fast_tools

    clock["value"] += 0.1
    await manager._evict_disk_idle_tools(now=clock["value"])
    assert state.tool is None
    assert disk_unloads == ["disk-idle"]

    load_count = DummyTool.load_counts.get(source, 0)
    result = await manager.call_tool("disk-idle", {"value": 2})
    assert result.value == 3
    assert state.tool is not None
    assert state.tool is not tool
    assert DummyTool.load_counts[source] == load_count + 1
    assert disk_loads[-1] == ("disk-idle", 0.0, "resident", "ok")
    await manager.aclose()


@pytest.mark.asyncio()
async def test_disk_idle_timer_starts_when_tool_is_displaced(
    monkeypatch,
    dummy_tool_factory,
):
    clock = {"value": 1_000.0}

    def fake_monotonic():
        return clock["value"]

    monkeypatch.setattr("nomad.torch_tool_manager.time.monotonic", fake_monotonic)

    device = make_accelerator_device(0)
    manager = TorchModelToolManager(
        ToolManagerConfig(idle_seconds=None, disk_idle_seconds=0.5),
        device_provider=lambda: [device],
    )
    first_tool = dummy_tool_factory(label="disk-displaced-first")
    second_tool = dummy_tool_factory(label="disk-displaced-second")
    manager.register_tool(
        "first",
        first_tool,
        source=DummyTool.clone_sources["disk-displaced-first"],
    )
    manager.register_tool(
        "second",
        second_tool,
        source=DummyTool.clone_sources["disk-displaced-second"],
    )

    await manager.call_tool("first", {"value": 1})

    clock["value"] += 1.0
    await manager.call_tool("second", {"value": 2})

    first_state = manager._tools["first"]
    assert first_state.tool is first_tool
    assert first_state.loaded_slots == set()
    assert first_state.assigned_slots == set()

    clock["value"] += 0.4
    await manager._evict_disk_idle_tools(now=clock["value"])
    assert first_state.tool is first_tool

    clock["value"] += 0.2
    await manager._evict_disk_idle_tools(now=clock["value"])
    assert first_state.tool is None
    await manager.aclose()


@pytest.mark.asyncio()
@pytest.mark.gpu
async def test_resident_instance_reused_after_resident_slot_eviction(
    dummy_tool_factory,
):
    fake_devices = make_accelerator_devices(2)
    manager = TorchModelToolManager(
        ToolManagerConfig(idle_seconds=0.0, max_devices_per_tool=2),
        device_provider=lambda: fake_devices,
    )
    tool = dummy_tool_factory(label="resident-reuse", delay=0.05, batch_size=1)
    source = DummyTool.clone_sources["resident-reuse"]
    manager.register_tool("resident-reuse", tool, source=source)

    await asyncio.gather(
        manager.call_tool("resident-reuse", {"value": 1}),
        manager.call_tool("resident-reuse", {"value": 2}),
    )

    state = manager._tools["resident-reuse"]
    resident_slot = state.resident_slot
    assert resident_slot is not None
    assert len(state.loaded_slots) == 2

    for index, slot in enumerate(manager._device_slots):
        slot.last_used = 0.0 if index == resident_slot else 10.0
    state.last_used = 0.0
    await manager._evict_idle_tools(now=1.0)

    assert state.tool is tool
    assert state.resident_slot is None
    assert resident_slot not in state.loaded_slots
    assert len(state.loaded_slots) == 1

    load_count = DummyTool.load_counts.get(source, 0)
    await asyncio.gather(
        manager.call_tool("resident-reuse", {"value": 3}),
        manager.call_tool("resident-reuse", {"value": 4}),
    )

    assert DummyTool.load_counts.get(source, 0) == load_count
    assert state.resident_slot is not None
    await manager.aclose()


@pytest.mark.asyncio()
@pytest.mark.gpu
async def test_oom_backoff_reduces_batch_size(dummy_tool_factory):
    device = make_accelerator_device(0)
    manager = TorchModelToolManager(device_provider=lambda: [device])
    tool = dummy_tool_factory(label="oom", batch_size=8, oom_threshold=2)
    manager.register_tool("oom", tool, source=DummyTool.clone_sources["oom"])

    tasks = [
        asyncio.create_task(manager.call_tool("oom", {"value": idx}))
        for idx in range(8)
    ]

    results = await asyncio.gather(*tasks)
    assert [result.value for result in results] == [idx + 1 for idx in range(8)]

    state = manager._tools["oom"]
    assert state.batch_size <= 2
    assert manager._inflight_by_tool["oom"] == 0
    assert tool.batch_size == state.batch_size
    assert tool.fm.forward_batch_sizes[:2] == [8, 4]
    assert all(size <= 2 for size in tool.fm.forward_batch_sizes[2:])

    await manager.aclose()


@pytest.mark.asyncio()
@pytest.mark.gpu
async def test_oom_at_batch_size_one_raises(dummy_tool_factory):
    device = make_accelerator_device(0)
    manager = TorchModelToolManager(device_provider=lambda: [device])
    tool = dummy_tool_factory(label="oom1", batch_size=1, oom_threshold=0)
    manager.register_tool("oom1", tool, source=DummyTool.clone_sources["oom1"])

    with pytest.raises(RuntimeError) as excinfo:
        await manager.call_tool("oom1", {"value": 42})

    assert "out of memory" in str(excinfo.value).lower()
    assert tool.batch_size == 1
    await manager.aclose()


@pytest.mark.asyncio()
@pytest.mark.gpu
async def test_fastmcp_integration(dummy_tool_factory):
    device = make_accelerator_device(0)
    manager = TorchModelToolManager(device_provider=lambda: [device])
    tool = dummy_tool_factory(label="fast")
    manager.register_tool("fast", tool, source=DummyTool.clone_sources["fast"])

    server = FastMCP()
    fast_tools = manager.add_to_fastmcp(server)
    assert "fast" in fast_tools

    tools = await server.list_tools()
    assert any(t.name == tool.name for t in tools)

    fast_tool = fast_tools["fast"]
    result = await fast_tool.fn(value=7)
    assert isinstance(result, DummyOutput)
    assert result.value == 8

    await manager.aclose()


@pytest.mark.asyncio()
async def test_tool_manager_records_core_metrics(dummy_tool_factory, monkeypatch):
    events: list[tuple[str, tuple, dict]] = []

    def capture(name):
        def _capture(*args, **kwargs):
            events.append((name, args, kwargs))

        return _capture

    monkeypatch.setattr(
        nomad_metrics,
        "record_tool_request",
        capture("request"),
    )
    monkeypatch.setattr(
        nomad_metrics,
        "record_tool_queue_wait",
        capture("queue_wait"),
    )
    monkeypatch.setattr(
        nomad_metrics,
        "record_tool_batch",
        capture("batch"),
    )
    monkeypatch.setattr(
        nomad_metrics,
        "record_tool_request_duration",
        capture("request_duration"),
    )

    device = torch.device("cpu")
    manager = TorchModelToolManager(device_provider=lambda: [device])
    tool = dummy_tool_factory(label="metrics", batch_size=2)
    manager.register_tool("metrics", tool, source=DummyTool.clone_sources["metrics"])

    result = await manager.call_tool("metrics", {"value": 10})

    assert result.value == 11
    assert any(event[0] == "request" and event[1] == ("metrics",) for event in events)
    assert any(
        event[0] == "queue_wait" and event[1][0] == "metrics" for event in events
    )
    assert any(
        event[0] == "batch"
        and event[1] == ("metrics", device)
        and event[2]["batch_size"] == 1
        and event[2]["status"] == "ok"
        for event in events
    )
    assert any(
        event[0] == "request_duration"
        and event[1][0] == "metrics"
        and event[2]["status"] == "ok"
        for event in events
    )

    await manager.aclose()


def test_tool_manager_observable_metrics(dummy_tool_factory):
    manager = TorchModelToolManager(device_provider=lambda: [torch.device("cpu")])
    tool = dummy_tool_factory(label="observable", batch_size=3)
    manager.register_tool(
        "observable", tool, source=DummyTool.clone_sources["observable"]
    )

    queue = list(manager.queue_length_observations())
    inflight = list(manager.inflight_observations())
    batch_sizes = list(manager.configured_batch_size_observations())
    resident_tools = list(manager.resident_tool_observations())
    devices = list(manager.device_busy_observations())
    utilization = list(manager.device_utilization_observations())

    assert queue[0].value == 0
    assert queue[0].attributes == {"tool": "observable"}
    assert inflight[0].value == 0
    assert inflight[0].attributes == {"tool": "observable"}
    assert batch_sizes[0].value == 3
    assert batch_sizes[0].attributes == {"tool": "observable"}
    assert resident_tools[0].value == 1
    assert resident_tools[0].attributes == {"tool": "observable"}
    assert devices[0].value == 0
    assert devices[0].attributes["device"] == "cpu"
    assert utilization == []
