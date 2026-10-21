from __future__ import annotations

import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from fastmcp import FastMCP
from pydantic import BaseModel, ValidationError

from nomad.config import ToolManagerConfig
from nomad.model_env import ModelEnvironment
from nomad.model_rpc import (
    ModelProcess,
    ModelRPCError,
    RemoteModel,
    inspect_remote_model,
)
from nomad.model_worker import ModelWorker
from nomad.torch_tool_manager import TorchModelToolManager


@dataclass(frozen=True)
class ExistingEnvironment:
    python: Path
    checksum: str = "existing"

    def ensure(self) -> Path:
        return self.python


def test_model_process_json_rpc_round_trip(monkeypatch, tmp_path: Path):
    module = tmp_path / "isolated_fixture.py"
    module.write_text(
        """
import os
from pydantic import BaseModel

class Input(BaseModel):
    value: int

class Output(BaseModel):
    value: int

class Module:
    def to(self, device):
        self.device = device
        return self

class Tool:
    name = "isolated"
    description = "isolated fixture"
    args_schema = Input
    output_schema = Output
    batch_size = 2

    def __init__(self):
        self.fm = Module()

    @classmethod
    def from_pretrained(cls, source):
        print("model stdout is not protocol output")
        os.write(1, b"native model stdout is not protocol output\\n")
        return cls()

    def batch_as_completed(self, inputs, max_concurency=None):
        for item in inputs:
            yield Output(value=item.value + 1)
""",
        encoding="utf-8",
    )
    current_pythonpath = str(tmp_path)
    if inherited := os.environ.get("PYTHONPATH"):
        current_pythonpath += os.pathsep + inherited
    monkeypatch.setenv("PYTHONPATH", current_pythonpath)
    environment = ExistingEnvironment(Path(sys.executable))

    remote = inspect_remote_model(
        environment=environment,  # type: ignore[arg-type]
        model_class="isolated_fixture.Tool",
        source="weights",
        tool_name=None,
        batch_size=None,
    )

    assert remote.name == "isolated"
    assert remote.args_schema["properties"]["value"]["type"] == "integer"
    with ModelProcess(environment) as process:  # type: ignore[arg-type]
        process.load(remote, "cpu")
        assert process.run_batch([{"value": 2}], batch_size=1) == [{"value": 3}]


def test_model_worker_validates_outputs_against_declared_schema():
    class Input(BaseModel):
        value: int

    class Output(BaseModel):
        value: int

    class InvalidOutputTool:
        args_schema = Input
        output_schema = Output

        @staticmethod
        def batch_as_completed(inputs, max_concurency=None):
            return [{"unexpected": item.value} for item in inputs]

    worker = ModelWorker()
    worker.tool = InvalidOutputTool()

    with pytest.raises(ValidationError, match="value"):
        worker.run_batch({"inputs": [{"value": 2}], "batch_size": 1})


@pytest.mark.asyncio
async def test_slot_reuses_worker_for_same_environment_and_replaces_for_new_one(
    monkeypatch,
):
    created = []

    class FakeProcess:
        def __init__(self, environment, *, timeout_seconds):
            self.environment = environment
            self.timeout_seconds = timeout_seconds
            self.alive = True
            self.closed = False
            self.pid = len(created) + 1
            self.requests = []
            created.append(self)

        def load(self, remote, device):
            self.remote = remote
            return {}

        def run_batch(self, inputs, *, batch_size):
            return [{"value": item["value"] + 1} for item in inputs]

        def request(self, method, params=None):
            self.requests.append(method)
            return None

        def close(self):
            self.closed = True
            self.alive = False

    monkeypatch.setattr("nomad.torch_tool_manager.ModelProcess", FakeProcess)
    first_env = ModelEnvironment(("same",))
    second_env = ModelEnvironment(("different",))

    def remote(name: str, environment: ModelEnvironment) -> RemoteModel:
        schema = {
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
        }
        return RemoteModel(
            environment=environment,
            model_class="fixture.Tool",
            source=name,
            name=name,
            description=name,
            batch_size=1,
            args_schema=schema,
            output_schema=schema,
        )

    manager = TorchModelToolManager(
        ToolManagerConfig(
            idle_seconds=None,
            disk_idle_seconds=30,
            gc_idle_seconds=30,
            venv_idle_seconds=30,
            rpc_timeout_seconds=12,
        ),
        device_provider=lambda: [torch.device("cpu")],
    )
    manager.register_remote_tool(remote("first", first_env))
    manager.register_remote_tool(remote("second", first_env))
    manager.register_remote_tool(remote("third", second_env))

    fast_tools = manager.add_to_fastmcp(FastMCP())
    assert await fast_tools["first"].fn(value=1) == {"value": 2}
    first_process = created[0]
    assert first_process.timeout_seconds == 12
    assert await manager.call_tool("second", {"value": 2}) == {"value": 3}
    assert created == [first_process]
    assert not first_process.closed

    assert await manager.call_tool("third", {"value": 3}) == {"value": 4}
    assert first_process.closed
    assert len(created) == 2

    dead_process = created[-1]
    dead_process.alive = False
    assert await manager.call_tool("third", {"value": 4}) == {"value": 5}
    assert dead_process.closed
    assert len(created) == 3

    slot = manager._device_slots[0]
    await manager._clear_cache_if_server_idle(now=manager._last_server_activity + 31)
    assert "clear_cache" in created[-1].requests

    await manager._offload_slot(0, expected_tool_name="third")
    state = manager._tools["third"]
    await manager._evict_disk_idle_tools(now=state.last_used + 31)
    assert "unload" in created[-1].requests
    assert slot.worker_tool is None
    assert slot.worker is created[-1]

    await manager._evict_idle_workers(now=slot.last_used + 31)
    assert created[-1].closed
    assert slot.worker is None
    await manager.aclose()


def test_device_memory_includes_host_and_model_worker(monkeypatch):
    manager = TorchModelToolManager(
        device_provider=lambda: [torch.device("cpu")],
    )
    device = torch.device("cuda:0")
    manager._device_slots[0].device = device

    class MemoryProcess:
        alive = True
        pid = 123
        request_count = 0

        def request(self, method, params=None):
            self.request_count += 1
            assert method == "device_memory"
            assert params == {"device": "cuda:0"}
            return {"allocated": 7, "reserved": 13}

    manager._device_slots[0].worker = MemoryProcess()  # type: ignore[assignment]
    monkeypatch.setattr(
        manager,
        "_host_device_memory_value",
        lambda requested_device, kind: {"allocated": 5, "reserved": 11}[kind],
    )

    assert manager._device_memory_value(device, "allocated") == 12
    assert manager._device_memory_value(device, "reserved") == 24
    assert manager._device_slots[0].worker.request_count == 1  # type: ignore[union-attr]


def test_concurrent_device_memory_metrics_share_worker_snapshot(monkeypatch):
    monkeypatch.setattr(
        "nomad.torch_tool_manager._DEVICE_MEMORY_SNAPSHOT_SECONDS", 0.01
    )
    manager = TorchModelToolManager(
        device_provider=lambda: [torch.device("cpu")],
    )
    device = torch.device("cuda:0")
    manager._device_slots[0].device = device
    request_entered = threading.Event()
    release_request = threading.Event()
    request_count = 0
    request_count_lock = threading.Lock()

    class MemoryProcess:
        alive = True
        pid = 123

        def request(self, method, params=None):
            nonlocal request_count
            with request_count_lock:
                request_count += 1
            request_entered.set()
            assert release_request.wait(timeout=2)
            return {"allocated": 7, "reserved": 13}

    manager._device_slots[0].worker = MemoryProcess()  # type: ignore[assignment]
    monkeypatch.setattr(manager, "_host_device_memory_value", lambda *args: 0)
    started = threading.Barrier(3)

    def read(kind: str) -> int | None:
        started.wait()
        return manager._device_memory_value(device, kind)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(read, kind) for kind in ("allocated", "reserved")]
        started.wait()
        assert request_entered.wait(timeout=2)
        time.sleep(0.05)
        assert request_count == 1
        release_request.set()
        assert [future.result(timeout=2) for future in futures] == [7, 13]
        assert request_count == 1


def test_rpc_timeout_includes_blocked_stdin_write():
    release_write = threading.Event()
    terminated = threading.Event()

    class BlockingStdin:
        def write(self, value):
            release_write.wait(timeout=2)

        def flush(self):
            pass

    class FakeProcess:
        pid = 123
        stdin = BlockingStdin()

        @staticmethod
        def poll():
            return None

    process = ModelProcess.__new__(ModelProcess)
    process.timeout_seconds = 0.05
    process._process = FakeProcess()  # type: ignore[assignment]
    process._request_id = 0
    process._lock = threading.Lock()

    def terminate() -> None:
        terminated.set()
        release_write.set()

    process._terminate_process = terminate  # type: ignore[method-assign]
    started = time.monotonic()

    with pytest.raises(ModelRPCError, match="timed out after 0.05 seconds"):
        process.request("blocked")

    assert time.monotonic() - started < 0.5
    assert terminated.is_set()


def test_close_terminates_worker_blocked_in_rpc(monkeypatch, tmp_path: Path):
    module = tmp_path / "blocking_fixture.py"
    module.write_text(
        """
import time
from pydantic import BaseModel

class Input(BaseModel):
    value: int

class Output(BaseModel):
    value: int

class Module:
    def to(self, device):
        return self

class Tool:
    name = "blocking"
    description = "blocking fixture"
    args_schema = Input
    output_schema = Output
    batch_size = 1

    def __init__(self):
        self.fm = Module()

    @classmethod
    def from_pretrained(cls, source):
        return cls()

    def batch_as_completed(self, inputs, max_concurency=None):
        time.sleep(60)
        return []
""",
        encoding="utf-8",
    )
    pythonpath = str(tmp_path)
    if inherited := os.environ.get("PYTHONPATH"):
        pythonpath += os.pathsep + inherited
    monkeypatch.setenv("PYTHONPATH", pythonpath)
    environment = ExistingEnvironment(Path(sys.executable))
    remote = inspect_remote_model(
        environment=environment,  # type: ignore[arg-type]
        model_class="blocking_fixture.Tool",
        source="weights",
        tool_name=None,
        batch_size=None,
    )
    process = ModelProcess(environment)  # type: ignore[arg-type]
    process.load(remote, "cpu")
    errors: list[Exception] = []

    def run_blocking_request() -> None:
        try:
            process.run_batch([{"value": 1}], batch_size=1)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    request_thread = threading.Thread(target=run_blocking_request)
    request_thread.start()
    time.sleep(0.1)
    started = time.monotonic()
    process.close()
    request_thread.join(timeout=3)

    assert time.monotonic() - started < 3
    assert not request_thread.is_alive()
    assert not process.alive
    assert errors

    with ModelProcess(
        environment,  # type: ignore[arg-type]
        timeout_seconds=5,
    ) as timed_process:
        timed_process.load(remote, "cpu")
        timed_process.timeout_seconds = 0.1
        with pytest.raises(ModelRPCError, match="timed out after 0.1 seconds"):
            timed_process.run_batch([{"value": 1}], batch_size=1)
        assert not timed_process.alive


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group behavior")
@pytest.mark.parametrize("crash_worker", [False, True])
def test_close_terminates_model_spawned_child(
    monkeypatch, tmp_path: Path, crash_worker: bool
):
    module = tmp_path / "child_fixture.py"
    module.write_text(
        """
import subprocess
import sys
from pathlib import Path
from pydantic import BaseModel

class Input(BaseModel):
    value: int

class Output(BaseModel):
    value: int

class Module:
    def to(self, device):
        return self

class Tool:
    name = "child"
    description = "child fixture"
    args_schema = Input
    output_schema = Output
    batch_size = 1

    def __init__(self):
        self.fm = Module()

    @classmethod
    def from_pretrained(cls, source):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        Path(source).write_text(str(child.pid))
        return cls()

    def batch_as_completed(self, inputs, max_concurency=None):
        return []
""",
        encoding="utf-8",
    )
    pythonpath = str(tmp_path)
    if inherited := os.environ.get("PYTHONPATH"):
        pythonpath += os.pathsep + inherited
    monkeypatch.setenv("PYTHONPATH", pythonpath)
    environment = ExistingEnvironment(Path(sys.executable))
    pid_file = tmp_path / "child.pid"
    schema = {"type": "object", "properties": {"value": {"type": "integer"}}}
    remote = RemoteModel(
        environment=environment,  # type: ignore[arg-type]
        model_class="child_fixture.Tool",
        source=str(pid_file),
        name="child",
        description="child fixture",
        batch_size=1,
        args_schema=schema,
        output_schema=schema,
    )
    process = ModelProcess(environment)  # type: ignore[arg-type]
    process.load(remote, "cpu")
    child_pid = int(pid_file.read_text())

    if crash_worker:
        process._process.kill()
        process._process.wait()
    process.close()

    deadline = time.monotonic() + 2
    while _process_exists(child_pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _process_exists(child_pid)


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    stat = Path(f"/proc/{pid}/stat")
    if stat.is_file() and stat.read_text().split()[2] == "Z":
        return False
    return True
