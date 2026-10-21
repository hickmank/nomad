from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from nomad._torch_module_compat import is_torch_module_tool_instance
from nomad.common.config_errors import ConfigError
from nomad.config import (
    ServerConfig,
    ToolConfig,
    ToolManagerConfig,
    TorchModuleConfig,
)
from nomad.fm_base_tool import TorchModuleTool


class DummyInput(BaseModel):
    value: int


class DummyOutput(BaseModel):
    value: int


class DummyTorchModuleTool(TorchModuleTool):
    args_schema: type[DummyInput] = DummyInput
    output_schema: type[DummyOutput] = DummyOutput

    def preprocess(self, inputs):
        return inputs

    def _forward(self, model_inputs):
        return model_inputs

    def postprocess(self, model_output):
        for item in model_output:
            yield DummyOutput(value=getattr(item, "value", 0))


def test_server_config_resolves_entries(monkeypatch):
    calls: list[tuple[str, dict]] = []

    class DummyModel:
        @classmethod
        def from_pretrained(cls, name_or_path, **kwargs):
            calls.append((name_or_path, kwargs))
            return DummyTorchModuleTool(
                fm=torch.nn.Linear(1, 1),
                name="dummy",
                description="desc",
                batch_size=kwargs.get("batch_size", 1),
                device=torch.device("cpu"),
            )

    def referenced_tool(value: int) -> int:
        return value

    class DummyTool:
        def __init__(self, factor: int = 1):
            self.factor = factor

        def __call__(self, value: int) -> int:
            return value * self.factor

    module_name = "test_nomad_dummy_config"
    dummy_module = types.ModuleType(module_name)
    dummy_module.DummyModel = DummyModel
    dummy_module.referenced_tool = referenced_tool
    dummy_module.DummyTool = DummyTool
    sys.modules[module_name] = dummy_module

    class FakeRepoSpec:
        @classmethod
        def parse(cls, spec: str, *, base_dir=None):
            assert spec == "foo"
            return types.SimpleNamespace(pull=lambda: Path("foo"))

    monkeypatch.setattr("nomad.config.RepoSpec", FakeRepoSpec)

    config_data = {
        "fmod_models": [
            {
                "model_class": f"{module_name}.DummyModel",
                "name_or_path": "foo",
                "tool_name": "dummy-tool",
                "batch_size": 8,
            }
        ],
        "tools": [
            {"tool": f"{module_name}.referenced_tool", "aliases": ["ref"]},
            {"tool": f"{module_name}.DummyTool", "tool_kwargs": {"factor": 3}},
        ],
        "tool_manager": {
            "idle_seconds": 10.0,
            "gc_idle_seconds": 30.0,
            "disk_idle_seconds": 60.0,
            "max_pending_per_tool": 5,
        },
    }

    config = ServerConfig.model_validate(config_data)

    assert isinstance(config.tool_manager, ToolManagerConfig)
    assert config.search_tool.expose is False
    assert config.search_tool.weights.prefix_name == 8.0
    manager = config.tool_manager.instantiate()
    try:
        assert manager.idle_seconds == 10.0
        assert manager.gc_idle_seconds == 30.0
        assert manager.disk_idle_seconds == 60.0
    finally:
        import asyncio

        asyncio.run(manager.aclose())

    assert len(config.fmod_models) == 1
    fm_config = config.fmod_models[0]
    assert isinstance(fm_config, TorchModuleConfig)
    model = fm_config.build_tool()
    assert isinstance(model, DummyTorchModuleTool)
    assert calls == [("foo", {})]

    class DummyFastMCPServer:
        def __init__(self):
            self.tools: list[object] = []

        def add_tool(self, tool):
            self.tools.append(tool)
            return tool

    fm_server = DummyFastMCPServer()
    registered_model_tool = fm_config.add_to_fastmcp(fm_server)
    assert registered_model_tool.name == "dummy_tool"
    assert fm_server.tools == [registered_model_tool]

    assert len(config.tools) == 2
    tool_config, class_tool_config = config.tools
    assert isinstance(tool_config, ToolConfig)
    assert isinstance(class_tool_config, ToolConfig)

    class DummyServer:
        def __init__(self):
            self.tools: list[tuple[object, dict]] = []

        def add_tool(self, tool, **kwargs):
            self.tools.append((tool, kwargs))

    server = DummyServer()
    tool_config.add_to_fastmcp(server)
    class_tool_config.add_to_fastmcp(server)

    assert server.tools[0][0].fn is referenced_tool
    assert server.tools[0][0].name == "test_nomad_dummy_configpreferenced_tool"
    assert server.tools[0][1] == {}

    instantiated_tool, kwargs = server.tools[1]
    assert callable(instantiated_tool.fn)
    assert instantiated_tool.fn(2) == 6
    assert instantiated_tool.name == "test_nomad_dummy_configpDummyTool"
    assert kwargs == {}


def test_tool_config_registration_raises_for_broken_tool_import():
    class DummyServer:
        def add_tool(self, tool):
            pass

    tool_config = ToolConfig(tool="missing.module.tool")

    with pytest.raises(ModuleNotFoundError):
        tool_config.add_to_fastmcp(DummyServer())


def test_tool_manager_config_rejects_unknown_options():
    with pytest.raises(ValidationError) as exc_info:
        ToolManagerConfig.model_validate({"typo_seconds": 10})

    assert "Extra inputs are not permitted" in str(exc_info.value)


@pytest.mark.parametrize(
    "field",
    ["idle_seconds", "gc_idle_seconds", "disk_idle_seconds", "venv_idle_seconds"],
)
def test_tool_manager_config_rejects_negative_seconds(field: str):
    with pytest.raises(ValidationError) as exc_info:
        ToolManagerConfig.model_validate({field: -1})

    assert "greater than or equal to 0" in str(exc_info.value)


def test_tool_manager_config_defaults_venv_idle_seconds():
    config = ToolManagerConfig()

    assert config.venv_idle_seconds == 1200.0
    assert config.rpc_timeout_seconds == 30.0


def test_tool_manager_config_rejects_removed_enabled_flag():
    with pytest.raises(ValidationError) as exc_info:
        ToolManagerConfig.model_validate({"enabled": False})

    assert "Extra inputs are not permitted" in str(exc_info.value)


@pytest.mark.parametrize("value", [0, -1])
def test_tool_manager_config_rejects_nonpositive_rpc_timeout(value: float):
    with pytest.raises(ValidationError) as exc_info:
        ToolManagerConfig.model_validate({"rpc_timeout_seconds": value})

    assert "greater than 0" in str(exc_info.value)


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ("./requirements.txt", "Invalid PEP 508 requirement"),
        ("-r requirements.txt", "Invalid PEP 508 requirement"),
        (["demo-package>=1", "./local"], "Invalid PEP 508 requirement"),
        ("requirements.txt", "Environment file paths are not supported"),
        ("pyproject.toml", "Environment file paths are not supported"),
        ("demo @ ./local", "direct references must use an absolute URL"),
        ("demo @ file:./local", "direct references must use an absolute URL"),
        ("demo @ https:relative", "direct references must use an absolute URL"),
        ("demo @ git+file:./local", "direct references must use an absolute URL"),
        ("demo @ ftp:relative", "direct references must use an absolute URL"),
        ("demo @ hg+https:relative", "direct references must use an absolute URL"),
        (r"demo @ C:\local", "direct references must use an absolute URL"),
        ("demo-package\n--extra-index-url https://example.com", "line breaks"),
        ("demo-package\0other-package", "NUL bytes"),
    ],
)
def test_torch_module_config_rejects_non_pep508_environments(env, message):
    with pytest.raises(ValidationError) as exc_info:
        TorchModuleConfig(
            model_class="demo.Tool",
            name_or_path="demo/model",
            env=env,
        )

    assert message in str(exc_info.value)


def test_torch_module_config_accepts_pep508_environment_list():
    config = TorchModuleConfig(
        model_class="demo.Tool",
        name_or_path="demo/model",
        env=["demo-package>=1", "other-package[extra]"],
    )

    assert config.env == ["demo-package>=1", "other-package[extra]"]


def test_torch_module_config_accepts_absolute_direct_reference():
    requirement = "demo-package @ file:///opt/demo-package"

    config = TorchModuleConfig(
        model_class="demo.Tool",
        name_or_path="demo/model",
        env=requirement,
    )

    assert config.env == requirement


def test_torch_module_config_accepts_legacy_ursa_tool(monkeypatch):
    class LegacyTorchModuleTool(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)

        fm: torch.nn.Module
        name: str | None = None
        description: str
        args_schema: type[BaseModel]
        output_schema: type[BaseModel]
        batch_size: int = 1
        device: torch.device = Field(default_factory=lambda: torch.device("cpu"))

        def model_post_init(self, __context) -> None:
            self.fm = self.fm.to(self.device)
            if self.name is None:
                self.name = self.__class__.__name__

        def add_to_fastmcp(self, server):
            raise AssertionError("Nomad should register legacy tools itself")

    legacy_ursa = types.ModuleType("ursa")
    legacy_ursa_tools = types.ModuleType("ursa.tools")
    legacy_ursa_fm = types.ModuleType("ursa.tools.fm_base_tool")
    legacy_ursa.__path__ = []
    legacy_ursa_tools.__path__ = []
    legacy_ursa_fm.TorchModuleTool = LegacyTorchModuleTool
    legacy_ursa.tools = legacy_ursa_tools
    legacy_ursa_tools.fm_base_tool = legacy_ursa_fm

    monkeypatch.setitem(sys.modules, "ursa", legacy_ursa)
    monkeypatch.setitem(sys.modules, "ursa.tools", legacy_ursa_tools)
    monkeypatch.setitem(sys.modules, "ursa.tools.fm_base_tool", legacy_ursa_fm)

    class LegacyCompatibleModel:
        @classmethod
        def from_pretrained(cls, name_or_path, **kwargs):
            assert name_or_path == "legacy-model"
            assert kwargs == {}
            return LegacyTorchModuleTool(
                fm=torch.nn.Identity(),
                name="legacy",
                description="legacy tool",
                args_schema=DummyInput,
                output_schema=DummyOutput,
                batch_size=2,
                device=torch.device("cpu"),
            )

    module_name = "test_nomad_legacy_ursa_config"
    dummy_module = types.ModuleType(module_name)
    dummy_module.LegacyCompatibleModel = LegacyCompatibleModel
    sys.modules[module_name] = dummy_module

    class FakeRepoSpec:
        @classmethod
        def parse(cls, spec: str, *, base_dir=None):
            assert spec == "legacy-model"
            return types.SimpleNamespace(pull=lambda: Path("legacy-model"))

    monkeypatch.setattr("nomad.config.RepoSpec", FakeRepoSpec)

    config = TorchModuleConfig(
        model_class=f"{module_name}.LegacyCompatibleModel",
        name_or_path="legacy-model",
        batch_size=2,
    )

    model = config.instantiate()
    assert isinstance(model, LegacyTorchModuleTool)
    assert is_torch_module_tool_instance(model)

    class DummyFastMCPServer:
        def __init__(self):
            self.tools: list[object] = []

        def add_tool(self, tool):
            self.tools.append(tool)
            return tool

    server = DummyFastMCPServer()
    registered_tool = config.add_to_fastmcp(server)
    assert registered_tool.name == "legacy"
    assert server.tools == [registered_tool]


def test_server_config_from_file_resolves_relative_model_paths_from_config_dir(
    monkeypatch, tmp_path: Path
):
    calls: list[tuple[Path | str, dict]] = []

    class DummyModel:
        @classmethod
        def from_pretrained(cls, name_or_path, **kwargs):
            calls.append((name_or_path, kwargs))
            return DummyTorchModuleTool(
                fm=torch.nn.Linear(1, 1),
                name="dummy",
                description="desc",
                batch_size=1,
                device=torch.device("cpu"),
            )

    module_name = "test_nomad_config_relative_from_file"
    dummy_module = types.ModuleType(module_name)
    dummy_module.DummyModel = DummyModel
    sys.modules[module_name] = dummy_module

    config_dir = tmp_path / "configs"
    model_dir = config_dir / "models" / "demo-model"
    model_dir.mkdir(parents=True)
    config_path = config_dir / "nomad.yml"
    config_path.write_text(
        "\n".join(
            [
                "tool_manager: {}",
                "tools: []",
                "fmod_models:",
                f"  - model_class: {module_name}.DummyModel",
                "    name_or_path: models/demo-model",
                "",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)

    config = ServerConfig.from_file(config_path)
    model = config.instantiate_model(config.fmod_models[0])

    assert isinstance(model, DummyTorchModuleTool)
    assert calls == [(str(model_dir), {})]


def test_torch_module_config_resolves_existing_local_path(tmp_path):
    calls: list[tuple[Path | str, dict]] = []

    class DummyModel:
        @classmethod
        def from_pretrained(cls, name_or_path, **kwargs):
            calls.append((name_or_path, kwargs))
            return DummyTorchModuleTool(
                fm=torch.nn.Linear(1, 1),
                name="dummy",
                description="desc",
                batch_size=1,
                device=torch.device("cpu"),
            )

    module_name = "test_nomad_config_local_path"
    dummy_module = types.ModuleType(module_name)
    dummy_module.DummyModel = DummyModel
    sys.modules[module_name] = dummy_module

    local_model_path = tmp_path / "model"
    local_model_path.mkdir()

    config = TorchModuleConfig(
        model_class=f"{module_name}.DummyModel",
        name_or_path=str(local_model_path),
    )

    model = config.instantiate()
    assert isinstance(model, DummyTorchModuleTool)
    assert calls == [(str(local_model_path), {})]


def test_torch_module_config_uses_repo_spec(monkeypatch):
    calls: list[tuple[Path | str, dict]] = []

    class DummyModel:
        @classmethod
        def from_pretrained(cls, name_or_path, **kwargs):
            calls.append((name_or_path, kwargs))
            return DummyTorchModuleTool(
                fm=torch.nn.Linear(1, 1),
                name="dummy",
                description="desc",
                batch_size=1,
                device=torch.device("cpu"),
            )

    module_name = "test_nomad_config_repo_spec"
    dummy_module = types.ModuleType(module_name)
    dummy_module.DummyModel = DummyModel
    sys.modules[module_name] = dummy_module

    pulled_path = Path("/tmp/fetched-model")

    class FakeParsedSpec:
        def pull(self):
            return pulled_path

    class FakeRepoSpec:
        @classmethod
        def parse(cls, spec: str, *, base_dir=None):
            assert spec == "git+https://example.com/model.git@main#weights"
            assert base_dir is None
            return FakeParsedSpec()

    monkeypatch.setattr("nomad.config.RepoSpec", FakeRepoSpec)

    config = TorchModuleConfig(
        model_class=f"{module_name}.DummyModel",
        name_or_path="git+https://example.com/model.git@main#weights",
        revision="abc123",
    )

    model = config.instantiate()
    assert isinstance(model, DummyTorchModuleTool)
    assert calls == [(str(pulled_path), {})]


def test_torch_module_config_uses_hf_fallback_through_repo_spec(monkeypatch):
    calls: list[tuple[Path | str, dict]] = []

    class DummyModel:
        @classmethod
        def from_pretrained(cls, name_or_path, **kwargs):
            calls.append((name_or_path, kwargs))
            return DummyTorchModuleTool(
                fm=torch.nn.Linear(1, 1),
                name="dummy",
                description="desc",
                batch_size=1,
                device=torch.device("cpu"),
            )

    module_name = "test_nomad_config_name_fallback"
    dummy_module = types.ModuleType(module_name)
    dummy_module.DummyModel = DummyModel
    sys.modules[module_name] = dummy_module

    class FakeRepoSpec:
        @classmethod
        def parse(cls, spec: str, *, base_dir=None):
            assert spec == "hf-org/my-model"
            return types.SimpleNamespace(pull=lambda: Path("/tmp/hf-model"))

    monkeypatch.setattr("nomad.config.RepoSpec", FakeRepoSpec)

    config = TorchModuleConfig(
        model_class=f"{module_name}.DummyModel",
        name_or_path="hf-org/my-model",
        cache_dir="/tmp/cache",
    )

    model = config.instantiate()
    assert isinstance(model, DummyTorchModuleTool)
    assert calls == [("/tmp/hf-model", {})]


def test_torch_module_config_uses_hf_repo_id_as_default_tool_name(monkeypatch):
    calls: list[tuple[Path | str, dict]] = []

    class DummyModel:
        @classmethod
        def from_pretrained(cls, name_or_path, **kwargs):
            calls.append((name_or_path, kwargs))
            return DummyTorchModuleTool(
                fm=torch.nn.Linear(1, 1),
                name="1af94460e96bcfeb796d475c3feadb55e241aa37",
                description="desc",
                batch_size=1,
                device=torch.device("cpu"),
            )

    module_name = "test_nomad_config_hf_tool_name"
    dummy_module = types.ModuleType(module_name)
    dummy_module.DummyModel = DummyModel
    sys.modules[module_name] = dummy_module

    class FakeRepoSpec:
        scheme = "hf"
        location = "mist-models/mist-28M-gzwqzpcr-qm8"

        def pull(self):
            return Path("/tmp/hf-model")

    class FakeRepoSpecFactory:
        @classmethod
        def parse(cls, spec: str, *, base_dir=None):
            assert spec == "mist-models/mist-28M-gzwqzpcr-qm8"
            return FakeRepoSpec()

    monkeypatch.setattr("nomad.config.RepoSpec", FakeRepoSpecFactory)

    config = TorchModuleConfig(
        model_class=f"{module_name}.DummyModel",
        name_or_path="mist-models/mist-28M-gzwqzpcr-qm8",
    )

    model = config.build_tool()
    assert model.name == "mist_models---mist_28M_gzwqzpcr_qm8"
    assert calls == [("/tmp/hf-model", {})]


def test_server_config_rejects_legacy_expose_search_tool_flag():
    with pytest.raises(ValueError, match="search_tool.expose"):
        ServerConfig.model_validate(
            {
                "fmod_models": [],
                "tools": [],
                "tool_manager": {},
                "expose_search_tool": True,
            }
        )


def test_server_config_from_file_missing_has_specific_error(tmp_path: Path):
    config_path = tmp_path / "missing.yml"

    with pytest.raises(ConfigError, match="Config file not found"):
        ServerConfig.from_file(config_path)


def test_server_config_from_file_malformed_yaml_has_location(tmp_path: Path):
    config_path = tmp_path / "nomad.yml"
    config_path.write_text("fmod_models:\n  - [\n", encoding="utf-8")

    with pytest.raises(ConfigError) as exc_info:
        ServerConfig.from_file(config_path)

    message = str(exc_info.value)
    assert "YAML syntax error" in message
    assert "line 3" in message


def test_server_config_from_file_rejects_non_mapping_root(tmp_path: Path):
    config_path = tmp_path / "nomad.yml"
    config_path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    with pytest.raises(ConfigError) as exc_info:
        ServerConfig.from_file(config_path)

    assert "Config root" in str(exc_info.value)
    assert "list" in str(exc_info.value)


def test_server_config_from_file_wrong_type_describes_expected_shape(
    tmp_path: Path,
):
    config_path = tmp_path / "nomad.yml"
    config_path.write_text(
        "fmod_models: nope\ntools: []\ntool_manager: {}\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError) as exc_info:
        ServerConfig.from_file(config_path)

    assert "`fmod_models` must be a list (got 'nope')" in str(exc_info.value)


def test_server_config_from_file_range_error_describes_bound(tmp_path: Path):
    config_path = tmp_path / "nomad.yml"
    config_path.write_text(
        "fmod_models: []\n"
        "tools: []\n"
        "tool_manager: {}\n"
        "search_tool: {candidate_limit: 0}\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError) as exc_info:
        ServerConfig.from_file(config_path)

    assert (
        "`search_tool.candidate_limit` must be greater than or equal to 1 (got 0)"
        in str(exc_info.value)
    )


def test_server_config_accepts_search_tool_weights():
    config = ServerConfig.model_validate(
        {
            "fmod_models": [],
            "tools": [],
            "tool_manager": {},
            "search_tool": {
                "expose": True,
                "candidate_limit": 75,
                "weights": {
                    "prefix_name": 9.0,
                    "suffix_name": 5.0,
                    "substring_name": 1.0,
                },
            },
        }
    )

    assert config.search_tool.expose is True
    assert config.search_tool.candidate_limit == 75
    assert config.search_tool.weights.prefix_name == 9.0
    assert config.search_tool.weights.suffix_name == 5.0
    assert config.search_tool.weights.substring_name == 1.0
