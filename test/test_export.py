from __future__ import annotations

import types
from pathlib import Path

import pytest
import yaml

from nomad import export as nomad_export
from nomad import hub
from nomad.export import ExportTarget, export_model_report, export_models_config


def test_export_models_config_builds_pep508_environments(
    monkeypatch,
    tmp_path: Path,
):
    built_environments = []

    def fake_ensure(self, *, cache_root=None):
        built_environments.append((self, cache_root))
        return cache_root / "venv" / "python"

    monkeypatch.setattr("nomad.export.ModelEnvironment.ensure", fake_ensure)
    config_dir = tmp_path / "config"
    model_dir = config_dir / "models" / "demo"
    model_dir.mkdir(parents=True)
    (model_dir / "weights.bin").write_text("weights", encoding="utf-8")

    config_path = config_dir / "nomad.yml"
    config_path.write_text(
        "\n".join(
            [
                "fmod_models:",
                "  - model_class: demo.Tool",
                "    name_or_path: models/demo",
                "    tool_name: first-model",
                "    env: demo-package==1.2",
                "  - model_class: demo.Tool",
                "    name_or_path: models/demo",
                "    tool_name: second-model",
                "    env:",
                "      - other-package>=2",
                "      - third-package[extra]",
                "",
            ]
        ),
        encoding="utf-8",
    )

    output_path = export_models_config(config_path, tmp_path / "bundle")
    exported = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    assert [entry["env"] for entry in exported["fmod_models"]] == [
        "demo-package==1.2",
        ["other-package>=2", "third-package[extra]"],
    ]
    assert [environment.requirements for environment, _ in built_environments] == [
        ("demo-package==1.2",),
        ("other-package>=2", "third-package[extra]"),
    ]
    assert all(
        environment.base_dir == config_dir for environment, _ in built_environments
    )
    assert [cache_root for _, cache_root in built_environments] == [
        tmp_path / "bundle",
        tmp_path / "bundle",
    ]


def test_export_models_config_can_skip_venvs(monkeypatch, tmp_path: Path):
    project_dir = tmp_path / "config"
    model_dir = project_dir / "models" / "demo"
    model_dir.mkdir(parents=True)
    (model_dir / "weights.bin").write_text("weights", encoding="utf-8")
    config_path = project_dir / "nomad.yml"
    config_path.write_text(
        "\n".join(
            [
                "fmod_models:",
                "  - model_class: demo_package.Tool",
                "    name_or_path: models/demo",
                "    env: demo-package==1.2",
                "",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "nomad.export.ModelEnvironment.ensure",
        lambda *args, **kwargs: pytest.fail("venv construction was not disabled"),
    )

    export_models_config(
        config_path,
        tmp_path / "bundle",
        build_venvs=False,
    )


def test_export_models_config_https_rewrites_git_sources_and_normalizes_hf(
    monkeypatch, tmp_path: Path
):
    config_dir = tmp_path / "config"
    local_model_dir = config_dir / "models" / "local-model"
    local_model_dir.mkdir(parents=True)
    (local_model_dir / "weights.safetensors").write_text("weights", encoding="utf-8")

    config_path = config_dir / "nomad.yml"
    config_path.write_text(
        "\n".join(
            [
                "tool_manager: {}",
                "tools: []",
                "fmod_models:",
                "  - model_class: tests.DummyModel",
                "    name_or_path: models/local-model",
                "  - model_class: tests.DummyModel",
                "    name_or_path: git+ssh://git@example.com:2222/org/repo.git@main#weights",
                "    tool_name: repo-weights",
                "  - model_class: tests.DummyModel",
                "    name_or_path: hf-org/my-model",
                "  - model_class: tests.DummyModel",
                f"    name_or_path: oras://registry.example.com/models/model@{'d' * 64}#weights",
                "    tool_name: oras-weights",
                "",
            ]
        ),
        encoding="utf-8",
    )

    copy_calls: list[tuple[Path, Path, bool]] = []

    def fake_copy_cow(source: Path, destination: Path, symlinks: bool = True):
        copy_calls.append((source, destination, symlinks))
        destination.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("nomad.hub.copy_cow", fake_copy_cow)
    monkeypatch.setattr(hub.GitRepoSpec, "cache_digest", lambda self: "a" * 40)
    monkeypatch.setattr(hub.HuggingFaceRepoSpec, "cache_digest", lambda self: "b" * 40)

    output_path = export_models_config(
        config_path,
        tmp_path / "bundle",
        to=ExportTarget.HTTPS,
    )
    rendered = output_path.read_text(encoding="utf-8")
    exported = yaml.safe_load(rendered)

    assert [entry["name_or_path"] for entry in exported["fmod_models"]] == [
        "file:models/local-model",
        f"git+https://example.com/org/repo.git@{'a' * 40}#weights",
        f"hf://hf-org/my-model@{'b' * 40}",
        f"oras://registry.example.com/models/model@{'d' * 64}#weights",
    ]
    assert copy_calls == [
        (
            local_model_dir,
            tmp_path / "bundle" / "models" / "local-model",
            True,
        )
    ]
    assert "# source: models/local-model" in rendered
    assert (
        "# source: git+ssh://git@example.com:2222/org/repo.git@main#weights" in rendered
    )
    assert "# source: hf-org/my-model" in rendered
    assert "# source: oras://registry.example.com/models/model" not in rendered


def test_export_models_config_oras_pushes_sources_and_rewrites_config(
    monkeypatch, tmp_path: Path
):
    config_dir = tmp_path / "config"
    local_model_dir = config_dir / "models" / "local-model"
    local_model_dir.mkdir(parents=True)
    (local_model_dir / "weights.safetensors").write_text("weights", encoding="utf-8")

    config_path = config_dir / "nomad.yml"
    config_path.write_text(
        "\n".join(
            [
                "tool_manager: {}",
                "tools: []",
                "fmod_models:",
                "  - model_class: tests.DummyModel",
                "    name_or_path: models/local-model",
                "    tool_name: weather-rollout",
                "  - model_class: tests.DummyModel",
                "    name_or_path: models/local-model",
                "    tool_name: weather-rollout-copy",
                "",
            ]
        ),
        encoding="utf-8",
    )

    pushes: list[tuple[Path, str]] = []

    def fake_oras_push(self, source: Path):
        pushes.append((source, self.location))
        _, tag = self.location.rsplit(":", 1)
        return hub.OrasRepoSpec(
            scheme="oras",
            location=hub.RepoSpec._oras_repository_ref(self.location),
            reference=f"sha256:{tag}",
        )

    uuid = types.SimpleNamespace(hex="1234567890abcdef1234567890abcdef")
    monkeypatch.setattr("nomad.hub.OrasRepoSpec.push", fake_oras_push)
    monkeypatch.setattr("nomad.export.uuid4", lambda: uuid)

    output_path = export_models_config(
        config_path,
        tmp_path / "bundle",
        to=ExportTarget.ORAS,
        oras_registry="oras://registry.example.com/scifm",
    )
    rendered = output_path.read_text(encoding="utf-8")
    exported = yaml.safe_load(rendered)

    assert pushes == [
        (
            local_model_dir,
            "registry.example.com/scifm:local-model-1234567890abcdef",
        ),
    ]
    assert [entry["name_or_path"] for entry in exported["fmod_models"]] == [
        "oras://registry.example.com/scifm@sha256:local-model-1234567890abcdef",
        "oras://registry.example.com/scifm@sha256:local-model-1234567890abcdef",
    ]
    assert "# source: models/local-model" in rendered


def test_export_models_config_rejects_duplicate_generated_tool_names(tmp_path: Path):
    config_dir = tmp_path / "config"
    first_model_dir = config_dir / "org-a" / "model"
    second_model_dir = config_dir / "org-b" / "model"
    first_model_dir.mkdir(parents=True)
    second_model_dir.mkdir(parents=True)
    (first_model_dir / "weights.safetensors").write_text("first", encoding="utf-8")
    (second_model_dir / "weights.safetensors").write_text("second", encoding="utf-8")

    config_path = config_dir / "nomad.yml"
    config_path.write_text(
        "\n".join(
            [
                "tool_manager: {}",
                "tools: []",
                "fmod_models:",
                "  - model_class: tests.DummyModel",
                "    name_or_path: org-a/model",
                "  - model_class: tests.DummyModel",
                "    name_or_path: org-b/model",
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate MCP tool name 'model'"):
        export_models_config(
            config_path,
            tmp_path / "bundle",
            to=ExportTarget.DISK,
        )


def test_oras_artifact_tag_uses_shortest_allowed_suffix_when_space_is_tight():
    artifact_name = "m" * (
        nomad_export.ORAS_TAG_MAX_CHARS - nomad_export.ORAS_TAG_SUFFIX_MIN_CHARS - 1
    )
    source_tag = "1234567890abcdef"

    assert (
        nomad_export._oras_artifact_tag(
            artifact_name=artifact_name,
            source_tag=source_tag,
        )
        == f"{artifact_name}-12345678"
    )


def test_oras_artifact_tag_uses_source_tag_for_long_names(caplog):
    artifact_name = "m" * (
        nomad_export.ORAS_TAG_MAX_CHARS - nomad_export.ORAS_TAG_SUFFIX_MIN_CHARS
    )
    source_tag = "1234567890abcdef"

    with caplog.at_level("WARNING", logger="nomad"):
        tag = nomad_export._oras_artifact_tag(
            artifact_name=artifact_name,
            source_tag=source_tag,
        )

    assert tag == source_tag
    assert "too long to append a source tag suffix" in caplog.text


def test_export_models_config_oras_requires_registry(tmp_path: Path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / "nomad.yml"
    config_path.write_text(
        "tool_manager: {}\ntools: []\nfmod_models: []\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="--oras-registry is required"):
        export_models_config(
            config_path,
            tmp_path / "bundle",
            to=ExportTarget.ORAS,
        )


def test_export_model_report_writes_cards_and_linked_descriptions(
    monkeypatch, tmp_path: Path
):
    config_path = tmp_path / "nomad.yml"
    config_path.write_text(
        "\n".join(
            [
                "tool_manager: {}",
                "tools:",
                "  - tool: tests.regular_report_tool",
                "    name: regular-tool",
                "fmod_models:",
                "  - model_class: tests.DummyModel",
                "    name_or_path: org/model-one",
                "    tool_name: model-one",
                "  - model_class: tests.DummyModel",
                "    name_or_path: org/model-two",
                "    tool_name: model-two",
                "",
            ]
        ),
        encoding="utf-8",
    )
    cards = {
        "model-one": "# Model One\n\nFirst card.\n",
        "model-two": "# Model Two\n\nSecond card.\n",
    }

    class DummyTool:
        def __init__(self, name: str):
            self.name = name
            self.description = f"Description for {name}."
            self.source = tmp_path / name

    def fake_build_remote(self, fm_config):
        return DummyTool(fm_config.tool_name)

    def fake_read_model_card(self, tool_name):
        return cards[tool_name]

    def regular_report_tool():
        """Describe the regular tool."""

    monkeypatch.setattr(
        "nomad.config.ToolConfig.fn", property(lambda self: regular_report_tool)
    )

    monkeypatch.setattr("nomad.config.ServerConfig.build_remote", fake_build_remote)
    monkeypatch.setattr(
        "nomad.export.ModelCardLocator.read_model_card", fake_read_model_card
    )

    readme = export_model_report(config_path, tmp_path / "report")

    assert (tmp_path / "report" / "model-one.md").read_text() == cards["model-one"]
    assert (tmp_path / "report" / "model-two.md").read_text() == cards["model-two"]
    assert readme.read_text() == (
        "# Nomad Tool Report\n"
        "\n"
        "## SciFM Tools\n"
        "\n"
        "### [model-one](model-one.md)\n"
        "\n"
        "Description for model-one.\n"
        "\n"
        "### [model-two](model-two.md)\n"
        "\n"
        "Description for model-two.\n"
        "\n"
        "## Other Tools\n"
        "\n"
        "### regular-tool\n"
        "\n"
        "Describe the regular tool.\n"
    )
