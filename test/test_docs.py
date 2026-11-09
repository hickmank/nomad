from __future__ import annotations

import re
import subprocess
from collections.abc import Iterator
from pathlib import Path

import yaml

from nomad.config import ServerConfig
from nomad.gateway.config import GatewayConfig
from nomad.metrics import metric_definitions, metrics_markdown_table

FENCED_BLOCK_RE = re.compile(
    r"^```(?P<lang>[A-Za-z0-9_-]+)?[^\n]*\n(?P<body>.*?)^```",
    re.MULTILINE | re.DOTALL,
)


def _fenced_blocks(path: Path) -> Iterator[tuple[str, str, int]]:
    text = path.read_text(encoding="utf-8")
    for match in FENCED_BLOCK_RE.finditer(text):
        lang = (match.group("lang") or "").lower()
        start_line = text.count("\n", 0, match.start()) + 1
        yield lang, match.group("body"), start_line


def _preceding_lines(path: Path, line_number: int, *, count: int = 4) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    start = max(0, line_number - count - 1)
    end = max(0, line_number - 1)
    return "\n".join(lines[start:end])


def _yaml_blocks(path: Path) -> Iterator[tuple[dict, int]]:
    for lang, body, start_line in _fenced_blocks(path):
        if lang not in {"yaml", "yml"}:
            continue
        data = yaml.safe_load(body)
        assert isinstance(data, dict), f"{path}:{start_line} is not a mapping"
        yield data, start_line


def _validate_classified_yaml_blocks(path: Path) -> dict[str, int]:
    counts = {"server": 0, "gateway": 0, "external": 0}

    for data, start_line in _yaml_blocks(path):
        if {"fmod_models", "tools", "tool_manager", "search_tool"} & data.keys():
            ServerConfig.model_validate(data)
            counts["server"] += 1
            continue

        if {"servers", "defaults", "middleware"} & data.keys():
            GatewayConfig.model_validate(data)
            counts["gateway"] += 1
            continue

        if "mcp_servers" in data:
            counts["external"] += 1
            continue

        raise AssertionError(
            f"{path}:{start_line} YAML block did not match a documented config schema"
        )

    return counts


def test_getting_started_gateway_yaml_examples_are_valid() -> None:
    docs_path = Path("docs/guides/getting-started.md")
    validated_blocks = 0

    for data, start_line in _yaml_blocks(docs_path):
        if "gateway.y" not in _preceding_lines(docs_path, start_line):
            continue

        assert "servers" in data, f"{docs_path}:{start_line} must use GatewayConfig"
        GatewayConfig.model_validate(data)
        validated_blocks += 1

    assert validated_blocks == 2


def test_reference_config_yaml_examples_are_valid() -> None:
    counts = _validate_classified_yaml_blocks(Path("docs/reference/config.md"))
    assert counts == {"server": 1, "gateway": 2, "external": 0}


def test_model_builder_yaml_examples_are_valid() -> None:
    counts = _validate_classified_yaml_blocks(Path("docs/guides/model-builder.md"))
    assert counts == {"server": 1, "gateway": 1, "external": 1}


def test_metric_docs_table_is_generated_from_definitions() -> None:
    for scope in ("serve", "gateway"):
        table = metrics_markdown_table(scope)
        assert "| Metric | Type | Unit | Description |" in table
        for definition in metric_definitions(scope):
            assert f"`{definition.name}`" in table
            assert f"| {definition.kind} |" in table
            assert f"`{definition.unit}`" in table
            assert definition.description in table


def test_generated_metric_docs_are_gitignored() -> None:
    result = subprocess.run(
        ["git", "check-ignore", "docs/reference/generated/metrics.md"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
