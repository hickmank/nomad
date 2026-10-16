# 09: Tests and Lint

Before you change Nomad itself, learn the checks that CI runs. They are
defined in `Justfile` and `.github/workflows/ci.yml`.

## Run the Checks

```shell
# Lint: formatting and static checks (prek runs the hooks in .pre-commit-config.yaml)
uv run --group dev just lint

# Full test suite (about 90 seconds on a laptop; CI adds -n auto)
uv run --group dev just test

# Faster: run the suite directly, or run one file, test, or keyword
./.venv/bin/python -m pytest -q
./.venv/bin/python -m pytest -q test/test_config.py
./.venv/bin/python -m pytest -q test/gateway/test_gateway.py -k allowlist

# Compatibility check against an older huggingface-hub
uv run --group dev just test-compat
```

A clean run currently gives `384 passed, 1 skipped`.

## Tour of the Tests

| File | What it covers |
| --- | --- |
| `test/test_config.py` | Config parsing, tool naming, and environment placeholders |
| `test/test_hub.py` | Model source resolution and caching (`hf://`, `git+`, `oras://`, local) |
| `test/test_torch_tool_manager_basic.py`, `test/test_tool_manager.py` | Batching, scheduling, and eviction (`gpu` marker for real-GPU tests) |
| `test/test_tool_search.py` | `search_tools` ranking |
| `test/test_well_format.py` | Tensor and WellFormat encoding and decoding |
| `test/test_export.py` | `nomad export` bundles |
| `test/test_docs.py` | Checks the YAML examples in the docs against the config schemas |
| `test/gateway/` | Gateway, sandbox, wrappers, logging, telemetry, and workspace |

Settings: `asyncio_mode = "auto"` (async tests need no decorator), and the
markers are `asyncio` and `gpu` (see `pyproject.toml`).

## Tests That Check the Docs

`test/test_docs.py` checks every YAML block in
`docs/guides/getting-started.md`, `docs/reference/config.md`, and
`docs/guides/model-builder.md` against `ServerConfig` or `GatewayConfig`. If
you change config behavior, update those docs or the test fails.

## Exercises

1. Run `test/test_config.py` with `-v` and pick one test to read closely.
2. Find the test that checks a `${VAR}` that is not set
   (`test/test_env.py`). Compare it with the docstring at
   `src/nomad/common/env.py:27`, which is wrong: it says the result is an
   empty string, but an error is raised. Fixing that docstring is a good
   first contribution.
3. Write a small test in `scratch/test_lesson.py` that builds a
   `ServerConfig` with your `lesson_tools` entry, then run it with
   `./.venv/bin/python -m pytest scratch/test_lesson.py`.
4. Break the formatting of a file on purpose, run `just lint`, and look at
   what Ruff fixes automatically. Revert the change afterwards.

## Docs

```shell
uv run --no-dev --group docs just docs         # build HTML (also syncs container/demo)
uv run --no-dev --group docs just docs --live  # rebuild as you edit
```

Next: [10 Your first model tool](10-first-model-tool.md).
