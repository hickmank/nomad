# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `nomad code-mode-exec` gained the ability to start and use a local nomad server ([#20]).

## [v0.2.1] - 2026-09-10

### Changed

- Enabled [OTEL](https://opentelemetry.io/) logging within the [nomad demo image](ghcr.io/lanl/nomad). Added a docker-compose with a reference observability stack ([#9]).
- Use the operating system trust store for Hugging Face, MCP/FastMCP, ORAS,
  Git, and OpenTelemetry HTTPS connections, ignoring alternate CA overrides
  inherited from the environment ([#10]).

### Fixed

- Fixed the workspace_root to default to CWD when launched with stdio ([#13])
- Script path is checked for existence prior to invocation ([#13])
- Demo container now uses environment variables to configure nomad, simplifying hosting ([#16])

## [v0.2.0] - 2026-07-29

### Added

- Added trusted publishing to PyPI and PDF documentation to GitHub releases ([#8])
- Added the `--report` flag to `nomad export` to generate model cards and a linked README for the configured models and tools ([#3])
- Added Agent Skills documentation and a SciFM-to-Nomad connector skill ([#5],[#6])

### Changed

- Renamed the Python distribution to `nomad-scifm`; the `nomad` import package and CLI are unchanged ([#8])
- Migrated gateway upstream connections and transport configuration to FastMCP ([#4])

### Fixed

- Fixed borked nomad executable path in Nomad Demo Dockerfile ([#2])

## [v0.1.0] - 2026-07-13

Initial Public Release of Nomad

<!-- Versions -->
[unreleased]: https://github.com/lanl/nomad/compare/v0.2.0...HEAD
[v0.2.0]: https://github.com/lanl/nomad/compare/v0.1.0...v0.2.0
[v0.1.0]: https://github.com/lanl/nomad/tree/v0.1.0

<!-- Pull Requests -->
[#2]: https://github.com/lanl/nomad/pull/2
[#3]: https://github.com/lanl/nomad/pull/3
[#4]: https://github.com/lanl/nomad/pull/4
[#5]: https://github.com/lanl/nomad/pull/5
[#6]: https://github.com/lanl/nomad/pull/6
[#8]: https://github.com/lanl/nomad/pull/8
[#9]: https://github.com/lanl/nomad/pull/9
[#10]: https://github.com/lanl/nomad/pull/10
[#20]: https://github.com/lanl/nomad/pull/20
