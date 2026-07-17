# Repository instructions

This file contains only repository-specific constraints that are easy to miss.
Project architecture, goals and roadmaps live in [`docs/README.md`](docs/README.md)
and should not be duplicated here.

## Scope and environment

- This repository contains the Hex3 application. `s2-voronoi` is an external
  dependency maintained in a separate repository.
- Development is performed in WSL2, but the interactive application must be run
  on Windows. WSL2 GPU passthrough does not correctly support the compute shaders
  used by the particle system.
- Use WSL2 for compilation, tests, formatting and static checks. Validate actual
  rendering, compute shaders and representative GPU performance on Windows.

## Commands

```bash
cargo build
cargo build --release
cargo test
cargo clippy
cargo fmt
```

The quarantined landform/orogen laboratory is excluded from the default product
build. Validate it explicitly when changing that subsystem:

```bash
cargo check --features research-landscape --all-targets
cargo test --features research-landscape
```

From a Windows terminal:

```powershell
cargo run --release --bin hex3
```

## Documentation authority

- Start at [`docs/README.md`](docs/README.md).
- Code and tests are authoritative for current behavior; canonical docs describe
  the accepted product path and may be corrected or fundamentally revised.
- Implemented, selectable, experimental, promoted and default are distinct
  statuses. Do not describe every available model as active product behavior.
- Files under `docs/archive/`, `docs/research/`, `docs/generated/` and dated
  audits are evidence or history, not current architecture.
- Keep physical/semantic state separate from renderer-only exaggeration and
  styling. Use the validation and presentation policies linked from the docs
  index when judging changes.
