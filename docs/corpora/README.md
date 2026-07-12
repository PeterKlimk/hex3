# Evaluation corpora

Corpus specifications are declarative JSON inputs to the CPU-only `corpus`
binary. Specifications are reviewed source; generated artifacts are ignored
under `artifacts/`.

Validate and list stable run identities without generation:

```bash
cargo run --bin corpus -- --spec docs/corpora/smoke-v1.json --dry-run
```

Run or resume the smoke tier:

```bash
cargo run --release --bin corpus -- --spec docs/corpora/smoke-v1.json
```

Use `--out PATH` to relocate artifacts and `--force` to replace completed runs.
Without `--force`, completed artifacts are skipped. Failed runs are published as
`<run-id>.failed`, cause a nonzero exit, and are retried later.

## Artifact contract v1

Each successful run is published atomically as:

```text
artifacts/evaluation/<corpus-id>/<run-id>/
  run-spec.json
  manifest.json
  metrics.json
  timings.json
  status.json
```

The corpus root also contains `index.json`, an atomically updated ledger of all
declared run IDs and their pending/completed/failed state. `summary.json`
flattens completed run configuration, metrics and timings for comparison.

`manifest.json` binds the effective world manifest to corpus/run identity and
schema versions. Metrics carry stable IDs plus value, unit, weighting,
aggregation and stage. Timings record wall time by major pipeline stage and
total. Status is written last inside the temporary directory before atomic
publication.

Run IDs hash normalized generation configuration plus corpus/metric schema
versions. Human labels are excluded. The v1 hash is deterministic FNV-1a over
serde JSON: an identity checksum, not a security primitive.

The initial adapter intentionally exposes only a small registered metric set.
Detailed `diagnose` reports are not scraped. Full exports, captured process
logs, memory use and matched visual evidence are not part of artifact v1.

`smoke-v1.json` uses tiny stage-4 worlds for schema, resume and cost testing. It
is not a quality corpus and must not support promotion decisions.

`reference-budget-v1.json` holds product coarse resolution and sweeps one fixed
seed across 250k, 1M and 4M fine-cell caps. Its purpose is to choose an
evaluation budget from runtime and resolution-sensitive metric behavior, not to
judge population quality.

`reference-v1.json` is the standing ten-seed product-baseline corpus. Its 1M
fine cap is justified by the
[reference-budget audit](../audits/reference-budget-2026-07-12.md).

`erosion-core-screen-v1.json` is a deliberately under-resolved, two-seed 250k
screen for cost/value differences among 50/100/200 steps and `n=1`/`n=2`.
The [result](../audits/erosion-core-screen-2026-07-12.md) found large
morphological differences, so none was promoted as an equivalent 1M candidate.
