# Evaluation tooling inventory

Status: code synthesis as of 2026-07-12.

## Reusable foundations

### Generation and provenance

The main binary supports deterministic seed, coarse cells, stage, fine scale and
cap, backend/model selection and JSON export. `RunManifest` records revision and
dirty state, units, seed, backend, cell counts, model/configuration, stages and
fine-cache outcome. JSON/gzip exports embed the manifest and detailed per-cell,
plate, tectonic, climate, hydrology and stage-separated fine fields.

This is a strong basis for corpus artifacts. Important current omissions are an
export schema version, corpus/run identity, baseline-parent relation, experiment
label, serialized timings/resource use and failure records.

The fine cache has content identity and outcome provenance. Cache reuse is also
important for repeatability because parallel fine-mesh welding is documented as
non-deterministic when a base is freshly rebuilt.

### Numerical comparison

- `tectonic_scorecard` is the only built-in multi-seed/multi-resolution runner.
  It is deliberately coarse-only and human-table oriented.
- `diagnose` can combine several audits after one expensive generation but runs
  only one world and emits human-readable text.
- `scripts/resolution_compare.py` reads JSON/gzip, computes area-weighted
  metrics, common-grid field correlation/nRMSE and resolution drift relative to
  seed spread. Its metric set is narrow and excludes most objects, causal
  relationships, topology, semantics, runtime and conservation.
- `compare_orogen_exports` compares two same-seed stage-4 exports, but validates
  only seed equality and is specialized to orogen fields.
- Tiny integration tests protect pipeline sanity and semantic invariants; they
  do not preserve corpus artifacts or characterize ensembles.

### Visual comparison

The headless sweep renderer supports fixed-camera parameter montages and can
reuse a fine base for erosion-only comparisons. It requires a working GPU and
therefore must run on Windows rather than the WSL development environment.
Images encode knob values but have no manifest sidecars or associated numerical
records. Automatically selected highland views bias inspection and are not
stable semantic targets across geography-changing configurations.

CPU-side map and terrain-analysis scripts can create WSL-safe plots from
exports, but they are analysis views rather than product-render evidence.

## Missing corpus harness capabilities

- declarative seeds × resolutions × configurations × stages specification;
- resumable ledger with pending/running/completed/failed states;
- atomic per-run artifact bundles and selective reruns;
- structured metrics from `diagnose` and scorecards;
- metric-registry/schema versions in outputs;
- automatic verification that an A/B differs only on declared axes;
- generic manifest-aware comparisons;
- serialized stage timings, memory/resource data and cache efficiency;
- matched manifest/export/numerical report/log/image bundles;
- cross-seed stable view definitions and image sidecars;
- output retention and migration policy.

The convenience `scripts/analyze` deletes its source export after one run and is
not suitable as a corpus runner.

## Minimal implementation implication

Do not begin by replacing all analysis code. The smallest useful foundation is:

1. a declarative corpus specification with stable run IDs;
2. one atomic artifact directory per run containing manifest, structured metric
   rows, logs/timings and optional export/views;
3. resumable execution and declared-axis comparison checks;
4. adapters that expose existing high-value metrics before adding new ones.

The reference fine resolution should be chosen only after smoke-tier timing and
artifact-size measurement. The eight-million-cell product cap is a guardrail,
not an automatic evaluation budget.

