# Plan

  Using plan skill to outline a canonical fallback for ambiguous vertex support with transitive-chain handling and order-independent decisions.

  ## Requirements
  ## Scope

  - In: fallback pipeline for ambiguous support sets; candidate expansion; triplet enumeration; clustering and canonicalization; optional KNN radius query.
  - Out: global algorithm rewrite; changing how base Voronoi clipping works.

  ## Files and entry points

  - src/geometry/gpu_voronoi/cell_builder.rs (support key computation and fallback trigger)
  - src/geometry/gpu_voronoi/mod.rs (plumbing/config for fallback, stats)
  - src/geometry/gpu_voronoi/cube_grid.rs or KNN provider (optional radius/ball query helpers)
  - src/geometry/gpu_voronoi/tests.rs (repro and validation)

  ## Data model / API changes

  - Optional: a small fallback config struct (thresholds, max candidates, enable/disable).
  - Optional: KNN provider addition for “within angular radius” queries; otherwise use full scan.

  ## Action items

  [ ] Define fallback trigger conditions (e.g., support size > 3, support_lo != support_hi, near-eps deltas).
  [ ] Implement candidate expansion strategy (full generator scan as baseline; add optional KNN radius query with completeness check).
  [ ] Enumerate candidate vertices by triplet intersection in f64; filter by half-space constraints.
  [ ] Build transitive clusters by angular distance (union-find); compute cluster reps (normalized average or best-fit).
  [ ] Decide canonical vertex count: 1 cluster with small diameter => single vertex; otherwise multiple (no merge or split support keys).
  [ ] Integrate fallback into support key construction; keep existing fast path for non-ambiguous cases.
  [ ] Add diagnostics (ambiguous count, fallback hits, worst cluster diameter).
  [ ] Add/adjust tests to reproduce the boundary issue and validate stability.

  ## Testing and validation

  - Run existing strict validation tests (e.g., cargo test --release strict_large_counts -- --ignored --nocapture).
  - Add a targeted regression test or fixture reproducing the transitive-chain case.
  - Compare before/after orphan edge counts on known problematic seeds.

  ## Risks and edge cases

  - Fallback is O(m^3) in candidate size; must be gated and capped.
  - Incomplete candidate set if radius query is too small (mitigate with completeness check or full scan).
  - Numeric noise in half-space filtering near the boundary (need tolerant eps).

  ## Open questions

  - What thresholds should define “ambiguous” (eps band, support_lo/hi gap, cluster diameter)?
  - Prefer full scan for correctness first, then add KNN radius query later, or implement KNN radius now?