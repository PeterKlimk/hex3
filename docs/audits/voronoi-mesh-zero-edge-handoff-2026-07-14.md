# Voronoi substrate A/B and zero-angle edge handoff

**Date:** 2026-07-14

**Verdict:** resolved for the registered Hex3 witness by upstream revision
`e8804a639ea3c989e1ce9ea44b4c66c5f2d7e060`; unchanged physical G0 now passes

**Contract:** [Modern Voronoi backend controlled upgrade](../research/voronoi-mesh-upgrade-ab-2026-07-14.md)

**Audience:** Hex3 and the separate `voronoi-mesh` implementation agent

**Resolution:** Hex3 revision `2f35b80` pins the corrected upstream output-
resolution path, independently rejects any reported or validated remaining
zero geometry, and passes the unchanged 250k product observation. See the
[completion audit](landform-product-g0s0-completion-2026-07-15.md). The report
below is retained as the pre-fix witness and design handoff.

## Executive result

Hex3's old dependency revision
`8ee131ca0b4415aa1638d02fc8542623a7c20eca` was replaced experimentally by
`voronoi-mesh` revision
`da99a8a6384f2fb641c0d3389a6787dec78fc5b4`. The candidate is integrated at
Hex3 revision `5fd4857`; backend failure provenance was added at `6acae71` and
the exact natural-edge witness at `bad52ae`.

The integration itself is materially better:

- Hex3 consumes the effective post-weld partition;
- upstream validation, reconciliation and repair outcomes remain observable;
- non-strict, residual, perturbed and incomplete adjacency outputs are rejected;
- `NO_NEIGHBOR` entries are no longer dropped;
- nearest-generator orphan links are no longer invented;
- exact adjacency, shared-edge, determinism and weld regressions pass;
- fine-cache version 14 prevents old geometry from bypassing the candidate.

However, the frozen seed-12345 250k product observation still stops before S0.
The modern backend returns a topology-clean mesh, but fine cell 57852 contains
an edge between two distinct vertex IDs stored at exactly the same f32
position. Hex3's unchanged physical G0 correctly rejects it. The candidate
therefore has **not** repaired the physical control-volume prerequisite and is
not promoted on that claim. It is retained on the feature branch as an
integration scaffold for an upstream correction; no morphology was observed.

## Reproduction

Environment and frozen product controls:

- WSL2 CPU, release build;
- Hex3 seed 12345;
- 100,000 requested coarse cells, convex-hull coarse backend;
- 250,000 fine cap, product adaptive sampler and clipped-Voronoi backend;
- legacy product orogen model, Stage 4, fine cache disabled;
- unchanged G0 geometry and tolerances;
- no renderer or relief scale involved.

Commands:

```bash
cargo build --release --bin landform_baseline
/usr/bin/time -v timeout 20m target/release/landform_baseline \
  artifacts/landforms/seed-12345-product-g0s0-250k-v0.json
```

Failure:

```text
fine G0 adaptation failed:
InvalidSphericalCellArea {
    cell: 57852,
    edge: Some(7),
    reason: DegenerateEdge,
}
```

The first clean run at Hex3 `5fd4857` took 29.65 seconds wall time and
537,208 KiB maximum RSS. A second provenance run at `6acae71` took 44.93
seconds and 540,624 KiB; scheduler/context-switch variation and added failure
diagnostics make this pair insufficient for a backend performance claim. No
artifact was published because G0 failed.

For comparison, the old backend failed the same frozen observation at fine
cell 199413 edge 6. The changed cell identity shows that the modern algorithms
changed the defect tail; it does not establish that the underlying physical
representation problem was removed.

## Backend report at failure

```text
original_points:                         256847
effective_points:                        256847
merged_points:                                0
weld_threshold:                    1.3486991e-6 chord
preferred_strictly_valid:                   true
preferred_cells:                          256847
preferred_vertices:                       513690
preferred_undirected_edges:               770535
pre_repair_edge_mismatch_count:                0
repair_attempted:                           false
repair_accepted:                            false
post_repair_unpaired_edge_count:                0
native_missing_neighbor_entries:                0
degeneracy_perturbation_applied:            false
```

This rules out four tempting but incorrect explanations for this occurrence:

- no input generator was welded, so public/effective cell aliasing is not the
  cause;
- no upstream mismatch or repair path fired;
- native adjacency is complete and reciprocal;
- Hex3 did not drop a face or synthesize connectivity.

## Exact natural witness

The failed directed edge is cell 57852 edge 7:

```text
cell cycle:
[186377, 186356, 186357, 186385, 186661, 186641, 186640, 186655]

directed edge: 186655 -> 186377
native edge-aligned neighbor: 57857
```

The reciprocal owner is present and topologically correct:

```text
cell 57857 cycle:
[186391, 186378, 186377, 186655, 186654, 186668]

directed edge: 186377 -> 186655
```

The two endpoint identities are distinct, but their stored positions are
bit-identical:

```text
vertex IDs: 186655, 186377
f32 xyz: [0.5441961288452148, 0.6795159578323364, 0.4920453131198883]
f32 bits: [0x3f0b5070, 0x3f2df4c2, 0x3efbed5d]
other vertex IDs with those exact bits: [186377, 186655]

normalized f64 xyz:
[0.5441961414264408, 0.6795159735420105, 0.4920453244954438]
f64 chord length: 0
f64 cross norm:   0
f64 dot:          0.9999999999999998
```

Owner generators:

```text
cell 57852:
  f32 [0.5443676710128784, 0.6787402629852295, 0.4929254353046417]
  bits [0x3f0b5bae, 0x3f2dc1ec, 0x3efc60b9]

cell 57857:
  f32 [0.5445437431335449, 0.6799840927124023, 0.49101296067237854]
  bits [0x3f0b6738, 0x3f2e1370, 0x3efb660d]
```

The owner adjacency lists are also ordinary and reciprocal:

```text
cell 57852: [55879, 55871, 55884, 57860, 57847, 57839, 57844, 57857]
cell 57857: [55887, 55879, 57852, 57844, 57849, 57869]
```

Each endpoint has the expected three incident cells. Their intersection is the
two edge owners; their union identifies the four-generator Delaunay core:

```text
vertex 186655 incidents: [57844, 57852, 57857]
vertex 186377 incidents: [55879, 57852, 57857]
four-cell union:         [55879, 57844, 57852, 57857]

cell 55879 generator:
  f32 [0.5433253049850464, 0.6803092956542969, 0.4919114410877228]
  bits [0x3f0b175e, 0x3f2e28c0, 0x3efbdbd1]

cell 57844 generator:
  f32 [0.5451860427856445, 0.6789295077323914, 0.4917590022087097]
  bits [0x3f0b9150, 0x3f2dce53, 0x3efbc7d6]
```

`RAYON_NUM_THREADS` was unset; the WSL process reported 12-way available
parallelism.

This is not a caller-selected “tiny edge” threshold. In the returned f32
embedding the edge has exactly zero physical extent. The topology describes a
two-owner logical edge, but the stored spherical subdivision maps both
endpoints to one direction. A consumer cannot assign it positive face length,
area-fan orientation or a finite-volume flux geometry.

## Upstream contract gap

At both the pinned candidate and the inspected local upstream head `4619d22`,
`ValidationReport::is_strictly_valid()` secures combinatorics and representation
invariants such as Euler closure, incidence, reciprocal orientation, sphere
membership, self loops and antipodal edges. It does not reject distinct vertex
IDs on the same positive ray. Current correctness prose allows
resolution-floor epsilon edges to be retained or collapsed, which is sensible
for **positive** representable edges but currently also permits a returned
zero-extent edge.

The narrow contract recommendation is:

> Every returned boundary edge has two distinct representable directions; a
> valid result may retain any positive edge, regardless of size, but an exact
> same-ray edge must be collapsed safely or cause a defined error.

This preserves the library's appropriate non-promise about minimum geometric
quality, exact combinatorics and exact vertex positions. It only closes the
gap between “embedded spherical subdivision” and its stored representation.

## Ranked upstream handoff

### P0: make exact angular collapse invalid

Add a stable strict-validation counter for unique undirected edges whose IDs
differ but whose stored endpoint directions are identical. Use a direction
predicate such as f64 `cross_norm == 0 && dot > 0`, or equivalently a zero
`atan2(cross_norm, dot)` result. Do not use `acos(dot)`: it rounds many small
positive angles to zero. Exact antipodes remain owned by the existing
antipodal/owner-plane logic.

The validator-only insertion points identified in the current upstream tree
are:

- `src/validation.rs::ValidationReport` and `invariant_issues()`;
- the unique-edge loop in `validate_impl`;
- `verify_sphere_fast` and `verify_sphere_effective_strict`, so report,
  opt-in return verification and repair acceptance share one invariant.

Validator-only work makes `compute_with_report` observable and rejectable by
Hex3, but it is not a complete plain-API guarantee if ordinary `compute` can
still return the mesh.

### P0: return a repaired/canonical mesh or a defined error

The physically natural interpretation is to contract only exact
representationally collapsed **boundary edges**, rewrite all incident cell
cycles transactionally and accept the result only after complete strict
revalidation. This converts an arbitrary zero-length Delaunay diagonal into a
higher-degree Voronoi vertex without moving geometry. Use deterministic
lowest-ID representatives and expose a contraction count or repair kind.

If exact contraction degenerates a cell or cannot preserve the manifold,
return a defined representation/computation error or escalate the owner
neighborhood through Local3d. Do not jitter returned vertices, broaden
generator welding, or silently drop the face.

Relevant current flow:

```text
live_dedup/assemble
  -> knn_clipping/compute::reconcile_edges
  -> edge_reconcile::reconcile_unresolved_edges
  -> summarize_topology
  -> maybe_repair_effective
  -> validation::verify_sphere_effective_strict
```

Existing merge machinery lives in
`edge_reconcile::{collect_merges, bound_merge_components,
apply_merges_in_place}`. Simply synthesizing a normal edge-mismatch record is
unlikely to work: both cells already agree on both endpoint identities, so the
current reconciliation logic sees a topologically paired edge and proposes no
union.

The smaller production alternative is to detect the collapsed edge during the
mandatory topology summary, send its owner pair through Local3d, and make the
plain return signal fail loudly if it remains. That is less targeted but still
valid-or-error.

### P1: keep positive-small geometry as telemetry

Do not add a global epsilon to strict validity. A separate optional geometry
quality report may expose minimum positive chord/angle, caller-selected
near-resolution counts, signed-area/orientation evidence and post-store
contractions. Useful physical resolution depends on cell density and consumer;
exact zero in the stored representation does not.

### P2: consider f64 output only if preservation is required

An f64 vertex representation or sidecar may preserve features lost on f32
storage, but it is a larger memory/API decision and is not required for the
default valid-or-error contract.

## Suggested upstream regressions

1. A topologically clean split-octahedron with one north-pole identity split
   into two IDs at the same position: current strict validation passes; the new
   report must find exactly one collapsed edge.
2. Same positive ray with differently scaled near-unit endpoints also rejects.
3. A one-ULP non-collinear positive edge passes, proving there is no hidden
   epsilon or `acos` floor.
4. A cocircular/tie fixture contracts to one degree-4+ vertex and remains
   manifold with Euler characteristic 2 and reciprocal edge-aligned adjacency.
5. An unsafe contraction returns a defined error, never panic or invalid
   success.
6. Exercise plain `compute`, `compute_with_report`, effective/welded output,
   serde roundtrip and embedded-sphere paths.
7. Retain this Hex3 natural witness as cross-repository integration evidence;
   after an upstream candidate is pinned, repeat the unchanged full 250k G0.
8. Existing adversarial and large robustness campaigns assert that strict
   validity implies zero collapsed edges while recording, not rejecting,
   positive-small edges.
9. Benchmark the clean fast path; the expected added work is one linear edge
   scan with no cold repair allocation.

The synthetic split-octahedron should be the small unit regression. The full
natural 256,847-site case is the authoritative integration regression. If an
upstream-only natural fixture is required, export the canonical generator
array—and preferably the raw pre-canonicalization sampled sites—as ordered
`[u32; 3]` f32 bits plus a hash before attempting reduction. Preserve the four
generators above, seed with two adjacency rings, and add a sparse full-sphere
support shell before deterministic chunk deletion. The minimization predicate
must be semantic (successful strict-topology output containing any distinct-ID
same-ray edge), not tied to these global indices. A local angular subset may
not reproduce because grid resolution and global assembly influence the
defect; do not claim a minimized natural fixture until it reproduces the same
exact collapse across repeated and selected Rayon thread counts.

## Pre-fix Hex3 disposition and unblock condition

The modern adapter work remains useful independently: it removes a fabricated
graph repair, secures effective-cell identity and gives upstream repair
provenance. It remains an **implemented candidate**, not a promoted proof of
physical G0. Do not revert to hiding old defects, and do not resume terrain or
morphology interpretation yet.

The next Hex3 action is intentionally small: receive an exact upstream commit,
pin it, run focused adapter and full regressions, then repeat the unchanged
seed-12345 command. Fine G0 completion is the unblock condition. Only then may
the already-frozen ancestry observation proceed to S0.
