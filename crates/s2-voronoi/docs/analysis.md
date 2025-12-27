# Precision fallback notes

Goal: keep the `f64` fast path, but never panic, by separating cell *construction* (topology) from
cell *predicates* (clipping decisions + certification) and making only the predicates adaptive.

## Prefer “filtered predicates” over “rebuild everything at higher precision”

Most failures are fundamentally “the sign / ordering is numerically uncertain”. You don’t need high
precision everywhere—just a way to decide a few comparisons robustly.

Pattern (CGAL-style):

- Compute predicate in `f64` + a conservative error bound (or interval).
- If `|value| > bound`: decision is certain.
- Else: recompute predicate with a slower exact method (or higher precision) and decide.

This keeps most cells on the current hot path.

## Make key predicates polynomial (avoid `normalize()` in anything you might certify)

If you want an “exact” fallback, `normalize()` is awkward (involves `sqrt`). The good news is that
many predicates don’t need it.

Two tricks:

- For bisector planes, keep an unnormalized normal `n = g - h`. The half-space test
  `dot(x, g - h) >= 0` is scale-invariant.
- Don’t normalize vertex directions for predicates either. If `v = cross(n1, n2)`, comparisons like
  `dot(unit(v), a) >= dot(unit(v), b)` reduce to `dot(v, a - b) >= 0` (normalization cancels).

This turns certification into signs of dot/cross/determinants (pure `+ - *`), which are ideal for
adaptive exact fallback.

## Determinants: how this ties into clipping + certification

### Half-space clipping on S²

On the unit sphere, “closer to generator `g` than to neighbor `h`” is:

- `dot(x, g) >= dot(x, h)`
- `dot(x, g - h) >= 0`

So each neighbor `h` gives a great-circle bisector plane through the origin with (unnormalized)
normal `n_h = g - h`, and the cell is the intersection of those spherical half-spaces.

### Vertex tests become determinant sign tests

Let `n_a = g - a` and `n_b = g - b` be two plane normals. The intersection direction is
`v_dir = cross(n_a, n_b)`. A vertex “inside/outside plane `n_c`” test is:

- `dot(unit(v_dir), n_c) >= 0`

But `unit(v_dir)` only scales by a positive factor, so the sign is the same as:

- `dot(v_dir, n_c) = dot(cross(n_a, n_b), n_c) = det(n_a, n_b, n_c)`

So “which side?” at a vertex is the sign of a 3×3 determinant.

### Certification “gap” is the same predicate family

In `F64CellBuilder::to_vertex_data_into()`, you compute a “gap”:

- `gap = dot(v, g) - dot(v, c) = dot(v, g - c)`

If `v` is the vertex direction from planes `n_a = g - a`, `n_b = g - b`, then the sign of `gap` is
the sign of:

- `dot(cross(n_a, n_b), g - c) = det(n_a, n_b, g - c)`

Interpretation (after full-scan / sufficient-neighbor guarantees):

- `det(...) > 0`: `c` is definitely not in the support set.
- `det(...) = 0`: true degeneracy; include `c` in the support set.
- `det(...) < 0`: contradiction with the already-clipped plane set. This should be treated as an
  invariant violation (or evidence that termination/candidate assumptions are unsound), not a
  request for more neighbors from certification.

## Sketch: predicate kernel API

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sign {
    Neg,
    Zero,
    Pos,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PredResult {
    Certain(Sign),
    Uncertain,
}

pub trait PredKernel {
    fn det3_sign(&self, a: glam::DVec3, b: glam::DVec3, c: glam::DVec3) -> PredResult;

    #[inline]
    fn dot_cross_sign(&self, u: glam::DVec3, v: glam::DVec3, w: glam::DVec3) -> PredResult {
        // det(u, v, w) = dot(cross(u, v), w)
        self.det3_sign(u, v, w)
    }
}
```

### Filtered `f64` determinant (fast path)

```rust
#[inline]
fn det3_f64(u: glam::DVec3, v: glam::DVec3, w: glam::DVec3) -> f64 {
    let (ux, uy, uz) = (u.x, u.y, u.z);
    let (vx, vy, vz) = (v.x, v.y, v.z);
    let (wx, wy, wz) = (w.x, w.y, w.z);

    ux * (vy * wz - vz * wy) - uy * (vx * wz - vz * wx) + uz * (vx * wy - vy * wx)
}

#[inline]
fn det3_err_bound(u: glam::DVec3, v: glam::DVec3, w: glam::DVec3) -> f64 {
    // Conservative bound; can be tightened later.
    let au = glam::DVec3::new(u.x.abs(), u.y.abs(), u.z.abs());
    let av = glam::DVec3::new(v.x.abs(), v.y.abs(), v.z.abs());
    let aw = glam::DVec3::new(w.x.abs(), w.y.abs(), w.z.abs());

    let m = au.x * (av.y * aw.z + av.z * aw.y)
        + au.y * (av.x * aw.z + av.z * aw.x)
        + au.z * (av.x * aw.y + av.y * aw.x);

    let k = 64.0;
    k * f64::EPSILON * m
}

pub struct F64FilteredKernel;

impl PredKernel for F64FilteredKernel {
    fn det3_sign(&self, u: glam::DVec3, v: glam::DVec3, w: glam::DVec3) -> PredResult {
        let d = det3_f64(u, v, w);
        if !d.is_finite() {
            return PredResult::Uncertain;
        }

        let b = det3_err_bound(u, v, w);
        if d > b {
            PredResult::Certain(Sign::Pos)
        } else if d < -b {
            PredResult::Certain(Sign::Neg)
        } else {
            PredResult::Uncertain
        }
    }
}
```

### Using determinant signs for support certification

```rust
fn certify_vertex_support(
    kernel: &impl PredKernel,
    g: glam::DVec3,
    a: glam::DVec3,
    b: glam::DVec3,
    candidates: impl Iterator<Item = (u32, glam::DVec3)>, // (idx, position)
) -> Result<Vec<u32>, ()> {
    let n_a = g - a;
    let n_b = g - b;

    let mut support = Vec::new();
    for (c_idx, c_pos) in candidates {
        let w = g - c_pos;
        match kernel.dot_cross_sign(n_a, n_b, w) {
            PredResult::Certain(Sign::Pos) => {} // excluded
            PredResult::Certain(Sign::Zero) => support.push(c_idx), // truly in support set
            PredResult::Certain(Sign::Neg) => return Err(()), // invariant violation
            PredResult::Uncertain => return Err(()), // retry with higher precision
        }
    }

    Ok(support)
}
```

## Multi-tier escalation without duplicating the builder

Implement 2–3 precision tiers only for predicates:

- Tier A: filtered `f64` + error bound.
- Tier B: “double-double” (Dekker / `TwoSum` / `TwoProd`) determinant sign.
- Tier C: exact sign (Shewchuk-style expansions, or exact rationals from float bit patterns).

Cell-level ladder:

1. Retry uncertain predicates with a higher tier (don’t rebuild the cell).
2. If still uncertifiable: return structured diagnostics (and optionally use qhull when enabled).

Neighbor acquisition is governed by termination + kNN schedule, not by certification.

 Below is a concrete “shape” you can implement (modules + types + call flow) that matches docs/
  analysis.md and keeps you from duplicating the same per-vertex/per-plane loops in 3 places.

  ## Modules / Responsibilities

  - src/knn_clipping/cell_builder.rs
      - Owns topology + stored geometry for a single cell (planes, vertex cycle, indices).
      - Exposes a read-only CellView adapter (no certification/termination policy here).
  - src/knn_clipping/predicates.rs
      - Defines PredKernel + Tier ladder + determinant sign helpers (filtered f64 → dd → exact-ish).
      - No knowledge of “Voronoi”, just robust sign tests.
  - src/knn_clipping/termination.rs
      - Computes the “unseen dot threshold” (like what we refactored) and returns
  NeighborCompleteness.
  - src/knn_clipping/certify.rs
      - Given a CellView, NeighborSet, and a PredKernel, computes VertexKeys + support sets, or
  returns a structured failure telling the caller what to do next.

  ## Core shared “view” (prevents duplicate loops)

  pub trait CellView {
      fn generator_index(&self) -> u32;
      fn generator(&self) -> glam::DVec3;

      fn plane_count(&self) -> usize;
      fn plane_neighbor_index(&self, pi: usize) -> u32;

      // IMPORTANT: for predicate friendliness this should be UNNORMALIZED:
      // n_pi = g - neighbor_pos
      fn plane_normal_unnorm(&self, pi: usize) -> glam::DVec3;

      fn vertex_count(&self) -> usize;
      fn vertex_def_planes(&self, vi: usize) -> (usize, usize);

      // For output only (can be normalized); predicates should prefer det on normals.
      fn vertex_pos_unit(&self, vi: usize) -> glam::DVec3;
  }

  Then add one helper trait (or free functions) that both termination/certification call:

  pub struct VertexContext {
      pub vi: usize,
      pub def_a: usize,
      pub def_b: usize,
      pub n_a: DVec3,
      pub n_b: DVec3,
      pub v_dir: DVec3,        // cross(n_a, n_b)
      pub cond: f64,           // |v_dir|
  }

  pub fn iter_vertex_context(cell: &impl CellView) -> impl Iterator<Item = VertexContext> + '_ { ... }

  This is the big “no-duplication” win: every place that needs conditioning, v_dir, defining planes,
  etc. gets it from one iterator.

  ## Predicate kernel + tiers (from docs/analysis.md)

  #[derive(Clone, Copy, Debug, PartialEq, Eq)]
  pub enum Sign { Neg, Zero, Pos }

  #[derive(Clone, Copy, Debug, PartialEq, Eq)]
  pub enum PredResult { Certain(Sign), Uncertain }

  pub trait PredKernel {
      fn det3_sign(&self, a: DVec3, b: DVec3, c: DVec3) -> PredResult;
  }

  #[derive(Clone, Copy, Debug, PartialEq, Eq)]
  pub enum PredTier { FilteredF64, DoubleDouble, Exact }

  pub struct KernelLadder { /* owns tier impls */ }

  impl KernelLadder {
      pub fn tiers(&self) -> impl Iterator<Item = (PredTier, &dyn PredKernel)> { ... }
  }

  ## Termination API (explicit contract)

  #[derive(Clone, Copy, Debug, PartialEq, Eq)]
  pub enum NeighborCompleteness {
      Incomplete,
      Complete,
      NotApplicable, // e.g. too few vertices or too ill-conditioned
  }

  pub struct TerminationBound {
      pub unseen_dot_threshold: f64, // if max_unseen_dot < this => complete
      pub tier: PredTier,            // optional: if you decide to use kernels here later
  }

  pub fn termination_bound(cell: &impl CellView) -> Option<TerminationBound>;
  pub fn check_termination(cell: &impl CellView, max_unseen_dot: f32) -> NeighborCompleteness;

  Notes:

  - If you keep termination purely geometric (triangle inequality), it doesn’t need PredKernel.
  - If you later fold in “predicate-based” vertex radius bounds, you can.

  ## Certification output and failure modes (actionable)

  You want certify to tell the caller what to do next.

  pub struct CertifiedCellVertices {
      pub vertices: Vec<(VertexKey, glam::Vec3)>,
      pub support_data_appended: std::ops::Range<u32>, // or just let caller manage Vec<u32>
  }

  pub enum CertifyError {
      NeedMorePrecision,  // Uncertain predicate
      InvariantViolation, // topology/predicate inconsistency (should be rare; indicates bug)
  }

  pub struct CertifyContext<'a> {
      pub completion: NeighborCompleteness,
      pub max_unseen_dot_bound: f32, // optional: for debugging/diagnostics
      pub candidate_planes: std::ops::Range<usize>, // typically 0..plane_count
      pub support_write: &'a mut Vec<u32>,
  }

  pub fn certify_vertices(
      cell: &impl CellView,
      kernel: &dyn PredKernel,
      ctx: CertifyContext<'_>,
  ) -> Result<CertifiedCellVertices, CertifyError>;

  ### What certify_vertices actually checks (determinant-based)

  For each vertex context (n_a, n_b):

  - For each plane c (non-defining):
      - Compute s = det(n_a, n_b, n_c) where n_c = g - c.
      - Interpret:
          - Pos: c excluded at this vertex direction (good).
          - Zero: true degeneracy => include c in support set.
          - Neg: contradiction => InvariantViolation (the current plane set is inconsistent).
          - Uncertain: NeedMorePrecision (retry with higher tier).
  - Key formation:
      - If support set size == 3 => VertexKey::Triplet(sorted)
      - Else => VertexKey::Support{start,len} appended to support_write

  This uses the same sign test family described in docs/analysis.md:33.

  ## Driving it from live_dedup.rs (ladder, no panics)

  High-level loop per cell (pseudo-structure, not exact code):

  // Phase A: build topology incrementally
  builder.reset(i, points[i]);
  for neighbor in knn_stream {
      builder.clip(...)
      update worst_cos
      if termination.should_check(...) {
          if check_termination(builder.view(), worst_cos) == Complete {
              break;
          }
      }
  }
  // If not complete by schedule limit -> do full scan fallback.

  for (tier, kernel) in ladder.tiers() {
      match certify_vertices(builder.view(), kernel, ctx.with_completion(completion)) {
          Ok(verts) => { use verts; break; }
          Err(CertifyError::NeedMorePrecision) => continue, // next tier
          Err(CertifyError::InvariantViolation) => { /* diagnostic / fallback */ }
      }
  }

  Key point: certification only asks for more precision or reports an invariant violation.

  ## Naming suggestions (so it stays readable)

  - NeighborCompleteness (termination result)
  - TerminationBound (explicit threshold type)
  - CertifyError::{NeedMorePrecision, InvariantViolation}
  - KernelLadder + PredTier
  - CellView + VertexContext + iter_vertex_context

  ## Where duplication disappears

  - Only CellView knows how to get n = g - c, vertex defining planes, conditioning, etc.
  - Only predicates.rs knows how to compute robust det3_sign.
  - Only certify.rs knows how to turn sign outcomes into support sets + keys.
  - Only termination.rs knows the spherical triangle-inequality bound (and optionally uses the shared
  conditioning summaries).

  If you want, I can tailor this sketch to your exact current data layout (e.g. reuse
  neighbor_positions vs switching to unnormalized normals) and show how the “support cutoff” concept
  maps into determinant-space without reintroducing normalize().
