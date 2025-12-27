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
- `det(...) < 0`: `c` beats `g` at that vertex direction (often means missing constraints /
  insufficient neighbors; escalate k / full scan rather than “more precision”).

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
            PredResult::Certain(Sign::Neg) => return Err(()), // contradiction / missing constraints
            PredResult::Uncertain => return Err(()), // escalate kernel tier
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
2. Increase k / extend neighbor schedule (missing a true neighbor can legitimately break clipping).
3. Full scan fallback (already present).
4. If still uncertifiable: return structured diagnostics (and optionally use qhull when enabled).
