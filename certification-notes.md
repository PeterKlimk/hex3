# Voronoi Vertex Certification Approaches

This note summarizes the certification strategies tried so far and the formal
justifications behind each one. The goal is to ensure any cell that touches a
vertex would resolve it the same way in parallel, and to defer regions that are
uncertifiable under floating-point error.

## Terminology

- `theta`: angle between the vertex direction and its owning generator.
- `eps`: angular error bound for a vertex (per-vertex; FP-driven).
- `cluster`: generators within `theta + 2*eps` of the vertex.
- `gap`: require the next-best generator to be outside `theta + 3*eps`.

## 1) Fixed dot-epsilon support set (baseline)

Summary:
- Support set S(v) = { g | max_dot - v·g <= SUPPORT_EPS_ABS }.
- Certification fails on either:
  - `ill_conditioned` intersections, or
  - `gap` if the next-best candidate is within a tiny dot threshold.

Formal justification:
- Uses a constant f32 dot-precision bound (SUPPORT_EPS_ABS).
- Assumes the candidate generator set is complete for the true support set.

## 2) Cluster-by-epsilon with per-vertex eps + epsilon-aware termination

Summary:
- Per-vertex eps = SUPPORT_VERTEX_ANGLE_EPS + drift_angle.
- Cluster threshold computed as cos(theta + 2*eps) in dot space.
- Gap threshold computed as cos(theta + 3*eps) in dot space.
- Termination uses `eps_cell` so candidates are complete for the needed radius.

Formal justification:
- Spherical triangle inequality: if a generator is within `theta + 2*eps`
  of the vertex, it must be within `2*theta + 2*eps` of the cell generator.
- Using `eps_cell` in termination makes `candidates_complete=1` meaningful
  for the epsilon-based cluster test.

## 3) Conditioning-scaled epsilon (unbounded 1/sin(phi))

Summary:
- Scale eps by conditioning of plane intersection:
  `eps_angle ∝ 1 / sin(phi)` where `phi` is angle between plane normals.
- Mark uncertifiable if `eps_angle > eps_cell`.

Formal justification:
- Intersection of two planes is sensitive to normal perturbation with
  condition number ~ 1/sin(phi).

## 4) Formal plane-normal error bound (current)

Summary:
- Compute plane normal error directly:
  - Build f64 normals from the same f32 inputs used in clipping.
  - Compute `plane_err = angle(n_f32, n_f64)` for each plane.
- Use conditioning to project this to vertex direction:
  `eps_angle = SUPPORT_VERTEX_ANGLE_EPS + drift_angle + plane_err / sin(phi)`.
- If `eps_angle > eps_cell`, mark the vertex uncertifiable.

Formal justification:
- Uses the actual measured FP error of plane normals, scaled by the
  conditioning of the intersection.
- Keeps eps FP-driven without unbounded growth.

## References

- Certification logic: `src/geometry/gpu_voronoi/cell_builder.rs`
- Constants: `src/geometry/gpu_voronoi/constants.rs`
- Correlation test: `src/geometry/gpu_voronoi/tests.rs`
- Strict validation: `src/geometry/validation.rs`
