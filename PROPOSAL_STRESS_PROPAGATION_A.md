# Proposal A: Plate-Constrained Screened Diffusion for Stress Propagation

This proposes replacing `src/geometry/tectonics.rs::propagate_stress` with a plate-constrained **screened diffusion** solve on the Voronoi cell adjacency graph. The intent is to keep linear superposition, produce realistic-looking interior stress fields, and reduce runtime from ~quadratic behavior.

## Background: Current Algorithm (for contrast)

Today, stress is computed as a dense superposition from all boundary cells on the same plate:

- Build `plate_boundaries[plate_id] = { b | |boundary_stress[b]| > eps }`
- For each cell `i` on plate `P`:
  - `stress[i] = Σ_{b ∈ plate_boundaries[P]} boundary_stress[b] * exp(-d_sphere(i,b)/L)`
  - where `d_sphere(i,b) = acos(clamp(dot(pos_i, pos_b)))`, `L = STRESS_DECAY_LENGTH`.

Issues:
- Runtime is roughly `O( N * B_plate )` (often heavy at 20k cells).
- Uses direct great-circle distance between generators, not distance/paths through the plate’s interior.
- Summing over boundary *cells* can make amplitude sensitive to boundary discretization/jaggedness unless normalized.

## Goals

- Preserve **linear superposition** (multiple boundary segments contribute additively).
- Constrain propagation **within a plate** (no cross-plate influence).
- Use a distance notion consistent with the **adjacency graph**.
- Be cheap: ~`O(E * iters)` with `E ~ O(N)`.
- Provide a knob that maps to a meaningful **propagation length**.

## Model: Screened Diffusion / Helmholtz on the Plate Graph

We model the propagated stress field `s` as the solution to a screened diffusion equation on each plate:

> (I − λ L) s = b

Where:
- `s[i]` is the propagated stress at cell `i`
- `b[i]` is boundary forcing (use `boundary_stress[i]`, or a slightly massaged variant, see below)
- `L` is a (possibly weighted) graph Laplacian over the Voronoi adjacency
- `λ` controls smoothing/decay length

Interpretation:
- `L` spreads values to neighbors (diffusion).
- The `I` term “screens” it, preventing unbounded spreading and enforcing decay away from sources.
- This approximates a continuous PDE `(1 − ℓ² ∇²) s = b` on the plate surface, with `ℓ` analogous to a decay length.

### Plate constraint

Solve independently per plate on the induced subgraph:
- Only include edges `(i,j)` where `cell_plate[i] == cell_plate[j]`.
- This guarantees no cross-plate bleeding.

## Discretization Details

### Adjacency and edge weights

Use the existing cell adjacency graph (`plates.adjacency` or similar).

Optional (recommended) weighting:
- Define edge length `w_ij` as the great-circle distance between generator points:
  - `θ_ij = acos(clamp(dot(pos_i, pos_j)))`
- Use `w_ij = 1 / max(θ_ij, eps)` as a conductance (closer neighbors couple more strongly).

Unweighted Laplacian also works and is simpler:
- `L(s)[i] = Σ_{j∈N(i)} (s[i] − s[j])`

### Forcing term `b`

Baseline:
- `b[i] = boundary_stress[i]` (already signed).

If boundary discretization causes amplitude issues, consider normalizing forcing by local boundary density:
- `b[i] = boundary_stress[i] / (deg(i) or local_edge_length_sum)`
but start with the baseline; your existing boundary stress already includes edge-length weighting.

### Mapping `λ` to a length scale

On a roughly regular graph, the screened diffusion kernel decays approximately exponentially with graph distance. A practical approach:
- Keep the user-facing constant `STRESS_DECAY_LENGTH` (in radians) as the desired length `ℓ`.
- Convert to a graph-scale parameter:
  - Let `θ̄` = median (or mean) neighbor distance within the plate.
  - Let `k = ℓ / θ̄` be “number of neighbor steps”.
  - Choose `λ` such that the half-life is around `k` steps.

Empirical starting point (tune visually):
- `λ ≈ (k²) / C`, with `C` in `[4, 16]` depending on weighting and solver. Start with `C = 8`.

This is intentionally pragmatic; exact matching isn’t necessary for good results.

## Solver Strategy

We need to solve a sparse SPD system per plate. Recommended: iterative relaxation (Jacobi or Gauss–Seidel), which is simple and stable for this structure.

### Weighted Jacobi (simple, parallel-friendly)

Let:
- `A = I − λ L`
- For each node `i`, define:
  - `diag[i] = 1 + λ Σ_j w_ij` (for Laplacian defined as `Σ w_ij (s[i] − s[j])`)
  - `offsum = λ Σ_j w_ij * s[j]`
Then one Jacobi update is:

> s_new[i] = ( b[i] + offsum ) / diag[i]

Optionally use damping:
- `s = (1 − ω) * s + ω * s_new`, `ω ∈ (0,1]` (e.g. `ω = 0.7`).

Iterations:
- Fixed small number (e.g. 30–80) is usually enough for “visual plausibility”.
- Stop early if max change < tolerance.

### Gauss–Seidel (often converges faster)

Same formula as Jacobi, but write updated `s[i]` in-place as you sweep nodes. This is slightly order-dependent; per-plate random or BFS ordering can reduce artifacts.

## Complexity

Let `N_p` be cells in a plate, `E_p` edges in its induced subgraph.

- Each iteration is `O(E_p)`.
- Total is `O( (Σ_p E_p) * iters ) ≈ O(N * iters)`.

Compared to the current `O(N * B)`, this should be a large win at high cell counts.

## Pseudocode

Per plate:

1. Collect `nodes = { i | cell_plate[i] == plate }`
2. Build neighbor lists restricted to `nodes` (filter adjacency)
3. Initialize `s[i] = b[i]` or `0`
4. Precompute `diag[i] = 1 + λ Σ_j w_ij`
5. Repeat `iters`:
   - For each `i` in `nodes`:
     - `offsum = λ Σ_j w_ij * s[j]`
     - `candidate = (b[i] + offsum) / diag[i]`
     - `s_new[i] = lerp(s[i], candidate, ω)`
   - Swap / assign `s = s_new`

Global output: concatenate `s` across all plates.

## Expected Visual/Behavioral Changes

- Stress fields become smoother and more “plate-interior coherent” because propagation uses plate connectivity.
- Multiple boundaries naturally superpose without explicitly summing over all boundary cells.
- Reduced sensitivity to jagged boundary discretization.
- Much lower runtime at higher cell counts.

## Why This Has High “Verisimilitude”

This is the discrete analogue of solving a simple elliptic PDE on a surface with boundary forcing. It captures:
- locality,
- superposition,
- attenuation with distance,
- plate-constrained propagation,
without pretending to be a full geophysical model.

