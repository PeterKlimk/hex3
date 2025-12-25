# Conservative bounds for S2-style nearest-neighbor (NN) search over cells

This note describes two bounds for pruning / best-first search over **S2-style cells** (spherical quadrilaterals bounded by **great-circle (geodesic) arcs**).

It focuses on bounding the **minimum spherical distance** from a query point `q` (unit vector) to **any point** in a cell `C`.

---

## Notation

- `q ∈ R³`, `|q| = 1` (query direction on the unit sphere)
- `C ⊂ S²` (an S2 cell / region on the unit sphere)
- For any point `x ∈ S²`, the spherical angle is  
  `angle(q, x) = arccos(q · x)`.

Define:

- `dotMax(C) = max_{x∈C} (q · x)`
- Then the true minimum angle from `q` to the cell is:

  `θ_min(q, C) = arccos(dotMax(C))`

This equivalence is the key trick: **minimizing distance is equivalent to maximizing dot product** on the unit sphere.

Pruning-friendly form (no trig needed):

- Keep current best NN angle `θ_best`, and store `cosBest = cos(θ_best)`.
- If you can compute an **upper bound** `UBdot(C) ≥ dotMax(C)`, then it is safe to prune if:

  `UBdot(C) ≤ cosBest`

(If even the best possible dot in the cell can't beat your current best, the cell can't contain a nearer neighbor.)

---

## How bound (2) differs from “distance to great-circle arc edges”

### Your “distance to arc” idea
You compute something like:

- `d_edge = min distance from q to each great-circle arc edge`
- `d_vertex = min distance from q to each vertex`
- `θ_min = min(d_edge, d_vertex)` (for convex cells, this can be exact)

This is geometrically direct but often costs:
- per-edge projection + clamping-to-arc logic
- trig (`acos`, `asin`, `atan2`) or robust angle computations
- extra normalization / branching

### Bound (2) below is *not* “distance to arcs”
Instead, it computes **the maximum dot** achievable on each edge (and at vertices), then converts (conceptually) to a distance bound.

- It uses dot products and (usually) one `sqrt` per “active” edge.
- It avoids `acos/asin` in the hot loop.
- It yields the same *information* as your arc-distance method (often the exact minimum distance), but via a cheaper quantity (`dotMax`) that is directly comparable to `cosBest`.

---

## (2) Tight bound via `dotMax` on vertices + geodesic edges (often exact on S2 cells)

### Precompute per edge
For each directed edge from vertex `a` to `b` (both unit vectors), precompute:

1) Great-circle plane normal (unit):
- `n = normalize(a × b)`

2) Two “arc gate” vectors (don’t need normalization):
- `g0 = n × a`
- `g1 = b × n`

These gates help test whether the **great-circle maximizer** lies on the **finite arc** from `a` to `b`.

### Query-time evaluation for one edge
Given `q`:

1) Compute signed distance to the edge plane:
- `dn = q · n`

2) Project `q` onto the edge plane:
- `p = q - dn * n`

3) Candidate maximizer on the *infinite* great circle is along `p`.
- If `|p|` is very small, the maximizer is ill-conditioned; just fall back to endpoints.

4) Arc-membership (“gate”) test:
- The projected maximizer lies on the arc if:

  `p · g0 ≥ 0` **and** `p · g1 ≥ 0`

5) If the gate test passes, then the maximum dot along that edge arc is:

- `dotEdge = |p| = sqrt(1 - dn²)`  (since `|q|=|n|=1`)

6) Otherwise:
- `dotEdge = max(q·a, q·b)` (best is at an endpoint)

### Cell bound
Compute:

- `dotV = max_i (q · v_i)` over the 4 cell vertices
- `dotE = max_j (dotEdge_j)` over the 4 edges

Then:

- `UBdot(C) = max(dotV, dotE)`

If computed exactly, `UBdot(C)` is actually `dotMax(C)` (not just an upper bound), so this gives the **exact** `θ_min` (up to floating-point error).

### Why it’s conservative on S2 cells
S2 cells are **spherically convex** regions bounded by 4 geodesic (great-circle) arcs. For such a region:

- A linear functional `x ↦ q·x` achieves its maximum on the boundary.
- The boundary is exactly the union of the 4 geodesic edges.
- On a geodesic edge (a great-circle arc), the maximum of `q·x` is either:
  - the great-circle maximizer (if it lies inside the arc), or
  - an endpoint (vertex).

So checking “(interior maximizer if on arc) else endpoints” per edge, plus vertices, gives `dotMax(C)`.

If you slightly *underestimate* `dotMax(C)` due to numerics, you might prune incorrectly. In practice:
- add a small epsilon slack, e.g. prune only if `UBdot ≤ cosBest - eps`
- and clamp `UBdot` into `[-1, 1]`.

---

## (3) Very cheap hemisphere-violation lower bound (looser, but safe)

### Idea
An S2 cell can be represented as an **intersection of 4 hemispheres**, one per edge plane.

Let each edge define a great-circle plane through the origin with an **inward** unit normal `n_i` such that:

- inside the cell implies `n_i · x ≤ 0` for all `i`  
  (Your sign convention may be flipped; just be consistent.)

Define the hemisphere for constraint `i`:

- `H_i = { x ∈ S² : n_i · x ≤ 0 }`

Then the cell is:

- `C = ⋂_i H_i`   (**intersection of hemispheres**)

### Bound construction
Distance from `q` to a hemisphere `H_i` is:

- `0` if `n_i · q ≤ 0` (q is already inside that hemisphere)
- otherwise the distance to the boundary great circle:

  `d(q, H_i) = arcsin(n_i · q)`  (assuming `n_i` is unit and `n_i·q > 0`)

Define:

- `s_i = max(0, n_i · q)`
- `LB_sin = max_i s_i`
- angle lower bound: `LB_angle = arcsin(LB_sin)`

You can avoid trig by comparing in “sin space”:

- maintain `sinBest = sin(θ_best)`
- prune if `LB_sin > sinBest + eps`

### Why it’s conservative (the key set argument)
Because the cell is an **intersection**:

- `C = ⋂_i H_i`

we have `C ⊆ H_i` for every `i`. For any sets `A ⊆ B`:

- `dist(q, A) ≥ dist(q, B)`  (it can only get farther when the target set shrinks)

Therefore:

- `dist(q, C) ≥ dist(q, H_i)` for every `i`
- taking the maximum over `i` preserves the inequality:

  `dist(q, C) ≥ max_i dist(q, H_i)`

So `max_i dist(q, H_i)` is a **conservative lower bound** on the true distance to the cell.

### Tradeoff vs (2)
- (3) is *very* cheap: just 4 dot products + a max.
- It can be loose because it ignores that you must satisfy **all** halfspaces at once.
- (2) is tighter (often exact) but costs more math per edge.

---

## Practical recipe: staged pruning
A common fast strategy:

1) Use a very cheap bound first (caps or hemisphere-violation (3)).
2) Only if it survives, compute the tight bound (2).
3) Only rarely compute “exact distance to arc” / trig, if you need an explicit angle value.

---

## Pseudocode sketches

### Bound (2): tight `UBdot`
```text
UBdot = -inf
for each vertex v:
    UBdot = max(UBdot, dot(q, v))

for each edge (a,b) with precomputed (n, g0, g1):
    dn = dot(q, n)
    p  = q - dn*n
    if |p| is tiny:
        UBdot = max(UBdot, dot(q,a), dot(q,b))
        continue

    if dot(p, g0) >= 0 and dot(p, g1) >= 0:
        UBdot = max(UBdot, sqrt(max(0, 1 - dn*dn)))
    else:
        UBdot = max(UBdot, dot(q,a), dot(q,b))
```

### Bound (3): hemisphere-violation in “sin space”
```text
LB_sin = 0
for each inward unit normal n_i:
    LB_sin = max(LB_sin, max(0, dot(n_i, q)))
```

---

## Assumptions (when this is valid)
These conservativeness claims rely on the **S2 cell model**:

- cell edges are **geodesic (great-circle) arcs**
- the cell is **spherically convex** (intersection of 4 hemispheres)

If your “bins” are something else (e.g., lat/lon rectangles on the sphere, or curved edges under a projection), then (2) and (3) may not hold without modification.
