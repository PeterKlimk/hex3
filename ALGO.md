## What the PDF’s algorithm does

The paper proposes a **GPU-focused, “meshless” 3D Voronoi** method that **does not build a global Delaunay/Voronoi connectivity structure**. Instead, it computes **each Voronoi cell independently** from a **small local neighbor set**, which avoids synchronization and supports high parallelism.  

The pipeline is explicitly two-stage: **(1) k-nearest neighbors** for every seed, then **(2) per-seed Voronoi cell construction + per-cell integrals**. 

---

## High-level algorithm (per the paper)

### Step 1 — GPU k-nearest neighbors (k-NN) via uniform grid + ring expansion

1. **Embed points in a 3D grid** (chosen so there are ~5 points per cell on average).
2. **Sort points by cell id** so each grid cell’s points are contiguous (fast retrieval).
3. For each query point `q`, **visit neighboring grid cells in concentric “rings”**, inserting candidate neighbors into a **fixed-size max-heap** of size `k`.
4. **Early stop** when the current farthest heap element is closer than the minimum possible distance to the next ring.  

Notes from the PDF:

* The full `k*n` neighbor index array can be huge (example: 10M points, k=40 ⇒ ~1.5GB just for indices), and the authors note an obvious optimization would be chunking/interleaving k-NN with cell construction, though they keep it simple and build the full array. 

---

### Step 2 — Per-point Voronoi cell as a half-space intersection (with early termination)

For a seed `i`, the Voronoi cell is represented as the **intersection of half-spaces** defined by bisector planes between `i` and other points. The algorithm:

1. Initializes the cell to the **global bounding box** of the domain.
2. Iteratively **clips the convex cell** by bisector planes for neighbors in **increasing distance order** (from the k-NN list).
3. Uses the **“radius of security”** criterion to stop early once further points cannot affect the cell.  

**Radius of security (key pruning idea):** while clipping, keep a bounding ball of the current cell around the seed; if the next neighbor is farther than **2 × (current bounding radius)**, its bisector cannot intersect the ball, so it cannot change the cell, and you can stop. 

---

## The core data structure (GPU-friendly “minimalist” cell representation)

Instead of storing an explicit polyhedron with variable-degree faces, the cell is stored in **dual form**:

* `P`: an array of **plane equations** for the current half-spaces.
* `T`: an array of **dual triangles**, where each triangle is a triplet of **plane IDs**; intersecting those 3 planes yields one Voronoi vertex position (computed on demand). 

Initialization from the bounding box creates 6 planes and a fixed initial triangle list `T`. 

This is designed so each GPU thread can run with **fixed maximum sizes** for `P`, `T`, and temporary sets, enabling constant per-thread memory. 

---

## Clipping a cell (the main geometric kernel)

To clip by a new plane `p`, the paper’s procedure (in dual terms) is:

1. Iterate dual triangles `(u,v,w)` in `T`:

   * Compute the corresponding primal vertex by intersecting planes `Pu,Pv,Pw`
   * If the vertex is on the “clipped out” side, remove that triangle from `T` and add it to a removed set `R`.
2. If anything was removed:

   * Append the new plane to `P`
   * Compute the boundary cycle `∂R` of the hole
   * For each boundary edge `(s,t)` create a new triangle `(s,t,p)` to “cap” the hole.  

### Boundary reconstruction

`computeBoundary(R)` merges triangle “elementary cycles” into a single **simple cycle** (a circular vertex list), because the hole boundary can then be represented compactly and updated efficiently. 

---

## Computing integrals without exporting cell geometry

After constructing a cell, the paper computes **volume and barycenter** by decomposing the cell into tetrahedra **on the fly**. Instead of triangulating faces explicitly, it iterates Voronoi vertices and defines a “dual geometry” using orthogonal projections of the seed onto planes and plane intersections; this yields tetrahedra whose signed volumes sum correctly even if parts lie outside the true cell (extra volume cancels via negative signed tetrahedra). 

---

## Failure modes and robustness strategy

The GPU approach uses bounded arrays; it can fail if:

* security radius not reached (insufficient neighbors),
* plane/vertex arrays overflow,
* boundary reconstruction error (numerical issues). 

The paper suggests fixing rare bad cells on CPU, either by re-running with robust predicates or by computing a local Delaunay triangulation of the seed + neighbors and taking its dual. 

---

## Optimizations described in the PDF

### 1) Tune the neighborhood size `k` (dominant performance lever)

The “most important parameter” is the **maximum number of neighbors** used for clipping; it directly affects:

* k-NN search work and memory,
* how many clipping planes are tested,
* global memory traffic.  

They emphasize that:

* `k` can be **small for good distributions** (e.g., blue noise),
* `k` must be larger for worse distributions (e.g., white noise),
  and tuning yields large speedups. 

### 2) Keep per-thread structures in fast GPU shared memory

The main GPU-specific optimization is storing the constant-size cell state (`T`, `R`, `P`, boundary list) in **shared memory**, and touching global memory mainly to stream neighbor points. 

### 3) Choose `max #P` and `max #T` to fit in shared memory

`max #P` (planes) and `max #T` (dual triangles / vertices) matter less for runtime than `k`, **as long as they fit in shared memory**; once they spill, performance degrades. 

### 4) Early exits reduce both clipping work and neighbor usage

Two major early exits:

* **k-NN ring traversal stop** once further rings cannot beat the current heap farthest. 
* **Security radius stop** during clipping once farther neighbors can no longer affect the cell. 

---

## Applicability constraints (important)

The method assumes point sets are **evenly distributed or smoothly varying** so each cell is determined by a relatively small local neighborhood (typical k cited in the paper is roughly 35–180). 
