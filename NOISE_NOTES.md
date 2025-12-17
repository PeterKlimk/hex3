## Noise system notes / TODOs

### Ridge anisotropy direction field
- Current approach infers an along-belt direction from the local gradient of `features.convergent`.
- This is a *line field* (180°): `v` and `-v` are equivalent.
- If artifacts appear from cell-to-cell sign flips, enforce a globally consistent sign by:
  - picking an arbitrary seed cell direction
  - BFS/DFS over the adjacency graph
  - flipping neighbor vectors to keep `dot(v_i, v_j) >= 0`

### Separate fabrics for arcs vs collisions
- Today the ridge fabric anisotropy is driven by convergence only.
- If arcs and collision belts should look different, split the mask/params into:
  - `collision_fabric` (thicker, higher amplitude, more coherent)
  - `arc_fabric` (narrower, more segmented / volcanic-line texture)

### Hills (pre-erosion seed, avoid double counting)
- Treat hills as *bedrock-scale variability* to seed drainage, not “finished” dissection.
- Prefer using tectonic fields (`convergent/divergent/activity`) as exclusion masks (suppress in active belts),
  not as positive “make hills here” drivers.
- Consider adding a low-frequency “erodibility/lithology” field that is reused by erosion later, so the same
  cause influences both initial relief and subsequent incision.

### Domain warping
- If adding warps, scale warp strength by a mask to avoid globally smearing tectonic signals.
- Start with 1 warp field and minimal octaves; avoid stacked warps unless necessary.
