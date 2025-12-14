# Proposal: Component-Based Waterbodies (Lakes/Wetlands) From Basin Water Levels

## Goal

Stop treating “one basin = one waterbody”, eliminate single-tile water specks as basins dry, and make rivers terminate consistently with what is rendered.

## What’s Subpar Today

- **Basin-wide depth gate tests capacity, not current lake.**
  - Current logic uses `max_depth = spill_elevation - bottom_elevation` to decide whether a lake can exist at all.
  - This answers “can this basin ever be deep?”, not “is there meaningful open water right now?”, and it permits tiny remnant puddles in deep basins.
- **Water classification implicitly assumes basin water is one contiguous lake.**
  - Cells are “lake water” if `elevation < basin.water_level`.
  - As water level drops below internal saddles, the wet set can legitimately split into multiple disconnected components; treating all of them as “lake” produces specks.
- **Rendering-only filtering risks invisible sinks.**
  - If tiny puddles are hidden only at render time, the hydrology still treats them as water sinks, so rivers can terminate in “invisible water”.
- **Small-lake behavior is sensitive to discretization.**
  - Integer surface targeting and hypsometry can yield one-cell residuals unless the “open water” decision is made using more robust geometry/volume criteria.

## Proposed Architecture Change (Keep Basin Water Balance; Derive Waterbodies)

### 1) Keep basins and the water-balance solver as-is

Preserve the current pipeline for:

- basin detection via priority-flood
- catchment computation
- overflow targets and overflow cascade
- one `water_level` per basin

Basins remain the unit of “container” and overflow.

### 2) Add a post-pass: extract connected “open-water bodies” from each basin

After basin `water_level`s are computed:

- For each basin, define the wet candidate set:
  - `W = { cell in basin | elevation[cell] < basin.water_level }`
- Split `W` into **connected components** using tessellation adjacency.
- Each component becomes a `WaterBody` candidate with *current* water metrics:
  - `area_cells`
  - `volume = Σ (water_level - elevation[cell])` over component cells
  - `max_depth = water_level - min_elev_in_component`
  - `mean_depth = volume / area_cells`

This explicitly allows a single basin to contain multiple distinct lakes at low stands.

### 3) Classify per component, not per basin

Replace “basin-level lake exists” logic with component-level rules:

- **Open water (Lake)** iff:
  - `area_cells >= A_min` AND (`mean_depth >= D_mean_min` OR `volume >= V_min`)
  - Optionally also require `max_depth >= D_max_min` as a secondary constraint.
- Otherwise classify as:
  - **Wetland/Mudflat** (optional third class), or
  - **Dry** if no intermediate state is desired.

This removes one-tile specks without deleting legitimate lakes elsewhere in the same basin.

### 4) Redefine “submerged / sink” to match what is rendered

Update the conceptual meaning of “submerged” so drainage and rivers remain consistent:

- `Ocean` is always submerged/sink.
- `Lake` components are submerged/sinks.
- `Wetland` (if implemented) is **not** a sink by default (rivers can continue across it), unless explicitly desired.

This prevents “invisible sinks” if wetlands/specks are not rendered as open water.

### 5) Replace or demote the basin-capacity `max_depth` gate

The basin-wide capacity test (`spill_elevation - bottom_elevation`) is not a good “open water exists” criterion.

- Remove it as the primary lake-formation test.
- If you keep a basin-level filter, use it only as a cheap early-out (e.g., ignore basins with negligible total relief), but rely on **component** metrics (area/volume/mean depth) for actual classification.

### 6) Optional: add a wetness band for “marshy” basin floors

If you want “kind-of wet” behavior where the basin floor can be damp while only some sub-depressions hold open water:

- Define a continuous wetness value for cells near the shoreline (slightly above/below `water_level`).
- Render wetness as marsh/mudflat tint/roughness, independent of open-water (Lake) classification.

## How This Changes Existing Knobs (Area / Depth / Volume)

- Keep a minimum area threshold, but apply it **per connected wet component**, not per basin.
- Replace “minimum lake depth” (as a single global gate) with more robust current-water criteria:
  - `mean_depth` threshold and/or
  - `volume` threshold,
  with optional `max_depth` as a secondary guard.

These metrics behave better on discrete Voronoi meshes than “deepest cell depth” alone.

## Expected Outcomes

- One-tile water specks are downgraded to wetland/dry instead of “lake”.
- A single spill-defined basin can contain multiple lakes at low stand (realistic split behavior).
- Rivers terminate only at true visible sinks (ocean + lake components), reducing short/messy fragmentation caused by tiny puddle sinks.

