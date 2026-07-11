# Rendering and presentation inventory

Status: working inventory, code-authoritative as of 2026-07-12. This document describes what exists; fidelity labels are provisional assessments, not design decisions.

## Executive summary

Hex3 already has a recognizable separation between modeled state and presentation. Elevation, water state, drainage, tectonic fields, and atmosphere live in `world`; `app` derives colors and render geometry; `render` owns wgpu pipelines; shaders perform projection, lighting, displacement, river reconstruction, and particles. Recent relief presets and renderer-only river-width controls make the physical/presentation distinction explicit.

The main presentation path is nevertheless not one coherent renderer. It is two surface paths plus specialized overlays:

- `UnifiedMesh` + `unified.wgsl` for Relief and the two wind views. This carries elevation and material, supports radial relief, and blends a baked river distance field into the terrain.
- A regenerated `VoronoiMesh` + `sphere.wgsl` for Terrain, Elevation, Plates, Noise, Hydrology, Features, and non-wind Climate views. It carries precomputed RGB but no elevation/material semantics.
- Lines/markers for cell edges, plate motion, non-unified globe rivers, and GPU particle trails.

This split explains several inconsistencies: rivers are presentation-path-dependent, non-relief modes cannot use relief-aware material lighting, map views of non-unified modes have no river line overlay, and the more diagnostic displaced-face hillshade exists but is enabled by the headless sweep rather than the interactive Relief view.

## Render modes and visible contract

Definitions and cycling are in `src/app/view.rs`; key handling is in `src/app/mod.rs`; color derivation is in `src/app/coloring.rs`; mesh selection is in `src/app/world.rs` and `src/app/state.rs`.

| Mode | World inputs | Visible role | Presentation/fidelity category | Current implementation and gaps |
|---|---|---|---|---|
| Relief (`1`) | active elevation, hydrology water states/levels, terrain colors | Main spectacle/product view: displaced terrain, lakes/ocean, draped rivers | Cartographic geometry plus visual styling, grounded in modeled fields | Uses unified mesh and configurable vertical exaggeration. Water surfaces are flattened to ocean/lake level. Interactive path does not enable displaced-face slope shading, despite shader support. |
| Terrain (`2`) | elevation, hydrology, local slope, flow accumulation | Flat “finished terrain” map/globe | Cartographic thematic view | Hypsometric color, optical-looking water palettes, snow by elevation, rock by slope, and a deliberately cheap flow-based green-valley moisture proxy. No relief. Rivers are lines only on globe. |
| Elevation (`3`) | active elevation | Raw elevation inspection | Diagnostic thematic view | Fixed piecewise palette, not a numeric legend or true-unit view. Hydrologic lake overlay is deliberately absent. |
| Plates (`4`) | plate assignment, dynamics, boundary relative velocity | Plate regions, boundary stress, velocities, Euler poles | Diagnostic/semantic overlay | Fixed-size arrows and pole markers communicate direction/identity, not physical scale. Overlays appear only on globe. Boundary edges appear only when global edge toggle is on. |
| Noise (`5`) | current noise outputs | Combined, Macro, ArcShape | Diagnostic thematic view | CPU recolors mesh on each cycle. The checked-in `layered.wgsl` describes older six-layer names and instant GPU switching; current application does not use that path. |
| Hydrology (`6`) | hydrology flow/water state | Flow accumulation/drainage inspection | Diagnostic thematic view | CPU palette; absent hydrology must be handled by coloring fallback. River network overlay is independently controlled and may visually duplicate/complicate flow coloring. |
| Features (`7`) | trench, arc, ridge, collision, activity | Tectonic field inspection | Diagnostic thematic view | CPU recoloring. Fixed normalization/palettes mean color is comparative rather than quantitative. |
| Climate (`8`) | temperature, surface/upper wind, uplift, precipitation | Atmosphere field inspection | Diagnostic plus spectacle | Surface/upper wind switch to unified relief terrain and GPU particles; other layers use flat colored mesh. Stage gating is handled by application state/color functions. |

Layer availability in current code differs from older summaries: Noise has only Combined/Macro/ArcShape; Climate has Temperature/Surface Wind/Upper Wind/Uplift/Precipitation.

## Views, projection, and navigation

`ViewMode` has Globe and Map (`src/app/view.rs`). Globe uses an orbit camera (`src/render/camera.rs`) with left-drag rotation and scroll zoom. Map uses an orthographic camera and an equirectangular projection performed in the vertex shaders. Per-vertex `wrap_offset` handles antimeridian triangles. Tab toggles views; globe-only input guards disable orbit dragging in map mode.

The map is a direct equirectangular thematic view: it has no equal-area projection, scale bar, graticule, labels, projection metadata, or zoom-dependent generalization. Relief displacement is bypassed in map projection even when the unified pipeline is selected, which is a sound separation. Plate arrows/poles, GPU wind particles, and line rivers are globe-only.

Category: cartographic projection and interaction, not a physical model. Cost is low (vertex-shader transform), with the usual severe high-latitude area/shape distortion. Validation is implicit via wrap-offset mesh handling; no projection image tests were found.

## Surface data flow and GPU architecture

1. `World` exposes the active stage/tessellation/elevation/hydrology.
2. `src/app/coloring.rs` derives per-cell RGB for thematic modes. `src/app/world.rs` triangulates the active Voronoi surface.
3. `generate_world_buffers` builds a dynamic colored mesh, a persistent unified relief mesh, edge/overlay buffers, two line-river buffers, and an 8192x4096 river SDF texture.
4. `AppState::render` chooses unified versus colored fill, globe versus map pipeline, overlays, and uniforms.
5. `src/render/renderer.rs` executes a depth-tested fill pass and applicable line/particle draws. `src/render/pipeline.rs` contains the pipeline construction; `src/render/context.rs` owns windowed or offscreen targets.

Colored meshes are regenerated on mode/layer changes and stage snaps (commented as roughly 5–10 ms, though fine-mesh logging shows memory is a first-class concern). The relief wireframe is deliberately lazy because at roughly 2.5 million fine cells it can occupy hundreds of MiB and overlap with mesh rebuilding badly enough to exhaust memory. Stage buffers are cached so pre-/post-erosion comparisons become instant after first construction (`src/app/state.rs`).

`src/render/elevation_map.rs` rasterizes the surface into a float cubemap. The GPU particle renderer samples it so trails follow relief without an equirectangular seam/pole singularity. This is infrastructure for presentation coupling, not an end-user elevation-map display.

## Relief and vertical exaggeration

`ReliefPreset` (`src/app/view.rs`) is explicitly presentation-only:

| Preset | Scale | Intended meaning |
|---|---:|---|
| Flat | 0 | No radial displacement |
| Physical 1x | `10 / PLANET_RADIUS_KM` | A 10 km normalized elevation maps to physical globe scale |
| Authentic | `RELIEF_SCALE`, currently 0.04 | Default legible cartographic relief |
| Dramatic | 0.08 | Showcase exaggeration |
| Custom | nonnegative CLI/runtime value | Experiments/sweeps |

`X` cycles presets. `--relief-scale`/sweep overrides become renderer state and do not alter world generation (`src/app/world.rs`, `src/app/sweep.rs`). The shader displaces radially by `1 + elevation * relief_scale`; ocean is rendered at radius 1 and lake cells at modeled water level. Relief edges are CPU-displaced with the same selected scale and invalidated when the scale changes. Wind particles receive the same relief scale and sample surface elevation.

This is the clearest existing implementation of physical grounding to cartographic spectacle. Remaining ambiguity: “Physical 1x” encodes a 10 km reference into a scale while model elevation values are normalized; the full unit conversion contract should be documented and tested end to end. The status line prints preset and scale, but there is no on-canvas exaggeration legend or capture metadata.

`unified.wgsl` can derive a facet normal from screen-space derivatives of displaced geometry (`slope_shading`). `src/app/sweep.rs` enables it for relief judging, but `AppState::render` never calls `with_slope_shading(true)`. Interactive Relief therefore shades with smooth sphere normals, weakening slope readability and making hypsometric color/shape carry most terrain perception. This looks like an unfinished diagnostic/presentation integration rather than a model problem.

## Terrain, ocean, lakes, coasts, and color

`src/app/coloring.rs` is a substantial semantic/cartographic layer:

- Land uses fixed hypsometric bands, then slope-based rock exposure and a fixed elevation snowline.
- Ocean and lake colors use nonlinear depth darkening with separate palettes and bright shallow margins, described as optical attenuation/turbidity but not an optical water simulation.
- A flow-accumulation “moisture proxy” greens gentle lowland channels. The code explicitly calls this cheap and subtle; it is an authentic visual hack, not climate state.
- The unified mesh labels Land/Ocean/Lake material and uses hydrologic lake levels. Coastal shared vertices combine land averages and water levels to avoid cracks/steps (`src/app/world.rs`, geometry mesh constructors).

No separate coastline geometry, foam, bathymetric shelf line, shoreline antialiasing, sea-level datum display, clouds, atmosphere shell, or biome/vegetation renderer exists. Snow is purely an elevation color threshold and does not consume modeled temperature/precipitation; this is a likely Pareto-positive future coupling if climate fields are trustworthy.

Potential leakage is limited but worth naming: cartographic slope/moisture/snow colors are derived from physical fields and remain render-only, which is healthy. However, screenshots of Terrain/Relief can easily be read as claims about climate or rock exposure when these are visual heuristics. The future semantic layer should identify them explicitly.

## Rivers

River presentation has two implementations (`src/app/world.rs`, `src/shaders/unified.wgsl`, `src/shaders/surface_line.wgsl`):

- Relief/wind unified views bake eligible centerline segments into an 8192x4096 equirectangular RGBA distance-field texture. R stores distance, G flow factor, B major-river membership. The shader reconstructs an antialiased, flow-tapered screen-space stroke, blends a reflective/glint water color into the terrain, and samples with longitude wrapping.
- Other globe views draw alpha-blended line segments between cell centers, displaced by the current surface-line uniform with a fixed z offset. Alpha is logarithmic in flow. Lake outflow paths are included.

`V` cycles Off/Major/All. Eligibility can use a physical catchment-area threshold (default documented as approximately global-map perennial-river scale) or legacy count-equivalent thresholds. `river_width_scale` is explicitly renderer-only and sweepable. The shader’s derivative conversion is specifically intended to keep widths in screen pixels rather than accidentally making them 7–21 km physical ribbons.

Category: drainage topology and flow are model-grounded; selection is semantic cartography; stroke width, opacity, water Fresnel/glint, and z offsets are visual hacks. This is an appropriate separation. Main costs are a fixed ~128 MiB uncompressed RGBA texture before overhead, CPU baking, and duplicate line buffers. There are no mipmaps; high-frequency minification and polar distortion merit visual testing. Non-unified Map views suppress line rivers and do not sample the river texture, so Terrain/Hydrology/etc. maps currently show no explicit river overlay. Rivers in unified Map can be texture-blended, though Relief itself is flattened by map projection.

River-specific diagnostics exist outside the renderer (`diagnose --river-audit` references in `src/app/world.rs`), and the sweep harness can vary renderer-only width. No automated image regression, seam/pole test, width-in-screen-pixels test, or visibility-at-zoom test was found.

## Plate and cell overlays

`src/app/visualization.rs` derives:

- Plate-boundary edge colors from relative Euler-pole velocity projected onto a local tangent normal: red convergent, blue divergent, yellow near-transform.
- Fixed angular-length velocity arrows on every boundary cell; hue encodes speed, but arrow length does not.
- Fixed-size lifted triangular Euler-pole markers colored by plate identity.

These are semantic diagnostics with authentic directional grounding and deliberately nonphysical sizing. Dense boundary-cell arrows can become cluttered at high resolution; no zoom-dependent thinning, legends, numeric units, selection, or vector-scale indicator exists. Plate overlays are globe-only. General cell edges toggle with `E`; relief edges are scale-matched and expensive, while flat-mode edges use ordinary line buffers.

## Atmosphere and wind particles

`src/render/particles.rs`, `wind_particles.wgsl`, and `particle_render.wgsl` implement the active wind visualization. A default 50,000 particles are initialized at cell centers. A compute shader advects each particle along the per-cell wind vector, tracks its current cell using local adjacency (constant-bounded lookup under the slow-motion assumption), retains a trail endpoint, ages/respawns particles, and dispatches 64 threads per workgroup. Rendering uses instanced two-vertex line trails.

Surface particles sample the elevation cubemap and use the terrain relief scale; upper-wind particles float at a fixed height. `W` enters/toggles surface and upper wind; particles render only on globe at Stage 2+ in the appropriate Climate layer. Speed and trail scales are presentation parameters. The field is model-grounded, while seeding, density, lifetime, speed, trail length, colors, and upper-air height are spectacle/flow-visualization choices.

This is a high-benefit GPU technique: O(particles) parallel work exposes a vector field more intuitively than arrows without feeding back into simulation. Costs include storage/compute each frame, elevation cubemap generation/update, and possible cell-tracking failure if a frame crosses beyond the searched neighborhood. No particle correctness/performance tests or map-mode equivalent were found. A legacy CPU-particle scene slot remains but is always passed as `None`.

## Lighting and materials

The unified shader has material identifiers for land, ocean, lake, river, and ice/snow. It supports:

- A stylized three-part “hemisphere” light: warm sun, cool sky ambient, warm ground bounce, plus wrapped diffuse.
- A simple ambient + directional diffuse alternative toggled with `H`.
- Optional displaced-facet hillshade.
- Ice/snow specular glint and river Fresnel/specular styling.

The colored `sphere.wgsl` path uses only simple ambient/diffuse and ignores the hemisphere toggle semantically even though application uniforms are shared by pipeline setup. Water is not geometrically reflective/refractive; there are no shadows, atmosphere scattering, tonemapping/exposure controls, PBR BRDFs, normal maps, or time-of-day. Category: visual hack / illustrative lighting, appropriate to the current board/globe aesthetic but not physically based rendering.

## Diagnostics, sweeps, staging, and export

- `Space` computes/advances Lithosphere → Atmosphere → pre-erosion Hydrosphere → Erosion. `Backspace` snaps to an already computed earlier stage. Cached buffers make direct pre/post comparisons practical.
- `src/app/sweep.rs` is a headless offscreen renderer producing PNG tiles and contact sheets for controlled parameter comparisons. It can isolate renderer-only `relief_scale` and `river_width_scale`, uses common cameras, and deliberately enables simple directional displaced-surface hillshade for relief judgment. This is the strongest current presentation validation tool.
- `Up/Down` changes hydrologic climate ratio at Stage 3+ and rebuilds buffers. This is a model/state adjustment exposed through visualization, not merely a display control.
- `D` exports gzipped JSON via `src/app/export.rs`. Export includes physical/model fields and lifecycle audits, not current camera, view mode, relief preset/scale, river-width scale, lighting, or render settings. Thus exported state cannot reproduce a screenshot’s presentation.
- Runtime FPS is displayed/logged through `AppState`; fine-mesh allocations are logged. No GPU timing, draw/triangle statistics, image regression suite, or reference captures were found.

## Controls in current code

| Input | Action |
|---|---|
| Left drag / scroll | Rotate globe / zoom |
| Tab | Globe ↔ equirectangular map |
| `1`–`8` | Relief, Terrain, Elevation, Plates, Noise, Hydrology, Features, Climate |
| Repeated `5`, `7`, `8` | Cycle that mode’s sublayers |
| `W` | Enter/toggle surface and upper wind |
| `E` | Toggle cell edges |
| `V` | Off/Major/All rivers |
| `X` | Flat/Physical/Authentic/Dramatic relief preset |
| `H` | Toggle hemisphere versus simple lighting (effective on unified shader) |
| Space / Backspace | Advance computation/view next stage / view prior computed stage |
| Up / Down | Adjust hydrologic climate ratio |
| `R` | Regenerate with random seed |
| `D` | Export world JSON gzip |
| Escape | Quit |

## Suspected model/presentation leakage and architectural debt

No clear case was found where relief scale or river stroke width feeds back into world generation; comments and override plumbing explicitly prevent it. The main risks are interpretive or structural:

1. The same word “elevation” spans normalized model values, physical-kilometre interpretation, color thresholds, and radial render displacement. The unit/conversion contract is not self-evident.
2. Snow, rock exposure, green valleys, optical water, and river reflectance can be mistaken for simulated fields. They should be recorded as semantic/cartographic derivations.
3. Presentation varies by pipeline rather than only by declared policy. River availability, lighting, and geometry differ between unified/colored and globe/map paths.
4. Relief wireframes are CPU-baked with a presentation scale while surface relief is shader-driven. Invalidation is implemented, but this duplicate authority is fragile.
5. `layered.wgsl` and associated layered uniforms/vertex layout appear to describe an older or unused instant-layer architecture while current mode switching regenerates CPU colors. Confirm and remove or revive during cleanup.
6. Elevation cubemap coastal interpolation is presentation infrastructure consumed by particles; it should not become an accidental alternate terrain truth.

## High-value gaps for the replacement architecture discussion

These are inventory findings, not yet a roadmap:

- A declared presentation profile that bundles relief, river generalization, lighting, overlays, and intended use (physical diagnostic, cartographic, dramatic), rather than independent implicit switches.
- A true-scale diagnostic view with explicit units, vertical-exaggeration readout, scale/legend, and perhaps slope-angle visualization.
- Interactive use of displaced-surface hillshade, with a conscious choice about stylized hemisphere lighting versus relief judgment.
- Consistent river policy across flat/globe/map modes and zoom levels; likely semantic generalization rather than one fixed world texture.
- Reproducible capture metadata connecting seed/stage/model parameters to camera and all presentation settings.
- Quantitative legends and normalization disclosure for elevation, features, climate, flow, and velocity diagnostics.
- Scale-dependent decluttering for plate arrows, cell edges, rivers, and future labels.
- Automated reference captures/image comparisons for antimeridian seams, poles, river widths, coast flattening, relief presets, pre/post erosion, and stage availability.
- Explicit retirement or integration of unused shader/pipeline paths and legacy CPU-particle interfaces.

## Validation presently visible in code

The presentation-specific unit test found is `ReliefPreset` scale/cycle stability in `src/app/view.rs`. Most validation is observational: sweep contact sheets, diagnostic/audit CLI paths, allocation logs, FPS, and interactive mode comparison. The renderer has good implementation comments around known defects (river physical-width inflation, fine-edge memory exhaustion, smooth-normal relief), but those fixes are not protected by automated visual or invariant tests.
