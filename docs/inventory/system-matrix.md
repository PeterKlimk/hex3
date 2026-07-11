# Cross-cutting system matrix

Status: synthesis of the code-grounded inventories on 2026-07-12. This is a
working current-state map, not a roadmap or promotion decision. Fidelity and
Pareto assessments are provisional and should be reviewed before they become
canonical architecture.

The maintained successors to these provisional judgments are now
[`system-assessments.md`](../system-assessments.md) and
[`experiment-registry.md`](../experiment-registry.md).

Source inventories:

- [`documentation.md`](documentation.md)
- [`world-systems.md`](world-systems.md)
- [`rendering.md`](rendering.md)

## Current pipeline

```text
coarse geometry
  -> plates + independent crust + synthetic Euler motion
  -> boundary kinematics + tectonic history/features/deformation choices
  -> coarse elevation and bathymetry
  -> zonal circulation + temperature + transported moisture
  -> adaptive fine geometry + synthesized substrate
  -> pre-erosion drainage, basins, lakes, and rivers
  -> fluvial/hillslope erosion + limited local climate feedback
  -> final drainage and water bodies
  -> semantic coloring/selection
  -> cartographic or diagnostic rendering
```

The runtime names these Stage 0 geometry, Stage 1 lithosphere, Stage 2
atmosphere, Stage 3 fine/pre-erosion hydrosphere, and Stage 4 erosion. The code
retains both Stage-3 and Stage-4 fine surfaces. Older three-stage descriptions
and comments that call erosion future work are stale.

## Decision matrix

| System | Product status | Fidelity shape | Main value / consumers | Main cost or risk | Important gap or question |
|---|---|---|---|---|---|
| Spherical tessellation | Product, two backends | Exact/approximate computational geometry; physically neutral | Shared spatial graph and area basis for every model | Fine graphs reach millions of cells; backend and welding reproducibility | Make resolution/convergence guarantees part of the architecture contract |
| Plate partition | Product | Procedural/authentic hack | Coherent motion regions, boundaries, downstream tectonic skeleton | No force or mantle basis; tunable flood-fill grammar | Is this sufficient as a world initializer even if time evolution deepens? |
| Crust and continents | Product | Authentic procedural hack | Mixed-crust plates, margins, elevation and interaction types | Default crust does not arise from tectonic history | Preserve useful independent initialization while deciding how history modifies it |
| Euler kinematics | Product | Physical rigid-rotation primitive with synthetic forcing | Meaningful relative motion and boundary rates | Random axes/speeds lack force balance | Force-derived motion is expensive; visible/emergent payoff must be established |
| Boundary classification | Product | Physically based kinematics plus classification heuristics | Shared cause for trenches, arcs, ridges, collision and diagnostics | Threshold/polarity rules can dominate morphology | Stabilize and document interaction semantics and units |
| History and deformation | Legacy product path plus many experiments | Authentic history proxy through experimental physically based material evolution | Potential geological time, conservation, uplift history and future coupling | Large conceptual/runtime surface; internally validated alternatives remain unpromoted | Separate accepted model from selectable experiments; define promotion evidence |
| Tectonic features | Product | Mixed physical quantities and calibrated response kernels | High-leverage bridge from motion/crust to terrain and fine allocation | Easy to stack prescribed responses and double-own shape | Declare ownership of mass, uplift, shape and presentation for each feature |
| Coarse elevation/bathymetry | Product with selectable alternatives | Mixed physical basis, authentic hacks and procedural texture | Global terrain envelope consumed by all later stages | Normalized units and legacy shaping complicate physical interpretation | Establish one explicit elevation/unit/sea-level contract |
| Temperature | Product | Physically informed authentic hack | Latitude, continentality and lapse effects for moisture/hydrology | Dimensionless, no energy balance or seasons | Likely adequate until ecology/ice requires more; validate that payoff assumption |
| Circulation and wind | Product | Prescribed physically inspired circulation | Coherent bands, moisture transport and compelling particle display | Not dynamic climate; limited longitudinal structure | Assess whether asymmetric circulation is Pareto-positive for geography |
| Moisture and precipitation | Product | Physical finite-volume transport skeleton with tuned sources/sinks | Rain shadows, drainage, erosion forcing and climate views | Iterative cost; normalized water budget and one-layer physics | Clarify physical claims, convergence, and climate re-equilibration boundaries |
| Adaptive fine mesh | Product | Compute-allocation/authentic hack | Spends resolution on visible terrain, drainage and erosion | Memory, nondeterministic construction, scale-sensitive numerics | Formalize coarse/fine ownership, cache provenance and cross-resolution gates |
| Fine structural substrate | Product scaffold with many neutral/default-off experiments | Authentic to visual substrate hacks | Supplies unresolved geologic grammar for erosion to organize | Many cheap mechanisms create high tuning and ownership cost | Decide the minimal justified substrate after accepted tectonic model is clear |
| Hydrology | Product before and after erosion | Physical topology plus equilibrium/cleanup hacks | Drainage hierarchy, basins, lakes, erosion routing and river semantics | Basin breaching and lake equilibrium are sensitive; no transient water cycle | Wetlands, groundwater and persistent water budget are absent; prioritize by payoff |
| Fluvial/hillslope erosion | Product | Physically based process model, aesthetically calibrated | High-emergence terrain dissection and drainage coupling | Dominant iterative cost; chronology and sediment continuity are incomplete | Persistent sediment, tectonic feedback and dimensional time are major possible extensions |
| Local fine climate feedback | Product with optional components | Authentic physical shortcut | Orographic/lee/lake effects without rerunning global climate | Cannot alter global circulation; mean is deliberately preserved | High-Pareto design if its limitations are made explicit |
| Glacial erosion | Implemented; active-default status needs confirmation | Early authentic process hack | Potential high visual payoff in alpine terrain | No ice dynamics, mass balance, widening or isostatic response | Determine current activation and whether a fuller glacier model earns its cost |
| Sediment/geological surface materials | Ledgers and partial deposition only | Incomplete physical bookkeeping | Would connect erosion, basins, deltas, lithology and tectonics | Persistent material transport/state would be substantial | Strong missing coupling candidate; scope by visible multi-system payoff |
| Ecology/biomes | Absent | — | Would turn climate/terrain/hydrology into recognizable world regions | Risks becoming categorical paint without seasonal/water grounding | Define whether semantic biomes or process ecology best serves the project |
| Semantic feature model | Mostly implicit/distributed | Authentic cartographic derivation | Needed to name/generalize ranges, rivers, basins, margins and climate regions | Currently embedded in thresholds, coloring and diagnostics | Make this an explicit architectural layer rather than another simulation stage |
| Relief presentation | Product | Explicit visual/cartographic hack grounded in model elevation | Makes planetary terrain legible; strong spectacle payoff | Unit conversion remains obscure; screenshots can misdiagnose terrain | Preserve physical/diagnostic/cartographic/dramatic profiles with metadata |
| River presentation | Product, path-dependent | Physical topology + semantic selection + visual stroke | Legible drainage at globe scale | Fixed ~128 MiB SDF, duplicate paths, inconsistent map availability | Unify scale-dependent river policy across view and render paths |
| Terrain materials/colors | Product | Semantic/cartographic heuristics | High visual payoff from inexpensive derivations | Snow, rock, valley green and optical water can be mistaken for simulated state | Label derivations and connect to climate only where model quality justifies it |
| Lighting | Product | Illustrative visual hack | Cheap depth/readability and board/globe character | Diverges between rendering paths; interactive relief omits existing facet shading | Define presentation profiles and intended diagnostic versus showcase lighting |
| Wind particles | Product | Model-grounded vector visualization plus spectacle | Excellent comprehension and “wow” per GPU cost | Globe-only, unvalidated tracking/performance edge cases | Retain; add correctness/performance tests before increasing complexity |
| Diagnostic render modes | Product | Thematic visualization | Essential inspection of plates, fields, climate and hydrology | Mostly qualitative palettes with no units/legends | Add quantitative legends, normalization disclosure and true-scale slope views |
| Staging, cache and sweeps | Product tooling | Infrastructure | Fast before/after comparison and tractable experimentation | Untyped stage state, authoritative cache versioning, split viewed/computed state | Canonicalize state transitions and reproducibility metadata |
| Export and audit tooling | Product/tooling | Evidence infrastructure | Numeric validation, ledgers, scorecards and external analysis | Presentation/camera/config provenance is incomplete | Adopt a common reproducibility envelope for data and screenshots |

## Cross-cutting findings

### 1. The accepted product is smaller than the implemented code surface

The repository contains thirteen selectable orogen models and many erosion,
substrate, climate and glacial experiments. “Implemented,” “available,”
“numerically evaluated,” “visually evaluated,” “promoted,” and “product
default” are distinct states. Existing documentation often conflates them.

A replacement experiment registry should track these states explicitly. Current
architecture documents should describe the product path first, then name
experimental branches without presenting their combined capability as the
normal simulation.

### 2. The coarse/fine split is the central compute bargain

Global tectonics and climate run on the coarse graph; adaptive millions-cell
geometry is reserved for drainage, erosion and rendering. Fine-base caching and
local rather than global climate feedback make iteration affordable. This is a
strong Pareto architecture, but it needs a clearer scale contract: which fields
are authoritative at each scale, how they transfer, and which results must
converge when resolution changes.

### 3. The model-to-presentation boundary is healthy but incomplete

Relief scale and river width are explicitly renderer-only and do not feed world
generation. That directly supports the thesis. Remaining problems are mostly
interpretive and structural: normalized elevation has an unclear end-to-end
unit contract; visual heuristics look like modeled materials; screenshot/export
metadata is incomplete; and presentation capabilities differ across two surface
render paths and globe/map modes.

### 4. A semantic layer exists in fragments

River eligibility, major outlets, terrain materials, plate overlays, lake state,
range/object diagnostics and thematic palettes already interpret model state.
They are not yet treated as one architectural layer. Making that layer explicit
would clarify what is physical, what is derived meaning, and what is merely how
that meaning is drawn. It would also support scale-dependent cartography and
future strategy-game or board-like views.

### 5. Missing couplings may outperform missing standalone systems

The most consequential gaps are interactions: persistent sediment connecting
erosion to basins/coasts/lithology; erosion/loading feeding flexure or crust;
surface water feeding climate; evolving tectonics competing with drainage; and
modeled climate informing snow/ice/ecology. These are not automatically roadmap
priorities, but they deserve comparison against standalone additions because
each could make several existing systems more coherent at once.

### 6. Validation maturity varies independently of physical sophistication

Geometry, conservation ledgers, resolution diagnostics and numerical invariants
are relatively strong. Product-level comparison to planetary observations is
limited, and rendering relies heavily on controlled sweeps and human judgment.
Every system assessment should therefore record physical basis, internal
correctness, empirical grounding, structural behavior and visual acceptance as
separate evidence dimensions.

## Documentation consequences

The inventory supports a replacement set with separate responsibilities:

1. human project entry point;
2. thesis and fidelity policy;
3. current architecture and data flow;
4. canonical stage/state and coarse/fine pipeline;
5. per-system model/fidelity assessments;
6. semantic and presentation architecture/contract;
7. validation and reproducibility policy;
8. experiment registry;
9. roadmap, gaps and Pareto decisions;
10. clearly subordinate archives for audits, old specs and research.

No existing files should be deleted or moved until the replacement documents
preserve their still-valid decisions, open questions and historical evidence.
