# Cross-system disposition

Status: **current project decision**, 2026-07-17.

This document closes the first cross-system disposition pass. It decides what
the product should retain, simplify, replace, quarantine, remove or research
after the geographic-coherence and mountain-organization work. It is not a
claim that every retained implementation is final, and it is not a cleanup
queue.

See the [project thesis](thesis.md), [model strategy](model-strategy.md),
[system assessments](system-assessments.md), [landscape strategy](landscape-strategy.md),
[gap analysis](gaps.md) and historical
[subtractive audit](subtractive-audit.md).

## Decision

Keep the cheap causal spine:

```text
spherical/adaptive geometry
  -> plate/crust setting and boundary kinematics
  -> coarse elevation and climate
  -> fine surface and drainage
  -> erosion/hillslope operators
  -> water and geographic objects
  -> declared cartographic presentation
```

Shrink the experiment and configuration surface around it. The project has
more valuable operators than valuable named model compositions.

The next bounded slice is **water geography**, not another mountain model and
not yet ecology or sediment. It joins climate source and runoff, connected
ocean identity, basin/lake state, outlets and spills, drainage-repair
provenance, river roles, coast/island hierarchy and scale-aware presentation.
This is primarily a truth-contract and semantic integration pass over cheap
systems already worth retaining. It does not require transient hydrology,
seasons, ocean circulation, groundwater or coastal morphodynamics.

If that slice establishes coherent inputs and readable objects, the first
actual world expansion should be a bounded **living surface**: continuous
ecological constraints, semantic regions, vegetation coverage and multiscale
presentation. Persistent sediment remains the strongest large physical
candidate, but follows a river/material/time design gate.

## Dispositions

The primary disposition applies to the current product role. The action may
preserve an implementation temporarily as a control while replacing its
long-term ownership.

| System or role | Disposition | Truth and payoff retained | Current limit and action |
|---|---|---|---|
| Primary spherical Voronoi geometry | **Retain** | Coherent spherical topology, area, distance and adjacency used everywhere | Keep the pinned validated backend and its independent Hex3 gates; backend work must answer a concrete robustness or performance need. |
| Alternative geometry backends | **Quarantine** | Dedicated comparison and integration value | Keep outside ordinary product choice unless an active question justifies the maintenance surface. |
| Coarse/fine allocation | **Simplify** | The best current compute bargain: broad causes on coarse state and visible drainage/erosion on fine state | Retain two scales, but compare the adaptive density prior with a simpler equal-budget allocation and declare transfer semantics per field. |
| Plate/crust initialization | **Retain** | Cheap continent, crust and margin variety with inspectable causes | Treat it as an initializer, not geological genesis. Future history may modify it without replacing it with mantle simulation. |
| Euler motion and boundary kinematics | **Retain** | Physically meaningful rigid motion, convergence, divergence, shear and polarity | Synthetic poles are not force-balanced history. Research force-derived motion only if several visible or downstream consequences need it. |
| Product boundary feature bridge | **Retain** | Present fronts, sign, rate, crust setting and polarity are high-leverage forcing inputs | Stop treating its scalar response as finished mountain terrain. Preserve the inputs while the eventual forcing organizer is replaced. |
| Scalar tectonic final-height ownership | **Replace** | Keep the legacy output as the usable product baseline and control | The generic capped/ribbon grammar is not a product foundation. A future replacement starts from real front topology, tectonic history and inherited material state; H/C/G/I do not advance. |
| History, lifecycle, carrier and named orogen ladder | **Quarantine** | Conservative transport, overlap, material-ledger, underthrust and relaxation operators remain a useful research library | Lifecycle does not consume generated reorganization history, several compositions are falsified, and the public ladder overstates product choice. Remove misleading aliases and retired named selectors after operator extraction. |
| Coarse elevation, bathymetry and datum | **Simplify** | Efficient global envelope with explicit physical elevation and crust-column conversions | Keep ownership and units explicit. Target-land-fraction sea level is a worldbuilding constraint, not conserved ocean volume; deeper loading/sea-level physics waits for sediment or ice consumers. |
| Temperature and prescribed circulation | **Retain** | Cheap latitude, elevation and circulation structure for water, ecology and presentation | Do not add a dynamic atmosphere. Test regional consequences before adding longitude-asymmetric heat or seasonality. |
| Moisture transport and precipitation | **Retain** | Affordable finite-volume transport connects ocean, wind and terrain to runoff | Its advantage over cheap climatology and its normalized units remain unproved. Water-geography controls decide whether to keep or simplify it. |
| Fine structural substrate | **Simplify** | Coarse-to-fine transfer and a minimal material/terrain substrate support the product | Quarantine inactive scarps, grain, meso relief, rebuilds and other overlapping shape owners outside product parameters and CLI. |
| Fluvial incision and hillslope operators | **Retain** | They create real drainage-related relief and conservative surface response | The 200-step hold-and-carve composition is calibrated, expensive and not geological time. Preserve operators; simplify the composition and evaluate individual diffusion/material/sink benefits only when tied to an object. |
| Default-off erosion, deposition and climate-feedback branches | **Quarantine** | Some branches remain useful isolated experiments | Zero defaults do not remove ownership and maintenance cost. Retire neutral or superseded paths rather than accumulating controls. |
| Hydrologic topology and routing | **Retain** | Basins, outlets, spills, accumulation and flow topology are a central emergence engine and future input | Keep the physical/topological core; clarify transient claims, runoff units and resolution effects instead of replacing it with a deeper water solver. |
| Equilibrium lakes and drainage integration | **Simplify** | Storage and terrain-repair hacks make difficult terrain usable and preserve exact sparse provenance | One lake-ratio dial is not whole-world climate. Bind lake state, outlet/spill relations and repair cuts to shared objects; judge object consequences before changing the hack. |
| Water-body and river semantics | **Retain** | Shared identity, mouths, reaches, hierarchy and catchment policy cheaply support diagnostics and rendering | Separate catchment/discharge importance, hierarchy, trunk length and cartographic importance. Add scale generalization and comparison-sidecar ancestry without a generic entity framework. |
| Coast and island semantics | **Research** | Cheap hierarchy/generalization can improve maps, inspection and later settlement immediately | Current coast is only the zero-elevation boundary. Derive semantic coasts/islands/straits now; dynamic coastal processes and deltas wait for sediment. |
| Drainage-repair provenance | **Retain** | Exact cuts explain where authoritative terrain was changed | Make repair contribution visible in river, lake, range and capture evidence. Do not call repair erosion or assume every repaired object is invalid. |
| Ecology/biome prototype | **Quarantine** | Continuous constraints and uncertainty are useful diagnostic scaffolding | It is implemented semantics, not an ecology stage. Do not calibrate labels or create product dependencies before climate/water correspondence. |
| Glacial shaping pass | **Quarantine** | A later cryosphere could provide high-latitude water and terrain consequences | The current pass lacks ice mass balance, dynamics, widening and loading; do not use it as generic mountain texture. |
| Relief, river styling, color and lighting | **Retain** | Authentic and Dramatic profiles make true physical state legible and appealing | Consolidate around semantic inputs and declared profiles. Fix scale selection and path parity before heavier material rendering. |
| GPU wind particles | **Retain** | Model-grounded motion has strong explanatory and spectacle value | Validate runtime and representative performance on Windows independently of the atmosphere's physical claim. |
| Stage orchestration, snapshots and cache | **Simplify** | Retained stages and caches support inspection and iteration | Keep snapshot semantics, but add a dependency graph only when ecology/sediment creates concrete invalidation pressure. Do not build framework first. |
| Diagnostics, corpus and experiment tooling | **Simplify** | Reproducible evidence and operator tests prevent false promotion | Move shared promoted measurements into library adapters; keep historical or invalid probes out of normal product configuration and contributor workflow. |
| Persistent sediment | **Research** | One conserved mobile/deposited material could connect erosion, basins, floodplains, deltas, coasts, soil and loading | Require one visible source-to-sink target, explicit time/units, river ownership and memory budget before implementation. No stratigraphic simulator. |
| Living surface | **Research** | Climate, terrain and water can yield major visual identity, ecological regions and inputs to resources/settlement | Conditional next expansion after water geography. Begin with continuous limitations, regions and coverage—not succession or individual-tree ecology. |
| Human geography | **Research** | Traversability, resources, routes and settlement can make geography consequential and produce board-like stories | Follow living geography with aggregate suitability/routes/settlement before agents, economies or full civilization simulation. |

## Shared unresolved seam: water geography

Several apparently separate defects share one ownership problem:

- climate currently decides moisture sources before connected-ocean identity;
- lake storage can change independently of precipitation, rivers and erosion;
- lake survival and area change materially with fine resolution;
- river “major” status conflates catchment, hierarchy, length and visibility;
- repair cuts can contribute to selected trunks and ranges without appearing in
  their semantic provenance;
- coast/island hierarchy and topology-aware generalization do not exist; and
- Globe/Map and surface paths do not consistently present the same objects.

A shared account should relate, without conflating:

```text
connected ocean / inland water identity
  -> moisture-source and runoff context
  -> basin, lake, outlet and spill state
  -> drainage supply and river-network roles
  -> sparse repair provenance
  -> coast, island and water-region semantics
  -> scale-dependent cartographic selection
```

It need not be one mutable mega-object. Physical state remains authoritative;
semantic objects and comparison-sidecar ancestry are deterministic derivatives.
Cross-resolution correspondence is evaluation evidence until a product
consumer demonstrates a need for persistent lineage.

Implementation progress: the first seam uses one connected-ocean classifier for
climate coast distance, moisture sourcing and hydrology. A compact derived
report covers aggregate water/land components, shoreline, basin/lake state,
distinct river roles, repair footprint and consistency. Dossier schema v3 adds
the first frozen-terrain conditional-climatology comparison. Its interpretation
on a representative-resolution multi-seed panel, coast geometry/generalization
and event-level repair provenance remain open.

## Next bounded slice: Water Geography V0

### Product question

Does the current transported climate create stable, useful water geography
beyond a much cheaper climatology, and can existing hydrology be expressed as
coherent, inspectable and legible river/lake/coast objects with honest repair
provenance?

### Scope

Use frozen terrain and the existing product hydrology. On a small set of
already informative worlds:

1. compare current climate/runoff with a latitude–elevation–coast-distance
   baseline;
2. use wind reversal, orographic terms and uniform runoff only as causal
   controls, not a parameter sweep;
3. use connected ocean identity consistently when attributing moisture source;
4. derive one shared water-geography report: oceans, lakes, basins, outlets,
   spills, river mouths/reaches/trunks, coast/island hierarchy and repair-cut
   contribution;
5. distinguish physical supply, network hierarchy, longest trunk and
   cartographic importance; and
6. inspect the same objects in Physical, Diagnostic and Cartographic views with
   scale-appropriate framing.

Do not add seasons, dynamic weather, ocean circulation, groundwater, wetlands,
vegetation, sediment transport, coastal evolution or erosion tuning inside
this slice.

### Decision

- Retain transported moisture if its barrier orientation, interior drying and
  downstream river/lake consequences are stable, causal and worth the small
  added cost; otherwise simplify to the baseline or a smaller authentic hack.
- Retain equilibrium/repair hacks only with explicit object provenance and
  acceptable topology; replace a hack only where the shared view locates a
  concrete failure.
- Advance the living-surface slice only when ecological inputs have declared
  meaning and water objects are coherent enough to consume.

## Pareto frontier after this decision

1. **Water Geography V0** — immediate enabling slice; low–medium cost and broad
   leverage.
2. **Living Surface V0** — likely first expansion; very high visible payoff,
   conditional on the water/climate result.
3. **Source-to-sink Sediment V0** — strongest large physical coupling, but
   requires a design gate for river ownership, time, mass and memory.
4. **Geography-to-human semantics** — aggregate productivity, traversability,
   routes and settlement after living geography.
5. **Reduced tectonic forcing replacement** — important but deferred until it
   can consume real product fronts/history/material state rather than another
   manufactured test field.

Dynamic atmosphere/ocean, full vegetation dynamics, individual-tree ecology,
full stratigraphy, generic glacial tuning, physically based material rendering,
force-derived global plates and full civilization/economics are dominated at
the current frontier.

## Stop rule

The disposition pass is complete enough to choose work. Do not turn it into a
permanent scoring exercise. A new product mechanism must replace an owner,
create a visible or reusable consequence and preserve the truth contract more
cheaply than the next-best alternative. Otherwise retain the simpler system or
stop.
