# Subtractive architecture audit

Status: active first-pass disposition audit, 2026-07-12. This document asks
which systems and code paths earn their claims, cost and architectural surface.
It complements the [evaluation synthesis](evaluation-synthesis.md) and
[gap analysis](gaps.md); it is not a deletion manifest by itself.

## Disposition vocabulary

| Disposition | Meaning |
|---|---|
| **Retain** | Current correspondence and payoff justify the product role |
| **Validate** | Plausible/high-value, but a named claim or cost remains unproven |
| **Simplify** | Useful result, excessive mechanism/state/configuration surface |
| **Quarantine** | Research code or evidence retained outside ordinary product architecture and CLI |
| **Remove/replace** | Invalid, unreachable, duplicated, misleading or no longer worth maintenance |

Evidence strength is recorded separately:

- **demonstrated** — code reachability, invariant, controlled result or corpus evidence;
- **strong inference** — code structure and existing evidence point one way, but
  an ablation is still required;
- **open question** — concern worth testing, not a negative verdict.

Quarantine is an architectural action, not a euphemism for failure. Useful
operators and test fixtures can survive after a failed named model stops being
a selectable world-generation path.

## Executive diagnosis

Hex3's likely Pareto core is substantially smaller than its compiled and
documented experiment surface:

```text
spherical/adaptive mesh
  -> rigid plate kinematics + product boundary features
  -> coarse elevation/climate forcing
  -> fine transfer
  -> stable drainage
  -> SFD stream-power incision + one hillslope relaxation
  -> final hydrology and shared semantic objects
  -> declared cartographic presentation
```

The main architectural problem is accumulation. Failed hypotheses, alternative
solvers, painted fine-relief systems and rendering predecessors remain inside
product enums, parameter structs, cache keys, CLI help, omnibus state objects
and contributor mental models. Zero defaults prevent runtime effects but do not
remove architecture cost.

## Demonstrated wrong, misleading or unreachable behavior

### Lifecycle does not consume generated motion-reorganization history

Carrier replay stores per-snapshot Euler poles, and the evolved-carrier solver
uses them. The lifecycle solver instead initializes present `Dynamics` poles,
advances those constant poles and changes motion only on continental merge. It
does not read `snapshot.plate_euler_poles`.

Therefore:

- a finite motion-coherence setting can be bit-identical to constant motion in
  lifecycle output;
- lifecycle is currently a fixed-pole/merge-driven material automaton, not a
  solver of the generated reorganization history;
- conservation tests establish operator correctness, not history correspondence.

Evidence: `src/world/history.rs:600-802`,
`src/world/deformation.rs:960-1070`, `2869-2905`. The existing reorganization
ledger test exercises evolved replay, not lifecycle (`deformation.rs:3334-3347`).

Disposition: **quarantine** lifecycle as a causal research engine and correct
all history claims. Continue only after a consequence ablation shows that a
real motion schedule changes retained ranges, basins or drainage enough to earn
the complexity.

### `HistoryCarrierThinSheet` is not a distinct moving-carrier solver

`HistoryCarrierThinSheet` and `HistoryThinSheet` route to the same
`solve_history_thin_sheet` implementation (`features.rs:949-957`). Any difference
comes from contact/history inputs, not a separate carrier thin-sheet mechanism.

Disposition: **remove/rename** the misleading alias unless a maintained ablation
requires it.

### Stale rendering paths were constructed but unreachable

- `LayeredGlobe`/`LayeredMap`, two pipelines, bind groups, `LayeredMesh`,
  `LayeredVertex` and `layered.wgsl` remain implemented. Current app and sweep
  selection choose only Globe/Map or Unified variants; no producer selects a
  layered fill (`app/state.rs:442-445`, `app/sweep.rs:240`).
- `RenderScene.wind_particles` and its CPU line draw remain, but every current
  scene passes `None`; the app explicitly labels it legacy
  (`app/state.rs:574`, `app/sweep.rs:251`).

Disposition: **removed** after reachability and full compile/test verification.
Unified globe/map rendering and GPU wind particles remain. A Windows runtime
check is still required because WSL cannot validate live wgpu pipeline creation.

### Invalid numerical interpretations already identified

The following should remain registered as rejected behavior:

- population slope–area regression as a river-grading gate;
- `acos(dot)` distance for near-coincident fine cells;
- cell-weighted planetary statistics on the adaptive mesh;
- selected river-cell support area interpreted as physical river area;
- within-world normalized aridity interpreted as absolute cross-world dryness;
- cartographically exaggerated geometry used as physical slope evidence;
- biome labels used to validate their climate inputs.

Disposition: **remove from canonical evaluation**; retain only explicit negative
evidence where historically useful.

## Tectonic/orogen family

### Product baseline

`Legacy` efficiently preserves tectonic setting and produces useful geography,
but it is a calibrated boundary-response terrain hack rather than geological
evolution.

Disposition: **retain honestly; validate** range placement, polarity, belt width
and morphology. Do not promote its plausible peaks as physical validation.

`LegacyYield` contains a reusable conservative yield-relaxation operator and may
remain one bounded candidate.

Disposition: **validate/park** until visual and range-object evidence exists.

### Thirteen-model public ladder

`OrogenModel` exposes thirteen variants through product CLI, diagnostic CLI,
cache identity, provenance and downstream match statements
(`elevation.rs:47-85`, `main.rs:20-86`, `diagnose.rs:17-42`). Most are parked or
falsified research rungs. The scorecard itself primarily reasons about legacy,
evolved carrier and lifecycle categories.

Disposition:

- retain `Legacy` in product configuration;
- possibly retain `LegacyYield` as an explicit candidate;
- **quarantine** all other selection behind research-only configuration/binaries;
- after extracting reusable operators, **remove named selectable models** whose
  hypotheses are falsified.

### Product pays for unused history/work

`World::generate_features` unconditionally computes `TectonicHistory`.
Non-carrier/default history reconstructs plate-seed adjacency through time, and
`FeatureFields` builds integrated work/episode maps even though legacy elevation
consumes only product arc/collision fields (`world/mod.rs:298-329`,
`history.rs:219-275,456-501`, `features.rs:274-310`,
`elevation.rs:681-695`).

Disposition: **simplify** through lazy/model-gated history and work generation.
Benchmark before claiming the saved time, but product architecture should not
own unused experiment state.

### Omnibus `FeatureFields`

One type owns product trench/arc/ridge/collision fields, conserved work,
episodes, thin-sheet state, carrier audits, lifecycle crust/age/weakness and
ledgers. Legacy construction allocates many zero experiment vectors
(`features.rs:41-123,945-1031`).

Disposition: **split** into product `BoundaryFeatureFields` and optional
experiment/lifecycle outputs. Absent experiments should allocate no state.

### Operators worth preserving

Preserve independently tested mechanisms even when their named terrain model is
removed:

- conservative graph scalar/yield relaxation;
- carrier scalar/vector projection;
- face-flux/CFL transport;
- deterministic overlap/gap resolution;
- explicit material ledgers;
- bounded underthrust and magma placement.

The history/deformation/features/elevation family is roughly 8,200 lines with
substantial invariant testing. That evidence earns an operator library, not a
permanent thirteen-way world-model surface.

## Fine substrate and erosion

### Defensible active core

| Mechanism | Disposition | Open evidence |
|---|---|---|
| Adaptive fine tessellation | **Retain/validate** | Compare current density prior with simpler equal-budget meshes |
| Coarse-to-fine field transfer | **Retain/validate** | Register intensive/content/flux semantics; current interpolation is not generally conservative |
| Pre/post fine hydrology | **Retain** | Memory cost and cross-stage identity |
| Priority flood and convergent flat routing | **Retain** | Resolution/topology controls |
| Precipitation-weighted SFD incision | **Retain/validate** | Step count, exponent and causal uplift/rainfall response |
| Linear hillslope relaxation | **Validate/simplify** | Visual/structural gain per Jacobi sweep |
| Geological erodibility proxy | **Validate** | Must cause craton/arc-conditional differences, not only global texture |
| Terminal sink fill | **Validate/remove if neutral** | Visible effect and honest material ledger |
| Fine cache | **Retain as tooling** | Keep separate from model architecture |
| Full pre and eroded surfaces retained | **Validate/simplify** | Interaction value versus peak/retained memory |

The reference corpus median is about 62 seconds in erosion versus roughly 3.1
seconds lithosphere, 1.6 atmosphere and 7.5 fine-pre. This establishes urgency,
not that erosion is intrinsically too expensive.

### Inactive experimental superstructure

Product structs and the main erosion path still carry default-off branches for:

- MFD routing;
- nonlinear hillslopes and confinement;
- uplift smoothing;
- en-route deposition;
- synthetic/structural lithology;
- climate–erosion feedback and lake evaporation;
- glacial sculpting;
- fault scarps, interior grain, strike bands and margin contrast;
- emergent demotion/rebuild and structured uplift;
- meso relief;
- drainage pulse/burn-in.

Together these contribute roughly 30 `ErosionParams`, 11
`FineStructureParams`, a large override CLI, cache-key fields and multiple
overlapping shape owners. They are inactive, not free.

Disposition: **quarantine** by moving them out of product parameter types and
primary CLI/help. Preserve an explicit experiment configuration only while its
evidence remains useful.

### Overlapping mountain-shape owners

Potential owners include coarse tectonic elevation, adaptive density, painted
fine scarps/grain/meso relief, emergent rebuilt uplift, and erosion/diffusion.
Zero defaults currently limit active overlap, but the architecture invites
stacked tuning.

Disposition: enforce:

- tectonics owns broad forcing/envelope/history;
- landscape evolution owns geomorphic organization;
- semantics owns range/plateau identity;
- presentation owns visibility.

Painted fine relief may remain a declared alternative authentic hack, not a
silent additive owner beside emergent uplift.

### Claims requiring correction

- `steps=200` and `dt=1` have no geological time unit; active erosion is a
  calibrated landscape operator.
- legacy uplift reinjects tectonic thickening per model step; it is
  hold-and-carve, not a shared tectonic/erosion clock.
- emergent rate self-calibrates from target/epoch, making “time” a build/carve
  dial.
- terminal lake levels are frozen from pre-erosion state, not dynamically
  coupled basin evolution.
- inverse-distance transfer is reasonable for intensive fields but does not
  establish conservative transfer of thickness/work/uplift quantities.
- continentality and arc/convergence produce a plausible erodibility proxy, not
  lithologic units or stratigraphy.

Disposition: keep fidelity labels at physically inspired/authentic reduced
model until controlled scaling and budget tests pass.

### Likely active-path performance debt

Code inspection identifies several ablation/optimization candidates:

- `deposit()` copies an N-cell sediment vector every step
  (`erosion.rs:2405-2408`);
- diffusion allocates roughly five N-cell vectors per step and performs six
  Jacobi sweeps (`erosion.rs:2181-2230`);
- immutable CSR geometry is cloned into `ErosionState`
  (`erosion.rs:763-768`);
- both full surfaces remain resident (`fine.rs:370-419`);
- transfer materializes an all-cell temporary struct before splitting fields
  (`fine.rs:2880-2969`);
- neighbor geometry construction creates small vectors per cell before
  flattening (`erosion.rs:2474-2507`);
- roughness diagnostics may execute inside generation depending on logging.

Disposition: **profile and simplify** after semantic ablations. Buffer reuse and
shared immutable geometry should be arithmetic-identical, low-risk wins if
phase timing confirms them.

## Atmosphere, climate and wind

The atmosphere/circulation/moisture family is about 1,700 lines. It contains a
real finite-volume moisture transport with CFL limiting, projected winds and
orographic/convergence rainout. Precipitation feeds hydrology, erosion and
ecology; wind also feeds visualization. This is not decorative-only complexity.

However, its added value over cheaper climatology has not been isolated. Missing
energy balance, seasonality, ocean heat transport and physical precipitation
units constrain its claim.

Disposition: **validate**, not remove. Run:

1. full model versus latitude/elevation + coast-distance precipitation baseline;
2. normal versus reversed wind over a fixed barrier;
3. topography/orographic terms on/off;
4. transported versus uniform runoff effects on rivers/lakes;
5. wind particles off while retaining physical wind consumers.

If transport does not create stable conditional rain-shadow, interior-drying
and hydrologic consequences, **simplify** to the cheaper authentic hack. Do not
add seasons/ocean circulation to rescue an unvalidated base.

GPU wind particles have spectacle value but no model effect. Their value should
be judged by frame/memory cost and human acceptance independently of atmosphere
correspondence.

Disposition: **validate as presentation**; never use particles to justify the
wind solver.

## Hydrology and semantics

Hydrology's topology, basins, spill relationships and final river network have
high downstream leverage. Retain the core. Basin integration, breach protection
and climate-ratio equilibrium are authentic hacks whose sensitivity should be
tested, not removed because they are non-transient.

Disposition: **retain/validate** core topology and isolate hacks through frozen
terrain rainfall/evaporation controls.

Replace resolution-dependent selected-cell metrics with catchment, reach,
mouth, basin and topology measures. Do not label flow accumulation as discharge
without runoff/time semantics.

`EcologySemantics` has no product renderer or physical feedback consumer. It is
a useful 347-line diagnostic prototype whose categorical calibration depends on
unvalidated climate.

Disposition: **quarantine as diagnostic semantics**; prevent product
dependencies and avoid calibration work until climate/landforms pass.

## Diagnostics, backends and stage architecture

### `diagnose`

The 4,000-line binary mixes registered measurements, known-invalid probes,
historical experiment controls and prose tables. Its content is valuable; its
ownership is not scalable.

Disposition: **split/simplify**. Move promoted metric implementations into
library adapters consumed by corpus and diagnostics. Keep experimental reports
explicitly unregistered.

### Dual Voronoi backends

The alternative backend is useful only while it answers robustness,
performance or integration questions. Maintaining an external dependency and
two generation paths indefinitely has cost.

Disposition: **validate** with topology, timing and downstream invariance. If it
does not serve an active product/research need, **quarantine/remove** from normal
CLI while retaining a dedicated integration benchmark.

### Stages and retained state

Pre/post snapshots are useful for inspection, but numbered stages risk implying
a permanently linear architecture. Full retained surfaces may be expensive at
the original 8M guardrail.

Disposition: **retain snapshot semantics; validate/simplify storage and naming**.
Future feedback should not be forced into additional numbered stages.

## Disposition matrix

| Area | Provisional disposition | Confidence |
|---|---|---|
| Computational geometry / primary Voronoi backend | Retain | High |
| Alternative Voronoi backend | Validate then quarantine/remove if purposeless | Medium |
| Rigid plate/Euler kinematics | Retain | High |
| Legacy orogen | Retain honestly, validate morphology | High |
| LegacyYield | Validate/park | Medium |
| Conserved/history/thin-sheet named terrain ladder | Quarantine; remove falsified named paths after operator extraction | High |
| Carrier/lifecycle operator library | Retain research operators | High |
| Lifecycle complete model | Quarantine fixed-pole engine | High |
| Adaptive fine substrate | Retain, validate allocation | High |
| Product SFD erosion core | Retain/validate/simplify | High |
| Default-off erosion/fine experiments | Quarantine | High |
| Sediment sink fill | Validate/remove if neutral | Medium |
| Atmosphere/moisture transport | Validate against cheap baseline | Medium |
| GPU wind particles | Validate presentation payoff | Medium |
| Hydrology core | Retain | High |
| Basin/lake integration hacks | Retain/validate sensitivity | Medium-high |
| Ecology prototype | Quarantine diagnostic | High |
| Shared water/river semantics | Retain | High |
| Layered rendering path | Removed; Windows runtime check pending | High |
| Legacy CPU wind draw path | Removed | High |
| Diagnostic monolith | Split/simplify | High |
| Full pre/post retained surfaces | Validate memory/payoff | Medium |

## Ordered ablation and cleanup queue

### Safe cleanup first

1. **Completed:** remove layered pipelines/mesh/shader and legacy CPU wind draw;
   run the remaining Windows runtime check.
2. Correct lifecycle/history naming and registry claims.
3. Hide research orogen rungs and parked erosion/fine controls from product CLI.
4. Split product feature state from optional experiment outputs.

### Cost/value ablations

1. **Screened:** erosion steps 50/100/200 on seed 12345 and 8675309. The
   [screening audit](audits/erosion-core-screen-2026-07-12.md) rejects 50/100 as
   neutral simplifications.
2. **Screened:** stream exponent `n=1` versus `n=2`; it is a materially different
   morphology, not a drop-in speed optimization.
3. Diffusion off/default and 2 versus 6 Jacobi sweeps.
4. Geological erodibility strength zero versus default.
5. Sink deposition zero versus default.
6. Equal-budget simple versus current adaptive-density prior.
7. Full climate versus cheap climatology; wind/orography controls.
8. Uniform versus climate-weighted runoff on frozen terrain.
9. Lifecycle constant versus actually reorganized motion, only after landform
   objects exist to measure consequences.

### Memory/allocation work

Measure peak RSS and phase allocations, then reuse sediment/diffusion buffers,
share immutable geometry and test eroded-only retention. Arithmetic-preserving
optimizations do not need new physical justification, but still require
byte/metric equivalence.

## Stop rule

No new mechanism should enter product configuration without naming what it
replaces. Promotion should normally reduce or strengthen the active path. A new
default added beside every prior owner is accumulation, not progress.
