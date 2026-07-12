# Experiment registry

This registry separates current product behavior from implemented alternatives.
It summarizes code and recorded evidence as of 2026-07-12; it is not a substitute
for the linked audits. Status terms follow the
[documentation policy](documentation-policy.md).

## Product baseline

| Area | Default/product state | Evidence/status |
|---|---|---|
| Orogen model | `legacy` | Explicit enum and CLI default; promoted product baseline |
| Fine terrain | Coarse physical envelope transferred without optional P1/meso base additions | All structural master amplitudes default to zero |
| Drainage | Barnes flat resolution + SFD | Product default; MFD exponent is zero |
| Erosion | 200 steps, stream power `m=0.5`, `n=2`, linear hillslope diffusion, 4 km² channel support | Product reset baseline; numerical maturity, not physical time |
| Uplift | Direct hold-and-carve source with scale `0.003` | Product baseline pending shared geological clock |
| Optional surface additions | Critical-slope, confinement, en-route deposition, lithology, climate feedback, glacial and drainage pulse neutral | Master controls default to zero; not product-active |
| Relief | `Authentic` code preset at scale `0.04` | Promoted cartographic presentation; not physical scale |
| River width | Renderer-only screen-space policy | Promoted separation; topology remains modeled |

## Orogen and tectonic-time models

All rows below are implemented and CLI-selectable through `--orogen-model` unless
stated otherwise. Only `legacy` is the default.

| Model | Mechanism/hypothesis | Current maturity | Registry decision |
|---|---|---|---|
| `legacy` | Historical arc/collision response within thickness/isostatic elevation assembly | Implemented, selectable, evaluated, promoted, default | Product control and baseline |
| `legacy-yield` | Conserved spreading only above a gravitational yield elevation | Implemented and numerically evaluated on limited seeds; visual acceptance recorded as pending in its source spec | Candidate/parked; do not imply promotion |
| `conserved-local` | Boundary crust volume retained at receiving cells | Implemented experimental isolator | Parked diagnostic rung |
| `conserved-feature-footprint` | Conserved volume redistributed through legacy feature footprint | Implemented experimental bridge | Parked diagnostic rung |
| `conserved-isotropic` | Conserved work redistributed by isotropic thin-sheet diffusion | Implemented and evaluated during model ladder | Parked; insufficient product shape evidence |
| `history-local` | Physical-rate episode work integrated over derived duration, no lateral relaxation | Implemented and falsified for promotion in recorded tectonic-time tests | Retain as diagnostic rung |
| `history-diffusive` | Episode work plus dimensioned lower-crust diffusion | Implemented and falsified for promotion | Retain as diagnostic rung |
| `history-material` | Episode work placed in finite-strain material footprint then diffused | Implemented and falsified for promotion | Retain only while useful to ongoing history research |
| `thin-sheet` | Velocity/continuity thin-sheet T0 prototype | Implemented, selectable, not promoted | Parked research prototype |
| `history-thin-sheet` | Physical-rate thin-sheet integrated over boundary episodes | Implemented, evaluated, not promoted | Active research rung only |
| `history-carrier-thin-sheet` | Thin sheet driven by moving low-resolution carrier history | Implemented, evaluated, not promoted | Active research rung only |
| `history-carrier-evolved` | Deformation accumulated on moving material parcels and projected to present | Implemented; conservation passes but resolution and morphology promotion gates fail | Falsified as product candidate in current form; keep for operator research |
| `history-carrier-lifecycle` | Forward carrier automaton with discrete ownership, continuous boundary work, underthrust/arc reservoirs and conservative ledgers | Implemented and checkpointed in `70ba33e`; improves height behavior and remap causality, while cross-resolution land/peak gates and broad-plateau outliers remain open | Causal engine retained; experimental, not promoted |

The experiments above contain reusable mechanisms even where their combined
terrain hypothesis failed. Conservation success does not override spatial
non-convergence, excessive relief, fragmented morphology or missing visual
acceptance.

### Carrier/lifecycle subexperiments

| Experiment | Status | Outcome |
|---|---|---|
| Carrier resolution sweep | Numerically evaluated | Failed: deformation concentration and morphology do not converge with carrier resolution |
| Exact boundary-velocity constraint | Falsified and reverted | Increased resolution divergence and peak concentration |
| Fixed-width boundary band | Falsified and reverted | Did not close promotion behavior |
| Same-clock coarse denudation | Implemented behind zero-default carrier control; evaluated | Bounded but insufficient; not a promoted erosion model |
| Finite-memory/coherent Euler motion | Implemented behind zero default; evaluated | Useful causal experiment, not a promotion; hard cutoff/reorganization shortcuts rejected |
| Forward lifecycle material automaton | Implemented and invariant-tested | Material/topology causal engine passes; direct full-column surface expression fails |
| Capacity-limited underthrust and arc sheets | Implemented and retained in lifecycle research | Improves causal allocation but does not close convergence/promotion gates |
| Binary nearest-cell lifecycle event pullback | Falsified | Resolution-dependent event measure |
| Fractional face work | Implemented | Better event accounting; lifecycle still not promotable |

### Post-`9849e4a` lifecycle checkpoint

Commit `70ba33e` contains a later hybrid architecture than checkpoint `9849e4a`
(`Require convergent lifecycle reactions`). It separates:

- discrete pullback for surface ownership and plate topology;
- continuous face-swept collision, subduction and spreading work;
- explicit conservative remap-displacement ledgers;
- bounded sheet/batholith placement for buried crust and magma;
- CFL-limited integration and deterministic collision merging.

Two fuller Eulerian alternatives were implemented experimentally and rejected:
one collapsed plate identity through material mixing; the other diffusively
smeared deformation globally with substantially higher runtime.

Recorded results for the retained hybrid show seed-12345 carrier-resolution
peaks of 12.55/9.23/10.64 km at 4k/8k/16k and land coverage of
38.3/30.0/29.7%. All ten canonical 8k seeds pass the peak-height gate, but two
retain broad mountain coverage (41.4% and 49.8% of land), and the strict
cross-resolution gates remain open. Material ledgers close near roundoff and
remap stacking is no longer the primary suspected cause.

The next local tectonic hypothesis is time-varying motion history: the forward
lifecycle solver currently uses constant Euler poles through the full lookback,
so a 25 Myr motion-coherence setting is bit-identical to infinite coherence.
That is a well-isolated continuation if this research resumes, but it is not the
active project roadmap priority while the observability/semantic foundation is
built.

Primary evidence: `docs/archive/specs/tectonic-time.md`,
`docs/archive/specs/thin-sheet-orogeny.md`, and
`docs/audits/tectonic-promotion-scorecard-2026-07-11.md`. These remain historical
logs; this registry owns current status wording.

## Fine substrate and erosion experiments

The product reset parks these mechanisms by setting their master controls to
zero. “Parked” means code remains selectable for controlled work; it does not
mean the underlying scientific mechanism is rejected.

| Mechanism | Control/default | Evidence maturity | Current status |
|---|---|---|---|
| Fault-front scarp/base offset | `FAULT_SCARP_HEIGHT = 0` | Implemented and previously visually/numerically explored | Parked; requires a justified fault owner and scale |
| Interior fine relief | `FINE_INTERIOR_RELIEF = 0` | Implemented/evaluated in fine-synthesis work | Parked procedural substrate |
| Strike-aligned bands | `FINE_FRONT_STRIKE_WEIGHT = 0` | Implemented/evaluated | Parked; orientation is grounded, relief amplitude/role not promoted |
| Active/passive margin contrast | `FINE_MARGIN_CONTRAST = 0` | Implemented/evaluated | Parked |
| Emergent orogen demotion/rebuild | `FINE_EMERGENT_LAMBDA = 0` | Implemented; original premise and several regimes evaluated | Parked/falsified as replacement premise under current solver |
| Structured O0 uplift | `FINE_EMERGENT_STRUCTURED = 0` | Implemented/evaluated | Parked; ownership overlaps accepted envelope/uplift |
| Meso uplift modulation | `FINE_MESO_RELIEF = 0` | Implemented, numerically promising in composed regimes, visual acceptance incomplete | Parked candidate |
| Meso base relief | `FINE_MESO_BASE_RELIEF = 0` | Implemented experiment | Parked as prescribed geometry |
| Drainage-aware uplift pulse | `EROSION_DRAINAGE_PULSE = 0` | Implemented and numerically evaluated; visual acceptance incomplete | Parked; risks circular ownership |
| Nonlinear critical-slope hillslopes | `EROSION_HILLSLOPE_CRITICAL_SLOPE = 0` | Physical mechanism implemented; isolated reset effect nearly inert | Parked until mesh/parameter regime justifies it |
| MFD routing/incision | `EROSION_MFD_EXPONENT = 0` | Implemented experimental route split; ladder incomplete | Parked; SFD remains product |
| Alluvial confinement gate | `EROSION_CONFINEMENT_SLOPE = 0` | Implemented experiment | Parked |
| Uplift-source smoothing | `EROSION_UPLIFT_SMOOTH_KM = 0` | Implemented and A/B evaluated | Parked experiment |
| En-route deposition | `EROSION_DEPOSITION_SLOPE = 0` | Implemented operator/ledger; product calibration absent | Parked; terminal sink fill remains separate |
| Synthetic lithologic variation | `EROSION_LITHO_SIGMA = 0` | Implemented | Parked pending grounded material field and payoff |
| Structural-grain erodibility | `EROSION_LITHO_GRAIN_STRENGTH = 0` | Implemented experiment | Parked pending demonstrated drainage/landform consequence |
| Orographic fine feedback | `OROGRAPHIC_PRECIP_STRENGTH = 0` | Implemented/evaluated in feedback work | Parked; retain architecture |
| Downwind rain shadow | `DOWNWIND_SHADOW_STRENGTH = 0` | Implemented/evaluated; reclassified as climate/river rather than mountain feature | Parked but potentially valid hydrology candidate |
| Lake humidity boost | `LAKE_EVAP_STRENGTH = 0` | Implemented | Parked |
| Glacial abrasion | `GLACIAL_K = 0` | Implemented early process hack | Parked; incomplete glacier morphology/physics |

The nonzero `EMERGENT_REBUILD_GAIN` does not activate emergent rebuilding while
its controlling demotion/structure path is neutral. Likewise supporting shape,
snowline or iteration constants do not make a zero-master mechanism active.

Primary product-reset evidence: `docs/archive/specs/terrain-reset.md`. Individual old
specs remain experiment history until archived.

## Hydrology experiments and accepted corrections

| Mechanism | Status | Registry decision |
|---|---|---|
| Basin integration/outlet carving | Implemented, evaluated and promoted | Product correctness/topology mechanism |
| Lake-aware breach protection and basin-aware carving | Implemented, evaluated and promoted | Product mechanism preserving lakes with outlets |
| Climate-ratio lake equilibrium control | Implemented and product-interactive | Modeled hydrology control, not presentation-only |
| Pluvial overflow selection criterion | Evaluated and rejected | Falsified as default because it over-integrated basins |
| Mega inland-sea correction | Problem observed; no accepted model | Proposed/open, not implemented product behavior |
| Hydrologic wetland objects | Not implemented | Proposed/absent; ecological semantics now expose only a provisional wetland suitability potential |

## Presentation experiments

| Mechanism | Status | Registry decision |
|---|---|---|
| Relief scale `0.20` (~127× in recorded comparison) | Evaluated and rejected as default | Historical dramatic distortion; not evidence against terrain |
| `Physical`, `Authentic` 0.04 and `Dramatic` 0.08 presets | Implemented/selectable; 0.04 promoted default | Product presentation policy, with terminology caveat |
| Screen-space draped river SDF | Implemented and promoted | Product Relief/wind river path; performance/generalization debt remains |
| Legacy world-space/fixed-texture river width behavior | Corrected | Falsified presentation interpretation; do not restore |
| Displaced-facet slope shading | Implemented and used by sweeps, not interactive Relief | Selectable internally/incompletely integrated; decision pending |
| Older layered shader path | Appears unused/stale | Verify, then remove or explicitly revive; not product capability |

## Promotion requirements

The full gate is defined by the [validation policy](validation.md). In summary,
an experiment may be proposed for product promotion only when its record states:

1. physical or authentic role and ownership;
2. exact active controls and default interaction;
3. isolated same-seed A/B against the current product baseline;
4. relevant invariants and numerical evidence;
5. cross-seed behavior;
6. resolution and cache/provenance conditions;
7. compute/memory change;
8. downstream consequences;
9. controlled visual review when appearance is part of its purpose;
10. which prior mechanism it replaces, if ownership would otherwise overlap.

Promotion should normally simplify or strengthen the product path. Adding a new
default alongside every mechanism it was meant to replace is not promotion; it
is accumulation.
