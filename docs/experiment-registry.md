# Experiment registry

This registry separates current product behavior from implemented alternatives.
It summarizes code and recorded evidence through 2026-07-15; it is not a substitute
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

### Orogen-organization proposals

These experiments address the diagnosed product morphology without assuming
that all broad plateaus are defects. Their implementation and disposition are
tracked independently below; none is promoted product behavior.

| ID | Isolated hypothesis | Required budget/control | Status |
|---|---|---|---|
| O1 | A peaked or asymmetric convergent-belt cross-section prevents the smooth cap from becoming the universal range grammar | Tested `legacy-peaked`: compact continental arc/collision profiles, positive work conserved separately per plate and feature type; oceanic arcs, erosion and presentation unchanged | Rejected after numerical and human visual review; implementation removed, evidence retained |
| O2 | Causally conditioned along-strike variation creates legitimate saddles and differentiated segments | Tested `legacy-segmented`: exact legacy cross-section; deterministic 400-km-smoothed convergence-orthogonality strength modifier; response conserved per connected boundary episode, feature type and receiver plate | Rejected after numerical and human visual review; implementation removed, evidence retained |
| O3A | Regional organization needs a structure-first joint massif/saddle/corridor scaffold rather than more scalar-envelope variation | Existing `MassifCorridor` base path alone at 0.05 versus legacy and P1a 0.0611; perturbed arms matched at 44 m structural RMS; unchanged common erosion/integration/presentation | Implemented/evaluated on seed 12345; provisionally negative pending human review: improves flat-cap coverage but does not beat isotropic on drainage organization or final relief |
| O3B | Regional organization should emerge from a frozen drainage-first scaffold | Isolate the existing A4 burn-in/trunk/interfluve idea from its parked emergent/O0 stack; compare the same objects/budgets as O3A | Paused as a product candidate after research reframe; retain as a synthetic-topology control for the organization-owner testbed |
| O3C | An explicit persistent crest/divide graph is necessary | Joint positive and negative objects, zero-net authoritative redistribution, not crest-only; compare with O3A/O3B | Reframed as the explicit-skeleton structural upper bound in the organization-owner testbed, not a presumptive product implementation |
| O4 | Existing erosion can dismantle a broad cap when repeated rebuild no longer suppresses reorganization | Fixed current cap; rebuild off/reduced; bounded drainage/base-level perturbation | Folded into the locked hold-and-carve/control family; secondary to the coupled representation decision |

Success requires authoritative pre/final morphology and at least one downstream
hydrology or climate organization signature to improve under the declared
budget. Peak count or render appeal alone is not sufficient. Research basis:
[`research/orogen-organization-2026-07-12.md`](research/orogen-organization-2026-07-12.md).

The zoomed-out disposition supersedes O3B/O3C as separate presumed next
implementations. The next planned discriminator is one organization-owner slice
with three comparable families: locked hold-and-carve H, reduced coupled C and
graph-first authentic G. Its shared admissible inputs, arm-neutral opportunity
calibration, resource ceiling and independent object extraction must be
preregistered before implementation. Status: planned discriminator;
comparison family selected, no arm implemented, evaluated or promoted. See the
[landscape organization strategy](landscape-strategy.md). The
[landform object packet v0](research/landform-object-packet-v0-2026-07-14.md)
now preregisters the common evidence vocabulary and rung order. It does not yet
change the status of any comparison arm. Its first executable
[G0/S0 contract](research/landform-object-packet-g0s0-2026-07-14.md) is now
implemented and passes its manufactured planar/spherical gates plus the first
unchanged 250k product observation. The
[D0 common drainage contract](research/landform-object-packet-d0-2026-07-15.md)
is now implemented and passes its bounded planar manufactured matrix. This
validates this bounded common evaluation implementation only, not product
hydrology or any comparison arm. The
[O0a relationship-probe contract](research/landform-object-packet-o0-2026-07-15.md)
is now preregistered; O0b correspondence and packet assembly remain
unregistered, and no comparison arm is promoted.

The first O1 run at 100k coarse/250k fine preserves peak and mountain-area
plausibility and modestly increases 25–100-km relief, but increases median/p90
summit-cap area and barely changes drainage orientation. It should not be tuned
or promoted on this evidence. Human visual review agreed; the selectable rung
was removed to avoid accumulating failed models. Matched packets are recorded in
`artifacts/captures/seed-12345-o1-peaked-ab/` and
`artifacts/captures/seed-12345-o1-peaked-range-ancestry/`.

The corrected first O2 run uses connected boundary episodes as its work owner;
an earlier per-fragment ledger was discarded because singleton fragments
cancelled their own modifiers. The retained experiment changes 95.9% of eligible
continental response cells, with p02/p98 effective multipliers near 0.959/1.041
and maximum deviation 10.3%. Despite that causal signal, cap metrics are mixed,
elongation declines slightly, drainage orientation is stable and matched terrain
remains very similar. Human review agreed; the rung was removed rather than
parked. Evidence:
`artifacts/captures/seed-12345-o2-segmented-ab/` and
`artifacts/captures/seed-12345-o2-segmented-range-ancestry/`.

The first O3A run isolates prescribed base geometry from the historical
emergent/O0/gain stack. `MassifCorridor=0.05` and isotropic P1a `0.0611` each
produce 44 m area-weighted structural RMS over the same process footprint.
MassifCorridor improves low-grade cap coverage and fragments basin topology less
than isotropic structure, but final relief, passes, river hierarchy and
strike-relative drainage do not show a consistent structured advantage. Matched
ancestry confirms the candidate exists in the pre-hydrology base and that common
erosion reduces the distinction. Evidence:
`artifacts/captures/seed-12345-o3a-organization/`,
`artifacts/captures/seed-12345-o3a-isotropic-ancestry/` and
`artifacts/captures/seed-12345-o3a-massif-corridor-ancestry/`.

The subsequent [mountain-system design basis](research/mountain-system-design-basis-2026-07-13.md)
changes the next decision. O3B should not be tuned or promoted in isolation:
reality and successful reduced terrain systems organize mountains through
heterogeneous time-varying forcing coupled to drainage, hillslopes and inherited
materials. Historical A4 remains useful as the cheap synthetic-topology control
against a reduced coupled landscape architecture and an explicit skeleton upper
bound.

The comparison contract is now frozen at the design level in the
[bounded orogen organization testbed](research/orogen-testbed-spec-2026-07-13.md).
It compares five representation families under common deformation work,
boundaries, runoff, material inputs and time: hold-and-carve, synthetic
drainage-first, reduced coupled landscape evolution, coupled evolution with a
shared province, and an explicit joint skeleton. The first implementation slice
contains only the dimensioned coupled U/L null cases and numerical ledgers; it
does not constitute an O3 promotion attempt.

Slice 1 is now implemented and screened. U/L forcing budgets, deterministic
routing, timestep response and ledgers pass, and full 10-Myr runs remain stable.
The representation fails its resolution gate: incision export decreases
strongly from 8→4→2 km and the discrepancy accumulates into roughly 50% final
relief differences between 8 and 4 km. The subsequent
[channel/surface decision](research/channel-surface-scaling-2026-07-13.md)
identifies a path-elevation/cell-mean semantic collision rather than a missing
calibration constant. The registered next rung is C0: cell-mean finite-volume
surface, face-flux/MFD routing, specific-discharge-driven effective areal
denudation, and resolution-stable outlet portals. It must pass analytic
boundary, routing, depression and manufactured-denudation tests before U/L is
rerun. Slice 2 remains blocked. See the
[Slice 1 audit](audits/orogen-testbed-slice1-2026-07-13.md).

Slice 1R has now implemented the first analytic pieces without changing U/L.
The P/pathway equilibrium, genuine full-hex boundary geometry, unfilled MFD
water conservation, plane/ridge specific-discharge convergence and isolated C0
manufactured denudation pass. The exact-rectangle boundary attempt was rejected
before use because its nominal faces did not match the full-cell finite-volume
domain. A separate portal-seeded fill and BFS flat potential now pass exact-flat
and nested-sill controls without mutating physical elevation; the exact linear
Dirichlet fixture also passes. Physical mean-surface gradients now pass affine
and smooth-radial controls. Genuine-boundary hillslope conservation now passes
with an explicitly linear external Dirichlet control. Radial/convergent
specific discharge, flow-aligned physical grade and coupled C0 remain blocked.
See the
[Slice 1R audit](audits/orogen-testbed-slice1r-2026-07-13.md).

The radial/convergent gate now passes away from the convergence locus but
shows that `|q_vector|` is not pointwise two-sided throughput there: it retains
only 16.1% of the chosen line-sink strength at 2 km. Candidate RMS/L1/support-
width replacements lack an invariant normalization at that singularity and are
not promoted. Retain the consistent LS vector/magnitude for resolved continuum
flow, validate integrated cut flux and a broad downstream reach, and record the
exact sink as a C0 truth limit. Do not tune `K` or introduce a local fallback.

The resolved broad downstream-reach control then passes: integrated cut flux
closes and local support-corrected vector/magnitude errors fall below 0.5% at
2 km in aligned/rotated cases. C0-V is retained for resolved continuum flow;
the exact line sink remains a declared truth limit. A separate coupled C0 arm
may now proceed without RMS/L1/cell-width switching.

That separate C0 solver is now implemented and passes transactional composition,
retry, no-pinning and elevation-volume-moment ledger controls. Its effective
denudation coefficient remains zero by default pending a dimensioned regime;
manufactured spatial/temporal convergence and a fluvial slope-response limit
remain prerequisites for short U/L. Implementation is not promotion.

The first frozen-regime 0.1 Myr U/L spatial screen fails. The 8/4 km responses
remain near 32–35 m relief, but 2 km reaches 87 m (U) and 180 m (L), with
maximum effective denudation increasing to 7.8/19.3 km/Myr and adaptive steps
collapsing. Water and solid ledgers close and no sinks occur. Cell diagnostics
then locate an open-base physical-grade defect and the already disclosed absent
fluvial slope CFL; smaller dt does not fix the boundary feedback. This invalid
screen is retained as diagnostic evidence rather than a C0 falsification.

The unchanged corrected rerun uses face-consistent portal grade and an enforced
slope Courant limit. U/L relief is tightly stable across 8/4/2 km (about
31–32 m), ledgers close within `1.9e-10 km³`, water closes and the 2 km runaway
is absent. Peak cellwise denudation nevertheless grows `4.4–5.1×`, export is
not yet asymptotic, and the accepted timestep scales approximately as `h²`.
Status: numerical correctness promoted; unregularized local support and compute
remain open. Next compare one fixed-physical-support C0 arm with the minimum C1
state before any 1/10 Myr run or parameter tuning.

That comparison is now evaluated under the frozen C0-Q16 preregistration.
Fixed-16-km supported intensity stabilizes q maxima to within roughly `3–6%`
per refinement, keeps relief stable and holds export drift below `0.9%`; raw
water and solid ledgers close. It is not promoted: export changes `20–24%` at
fixed `K`, 2 km global-filter runtime remains near unfiltered C0, and isotropic
support has no drainage-divide ownership. Status: retain Q16 as diagnostic
evidence and advance only a manufactured minimum C1 `{z_bar,z_c,f_c}` volume-
mixing fixture. Long U/L and product integration remain blocked.

The minimum C1 fixture now passes that gate. A 128 km × 0.2 km prescribed reach
retains identical `sum(wL)=25.6 km²`, channel evolution and `2.56 km³` export
at 8/4/2 km; cell-mean volume closes, internal compartment transfer cancels and
zero width is an exact C0 reduction. Status: promote `{z_bar,z_c,f_c}` only as
the next isolated routed-testbed representation. Network/reach and physical
width ownership remain absent; sediment, valleys, ecology, long U/L and product
integration remain blocked.

The prescribed routed C1 fixture also passes. At 8/4/2 km it retains 288 km of
stable reach identity, 48 km² channel area, exact registered outlet allocation
and invariant `0.0002018352 km³` manufactured export. A B:C→D capture preserves
all C1 state bits and shifts only downstream C/D flow/response; overlap remap
preserves compartment moments without cross-reach transfer. Status: semantic
receiver graphs are a viable isolated C1 state owner. Network generation and
width evolution remain unselected; long U/L, sediment/ecology and product
integration remain blocked pending that owner decision.

The subsequent ownership review selects a layered testbed architecture rather
than one exclusive owner. Conservative continuum flow owns instantaneous water
supply; a dimensioned discharge-slope/resistance closure proposes active
channel support; a sparse active/dormant reach graph owns lineage and C1 state.
Semantic importance and presentation stay derived. Status: architecture
selected for one manufactured discriminator, not implemented or product-
promoted. Compare snapshot rebuilding with hysteretic correspondence under
sub-cell receiver jitter and one real capture, using a prescribed skeleton only
as the upper-bound control. Physical initiation parameters, width evolution,
long U/L, sediment/ecology and product integration remain blocked. See the
[drainage-network ownership decision](research/drainage-network-ownership-2026-07-13.md).

The narrower M0 implementation now passes prescribed-observation hysteresis,
unique-anchor correspondence, one-capture, invalid-cycle and unchanged-interval
C1 composition controls at 8/4/2 km. It deliberately stops before reach
birth/retirement because those require explicit attached-state ledgers. Status:
retain as isolated mechanism evidence only. Since channel support and physical
anchors are prescribed and S0 crosses the initiation threshold by construction,
M0 does not execute the preregistered discriminator, demonstrate continuum-
derived network convergence or show that persistence earns its cost. The next
decision is the MFD-to-sparse-candidate extraction rung, compared with
production SFD extraction and the prescribed control. Evidence:
[M0 audit](audits/channel-ownership-memory-m0-2026-07-13.md).

The first state-free R0 implementation attempt invalidated its own
discriminator before a valid arm comparison. The registered centre-to-centre
V length gate is mathematically unattainable on the chosen regular hex
orientation (`1/cos(30°)` gives the observed 15.47% asymptotic excess), P0 and
M0 are aliased by equal face widths, and the draft Y construction is not smooth
at the junction. Status: invalid/inconclusive; no implementation retained and
no arm selected or rejected. M1 remains an unearned extra full-domain mechanism,
not a falsified one. Next preregister a narrow irregular-S2-Voronoi comparison
which separates path ownership from face-crossing/within-cell geometry. It
still cannot promote initiation, width, lineage or C1 state. Evidence:
[R0 specification](research/channel-extraction-r0-2026-07-13.md) and
[R0 audit](audits/channel-extraction-r0-2026-07-13.md).

R1a is preregistered on a guarded, Earth-radius local S2 Voronoi cap at fixed
8/4/2 km spacing. It compares only path-local physical-grade P0 and dominant-
integrated-flux M0 on an affine direction control and smooth quadratic V,
while scoring cell graph and selected-face midpoint geometry separately. The
G0 product-backend cap and planar finite-volume adapter now pass deterministic
rebuild, reciprocal two-vertex topology, positive unequal area/face geometry,
projection and eight-versus-ten-guard gates. Status: geometry substrate passed;
exact registered A/V/B polygon means, one immutable conservative route per case
and P0/M0 local ranks are now implemented. Domain ranks disagree broadly and
six A/V cases disagree at their prescribed head, passing the existential
visited-cell anti-alias subgate. Path-local tracers, portal termination and
C0/F0 physical gates have now also been implemented and evaluated. Status:
completed negative experiment. P0 and M0 both fail the
required affine-plus-valley contract, so neither is selected. The affine
control exposes polygon-mean state interpreted through generator-based two-
point geometry; some rotated paths terminate at genuine sinks and successful
paths drift materially. A broad corridor remains report-only. M1, analytic Y,
initiation, confluence ownership, persistence and C1 coupling are ineligible.
Research
establishes that distinct trajectories of a smooth steady gradient field
cannot merge and share a suffix, so confluence receives a later conservative
topology/morphology gate rather than another fictitious exact Y. See the
[geometry basis](research/channel-centerline-geometry-basis-2026-07-13.md),
[R1a specification](research/channel-extraction-r1a-2026-07-13.md) and
[G0 audit](audits/channel-extraction-r1a-g0-2026-07-13.md), plus the
[input/rank audit](audits/channel-extraction-r1a-input-rank-precheck-2026-07-13.md)
and [path audit](audits/channel-extraction-r1a-path-2026-07-14.md).

The generator-point causal rung is evaluated. It pairs each
registered polygon-mean affine case with exact values of the same plane at the
Voronoi generators, then independently recomputes a route and P0/M0 ranks with
the same algorithms, tracer and F0 subgates. It is report-only and cannot
promote an arm or weaken polygon-mean physical semantics. Paired termination
repairs identify the polygon-mean/generator-geometry interaction as load-
bearing. Geometry attribution uses only complete paired-success sets; censored
comparisons remain descriptive. Result: neither arm repairs a termination. P0
retains six rotated failures and M0 worsens from four to six; both generator
arms also fail the cross-track and length measures. Alternate state placement
is insufficient, so neither arm is rescued. A linear-exact gradient feeding the
same affine maximum-face P0 would be redundant; the next nonredundant candidate
is a separately preregistered affine, entry-point-aware continuous crossing.
See the [generator-point control](research/channel-extraction-r1a-generator-control-2026-07-14.md)
and [audit](audits/channel-extraction-r1a-generator-control-2026-07-14.md).

The affine continuous-crossing discriminator is evaluated but causally
incomplete. It compares an analytic downhill ray with a local polygon-mean/
polygon-centroid linear reconstruction over the same 12 A cases. Both start at
the physical head and traverse actual Voronoi segments at actual intersection
points. A checked segment-to-CSR/boundary context, all-cell reconstruction gate,
explicit vertex/collinear ambiguity and semantic portal termination precede
any causal claim. X0 passes 12/12 and validates the registered cap plus analytic
continuous traversal. X1 passes all 11 judged cases with X0-identical face
sequences, and internal-score equivalence passes 12/12, but the frozen all-cell
reconstruction misses its `1e-10` gate once at 2 km. That case is not judged;
there is no X1 traversal failure. Status: incomplete, no product arm promoted
and the maximum-face/F0 bundle localization remains unclaimed. The next rung is
a separately preregistered stable linear-consistency control, not V, RT0 or
product integration. See the
[crossing specification](research/channel-extraction-r1a-affine-crossing-2026-07-14.md)
and [audit](audits/channel-extraction-r1a-affine-crossing-2026-07-14.md).

The stable affine-reconstruction control is evaluated. It crosses registered
polygon-mean differences versus a direct
affine centroid-difference oracle with the frozen normal-equation solve versus
a fixed-order streaming Givens QR solve. RN must reproduce the incomplete
parent baseline exactly; only registered-input RQ can repair the physical-input
prerequisite and complete the earlier causal claim. Oracle arms are numerical
attribution controls, not product states. X0, the checked cap, stencil,
singularity gate, crossing and `1e-10` all-cell threshold remain unchanged. See
the [stable reconstruction specification](research/channel-extraction-r1a-stable-reconstruction-2026-07-14.md).
Result: RN and RQ both remain 11/12 and fail the same 2 km reconstruction;
ON and OQ pass 12/12 at machine precision. Every judged crossing and every
score-equivalence audit passes. The stencil is well-conditioned and QR changes
registered gradients only around `1e-15`; registered mean/difference numerics,
not the solve, remain load-bearing. Status: completed negative solve control;
QR not promoted and graph-bundle localization still formally incomplete. See
the [audit](audits/channel-extraction-r1a-stable-reconstruction-2026-07-14.md).

### Carrier/lifecycle subexperiments

| Experiment | Status | Outcome |
|---|---|---|
| Carrier resolution sweep | Numerically evaluated | Failed: deformation concentration and morphology do not converge with carrier resolution |
| Exact boundary-velocity constraint | Falsified and reverted | Increased resolution divergence and peak concentration |
| Fixed-width boundary band | Falsified and reverted | Did not close promotion behavior |
| Same-clock coarse denudation | Implemented behind zero-default carrier control; evaluated | Bounded but insufficient; not a promoted erosion model |
| Finite-memory/coherent Euler motion | Implemented behind zero default; evaluated in evolved-carrier replay | Useful causal experiment for evolved replay, not consumed by the lifecycle solver; hard cutoff/reorganization shortcuts rejected |
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
| Older layered shader path | Removed after reachability/compile verification | No product capability; unified and flat paths remain |
| Legacy CPU wind line draw | Removed after every scene supplied `None` | GPU wind particles remain the active presentation |

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
