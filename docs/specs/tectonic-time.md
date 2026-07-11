# Tectonic time and boundary history

Status: T0, stationary-topology T1a, seed-Voronoi T1b, full-resolution material-
raster T1c, and low-resolution fixed-carrier T1d implemented; scalar T2 rungs
falsified; physical-clock thin-sheet T2c and moving-carrier T2d implemented but not
promoted; bounded forward lifecycle T4a passes material/topology invariants but its
direct underthrust-to-surface response is falsified; concurrent surface processes pending.

## Problem

The generator currently mixes four unrelated clocks:

- Euler-pole angular velocities are dimensionless;
- legacy arc/collision fields become height immediately;
- conservation experiments multiply flux by `TECTONIC_ACCUMULATION_TIME = 0.05`;
- fine erosion runs `steps * dt = 200` dimensionless time while direct uplift adds a
  fixed fraction of the already-solved tectonic load each step.

Consequently, a young collision, a mature flux-balanced range, an old decaying belt,
and a broad long-lived plateau can differ only through spatial tuning. Fine-terrain
systems have been compensating for missing history.

## State belongs to contacts, not just plates

A plate can participate simultaneously in an old passive margin, a young collision,
and a mature subduction zone. One `plate_age` cannot determine their landscapes.

The primary historical object must be a **boundary episode**, keyed by plate pair and
connected boundary chain:

```text
BoundaryEpisode {
    plate_pair
    chain_id
    kind_history          // convergent, divergent, transform, inactive
    polarity_history
    start_myr
    end_myr_or_active
    integrated_normal_displacement_km
    integrated_shear_displacement_km
    integrated_crust_flux_km3_per_km
}
```

Plate birth age remains useful for oceanic lithosphere and lifecycle constraints, but
orogen maturity comes from the duration and flux of its contact episode.

## Units

Introduce one explicit geological unit system:

- distance: km;
- time: Myr;
- plate angular velocity: rad/Myr;
- linear plate velocity: km/Myr (1 cm/yr = 10 km/Myr);
- crustal supply: km³/Myr or thickness-km/Myr after footprint distribution;
- erosion/uplift: km/Myr;
- hillslope transport: km²/Myr.

`FEATURE_FORCE_SCALE`, `TECTONIC_ACCUMULATION_TIME`, `EROSION_UPLIFT_SCALE`, and
dimensionless erosion `dt` then become migration shims to remove, not permanent dials.

## Recommended implementation ladder

### T0 — unit conversion, bit-gated

Add `TectonicClock` and convert generated Euler speeds to rad/Myr using a documented
plate-speed prior. Preserve the current world behind a legacy clock flag while tests
establish conversion identities.

No terrain changes in this rung.

**Implemented 2026-07-11.** Normalized legacy velocities remain bit-identical. New code
and exports carry a 10 cm/yr maximum prior, km/Myr and rad/Myr conversions, and an
explicit 100 Myr / 1 Myr-sample clock.

### T1 — constant-velocity boundary replay

Reconstruct a bounded lookback history on the coarse tectonic mesh:

1. Start from present plate domains and Euler poles.
2. Back-rotate plate material in fixed geological increments.
3. Resolve overlaps/gaps deterministically on the sphere.
4. Recompute adjacency, boundary kind, normal velocity, and polarity at each slice.
5. Chain compatible edges by plate pair and topology.
6. Match chains between adjacent slices and accumulate boundary episodes.

This assumes constant Euler poles and no plate birth/death. It is not a complete plate
cycle, but it derives contact duration from kinematics rather than sampling an arbitrary
"mountain age". Store the approximation explicitly in exported provenance.

**T1a implemented.** Present boundary edges are chained
by plate pair and connected topology. A stationary-topology residence bound uses
`min(chain_length / mean_relative_speed, lookback)`. This is exported and audited with
`--tectonic-history-audit`. It is useful provenance but is NOT sufficient history:
across seeds 12345/777/4242, median duration is 86–91 Myr and major contacts accumulate
~3,300–5,100 km of normal displacement. Holding present contacts for that residence time
systematically over-accumulates shortening. Full back-rotation/topology matching remains
the required T1 completion; do not tune terrain to compensate for T1a.

**T1b implemented.** The actual plate-generation seed positions are retained, back-
rotated under physical Euler rates at 1 Myr intervals, and their spherical Voronoi
adjacency is reconstructed by convex hull. A present plate-pair episode cannot predate
the first lookback slice where that pair ceases to be adjacent. Duration is the minimum
of this topology age and the T1a residence bound. This is not yet material-domain
advection—the noisy margin and crust field are not replayed—but the clock changes only
through kinematics. Seed 12345's largest pillar-contributing contact falls from ~5050
to ~1500 km integrated convergence; duration p10/p50/p90 becomes 0.5/55.5/100 Myr.

**T1c experimental replay implemented.** `history-material` back-rotates every coarse
plate cell as a material parcel, rasterizes overlaps by nearest parcel, fills gaps by
deterministic graph propagation, and measures pair adjacency on the reconstructed noisy
domains. This is intentionally not the product path because it costs O(cells × history
steps). It falsifies the optimistic T1b explanation for seed 12345: the pillar contact is
52.5 Myr / 2863 km rather than 27.5 Myr / 1500 km. Better domain topology makes the
over-accumulation worse, so topology age alone is not the missing regulator.

**T1d fixed-carrier replay implemented.** `history-carrier-thin-sheet` constructs one
immutable 8,192-cell (~250 km spacing) Voronoi carrier and transfers one present-day
plate/crust parcel to every site. It then back-rotates those parcels under their Euler
poles through 51 snapshots at 2 Myr intervals. Geometry is never rebuilt. Each snapshot
retains the raw occupancy field (zero = gap, >1 = overlap), exact gap indices and all
overlapping parcel/plate/crust identities, deterministic filled surface plate/crust
ownership, pair adjacency, kinematic pair kind, and topology changes. The
existing episode ledger consumes its contiguous contact ages, and the existing history
thin-sheet solver consumes those episodes; the product default is untouched.

The 2 Myr interval is a sampling bound rather than a tuned dial: at the 100 km/Myr speed
ceiling a parcel travels at most 200 km, below the 249 km mean carrier spacing. Exact
states at shared times are invariant when the interval is subdivided in tests. Parcel
count is exact: gap count equals overlap excess at every slice. On seeds 12345/777/4242,
the carrier itself builds in 0.127/0.129/0.123 seconds in release. Mean gap/overlap
fractions are large (28%/28%, 28%/28%, and 31%/31% respectively), which is useful
physics provenance: rigidly rotating present domains do not tile their reconstructed
past without deformation, crust creation, or destruction.

For seed 12345 the important 1-5 contact is 53 Myr / 2,891 km, essentially confirming
the full-resolution T1c result (52.5 Myr / 2,863 km) at a small fraction of its raster
cost. Thin-sheet coarse peaks for seeds 12345/777/4242 become 13.9/15.6/11.0 km; seed
12345 at 50k source cells is 12.7 km (~9% lower), so the existing resolution gate remains
open (and source plate geography itself also changes with resolution). This does **not**
complete dynamic tectonics: the thin-sheet forcing still acts on present-day
high-resolution boundary segments and only their clocks come from the carrier. The next
architectural rung must solve deformation on each carrier snapshot (or transfer each
snapshot's forcing), advect inherited crust state forward, and only then couple erosion.

### T2 — integrate crustal work

Feature computation must emit rates. Integrate each episode's convergent crust flux over
its actual duration, then distribute it across a deformation footprint whose width is a
material/rheological property. Broad old collision zones may become plateaus; young or
narrow contacts remain ranges.

Subduction and collision must have distinct grammars:

- subduction: polarity-aware forearc/arc/backarc system;
- continental collision: bilateral shortening unless history establishes asymmetry;
- transform: no generic vertical relief;
- inactive episodes: inherited crust/topography, zero present uplift.

**Experimental rungs implemented and falsified.** `history-local` integrates physical
closing rate × episode duration and leaves conserved work in receiving cells; seed 12345
reaches ~1097 km peaks. `history-diffusive` conserves the same work and spreads it with
an explicit 1000 km²/Myr lower-crust mobility over the work-weighted episode duration;
it reaches ~203 km peaks and 25% mountain land under T1a. T1b reduces integrated work,
but the diffusive result still reaches ~140 km and 22% mountain land. The remaining
failure is not a surface-shape problem: continent collision currently retains all
closing volume, and erosion is applied after rather than throughout growth. Both models
remain runtime-selectable and cannot be promoted. No height cap, gain reduction, or
mobility retune should hide this failure.

**T2b implemented.** Integrated work now retains a sparse per-episode ledger, and
`history-diffusive` relaxes each contact over that contact's own duration before
superposing the conserved thickness fields. The former global work-weighted mean age
was causally wrong: it made young and old contacts share one clock. The aggregate and
episode ledgers are compared in `--tectonic-history-audit`; their residual is a standing
conservation gate. This correction is architectural, not a candidate default, and does
not address the all-shortening-is-retained assumption.

**Finite-strain material-footprint rung implemented and falsified.** `history-material`
remaps each episode's conserved work into a same-plate/same-crust footprint whose target
area is `work / reference_crust_thickness`, then applies episode-local lower-crust
diffusion. Seed 12345 still peaks at 54.6 km; only 84.7% of requested footprint capacity
is available in the receiving domains. This is a direct diagnostic that the scalar
source repeatedly spends more convergence than the rigid plate/material model can
accommodate. It must not be fixed by widening the kernel.

**T2c physical-clock thin sheet implemented.** `history-thin-sheet` treats boundary
closure as traction, solves internal sheet velocity, and integrates conservative crustal
continuity over episode start/stop intervals in Myr. Collision only redistributes existing
crust; retained arc magma is the sole positive mass source. Stress transmission is an
explicit 440 km viscosity/drag length and gravitational mobility is 1000 km²/Myr; the
historical dimensionless duration and yield-height threshold are not used. Coarse peaks
for seeds 12345/777/4242 are 11.4/14.2/10.9 km, and seed 12345 at 50k cells is 12.5 km.
That is the first correct order of magnitude, but the ~10% resolution shift and lack of
concurrent erosion keep it experimental. A mass ledger audits final thickness change
against retained magma; collision transport must be globally neutral.

This pivot is physically motivated rather than numeric convenience. India–Asia studies
separate thousands of kilometres of plate convergence from only hundreds of kilometres
of distributed intra-Asian shortening; underthrusting, subduction, and lateral extrusion
accommodate the rest. Plate closure therefore cannot itself be a retained-crust source:

- https://doi.org/10.1029/2010JB008051
- https://doi.org/10.1002/2016JB013337
- https://doi.org/10.1016/j.tecto.2021.229081

The current legacy envelope can remain as a comparison rung, but not as both an
instantaneous height prescription and an uplift-rate source.

**T2d moving-carrier evolution implemented.** `history-carrier-evolved` uses the T1d
8,192-cell snapshots as actual forcing geography rather than only as a contact clock.
For every 2 Myr interval, oldest to present, it reconstructs plate/crust boundary edges,
classifies the whole plate-pair regime, derives ocean/continent and motion-voted
ocean/ocean polarity, solves the thin-sheet velocity on that snapshot, and integrates
continuity. Conserved thickness volume and accumulated strain live on material parcels:
the filled surface owner receives each cell's deformation and arc magma, hidden overlap
parcels retain their state, and gap-filled cells share their owner's finite parcel volume
rather than duplicating it. Historical compression axes rotate forward with their Euler
plate. There is no dynamic parcel birth/death, multilayer underthrust stack, sediment,
or geological erosion in this bounded rung.

Collision flux is globally neutral; retained arc magma is the only positive material
source. Release results for seeds 12345/777/4242 are:

| seed | carrier build | evolution | peak | arc addition | relative mass residual | historical receiver events outside current support |
|---:|---:|---:|---:|---:|---:|---:|
| 12345 | 0.132 s | 0.846 s | 13.9 km | 0.3744 | 1.06e-6 | 49.9% |
| 777 | 0.151 s | 0.889 s | 12.0 km | 0.3143 | 1.82e-6 | 60.2% |
| 4242 | 0.146 s | 0.915 s | 16.8 km | 0.3327 | 4.99e-6 | 56.4% |

The stationary carrier-clock peaks were 13.9/15.6/11.0 km, so moving forcing is
load-bearing even where one seed's maximum happens to be unchanged. The receiver-event
metric is material-domain based: it asks whether the historically forced parcel belongs
to the present receiver support, not merely whether a fixed spatial cell was reused.
The solver also projects inherited thickness/strain/fabric and the present physical
thickness tendency onto the 100k terrain mesh. Present tendency remains diagnostic and
is not fed into the legacy dimensionless erosion loop.

Small-mesh regression tests require bit-deterministic replay/evolution, conservative
parcel mass, and stable 1/2/4 Myr subdivision (relative thickness RMS under 10% for
1-to-2 Myr and under 20% for 2-to-4 Myr). This is authentic qualitative history, not a
plate lifecycle: overlaps and gaps inform surface ownership/polarity but are not yet
resolved through crust production, consumption, or stacked underthrust sheets.

### T3 — run surface processes on the same clock

For each history interval, evolve elevation with that interval's uplift field,
precipitation, stream-power incision, hillslope transport, and sediment routing.
Growing, steady, and decaying orogens then emerge from the ratio of supply to removal
rather than from separate visual presets.

The adaptive fine mesh need not exist for the full replay. Integrate tectonic/crustal
history on the coarse mesh, then run a final higher-resolution surface-process window
using inherited coarse topography and current/recent uplift rates.

**Clock-safety gate implemented.** `ElevationFields` now distinguishes accumulated
tectonic state from the legacy dimensionless uplift source. History/conservation/thin-
sheet experiments enter the existing fine erosion loop with zero tectonic rebuild,
rather than incorrectly adding a fixed fraction of their entire 50–100 Myr crust load
on every fine step. This is deliberately not T3: inherited relief can be carved, but
present uplift remains off until the thin-sheet solver exports a km/Myr rate and erosion
has a Myr timestep. The legacy product path retains its original floating-point source
expression.

Physical observations support a regime-aware rate model rather than one global erosion
gain: rapidly incising Himalayan rivers can reach 2–12 mm/yr, while preserved plateau
surfaces can remain below 250 m/Myr for tens of Myr.

- https://doi.org/10.1038/379505a0
- https://doi.org/10.1038/ngeo503

**T3a capacity experiment implemented, identity-gated.** The moving carrier can now
apply an optional surface-lowering ceiling in km/Myr after each tectonic interval.
Airy isostasy converts that rate to crust-thickness loss; only thickness above each
material parcel's undeformed reference column is eligible. Removed volume enters an
explicit unresolved-sediment ledger. The default rate is zero and preserves the prior
carrier path exactly. This is intentionally not stream-power erosion: the carrier has
neither precipitation nor resolved drainage, so the parameter is an upper-bound A/B,
not a new product constant.

At canonical 8192 cells, ten-seed sweeps at 0/0.05/0.10 km/Myr reduced median peak
height from 15.98 to 14.53 to 13.14 km, but FAIL counts only moved from 6/10 to 6/10
to 4/10. Median mountain-land coverage was 40.8/41.0/39.2%, so the broad-load problem
did not respond materially. The next T3 rung must condition removal on relief/drainage
and resolve sediment deposition; increasing the uniform ceiling until all peaks pass
would be target tuning.

**Coherent-motion audit implemented.** The scorecard can override geological lookback
without changing product defaults. At 25/50/75/100 Myr with denudation disabled,
ten-seed median peaks were 6.68/9.87/13.06/15.98 km, but median mountain-land coverage
was already 35.0% at 25 Myr and saturated near 40% by 50–100 Myr. Thus constant Euler
motion over the full 100 Myr is causal for extreme stacking, not for most of the broad
footprint. No shorter duration is selected: the next physical-history experiment must
retain 100 Myr while allowing plate velocities to reorganize through time.

**Finite-memory motion A/B implemented, identity-gated.** Historical Euler vectors can
now reorganize at deterministic continuous-time Markov events while retaining the same
isotropic axis/speed prior. Event ages do not depend on carrier snapshot subdivision;
the same piecewise rotations drive both material positions and boundary forcing. Zero
means infinite coherence and preserves the previous carrier result. At 15/30/60 Myr
mean coherence, median integrated rotation remains 0.77/0.74/0.75 rad versus 0.73 for
constant motion, while median peaks fall to 10.54/12.26/14.02 km. Median mountain-land
coverage worsens to 47.6/47.1/44.1% versus 40.8%, because authentic forcing migration
distributes work instead of repeatedly stacking it. No coherence time is selected;
deformation support must be fixed independently of this history mechanism.

### T4 — plate lifecycle

Only after T1–T3 work should the model add ridge birth, subduction consumption, plate
splitting/merging, changing Euler poles, and true oceanic crust ages. This is the full
plate-cycle rung and will intentionally change every seed again.

**T4a bounded forward carrier lifecycle implemented and falsified.** The identity-gated
`history-carrier-lifecycle` model treats the generated carrier plate/crust layout as the
oldest state and evolves it forward for 100 Myr on the immutable 8192-cell mesh. State is
limited to motion owner, crust class, physical ocean age, component-resolved crust/root
volume, persistent weakness/fabric, and explicit material/area reservoirs. It does not
claim sediments, multilayer slab geometry, plate splitting, or a full mantle cycle.

The first forward implementation used parcel splatting plus graph gap fill. It conserved
global volume but failed a rigid-motion null case and produced a 14.43-thickness-unit
numerical remap maximum (seed 12345 peak 51 km; 85.7% of land above 2 km). It was removed,
not calibrated. The retained operator is topology-aware pullback: every destination cell
is inverse-rotated through every active plate's interval motion and admits a plate only
when the sampled source belonged to it. Plate/component totals are conservatively
normalized after sampling. Multiple admissions are genuine boundary overlap; no
admissions are genuine opening. A single rigid uniform plate now remains uniform within
1e-5 over 100 Myr with zero lifecycle work.

Lifecycle reactions are deliberately small:

- divergent ocean/ocean openings create zero-age ocean crust and attach it to an adjacent
  motion domain;
- oceanic overlap is consumed into an explicit reservoir and only the existing retained
  magmatic fraction returns to the overriding material;
- continent/continent overlap conserves both continental volumes as underthrust root and
  writes persistent suture weakness/fabric;
- continental pairs merge motion only after edge-length-weighted positive normal
  convergence on their actual pre-remap continental contact accumulates to one carrier
  spacing; transform path length contributes zero, and the merged Euler vector is
  material-area weighted;
- sub-cell domains receive one deterministic pullback support cell so they disappear only
  through explicit consumption or merge, never sampling loss. No splits were required in
  the audited worlds.

The evolved final crust classification, thickness/root, ocean age, fabric/weakness, and
present thickness tendency project to the 100k terrain mesh. Source-layout trench, arc,
ridge, collision, rift, and craton-macro fields remain available as diagnostics but are
explicitly zeroed in lifecycle elevation assembly. They do not force lifecycle terrain.

A strict rerun after the positive-normal-convergence merge correction gives the committed
direct-column baseline a median peak of 33.48 km (range 1.50–334.67; 9 FAIL / 0 WARN /
1 PASS) and median mountain-land coverage about 0.65%. Maximum local underthrust is
2.51–47.22 reference thickness units versus 0.028–0.892 residual remap. The earlier
14.35 km table predated that root-review correction and is superseded. The lifecycle is
not promotable: direct full-column expression creates localized excessive roots while
generating too little broad mountain area.

Across the same ten seeds, created/consumed ocean volumes span 0.047–0.536 / 0.013–0.378,
continental underthrust 0.0053–0.0606, retained magma 0.0019–0.0567, active sutures 0–24,
merges 0–6, and final motion domains 8–13. Lifecycle solve cost is 0.78–1.08 seconds in
release (carrier construction separate). All material and continental ledgers close near
floating-point roundoff; final surface ownership has no ghost overlaps.
Persistent weakness is transported as an intensive connected-component field: every
suture retains deterministic support through rigid pullback, damage uses max semantics,
fabric rotates with its plate, and collision-deposit counts cannot disappear through
nearest-cell sampling.

**T4b capacity-limited buried sheet implemented.** Continental collision now transfers
the losing material into a separate deposit event rather than adding it at the contact.
The receiving domain admits at most one normal reference-crust layer per carrier cell;
overflow advances through graph rings with convergence alignment providing the ordering
inside each ring. Thus sheet area is volume divided by reference thickness, not a painted
width. Pullback reconcentration is re-extracted and propagated every interval. Any volume
blocked by finite receiving geometry remains in an explicit foundering reservoir included
in both material ledgers but excluded from surface isostasy.

At canonical 8k, ten seeds improve from the committed direct-column median/range of
33.48/1.50–334.67 km to 8.71/7.08–20.21 km; gates move from 9/0/1 to 3/1/6 and median
mountain-land coverage from about 0.65% to 2.75%. Every seed respects a maximum buried
layer fraction of 1.0, sheet footprint spans 0.0079–0.2195 steradians, foundering spans
0–0.0304 volume units, and ledgers remain at roundoff. This is a retained causal rung,
not a promotion. Remaining failures are now magma-localization dominated (maximum local
magma 1.89–2.65 thickness units in the three FAIL seeds), which must be specified and
audited independently.

## Required invariants

- Integrated crustal addition equals boundary flux integrated over time, within solver
  tolerance.
- Present uplift is zero on inactive contacts even when inherited mountains remain.
- Uniform time subdivision leaves the result stable.
- Fixed physical duration is stable under coarse/fine mesh refinement.
- Collision and subduction cannot both spend the same convergence flux independently.
- Sea-level/land-volume changes have an explicit crustal or sediment ledger.
- Fine stochastic perturbation amplitude tends toward zero without changing resolved
  relief statistics.
- Rendering parameters never enter a generation or acceptance gate.

## Promotion scorecard

The headless `tectonic_scorecard` binary now sweeps seeds and carrier resolution
independently of the 100k terrain mesh. The first standing audit is recorded in
`docs/audits/tectonic-promotion-scorecard-2026-07-11.md` and falsifies the moving-
carrier rung: peak drift is 4.1–19.4 km across 4k/8k/16k carriers, six of ten 8k
seeds exceed 14 km, and mountain-land coverage reaches 33–51%. Mass conservation
passes, isolating the failure to deformation concentration/parcel projection rather
than leakage. The subsequent T3a experiment is retained only as an identity-gated
capacity diagnostic; it is not a promoted erosion model.

The subsequent five-rung operator ladder localizes the first failure to one 2 Myr
step from uniform crust. Boundary length grows from 417k to 647k km and the step maximum
from 0.127 to 0.249 across 4k→16k carriers; carrier-native and 100k-projected maxima are
identical. The pending fix is therefore a finite-volume/fixed-physical-width boundary
traction, not parcel remapping, projection, or erosion.

An exact boundary-velocity (Dirichlet) A/B was implemented and reverted after worsening
the one-step maximum span from 0.122 to 0.330 and seed-12345 peaks to
14.2/17.7/20.3 km. It preserved the carrier's increasingly jagged boundary geometry too
faithfully. Fixed-physical-scale boundary geometry must precede another forcing-law
change.

A subsequent 440 km fixed-width boundary band was also implemented and removed. Both
nearest-seed and edge-weighted blended variants failed; the blended one-step maximum
span was 0.139 versus the original 0.122. Transport/magma isolation shows transport
sets the maximum (0.197/0.250/0.336 across 4k/8k/16k), not arc addition. Since the
coarsest mesh barely resolves the physical band, carrier resolution is now treated as
part of the procedural model rather than a continuum claim. The canonical 8192-cell
carrier remains explicitly experimental and non-converged.

## Migration

The 2026-07-11 reset direct-uplift path is the temporary product baseline because it has
one causal direction and no target reconstruction. It is not the final physical model:
`EROSION_UPLIFT_SCALE * 200` still substitutes for an active duration.

Implementation should replace that product with episode-integrated rates, rather than
retuning either constant.
