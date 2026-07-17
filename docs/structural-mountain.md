# Structural Mountain V0 contract

Status: **source-first target and legacy observation binding accepted; internal
organization and terrain replacement not implemented or promoted**, 2026-07-18.

This document defines the one product-native mountain comparison authorized by
the [roadmap](roadmap.md). It replaces the old H/C/G campaign as the active
instruction. Those planar experiments remain useful research history; they are
not a queue of unfinished promotion work.

## Product outcome and precise defect

Hex3 needs convergent belts that read as organized mountain systems in Physical
terrain and remain legible under Authentic presentation. They should provide
causally placed massifs, finite range ends, a main divide, branch ridges,
saddles/lows, valleys and drainage basins that later systems can consume.

The defect is not that plateaus exist. Broad high surfaces are legitimate under
distributed shortening, strong material, aridity or internal drainage. The
defect is that the current smooth, capped distance-band response is the generic
grammar of convergence. Erosion adds strong local relief but mostly carves the
inherited roof. Relief exaggeration makes its margins attractive while making
the missing internal hierarchy harder to notice.

The earlier apparent pillar is a separate presentation warning. It was a broad,
supported structure made implausible by roughly hundredfold relief
exaggeration, not proof that the physical terrain needed to be flattened.

## Selected architecture

Build one **finite-segment uplift organizer with drainage-conditioned terrain
response**. It is an authentic reduction of linked fault/fold-system envelopes,
not an explicit fault, crustal-rheology or geological-time simulation.

```text
generated convergent front + plate/crust setting + bounded history evidence
  -> sparse finite deformation segments and transfer relationships
  -> budget-closed, finite, polarity-aware cumulative uplift opportunity
  -> routed incision and hillslope response on one authoritative surface
  -> drainage basins and watershed divides
  -> physical terrain + derived belt evidence
  -> declared cartographic presentation
```

This preserves the three relationships with the strongest benefit-to-cost
case:

1. finite deformation segments supply a prior for uplift maxima, tapered ends
   and intersegment lows;
2. polarity orients the receiving side, while an independently supported
   cross-belt uplift gradient may create unequal flanks; and
3. drainage competition plus hillslope response turns that opportunity into
   valleys and a branching divide/ridge hierarchy.

Real fault systems show finite nested segments and displacement deficits near
intersegment zones
([Manighetti et al.](https://doi.org/10.1002/2014GC005691)). Reduced landscape
models show that incision and hillslope transport can establish ridge/valley
spacing and asymmetric divides
([Perron et al.](https://www.nature.com/articles/nature08174),
[He et al.](https://www.nature.com/articles/s41467-020-20748-2)). Graphics
systems reach a compatible authentic reduction by authoring sparse structure in
the uplift domain and allowing routing/erosion to own the compatible terrain
([Cordonnier et al.](https://doi.org/10.1111/cgf.12820),
[Schott et al.](https://doi.org/10.1145/3592787)).

The mapped-fault evidence is primarily extensional; V0 uses finite segmentation
as a modeling analogy, not a universal convergent-belt law. The landscape
models do not guarantee a branching massif-scale hierarchy from arbitrary
forcing, either—that hierarchy remains an outcome the candidate must earn.
Graphics systems establish a practical architecture precedent, not the
geological truth of Hex3's automatic segment placement.

No independent ridge noise, arbitrary pass locations or renderer relief enter
the world model. A graph-first drainage reconstruction remains a fallback
architecture only if the selected response cannot earn its cost; it is not a
second arm in this slice.

## Existing product inputs

The slice consumes evidence already computed or derivable in the ordinary
product path:

- shared Voronoi geometry, area and physical distance;
- connected convergent boundary edges with plate pair, crust pair, convergence,
  shear, boundary length and subduction polarity;
- shared boundary-edge arcs and polarity currently exposed through
  `OrogenFronts`; its chain IDs and coordinates remain experimental inputs, not
  accepted belt authority;
- episode identity, duration and integrated normal/shear displacement from the
  default `TectonicHistory`;
- continental/oceanic identity, crustal-margin distance and craton identity;
- base elevation, common sea-level datum, runoff and hydrologic policy; and
- hydrologic repair provenance.

Default history is a synthetic reduced backrotation, not geological
reconstruction. V0 may use its episode grouping and relative displacement
ordering; it may not narrate exact past geography. Craton and crust transitions
may explain segment placement or termination when they actually intersect the
belt. V0 does not add an independent weakness, competence or lithology-noise
field.

The product-facing front record owns chain grouping and one directed
along-strike coordinate independently. Existing `arc_u` and endpoint-ordered
`u_lin` have different semantics, and current degree-2 chains do not split
automatically at episode, regime or receiving-side changes. V0 may reuse the
shared-edge geometry, but it must make those causal boundaries part of the new
record rather than promoting a P1/Massif helper by accident.

## Mechanical implementation checkpoint

`world::structural_mountain` now implements the deliberately inert first code
boundary. It collects exact shared Voronoi arcs from the same boundary snapshot
used by history, converts chord length to physical great-circle length, resolves
edge-local subduction sides, and counts every plate-boundary arm so a hidden
transform/divergent third arm terminates a structural chain.

The compiler groups independently of `OrogenFronts` and continues only through
degree-two vertices with identical episode, regime, plate/crust pairing and side
semantics. Each open causal chain becomes one finite parent. A full-cosine
along-strike taper redistributes its declared shortening-area opportunity and is
renormalized per segment; the ledger closes without creating opportunity.
Closed loops, zero-opportunity sources, disconnected parents and missing or
inconsistent source evidence remain explicit outcomes rather than geometry
fallbacks. Source, accepted and omitted opportunity are recorded separately so
an omitted positive source cannot disappear behind closure of the accepted
subset.

This is not yet an uplift field. The ledger quantity is
`edge length × positive local convergence × episode duration` in km² of
shortening-area opportunity—not tectonic work, uplift volume, elevation or a
terrain-response calibration. No terrain, cache, default, erosion or renderer
path consumes it. Manufactured tests cover finite support, causal transitions,
input/endpoint reversal, collision/subduction semantics, hidden third arms,
disconnected parents and typed omissions; a generated product-input smoke test
covers the real collector.

## Organization contract

For the fixed belt's causally attributed front set, compile a small
deterministic graph:

- **segments** retain parent front interval, receiving side where defined,
  start/end, length, width, polarity where defined, episode,
  maturity/uplift-opportunity share and active/abandoned status if the accepted
  history supports it;
- **nodes** retain finite tips, bends, overlaps/linkages and transfer zones;
- **budget ownership** is partitioned once among segments and closes to the
  declared belt uplift-opportunity total; overlap cannot manufacture
  opportunity; and
- **field compilation** emits a continuous cumulative uplift opportunity with
  finite tapered support, varying width, disclosed overlap and lower support at
  transfer zones.

Segment width begins from the frozen physical decay scale already used by the
matching legacy feature type. It may vary only where crust, convergence or
episode evidence supplies an ordering; otherwise it remains uniform and the
candidate may fail. Width, bend and child thresholds cannot be adjusted after
viewing terrain to force the required hierarchy.

Segmentation must change geometry, not just multiply one ribbon. It may use
front ends, strong bends, episode boundaries, convergence/displacement extrema
and real crust transitions. A long uniform chain must still receive a finite-
support prior, because the earlier F control established that exact infinite
roofs are an artifact of the representation. It must not be split by
independent random samples merely to increase variety.

A genuinely uniform finite chain receives finite ends, not fabricated internal
segments. If available front, history and crust evidence cannot support an
internal high/low hierarchy on the fixed belt, the candidate fails this
contract with `insufficient-causal-segmentation`.

Where subduction polarity defines a receiving side, it orients the cross-belt
opportunity consistently. Polarity alone does not select an uplift gradient,
maximum or width asymmetry. Those require separate convergence, history or
crust evidence. Continental collision has no automatically correct overriding
side; V0 must remain symmetric there unless kinematic/history evidence supplies
a defensible vergence. It may report missing asymmetry rather than fabricate
one.

The organizer emits uplift opportunity, not finished mountain elevation, ridge
pixels or named passes. Its sparse graph remains provenance. Drainage-derived
watersheds, not the authored segment graph, own final divide semantics. A
divide is called a branch ridge only where final elevation independently
supports a crest; not every divide is a pronounced ridge, and not every ridge
is a watershed divide.

## Single-owner terrain boundary

The candidate replaces both current convergent mountain writers inside the
comparison domain:

1. legacy `arc + collision` direct tectonic thickening in the fine starting
   surface; and
2. the equivalent repeated legacy uplift used by hold-and-carve erosion.

It must not add structural relief on top of either. Other terrain owners—crustal
base, ridge/trench/rift terms, craton relief and the shared sea-level datum—stay
fixed. Hydrologic outlet integration remains a later explicit repair writer.
The current erosion API combines convergent thickening and rift response behind
one legacy-uplift switch, so implementation must split those sources: the
candidate removes only repeated convergent uplift while leaving the declared
rift owner unchanged.

The candidate starts from the same non-orogenic base and applies the compiled
cumulative uplift through the existing routed incision and hillslope operators
over one fixed product response schedule. That schedule remains an authentic
calibrated response, not geological time. Routing is recomputed as terrain
changes; erosion is allowed to express or modify supplied organization but is
not asked to invent its first-order segmentation.

Candidate pre-hydrology and local erosion base levels are recomputed from the
candidate starting surface. Reusing legacy pre-hydrology would leak the removed
terrain owner into the comparison. Uplift support is the organizer's finite
receiving-crust/material support, not the legacy `base > 0` land gate; supported
cells may rise from below the fixed sea datum. This requires an explicit source
support API and must not inherit the parked emergent target-land floors.

Because subtracting post-datum legacy height does not commute with the original
sea-level solve, the packet reports area-weighted land-to-water and
water-to-land flips, candidate-domain boundary continuity and every changed
coast cell. A qualifying drainage consequence must lie in the stable-land core
and cannot be owned by a crop seam or datum-induced coastline change.

One conversion from episode/front convergence/displacement evidence to
cumulative uplift is frozen before viewing the candidate. At the reference
belt it matches the legacy control's belt-integrated positive initial
tectonic-thickness opportunity, not its final peak, cap fraction or rendered
relief. Candidate denudation is an outcome. The legacy direct-height
contribution and gross repeated uplift addition are reported separately rather
than mislabeled as the same physical work ledger.

For the first discriminator, the legacy coarse terrain may remain frozen solely
to preserve upstream atmosphere and adaptive-mesh allocation. The candidate
must subtract its interpolated legacy convergent contribution before fine
surface response. This makes the discriminator cheaper and causal, but also
non-promotable: a passing owner must subsequently provide the coarse preview or
rerun every upstream consumer affected by the changed terrain.

The first discriminator is explicitly uncached and constructed downstream of a
loaded legacy fine base. Its sidecar hashes the consumed front, episode, crust
and base arrays. If the owner passes and enters `FineBase`, promotion work must
version the fine-cache schema and key every consumed history/front input.

## One fixed generated belt

Use seed `12345`, legacy Stage 4, 100,000 coarse cells and the 250,000 fine-cell
budget. The fixed observation is the dossier's **broadest range** target:

| Property | Frozen value |
|---|---:|
| Anchor | 39.93854° N, 37.91544° E |
| Legacy component area | 1.52 million km² |
| Legacy length × mean width | 2,439 × 622 km |
| Legacy pre/final peak | 3.26 / 4.57 km |
| Legacy integration-cut cells | 50 |

Recover the fixed target by its frozen unit-vector anchor on the unchanged
legacy Stage-4 surface, not by rerunning the `Broadest` rank. The core is the
connected `>= 1.5 km` component containing the nearest fine generator. Its
contributing-catchment buffer is the union of exact legacy drainage catchments
that receive runoff from at least one core cell: non-core land sharing a
terminal water/sink cell with core runoff. This includes other tributaries in
those basins; a reverse closure into the high core is generally empty and is not
the intended relationship. This is not a tunable kilometre dilation. Freeze
sorted core/buffer identities, terminal ownership, actual integration cuts and
breached-source provenance before candidate terrain exists. The old dossier
retains only a 64-cell geometry sample, so the first current-baseline replay
must write the complete identities and verify correspondence with the frozen
anchor and summary values.

Before compiling segments, attribute the target core's legacy **geometric
source ancestry**. The receiving catchment is an evaluation domain for later
drainage consequence, not a source selector: other orogens may share its
downstream basins without owning this range. For every coarse cell read by fine
interpolation over the frozen core, reproduce the legacy plate-restricted
arc/collision distance source:
subduction owns only the receiving-side arc seed; continental collision admits
both seeded sides and invents no receiving plate. Ties use the lower canonical
boundary-cell pair. Include every eligible front co-aggregated at an owning seed
cell, because legacy forcing averages those edges before propagation. The union
is closed only across deterministic shortest pair-convergent exact-front paths
between direct sources inside the same episode/plate component. A locally weak
bridge may represent a transfer low; it is bridge provenance, not a new legacy
amplitude owner. The result must form one coherent
linked plate/episode belt graph. If it does not, the
fixed target is inadmissible and V0 stops rather than choosing one globally
nearest front or a friendlier range.

This is deliberately not additive height attribution. Legacy seed averaging,
normalized plate-wide diffusion, nearest-distance geometry, square-root caps,
oceanic arc noise, fine interpolation and later erosion destroy per-edge height
shares; the diffusion kernel also has no finite support cutoff. Record the
larger same-plate dependency roster as non-attributable legacy amplitude mixing,
but do not let remote diffuse dependencies define the new belt. Leave-one-edge-
out deltas are unnecessary unless geometric ancestry proves ambiguous; they are
signed counterfactual sensitivity, not physical budget shares. Structural V0's
separate shortening-opportunity ledger remains the candidate's budget owner.

This target is already human-reviewed, strongly convergence-associated and
visibly exhibits the long-tableland grammar. It is not claimed to represent all
orogen families. A first pass authorizes a small global corpus later; a failure
does not authorize choosing a friendlier belt.

## Fixed attribution verdict

The first current-baseline replay stops this V0 before terrain response. At
100,000 coarse cells and 255,866 actual fine cells, anchor-first recovery finds
the reviewed object at fine cell `65597`: 10,975 core cells, 1.533 million km²,
a 4.572 km peak and 61 integration-cut cells. This corresponds closely to the
old sampled dossier despite the intentionally regenerated v15 fine mesh. Its
outlet-defined receiving catchments add 9,147 non-core land cells and 6.284
million km²; they are frozen for consequence evaluation but do not select
sources.

Core-only legacy ancestry has 39 geometric owner edges and 52 direct edges after
co-seed inclusion. Exact-front topology closure adds 31 intervening edges. The
83-edge selected set equals the same-plate diffuse dependency roster, yet the
mechanical compiler still produces three disconnected finite parents in one
coarse history episode: a 3,282 km continental-collision front (69 edges) and
two separate subduction fronts of 551 km (11 edges) and 79 km (2 edges), plus
one zero-opportunity omission. One transfer/junction link does not connect all
three. Both opportunity ledgers close near roundoff.

Therefore the fixed elevation component is **inadmissible as one structural
belt** under this contract. The legacy scalar field and the `>= 1.5 km`
component threshold have merged relief from distinct exact-front systems; the
component's strong aggregate convergence association did not establish a
single causal range identity. No candidate terrain, tuning, cache change or
renderer comparison is authorized from this target. The complete frozen domain
and attribution are retained in
[`generated/structural-mountain-seed-12345-attribution-v0.json`](generated/structural-mountain-seed-12345-attribution-v0.json).

This is a target-selection architecture finding, not evidence that finite-front
organization failed. The revised slice below begins source-first from one
coherent generated front/belt graph, then measures legacy and candidate terrain
attached to that same cause. It does not browse current height components for a
friendlier example.

## Revised source-first target

The target identity is now fixed before fine terrain, climate, hydrology,
erosion or rendering exists. Compile every generated convergent front on seed
`12345` at 100,000 coarse cells, then form exact-link connected components with
one history episode and one plate pair. Connectivity is limited to finite
parents capable of building continental relief: continental collision, or
subduction with continental crust on the receiving plate. An oceanic-receiving
continuation remains tectonic context but cannot make a terrestrial mountain
belt global.

Rank eligible components by total declared shortening-area opportunity, then
length and lower stable source ID. This is a pre-terrain prior for source extent
and forcing, not predicted height, tectonic work or a promise of good
morphology. Segment count is deliberately absent from the ranking: a uniform
strong source is allowed to expose `insufficient-causal-segmentation` rather
than losing to a visually convenient graph.

The first implementation exposed and corrected an overly broad intermediate
rule. Plate-pair/episode connectedness alone selected 13,028 km of interface
because one continental segment imported four oceanic-receiving continuations.
Applying the material/receiving-side termination above—still without observing
terrain—produces 45 source components, of which 21 can build continental
relief. The selected component is episode `0`, plate pair `[0, 2]`, stable ID
`[91484, 91861]`:

- 72 exact source edges and 3,253.7 km total length;
- 15.122 million km² of declared shortening-area opportunity;
- one 3,199.8 km continental-collision parent with 70 edges; and
- two continental-receiving subduction transitions of 38.4 and 15.5 km, one
  edge each.

The selected system is exactly linked and therefore source-admissible, but it
is strongly dominated by one uniform causal parent. `CausallySegmented` here
means the source graph contains real regime/material transitions; it does not
yet establish the internal high/low hierarchy required of the eventual
organizer. The long collision parent may still contain defensible bend or
kinematic extrema, which the mechanical compiler has not yet interpreted.

The previously reviewed 3,282 km collision parent is independently recovered
as source rank three and has zero edge overlap with the new selected target.
That is expected: the old observation is retained as evidence of the scalar
field's merging behavior, not recycled as the replacement target.

The complete source-only ranking and selected provenance are frozen in
[`generated/structural-mountain-seed-12345-source-target-v0.json`](generated/structural-mountain-seed-12345-source-target-v0.json).
The next boundary is to derive the selected source's legacy observation domain
from exact legacy response ownership. Elevation may describe that already-fixed
domain; it may not select or expand the source. No terrain replacement is
authorized until that binding establishes what legacy relief and downstream
geography actually correspond to this cause.

## Source-owned legacy observation verdict

The binding reproduces the legacy writer rather than approximating it. Every
coarse `arc` or `collision` response sample is assigned to its deterministic
plate-restricted geometric owner and complete co-seed roster. The fine response
is decomposed with the product's exact nearest-coarse-plus-neighbours inverse-
distance interpolation into pure selected-source, mixed-seed and other-source
geometry. A fine cell is strictly owned only where pure selected response
exceeds mixed plus other response. Mixed ownership remains a typed ambiguity
and cannot enlarge the target.

On the fixed 255,866-cell fine world, 67 of the selected source's 72 edges are
eligible legacy response seeds; five locally weak structural edges have no
legacy response under the historical convergence threshold. There are no
external co-seeders and no ambiguous-majority fine cells. The decomposition
reconstructs the transferred product `arc + collision` field with maximum
absolute error `1.79e-7`; its area-weighted residual is `0.0029` against roughly
9.86 million response-area units.

Strict geometric ownership contains 16,599 fine cells and 7.458 million km².
That is intentionally not a terrain crop: the legacy Gaussian response has no
semantic cutoff, so geometrically owned tails extend far into the plate. Once
the source is fixed, the existing 1.5 km observation threshold may describe
its visible consequence without changing source identity. It yields 8,659
strict-owned cells and 1.445 million km² across ten components. One component
contains 8,583 cells and 1.436 million km²—99.1% of visible cells and 99.4% of
visible area. The nine residual fragments total only 76 cells.

The dominant legacy observation peaks at 9.044 km near 67.65° S, 47.41° E.
Only one visible cell is an integration cut and one is an integration-breached
source, so its visible extent is not owned by hydrologic repair. The binding
therefore passes: the terrain-blind source has one coherent, dominant visible
legacy consequence and is not rescued by diffuse or mixed attribution.

The complete strict domain and dominant visible component are frozen in
[`generated/structural-mountain-seed-12345-observation-binding-v0.json`](generated/structural-mountain-seed-12345-observation-binding-v0.json).
The next gate is internal source organization. Before terrain work, inspect the
70-edge collision parent's generated curvature, local convergence and bounded
history displacement for real extrema or transitions that could justify more
than one along-strike high/low relationship. If that evidence is absent, report
`insufficient-causal-segmentation`; do not manufacture subsegments or tune a
pretty field.

## Evidence packet

Compare only unchanged legacy and the one candidate on the same mesh, datum,
runoff, erosion settings, hydrologic policy, crop and cameras.

Required non-presentation evidence:

- source fronts, episode/crust inputs and compiled segment graph;
- per-segment and belt uplift-opportunity budget closure, field support and
  conversion provenance;
- starting surface after removal of legacy convergent ownership;
- cumulative uplift opportunity and final physical elevation;
- along- and cross-belt profiles through segment maxima and one transfer low;
- multiscale relief, summit-cap/low-grade descriptors and peak/saddle
  persistence as supporting measurements;
- drainage basins, major trunks, confluences and watershed divides over the
  final surface;
- alignment/correspondence between segment organization, final divides and
  longitudinal/transverse channels;
- drainage-integration repair overlap;
- area-weighted land-mask flips and comparison-domain boundary continuity; and
- wall time, peak memory and resolution-sensitive assumptions.

Required matched views are:

1. Physical 1× terrain with no snow/color saturation;
2. Diagnostic slope/hillshade plus uplift segments;
3. Diagnostic drainage basins, major channels and divides; and
4. Authentic presentation with the same declared relief and river policy.

Dramatic presentation is optional celebration after a decision, not admission
evidence. No new blind-review protocol, artifact registry or promotion schema is
required for this slice. One sidecar with source revision, effective settings,
ledger/cost summaries and image names is enough.

## Pass, kill and cost rules

The candidate passes this belt only if all of the following hold:

- **ownership:** legacy convergent height and repeated uplift are absent from
  the candidate final-surface path;
- **causal structure:** finite segment geometry produces corresponding finite
  relief maxima, ends or transfer lows without isolated beads or a regular
  dune/corduroy field;
- **internal hierarchy:** the physical surface has a connected but segmented
  main watershed divide, subordinate basin-bounding branches and more than one
  meaningful along-strike high/low relationship; a smooth elongated massif is
  still failure;
- **downstream consequence:** at least one organized low or flank difference
  changes basin ownership, trunk orientation or a defensible cross-belt channel
  relationship relative to legacy;
- **numerical restraint:** the declared uplift opportunity closes, no hidden
  height clamp or post-run relief normalization is applied, and peak/hypsometry
  remain usable rather than merely matching a threshold;
- **visual judgment:** Physical profiles, slope and diagnostic geometry confirm
  the morphology and introduce no seam, pillar or faceting regression; human
  review prefers the candidate in Authentic without relief scale hiding the
  physical result; and
- **cost:** compilation is linear in source-front edges plus belt cells,
  transient organization state is at most one fine-cell scalar plus the sparse
  graph, and the complete path stays close to the measured legacy Stage-3/4
  baseline because it reuses the same response operators. Report exact wall
  time and peak memory. If organization cost is no longer negligible beside
  existing routing/erosion, stop before optimizing unless it buys a distinct
  downstream consequence unavailable to the cheaper organizer.

Stop after the first matched result if it remains a long roof, becomes
disconnected blobs/spikes, relies on amplitude or presentation, or changes only
descriptive cap/peak metrics. Diagnose whether the organizer or surface
response failed, but do not start a parameter sweep. A second implementation is
authorized only by a named missing causal relationship, not by a desire for a
prettier image.

## Deliberate omissions

V0 does not add explicit faults, continuum crustal mechanics, a new tectonic
history solver, sediment, landslide events, flexure, glaciers, dynamic climate,
biome feedback, roads or culture. Existing cheap runoff and hydrology are
inputs/consumers and must remain available to later stages. A coherent material
or erodibility field is the most plausible second rung, but only if the fixed
belt shows that real crust transitions cannot supply enough organization.

`segment-transfer-low` is forcing provenance; `divide-saddle` is derived
final-surface topology. One does not imply the other without measured
correspondence. Neither is automatically a pass, wind gap, water gap, crossing
or chokepoint. Those names require their own consumer and topology, just as the
completed Consequential Geography slice required conservative route-relative
semantics.

## Consequence of the decision

The next implementation task is small: expose a product-facing front/episode
record, compile the selected belt's finite segment graph and uplift opportunity,
and prove its ledger on manufactured chains before touching terrain response.
It is not to revive O1/O2/O3, H/C/G, carrier tuning or the full planar evidence
campaign.
