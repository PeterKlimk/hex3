# Orogen organization: causal reference and reduced experiments — 2026-07-12

Status: synthesized research note. It sharpens the active mountain experiment;
it does not promote a replacement model.

## Decision and failure

Hex3's defect is not that broad, low-relief highlands exist. Distributed
shortening, resistant interiors and internal drainage can produce real broad
plateaus. The defect under test is narrower: the product path uses a smooth,
capped, approximately equal-width response as the generic grammar of
convergent mountain belts.

The decision is whether a cheap, causally conditioned construction can produce
several legitimate orogen families—including occasional plateaus—without a
full geodynamic model.

## How reality avoids a universal smooth cap

Construction and destruction are separate causes.

**Constructional organization** comes from localized and migrating deformation,
fault and fold systems, inherited weak structures, lithospheric strength,
subduction polarity, obliquity and along-strike changes in convergence. These
create narrow or broad belts, asymmetric flanks, offset axes, saddles,
intermontane lows and segmented deformation rather than one cylindrical
response. Observed Himalayan structure, for example, is segmented by inherited
lithospheric structure ([study](https://www.nature.com/articles/srep33866));
numerical collision experiments likewise find that inherited rift geometry and
polarity affect whether mountain belts remain continuous or become
non-cylindrical ([study](https://www.nature.com/articles/s41467-025-66695-8)).

**Erosional organization** then routes branching channels, migrates divides,
captures basins and retreats or fails hillslopes. It can dissect or destroy a
plateau margin, but arid, internally drained or resistant plateaus can retain
low-relief interiors. Puna basin reintegration illustrates how capture and
external drainage can initiate rapid plateau incision
([study](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2021JF006147)).

This distinction fits the Hex3 ancestry evidence: erosion can create substantial
local relief while retaining the smooth constructional envelope it was given.

## How simulations and graphics compress the causes

Landscape-evolution models commonly combine prescribed spatially and temporally
varying displacement with routed stream-power incision and hillslope transport.
[FastScape](https://fastscape.readthedocs.io/en/latest/models.html) exposes the
minimal block-uplift, fluvial-incision and diffusion model;
[Badlands](https://badlands.readthedocs.io/en/latest/) accepts varying horizontal
and vertical tectonics, climate, incision, hillslopes and sediment transport.
Coupled geodynamic models can supply strain- or fault-derived displacement, but
Hex3 need not reproduce their mechanics to preserve localized axes,
segmentation, polarity, intervening lows and heterogeneous width.

Graphics approaches often preserve a large-scale scaffold and spend cheaper
processes on its expression. Cordonnier et al. combine an uplift field with
fluvial erosion for large terrain
([paper](https://diglib.eg.org/items/13e52c36-0200-4652-aacf-17aa3098c5fd));
Michel et al. construct folded mountain organization from sparse vector input
before erosion ([paper](https://diglib.eg.org/items/5ecd72bf-b3a5-46be-b108-54e556b23f99)).
This becomes an authentic Hex3 hack when the scaffold is conditioned on plate
causes and changes the authoritative surface consumed by later systems.

## Causal signatures to preserve

A candidate should be judged on consequences rather than resemblance alone:

- a connected but segmented crest/divide organization;
- meaningful along-strike variation in height, width and continuity;
- asymmetric flanks tied to tectonic polarity where applicable;
- subsidiary ridges, saddles and foreland or intermontane lows;
- fewer universal large low-slope caps, without forbidding genuine plateaus;
- hydrologic basins and crossings organized by the resulting divides and lows;
- rain-shadow continuity and leakage controlled by flanks and passes;
- elevation/aspect connectivity suitable for future biome and ecology stages.

Adding micro-peaks to an unchanged broad cap does not satisfy these signatures.
Nor does a cosmetic field that fails to affect authoritative routing.

## Budget-matched discriminating experiments

### O1 — cross-section geometry

At fixed seed, replace only the smooth capped cross-section with a peaked or
asymmetric profile. Conserve belt-integrated positive tectonic work and match
global land hypsometry closely; keep erosion, integration and presentation
identical.

Compare the three named seed-12345 ranges before erosion and at final state:
cap area, low-grade summit coverage, width distribution, flank asymmetry,
crest/divide organization and fixed-camera morphology.

### O2 — along-strike segmentation

Independently modulate the existing source along the boundary with a
deterministic, low-frequency, zero-mean signal conditioned on boundary
kinematics or geometry. It may create saddles, gaps, offsets and width changes,
but must conserve total positive tectonic work.

Measure range continuity and variation plus river crossings, basin hierarchy
and rain-shadow leakage. Peak count alone is explicitly insufficient.

### O3 — structured fine redistribution

Redistribute elevation within the same broad envelope through causal crest axes
and adjacent lows, with zero net added uplift. Compare it with equally powered
isotropic structure. A useful structural layer should change divide alignment
and longitudinal/transverse drainage; equivalent results imply mere texture.

### O4 — erosional capability control

Hold the current cap fixed, disable or strongly reduce legacy rebuilding, and
introduce only a modest seeded drainage/base-level corridor. This asks whether
the existing erosion system can branch, migrate divides and break up the cap,
or merely roughen it. Keep this to the three named ranges at the diagnostic
budget.

## Recommendation

Run O1 and O2 first and independently. They test the diagnosed constructional
owner more directly and cheaply than adding geological machinery. Promotion
requires a change in authoritative pre/final morphology and at least one useful
downstream organization signature under a conserved budget. Improved relief or
render appeal alone does not pass.

## First O1 result

The tested rung (`legacy-peaked`) used compact biquadratic
continental profiles and normalizes positive arc and collision work separately
per plate. On seed 12345 at 100k coarse/250k fine it preserves plausible peak
height and mountain area and raises some regional-relief percentiles, but summit
cap area worsens, drainage orientation is nearly unchanged and matched views
remain very similar. This falsifies that compact normal cross-section geometry
alone, in this form, is sufficient. It does not falsify segmentation or causal
crest organization. Human review agreed with rejection, and the implementation
was removed rather than parked. Proceed to O2 independently rather than tuning
O1 widths.

## First O2 result

The tested `legacy-segmented` rung keeps the exact legacy Gaussian normal
profile and changes only along-strike continental response strength. The driver
is local convergence orthogonality smoothed over 400 km on each connected
boundary episode; positive response is conserved independently by episode,
feature type and receiving plate.

Seed 12345 contains real kinematic variation: episode-kind orthogonality span is
0.57/0.92 at p50/p90, and the conserved response multiplier reaches roughly
0.959/1.041 at p02/p98 with a 10.3% maximum deviation. Nevertheless the 100k
coarse/250k fine result leaves peak, mountain area, regional relief and drainage
organization nearly unchanged; summit-cap changes are mixed and elongation
slightly declines. Matched views retain the broad tableland grammar.

This is evidence against present-day orthogonality-driven **strength-only**
segmentation as the missing mechanism. It does not reject time-varying or
inherited segmentation, explicit offsets/width changes, or a causal
crest/divide hierarchy. Do not infer that the right response is a larger
arbitrary multiplier.

Human visual review agreed with rejection, and the implementation was removed.

## Historical precursor correction

A literal crest-skeleton O3 would skip unresolved evidence already present in
the repository history. P1 and generic meso fold trains added positive ridge
fabric but did not organize drainage and often read as corduroy. O0 combined a
front crest with scalar along-strike segmentation. Neither supplied an internal
crest/divide/valley hierarchy.

Two parked mechanisms are closer to the functional decision:

- `MassifCorridor` builds unequal massifs and branching oblique/transverse low
  corridors in the pre-hydrology base. It is a structure-first authentic hack,
  numerically evaluated but never given a decisive current visual/ownership
  disposition.
- A4 burns in drainage, selects major trunks, shifts uplift away from trunks and
  toward interfluves, then freezes the field for a final epoch. It is a
  drainage-first organization probe, but its implementation is coupled to the
  parked emergent/O0 path and risks circular ownership.

The next experiment should compare organization owners before inventing another
generator: legacy/process-only, equal-power isotropic, structure-first
massif/corridor and drainage-first frozen scaffold. An explicit joint
crest/divide graph becomes justified only if those arms demonstrate a functional
topology benefit that existing procedural state cannot retain or expose as
stable semantic objects.

## First O3A result

The structure-first reuse arm now has a fair amplitude control. On seed 12345 at
100k coarse/250k fine, `MassifCorridor` base relief 0.05 and isotropic P1a
relief 0.0611 each contribute 44 m area-weighted RMS over the fixed tectonic
process footprint. All other structural masters are zero and both feed the same
pre-hydrology surface, erosion, integration and renderer.

MassifCorridor reduces low-grade cap coverage and produces fewer basin fragments
than equal-energy isotropic structure, so its object conditioning is not wholly
inert. It does not, however, produce a distinct functional mountain class:
final multiscale relief and drainage orientation are comparable to isotropic,
pass/river hierarchy does not improve consistently, and matched terrain retains
the broad tableland grammar. Ancestry views show the prescribed field in the
fine base and substantially weaker differentiation after common erosion.

Do not optimize its amplitude. This first result makes the drainage-first frozen
scaffold the more informative remaining architecture arm. It also establishes a
bar for any future joint graph: it must outperform equal-energy isotropic relief
in authoritative channel/divide topology, not merely make the base look more
structured.
