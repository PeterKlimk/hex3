# Regional deformation support

Status: **current proposed architecture; research-only vertical slice selected;
not implemented or promoted**, 2026-07-23.

This document defines the missing seam between corrected plate-boundary work and
the coupled landscape response. It refines the
[terrain architecture design space](terrain-architecture.md) after human review
of the conservative finite-age source comparison. Legacy remains the usable
product terrain.

## Decision

Retain signed aggregation before positive classification as a necessary source
operation. Do not promote its current nearest-front terrain adapter as a terrain
architecture. It removes the false edge-scale comb but exposes a broad smooth
mound because one scalar boundary rate is still extruded through one static
cross-front profile.

The next owner is a **regional deformation program**:

```text
corrected signed boundary work
  + finite front topology, side and regime
  + supported material/inherited relationships
  + a small relative deformation history
    -> conservative two-dimensional deformation-opportunity frames
    -> explicit opportunity-to-rock-uplift adapter
    -> coupled drainage, incision and hillslope response
```

This is not a request for full geodynamics. It is the cheapest intermediate
state capable of representing finite structures, overlap, linkage, retirement
and migration without painting final ridges or valleys.

Do not begin with a product framework, a parameter ladder or a multi-seed
campaign. First run one fixed-budget mechanism discriminator on seed `8675309`,
episode `9`, at the ordinary fine resolution.

## Evidence that changes the decision

The raw finite-age Slice A source clipped convergence independently on every
Voronoi edge. Conservative signed aggregation at one collision width fixes that
operation order and closes the signed flux ledger near roundoff. In the coupled
episode-9 comparison, owner-rate maxima fall `73 -> 12`, final-crest maxima fall
`72 -> 29`, and final mean spacing expands `176 -> 520 km`.

Human review agrees with the preliminary visual diagnosis: the corrected arm
removes the conspicuous repeated teeth, but replaces them with an over-smooth
regional mound rather than a convincing organized range. The operation is
therefore retained as a source correction and rejected as a sufficient terrain
owner. No broader seed confirmation or aggregation-width tuning is warranted.

The reported `105` scheduled-uplift maxima are not the primary visual defect.
That field also contains the target-land recovery floor, asymmetric profile,
support sampling and masking at a much lower prominence threshold; only `29`
maxima survive into final terrain, which reads smooth. The architectural
failure is more direct: after coherent along-strike rates are installed, the
current adapter has almost no regional state left with which to organize a
range.

## Research synthesis

Scientific landscape-evolution systems normally accept tectonics as an
external spatial and temporal field rather than derive it from continuum
geodynamics. eSCAPE models the surface response to imposed tectonic and climate
forcing; FastScape accepts per-node uplift and horizontal velocity fields; and
Badlands consumes sequences of displacement maps, with the large complexity
jump occurring when horizontal material motion requires remeshing
([eSCAPE](https://doi.org/10.5194/gmd-12-4165-2019),
[FastScapeLib](https://fastscape.org/fastscapelib-fortran/),
[Badlands](https://badlands.readthedocs.io/en/latest/xml.html)). CHILD and
Landlab likewise expose analytic, block, fold, fault and time-varying uplift
prescriptions as reduced forcing models rather than final terrain
([CHILD guide](https://csdms.colorado.edu/w/images/Child_users_guide.pdf),
[Landlab NormalFault](https://landlab.readthedocs.io/en/latest/tutorials/normal_fault/normal_fault_component_tutorial.html)).

The useful physical consequences are also consistent across reduced fault and
fold studies:

- finite structures grow laterally, overlap and link, producing ends, relays,
  gaps and changing drainage rather than one permanently active band
  ([Cowie et al. 2006](https://www.research.ed.ac.uk/en/publications/investigating-the-surface-process-response-to-fault-interaction-a/));
- migrating uplift leaves different lags in channels, relief and hillslopes
  ([Clubb et al. 2020](https://www.research.ed.ac.uk/en/publications/differences-in-channel-and-hillslope-geometry-record-a-migrating-/));
- horizontal advection can move divides and alter flank geometry, so pure
  vertical extrusion is a declared approximation rather than a universal model
  ([Miller et al. 2007](https://ftp.ems.psu.edu/pub/geosc/sling/PUBLICATIONS_SLINGERLAND/2001-2010/Milleretal2007.pdf),
  [He et al. 2021](https://doi.org/10.1038/s41467-020-20748-2)); and
- a moving smooth uplift wave is a useful regional envelope but explicitly
  smooths discontinuous fault deformation and does not reproduce material
  advection across a ramp
  ([Li et al. 2025](https://www.fs.usda.gov/rm/pubs_journals/2025/rmrs_2025_li_y001.pdf)).

Graphics provides a particularly relevant authentic-hack precedent.
*Sculpting Mountains* separates approximately conserved crustal thickening, a
main fault and material-controlled folds from the uplift map subsequently
consumed by erosion. The fault/fold system is procedural and incomplete, but it
organizes forcing rather than stamping final ridges
([Cordonnier et al. 2018](https://doi.org/10.1109/TVCG.2017.2689022)). Earlier
uplift-first terrain work similarly treats a two-dimensional uplift map as the
interface to stream-power drainage and reconstruction
([Cordonnier et al. 2016](https://doi.org/10.1111/cgf.12820)).

These precedents do not license periodic folds. Regular bands can become
corduroy, a moving smooth wave can remain a moving mound, and a library of
analytic kernels can become experimental bureaucracy. Drainage-first synthesis
and explicit ridge/divide trees remain valuable topology upper bounds and
possible sub-grid completion methods, not evidence for the physical owner
([Génevaux et al. 2013](https://doi.org/10.1145/2461912.2461996),
[Argudo et al. 2019](https://doi.org/10.1145/3355089.3356535)). Multiscale
erosion or structured procedural ravines can later buy back local detail, but
they preserve the coarse regional model, including its mistakes
([Schott et al. 2024](https://doi.org/10.1145/3658200),
[controlled procedural patterns](https://doi.org/10.1111/cgf.14992)).

## Ownership boundary

`RegionalDeformationProgram` owns only the map from one-dimensional boundary
shortening opportunity to persistent two-dimensional deformation opportunity.
It does not own:

- final terrain height;
- drainage, erosion or hillslope response;
- cartographic relief or fine visual texture;
- crust creation, underthrust volume or isostatic conversion; or
- a claim that synthetic relative epochs reconstruct geological history.

The seam deliberately has two quantities. Boundary shortening flux has units
of `km²/Myr`; distributing it over receiving area produces a deformation-rate
density with units `1/Myr`. Calling that field rock uplift would require an
effective thickness, retention and surface-response conversion that Hex3 has
not selected. The first terrain discriminator may use a separately named
adapter that normalizes the spatial program to the same demoted-Legacy builder
volume. That is a fixed-budget morphology comparison, not physical mass
closure.

## Minimum program state

The sparse program should retain, per deformation element:

- stable element, parent segment, epoch and source-edge identity;
- active interval and growth, linkage or retirement state;
- finite along-strike support with true ends;
- receiving side, regime, vergence and a finite cross-strike support;
- corrected shortening-opportunity budget and any explicit unallocated share;
- optional inherited-structure relationship actually consumed; and
- accumulated deformation age/opportunity plus a local axial fabric.

Frames evaluated on the process mesh expose rate density, active-support
fraction, fabric, regime/side and sparse contribution weights. Multiple elements
add or partition work; they do not compete through a nearest-front winner.
Large per-epoch meshes need not be retained: a small sparse program can stream
one frame and its provenance into the existing erosion epoch.

The program consumes no elevation, current land mask, drainage or renderer
state. Submarine support remains eligible, and the fine mesh cannot choose the
geology it later resolves.

## Source and history rules

Reuse the current source machinery rather than recreating it:

1. `StructuralMountainGraph` supplies exact finite segments, tips, links,
   regimes, sides, ordered edge provenance and parent ledgers.
2. The accepted signed-before-clipping operation supplies corrected local rates.
   Positive admission occurs once when regional support is allocated.
3. Plate/crust ownership bounds connected receiving material.
4. `LithosphereInheritanceV0` may redirect, connect or localize support only
   where a named generated relationship genuinely intersects the active source.
   Proximity and unnamed contacts are neutral.

Current `TectonicHistory` supplies fixed present geometry and relative onset
only. It cannot support a claim of migration, reversal or abandonment. The
first mechanism discriminator should therefore use an explicitly labelled
**reduced kinematic counterfactual**, not silently reinterpret current episode
ages. It partitions one real parent's corrected work among a small sequence of
finite elements that grow, overlap/link and shift cross-front through a few
relative epochs. It tests whether this missing causal consequence matters; it
is not yet the product history generator.

If that counterfactual earns a terrain consequence, the next source decision is
whether its program should come from a compact birth/growth/link prior or from
the existing carrier replay's changing material/contact frames. The carrier is
a useful history reference, but its resolution sensitivity, stochastic Euler
reorganization and remap cost prevent selecting it in advance. Do not revive
the lifecycle terrain model wholesale.

## Conservation and topology invariants

For every processed parent, epoch and receiving side:

```text
sum(cell_area_km2 * opportunity_rate_density_per_myr)
  + explicit_unallocated_km2_per_myr
  = sum(source_edge_length_km * max(corrected_signed_rate, 0))
```

The program must additionally satisfy:

- parent and global ledgers close before any Legacy-budget adapter;
- source-edge subdivision cannot create a support maximum or change total work;
- support remains connected through eligible receiving material;
- true tips bound support and declared links alone permit transfer or
  continuation;
- subduction loads the overriding/receiving domain; collision uses an explicit
  two-side rule rather than silently treating both sides as one gentle flank;
- overlaps are additive within a conserved partition, not max-blended;
- inheritance is budget-neutral and bit-neutral when no named relationship is
  consumed;
- coarse-to-fine resampling preserves integrated opportunity and provenance;
  and
- input ordering and exact repetition are deterministic.

Unsupported work is typed and visible. At minimum distinguish historical
support unavailable, a closed loop without a finite parent, no eligible
receiving material, ambiguous side semantics, disconnected support,
unsupported inheritance and underresolved fine support. Raw-fallback terrain is
not an omission policy.

## One bounded vertical slice

Use seed `8675309`, episode `9`, the ordinary roughly 256k-cell fine surface and
the accepted 127 km signed source operation. Compare only:

- **control:** current conservative rate, one nearest present-front owner and
  static asymmetric profile;
- **counterfactual:** the identical corrected parent work and age envelope,
  conservatively partitioned among a small finite growth/link/retirement
  sequence with cross-front movement and additive overlap.

Before terrain, show the two-dimensional support, fabric and provenance; prove
per-parent closure, subdivision invariance and the absence of nearest-owner
bisector seams. Confirm whether the selected parent consumes any named
inheritance. If it does not, inheritance must be exactly neutral.

Then run one terrain pair with identical non-orogenic base, precipitation,
erosion response, total Legacy-derived builder budget and physical/diagnostic
presentation. Human review remains necessary.

The counterfactual passes only if it retains comb removal and changes a
regional object relationship: finite range/massif ends, a supported saddle or
low, drainage-basin/divide organization, or coherent transverse/longitudinal
trunk response. It need not create sharp peaks. An honestly continuous parent
may remain a broad massif when the support program supplies no reason to split
it.

Kill the mechanism if it produces another smooth mound, periodic ribs, beads,
isolated hot spots, owner seams or only fine texture; if organization depends on
inheritance absent from the generated source; or if the existing response
erases materially different support. Do not respond with a width or epoch-count
sweep.

## Deliberate omissions and later representation

V0 omits stress balance, critical-wedge solving, earthquake cycles, explicit
fault planes, full horizontal terrain advection, calibrated crustal thickness
conversion and full plate reconstruction. A single persistent deformation-age
or material-provenance field is preferable to advecting the whole terrain until
horizontal motion demonstrates visible or downstream value.

Old `MassifCorridor`, fold-grain and O0 fields remain authored-pattern controls.
A4 retains the useful idea that drainage can redistribute later forcing, but it
does not supply tectonic history. None should be enabled inside this slice.
Conditional multiscale erosion or structured ravine synthesis belongs after
regional support passes and only for detail that the physical mesh need not own.
