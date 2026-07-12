# Channel support, surface meaning and mesh convergence — 2026-07-13

Status: **design decision for the bounded orogen testbed**. This note refines
the Slice 1 failure diagnosis and defines the numerical work that must precede
another U/L experiment. It is not yet a product architecture decision.

Related evidence:

- [mountain-system design basis](mountain-system-design-basis-2026-07-13.md)
- [bounded orogen testbed](orogen-testbed-spec-2026-07-13.md)
- [Slice 1 audit](../audits/orogen-testbed-slice1-2026-07-13.md)

## Decision

Use a **cell-mean finite-volume surface with effective areal fluvial
denudation** as the next testbed representation, called **C0** below. Water is
routed as face flux on the physical mesh; fluvial forcing is expressed through
a convergent specific-discharge field rather than raw accumulated cell
discharge. Lowering a cell then removes the declared solid volume
`E * cell_area * dt`.

C0 is deliberately a regional, coarse-grained landscape model. At 2–8 km it
may claim mean rock-surface evolution, drainage organization, water balance and
solid-volume balance. It may not claim to resolve channel-bed elevation,
channel width, gorge geometry, floodplains or valley cross-sections.

Do not repair Slice 1 by scaling `K` with cell size or by attaching a width to
the current single elevation field. Those changes can make one budget agree
while leaving the represented physical quantity ambiguous.

Retain two alternatives as explicit controls or later branches:

- **P — pathway surface:** elevation is the dominant flow-path or channel-bed
  sample used by FastScape-style stream-power solvers. It is useful for profile
  and topology experiments, but cell-area multiplication does not turn its
  incision into a physical solid-volume ledger.
- **C1 — dual subgrid channel:** each coarse cell has at least mean surface,
  channel-bed elevation, reach length and effective width, plus an explicit
  bank/valley coupling law. This is justified only if C0 passes its numerical
  contracts but fails because bed–interfluve disequilibrium is a necessary
  state.

Mobile sediment or cover is **C2**, orthogonal to this choice. Separating
bedrock from alluvium can represent storage and cover effects, but does not by
itself define the spatial support of a channel inside a coarse cell.

## Why Slice 1 closes its ledger but does not converge

The implementation combines three individually reasonable ideas with
incompatible meanings:

1. priority-flood/SFD routing selects a one-cell-wide path;
2. implicit stream-power incision computes a vertical change appropriate to a
   channel or dominant pathway;
3. the volume ledger multiplies that change by the entire finite-volume cell
   area.

On an established one-cell reach, the implied eroding width is proportional to
mesh spacing. Refinement therefore narrows the implicit river and changes gross
export even when the profile law is stable. In headwaters there is an
additional scale dependence: local source discharge is runoff times cell area,
nearly every receiver-bearing cell is incision-eligible, and SFD creates a
resolution-dependent population of parallel paths and merger points.

The ledger proves that each mesh accounts exactly for the volume its own
operator chose to remove. It does not prove that different meshes chose the
same physical removal.

The open boundary adds another changing quantity. Every north/south boundary
cell is currently a receiverless outlet and a full cell row is held at base
level. Refinement changes outlet count and spacing, catchment partitioning, the
width of the base-level reservoir and the effective location of the boundary.
This is not a floating-point defect; it is an unresolved boundary model.

The mesh geometry, area-normalized uplift forcing and conservative hillslope
flux do not show the same semantic mismatch. They should be retained while the
fluvial and boundary contracts are replaced.

## Correspondence with established models

There are two legitimate traditions which must not be silently merged.

FastScape's implicit algorithm treats the elevation at a node along a drainage
tree as the quantity evolved by a stream-power path law. Its numerical strength
is rapid, stable profile evolution, not a declaration that the channel occupies
the node's full control area. Braun and Willett describe that path-based
implicit formulation in
[FastScape](https://doi.org/10.1016/j.geomorph.2012.10.008).

CHILD instead describes node elevation as the average elevation of a Voronoi
cell for bulk mass-balance updates, while its channel geometry is a separate
subgrid concept. That distinction is explicit in the
[CHILD model description](https://csdms.colorado.edu/w/images/Child_users_guide.pdf).

At landscape-model resolutions, real channels are normally subpixel. Stark and
Stark frame channelization and the channel–hillslope transition as a subgrid
problem rather than a reason to treat each pixel as a literal river cross
section ([2001](https://doi.org/10.1029/2000WR900414)). Pelletier shows that
using specific catchment area—contributing area per unit contour width—improves
the scaling of continuum erosion laws over raw contributing area
([2010](https://doi.org/10.1029/2009JF001435)).

Hergarten analyzes the exact coupled-model artifact at issue here: rivers
represented as one-pixel lines make fluvial versus hillslope response depend on
pixel size. The proposed threshold-based scaling is a useful inexpensive
alternative, but the paper also makes clear that there is no universal channel
width correction for every grid model
([2020](https://doi.org/10.5194/esurf-8-367-2020)). Kwang and Parker further show
that common stream-power exponents can have scale and ridge singularities even
apart from implementation bugs
([2017](https://doi.org/10.5194/esurf-5-807-2017)).

Explicit width and lateral-erosion models exist, but width becomes useful state
only with a closure law and cross-section consequences. Examples include
dynamic width adjustment in [Attal et al.
2008](https://doi.org/10.1029/2007JF000893) and coupled vertical/lateral erosion
in [Langston and Tucker
2018](https://doi.org/10.5194/esurf-6-1-2018). Likewise, SPACE adds bedrock,
sediment and cover interaction but still does not make a one-cell channel a
resolved valley cross-section ([Shobe et al.
2017](https://doi.org/10.5194/gmd-10-4577-2017)).

The lesson is not that one literature family is correct and another is wrong.
It is that the state meaning, spatial support and ledger must come from the same
family.

## C0 state and operator contract

The authoritative scalar is renamed conceptually to make its meaning explicit:

```text
mean_bedrock_elevation_km[cell]
```

For each routing update:

1. derive a routing surface from the physical mean surface; depression filling
   may alter only this derived field;
2. distribute local runoff through downhill faces with a finite-volume
   multiple-flow-direction rule;
3. record face water flux `Q_face` in `km³/Myr` and close the water balance;
4. derive specific discharge from physical support, beginning with
   `q_face = Q_face / face_width` in `km²/Myr`;
5. combine incident face values into a declared cell field and compute an
   effective areal denudation rate, for example `E = K q^m S^n`, in `km/Myr`;
6. lower the mean surface and record `E * cell_area * dt` as solid export.

The exact cell reduction of face flux and slope is not selected by prose. Plane,
radial and convergent manufactured cases below must choose it. Simply dividing
SFD contributing area by nominal cell spacing would preserve the same hidden
grid dependence under a new name.

MFD is the research default because a continuum specific-area field requires
distributed face support. SFD remains a useful limiting/control routing model,
especially after a physical channel-initiation rule exists. eSCAPE provides a
relevant finite-volume/MFD landscape-evolution precedent
([Salles and Hardiman
2019](https://doi.org/10.5194/gmd-12-4165-2019)).

A channelization transition may later use a threshold with physical units and
a smooth or explicit interpretation. It may not be expressed as a cell count,
and it must be held fixed in physical space during refinement.

### Scalar intensity at convergence

C0 uses the magnitude of the consistently reconstructed continuum water-flux
vector as its local specific-discharge intensity. This is authoritative only
where the face samples admit one resolved local vector. It is not silently
replaced by absolute-face, RMS or cell-width discharge at junctions: those are
different constitutive closures with mesh- and support-dependent
normalizations.

At an unresolved confluence or exact line sink, opposing face fluxes need not
admit one local vector and its magnitude may cancel. Such a point is a known
truth limit, not a place to fit `K`. Validate it with integrated flux across a
declared physical/graph cut and with a resolved downstream reach; do not score
an invented pointwise specific discharge at the singular junction. If this
localized limitation materially prevents the required valley/range
organization, compare a fixed-physical-scale filtered C0 arm or explicit C1
channel/skeleton state as separate representations rather than switching
locally by heuristic.

This follows the consistent continuum-flux interpretation discussed by
[Porporato et al. 2024](https://doi.org/10.5194/esurf-12-995-2024). Traditional
specific catchment area remains discharge per contour width, but the contour or
resolved support must be declared; it is not physical river width.

### Frozen first smoke normalization

The coupled solver's default coefficient remains zero. For the first unseen
short U/L smoke only, preregister the normalized direct-unit-stream-power arm:

```text
R0 = 500 km/Myr                 (0.5 m/yr runoff depth)
a0 = 100 km                     (specific contributing area)
q0 = R0 a0 = 50,000 km²/Myr
S0 = 0.02
E0 = 0.1 km/Myr                 (0.1 mm/yr)
m = 1, n = 1
K = E0 / (q0 S0) = 1e-4 km^-1
D = 0.1 km²/Myr, Sc = 0.7
```

Conceptually record the law as
`E = E0 (q/q0)^m (S/S0)^n`, even if code stores the equivalent `K`. Freeze all
six reference quantities before viewing output. `E0` is an order-of-magnitude
response-time prior, uncertain by at least a factor of ten, not an Earth fit or
product default.

Here `m=1` is the causal first hypothesis because `q` already means discharge
per flow width: unit stream power is proportional to `qS`. The familiar
area-law exponent near `0.5` already absorbs runoff and hydraulic/channel-width
scaling; reusing it on explicit `q` repeats that abstraction. See
[Tucker and Whipple 2002](https://doi.org/10.1029/2001JB000162) and
[Finnegan et al. 2005](https://doi.org/10.1130/G21171.1).

At the reference point, 0.1 Myr removes about 10 m and 1 Myr about 100 m. With
`K q0 = 5 km/Myr`, an `n=1` profile response time is roughly 4 Myr over 20 km
and 20 Myr over 100 km. Therefore 0.1–1 Myr U/L is a transient numerical and
organization smoke, not mature-mountain validation. `D=0.1` is likewise an
effective coarse closure: it modestly damps the shortest mesh wavelengths in a
1 Myr run but should not organize a regional range by itself.

## Outlet and domain contract

Convergence cases use fixed outlet **portals**, not one semantic outlet per
boundary cell. For the hex testbed, the conservative domain is the union of
complete hexagonal control volumes selected to approximate a fixed target
rectangle. Its actual area, boundary arc and offset from the target are recorded
at every resolution; morphology is scored inside a fixed physical buffer.

- A portal has a stable identifier, projected coordinate span, base level and
  boundary type.
- Genuine exposed hex faces intersecting one portal share its identity and
  apportion its flux; refinement may change face count and sawtooth arc length
  but not portal topology or projected span.
- Dirichlet base level is applied at the physical boundary face or a ghost
  state using the half-cell distance. A full interior row is not pinned.
- Closed faces carry zero normal water and sediment/solid flux as declared.
- Exact rectangular Dirichlet-flux formulas use a small 1-D or Cartesian
  manufactured fixture. A true rectangular hex domain would require a complete
  cut-cell/Voronoi mesh, not nominal faces attached to full cells; that machinery
  is deferred unless boundary sensitivity becomes a discriminator.
- A continuously open coastline is a separate physical experiment, not the
  convergence boundary for the bounded orogen cases.
- Metrics use the specified central scored window; buffer cells remain visible
  to conservation audits but cannot determine morphology scores.

## Analytic gate before U/L

The next implementation slice is numerical infrastructure, not another terrain
comparison. Tests are ordered so later evidence cannot hide an earlier semantic
failure.

### A. Pathway-law control

On a fixed one-dimensional receiver chain with prescribed area/discharge,
uplift and base level, reproduce the implicit transient recurrence and the
steady profile

```text
S = (U / (K A^m))^(1/n).
```

This validates the existing implicit solver as a P/pathway operator. The test
does not multiply its change by cell area or claim a physical incision volume.

### B. Boundary and portal geometry

- on an exact 1-D/Cartesian fixture, diffuse a linear ramp against a Dirichlet
  ghost boundary and recover the known boundary flux at 8/4/2 km;
- route uniform supply to one and then two fixed physical portals;
- require stable portal identities and projected spans, conservative total
  discharge, and stable buffered basin assignment under refinement and mesh
  rotation; report the actual hex-domain area and boundary offset.

### C. Finite-volume water routing

Use planar, radially divergent, radially convergent and rotated-mesh surfaces.
Require water closure and convergence of the analytic or high-resolution
specific-discharge field. Compare SFD, two-direction and MFD rules; select the
cheapest rule that meets the field and symmetry errors.

### D. Depressions and flats

Use a known sill, nested bowls and flat outlets. Require invariant outlet
selection and water balance while byte-checking that priority filling never
changes the physical mean surface.

### E. Manufactured C0 denudation

Prescribe a smooth specific-discharge/slope field with known effective
denudation. Require the denudation field, exported volume and relief response to
converge monotonically from 8→4→2 km. Repeat with routing enabled only after the
prescribed-field test passes.

### F. Coupled U/L

Only then rerun the coupled cases. Compare the scored interior, ledgers and
object identities; do not require a visually satisfying range from the
numerical null.

An optional C1 discriminator may prescribe a straight subgrid reach and verify
`sum(E * width * reach_length * dt)` plus its transfer to mean cell volume. It
belongs after C0 unless an earlier test proves that no meaningful cell-mean
closure exists.

## Stop and promotion rules

- If portal geometry or specific discharge does not converge, stop before
  fitting `K` or uplift.
- If P converges as a profile model but its inferred cell volume does not,
  retain it only as a pathway control.
- If C0 passes analytic and U/L numerical gates, it becomes the coupled arm for
  the organization testbed—not automatically the Hex3 product solver.
- If C0 is numerically sound but later lacks necessary channel/interfluve
  memory, compare C1 against P and an authentic explicit drainage-skeleton
  control.
- Add mobile sediment only to answer storage, cover, deposition or
  source-to-sink questions. It is not a convergence patch.

## Consequence for the larger project

C0 matches Hex3's present need better than a simulation-grade channel model:
one authoritative regional surface can feed hydrology, climate, ecology,
traversability and cartography, while its costs and truth limits remain clear.
It preserves the physically important causal direction—uplift, runoff and
material transport evolve surface geography—without pretending that a 2–8 km
cell resolves a river valley.

That is an authentic coarse-grained model, not a promise that more process
detail is always better. If a cheaper explicit graph later produces stronger
and more stable geographic organization, the testbed is designed to reveal
that rather than protect C0.

## Implementation evidence and next discriminator

The analytic C0 stack and the corrected 0.1 Myr U/L screen now pass the
conservation, boundary and integrated-relief gates. The correction matters:
physical outgoing-face grade including portal datum plus a routed-distance
slope Courant limit removes the former 2 km runaway. Across 8/4/2 km, corrected
relief remains within `0.40%` per refinement and ledgers close within
`1.9e-10 km³`.

The short screen also isolates what C0 does not resolve. Peak cellwise
denudation grows `4.4–5.1×` across 8→2 km, total export is not yet asymptotic,
and explicit stable work scales approximately with `h^-2` in time as well as
the growing cell count. The evidence therefore supports C0 as a conservative
regional surface arm, but not yet as a resolution-independent local channel
intensity model.

The next discriminator is one representation comparison with frozen forcing
and `K`, not a calibration sweep:

1. retain unfiltered C0 as the control;
2. apply one fixed-kilometre scalar Helmholtz support closure whose filtered
   quantity and downstream meaning are declared before the run;
3. specify the minimum C1 memory (`z_mean`, channel-bed elevation, width/reach
   measure, and exchange law) needed by sediment, valleys and ecology, and
   estimate its cost without implementing the full model.

Compare relief and export convergence, peak/local intensity, drainage identity,
water/solid closure and accepted work. A filter wins only if it regularizes
local support and cost without moving drainage semantics or smearing erosion
across interfluves. Otherwise the result is evidence for C1 or an explicit
drainage skeleton—not a reason to make the filter wider or retune `K`.
