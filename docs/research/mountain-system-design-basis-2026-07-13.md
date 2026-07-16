# Mountain-system design basis — 2026-07-13

Status: research synthesis and architecture basis. It supersedes the assumption
that the next useful step is another mountain-shape parameter sweep. It does not
promote a replacement implementation.

## Decision

Pause O3B implementation and further tuning of the current mountain-shape
family. O1, O2 and O3A tested cross-section, scalar segmentation and prescribed
fine structure. Their negative results are better explained as a missing causal
representation than as three unlucky parameter choices.

The next decision is which **organization owner** Hex3 should use. Research
supports a reduced coupled system in which tectonics supplies persistent,
heterogeneous rock-uplift and material structure; drainage and hillslopes
coevolve with that forcing; and surface elevation is the outcome. A synthetic
drainage graph and an explicit ridge/valley skeleton remain valuable controls,
not presumed product solutions.

## 2026-07-16 discriminator update

The thin 4 km H/C/G discriminator has resolved the first architectural question
negatively. It did not identify a winning organization owner because the shared
forcing representation lacks the regional degrees of freedom that the owners
were meant to organize.

The linked case consists of two geometrically equal `350 × 100 km` support
segments, one synchronous episode, equal work shares and cosine taper confined
to the terminal 22% of each segment. The central 56%—about `196 km`—is exactly
constant along strike. Segment links and vergence are retained but ignored by
the compiler, horizontal velocity is zero, material and runoff are homogeneous,
and the overlap is stencil addition rather than a mechanical transfer zone.

The sharp matched diagnostic and physical profiles show the consequence:

- H restores the pointwise displacement target before every carve pass and
  explicitly maintains the long roof;
- C recomputes routing and erosion as the surface evolves, but receives the
  same static spatial stencil and has no inherited localization, moving support
  or persistent channel state; it adds local texture without breaking the roof;
- G uses the displacement opportunity both to plan its minimax forest and to
  scale reconstructed edge rises, converting the roof into steps and spikes;
  and
- one release realization costs about `323 s` for H, `636 s` for C and `2.25 s`
  for G. G's speed does not compensate for failed morphology, while C must
  demonstrate much more value before its cost is justified.

Evidence is retained in
`artifacts/captures/orogen-owner-hcg-4km-diagnostic-2026-07-16/`. This is a
negative result for the shared static-ribbon representation and the current G
reconstruction. H remains a null/control. The broader reduced-coupled family
remains plausible, but the present C instantiation is not a viable product
candidate or evidence that erosion alone should create massifs from a uniform
ribbon.

### Refined physical and production prior

Real and simulated belts generally acquire first-order along-strike structure
before surface erosion finishes the terrain. Persistent inherited weakness,
strong blocks, fault-system localization, bends and stepovers, differential
collision onset and changing motion produce finite active domains, relay zones
and abandoned structures. Drainage competition makes those differences legible
and can amplify them; it should not carry the entire burden of symmetry
breaking. Inherited-rift models produce non-cylindrical belts and topographic
transition lows ([Gouiza and Naliboff 2021](https://www.nature.com/articles/s41467-021-24945-5),
[linked-rift inversion study](https://www.nature.com/articles/s41467-025-66695-8));
fault observations and models retain nested segments, displacement maxima and
intersegment lows ([Manighetti et al. 2015](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2014GC005691)).

Graphics reaches a compatible reduction. Schott et al. author hierarchical
curve-based **uplift** segments with varying width and along-strike intensity,
then let stream-power erosion construct terrain. Their important lesson is that
warping or blending final elevation damages drainage and proportions, whereas
structure placed in the uplift domain survives the physical response
([Large-scale Terrain Authoring through Interactive Erosion Simulation](https://artefacts-discovery.researcher.life/full_text/DA-2/33/332155303592301b8a92eeb517b04810/full_text/394ff426518242669ff7e43ffbfc4951.pdf)).
Minimal uplift–incision–creep models likewise show how drainage competition
creates finite ridge/valley spacing once meaningful forcing and initial
asymmetry exist ([Perron et al. 2009](https://www.nature.com/articles/nature08174)).

### Revised minimum mechanism

Introduce a persistent **orogen organization graph** between boundary history
and surface forcing. It is an authentic reduction of fault-system envelopes,
not an explicit fault or geodynamic solver.

On plate-bound material retain a small coherent inheritance field: competence
or weakness, preferred fabric and a few strong blocks or weak lineaments. On
each materially convergent boundary chain retain cumulative work,
onset/maturity, local convergence and obliquity through time,
damage/localization memory, active segment identity, endpoints, polarity and
linkage/transfer relationships.

The graph should:

1. nucleate separated candidates from convergence work, inherited weakness and
   fabric/stress alignment;
2. grow a sparse hierarchy of overlapping finite parent/child segments rather
   than independent Gaussian bumps;
3. stop, bend, split or link segments at coherent material and kinematic
   transitions, leaving finite relay zones and potential passes;
4. make segments compete for one shortening/material-work budget;
5. emit finite-width, polarity-aware rock-uplift/loading patches with conserved
   integral work, never direct terrain height; and
6. activate, migrate and abandon patches over a small number of meaningful
   tectonic epochs, after which existing routing and erosion respond.

At 4 km, useful starting *priors*, not constants, are inheritance correlation
lengths of roughly 30–150 km, segment envelopes of 50–300 km and transfer lows
at least 12–30 km wide. Graph construction should be approximately linear in
cells plus active boundary faces and negligible beside repeated global routing.

### Compiler and response results

The compiler-only `B/F/I` comparison is complete at 4 km. `B` is the accepted
flat-interior ribbon; `F` spends the same two 50/50 work ledgers through a full
cosine over each finite parent; `I` compiles finite children from an independent
96-km coherent weakness/fabric lattice. The compiler API cannot receive
elevation, drainage, runoff or landform objects. All fields close the same
`100,625 km³` rock-volume budget near roundoff, including the two original
`50,312.5 km³` parent ledgers. They compile in about `0.020 s` in release.

The cheap `F` control is the important positive result: it removes the exact
roof and emits two coherent range-scale massifs with a transfer low. `I`
concentrates the same budget into four separated hot spots with displacement
maxima around `7.59 km`, compared with `5.70 km` for `F` and `3.66 km` for `B`.
Its causes are
auditable, but its disconnected bead grammar fails the preregistered test that
inheritance earn complexity beyond one honest finite-slip prior. Do not tune or
promote this manufactured `I` generator. Evidence is in
`artifacts/captures/orogen-organization-bfi-parent-ledger-v1-4km-2026-07-17/`.

The one-shot `F -> H/C` response is also complete. Its independently compiled F
field agrees with the compiler-only F probe to `3.11e-15 km` maximum error and
again closes the `100,625 km³` budget near roundoff. Finite tips replace the
infinite roof with two compact elongated massifs, but do not by themselves emit
a differentiated range grammar. C removes `25,345 km³`, lengthens the largest
portal trunk from H's `426 km` to `530 km`, and creates substantially more
whole-domain surface critical points. Because the all-positive domain makes the
critical-point population nonlocal, that count is not evidence of F-massif
organization. The common physical view remains a pair of smooth massifs without
clear valley/divide/pass organization beyond H. C's
`746.10 s` runtime versus H's `336.43 s` does not earn that response.

H continually restores its target and therefore records `130,385 km³` of gross
restoration, while C receives the declared work gradually and preserves process
losses. The comparison discriminates mechanisms; it does not rank two
work-identical product models. Its architectural result is still useful: the
static ribbon is not a product forcing owner, and more downstream coupling is
not the first repair. When mountain work resumes, derive an auditable reduced
forcing organizer from real front chains, carrier history and inherited
material state. Keep H as a null/control and C as reusable research machinery,
not selected product behavior. Evidence is in
`artifacts/captures/orogen-f-hc-response-4km-2026-07-17/`.
That sidecar records revision `8d8ae5e` with `source_dirty: true` and is not a
promotion-grade reproducibility anchor; the committed runner and tests retain
the experiment definition, while its one-shot measurements remain bounded
decision evidence.

Drainage memory, nonlinear hillslopes, analytical or implicit stream-power
acceleration and one mobile sediment quantity remain possible later rungs.
Flexure is broad load response and cannot create the missing segmentation;
slab tears are exceptional events, not a universal pass generator. None belongs
in the first causal test.

## Precise defect

Reality and higher-fidelity models do not forbid plateaus. Broad high surfaces
are legitimate under distributed shortening, aridity, internal drainage,
resistant material or surface processes unable to keep pace. Coupled
geodynamic–landscape experiments themselves produce different plateau and belt
regimes as erodibility changes
([Theodoratos et al. 2024](https://www.nature.com/articles/s41467-024-54690-4)).

Hex3's defect is that an approximately smooth, capped distance response is the
generic grammar of convergence. Added noise, crest sharpening and scalar
along-strike modulation perturb that grammar without replacing it. The target
is therefore not “fewer plateaus”; it is several causally distinct mountain
families with internal range, divide, pass, valley and basin organization.

## Full causal reference

The reference system is deliberately more complete than the likely product
model:

```text
plate-motion and collision history
  + inherited crustal weakness, rheology and material provinces
  -> linked and migrating deformation episodes
  -> rock uplift, horizontal advection and crustal loading
  <-> drainage growth, divide migration and capture
  <-> channel incision, sediment transport and valley base level
  <-> threshold hillslope failure and sediment delivery
  -> surface elevation and relief
  -> regional erosion/deposition loads and optional isostatic response

climate/runoff modulates the surface-process loop
glaciers replace parts of it only in an appropriate climatic regime
```

This graph is a design reference, not a promise to simulate every node. A
reduction is authentic when it preserves the state and dependencies responsible
for visible geography and useful downstream consequences.

## Physical priors worth preserving

### Tectonics supplies forcing and inheritance, not finished terrain

Rock uplift, surface uplift and exhumation are distinct. The useful surface
balance is approximately `surface change = rock uplift - denudation + deposition`,
with broad load response where justified. Treating tectonic work as direct DEM
height skips the competition that creates relief
([England and Molnar 1990](https://doi.org/10.1130/0091-7613(1990)018%3C1173:SUUORA%3E2.3.CO;2)).

Real deformation is segmented, directional, mobile and inherited. Faults and
fold systems grow, link, terminate and reactivate; earlier structures can
condition later deformation. Their correlated displacement fields create range
ends, saddles, transfer zones and intermontane lows. Numerical experiments show
that inherited rift geometry affects orogen continuity, polarity, river
orientation and intervening topographic lows
([Le Breton et al. 2025](https://www.nature.com/articles/s41467-025-66695-8)).

The reduced prior is not explicit earthquake or continuum mechanics. It is a
small persistent set of linked deformation episodes or patches with:

- age and active interval;
- receiving plate/material identity;
- strike, vergence and optional advection;
- tapered along-strike displacement and finite cross-strike width;
- linkage, termination and inherited-weak-zone relationships;
- an output in rock-uplift/loading units rather than final height.

### Geological heterogeneity is shared state

Weak structures and material contrasts affect both deformation localization
and erosion. Modeled strength fields can turn homogeneous dendritic drainage
into fault-parallel high-order channels with short transverse tributaries;
channels crossing strength gradients acquire persistent knickpoints
([Roy et al. 2015](https://agupubs.onlinelibrary.wiley.com/doi/10.1002/2014JF003281)).

The useful cheap representation is a small number of coherent geological
provinces and weak structures. The same retained state may condition deformation,
channel erodibility, hillslope failure threshold and later soils/resources.
Independent noise sampled separately for height and erodibility is texture, not
geological inheritance.

### Drainage partitions the belt while it grows

Channels are not decoration applied to a completed mountain. Incision lowers
the base level for hillslopes; capture reallocates drainage area and erosive
power; divides migrate in response to uplift gradients, advection, erodibility
and climate. River profiles can adjust faster than divides, leaving meaningful
transient states
([Whipple et al. 2017](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2016JF003973)).
Mountain asymmetry and divide position also respond systematically to uplift
gradients and horizontal advection
([He et al. 2021](https://www.nature.com/articles/s41467-020-20748-2)).

Therefore uplift and drainage must alternate or be solved together long enough
for trunk incision, headward growth, divide migration and occasional capture to
change authoritative terrain. A frozen drainage scaffold can test the value of
topology, but it is not the general physical prior.

### Hillslopes fail near material-dependent thresholds

Linear diffusion is a useful low-relief approximation but a poor universal
mountain law. Landslide occurrence rises nonlinearly at steep slopes, so stronger
forcing often increases failure frequency and sediment delivery more than slope
angle ([Larsen and Montgomery 2012](https://www.nature.com/articles/ngeo1479)).

A Pareto surface model should use nonlinear slope-dependent transport or a
bounded episodic-failure shortcut, ideally with a material-dependent threshold.
This can retain sharp divides while limiting pillars without globally rounding
every convexity.

### Sediment creates transience and two-way coupling

Landslide sediment can choke or armor channels when supply exceeds transport
capacity, or promote incision as abrasive tools. Detachment-limited and
transport-limited laws may resemble one another near steady state but diverge
during tectonic and climatic transients. HyLands demonstrates a reduced
landslide–river sediment coupling
([Campforts et al. 2020](https://gmd.copernicus.org/articles/13/3863/2020/));
eSCAPE demonstrates a scalable landscape model combining stream power,
hillslopes and source-to-sink transport under varying tectonic and climatic
forcing ([Salles 2019](https://gmd.copernicus.org/articles/12/4165/2019/)).

Full stratigraphy and grain classes are unnecessary for the first architecture.
A single mobile sediment/cover quantity with finite transport and deposition may
capture much of the missing transient behavior and has unusually broad future
value for basins, plains, coasts and soils.

### Climate is a conditioner, not a generic mountain fix

Runoff and orographic precipitation can create flank asymmetry and alter divide
position, but precipitation, erosion and relief need not vary monotonically
because channel geometry, sediment and tectonic transport mediate the response
([Burbank et al. 2003](https://www.nature.com/articles/nature02187)). Hex3's
cheap climate and hydrology are therefore valuable inputs. They should modulate
the coupled surface system rather than directly paint a mountain style.

Glaciation is a later regime-specific owner. It can focus erosion near an
equilibrium-line altitude and alter valley/divide form, but it should not be
used to repair generic non-glacial ranges
([Thomson et al. 2010](https://www.nature.com/articles/nature09365)).

### Isostasy is broad load response

Erosion and deposition change loads, but cell-local rebound is the wrong prior
and risks recreating the historical pillar failure. If retained, response should
be broad, conservative and driven by integrated basin-scale mass redistribution.

## What other terrain systems preserve

Graphics and production systems reach the same conclusion from a different
direction: convincing hierarchy is normally owned by a sparse structural
system, with noise used below that scale.

| Approach | Fidelity and benefit | Principal limitation | Hex3 use |
|---|---|---|---|
| Ridged noise/domain warping | Very cheap visual variation | No basin, divide, pass or counterfactual guarantees | Presentation/fine texture only after structure exists |
| Geological/province masks plus multiscale erosion | Authentic production hack; coherent regional families | Authored masks can lack causal history | Candidate retained material/province prior |
| Explicit ridge/valley curves | Cheap, controllable semantic skeleton | Can declare the desired answer | Upper-bound/control and possible reconstruction layer |
| Drainage-first synthesis | Native rivers, watersheds and interfluves | Tectonic relationship is imposed rather than emergent | Strong topology control arm |
| Uplift plus stream-power graph | Reduced physical loop with shared rivers, ridges and watersheds | Prescribed uplift and simplified hillslopes | Strongest near-term architecture precedent |
| Coupled crust/strata plus erosion | Coherent folds, fronts, materials and differential response | More state and compute; surface evolution still required | Research reference; reduce to episodes/provinces |
| DEM/example synthesis | Imports realistic finite-scale morphology | Weak causal and counterfactual meaning | Visual upper bound, not world-model owner |

Cordonnier et al. combine an uplift field, stream-power evolution on a stream
graph and terrain reconstruction, producing shared rivers, watersheds and ridges
at relatively low cost
([2016 paper](https://www.cs.purdue.edu/cgvlab/www/resources/papers/Cordonnier-Computer_Graphics_Forum-2016-Large_Scale_Terrain_Generation_from_Tectonic_Uplift_and_Fluvial_.pdf)).
Génevaux et al. instead make the hierarchical drainage network authoritative and
construct compatible valleys and interfluves
([2013 paper](https://www.cs.purdue.edu/cgvlab/www/resources/papers/Genevaux-ACM_Trans_Graph-2013-Terrain_Generation_Using_Procedural_Models_Based_on_Hydrology.pdf)).
Explicit feature-curve and joint ridge/river methods show the cheap structural
upper bound, but largely obtain correctness by construction
([Hnaidi et al. 2010](https://onlinelibrary.wiley.com/doi/10.1111/j.1467-8659.2010.01806.x),
[Belhadj and Audibert 2005](https://doi.org/10.1145/1101389.1101479)).

Production tools are openly hybrid. Houdini begins with coarse massing and
structured seeds, then upsamples and reapplies scale-specific erosion; it also
retains flow, debris and sediment products for later use
([terrain creation](https://www.sidefx.com/docs/houdini/heightfields/creation.html),
[erosion](https://www.sidefx.com/docs/houdini/heightfields/erosion.html)). This
is not a physical reference, but it is evidence against asking one universal
height field or erosion pass to own every landform family.

## Current Hex3 ownership mismatch

The implemented path collapses useful causal state too early:

```text
plate/crust/relative motion
  -> boundary classification and distance-response fields
  -> scalar arc + collision thickening
  -> coarse crust thickness and elevation
  -> interpolated fine elevation
  -> stream-power incision + linear diffusion + repeated legacy rebuild
  -> hydrology integration cuts
  -> final scalar elevation
  -> threshold-derived mountain diagnostics
```

The code contains stronger priors, but the product path does not use them as
surface-organization owners:

- `src/world/features.rs` retains episode work, contact duration, physical
  uplift rate and experimental lifecycle/material state alongside product
  fields, yet legacy arcs and collisions are still fixed analytic distance
  profiles;
- `src/world/elevation.rs` combines legacy arc and collision response into one
  `tectonic_thickening` scalar before Airy conversion;
- `src/world/fine.rs` retains boundary-front geometry, polarity and along-strike
  coordinates, but all default fine structural generators are neutral and the
  authoritative base is interpolated coarse elevation;
- `src/world/erosion.rs` reinjects the same legacy tectonic thickening during
  the numerical erosion epoch. This is the calibrated “hold-and-carve” owner,
  not rock uplift on a shared geological clock;
- `src/world/hydrology.rs` ultimately mutates terrain through integration cuts,
  while pre-erosion hydrology also supplies frozen base levels to erosion;
- mountain semantics remain diagnostic thresholds rather than shared range,
  divide, crest, pass and valley objects.

The ancestry evidence in commit `60b9d84` establishes that the broad tableland
already exists in coarse elevation. Fine interpolation preserves it, erosion
dissects it, repeated uplift reinforces it, and presentation merely amplifies
it. Commit `6cae7cd` adds an equally important negative result: the erosion
solver can dissect supplied relief but did not organize a smooth uplift dome
into a convincing range by itself.

This suggests the following dispositions:

- **Retain:** spherical geometry, plate/crust identity, boundary sign/rate/polarity,
  boundary-front primitives, climate/runoff, hydrologic topology, incision and
  hillslope operator libraries, provenance, and explicit presentation separation.
- **Rework:** convert tectonic output from direct mountain height into persistent
  deformation/rock-uplift forcing; make landscape evolution the sole owner of
  fine ridges, valleys and divides; give geological provinces shared identity;
  name drainage integration as terrain repair; derive reusable landform semantics.
- **Quarantine/remove:** overlapping default-off height generators, duplicate
  mountain definitions, unconditional legacy history products, and repeated
  legacy rebuild once a real forcing owner exists.
- **Validate before retaining:** preview-flow-driven adaptive refinement, which
  may preferentially preserve morphology already predicted by the coarse model,
  and coarse-to-fine transfer of work/thickness quantities without declared
  conservative semantics.

## Candidate architecture families

### A. Current hold-and-carve baseline

Tectonics creates a broad height envelope; fine erosion repeatedly receives a
static rebuild term. This remains the control because it is implemented and
cheap. Its physical mismatch is direct uplift-to-height ownership, fixed forcing
and limited inherited material structure. O1–O3A show that additional shape
fields do not repair that mismatch.

**Disposition:** retain only as control while a replacement is evaluated.

### B. Synthetic drainage-first world

Generate a hierarchical drainage graph conditioned on coasts, broad relief and
tectonic setting, then reconstruct compatible valleys, divides and ridges. This
is cheap, legible and produces reusable semantic topology.

**Disposition:** build only as an explicit authentic-hack/control arm. Its value
is to reveal how much product quality comes from correct topology; it should not
silently become evidence for physical causality.

### C. Reduced coupled landscape evolution

Convert tectonic state into a time-varying rock-uplift/advection field. Coevolve
drainage, stream-power incision and nonlinear hillslope transport. Derive ridges,
passes and basins from the resulting watershed topology.

**Disposition:** preferred minimum physical architecture. It preserves the
important feedback without continuum geodynamics.

### D. Coupled landscape evolution with inherited provinces

Add a small persistent geology graph/field shared by deformation localization,
erodibility and slope threshold. Add one mobile sediment/cover quantity only if
the simpler coupled model lacks persistent transient behavior.

**Disposition:** likely target architecture, reached incrementally from C rather
than built as a complete geology simulator.

### E. Explicit joint ridge/valley skeleton

Construct dual ridge and drainage graphs conditioned on tectonic episodes, then
reconstruct a surface between them.

**Disposition:** upper-bound and fallback authentic hack. Promote only if the
coupled model cannot provide usable topology within budget and if the authored
graph preserves honest causes and downstream semantics.

## Deliberate omissions

The initial coupled architecture need not include:

- mantle convection or predictive plate forces;
- continuum thermo-mechanical rheology;
- earthquake cycles or explicit fault-plane meshes;
- full 3-D stratigraphy and grain-size distributions;
- detailed flexural solving;
- two-way surface-process feedback into deep tectonics;
- glaciers outside appropriate climate regimes.

It should preserve their relevant signatures where needed: finite linked
segments, changing forcing, inherited provinces, nonlinear slope limits,
drainage reorganization and finite sediment export.

## Discriminating program before product implementation

Do not begin with a global seed sweep. First construct an idealized bounded
orogen testbed with the same graph operators intended for the sphere:

1. one uniform uplift block to establish the scientific null;
2. two linked tapered deformation segments with a termination/transfer zone;
3. the same forcing crossing one inherited weak province;
4. a forcing reorganization that migrates or deactivates a segment;
5. wet/dry climate controls without changing tectonics.

Compare representation families, not amplitudes:

- current height-envelope hold-and-carve;
- synthetic drainage-first graph;
- coupled uplift/drainage/hillslope model;
- coupled model plus shared province state;
- explicit ridge/valley skeleton as an upper bound.

The testbed must expose state through time. Success is not one attractive final
DEM. It requires:

- linked massifs, range ends, saddles and passes corresponding to forcing;
- drainage basins that grow, compete and occasionally capture;
- longitudinal/transverse channel relationships appropriate to structure;
- nonlinear slope limitation without global cap rounding;
- persistence and decay after forcing changes;
- coherent river/divide/ridge graphs reusable by climate, cartography, ecology
  and later settlement;
- conserved or explicitly bounded uplift, erosion and sediment/load ledgers;
- resolution and timestep response adequate for the claimed reduced model;
- a visible advantage over the synthetic-topology control commensurate with
  added compute.

Only after one representation passes these causal cases should it be mapped to
the ten-seed planet corpus and tuned for useful regimes.

## Immediate recommendation

Do not implement historical A4 as the next product candidate. Preserve it as a
drainage-first control concept. First specify the organization-owner testbed and
the minimum state contract for deformation episodes, rock uplift, geological
provinces, evolving drainage, threshold hillslopes and optional mobile sediment.

The likely product direction is C evolving toward D: a reduced coupled
landscape model, conditioned by Hex3's plate history and cheap climate, with
noise restricted to unresolved texture and presentation kept separate. This is
a hypothesis to test, not a mandate to retain the current stages or terrain
representation.
