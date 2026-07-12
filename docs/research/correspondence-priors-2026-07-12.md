# Correspondence priors for reduced planet models — 2026-07-12

Status: synthesized research note. It informs the active evaluation but does not
promote mechanisms or thresholds.

## Central prior

An authentic reduced model preserves important causal relationships, invariants
and scales of the system it names. It need not predict Earth in detail. A more
elaborate model is not more authentic when its added state has no readable
effect on geography or downstream systems.

Evaluate correspondence in ascending levels:

1. **Kinematic/structural:** topology, sign, locality and ordering;
2. **Budgetary:** conservation of declared matter, water or flux;
3. **Scaling:** response to length, time and forcing;
4. **Statistical:** conditional distributions and spatial spectra;
5. **Predictive:** calibrated magnitudes, uncertainty and out-of-sample behavior.

Hex3 can plausibly make level 1–3 claims for selected subsystems. Attractive
Earth-like marginal distributions do not establish simulation-grade behavior.

Prefer conditional priors over global resemblance:

- elevation and belt width given convergence and duration;
- precipitation given wind, terrain and water supply;
- channel slope/profile given catchment, uplift and erodibility;
- lake state given basin geometry and water balance;
- semantic feature identity given remeshing and stage changes.

## Scientific correspondence

### Plate kinematics and deformation

Rigid Euler rotations are a valid kinematic abstraction for plate interiors,
but they are not geodynamics. GPlates distinguishes rigid domains from explicit
deforming networks and tracks strain/thickness only where deformation is
declared ([Müller et al. 2018](https://doi.org/10.1029/2018GC007584),
[Gurnis et al. 2018](https://par.nsf.gov/servlets/purl/10129487)).

Useful priors are interior rigidity, boundary-localized relative motion,
material identity, finite deforming width and topological reorganization. Final
mountain appearance is not the primary test of plate kinematics. A history
solver earns its cost only if reorganized motion changes retained geology,
range structure, basins or drainage—not only internal ledgers.

### Tectonic terrain

Convergence predicts tectonic setting and polarity more directly than height or
morphology. Deformable reconstructions use accumulated strain, thinning and
crustal thickness as state, while plateau scale reflects distributed shortening
over time ([GPlates deformation model](https://www.gplates.org/docs/user-manual/crustaldeformation/),
[King et al. 2022](https://doi.org/10.1029/2022GC010372)).

Hex3 correspondence should therefore test signed convergence × duration,
finite belt width, crust/material budget, subduction polarity, ocean age/depth
and response to reorganization. Boundary kernels that merely place elevation
belong to the authentic-hack class even if their setting is physically grounded.

### Climate and moisture

A steady transport model can be a mechanism-informed climatology proxy when
wind transports moisture and declared evaporation/rainout terms close a budget.
It is not a climate simulation without energy balance, vertical structure,
radiation, seasonal dynamics and ocean heat transport. Even intermediate
complexity models such as ExoPlaSim solve 3-D circulation while simplifying
physics ([Paradise et al. 2022](https://doi.org/10.1093/mnras/stab2585)).

The smallest correspondence battery is aquaplanet → flat mixed land/ocean → one
mountain barrier, with moisture closure, symmetry, wind reversal and
topography-removal tests. Rain shadows should change sides when wind reverses.

### Hydrology

Terrain-derived routing and depression hierarchy are physically grounded
topology. Flow accumulation is catchment supply, not discharge, unless runoff
and storage are represented. Reduced hydrological networks join runoff,
precipitation, evaporation, rivers, lakes and wetlands through water balance
([Coe 2000](https://doi.org/10.1175/1520-0442(2000)013%3C0686:MTHSAT%3E2.0.CO;2)).

Correspondence tests include graph conservation, outlet completeness, stable
catchments under remesh, spill-level consistency, endorheic response and the
difference between uniform-area and precipitation-weighted networks.

### Landscape evolution

Stream-power incision plus diffusive hillslopes is a standard reduced
landscape-evolution model. eSCAPE explicitly presents such laws as simplified
mechanisms reproducing first-order complexity, commonly under prescribed uplift
and rainfall ([Salles 2019](https://doi.org/10.5194/gmd-12-4165-2019)). The key
correspondence is `elevation change = uplift − incision + transport`, not an
Earth-like slope histogram.

Hex3 should isolate uplift, incision and diffusion; test timestep/resolution
behavior; condition river-profile tests on forcing; and ask whether drainage
reorganizes. Existing coupling must be evaluated before assuming another
erosion mechanism is missing.

Sediment is a consequential omission. Models such as SPACE preserve bedrock,
sediment transport and alluvial thickness together, while landslide sediment
can inhibit incision and alter channel behavior
([Shobe et al. 2017](https://doi.org/10.5194/gmd-10-4577-2017),
[Campforts et al. 2020](https://doi.org/10.5194/gmd-13-3863-2020)). Without this,
Hex3 erosion should be described as bedrock-style morphological evolution.

### Remapping and scale coupling

Coupling must declare each exchanged quantity as intensive state, extensive
content or flux. Conservative remapping protects integrals but may still damage
support, extrema and monotonicity ([Taylor 2024](https://doi.org/10.5194/gmd-17-415-2024),
[Mahadevan et al. 2020](https://doi.org/10.5194/gmd-13-2355-2020)). Synthetic
constant, impulse, coastline and smooth-wave fields should be remapped
coarse→fine→coarse and scored separately for integral error, overshoot, support
growth and spectral loss.

## Authentic game/world-generation hacks

The graphics literature offers a useful test: preserve a causal skeleton even
when the constitutive model is cheap.

- Cortial et al.'s procedural tectonic planets retain moving identity,
  boundary-relative events and geological history while using graphics-oriented
  reduced behavior ([Cortial et al. 2019](https://perso.liris.cnrs.fr/eric.galin/Articles/2019-planets.pdf)).
- Coupled tectonic uplift and fluvial erosion produces drainage-shaped mountain
  systems rather than boundary height masks
  ([Cordonnier et al. 2016](https://www.cs.purdue.edu/cgvlab/www/resources/papers/Cordonnier-Computer_Graphics_Forum-2016-Large_Scale_Terrain_Generation_from_Tectonic_Uplift_and_Fluvial_.pdf)).
- Hydrology-first terrain modeling builds legible valley geometry around an
  authoritative drainage hierarchy, a strong topology-preserving hack
  ([Génevaux et al. 2013](https://cgvlab.github.io/cgvlab/www/publications/Genevaux13ToG/)).
- Dwarf Fortress gains value from staged downstream dependence and history,
  including pragmatic smoothing/rejection and river carving; its authenticity
  is systemic rather than predictive
  ([world-generation process](https://dwarffortresswiki.org/index.php/World_generation#The_generation_process)).
- Red Blob's polygonal maps demonstrate how topology and semantic hierarchy can
  be more legible than physical provenance, making them a useful cheap baseline
  rather than a physical reference
  ([polygon map generation](https://www.redblobgames.com/maps/mapgen2/),
  [Mapgen4](https://www.redblobgames.com/maps/mapgen4/)).

High-prior authentic hacks for Hex3 preserve topology and rank: semantic river
taper, valley geometry around authoritative reaches, scale-dependent feature
generalization and named causal provenance. Distribution forcing is legitimate
as a declared product prior, not evidence that a physical model is correct.

## Rendering and cartographic correspondence

Cartography is controlled semantic abstraction, not a rival physical model.
Maintain one authoritative surface and disclose every scale-specific transform.

- Vertical exaggeration preserves ordering/location but changes slope,
  prominence and silhouette. USGS products use task-specific factors and report
  blocky artifacts at excessive values
  ([Idaho example](https://pubs.usgs.gov/of/2003/of03-471/stanford2/index.html),
  [IFSAR example](https://pubs.usgs.gov/of/2004/1451/garrity/)).
- Multidirectional hillshade reduces orientation blindness without changing
  terrain geometry ([USGS multidirectional relief](https://pubs.usgs.gov/of/1992/of92-422/)).
- Multiscale terrain generalization should preserve ridge/valley skeleton rather
  than merely blur elevation
  ([Jenny et al.](https://research.monash.edu/en/publications/terrain-generalization-with-multi-scale-pyramids-constrained-by-c/)).
- River generalization should preserve network character and hierarchy, not
  select independent lines by one global threshold
  ([Zhang and Guilbert 2016](https://doi.org/10.3390/ijgi5120230),
  [Natural Earth river ranks](https://www.naturalearthdata.com/downloads/50m-physical-vectors/50m-rivers-lake-centerlines/)).

The immediate prior is not “add PBR.” It is: add transform provenance,
physical-radius descriptors, matched physical/cartographic views and stable
landform/network semantics. Only then can visual evidence distinguish missing
morphology from poor communication.

## Pareto-ranked missing or incomplete systems

| Candidate | Why it matters | Prior posture |
|---|---|---|
| Landform semantics: ranges, plateaus, ridges, valleys, divides, peaks/passes | Shared bridge for metrics, causal tests, rendering, labels, ecology and later civilization | Highest leverage; evaluate/extract next |
| Presentation transform ledger and matched physical/cartographic views | Prevents false physical inference and makes visual evaluation reproducible | Very low cost, immediate |
| Controlled correspondence tests for current uplift–erosion coupling | Determines whether an expensive existing system earns its claim before replacement | Immediate evaluation, not new feature |
| Sediment/source-to-sink ledger and minimal cover/deposition state | Major missing geomorphic cause; enables plains, deltas, fans and plateau-edge behavior | High potential, research/design gate before implementation |
| Time-varying tectonic motion and readable geological inheritance | Removes permanent forcing; worthwhile only if visible/semantic consequences survive | Conditional on lifecycle ablation evidence |
| Hydrologic storage/water budget, wetlands and runoff semantics | Separates catchment area from discharge and supports ecology | Targeted addition after frozen-terrain tests |
| Coast/island hierarchy and scale generalization | High board/globe legibility and future region leverage | Moderate cost, presentation/semantic work |
| Ocean heat transport, seasonality and soils | Scientifically important for ecology | High cost/prerequisite burden; do not add until climate correspondence tests show need |
| Vegetation/civilization | High spectacle and downstream value | Paused until environmental and geographic semantics are trustworthy |

This ranking is intentionally not a completeness checklist. Mantle convection,
full GCMs, groundwater and individual-organism simulation remain poor near-term
tradeoffs unless evaluation exposes a dependency that cheaper mechanisms cannot
satisfy.

