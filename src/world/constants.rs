//! Constants for world generation and simulation.

// Crust generation (continents are independent of plates)
//
// These are first-class worldbuilding knobs: one craton at high coverage is a
// Pangaea world, many small cratons at low coverage is an archipelago world.

/// Target fraction of surface area that should be continental crust.
pub const CONTINENTAL_FRACTION: f32 = 0.30;

/// Number of cratons (continental nuclei) to grow.
pub const NUM_CRATONS: usize = 5;

/// Log-normal sigma for craton target sizes. Higher = more varied continent sizes.
pub const CRATON_SIZE_SIGMA: f32 = 0.6;

/// Max ratio between largest and smallest craton target size.
pub const CRATON_SIZE_MAX_RATIO: f32 = 6.0;

/// Minimum spacing between craton seeds, as a fraction of the ideal spacing
/// for evenly distributed seeds.
pub const CRATON_SEED_SPACING_FRACTION: f32 = 0.6;

/// fBm frequency for craton edge noise (coastline irregularity).
pub const CRATON_NOISE_FREQUENCY: f64 = 2.5;

/// fBm octaves for craton edge noise.
pub const CRATON_NOISE_OCTAVES: usize = 4;

/// Weight of noise in craton growth priority. Higher = more ragged coastlines.
pub const CRATON_NOISE_WEIGHT: f32 = 1.2;

/// Scale factor for boundary forcing used by tectonic features (arcs, trenches, ridges).
pub const FEATURE_FORCE_SCALE: f32 = 35.0;

/// Reference boundary spacing (mean neighbor arc distance, radians) at the design
/// resolution (~100k coarse cells, measured 0.012558). The magnitude-feature seeds
/// (trench/forearc/arc/ridge/collision/rift) are an edge-length-weighted MEAN rate
/// times this FIXED length — an intensive quantity, so per-cell forcing (and thus
/// feature heights) is independent of cell count. Multiplying by this constant
/// keeps the absolute forcing scale matched to the historical area-scaled-sum
/// formulation at the design resolution, so the existing `*_SENSITIVITY` / `*_MAX`
/// constants need no re-tuning. (Do NOT replace with the live `mean_neighbor_dist`
/// — that would reintroduce the 1/sqrt(N) resolution dependence this removes.)
pub const FEATURE_FORCE_REF_SPACING: f32 = 0.012558;

// Screened diffusion solver parameters

/// Damping factor for Gauss-Seidel iteration (0 < ω ≤ 1). Lower = more stable.
pub const DIFFUSION_DAMPING: f32 = 0.8;

/// Maximum iterations for diffusion solver.
pub const DIFFUSION_MAX_ITERS: usize = 50;

/// Convergence tolerance for early termination.
pub const DIFFUSION_TOLERANCE: f32 = 0.001;

// Isostasy: elevation derives from crust thickness via the Airy relation
// (columns float on the mantle; with uniform densities the relation is
// linear). The slope/offset are derived from two anchor points:
// (CRUST_THICKNESS_CONTINENTAL -> CONTINENTAL_BASE) and
// (CRUST_THICKNESS_OCEANIC -> ABYSSAL_DEPTH).
//
// CANONICAL VERTICAL SCALE: elevation is a normalized unit-sphere field (sea
// level = 0); 1 elevation unit ≈ 10 km. This is the single reference for every
// "≈ N m" annotation below (it is the scale the ocean anchors and the
// land↔ocean isostasy gap imply). The world is physically inspired, not
// Earth-accurate, so this is for intuition/tuning only — nothing in the code
// reads a vertical metres conversion; horizontal distances use PLANET_RADIUS_KM.

/// Reference thickness of undisturbed continental crust (definition scale).
pub const CRUST_THICKNESS_CONTINENTAL: f32 = 1.0;

/// Reference thickness of oceanic crust.
pub const CRUST_THICKNESS_OCEANIC: f32 = 0.25;

/// Elevation of reference continental crust (~800m above sea level, at 10 km/unit).
/// Isostasy anchor point.
pub const CONTINENTAL_BASE: f32 = 0.08;

/// Macro-scale crust thickness variation amplitude (thickness units).
/// Replaces additive macro elevation noise: thick cratonic cores and thin
/// interior basins, automatically isostatically compensated.
pub const MACRO_THICKNESS_AMPLITUDE: f32 = 0.13;

// Continental rifting. Thinning derives from the actual opening rate at
// the boundary (per-edge kinematics, so along-strike variation comes from
// Euler-pole geometry), localized into a narrow valley by lithospheric
// necking, with flexural/unloading shoulder uplift on the flanks.

/// Sensitivity of crustal thinning to rift opening forcing (sqrt response).
pub const RIFT_SENSITIVITY: f32 = 0.10;

/// Maximum axial crustal thinning (thickness units). Strong rifts subside
/// below sea level (future oceans / Red Sea stage).
pub const RIFT_MAX_THINNING: f32 = 0.35;

/// Half-width of the rift valley thinning core (radians). Strain localizes
/// by lithospheric necking; real rift valleys are 50-80 km wide. 0.018 rad
/// ~ 115 km, near the resolution floor at 100k cells.
pub const RIFT_VALLEY_WIDTH: f32 = 0.018;

/// Distance from the rift axis to the crest of the uplifted shoulders
/// (radians). ~0.035 rad ~ 220 km.
pub const RIFT_SHOULDER_OFFSET: f32 = 0.035;

/// Width of the shoulder uplift band (radians).
pub const RIFT_SHOULDER_WIDTH: f32 = 0.025;

/// Shoulder thickening as a fraction of axial thinning magnitude.
/// Real rift shoulders rise a substantial fraction of the graben depth
/// (flexural unloading + thermal support).
pub const RIFT_SHOULDER_RATIO: f32 = 0.4;

/// Target fraction of surface area above sea level. Sea level is solved
/// (uniform elevation shift) so the coastline lands here exactly,
/// independent of seed. Distinct from CONTINENTAL_FRACTION (crust area):
/// the difference is submerged shelf.
pub const LAND_FRACTION: f32 = 0.26;

/// Depth at mid-ocean ridge crests (young, hot oceanic crust).
/// Represents ~2500m below sea level.
pub const RIDGE_CREST_DEPTH: f32 = -0.25;

/// Depth of old oceanic crust far from ridges (abyssal plain).
/// Represents ~4500m below sea level (thermally subsided).
pub const ABYSSAL_DEPTH: f32 = -0.45;

/// Ocean depth for crust on plates with no ridge (no age information).
/// Between ridge crest and abyssal so these basins keep noise-driven variety
/// instead of sitting uniformly at maximum depth.
pub const NO_RIDGE_DEPTH: f32 = -0.38;

/// Characteristic distance for oceanic thermal subsidence (radians).
/// 1.5 rad ≈ 9550 km on Earth. Ocean floor deepens from ridge crest
/// to abyssal depth over roughly this distance (sqrt decay).
pub const THERMAL_SUBSIDENCE_WIDTH: f32 = 1.5;

/// Width of continental shelf transition on passive margins (radians).
/// 0.05 rad ≈ 320 km on Earth (Atlantic-style wide shelf). Controls how far
/// inland the crust thickness ramp extends before reaching full continental
/// thickness.
pub const PASSIVE_SHELF_WIDTH: f32 = 0.05;

/// Width of continental shelf transition on active margins (radians).
/// 0.015 rad ≈ 95 km on Earth (Andes-style narrow shelf). Used where the
/// margin sits near a convergent plate boundary.
pub const ACTIVE_SHELF_WIDTH: f32 = 0.015;

/// Width of oceanic transition from margin to abyssal plain on passive
/// margins (radians). 0.08 rad ≈ 510 km on Earth. Represents continental
/// slope (~100 km) plus continental rise (~400 km).
pub const PASSIVE_OCEANIC_TRANSITION_WIDTH: f32 = 0.08;

/// Width of oceanic transition on active margins (radians).
/// 0.03 rad ≈ 190 km. Steep descent where a trench sits offshore.
pub const ACTIVE_OCEANIC_TRANSITION_WIDTH: f32 = 0.03;

/// Angular velocity range for random Euler poles.
pub const MAX_ANGULAR_VELOCITY: f32 = 1.0;

// Plate interaction multipliers (convergent)
//
// These scale *uplift forcing* used for uplift-style convergent features
// (continental collision, overriding-side cordilleras / arcs).
//
// Trench forcing uses separate subduction multipliers so that trench depth is
// not accidentally suppressed on the subducting oceanic side.

/// Cont+Cont: Himalayas-scale, highest mountains.
pub const CONV_CONT_CONT: f32 = 1.5;

/// Ocean+Ocean: Volcanic island arc.
/// Higher than before to enable island formation above deep ocean floor.
pub const CONV_OCEAN_OCEAN: f32 = 0.8;

/// Cont side of Cont+Ocean: Andes-style coastal mountains.
pub const CONV_CONT_OCEAN: f32 = 1.2;

/// Ocean side of Cont+Ocean: minimal - subducting plate goes down.
pub const CONV_OCEAN_CONT: f32 = 0.1;

// Subduction / trench forcing multipliers (convergent)
//
// These scale the *subducting-side* forcing that drives trench depth.
// Trench depth is further modulated by slab age (see `TRENCH_AGE_*`).
pub const SUBD_OCEAN_CONT: f32 = 1.0;
pub const SUBD_OCEAN_OCEAN: f32 = 1.0;

// Plate interaction multipliers (divergent)

/// Cont+Cont: East African Rift, Red Sea.
pub const DIV_CONT_CONT: f32 = 0.6;

/// Ocean+Ocean: Mid-Atlantic Ridge.
pub const DIV_OCEAN_OCEAN: f32 = 0.5;

/// Cont side of Cont+Ocean: modest rifting at passive margin.
pub const DIV_CONT_OCEAN: f32 = 0.1;

// Plate generation tuning

/// Fraction of ideal seed spacing to use as minimum distance.
pub const SEED_SPACING_FRACTION: f32 = 0.5;

/// Log-normal spread for plate target sizes.
pub const TARGET_SIZE_SIGMA: f32 = 0.4;

/// Maximum ratio between largest and smallest plate target size.
pub const TARGET_SIZE_MAX_RATIO: f32 = 4.0;

// Relief rendering

/// Scale factor for elevation displacement in relief view.
pub const RELIEF_SCALE: f32 = 0.2;

/// Weight of noise vs distance in cell priority.
pub const NOISE_WEIGHT: f32 = 1.0;

/// Bonus per same-plate neighbor when claiming a cell.
pub const NEIGHBOR_BONUS: f32 = 0.1;

/// Base frequency for fBm noise on sphere.
pub const NOISE_FREQUENCY: f64 = 2.0;

/// Number of octaves for fBm noise.
pub const NOISE_OCTAVES: usize = 4;

// Elevation noise (post erosion-v2). Only the macro layer remains in the
// simulation: hills and ridge were appearance-paint that erosion now supersedes
// (docs/specs/erosion-v2.md "Noise philosophy") and were retired; cosmetic micro
// texture was removed entirely. Macro is a crust-thickness perturbation
// (isostatically compensated — the model of "the crust isn't uniform"); it does
// not paint terrain relief.

// --- Macro layer (craton-structure crust-thickness perturbation) ---
// Replaces free position-fBm with a structure-derived field: each continental
// craton is a thick interior that tapers to its margins (an isostatic dome →
// elevated shield), with a per-craton base amplitude (older/thicker shields vs
// thinner cratons) and a decorrelated interior fBm for intracratonic basins and
// swells. Oceanic cells get no perturbation (their relief is thermal + ridge).
// Overall scale is MACRO_THICKNESS_AMPLITUDE.

/// Distance scale (radians) over which the craton thickness dome saturates from
/// margin (thin) to interior (full). 0.15 rad ≈ 955 km: interiors of large
/// continents reach near-full thickness; small cratons stay margin-dominated.
pub const MACRO_CRATON_DECAY: f32 = 0.15;
/// Per-craton base thickness amplitude range (multiplies the dome). The spread
/// gives thicker shields vs thinner cratons; both positive, so interiors thicken.
pub const MACRO_CRATON_AMP_MIN: f32 = 0.4;
pub const MACRO_CRATON_AMP_MAX: f32 = 1.0;
/// Frequency of the intracratonic interior fBm (basins/swells within a craton).
pub const MACRO_INTERIOR_FREQUENCY: f64 = 1.5;
/// Octaves for the interior fBm (few = smooth).
pub const MACRO_INTERIOR_OCTAVES: usize = 2;
/// Interior fBm amplitude relative to the per-craton base; can locally invert the
/// dome into a basin. 0.5 = basins/swells up to half the base amplitude.
pub const MACRO_INTERIOR_RELIEF: f32 = 0.5;

// Climate mechanisms

/// Distance scale for land-ocean thermal contrast continentality (radians).
/// 0.10 rad ~= 637 km on Earth: shelf seas and nearby coasts stay moderated,
/// while continental interiors approach the full contrast.
pub const CONTINENTALITY_DISTANCE_SCALE: f32 = 0.10;
/// Maximum amplification of latitude-only temperature deviation in continental interiors.
pub const CONTINENTALITY_AMP: f32 = 0.35;
/// Exponential sensitivity of basin evaporation to mean catchment temperature.
/// A basin at global mean temperature keeps the global climate ratio unchanged.
pub const EVAP_TEMP_SENSITIVITY: f32 = 1.0;

// Boundary-anchored elevation features (minimal bathymetry/orogeny model)
//
// These are applied as additive terms during elevation generation, using
// distance-to-boundary fields derived from plate kinematics.

/// Trench depth sensitivity (uses sqrt response of boundary forcing).
pub const TRENCH_SENSITIVITY: f32 = 0.06;
/// Maximum trench depth (positive magnitude; applied as negative elevation).
pub const TRENCH_MAX_DEPTH: f32 = 0.18;
/// Multiplier applied to trench forcing for very young oceanic lithosphere (near ridges).
pub const TRENCH_AGE_YOUNG_MULT: f32 = 0.7;
/// Multiplier applied to trench forcing for old oceanic lithosphere (far from ridges).
pub const TRENCH_AGE_OLD_MULT: f32 = 1.3;
/// Flexural parameter alpha for the subducting plate (radians).
/// Sets the whole trench-to-outer-rise geometry: trench wall zero-crossing
/// at 2.36*alpha, outer rise crest at pi*alpha (4.3% of trench depth, up).
/// 0.018 rad ~= 115 km on Earth -> outer rise ~360 km from the axis.
pub const TRENCH_FLEX_ALPHA: f32 = 0.018;
/// Alpha multiplier for young (near-ridge) lithosphere: hot, thin, floppy.
pub const TRENCH_FLEX_ALPHA_YOUNG_MULT: f32 = 0.6;
/// Alpha multiplier for old lithosphere: cold, thick, stiff -> wide flexure.
pub const TRENCH_FLEX_ALPHA_OLD_MULT: f32 = 1.4;
/// Flexural parameter for the overriding plate's forearc (radians).
/// Recovery is ~complete by pi/2*alpha_f ~= 0.024 rad, landward of which
/// the arc Gaussian (peak 0.04-0.05 rad) takes over.
pub const FOREARC_ALPHA: f32 = 0.015;
/// Fraction of the trench-axis depth inherited by the overriding plate edge.
pub const FOREARC_COUPLING: f32 = 0.8;
/// Reference ridge opening rate for distance-to-age conversion.
/// When local spreading rate equals this value, ridge distance maps to the
/// previous implicit-age behavior; faster ridges keep ocean floor young wider.
pub const OCEAN_SPREADING_REFERENCE_RATE: f32 = 1.0;

/// Volcanic arc / cordillera uplift sensitivity (sqrt response of boundary forcing).
///
/// Split by overriding plate type:
/// - Continental: cordillera-style uplift
/// - Oceanic: island-arc uplift, needs high sensitivity to overcome deep ocean base (-0.45)
pub const ARC_CONT_SENSITIVITY: f32 = 0.12;
pub const ARC_OCEAN_SENSITIVITY: f32 = 1.2;

/// Maximum arc uplift (cap applied after sqrt response).
pub const ARC_CONT_MAX_UPLIFT: f32 = 0.48;
/// Oceanic arc max is lower to avoid mountains in shallow water near ridges.
pub const ARC_OCEAN_MAX_UPLIFT: f32 = 0.40;

/// Peak offset of arc uplift inland from the boundary (radians).
/// Continental: 0.05 rad ≈ 319 km on Earth (large-end: 200-350+ km inland).
/// Oceanic: 0.04 rad ≈ 255 km on Earth.
pub const ARC_CONT_PEAK_DIST: f32 = 0.05;
pub const ARC_OCEAN_PEAK_DIST: f32 = 0.04;

/// Arc band width (radians).
/// Continental: 0.05 rad ≈ 319 km on Earth; oceanic: 0.04 rad ≈ 255 km.
/// Wider band = more cells in the arc belt.
/// Note: Real volcanic arcs are narrower (50-150 km), but wider values help visibility.
pub const ARC_CONT_WIDTH: f32 = 0.05;
pub const ARC_OCEAN_WIDTH: f32 = 0.04;

// Oceanic arc noise (multiplicative modulation for island clustering).
// Noise determines which parts of the arc form islands vs remain underwater.
/// Seed for arc noise.
pub const ARC_NOISE_SEED: u32 = 0xA16C_0B3D;
/// Frequency for island-scale variation (lower = larger island groups).
pub const ARC_NOISE_FREQ: f64 = 8.0;
/// Number of octaves for arc noise.
pub const ARC_NOISE_OCTAVES: usize = 3;
/// Noise threshold for island formation.
/// Arc is multiplied by smoothstep(noise, threshold - width, threshold + width).
/// Positive = fewer islands, negative = more islands.
pub const ARC_ISLAND_THRESHOLD: f32 = -0.2;
/// Transition width for island formation smoothstep.
/// Larger = smoother transitions, smaller = sharper island boundaries.
pub const ARC_ISLAND_TRANSITION: f32 = 0.5;

/// Maximum volcanic island height (soft cap using tanh).
/// Represents equilibrium between volcanic construction and erosion/subsidence.
/// Islands above this height are smoothly compressed: H * tanh(h/H).
/// 0.15 gives realistic volcanic island elevations (comparable to Hawaii's ~0.1-0.15 normalized).
pub const VOLCANIC_ISLAND_MAX_HEIGHT: f32 = 0.15;

/// Mid-ocean ridge uplift sensitivity (sqrt response of boundary forcing).
pub const RIDGE_SENSITIVITY: f32 = 0.006;
/// Maximum ridge uplift.
pub const RIDGE_MAX_UPLIFT: f32 = 0.02;
/// Ridge decay length from the boundary (radians).
/// Note: broad ridge swell is already captured by `thermal_oceanic_depth(ridge_distance)`;
/// this term is meant to add a narrower axial high on top.
/// 0.015 rad ≈ 96 km on Earth.
pub const RIDGE_DECAY: f32 = 0.015;

/// Continental collision uplift sensitivity (sqrt response).
pub const COLLISION_SENSITIVITY: f32 = 0.10;
/// Maximum collision uplift (Himalaya-scale).
pub const COLLISION_MAX_UPLIFT: f32 = 0.35;
/// Collision band width (radians).
/// 0.02 rad ≈ 127 km on Earth. Gives ~250-350 km effective mountain range width.
pub const COLLISION_WIDTH: f32 = 0.02;
/// Collision peak offset from boundary (radians).
/// 0.015 rad ≈ 96 km on Earth. Places peak near boundary, not far inland.
pub const COLLISION_PEAK_DIST: f32 = 0.015;

/// Decay length for tectonic activity field (radians).
/// Controls how far "tectonically active" influence spreads from boundaries.
pub const ACTIVITY_INFLUENCE_LENGTH: f32 = 0.05;
pub const CONVERGENT_INFLUENCE_LENGTH: f32 = 0.06;
pub const DIVERGENT_INFLUENCE_LENGTH: f32 = 0.06;

// Boundary classification thresholds

/// Normal velocity threshold for transform classification.
/// If |convergence| < this AND shear dominates, classify as transform.
pub const TRANSFORM_NORMAL_THRESHOLD: f32 = 0.02;

/// Ratio threshold: shear must exceed convergence by this factor to be transform.
pub const TRANSFORM_RATIO: f32 = 2.0;

// Plate-pair regime classification (ridge vs subduction vs transform)
//
// We classify the boundary regime per touching plate pair by aggregating kinematics over
// all edges between the two plates, instead of per edge. This prevents tiny sign-flip
// patches (from boundary geometry noise) from spawning large boundary-anchored features.

/// Minimum total boundary length (radians) required to classify a plate-pair as
/// convergent or divergent; shorter contacts are treated as transform/inactive.
///
/// 0.05 rad ≈ 318 km on Earth.
pub const PLATE_PAIR_MIN_BOUNDARY_LENGTH: f32 = 0.05;

/// Minimum boundary length (radians) that must have consistent-sign normal motion
/// (|convergence| >= TRANSFORM_NORMAL_THRESHOLD) for a plate-pair to be classified as
/// convergent or divergent.
///
/// 0.03 rad ≈ 191 km on Earth.
pub const PLATE_PAIR_MIN_ACTIVE_LENGTH: f32 = 0.03;

// =============================================================================
// Atmosphere constants (Stage 2: Wind simulation)
// =============================================================================

// --- Wind forcing ---

/// Scale factor for pressure gradient → wind velocity.
pub const PRESSURE_WIND_SCALE: f32 = 0.3;

/// How strongly Coriolis turns pressure-driven flow toward geostrophic balance.
///
/// 0.0 = purely down-gradient (toward low pressure).
/// 1.0 = purely geostrophic (parallel to isobars).
pub const GEOSTROPHIC_BALANCE: f32 = 0.85;

/// Weight of pressure-gradient component in wind blend.
pub const PRESSURE_WEIGHT: f32 = 0.4;

// --- Meridional overturning circulation ---

/// Planet rotation rate relative to Earth. Drives circulation-cell width,
/// Coriolis turning strength, and upper-level jet positions coherently.
pub const PLANET_ROTATION_RATE: f32 = 1.0;

/// Scale from prescribed streamfunction transport to surface meridional wind.
pub const CIRC_MERIDIONAL_SCALE: f32 = 0.20;

/// Linear surface friction used in u = (f / eps) * v.
pub const CIRC_FRICTION: f32 = 1.0;

/// Hadley-cell streamfunction amplitude.
pub const CIRC_HADLEY_AMPLITUDE: f32 = 1.0;

/// Ferrel-cell streamfunction amplitude.
pub const CIRC_FERREL_AMPLITUDE: f32 = 0.5;

/// Polar-cell streamfunction amplitude.
pub const CIRC_POLAR_AMPLITUDE: f32 = 0.25;

/// Scale for angular-momentum upper wind. Relative to Earth surface speed.
pub const OMEGA_SURFACE_SPEED: f32 = 0.45;

/// Visualization clamp for angular-momentum upper wind.
pub const UPPER_WIND_MAX_SPEED: f32 = 0.45;

/// Upper-level amplification of surface zonal wind in thermally indirect
/// (eddy-driven, roughly barotropic) cells — the mid-latitude jet aloft.
pub const UPPER_BAROTROPIC_FACTOR: f32 = 2.0;

// --- Uplift proxy ---

/// Weight for convergence-based uplift (mass continuity proxy).
pub const UPLIFT_CONVERGENCE_WEIGHT: f32 = 0.8;

/// Weight for orographic uplift (upslope flow).
pub const UPLIFT_OROGRAPHIC_WEIGHT: f32 = 1.2;

/// Weight for signed large-scale circulation uplift/subsidence.
pub const UPLIFT_CIRCULATION_WEIGHT: f32 = 0.7;

/// Percentile used to normalize uplift into 0..1 for visualization.
pub const UPLIFT_NORM_PERCENTILE: f32 = 0.95;

// --- Terrain effects (before projection) ---

/// How much terrain slope blocks uphill wind.
/// block_factor = min(gradient * UPHILL_BLOCKING, 1.0)
/// At 1.0: full blocking at 45° slopes (gradient=1.0), partial at gentler slopes.
pub const UPHILL_BLOCKING: f32 = 1.0;

/// Katabatic (downhill) wind acceleration strength.
/// Cold air drainage down slopes. katabatic_wind = gradient * KATABATIC_STRENGTH
/// At 0.05: steep slopes (gradient=1.0) add ~0.05 to wind magnitude,
/// comparable to but not overwhelming background winds (~0.1-0.3).
pub const KATABATIC_STRENGTH: f32 = 0.1;

// --- Projection solver ---

/// Power for cosine permeability law: perm = cos^p(atan(gradient)) = 1/(1+g²)^(p/2)
/// At p=2: gradient=0.5 → perm=0.80, gradient=1.0 (45°) → perm=0.50,
/// gradient=2.0 → perm=0.20, gradient=4.0 → perm=0.06.
pub const PERMEABILITY_POWER: f32 = 2.0;

/// Elevation at which terrain exerts full orographic uplift on wind.
/// Below this, uplift scales down linearly so coastal shelf steps do not
/// out-rain interior mountain ranges.
pub const OROGRAPHIC_FULL_HEIGHT: f32 = 0.15;

/// Base number of SOR sweeps for the wind-projection Poisson solve, applied at
/// [`PROJECTION_REFERENCE_SPACING`]. The actual count scales up on finer meshes
/// (see below).
pub const PROJECTION_ITERATIONS: usize = 50;

/// Mean-neighbour spacing (radians) at which [`PROJECTION_ITERATIONS`] sweeps
/// converge the projection well. ~the 50k-cell spacing, where 50 sweeps already
/// converge the uplift field. Finer meshes scale sweeps by (this / spacing)² —
/// the projection is an UNSCREENED Poisson solve whose low-frequency Gauss-Seidel
/// error decays ~O(1/h²) per sweep, so a fixed count under-develops large-scale
/// `phi` (hence uplift) as resolution rises. Quadratic scaling holds the
/// convergence level — and thus the field — mesh-independent.
pub const PROJECTION_REFERENCE_SPACING: f32 = 0.018;

/// Hard cap on the adaptive projection sweep count (bounds cost at very fine
/// coarse meshes; the solve is on the coarse mesh only).
pub const PROJECTION_MAX_ITERATIONS: usize = 800;

/// SOR relaxation factor (1.0-1.9, higher = faster but less stable).
pub const SOR_OMEGA: f32 = 1.0;

// Moisture & precipitation.

/// Reference advective timestep that the per-iteration reaction rates
/// (`EVAPORATION_RATE`, the `RAINOUT_*` family, `MOISTURE_DIFFUSIVITY`) are
/// calibrated against. The advection pass advances physical time `dt =
/// MOISTURE_CFL/max_outflow` per iteration, but those reaction/mixing constants
/// were tuned as per-iteration fractions (an implicit `dt = 1`). To make the
/// reaction:advection balance — and thus how far moisture penetrates inland —
/// independent of mesh resolution and wind speed (both of which move `dt`), the
/// reactions are scaled by `dt / MOISTURE_DT_REF`. At the design coarse
/// resolution (100k cells, where the live `dt ≈ 0.01`) the factor is ≈1, so the
/// generated world's climate is unchanged; at other resolutions it scales
/// correctly instead of drifting. This anchors the constants — changing it
/// rescales every reaction rate. (It also removes the old spurious dependence of
/// the climate on a world's peak wind, which set the CFL `dt`.)
pub const MOISTURE_DT_REF: f32 = 0.01;

/// Hard cap on moisture relaxation iterations. The loop runs to a steady state
/// (see `MOISTURE_CONV_TOL`) rather than a fixed count, because the reaction
/// timestep `dt/MOISTURE_DT_REF` shrinks with resolution, so the iteration count
/// to converge grows. At the 100k design resolution it converges in ~50 iters
/// and breaks early; this cap only bounds far finer meshes.
pub const MOISTURE_MAX_ITERATIONS: usize = 2000;

/// Steady-state tolerance: the relaxation stops once the max per-iteration change
/// in cell moisture falls below this fraction of the mean moisture.
pub const MOISTURE_CONV_TOL: f32 = 1.0e-4;

/// Iterations after convergence to average precipitation over.
pub const MOISTURE_AVG_WINDOW: usize = 20;

/// Fraction of a water cell's capacity deficit replenished per iteration.
pub const EVAPORATION_RATE: f32 = 0.5;

/// Moisture carrying capacity of cold air (temperature 0).
pub const MOISTURE_CAP_COLD: f32 = 0.35;

/// Moisture carrying capacity of warm air (temperature 1).
pub const MOISTURE_CAP_WARM: f32 = 1.0;

/// Global planet wetness: the target area-weighted LAND mean of the precipitation
/// field. The moisture model produces a *relative* rainfall pattern; this sets its
/// absolute level, which is what was lost when precip is renormalized. It scales
/// every absolute water consumer — fluvial discharge (so a wetter planet incises
/// deeper, drives bigger rivers) and lake catchment budgets (bigger lakes) — while
/// leaving the channel-initiation support area (a geometric threshold) invariant,
/// so drainage-network *density* is unchanged; only its vigour scales.
///
/// Distinct from the hydrology `climate_ratio` (the lake P/E balance, which only
/// moves lake levels). 1.0 = today's behaviour; >1 = wet world (lush, dissected,
/// many lakes); <1 = arid world (sparse rivers, shrunken lakes, less erosion).
pub const PRECIP_GLOBAL_SCALE: f32 = 1.0;

/// Baseline fraction of airborne moisture raining out per iteration.
/// Controls how far moisture travels inland before drying out.
pub const RAINOUT_BASE: f32 = 0.025;

/// Additional rainout per unit of uplift (orographic + convergence rain).
pub const RAINOUT_OROGRAPHIC: f32 = 0.15;

/// Convective rainout coefficient: warm, humid air rains on its own.
/// Applied as RAINOUT_CONVECTIVE * humidity^2 * temperature, so tropical
/// moist air rains (rainforests) while cold or dry air does not.
pub const RAINOUT_CONVECTIVE: f32 = 0.12;

/// Fraction of land precipitation re-evaporated into the air column
/// (evapotranspiration recycling; lets rain propagate into deep interiors).
pub const MOISTURE_RECYCLE_FRACTION: f32 = 0.5;

/// CFL number for moisture advection: the largest fraction of any cell's
/// moisture that may be exported per iteration. Controls the effective
/// timestep (dt = CFL / max outflow rate).
pub const MOISTURE_CFL: f32 = 0.8;

/// Fraction of over-capacity moisture raining out per iteration.
/// Below 1.0 so cold regions don't flash-dump all arriving moisture at once.
pub const OVERFLOW_RAINOUT: f32 = 0.3;

/// Eddy diffusivity for moisture transport, in radians^2 per iteration.
/// Models horizontal turbulent mixing. Resolution-independent: the
/// per-iteration mixing fraction is diffusivity / cell_spacing^2 (clamped
/// for stability, so very high resolutions under-diffuse slightly rather
/// than going unstable).
pub const MOISTURE_DIFFUSIVITY: f32 = 5.0e-5;

// =============================================================================
// Fine mesh refinement (Stage 3 erosion infrastructure)
// =============================================================================

/// Planet radius in km, used to report angular sizes (radians on the unit
/// sphere) as physical distances. Defaults to Earth (6371 km); the rest of the
/// codebase's "~N km on Earth" comments assume this value. A fantasy planet
/// would change this — it only scales the human-readable km readouts, not the
/// simulation, which is unit-sphere throughout.
pub const PLANET_RADIUS_KM: f32 = 6371.0;

// The fine mesh is sized by an ABSOLUTE target cell size (km) per location, not
// a relative density with a magic ratio cap. The cell count is therefore not a
// target — it EMERGES from integrating 1/size^2 over the terrain, so a
// mountainous world needs more cells than a flat one. The three cell-size scales
// below are meaningful lengths; the density ratio between them is derived, not
// chosen. FINE_MAX_CELLS is a guardrail, not a budget.

/// Finest fine-mesh cell size, in km (mountains / steepest ground / main
/// channels). Physically this is roughly the channel-initiation / hillslope
/// length below which erosion is sub-grid, so it is a real floor, not a knob to
/// chase. ~1.5 km resolves drainage-network dissection.
pub const FINE_MOUNTAIN_CELL_KM: f32 = 1.5;

/// Coarsest land cell size, in km (flat plains floor). Gentle, low-flow land
/// carries little erosion detail, so it sits here. Raised 12->20 by the 2026-06
/// density audit: plains/upland were ~72% of the cell budget at 8-16x lower
/// relief-per-cell than mountains (over-resolved). See
/// docs/density-audit-2026-06-15.md.
pub const FINE_PLAINS_CELL_KM: f32 = 20.0;

/// Ocean-floor cell size, in km. Featureless seabed needs almost nothing, and
/// erosion skips it. Raised 60->120 (density audit): a weak lever on count
/// (~9% of cells) but free — coarsen the underwater floor we don't render in
/// detail. (A near-shore band may be wanted later if coastlines look blocky.)
pub const FINE_OCEAN_CELL_KM: f32 = 120.0;

/// Hard ceiling on fine-mesh cell count (memory/perf guardrail). The count is
/// NOT a target: it emerges from the resolution criterion. If the emergent count
/// exceeds this (e.g. mountains nearly everywhere), the whole size field is
/// uniformly coarsened to fit — graceful degradation, proportions intact — and a
/// warning is logged. Raise it (with the s2-voronoi budget in mind) to allow
/// genuinely finer/larger meshes.
pub const FINE_MAX_CELLS: usize = 8_000_000;

/// Exponent applied to each normalized refinement feature (slope, flow,
/// activity). 1.0 = linear (gentle terrain refines quickly); >1.0 keeps
/// flat/gentle land near the plains size and concentrates resolution on the
/// steepest ground and main channels. 2.0 = square, 3.0 = cube. Raised 3->4 by
/// the density audit: sharpening the demand curve concentrates the budget on
/// mountains (which carry the relief and were under-allocated — achieving ~3.9km
/// vs the 1.5km knob) while gentle terrain coarsens. With the plains/ocean
/// changes this is ~-44% cells, mountain share 18.6%->26.8%, grading preserved.
pub const FINE_DENSITY_FEATURE_EXPONENT: f32 = 4.0;

/// Relative importance of coarse slope in the refinement demand (0..1 blend
/// weight; only the ratio between the three weights matters — absolute scale
/// comes from the cell-size constants above).
pub const FINE_SLOPE_DENSITY_WEIGHT: f32 = 8.0;

/// Relative importance of log-scaled coarse flow accumulation in the demand.
pub const FINE_FLOW_DENSITY_WEIGHT: f32 = 18.0;

/// Relative importance of tectonic activity/uplift forcing in the demand.
pub const FINE_ACTIVITY_DENSITY_WEIGHT: f32 = 6.0;

/// Particle-repulsion relaxation passes applied to the thinned fine points to
/// turn white-noise (sliver-prone) sampling into adaptive blue noise. 0 = off
/// (raw thinning). Each pass rebuilds a kd-tree and does a density-aware tangent
/// push (~1.5-2.5s/pass at 2.5M). A rho=nn/sqrt(area) sweep (June 2026, 2.25M,
/// see the mesh-quality probe) shows a sharp knee: 0->2 passes does almost all
/// the regularization (slivers<0.4 1.88%->0.65%, mean rho 0.76->0.91), and 2->5
/// is diminishing (mean 0.91->0.95). 3 captures ~95% of the 5-pass quality on
/// every metric for ~5s less. See src/bin/sample_experiment.rs.
pub const FINE_RELAX_PASSES: usize = 3;

// --- Fluvial erosion on the fine mesh (docs/specs/erosion.md) ---------------
//
// Detachment-limited stream power (Braun & Willett 2013 implicit, n = 1) plus
// linear hillslope diffusion, run on the fine mesh between fine-elevation
// refinement and final hydrology. The loop modifies CRUST THICKNESS; elevation
// responds isostatically (Airy slope) so the result composes with the
// thickness-based machinery the rest of the world uses. All lengths are on the
// unit sphere (arc radians / steradians); K, the diffusivity, and the uplift
// scale therefore carry the unit bookkeeping and are the primary VISUAL knobs
// (tune by eye on real renders, not against a numeric target).

/// Number of implicit incision timesteps. The implicit scheme is
/// unconditionally stable, so a few dozen large steps suffice. With uplift ON
/// (the Phase 3 "Hold & carve" default) the step count sets how far the orogen
/// evolves: uplift holds the divides while channels grade, so RELIEF builds with
/// steps and then plateaus as the profile approaches graded. At ~60 the reference
/// world (seed 12345) sits in that plateau — peaks held above the un-eroded base
/// (max ~+15%), valleys graded — and the OUTCOME is resolution-robust: the eroded
/// height distribution (mean/p90/p99/max, land volume) is stable to a few percent
/// across 300k–2.5M cells, even though gross incision *throughput* falls with
/// resolution. Judge the balance by that held-peaks/bounded-relief outcome, NOT by
/// "uplift-in ≈ eroded" (a 300k coincidence: the throughput ledger is resolution-
/// dependent and Phase-2 deposition sits on its denudation side). Fewer steps =
/// less relief; far more = over-graded lowlands. With uplift OFF (relaxation) the
/// count is just a visual stopping time. Per-step cost is dominated by re-routing,
/// so this is also the runtime knob.
// DEFAULT 60→200 (2026-06-21): the emergent build needs more carving time (steps spreads
// the same total uplift, so more steps = more dissection of the rising orogen) — 200 was
// the visual-validated budget. This ~3× the erosion stage's time; `steps` is the primary
// quality↔speed dial (lower = faster, less mature/dissected).
pub const EROSION_STEPS: usize = 200;

/// Timestep (dimensionless; folded into K / uplift scale). Kept explicit so the
/// Braun & Willett formula reads as written. Larger dt relaxes faster per step.
pub const EROSION_DT: f32 = 1.0;

/// Stream-power area exponent m in E = K A^m S^n. 0.5 is the canonical value;
/// raising it makes large rivers dominate incision.
pub const EROSION_M: f32 = 0.5;

/// Stream-power SLOPE exponent n in E = K A^m S^n. n = 1 is linear (the original,
/// chosen for the closed-form implicit solve) and gives the roundest, softest valley
/// profiles. n ≈ 1.5–2 (real bedrock channels) yields sharper valley walls, crisper
/// divides, and threshold-like response to uplift — the shape control for "ranges, not
/// bumps". n ≠ 1 uses a per-cell Newton solve of the implicit step (a few iterations).
pub const EROSION_N: f32 = 2.0;

/// Max Newton iterations for the n ≠ 1 implicit incision solve (per channel cell per
/// step). The step is small and F is convex, so it converges in ~3–4; 8 is headroom.
pub const EROSION_INCISION_NEWTON_ITERS: usize = 8;

/// Stream-power coefficient K (erodibility). Drainage area A is "wet area" in
/// steradians and the receiver distance is in arc radians, so K is small. The
/// master knob for how deeply rivers dissect: K carves valleys (high drainage
/// area) and strips lowlands, but holds peaks/divides (~zero drainage area) — so
/// it raises relief, it doesn't lower summits (that's `EROSION_DIFFUSIVITY`).
///
/// Balanced AGAINST diffusivity. With uplift off (relaxation), incision is the
/// only thing creating relief and hillslope diffusion the only thing destroying
/// it — so the regime is set by their ratio, not K alone. At the old K=1e-2 with
/// D=2e-7, diffusion won and erosion just SMOOTHED (every slope percentile fell
/// 25-60%). Raising K to 4e-2 and dropping D 10x flips it to incision-dominated:
/// slopes rise above base (p90/p99 +15/+17%), valleys cut while divides hold —
/// dissection. Denudation ~11% of land volume (a substantial erosion epoch).
/// Up = deeper canyons; pair with EROSION_DIFFUSIVITY (the smoothing knob).
pub const EROSION_K: f32 = 4.0e-2;

/// Hillslope diffusivity (soil creep). Rounds off ridgecrests and fills the
/// finest valleys; 0 disables diffusion (rivers only). Solved IMPLICITLY
/// (backward Euler), so it is unconditionally stable regardless of the finest
/// cell size — no CFL substep blow-up on sliver cells. Units: steradian/step.
///
/// This is the SMOOTHING knob; it fights incision (EROSION_K) for control of the
/// relief. The old 2e-7 over-smoothed in the relaxation regime (mountains went
/// completely smooth). Lowered 10x to 2e-8: just enough to tame cell-scale
/// spikes (eroded max-slope stays at/below base) without erasing the dissection
/// incision produces. Up = rounder/smoother; toward 0 = sharper but speckle-prone.
pub const EROSION_DIFFUSIVITY: f32 = 2.0e-8;

/// Roering nonlinear-hillslope critical slope S_c (escalation #2,
/// docs/specs/erosion-escalations.md). Linear creep (EROSION_DIFFUSIVITY) rounds
/// every ridge into mush and can't hold a crest against constantly-regenerated
/// relief without over-smoothing everything. The Roering, Kirchner & Dietrich
/// (1999) flux law `q = -D ∇h / (1 - (|∇h|/S_c)²)` instead lets the creep
/// diffusivity blow up as the slope approaches a critical angle S_c: gentle
/// slopes creep ~linearly (`q ≈ -D∇h`), but slopes near S_c transport so fast
/// they can't get steeper — so hillslopes become PLANAR and ridgelines CRISP
/// (the angle-of-repose facet) instead of rounded. This gives channel initiation
/// a credible, physically-limited hillslope to compete against (must precede the
/// channelization escalation #4).
///
/// Implemented as a slope-dependent conductivity multiplier `κ = 1/(1-(S/S_c)²)`
/// on the existing implicit edge weights, with the ratio `S/S_c` regularized
/// (clamped below 1) so the flux is bounded near S_c — the explicit singularity
/// the trap warns about can't fire. `κ ≡ 1` when this is 0, recovering the exact
/// linear diffusion above, so 0 = OFF (current behaviour).
///
/// UNITS: code slope = Δelevation / Δradian (the diffusion's native units, same
/// as EROSION_DEPOSITION_SLOPE), NOT a dimensionless grade. Convert a physical
/// hillslope grade (rise/run) via `S_code ≈ grade · R / (m per elev-unit)` =
/// `grade · 6.371e6 / 1e4 ≈ grade · 637` (canonical ~10 km/elev-unit). So a
/// ~30-45° angle-of-repose grade (0.6-1.0) is S_code ≈ 380-640. CAVEAT: this acts
/// on CELL-TO-CELL slopes (~1.5 km mountain cells), which are gentler than true
/// sub-cell hillslope angles — the eroded mesh's steepest cell slopes sit near
/// S_code ~290 (grade ~0.45) — so to bite on the steepest cells the starting band
/// is lower, S_code ~150-300 (grade ~0.24-0.47). Default off; calibrate via
/// `--erosion-hillslope-crit` against the curv-rms / peak% roughness counters
/// (crisper, more planar ridges should LOWER cell-scale curvature, unlike the #1
/// source smoothing which was a no-op). Watch the trap: too low → a
/// "critical-slope-everywhere" tiled-facet world.
pub const EROSION_HILLSLOPE_CRITICAL_SLOPE: f32 = 200.0;

/// Jacobi sweeps used to approximate the implicit diffusion solve each step.
/// More = closer to the exact backward-Euler smoothing. Verified ample at 6
/// (seed 12345, --fine-max 300000, --erosion-diffusion-iters sweep): 6 -> 24
/// gives bit-identical roughness and a diffused-net mass residual ~1e-10 of the
/// eroded volume, so the gentle hillslope diffusion (D ~ 2e-8) already reaches
/// its fixed point in <=6 sweeps even on the finest cells.
pub const EROSION_DIFFUSION_ITERS: usize = 6;

/// Re-route (priority-flood + drainage-area accumulation) every N incision
/// steps rather than every step. Routing dominates the loop cost (~75%), and
/// the drainage network is quasi-static over a few incision steps: channels
/// deepen along fixed paths, and the incision slope-guard safely skips any cell
/// whose stale receiver is no longer downhill. The network still reorganizes at
/// each re-route. 1 = re-route every step (exact, slowest). Higher = faster,
/// staler network between re-routes.
pub const EROSION_REROUTE_INTERVAL: usize = 6;

/// Barnes-style convergent flat resolution (Barnes, Lehman & Mulla 2014): on a
/// priority-flood-filled surface, drain flat interiors with a synthetic descent
/// toward outlets and away from higher walls, instead of the bare priority-flood
/// wavefront (`flood_parent`) which spirals and gets incised as spiral grooves.
/// `false` reproduces the old wavefront routing (for A/B). See
/// docs/specs/erosion-routing-ladder.md (Rung 1).
pub const EROSION_FLAT_RESOLUTION: bool = true;

/// Multiple-flow-direction exponent (Rung 2 area + Rung 3 incision). Each cell
/// spreads its flow — and its stream-power incision — to all downslope neighbours
/// weighted by `slope^p`; `p` is the SFD↔MFD dial: `p → ∞` recovers single-flow
/// (sharp, 1-cell channels = the perforation), `p ≈ 1` is fully dispersive
/// (smoother, valleys re-converge). `0` = pure SFD (single-receiver area +
/// incision, the pre-Rung-2 behaviour).
///
/// DEFAULT OFF. Rung 3 hypothesised MFD incision would break the 1-cell
/// perforation. It does NOT: sweeping p at 600k AND at full ~2.5M (seed 12345)
/// moved curv-rms only ~3% and left elevation unchanged — the cell-scale roughness
/// erosion adds (curv-rms ~7-8x base→eroded) is INVARIANT to SFD↔MFD, to diffusion
/// magnitude (Rung 0), and to reroute frequency. So the swiss-cheese is not
/// routing-driven; the cause is elsewhere in the loop (deposition / diffusion
/// solver / fold-back — under investigation). This MFD path is correct LEM practice
/// (divergent flow, mesh-insensitive) and kept as a toggle, but it is not the fix,
/// so it stays off (it adds a per-reroute sort + CSR for ~no gain on this artifact).
/// `p → ∞` ≈ SFD, `p ≈ 1` fully dispersive. A/B with `--erosion-mfd-exponent`.
/// See docs/specs/erosion-routing-ladder.md.
pub const EROSION_MFD_EXPONENT: f32 = 0.0;

/// Plains alluvial regime gate (docs/specs/erosion-valleys-not-channels.md,
/// Phase 1). Stream-power incision is faded by a confinement factor
/// `C = smoothstep(0, this, channel_slope)` in [0,1]: full incision (C=1) on
/// bedrock/mountain channels at/above this slope, fading to none (C=0) on gentle
/// alluvial plains — where the transport-aware deposition then grades the lowland
/// to a floodplain instead of erosion gouging a cell-wide ditch ("huge one-cell
/// river"). Channel slope is Δelev/Δkm (elev-units per km); the land distribution
/// runs p50 ~4e-4, p90 ~2e-3, so 5e-4 (~0.5% gradient) sits at the plains/upland
/// divide. `0` disables the gate (C=1 everywhere = the pre-Phase-1 behaviour).
/// DEFAULT OFF. The gate is physically correct for the plains-floodplain goal, but
/// the A/B (2026-06-16) showed it is NOT the swiss-cheese fix: at honest channel
/// slopes (≤3e-3) it only fades gentle PLAINS channels and does ~nothing visible,
/// because the "too busy/sharp" dissection lives in steeper UPLAND incision. Only
/// an absurd value (~3e-1, a 300% gradient — i.e. incision off everywhere) removes
/// it, and that flattens mountains too. So this stays off and dormant (kept for
/// docs/specs/erosion-valleys-not-channels.md Phase 1 / plains floodplains); the
/// actual dissection-texture levers are K, diffusivity, and channel support.
pub const EROSION_CONFINEMENT_SLOPE: f32 = 0.0;

/// Scales tectonic uplift added to crust thickness each step. Source is the
/// transferred feature forcing: (arc + collision) are elevation magnitudes
/// (converted to thickness by dividing the Airy slope) and rift_delta is
/// already a signed thickness delta. U_thickness = SCALE * ((arc+collision)/slope
/// + rift_delta).
///
/// Phase 3 "Hold & carve" (erosion-v2): uplift is ON, calibrated so it roughly
/// BALANCES erosion — divides are held (peaks ~preserved) while channels carve
/// graded valleys, so RELIEF increases instead of the whole orogen decaying
/// (relaxation) or inflating (over-uplift, the old double-count). It must not
/// re-inject the full orogen — the coarse elevation already encodes the static
/// height — so SCALE buys only the ongoing uplift during the erosion epoch and
/// must stay small.
///
/// Calibration (seed 12345, sweeps on --erosion-uplift-scale / --erosion-steps):
/// at 0.003 with the default 60 steps the eroded surface holds high terrain above
/// the un-eroded base (p90/p99/max raised, peak ~+15%) while valleys incise — the
/// "Hold & carve" outcome. Verified resolution-robust: that height distribution is
/// stable to a few percent across 300k–2.5M cells (see EROSION_STEPS). Read the
/// held-peaks/bounded-relief OUTCOME, not the throughput ledger — gross eroded
/// falls with resolution and Phase-2 deposition feeds it, so "uplift-in ≈ eroded"
/// is not the invariant. The step count co-sets how far relief evolves (see
/// EROSION_STEPS), so move them together. 0 = relaxation only (peaks decay); up =
/// taller, more actively-rising orogens.
pub const EROSION_UPLIFT_SCALE: f32 = 0.003;

/// Forcing-smoothing length (km) for the tectonic uplift SOURCE, applied once in
/// `ErosionState::new` before the step loop. Escalation #1
/// (docs/specs/erosion-uplift-smoothing.md): the per-step uplift source
/// `u_thick = SCALE·((arc+collision)/slope + rift_delta)` is built from coarse
/// feature fields interpolated onto the fine mesh; if that handoff carries
/// wavelengths below the landform scale, erosion re-injects high-frequency
/// tectonic "work" every step and the orogens equilibrate with cell-scale chatter
/// ("swiss cheese"). Component-isolation pinned uplift as the dominant mountain-top
/// bump source. So we smooth the FORCING — never the final elevation — over a
/// physically-named length: the sub-grid orogenic forcing width (real tectonic
/// uplift is smooth at cell scale; sub-cell variation is unmodelled detail).
///
/// Mechanism: a conservative, area-weighted graph diffusion of `u_thick` within
/// the LAND mask (no-flux at the coastline, so uplift never bleeds onto submerged
/// cells and the integrated area-weighted uplift is preserved exactly). Because
/// the diffusion is linear, smoothing the combined source is identical to
/// smoothing arc/collision/rift_delta separately and recombining — the signed
/// rift thinning superposes correctly with the positive orogen uplift (no
/// magnitude blur, so the trap of rift thinning bleeding into orogen uplift does
/// not arise). The length is the kernel's Gaussian std-dev (variance 2·D·τ per
/// axis), so e.g. 30 km smooths sub-30 km source chatter while leaving the macro
/// orogen front intact.
///
/// 0 = off (current behaviour — the source enters erosion unsmoothed). Calibrate
/// to the cell/landform scale (a few × the fine mountain-cell size): enough to
/// kill sub-landform speckle (`morans_i(u_thick)` rises toward smooth), not so
/// much that it rounds the genuine orogen front or flattens peaks. Exposed via
/// `ErosionParams` + diagnose/app (`--erosion-uplift-smooth`) for A/B.
pub const EROSION_UPLIFT_SMOOTH_KM: f32 = 0.0;

/// Fraction of a terminal sink's depth-to-base-level (sea level, or a lake
/// surface) that routed sediment may fill (delta / lake-infill building at the
/// mouth). Sediment beyond this cap is "lost to the ocean" for the mass-balance
/// log. Keeps deposition from breaching the local water surface.
pub const EROSION_DEPOSIT_FILL_FRACTION: f32 = 0.5;

/// Depositional repose slope (elevation per radian) for en-route aggradation. As
/// sediment routes to the sea/lakes, a reach aggrades toward the surface that
/// grades to its receiver at this slope — so low-gradient reaches (valley floors,
/// alluvial fans where mountains meet plains, lake/coast margins) fill while steep
/// reaches pass their load downstream. This is the transport-limited half of the
/// sediment cycle: it builds floodplains, fans, foreland-style fills, and broad
/// deltas rather than dumping each catchment's load at one mouth.
///
/// Units match the incision slope (Δelevation / Δarc-radian). Larger = sediment
/// aggrades steeper reaches too (more pervasive valley fill, less relief); smaller
/// = deposition confined to the flattest ground (closer to sink-fill only). 0
/// disables en-route deposition. The primary deposition knob; sweep with diagnose
/// --erosion-deposition-slope and read the mass ledger (lost-to-ocean should drop)
/// alongside the maps. Starting value — calibrate by eye, not to a target.
///
/// PHYSICAL SCALE (the knob's real meaning, cell-size-independent): a reach
/// aggrades only where its bed grade is gentler than this slope. Converting to a
/// surface grade, `grade ≈ SLOPE · (m per elev-unit) / R` with ~10 km/unit (the
/// canonical vertical scale) and R = 6.371e6 m, so `grade ≈ SLOPE × 0.16%`.
/// The old value 1.0 was a ~0.16% repose grade — only deltas/lake floors are that
/// flat, so en-route deposition was effectively inert (sink-fill only) at the
/// fine mesh's radian-scale receiver distances. Floodplains and alluvial fans sit
/// at ~0.5–2%, i.e. SLOPE ≈ 3–13. 6.0 (~0.9% grade) is a conservative starting
/// point in that band — meaningful valley-floor / fan aggradation without
/// over-filling; raise toward 13 for more pervasive fill. Calibrate on renders.
pub const EROSION_DEPOSITION_SLOPE: f32 = 6.0;

/// Channel-initiation support area (km²) at MEAN land wetness. Stream-power
/// incision acts only on cells whose precip-weighted drainage (discharge)
/// exceeds the discharge of this geometric area at the mean land precipitation;
/// below it the cell is a hillslope and erodes by hillslope diffusion only (the
/// channel head). Without this, stream power incises everywhere there is any
/// drainage — including hillslopes that should not channelize — so valleys lack
/// a clean head and divides get nibbled.
///
/// Because the threshold is on discharge (not bare area), the GEOMETRIC support
/// shrinks in wetter-than-mean regions and grows in arid ones: channels start
/// higher up where it rains -> denser networks in wet regions, sparser in
/// deserts, with no extra knob. A real channel-initiation support area on Earth
/// spans ~10⁻²–10¹ km² (humid → arid); at the fine mesh's ~1.5 km mountain cells
/// this gates the few-cell hillslope band between channel heads. The primary
/// drainage-density knob — up = sparser/blockier networks, down = finer
/// dissection (0 disables). 0 = off.
/// Lowered 30→4 (2026-06-20): on the structured-emergent orogens the high massif was
/// UNDER-dissected (channels didn't reach the divides → active uplift made hillslope
/// spikes); ~3-5 km² lets channels initiate higher and carve coherent ridge-and-valley
/// into the high terrain instead of spikes (visual-settled).
pub const EROSION_CHANNEL_SUPPORT_KM2: f32 = 4.0;

// --- Lithologic erodibility K(x) (role-1 fine-mesh seed) -------------------
//
// Real terrain gets much of its character from rock that resists erosion
// unevenly (hard ridges, soft basins, knickpoints at contacts). Rather than
// paint that grain (the retired ridge layer), perturb the incision coefficient
// K with an fBm "rock varies" field evaluated at fine cell centers and let
// erosion organize it — the grain then emerges ALIGNED with drainage. The field
// is normalized to unit area-weighted mean over land, so sigma only
// REDISTRIBUTES incision (harder rock holds relief, softer strips it) without
// changing total denudation. Applies to stream-power incision only; hillslope
// diffusion stays uniform.

/// Log-amplitude of lithologic erodibility variation (the contrast knob). The
/// per-cell multiplier is exp(sigma * fbm), so sigma = 0.7 spans ~exp(±0.7) =
/// 0.5–2.0 (≈4x hard-to-soft). 0 = uniform K (no lithology). The effect is a
/// pure redistribution, so it is subtle at low total erosion — raise it (or K)
/// for bolder differential dissection. Sweep with diagnose --erosion-litho-sigma.
pub const EROSION_LITHO_SIGMA: f32 = 0.7;

/// Base spatial frequency of lithologic units (fBm on the unit sphere). ~12 puts
/// the coarsest units near terrane scale (~500 km); octaves carry it down to
/// formation scale (~30–60 km) so individual valleys cross different rock.
pub const EROSION_LITHO_FREQUENCY: f64 = 12.0;

/// fBm octaves for the lithology field (multi-scale terrane → formation grain).
pub const EROSION_LITHO_OCTAVES: usize = 4;

/// Log-amplitude of the GEOLOGY-tied erodibility contrast, layered on top of the
/// fBm texture (added in log-K space). Deep continental interiors are old, hard
/// cratonic basement (lower K) and volcanic arcs are fresh, fractured terrain
/// (higher K): per-cell `geo_log = STRENGTH * (arc_norm − continentality)`, so
/// 0.5 spans ~exp(±0.5) ≈ 0.6–1.6x on top of the fBm. Unlike the free fBm, this
/// grain follows the world's geology (same continentality/arc fields elevation
/// uses). Normalized to unit land mean downstream (pure redistribution). 0 =
/// fBm-only lithology.
pub const EROSION_LITHO_GEO_STRENGTH: f32 = 0.5;

/// Log-amplitude of the STRUCTURAL-GRAIN erodibility contrast: alternating hard/
/// soft bands in convergent belts, `grain_log = STRENGTH · conv · sin(τ·conv /
/// WAVELENGTH)`, where `conv` is the land-normalized convergence. The bands strike
/// along iso-convergent contours (≈ the collision front / fold-axis strike) and
/// strengthen toward the suture — a fold-and-thrust fabric the incision can express
/// as ridge-and-valley / trellis drainage. EXPERIMENTAL (trellis is an outcome to
/// test, not a promise). 0 = no grain. Sweep with diagnose --litho-grain-strength.
pub const EROSION_LITHO_GRAIN_STRENGTH: f32 = 0.6;

/// Fold wavelength of the structural grain, in units of normalized convergence
/// (0..1 across a belt). ~0.15 gives ~6–7 bands across the convergent gradient,
/// tightest near the suture (where the convergence gradient is steepest). Smaller
/// = more, tighter folds.
pub const EROSION_FOLD_WAVELENGTH: f32 = 0.15;

// --- Orographic precipitation feedback (climate↔erosion) ---------------------
//
// After erosion carves the fine relief, the coarse precipitation is modulated by
// the orographic forcing (wind·∇elev) on the NEW relief: windward slopes wetter,
// lee drier (rain shadows behind the carved ranges). The coarse moisture model
// already handled the large-scale transport; this adds the fine-scale detail the
// coarse mesh couldn't resolve. Renormalized to land mean 1.0 so hydrology
// budgets are unchanged in the mean.

/// Strength of the orographic precip modulation. The per-cell factor is
/// (1 + STRENGTH * signed_orographic_norm), clamped to [MIN, MAX]. 0 disables it
/// (precip = coarse). Up = sharper rain shadows. Conservative default — the
/// effect is local (windward/lee of a range) so it's judged on maps, not the
/// global drainage-density probe; sweep with diagnose --erosion-orographic-strength.
pub const OROGRAPHIC_PRECIP_STRENGTH: f32 = 0.6;
/// Lower clamp on the modulation factor (driest lee = MIN x coarse precip).
pub const OROGRAPHIC_PRECIP_MIN: f32 = 0.3;
/// Upper clamp on the modulation factor (wettest windward = MAX x coarse precip).
pub const OROGRAPHIC_PRECIP_MAX: f32 = 2.5;

// --- Downwind rain shadow (side-aware lee drying; docs/specs/rain-shadow.md) ---
// The local orographic term above is purely per-cell, so a flat lee basin behind a crest
// (wind·∇elev≈0) snaps back to coarse precip. This propagates each cell's LOCAL lee
// dry-anomaly DOWNWIND along the per-cell wind in one ordered O(N) DAG sweep (NOT CFL
// advection — that over-dried the fine mesh), so the lee progressively dries. Mean-invariant
// (renormalized), so only the spatial distribution shifts windward-wet / lee-dry.
/// Master strength: fraction of the propagated downwind dry-anomaly applied. 0 = OFF
/// (bit-identical to local-only). Up = stronger progressive lee drying.
pub const DOWNWIND_SHADOW_STRENGTH: f32 = 0.0;
/// Fetch decay length in KM (converted to unit-sphere chord via PLANET_RADIUS_KM): the
/// anomaly carried per hop decays as exp(−chord/fetch), so a plume dies after a bounded arc
/// length independent of cell count (resolution-robust — the fix for the over-dry trap).
pub const DOWNWIND_SHADOW_FETCH_KM: f32 = 300.0;
/// Hard floor as a fraction of coarse precip (≤ OROGRAPHIC_PRECIP_MIN so it never raises the
/// local lee minimum). Applied pre-renorm; audited post-renorm.
pub const DOWNWIND_SHADOW_FLOOR: f32 = 0.25;
/// Minimum cos(wind[i], i→neighbor) for a neighbor to count as downwind (0.5 ≈ ±60° cone).
pub const DOWNWIND_CONE_COS: f32 = 0.5;

/// Coupled erode↔precip feedback passes. Each pass re-carves the base relief with
/// the rain-shadow precipitation from the previous pass, so windward flanks
/// (wetter) dissect more than lee. 1 = erode once (precip still recomputed for
/// hydrology, but no erosion feedback); 2 = one feedback pass. Cost is ~linear
/// (each pass is a full erosion run), so keep it small. Dial to 1 if stage-4 is
/// too slow at full resolution.
pub const EROSION_PRECIP_OUTER_ITERS: usize = 2;

/// Lakes as evaporation sources: standing water adds local humidity, boosting
/// precipitation in a halo around lakes. The per-cell factor is (1 + STRENGTH *
/// lake_halo), renormalized to land mean 1.0. Local only (the fine mesh doesn't
/// re-advect moisture), applied once after the lakes are known, so no runaway
/// lake growth. 0 disables.
pub const LAKE_EVAP_STRENGTH: f32 = 0.5;
/// Diffusion steps that spread the lake humidity into the surrounding halo.
pub const LAKE_EVAP_DIFFUSE_STEPS: usize = 5;

// --- Glacial erosion (v1: snowline-driven ice over-deepening) -----------------
//
// A post-fluvial pass on the fine mesh. Cells above a latitude-dependent snowline
// accumulate ice; ice routes downslope (reusing the fluvial routing) and abrades
// its bed in proportion to ice flux, so trunk glaciers over-deepen into U-troughs
// and tarn basins while the low-flux peaks/divides between them erode little and
// stand out sharper (arêtes, horns). Works in elevation space (no isostatic
// response in v1). U-shape valley *widening* is the deferred v2 piece.

/// Glacial abrasion coefficient (ice-flux → bed lowering). The master "how
/// glacial" knob. 0 disables the whole pass (no glaciers). Up = deeper troughs /
/// more over-deepening. Sweep with diagnose --glacial-k and read the glaciated
/// coverage + relief change it logs (a reference, not a target).
///
/// DEFAULT-OFF (2026-06-20). The v1 single-flow pass injects cell-scale roughness
/// ("prickle": curv-rms ~8x, glacial-flux checker ~49%) instead of clean glacial
/// landforms — it point-routes ice like water (no glacier width / ice-surface
/// slope). Disabled until the OpenLEM-style glacial stream-power rework lands; see
/// docs/specs/erosion-glacial-streampower.md. Cosmetic snow caps (coloring.rs
/// `apply_snow_cap`) are independent of this and still render. Set >0 to A/B the v1
/// pass; production stays fluvial-only for now.
pub const GLACIAL_K: f32 = 0.0;

/// Glacial sub-steps (re-route + accumulate + abrade each), letting troughs and
/// over-deepened basins develop. A handful suffices; the routing dominates cost.
pub const GLACIAL_STEPS: usize = 6;

/// Snowline (equilibrium-line) ELEVATION at the equator, in the normalized
/// elevation units (sea level 0). Only terrain above the local snowline glaciates,
/// so near the equator just the highest peaks carry ice.
pub const GLACIAL_SNOWLINE_EQUATOR: f32 = 0.30;

/// Snowline ELEVATION at the poles. Lower than the equator (cold), so polar
/// uplands glaciate down toward the coast. The per-cell snowline interpolates as
/// EQUATOR + (POLE − EQUATOR)·sin²(lat), latitude from the cell position (this
/// keeps elevation out of the snowline so it isn't double-counted against the
/// ice-load term, unlike the lapse-baked temperature field).
pub const GLACIAL_SNOWLINE_POLE: f32 = 0.02;

/// Ice ablation rate below the snowline: ice flux loses ABLATION × (depth below
/// snowline) per cell, so glaciers extend a bounded tongue below the snowline
/// rather than flowing forever. Up = shorter tongues (ice melts faster).
pub const GLACIAL_ABLATION: f32 = 6.0;

/// Maximum reverse gradient (elevation) glacial abrasion may carve below a cell's
/// receiver — the over-deepening that makes closed rock basins (tarns / paternoster
/// steps) which later fill as lakes. Capped so over-deepening stays bounded; never
/// carves below sea level. Up = deeper tarns.
pub const GLACIAL_OVERDEEPEN_MAX: f32 = 0.012;

// --- Fault range fronts (v1 proxy: sharpen active orogen margins) -------------
//
// The lithosphere is a smooth isostatic field, so range fronts grade out softly —
// real fault-bounded ranges rise abruptly along a near-linear front. Rather than a
// discrete fault-trace generator (deferred), this imposes a localized scarp on the
// fine base BEFORE erosion, at the contour of the orogen forcing (collision +
// convergence) — which follows the plate boundary, so the front is boundary-seeded.
// The scarp is the derivative-of-Gaussian of the (normalized) forcing across that
// contour: footwall edge up, hanging-wall/basin edge down, tapering to zero in the
// deep interior and far basin (a steepening, not a second uplift). Erosion then cuts
// canyons through the steepened front and the triangular facets EMERGE between them.

/// Scarp relief (elevation units) imposed at the active range front. The master
/// knob; 0 disables the pass (no front sharpening). Peak offset is ~0.6x this at
/// the band centre. Sweep with diagnose --fault-scarp.
// DEFAULT 0.04→0.0 (2026-06-21): painted range-front scarps OFF under the emergent default
// (the structured uplift builds the front). Painted A/B only.
pub const FAULT_SCARP_HEIGHT: f32 = 0.0;

/// Front location: the fraction of the land-max orogen forcing whose contour marks
/// the range front (where the scarp centres). ~0.4 puts it on the flank of the
/// orogen where it meets the lowland.
pub const FAULT_FRONT_THRESHOLD: f32 = 0.4;

/// Front width in normalized-forcing units: how far either side of the threshold
/// contour the scarp reaches. Smaller = sharper, more sharply localized front.
pub const FAULT_FRONT_BAND: f32 = 0.2;

/// Hanging-wall (basin) subsidence as a fraction of footwall uplift. Real range
/// fronts express footwall uplift far more than basin drop (basins fill with
/// sediment), and a symmetric scarp over-drops the lowlands and drowns coastal
/// land, so the basin side is damped well below 1.0.
pub const FAULT_BASIN_DROP_FRAC: f32 = 0.25;

// --- Fine interior structural relief (erosion-v2 Phase 1 / P1a) ---
// The fine base is the coarse elevation INTERPOLATED onto fine cells, so orogen
// interiors are smooth flat-topped plateaus (the coarse forcing saturates there):
// no drainage gradient, so erosion spirals into cottage-cheese instead of
// dissecting real ranges (root cause #1, [[hex3-summit-cottage-cheese]]). This is
// a mid-band structural HEIGHT term (fault-block / fold grain) added to the base
// BEFORE pre-hydrology, gated to high orogen interiors, made coarse-cell-local
// zero-mean so it adds sub-coarse structure WITHOUT shifting the coarse datum /
// land fraction. It is the SUBSTRATE erosion carves, not painted final relief.
// P1a is isotropic fBm (soft, no strike-aware fronts — that's P1b); the height
// term is what fix #7 (codex) requires: erodibility alone gives a flat summit no
// gradient. Sweep amplitude with `--sweep interior_relief` / diagnose.

/// Amplitude (elevation units) of the interior structural relief. Zero-mean per
/// coarse cell, so this is the ~peak fBm excursion (≈ this × 10 km). 0 = off
/// (pure interpolant — the old flat-top behaviour). The master P1a knob.
// DEFAULT 0.04→0.005 (2026-06-21): under the emergent default this is just a FAINT seed
// to break drainage symmetry (erosion builds the relief). For the painted A/B path set it
// back to ~0.04 (the structural-substrate amplitude).
pub const FINE_INTERIOR_RELIEF: f32 = 0.005;

/// Base spatial frequency of the interior fBm (cell_center is a unit vector, so
/// the base-octave wavelength ≈ 1/this rad; 160 ≈ 40 km, squarely in the mid-band
/// between the coarse Nyquist (~70 km) and the channel scale (~few km)).
pub const FINE_INTERIOR_FREQUENCY: f64 = 160.0;

/// Octaves of the interior fBm. 5 reaches ≈ 40/16 ≈ 2.5 km at the finest octave,
/// down to the channel scale erosion organizes on.
pub const FINE_INTERIOR_OCTAVES: usize = 5;

/// Elevation gate (elevation units): structure fades in over
/// [`MIN`, `MIN`+`BAND`]. Keeps the (signed) relief on high orogen terrain well
/// above sea level so the zero-mean negative excursions never flip a near-coast
/// land cell to ocean (land fraction preserved by construction, not by clamping —
/// clamping would re-introduce a positive bias). MIN must exceed the worst-case
/// downward excursion (~2× amplitude after the per-cell mean subtraction) for the
/// default amplitude; a high-amplitude sweep may show small land drift (caught by
/// the area-weighted drift check).
pub const FINE_INTERIOR_MIN_ELEV: f32 = 0.10;
pub const FINE_INTERIOR_ELEV_BAND: f32 = 0.08;

/// Orogen-forcing gate: structure fades in over [`THRESHOLD`, `THRESHOLD`+`BAND`]
/// of the land-normalized (collision+convergent+arc) forcing, so faults/folds live
/// in convergence belts and cratons stay quiet (per the noise philosophy).
pub const FINE_INTERIOR_FORCING_THRESHOLD: f32 = 0.15;
pub const FINE_INTERIOR_FORCING_BAND: f32 = 0.35;

/// Tolerance for the area-weighted land-fraction drift the structural relief may
/// introduce (interior grain + scarps) before it's flagged as invalidating the
/// coarse atmosphere's land/ocean mask. The relief is zero-mean + gated above sea
/// level, so well-behaved knobs sit far below this; it catches high-amplitude
/// sweeps flipping near-coast cells.
pub const FINE_STRUCTURE_LAND_DRIFT_TOL: f32 = 1e-3;

// --- Fine strike-aware structural relief (erosion-v2 Phase 1 / P1b) ---
// P1a's interior grain is isotropic fBm (the transferred fields carry no boundary
// strike). P1b orients it ALONG the orogen front using the coarse convergent plate
// boundaries as primitives: the signed great-circle distance to the nearest
// compatible front is a globally-consistent field whose iso-distance contours run
// PARALLEL to the front, so a banded function of that distance gives ridge-and-
// valley grain striking along the range (fold-and-thrust fabric) — without any
// per-cell strike vector (which would seam at kinks). Near a front the relief blends
// toward the bands; far from any front it falls back to P1a isotropic grain. Magnitude
// stays gated by the P1a elevation+forcing gates (primitives drive ORIENTATION, not a
// new uplift model). See docs/specs/erosion-fine-synthesis.md rung P1b.

/// Blend weight (0..1) of the strike-banded grain vs P1a isotropic grain at a front
/// (faded by distance to the front). 0 = pure isotropic (P1a); 1 = pure bands. The
/// master P1b knob.
// DEFAULT 0.7→0.0 (2026-06-21): P1b strike-banding is OFF — it painted global "sand dune"
// corduroy (global structure can't be painted; tectonics/O0 owns it). Painted A/B only.
pub const FINE_FRONT_STRIKE_WEIGHT: f32 = 0.0;

/// Across-front band frequency (cycles per radian of front distance): the ridge-and-
/// valley wavelength of the fold grain. 200 ≈ 32 km bands (mid-band, dissectable).
pub const FINE_FRONT_BAND_FREQUENCY: f64 = 200.0;

/// Front influence radius (radians): beyond this great-circle distance from any
/// compatible front a cell gets no banding and falls back to P1a isotropic grain.
/// 0.09 rad ≈ 573 km spans a typical orogen interior from its front.
pub const FINE_FRONT_INFLUENCE_RADIUS: f32 = 0.09;

/// Along-strike warp of the bands (radians, added to the front distance before
/// banding) so ridges undulate instead of being perfectly parallel contour lines.
/// ≈ 25 km, about one band wavelength.
pub const FINE_FRONT_WARP: f32 = 0.004;
/// Spatial frequency of the along-strike warp fBm (low = long, smooth undulations).
pub const FINE_FRONT_WARP_FREQUENCY: f64 = 40.0;

/// Extra gather radius (radians) added to the influence radius when collecting
/// candidate front arcs: an arc whose midpoint anchor sits just past the influence
/// radius but whose body enters it is still considered. ≈ one coarse half-edge.
pub const FINE_FRONT_GATHER_MARGIN: f32 = 0.012;

// --- Fine active/passive margin contrast (erosion-v2 Phase 1 / P1c) ---
// The coarse model already widens the SHELF by margin activity (continentality →
// crust thickness → bathymetry). P1c adds the sub-coarse STRUCTURAL contrast at the
// continental margin: where high terrain meets the coast near a convergent front
// (ACTIVE margin, Andes-type), sharpen the structural relief so ranges rise steeply
// from the shore; on a PASSIVE coast (no convergence) damp it so the coastal upland
// stays quiet. Keyed on the interpolated raw `Crust.signed_margin_distance` (radians
// from the coast; + continental, − oceanic) × the transferred convergent forcing.
// Modulates AMPLITUDE within the existing elevation gate (land-safe), so it never
// touches the land/ocean mask. 0 contrast reduces to P1b exactly.

/// Master P1c knob: strength of the margin contrast (0 = off → P1b; 1 = full).
// DEFAULT 1.0→0.0 (2026-06-21): P1c margin contrast OFF under the emergent default (it
// modulated the painted relief; emergent owns the structure). Painted A/B only.
pub const FINE_MARGIN_CONTRAST: f32 = 0.0;

/// Coastal band width (radians of margin distance): the modulation is full at the
/// coast (margin ≈ 0) and fades to neutral by this distance inland. 0.05 ≈ 318 km.
pub const FINE_MARGIN_WIDTH: f32 = 0.05;

/// Relief amplitude scale at the coast for a fully ACTIVE margin (convergent front at
/// the shore): >1 sharpens the coastal front.
pub const FINE_MARGIN_ACTIVE_FACTOR: f32 = 1.3;
/// Relief amplitude scale at the coast for a fully PASSIVE margin: <1 quiets it.
pub const FINE_MARGIN_PASSIVE_FACTOR: f32 = 0.5;

// --- Emergent orogens (erosion-v3 prototype) ---
// Demote the orogen peak from the static fine base and re-supply it as active uplift, so
// fine erosion BUILDS dissected ranges instead of dissecting a pre-baked flat plateau.
// See docs/specs/erosion-v3-emergent-orogens.md. `lambda` = fraction of the orogen peak
// (arc+collision elevation) removed from the base envelope and rebuilt by uplift over the
// erosion epoch. 0 = off (current painted/postprocessor behaviour); 0.25-0.5 = the
// prototype's hedge; 1.0 = build the orogen entirely from uplift (riskiest).
// Rebuild calibration: uplift_scale * steps * dt ~= lambda (today 0.18); raise
// EROSION_UPLIFT_SCALE accordingly when emergent.
// DEFAULT (2026-06-21): emergent orogens are the PRODUCT path (validated: structured
// uplift + n>1 + channel_support≈4 beats painted P1 dunes + the smooth dome). Set 0 to
// fall back to the painted path (also requires the painted knobs — interior_relief 0.04,
// front_strike_weight 0.7, etc.). See docs/specs/orogen-structure.md.
pub const FINE_EMERGENT_LAMBDA: f32 = 0.5;

// --- O0: structured (asymmetric + segmented) emergent uplift (orogen-structure.md) ---
// Replace the uniform per-cell rebuild of the demoted orogen with a TECTONIC uplift
// SHAPE: an asymmetric front profile (steep narrow foreland flank → crest at the front
// → gentle wide hinterland, keyed on overriding-vs-foreland plate side) × a low-freq
// along-strike SEGMENTATION (proxy noise; the range plunges/segments) × the demoted
// forcing. Volume-normalized so total uplift is preserved (height stays sane), but the
// DISTRIBUTION is tectonic. Drives emergent + n>1. 0 = off (uniform v3 rebuild).

/// Master O0 knob: blend of structured shape vs uniform rebuild. 0 = uniform (v3); 1 =
/// fully structured. The decisive-hack default for an A/B run is 1.
pub const FINE_EMERGENT_STRUCTURED: f32 = 1.0;

/// Asymmetric front profile widths (radians of front-normal distance): steep narrow rise
/// on the FORELAND side (≈64 km), gentle wide taper into the HINTERLAND (≈320 km). The
/// crest sits at the front (v=0).
pub const FINE_OROGEN_FORELAND_WIDTH: f32 = 0.01;
pub const FINE_OROGEN_HINTERLAND_WIDTH: f32 = 0.05;

/// Along-strike segmentation proxy frequency (low → long plunging segments). O0 smoke-test
/// only — NOT a real along-front coordinate (see spec; real arc-length chaining is O3).
pub const FINE_OROGEN_SEGMENT_FREQUENCY: f64 = 14.0;
/// Minimum segmentation amplitude (so segments thin/plunge but the belt doesn't fully
/// vanish into disconnected blobs at O0). Segmentation spans [MIN, 1].
pub const FINE_OROGEN_SEGMENT_MIN: f32 = 0.15;

// --- Candidate A: meso structured emergent uplift (relief-spectrum-redesign.md) ---
// Modulates the O0 uplift SHAPE before erosion's volume normalization, so it
// redistributes uplift into 10-50 km proto-ridges/valleys instead of adding height.

/// Master Candidate A knob: modulation depth of the meso uplift-shape field. 0 = off.
pub const FINE_MESO_RELIEF: f32 = 0.0;
/// Candidate A' base-elevation meso relief amplitude, in elevation units. 0 = off.
pub const FINE_MESO_BASE_RELIEF: f32 = 0.0;
/// Cross-strike fold-train wavelength in km. 25 km sits in the target meso band.
pub const FINE_MESO_WAVELENGTH_KM: f32 = 25.0;
/// Along-strike phase modulation frequency (cycles/radian of chained front arc).
pub const FINE_MESO_PHASE_FREQUENCY: f64 = 18.0;
/// Along-strike spur/gap frequency, roughly 2x the O0 segmentation frequency.
pub const FINE_MESO_SPUR_FREQUENCY: f64 = 28.0;
/// Maximum phase warp, in fold cycles, from the along-strike fBm.
pub const FINE_MESO_PHASE_WARP: f32 = 1.25;
/// Local wavelength jitter from the phase fBm. Keeps bands from becoming corduroy.
pub const FINE_MESO_WAVELENGTH_JITTER: f32 = 0.35;
/// Minimum fold-train amplitude through along-strike gaps.
pub const FINE_MESO_AMP_MIN: f32 = 0.10;
/// Minority isotropic 3D noise blended into the front-coordinate fold field.
pub const FINE_MESO_ISO_WEIGHT: f32 = 0.20;
/// Spatial frequency of the isotropic meso component on the unit sphere.
pub const FINE_MESO_ISO_FREQUENCY: f64 = 170.0;
/// Master irregularity dial (0..1) for the fold train. 0 reproduces the plain
/// 1-D field ("ridges too consistent" — the 2026-07-10 visual verdict); >0 scales
/// three metronome-breakers together: cross-strike decorrelation, a second
/// incommensurate fold octave, and crest sharpening.
pub const FINE_MESO_IRREGULARITY: f32 = 0.7;
/// Cross-strike decorrelation length of the phase/spur modulation, in fold
/// wavelengths: adjacent ridges stop being phase-locked copies past ~this many λ.
pub const FINE_MESO_DECOR_WAVELENGTHS: f32 = 2.5;
/// Second fold octave wavelength ratio (incommensurate — golden ratio — so the
/// beat pattern against the primary never repeats).
pub const FINE_MESO_OCTAVE2_RATIO: f32 = 0.618;
/// Second fold octave amplitude at full irregularity, relative to the primary.
pub const FINE_MESO_OCTAVE2_AMP: f32 = 0.6;
/// Crest/valley sharpening at full irregularity: |fold|^(1-this) pulls the
/// cross-section away from the symmetric-sine "dune" profile.
pub const FINE_MESO_SHARPEN: f32 = 0.35;

// -- Massif-corridor meso style (A2+A3, relief-spectrum spec §13). The alpine
// default: irregular uplift massifs separated by branching low-uplift valley
// corridors, so ridges and valleys AGREE (the fold train's measured failure was
// "stamped ribs + indifferent drainage"). All scales derive from
// FINE_MESO_WAVELENGTH_KM so one dial scales the construction.

/// Meso construction style: 0 = fold train (foreland/fold-thrust preset),
/// 1 = massif-corridor (alpine default).
pub const FINE_MESO_STYLE: usize = 1;
/// Massif lattice period along strike, in fold wavelengths (~40 km at λ=25).
pub const FINE_MESO_MASSIF_PERIOD: f32 = 1.6;
/// Corridor lattice period along strike, in fold wavelengths (~45 km at λ=25;
/// ≈ Hovius outlet spacing for these hinterland widths).
pub const FINE_MESO_CORRIDOR_PERIOD: f32 = 1.8;
/// Fraction of corridors that cross the crest into the foreland (water gaps /
/// antecedent trunks); the rest fade out below the crest (interdigitation).
pub const FINE_MESO_CORRIDOR_CROSS_FRACTION: f32 = 0.15;
/// Corridor obliquity range vs the cross-strike normal, degrees.
pub const FINE_MESO_CORRIDOR_OBLIQUITY_DEG: (f32, f32) = (20.0, 40.0);
/// Minimum Gaussian sigma for massifs/corridors, km (mesh floor ~2 cells).
pub const FINE_MESO_MIN_SIGMA_KM: f32 = 8.0;
/// Cap on the POSITIVE (massif) side of the field. Uncapped massifs inflate
/// peaks (12-18 km, "reach the heavens" — user-rejected 2026-07-10); relief must
/// come from corridors cutting DOWN at a fixed peak budget, not summits up.
pub const FINE_MESO_MASSIF_CAP: f32 = 0.2;
/// Gain on the corridor (negative) side: >1 lets corridor cores drive local
/// uplift to zero at high meso_relief (deep valley seeds).
pub const FINE_MESO_CORRIDOR_GAIN: f32 = 1.6;

/// Emergent builder over-rebuild gain. The builder uplift rebuilds the demoted
/// envelope (`coarse_target − base`) over the erosion epoch; >1 compensates the
/// material erosion removes WHILE building, so the eroded orogen lands near the coarse
/// target rather than under it. ~1.2 ≈ +20%. Self-calibrating: the per-cell uplift rate
/// is `gain·(target−base)/(steps·dt·slope)`, so `steps` becomes a pure build-vs-carve
/// dial (more steps = same total uplift, more carving time) without moving the height.
pub const EMERGENT_REBUILD_GAIN: f32 = 1.2;

/// Land floor for the O0 structured builder (elevation units ≈ 50 m): every demoted
/// target-land cell is uplifted to at least this above sea before the shaped excess is
/// distributed, so structured (asymmetric/segmented) uplift can't leave demoted-below-sea
/// cells submerged (no land loss). See erosion.rs structured builder.
pub const EMERGENT_LAND_FLOOR_MARGIN: f32 = 0.005;

// --- A4: two-stage drainage-aware uplift pulse (meso-a4-drainage-pulse.md) ---
// The uplift-shape channel's measured ceiling: the volume-conserving builder repays
// any meso-carved volume as global uplift, so valley depth bought via the shape
// arrives as peak height. A4 escapes the coupling by removing volume AFTER the
// height budget is set: a burn-in erosion epoch self-organizes drainage, the stable
// trunk network is extracted, and a ZERO-MEAN multiplicative uplift modifier (low
// along trunks, high on interfluves) drives a short frozen final epoch — valleys
// deepen by incision along the organized network, not by uplift deficit the
// normalizer refunds. ONE feedback pass only (continuous feedback locks into
// exaggerated spokes — GPT 5.6 consult §4.3).

/// Master A4 dial: depth of the drainage-aware uplift modulation. 0 = off (the
/// erosion pipeline is byte-identical to the single-pass path). At 1.0, trunk
/// cores get ×(1−TRUNK_SUPPRESS) uplift and far interfluves ×(1+INTERFLUVE_BOOST),
/// before per-orogen mean normalization.
pub const EROSION_DRAINAGE_PULSE: f32 = 0.0;
/// Burn-in epoch steps: enough for drainage TOPOLOGY to self-organize (not
/// landscape maturity — consult: ~60-100).
pub const EROSION_PULSE_BURNIN_STEPS: usize = 80;
/// Gaussian falloff sigma (km) of the trunk-proximity field. Keeps the modifier
/// in the 10-40 km meso band; nothing below ~8 km (mesh floor ~3.9 km cells).
pub const EROSION_PULSE_SMOOTH_KM: f32 = 15.0;
/// Minimum Strahler order (on the burn-in SFD channel network) for a cell to
/// count as a major trunk. Order ≥3 = the stable through-going drainage.
pub const EROSION_PULSE_TRUNK_ORDER: u32 = 3;
/// Uplift suppression on trunk cores at dial 1.0 (consult: ~0.5-0.7× on trunks).
pub const EROSION_PULSE_TRUNK_SUPPRESS: f32 = 0.4;
/// Uplift boost on far interfluves/massif cores at dial 1.0 (consult: ~1.1-1.3×).
pub const EROSION_PULSE_INTERFLUVE_BOOST: f32 = 0.25;
