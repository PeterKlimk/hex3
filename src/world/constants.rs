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

/// Reference thickness of undisturbed continental crust (definition scale).
pub const CRUST_THICKNESS_CONTINENTAL: f32 = 1.0;

/// Reference thickness of oceanic crust.
pub const CRUST_THICKNESS_OCEANIC: f32 = 0.25;

/// Elevation of reference continental crust (~500m above sea level).
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
/// Represents ~4500-5000m below sea level (thermally subsided).
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

// Sqrt-based elevation response parameters (continental crust)

/// Scale factor for compression → mountain height (continental).
pub const CONT_COMPRESSION_SENS: f32 = 0.4;

/// Scale factor for tension → rift depth (continental).
pub const CONT_TENSION_SENS: f32 = 0.3;

/// Maximum mountain height from compression (continental).
pub const CONT_MAX_MOUNTAIN: f32 = 0.8;

/// Maximum rift depth from tension (continental).
pub const CONT_MAX_RIFT: f32 = 0.2;

// Oceanic crust parameters

/// Scale factor for stress → oceanic uplift.
pub const OCEAN_SENSITIVITY: f32 = 0.12;

/// Maximum oceanic uplift from compression (volcanic edifice - can create islands).
pub const OCEAN_COMPRESSION_MAX: f32 = 0.25;

/// Maximum oceanic uplift from tension (isostatic limit - stays underwater).
pub const OCEAN_TENSION_MAX: f32 = 0.12;

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

/// Ocean side of Cont+Ocean: thermal uplift near margin.
pub const DIV_OCEAN_CONT: f32 = 0.3;

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

// Multi-layer elevation noise system
//
// Four layers with different scales and purposes:
// - Macro: continental-scale tilt (very smooth, large features) - PRIMARY vertical contributor
// - Hills: regional rolling terrain (medium scale) - secondary character
// - Ridges: mountain grain/spines (RidgedMulti, high freq) - ruggedness in high-stress areas
// - Micro: surface texture (fine detail, cosmetic only)
//
// Design principles:
// - Macro > Hills amplitude (broad plateaus/basins, not just regional bumpiness)
// - Ridges freq > Hills freq (ridges are finer grain than regional hills)
// - Ridges octaves low (2-3) to avoid "crinkly everywhere" pocketing
// - Micro very small, mostly cosmetic

// --- Stress modulation ---
/// Lower stress threshold for regime weighting.
/// Used with `STRESS_HIGH_THRESHOLD` in a smoothstep to map stress → 0..1.
pub const STRESS_LOW_THRESHOLD: f32 = 0.05;
/// Upper stress threshold for regime weighting smoothstep.
pub const STRESS_HIGH_THRESHOLD: f32 = 0.4;

// --- Macro layer (continental tilt) ---
/// Frequency for macro layer (very low = large features).
pub const MACRO_FREQUENCY: f64 = 1.1;
/// Octaves for macro layer (few = smooth).
pub const MACRO_OCTAVES: usize = 2;
/// Amplitude multiplier for oceanic plates (flatter ocean floor).
pub const MACRO_OCEANIC_MULT: f32 = 0.5;

// --- Hills layer (regional terrain) ---
/// Base amplitude for hills layer - secondary to macro.
pub const HILLS_AMPLITUDE: f32 = 0.07;
/// Frequency for hills layer.
pub const HILLS_FREQUENCY: f64 = 3.0;
/// Octaves for hills layer.
pub const HILLS_OCTAVES: usize = 3;
/// Amplitude multiplier for oceanic plates.
pub const HILLS_OCEANIC_MULT: f32 = 0.2;
// Hills are suppressed in active compressional orogens (see TerrainNoise::sample).
/// Downward bias applied to continental hills in extensional regimes.
/// Helps suggest rift basins/grabens in bedrock before erosion/sedimentation.
pub const HILLS_EXT_BIAS: f32 = 0.25;

// --- Ridge layer (mountain detail noise) ---
// Simple 3D noise, biased upward, modulated by convergence.

/// Base amplitude for ridge layer.
pub const RIDGE_AMPLITUDE: f32 = 0.2;
/// Octaves for ridge noise.
pub const RIDGE_OCTAVES: usize = 2;
/// Base frequency for ridge noise.
pub const RIDGE_FREQUENCY: f64 = 6.0;
/// Amplitude multiplier for oceanic plates (weaker offshore).
pub const RIDGE_OCEANIC_MULT: f32 = 0.15;

// --- Micro layer (surface texture) ---
/// Base amplitude for micro layer - cosmetic only.
/// Note: For unified shader path, see MICRO_AMPLITUDE in unified.wgsl
pub const MICRO_AMPLITUDE: f32 = 0.02;
/// Frequency for micro layer (high = fine detail).
pub const MICRO_FREQUENCY: f64 = 16.0;
/// Octaves for micro layer.
pub const MICRO_OCTAVES: usize = 2;
/// Amplitude multiplier for underwater areas.
pub const MICRO_UNDERWATER_MULT: f32 = 0.8;

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
/// 0.045 rad ≈ 287 km on Earth (large-end: 200-350+ km inland).
pub const ARC_CONT_PEAK_DIST: f32 = 0.05;
pub const ARC_OCEAN_PEAK_DIST: f32 = 0.04;

/// Arc band width (radians).
/// 0.060 rad ≈ 382 km on Earth. Wider band = more cells in the arc belt.
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
pub const TRANSFORM_INFLUENCE_LENGTH: f32 = 0.05;

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

/// Number of SOR iterations for projection solver.
pub const PROJECTION_ITERATIONS: usize = 50;

/// SOR relaxation factor (1.0-1.9, higher = faster but less stable).
pub const SOR_OMEGA: f32 = 1.0;

// Moisture & precipitation.

/// Number of moisture advection iterations (steady-state relaxation).
pub const MOISTURE_ITERATIONS: usize = 80;

/// Iterations at the end of the run to average precipitation over.
pub const MOISTURE_AVG_WINDOW: usize = 20;

/// Fraction of a water cell's capacity deficit replenished per iteration.
pub const EVAPORATION_RATE: f32 = 0.5;

/// Moisture carrying capacity of cold air (temperature 0).
pub const MOISTURE_CAP_COLD: f32 = 0.35;

/// Moisture carrying capacity of warm air (temperature 1).
pub const MOISTURE_CAP_WARM: f32 = 1.0;

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
/// carries little erosion detail, so it sits here.
pub const FINE_PLAINS_CELL_KM: f32 = 12.0;

/// Ocean-floor cell size, in km. Featureless seabed needs almost nothing.
pub const FINE_OCEAN_CELL_KM: f32 = 60.0;

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
/// steepest ground and main channels. 2.0 = square, 3.0 = cube.
pub const FINE_DENSITY_FEATURE_EXPONENT: f32 = 3.0;

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
/// unconditionally stable, so a few dozen large steps suffice to relax toward
/// the U/K equilibrium; more steps = closer to graded profiles, linear cost.
/// The dominant per-step cost is re-routing (priority flood), so this is also
/// the main runtime knob.
pub const EROSION_STEPS: usize = 30;

/// Timestep (dimensionless; folded into K / uplift scale). Kept explicit so the
/// Braun & Willett formula reads as written. Larger dt relaxes faster per step.
pub const EROSION_DT: f32 = 1.0;

/// Stream-power area exponent m in E = K A^m S^n (n = 1 here). 0.5 is the
/// canonical value; raising it makes large rivers dominate incision.
pub const EROSION_M: f32 = 0.5;

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

/// Jacobi sweeps used to approximate the implicit diffusion solve each step.
/// More = closer to the exact backward-Euler smoothing; 8 is ample for the
/// gentle rounding hillslope diffusion provides (it need not fully converge).
pub const EROSION_DIFFUSION_ITERS: usize = 6;

/// Re-route (priority-flood + drainage-area accumulation) every N incision
/// steps rather than every step. Routing dominates the loop cost (~75%), and
/// the drainage network is quasi-static over a few incision steps: channels
/// deepen along fixed paths, and the incision slope-guard safely skips any cell
/// whose stale receiver is no longer downhill. The network still reorganizes at
/// each re-route. 1 = re-route every step (exact, slowest). Higher = faster,
/// staler network between re-routes.
pub const EROSION_REROUTE_INTERVAL: usize = 6;

/// Scales tectonic uplift added to crust thickness each step. Source is the
/// transferred feature forcing: (arc + collision) are elevation magnitudes
/// (converted to thickness by dividing the Airy slope) and rift_delta is
/// already a signed thickness delta. U_thickness = SCALE * ((arc+collision)/slope
/// + rift_delta).
///
/// MUST be 0 for the one-shot stage-4 erosion pass. The coarse elevation already
/// encodes the orogen (collision/arc -> crust thickness -> mountain height at
/// stage 1), so re-injecting tectonic uplift here DOUBLE-COUNTS it: measured at
/// 0.02 the uplift injected ~99x the volume incision removes, inflating mountains
/// ~+26% in bulk / +36% at the peaks instead of eroding them. Stage 4 is
/// erosional RELAXATION (incision + hillslope diffusion carving the existing
/// orogen toward graded profiles). The uplift mechanism (`u_thick` in
/// `erosion.rs`) stays wired for a FUTURE coupled tectonics<->erosion
/// time-evolution loop (the A'2 thread in docs/ideas.md), where it belongs and
/// must be balanced against K; set it >0 only there.
pub const EROSION_UPLIFT_SCALE: f32 = 0.0;

/// Fraction of a coastal sink's depth-to-sea-level that routed sediment may fill
/// (delta building at river mouths). Sediment beyond this cap is "lost to the
/// ocean" for the mass-balance log. Keeps deposition from breaching sea level.
pub const EROSION_DEPOSIT_FILL_FRACTION: f32 = 0.5;

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
pub const EROSION_CHANNEL_SUPPORT_KM2: f32 = 30.0;
