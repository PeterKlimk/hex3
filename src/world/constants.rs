//! Constants for world generation and simulation.

/// Target fraction of surface area that should be continental.
pub const CONTINENTAL_FRACTION: f32 = 0.30;

/// Scale factor for boundary forcing used by tectonic features (arcs, trenches, ridges).
pub const FEATURE_FORCE_SCALE: f32 = 35.0;

// Screened diffusion solver parameters

/// Damping factor for Gauss-Seidel iteration (0 < ω ≤ 1). Lower = more stable.
pub const DIFFUSION_DAMPING: f32 = 0.8;

/// Maximum iterations for diffusion solver.
pub const DIFFUSION_MAX_ITERS: usize = 50;

/// Convergence tolerance for early termination.
pub const DIFFUSION_TOLERANCE: f32 = 0.001;

/// Base elevation for continental plate interiors.
/// Represents typical continental elevation (~500m above sea level).
pub const CONTINENTAL_BASE: f32 = 0.05;

/// Depth at mid-ocean ridge crests (young, hot oceanic crust).
/// Represents ~2500m below sea level.
pub const RIDGE_CREST_DEPTH: f32 = -0.25;

/// Depth of old oceanic crust far from ridges (abyssal plain).
/// Represents ~4500-5000m below sea level (thermally subsided).
pub const ABYSSAL_DEPTH: f32 = -0.45;

/// Characteristic distance for oceanic thermal subsidence (radians).
/// 1.5 rad ≈ 9550 km on Earth. Ocean floor deepens from ridge crest
/// to abyssal depth over roughly this distance (sqrt decay).
pub const THERMAL_SUBSIDENCE_WIDTH: f32 = 1.5;

/// Elevation at continent-ocean margin (continental shelf edge).
/// Both continental and oceanic crust meet at this depth, representing
/// the outer continental shelf (~200-400m below sea level on Earth).
/// This allows continental shelves to be submerged while oceanic crust
/// rises slightly toward the margin.
pub const MARGIN_DEPTH: f32 = -0.035;

/// Width of continental shelf transition (radians).
/// 0.04 rad ≈ 255 km on Earth. Controls how far inland the shelf/coast
/// transition extends before reaching full continental base elevation.
pub const CONTINENTAL_SHELF_WIDTH: f32 = 0.04;

/// Width of oceanic transition from margin to abyssal plain (radians).
/// 0.08 rad ≈ 510 km on Earth. Represents continental slope (~100 km)
/// plus continental rise (~400 km) - the gradual descent to abyssal depths.
pub const OCEANIC_TRANSITION_WIDTH: f32 = 0.08;

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

/// Cont+Cont: Himalayas-scale, highest mountains.
pub const CONV_CONT_CONT: f32 = 1.5;

/// Ocean+Ocean: Volcanic island arc.
/// Higher than before to enable island formation above deep ocean floor.
pub const CONV_OCEAN_OCEAN: f32 = 0.8;

/// Cont side of Cont+Ocean: Andes-style coastal mountains.
pub const CONV_CONT_OCEAN: f32 = 1.2;

/// Ocean side of Cont+Ocean: minimal - subducting plate goes down.
pub const CONV_OCEAN_CONT: f32 = 0.1;

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
/// Base amplitude for macro layer - PRIMARY vertical contributor.
pub const MACRO_AMPLITUDE: f32 = 0.12;
/// Frequency for macro layer (very low = large features).
pub const MACRO_FREQUENCY: f64 = 0.7;
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

// --- Ridge layer (mountain grain) ---
/// Base amplitude for ridge layer.
pub const RIDGE_AMPLITUDE: f32 = 0.14;
/// Frequency for ridge layer - HIGHER than hills for fine mountain grain.
pub const RIDGE_FREQUENCY: f64 = 6.0;
/// Octaves for ridge layer - keep LOW to avoid pocketing (2-3).
pub const RIDGE_OCTAVES: usize = 3;
/// Minimum ridge contribution (even in low-stress areas).
pub const RIDGE_MIN_FACTOR: f32 = 0.1;
/// Amplitude multiplier for oceanic plates (weaker offshore).
pub const RIDGE_OCEANIC_MULT: f32 = 0.15;
// Ridges amplified by stress: amp *= (RIDGE_MIN_FACTOR + (1-RIDGE_MIN_FACTOR) * stress_factor)

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

// Boundary-anchored elevation features (minimal bathymetry/orogeny model)
//
// These are applied as additive terms during elevation generation, using
// distance-to-boundary fields derived from plate kinematics.

/// Trench depth sensitivity (uses sqrt response of boundary forcing).
pub const TRENCH_SENSITIVITY: f32 = 0.06;
/// Maximum trench depth (positive magnitude; applied as negative elevation).
pub const TRENCH_MAX_DEPTH: f32 = 0.18;
/// Trench decay length from the boundary (radians).
/// 0.020 rad ≈ 127 km on Earth.
pub const TRENCH_DECAY: f32 = 0.020;

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
pub const ACTIVITY_DECAY_LENGTH: f32 = 0.05;

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

/// Weight of zonal (trade winds, westerlies) component in wind blend.
pub const ZONAL_WEIGHT: f32 = 0.6;

/// Weight of pressure-gradient component in wind blend.
pub const PRESSURE_WEIGHT: f32 = 0.4;

/// Strength multiplier for zonal wind patterns.
pub const ZONAL_STRENGTH: f32 = 0.3;

/// Surface wind Coriolis deflection angle (radians).
/// Surface wind deflects ~45° from geostrophic flow (not full 90°).
pub const SURFACE_CORIOLIS_ANGLE: f32 = 0.785; // 45 degrees

// --- Terrain effects (before projection) ---

/// How much terrain slope blocks uphill wind.
/// Higher values = steeper slopes block more wind.
pub const UPHILL_BLOCKING: f32 = 3.0;

/// Katabatic (downhill) wind acceleration strength.
/// Cold air drainage down slopes.
pub const KATABATIC_STRENGTH: f32 = 0.2;

// --- Projection solver ---

/// Terrain resistance for edge weights in projection.
/// Higher = mountains more impermeable to airflow routing.
/// weight = exp(-max_elev * TERRAIN_RESISTANCE)
pub const TERRAIN_RESISTANCE: f32 = 4.0;

/// Number of SOR iterations for projection solver.
pub const PROJECTION_ITERATIONS: usize = 50;

/// SOR relaxation factor (1.0-1.9, higher = faster but less stable).
pub const SOR_OMEGA: f32 = 1.5;
