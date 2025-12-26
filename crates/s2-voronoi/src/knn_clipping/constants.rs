//! Shared constants for kNN clipping Voronoi construction and validation.

/// Treat generators as coincident when their dot product differs from 1 by this amount.
///
/// For unit vectors, `1 - dot ≈ (distance^2) / 2`. This threshold is derived from
/// f32 rounding on normalized inputs, with a small safety multiplier.
/// This is only for duplicate-generator handling; it does not address vertex degeneracy.
pub const COINCIDENT_DOT_TOL: f32 = 64.0 * f32::EPSILON * f32::EPSILON;

/// Squared Euclidean distance threshold for coincident generators.
#[inline]
pub const fn coincident_distance_sq() -> f32 {
    2.0 * COINCIDENT_DOT_TOL
}

/// Euclidean distance threshold for coincident generators.
#[inline]
pub fn coincident_distance() -> f32 {
    coincident_distance_sq().sqrt()
}

// No other shared epsilons remain; support thresholds are derived in the builder.
