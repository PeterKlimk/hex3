//! Shared constants for GPU-style Voronoi construction and validation.

/// Minimum distance between generator and neighbor for a valid bisector.
/// Points closer than this are merged before Voronoi computation to prevent:
/// 1. Numerically unstable bisector planes
/// 2. Multiple close neighbors collectively killing cells
/// 3. Orphan edges from inconsistent cell topologies
pub const MIN_BISECTOR_DISTANCE: f32 = 1e-5;

/// Fraction of mean generator spacing used for near-duplicate thresholds in tests.
pub const VERTEX_WELD_FRACTION: f32 = 0.01;

/// Support-set epsilon in dot space (absolute).
///
/// This defines the ambiguity zone for support set membership:
/// a generator C is in the support set if dot(V,G) - dot(V,C) ≤ SUPPORT_EPS_ABS.
///
/// For f64 vertex computation with ~2e-15 position error:
/// - Gap error ≈ 2e-15 × |G-C| ≤ 4e-15
/// - Two cells computing "same" vertex may get gaps differing by ~8e-15
///
/// We use 1e-12 for ~100x safety margin over f64 arithmetic error.
pub const SUPPORT_EPS_ABS: f64 = 1e-12;

/// Additional margin when certifying a bounded support cluster.
///
/// This is intentionally non-zero to avoid razor-thin pass/fail boundaries when
/// comparing near-ties in dot space under accumulated numeric error.
pub const SUPPORT_CERT_MARGIN_ABS: f64 = SUPPORT_EPS_ABS;

/// Cluster radius for ambiguity certification (radians).
pub const SUPPORT_CLUSTER_RADIUS_ANGLE: f64 = 1e-7;
/// Approximate angular error scale from f32 inputs, used for adaptive cluster bounds.
pub const SUPPORT_VERTEX_ANGLE_EPS: f64 = f32::EPSILON as f64 * 8.0;

#[inline]
pub fn support_cluster_drift_dot() -> f64 {
    2.0 * (SUPPORT_CLUSTER_RADIUS_ANGLE * 0.5).sin()
}

// Epsilon values for numerical stability (f32 units).
pub(crate) const EPS_PLANE_CONTAINS: f32 = 1e-7;
pub(crate) const EPS_TERMINATION_MARGIN: f32 = 1e-7;
