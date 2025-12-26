#![feature(portable_simd)]

//! Spherical Voronoi diagrams on the unit sphere (S2).
//!
//! This crate computes Voronoi diagrams for points on the unit sphere using
//! a kNN-driven half-space clipping algorithm.
//!
//! # Example
//!
//! ```
//! use s2_voronoi::{compute, UnitVec3};
//!
//! // Generate some points on the unit sphere
//! let points = vec![
//!     UnitVec3::new(1.0, 0.0, 0.0),
//!     UnitVec3::new(0.0, 1.0, 0.0),
//!     UnitVec3::new(0.0, 0.0, 1.0),
//!     UnitVec3::new(-1.0, 0.0, 0.0),
//!     UnitVec3::new(0.0, -1.0, 0.0),
//!     UnitVec3::new(0.0, 0.0, -1.0),
//! ];
//!
//! let output = compute(&points).expect("computation should succeed");
//! assert_eq!(output.diagram.num_cells(), 6);
//! ```

mod diagram;
mod error;
mod types;

// Internal modules
pub(crate) mod cube_grid;
pub(crate) mod knn_clipping;

pub use diagram::{CellView, SphericalVoronoi, VoronoiCell};
pub use error::VoronoiError;
pub use types::{UnitVec3, UnitVec3Like};

/// Output from Voronoi computation, including diagram and diagnostics.
#[derive(Debug, Clone)]
pub struct VoronoiOutput {
    /// The computed Voronoi diagram.
    pub diagram: SphericalVoronoi,
    /// Diagnostic information about the computation.
    pub diagnostics: VoronoiDiagnostics,
}

/// Diagnostic information from Voronoi computation.
///
/// In the current implementation, some cells may be degenerate or invalid.
/// Once precision fallbacks are implemented, these should always be empty.
#[derive(Debug, Clone, Default)]
pub struct VoronoiDiagnostics {
    /// Cell indices with fewer than 3 vertices (invalid for rendering/adjacency).
    pub bad_cells: Vec<usize>,
    /// Cell indices with duplicate vertex indices (degenerate polygons).
    pub degenerate_cells: Vec<usize>,
}

impl VoronoiDiagnostics {
    /// Returns true if no issues were detected.
    pub fn is_clean(&self) -> bool {
        self.bad_cells.is_empty() && self.degenerate_cells.is_empty()
    }
}

/// Configuration for Voronoi computation.
#[derive(Debug, Clone)]
pub struct VoronoiConfig {
    // Future: termination config, knn schedule, preprocessing options
    _private: (),
}

impl Default for VoronoiConfig {
    fn default() -> Self {
        Self { _private: () }
    }
}

/// Compute a spherical Voronoi diagram with default settings.
///
/// Returns a diagram plus diagnostics. Errors are reserved for invalid inputs
/// (e.g., insufficient points) or unrecoverable internal failures.
pub fn compute<P: UnitVec3Like>(points: &[P]) -> Result<VoronoiOutput, VoronoiError> {
    compute_with(points, VoronoiConfig::default())
}

/// Compute a spherical Voronoi diagram with explicit configuration.
pub fn compute_with<P: UnitVec3Like>(
    points: &[P],
    _config: VoronoiConfig,
) -> Result<VoronoiOutput, VoronoiError> {
    use glam::Vec3;

    if points.len() < 4 {
        return Err(VoronoiError::InsufficientPoints(points.len()));
    }

    // Convert input points to Vec3 for the backend
    let vec3_points: Vec<Vec3> = points
        .iter()
        .map(|p| Vec3::new(p.x(), p.y(), p.z()))
        .collect();

    // Call knn_clipping backend
    let diagram = knn_clipping::compute_voronoi_gpu_style(&vec3_points);

    // Collect diagnostics
    let mut diagnostics = VoronoiDiagnostics::default();
    for (i, cell) in diagram.iter_cells().enumerate() {
        if cell.len() < 3 {
            diagnostics.bad_cells.push(i);
        }
        // Check for duplicate vertex indices
        let indices = cell.vertex_indices;
        if indices.len() > 1 {
            let mut sorted: Vec<u32> = indices.to_vec();
            sorted.sort_unstable();
            for w in sorted.windows(2) {
                if w[0] == w[1] {
                    diagnostics.degenerate_cells.push(i);
                    break;
                }
            }
        }
    }

    Ok(VoronoiOutput {
        diagram,
        diagnostics,
    })
}
