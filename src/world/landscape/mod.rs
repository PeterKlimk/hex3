//! Experimental, dimensioned landscape-evolution testbed.
//!
//! This module is deliberately independent of [`super::World`]. Tectonic
//! scenarios produce rock-velocity fields; only the landscape solver may turn
//! those velocities into authoritative elevation.

mod boundary_fixture;
mod c0_solver;
mod continuum;
mod denudation;
mod filter;
mod forcing;
mod gradient;
mod hillslope;
mod mesh;
mod solver;

pub use boundary_fixture::{linear_diffusive_boundary_flux_km3_myr, BoundaryFluxFixtureError};
pub use c0_solver::{
    C0ElevationVolumeMomentLedger, C0LandscapeError, C0LandscapeParams, C0LandscapeSolver,
    C0LandscapeState, C0OperatorLimits, C0StepDiagnostics, C0TimestepLimiter, C0WaterDiagnostics,
};
pub use continuum::{ContinuumFlowError, DepressionRoutingSurface, FaceFlowCache, FlowPartition};
pub use denudation::{
    apply_effective_areal_denudation, EffectiveArealDenudationError,
    EffectiveArealDenudationParams, EffectiveArealDenudationResult,
};
pub use filter::{
    apply_scalar_helmholtz_filter, HelmholtzBoundaryMode, HelmholtzFilterAudit,
    HelmholtzFilterError, HelmholtzFilterParams, HelmholtzFilterResult,
};
pub use forcing::{
    linked_scenario, uniform_scenario, DeformationEpisode, DeformationEvaluator, DeformationFrame,
    DeformationSegment, EpisodeId, ForcingError, LandscapeScenario, SegmentGeometry, SegmentId,
    SegmentLink, SegmentLinkKind, SegmentShare, SupportStencil, Taper,
    REFERENCE_ROCK_VOLUME_RATE_KM3_MYR,
};
pub use gradient::{
    flow_aligned_physical_grade, reconstruct_mean_surface_gradient, FlowAlignedGradeError,
    MeanSurfaceGradient, SurfaceGradientError,
};
pub use hillslope::{
    apply_conservative_hillslope_step, ConservativeHillslopeError, ConservativeHillslopeParams,
    ConservativeHillslopeStep, PortalSolidTransfer,
};
pub use mesh::{
    BoundaryCondition, BoundaryFaceCondition, BoundarySide, LandscapeBoundaryFace, LandscapeMesh,
    LandscapeMeshError, OutletPortal, OutletPortalId,
};
pub use solver::{
    DrainageCache, LandscapeError, LandscapeLedger, LandscapeParams, LandscapeSnapshot,
    LandscapeSolver, LandscapeState, StepDiagnostics, TimestepAudit, TimestepLimiter,
};
