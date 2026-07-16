//! Direct common evidence for the disposable 4 km H/C/G discriminator.
//!
//! This composes the existing S0, D0, and reference O0a builders in memory.
//! It is intentionally not a packet, campaign artifact, or promotion result.

use super::organization_owner::ThinG4KmObservationV0;
use super::organization_owner_c::ThinC4KmObservationV0;
use super::organization_owner_h::ThinH4KmObservationV0;
use super::{LinkedResolutionInputV0, LinkedSharedInputBundleV0};
use crate::world::landforms::{
    adapt_landscape_graph_v0, build_evaluation_drainage_v0, build_landform_relationships_v0,
    build_regular_hex_control_volumes_v0, build_surface_hierarchy_v0, relationship_graph_hash_v0,
    DrainageConfigV0, EvaluationDrainageV0, LandformRelationshipConfigV0, LandformRelationshipsV0,
    PacketGeometryIdentityV0, SurfaceHierarchyConfigV0, SurfaceHierarchyV0,
};
use crate::world::landscape::organization_artifact::OrganizationArmV0;
use serde::Serialize;
use std::fmt;

pub const THIN_HCG_COMMON_EVIDENCE_SCHEMA_V0: &str = "orogen-owner-thin-hcg-common-evidence-v0";
pub const THIN_HC_SIDECAR_COMMON_EVIDENCE_SCHEMA_V0: &str =
    "orogen-owner-thin-hc-sidecar-common-evidence-v0";
const TARGET_SPACING_KM: f64 = 4.0;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ThinArmCommonEvidenceV0 {
    pub arm: OrganizationArmV0,
    pub physical_elevation_component_hash: u64,
    pub surface_hierarchy: SurfaceHierarchyV0,
    pub drainage: EvaluationDrainageV0,
    pub relationships: LandformRelationshipsV0,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ThinHcgCommonEvidenceV0 {
    pub schema_version: String,
    pub warning: String,
    pub input_bundle_hash: u64,
    pub input_resolution_hash: u64,
    pub relationship_graph_hash: u64,
    pub h: ThinArmCommonEvidenceV0,
    pub c: ThinArmCommonEvidenceV0,
    pub g: ThinArmCommonEvidenceV0,
}

/// Common S0/D0/O0a evidence for two noncanonical terrain responses which
/// reuse the accepted geometry, initial conditions and runoff but replace the
/// registered forcing through an explicitly bound sidecar.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ThinHcSidecarCommonEvidenceV0 {
    pub schema_version: String,
    pub warning: String,
    pub input_bundle_hash: u64,
    pub input_resolution_hash: u64,
    pub forcing_binding_hash: u64,
    pub relationship_graph_hash: u64,
    pub h: ThinArmCommonEvidenceV0,
    pub c: ThinArmCommonEvidenceV0,
}

struct CommonEvidenceContextV0 {
    graph: crate::world::landforms::EvaluationSurfaceGraphV0,
    graph_hash: u64,
    scored_cell: Vec<bool>,
    surface_config: SurfaceHierarchyConfigV0,
    drainage_config: DrainageConfigV0,
    geometry_identity: PacketGeometryIdentityV0,
    relationship_config: LandformRelationshipConfigV0,
}

/// Build matched H/C common evidence without asserting the frozen HCG owner
/// identities. `forcing_binding_hash` is caller-owned and must bind the
/// sidecar compiler/result used by both response surfaces.
#[allow(clippy::too_many_arguments)]
pub fn build_thin_hc_sidecar_common_evidence_v0(
    bundle: &LinkedSharedInputBundleV0,
    forcing_binding_hash: u64,
    h_physical_elevation_component_hash: u64,
    h_elevation_km: &[f64],
    c_physical_elevation_component_hash: u64,
    c_elevation_km: &[f64],
) -> Result<ThinHcSidecarCommonEvidenceV0, ThinComparisonErrorV0> {
    let input = bundle
        .resolutions
        .iter()
        .find(|value| value.nominal_spacing_km.to_bits() == TARGET_SPACING_KM.to_bits())
        .ok_or_else(|| fail("accepted bundle has no exact 4 km resolution"))?;
    validate_sidecar_surface(input, OrganizationArmV0::H, h_elevation_km)?;
    validate_sidecar_surface(input, OrganizationArmV0::C, c_elevation_km)?;
    let context = common_evidence_context(input)?;
    let h = build_arm_evidence(
        OrganizationArmV0::H,
        h_physical_elevation_component_hash,
        h_elevation_km,
        input,
        &context.graph,
        &context.scored_cell,
        context.surface_config,
        context.drainage_config,
        context.geometry_identity,
        context.relationship_config,
    )?;
    let c = build_arm_evidence(
        OrganizationArmV0::C,
        c_physical_elevation_component_hash,
        c_elevation_km,
        input,
        &context.graph,
        &context.scored_cell,
        context.surface_config,
        context.drainage_config,
        context.geometry_identity,
        context.relationship_config,
    )?;
    Ok(ThinHcSidecarCommonEvidenceV0 {
        schema_version: THIN_HC_SIDECAR_COMMON_EVIDENCE_SCHEMA_V0.into(),
        warning: "DISPOSABLE SIDECAR COMMON EVIDENCE: not an accepted input, campaign packet or promotion result".into(),
        input_bundle_hash: bundle.derived_bundle_hash,
        input_resolution_hash: input.derived_resolution_hash,
        forcing_binding_hash,
        relationship_graph_hash: context.graph_hash,
        h,
        c,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThinComparisonErrorV0(pub String);

impl fmt::Display for ThinComparisonErrorV0 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ThinComparisonErrorV0 {}

pub fn build_thin_hcg_common_evidence_v0(
    bundle: &LinkedSharedInputBundleV0,
    h: &ThinH4KmObservationV0,
    c: &ThinC4KmObservationV0,
    g: &ThinG4KmObservationV0,
) -> Result<ThinHcgCommonEvidenceV0, ThinComparisonErrorV0> {
    let input = bundle
        .resolutions
        .iter()
        .find(|value| value.nominal_spacing_km.to_bits() == TARGET_SPACING_KM.to_bits())
        .ok_or_else(|| fail("accepted bundle has no exact 4 km resolution"))?;
    validate_owner_bindings(bundle, input, h, c, g)?;

    let context = common_evidence_context(input)?;

    let h_evidence = build_arm_evidence(
        OrganizationArmV0::H,
        h.final_elevation_component_hash,
        &h.final_elevation_km,
        input,
        &context.graph,
        &context.scored_cell,
        context.surface_config,
        context.drainage_config,
        context.geometry_identity,
        context.relationship_config,
    )?;
    let c_evidence = build_arm_evidence(
        OrganizationArmV0::C,
        c.final_elevation_component_hash,
        &c.final_elevation_km,
        input,
        &context.graph,
        &context.scored_cell,
        context.surface_config,
        context.drainage_config,
        context.geometry_identity,
        context.relationship_config,
    )?;
    let g_evidence = build_arm_evidence(
        OrganizationArmV0::G,
        g.final_elevation_component_hash,
        &g.final_elevation_km,
        input,
        &context.graph,
        &context.scored_cell,
        context.surface_config,
        context.drainage_config,
        context.geometry_identity,
        context.relationship_config,
    )?;

    Ok(ThinHcgCommonEvidenceV0 {
        schema_version: THIN_HCG_COMMON_EVIDENCE_SCHEMA_V0.into(),
        warning: "DISPOSABLE COMMON EVIDENCE: not a campaign packet or promotion result".into(),
        input_bundle_hash: bundle.derived_bundle_hash,
        input_resolution_hash: input.derived_resolution_hash,
        relationship_graph_hash: context.graph_hash,
        h: h_evidence,
        c: c_evidence,
        g: g_evidence,
    })
}

fn common_evidence_context(
    input: &LinkedResolutionInputV0,
) -> Result<CommonEvidenceContextV0, ThinComparisonErrorV0> {
    if input.nominal_spacing_km.to_bits() != TARGET_SPACING_KM.to_bits() {
        return Err(fail("common evidence requires exact 4 km geometry"));
    }
    let surface_config = SurfaceHierarchyConfigV0::default();
    let drainage_config = DrainageConfigV0::default();
    let relationship_config = LandformRelationshipConfigV0::default();
    let controls = build_regular_hex_control_volumes_v0(&input.mesh, &surface_config)
        .map_err(|error| fail(format!("control-volume construction failed: {error}")))?;
    let graph = adapt_landscape_graph_v0(&input.mesh, &controls, &surface_config)
        .map_err(|error| fail(format!("landscape graph adaptation failed: {error}")))?;
    let graph_hash = relationship_graph_hash_v0(&graph)
        .map_err(|error| fail(format!("relationship graph hash failed: {error}")))?;
    let geometry_identity = PacketGeometryIdentityV0::LandscapeRegularPlanar {
        nominal_spacing_km: TARGET_SPACING_KM,
        canonical_graph_hash: graph_hash,
    };
    let scored_cell = vec![true; graph.cell_count()];
    Ok(CommonEvidenceContextV0 {
        graph,
        graph_hash,
        scored_cell,
        surface_config,
        drainage_config,
        geometry_identity,
        relationship_config,
    })
}

fn validate_sidecar_surface(
    input: &LinkedResolutionInputV0,
    arm: OrganizationArmV0,
    elevation_km: &[f64],
) -> Result<(), ThinComparisonErrorV0> {
    if input.nominal_spacing_km.to_bits() != TARGET_SPACING_KM.to_bits()
        || elevation_km.len() != input.mesh.cell_count()
        || elevation_km.iter().any(|value| !value.is_finite())
    {
        return Err(fail(format!(
            "{arm:?} sidecar surface does not fit the shared finite 4 km geometry"
        )));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn build_arm_evidence(
    arm: OrganizationArmV0,
    physical_elevation_component_hash: u64,
    elevation_km: &[f64],
    input: &LinkedResolutionInputV0,
    graph: &crate::world::landforms::EvaluationSurfaceGraphV0,
    scored_cell: &[bool],
    surface_config: SurfaceHierarchyConfigV0,
    drainage_config: DrainageConfigV0,
    geometry_identity: PacketGeometryIdentityV0,
    relationship_config: LandformRelationshipConfigV0,
) -> Result<ThinArmCommonEvidenceV0, ThinComparisonErrorV0> {
    let hierarchy = build_surface_hierarchy_v0(graph, elevation_km, scored_cell, surface_config)
        .map_err(|error| fail(format!("{arm:?} S0 failed: {error}")))?;
    let drainage = build_evaluation_drainage_v0(
        graph,
        elevation_km,
        &input.local_runoff_supply_km3_myr,
        drainage_config,
    )
    .map_err(|error| fail(format!("{arm:?} D0 failed: {error}")))?;
    let relationships = build_landform_relationships_v0(
        graph,
        elevation_km,
        scored_cell,
        &input.local_runoff_supply_km3_myr,
        surface_config,
        drainage_config,
        &hierarchy,
        &drainage,
        geometry_identity,
        relationship_config,
    )
    .map_err(|error| fail(format!("{arm:?} O0a failed: {error}")))?;
    Ok(ThinArmCommonEvidenceV0 {
        arm,
        physical_elevation_component_hash,
        surface_hierarchy: hierarchy,
        drainage,
        relationships,
    })
}

fn validate_owner_bindings(
    bundle: &LinkedSharedInputBundleV0,
    input: &LinkedResolutionInputV0,
    h: &ThinH4KmObservationV0,
    c: &ThinC4KmObservationV0,
    g: &ThinG4KmObservationV0,
) -> Result<(), ThinComparisonErrorV0> {
    let n = input.mesh.cell_count();
    let expected_bundle = bundle.derived_bundle_hash;
    let expected_resolution = input.derived_resolution_hash;
    let bindings = [
        (
            OrganizationArmV0::H,
            &h.identity,
            h.final_elevation_km.len(),
        ),
        (
            OrganizationArmV0::C,
            &c.base_identity,
            c.final_elevation_km.len(),
        ),
        (
            OrganizationArmV0::G,
            &g.identity,
            g.final_elevation_km.len(),
        ),
    ];
    for (arm, identity, length) in bindings {
        if identity.arm != arm
            || identity.input_bundle_hash != expected_bundle
            || identity.input_resolution_hash != expected_resolution
            || identity.nominal_spacing_km.to_bits() != TARGET_SPACING_KM.to_bits()
            || length != n
        {
            return Err(fail(format!(
                "{arm:?} observation does not bind the shared 4 km input"
            )));
        }
    }
    for (arm, elevation) in [
        (OrganizationArmV0::H, h.final_elevation_km.as_slice()),
        (OrganizationArmV0::C, c.final_elevation_km.as_slice()),
        (OrganizationArmV0::G, g.final_elevation_km.as_slice()),
    ] {
        if elevation.iter().any(|value| !value.is_finite()) {
            return Err(fail(format!(
                "{arm:?} final surface contains a nonfinite value"
            )));
        }
    }
    Ok(())
}

fn fail(message: impl Into<String>) -> ThinComparisonErrorV0 {
    ThinComparisonErrorV0(message.into())
}
