//! Run one H and one C response to the work-matched full-cosine F forcing.

use std::fs::{self, File};
use std::io::BufWriter;
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use hex3::world::landscape::organization_comparison::build_thin_hc_sidecar_common_evidence_v0;
use hex3::world::landscape::organization_output::{
    build_thin_hc_sidecar_numerical_output_v0, thin_c_numerical_work_v0, thin_h_numerical_work_v0,
    ThinHcSidecarNumericalOutputV0,
};
use hex3::world::landscape::organization_owner_c::{
    derive_thin_c_experimental_forcing_binding_4km_v0, run_thin_c_experimental_forcing_4km_v0,
    ThinC4KmObservationV0, ThinCExperimentalForcingBindingV0,
};
use hex3::world::landscape::organization_owner_h::{
    derive_thin_h_experimental_forcing_binding_4km_v0, run_thin_h_experimental_forcing_4km_v0,
    ThinH4KmObservationV0, ThinHExperimentalForcingBindingV0,
};
use hex3::world::landscape::organization_render::{
    write_thin_three_surface_diagnostic_png_v0, ThinHcgDiagnosticConfigV0,
    ThinHcgDiagnosticMetadataV0,
};
use hex3::world::landscape::orogen_organization_graph::{
    compile_organization_graph_v0, InheritanceModeV0, OrganizationCompilerConfigV0,
    OrganizationCompilerInputV0, OrganizationSourceLinkV0, OrganizationSourceSegmentV0,
    ParentWorkV0,
};
use hex3::world::landscape::{
    build_linked_shared_input_bundle_v0, DeformationEvaluator, LinkedResolutionInputV0,
    LinkedSharedInputBundleV0, SegmentId, Taper,
};
use serde::Serialize;

const FILE_SCHEMA_V0: &str = "orogen-finite-forcing-hc-response-file-v0";
const SURFACE_SCHEMA_V0: &str = "orogen-finite-forcing-hc-surfaces-v0";

#[derive(Debug, Parser)]
#[command(about = "Run one H and one C response to the 4 km full-cosine F forcing")]
struct Cli {
    /// New directory receiving comparison.json, diagnostic.png and surfaces.bin.
    #[arg(long)]
    output_dir: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct FForcingBindingV0 {
    accepted_bundle_hash: u64,
    accepted_resolution_hash: u64,
    organization_probe_hash: u64,
    scenario_hash: u64,
    compiled_stencils_hash: u64,
    evaluator_chronology_hash: u64,
    cumulative_displacement_hash: u64,
    synthetic_input_bundle_hash: u64,
    synthetic_input_resolution_hash: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct ParentWorkLedgerV0 {
    parent_id: SegmentId,
    declared_work_km3: f64,
    compiled_work_km3: f64,
    closure_error_km3: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct FForcingAuditV0 {
    total_declared_work_km3: f64,
    total_compiled_work_km3: f64,
    total_closure_error_km3: f64,
    parent_ledgers: Vec<ParentWorkLedgerV0>,
    maximum_displacement_km: f64,
    maximum_abs_difference_from_bfi_f_km: f64,
    area_weighted_rms_difference_from_bfi_f_km: f64,
    full_cosine_parent_count: u64,
    horizontal_velocity_policy: &'static str,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct ResidualSummaryV0 {
    definition: String,
    area_weighted_mean_km: f64,
    area_weighted_rms_km: f64,
    maximum_abs_km: f64,
    signed_volume_km3: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct TimingsV0 {
    linked_input_seconds: f64,
    f_compiler_seconds: f64,
    h_seconds: f64,
    c_seconds: f64,
    common_evidence_seconds: f64,
    numerical_summary_seconds: f64,
    render_seconds: f64,
    total_seconds: f64,
}

#[derive(Debug, Serialize)]
struct ComparisonFileV0<'a> {
    schema_version: &'static str,
    warning: &'static str,
    source_revision: &'static str,
    source_dirty: bool,
    forcing: &'a FForcingBindingV0,
    forcing_audit: &'a FForcingAuditV0,
    timings: TimingsV0,
    render: ThinHcgDiagnosticMetadataV0,
    target_minus_h: ResidualSummaryV0,
    target_minus_c: ResidualSummaryV0,
    h_minus_c: ResidualSummaryV0,
    relationship_graph_hash: u64,
    numerical: ThinHcSidecarNumericalOutputV0,
    h: &'a ThinH4KmObservationV0,
    c: &'a ThinC4KmObservationV0,
}

#[derive(Serialize)]
struct SurfaceCacheV0<'a> {
    schema_version: &'static str,
    forcing: &'a FForcingBindingV0,
    target_elevation_km: &'a [f64],
    h_elevation_km: &'a [f64],
    c_elevation_km: &'a [f64],
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    fs::create_dir(&cli.output_dir)?;
    let total_started = Instant::now();

    let started = Instant::now();
    let bundle = build_linked_shared_input_bundle_v0()?;
    let input = exact_4km_input(&bundle)?;
    let linked_input_seconds = started.elapsed().as_secs_f64();

    let started = Instant::now();
    let (evaluator, displacement, forcing, forcing_audit) = compile_f_forcing(&bundle, input)?;
    let f_compiler_seconds = started.elapsed().as_secs_f64();

    let started = Instant::now();
    let h = run_thin_h_experimental_forcing_4km_v0(
        &bundle,
        &displacement,
        ThinHExperimentalForcingBindingV0 {
            synthetic_input_bundle_hash: forcing.synthetic_input_bundle_hash,
            synthetic_input_resolution_hash: forcing.synthetic_input_resolution_hash,
            cumulative_displacement_component_hash: forcing.cumulative_displacement_hash,
        },
    )?;
    let h_seconds = started.elapsed().as_secs_f64();

    let started = Instant::now();
    let c = run_thin_c_experimental_forcing_4km_v0(
        &bundle,
        &evaluator,
        &displacement,
        ThinCExperimentalForcingBindingV0 {
            synthetic_input_bundle_hash: forcing.synthetic_input_bundle_hash,
            synthetic_input_resolution_hash: forcing.synthetic_input_resolution_hash,
            compiled_stencils_component_hash: forcing.compiled_stencils_hash,
            frame_witnesses_component_hash: forcing.evaluator_chronology_hash,
            cumulative_displacement_component_hash: forcing.cumulative_displacement_hash,
        },
    )?;
    let c_seconds = started.elapsed().as_secs_f64();

    let target_elevation_km = input
        .initial_elevation_km
        .iter()
        .zip(&displacement)
        .map(|(initial, uplift)| initial + uplift)
        .collect::<Vec<_>>();
    let started = Instant::now();
    let evidence = build_thin_hc_sidecar_common_evidence_v0(
        &bundle,
        forcing.synthetic_input_bundle_hash,
        h.final_elevation_component_hash,
        &h.final_elevation_km,
        c.final_elevation_component_hash,
        &c.final_elevation_km,
    )?;
    let common_evidence_seconds = started.elapsed().as_secs_f64();

    let started = Instant::now();
    let numerical = build_thin_hc_sidecar_numerical_output_v0(
        &bundle,
        &evidence,
        h.final_elevation_component_hash,
        &h.final_elevation_km,
        thin_h_numerical_work_v0(&h),
        c.final_elevation_component_hash,
        &c.final_elevation_km,
        thin_c_numerical_work_v0(&c),
    )?;
    let numerical_summary_seconds = started.elapsed().as_secs_f64();

    let forcing_support = displacement
        .iter()
        .map(|value| *value > 0.0)
        .collect::<Vec<_>>();
    let profile_forcing = evaluator
        .support_stencils()
        .iter()
        .map(|stencil| stencil.weight_per_km2.as_slice())
        .collect::<Vec<_>>();
    let started = Instant::now();
    let render = write_thin_three_surface_diagnostic_png_v0(
        cli.output_dir.join("diagnostic.png"),
        "Full-cosine F forcing: target, H response, C response",
        &input.mesh,
        &forcing_support,
        &displacement,
        &profile_forcing,
        [
            &target_elevation_km,
            &h.final_elevation_km,
            &c.final_elevation_km,
        ],
        ["F target", "H(F)", "C(F)"],
        ThinHcgDiagnosticConfigV0::default(),
    )?;
    let render_seconds = started.elapsed().as_secs_f64();

    let surface_cache = SurfaceCacheV0 {
        schema_version: SURFACE_SCHEMA_V0,
        forcing: &forcing,
        target_elevation_km: &target_elevation_km,
        h_elevation_km: &h.final_elevation_km,
        c_elevation_km: &c.final_elevation_km,
    };
    fs::write(
        cli.output_dir.join("surfaces.bin"),
        bincode::serialize(&surface_cache)?,
    )?;
    let timings = TimingsV0 {
        linked_input_seconds,
        f_compiler_seconds,
        h_seconds,
        c_seconds,
        common_evidence_seconds,
        numerical_summary_seconds,
        render_seconds,
        total_seconds: total_started.elapsed().as_secs_f64(),
    };
    let output = ComparisonFileV0 {
        schema_version: FILE_SCHEMA_V0,
        warning:
            "DISPOSABLE F RESPONSE COMPARISON: not an accepted input, campaign or promotion result",
        source_revision: env!("HEX3_GIT_REVISION"),
        source_dirty: env!("HEX3_GIT_DIRTY") == "true",
        forcing: &forcing,
        forcing_audit: &forcing_audit,
        timings,
        render,
        target_minus_h: residual_summary(
            "(z0 + F) - H(F)",
            &target_elevation_km,
            &h.final_elevation_km,
            &input.mesh.cell_area_km2,
        )?,
        target_minus_c: residual_summary(
            "(z0 + F) - C(F)",
            &target_elevation_km,
            &c.final_elevation_km,
            &input.mesh.cell_area_km2,
        )?,
        h_minus_c: residual_summary(
            "H(F) - C(F)",
            &h.final_elevation_km,
            &c.final_elevation_km,
            &input.mesh.cell_area_km2,
        )?,
        relationship_graph_hash: evidence.relationship_graph_hash,
        numerical,
        h: &h,
        c: &c,
    };
    serde_json::to_writer_pretty(
        BufWriter::new(File::create(cli.output_dir.join("comparison.json"))?),
        &output,
    )?;
    Ok(())
}

#[allow(clippy::type_complexity)]
fn compile_f_forcing(
    bundle: &LinkedSharedInputBundleV0,
    input: &LinkedResolutionInputV0,
) -> Result<
    (
        DeformationEvaluator,
        Vec<f64>,
        FForcingBindingV0,
        FForcingAuditV0,
    ),
    Box<dyn std::error::Error>,
> {
    let mut scenario = bundle.declaration.scenario.clone();
    for segment in &mut scenario.segments {
        segment.along_strike_taper = Taper::CosineEnds { end_fraction: 0.5 };
    }
    scenario.id = "F-full-cosine-finite-parents-v0".into();
    let evaluator = scenario.compile(&input.mesh)?;
    let parent_work = parent_work(bundle);
    let mut displacement = vec![0.0; input.mesh.cell_count()];
    let mut parent_ledgers = Vec::new();
    for stencil in evaluator.support_stencils() {
        let declared = parent_work
            .iter()
            .find(|work| work.parent_id == stencil.segment_id)
            .ok_or("F stencil has no declared parent work")?
            .work_km3;
        let mut compiled = 0.0;
        for ((value, weight), area) in displacement
            .iter_mut()
            .zip(&stencil.weight_per_km2)
            .zip(&input.mesh.cell_area_km2)
        {
            let depth = declared * weight;
            *value += depth;
            compiled += depth * area;
        }
        parent_ledgers.push(ParentWorkLedgerV0 {
            parent_id: stencil.segment_id,
            declared_work_km3: declared,
            compiled_work_km3: compiled,
            closure_error_km3: compiled - declared,
        });
    }
    let total_compiled: f64 = displacement
        .iter()
        .zip(&input.mesh.cell_area_km2)
        .map(|(depth, area)| depth * area)
        .sum();
    let organization_input = organization_input(bundle, input, parent_work.clone());
    let probe = compile_organization_graph_v0(
        &organization_input,
        OrganizationCompilerConfigV0::default(),
        InheritanceModeV0::CoherentLattice,
    )?;
    let (max_difference, rms_difference) = difference_metrics(
        &displacement,
        &probe.finite_displacement_km,
        &input.mesh.cell_area_km2,
    )?;
    if max_difference > 1e-12 {
        return Err(format!("scenario F and BFI F differ by {max_difference:.17} km").into());
    }
    let organization_probe_hash = fnv1a64(&bincode::serialize(&probe)?);
    let scenario_hash = hash_value("orogen-f-scenario-v0", &scenario)?;
    let h_binding = derive_thin_h_experimental_forcing_binding_4km_v0(bundle, &displacement)?;
    let c_binding =
        derive_thin_c_experimental_forcing_binding_4km_v0(bundle, &evaluator, &displacement)?;
    if h_binding.synthetic_input_bundle_hash != c_binding.synthetic_input_bundle_hash
        || h_binding.synthetic_input_resolution_hash != c_binding.synthetic_input_resolution_hash
        || h_binding.cumulative_displacement_component_hash
            != c_binding.cumulative_displacement_component_hash
    {
        return Err("H and C derived different F input identities".into());
    }
    let forcing = FForcingBindingV0 {
        accepted_bundle_hash: bundle.derived_bundle_hash,
        accepted_resolution_hash: input.derived_resolution_hash,
        organization_probe_hash,
        scenario_hash,
        compiled_stencils_hash: c_binding.compiled_stencils_component_hash,
        evaluator_chronology_hash: c_binding.frame_witnesses_component_hash,
        cumulative_displacement_hash: c_binding.cumulative_displacement_component_hash,
        synthetic_input_bundle_hash: h_binding.synthetic_input_bundle_hash,
        synthetic_input_resolution_hash: h_binding.synthetic_input_resolution_hash,
    };
    let audit = FForcingAuditV0 {
        total_declared_work_km3: bundle.declaration.analytic_rock_volume_km3,
        total_compiled_work_km3: total_compiled,
        total_closure_error_km3: total_compiled - bundle.declaration.analytic_rock_volume_km3,
        parent_ledgers,
        maximum_displacement_km: displacement.iter().copied().fold(0.0, f64::max),
        maximum_abs_difference_from_bfi_f_km: max_difference,
        area_weighted_rms_difference_from_bfi_f_km: rms_difference,
        full_cosine_parent_count: scenario.segments.len() as u64,
        horizontal_velocity_policy: "exact positive zero everywhere",
    };
    Ok((evaluator, displacement, forcing, audit))
}

fn organization_input(
    bundle: &LinkedSharedInputBundleV0,
    input: &LinkedResolutionInputV0,
    parent_work_km3: Vec<ParentWorkV0>,
) -> OrganizationCompilerInputV0 {
    OrganizationCompilerInputV0 {
        nominal_spacing_km: input.nominal_spacing_km,
        mesh: input.mesh.clone(),
        source_segments: bundle
            .declaration
            .scenario
            .segments
            .iter()
            .map(|segment| OrganizationSourceSegmentV0 {
                id: segment.id,
                start_km: [segment.geometry.start_km.x, segment.geometry.start_km.y],
                end_km: [segment.geometry.end_km.x, segment.geometry.end_km.y],
                width_km: segment.width_km,
                vergence_xy: [segment.vergence.x as f64, segment.vergence.y as f64],
                links: segment
                    .links
                    .iter()
                    .map(|link| OrganizationSourceLinkV0 {
                        other: link.other,
                        kind: link.kind,
                    })
                    .collect(),
            })
            .collect(),
        baseline_stencils: input.compiled_stencils.clone(),
        parent_work_km3,
        total_work_km3: bundle.declaration.analytic_rock_volume_km3,
        source_bundle_hash: bundle.derived_bundle_hash,
        source_resolution_hash: input.derived_resolution_hash,
    }
}

fn parent_work(bundle: &LinkedSharedInputBundleV0) -> Vec<ParentWorkV0> {
    bundle
        .declaration
        .work_ledgers
        .iter()
        .filter_map(|ledger| {
            ledger.segment_id.map(|parent_id| ParentWorkV0 {
                parent_id,
                work_km3: ledger.positive_rock_volume_km3,
            })
        })
        .collect()
}

fn exact_4km_input(
    bundle: &LinkedSharedInputBundleV0,
) -> Result<&LinkedResolutionInputV0, Box<dyn std::error::Error>> {
    bundle
        .resolutions
        .iter()
        .find(|value| value.nominal_spacing_km.to_bits() == 4.0_f64.to_bits())
        .ok_or_else(|| "accepted bundle has no exact 4 km input".into())
}

fn residual_summary(
    definition: &str,
    lhs: &[f64],
    rhs: &[f64],
    area: &[f64],
) -> Result<ResidualSummaryV0, Box<dyn std::error::Error>> {
    if lhs.len() != rhs.len() || lhs.len() != area.len() || lhs.is_empty() {
        return Err("residual arrays have unequal or empty shapes".into());
    }
    let total_area: f64 = area.iter().sum();
    let mut signed_volume = 0.0;
    let mut squares = 0.0;
    let mut maximum: f64 = 0.0;
    for ((left, right), cell_area) in lhs.iter().zip(rhs).zip(area) {
        let difference = left - right;
        if !difference.is_finite() || !cell_area.is_finite() || *cell_area <= 0.0 {
            return Err("residual arrays contain invalid values".into());
        }
        signed_volume += difference * cell_area;
        squares += difference * difference * cell_area;
        maximum = maximum.max(difference.abs());
    }
    Ok(ResidualSummaryV0 {
        definition: definition.into(),
        area_weighted_mean_km: signed_volume / total_area,
        area_weighted_rms_km: (squares / total_area).sqrt(),
        maximum_abs_km: maximum,
        signed_volume_km3: signed_volume,
    })
}

fn difference_metrics(
    lhs: &[f64],
    rhs: &[f64],
    area: &[f64],
) -> Result<(f64, f64), Box<dyn std::error::Error>> {
    let summary = residual_summary("F scenario - BFI F", lhs, rhs, area)?;
    Ok((summary.maximum_abs_km, summary.area_weighted_rms_km))
}

fn hash_value<T: Serialize>(domain: &str, value: &T) -> Result<u64, bincode::Error> {
    Ok(fnv1a64(&bincode::serialize(&(domain, value))?))
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}
