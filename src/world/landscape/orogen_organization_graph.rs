//! Compiler-only probe for finite, inheritance-conditioned orogen forcing.
//!
//! This module deliberately cannot see terrain elevation, drainage, runoff, or
//! extracted landforms. It compares the accepted linked ribbon (B), the same
//! parents with a full finite-segment taper (F), and finite child patches
//! selected from independent plate-material inheritance (I). All three spend
//! the same declared rock-volume budget and emit displacement, never height.

use std::fmt;

use glam::DVec2;
use serde::{Deserialize, Serialize};

use super::{LandscapeMesh, SegmentId, SegmentLinkKind, SupportStencil};

pub const ORGANIZATION_GRAPH_SCHEMA_V0: &str = "orogen-organization-graph-compiler-v0";

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct OrganizationCompilerConfigV0 {
    pub inheritance_seed: u64,
    pub inheritance_lattice_km: f64,
    pub chain_sample_km: f64,
    pub local_maximum_radius_km: f64,
    pub suppression_radius_km: f64,
    pub maximum_children_per_parent: usize,
    pub minimum_child_length_km: f64,
    pub maximum_child_length_km: f64,
    pub vergence_shift_km: f64,
    pub profile_bin_km: f64,
}

impl Default for OrganizationCompilerConfigV0 {
    fn default() -> Self {
        Self {
            inheritance_seed: 0x6f72_6f67_656e_7630,
            inheritance_lattice_km: 96.0,
            chain_sample_km: 4.0,
            local_maximum_radius_km: 40.0,
            suppression_radius_km: 80.0,
            maximum_children_per_parent: 3,
            minimum_child_length_km: 50.0,
            maximum_child_length_km: 200.0,
            vergence_shift_km: 20.0,
            profile_bin_km: 8.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InheritanceModeV0 {
    CoherentLattice,
    HomogeneousAblation,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OrganizationSourceSegmentV0 {
    pub id: SegmentId,
    pub start_km: [f64; 2],
    pub end_km: [f64; 2],
    pub width_km: f64,
    /// Unit direction on the planar fixture. This is polarity/side semantics,
    /// not a rendering offset chosen from terrain.
    pub vergence_xy: [f64; 2],
    pub links: Vec<OrganizationSourceLinkV0>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct OrganizationSourceLinkV0 {
    pub other: SegmentId,
    pub kind: SegmentLinkKind,
}

/// Narrow compiler input. The absence of terrain and drainage fields is an
/// enforced causal boundary, not a caller convention.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OrganizationCompilerInputV0 {
    pub nominal_spacing_km: f64,
    pub mesh: LandscapeMesh,
    pub source_segments: Vec<OrganizationSourceSegmentV0>,
    pub baseline_stencils: Vec<SupportStencil>,
    pub parent_work_km3: Vec<ParentWorkV0>,
    pub total_work_km3: f64,
    pub source_bundle_hash: u64,
    pub source_resolution_hash: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ParentWorkV0 {
    pub parent_id: SegmentId,
    pub work_km3: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OrganizationGraphProbeV0 {
    pub schema_version: String,
    pub config: OrganizationCompilerConfigV0,
    pub inheritance_mode: InheritanceModeV0,
    pub source_bundle_hash: u64,
    pub source_resolution_hash: u64,
    pub graph: OrganizationGraphV0,
    pub inheritance: InheritanceRasterV0,
    pub baseline_displacement_km: Vec<f64>,
    pub finite_displacement_km: Vec<f64>,
    pub inherited_displacement_km: Option<Vec<f64>>,
    pub ledgers: Vec<OrganizationFieldLedgerV0>,
    pub parent_ledgers: Vec<OrganizationParentLedgerV0>,
    pub profiles: Vec<ParentLongitudinalProfileV0>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OrganizationGraphV0 {
    pub parents: Vec<OrganizationParentNodeV0>,
    pub children: Vec<OrganizationChildNodeV0>,
    pub edges: Vec<OrganizationGraphEdgeV0>,
    pub inherited_localization: bool,
    pub no_localization_reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OrganizationParentNodeV0 {
    pub id: SegmentId,
    pub length_km: f64,
    pub declared_work_km3: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OrganizationChildNodeV0 {
    pub id: u32,
    pub parent_id: SegmentId,
    pub ordinal: u32,
    pub interval_km: [f64; 2],
    pub nucleus_km: f64,
    pub nucleus_score: f64,
    pub mean_score: f64,
    pub work_share: f64,
    pub support_area_integral: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrganizationGraphEdgeKindV0 {
    ParentChild,
    Continuation,
    Transfer,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OrganizationGraphEdgeV0 {
    pub from: String,
    pub to: String,
    pub kind: OrganizationGraphEdgeKindV0,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InheritanceRasterV0 {
    pub weakness: Vec<f64>,
    pub fabric_axis_xy: Vec<[f64; 2]>,
    pub generator: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OrganizationFieldLedgerV0 {
    pub field: String,
    pub integrated_work_km3: Option<f64>,
    pub closure_error_km3: Option<f64>,
    pub maximum_displacement_km: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OrganizationParentLedgerV0 {
    pub parent_id: SegmentId,
    pub declared_work_km3: f64,
    pub baseline_work_km3: f64,
    pub finite_work_km3: f64,
    pub inherited_work_km3: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParentLongitudinalProfileV0 {
    pub parent_id: SegmentId,
    pub bin_edges_km: Vec<f64>,
    pub baseline_work_km3: Vec<f64>,
    pub finite_work_km3: Vec<f64>,
    pub inherited_work_km3: Option<Vec<f64>>,
    pub sample_arclength_km: Vec<f64>,
    pub sample_weakness: Vec<f64>,
    pub sample_fabric_alignment_sq: Vec<f64>,
    pub sample_localization_score: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrganizationCompilerErrorV0(pub String);

impl fmt::Display for OrganizationCompilerErrorV0 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for OrganizationCompilerErrorV0 {}

#[derive(Debug, Clone)]
struct Patch {
    parent_id: SegmentId,
    share: f64,
    stencil: Vec<f64>,
}

#[derive(Debug, Clone)]
struct ChainSamples {
    arclength_km: Vec<f64>,
    weakness: Vec<f64>,
    alignment_sq: Vec<f64>,
    score: Vec<f64>,
}

pub fn compile_organization_graph_v0(
    input: &OrganizationCompilerInputV0,
    config: OrganizationCompilerConfigV0,
    inheritance_mode: InheritanceModeV0,
) -> Result<OrganizationGraphProbeV0, OrganizationCompilerErrorV0> {
    validate_input(input, config)?;
    let mut parents = input.source_segments.clone();
    parents.sort_by_key(|segment| segment.id.0);

    let baseline_patches = baseline_patches(input, &parents)?;
    let finite_patches = finite_parent_patches(input, &parents)?;
    let baseline_displacement_km = combine_patches(
        input.mesh.cell_count(),
        &baseline_patches,
        input.total_work_km3,
    );
    let finite_displacement_km = combine_patches(
        input.mesh.cell_count(),
        &finite_patches,
        input.total_work_km3,
    );

    let inheritance = inheritance_raster(input, config, inheritance_mode);
    let mut samples = Vec::with_capacity(parents.len());
    for parent in &parents {
        samples.push(sample_chain(parent, config, inheritance_mode));
    }

    let (graph, inherited_patches) = if inheritance_mode == InheritanceModeV0::HomogeneousAblation {
        (
            graph_without_children(
                input,
                &parents,
                "homogeneous inheritance has no localization",
            ),
            None,
        )
    } else {
        let compiled = compile_inherited_children(input, &parents, &samples, config)?;
        (compiled.0, Some(compiled.1))
    };
    let inherited_displacement_km = inherited_patches
        .as_ref()
        .map(|patches| combine_patches(input.mesh.cell_count(), patches, input.total_work_km3));

    let ledgers = vec![
        ledger(
            "B: accepted ribbon",
            &baseline_displacement_km,
            &input.mesh.cell_area_km2,
            input.total_work_km3,
        ),
        ledger(
            "F: full-cosine finite parents",
            &finite_displacement_km,
            &input.mesh.cell_area_km2,
            input.total_work_km3,
        ),
        match &inherited_displacement_km {
            Some(field) => ledger(
                "I: inheritance-conditioned children",
                field,
                &input.mesh.cell_area_km2,
                input.total_work_km3,
            ),
            None => OrganizationFieldLedgerV0 {
                field: "I: no inherited localization".into(),
                integrated_work_km3: None,
                closure_error_km3: None,
                maximum_displacement_km: None,
            },
        },
    ];

    let profiles = build_profiles(
        input,
        &parents,
        &samples,
        &baseline_patches,
        &finite_patches,
        inherited_patches.as_deref(),
        config.profile_bin_km,
    );
    let parent_ledgers = profiles
        .iter()
        .map(|profile| OrganizationParentLedgerV0 {
            parent_id: profile.parent_id,
            declared_work_km3: parent_work(input, profile.parent_id).unwrap_or(f64::NAN),
            baseline_work_km3: profile.baseline_work_km3.iter().sum(),
            finite_work_km3: profile.finite_work_km3.iter().sum(),
            inherited_work_km3: profile
                .inherited_work_km3
                .as_ref()
                .map(|values| values.iter().sum()),
        })
        .collect();
    Ok(OrganizationGraphProbeV0 {
        schema_version: ORGANIZATION_GRAPH_SCHEMA_V0.into(),
        config,
        inheritance_mode,
        source_bundle_hash: input.source_bundle_hash,
        source_resolution_hash: input.source_resolution_hash,
        graph,
        inheritance,
        baseline_displacement_km,
        finite_displacement_km,
        inherited_displacement_km,
        ledgers,
        parent_ledgers,
        profiles,
    })
}

fn validate_input(
    input: &OrganizationCompilerInputV0,
    config: OrganizationCompilerConfigV0,
) -> Result<(), OrganizationCompilerErrorV0> {
    input
        .mesh
        .validate()
        .map_err(|error| OrganizationCompilerErrorV0(error.to_string()))?;
    if input.source_segments.is_empty()
        || input.baseline_stencils.len() != input.source_segments.len()
        || input.parent_work_km3.len() != input.source_segments.len()
        || !input.total_work_km3.is_finite()
        || input.total_work_km3 <= 0.0
    {
        return Err(error("invalid source segment/work cardinality"));
    }
    let positive = [
        config.inheritance_lattice_km,
        config.chain_sample_km,
        config.local_maximum_radius_km,
        config.suppression_radius_km,
        config.minimum_child_length_km,
        config.maximum_child_length_km,
        config.profile_bin_km,
    ];
    if positive
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
        || config.minimum_child_length_km >= config.maximum_child_length_km
        || config.maximum_children_per_parent == 0
        || !config.vergence_shift_km.is_finite()
    {
        return Err(error("invalid compiler configuration"));
    }
    let mut ids: Vec<_> = input
        .source_segments
        .iter()
        .map(|segment| segment.id.0)
        .collect();
    ids.sort_unstable();
    ids.dedup();
    if ids.len() != input.source_segments.len() {
        return Err(error("duplicate source segment id"));
    }
    for segment in &input.source_segments {
        let axis = vec2(segment.end_km) - vec2(segment.start_km);
        if !axis.is_finite() || axis.length() <= 0.0 || segment.width_km <= 0.0 {
            return Err(error("invalid source segment geometry"));
        }
        let stencil = input
            .baseline_stencils
            .iter()
            .find(|stencil| stencil.segment_id == segment.id)
            .ok_or_else(|| error("missing baseline stencil"))?;
        if stencil.weight_per_km2.len() != input.mesh.cell_count() {
            return Err(error("baseline stencil length mismatch"));
        }
        if stencil
            .weight_per_km2
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
        {
            return Err(error("baseline stencil has invalid weights"));
        }
        let stencil_integral: f64 = stencil
            .weight_per_km2
            .iter()
            .zip(&input.mesh.cell_area_km2)
            .map(|(weight, area)| weight * area)
            .sum();
        if !stencil_integral.is_finite() || (stencil_integral - 1.0).abs() > 1e-8 {
            return Err(error("baseline stencil is not area normalized"));
        }
        if input
            .parent_work_km3
            .iter()
            .find(|work| work.parent_id == segment.id)
            .is_none()
        {
            return Err(error("missing parent work"));
        }
    }
    if input
        .parent_work_km3
        .iter()
        .any(|work| !work.work_km3.is_finite() || work.work_km3 <= 0.0)
    {
        return Err(error("invalid parent work"));
    }
    let sum: f64 = input.parent_work_km3.iter().map(|work| work.work_km3).sum();
    if (sum - input.total_work_km3).abs() > 1e-8 * input.total_work_km3 {
        return Err(error("parent work does not close to total"));
    }
    Ok(())
}

fn baseline_patches(
    input: &OrganizationCompilerInputV0,
    parents: &[OrganizationSourceSegmentV0],
) -> Result<Vec<Patch>, OrganizationCompilerErrorV0> {
    parents
        .iter()
        .map(|parent| {
            let stencil = input
                .baseline_stencils
                .iter()
                .find(|stencil| stencil.segment_id == parent.id)
                .ok_or_else(|| error("missing baseline stencil"))?;
            Ok(Patch {
                parent_id: parent.id,
                share: parent_work(input, parent.id)? / input.total_work_km3,
                stencil: stencil.weight_per_km2.clone(),
            })
        })
        .collect()
}

fn finite_parent_patches(
    input: &OrganizationCompilerInputV0,
    parents: &[OrganizationSourceSegmentV0],
) -> Result<Vec<Patch>, OrganizationCompilerErrorV0> {
    parents
        .iter()
        .map(|parent| {
            let stencil = compile_stencil(
                &input.mesh,
                vec2(parent.start_km),
                vec2(parent.end_km),
                parent.width_km,
            )?;
            Ok(Patch {
                parent_id: parent.id,
                share: parent_work(input, parent.id)? / input.total_work_km3,
                stencil,
            })
        })
        .collect()
}

fn sample_chain(
    parent: &OrganizationSourceSegmentV0,
    config: OrganizationCompilerConfigV0,
    mode: InheritanceModeV0,
) -> ChainSamples {
    let start = vec2(parent.start_km);
    let axis = vec2(parent.end_km) - start;
    let length = axis.length();
    let unit = axis / length;
    let count = (length / config.chain_sample_km).ceil() as usize + 1;
    let mut result = ChainSamples {
        arclength_km: Vec::with_capacity(count),
        weakness: Vec::with_capacity(count),
        alignment_sq: Vec::with_capacity(count),
        score: Vec::with_capacity(count),
    };
    for index in 0..count {
        let s = (index as f64 * config.chain_sample_km).min(length);
        let point = start + unit * s;
        let (weakness, fabric) = inherited_value(point, config, mode, unit);
        let alignment_sq = fabric.dot(unit).powi(2).clamp(0.0, 1.0);
        let score = (0.25 + 0.75 * weakness) * (0.5 + 0.5 * alignment_sq);
        result.arclength_km.push(s);
        result.weakness.push(weakness);
        result.alignment_sq.push(alignment_sq);
        result.score.push(score);
    }
    result
}

fn inheritance_raster(
    input: &OrganizationCompilerInputV0,
    config: OrganizationCompilerConfigV0,
    mode: InheritanceModeV0,
) -> InheritanceRasterV0 {
    let mut weakness = Vec::with_capacity(input.mesh.cell_count());
    let mut fabric_axis_xy = Vec::with_capacity(input.mesh.cell_count());
    for center in &input.mesh.cell_center_km {
        let (w, f) = inherited_value(DVec2::new(center.x, center.y), config, mode, DVec2::X);
        weakness.push(w);
        fabric_axis_xy.push([f.x, f.y]);
    }
    InheritanceRasterV0 {
        weakness,
        fabric_axis_xy,
        generator: match mode {
            InheritanceModeV0::CoherentLattice => format!(
                "independent anchored {:.17} km SplitMix64 value lattice; seed {}; bilinear interpolation",
                config.inheritance_lattice_km, config.inheritance_seed
            ),
            InheritanceModeV0::HomogeneousAblation => "homogeneous inheritance ablation".into(),
        },
    }
}

fn inherited_value(
    point: DVec2,
    config: OrganizationCompilerConfigV0,
    mode: InheritanceModeV0,
    homogeneous_fabric: DVec2,
) -> (f64, DVec2) {
    if mode == InheritanceModeV0::HomogeneousAblation {
        return (0.5, homogeneous_fabric);
    }
    let q = point / config.inheritance_lattice_km;
    let ix = q.x.floor() as i64;
    let iy = q.y.floor() as i64;
    let tx = q.x - ix as f64;
    let ty = q.y - iy as f64;
    let mut weakness = 0.0;
    let mut doubled_axis = DVec2::ZERO;
    for dx in 0..=1 {
        for dy in 0..=1 {
            let weight = if dx == 0 { 1.0 - tx } else { tx } * if dy == 0 { 1.0 - ty } else { ty };
            let x = ix + dx;
            let y = iy + dy;
            let h0 = lattice_hash(config.inheritance_seed, x, y, 0);
            let h1 = lattice_hash(config.inheritance_seed, x, y, 1);
            weakness += weight * hash_unit(h0);
            let angle2 = std::f64::consts::TAU * hash_unit(h1);
            doubled_axis += weight * DVec2::new(angle2.cos(), angle2.sin());
        }
    }
    let angle = 0.5 * doubled_axis.y.atan2(doubled_axis.x);
    (
        weakness.clamp(0.0, 1.0),
        DVec2::new(angle.cos(), angle.sin()),
    )
}

fn compile_inherited_children(
    input: &OrganizationCompilerInputV0,
    parents: &[OrganizationSourceSegmentV0],
    samples: &[ChainSamples],
    config: OrganizationCompilerConfigV0,
) -> Result<(OrganizationGraphV0, Vec<Patch>), OrganizationCompilerErrorV0> {
    let mut children = Vec::new();
    let mut patch_specs = Vec::new();
    let mut next_id = 0_u32;
    for (parent, chain) in parents.iter().zip(samples) {
        let nuclei = select_nuclei(chain, config);
        if nuclei.is_empty() {
            return Err(error("coherent fixture produced no inherited localization"));
        }
        let boundaries = child_boundaries(chain, &nuclei);
        let start = vec2(parent.start_km);
        let axis = vec2(parent.end_km) - start;
        let length = axis.length();
        let unit = axis / length;
        let vergence = normalized_or_zero(vec2(parent.vergence_xy));
        for (ordinal, (&nucleus, assigned_interval)) in nuclei.iter().zip(boundaries).enumerate() {
            // A nearest-nucleus basin is only a candidate domain. A finite
            // structure may terminate inside it: intersect the basin with the
            // declared maximum envelope centred on its causal nucleus instead
            // of stretching a child to fill an otherwise inactive chain end.
            let interval = finite_interval(
                chain,
                assigned_interval,
                nucleus,
                config.maximum_child_length_km,
            );
            let a = chain.arclength_km[interval.0];
            let b = chain.arclength_km[interval.1];
            let child_length = b - a;
            if child_length < config.minimum_child_length_km
                || child_length > config.maximum_child_length_km
            {
                return Err(error(format!(
                    "manufactured parent {} child {ordinal} length {child_length:.3} km outside [{:.3}, {:.3}] km; nuclei={:?}, interval_indices={interval:?}",
                    parent.id.0,
                    config.minimum_child_length_km,
                    config.maximum_child_length_km,
                    nuclei
                )));
            }
            let mean_score = chain.score[interval.0..=interval.1].iter().sum::<f64>()
                / (interval.1 - interval.0 + 1) as f64;
            let shifted_start = start + unit * a + vergence * config.vergence_shift_km;
            let shifted_end = start + unit * b + vergence * config.vergence_shift_km;
            patch_specs.push((
                parent.id,
                next_id,
                ordinal as u32,
                shifted_start,
                shifted_end,
                a,
                b,
                chain.arclength_km[nucleus],
                chain.score[nucleus],
                mean_score,
                child_length * mean_score,
                parent.width_km,
            ));
            next_id += 1;
        }
    }
    let mut parent_raw_sums = Vec::with_capacity(parents.len());
    for parent in parents {
        let raw_sum: f64 = patch_specs
            .iter()
            .filter(|spec| spec.0 == parent.id)
            .map(|spec| spec.10)
            .sum();
        if !raw_sum.is_finite() || raw_sum <= 0.0 {
            return Err(error("invalid inherited child work scores"));
        }
        parent_raw_sums.push((parent.id, raw_sum));
    }
    let mut patches = Vec::with_capacity(patch_specs.len());
    for spec in patch_specs {
        let stencil = compile_stencil(&input.mesh, spec.3, spec.4, spec.11)?;
        let area_integral: f64 = stencil
            .iter()
            .zip(&input.mesh.cell_area_km2)
            .map(|(weight, area)| weight * area)
            .sum();
        let parent_raw_sum = parent_raw_sums
            .iter()
            .find(|(parent_id, _)| *parent_id == spec.0)
            .map(|(_, sum)| *sum)
            .ok_or_else(|| error("missing inherited parent score total"))?;
        let share = parent_work(input, spec.0)? / input.total_work_km3 * spec.10 / parent_raw_sum;
        children.push(OrganizationChildNodeV0 {
            id: spec.1,
            parent_id: spec.0,
            ordinal: spec.2,
            interval_km: [spec.5, spec.6],
            nucleus_km: spec.7,
            nucleus_score: spec.8,
            mean_score: spec.9,
            work_share: share,
            support_area_integral: area_integral,
        });
        patches.push(Patch {
            parent_id: spec.0,
            share,
            stencil,
        });
    }
    let mut graph = graph_without_children(input, parents, "");
    graph.inherited_localization = true;
    graph.no_localization_reason = None;
    graph.children = children;
    for child in &graph.children {
        graph.edges.push(OrganizationGraphEdgeV0 {
            from: format!("parent:{}", child.parent_id.0),
            to: format!("child:{}", child.id),
            kind: OrganizationGraphEdgeKindV0::ParentChild,
        });
    }
    for pair in graph.children.windows(2) {
        if pair[0].parent_id == pair[1].parent_id {
            graph.edges.push(OrganizationGraphEdgeV0 {
                from: format!("child:{}", pair[0].id),
                to: format!("child:{}", pair[1].id),
                kind: OrganizationGraphEdgeKindV0::Continuation,
            });
        }
    }
    Ok((graph, patches))
}

fn graph_without_children(
    input: &OrganizationCompilerInputV0,
    parents: &[OrganizationSourceSegmentV0],
    reason: &str,
) -> OrganizationGraphV0 {
    let mut edges = Vec::new();
    for parent in parents {
        for link in &parent.links {
            if parent.id.0 < link.other.0 && link.kind == SegmentLinkKind::Transfer {
                edges.push(OrganizationGraphEdgeV0 {
                    from: format!("parent:{}", parent.id.0),
                    to: format!("parent:{}", link.other.0),
                    kind: OrganizationGraphEdgeKindV0::Transfer,
                });
            }
        }
    }
    OrganizationGraphV0 {
        parents: parents
            .iter()
            .map(|parent| OrganizationParentNodeV0 {
                id: parent.id,
                length_km: (vec2(parent.end_km) - vec2(parent.start_km)).length(),
                declared_work_km3: parent_work(input, parent.id).unwrap_or(0.0),
            })
            .collect(),
        children: Vec::new(),
        edges,
        inherited_localization: false,
        no_localization_reason: Some(reason.into()),
    }
}

fn select_nuclei(chain: &ChainSamples, config: OrganizationCompilerConfigV0) -> Vec<usize> {
    let min = chain.score.iter().copied().fold(f64::INFINITY, f64::min);
    let max = chain
        .score
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    if max - min <= 1e-12 * max.abs().max(1.0) {
        return Vec::new();
    }
    let radius = (config.local_maximum_radius_km / config.chain_sample_km).ceil() as usize;
    let mut candidates = Vec::new();
    // A nucleus must own the complete declared comparison window. Treating a
    // one-sided endpoint sample as a local maximum would manufacture range-end
    // peaks from missing evidence rather than inherited localization.
    if chain.score.len() <= 2 * radius {
        return Vec::new();
    }
    for i in radius..chain.score.len() - radius {
        let lo = i - radius;
        let hi = i + radius;
        let is_maximum = (lo..=hi).all(|j| {
            chain.score[j] < chain.score[i]
                || (chain.score[j].to_bits() == chain.score[i].to_bits() && j >= i)
        });
        if is_maximum {
            candidates.push(i);
        }
    }
    candidates.sort_by(|&a, &b| {
        chain.score[b]
            .total_cmp(&chain.score[a])
            .then_with(|| a.cmp(&b))
    });
    let mut accepted: Vec<usize> = Vec::new();
    for candidate in candidates {
        if accepted.iter().all(|&other| {
            (chain.arclength_km[candidate] - chain.arclength_km[other]).abs()
                >= config.suppression_radius_km
        }) {
            accepted.push(candidate);
            if accepted.len() == config.maximum_children_per_parent {
                break;
            }
        }
    }
    accepted.sort_unstable();
    accepted
}

fn child_boundaries(chain: &ChainSamples, nuclei: &[usize]) -> Vec<(usize, usize)> {
    let mut cuts = Vec::new();
    for pair in nuclei.windows(2) {
        let mut minimum = pair[0] + 1;
        for i in pair[0] + 1..pair[1] {
            if chain.score[i].total_cmp(&chain.score[minimum]).is_lt() {
                minimum = i;
            }
        }
        cuts.push(minimum);
    }
    nuclei
        .iter()
        .enumerate()
        .map(|(index, _)| {
            let start = if index == 0 { 0 } else { cuts[index - 1] };
            let end = if index == nuclei.len() - 1 {
                chain.score.len() - 1
            } else {
                cuts[index]
            };
            (start, end)
        })
        .collect()
}

fn finite_interval(
    chain: &ChainSamples,
    assigned: (usize, usize),
    nucleus: usize,
    maximum_length_km: f64,
) -> (usize, usize) {
    let half = 0.5 * maximum_length_km;
    let nucleus_s = chain.arclength_km[nucleus];
    let lower = (nucleus_s - half).max(chain.arclength_km[assigned.0]);
    let upper = (nucleus_s + half).min(chain.arclength_km[assigned.1]);
    let start = (assigned.0..=assigned.1)
        .find(|&index| chain.arclength_km[index] >= lower)
        .unwrap_or(assigned.0);
    let end = (assigned.0..=assigned.1)
        .rev()
        .find(|&index| chain.arclength_km[index] <= upper)
        .unwrap_or(assigned.1);
    (start, end)
}

fn compile_stencil(
    mesh: &LandscapeMesh,
    start: DVec2,
    end: DVec2,
    width_km: f64,
) -> Result<Vec<f64>, OrganizationCompilerErrorV0> {
    let axis = end - start;
    let length = axis.length();
    let unit = axis / length;
    let mut weights = Vec::with_capacity(mesh.cell_count());
    for center in &mesh.cell_center_km {
        let p = DVec2::new(center.x, center.y) - start;
        let along = p.dot(unit) / length;
        let cross = (p - unit * p.dot(unit)).length();
        let cross_weight = if cross >= 0.5 * width_km {
            0.0
        } else {
            0.5 * (1.0 + (std::f64::consts::PI * cross / (0.5 * width_km)).cos())
        };
        let along_weight = if !(0.0..=1.0).contains(&along) {
            0.0
        } else {
            // `Taper::CosineEnds { end_fraction: 0.5 }`: finite-slip control
            // with no exactly flat interior.
            0.5 * (1.0 - (std::f64::consts::TAU * along).cos())
        };
        weights.push(cross_weight * along_weight);
    }
    let integral: f64 = weights
        .iter()
        .zip(&mesh.cell_area_km2)
        .map(|(weight, area)| weight * area)
        .sum();
    if !integral.is_finite() || integral <= 0.0 {
        return Err(error("organization patch has empty support"));
    }
    for weight in &mut weights {
        *weight /= integral;
    }
    Ok(weights)
}

fn combine_patches(cell_count: usize, patches: &[Patch], total_work_km3: f64) -> Vec<f64> {
    let mut field = vec![0.0; cell_count];
    for patch in patches {
        for (value, weight) in field.iter_mut().zip(&patch.stencil) {
            *value += total_work_km3 * patch.share * weight;
        }
    }
    field
}

fn build_profiles(
    input: &OrganizationCompilerInputV0,
    parents: &[OrganizationSourceSegmentV0],
    samples: &[ChainSamples],
    baseline: &[Patch],
    finite: &[Patch],
    inherited: Option<&[Patch]>,
    bin_km: f64,
) -> Vec<ParentLongitudinalProfileV0> {
    parents
        .iter()
        .zip(samples)
        .map(|(parent, chain)| {
            let length = (vec2(parent.end_km) - vec2(parent.start_km)).length();
            let bin_count = (length / bin_km).ceil() as usize;
            let bin_edges_km = (0..=bin_count)
                .map(|index| (index as f64 * bin_km).min(length))
                .collect::<Vec<_>>();
            ParentLongitudinalProfileV0 {
                parent_id: parent.id,
                baseline_work_km3: bin_patch_work(input, parent, baseline, bin_count, bin_km),
                finite_work_km3: bin_patch_work(input, parent, finite, bin_count, bin_km),
                inherited_work_km3: inherited
                    .map(|patches| bin_patch_work(input, parent, patches, bin_count, bin_km)),
                bin_edges_km,
                sample_arclength_km: chain.arclength_km.clone(),
                sample_weakness: chain.weakness.clone(),
                sample_fabric_alignment_sq: chain.alignment_sq.clone(),
                sample_localization_score: chain.score.clone(),
            }
        })
        .collect()
}

fn bin_patch_work(
    input: &OrganizationCompilerInputV0,
    parent: &OrganizationSourceSegmentV0,
    patches: &[Patch],
    bin_count: usize,
    bin_km: f64,
) -> Vec<f64> {
    let start = vec2(parent.start_km);
    let axis = vec2(parent.end_km) - start;
    let length = axis.length();
    let unit = axis / length;
    let mut bins = vec![0.0; bin_count];
    for patch in patches.iter().filter(|patch| patch.parent_id == parent.id) {
        for ((center, area), weight) in input
            .mesh
            .cell_center_km
            .iter()
            .zip(&input.mesh.cell_area_km2)
            .zip(&patch.stencil)
        {
            // Attribute every cell-volume contribution to its source parent.
            // Cross-width support and the polarity shift can project a cell a
            // few kilometres beyond the parent endpoint; clamping it to the
            // terminal bin preserves the exact 2-D parent ledger.
            let along = (DVec2::new(center.x, center.y) - start)
                .dot(unit)
                .clamp(0.0, length);
            let bin = ((along / bin_km).floor() as usize).min(bin_count - 1);
            bins[bin] += input.total_work_km3 * patch.share * weight * area;
        }
    }
    bins
}

fn ledger(name: &str, field: &[f64], area: &[f64], declared: f64) -> OrganizationFieldLedgerV0 {
    let actual: f64 = field
        .iter()
        .zip(area)
        .map(|(value, area)| value * area)
        .sum();
    OrganizationFieldLedgerV0 {
        field: name.into(),
        integrated_work_km3: Some(actual),
        closure_error_km3: Some(actual - declared),
        maximum_displacement_km: field.iter().copied().reduce(f64::max),
    }
}

fn parent_work(
    input: &OrganizationCompilerInputV0,
    id: SegmentId,
) -> Result<f64, OrganizationCompilerErrorV0> {
    input
        .parent_work_km3
        .iter()
        .find(|work| work.parent_id == id)
        .map(|work| work.work_km3)
        .ok_or_else(|| error("missing parent work"))
}

fn vec2(value: [f64; 2]) -> DVec2 {
    DVec2::new(value[0], value[1])
}

fn normalized_or_zero(value: DVec2) -> DVec2 {
    if value.is_finite() && value.length_squared() > 0.0 {
        value.normalize()
    } else {
        DVec2::ZERO
    }
}

fn lattice_hash(seed: u64, x: i64, y: i64, stream: u64) -> u64 {
    let mixed = seed
        ^ (x as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ (y as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9)
        ^ stream.wrapping_mul(0x94d0_49bb_1331_11eb);
    splitmix64(mixed)
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn hash_unit(hash: u64) -> f64 {
    (hash >> 11) as f64 / (1_u64 << 53) as f64
}

fn error(message: impl Into<String>) -> OrganizationCompilerErrorV0 {
    OrganizationCompilerErrorV0(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::landscape::{
        build_linked_shared_input_bundle_v0, LINKED_INPUT_BUNDLE_HASH_V0,
    };
    use std::sync::OnceLock;

    fn accepted_input() -> &'static OrganizationCompilerInputV0 {
        static INPUT: OnceLock<OrganizationCompilerInputV0> = OnceLock::new();
        INPUT.get_or_init(build_accepted_input)
    }

    fn build_accepted_input() -> OrganizationCompilerInputV0 {
        let bundle = build_linked_shared_input_bundle_v0().unwrap();
        let resolution = bundle
            .resolutions
            .iter()
            .find(|value| value.nominal_spacing_km.to_bits() == 4.0_f64.to_bits())
            .unwrap();
        let source_segments = bundle
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
            .collect();
        let parent_work_km3 = bundle
            .declaration
            .work_ledgers
            .iter()
            .filter_map(|ledger| {
                ledger.segment_id.map(|parent_id| ParentWorkV0 {
                    parent_id,
                    work_km3: ledger.positive_rock_volume_km3,
                })
            })
            .collect();
        OrganizationCompilerInputV0 {
            nominal_spacing_km: 4.0,
            mesh: resolution.mesh.clone(),
            source_segments,
            baseline_stencils: resolution.compiled_stencils.clone(),
            parent_work_km3,
            total_work_km3: bundle.declaration.analytic_rock_volume_km3,
            source_bundle_hash: bundle.derived_bundle_hash,
            source_resolution_hash: resolution.derived_resolution_hash,
        }
    }

    #[test]
    fn b_f_and_i_close_the_same_work() {
        let input = accepted_input();
        let probe = compile_organization_graph_v0(
            input,
            OrganizationCompilerConfigV0::default(),
            InheritanceModeV0::CoherentLattice,
        )
        .unwrap();
        for ledger in &probe.ledgers {
            assert!(ledger.closure_error_km3.unwrap().abs() < 1e-8);
        }
        let profile_b: f64 = probe
            .profiles
            .iter()
            .flat_map(|profile| &profile.baseline_work_km3)
            .sum();
        let profile_f: f64 = probe
            .profiles
            .iter()
            .flat_map(|profile| &profile.finite_work_km3)
            .sum();
        let profile_i: f64 = probe
            .profiles
            .iter()
            .flat_map(|profile| profile.inherited_work_km3.as_ref().unwrap())
            .sum();
        for profile_total in [profile_b, profile_f, profile_i] {
            assert!((profile_total - input.total_work_km3).abs() < 1e-8);
        }
        for ledger in &probe.parent_ledgers {
            for actual in [
                ledger.baseline_work_km3,
                ledger.finite_work_km3,
                ledger.inherited_work_km3.unwrap(),
            ] {
                assert!((actual - ledger.declared_work_km3).abs() < 1e-8);
            }
        }
        assert!(probe.graph.inherited_localization);
        assert!(!probe.graph.children.is_empty());
    }

    #[test]
    fn finite_control_has_no_flat_analytic_interior() {
        let values: Vec<_> = (1..100)
            .map(|index| 0.5 * (1.0 - (std::f64::consts::TAU * index as f64 / 100.0).cos()))
            .collect();
        assert!(values
            .windows(2)
            .all(|pair| pair[0].to_bits() != pair[1].to_bits()));
    }

    #[test]
    fn inherited_compilation_is_deterministic() {
        let input = accepted_input();
        let first = compile_organization_graph_v0(
            input,
            OrganizationCompilerConfigV0::default(),
            InheritanceModeV0::CoherentLattice,
        )
        .unwrap();
        let second = compile_organization_graph_v0(
            input,
            OrganizationCompilerConfigV0::default(),
            InheritanceModeV0::CoherentLattice,
        )
        .unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn homogeneous_inheritance_explicitly_refuses_segmentation() {
        let input = accepted_input();
        let probe = compile_organization_graph_v0(
            input,
            OrganizationCompilerConfigV0::default(),
            InheritanceModeV0::HomogeneousAblation,
        )
        .unwrap();
        assert!(!probe.graph.inherited_localization);
        assert!(probe.graph.children.is_empty());
        assert!(probe.inherited_displacement_km.is_none());
        assert!(probe.graph.no_localization_reason.is_some());
    }

    #[test]
    fn accepted_input_hash_is_unchanged() {
        assert_eq!(
            build_linked_shared_input_bundle_v0()
                .unwrap()
                .derived_bundle_hash,
            LINKED_INPUT_BUNDLE_HASH_V0
        );
    }

    #[test]
    fn invalid_public_input_is_rejected() {
        let mut invalid_work = accepted_input().clone();
        invalid_work.parent_work_km3[0].work_km3 = f64::NAN;
        assert!(compile_organization_graph_v0(
            &invalid_work,
            OrganizationCompilerConfigV0::default(),
            InheritanceModeV0::CoherentLattice,
        )
        .is_err());

        let mut invalid_stencil = accepted_input().clone();
        invalid_stencil.baseline_stencils[0].weight_per_km2[0] = -1.0;
        assert!(compile_organization_graph_v0(
            &invalid_stencil,
            OrganizationCompilerConfigV0::default(),
            InheritanceModeV0::CoherentLattice,
        )
        .is_err());
    }
}
