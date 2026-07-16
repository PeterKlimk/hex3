//! Stable, CPU-only spatial evidence packet for planet autopsies.
//!
//! The packet selects a deliberately small set of semantic objects and retains
//! enough spatial and causal context to drive diagnostics and matched renders.
//! It does not classify the whole planet or make a realism judgment.

use serde::Serialize;

use super::diagnostics::{measure_components, ComponentStats};
use super::{
    elevation_to_km, ClimatologyNullReport, Elevation, RiverMouth, RiverNetwork,
    RiverThresholdPolicy, RunManifest, SemanticWaterKind, Tessellation, WaterBodySemantics,
    WaterGeographyReport, World, PLANET_RADIUS_KM,
};

pub const DOSSIER_SCHEMA_VERSION: u32 = 3;
const MOUNTAIN_ELEVATION_KM: f32 = 1.5;
const SIGNIFICANT_MOUNTAIN_AREA_KM2: f32 = 20_000.0;
const TARGET_LIMIT: usize = 3;
const GEOMETRY_SAMPLE_LIMIT: usize = 64;
const RIVER_SAMPLE_LIMIT: usize = 128;

#[derive(Debug, Serialize)]
pub struct DossierPacket {
    pub schema_version: u32,
    pub manifest: RunManifest,
    pub selection_contract: SelectionContract,
    pub mountains: Vec<MountainTarget>,
    pub lakes: Vec<LakeTarget>,
    pub rivers: Vec<RiverTarget>,
    pub water_geography: WaterGeographyReport,
    pub climatology_null: ClimatologyNullReport,
    pub hydrology_provenance: HydrologyProvenance,
    pub caveats: Vec<&'static str>,
    pub future_gaps: Vec<&'static str>,
}

#[derive(Debug, Serialize)]
pub struct SelectionContract {
    pub target_limit_per_kind: usize,
    pub mountain_threshold_km: f32,
    pub significant_mountain_area_km2: f32,
    pub river_policy: RiverThresholdPolicy,
    pub geometry_sample_limit: usize,
    pub coordinates: &'static str,
    pub focus_radius_unit: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct SpatialAnchor {
    pub cell: usize,
    pub latitude_deg: f32,
    pub longitude_deg: f32,
    pub unit_xyz: [f32; 3],
    pub elevation_km: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct ClimateAtAnchor {
    pub temperature_normalized: f32,
    pub precipitation_relative_to_global_mean: f32,
}

#[derive(Debug, Serialize)]
pub struct TectonicAssociation {
    pub dominant_plate: u32,
    pub collision_cell_fraction: f32,
    pub arc_cell_fraction: f32,
    pub mean_collision_response: f32,
    pub mean_arc_response: f32,
    pub mean_convergent_influence: f32,
    pub mean_coarse_tectonic_activity: f32,
    pub mean_tectonic_thickening: f32,
    pub mean_tectonic_uplift_rate: f32,
}

#[derive(Debug, Serialize)]
pub struct HydrologyProvenance {
    pub integration_cut_cell_count: usize,
    pub maximum_integration_cut_depth_km: f32,
}

#[derive(Debug, Serialize)]
pub struct MountainTarget {
    pub rank: usize,
    pub selection_role: MountainSelectionRole,
    pub anchor_peak: SpatialAnchor,
    pub focus_radius_km: f32,
    pub area_km2: f32,
    pub extent_km: f32,
    pub mean_width_km: f32,
    pub elongation: f32,
    pub cell_count: usize,
    pub geometry_sample: Vec<SpatialAnchor>,
    pub pre_erosion_peak_elevation_km: f32,
    pub final_peak_elevation_km: f32,
    pub tectonics: TectonicAssociation,
    pub climate: ClimateAtAnchor,
    pub drainage_integration_cut_cells: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum MountainSelectionRole {
    HighestPeak,
    LargestArea,
    Broadest,
}

#[derive(Debug, Serialize)]
pub struct LakeBasinData {
    pub basin_id: usize,
    pub basin_cell_count: usize,
    pub catchment_discharge_equivalent_km2: f32,
    pub bottom_elevation_km: f32,
    pub spill_elevation_km: f32,
    pub water_level_km: f32,
    pub mean_temperature_normalized: f32,
    pub evaporation_factor: f32,
    pub overflowing: bool,
}

#[derive(Debug, Serialize)]
pub struct LakeTarget {
    pub rank: usize,
    pub selection_role: LakeSelectionRole,
    pub semantic_anchor: SpatialAnchor,
    pub focus_radius_km: f32,
    pub area_km2: f32,
    pub surface_elevation_km: f32,
    pub max_depth_km: f32,
    pub outlet: super::WaterOutlet,
    pub basin: LakeBasinData,
    pub geometry_sample: Vec<SpatialAnchor>,
    pub climate: ClimateAtAnchor,
    pub drainage_integration_cut_cells: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum LakeSelectionRole {
    LargestArea,
    LargestTerminal,
    HighestSurface,
}

#[derive(Debug, Serialize)]
pub struct RiverTarget {
    pub rank: usize,
    pub selection_role: RiverSelectionRole,
    pub mouth: SpatialAnchor,
    pub head: SpatialAnchor,
    pub focus_radius_km: f32,
    pub mouth_kind: RiverMouth,
    pub catchment_discharge_equivalent_km2: f32,
    pub strahler_order_at_mouth: u8,
    pub is_major: bool,
    pub trunk_cell_count: usize,
    pub trunk_length_km_approx: f32,
    pub trunk_geometry: Vec<SpatialAnchor>,
    pub mouth_climate: ClimateAtAnchor,
    pub head_climate: ClimateAtAnchor,
    pub trunk_integration_cut_cells: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum RiverSelectionRole {
    HighestDischarge,
    LongestTrunk,
    HighestStrahlerOrder,
}

impl DossierPacket {
    /// Build a deterministic object packet from the currently active fine
    /// surface. Stage 3 and stage 4 are both supported and identified by the
    /// manifest; callers normally use the retained stage-4 product surface.
    pub fn build(world: &World) -> Result<Self, String> {
        let fine = world
            .fine
            .as_ref()
            .ok_or_else(|| "dossier requires a generated fine stage (stage 3 or 4)".to_string())?;
        let tess = fine.tessellation();
        let surface = fine.surface_for(world.view_stage());
        let elevation = &surface.elevation.values;
        let hydrology = &surface.hydrology;
        if elevation.len() != tess.num_cells() {
            return Err("active fine elevation/tessellation length mismatch".into());
        }

        let water = WaterBodySemantics::build(tess, hydrology);
        let river_policy = RiverThresholdPolicy::default();
        let network = RiverNetwork::build(tess, hydrology, &water, river_policy);
        let water_geography = WaterGeographyReport::build(tess, hydrology, &water, &network)?;
        let pre_hydrology_values: Vec<f32> = (0..tess.num_cells())
            .map(|cell| hydrology.pre_integration_elevation(cell))
            .collect();
        let pre_hydrology_elevation = Elevation::refine_from_base(tess, &pre_hydrology_values);
        let climatology_null = ClimatologyNullReport::build(
            tess,
            &pre_hydrology_elevation,
            &fine.base.fields.elevation_fields.continentality,
            &surface.temperature,
            &surface.precipitation,
            hydrology,
            &network,
            &water_geography,
            river_policy,
        )?;
        let mountains = build_mountains(
            world,
            tess,
            elevation,
            &surface.temperature,
            &surface.precipitation,
        );
        let lakes = build_lakes(
            tess,
            elevation,
            &surface.temperature,
            &surface.precipitation,
            hydrology,
            &water,
        );
        let rivers = build_rivers(
            tess,
            elevation,
            &surface.temperature,
            &surface.precipitation,
            hydrology,
            &water,
            &network,
        );

        Ok(Self {
            schema_version: DOSSIER_SCHEMA_VERSION,
            manifest: world.manifest(),
            selection_contract: SelectionContract {
                target_limit_per_kind: TARGET_LIMIT,
                mountain_threshold_km: MOUNTAIN_ELEVATION_KM,
                significant_mountain_area_km2: SIGNIFICANT_MOUNTAIN_AREA_KM2,
                river_policy,
                geometry_sample_limit: GEOMETRY_SAMPLE_LIMIT,
                coordinates: "latitude/longitude degrees plus unit-sphere xyz",
                focus_radius_unit: "km on the modeled planet surface",
            },
            mountains,
            lakes,
            rivers,
            water_geography,
            climatology_null,
            hydrology_provenance: HydrologyProvenance {
                integration_cut_cell_count: hydrology.integration_cut_count(),
                maximum_integration_cut_depth_km: hydrology
                    .integration_cuts()
                    .map(|(cell, _, _)| elevation_to_km(hydrology.integration_cut_depth(cell)))
                    .fold(0.0, f32::max),
            },
            caveats: vec![
                "Targets are deterministic diagnostic selections, not stable geographic names.",
                "Mountain components use an absolute 1.5 km mask and therefore conflate orogens with elevated interiors.",
                "Catchment area is precipitation-weighted discharge expressed as an equivalent km2 at unit wetness.",
                "Climate values are normalized model fields intended as cheap downstream inputs, not observed SI climatology.",
                "Geometry arrays are capped samples; areas and component measurements use the complete objects.",
                "Mountain association means and fractions are fine-cell weighted on an adaptive mesh, not area weighted.",
                "The climatology null is fitted in-sample from this world's product precipitation; it is a diagnostic projection, not an independently generative replacement climate.",
                "Conditional-null fit statistics must be read with the reported occupied-bin area support; sparse joint bins can overstate in-sample explanatory power.",
                "The frozen-terrain comparison does not rewind climate-shaped mesh refinement, erosion, or other upstream history.",
            ],
            future_gaps: vec![
                "Coast component geometry, straits, adjacency and topology-aware generalization are not yet represented.",
                "Persistent cross-run object identity requires spatial matching beyond these per-run ranks.",
                "Matched physical, diagnostic, cartographic, and dramatic renders are produced by a separate presentation tool.",
            ],
        })
    }
}

fn build_mountains(
    world: &World,
    tess: &Tessellation,
    elevation: &[f32],
    temperature: &[f32],
    precipitation: &[f32],
) -> Vec<MountainTarget> {
    let fine = world.fine.as_ref().expect("checked by caller");
    let fields = &fine.base.fields.elevation_fields;
    let plates = world.plates.as_ref().expect("fine world requires plates");
    let mask: Vec<bool> = elevation
        .iter()
        .map(|&value| elevation_to_km(value) >= MOUNTAIN_ELEVATION_KM)
        .collect();
    let components = measure_components(tess, &mask);
    let components: Vec<_> = components
        .into_iter()
        .filter(|component| component.area_km2 >= SIGNIFICANT_MOUNTAIN_AREA_KM2)
        .map(|component| {
            let (anchor, peak) = peak_cell(&component.cells, elevation);
            (component, anchor, peak)
        })
        .collect();
    let selected = select_mountain_candidates(components);

    selected
        .into_iter()
        .enumerate()
        .map(|(rank, ((component, anchor, peak), selection_role))| {
            let coarse_cells: Vec<usize> = component
                .cells
                .iter()
                .map(|&cell| fine.base.coarse_cell[cell])
                .collect();
            let dominant_plate =
                dominant_u32(coarse_cells.iter().map(|&cell| plates.cell_plate[cell]));
            let mean = |field: &[f32]| {
                component.cells.iter().map(|&cell| field[cell]).sum::<f32>()
                    / component.cells.len() as f32
            };
            let pre_erosion_peak = component
                .cells
                .iter()
                .map(|&cell| fine.pre.elevation.values[cell])
                .fold(f32::NEG_INFINITY, f32::max);
            let coarse_features = world
                .features
                .as_ref()
                .expect("fine world requires features");
            MountainTarget {
                rank: rank + 1,
                selection_role,
                anchor_peak: spatial_anchor(tess, elevation, anchor),
                focus_radius_km: (component.length_km * 0.6).max(150.0),
                area_km2: component.area_km2,
                extent_km: component.length_km,
                mean_width_km: component.width_km,
                elongation: component.elongation(),
                cell_count: component.cells.len(),
                geometry_sample: sample_anchors(
                    tess,
                    elevation,
                    &component.cells,
                    GEOMETRY_SAMPLE_LIMIT,
                    Some(anchor),
                ),
                pre_erosion_peak_elevation_km: elevation_to_km(pre_erosion_peak),
                final_peak_elevation_km: elevation_to_km(peak),
                tectonics: TectonicAssociation {
                    dominant_plate,
                    collision_cell_fraction: component
                        .fraction_where(|cell| fields.collision[cell] > 0.02),
                    arc_cell_fraction: component.fraction_where(|cell| fields.arc[cell] > 0.02),
                    mean_collision_response: mean(&fields.collision),
                    mean_arc_response: mean(&fields.arc),
                    mean_convergent_influence: mean(&fields.convergent),
                    mean_coarse_tectonic_activity: coarse_cells
                        .iter()
                        .map(|&cell| coarse_features.activity[cell])
                        .sum::<f32>()
                        / coarse_cells.len() as f32,
                    mean_tectonic_thickening: mean(&fields.tectonic_thickening),
                    mean_tectonic_uplift_rate: mean(&fields.tectonic_uplift_rate),
                },
                climate: climate_at(anchor, temperature, precipitation),
                drainage_integration_cut_cells: component
                    .cells
                    .iter()
                    .filter(|&&cell| surface_hydrology(world).was_lowered_by_integration(cell))
                    .count(),
            }
        })
        .collect()
}

type MountainCandidate = (ComponentStats, usize, f32);

fn select_mountain_candidates(
    mut components: Vec<MountainCandidate>,
) -> Vec<(MountainCandidate, MountainSelectionRole)> {
    let mut selected = Vec::with_capacity(TARGET_LIMIT);
    if let Some(index) = components
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            a.2.total_cmp(&b.2)
                .then_with(|| a.0.area_km2.total_cmp(&b.0.area_km2))
                .then_with(|| b.0.cells[0].cmp(&a.0.cells[0]))
        })
        .map(|(index, _)| index)
    {
        selected.push((components.remove(index), MountainSelectionRole::HighestPeak));
    }
    if let Some(index) = components
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            a.0.area_km2
                .total_cmp(&b.0.area_km2)
                .then_with(|| a.2.total_cmp(&b.2))
                .then_with(|| b.0.cells[0].cmp(&a.0.cells[0]))
        })
        .map(|(index, _)| index)
    {
        selected.push((components.remove(index), MountainSelectionRole::LargestArea));
    }
    if let Some(index) = components
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            a.0.width_km
                .total_cmp(&b.0.width_km)
                .then_with(|| a.0.area_km2.total_cmp(&b.0.area_km2))
                .then_with(|| b.0.cells[0].cmp(&a.0.cells[0]))
        })
        .map(|(index, _)| index)
    {
        selected.push((components.remove(index), MountainSelectionRole::Broadest));
    }

    selected
}

fn build_lakes(
    tess: &Tessellation,
    elevation: &[f32],
    temperature: &[f32],
    precipitation: &[f32],
    hydrology: &super::Hydrology,
    water: &WaterBodySemantics,
) -> Vec<LakeTarget> {
    let mut bodies: Vec<_> = water
        .bodies
        .iter()
        .filter(|body| body.kind == SemanticWaterKind::Lake)
        .collect();
    let mut selected = Vec::with_capacity(TARGET_LIMIT);
    if let Some(index) = bodies
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            a.area_km2
                .total_cmp(&b.area_km2)
                .then_with(|| b.id.anchor_cell.cmp(&a.id.anchor_cell))
        })
        .map(|(index, _)| index)
    {
        selected.push((bodies.remove(index), LakeSelectionRole::LargestArea));
    }
    if let Some(index) = bodies
        .iter()
        .enumerate()
        .filter(|(_, body)| matches!(body.outlet, super::WaterOutlet::Terminal))
        .max_by(|(_, a), (_, b)| {
            a.area_km2
                .total_cmp(&b.area_km2)
                .then_with(|| b.id.anchor_cell.cmp(&a.id.anchor_cell))
        })
        .map(|(index, _)| index)
    {
        selected.push((bodies.remove(index), LakeSelectionRole::LargestTerminal));
    }
    if let Some(index) = bodies
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            a.surface_elevation_km
                .total_cmp(&b.surface_elevation_km)
                .then_with(|| a.area_km2.total_cmp(&b.area_km2))
                .then_with(|| b.id.anchor_cell.cmp(&a.id.anchor_cell))
        })
        .map(|(index, _)| index)
    {
        selected.push((bodies.remove(index), LakeSelectionRole::HighestSurface));
    }
    selected
        .into_iter()
        .enumerate()
        .filter_map(|(rank, (body, selection_role))| {
            let basin_id = body.id.basin_id?;
            let basin = hydrology.basins.get(basin_id)?;
            let mut body_mask = vec![false; tess.num_cells()];
            for &cell in &body.cells {
                body_mask[cell] = true;
            }
            let extent = measure_components(tess, &body_mask)
                .first()
                .map_or(0.0, |component| component.length_km);
            Some(LakeTarget {
                rank: rank + 1,
                selection_role,
                semantic_anchor: spatial_anchor(tess, elevation, body.id.anchor_cell),
                focus_radius_km: (extent * 0.75).max(75.0),
                area_km2: body.area_km2,
                surface_elevation_km: body.surface_elevation_km,
                max_depth_km: body.max_depth_km,
                outlet: body.outlet,
                basin: LakeBasinData {
                    basin_id,
                    basin_cell_count: basin.cells.len(),
                    catchment_discharge_equivalent_km2: basin.catchment_area
                        * PLANET_RADIUS_KM.powi(2),
                    bottom_elevation_km: elevation_to_km(basin.bottom_elevation),
                    spill_elevation_km: elevation_to_km(basin.spill_elevation),
                    water_level_km: elevation_to_km(basin.water_level),
                    mean_temperature_normalized: basin.mean_temperature,
                    evaporation_factor: basin.evaporation_factor,
                    overflowing: basin.is_overflowing(),
                },
                geometry_sample: sample_anchors(
                    tess,
                    elevation,
                    &body.cells,
                    GEOMETRY_SAMPLE_LIMIT,
                    Some(body.id.anchor_cell),
                ),
                climate: climate_at(body.id.anchor_cell, temperature, precipitation),
                drainage_integration_cut_cells: body
                    .cells
                    .iter()
                    .filter(|&&cell| hydrology.was_lowered_by_integration(cell))
                    .count(),
            })
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn build_rivers(
    tess: &Tessellation,
    elevation: &[f32],
    temperature: &[f32],
    precipitation: &[f32],
    hydrology: &super::Hydrology,
    water: &WaterBodySemantics,
    network: &RiverNetwork,
) -> Vec<RiverTarget> {
    let mut candidates: Vec<_> = network
        .mouths
        .iter()
        .copied()
        .map(|mouth| {
            let trunk = main_trunk(mouth, hydrology, network);
            let length_km = polyline_length_km(tess, &trunk);
            RiverCandidate {
                mouth,
                trunk,
                length_km,
            }
        })
        .collect();
    let mut selected = Vec::with_capacity(TARGET_LIMIT);
    if let Some(index) = candidates
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            hydrology.flow_accumulation[a.mouth]
                .total_cmp(&hydrology.flow_accumulation[b.mouth])
                .then_with(|| b.mouth.cmp(&a.mouth))
        })
        .map(|(index, _)| index)
    {
        selected.push((
            candidates.remove(index),
            RiverSelectionRole::HighestDischarge,
        ));
    }
    if let Some(index) = candidates
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            a.length_km
                .total_cmp(&b.length_km)
                .then_with(|| {
                    hydrology.flow_accumulation[a.mouth]
                        .total_cmp(&hydrology.flow_accumulation[b.mouth])
                })
                .then_with(|| b.mouth.cmp(&a.mouth))
        })
        .map(|(index, _)| index)
    {
        selected.push((candidates.remove(index), RiverSelectionRole::LongestTrunk));
    }
    if let Some(index) = candidates
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            network.strahler_order[a.mouth]
                .cmp(&network.strahler_order[b.mouth])
                .then_with(|| {
                    hydrology.flow_accumulation[a.mouth]
                        .total_cmp(&hydrology.flow_accumulation[b.mouth])
                })
                .then_with(|| b.mouth.cmp(&a.mouth))
        })
        .map(|(index, _)| index)
    {
        selected.push((
            candidates.remove(index),
            RiverSelectionRole::HighestStrahlerOrder,
        ));
    }
    selected
        .into_iter()
        .enumerate()
        .map(|(rank, (candidate, selection_role))| {
            let RiverCandidate {
                mouth,
                trunk,
                length_km,
            } = candidate;
            let head = trunk[0];
            let mouth_kind = match hydrology.downstream(mouth) {
                Some(next) if hydrology.is_ocean(next) => {
                    RiverMouth::Ocean(water.cell_body[next].expect("ocean has semantic body"))
                }
                Some(next) if hydrology.is_lake_water(next) => {
                    RiverMouth::Lake(water.cell_body[next].expect("lake has semantic body"))
                }
                Some(_) => RiverMouth::Confluence,
                None => RiverMouth::Inland,
            };
            RiverTarget {
                rank: rank + 1,
                selection_role,
                mouth: spatial_anchor(tess, elevation, mouth),
                head: spatial_anchor(tess, elevation, head),
                focus_radius_km: (length_km * 0.6).max(200.0),
                mouth_kind,
                catchment_discharge_equivalent_km2: hydrology.flow_accumulation[mouth]
                    * PLANET_RADIUS_KM.powi(2),
                strahler_order_at_mouth: network.strahler_order[mouth],
                is_major: network.major_cells[mouth],
                trunk_cell_count: trunk.len(),
                trunk_length_km_approx: length_km,
                trunk_geometry: sample_anchors(
                    tess,
                    elevation,
                    &trunk,
                    RIVER_SAMPLE_LIMIT,
                    Some(mouth),
                ),
                mouth_climate: climate_at(mouth, temperature, precipitation),
                head_climate: climate_at(head, temperature, precipitation),
                trunk_integration_cut_cells: trunk
                    .iter()
                    .filter(|&&cell| hydrology.was_lowered_by_integration(cell))
                    .count(),
            }
        })
        .collect()
}

struct RiverCandidate {
    mouth: usize,
    trunk: Vec<usize>,
    length_km: f32,
}

fn main_trunk(mouth: usize, hydrology: &super::Hydrology, network: &RiverNetwork) -> Vec<usize> {
    let mut trunk = vec![mouth];
    let mut current = mouth;
    while let Some(&up) = network.upstream[current].iter().max_by(|&&a, &&b| {
        hydrology.flow_accumulation[a]
            .total_cmp(&hydrology.flow_accumulation[b])
            .then_with(|| b.cmp(&a))
    }) {
        trunk.push(up);
        current = up;
    }
    trunk.reverse();
    trunk
}

fn surface_hydrology(world: &World) -> &super::Hydrology {
    let fine = world.fine.as_ref().expect("fine world required");
    &fine.surface_for(world.view_stage()).hydrology
}

fn spatial_anchor(tess: &Tessellation, elevation: &[f32], cell: usize) -> SpatialAnchor {
    let point = tess.cell_center(cell);
    SpatialAnchor {
        cell,
        latitude_deg: point.y.clamp(-1.0, 1.0).asin().to_degrees(),
        longitude_deg: point.z.atan2(point.x).to_degrees(),
        unit_xyz: point.to_array(),
        elevation_km: elevation_to_km(elevation[cell]),
    }
}

fn climate_at(cell: usize, temperature: &[f32], precipitation: &[f32]) -> ClimateAtAnchor {
    ClimateAtAnchor {
        temperature_normalized: temperature[cell],
        precipitation_relative_to_global_mean: precipitation[cell],
    }
}

fn peak_cell(cells: &[usize], elevation: &[f32]) -> (usize, f32) {
    let cell = cells
        .iter()
        .copied()
        .max_by(|&a, &b| {
            elevation[a]
                .total_cmp(&elevation[b])
                .then_with(|| b.cmp(&a))
        })
        .expect("component is non-empty");
    (cell, elevation[cell])
}

fn sample_anchors(
    tess: &Tessellation,
    elevation: &[f32],
    cells: &[usize],
    limit: usize,
    required: Option<usize>,
) -> Vec<SpatialAnchor> {
    let indices = capped_sample(cells, limit, required);
    indices
        .into_iter()
        .map(|cell| spatial_anchor(tess, elevation, cell))
        .collect()
}

fn capped_sample(cells: &[usize], limit: usize, required: Option<usize>) -> Vec<usize> {
    if limit == 0 || cells.is_empty() {
        return Vec::new();
    }
    let mut sorted = cells.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    if sorted.len() > limit {
        let last = sorted.len() - 1;
        sorted = (0..limit)
            .map(|index| sorted[index * last / (limit - 1).max(1)])
            .collect();
    }
    if let Some(cell) = required {
        if cells.contains(&cell) && !sorted.contains(&cell) {
            if sorted.len() == limit {
                sorted.pop();
            }
            sorted.push(cell);
            sorted.sort_unstable();
        }
    }
    sorted
}

fn dominant_u32(values: impl Iterator<Item = u32>) -> u32 {
    let mut counts = std::collections::BTreeMap::<u32, usize>::new();
    for value in values {
        *counts.entry(value).or_default() += 1;
    }
    counts
        .into_iter()
        .max_by(|(a_value, a_count), (b_value, b_count)| {
            a_count.cmp(b_count).then_with(|| b_value.cmp(a_value))
        })
        .map_or(0, |(value, _)| value)
}

fn polyline_length_km(tess: &Tessellation, cells: &[usize]) -> f32 {
    cells
        .windows(2)
        .map(|pair| {
            tess.cell_center(pair[0])
                .dot(tess.cell_center(pair[1]))
                .clamp(-1.0, 1.0)
                .acos()
                * PLANET_RADIUS_KM
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capped_sample_is_deterministic_capped_and_keeps_anchor() {
        let cells: Vec<usize> = (0..100).rev().collect();
        let sample = capped_sample(&cells, 8, Some(51));
        assert_eq!(sample.len(), 8);
        assert!(sample.contains(&51));
        assert_eq!(sample, capped_sample(&cells, 8, Some(51)));
        assert!(sample.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn dominant_value_breaks_ties_toward_lower_identity() {
        assert_eq!(dominant_u32([7, 2, 7, 2].into_iter()), 2);
    }

    #[test]
    fn mountain_selection_uses_distinct_peak_area_and_breadth_roles() {
        let component = |cell, area, width| ComponentStats {
            cells: vec![cell],
            area_km2: area,
            length_km: 100.0,
            width_km: width,
        };
        let selected = select_mountain_candidates(vec![
            (component(1, 10.0, 1.0), 1, 9.0),
            (component(2, 100.0, 2.0), 2, 5.0),
            (component(3, 20.0, 20.0), 3, 4.0),
            (component(4, 15.0, 3.0), 4, 3.0),
        ]);
        assert_eq!(selected.len(), 3);
        assert_eq!(selected[0].0 .1, 1);
        assert_eq!(selected[0].1, MountainSelectionRole::HighestPeak);
        assert_eq!(selected[1].0 .1, 2);
        assert_eq!(selected[1].1, MountainSelectionRole::LargestArea);
        assert_eq!(selected[2].0 .1, 3);
        assert_eq!(selected[2].1, MountainSelectionRole::Broadest);
    }

    #[test]
    fn packet_requires_fine_state() {
        let world = World::new(1, 32, 0);
        assert!(DossierPacket::build(&world).is_err());
    }
}
