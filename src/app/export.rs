//! World data export for external analysis.

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use std::time::Instant;

use flate2::write::GzEncoder;
use flate2::Compression;
use glam::Vec3;
use serde::Serialize;

use hex3::world::{CrustType, World};

/// Export world data to a JSON file (optionally gzipped).
///
/// Export always reflects the LATEST computed stage, not whatever stage is
/// currently being viewed: the view cap is lifted for the read (then restored),
/// so a stage-back navigation can't desync the `active_*` arrays from the
/// fine-mesh arrays read directly. Takes `&mut World` only to toggle that cap.
pub fn export_world(world: &mut World, seed: u64, path: &Path) {
    print!("Exporting to {}... ", path.display());
    let start = Instant::now();

    let saved_view = world.view_stage();
    world.set_view_stage(u32::MAX);
    let data = WorldExport::from_world(world, seed);
    world.set_view_stage(saved_view);

    let file = File::create(path).expect("Failed to create export file");

    // Check if we should gzip based on extension
    let is_gzip = path.extension().map(|ext| ext == "gz").unwrap_or(false);

    if is_gzip {
        let encoder = GzEncoder::new(BufWriter::new(file), Compression::default());
        serde_json::to_writer(encoder, &data).expect("Failed to write JSON");
    } else {
        let writer = BufWriter::new(file);
        serde_json::to_writer(writer, &data).expect("Failed to write JSON");
    }

    println!("{:.1}ms", start.elapsed().as_secs_f64() * 1000.0);
}

#[derive(Serialize)]
struct WorldExport {
    metadata: Metadata,
    cells: CellData,
    plates: Vec<PlateData>,
    boundary_episodes: Vec<BoundaryEpisodeData>,
}

#[derive(Serialize)]
struct Metadata {
    seed: u64,
    num_cells: usize,
    num_plates: usize,
    stage: u32,
    mean_neighbor_dist: f32,
    mean_cell_area: f32,
    orogen_model: String,
    tectonic_lookback_myr: f32,
    tectonic_step_myr: f32,
    max_plate_speed_km_per_myr: f32,
    boundary_episode_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    tectonic_carrier: Option<CarrierReplayData>,
}

#[derive(Serialize)]
struct CarrierReplayData {
    cells: usize,
    mean_spacing_km: f32,
    step_myr: f32,
    snapshots: usize,
    mean_gap_fraction: f32,
    mean_overlap_excess_fraction: f32,
    topology_changes: usize,
}

#[derive(Serialize)]
struct CellData {
    elevation: Vec<f32>,
    plate_id: Vec<u32>,
    crust_type: Vec<u8>, // 0 = continental, 1 = oceanic (per-cell, independent of plates)
    area: Vec<f32>,
    latitude: Vec<f32>,
    longitude: Vec<f32>,

    features: FeatureData,
    noise: NoiseData,

    /// Stage-separated topography for structural survival/A-B analysis.
    #[serde(skip_serializing_if = "Option::is_none")]
    stages: Option<StageData>,

    #[serde(skip_serializing_if = "Option::is_none")]
    density: Option<Vec<f32>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    atmosphere: Option<AtmosphereData>,

    #[serde(skip_serializing_if = "Option::is_none")]
    hydrology: Option<HydrologyData>,
}

#[derive(Serialize)]
struct AtmosphereData {
    temperature: Vec<f32>,
    uplift: Vec<f32>,
    precipitation: Vec<f32>,
    wind: Vec<f32>,
    wind_speed: Vec<f32>,
    upper_wind: Vec<f32>,
    upper_wind_speed: Vec<f32>,
}

#[derive(Serialize)]
struct FeatureData {
    trench: Vec<f32>,
    arc: Vec<f32>,
    ridge: Vec<f32>,
    collision: Vec<f32>,
    activity: Vec<f32>,
    convergent: Vec<f32>,
    divergent: Vec<f32>,
    ridge_distance: Vec<f32>,
    ridge_age_distance: Vec<f32>,
    ridge_spreading_rate: Vec<f32>,
    collision_distance: Vec<f32>,
    tectonic_crust_flux: Vec<f32>,
    tectonic_crust_work: Vec<f32>,
    tectonic_work_mean_duration_myr: f32,
    thin_sheet_material_added: f64,
    thin_sheet_material_residual: f64,
    tectonic_uplift_rate: Vec<f32>,
    carrier_evolution_seconds: f32,
    carrier_moving_forcing_fraction: f32,
}

#[derive(Serialize)]
struct StageData {
    /// Interpolated coarse target on the fine mesh.
    coarse_envelope: Vec<f32>,
    /// Fine substrate after demotion and structural synthesis, before erosion.
    fine_base: Vec<f32>,
    /// Stage-3 surface used by pre-erosion hydrology.
    pre_erosion: Vec<f32>,
    /// Stage-4 surface, when generated.
    #[serde(skip_serializing_if = "Option::is_none")]
    eroded: Option<Vec<f32>>,
    /// Conserved/legacy-equivalent tectonic thickness transferred to fine cells.
    tectonic_thickening: Vec<f32>,
    tectonic_strain: Vec<f32>,
    compression_axis: Vec<Vec3>,
    /// `fine_base - coarse_envelope`: detail added or demoted before erosion.
    structural_delta: Vec<f32>,
    /// `eroded - fine_base`, when stage 4 exists.
    #[serde(skip_serializing_if = "Option::is_none")]
    erosion_delta: Option<Vec<f32>>,
}

#[derive(Serialize)]
struct NoiseData {
    combined: Vec<f32>,
    macro_layer: Vec<f32>,
}

#[derive(Serialize)]
struct HydrologyData {
    flow_accumulation: Vec<f32>,
    is_lake: Vec<bool>,
    lake_surface: Vec<Option<f32>>,
}

#[derive(Serialize)]
struct PlateData {
    id: usize,
    continental_fraction: f32,
    cell_count: usize,
    euler_pole: [f32; 3],
    angular_velocity: f32,
    angular_velocity_rad_per_myr: f32,
    max_surface_speed_km_per_myr: f32,
}

#[derive(Serialize)]
struct BoundaryEpisodeData {
    id: usize,
    plate_a: usize,
    plate_b: usize,
    kind: String,
    subducting_plate: Option<usize>,
    edge_count: usize,
    length_km: f32,
    mean_convergence_km_per_myr: f32,
    mean_shear_km_per_myr: f32,
    mean_relative_speed_km_per_myr: f32,
    duration_myr: f32,
    integrated_normal_displacement_km: f32,
    integrated_shear_displacement_km: f32,
    history_model: String,
}

impl WorldExport {
    fn from_world(world: &World, seed: u64) -> Self {
        let tessellation = world.active_tessellation();
        let elevation = world.active_elevation().expect("Elevation not generated");
        let hydrology = world.active_hydrology();
        let fine = world.fine.as_ref();
        let num_cells = tessellation.num_cells();
        let cell_areas = tessellation.cell_areas();
        let mean_area = tessellation.mean_cell_area();

        // Compute mean neighbor distance
        let mean_neighbor_dist = compute_mean_neighbor_dist(tessellation);

        // Get references to required data (these should always be present after stage 1)
        let plates = world.plates.as_ref().expect("Plates not generated");
        let crust = world.crust.as_ref().expect("Crust not generated");
        let dynamics = world.dynamics.as_ref().expect("Dynamics not generated");
        let features = world.features.as_ref().expect("Features not generated");

        // Build cell data arrays
        let mut elevation_vec = Vec::with_capacity(num_cells);
        let mut plate_id = Vec::with_capacity(num_cells);
        let mut crust_type = Vec::with_capacity(num_cells);
        let mut latitude = Vec::with_capacity(num_cells);
        let mut longitude = Vec::with_capacity(num_cells);

        for i in 0..num_cells {
            elevation_vec.push(elevation.values[i]);
            let coarse_i = fine.map(|f| f.coarse_cell()[i]).unwrap_or(i);
            plate_id.push(plates.cell_plate[coarse_i]);

            let continental = fine
                .map(|f| f.fields().elevation_fields.continentality[i] >= 0.5)
                .unwrap_or_else(|| crust.crust_type(i) == CrustType::Continental);
            crust_type.push(if continental { 0 } else { 1 });

            // y is the pole axis (matches the simulation and map projection)
            let center = tessellation.cell_center(i);
            latitude.push(center.y.asin());
            longitude.push(center.z.atan2(center.x));
        }

        // Features
        let features_data = if let Some(fine) = fine {
            let map_feature = |field: &[f32]| -> Vec<f32> {
                fine.coarse_cell().iter().map(|&c| field[c]).collect()
            };
            FeatureData {
                trench: fine.fields().elevation_fields.trench.clone(),
                arc: map_feature(&features.arc),
                ridge: fine.fields().elevation_fields.ridge.clone(),
                collision: map_feature(&features.collision),
                activity: map_feature(&features.activity),
                convergent: fine.fields().elevation_fields.convergent.clone(),
                divergent: fine.fields().elevation_fields.divergent.clone(),
                ridge_distance: map_feature(&features.ridge_distance),
                ridge_age_distance: fine.fields().elevation_fields.ridge_age_distance.clone(),
                ridge_spreading_rate: map_feature(&features.ridge_spreading_rate),
                collision_distance: map_feature(&features.collision_distance),
                tectonic_crust_flux: map_feature(&features.tectonic_crust_flux),
                tectonic_crust_work: map_feature(&features.tectonic_crust_work),
                tectonic_work_mean_duration_myr: features.tectonic_work_mean_duration_myr,
                thin_sheet_material_added: features.thin_sheet_material_added,
                thin_sheet_material_residual: features.thin_sheet_material_residual,
                tectonic_uplift_rate: map_feature(&features.tectonic_uplift_rate),
                carrier_evolution_seconds: features.carrier_evolution_seconds,
                carrier_moving_forcing_fraction: features.carrier_moving_forcing_fraction,
            }
        } else {
            FeatureData {
                trench: features.trench.clone(),
                arc: features.arc.clone(),
                ridge: features.ridge.clone(),
                collision: features.collision.clone(),
                activity: features.activity.clone(),
                convergent: features.convergent.clone(),
                divergent: features.divergent.clone(),
                ridge_distance: features.ridge_distance.clone(),
                ridge_age_distance: features.ridge_age_distance.clone(),
                ridge_spreading_rate: features.ridge_spreading_rate.clone(),
                collision_distance: features.collision_distance.clone(),
                tectonic_crust_flux: features.tectonic_crust_flux.clone(),
                tectonic_crust_work: features.tectonic_crust_work.clone(),
                tectonic_work_mean_duration_myr: features.tectonic_work_mean_duration_myr,
                thin_sheet_material_added: features.thin_sheet_material_added,
                thin_sheet_material_residual: features.thin_sheet_material_residual,
                tectonic_uplift_rate: features.tectonic_uplift_rate.clone(),
                carrier_evolution_seconds: features.carrier_evolution_seconds,
                carrier_moving_forcing_fraction: features.carrier_moving_forcing_fraction,
            }
        };

        let stages = fine.map(|fine| {
            let eroded = fine
                .eroded
                .as_ref()
                .map(|surface| surface.elevation.values.clone());
            let structural_delta = fine
                .base
                .base_elevation
                .iter()
                .zip(fine.base.coarse_base_elevation.iter())
                .map(|(&base, &coarse)| base - coarse)
                .collect();
            let erosion_delta = eroded.as_ref().map(|values| {
                values
                    .iter()
                    .zip(fine.base.base_elevation.iter())
                    .map(|(&surface, &base)| surface - base)
                    .collect()
            });
            StageData {
                coarse_envelope: fine.base.coarse_base_elevation.clone(),
                fine_base: fine.base.base_elevation.clone(),
                pre_erosion: fine.pre.elevation.values.clone(),
                eroded,
                tectonic_thickening: fine
                    .base
                    .fields
                    .elevation_fields
                    .tectonic_thickening
                    .clone(),
                tectonic_strain: fine.base.fields.elevation_fields.tectonic_strain.clone(),
                compression_axis: fine.base.fields.elevation_fields.compression_axis.clone(),
                structural_delta,
                erosion_delta,
            }
        });

        // Noise (combined contribution)
        let noise = NoiseData {
            combined: elevation.noise_contribution.clone(),
            macro_layer: elevation.noise_layers.macro_layer.clone(),
        };

        // Hydrology (if available)
        let atmosphere_data = world.atmosphere.as_ref().map(|a| {
            let mut wind = Vec::with_capacity(num_cells);
            let mut wind_speed = Vec::with_capacity(num_cells);
            let mut upper_wind = Vec::with_capacity(num_cells);
            let mut upper_wind_speed = Vec::with_capacity(num_cells);

            for i in 0..num_cells {
                let coarse_i = fine.map(|f| f.coarse_cell()[i]).unwrap_or(i);
                let east = tangent_east(tessellation.cell_center(i));
                wind.push(a.wind[coarse_i].dot(east));
                wind_speed.push(a.wind[coarse_i].length());
                upper_wind.push(a.upper_wind[coarse_i].dot(east));
                upper_wind_speed.push(a.upper_wind[coarse_i].length());
            }

            AtmosphereData {
                temperature: world.active_temperature().unwrap().to_vec(),
                uplift: world.active_uplift().unwrap().to_vec(),
                precipitation: world.active_precipitation().unwrap().to_vec(),
                wind,
                wind_speed,
                upper_wind,
                upper_wind_speed,
            }
        });

        let hydrology_data = hydrology.map(|h| {
            let mut flow_accumulation = Vec::with_capacity(num_cells);
            let mut is_lake = Vec::with_capacity(num_cells);
            let mut lake_surface = Vec::with_capacity(num_cells);

            for i in 0..num_cells {
                flow_accumulation.push(h.flow_accumulation[i]);

                let basin_idx = h.basin_id[i];
                if let Some(idx) = basin_idx {
                    let basin = &h.basins[idx];
                    if basin.has_water() {
                        is_lake.push(true);
                        lake_surface.push(Some(basin.water_level));
                    } else {
                        is_lake.push(false);
                        lake_surface.push(None);
                    }
                } else {
                    is_lake.push(false);
                    lake_surface.push(None);
                }
            }

            HydrologyData {
                flow_accumulation,
                is_lake,
                lake_surface,
            }
        });

        // Plate data
        let mut plates_data = Vec::with_capacity(plates.num_plates);
        for pid in 0..plates.num_plates {
            let euler = dynamics.euler_pole(pid);

            // Count cells in this plate, and how many carry continental crust
            let mut cell_count = 0usize;
            let mut continental_count = 0usize;
            for (i, &p) in plates.cell_plate.iter().enumerate() {
                if p as usize == pid {
                    cell_count += 1;
                    if crust.is_continental(i) {
                        continental_count += 1;
                    }
                }
            }

            plates_data.push(PlateData {
                id: pid,
                continental_fraction: if cell_count > 0 {
                    continental_count as f32 / cell_count as f32
                } else {
                    0.0
                },
                cell_count,
                euler_pole: [euler.axis.x, euler.axis.y, euler.axis.z],
                angular_velocity: euler.angular_velocity,
                angular_velocity_rad_per_myr: euler.angular_velocity_rad_per_myr(),
                max_surface_speed_km_per_myr: euler.angular_velocity.abs()
                    * hex3::world::MAX_PLATE_SPEED_KM_PER_MYR,
            });
        }

        let boundary_episodes: Vec<_> = world
            .tectonic_history
            .as_ref()
            .map(|history| {
                history
                    .episodes
                    .iter()
                    .map(|episode| BoundaryEpisodeData {
                        id: episode.id,
                        plate_a: episode.plate_a,
                        plate_b: episode.plate_b,
                        kind: format!("{:?}", episode.kind).to_lowercase(),
                        subducting_plate: episode.subducting_plate,
                        edge_count: episode.edge_count,
                        length_km: episode.length_km,
                        mean_convergence_km_per_myr: episode.mean_convergence_km_per_myr,
                        mean_shear_km_per_myr: episode.mean_shear_km_per_myr,
                        mean_relative_speed_km_per_myr: episode.mean_relative_speed_km_per_myr,
                        duration_myr: episode.duration_myr,
                        integrated_normal_displacement_km: episode
                            .integrated_normal_displacement_km,
                        integrated_shear_displacement_km: episode.integrated_shear_displacement_km,
                        history_model: format!("{:?}", episode.model).to_lowercase(),
                    })
                    .collect()
            })
            .unwrap_or_default();

        let tectonic_carrier = world
            .tectonic_history
            .as_ref()
            .and_then(|history| history.carrier_replay.as_ref())
            .map(|carrier| {
                let snapshot_count = carrier.snapshots.len().max(1) as f32;
                CarrierReplayData {
                    cells: carrier.num_cells,
                    mean_spacing_km: carrier.mean_spacing_km,
                    step_myr: carrier.step_myr,
                    snapshots: carrier.snapshots.len(),
                    mean_gap_fraction: carrier
                        .snapshots
                        .iter()
                        .map(|snapshot| snapshot.gap_cells as f32 / carrier.num_cells as f32)
                        .sum::<f32>()
                        / snapshot_count,
                    mean_overlap_excess_fraction: carrier
                        .snapshots
                        .iter()
                        .map(|snapshot| snapshot.overlap_excess as f32 / carrier.num_cells as f32)
                        .sum::<f32>()
                        / snapshot_count,
                    topology_changes: carrier
                        .snapshots
                        .iter()
                        .map(|snapshot| snapshot.topology_changes_from_previous)
                        .sum(),
                }
            });

        Self {
            metadata: Metadata {
                seed,
                num_cells,
                num_plates: plates.num_plates,
                stage: world.current_stage(),
                mean_neighbor_dist,
                mean_cell_area: mean_area,
                orogen_model: world.orogen_model.to_string(),
                tectonic_lookback_myr: dynamics.clock.lookback_myr,
                tectonic_step_myr: world
                    .tectonic_history
                    .as_ref()
                    .map(|history| history.step_myr)
                    .unwrap_or(dynamics.clock.step_myr),
                max_plate_speed_km_per_myr: hex3::world::MAX_PLATE_SPEED_KM_PER_MYR,
                boundary_episode_count: boundary_episodes.len(),
                tectonic_carrier,
            },
            cells: CellData {
                elevation: elevation_vec,
                plate_id,
                crust_type,
                area: cell_areas,
                latitude,
                longitude,
                features: features_data,
                noise,
                stages,
                density: fine.map(|f| f.density().to_vec()),
                atmosphere: atmosphere_data,
                hydrology: hydrology_data,
            },
            plates: plates_data,
            boundary_episodes,
        }
    }
}

fn tangent_east(pos: Vec3) -> Vec3 {
    let east = Vec3::Y.cross(pos);
    let len = east.length();
    if len < 1e-6 {
        Vec3::X
    } else {
        east / len
    }
}

fn compute_mean_neighbor_dist(tessellation: &hex3::world::Tessellation) -> f32 {
    let mut total_dist: f32 = 0.0;
    let mut count: usize = 0;

    for i in 0..tessellation.num_cells() {
        let pos_i = tessellation.cell_center(i);
        for &j in tessellation.neighbors(i) {
            if j > i {
                let pos_j = tessellation.cell_center(j);
                let dist = pos_i.dot(pos_j).clamp(-1.0, 1.0).acos();
                total_dist += dist;
                count += 1;
            }
        }
    }

    if count > 0 {
        total_dist / count as f32
    } else {
        0.03
    }
}
