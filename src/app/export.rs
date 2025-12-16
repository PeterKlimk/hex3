//! World data export for external analysis.

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use std::time::Instant;

use flate2::write::GzEncoder;
use flate2::Compression;
use serde::Serialize;

use hex3::world::{PlateType, World};

/// Export world data to a JSON file (optionally gzipped).
pub fn export_world(world: &World, seed: u64, path: &Path) {
    print!("Exporting to {}... ", path.display());
    let start = Instant::now();

    let data = WorldExport::from_world(world, seed);

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
}

#[derive(Serialize)]
struct Metadata {
    seed: u64,
    num_cells: usize,
    num_plates: usize,
    stage: u32,
    mean_neighbor_dist: f32,
    mean_cell_area: f32,
}

#[derive(Serialize)]
struct CellData {
    elevation: Vec<f32>,
    plate_id: Vec<u32>,
    plate_type: Vec<u8>, // 0 = continental, 1 = oceanic
    area: Vec<f32>,
    latitude: Vec<f32>,

    features: FeatureData,
    noise: NoiseData,

    #[serde(skip_serializing_if = "Option::is_none")]
    hydrology: Option<HydrologyData>,
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
    transform: Vec<f32>,
    ridge_distance: Vec<f32>,
}

#[derive(Serialize)]
struct NoiseData {
    combined: Vec<f32>,
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
    plate_type: String,
    cell_count: usize,
    euler_pole: [f32; 3],
    angular_velocity: f32,
}

impl WorldExport {
    fn from_world(world: &World, seed: u64) -> Self {
        let num_cells = world.tessellation.num_cells();
        let cell_areas = world.tessellation.cell_areas();
        let mean_area = world.tessellation.mean_cell_area();

        // Compute mean neighbor distance
        let mean_neighbor_dist = compute_mean_neighbor_dist(&world.tessellation);

        // Get references to required data (these should always be present after stage 1)
        let plates = world.plates.as_ref().expect("Plates not generated");
        let dynamics = world.dynamics.as_ref().expect("Dynamics not generated");
        let features = world.features.as_ref().expect("Features not generated");
        let elevation = world.elevation.as_ref().expect("Elevation not generated");

        // Build cell data arrays
        let mut elevation_vec = Vec::with_capacity(num_cells);
        let mut plate_id = Vec::with_capacity(num_cells);
        let mut plate_type = Vec::with_capacity(num_cells);
        let mut latitude = Vec::with_capacity(num_cells);

        for i in 0..num_cells {
            elevation_vec.push(elevation.values[i]);
            plate_id.push(plates.cell_plate[i]);

            let ptype = dynamics.plate_type(plates.cell_plate[i] as usize);
            plate_type.push(if ptype == PlateType::Continental {
                0
            } else {
                1
            });

            // Latitude from cell center (z coordinate on unit sphere)
            let center = world.tessellation.cell_center(i);
            latitude.push(center.z.asin());
        }

        // Features
        let features_data = FeatureData {
            trench: features.trench.clone(),
            arc: features.arc.clone(),
            ridge: features.ridge.clone(),
            collision: features.collision.clone(),
            activity: features.activity.clone(),
            convergent: features.convergent.clone(),
            divergent: features.divergent.clone(),
            transform: features.transform.clone(),
            ridge_distance: features.ridge_distance.clone(),
        };

        // Noise (combined contribution)
        let noise = NoiseData {
            combined: elevation.noise_contribution.clone(),
        };

        // Hydrology (if available)
        let hydrology_data = world.hydrology.as_ref().map(|h| {
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
            let ptype = dynamics.plate_type(pid);
            let euler = dynamics.euler_pole(pid);

            // Count cells in this plate
            let cell_count = plates
                .cell_plate
                .iter()
                .filter(|&&p| p as usize == pid)
                .count();

            plates_data.push(PlateData {
                id: pid,
                plate_type: if ptype == PlateType::Continental {
                    "continental".to_string()
                } else {
                    "oceanic".to_string()
                },
                cell_count,
                euler_pole: [euler.axis.x, euler.axis.y, euler.axis.z],
                angular_velocity: euler.angular_velocity,
            });
        }

        Self {
            metadata: Metadata {
                seed,
                num_cells,
                num_plates: plates.num_plates,
                stage: world.current_stage(),
                mean_neighbor_dist,
                mean_cell_area: mean_area,
            },
            cells: CellData {
                elevation: elevation_vec,
                plate_id,
                plate_type,
                area: cell_areas,
                latitude,
                features: features_data,
                noise,
                hydrology: hydrology_data,
            },
            plates: plates_data,
        }
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
