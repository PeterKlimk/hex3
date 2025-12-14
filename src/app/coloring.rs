//! Color functions for visualization.
//!
//! This module contains all color conversion and cell coloring logic,
//! separated from the domain models.

use glam::Vec3;

use super::view::NoiseLayer;
use hex3::geometry::Material;
use hex3::world::{CellWaterState, PlateType, World};

/// Convert HSL to RGB.
pub fn hsl_to_rgb(h: f32, s: f32, l: f32) -> Vec3 {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let h_prime = h / 60.0;
    let x = c * (1.0 - (h_prime % 2.0 - 1.0).abs());

    let (r1, g1, b1) = match h_prime as i32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };

    let m = l - c / 2.0;
    Vec3::new(r1 + m, g1 + m, b1 + m)
}

/// Color for ocean water based on depth.
fn ocean_color(depth: f32) -> Vec3 {
    // Optical attenuation: depth response is strongly non-linear.
    // This avoids requiring per-world normalization while still giving contrast
    // across the typical depth range.
    const DEPTH_SCALE: f32 = 0.22;
    let d = depth.max(0.0);
    let t = 1.0 - (-(d / DEPTH_SCALE)).exp();

    // Slightly turquoise shallow water; deep water trends toward navy.
    let shallow = Vec3::new(0.10, 0.35, 0.55);
    let deep = Vec3::new(0.01, 0.04, 0.16);

    // Optional near-shore shelf brightening (very shallow scattering).
    let shelf = smoothstep(0.00, 0.03, d);
    let shelf_color = Vec3::new(0.12, 0.55, 0.62);
    shelf_color.lerp(shallow, shelf).lerp(deep, t)
}

/// Color for lake water based on depth.
fn lake_color(depth: f32) -> Vec3 {
    // Lakes are typically shallower and more optically "thick" (turbidity),
    // so they darken faster with depth than the open ocean.
    const DEPTH_SCALE: f32 = 0.07;
    let d = depth.max(0.0);
    let t = 1.0 - (-(d / DEPTH_SCALE)).exp();

    // Lakes lean slightly greener and lighter than ocean water.
    let shallow = Vec3::new(0.20, 0.55, 0.60);
    let deep = Vec3::new(0.06, 0.25, 0.35);

    // Keep a brighter shoreline band for readability.
    let shelf = smoothstep(0.00, 0.015, d);
    let shelf_color = Vec3::new(0.30, 0.65, 0.62);
    shelf_color.lerp(shallow, shelf).lerp(deep, t)
}

/// Color for land based on elevation (hypsometric tinting).
/// Matches the elevation_to_color palette for consistency.
/// Does NOT include snow caps - those are added separately.
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    if edge0 == edge1 {
        return if x < edge0 { 0.0 } else { 1.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Base land tint (no snow, no slope/moisture effects).
///
/// Designed to avoid the "tan blob" look by:
/// - spending more of the range in ochre/brown transitions
/// - desaturating toward rocky grays before snow
fn land_base_color(elevation: f32) -> Vec3 {
    // Most generated land tends to live in ~0..0.6; clamping here avoids
    // weird parameter choices pushing everything into a single extreme.
    let e = elevation.clamp(0.0, 0.8);

    if e < 0.06 {
        // Lowlands: deep green → green
        let t = e / 0.06;
        hsl_to_rgb(115.0, 0.55, 0.28).lerp(hsl_to_rgb(112.0, 0.50, 0.34), t)
    } else if e < 0.16 {
        // Plains: green → yellow-green
        let t = (e - 0.06) / 0.10;
        hsl_to_rgb(112.0, 0.50, 0.34).lerp(hsl_to_rgb(88.0, 0.45, 0.43), t)
    } else if e < 0.28 {
        // Dry grass / foothills: yellow-green → ochre
        let t = (e - 0.16) / 0.12;
        hsl_to_rgb(88.0, 0.45, 0.43).lerp(hsl_to_rgb(52.0, 0.40, 0.50), t)
    } else if e < 0.42 {
        // Uplands: ochre → brown
        let t = (e - 0.28) / 0.14;
        hsl_to_rgb(52.0, 0.40, 0.50).lerp(hsl_to_rgb(32.0, 0.33, 0.45), t)
    } else if e < 0.58 {
        // Mountains: brown → gray-brown (less saturated, more rock-like)
        let t = (e - 0.42) / 0.16;
        hsl_to_rgb(32.0, 0.33, 0.45).lerp(hsl_to_rgb(26.0, 0.14, 0.46), t)
    } else {
        // High terrain pre-snow: gray-brown → cool gray
        let t = ((e - 0.58) / 0.22).clamp(0.0, 1.0);
        hsl_to_rgb(26.0, 0.14, 0.46).lerp(hsl_to_rgb(220.0, 0.06, 0.54), t)
    }
}

/// Global snow line elevation.
/// Only the highest peaks get snow (peaks typically reach 0.3-0.5).
const SNOW_LINE: f32 = 0.28;

/// Apply snow cap coloring based on elevation.
/// Returns snow-adjusted color with smooth transition.
fn apply_snow_cap(base_color: Vec3, elevation: f32) -> Vec3 {
    let snow_transition = 0.12; // Elevation range for gradual transition

    if elevation < SNOW_LINE {
        base_color
    } else {
        // Blend to white above snow line
        let t = ((elevation - SNOW_LINE) / snow_transition).clamp(0.0, 1.0);
        let snow_color = Vec3::new(0.95, 0.97, 1.0); // Slightly blue-white snow
        base_color.lerp(snow_color, t)
    }
}

/// Compute slope steepness for a cell based on elevation gradient.
/// Returns a value from 0.0 (flat) to 1.0 (very steep).
pub fn compute_slope(world: &World, cell_idx: usize) -> f32 {
    let elevation = world
        .elevation
        .as_ref()
        .expect("Elevation must be generated");

    let cell_elev = elevation.values[cell_idx];
    let cell_pos = world.tessellation.cell_center(cell_idx);
    let neighbors = world.tessellation.neighbors(cell_idx);

    if neighbors.is_empty() {
        return 0.0;
    }

    // Compute maximum elevation difference per unit distance (steepest slope)
    let mut max_slope = 0.0f32;
    for &n in neighbors {
        let neighbor_elev = elevation.values[n];
        let neighbor_pos = world.tessellation.cell_center(n);

        // Arc distance between cell centers
        let dist = cell_pos.dot(neighbor_pos).clamp(-1.0, 1.0).acos();
        if dist < 1e-6 {
            continue;
        }

        // Slope = elevation change / horizontal distance
        let elev_diff = (neighbor_elev - cell_elev).abs();
        let slope = elev_diff / dist;
        max_slope = max_slope.max(slope);
    }

    // Normalize: slope of ~0.5 (about 27 degrees) starts looking steep
    // slope of ~1.0 (45 degrees) is very steep
    (max_slope * 2.0).clamp(0.0, 1.0)
}

/// Get the color for a cell in Terrain mode (default view).
///
/// Uses water state classification from hydrology.
/// This is the primary view that shows the "finished" terrain.
/// Micro noise is added to land elevation for color variation (cosmetic only).
/// Snow caps are applied based on elevation and latitude.
pub fn cell_color_terrain(world: &World, cell_idx: usize) -> Vec3 {
    let elevation = world
        .elevation
        .as_ref()
        .expect("Elevation must be generated");

    // Check water state if hydrology is available
    if let Some(hydrology) = &world.hydrology {
        let depth = hydrology.water_depth(cell_idx);
        match hydrology.water_state(cell_idx) {
            CellWaterState::Ocean => return ocean_color(depth),
            CellWaterState::LakeWater => return lake_color(depth),
            CellWaterState::DryBasin | CellWaterState::Land => {}
        }
    }

    // Land (or dry basin) - use elevation + micro noise for color variation
    let visual_elevation = elevation.values[cell_idx]
        + elevation.noise_layers.micro_layer[cell_idx]
        + 0.25 * elevation.noise_layers.hills_layer[cell_idx];

    // Slope for rock exposure.
    let slope = compute_slope(world, cell_idx);

    // Cheap moisture proxy: flow accumulation (if hydrology exists).
    // Purposefully subtle: greener valleys without requiring a full climate model.
    let moisture = world.hydrology.as_ref().map_or(0.0, |hydrology| {
        let flow = hydrology.flow_accumulation[cell_idx].max(1.0);
        let ln_flow = flow.ln();
        let ln_n = (world.num_cells() as f32).ln().max(1.0);
        let flow_t = (ln_flow / ln_n).clamp(0.0, 1.0);

        // Emphasize higher-flow channels; suppress on steep slopes and at high elevation.
        let channel = smoothstep(0.35, 0.85, flow_t);
        let lowland = 1.0 - smoothstep(0.22, 0.45, visual_elevation);
        let gentle = 1.0 - smoothstep(0.25, 0.65, slope);
        (channel * lowland * gentle).clamp(0.0, 1.0)
    });

    // Base hypsometric tint.
    let mut base_color = land_base_color(visual_elevation);

    // Moisture pushes lowlands/valleys a bit greener and darker.
    if moisture > 0.0 {
        let wet_tint = Vec3::new(0.08, 0.14, 0.06);
        base_color = (base_color + wet_tint * moisture).clamp(Vec3::ZERO, Vec3::ONE);
        base_color = base_color.lerp(Vec3::new(0.22, 0.40, 0.22), 0.25 * moisture);
    }

    // Rock exposure: increase with slope and with elevation (more barren mountains).
    let rockiness = smoothstep(0.12, 0.70, slope) * smoothstep(0.18, 0.50, visual_elevation);
    let rock_color = Vec3::new(0.42, 0.41, 0.40).lerp(Vec3::new(0.58, 0.60, 0.62), rockiness);
    let sloped_color = base_color.lerp(rock_color, 0.85 * rockiness);

    // Apply snow caps on high peaks
    apply_snow_cap(sloped_color, visual_elevation)
}

/// Convert elevation to color using hypsometric tinting.
/// Calibrated for actual ranges: ocean -0.25 to 0, land 0 to ~0.6
pub fn elevation_to_color(elevation: f32) -> Vec3 {
    let e = elevation.clamp(-0.3, 0.8);

    if e < -0.18 {
        // Deep ocean: dark blue
        let t = (e + 0.3) / 0.12;
        Vec3::new(0.02, 0.05, 0.18).lerp(Vec3::new(0.05, 0.10, 0.30), t)
    } else if e < -0.05 {
        // Shallow ocean: medium blue
        let t = (e + 0.18) / 0.13;
        Vec3::new(0.05, 0.10, 0.30).lerp(Vec3::new(0.12, 0.25, 0.50), t)
    } else if e < 0.0 {
        // Continental shelf: light blue
        let t = (e + 0.05) / 0.05;
        Vec3::new(0.12, 0.25, 0.50).lerp(Vec3::new(0.20, 0.40, 0.55), t)
    } else if e < 0.08 {
        // Coastal lowlands: green
        let t = e / 0.08;
        Vec3::new(0.20, 0.48, 0.25).lerp(Vec3::new(0.30, 0.55, 0.30), t)
    } else if e < 0.20 {
        // Plains/hills: yellow-green
        let t = (e - 0.08) / 0.12;
        Vec3::new(0.30, 0.55, 0.30).lerp(Vec3::new(0.55, 0.55, 0.35), t)
    } else if e < 0.35 {
        // Highlands: tan
        let t = (e - 0.20) / 0.15;
        Vec3::new(0.55, 0.55, 0.35).lerp(Vec3::new(0.55, 0.48, 0.40), t)
    } else if e < 0.50 {
        // Mountains: brown-gray
        let t = (e - 0.35) / 0.15;
        Vec3::new(0.55, 0.48, 0.40).lerp(Vec3::new(0.55, 0.52, 0.48), t)
    } else {
        // High peaks: gray to white
        let t = ((e - 0.50) / 0.30).clamp(0.0, 1.0);
        Vec3::new(0.55, 0.52, 0.48).lerp(Vec3::new(0.95, 0.97, 1.0), t)
    }
}

/// Get the color for a cell based on its elevation (hypsometric tinting).
pub fn cell_color_elevation(world: &World, cell_idx: usize) -> Vec3 {
    let elevation = world
        .elevation
        .as_ref()
        .expect("Elevation must be generated")
        .values[cell_idx];
    elevation_to_color(elevation)
}

/// Get the color for a cell based on its noise contribution.
pub fn cell_color_noise(world: &World, cell_idx: usize, layer: NoiseLayer) -> Vec3 {
    let elevation = world
        .elevation
        .as_ref()
        .expect("Elevation must be generated");

    let noise = match layer {
        NoiseLayer::Combined => elevation.noise_contribution[cell_idx],
        NoiseLayer::Macro => elevation.noise_layers.macro_layer[cell_idx],
        NoiseLayer::Hills => elevation.noise_layers.hills_layer[cell_idx],
        NoiseLayer::Ridges => elevation.noise_layers.ridge_layer[cell_idx],
        NoiseLayer::Micro => elevation.noise_layers.micro_layer[cell_idx],
    };

    // Scale factor varies by layer (smaller layers need more amplification to be visible)
    let scale = match layer {
        NoiseLayer::Combined => 10.0,
        NoiseLayer::Macro => 15.0,
        NoiseLayer::Hills => 12.0,
        NoiseLayer::Ridges => 10.0,
        NoiseLayer::Micro => 50.0, // Micro is very small, needs big scale
    };

    let t = (noise.abs() * scale).min(1.0);
    if noise >= 0.0 {
        Vec3::new(0.3 * (1.0 - t), 0.3 + 0.7 * t, 0.3 * (1.0 - t))
    } else {
        Vec3::new(0.3 + 0.7 * t, 0.3 * (1.0 - t), 0.3 + 0.7 * t)
    }
}

/// Get the color for a cell based on its plate assignment.
pub fn cell_color_plate(world: &World, cell_idx: usize) -> Vec3 {
    let plates = world.plates.as_ref().expect("Plates must be generated");
    let dynamics = world.dynamics.as_ref().expect("Dynamics must be generated");

    let plate_id = plates.cell_plate[cell_idx] as usize;
    let plate_type = dynamics.plate_type(plate_id);

    let continental_plates: Vec<usize> = (0..plates.num_plates)
        .filter(|&p| dynamics.plate_type(p) == PlateType::Continental)
        .collect();
    let oceanic_plates: Vec<usize> = (0..plates.num_plates)
        .filter(|&p| dynamics.plate_type(p) == PlateType::Oceanic)
        .collect();

    match plate_type {
        PlateType::Continental => {
            let idx = continental_plates
                .iter()
                .position(|&p| p == plate_id)
                .unwrap_or(0);
            let t = if continental_plates.len() > 1 {
                idx as f32 / (continental_plates.len() - 1) as f32
            } else {
                0.5
            };
            let hue = 30.0 + t * 60.0;
            hsl_to_rgb(hue, 0.5, 0.5)
        }
        PlateType::Oceanic => {
            let idx = oceanic_plates
                .iter()
                .position(|&p| p == plate_id)
                .unwrap_or(0);
            let t = if oceanic_plates.len() > 1 {
                idx as f32 / (oceanic_plates.len() - 1) as f32
            } else {
                0.5
            };
            let hue = 180.0 + t * 60.0;
            hsl_to_rgb(hue, 0.5, 0.4)
        }
    }
}

/// Get the color for a cell based on its stress value.
pub fn cell_color_stress(world: &World, cell_idx: usize) -> Vec3 {
    let stress = world
        .stress
        .as_ref()
        .expect("Stress must be generated")
        .cell_stress[cell_idx];

    if stress > 0.0 {
        let t = (stress / 0.5).clamp(0.0, 1.0);
        Vec3::new(0.2 + t * 0.8, 0.2, 0.2)
    } else {
        let t = (-stress / 0.3).clamp(0.0, 1.0);
        Vec3::new(0.2, 0.2, 0.2 + t * 0.8)
    }
}

/// Get the color for a cell based on hydrology data.
///
/// Shows:
/// - Ocean: deep blue
/// - Lakes: lighter blue (intensity by depth)
/// - Dry basins: tan/beige (playa-like)
/// - Land: green/brown tinted by flow accumulation
pub fn cell_color_hydrology(world: &World, cell_idx: usize) -> Vec3 {
    // Hydrology view requires hydrology to be generated
    let Some(hydrology) = &world.hydrology else {
        // Fallback to basic land color
        return Vec3::new(0.3, 0.5, 0.3);
    };

    let depth = hydrology.water_depth(cell_idx);
    match hydrology.water_state(cell_idx) {
        CellWaterState::Ocean => {
            ocean_color(depth)
        }
        CellWaterState::LakeWater => {
            lake_color(depth)
        }
        CellWaterState::DryBasin => {
            // Dry basin - tan/beige (playa, salt flat)
            Vec3::new(0.75, 0.70, 0.55)
        }
        CellWaterState::Land => {
            // Land - color by flow accumulation (drainage intensity)
            // Low flow = drier uplands (tan/brown)
            // High flow = wetter valleys (greener)
            let flow = hydrology.flow_accumulation[cell_idx];
            let flow_t = (flow.ln().max(0.0) / 7.0).clamp(0.0, 1.0);
            let dry = Vec3::new(0.6, 0.55, 0.4); // tan/brown (ridges, divides)
            let wet = Vec3::new(0.25, 0.45, 0.25); // green (valleys, drainage)
            dry.lerp(wet, flow_t)
        }
    }
}

/// Get the material type for a cell based on its water state.
///
/// This determines how lighting is applied in the unified shader:
/// - Land: matte diffuse only
/// - Ocean: diffuse + specular + fresnel
/// - Lake: diffuse + specular + fresnel (subtler)
pub fn cell_material(world: &World, cell_idx: usize) -> Material {
    if let Some(hydrology) = &world.hydrology {
        match hydrology.water_state(cell_idx) {
            CellWaterState::Ocean => Material::Ocean,
            CellWaterState::LakeWater => Material::Lake,
            CellWaterState::DryBasin | CellWaterState::Land => Material::Land,
        }
    } else {
        // Without hydrology, use simple elevation test
        let elevation = world
            .elevation
            .as_ref()
            .expect("Elevation must be generated")
            .values[cell_idx];
        if elevation < 0.0 {
            Material::Ocean
        } else {
            Material::Land
        }
    }
}
