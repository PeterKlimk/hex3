//! Guarded local S2-Voronoi cap for irregular-mesh landscape experiments.
//!
//! This is a product-backend mechanism fixture, not a claim that its local
//! jitter process reproduces the product's global generator statistics.

use std::{collections::HashMap, fmt};

use glam::{DVec2, DVec3, Vec3};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::world::{Tessellation, PLANET_RADIUS_KM};

use super::{
    BoundaryCondition, BoundaryFaceCondition, BoundarySide, LandscapeBoundaryFace, LandscapeMesh,
    OutletPortal, OutletPortalId,
};

pub const R1_CAP_WIDTH_KM: f64 = 256.0;
pub const R1_CAP_HEIGHT_KM: f64 = 224.0;
pub const R1_CAP_GUARD_SPACINGS: usize = 8;
pub const R1_CAP_PORTAL_ID: OutletPortalId = OutletPortalId(401);
pub const R1_CAP_PORTAL_HALF_WIDTH_KM: f64 = 80.0;
pub const R1_CAP_PORTAL_BASE_LEVEL_KM: f32 = 1.0;

const SITE_SEED: u64 = 0x5231_A11C_E001;
const Q_HASH: u64 = 0x9E37_79B9_7F4A_7C15;
const R_HASH: u64 = 0xD1B5_4A32_D192_ED03;
const JITTER_RADIUS_FRACTION: f64 = 0.18;
const FAR_FIELD_SITE_COUNT: usize = 128;
const FAR_FIELD_NORTH_EXCLUSION_RAD: f64 = 0.08;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VoronoiCapConfig {
    pub spacing_km: f64,
    pub guard_spacings: usize,
}

impl VoronoiCapConfig {
    pub fn r1(spacing_km: f64) -> Self {
        Self {
            spacing_km,
            guard_spacings: R1_CAP_GUARD_SPACINGS,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct VoronoiCapAudit {
    pub input_site_count: usize,
    pub retained_cell_count: usize,
    pub internal_undirected_faces: usize,
    pub boundary_faces: usize,
    pub portal_faces: usize,
    pub min_cell_area_km2: f64,
    pub cell_area_p10_km2: f64,
    pub cell_area_p50_km2: f64,
    pub cell_area_p90_km2: f64,
    pub min_internal_face_width_km: f64,
    pub internal_face_width_p10_km: f64,
    pub internal_face_width_p50_km: f64,
    pub internal_face_width_p90_km: f64,
    pub max_internal_face_width_km: f64,
    pub generator_centroid_offset_p50_km: f64,
    pub generator_centroid_offset_p95_km: f64,
    pub max_edge_projection_relative_error: f64,
    pub max_center_distance_projection_relative_error: f64,
    pub max_face_midpoint_projection_error_km: f64,
    pub max_cell_area_projection_relative_error: f64,
    pub total_projected_area_km2: f64,
    pub total_spherical_area_km2: f64,
    pub total_area_projection_relative_error: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VoronoiCapFixture {
    pub config: VoronoiCapConfig,
    pub mesh: LandscapeMesh,
    /// Retained product-backend generators before tangent projection.
    pub cell_center_unit: Vec<Vec3>,
    /// Retained product-backend polygon vertices before tangent projection.
    pub cell_polygons_unit: Vec<Vec<Vec3>>,
    /// Projected polygon vertices in the same tangent coordinates as the mesh.
    pub cell_polygons_km: Vec<Vec<DVec2>>,
    /// Shared-face midpoint aligned with the mesh's directed CSR edges.
    pub edge_face_midpoint_km: Vec<DVec3>,
    /// Exact directed shared-face endpoints aligned with the mesh's directed
    /// CSR edges and oriented in the owning cell's polygon order.
    pub edge_face_endpoints_km: Vec<[DVec3; 2]>,
    /// Exact directed boundary-face endpoints aligned with
    /// `mesh.boundary_faces` and oriented in the owning cell's polygon order.
    pub boundary_face_endpoints_km: Vec<[DVec3; 2]>,
    pub audit: VoronoiCapAudit,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VoronoiCapError(pub String);

impl fmt::Display for VoronoiCapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for VoronoiCapError {}

/// Build the preregistered guarded irregular S2 cap and its planar FV adapter.
pub fn build_r1_voronoi_cap(
    config: VoronoiCapConfig,
) -> Result<VoronoiCapFixture, VoronoiCapError> {
    validate_config(config)?;
    let sites = cap_sites(config);
    let tessellation = Tessellation::from_points_knn_clipping(sites);
    adapt_cap(config, &tessellation)
}

fn validate_config(config: VoronoiCapConfig) -> Result<(), VoronoiCapError> {
    if ![8.0, 4.0, 2.0].contains(&config.spacing_km) {
        return Err(VoronoiCapError(
            "R1 cap spacing must be one of the registered 8, 4 or 2 km values".into(),
        ));
    }
    if ![R1_CAP_GUARD_SPACINGS, 10].contains(&config.guard_spacings) {
        return Err(VoronoiCapError(
            "R1 cap guard must be the registered eight- or ten-spacing value".into(),
        ));
    }
    Ok(())
}

fn cap_sites(config: VoronoiCapConfig) -> Vec<Vec3> {
    let spacing = config.spacing_km;
    let guard = config.guard_spacings as f64 * spacing;
    let half_width = 0.5 * R1_CAP_WIDTH_KM + guard;
    let half_height = 0.5 * R1_CAP_HEIGHT_KM + guard;
    let row_step = 0.5 * 3.0_f64.sqrt() * spacing;
    let min_r = (-half_height / row_step).floor() as i64 - 1;
    let max_r = (half_height / row_step).ceil() as i64 + 1;
    let mut sites = Vec::new();
    for r in min_r..=max_r {
        let y = r as f64 * row_step;
        if y < -half_height || y > half_height {
            continue;
        }
        let row_offset = 0.5 * r as f64 * spacing;
        let min_q = ((-half_width - row_offset) / spacing).floor() as i64 - 1;
        let max_q = ((half_width - row_offset) / spacing).ceil() as i64 + 1;
        for q in min_q..=max_q {
            let x = q as f64 * spacing + row_offset;
            if x < -half_width || x > half_width {
                continue;
            }
            let seed =
                SITE_SEED ^ (q as u64).wrapping_mul(Q_HASH) ^ (r as u64).wrapping_mul(R_HASH);
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let angle = rng.gen::<f64>() * std::f64::consts::TAU;
            let radius = JITTER_RADIUS_FRACTION * spacing * rng.gen::<f64>().sqrt();
            sites.push(tangent_exp_map(DVec2::new(
                x + radius * angle.cos(),
                y + radius * angle.sin(),
            )));
        }
    }

    let golden_angle = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    for k in 0..FAR_FIELD_SITE_COUNT {
        let z = 1.0 - 2.0 * (k as f64 + 0.5) / FAR_FIELD_SITE_COUNT as f64;
        if z.acos() < FAR_FIELD_NORTH_EXCLUSION_RAD {
            continue;
        }
        let radial = (1.0 - z * z).sqrt();
        let longitude = golden_angle * k as f64;
        sites.push(Vec3::new(
            (radial * longitude.cos()) as f32,
            (radial * longitude.sin()) as f32,
            z as f32,
        ));
    }
    sites
}

fn tangent_exp_map(point_km: DVec2) -> Vec3 {
    let radius = f64::from(PLANET_RADIUS_KM);
    let distance = point_km.length();
    if distance == 0.0 {
        return Vec3::Z;
    }
    let angle = distance / radius;
    let radial = angle.sin() / distance;
    Vec3::new(
        (point_km.x * radial) as f32,
        (point_km.y * radial) as f32,
        angle.cos() as f32,
    )
    .normalize()
}

fn tangent_log_map(point: Vec3) -> Result<DVec2, VoronoiCapError> {
    if !point.is_finite() {
        return Err(VoronoiCapError("non-finite spherical point".into()));
    }
    let z = f64::from(point.z);
    let horizontal = f64::from(point.x).hypot(f64::from(point.y));
    if horizontal <= 1.0e-15 {
        if z > 0.0 {
            return Ok(DVec2::ZERO);
        }
        return Err(VoronoiCapError(
            "south-pole geometry cannot be tangent-projected".into(),
        ));
    }
    // atan2 is equivalent to acos(z) for an exact unit vector but remains
    // well-conditioned near the tangent point when the backend stores the
    // unit sphere in f32.
    let angle = horizontal.atan2(z);
    let scale = f64::from(PLANET_RADIUS_KM) * angle / horizontal;
    Ok(DVec2::new(
        f64::from(point.x) * scale,
        f64::from(point.y) * scale,
    ))
}

fn adapt_cap(
    config: VoronoiCapConfig,
    tessellation: &Tessellation,
) -> Result<VoronoiCapFixture, VoronoiCapError> {
    let half_width = 0.5 * R1_CAP_WIDTH_KM;
    let half_height = 0.5 * R1_CAP_HEIGHT_KM;
    let mut generator_xy = Vec::with_capacity(tessellation.num_cells());
    for &generator in &tessellation.voronoi.generators {
        generator_xy.push(tangent_log_map(generator)?);
    }
    let retained_old: Vec<usize> = generator_xy
        .iter()
        .enumerate()
        .filter_map(|(cell, point)| {
            (point.x >= -half_width
                && point.x <= half_width
                && point.y >= -half_height
                && point.y <= half_height)
                .then_some(cell)
        })
        .collect();
    if retained_old.is_empty() {
        return Err(VoronoiCapError("cap retained no cells".into()));
    }
    let mut old_to_new = vec![None; tessellation.num_cells()];
    for (new, &old) in retained_old.iter().enumerate() {
        old_to_new[old] = Some(new);
    }

    let mut vertex_xy = vec![None; tessellation.voronoi.vertices.len()];
    for &old in &retained_old {
        for &vertex in tessellation.voronoi.cell(old).vertex_indices {
            let slot = &mut vertex_xy[vertex as usize];
            if slot.is_none() {
                *slot = Some(tangent_log_map(
                    tessellation.voronoi.vertices[vertex as usize],
                )?);
            }
        }
    }

    let mut edge_owners: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    for (old, retained_index) in old_to_new.iter().enumerate() {
        let vertices = tessellation.voronoi.cell(old).vertex_indices;
        if vertices.len() < 3 {
            if retained_index.is_some() {
                return Err(VoronoiCapError(format!(
                    "retained cell {old} has fewer than three vertices"
                )));
            }
            continue;
        }
        for edge in polygon_edges(vertices) {
            edge_owners.entry(edge).or_default().push(old);
        }
    }

    let spherical_areas = tessellation.cell_areas_ref();
    let mut cell_center_km = Vec::with_capacity(retained_old.len());
    let mut cell_center_unit = Vec::with_capacity(retained_old.len());
    let mut cell_area_km2 = Vec::with_capacity(retained_old.len());
    let mut cell_polygons_unit = Vec::with_capacity(retained_old.len());
    let mut cell_polygons_km = Vec::with_capacity(retained_old.len());
    let mut generator_centroid_offsets_km = Vec::with_capacity(retained_old.len());
    let mut total_spherical_area_km2 = 0.0;
    let mut min_cell_area_km2 = f64::INFINITY;
    let mut max_cell_area_projection_relative_error = 0.0_f64;
    for &old in &retained_old {
        let center = generator_xy[old];
        cell_center_km.push(center.extend(0.0));
        cell_center_unit.push(tessellation.voronoi.generators[old]);
        let polygon_unit: Vec<_> = tessellation
            .voronoi
            .cell(old)
            .vertex_indices
            .iter()
            .map(|&vertex| tessellation.voronoi.vertices[vertex as usize])
            .collect();
        let polygon: Vec<_> = tessellation
            .voronoi
            .cell(old)
            .vertex_indices
            .iter()
            .map(|&vertex| {
                vertex_xy[vertex as usize]
                    .ok_or_else(|| VoronoiCapError("retained vertex was not projected".into()))
            })
            .collect::<Result<_, _>>()?;
        let area = polygon_area(&polygon);
        if !area.is_finite() || area <= 0.0 {
            return Err(VoronoiCapError(format!(
                "retained cell {old} has invalid projected area {area}"
            )));
        }
        min_cell_area_km2 = min_cell_area_km2.min(area);
        cell_area_km2.push(area);
        generator_centroid_offsets_km.push(center.distance(polygon_centroid(&polygon)?));
        cell_polygons_unit.push(polygon_unit);
        cell_polygons_km.push(polygon);
        let spherical_area_km2 =
            f64::from(spherical_areas[old]) * f64::from(PLANET_RADIUS_KM).powi(2);
        max_cell_area_projection_relative_error = max_cell_area_projection_relative_error
            .max((area - spherical_area_km2).abs() / spherical_area_km2.max(f64::MIN_POSITIVE));
        total_spherical_area_km2 += spherical_area_km2;
    }

    let mut edge_offsets = Vec::with_capacity(retained_old.len() + 1);
    let mut edge_neighbor = Vec::new();
    let mut edge_distance_km = Vec::new();
    let mut edge_face_width_km = Vec::new();
    let mut edge_outward_tangent = Vec::new();
    let mut edge_face_midpoint_km = Vec::new();
    let mut edge_face_endpoints_km = Vec::new();
    let mut boundary_faces = Vec::new();
    let mut boundary_face_endpoints_km = Vec::new();
    let mut boundary = vec![BoundaryCondition::Interior; retained_old.len()];
    let mut min_internal_face_width_km = f64::INFINITY;
    let mut max_internal_face_width_km = 0.0_f64;
    let mut internal_face_widths_km = Vec::new();
    let mut internal_pair_edge = HashMap::new();
    let mut max_edge_projection_relative_error = 0.0_f64;
    let mut max_center_distance_projection_relative_error = 0.0_f64;
    let mut max_face_midpoint_projection_error_km = 0.0_f64;
    let mut internal_directed_faces = 0usize;
    let mut portal_faces = 0usize;

    for (new, &old) in retained_old.iter().enumerate() {
        edge_offsets.push(edge_neighbor.len() as u32);
        let vertices = tessellation.voronoi.cell(old).vertex_indices;
        for (va, vb) in polygon_edges_oriented(vertices) {
            let key = canonical_edge(va, vb);
            let owners = edge_owners.get(&key).ok_or_else(|| {
                VoronoiCapError(format!("missing ownership for Voronoi edge {key:?}"))
            })?;
            if owners.len() != 2 {
                return Err(VoronoiCapError(format!(
                    "retained edge {key:?} has {} owners",
                    owners.len()
                )));
            }
            let neighbor_old = owners
                .iter()
                .copied()
                .find(|&owner| owner != old)
                .ok_or_else(|| VoronoiCapError("edge owner is not distinct".into()))?;
            let a = vertex_xy[va as usize]
                .ok_or_else(|| VoronoiCapError("retained edge vertex was not projected".into()))?;
            let b = vertex_xy[vb as usize]
                .ok_or_else(|| VoronoiCapError("retained edge vertex was not projected".into()))?;
            let midpoint = 0.5 * (a + b);
            let spherical_midpoint_unit = spherical_midpoint(
                tessellation.voronoi.vertices[va as usize],
                tessellation.voronoi.vertices[vb as usize],
            )?;
            let spherical_midpoint_xy = tangent_log_map(spherical_midpoint_unit)?;
            max_face_midpoint_projection_error_km =
                max_face_midpoint_projection_error_km.max(midpoint.distance(spherical_midpoint_xy));
            let width = a.distance(b);
            if !width.is_finite() || width <= 0.0 {
                return Err(VoronoiCapError("invalid projected face width".into()));
            }
            let spherical_width = spherical_arc_km(
                tessellation.voronoi.vertices[va as usize],
                tessellation.voronoi.vertices[vb as usize],
            );
            let projection_error =
                (width - spherical_width).abs() / spherical_width.max(f64::MIN_POSITIVE);
            max_edge_projection_relative_error =
                max_edge_projection_relative_error.max(projection_error);

            if let Some(neighbor_new) = old_to_new[neighbor_old] {
                let owner_pair = if old < neighbor_old {
                    (old, neighbor_old)
                } else {
                    (neighbor_old, old)
                };
                if let Some(&previous_edge) = internal_pair_edge.get(&owner_pair) {
                    if previous_edge != key {
                        return Err(VoronoiCapError(format!(
                            "retained cell pair {owner_pair:?} owns multiple Voronoi faces"
                        )));
                    }
                } else {
                    internal_pair_edge.insert(owner_pair, key);
                }
                let delta = generator_xy[neighbor_old] - generator_xy[old];
                let distance = delta.length();
                if !distance.is_finite() || distance <= 0.0 {
                    return Err(VoronoiCapError("invalid projected center distance".into()));
                }
                edge_neighbor.push(neighbor_new as u32);
                edge_distance_km.push(distance as f32);
                edge_face_width_km.push(width as f32);
                edge_outward_tangent.push(delta.extend(0.0).normalize().as_vec3());
                edge_face_midpoint_km.push(midpoint.extend(0.0));
                edge_face_endpoints_km.push([a.extend(0.0), b.extend(0.0)]);
                min_internal_face_width_km = min_internal_face_width_km.min(width);
                max_internal_face_width_km = max_internal_face_width_km.max(width);
                // Store each physical face once even though the mesh CSR is directed.
                if old < neighbor_old {
                    internal_face_widths_km.push(width);
                }
                let spherical_distance = spherical_arc_km(
                    tessellation.voronoi.generators[old],
                    tessellation.voronoi.generators[neighbor_old],
                );
                max_center_distance_projection_relative_error =
                    max_center_distance_projection_relative_error.max(
                        (distance - spherical_distance).abs()
                            / spherical_distance.max(f64::MIN_POSITIVE),
                    );
                internal_directed_faces += 1;
            } else {
                let neighbor = generator_xy[neighbor_old];
                let side = boundary_side(neighbor, half_width, half_height);
                let span = match side {
                    BoundarySide::North | BoundarySide::South => (a.x.min(b.x), a.x.max(b.x)),
                    BoundarySide::East | BoundarySide::West => (a.y.min(b.y), a.y.max(b.y)),
                };
                if span.1 - span.0 <= 1.0e-12 {
                    return Err(VoronoiCapError("degenerate projected boundary span".into()));
                }
                let is_portal = neighbor.y < -half_height
                    && midpoint.x >= -R1_CAP_PORTAL_HALF_WIDTH_KM
                    && midpoint.x <= R1_CAP_PORTAL_HALF_WIDTH_KM;
                let condition = if is_portal {
                    portal_faces += 1;
                    boundary[new] = BoundaryCondition::OpenBaseLevel {
                        elevation_km: R1_CAP_PORTAL_BASE_LEVEL_KM,
                    };
                    BoundaryFaceCondition::OpenBaseLevel {
                        portal_id: R1_CAP_PORTAL_ID,
                        elevation_km: R1_CAP_PORTAL_BASE_LEVEL_KM,
                    }
                } else {
                    if !matches!(boundary[new], BoundaryCondition::OpenBaseLevel { .. }) {
                        boundary[new] = BoundaryCondition::Closed;
                    }
                    BoundaryFaceCondition::Closed
                };
                let outward = neighbor - generator_xy[old];
                let center_distance_km = generator_xy[old].distance(midpoint);
                let spherical_center_distance = spherical_arc_km(
                    tessellation.voronoi.generators[old],
                    spherical_midpoint_unit,
                );
                max_center_distance_projection_relative_error =
                    max_center_distance_projection_relative_error.max(
                        (center_distance_km - spherical_center_distance).abs()
                            / spherical_center_distance.max(f64::MIN_POSITIVE),
                    );
                boundary_faces.push(LandscapeBoundaryFace {
                    cell: new as u32,
                    side,
                    center_km: midpoint.extend(0.0),
                    outward_normal: outward.extend(0.0).normalize(),
                    width_km: width,
                    projected_span_start_km: span.0,
                    projected_span_end_km: span.1,
                    center_distance_km,
                    condition,
                });
                boundary_face_endpoints_km.push([a.extend(0.0), b.extend(0.0)]);
            }
        }
    }
    edge_offsets.push(edge_neighbor.len() as u32);
    if portal_faces == 0 {
        return Err(VoronoiCapError("cap produced no portal faces".into()));
    }
    if !internal_directed_faces.is_multiple_of(2) {
        return Err(VoronoiCapError(
            "internal directed face count is not reciprocal".into(),
        ));
    }
    if internal_pair_edge.len() * 2 != internal_directed_faces {
        return Err(VoronoiCapError(
            "internal cell-pair and directed-face counts disagree".into(),
        ));
    }

    let mesh = LandscapeMesh {
        cell_center_km,
        cell_area_km2,
        edge_offsets,
        edge_neighbor,
        edge_distance_km,
        edge_face_width_km,
        edge_outward_tangent,
        boundary,
        boundary_faces,
        outlet_portals: vec![OutletPortal {
            id: R1_CAP_PORTAL_ID,
            side: BoundarySide::South,
            span_start_km: -R1_CAP_PORTAL_HALF_WIDTH_KM,
            span_end_km: R1_CAP_PORTAL_HALF_WIDTH_KM,
            base_level_km: R1_CAP_PORTAL_BASE_LEVEL_KM,
        }],
    };
    mesh.validate()
        .map_err(|error| VoronoiCapError(error.to_string()))?;
    let total_projected_area_km2: f64 = mesh.cell_area_km2.iter().sum();
    let total_area_projection_relative_error =
        (total_projected_area_km2 - total_spherical_area_km2).abs()
            / total_spherical_area_km2.max(f64::MIN_POSITIVE);
    let audit = VoronoiCapAudit {
        input_site_count: tessellation.num_cells(),
        retained_cell_count: mesh.cell_count(),
        internal_undirected_faces: internal_pair_edge.len(),
        boundary_faces: mesh.boundary_faces.len(),
        portal_faces,
        min_cell_area_km2,
        cell_area_p10_km2: quantile(&mesh.cell_area_km2, 0.10)?,
        cell_area_p50_km2: quantile(&mesh.cell_area_km2, 0.50)?,
        cell_area_p90_km2: quantile(&mesh.cell_area_km2, 0.90)?,
        min_internal_face_width_km,
        internal_face_width_p10_km: quantile(&internal_face_widths_km, 0.10)?,
        internal_face_width_p50_km: quantile(&internal_face_widths_km, 0.50)?,
        internal_face_width_p90_km: quantile(&internal_face_widths_km, 0.90)?,
        max_internal_face_width_km,
        generator_centroid_offset_p50_km: quantile(&generator_centroid_offsets_km, 0.50)?,
        generator_centroid_offset_p95_km: quantile(&generator_centroid_offsets_km, 0.95)?,
        max_edge_projection_relative_error,
        max_center_distance_projection_relative_error,
        max_face_midpoint_projection_error_km,
        max_cell_area_projection_relative_error,
        total_projected_area_km2,
        total_spherical_area_km2,
        total_area_projection_relative_error,
    };
    Ok(VoronoiCapFixture {
        config,
        mesh,
        cell_center_unit,
        cell_polygons_unit,
        cell_polygons_km,
        edge_face_midpoint_km,
        edge_face_endpoints_km,
        boundary_face_endpoints_km,
        audit,
    })
}

fn polygon_edges(vertices: &[u32]) -> impl Iterator<Item = (u32, u32)> + '_ {
    polygon_edges_oriented(vertices).map(|(a, b)| canonical_edge(a, b))
}

fn polygon_edges_oriented(vertices: &[u32]) -> impl Iterator<Item = (u32, u32)> + '_ {
    vertices
        .iter()
        .copied()
        .zip(vertices.iter().copied().cycle().skip(1))
        .take(vertices.len())
}

fn canonical_edge(a: u32, b: u32) -> (u32, u32) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

fn polygon_area(vertices: &[DVec2]) -> f64 {
    0.5 * vertices
        .iter()
        .copied()
        .zip(vertices.iter().copied().cycle().skip(1))
        .take(vertices.len())
        .map(|(a, b)| a.perp_dot(b))
        .sum::<f64>()
        .abs()
}

fn polygon_centroid(vertices: &[DVec2]) -> Result<DVec2, VoronoiCapError> {
    let cross_sum: f64 = vertices
        .iter()
        .copied()
        .zip(vertices.iter().copied().cycle().skip(1))
        .take(vertices.len())
        .map(|(a, b)| a.perp_dot(b))
        .sum();
    if !cross_sum.is_finite() || cross_sum.abs() <= f64::MIN_POSITIVE {
        return Err(VoronoiCapError(
            "cannot compute centroid of degenerate polygon".into(),
        ));
    }
    let weighted_sum = vertices
        .iter()
        .copied()
        .zip(vertices.iter().copied().cycle().skip(1))
        .take(vertices.len())
        .fold(DVec2::ZERO, |sum, (a, b)| sum + (a + b) * a.perp_dot(b));
    Ok(weighted_sum / (3.0 * cross_sum))
}

fn quantile(values: &[f64], probability: f64) -> Result<f64, VoronoiCapError> {
    if values.is_empty() {
        return Err(VoronoiCapError("cannot quantify an empty sample".into()));
    }
    let mut sorted = values.to_vec();
    if sorted.iter().any(|value| !value.is_finite()) {
        return Err(VoronoiCapError("non-finite audit sample".into()));
    }
    sorted.sort_by(f64::total_cmp);
    let position = probability.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let fraction = position - lower as f64;
    Ok(sorted[lower] + fraction * (sorted[upper] - sorted[lower]))
}

fn spherical_arc_km(a: Vec3, b: Vec3) -> f64 {
    let a = DVec3::new(f64::from(a.x), f64::from(a.y), f64::from(a.z)).normalize();
    let b = DVec3::new(f64::from(b.x), f64::from(b.y), f64::from(b.z)).normalize();
    let chord = (a - b).length().clamp(0.0, 2.0);
    2.0 * (0.5 * chord).asin() * f64::from(PLANET_RADIUS_KM)
}

fn spherical_midpoint(a: Vec3, b: Vec3) -> Result<Vec3, VoronoiCapError> {
    let sum = a.normalize() + b.normalize();
    if !sum.is_finite() || sum.length_squared() <= 1.0e-20 {
        return Err(VoronoiCapError(
            "cannot construct midpoint of antipodal face vertices".into(),
        ));
    }
    Ok(sum.normalize())
}

fn boundary_side(point: DVec2, half_width: f64, half_height: f64) -> BoundarySide {
    let candidates = [
        ((-point.y - half_height).max(0.0), BoundarySide::South),
        ((point.y - half_height).max(0.0), BoundarySide::North),
        ((-point.x - half_width).max(0.0), BoundarySide::West),
        ((point.x - half_width).max(0.0), BoundarySide::East),
    ];
    candidates
        .into_iter()
        .max_by(|(a, _), (b, _)| a.total_cmp(b))
        .map(|(_, side)| side)
        .unwrap_or(BoundarySide::South)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cmp_dvec3(a: &DVec3, b: &DVec3) -> std::cmp::Ordering {
        a.x.total_cmp(&b.x)
            .then(a.y.total_cmp(&b.y))
            .then(a.z.total_cmp(&b.z))
    }

    fn cmp_dvec2(a: &DVec2, b: &DVec2) -> std::cmp::Ordering {
        a.x.total_cmp(&b.x).then(a.y.total_cmp(&b.y))
    }

    fn cmp_vec3(a: &Vec3, b: &Vec3) -> std::cmp::Ordering {
        a.x.total_cmp(&b.x)
            .then(a.y.total_cmp(&b.y))
            .then(a.z.total_cmp(&b.z))
    }

    fn assert_g0_geometry_gates(fixture: &VoronoiCapFixture) {
        fixture.mesh.validate().unwrap();
        assert_eq!(
            fixture.audit.internal_undirected_faces * 2,
            fixture.mesh.edge_neighbor.len()
        );
        assert_eq!(
            fixture.edge_face_midpoint_km.len(),
            fixture.mesh.edge_neighbor.len()
        );
        assert_eq!(
            fixture.edge_face_endpoints_km.len(),
            fixture.mesh.edge_neighbor.len()
        );
        assert_eq!(
            fixture.boundary_face_endpoints_km.len(),
            fixture.mesh.boundary_faces.len()
        );
        assert_eq!(fixture.cell_polygons_km.len(), fixture.mesh.cell_count());
        assert_eq!(fixture.cell_polygons_unit.len(), fixture.mesh.cell_count());
        assert_eq!(fixture.cell_center_unit.len(), fixture.mesh.cell_count());
        assert!(fixture.audit.portal_faces > 0);
        assert!(fixture.audit.min_cell_area_km2 > 0.0);
        assert!(fixture.audit.cell_area_p90_km2 > fixture.audit.cell_area_p10_km2);
        assert!(fixture.audit.min_internal_face_width_km > 0.0);
        assert!(
            fixture.audit.internal_face_width_p90_km > fixture.audit.internal_face_width_p10_km
        );
        assert!(
            fixture.audit.max_internal_face_width_km > fixture.audit.min_internal_face_width_km
        );
        assert!(fixture.audit.generator_centroid_offset_p50_km.is_finite());
        assert!(
            fixture.audit.generator_centroid_offset_p95_km
                >= fixture.audit.generator_centroid_offset_p50_km
        );
        assert!(
            fixture.audit.max_edge_projection_relative_error < 1.0e-3,
            "max edge projection error: {:#?}",
            fixture.audit
        );
        assert!(fixture
            .audit
            .max_center_distance_projection_relative_error
            .is_finite());
        assert!(fixture
            .audit
            .max_face_midpoint_projection_error_km
            .is_finite());
        assert!(fixture
            .audit
            .max_cell_area_projection_relative_error
            .is_finite());
        assert!(
            fixture.audit.total_area_projection_relative_error < 2.0e-3,
            "total area projection error: {:#?}",
            fixture.audit
        );
        assert!(fixture.cell_polygons_km.iter().all(|polygon| {
            polygon.len() >= 3 && polygon_area(polygon).is_finite() && polygon_area(polygon) > 0.0
        }));
        assert!(fixture
            .cell_center_unit
            .iter()
            .all(|center| (center.length() - 1.0).abs() <= 1.0e-6));

        let mut directed_edges = HashMap::new();
        for cell in 0..fixture.mesh.cell_count() {
            let polygon = &fixture.cell_polygons_km[cell];
            for edge in fixture.mesh.edge_offsets[cell] as usize
                ..fixture.mesh.edge_offsets[cell + 1] as usize
            {
                let endpoints = fixture.edge_face_endpoints_km[edge];
                assert_eq!(
                    fixture.edge_face_midpoint_km[edge],
                    0.5 * (endpoints[0] + endpoints[1])
                );
                let endpoint_width = endpoints[0].distance(endpoints[1]);
                let stored_width = f64::from(fixture.mesh.edge_face_width_km[edge]);
                assert!(
                    (endpoint_width - stored_width).abs()
                        <= 2.0 * f64::from(f32::EPSILON) * endpoint_width.max(f64::MIN_POSITIVE)
                );
                assert!(polygon
                    .iter()
                    .copied()
                    .zip(polygon.iter().copied().cycle().skip(1))
                    .take(polygon.len())
                    .any(|(a, b)| endpoints == [a.extend(0.0), b.extend(0.0)]));
                let neighbor = fixture.mesh.edge_neighbor[edge] as usize;
                assert!(directed_edges.insert((cell, neighbor), edge).is_none());
            }
        }
        for (&(cell, neighbor), &edge) in &directed_edges {
            let reciprocal = directed_edges[&(neighbor, cell)];
            let endpoints = fixture.edge_face_endpoints_km[edge];
            let reciprocal_endpoints = fixture.edge_face_endpoints_km[reciprocal];
            assert_eq!(
                endpoints,
                [reciprocal_endpoints[1], reciprocal_endpoints[0]]
            );
        }

        for (face, &endpoints) in fixture
            .mesh
            .boundary_faces
            .iter()
            .zip(&fixture.boundary_face_endpoints_km)
        {
            assert_eq!(face.center_km, 0.5 * (endpoints[0] + endpoints[1]));
            assert!((face.width_km - endpoints[0].distance(endpoints[1])).abs() <= 1.0e-12);
            let polygon = &fixture.cell_polygons_km[face.cell as usize];
            assert!(polygon
                .iter()
                .copied()
                .zip(polygon.iter().copied().cycle().skip(1))
                .take(polygon.len())
                .any(|(a, b)| endpoints == [a.extend(0.0), b.extend(0.0)]));
        }
    }

    fn assert_guard_invariant(eight: &VoronoiCapFixture, ten: &VoronoiCapFixture, spacing_km: f64) {
        let length_tolerance = 1.0e-3 * spacing_km;
        // Area is two-dimensional, so use the length gate times the nominal
        // cell scale. Vertex-by-vertex length checks independently constrain
        // the polygon geometry.
        let area_tolerance = length_tolerance * spacing_km;
        assert_eq!(eight.mesh.cell_count(), ten.mesh.cell_count());
        assert_eq!(
            eight.audit.internal_undirected_faces,
            ten.audit.internal_undirected_faces
        );
        assert_eq!(
            eight.mesh.boundary_faces.len(),
            ten.mesh.boundary_faces.len()
        );

        let mut eight_cells: Vec<_> = (0..eight.mesh.cell_count()).collect();
        let mut ten_cells: Vec<_> = (0..ten.mesh.cell_count()).collect();
        eight_cells.sort_by(|&a, &b| {
            cmp_dvec3(&eight.mesh.cell_center_km[a], &eight.mesh.cell_center_km[b])
        });
        ten_cells
            .sort_by(|&a, &b| cmp_dvec3(&ten.mesh.cell_center_km[a], &ten.mesh.cell_center_km[b]));

        for (&a, &b) in eight_cells.iter().zip(&ten_cells) {
            assert!(
                eight.mesh.cell_center_km[a].distance(ten.mesh.cell_center_km[b])
                    <= length_tolerance
            );
            assert!(
                eight.cell_center_unit[a].distance(ten.cell_center_unit[b]) <= 1.0e-6,
                "unit generator changed at spacing {spacing_km}"
            );
            assert!(
                (eight.mesh.cell_area_km2[a] - ten.mesh.cell_area_km2[b]).abs() <= area_tolerance
            );

            let mut polygon_a = eight.cell_polygons_km[a].clone();
            let mut polygon_b = ten.cell_polygons_km[b].clone();
            polygon_a.sort_by(cmp_dvec2);
            polygon_b.sort_by(cmp_dvec2);
            assert_eq!(polygon_a.len(), polygon_b.len());
            for (va, vb) in polygon_a.iter().zip(&polygon_b) {
                assert!(va.distance(*vb) <= length_tolerance);
            }

            let mut polygon_unit_a = eight.cell_polygons_unit[a].clone();
            let mut polygon_unit_b = ten.cell_polygons_unit[b].clone();
            polygon_unit_a.sort_by(cmp_vec3);
            polygon_unit_b.sort_by(cmp_vec3);
            assert_eq!(polygon_unit_a.len(), polygon_unit_b.len());
            for (va, vb) in polygon_unit_a.iter().zip(&polygon_unit_b) {
                assert!(va.distance(*vb) <= 1.0e-6);
            }

            let start_a = eight.mesh.edge_offsets[a] as usize;
            let end_a = eight.mesh.edge_offsets[a + 1] as usize;
            let start_b = ten.mesh.edge_offsets[b] as usize;
            let end_b = ten.mesh.edge_offsets[b + 1] as usize;
            let mut edges_a: Vec<_> = (start_a..end_a).collect();
            let mut edges_b: Vec<_> = (start_b..end_b).collect();
            edges_a.sort_by(|&ea, &eb| {
                cmp_dvec3(
                    &eight.mesh.cell_center_km[eight.mesh.edge_neighbor[ea] as usize],
                    &eight.mesh.cell_center_km[eight.mesh.edge_neighbor[eb] as usize],
                )
            });
            edges_b.sort_by(|&ea, &eb| {
                cmp_dvec3(
                    &ten.mesh.cell_center_km[ten.mesh.edge_neighbor[ea] as usize],
                    &ten.mesh.cell_center_km[ten.mesh.edge_neighbor[eb] as usize],
                )
            });
            assert_eq!(edges_a.len(), edges_b.len());
            for (&ea, &eb) in edges_a.iter().zip(&edges_b) {
                let neighbor_a = eight.mesh.edge_neighbor[ea] as usize;
                let neighbor_b = ten.mesh.edge_neighbor[eb] as usize;
                assert!(
                    eight.mesh.cell_center_km[neighbor_a]
                        .distance(ten.mesh.cell_center_km[neighbor_b])
                        <= length_tolerance
                );
                assert!(
                    (f64::from(eight.mesh.edge_distance_km[ea])
                        - f64::from(ten.mesh.edge_distance_km[eb]))
                    .abs()
                        <= length_tolerance
                );
                assert!(
                    (f64::from(eight.mesh.edge_face_width_km[ea])
                        - f64::from(ten.mesh.edge_face_width_km[eb]))
                    .abs()
                        <= length_tolerance
                );
                assert!(
                    eight.edge_face_midpoint_km[ea].distance(ten.edge_face_midpoint_km[eb])
                        <= length_tolerance
                );
                for endpoint in 0..2 {
                    assert!(
                        eight.edge_face_endpoints_km[ea][endpoint]
                            .distance(ten.edge_face_endpoints_km[eb][endpoint])
                            <= length_tolerance
                    );
                }
                assert!(
                    eight.mesh.edge_outward_tangent[ea].distance(ten.mesh.edge_outward_tangent[eb])
                        <= 1.0e-6
                );
            }
        }

        let boundary_key = |fixture: &VoronoiCapFixture, face: &LandscapeBoundaryFace| {
            (
                fixture.mesh.cell_center_km[face.cell as usize],
                face.center_km,
            )
        };
        let mut boundary_a: Vec<_> = eight.mesh.boundary_faces.iter().collect();
        let mut boundary_b: Vec<_> = ten.mesh.boundary_faces.iter().collect();
        boundary_a.sort_by(|a, b| {
            let (cell_a, face_a) = boundary_key(eight, a);
            let (cell_b, face_b) = boundary_key(eight, b);
            cmp_dvec3(&cell_a, &cell_b).then(cmp_dvec3(&face_a, &face_b))
        });
        boundary_b.sort_by(|a, b| {
            let (cell_a, face_a) = boundary_key(ten, a);
            let (cell_b, face_b) = boundary_key(ten, b);
            cmp_dvec3(&cell_a, &cell_b).then(cmp_dvec3(&face_a, &face_b))
        });
        for (a, b) in boundary_a.iter().zip(&boundary_b) {
            let index_a = eight
                .mesh
                .boundary_faces
                .iter()
                .position(|face| std::ptr::eq(face, *a))
                .unwrap();
            let index_b = ten
                .mesh
                .boundary_faces
                .iter()
                .position(|face| std::ptr::eq(face, *b))
                .unwrap();
            assert!(
                eight.mesh.cell_center_km[a.cell as usize]
                    .distance(ten.mesh.cell_center_km[b.cell as usize])
                    <= length_tolerance
            );
            assert_eq!(a.side, b.side);
            assert_eq!(a.condition, b.condition);
            assert!(a.center_km.distance(b.center_km) <= length_tolerance);
            assert!(a.outward_normal.distance(b.outward_normal) <= 1.0e-6);
            assert!((a.width_km - b.width_km).abs() <= length_tolerance);
            assert!((a.center_distance_km - b.center_distance_km).abs() <= length_tolerance);
            assert!(
                (a.projected_span_start_km - b.projected_span_start_km).abs() <= length_tolerance
            );
            assert!((a.projected_span_end_km - b.projected_span_end_km).abs() <= length_tolerance);
            for endpoint in 0..2 {
                assert!(
                    eight.boundary_face_endpoints_km[index_a][endpoint]
                        .distance(ten.boundary_face_endpoints_km[index_b][endpoint])
                        <= length_tolerance
                );
            }
        }
    }

    #[test]
    fn r1_cap_builds_valid_irregular_eight_km_mesh() {
        let fixture = build_r1_voronoi_cap(VoronoiCapConfig::r1(8.0)).unwrap();
        assert_g0_geometry_gates(&fixture);
    }

    #[test]
    #[ignore = "full 8/4/2 km guard matrix takes roughly two minutes in debug"]
    fn r1_cap_passes_g0_geometry_and_guard_gates_at_all_spacings() {
        for spacing_km in [8.0, 4.0, 2.0] {
            let eight = build_r1_voronoi_cap(VoronoiCapConfig::r1(spacing_km)).unwrap();
            let rebuild = build_r1_voronoi_cap(VoronoiCapConfig::r1(spacing_km)).unwrap();
            let ten = build_r1_voronoi_cap(VoronoiCapConfig {
                spacing_km,
                guard_spacings: 10,
            })
            .unwrap();
            assert_eq!(eight, rebuild);
            assert_g0_geometry_gates(&eight);
            assert_g0_geometry_gates(&ten);
            assert_guard_invariant(&eight, &ten, spacing_km);
            eprintln!("spacing {spacing_km} km, guard 8: {:#?}", eight.audit);
            eprintln!("spacing {spacing_km} km, guard 10: {:#?}", ten.audit);
        }
    }

    #[test]
    fn r1_cap_rebuild_is_bit_deterministic() {
        let a = build_r1_voronoi_cap(VoronoiCapConfig::r1(8.0)).unwrap();
        let b = build_r1_voronoi_cap(VoronoiCapConfig::r1(8.0)).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn r1_cap_rejects_unregistered_geometry() {
        assert!(build_r1_voronoi_cap(VoronoiCapConfig::r1(3.0)).is_err());
        assert!(build_r1_voronoi_cap(VoronoiCapConfig {
            spacing_km: 4.0,
            guard_spacings: 9,
        })
        .is_err());
    }
}
