//! Arm-neutral physical surface graph and peak-saddle hierarchy.
//!
//! This is a partial G0/S0 slice of the packet preregistered in
//! `docs/research/landform-object-packet-g0s0-2026-07-14.md`. It implements the
//! common graph types, explicit planar and product-spherical G0 adapters, the
//! exact-level peak-saddle split forest and planar and spherical reference-
//! highland measurements. Inputs are limited to physical geometry, elevation,
//! a scored mask and the frozen extractor configuration.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::fmt;

use bincode::Options;
use glam::{DVec2, DVec3};
use serde::{Deserialize, Serialize};

use super::landscape::{
    BoundaryFaceCondition, LandscapeMesh, LandscapeMeshError, OutletPortalId, VoronoiCapFixture,
};
use super::{Tessellation, PLANET_RADIUS_KM};

pub const G0S0_SCHEMA_VERSION: &str = "landform-g0s0-v0";
pub const G0S0_HASH_VERSION: &str = "fnv1a64-bincode-fixint-le-v0";

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum EvaluationDomainV0 {
    Planar,
    Spherical { radius_km: f64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum EvaluationBoundaryConditionV0 {
    Closed,
    OpenBaseLevel { portal_id: u32, elevation_km: f64 },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvaluationBoundarySegmentV0 {
    pub id: u32,
    pub owner_cell: u32,
    /// Directed in the owning cell's counter-clockwise polygon order.
    pub endpoints_km: [DVec3; 2],
    pub physical_length_km: f64,
    pub projected_span_km: Option<[f64; 2]>,
    pub condition: EvaluationBoundaryConditionV0,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvaluationSurfaceGraphV0 {
    pub domain: EvaluationDomainV0,
    pub cell_center_km: Vec<DVec3>,
    pub cell_area_km2: Vec<f64>,
    pub cell_polygon_offsets: Vec<u32>,
    pub cell_polygon_vertices_km: Vec<DVec3>,
    pub edge_offsets: Vec<u32>,
    pub edge_neighbor: Vec<u32>,
    pub edge_reciprocal: Vec<u32>,
    pub edge_distance_km: Vec<f64>,
    pub edge_shared_width_km: Vec<f64>,
    pub edge_face_endpoints_km: Vec<[DVec3; 2]>,
    pub boundary_segments: Vec<EvaluationBoundarySegmentV0>,
}

impl EvaluationSurfaceGraphV0 {
    pub fn cell_count(&self) -> usize {
        self.cell_center_km.len()
    }

    pub fn polygon(&self, cell: usize) -> &[DVec3] {
        let start = self.cell_polygon_offsets[cell] as usize;
        let end = self.cell_polygon_offsets[cell + 1] as usize;
        &self.cell_polygon_vertices_km[start..end]
    }

    pub fn validate(&self, config: &SurfaceHierarchyConfigV0) -> Result<(), LandformError> {
        validate_graph(self, config)
    }
}

/// Explicit control-volume geometry required to adapt a general planar
/// [`LandscapeMesh`]. Arrays of face endpoints align with the mesh's native
/// directed CSR and boundary-face arrays.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LandscapeControlVolumesV0 {
    pub cell_polygon_offsets: Vec<u32>,
    pub cell_polygon_vertices_km: Vec<DVec3>,
    pub edge_face_endpoints_km: Vec<[DVec3; 2]>,
    pub boundary_face_endpoints_km: Vec<[DVec3; 2]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SurfaceHierarchyConfigV0 {
    pub closure_level_km: f64,
    pub reference_persistence_km: f64,
    pub reference_min_footprint_km2: f64,
    pub persistence_sensitivity_km: [f64; 2],
    pub footprint_sensitivity_km2: [f64; 2],
    pub local_relief_radii_km: [f64; 3],
    pub summit_cap_depths_km: [f64; 3],
    pub gentle_grade_thresholds: [f64; 3],
    pub endpoint_match_abs_km: f64,
    pub planar_area_match_relative: f64,
    pub sphere_area_closure_relative: f64,
    pub linear_rank_relative: f64,
    pub orientation_ambiguity_anisotropy: f64,
    pub spherical_nonlocal_radius_rad: f64,
    pub schema_version: &'static str,
    pub hash_version: &'static str,
}

impl Default for SurfaceHierarchyConfigV0 {
    fn default() -> Self {
        Self {
            closure_level_km: 0.0,
            reference_persistence_km: 0.10,
            reference_min_footprint_km2: 2_500.0,
            persistence_sensitivity_km: [0.05, 0.20],
            footprint_sensitivity_km2: [1_250.0, 5_000.0],
            local_relief_radii_km: [25.0, 50.0, 100.0],
            summit_cap_depths_km: [0.25, 0.50, 1.00],
            gentle_grade_thresholds: [0.005, 0.010, 0.020],
            endpoint_match_abs_km: 1.0e-8,
            planar_area_match_relative: 1.0e-10,
            sphere_area_closure_relative: 1.0e-6,
            linear_rank_relative: 1.0e-12,
            orientation_ambiguity_anisotropy: 0.10,
            spherical_nonlocal_radius_rad: 0.50,
            schema_version: G0S0_SCHEMA_VERSION,
            hash_version: G0S0_HASH_VERSION,
        }
    }
}

impl SurfaceHierarchyConfigV0 {
    fn validate(&self) -> Result<(), LandformError> {
        let registered = Self::default();
        let finite = [
            self.closure_level_km,
            self.reference_persistence_km,
            self.reference_min_footprint_km2,
            self.endpoint_match_abs_km,
            self.planar_area_match_relative,
            self.sphere_area_closure_relative,
            self.linear_rank_relative,
            self.orientation_ambiguity_anisotropy,
            self.spherical_nonlocal_radius_rad,
        ]
        .into_iter()
        .chain(self.persistence_sensitivity_km)
        .chain(self.footprint_sensitivity_km2)
        .chain(self.local_relief_radii_km)
        .chain(self.summit_cap_depths_km)
        .chain(self.gentle_grade_thresholds)
        .all(f64::is_finite);
        if !finite {
            return Err(LandformError::NonFiniteConfiguration);
        }
        // The translation fixture may translate closure, but no extraction knob
        // is runtime-tunable under schema v0.
        let mut translated = *self;
        translated.closure_level_km = registered.closure_level_km;
        if translated != registered {
            return Err(LandformError::UnregisteredConfiguration);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LandformError {
    EmptyGraph,
    EmptyScoredMask,
    LengthMismatch(&'static str),
    NonFiniteGeometry,
    NonFiniteElevation,
    NonFiniteConfiguration,
    UnregisteredConfiguration,
    NonPositiveMeasure,
    MalformedCsr,
    MalformedPolygonOffsets,
    InvalidPolygon {
        cell: usize,
    },
    InvalidPolygonWinding {
        cell: usize,
    },
    NonCanonicalGeometry,
    PlanarAreaMismatch {
        cell: usize,
    },
    SelfEdge {
        cell: usize,
    },
    DuplicateNeighbor {
        cell: usize,
        neighbor: usize,
    },
    MissingReciprocal {
        cell: usize,
        neighbor: usize,
    },
    NonUniqueReciprocal {
        cell: usize,
        neighbor: usize,
    },
    ReciprocalGeometryMismatch {
        cell: usize,
        neighbor: usize,
    },
    FaceNotBackedByPolygon {
        cell: usize,
        neighbor: usize,
    },
    MissingPolygonEdgeBacking {
        cell: usize,
        edge: usize,
    },
    OverlappingPolygonEdgeBacking {
        cell: usize,
        edge: usize,
    },
    InvalidBoundarySegment {
        segment: usize,
    },
    DuplicateBoundarySegment,
    MissingControlVolumeGeometry,
    NonRegularHexGeometry,
    NonPhysicalAdjacency,
    InvalidSphericalGeometry,
    UnsupportedDomain,
    SphereAreaClosure,
    OperatorGeometryMismatch {
        cell: usize,
        neighbor: usize,
    },
    NonFiniteDerived {
        measurement: &'static str,
        cell: usize,
    },
    Overflow,
    NegativePersistence,
    MultipleExclusiveOwners {
        cell: usize,
    },
    CyclicParentage,
    FootprintAreaMismatch {
        peak: usize,
    },
    InvalidPlanarMoments {
        peak: usize,
    },
    InvalidSphericalMoments {
        peak: usize,
    },
    Serialization(String),
}

impl fmt::Display for LandformError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for LandformError {}

impl From<LandscapeMeshError> for LandformError {
    fn from(_: LandscapeMeshError) -> Self {
        LandformError::NonRegularHexGeometry
    }
}

fn canonical_zero(value: f64) -> f64 {
    if value == 0.0 {
        0.0
    } else {
        value
    }
}

fn canonical_point(point: DVec3) -> DVec3 {
    DVec3::new(
        canonical_zero(point.x),
        canonical_zero(point.y),
        canonical_zero(point.z),
    )
}

fn has_canonical_zero(value: f64) -> bool {
    value != 0.0 || value.to_bits() == 0.0f64.to_bits()
}

fn point_has_canonical_zero(point: DVec3) -> bool {
    has_canonical_zero(point.x) && has_canonical_zero(point.y) && has_canonical_zero(point.z)
}

fn point_key(point: DVec3) -> (u64, u64, u64) {
    let point = canonical_point(point);
    (point.x.to_bits(), point.y.to_bits(), point.z.to_bits())
}

fn weld_planar_endpoints(endpoint_sets: &mut [&mut Vec<[DVec3; 2]>], tolerance: f64) {
    let mut points = endpoint_sets
        .iter()
        .flat_map(|set| set.iter().flat_map(|endpoints| endpoints.iter().copied()))
        .collect::<Vec<_>>();
    points.sort_by(|a, b| {
        a.x.total_cmp(&b.x)
            .then_with(|| a.y.total_cmp(&b.y))
            .then_with(|| a.z.total_cmp(&b.z))
    });
    points.dedup_by_key(|point| point_key(*point));

    let bucket_key = |point: DVec3| {
        (
            (point.x / tolerance).floor() as i64,
            (point.y / tolerance).floor() as i64,
            (point.z / tolerance).floor() as i64,
        )
    };
    let mut buckets = BTreeMap::<(i64, i64, i64), Vec<DVec3>>::new();
    let mut replacements = BTreeMap::new();
    for point in points {
        let (bx, by, bz) = bucket_key(point);
        let mut replacement = None;
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(candidates) = buckets.get(&(bx + dx, by + dy, bz + dz)) {
                        for &candidate in candidates {
                            if points_match(point, candidate, tolerance)
                                && replacement.is_none_or(|current: DVec3| {
                                    point_key(candidate) < point_key(current)
                                })
                            {
                                replacement = Some(candidate);
                            }
                        }
                    }
                }
            }
        }
        let replacement = replacement.unwrap_or_else(|| {
            buckets.entry((bx, by, bz)).or_default().push(point);
            point
        });
        replacements.insert(point_key(point), replacement);
    }
    for endpoint_set in endpoint_sets {
        for endpoints in endpoint_set.iter_mut() {
            for point in endpoints {
                *point = replacements[&point_key(*point)];
            }
        }
    }
}

fn points_match(a: DVec3, b: DVec3, tolerance: f64) -> bool {
    a.distance(b) <= tolerance
}

fn endpoints_match_directed(a: [DVec3; 2], b: [DVec3; 2], tolerance: f64) -> bool {
    points_match(a[0], b[0], tolerance) && points_match(a[1], b[1], tolerance)
}

fn planar_polygon_signed_area(polygon: &[DVec3]) -> f64 {
    0.5 * polygon
        .iter()
        .copied()
        .zip(polygon.iter().copied().cycle().skip(1))
        .take(polygon.len())
        .map(|(a, b)| a.x * b.y - a.y * b.x)
        .sum::<f64>()
}

fn rotate_polygon_to_canonical_start(polygon: &mut [DVec3]) {
    if let Some((start, _)) = polygon
        .iter()
        .enumerate()
        .min_by_key(|(_, point)| point_key(**point))
    {
        polygon.rotate_left(start);
    }
}

fn edge_range(offsets: &[u32], cell: usize) -> std::ops::Range<usize> {
    offsets[cell] as usize..offsets[cell + 1] as usize
}

fn directed_polygon_edge_index(
    polygon: &[DVec3],
    endpoints: [DVec3; 2],
    tolerance: f64,
) -> Option<usize> {
    polygon
        .iter()
        .copied()
        .zip(polygon.iter().copied().cycle().skip(1))
        .take(polygon.len())
        .position(|(a, b)| {
            points_match(a, endpoints[0], tolerance) && points_match(b, endpoints[1], tolerance)
        })
}

/// Build the exact companion geometry for the current regular planar hex
/// constructor. Irregular meshes must supply their own polygons and endpoints.
pub fn build_regular_hex_control_volumes_v0(
    mesh: &LandscapeMesh,
    config: &SurfaceHierarchyConfigV0,
) -> Result<LandscapeControlVolumesV0, LandformError> {
    config.validate()?;
    mesh.validate()?;
    let n = mesh.cell_count();
    if mesh
        .cell_center_km
        .iter()
        .any(|p| !p.is_finite() || p.z != 0.0)
    {
        return Err(LandformError::NonRegularHexGeometry);
    }

    let mut edge_face_endpoints_km = Vec::with_capacity(mesh.edge_neighbor.len());
    let mut vertices_by_cell = vec![Vec::<DVec3>::new(); n];
    for (cell, cell_vertices) in vertices_by_cell.iter_mut().enumerate() {
        for edge in edge_range(&mesh.edge_offsets, cell) {
            let neighbor = mesh.edge_neighbor[edge] as usize;
            let delta = mesh.cell_center_km[neighbor] - mesh.cell_center_km[cell];
            let distance = delta.length();
            if !distance.is_finite() || distance <= 0.0 {
                return Err(LandformError::NonRegularHexGeometry);
            }
            let normal = delta / distance;
            let stored_normal = mesh.edge_outward_tangent[edge].as_dvec3();
            if normal.distance(stored_normal) > 1.0e-6
                || (distance - f64::from(mesh.edge_distance_km[edge])).abs() > 1.0e-5
            {
                return Err(LandformError::NonRegularHexGeometry);
            }
            let tangent = DVec3::new(-normal.y, normal.x, 0.0);
            let center = 0.5 * (mesh.cell_center_km[cell] + mesh.cell_center_km[neighbor]);
            let physical_width = distance / 3.0_f64.sqrt();
            let stored_width = f64::from(mesh.edge_face_width_km[edge]);
            let quantization_tolerance =
                2.0 * f64::from(f32::EPSILON) * physical_width.abs().max(f64::MIN_POSITIVE);
            if (stored_width - physical_width).abs() > quantization_tolerance {
                return Err(LandformError::NonRegularHexGeometry);
            }
            let half = 0.5 * physical_width;
            let endpoints = [center - half * tangent, center + half * tangent];
            edge_face_endpoints_km.push(endpoints);
            cell_vertices.extend(endpoints);
        }
    }

    let mut boundary_face_endpoints_km = Vec::with_capacity(mesh.boundary_faces.len());
    for face in &mesh.boundary_faces {
        let normal = face.outward_normal.normalize();
        let tangent = DVec3::new(-normal.y, normal.x, 0.0);
        let endpoints = [
            face.center_km - 0.5 * face.width_km * tangent,
            face.center_km + 0.5 * face.width_km * tangent,
        ];
        boundary_face_endpoints_km.push(endpoints);
        vertices_by_cell[face.cell as usize].extend(endpoints);
    }

    // Internal faces now use the exact f64 regular-hex geometry from center
    // spacing; weld only ordinary last-bit differences where independently
    // constructed internal and boundary endpoints represent the same vertex.
    let weld_tolerance = config.endpoint_match_abs_km;
    weld_planar_endpoints(
        &mut [&mut edge_face_endpoints_km, &mut boundary_face_endpoints_km],
        weld_tolerance,
    );
    for vertices in &mut vertices_by_cell {
        vertices.clear();
    }
    for cell in 0..n {
        for edge in edge_range(&mesh.edge_offsets, cell) {
            vertices_by_cell[cell].extend(edge_face_endpoints_km[edge]);
        }
    }
    for (face, endpoints) in mesh.boundary_faces.iter().zip(&boundary_face_endpoints_km) {
        vertices_by_cell[face.cell as usize].extend(*endpoints);
    }

    let tolerance = config.endpoint_match_abs_km;
    let mut cell_polygon_offsets = Vec::with_capacity(n + 1);
    let mut cell_polygon_vertices_km = Vec::new();
    for (cell, vertices) in vertices_by_cell.iter_mut().enumerate() {
        let center = mesh.cell_center_km[cell];
        vertices.sort_by(|a, b| {
            (a.y - center.y)
                .atan2(a.x - center.x)
                .total_cmp(&(b.y - center.y).atan2(b.x - center.x))
                .then_with(|| point_key(*a).cmp(&point_key(*b)))
        });
        let mut unique = Vec::with_capacity(vertices.len());
        for &vertex in vertices.iter() {
            if unique
                .last()
                .is_none_or(|&last| !points_match(last, vertex, tolerance))
            {
                unique.push(canonical_point(vertex));
            }
        }
        if unique.len() >= 2 && points_match(unique[0], *unique.last().unwrap(), tolerance) {
            unique.pop();
        }
        if unique.len() < 6 || planar_polygon_signed_area(&unique) <= 0.0 {
            return Err(LandformError::NonRegularHexGeometry);
        }
        rotate_polygon_to_canonical_start(&mut unique);
        cell_polygon_offsets.push(
            u32::try_from(cell_polygon_vertices_km.len()).map_err(|_| LandformError::Overflow)?,
        );
        cell_polygon_vertices_km.extend(unique);
    }
    cell_polygon_offsets
        .push(u32::try_from(cell_polygon_vertices_km.len()).map_err(|_| LandformError::Overflow)?);

    Ok(LandscapeControlVolumesV0 {
        cell_polygon_offsets,
        cell_polygon_vertices_km,
        edge_face_endpoints_km,
        boundary_face_endpoints_km,
    })
}

/// Adapt a planar landscape operator mesh without reconstructing or guessing
/// any control-volume geometry.
pub fn adapt_landscape_graph_v0(
    mesh: &LandscapeMesh,
    controls: &LandscapeControlVolumesV0,
    config: &SurfaceHierarchyConfigV0,
) -> Result<EvaluationSurfaceGraphV0, LandformError> {
    config.validate()?;
    mesh.validate().map_err(|_| LandformError::MalformedCsr)?;
    let n = mesh.cell_count();
    if controls.cell_polygon_offsets.len() != n + 1
        || controls.edge_face_endpoints_km.len() != mesh.edge_neighbor.len()
        || controls.boundary_face_endpoints_km.len() != mesh.boundary_faces.len()
    {
        return Err(LandformError::MissingControlVolumeGeometry);
    }
    if controls.cell_polygon_offsets.first() != Some(&0)
        || controls
            .cell_polygon_offsets
            .windows(2)
            .any(|pair| pair[0] > pair[1])
        || controls.cell_polygon_offsets[n] as usize != controls.cell_polygon_vertices_km.len()
    {
        return Err(LandformError::MalformedPolygonOffsets);
    }

    let mut cell_polygon_offsets = Vec::with_capacity(n + 1);
    let mut cell_polygon_vertices_km = Vec::new();
    let mut cell_area_km2 = Vec::with_capacity(n);
    for cell in 0..n {
        let start = controls.cell_polygon_offsets[cell] as usize;
        let end = controls.cell_polygon_offsets[cell + 1] as usize;
        let mut polygon: Vec<DVec3> = controls.cell_polygon_vertices_km[start..end]
            .iter()
            .copied()
            .map(canonical_point)
            .collect();
        let polygon_area = planar_polygon_signed_area(&polygon);
        if polygon.len() < 3
            || polygon.iter().any(|p| !p.is_finite() || p.z != 0.0)
            || polygon_area <= 0.0
        {
            return Err(LandformError::InvalidPolygon { cell });
        }
        let native_area = mesh.cell_area_km2[cell];
        let area_relative =
            (polygon_area - native_area).abs() / native_area.abs().max(f64::MIN_POSITIVE);
        if !native_area.is_finite()
            || native_area <= 0.0
            || area_relative > config.planar_area_match_relative
        {
            return Err(LandformError::PlanarAreaMismatch { cell });
        }
        cell_area_km2.push(canonical_zero(native_area));
        rotate_polygon_to_canonical_start(&mut polygon);
        cell_polygon_offsets.push(
            u32::try_from(cell_polygon_vertices_km.len()).map_err(|_| LandformError::Overflow)?,
        );
        cell_polygon_vertices_km.extend(polygon);
    }
    cell_polygon_offsets
        .push(u32::try_from(cell_polygon_vertices_km.len()).map_err(|_| LandformError::Overflow)?);

    #[derive(Clone)]
    struct EdgeRecord {
        neighbor: u32,
        distance: f64,
        width: f64,
        endpoints: [DVec3; 2],
    }
    let mut records_by_cell = Vec::with_capacity(n);
    for cell in 0..n {
        let mut records = Vec::new();
        for source_edge in edge_range(&mesh.edge_offsets, cell) {
            let neighbor = mesh.edge_neighbor[source_edge] as usize;
            let displacement = mesh.cell_center_km[neighbor] - mesh.cell_center_km[cell];
            let physical_distance = displacement.length();
            let stored_distance = f64::from(mesh.edge_distance_km[source_edge]);
            let stored_normal = mesh.edge_outward_tangent[source_edge].as_dvec3();
            let distance_tolerance =
                2.0 * f64::from(f32::EPSILON) * physical_distance.abs().max(f64::MIN_POSITIVE);
            let direction_tolerance = 8.0 * f64::from(f32::EPSILON);
            if !physical_distance.is_finite()
                || physical_distance <= 0.0
                || (stored_distance - physical_distance).abs() > distance_tolerance
                || stored_normal.distance(displacement / physical_distance) > direction_tolerance
            {
                return Err(LandformError::OperatorGeometryMismatch { cell, neighbor });
            }
            let endpoints = controls.edge_face_endpoints_km[source_edge].map(canonical_point);
            let face = endpoints[1] - endpoints[0];
            let face_length = face.length();
            let stored_width = f64::from(mesh.edge_face_width_km[source_edge]);
            let width_tolerance =
                2.0 * f64::from(f32::EPSILON) * face_length.abs().max(f64::MIN_POSITIVE);
            if !face_length.is_finite()
                || face_length <= 0.0
                || (stored_width - face_length).abs() > width_tolerance
            {
                return Err(LandformError::OperatorGeometryMismatch { cell, neighbor });
            }
            records.push(EdgeRecord {
                neighbor: mesh.edge_neighbor[source_edge],
                distance: stored_distance,
                width: stored_width,
                endpoints,
            });
        }
        records.sort_by_key(|record| record.neighbor);
        if records
            .windows(2)
            .any(|pair| pair[0].neighbor == pair[1].neighbor)
        {
            let neighbor = records
                .windows(2)
                .find(|pair| pair[0].neighbor == pair[1].neighbor)
                .unwrap()[0]
                .neighbor as usize;
            return Err(LandformError::DuplicateNeighbor { cell, neighbor });
        }
        records_by_cell.push(records);
    }

    let mut edge_offsets = Vec::with_capacity(n + 1);
    let mut edge_neighbor = Vec::new();
    let mut edge_distance_km = Vec::new();
    let mut edge_shared_width_km = Vec::new();
    let mut edge_face_endpoints_km = Vec::new();
    for records in &records_by_cell {
        edge_offsets.push(u32::try_from(edge_neighbor.len()).map_err(|_| LandformError::Overflow)?);
        for record in records {
            edge_neighbor.push(record.neighbor);
            edge_distance_km.push(canonical_zero(record.distance));
            edge_shared_width_km.push(canonical_zero(record.width));
            edge_face_endpoints_km.push(record.endpoints);
        }
    }
    edge_offsets.push(u32::try_from(edge_neighbor.len()).map_err(|_| LandformError::Overflow)?);

    let mut directed = BTreeMap::new();
    for cell in 0..n {
        for edge in edge_range(&edge_offsets, cell) {
            let neighbor = edge_neighbor[edge] as usize;
            if directed.insert((cell, neighbor), edge).is_some() {
                return Err(LandformError::DuplicateNeighbor { cell, neighbor });
            }
        }
    }
    let mut edge_reciprocal = vec![0u32; edge_neighbor.len()];
    for (&(cell, neighbor), &edge) in &directed {
        let reverse = directed
            .get(&(neighbor, cell))
            .copied()
            .ok_or(LandformError::MissingReciprocal { cell, neighbor })?;
        edge_reciprocal[edge] = u32::try_from(reverse).map_err(|_| LandformError::Overflow)?;
    }

    let mut boundary_records: Vec<_> = mesh
        .boundary_faces
        .iter()
        .zip(&controls.boundary_face_endpoints_km)
        .enumerate()
        .map(|(index, (face, endpoints))| {
            let endpoints = endpoints.map(canonical_point);
            let midpoint = 0.5 * (endpoints[0] + endpoints[1]);
            let length = endpoints[0].distance(endpoints[1]);
            if !length.is_finite()
                || length <= 0.0
                || (length - face.width_km).abs() > config.endpoint_match_abs_km
                || midpoint.distance(face.center_km) > config.endpoint_match_abs_km
            {
                return Err(LandformError::InvalidBoundarySegment { segment: index });
            }
            let condition = match face.condition {
                BoundaryFaceCondition::Closed => EvaluationBoundaryConditionV0::Closed,
                BoundaryFaceCondition::OpenBaseLevel {
                    portal_id: OutletPortalId(id),
                    elevation_km,
                } => EvaluationBoundaryConditionV0::OpenBaseLevel {
                    portal_id: id,
                    elevation_km: canonical_zero(f64::from(elevation_km)),
                },
            };
            Ok((
                face.cell,
                endpoints,
                face.width_km,
                Some([
                    canonical_zero(face.projected_span_start_km),
                    canonical_zero(face.projected_span_end_km),
                ]),
                condition,
            ))
        })
        .collect::<Result<_, LandformError>>()?;
    boundary_records.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then_with(|| point_key(a.1[0]).cmp(&point_key(b.1[0])))
            .then_with(|| point_key(a.1[1]).cmp(&point_key(b.1[1])))
    });
    let boundary_segments = boundary_records
        .into_iter()
        .enumerate()
        .map(
            |(id, (owner_cell, endpoints_km, physical_length_km, projected_span_km, condition))| {
                Ok(EvaluationBoundarySegmentV0 {
                    id: u32::try_from(id).map_err(|_| LandformError::Overflow)?,
                    owner_cell,
                    endpoints_km,
                    physical_length_km: canonical_zero(physical_length_km),
                    projected_span_km,
                    condition,
                })
            },
        )
        .collect::<Result<Vec<_>, LandformError>>()?;

    let graph = EvaluationSurfaceGraphV0 {
        domain: EvaluationDomainV0::Planar,
        cell_center_km: mesh
            .cell_center_km
            .iter()
            .copied()
            .map(canonical_point)
            .collect(),
        cell_area_km2,
        cell_polygon_offsets,
        cell_polygon_vertices_km,
        edge_offsets,
        edge_neighbor,
        edge_reciprocal,
        edge_distance_km,
        edge_shared_width_km,
        edge_face_endpoints_km,
        boundary_segments,
    };
    graph.validate(config)?;
    Ok(graph)
}

/// Adapt the retained, exact projected control volumes of the registered
/// irregular product-Voronoi cap. This is intentionally a fixture seam rather
/// than a license to reconstruct arbitrary planar Voronoi polygons.
pub fn adapt_projected_voronoi_cap_graph_v0(
    fixture: &VoronoiCapFixture,
    config: &SurfaceHierarchyConfigV0,
) -> Result<EvaluationSurfaceGraphV0, LandformError> {
    let mut cell_polygon_offsets = Vec::with_capacity(fixture.cell_polygons_km.len() + 1);
    let mut cell_polygon_vertices_km = Vec::new();
    for polygon in &fixture.cell_polygons_km {
        cell_polygon_offsets.push(
            u32::try_from(cell_polygon_vertices_km.len()).map_err(|_| LandformError::Overflow)?,
        );
        cell_polygon_vertices_km.extend(
            polygon
                .iter()
                .map(|point| DVec3::new(point.x, point.y, 0.0)),
        );
    }
    cell_polygon_offsets
        .push(u32::try_from(cell_polygon_vertices_km.len()).map_err(|_| LandformError::Overflow)?);
    let controls = LandscapeControlVolumesV0 {
        cell_polygon_offsets,
        cell_polygon_vertices_km,
        edge_face_endpoints_km: fixture.edge_face_endpoints_km.clone(),
        boundary_face_endpoints_km: fixture.boundary_face_endpoints_km.clone(),
    };
    adapt_landscape_graph_v0(&fixture.mesh, &controls, config)
}

/// Adapt the product tessellation into an exact closed-sphere G0 graph.
///
/// Polygon vertex IDs, rather than coordinate coincidence or generator
/// proximity, are the authority for physical face ownership. Product
/// adjacency is accepted only when it is exactly the same validated neighbor
/// set as that ownership graph.
pub fn adapt_product_tessellation_graph_v0(
    tessellation: &Tessellation,
    config: &SurfaceHierarchyConfigV0,
) -> Result<EvaluationSurfaceGraphV0, LandformError> {
    config.validate()?;
    let n = tessellation.voronoi.num_cells();
    if n == 0 {
        return Err(LandformError::EmptyGraph);
    }
    if tessellation.voronoi.generators.len() != n
        || !tessellation.voronoi.has_canonical_cell_layout()
        || !tessellation.adjacency.has_canonical_layout(n)
    {
        return Err(LandformError::MalformedCsr);
    }
    let radius_km = f64::from(PLANET_RADIUS_KM);
    let centers = tessellation
        .voronoi
        .generators
        .iter()
        .copied()
        .map(|point| spherical_source_point(point.as_dvec3(), radius_km))
        .collect::<Result<Vec<_>, _>>()?;
    let vertices = tessellation
        .voronoi
        .vertices
        .iter()
        .copied()
        .map(|point| spherical_source_point(point.as_dvec3(), radius_km))
        .collect::<Result<Vec<_>, _>>()?;

    #[derive(Clone, Copy)]
    struct FaceUse {
        cell: usize,
        start: u32,
        end: u32,
    }
    let mut ownership = BTreeMap::<(u32, u32), Vec<FaceUse>>::new();
    let mut cell_polygon_offsets = Vec::with_capacity(n + 1);
    let mut cell_polygon_vertices_km = Vec::new();
    let mut cell_area_km2 = Vec::with_capacity(n);
    for (cell, &center) in centers.iter().enumerate() {
        let source_cell = tessellation
            .voronoi
            .try_cell(cell)
            .ok_or(LandformError::MalformedPolygonOffsets)?;
        if source_cell.vertex_indices.len() < 3 {
            return Err(LandformError::InvalidPolygon { cell });
        }
        let mut distinct = BTreeSet::new();
        let mut polygon = Vec::with_capacity(source_cell.vertex_indices.len());
        for &vertex_id in source_cell.vertex_indices {
            let vertex = vertices
                .get(vertex_id as usize)
                .copied()
                .ok_or(LandformError::InvalidPolygon { cell })?;
            if !distinct.insert(vertex_id) {
                return Err(LandformError::InvalidPolygon { cell });
            }
            polygon.push(vertex);
        }
        for (&start, &end) in source_cell.vertex_indices.iter().zip(
            source_cell
                .vertex_indices
                .iter()
                .cycle()
                .skip(1)
                .take(source_cell.vertex_indices.len()),
        ) {
            if start == end {
                return Err(LandformError::InvalidPolygon { cell });
            }
            ownership
                .entry((start.min(end), start.max(end)))
                .or_default()
                .push(FaceUse { cell, start, end });
        }
        let area = spherical_cell_area_km2(center, &polygon, radius_km)?;
        cell_area_km2.push(canonical_zero(area));
        rotate_polygon_to_canonical_start(&mut polygon);
        cell_polygon_offsets.push(
            u32::try_from(cell_polygon_vertices_km.len()).map_err(|_| LandformError::Overflow)?,
        );
        cell_polygon_vertices_km.extend(polygon);
    }
    cell_polygon_offsets
        .push(u32::try_from(cell_polygon_vertices_km.len()).map_err(|_| LandformError::Overflow)?);

    #[derive(Clone)]
    struct EdgeRecord {
        neighbor: u32,
        distance: f64,
        width: f64,
        endpoints: [DVec3; 2],
    }
    let mut records_by_cell = vec![Vec::<EdgeRecord>::new(); n];
    for uses in ownership.values() {
        if uses.len() != 2
            || uses[0].cell == uses[1].cell
            || uses[0].start != uses[1].end
            || uses[0].end != uses[1].start
        {
            return Err(LandformError::NonPhysicalAdjacency);
        }
        for (owner, other) in [(uses[0], uses[1]), (uses[1], uses[0])] {
            let distance = spherical_arc_km(centers[owner.cell], centers[other.cell], radius_km)?;
            let endpoints = [vertices[owner.start as usize], vertices[owner.end as usize]];
            let width = spherical_arc_km(endpoints[0], endpoints[1], radius_km)?;
            records_by_cell[owner.cell].push(EdgeRecord {
                neighbor: u32::try_from(other.cell).map_err(|_| LandformError::Overflow)?,
                distance,
                width,
                endpoints,
            });
        }
    }

    for (cell, records) in records_by_cell.iter_mut().enumerate() {
        records.sort_by_key(|record| record.neighbor);
        if records
            .windows(2)
            .any(|pair| pair[0].neighbor == pair[1].neighbor)
        {
            return Err(LandformError::NonPhysicalAdjacency);
        }
        let mut stored = tessellation
            .try_neighbors(cell)
            .ok_or(LandformError::MalformedCsr)?
            .to_vec();
        if stored
            .iter()
            .any(|&neighbor| neighbor >= n || neighbor == cell)
        {
            return Err(LandformError::NonPhysicalAdjacency);
        }
        stored.sort_unstable();
        if stored.windows(2).any(|pair| pair[0] == pair[1])
            || !stored
                .iter()
                .copied()
                .eq(records.iter().map(|record| record.neighbor as usize))
        {
            return Err(LandformError::NonPhysicalAdjacency);
        }
    }

    let mut edge_offsets = Vec::with_capacity(n + 1);
    let mut edge_neighbor = Vec::new();
    let mut edge_distance_km = Vec::new();
    let mut edge_shared_width_km = Vec::new();
    let mut edge_face_endpoints_km = Vec::new();
    for records in &records_by_cell {
        edge_offsets.push(u32::try_from(edge_neighbor.len()).map_err(|_| LandformError::Overflow)?);
        for record in records {
            edge_neighbor.push(record.neighbor);
            edge_distance_km.push(canonical_zero(record.distance));
            edge_shared_width_km.push(canonical_zero(record.width));
            edge_face_endpoints_km.push(record.endpoints.map(canonical_point));
        }
    }
    edge_offsets.push(u32::try_from(edge_neighbor.len()).map_err(|_| LandformError::Overflow)?);

    let mut directed = BTreeMap::new();
    for cell in 0..n {
        for edge in edge_range(&edge_offsets, cell) {
            directed.insert((cell, edge_neighbor[edge] as usize), edge);
        }
    }
    let mut edge_reciprocal = vec![0; edge_neighbor.len()];
    for (&(cell, neighbor), &edge) in &directed {
        let reciprocal = directed
            .get(&(neighbor, cell))
            .copied()
            .ok_or(LandformError::NonPhysicalAdjacency)?;
        edge_reciprocal[edge] = u32::try_from(reciprocal).map_err(|_| LandformError::Overflow)?;
    }

    let graph = EvaluationSurfaceGraphV0 {
        domain: EvaluationDomainV0::Spherical { radius_km },
        cell_center_km: centers,
        cell_area_km2,
        cell_polygon_offsets,
        cell_polygon_vertices_km,
        edge_offsets,
        edge_neighbor,
        edge_reciprocal,
        edge_distance_km,
        edge_shared_width_km,
        edge_face_endpoints_km,
        boundary_segments: Vec::new(),
    };
    graph.validate(config)?;
    Ok(graph)
}

fn validate_graph(
    graph: &EvaluationSurfaceGraphV0,
    config: &SurfaceHierarchyConfigV0,
) -> Result<(), LandformError> {
    config.validate()?;
    let n = graph.cell_count();
    if n == 0 {
        return Err(LandformError::EmptyGraph);
    }
    if graph.cell_area_km2.len() != n {
        return Err(LandformError::LengthMismatch("cell_area_km2"));
    }
    if graph.cell_polygon_offsets.len() != n + 1
        || graph.cell_polygon_offsets.first() != Some(&0)
        || graph
            .cell_polygon_offsets
            .windows(2)
            .any(|pair| pair[0] > pair[1])
        || graph.cell_polygon_offsets[n] as usize != graph.cell_polygon_vertices_km.len()
    {
        return Err(LandformError::MalformedPolygonOffsets);
    }
    if graph.edge_offsets.len() != n + 1
        || graph.edge_offsets.first() != Some(&0)
        || graph.edge_offsets.windows(2).any(|pair| pair[0] > pair[1])
    {
        return Err(LandformError::MalformedCsr);
    }
    let m = graph.edge_neighbor.len();
    if graph.edge_offsets[n] as usize != m
        || graph.edge_reciprocal.len() != m
        || graph.edge_distance_km.len() != m
        || graph.edge_shared_width_km.len() != m
        || graph.edge_face_endpoints_km.len() != m
    {
        return Err(LandformError::MalformedCsr);
    }
    if graph.cell_center_km.iter().any(|p| !p.is_finite())
        || graph
            .cell_polygon_vertices_km
            .iter()
            .any(|p| !p.is_finite())
        || graph.cell_area_km2.iter().any(|a| !a.is_finite())
        || graph.edge_distance_km.iter().any(|d| !d.is_finite())
        || graph.edge_shared_width_km.iter().any(|w| !w.is_finite())
        || graph
            .edge_face_endpoints_km
            .iter()
            .flatten()
            .any(|p| !p.is_finite())
    {
        return Err(LandformError::NonFiniteGeometry);
    }
    if graph.cell_area_km2.iter().any(|&a| a <= 0.0)
        || graph.edge_distance_km.iter().any(|&d| d <= 0.0)
        || graph.edge_shared_width_km.iter().any(|&w| w <= 0.0)
    {
        return Err(LandformError::NonPositiveMeasure);
    }
    if graph
        .cell_center_km
        .iter()
        .chain(&graph.cell_polygon_vertices_km)
        .chain(graph.edge_face_endpoints_km.iter().flatten())
        .any(|&point| !point_has_canonical_zero(point))
        || matches!(graph.domain, EvaluationDomainV0::Spherical { radius_km } if !has_canonical_zero(radius_km))
    {
        return Err(LandformError::NonCanonicalGeometry);
    }

    let spherical_radius = match graph.domain {
        EvaluationDomainV0::Planar => None,
        EvaluationDomainV0::Spherical { radius_km } if radius_km.is_finite() && radius_km > 0.0 => {
            Some(radius_km)
        }
        EvaluationDomainV0::Spherical { .. } => {
            return Err(LandformError::InvalidSphericalGeometry);
        }
    };
    if spherical_radius.is_some() && !graph.boundary_segments.is_empty() {
        return Err(LandformError::InvalidSphericalGeometry);
    }

    let tolerance = config.endpoint_match_abs_km;
    if let Some(radius_km) = spherical_radius {
        if graph
            .cell_center_km
            .iter()
            .chain(&graph.cell_polygon_vertices_km)
            .chain(graph.edge_face_endpoints_km.iter().flatten())
            .any(|point| (point.length() - radius_km).abs() > tolerance)
        {
            return Err(LandformError::InvalidSphericalGeometry);
        }
    }
    let mut edge_backing_counts: Vec<Vec<u8>> = (0..n)
        .map(|cell| vec![0; graph.polygon(cell).len()])
        .collect();
    for cell in 0..n {
        let polygon = graph.polygon(cell);
        if polygon.len() < 3
            || polygon
                .windows(2)
                .any(|pair| points_match(pair[0], pair[1], tolerance))
            || points_match(polygon[0], *polygon.last().unwrap(), tolerance)
        {
            return Err(LandformError::InvalidPolygon { cell });
        }
        if polygon
            .iter()
            .min_by_key(|point| point_key(**point))
            .is_some_and(|minimum| point_key(*minimum) != point_key(polygon[0]))
        {
            return Err(LandformError::NonCanonicalGeometry);
        }
        if matches!(graph.domain, EvaluationDomainV0::Planar) {
            if polygon.iter().any(|p| p.z != 0.0) {
                return Err(LandformError::InvalidPolygon { cell });
            }
            let area = planar_polygon_signed_area(polygon);
            if area <= 0.0 {
                return Err(LandformError::InvalidPolygonWinding { cell });
            }
            let relative = (area - graph.cell_area_km2[cell]).abs()
                / graph.cell_area_km2[cell].max(f64::MIN_POSITIVE);
            if relative > config.planar_area_match_relative {
                return Err(LandformError::PlanarAreaMismatch { cell });
            }
        } else if let Some(radius_km) = spherical_radius {
            let area = spherical_cell_area_km2(graph.cell_center_km[cell], polygon, radius_km)?;
            let relative = (area - graph.cell_area_km2[cell]).abs()
                / graph.cell_area_km2[cell].max(f64::MIN_POSITIVE);
            if relative > config.sphere_area_closure_relative {
                return Err(LandformError::InvalidSphericalGeometry);
            }
        }

        let mut previous = None;
        for edge in edge_range(&graph.edge_offsets, cell) {
            let neighbor = graph.edge_neighbor[edge] as usize;
            if neighbor >= n || neighbor == cell {
                return Err(LandformError::SelfEdge { cell });
            }
            if previous.is_some_and(|value| value >= neighbor) {
                return Err(LandformError::DuplicateNeighbor { cell, neighbor });
            }
            previous = Some(neighbor);
            let reciprocal = graph.edge_reciprocal[edge] as usize;
            if reciprocal >= m
                || graph.edge_neighbor[reciprocal] as usize != cell
                || graph.edge_reciprocal[reciprocal] as usize != edge
            {
                return Err(LandformError::MissingReciprocal { cell, neighbor });
            }
            let reverse_endpoints = [
                graph.edge_face_endpoints_km[reciprocal][1],
                graph.edge_face_endpoints_km[reciprocal][0],
            ];
            if (graph.edge_distance_km[edge] - graph.edge_distance_km[reciprocal]).abs() > tolerance
                || (graph.edge_shared_width_km[edge] - graph.edge_shared_width_km[reciprocal]).abs()
                    > tolerance
                || !endpoints_match_directed(
                    graph.edge_face_endpoints_km[edge],
                    reverse_endpoints,
                    tolerance,
                )
            {
                return Err(LandformError::ReciprocalGeometryMismatch { cell, neighbor });
            }
            let polygon_edge =
                directed_polygon_edge_index(polygon, graph.edge_face_endpoints_km[edge], tolerance)
                    .ok_or(LandformError::FaceNotBackedByPolygon { cell, neighbor })?;
            edge_backing_counts[cell][polygon_edge] = edge_backing_counts[cell][polygon_edge]
                .checked_add(1)
                .ok_or(LandformError::OverlappingPolygonEdgeBacking {
                    cell,
                    edge: polygon_edge,
                })?;
            if edge_backing_counts[cell][polygon_edge] > 1 {
                return Err(LandformError::OverlappingPolygonEdgeBacking {
                    cell,
                    edge: polygon_edge,
                });
            }
            let endpoint_width = match graph.domain {
                EvaluationDomainV0::Planar => graph.edge_face_endpoints_km[edge][0]
                    .distance(graph.edge_face_endpoints_km[edge][1]),
                EvaluationDomainV0::Spherical { radius_km } => spherical_arc_km(
                    graph.edge_face_endpoints_km[edge][0],
                    graph.edge_face_endpoints_km[edge][1],
                    radius_km,
                )?,
            };
            let width_tolerance = if matches!(graph.domain, EvaluationDomainV0::Planar) {
                tolerance.max(
                    2.0 * f64::from(f32::EPSILON) * endpoint_width.abs().max(f64::MIN_POSITIVE),
                )
            } else {
                tolerance
            };
            if (endpoint_width - graph.edge_shared_width_km[edge]).abs() > width_tolerance {
                return Err(LandformError::ReciprocalGeometryMismatch { cell, neighbor });
            }
            if matches!(graph.domain, EvaluationDomainV0::Planar) {
                let displacement = graph.cell_center_km[neighbor] - graph.cell_center_km[cell];
                let physical_distance = displacement.length();
                let distance_tolerance =
                    2.0 * f64::from(f32::EPSILON) * physical_distance.abs().max(f64::MIN_POSITIVE);
                if !physical_distance.is_finite()
                    || physical_distance <= 0.0
                    || (graph.edge_distance_km[edge] - physical_distance).abs() > distance_tolerance
                {
                    return Err(LandformError::OperatorGeometryMismatch { cell, neighbor });
                }
            } else if let Some(radius_km) = spherical_radius {
                let center_distance = spherical_arc_km(
                    graph.cell_center_km[cell],
                    graph.cell_center_km[neighbor],
                    radius_km,
                )?;
                if (center_distance - graph.edge_distance_km[edge]).abs() > tolerance {
                    return Err(LandformError::InvalidSphericalGeometry);
                }
            }
        }
    }

    let mut previous_boundary_key = None;
    for (index, segment) in graph.boundary_segments.iter().enumerate() {
        if segment.id as usize != index
            || segment.owner_cell as usize >= n
            || !segment.physical_length_km.is_finite()
            || segment.physical_length_km <= 0.0
            || segment.endpoints_km.iter().any(|p| !p.is_finite())
            || segment
                .endpoints_km
                .iter()
                .any(|&point| !point_has_canonical_zero(point))
            || segment.projected_span_km.is_some_and(|span| {
                span.into_iter()
                    .any(|value| !value.is_finite() || !has_canonical_zero(value))
            })
        {
            return Err(LandformError::InvalidBoundarySegment { segment: index });
        }
        let owner = segment.owner_cell as usize;
        let polygon_edge =
            directed_polygon_edge_index(graph.polygon(owner), segment.endpoints_km, tolerance)
                .ok_or(LandformError::InvalidBoundarySegment { segment: index })?;
        edge_backing_counts[owner][polygon_edge] = edge_backing_counts[owner][polygon_edge]
            .checked_add(1)
            .ok_or(LandformError::OverlappingPolygonEdgeBacking {
                cell: owner,
                edge: polygon_edge,
            })?;
        if edge_backing_counts[owner][polygon_edge] > 1 {
            return Err(LandformError::OverlappingPolygonEdgeBacking {
                cell: owner,
                edge: polygon_edge,
            });
        }
        let length = segment.endpoints_km[0].distance(segment.endpoints_km[1]);
        if matches!(graph.domain, EvaluationDomainV0::Planar)
            && (length - segment.physical_length_km).abs() > tolerance
        {
            return Err(LandformError::InvalidBoundarySegment { segment: index });
        }
        let key = (
            segment.owner_cell,
            point_key(segment.endpoints_km[0]),
            point_key(segment.endpoints_km[1]),
        );
        if previous_boundary_key.is_some_and(|previous| previous >= key) {
            return Err(LandformError::DuplicateBoundarySegment);
        }
        previous_boundary_key = Some(key);
        if let EvaluationBoundaryConditionV0::OpenBaseLevel { elevation_km, .. } = segment.condition
        {
            if !elevation_km.is_finite() || !has_canonical_zero(elevation_km) {
                return Err(LandformError::InvalidBoundarySegment { segment: index });
            }
        }
    }
    for (cell, counts) in edge_backing_counts.into_iter().enumerate() {
        if let Some(edge) = counts.iter().position(|&count| count == 0) {
            return Err(LandformError::MissingPolygonEdgeBacking { cell, edge });
        }
    }
    if let Some(radius_km) = spherical_radius {
        let total_area = compensated_sum(graph.cell_area_km2.iter().copied());
        let expected_area = 4.0 * std::f64::consts::PI * radius_km * radius_km;
        if (total_area - expected_area).abs() / expected_area > config.sphere_area_closure_relative
        {
            return Err(LandformError::SphereAreaClosure);
        }
        let mut reached = vec![false; n];
        let mut queue = VecDeque::from([0usize]);
        reached[0] = true;
        while let Some(cell) = queue.pop_front() {
            for edge in edge_range(&graph.edge_offsets, cell) {
                let neighbor = graph.edge_neighbor[edge] as usize;
                if !reached[neighbor] {
                    reached[neighbor] = true;
                    queue.push_back(neighbor);
                }
            }
        }
        if reached.iter().any(|&value| !value) {
            return Err(LandformError::NonPhysicalAdjacency);
        }
    }
    Ok(())
}

fn spherical_source_point(point: DVec3, radius_km: f64) -> Result<DVec3, LandformError> {
    if !point.is_finite()
        || !radius_km.is_finite()
        || radius_km <= 0.0
        || point.length_squared() == 0.0
    {
        return Err(LandformError::InvalidSphericalGeometry);
    }
    Ok(canonical_point(point.normalize() * radius_km))
}

fn spherical_cell_area_km2(
    center: DVec3,
    polygon: &[DVec3],
    radius_km: f64,
) -> Result<f64, LandformError> {
    if polygon.len() < 3 {
        return Err(LandformError::InvalidSphericalGeometry);
    }
    let center = center.normalize();
    let mut terms = Vec::with_capacity(polygon.len());
    for (&a, &b) in polygon.iter().zip(polygon.iter().cycle().skip(1)) {
        let (a, b) = (a.normalize(), b.normalize());
        let cross = a.cross(b);
        let angle = cross.length().atan2(a.dot(b));
        let determinant = center.dot(cross);
        if !angle.is_finite()
            || angle <= 0.0
            || angle >= std::f64::consts::PI
            || !determinant.is_finite()
            || determinant <= 0.0
        {
            return Err(LandformError::InvalidSphericalGeometry);
        }
        let denominator = 1.0 + center.dot(a) + a.dot(b) + b.dot(center);
        let solid_angle = 2.0 * determinant.atan2(denominator);
        if !solid_angle.is_finite() || solid_angle <= 0.0 {
            return Err(LandformError::InvalidSphericalGeometry);
        }
        terms.push(solid_angle);
    }
    let area = compensated_sum(terms) * radius_km * radius_km;
    if !area.is_finite() || area <= 0.0 {
        return Err(LandformError::InvalidSphericalGeometry);
    }
    Ok(area)
}

fn compensated_sum(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut sum = 0.0;
    let mut correction = 0.0;
    for value in values {
        let adjusted = value - correction;
        let next = sum + adjusted;
        correction = (next - sum) - adjusted;
        sum = next;
    }
    sum
}

fn spherical_arc_km(a: DVec3, b: DVec3, radius_km: f64) -> Result<f64, LandformError> {
    if !radius_km.is_finite()
        || radius_km <= 0.0
        || a.length_squared() == 0.0
        || b.length_squared() == 0.0
    {
        return Err(LandformError::InvalidSphericalGeometry);
    }
    let (a, b) = (a.normalize(), b.normalize());
    let angle = a.cross(b).length().atan2(a.dot(b));
    if !angle.is_finite() || angle <= 0.0 || angle >= std::f64::consts::PI {
        return Err(LandformError::InvalidSphericalGeometry);
    }
    Ok(radius_km * angle)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SaddleNodeV0 {
    pub id: u32,
    pub elevation_km: f64,
    pub anchor_cell: u32,
    pub flat_centroid_km: DVec3,
    pub flat_saddle_cells: Vec<u32>,
    pub elder_peak: u32,
    pub losing_peaks: Vec<u32>,
    pub equal_elder_ambiguous: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PeakBranchV0 {
    pub id: u32,
    pub peak_elevation_km: f64,
    pub anchor_cell: u32,
    pub flat_centroid_km: DVec3,
    pub flat_maximum_cells: Vec<u32>,
    pub parent_peak: Option<u32>,
    pub key_saddle: Option<u32>,
    pub persistence_km: f64,
    pub root_closure: bool,
    pub equal_elder_ambiguous: bool,
    /// Cells assigned directly to this peak branch when their exact level batch
    /// activated. These sets partition the active domain.
    pub exclusive_cells: Vec<u32>,
    /// Exact nested branch footprint reconstructed from exclusive ownership and
    /// losing-child relations.
    pub footprint_members: Vec<u32>,
    pub footprint_area_km2: f64,
    pub union_boundary_edges: Vec<u32>,
    pub physical_boundary_segments: Vec<u32>,
    pub scored_boundary_contact: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HighlandPopulationsV0 {
    pub reference: Vec<u32>,
    pub persistence_low: Vec<u32>,
    pub persistence_high: Vec<u32>,
    pub footprint_low: Vec<u32>,
    pub footprint_high: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlanarFootprintGeometryV0 {
    pub area_km2: f64,
    pub centroid_km: DVec3,
    /// Row-major planar area covariance `[xx, xy, yx, yy]`.
    pub covariance_km2: [f64; 4],
    pub equivalent_ellipse_length_km: f64,
    pub equivalent_ellipse_width_km: f64,
    pub anisotropy: f64,
    /// Sign-canonical principal eigenvector in the physical XY frame.
    pub principal_axis: DVec3,
    pub orientation_ambiguous: bool,
    pub two_sweep_extent_km: Option<f64>,
    pub mean_width_km: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LocalReliefSummaryV0 {
    pub radius_km: f64,
    pub area_weighted_p50_km: f64,
    pub area_weighted_p90_km: f64,
    /// Footprint members whose registered-radius neighborhood intersects a
    /// physical or scored-domain boundary.
    pub truncated_member_cells: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GentleFractionV0 {
    pub grade_threshold: f64,
    pub fraction: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SummitCapSummaryV0 {
    pub depth_km: f64,
    pub area_km2: f64,
    pub fraction: f64,
    pub valid_grade_fraction: f64,
    pub gentle_fractions: Vec<GentleFractionV0>,
    pub cap_merge_censored: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlanarHighlandMeasurementsV0 {
    pub footprint_geometry: PlanarFootprintGeometryV0,
    pub local_relief: Vec<LocalReliefSummaryV0>,
    /// Footprint members whose face-weighted least-squares gradient is
    /// unavailable because its normal matrix is rank deficient.
    pub rank_deficient_grade_cells: Vec<u32>,
    pub summit_caps: Vec<SummitCapSummaryV0>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SphericalLocalFootprintGeometryV0 {
    pub area_km2: f64,
    pub centroid_km: DVec3,
    pub projected_centroid_km: [f64; 2],
    /// Row-major covariance in the deterministic tangent basis `[xx, xy, yx, yy]`.
    pub tangent_covariance_km2: [f64; 4],
    pub equivalent_ellipse_length_km: f64,
    pub equivalent_ellipse_width_km: f64,
    pub anisotropy: f64,
    /// Sign-canonical principal tangent in the global physical frame.
    pub principal_axis: DVec3,
    pub orientation_ambiguous: bool,
    pub maximum_angular_radius_rad: f64,
    pub spherical_nonlocal_warning: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SphericalFootprintGeometryV0 {
    Local(SphericalLocalFootprintGeometryV0),
    NonLocalGeometry,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SphericalHighlandMeasurementsV0 {
    pub footprint_geometry: SphericalFootprintGeometryV0,
    pub two_sweep_extent_km: Option<f64>,
    pub mean_width_km: Option<f64>,
    pub local_relief: Vec<LocalReliefSummaryV0>,
    pub rank_deficient_grade_cells: Vec<u32>,
    pub summit_caps: Vec<SummitCapSummaryV0>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum HighlandMeasurementsV0 {
    Planar(Box<PlanarHighlandMeasurementsV0>),
    Spherical(Box<SphericalHighlandMeasurementsV0>),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HighlandFeatureV0 {
    pub peak_id: u32,
    pub measurements: HighlandMeasurementsV0,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
/// Structural peak-saddle output and reference morphology on either registered
/// physical domain.
pub struct SurfaceHierarchyV0 {
    pub schema_version: String,
    pub hash_version: String,
    pub peaks: Vec<PeakBranchV0>,
    pub saddles: Vec<SaddleNodeV0>,
    pub roots: Vec<u32>,
    /// Canonical peak ID that exclusively owns each active cell; background is
    /// `None`.
    pub cell_peak_owner: Vec<Option<u32>>,
    pub populations: HighlandPopulationsV0,
    pub reference_highlands: Vec<HighlandFeatureV0>,
    pub derived_evidence_hash: u64,
}

#[derive(Clone)]
struct TempPeak {
    peak_elevation_km: f64,
    anchor_cell: usize,
    flat_centroid_km: DVec3,
    flat_maximum_cells: Vec<usize>,
    parent_peak: Option<usize>,
    key_saddle: Option<usize>,
    persistence_km: Option<f64>,
    root_closure: bool,
    equal_elder_ambiguous: bool,
}

#[derive(Clone)]
struct TempSaddle {
    elevation_km: f64,
    anchor_cell: usize,
    flat_centroid_km: DVec3,
    flat_saddle_cells: Vec<usize>,
    elder_peak: usize,
    losing_peaks: Vec<usize>,
    equal_elder_ambiguous: bool,
}

fn weighted_centroid(graph: &EvaluationSurfaceGraphV0, cells: &[usize]) -> DVec3 {
    let mut weighted = DVec3::ZERO;
    let mut area = 0.0;
    for &cell in cells {
        weighted += graph.cell_area_km2[cell] * graph.cell_center_km[cell];
        area += graph.cell_area_km2[cell];
    }
    canonical_point(weighted / area)
}

fn cmp_point_lex(a: DVec3, b: DVec3) -> std::cmp::Ordering {
    a.x.total_cmp(&b.x)
        .then_with(|| a.y.total_cmp(&b.y))
        .then_with(|| a.z.total_cmp(&b.z))
}

fn find_survivor(parent: &mut [usize], mut peak: usize) -> usize {
    let mut root = peak;
    while parent[root] != root {
        root = parent[root];
    }
    while parent[peak] != peak {
        let next = parent[peak];
        parent[peak] = root;
        peak = next;
    }
    root
}

fn elder_cmp(peaks: &[TempPeak], a: usize, b: usize) -> std::cmp::Ordering {
    peaks[a]
        .peak_elevation_km
        .total_cmp(&peaks[b].peak_elevation_km)
        .then_with(|| cmp_point_lex(peaks[b].flat_centroid_km, peaks[a].flat_centroid_km))
        .then_with(|| peaks[b].anchor_cell.cmp(&peaks[a].anchor_cell))
}

/// Build the complete, unpruned exact-level split forest and the frozen
/// reference/sensitivity populations for the partial structural slice.
pub fn build_surface_hierarchy_v0(
    graph: &EvaluationSurfaceGraphV0,
    elevation_km: &[f64],
    scored_cell: &[bool],
    mut config: SurfaceHierarchyConfigV0,
) -> Result<SurfaceHierarchyV0, LandformError> {
    config.closure_level_km = canonical_zero(config.closure_level_km);
    graph.validate(&config)?;
    config.validate()?;
    let n = graph.cell_count();
    if elevation_km.len() != n {
        return Err(LandformError::LengthMismatch("elevation_km"));
    }
    if scored_cell.len() != n {
        return Err(LandformError::LengthMismatch("scored_cell"));
    }
    if !scored_cell.iter().any(|&scored| scored) {
        return Err(LandformError::EmptyScoredMask);
    }
    if elevation_km.iter().any(|value| !value.is_finite()) {
        return Err(LandformError::NonFiniteElevation);
    }
    let elevation: Vec<f64> = elevation_km.iter().copied().map(canonical_zero).collect();
    let closure = canonical_zero(config.closure_level_km);
    let is_active_input: Vec<bool> = (0..n)
        .map(|cell| scored_cell[cell] && elevation[cell] > closure)
        .collect();

    let mut order: Vec<usize> = (0..n).filter(|&cell| is_active_input[cell]).collect();
    order.sort_by(|&a, &b| {
        elevation[b]
            .total_cmp(&elevation[a])
            .then_with(|| a.cmp(&b))
    });
    let mut active = vec![false; n];
    let mut active_branch = vec![None::<usize>; n];
    let mut exclusive_owner = vec![None::<usize>; n];
    let mut peaks = Vec::<TempPeak>::new();
    let mut saddles = Vec::<TempSaddle>::new();
    let mut survivor_parent = Vec::<usize>::new();
    // Every active cell belongs to exactly one elevation batch, so one
    // lifetime visited bitset is sufficient. Reallocating and clearing an
    // n-cell bitset for every distinct sampled height makes smooth analytic
    // fields accidentally quadratic in the number of cells.
    let mut level_visited = vec![false; n];

    let mut group_start = 0usize;
    while group_start < order.len() {
        let bits = elevation[order[group_start]].to_bits();
        let mut group_end = group_start + 1;
        while group_end < order.len() && elevation[order[group_end]].to_bits() == bits {
            group_end += 1;
        }
        let level_cells = &order[group_start..group_end];
        let mut components = Vec::<Vec<usize>>::new();
        for &start in level_cells {
            if level_visited[start] {
                continue;
            }
            level_visited[start] = true;
            let mut queue = VecDeque::from([start]);
            let mut component = Vec::new();
            while let Some(cell) = queue.pop_front() {
                component.push(cell);
                for edge in edge_range(&graph.edge_offsets, cell) {
                    let neighbor = graph.edge_neighbor[edge] as usize;
                    if is_active_input[neighbor]
                        && elevation[neighbor].to_bits() == bits
                        && !level_visited[neighbor]
                    {
                        level_visited[neighbor] = true;
                        queue.push_back(neighbor);
                    }
                }
            }
            component.sort_unstable();
            components.push(component);
        }
        components.sort_by_key(|component| component[0]);

        // Freeze every incidence with the strictly higher superlevel set before
        // resolving anything at this elevation. Disconnected flats can still
        // be parts of one simultaneous saddle event when they touch the same
        // higher branches; processing them independently would make topology
        // and loser pairing depend on component/min-cell order.
        let mut component_touching = Vec::<Vec<usize>>::with_capacity(components.len());
        for component in &components {
            let mut touching = BTreeSet::new();
            for &cell in component {
                for edge in edge_range(&graph.edge_offsets, cell) {
                    let neighbor = graph.edge_neighbor[edge] as usize;
                    if active[neighbor] {
                        let branch = active_branch[neighbor]
                            .ok_or(LandformError::MultipleExclusiveOwners { cell: neighbor })?;
                        touching.insert(find_survivor(&mut survivor_parent, branch));
                    }
                }
            }
            component_touching.push(touching.into_iter().collect());
        }

        let mut branch_components = BTreeMap::<usize, Vec<usize>>::new();
        for (component, touching) in component_touching.iter().enumerate() {
            for &branch in touching {
                branch_components.entry(branch).or_default().push(component);
            }
        }
        let mut component_visited = vec![false; components.len()];
        let mut events = Vec::<Vec<usize>>::new();
        for start in 0..components.len() {
            if component_visited[start] {
                continue;
            }
            component_visited[start] = true;
            let mut queue = VecDeque::from([start]);
            let mut event = Vec::new();
            while let Some(component) = queue.pop_front() {
                event.push(component);
                for branch in &component_touching[component] {
                    for &linked in &branch_components[branch] {
                        if !component_visited[linked] {
                            component_visited[linked] = true;
                            queue.push_back(linked);
                        }
                    }
                }
            }
            event.sort_unstable();
            events.push(event);
        }
        events.sort_by_key(|event| components[event[0]][0]);

        for event in events {
            let mut event_cells = event
                .iter()
                .flat_map(|&component| components[component].iter().copied())
                .collect::<Vec<_>>();
            event_cells.sort_unstable();
            let touching = event
                .iter()
                .flat_map(|&component| component_touching[component].iter().copied())
                .collect::<BTreeSet<_>>();
            let branch = match touching.len() {
                0 => {
                    let peak = peaks.len();
                    let anchor_cell = event_cells[0];
                    peaks.push(TempPeak {
                        peak_elevation_km: elevation[anchor_cell],
                        anchor_cell,
                        flat_centroid_km: weighted_centroid(graph, &event_cells),
                        flat_maximum_cells: event_cells.clone(),
                        parent_peak: None,
                        key_saddle: None,
                        persistence_km: None,
                        root_closure: false,
                        equal_elder_ambiguous: false,
                    });
                    survivor_parent.push(peak);
                    peak
                }
                1 => *touching.first().unwrap(),
                _ => {
                    let candidates: Vec<usize> = touching.into_iter().collect();
                    let elder = candidates
                        .iter()
                        .copied()
                        .max_by(|&a, &b| elder_cmp(&peaks, a, b))
                        .unwrap();
                    let equal_elder_ambiguous = candidates.iter().any(|&candidate| {
                        candidate != elder
                            && peaks[candidate].peak_elevation_km.to_bits()
                                == peaks[elder].peak_elevation_km.to_bits()
                    });
                    let saddle_id = saddles.len();
                    let mut losing = Vec::new();
                    for candidate in candidates {
                        if candidate == elder {
                            continue;
                        }
                        let persistence =
                            peaks[candidate].peak_elevation_km - elevation[event_cells[0]];
                        if persistence < 0.0 {
                            return Err(LandformError::NegativePersistence);
                        }
                        peaks[candidate].parent_peak = Some(elder);
                        peaks[candidate].key_saddle = Some(saddle_id);
                        peaks[candidate].persistence_km = Some(canonical_zero(persistence));
                        peaks[candidate].equal_elder_ambiguous |= equal_elder_ambiguous;
                        survivor_parent[candidate] = elder;
                        losing.push(candidate);
                    }
                    losing.sort_unstable();
                    saddles.push(TempSaddle {
                        elevation_km: elevation[event_cells[0]],
                        anchor_cell: event_cells[0],
                        flat_centroid_km: weighted_centroid(graph, &event_cells),
                        flat_saddle_cells: event_cells.clone(),
                        elder_peak: elder,
                        losing_peaks: losing,
                        equal_elder_ambiguous,
                    });
                    elder
                }
            };
            for &cell in &event_cells {
                if exclusive_owner[cell].replace(branch).is_some() {
                    return Err(LandformError::MultipleExclusiveOwners { cell });
                }
                active_branch[cell] = Some(branch);
            }
        }
        for &cell in level_cells {
            active[cell] = true;
        }
        group_start = group_end;
    }

    let mut roots = BTreeSet::new();
    for &branch in &active_branch {
        if let Some(branch) = branch {
            roots.insert(find_survivor(&mut survivor_parent, branch));
        }
    }
    for &root in &roots {
        let persistence = peaks[root].peak_elevation_km - closure;
        if persistence < 0.0 {
            return Err(LandformError::NegativePersistence);
        }
        peaks[root].persistence_km = Some(canonical_zero(persistence));
        peaks[root].root_closure = true;
    }

    canonicalize_hierarchy(
        graph,
        &elevation,
        scored_cell,
        &config,
        peaks,
        saddles,
        roots,
        exclusive_owner,
    )
}

#[allow(clippy::too_many_arguments)]
fn canonicalize_hierarchy(
    graph: &EvaluationSurfaceGraphV0,
    elevation: &[f64],
    scored_cell: &[bool],
    config: &SurfaceHierarchyConfigV0,
    temp_peaks: Vec<TempPeak>,
    temp_saddles: Vec<TempSaddle>,
    temp_roots: BTreeSet<usize>,
    temp_owner: Vec<Option<usize>>,
) -> Result<SurfaceHierarchyV0, LandformError> {
    let mut peak_order: Vec<usize> = (0..temp_peaks.len()).collect();
    peak_order.sort_by(|&a, &b| {
        temp_peaks[b]
            .peak_elevation_km
            .total_cmp(&temp_peaks[a].peak_elevation_km)
            .then_with(|| {
                cmp_point_lex(
                    temp_peaks[a].flat_centroid_km,
                    temp_peaks[b].flat_centroid_km,
                )
            })
            .then_with(|| temp_peaks[a].anchor_cell.cmp(&temp_peaks[b].anchor_cell))
    });
    let mut peak_remap = vec![0usize; temp_peaks.len()];
    for (new, &old) in peak_order.iter().enumerate() {
        peak_remap[old] = new;
    }
    let mut saddle_order: Vec<usize> = (0..temp_saddles.len()).collect();
    saddle_order.sort_by(|&a, &b| {
        temp_saddles[b]
            .elevation_km
            .total_cmp(&temp_saddles[a].elevation_km)
            .then_with(|| {
                cmp_point_lex(
                    temp_saddles[a].flat_centroid_km,
                    temp_saddles[b].flat_centroid_km,
                )
            })
            .then_with(|| {
                temp_saddles[a]
                    .anchor_cell
                    .cmp(&temp_saddles[b].anchor_cell)
            })
    });
    let mut saddle_remap = vec![0usize; temp_saddles.len()];
    for (new, &old) in saddle_order.iter().enumerate() {
        saddle_remap[old] = new;
    }

    let parent: Vec<Option<usize>> = peak_order
        .iter()
        .map(|&old| temp_peaks[old].parent_peak.map(|p| peak_remap[p]))
        .collect();
    for start in 0..parent.len() {
        let mut seen = BTreeSet::new();
        let mut current = Some(start);
        while let Some(peak) = current {
            if !seen.insert(peak) {
                return Err(LandformError::CyclicParentage);
            }
            current = parent[peak];
        }
    }

    let cell_peak_owner: Vec<Option<u32>> = temp_owner
        .iter()
        .map(|owner| {
            owner
                .map(|old| u32::try_from(peak_remap[old]).map_err(|_| LandformError::Overflow))
                .transpose()
        })
        .collect::<Result<_, _>>()?;
    let mut exclusive = vec![Vec::<u32>::new(); temp_peaks.len()];
    for (cell, owner) in cell_peak_owner.iter().enumerate() {
        if let Some(owner) = owner {
            exclusive[*owner as usize]
                .push(u32::try_from(cell).map_err(|_| LandformError::Overflow)?);
        }
    }
    let mut footprints = vec![Vec::<u32>::new(); temp_peaks.len()];
    for (cell, owner) in cell_peak_owner.iter().enumerate() {
        let Some(mut peak) = owner.map(|value| value as usize) else {
            continue;
        };
        loop {
            footprints[peak].push(u32::try_from(cell).map_err(|_| LandformError::Overflow)?);
            let Some(next) = parent[peak] else {
                break;
            };
            peak = next;
        }
    }

    let mut peaks = Vec::with_capacity(temp_peaks.len());
    for (new, &old) in peak_order.iter().enumerate() {
        let source = &temp_peaks[old];
        let footprint_area_km2: f64 = footprints[new]
            .iter()
            .map(|&cell| graph.cell_area_km2[cell as usize])
            .sum();
        if !footprint_area_km2.is_finite() || footprint_area_km2 <= 0.0 {
            return Err(LandformError::FootprintAreaMismatch { peak: new });
        }
        let mut in_footprint = vec![false; graph.cell_count()];
        for &cell in &footprints[new] {
            in_footprint[cell as usize] = true;
        }
        let mut union_boundary_edges = Vec::new();
        let mut scored_boundary_contact = false;
        for &cell_u32 in &footprints[new] {
            let cell = cell_u32 as usize;
            for edge in edge_range(&graph.edge_offsets, cell) {
                let neighbor = graph.edge_neighbor[edge] as usize;
                if !in_footprint[neighbor] {
                    union_boundary_edges
                        .push(u32::try_from(edge).map_err(|_| LandformError::Overflow)?);
                }
                if !scored_cell[neighbor] {
                    scored_boundary_contact = true;
                }
            }
        }
        union_boundary_edges.sort_unstable();
        union_boundary_edges.dedup();
        let mut physical_boundary_segments: Vec<u32> = graph
            .boundary_segments
            .iter()
            .filter(|segment| in_footprint[segment.owner_cell as usize])
            .map(|segment| segment.id)
            .collect();
        physical_boundary_segments.sort_unstable();
        if !physical_boundary_segments.is_empty() {
            scored_boundary_contact = true;
        }
        let mut flat_maximum_cells: Vec<u32> = source
            .flat_maximum_cells
            .iter()
            .map(|&cell| u32::try_from(cell).map_err(|_| LandformError::Overflow))
            .collect::<Result<_, _>>()?;
        flat_maximum_cells.sort_unstable();
        peaks.push(PeakBranchV0 {
            id: u32::try_from(new).map_err(|_| LandformError::Overflow)?,
            peak_elevation_km: canonical_zero(source.peak_elevation_km),
            anchor_cell: u32::try_from(source.anchor_cell).map_err(|_| LandformError::Overflow)?,
            flat_centroid_km: source.flat_centroid_km,
            flat_maximum_cells,
            parent_peak: source
                .parent_peak
                .map(|p| u32::try_from(peak_remap[p]).map_err(|_| LandformError::Overflow))
                .transpose()?,
            key_saddle: source
                .key_saddle
                .map(|s| u32::try_from(saddle_remap[s]).map_err(|_| LandformError::Overflow))
                .transpose()?,
            persistence_km: source
                .persistence_km
                .ok_or(LandformError::NegativePersistence)?,
            root_closure: source.root_closure,
            equal_elder_ambiguous: source.equal_elder_ambiguous,
            exclusive_cells: exclusive[new].clone(),
            footprint_members: footprints[new].clone(),
            footprint_area_km2,
            union_boundary_edges,
            physical_boundary_segments,
            scored_boundary_contact,
        });
    }

    let mut saddles = Vec::with_capacity(temp_saddles.len());
    for (new, &old) in saddle_order.iter().enumerate() {
        let source = &temp_saddles[old];
        let mut cells: Vec<u32> = source
            .flat_saddle_cells
            .iter()
            .map(|&cell| u32::try_from(cell).map_err(|_| LandformError::Overflow))
            .collect::<Result<_, _>>()?;
        cells.sort_unstable();
        let mut losing: Vec<u32> = source
            .losing_peaks
            .iter()
            .map(|&peak| u32::try_from(peak_remap[peak]).map_err(|_| LandformError::Overflow))
            .collect::<Result<_, _>>()?;
        losing.sort_unstable();
        saddles.push(SaddleNodeV0 {
            id: u32::try_from(new).map_err(|_| LandformError::Overflow)?,
            elevation_km: source.elevation_km,
            anchor_cell: u32::try_from(source.anchor_cell).map_err(|_| LandformError::Overflow)?,
            flat_centroid_km: source.flat_centroid_km,
            flat_saddle_cells: cells,
            elder_peak: u32::try_from(peak_remap[source.elder_peak])
                .map_err(|_| LandformError::Overflow)?,
            losing_peaks: losing,
            equal_elder_ambiguous: source.equal_elder_ambiguous,
        });
    }
    let mut roots: Vec<u32> = temp_roots
        .into_iter()
        .map(|old| u32::try_from(peak_remap[old]).map_err(|_| LandformError::Overflow))
        .collect::<Result<_, _>>()?;
    roots.sort_unstable();

    let populations = build_populations(&peaks, config);
    let reference_highlands = build_reference_highlands(
        graph,
        elevation,
        scored_cell,
        config,
        &peaks,
        &populations.reference,
    )?;
    let mut hierarchy = SurfaceHierarchyV0 {
        schema_version: config.schema_version.to_owned(),
        hash_version: config.hash_version.to_owned(),
        peaks,
        saddles,
        roots,
        cell_peak_owner,
        populations,
        reference_highlands,
        derived_evidence_hash: 0,
    };
    let first = canonical_hierarchy_bytes(graph, elevation, scored_cell, config, &hierarchy)?;
    let second = canonical_hierarchy_bytes(graph, elevation, scored_cell, config, &hierarchy)?;
    if first != second {
        return Err(LandformError::Serialization(
            "immediate canonical serialization mismatch".into(),
        ));
    }
    hierarchy.derived_evidence_hash = fnv1a64(&first);
    Ok(hierarchy)
}

fn build_populations(
    peaks: &[PeakBranchV0],
    config: &SurfaceHierarchyConfigV0,
) -> HighlandPopulationsV0 {
    let select = |persistence: f64, area: f64| {
        peaks
            .iter()
            .filter(|peak| peak.persistence_km >= persistence && peak.footprint_area_km2 >= area)
            .map(|peak| peak.id)
            .collect()
    };
    HighlandPopulationsV0 {
        reference: select(
            config.reference_persistence_km,
            config.reference_min_footprint_km2,
        ),
        persistence_low: select(
            config.persistence_sensitivity_km[0],
            config.reference_min_footprint_km2,
        ),
        persistence_high: select(
            config.persistence_sensitivity_km[1],
            config.reference_min_footprint_km2,
        ),
        footprint_low: select(
            config.reference_persistence_km,
            config.footprint_sensitivity_km2[0],
        ),
        footprint_high: select(
            config.reference_persistence_km,
            config.footprint_sensitivity_km2[1],
        ),
    }
}

fn planar_polygon_raw_moments_about(polygon: &[DVec3], origin: DVec3) -> [f64; 6] {
    let mut area_twice = 0.0;
    let mut first_x_six = 0.0;
    let mut first_y_six = 0.0;
    let mut second_x_twelve = 0.0;
    let mut second_y_twelve = 0.0;
    let mut second_xy_twenty_four = 0.0;
    for (a, b) in polygon
        .iter()
        .copied()
        .zip(polygon.iter().copied().cycle().skip(1))
        .take(polygon.len())
    {
        let a = a - origin;
        let b = b - origin;
        let cross = a.x * b.y - b.x * a.y;
        area_twice += cross;
        first_x_six += (a.x + b.x) * cross;
        first_y_six += (a.y + b.y) * cross;
        second_x_twelve += (a.x * a.x + a.x * b.x + b.x * b.x) * cross;
        second_y_twelve += (a.y * a.y + a.y * b.y + b.y * b.y) * cross;
        second_xy_twenty_four +=
            (2.0 * a.x * a.y + a.x * b.y + b.x * a.y + 2.0 * b.x * b.y) * cross;
    }
    [
        0.5 * area_twice,
        first_x_six / 6.0,
        first_y_six / 6.0,
        second_x_twelve / 12.0,
        second_xy_twenty_four / 24.0,
        second_y_twelve / 12.0,
    ]
}

#[cfg(test)]
fn planar_polygon_raw_moments(polygon: &[DVec3]) -> [f64; 6] {
    let origin = polygon.first().copied().unwrap_or(DVec3::ZERO);
    let local = planar_polygon_raw_moments_about(polygon, origin);
    let area = local[0];
    let first_x = local[1] + origin.x * area;
    let first_y = local[2] + origin.y * area;
    [
        area,
        first_x,
        first_y,
        local[3] + 2.0 * origin.x * local[1] + origin.x * origin.x * area,
        local[4] + origin.x * local[2] + origin.y * local[1] + origin.x * origin.y * area,
        local[5] + 2.0 * origin.y * local[2] + origin.y * origin.y * area,
    ]
}

fn measure_planar_footprint(
    graph: &EvaluationSurfaceGraphV0,
    members: &[u32],
    config: &SurfaceHierarchyConfigV0,
    peak_id: usize,
) -> Result<PlanarFootprintGeometryV0, LandformError> {
    let origin_cell = members.iter().copied().min().unwrap() as usize;
    let origin = graph.cell_center_km[origin_cell];
    let mut raw = [0.0; 6];
    for &cell in members {
        let moments = planar_polygon_raw_moments_about(graph.polygon(cell as usize), origin);
        for (sum, value) in raw.iter_mut().zip(moments) {
            *sum += value;
        }
    }
    let area = raw[0];
    if !area.is_finite() || area <= 0.0 || raw.iter().any(|value| !value.is_finite()) {
        return Err(LandformError::InvalidPlanarMoments { peak: peak_id });
    }
    let authoritative_area = members
        .iter()
        .map(|&cell| graph.cell_area_km2[cell as usize])
        .sum::<f64>();
    let area_relative_error =
        (area - authoritative_area).abs() / authoritative_area.max(f64::MIN_POSITIVE);
    if area_relative_error > config.planar_area_match_relative {
        return Err(LandformError::FootprintAreaMismatch { peak: peak_id });
    }
    let local_centroid_x = raw[1] / area;
    let local_centroid_y = raw[2] / area;
    let raw_second_x = raw[3] / area;
    let raw_second_y = raw[5] / area;
    let numeric_scale = raw_second_x
        .abs()
        .max(raw_second_y.abs())
        .max(f64::MIN_POSITIVE);
    let negative_tolerance = config.linear_rank_relative * numeric_scale;
    let clamp_covariance = |value: f64| {
        if value < -negative_tolerance {
            Err(LandformError::InvalidPlanarMoments { peak: peak_id })
        } else {
            Ok(value.max(0.0))
        }
    };
    let covariance_xx = clamp_covariance(raw_second_x - local_centroid_x * local_centroid_x)?;
    let covariance_xy = raw[4] / area - local_centroid_x * local_centroid_y;
    let covariance_yy = clamp_covariance(raw_second_y - local_centroid_y * local_centroid_y)?;
    let trace = covariance_xx + covariance_yy;
    let discriminant = (covariance_xx - covariance_yy).hypot(2.0 * covariance_xy);
    let lambda_1 = 0.5 * (trace + discriminant);
    let mut lambda_2 = 0.5 * (trace - discriminant);
    let eigenvalue_tolerance = config.linear_rank_relative * lambda_1.max(numeric_scale);
    if lambda_2 < -eigenvalue_tolerance {
        return Err(LandformError::InvalidPlanarMoments { peak: peak_id });
    }
    if lambda_2 < 0.0 {
        lambda_2 = 0.0;
    }
    if !lambda_1.is_finite() || !lambda_2.is_finite() || lambda_1 <= 0.0 {
        return Err(LandformError::InvalidPlanarMoments { peak: peak_id });
    }
    let anisotropy = canonical_zero((lambda_1 - lambda_2) / (lambda_1 + lambda_2));
    let candidate_a = DVec3::new(covariance_xy, lambda_1 - covariance_xx, 0.0);
    let candidate_b = DVec3::new(lambda_1 - covariance_yy, covariance_xy, 0.0);
    let mut principal = if candidate_a.length_squared() > candidate_b.length_squared()
        && candidate_a.length_squared() > 0.0
    {
        candidate_a.normalize()
    } else if candidate_b.length_squared() > 0.0 {
        candidate_b.normalize()
    } else if covariance_xx >= covariance_yy {
        DVec3::X
    } else {
        DVec3::Y
    };
    if principal.x < 0.0 || (principal.x == 0.0 && principal.y < 0.0) {
        principal = -principal;
    }
    principal = canonical_point(principal);

    let start = members.iter().copied().min().unwrap() as usize;
    let farthest = |origin: usize| {
        let mut selected = members[0] as usize;
        let mut selected_distance = -1.0;
        for &candidate_u32 in members {
            let candidate = candidate_u32 as usize;
            let distance =
                graph.cell_center_km[origin].distance_squared(graph.cell_center_km[candidate]);
            if distance > selected_distance
                || (distance.to_bits() == selected_distance.to_bits() && candidate < selected)
            {
                selected = candidate;
                selected_distance = distance;
            }
        }
        selected
    };
    let first = farthest(start);
    let second = farthest(first);
    let extent = canonical_zero(graph.cell_center_km[first].distance(graph.cell_center_km[second]));
    let (two_sweep_extent_km, mean_width_km) = if extent > 0.0 {
        (Some(extent), Some(canonical_zero(area / extent)))
    } else {
        (None, None)
    };

    Ok(PlanarFootprintGeometryV0 {
        area_km2: canonical_zero(area),
        centroid_km: canonical_point(DVec3::new(
            origin.x + local_centroid_x,
            origin.y + local_centroid_y,
            0.0,
        )),
        covariance_km2: [
            canonical_zero(covariance_xx),
            canonical_zero(covariance_xy),
            canonical_zero(covariance_xy),
            canonical_zero(covariance_yy),
        ],
        equivalent_ellipse_length_km: canonical_zero(4.0 * lambda_1.sqrt()),
        equivalent_ellipse_width_km: canonical_zero(4.0 * lambda_2.sqrt()),
        anisotropy,
        principal_axis: principal,
        orientation_ambiguous: anisotropy < config.orientation_ambiguity_anisotropy,
        two_sweep_extent_km,
        mean_width_km,
    })
}

fn sign_canonical_direction(mut direction: DVec3) -> DVec3 {
    if direction.x < 0.0
        || (direction.x == 0.0 && direction.y < 0.0)
        || (direction.x == 0.0 && direction.y == 0.0 && direction.z < 0.0)
    {
        direction = -direction;
    }
    canonical_point(direction)
}

fn spherical_tangent_basis(center: DVec3) -> Result<[DVec3; 2], LandformError> {
    if !center.is_finite() || center.length_squared() == 0.0 {
        return Err(LandformError::InvalidSphericalGeometry);
    }
    let center = center.normalize();
    let axes = [DVec3::X, DVec3::Y, DVec3::Z];
    let mut selected = 0usize;
    let mut alignment = center.dot(axes[0]).abs();
    for (index, axis) in axes.iter().enumerate().skip(1) {
        let candidate = center.dot(*axis).abs();
        if candidate < alignment {
            selected = index;
            alignment = candidate;
        }
    }
    let e1 = sign_canonical_direction(axes[selected].cross(center).normalize());
    let e2 = canonical_point(center.cross(e1));
    if !e1.is_finite() || !e2.is_finite() {
        return Err(LandformError::InvalidSphericalGeometry);
    }
    Ok([e1, e2])
}

/// Azimuthal-equidistant log coordinates. `None` is the registered antipodal
/// outcome; malformed or non-finite geometry remains fatal.
fn spherical_log_xy(
    center: DVec3,
    target: DVec3,
    radius_km: f64,
    basis: [DVec3; 2],
    rank_tolerance: f64,
) -> Result<Option<DVec2>, LandformError> {
    if !center.is_finite()
        || !target.is_finite()
        || center.length_squared() == 0.0
        || target.length_squared() == 0.0
        || !radius_km.is_finite()
        || radius_km <= 0.0
    {
        return Err(LandformError::InvalidSphericalGeometry);
    }
    let center = center.normalize();
    let target = target.normalize();
    let dot = center.dot(target).clamp(-1.0, 1.0);
    let cross_length = center.cross(target).length();
    let theta = cross_length.atan2(dot);
    if !theta.is_finite() {
        return Err(LandformError::InvalidSphericalGeometry);
    }
    if std::f64::consts::PI - theta <= rank_tolerance {
        return Ok(None);
    }
    if theta == 0.0 {
        return Ok(Some(DVec2::ZERO));
    }
    let tangent = target - dot * center;
    let tangent_length = tangent.length();
    if !tangent_length.is_finite() || tangent_length == 0.0 {
        return Err(LandformError::InvalidSphericalGeometry);
    }
    let displacement = radius_km * theta * (tangent / tangent_length);
    let result = DVec2::new(displacement.dot(basis[0]), displacement.dot(basis[1]));
    if !result.is_finite() {
        return Err(LandformError::InvalidSphericalGeometry);
    }
    Ok(Some(result))
}

fn cross_2d(a: DVec2, b: DVec2) -> f64 {
    a.x * b.y - a.y * b.x
}

fn point_on_closed_segment_2d(point: DVec2, a: DVec2, b: DVec2) -> bool {
    cross_2d(b - a, point - a) == 0.0
        && point.x >= a.x.min(b.x)
        && point.x <= a.x.max(b.x)
        && point.y >= a.y.min(b.y)
        && point.y <= a.y.max(b.y)
}

fn closed_segments_intersect_2d(a: DVec2, b: DVec2, c: DVec2, d: DVec2) -> bool {
    let ab_c = cross_2d(b - a, c - a);
    let ab_d = cross_2d(b - a, d - a);
    let cd_a = cross_2d(d - c, a - c);
    let cd_b = cross_2d(d - c, b - c);
    (ab_c.signum() != ab_d.signum()
        && cd_a.signum() != cd_b.signum()
        && ab_c != 0.0
        && ab_d != 0.0
        && cd_a != 0.0
        && cd_b != 0.0)
        || (ab_c == 0.0 && point_on_closed_segment_2d(c, a, b))
        || (ab_d == 0.0 && point_on_closed_segment_2d(d, a, b))
        || (cd_a == 0.0 && point_on_closed_segment_2d(a, c, d))
        || (cd_b == 0.0 && point_on_closed_segment_2d(b, c, d))
}

fn projected_polygon_is_simple(polygon: &[DVec2]) -> bool {
    let n = polygon.len();
    if n < 3
        || polygon.iter().any(|point| !point.is_finite())
        || (0..n).any(|index| polygon[index] == polygon[(index + 1) % n])
    {
        return false;
    }
    for first in 0..n {
        let first_next = (first + 1) % n;
        for second in (first + 1)..n {
            let second_next = (second + 1) % n;
            if first == second
                || first_next == second
                || second_next == first
                || (first == 0 && second_next == 0)
            {
                continue;
            }
            if closed_segments_intersect_2d(
                polygon[first],
                polygon[first_next],
                polygon[second],
                polygon[second_next],
            ) {
                return false;
            }
        }
    }
    true
}

fn spherical_polygon_centroid_direction(
    polygon: &[DVec3],
    peak_id: usize,
) -> Result<DVec3, LandformError> {
    let mut resultant = DVec3::ZERO;
    for (&a, &b) in polygon.iter().zip(polygon.iter().cycle().skip(1)) {
        let (a, b) = (a.normalize(), b.normalize());
        let cross = a.cross(b);
        let cross_length = cross.length();
        let angle = cross_length.atan2(a.dot(b));
        if !angle.is_finite() || !cross_length.is_finite() || cross_length == 0.0 {
            return Err(LandformError::InvalidSphericalMoments { peak: peak_id });
        }
        resultant += angle * (cross / cross_length);
    }
    if !resultant.is_finite() || resultant.length_squared() == 0.0 {
        return Err(LandformError::InvalidSphericalMoments { peak: peak_id });
    }
    Ok(resultant.normalize())
}

fn spherical_two_sweep_extent(
    graph: &EvaluationSurfaceGraphV0,
    members: &[u32],
    area_km2: f64,
    radius_km: f64,
) -> Result<(Option<f64>, Option<f64>), LandformError> {
    let start = members.iter().copied().min().unwrap() as usize;
    let farthest = |origin: usize| -> Result<usize, LandformError> {
        let mut selected = members[0] as usize;
        let mut selected_distance = -1.0;
        for &candidate_u32 in members {
            let candidate = candidate_u32 as usize;
            let distance = if candidate == origin {
                0.0
            } else {
                radius_km
                    * spherical_angle_rad(
                        graph.cell_center_km[origin],
                        graph.cell_center_km[candidate],
                    )?
            };
            if distance > selected_distance
                || (distance.to_bits() == selected_distance.to_bits() && candidate < selected)
            {
                selected = candidate;
                selected_distance = distance;
            }
        }
        Ok(selected)
    };
    let first = farthest(start)?;
    let second = farthest(first)?;
    let extent = if first == second {
        0.0
    } else {
        radius_km * spherical_angle_rad(graph.cell_center_km[first], graph.cell_center_km[second])?
    };
    if extent > 0.0 {
        Ok((
            Some(canonical_zero(extent)),
            Some(canonical_zero(area_km2 / extent)),
        ))
    } else {
        Ok((None, None))
    }
}

fn measure_spherical_footprint(
    graph: &EvaluationSurfaceGraphV0,
    members: &[u32],
    config: &SurfaceHierarchyConfigV0,
    peak_id: usize,
) -> Result<(SphericalFootprintGeometryV0, Option<f64>, Option<f64>), LandformError> {
    let EvaluationDomainV0::Spherical { radius_km } = graph.domain else {
        return Err(LandformError::UnsupportedDomain);
    };
    let authoritative_area = compensated_sum(
        members
            .iter()
            .map(|&cell| graph.cell_area_km2[cell as usize]),
    );
    if !authoritative_area.is_finite() || authoritative_area <= 0.0 {
        return Err(LandformError::InvalidSphericalMoments { peak: peak_id });
    }
    let mut resultant = DVec3::ZERO;
    let mut resultant_correction = DVec3::ZERO;
    for &cell in members {
        let cell = cell as usize;
        let value = graph.cell_area_km2[cell]
            * spherical_polygon_centroid_direction(graph.polygon(cell), peak_id)?;
        let adjusted = value - resultant_correction;
        let next = resultant + adjusted;
        resultant_correction = (next - resultant) - adjusted;
        resultant = next;
    }
    if !resultant.is_finite() {
        return Err(LandformError::InvalidSphericalMoments { peak: peak_id });
    }
    let extent = spherical_two_sweep_extent(graph, members, authoritative_area, radius_km)?;
    if resultant.length() / authoritative_area <= config.linear_rank_relative {
        return Ok((
            SphericalFootprintGeometryV0::NonLocalGeometry,
            extent.0,
            extent.1,
        ));
    }
    let center = resultant.normalize();
    let basis = spherical_tangent_basis(center)?;
    let mut raw = [0.0; 6];
    let mut maximum_angular_radius_rad = 0.0_f64;
    for &cell in members {
        let cell = cell as usize;
        let mut projected = Vec::with_capacity(graph.polygon(cell).len());
        for &vertex in graph.polygon(cell) {
            let unit = vertex.normalize();
            let theta = center.cross(unit).length().atan2(center.dot(unit));
            maximum_angular_radius_rad = maximum_angular_radius_rad.max(theta);
            let Some(point) =
                spherical_log_xy(center, unit, radius_km, basis, config.linear_rank_relative)?
            else {
                return Ok((
                    SphericalFootprintGeometryV0::NonLocalGeometry,
                    extent.0,
                    extent.1,
                ));
            };
            projected.push(point);
        }
        if !projected_polygon_is_simple(&projected) {
            return Ok((
                SphericalFootprintGeometryV0::NonLocalGeometry,
                extent.0,
                extent.1,
            ));
        }
        let projected_3d = projected
            .iter()
            .map(|point| DVec3::new(point.x, point.y, 0.0))
            .collect::<Vec<_>>();
        let moments = planar_polygon_raw_moments_about(&projected_3d, DVec3::ZERO);
        if !moments[0].is_finite() || moments[0] <= 0.0 {
            return Ok((
                SphericalFootprintGeometryV0::NonLocalGeometry,
                extent.0,
                extent.1,
            ));
        }
        let scale = graph.cell_area_km2[cell] / moments[0];
        if !scale.is_finite() || moments.iter().any(|value| !value.is_finite()) {
            return Err(LandformError::InvalidSphericalMoments { peak: peak_id });
        }
        for (sum, value) in raw.iter_mut().zip(moments) {
            *sum += scale * value;
        }
    }
    if raw.iter().any(|value| !value.is_finite()) {
        return Err(LandformError::InvalidSphericalMoments { peak: peak_id });
    }
    if (raw[0] - authoritative_area).abs() / authoritative_area
        > config.sphere_area_closure_relative
    {
        return Err(LandformError::FootprintAreaMismatch { peak: peak_id });
    }
    let area = raw[0];
    let centroid_x = raw[1] / area;
    let centroid_y = raw[2] / area;
    let raw_second_x = raw[3] / area;
    let raw_second_y = raw[5] / area;
    let numeric_scale = raw_second_x
        .abs()
        .max(raw_second_y.abs())
        .max(f64::MIN_POSITIVE);
    let negative_tolerance = config.linear_rank_relative * numeric_scale;
    let clamp_covariance = |value: f64| {
        if value < -negative_tolerance {
            Err(LandformError::InvalidSphericalMoments { peak: peak_id })
        } else {
            Ok(value.max(0.0))
        }
    };
    let covariance_xx = clamp_covariance(raw_second_x - centroid_x * centroid_x)?;
    let covariance_xy = raw[4] / area - centroid_x * centroid_y;
    let covariance_yy = clamp_covariance(raw_second_y - centroid_y * centroid_y)?;
    let trace = covariance_xx + covariance_yy;
    let discriminant = (covariance_xx - covariance_yy).hypot(2.0 * covariance_xy);
    let lambda_1 = 0.5 * (trace + discriminant);
    let mut lambda_2 = 0.5 * (trace - discriminant);
    let eigenvalue_tolerance = config.linear_rank_relative * lambda_1.max(numeric_scale);
    if lambda_2 < -eigenvalue_tolerance {
        return Err(LandformError::InvalidSphericalMoments { peak: peak_id });
    }
    lambda_2 = lambda_2.max(0.0);
    if !lambda_1.is_finite() || !lambda_2.is_finite() || lambda_1 <= 0.0 {
        return Err(LandformError::InvalidSphericalMoments { peak: peak_id });
    }
    let anisotropy = canonical_zero((lambda_1 - lambda_2) / (lambda_1 + lambda_2));
    let candidate_a = DVec2::new(covariance_xy, lambda_1 - covariance_xx);
    let candidate_b = DVec2::new(lambda_1 - covariance_yy, covariance_xy);
    let principal_2d = if candidate_a.length_squared() > candidate_b.length_squared()
        && candidate_a.length_squared() > 0.0
    {
        candidate_a.normalize()
    } else if candidate_b.length_squared() > 0.0 {
        candidate_b.normalize()
    } else if covariance_xx >= covariance_yy {
        DVec2::X
    } else {
        DVec2::Y
    };
    let principal_axis =
        sign_canonical_direction(principal_2d.x * basis[0] + principal_2d.y * basis[1]);
    let local = SphericalLocalFootprintGeometryV0 {
        area_km2: canonical_zero(area),
        centroid_km: canonical_point(radius_km * center),
        projected_centroid_km: [canonical_zero(centroid_x), canonical_zero(centroid_y)],
        tangent_covariance_km2: [
            canonical_zero(covariance_xx),
            canonical_zero(covariance_xy),
            canonical_zero(covariance_xy),
            canonical_zero(covariance_yy),
        ],
        equivalent_ellipse_length_km: canonical_zero(4.0 * lambda_1.sqrt()),
        equivalent_ellipse_width_km: canonical_zero(4.0 * lambda_2.sqrt()),
        anisotropy,
        principal_axis,
        orientation_ambiguous: anisotropy < config.orientation_ambiguity_anisotropy,
        maximum_angular_radius_rad: canonical_zero(maximum_angular_radius_rad),
        spherical_nonlocal_warning: maximum_angular_radius_rad
            > config.spherical_nonlocal_radius_rad,
    };
    Ok((
        SphericalFootprintGeometryV0::Local(local),
        extent.0,
        extent.1,
    ))
}

fn physical_grades(
    graph: &EvaluationSurfaceGraphV0,
    elevation: &[f64],
    scored_cell: &[bool],
    config: &SurfaceHierarchyConfigV0,
) -> Result<Vec<Option<f64>>, LandformError> {
    let mut grades = vec![None; graph.cell_count()];
    for cell in 0..graph.cell_count() {
        if !scored_cell[cell] {
            continue;
        }
        let mut matrix_xx = 0.0;
        let mut matrix_xy = 0.0;
        let mut matrix_yy = 0.0;
        let mut rhs_x = 0.0;
        let mut rhs_y = 0.0;
        let spherical_basis = match graph.domain {
            EvaluationDomainV0::Planar => None,
            EvaluationDomainV0::Spherical { .. } => {
                Some(spherical_tangent_basis(graph.cell_center_km[cell])?)
            }
        };
        for edge in edge_range(&graph.edge_offsets, cell) {
            let neighbor = graph.edge_neighbor[edge] as usize;
            if !scored_cell[neighbor] {
                continue;
            }
            let displacement = match graph.domain {
                EvaluationDomainV0::Planar => {
                    let displacement = graph.cell_center_km[neighbor] - graph.cell_center_km[cell];
                    DVec2::new(displacement.x, displacement.y)
                }
                EvaluationDomainV0::Spherical { radius_km } => spherical_log_xy(
                    graph.cell_center_km[cell],
                    graph.cell_center_km[neighbor],
                    radius_km,
                    spherical_basis.expect("spherical cells have a tangent basis"),
                    config.linear_rank_relative,
                )?
                .ok_or(LandformError::InvalidSphericalGeometry)?,
            };
            let weight = graph.edge_shared_width_km[edge] / graph.edge_distance_km[edge];
            let dz = elevation[neighbor] - elevation[cell];
            matrix_xx += weight * displacement.x * displacement.x;
            matrix_xy += weight * displacement.x * displacement.y;
            matrix_yy += weight * displacement.y * displacement.y;
            rhs_x += weight * displacement.x * dz;
            rhs_y += weight * displacement.y * dz;
        }
        let trace = matrix_xx + matrix_yy;
        let discriminant = (matrix_xx - matrix_yy).hypot(2.0 * matrix_xy);
        let larger = 0.5 * (trace + discriminant);
        let smaller = 0.5 * (trace - discriminant);
        if !matrix_xx.is_finite()
            || !matrix_xy.is_finite()
            || !matrix_yy.is_finite()
            || !rhs_x.is_finite()
            || !rhs_y.is_finite()
            || !larger.is_finite()
            || !smaller.is_finite()
        {
            return Err(LandformError::NonFiniteDerived {
                measurement: match graph.domain {
                    EvaluationDomainV0::Planar => "planar_grade",
                    EvaluationDomainV0::Spherical { .. } => "spherical_grade",
                },
                cell,
            });
        }
        if larger <= 0.0 || smaller <= config.linear_rank_relative * larger {
            continue;
        }
        let determinant = matrix_xx * matrix_yy - matrix_xy * matrix_xy;
        if !determinant.is_finite() || determinant <= 0.0 {
            return Err(LandformError::NonFiniteDerived {
                measurement: match graph.domain {
                    EvaluationDomainV0::Planar => "planar_grade",
                    EvaluationDomainV0::Spherical { .. } => "spherical_grade",
                },
                cell,
            });
        }
        let gradient_x = (matrix_yy * rhs_x - matrix_xy * rhs_y) / determinant;
        let gradient_y = (matrix_xx * rhs_y - matrix_xy * rhs_x) / determinant;
        let grade = canonical_zero(gradient_x.hypot(gradient_y));
        if !grade.is_finite() {
            return Err(LandformError::NonFiniteDerived {
                measurement: match graph.domain {
                    EvaluationDomainV0::Planar => "planar_grade",
                    EvaluationDomainV0::Spherical { .. } => "spherical_grade",
                },
                cell,
            });
        }
        grades[cell] = Some(grade);
    }
    Ok(grades)
}

#[cfg(test)]
fn planar_grades(
    graph: &EvaluationSurfaceGraphV0,
    elevation: &[f64],
    scored_cell: &[bool],
    config: &SurfaceHierarchyConfigV0,
) -> Result<Vec<Option<f64>>, LandformError> {
    physical_grades(graph, elevation, scored_cell, config)
}

fn point_segment_distance(point: DVec3, endpoints: [DVec3; 2]) -> f64 {
    let segment = endpoints[1] - endpoints[0];
    let length_squared = segment.length_squared();
    if length_squared == 0.0 {
        return point.distance(endpoints[0]);
    }
    let t = ((point - endpoints[0]).dot(segment) / length_squared).clamp(0.0, 1.0);
    point.distance(endpoints[0] + t * segment)
}

fn spherical_angle_rad(a: DVec3, b: DVec3) -> Result<f64, LandformError> {
    if !a.is_finite() || !b.is_finite() || a.length_squared() == 0.0 || b.length_squared() == 0.0 {
        return Err(LandformError::InvalidSphericalGeometry);
    }
    let (a, b) = (a.normalize(), b.normalize());
    let angle = a.cross(b).length().atan2(a.dot(b));
    if !angle.is_finite() {
        return Err(LandformError::InvalidSphericalGeometry);
    }
    Ok(angle)
}

fn spherical_point_minor_arc_distance_km(
    point: DVec3,
    endpoints: [DVec3; 2],
    radius_km: f64,
) -> Result<f64, LandformError> {
    let point = point.normalize();
    let a = endpoints[0].normalize();
    let b = endpoints[1].normalize();
    let cross = a.cross(b);
    let cross_length = cross.length();
    let delta = cross_length.atan2(a.dot(b));
    if !delta.is_finite()
        || delta <= 0.0
        || delta >= std::f64::consts::PI
        || !cross_length.is_finite()
        || cross_length == 0.0
        || !radius_km.is_finite()
        || radius_km <= 0.0
    {
        return Err(LandformError::InvalidSphericalGeometry);
    }
    let normal = cross / cross_length;
    let projected = point - point.dot(normal) * normal;
    let mut minimum = spherical_angle_rad(point, a)?.min(spherical_angle_rad(point, b)?);
    if projected.length_squared() > 0.0 && projected.is_finite() {
        let projection = projected.normalize();
        for candidate in [projection, -projection] {
            let mut parameter = normal.dot(a.cross(candidate)).atan2(a.dot(candidate));
            if parameter < 0.0 {
                parameter += std::f64::consts::TAU;
            }
            if parameter <= delta {
                minimum = minimum.min(spherical_angle_rad(point, candidate)?);
            }
        }
    }
    Ok(canonical_zero(radius_km * minimum))
}

struct PlanarReliefFields {
    relief_by_radius: Vec<Vec<Option<f64>>>,
    truncated_by_radius: Vec<Vec<bool>>,
}

fn planar_relief_fields(
    graph: &EvaluationSurfaceGraphV0,
    elevation: &[f64],
    scored_cell: &[bool],
    config: &SurfaceHierarchyConfigV0,
    query_cells: &[u32],
) -> PlanarReliefFields {
    let radii = config.local_relief_radii_km;
    // Exact disk queries can consume an entire bucket's extrema when its
    // bounding box lies inside the disk. Only the intersected rim enumerates
    // individual cells, avoiding work proportional to R² / h² for every
    // footprint member on fine analytic meshes.
    let bucket_size = radii[0] / 3.0;
    struct ReliefBucket {
        cells: Vec<usize>,
        minimum: f64,
        maximum: f64,
    }
    let mut buckets = BTreeMap::<(i64, i64), ReliefBucket>::new();
    for (cell, &scored) in scored_cell.iter().enumerate() {
        if scored {
            let center = graph.cell_center_km[cell];
            let bucket = buckets
                .entry((
                    (center.x / bucket_size).floor() as i64,
                    (center.y / bucket_size).floor() as i64,
                ))
                .or_insert_with(|| ReliefBucket {
                    cells: Vec::new(),
                    minimum: f64::INFINITY,
                    maximum: f64::NEG_INFINITY,
                });
            bucket.cells.push(cell);
            bucket.minimum = bucket.minimum.min(elevation[cell]);
            bucket.maximum = bucket.maximum.max(elevation[cell]);
        }
    }
    let scored_boundaries = graph
        .boundary_segments
        .iter()
        .map(|segment| segment.endpoints_km)
        .chain((0..graph.cell_count()).flat_map(|cell| {
            edge_range(&graph.edge_offsets, cell).filter_map(move |edge| {
                let neighbor = graph.edge_neighbor[edge] as usize;
                (scored_cell[cell] && !scored_cell[neighbor])
                    .then_some(graph.edge_face_endpoints_km[edge])
            })
        }))
        .collect::<Vec<_>>();
    let boundary_bucket_size = *radii.last().unwrap();
    let mut boundary_buckets = BTreeMap::<(i64, i64), Vec<[DVec3; 2]>>::new();
    for segment in scored_boundaries {
        let min_x = segment[0].x.min(segment[1].x);
        let max_x = segment[0].x.max(segment[1].x);
        let min_y = segment[0].y.min(segment[1].y);
        let max_y = segment[0].y.max(segment[1].y);
        let first_x = (min_x / boundary_bucket_size).floor() as i64;
        let last_x = (max_x / boundary_bucket_size).floor() as i64;
        let first_y = (min_y / boundary_bucket_size).floor() as i64;
        let last_y = (max_y / boundary_bucket_size).floor() as i64;
        for bx in first_x..=last_x {
            for by in first_y..=last_y {
                boundary_buckets.entry((bx, by)).or_default().push(segment);
            }
        }
    }

    let mut relief_by_radius = vec![vec![None; graph.cell_count()]; radii.len()];
    let mut truncated_by_radius = vec![vec![false; graph.cell_count()]; radii.len()];
    for &cell_u32 in query_cells {
        let cell = cell_u32 as usize;
        debug_assert!(scored_cell[cell]);
        let center = graph.cell_center_km[cell];
        let bucket = (
            (center.x / bucket_size).floor() as i64,
            (center.y / bucket_size).floor() as i64,
        );
        let boundary_bucket = (
            (center.x / boundary_bucket_size).floor() as i64,
            (center.y / boundary_bucket_size).floor() as i64,
        );
        let mut boundary_distance = f64::INFINITY;
        for bx in boundary_bucket.0 - 1..=boundary_bucket.0 + 1 {
            for by in boundary_bucket.1 - 1..=boundary_bucket.1 + 1 {
                if let Some(segments) = boundary_buckets.get(&(bx, by)) {
                    for &segment in segments {
                        boundary_distance =
                            boundary_distance.min(point_segment_distance(center, segment));
                    }
                }
            }
        }
        for (radius_index, &radius) in radii.iter().enumerate() {
            let reach = (radius / bucket_size).ceil() as i64;
            let mut minimum = f64::INFINITY;
            let mut maximum = f64::NEG_INFINITY;
            for bx in bucket.0 - reach..=bucket.0 + reach {
                for by in bucket.1 - reach..=bucket.1 + reach {
                    if let Some(candidate_bucket) = buckets.get(&(bx, by)) {
                        let x0 = bx as f64 * bucket_size;
                        let x1 = x0 + bucket_size;
                        let y0 = by as f64 * bucket_size;
                        let y1 = y0 + bucket_size;
                        let farthest_x = (center.x - x0).abs().max((center.x - x1).abs());
                        let farthest_y = (center.y - y0).abs().max((center.y - y1).abs());
                        let farthest_squared = farthest_x * farthest_x + farthest_y * farthest_y;
                        if farthest_squared <= radius * radius {
                            minimum = minimum.min(candidate_bucket.minimum);
                            maximum = maximum.max(candidate_bucket.maximum);
                            continue;
                        }
                        let nearest_x = center.x.clamp(x0, x1);
                        let nearest_y = center.y.clamp(y0, y1);
                        if (center.x - nearest_x).powi(2) + (center.y - nearest_y).powi(2)
                            > radius * radius
                        {
                            continue;
                        }
                        for &candidate in &candidate_bucket.cells {
                            if center.distance(graph.cell_center_km[candidate]) <= radius {
                                minimum = minimum.min(elevation[candidate]);
                                maximum = maximum.max(elevation[candidate]);
                            }
                        }
                    }
                }
            }
            relief_by_radius[radius_index][cell] = Some(canonical_zero(maximum - minimum));
            truncated_by_radius[radius_index][cell] = boundary_distance <= radius;
        }
    }
    PlanarReliefFields {
        relief_by_radius,
        truncated_by_radius,
    }
}

fn spherical_relief_fields(
    graph: &EvaluationSurfaceGraphV0,
    elevation: &[f64],
    scored_cell: &[bool],
    config: &SurfaceHierarchyConfigV0,
    query_cells: &[u32],
) -> Result<PlanarReliefFields, LandformError> {
    let EvaluationDomainV0::Spherical { radius_km } = graph.domain else {
        return Err(LandformError::UnsupportedDomain);
    };
    let maximum_radius = *config.local_relief_radii_km.last().unwrap();
    let bucket_size = maximum_radius;
    let bucket_key = |point: DVec3| {
        (
            (point.x / bucket_size).floor() as i64,
            (point.y / bucket_size).floor() as i64,
            (point.z / bucket_size).floor() as i64,
        )
    };
    let mut center_buckets = HashMap::<(i64, i64, i64), Vec<usize>>::new();
    for (cell, &scored) in scored_cell.iter().enumerate() {
        if scored {
            center_buckets
                .entry(bucket_key(graph.cell_center_km[cell]))
                .or_default()
                .push(cell);
        }
    }

    let scored_boundaries = graph
        .boundary_segments
        .iter()
        .map(|segment| segment.endpoints_km)
        .chain((0..graph.cell_count()).flat_map(|cell| {
            edge_range(&graph.edge_offsets, cell).filter_map(move |edge| {
                let neighbor = graph.edge_neighbor[edge] as usize;
                (scored_cell[cell] && !scored_cell[neighbor])
                    .then_some(graph.edge_face_endpoints_km[edge])
            })
        }))
        .collect::<Vec<_>>();
    let mut maximum_boundary_half_width = 0.0_f64;
    let mut boundary_buckets = HashMap::<(i64, i64, i64), Vec<[DVec3; 2]>>::new();
    for endpoints in scored_boundaries {
        let midpoint_direction = endpoints[0].normalize() + endpoints[1].normalize();
        if !midpoint_direction.is_finite() || midpoint_direction.length_squared() == 0.0 {
            return Err(LandformError::InvalidSphericalGeometry);
        }
        let midpoint = radius_km * midpoint_direction.normalize();
        let half_width = 0.5 * radius_km * spherical_angle_rad(endpoints[0], endpoints[1])?;
        maximum_boundary_half_width = maximum_boundary_half_width.max(half_width);
        boundary_buckets
            .entry(bucket_key(midpoint))
            .or_default()
            .push(endpoints);
    }
    let mut relief_by_radius =
        vec![vec![None; graph.cell_count()]; config.local_relief_radii_km.len()];
    let mut truncated_by_radius =
        vec![vec![false; graph.cell_count()]; config.local_relief_radii_km.len()];
    let center_reach = (maximum_radius / bucket_size).ceil() as i64 + 1;
    let boundary_reach =
        ((maximum_radius + maximum_boundary_half_width) / bucket_size).ceil() as i64 + 1;
    for &cell_u32 in query_cells {
        let cell = cell_u32 as usize;
        debug_assert!(scored_cell[cell]);
        let center = graph.cell_center_km[cell];
        let key = bucket_key(center);
        let mut minimum = vec![f64::INFINITY; config.local_relief_radii_km.len()];
        let mut maximum = vec![f64::NEG_INFINITY; config.local_relief_radii_km.len()];
        for bx in key.0 - center_reach..=key.0 + center_reach {
            for by in key.1 - center_reach..=key.1 + center_reach {
                for bz in key.2 - center_reach..=key.2 + center_reach {
                    let Some(candidates) = center_buckets.get(&(bx, by, bz)) else {
                        continue;
                    };
                    for &candidate in candidates {
                        let distance = radius_km
                            * spherical_angle_rad(center, graph.cell_center_km[candidate])?;
                        for (radius_index, &radius) in
                            config.local_relief_radii_km.iter().enumerate()
                        {
                            if distance <= radius {
                                minimum[radius_index] =
                                    minimum[radius_index].min(elevation[candidate]);
                                maximum[radius_index] =
                                    maximum[radius_index].max(elevation[candidate]);
                            }
                        }
                    }
                }
            }
        }
        let mut boundary_distance = f64::INFINITY;
        for bx in key.0 - boundary_reach..=key.0 + boundary_reach {
            for by in key.1 - boundary_reach..=key.1 + boundary_reach {
                for bz in key.2 - boundary_reach..=key.2 + boundary_reach {
                    let Some(segments) = boundary_buckets.get(&(bx, by, bz)) else {
                        continue;
                    };
                    for &endpoints in segments {
                        boundary_distance = boundary_distance.min(
                            spherical_point_minor_arc_distance_km(center, endpoints, radius_km)?,
                        );
                    }
                }
            }
        }
        for (radius_index, &radius) in config.local_relief_radii_km.iter().enumerate() {
            if !minimum[radius_index].is_finite() || !maximum[radius_index].is_finite() {
                return Err(LandformError::NonFiniteDerived {
                    measurement: "spherical_relief",
                    cell,
                });
            }
            relief_by_radius[radius_index][cell] = Some(canonical_zero(
                maximum[radius_index] - minimum[radius_index],
            ));
            truncated_by_radius[radius_index][cell] = boundary_distance <= radius;
        }
    }
    Ok(PlanarReliefFields {
        relief_by_radius,
        truncated_by_radius,
    })
}

fn weighted_quantile(mut values: Vec<(f64, u32, f64)>, fraction: f64) -> f64 {
    values.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    let target = fraction * values.iter().map(|value| value.2).sum::<f64>();
    let mut cumulative = 0.0;
    for (value, _, area) in values {
        cumulative += area;
        if cumulative >= target {
            return canonical_zero(value);
        }
    }
    unreachable!("a nonempty weighted sample reaches its total area")
}

fn summit_cap_summaries(
    graph: &EvaluationSurfaceGraphV0,
    elevation: &[f64],
    grades: &[Option<f64>],
    peak: &PeakBranchV0,
    config: &SurfaceHierarchyConfigV0,
) -> Vec<SummitCapSummaryV0> {
    config
        .summit_cap_depths_km
        .iter()
        .map(|&depth| {
            let cap = peak
                .footprint_members
                .iter()
                .copied()
                .filter(|&cell| elevation[cell as usize] >= peak.peak_elevation_km - depth)
                .collect::<Vec<_>>();
            let cap_area = cap
                .iter()
                .map(|&cell| graph.cell_area_km2[cell as usize])
                .sum::<f64>();
            let valid_area = cap
                .iter()
                .filter(|&&cell| grades[cell as usize].is_some())
                .map(|&cell| graph.cell_area_km2[cell as usize])
                .sum::<f64>();
            let gentle_fractions = config
                .gentle_grade_thresholds
                .iter()
                .map(|&threshold| {
                    let gentle_area = cap
                        .iter()
                        .filter(|&&cell| {
                            grades[cell as usize].is_some_and(|grade| grade <= threshold)
                        })
                        .map(|&cell| graph.cell_area_km2[cell as usize])
                        .sum::<f64>();
                    GentleFractionV0 {
                        grade_threshold: threshold,
                        fraction: canonical_zero(gentle_area / cap_area),
                    }
                })
                .collect();
            SummitCapSummaryV0 {
                depth_km: depth,
                area_km2: canonical_zero(cap_area),
                fraction: canonical_zero(cap_area / peak.footprint_area_km2),
                valid_grade_fraction: canonical_zero(valid_area / cap_area),
                gentle_fractions,
                cap_merge_censored: depth >= peak.persistence_km,
            }
        })
        .collect()
}

fn build_reference_highlands(
    graph: &EvaluationSurfaceGraphV0,
    elevation: &[f64],
    scored_cell: &[bool],
    config: &SurfaceHierarchyConfigV0,
    peaks: &[PeakBranchV0],
    reference: &[u32],
) -> Result<Vec<HighlandFeatureV0>, LandformError> {
    if reference.is_empty() {
        return Ok(Vec::new());
    }
    let grades = physical_grades(graph, elevation, scored_cell, config)?;
    let measured_cells = reference
        .iter()
        .flat_map(|&peak_id| peaks[peak_id as usize].footprint_members.iter().copied())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let relief = match graph.domain {
        EvaluationDomainV0::Planar => {
            planar_relief_fields(graph, elevation, scored_cell, config, &measured_cells)
        }
        EvaluationDomainV0::Spherical { .. } => {
            spherical_relief_fields(graph, elevation, scored_cell, config, &measured_cells)?
        }
    };
    reference
        .iter()
        .map(|&peak_id| {
            let peak = &peaks[peak_id as usize];
            let local_relief = config
                .local_relief_radii_km
                .iter()
                .enumerate()
                .map(|(radius_index, &radius)| {
                    let samples = peak
                        .footprint_members
                        .iter()
                        .map(|&cell| {
                            (
                                relief.relief_by_radius[radius_index][cell as usize]
                                    .expect("active footprint cells are scored"),
                                cell,
                                graph.cell_area_km2[cell as usize],
                            )
                        })
                        .collect::<Vec<_>>();
                    let truncated_member_cells = peak
                        .footprint_members
                        .iter()
                        .copied()
                        .filter(|&cell| relief.truncated_by_radius[radius_index][cell as usize])
                        .collect();
                    LocalReliefSummaryV0 {
                        radius_km: radius,
                        area_weighted_p50_km: weighted_quantile(samples.clone(), 0.50),
                        area_weighted_p90_km: weighted_quantile(samples, 0.90),
                        truncated_member_cells,
                    }
                })
                .collect();
            let rank_deficient_grade_cells = peak
                .footprint_members
                .iter()
                .copied()
                .filter(|&cell| grades[cell as usize].is_none())
                .collect();
            let summit_caps = summit_cap_summaries(graph, elevation, &grades, peak, config);
            let measurements = match graph.domain {
                EvaluationDomainV0::Planar => {
                    let footprint_geometry = measure_planar_footprint(
                        graph,
                        &peak.footprint_members,
                        config,
                        peak_id as usize,
                    )?;
                    HighlandMeasurementsV0::Planar(Box::new(PlanarHighlandMeasurementsV0 {
                        footprint_geometry,
                        local_relief,
                        rank_deficient_grade_cells,
                        summit_caps,
                    }))
                }
                EvaluationDomainV0::Spherical { .. } => {
                    let (footprint_geometry, two_sweep_extent_km, mean_width_km) =
                        measure_spherical_footprint(
                            graph,
                            &peak.footprint_members,
                            config,
                            peak_id as usize,
                        )?;
                    HighlandMeasurementsV0::Spherical(Box::new(SphericalHighlandMeasurementsV0 {
                        footprint_geometry,
                        two_sweep_extent_km,
                        mean_width_km,
                        local_relief,
                        rank_deficient_grade_cells,
                        summit_caps,
                    }))
                }
            };
            Ok(HighlandFeatureV0 {
                peak_id,
                measurements,
            })
        })
        .collect()
}

fn canonical_hierarchy_bytes(
    graph: &EvaluationSurfaceGraphV0,
    elevation: &[f64],
    scored_cell: &[bool],
    config: &SurfaceHierarchyConfigV0,
    hierarchy: &SurfaceHierarchyV0,
) -> Result<Vec<u8>, LandformError> {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_little_endian()
        .serialize(&(
            graph,
            elevation,
            scored_cell,
            config,
            &hierarchy.schema_version,
            &hierarchy.hash_version,
            &hierarchy.peaks,
            &hierarchy.saddles,
            &hierarchy.roots,
            &hierarchy.cell_peak_owner,
            &hierarchy.populations,
            &hierarchy.reference_highlands,
        ))
        .map_err(|error| LandformError::Serialization(error.to_string()))
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(test)]
#[path = "landforms/analytic_tests.rs"]
mod analytic_tests;

#[cfg(test)]
#[path = "landforms/spherical_tests.rs"]
mod spherical_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::fibonacci_sphere_points_with_rng;
    use crate::geometry::{SphericalVoronoi, VoronoiCell};
    use crate::world::CellAdjacency;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn voronoi_cap_controls(fixture: &VoronoiCapFixture) -> LandscapeControlVolumesV0 {
        let mut cell_polygon_offsets = Vec::with_capacity(fixture.cell_polygons_km.len() + 1);
        let mut cell_polygon_vertices_km = Vec::new();
        for polygon in &fixture.cell_polygons_km {
            cell_polygon_offsets.push(cell_polygon_vertices_km.len() as u32);
            cell_polygon_vertices_km.extend(
                polygon
                    .iter()
                    .map(|point| DVec3::new(point.x, point.y, 0.0)),
            );
        }
        cell_polygon_offsets.push(cell_polygon_vertices_km.len() as u32);
        LandscapeControlVolumesV0 {
            cell_polygon_offsets,
            cell_polygon_vertices_km,
            edge_face_endpoints_km: fixture.edge_face_endpoints_km.clone(),
            boundary_face_endpoints_km: fixture.boundary_face_endpoints_km.clone(),
        }
    }

    fn chain_graph(cell_count: usize, scale: f64) -> EvaluationSurfaceGraphV0 {
        assert!(cell_count >= 2);
        let mut polygon_offsets = Vec::with_capacity(cell_count + 1);
        let mut polygon_vertices = Vec::with_capacity(4 * cell_count);
        for cell in 0..cell_count {
            polygon_offsets.push(polygon_vertices.len() as u32);
            let x0 = cell as f64 * scale;
            let x1 = x0 + scale;
            polygon_vertices.extend([
                DVec3::new(x0, 0.0, 0.0),
                DVec3::new(x1, 0.0, 0.0),
                DVec3::new(x1, scale, 0.0),
                DVec3::new(x0, scale, 0.0),
            ]);
        }
        polygon_offsets.push(polygon_vertices.len() as u32);

        let mut edge_offsets = Vec::with_capacity(cell_count + 1);
        let mut edge_neighbor = Vec::new();
        let mut edge_distance = Vec::new();
        let mut edge_width = Vec::new();
        let mut edge_endpoints = Vec::new();
        for cell in 0..cell_count {
            edge_offsets.push(edge_neighbor.len() as u32);
            if cell > 0 {
                let x = cell as f64 * scale;
                edge_neighbor.push((cell - 1) as u32);
                edge_distance.push(scale);
                edge_width.push(scale);
                edge_endpoints.push([DVec3::new(x, scale, 0.0), DVec3::new(x, 0.0, 0.0)]);
            }
            if cell + 1 < cell_count {
                let x = (cell + 1) as f64 * scale;
                edge_neighbor.push((cell + 1) as u32);
                edge_distance.push(scale);
                edge_width.push(scale);
                edge_endpoints.push([DVec3::new(x, 0.0, 0.0), DVec3::new(x, scale, 0.0)]);
            }
        }
        edge_offsets.push(edge_neighbor.len() as u32);
        let mut lookup = BTreeMap::new();
        for cell in 0..cell_count {
            for edge in edge_range(&edge_offsets, cell) {
                lookup.insert((cell, edge_neighbor[edge] as usize), edge);
            }
        }
        let edge_reciprocal = (0..edge_neighbor.len())
            .map(|edge| {
                let cell = edge_offsets
                    .partition_point(|&offset| offset as usize <= edge)
                    .saturating_sub(1);
                lookup[&(edge_neighbor[edge] as usize, cell)] as u32
            })
            .collect();

        let mut boundary_segments = Vec::new();
        for cell in 0..cell_count {
            let x0 = cell as f64 * scale;
            let x1 = x0 + scale;
            let mut segments = vec![
                [DVec3::new(x0, 0.0, 0.0), DVec3::new(x1, 0.0, 0.0)],
                [DVec3::new(x1, scale, 0.0), DVec3::new(x0, scale, 0.0)],
            ];
            if cell == 0 {
                segments.push([DVec3::new(x0, scale, 0.0), DVec3::new(x0, 0.0, 0.0)]);
            }
            if cell + 1 == cell_count {
                segments.push([DVec3::new(x1, 0.0, 0.0), DVec3::new(x1, scale, 0.0)]);
            }
            for endpoints in segments {
                boundary_segments.push(EvaluationBoundarySegmentV0 {
                    id: 0,
                    owner_cell: cell as u32,
                    endpoints_km: endpoints,
                    physical_length_km: scale,
                    projected_span_km: None,
                    condition: EvaluationBoundaryConditionV0::Closed,
                });
            }
        }
        boundary_segments.sort_by_key(|segment| {
            (
                segment.owner_cell,
                point_key(segment.endpoints_km[0]),
                point_key(segment.endpoints_km[1]),
            )
        });
        for (id, segment) in boundary_segments.iter_mut().enumerate() {
            segment.id = id as u32;
        }

        EvaluationSurfaceGraphV0 {
            domain: EvaluationDomainV0::Planar,
            cell_center_km: (0..cell_count)
                .map(|cell| DVec3::new((cell as f64 + 0.5) * scale, 0.5 * scale, 0.0))
                .collect(),
            cell_area_km2: vec![scale * scale; cell_count],
            cell_polygon_offsets: polygon_offsets,
            cell_polygon_vertices_km: polygon_vertices,
            edge_offsets,
            edge_neighbor,
            edge_reciprocal,
            edge_distance_km: edge_distance,
            edge_shared_width_km: edge_width,
            edge_face_endpoints_km: edge_endpoints,
            boundary_segments,
        }
    }

    fn square_cycle_graph(physical_cell_by_index: [usize; 4]) -> EvaluationSurfaceGraphV0 {
        let physical_origin = [
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(1.0, 1.0, 0.0),
            DVec3::new(0.0, 1.0, 0.0),
        ];
        let polygons = physical_cell_by_index.map(|physical| {
            let origin = physical_origin[physical];
            [
                origin,
                origin + DVec3::X,
                origin + DVec3::X + DVec3::Y,
                origin + DVec3::Y,
            ]
        });

        let mut edge_owners =
            BTreeMap::<((u64, u64, u64), (u64, u64, u64)), Vec<(usize, [DVec3; 2])>>::new();
        for (cell, polygon) in polygons.iter().enumerate() {
            for edge in 0..polygon.len() {
                let endpoints = [polygon[edge], polygon[(edge + 1) % polygon.len()]];
                let keys = [point_key(endpoints[0]), point_key(endpoints[1])];
                let key = if keys[0] < keys[1] {
                    (keys[0], keys[1])
                } else {
                    (keys[1], keys[0])
                };
                edge_owners.entry(key).or_default().push((cell, endpoints));
            }
        }

        let mut internal = vec![Vec::<(usize, [DVec3; 2])>::new(); 4];
        let mut boundary_segments = Vec::new();
        for owners in edge_owners.values() {
            match owners.as_slice() {
                [(cell, endpoints)] => boundary_segments.push(EvaluationBoundarySegmentV0 {
                    id: 0,
                    owner_cell: *cell as u32,
                    endpoints_km: *endpoints,
                    physical_length_km: 1.0,
                    projected_span_km: None,
                    condition: EvaluationBoundaryConditionV0::Closed,
                }),
                [(a, a_endpoints), (b, b_endpoints)] => {
                    internal[*a].push((*b, *a_endpoints));
                    internal[*b].push((*a, *b_endpoints));
                }
                _ => panic!("square fixture has a non-manifold edge"),
            }
        }
        for records in &mut internal {
            records.sort_by_key(|record| record.0);
        }

        let mut edge_offsets = Vec::with_capacity(5);
        let mut edge_neighbor = Vec::new();
        let mut edge_face_endpoints_km = Vec::new();
        for records in &internal {
            edge_offsets.push(edge_neighbor.len() as u32);
            for &(neighbor, endpoints) in records {
                edge_neighbor.push(neighbor as u32);
                edge_face_endpoints_km.push(endpoints);
            }
        }
        edge_offsets.push(edge_neighbor.len() as u32);
        let mut directed = BTreeMap::new();
        for cell in 0..4 {
            for edge in edge_range(&edge_offsets, cell) {
                directed.insert((cell, edge_neighbor[edge] as usize), edge);
            }
        }
        let edge_reciprocal = (0..edge_neighbor.len())
            .map(|edge| {
                let cell = edge_offsets
                    .partition_point(|&offset| offset as usize <= edge)
                    .saturating_sub(1);
                directed[&(edge_neighbor[edge] as usize, cell)] as u32
            })
            .collect::<Vec<_>>();

        boundary_segments.sort_by_key(|segment| {
            (
                segment.owner_cell,
                point_key(segment.endpoints_km[0]),
                point_key(segment.endpoints_km[1]),
            )
        });
        for (id, segment) in boundary_segments.iter_mut().enumerate() {
            segment.id = id as u32;
        }

        EvaluationSurfaceGraphV0 {
            domain: EvaluationDomainV0::Planar,
            cell_center_km: polygons
                .iter()
                .map(|polygon| (polygon[0] + polygon[2]) * 0.5)
                .collect(),
            cell_area_km2: vec![1.0; 4],
            cell_polygon_offsets: vec![0, 4, 8, 12, 16],
            cell_polygon_vertices_km: polygons.into_iter().flatten().collect(),
            edge_offsets,
            edge_neighbor,
            edge_reciprocal,
            edge_distance_km: vec![1.0; edge_face_endpoints_km.len()],
            edge_shared_width_km: vec![1.0; edge_face_endpoints_km.len()],
            edge_face_endpoints_km,
            boundary_segments,
        }
    }

    fn single_polygon_graph(mut polygon: Vec<DVec3>) -> EvaluationSurfaceGraphV0 {
        if planar_polygon_signed_area(&polygon) < 0.0 {
            polygon.reverse();
        }
        for point in &mut polygon {
            *point = canonical_point(*point);
        }
        rotate_polygon_to_canonical_start(&mut polygon);
        let moments = planar_polygon_raw_moments(&polygon);
        let center = DVec3::new(moments[1] / moments[0], moments[2] / moments[0], 0.0);
        let mut boundary_segments = polygon
            .iter()
            .copied()
            .zip(polygon.iter().copied().cycle().skip(1))
            .take(polygon.len())
            .map(|(a, b)| EvaluationBoundarySegmentV0 {
                id: 0,
                owner_cell: 0,
                endpoints_km: [a, b],
                physical_length_km: a.distance(b),
                projected_span_km: None,
                condition: EvaluationBoundaryConditionV0::Closed,
            })
            .collect::<Vec<_>>();
        boundary_segments.sort_by_key(|segment| {
            (
                segment.owner_cell,
                point_key(segment.endpoints_km[0]),
                point_key(segment.endpoints_km[1]),
            )
        });
        for (id, segment) in boundary_segments.iter_mut().enumerate() {
            segment.id = id as u32;
        }
        EvaluationSurfaceGraphV0 {
            domain: EvaluationDomainV0::Planar,
            cell_center_km: vec![canonical_point(center)],
            cell_area_km2: vec![moments[0]],
            cell_polygon_offsets: vec![0, polygon.len() as u32],
            cell_polygon_vertices_km: polygon,
            edge_offsets: vec![0, 0],
            edge_neighbor: Vec::new(),
            edge_reciprocal: Vec::new(),
            edge_distance_km: Vec::new(),
            edge_shared_width_km: Vec::new(),
            edge_face_endpoints_km: Vec::new(),
            boundary_segments,
        }
    }

    fn hierarchy(elevation: &[f64]) -> SurfaceHierarchyV0 {
        let graph = chain_graph(elevation.len(), 60.0);
        build_surface_hierarchy_v0(
            &graph,
            elevation,
            &vec![true; elevation.len()],
            SurfaceHierarchyConfigV0::default(),
        )
        .unwrap()
    }

    #[test]
    fn chain_fixture_has_one_saddle_and_exact_nested_footprints() {
        let result = hierarchy(&[5.0, 3.0, 4.0]);
        assert_eq!(result.peaks.len(), 2);
        assert_eq!(result.saddles.len(), 1);
        assert_eq!(result.roots, vec![0]);
        assert_eq!(result.saddles[0].elder_peak, 0);
        assert_eq!(result.saddles[0].losing_peaks, vec![1]);
        assert_eq!(result.peaks[1].persistence_km, 1.0);
        assert_eq!(result.peaks[1].footprint_members, vec![2]);
        assert_eq!(result.peaks[0].footprint_members, vec![0, 1, 2]);
        assert_eq!(result.cell_peak_owner, vec![Some(0), Some(0), Some(1)]);
        assert_eq!(result.populations.reference, vec![0, 1]);
        assert_eq!(result.reference_highlands.len(), 2);
        let HighlandMeasurementsV0::Planar(measurements) =
            &result.reference_highlands[0].measurements
        else {
            panic!("planar fixture must produce planar measurements");
        };
        assert_eq!(measurements.local_relief.len(), 3);
        assert_eq!(measurements.summit_caps.len(), 3);
        assert!(measurements
            .summit_caps
            .iter()
            .all(|cap| cap.gentle_fractions.len() == 3));
    }

    #[test]
    fn exact_flat_maximum_births_one_peak() {
        let result = hierarchy(&[5.0, 5.0, 4.0]);
        assert_eq!(result.peaks.len(), 1);
        assert!(result.saddles.is_empty());
        assert_eq!(result.peaks[0].flat_maximum_cells, vec![0, 1]);
        assert_eq!(result.peaks[0].exclusive_cells, vec![0, 1, 2]);
    }

    #[test]
    fn disconnected_equal_saddle_supports_form_one_reindex_invariant_event() {
        fn extract(physical_cell_by_index: [usize; 4]) -> (SurfaceHierarchyV0, Vec<usize>) {
            let graph = square_cycle_graph(physical_cell_by_index);
            let physical_elevation = [5.0, 3.0, 4.0, 3.0];
            let elevation = physical_cell_by_index.map(|physical| physical_elevation[physical]);
            let result = build_surface_hierarchy_v0(
                &graph,
                &elevation,
                &[true; 4],
                SurfaceHierarchyConfigV0::default(),
            )
            .unwrap();
            let mut physical_support = result.saddles[0]
                .flat_saddle_cells
                .iter()
                .map(|&cell| physical_cell_by_index[cell as usize])
                .collect::<Vec<_>>();
            physical_support.sort_unstable();
            (result, physical_support)
        }

        let (base, base_support) = extract([0, 1, 2, 3]);
        let (reindexed, reindexed_support) = extract([2, 3, 0, 1]);

        assert_eq!(base.peaks.len(), 2);
        assert_eq!(base.saddles.len(), 1);
        assert_eq!(base.roots.len(), 1);
        assert_eq!(base.saddles[0].losing_peaks.len(), 1);
        assert_eq!(base_support, vec![1, 3]);
        assert_eq!(reindexed_support, base_support);
        assert_eq!(reindexed.peaks.len(), base.peaks.len());
        assert_eq!(reindexed.saddles.len(), base.saddles.len());
        assert_eq!(reindexed.roots.len(), base.roots.len());
        assert_eq!(reindexed.saddles[0].elevation_km, 3.0);
        assert_eq!(
            reindexed.saddles[0].flat_centroid_km,
            DVec3::new(1.0, 1.0, 0.0)
        );
        assert_eq!(
            reindexed.saddles[0].flat_centroid_km,
            base.saddles[0].flat_centroid_km
        );
        let base_loser = base.saddles[0].losing_peaks[0] as usize;
        let reindexed_loser = reindexed.saddles[0].losing_peaks[0] as usize;
        assert_eq!(base.peaks[base_loser].peak_elevation_km, 4.0);
        assert_eq!(reindexed.peaks[reindexed_loser].peak_elevation_km, 4.0);
    }

    #[test]
    fn equal_elder_uses_centroid_and_records_ambiguity() {
        let result = hierarchy(&[4.0, 3.0, 4.0]);
        assert_eq!(result.saddles.len(), 1);
        assert_eq!(result.saddles[0].elder_peak, 0);
        assert!(result.saddles[0].equal_elder_ambiguous);
        assert!(result.peaks[1].equal_elder_ambiguous);
        assert_eq!(result.peaks[0].anchor_cell, 0);
    }

    #[test]
    fn closure_is_strict_and_can_leave_multiple_roots() {
        let result = hierarchy(&[2.0, 0.0, 1.0]);
        assert_eq!(result.peaks.len(), 2);
        assert_eq!(result.roots.len(), 2);
        assert!(result.saddles.is_empty());
        assert_eq!(result.cell_peak_owner[1], None);

        let joined = hierarchy(&[2.0, 0.5, 1.0]);
        assert_eq!(joined.roots.len(), 1);
        assert_eq!(joined.saddles.len(), 1);
        assert_eq!(joined.saddles[0].elevation_km, 0.5);
    }

    #[test]
    fn no_cell_above_closure_is_valid_empty_forest() {
        let result = hierarchy(&[0.0, -1.0, -0.0]);
        assert!(result.peaks.is_empty());
        assert!(result.saddles.is_empty());
        assert!(result.roots.is_empty());
        assert_eq!(result.cell_peak_owner, vec![None, None, None]);
    }

    #[test]
    fn elevation_and_closure_translation_preserve_relative_hierarchy() {
        let graph = chain_graph(3, 60.0);
        let base = build_surface_hierarchy_v0(
            &graph,
            &[5.0, 3.0, 4.0],
            &[true; 3],
            SurfaceHierarchyConfigV0::default(),
        )
        .unwrap();
        let mut translated_config = SurfaceHierarchyConfigV0::default();
        translated_config.closure_level_km = 7.0;
        let translated =
            build_surface_hierarchy_v0(&graph, &[12.0, 10.0, 11.0], &[true; 3], translated_config)
                .unwrap();
        assert_eq!(base.peaks.len(), translated.peaks.len());
        assert_eq!(base.saddles.len(), translated.saddles.len());
        assert_eq!(base.cell_peak_owner, translated.cell_peak_owner);
        assert_eq!(
            base.peaks
                .iter()
                .map(|p| p.persistence_km)
                .collect::<Vec<_>>(),
            translated
                .peaks
                .iter()
                .map(|p| p.persistence_km)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn hierarchy_and_hash_repeat_bit_identically() {
        let a = hierarchy(&[5.0, 3.0, 4.0]);
        let b = hierarchy(&[5.0, 3.0, 4.0]);
        assert_eq!(a, b);
        assert_ne!(a.derived_evidence_hash, 0);
    }

    #[test]
    fn planar_rectangle_moments_rotate_and_disk_orientation_is_ambiguous() {
        let config = SurfaceHierarchyConfigV0::default();
        let rectangle = [
            DVec3::new(-4.0, -1.0, 0.0),
            DVec3::new(4.0, -1.0, 0.0),
            DVec3::new(4.0, 1.0, 0.0),
            DVec3::new(-4.0, 1.0, 0.0),
        ];
        let graph = single_polygon_graph(rectangle.to_vec());
        graph.validate(&config).unwrap();
        let axis_aligned = measure_planar_footprint(&graph, &[0], &config, 0).unwrap();
        assert!((axis_aligned.area_km2 - 16.0).abs() < 1.0e-12);
        assert!(axis_aligned.centroid_km.length() < 1.0e-12);
        assert!((axis_aligned.covariance_km2[0] - 64.0 / 12.0).abs() < 1.0e-12);
        assert!((axis_aligned.covariance_km2[3] - 4.0 / 12.0).abs() < 1.0e-12);
        assert_eq!(axis_aligned.principal_axis, DVec3::X);
        assert!(!axis_aligned.orientation_ambiguous);

        let angle = std::f64::consts::PI / 6.0;
        let (sin, cos) = angle.sin_cos();
        let rotated_polygon = rectangle.map(|point| {
            DVec3::new(
                cos * point.x - sin * point.y,
                sin * point.x + cos * point.y,
                0.0,
            )
        });
        let rotated_graph = single_polygon_graph(rotated_polygon.to_vec());
        rotated_graph.validate(&config).unwrap();
        let rotated = measure_planar_footprint(&rotated_graph, &[0], &config, 0).unwrap();
        assert!((rotated.area_km2 - axis_aligned.area_km2).abs() < 1.0e-12);
        assert!(
            (rotated.equivalent_ellipse_length_km - axis_aligned.equivalent_ellipse_length_km)
                .abs()
                < 1.0e-12
        );
        assert!(
            (rotated.equivalent_ellipse_width_km - axis_aligned.equivalent_ellipse_width_km).abs()
                < 1.0e-12
        );
        assert!(rotated.principal_axis.dot(DVec3::new(cos, sin, 0.0)) > 1.0 - 1.0e-12);

        let translation = DVec3::new(100_000_000.0, -200_000_000.0, 0.0);
        let translated_graph =
            single_polygon_graph(rectangle.map(|point| point + translation).to_vec());
        let translated = measure_planar_footprint(&translated_graph, &[0], &config, 0).unwrap();
        assert!((translated.area_km2 - axis_aligned.area_km2).abs() < 1.0e-12);
        assert!(translated.centroid_km.distance(translation) < 1.0e-12);
        assert_eq!(translated.covariance_km2, axis_aligned.covariance_km2);
        assert_eq!(
            translated.equivalent_ellipse_length_km,
            axis_aligned.equivalent_ellipse_length_km
        );
        assert_eq!(
            translated.equivalent_ellipse_width_km,
            axis_aligned.equivalent_ellipse_width_km
        );

        let disk_polygon = (0..128)
            .map(|index| {
                let angle = std::f64::consts::TAU * index as f64 / 128.0;
                DVec3::new(5.0 * angle.cos(), 5.0 * angle.sin(), 0.0)
            })
            .collect();
        let disk_graph = single_polygon_graph(disk_polygon);
        disk_graph.validate(&config).unwrap();
        let disk = measure_planar_footprint(&disk_graph, &[0], &config, 0).unwrap();
        assert!(disk.anisotropy < 1.0e-12);
        assert!(disk.orientation_ambiguous);
        assert!(
            (disk.equivalent_ellipse_length_km - disk.equivalent_ellipse_width_km).abs() < 1.0e-12
        );
    }

    #[test]
    fn face_weighted_least_squares_reconstructs_affine_grade() {
        let graph = square_cycle_graph([0, 1, 2, 3]);
        let elevation = graph
            .cell_center_km
            .iter()
            .map(|center| 0.006 * center.x + 0.008 * center.y + 2.0)
            .collect::<Vec<_>>();
        let grades = planar_grades(
            &graph,
            &elevation,
            &[true; 4],
            &SurfaceHierarchyConfigV0::default(),
        )
        .unwrap();
        assert!(grades
            .iter()
            .all(|grade| (grade.unwrap() - 0.010).abs() < 1.0e-12));
    }

    #[test]
    fn broad_cap_has_more_area_and_gentle_surface_than_narrow_cap() {
        fn fixture_peak() -> PeakBranchV0 {
            PeakBranchV0 {
                id: 0,
                peak_elevation_km: 2.0,
                anchor_cell: 0,
                flat_centroid_km: DVec3::new(0.5, 0.5, 0.0),
                flat_maximum_cells: vec![0],
                parent_peak: None,
                key_saddle: None,
                persistence_km: 0.5,
                root_closure: true,
                equal_elder_ambiguous: false,
                exclusive_cells: vec![0, 1, 2, 3],
                footprint_members: vec![0, 1, 2, 3],
                footprint_area_km2: 4.0,
                union_boundary_edges: Vec::new(),
                physical_boundary_segments: Vec::new(),
                scored_boundary_contact: false,
            }
        }
        let graph = square_cycle_graph([0, 1, 2, 3]);
        let config = SurfaceHierarchyConfigV0::default();
        let peak = fixture_peak();
        let narrow = summit_cap_summaries(
            &graph,
            &[2.0, 1.4, 0.8, 1.4],
            &[Some(0.015); 4],
            &peak,
            &config,
        );
        let broad = summit_cap_summaries(
            &graph,
            &[2.0, 1.9, 1.8, 1.9],
            &[Some(0.0), Some(0.005), Some(0.008), Some(0.005)],
            &peak,
            &config,
        );
        for (narrow_cap, broad_cap) in narrow.iter().zip(&broad) {
            assert!(broad_cap.area_km2 > narrow_cap.area_km2);
            assert!(
                broad_cap.gentle_fractions[0].fraction > narrow_cap.gentle_fractions[0].fraction
            );
            assert!(
                broad_cap.gentle_fractions[1].fraction > narrow_cap.gentle_fractions[1].fraction
            );
            assert!(
                broad_cap.gentle_fractions[2].fraction >= narrow_cap.gentle_fractions[2].fraction
            );
        }
        assert!(!broad[0].cap_merge_censored);
        assert!(broad[1].cap_merge_censored);
        assert!(broad[2].cap_merge_censored);
    }

    #[test]
    fn regular_hex_companion_round_trips_through_general_adapter() {
        let config = SurfaceHierarchyConfigV0::default();
        let mesh = LandscapeMesh::uniform_planar_hex_with_portals(48.0, 40.0, 4.0, &[]).unwrap();
        let controls = build_regular_hex_control_volumes_v0(&mesh, &config).unwrap();
        let graph = adapt_landscape_graph_v0(&mesh, &controls, &config).unwrap();
        assert_eq!(graph.cell_count(), mesh.cell_count());
        assert_eq!(graph.edge_neighbor.len(), mesh.edge_neighbor.len());
        graph.validate(&config).unwrap();
    }

    #[test]
    fn irregular_product_voronoi_cap_round_trips_exact_control_volumes() {
        let config = SurfaceHierarchyConfigV0::default();
        let fixture = super::super::landscape::build_r1_voronoi_cap(
            super::super::landscape::VoronoiCapConfig::r1(8.0),
        )
        .unwrap();
        let graph = adapt_projected_voronoi_cap_graph_v0(&fixture, &config).unwrap();
        assert_eq!(graph.cell_count(), fixture.mesh.cell_count());
        assert_eq!(graph.edge_neighbor.len(), fixture.mesh.edge_neighbor.len());
        assert_eq!(
            graph.boundary_segments.len(),
            fixture.mesh.boundary_faces.len()
        );
        graph.validate(&config).unwrap();

        let max_nonorthogonality = (0..fixture.mesh.cell_count())
            .flat_map(|cell| {
                edge_range(&fixture.mesh.edge_offsets, cell).map(move |edge| (cell, edge))
            })
            .map(|(cell, edge)| {
                let neighbor = fixture.mesh.edge_neighbor[edge] as usize;
                let displacement =
                    fixture.mesh.cell_center_km[neighbor] - fixture.mesh.cell_center_km[cell];
                let endpoints = fixture.edge_face_endpoints_km[edge];
                let face = endpoints[1] - endpoints[0];
                displacement.normalize().dot(face.normalize()).abs()
            })
            .fold(0.0_f64, f64::max);
        assert!(max_nonorthogonality > 1.0e-4);

        let controls = voronoi_cap_controls(&fixture);
        let mut bad_area = fixture.mesh.clone();
        bad_area.cell_area_km2[0] *= 1.01;
        assert!(matches!(
            adapt_landscape_graph_v0(&bad_area, &controls, &config),
            Err(LandformError::PlanarAreaMismatch { cell: 0 })
        ));

        let mut bad_width = fixture.mesh.clone();
        bad_width.edge_face_width_km[0] *= 1.01;
        assert!(matches!(
            adapt_landscape_graph_v0(&bad_width, &controls, &config),
            Err(LandformError::OperatorGeometryMismatch { .. })
        ));

        let mut bad_controls = controls;
        bad_controls.edge_face_endpoints_km[0][0].x += 0.01;
        assert!(matches!(
            adapt_landscape_graph_v0(&fixture.mesh, &bad_controls, &config),
            Err(LandformError::OperatorGeometryMismatch { .. })
        ));

        let controls = voronoi_cap_controls(&fixture);
        let mut bad_boundary = fixture.mesh.clone();
        bad_boundary.boundary_faces[0].center_km.x += 0.01;
        assert!(matches!(
            adapt_landscape_graph_v0(&bad_boundary, &controls, &config),
            Err(LandformError::InvalidBoundarySegment { .. })
        ));
    }

    #[test]
    fn product_sphere_g0_uses_physical_faces_and_rejects_adversarial_geometry() {
        let config = SurfaceHierarchyConfigV0::default();
        let mut rng = ChaCha8Rng::seed_from_u64(0x0006_0305_f470);
        let points = fibonacci_sphere_points_with_rng(128, 0.0, &mut rng);
        let mut tessellation = Tessellation::from_points_knn_clipping(points);
        let graph = adapt_product_tessellation_graph_v0(&tessellation, &config).unwrap();
        assert!(matches!(
            graph.domain,
            EvaluationDomainV0::Spherical { radius_km }
                if radius_km == f64::from(PLANET_RADIUS_KM)
        ));
        assert!(graph.boundary_segments.is_empty());
        graph.validate(&config).unwrap();

        let mut off_sphere = graph.clone();
        off_sphere.cell_center_km[0] *= 1.000_001;
        assert!(matches!(
            off_sphere.validate(&config),
            Err(LandformError::InvalidSphericalGeometry)
        ));

        let start = graph.cell_polygon_offsets[0] as usize;
        let end = graph.cell_polygon_offsets[1] as usize;
        let reversed_polygon = graph.cell_polygon_vertices_km[start..end]
            .iter()
            .copied()
            .rev()
            .collect::<Vec<_>>();
        assert_eq!(
            spherical_cell_area_km2(
                graph.cell_center_km[0],
                &reversed_polygon,
                f64::from(PLANET_RADIUS_KM),
            ),
            Err(LandformError::InvalidSphericalGeometry)
        );
        let mut reversed = graph.clone();
        reversed.cell_polygon_vertices_km[start..end].reverse();
        rotate_polygon_to_canonical_start(&mut reversed.cell_polygon_vertices_km[start..end]);
        assert!(reversed.validate(&config).is_err());

        let n = tessellation.num_cells();
        let mut adjacency = (0..n)
            .map(|cell| tessellation.neighbors(cell).to_vec())
            .collect::<Vec<_>>();
        let mut hidden_neighbor_data = vec![0];
        let mut hidden_offsets = vec![1];
        for neighbors in &adjacency {
            hidden_neighbor_data.extend(neighbors);
            hidden_offsets.push(hidden_neighbor_data.len());
        }
        tessellation.adjacency =
            CellAdjacency::from_raw_parts(hidden_offsets, hidden_neighbor_data);
        assert!(matches!(
            adapt_product_tessellation_graph_v0(&tessellation, &config),
            Err(LandformError::MalformedCsr)
        ));
        tessellation.adjacency = CellAdjacency::from_vecs(adjacency.clone());

        let generators = tessellation.voronoi.generators.clone();
        let vertices = tessellation.voronoi.vertices.clone();
        let cell_data = (0..n)
            .map(|cell| {
                tessellation
                    .voronoi
                    .try_cell(cell)
                    .unwrap()
                    .vertex_indices
                    .to_vec()
            })
            .collect::<Vec<_>>();
        let mut hidden_indices = vec![0];
        let mut cells = Vec::with_capacity(n);
        for indices in &cell_data {
            cells.push(VoronoiCell::new(
                hidden_indices.len() as u32,
                indices.len() as u16,
            ));
            hidden_indices.extend(indices);
        }
        tessellation.voronoi = SphericalVoronoi::from_raw_parts(
            generators.clone(),
            vertices.clone(),
            cells,
            hidden_indices,
        );
        assert!(matches!(
            adapt_product_tessellation_graph_v0(&tessellation, &config),
            Err(LandformError::MalformedCsr)
        ));
        tessellation.voronoi = SphericalVoronoi::new(generators, vertices, cell_data);

        let (a, b) = (0..n)
            .flat_map(|a| ((a + 1)..n).map(move |b| (a, b)))
            .find(|&(a, b)| !adjacency[a].contains(&b))
            .unwrap();
        adjacency[a].push(b);
        adjacency[b].push(a);
        tessellation.adjacency = CellAdjacency::from_vecs(adjacency);
        assert!(matches!(
            adapt_product_tessellation_graph_v0(&tessellation, &config),
            Err(LandformError::NonPhysicalAdjacency)
        ));
    }

    #[test]
    fn planar_operator_geometry_is_secured_before_measurement() {
        let config = SurfaceHierarchyConfigV0::default();
        let mesh = LandscapeMesh::uniform_planar_hex_with_portals(48.0, 40.0, 4.0, &[]).unwrap();
        let controls = build_regular_hex_control_volumes_v0(&mesh, &config).unwrap();

        let mut bad_distance = mesh.clone();
        bad_distance.edge_distance_km[0] *= 1.01;
        assert!(matches!(
            adapt_landscape_graph_v0(&bad_distance, &controls, &config),
            Err(LandformError::OperatorGeometryMismatch { .. })
        ));

        let mut bad_normal = mesh.clone();
        bad_normal.edge_outward_tangent[0] = -bad_normal.edge_outward_tangent[0];
        assert!(matches!(
            adapt_landscape_graph_v0(&bad_normal, &controls, &config),
            Err(LandformError::OperatorGeometryMismatch { .. })
        ));

        let mut direct_graph = adapt_landscape_graph_v0(&mesh, &controls, &config).unwrap();
        let reciprocal = direct_graph.edge_reciprocal[0] as usize;
        direct_graph.edge_distance_km[0] *= 1.01;
        direct_graph.edge_distance_km[reciprocal] *= 1.01;
        assert!(matches!(
            direct_graph.validate(&config),
            Err(LandformError::OperatorGeometryMismatch { .. })
        ));
    }

    #[test]
    fn numerical_grade_failure_is_not_rank_deficiency() {
        let graph = square_cycle_graph([0, 1, 2, 3]);
        assert!(matches!(
            planar_grades(
                &graph,
                &[f64::MAX, -f64::MAX, f64::MAX, -f64::MAX],
                &[true; 4],
                &SurfaceHierarchyConfigV0::default(),
            ),
            Err(LandformError::NonFiniteDerived {
                measurement: "planar_grade",
                ..
            })
        ));
    }

    #[test]
    fn bucketed_relief_matches_exact_disk_scan_and_only_materializes_queries() {
        let config = SurfaceHierarchyConfigV0::default();
        let mesh = LandscapeMesh::uniform_planar_hex_with_portals(320.0, 280.0, 4.0, &[]).unwrap();
        let controls = build_regular_hex_control_volumes_v0(&mesh, &config).unwrap();
        let graph = adapt_landscape_graph_v0(&mesh, &controls, &config).unwrap();
        let elevation = graph
            .cell_center_km
            .iter()
            .map(|center| {
                0.3 * (center.x / 37.0).sin() + 0.2 * (center.y / 29.0).cos() - 0.001 * center.x
            })
            .collect::<Vec<_>>();
        let nearest = |target: DVec3| {
            graph
                .cell_center_km
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    a.distance_squared(target)
                        .total_cmp(&b.distance_squared(target))
                })
                .unwrap()
                .0 as u32
        };
        let queries = [nearest(DVec3::ZERO), nearest(DVec3::new(90.0, 40.0, 0.0))];
        let scored = vec![true; graph.cell_count()];
        let fields = planar_relief_fields(&graph, &elevation, &scored, &config, &queries);

        for (radius_index, &radius) in config.local_relief_radii_km.iter().enumerate() {
            for &query in &queries {
                let center = graph.cell_center_km[query as usize];
                let mut minimum = f64::INFINITY;
                let mut maximum = f64::NEG_INFINITY;
                for (candidate, &candidate_center) in graph.cell_center_km.iter().enumerate() {
                    if center.distance(candidate_center) <= radius {
                        minimum = minimum.min(elevation[candidate]);
                        maximum = maximum.max(elevation[candidate]);
                    }
                }
                assert_eq!(
                    fields.relief_by_radius[radius_index][query as usize],
                    Some(canonical_zero(maximum - minimum))
                );
                let boundary_distance = graph
                    .boundary_segments
                    .iter()
                    .map(|segment| point_segment_distance(center, segment.endpoints_km))
                    .fold(f64::INFINITY, f64::min);
                assert_eq!(
                    fields.truncated_by_radius[radius_index][query as usize],
                    boundary_distance <= radius
                );
            }
        }
        let unqueried = (0..graph.cell_count())
            .find(|cell| !queries.contains(&(*cell as u32)))
            .unwrap();
        assert!(fields
            .relief_by_radius
            .iter()
            .all(|radius| radius[unqueried].is_none()));
    }

    #[test]
    fn spherical_builder_rejects_relabelled_planar_geometry() {
        let config = SurfaceHierarchyConfigV0::default();
        let mut graph = chain_graph(2, 60.0);
        graph.domain = EvaluationDomainV0::Spherical { radius_km: 6371.0 };
        assert_eq!(
            build_surface_hierarchy_v0(&graph, &[2.0, 1.0], &[true; 2], config),
            Err(LandformError::InvalidSphericalGeometry)
        );
    }

    #[test]
    fn malformed_geometry_and_nonfinite_elevation_are_typed_errors() {
        let config = SurfaceHierarchyConfigV0::default();
        let mut graph = chain_graph(3, 60.0);
        graph.edge_reciprocal[0] = 0;
        assert!(matches!(
            graph.validate(&config),
            Err(LandformError::MissingReciprocal { .. })
        ));

        let graph = chain_graph(3, 60.0);
        assert_eq!(
            build_surface_hierarchy_v0(&graph, &[1.0, f64::NAN, 2.0], &[true; 3], config),
            Err(LandformError::NonFiniteElevation)
        );
    }
}
