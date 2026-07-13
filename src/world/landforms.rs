//! Arm-neutral physical surface graph and peak-saddle hierarchy.
//!
//! This is a partial planar structural G0/S0 slice of the packet preregistered
//! in `docs/research/landform-object-packet-g0s0-2026-07-14.md`. It implements
//! the common graph types, the explicit planar adapter and the exact-level
//! peak-saddle split forest. It is not the complete packet: spherical product
//! adaptation and the frozen morphology/measurement outputs remain absent.
//! Inputs are limited to physical geometry, elevation, a scored mask and the
//! frozen extractor configuration.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;

use bincode::Options;
use glam::DVec3;
use serde::{Deserialize, Serialize};

use super::landscape::{BoundaryFaceCondition, LandscapeMesh, LandscapeMeshError, OutletPortalId};

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
    InvalidPolygon { cell: usize },
    InvalidPolygonWinding { cell: usize },
    NonCanonicalGeometry,
    PlanarAreaMismatch { cell: usize },
    SelfEdge { cell: usize },
    DuplicateNeighbor { cell: usize, neighbor: usize },
    MissingReciprocal { cell: usize, neighbor: usize },
    NonUniqueReciprocal { cell: usize, neighbor: usize },
    ReciprocalGeometryMismatch { cell: usize, neighbor: usize },
    FaceNotBackedByPolygon { cell: usize, neighbor: usize },
    MissingPolygonEdgeBacking { cell: usize, edge: usize },
    OverlappingPolygonEdgeBacking { cell: usize, edge: usize },
    InvalidBoundarySegment { segment: usize },
    DuplicateBoundarySegment,
    MissingControlVolumeGeometry,
    NonRegularHexGeometry,
    NonPhysicalAdjacency,
    InvalidSphericalGeometry,
    SphereAreaClosure,
    Overflow,
    NegativePersistence,
    MultipleExclusiveOwners { cell: usize },
    CyclicParentage,
    FootprintAreaMismatch { peak: usize },
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
    for cell in 0..n {
        let start = controls.cell_polygon_offsets[cell] as usize;
        let end = controls.cell_polygon_offsets[cell + 1] as usize;
        let mut polygon: Vec<DVec3> = controls.cell_polygon_vertices_km[start..end]
            .iter()
            .copied()
            .map(canonical_point)
            .collect();
        if polygon.len() < 3
            || polygon.iter().any(|p| !p.is_finite() || p.z != 0.0)
            || planar_polygon_signed_area(&polygon) <= 0.0
        {
            return Err(LandformError::InvalidPolygon { cell });
        }
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
            let endpoints = controls.edge_face_endpoints_km[source_edge].map(canonical_point);
            records.push(EdgeRecord {
                neighbor: mesh.edge_neighbor[source_edge],
                distance: f64::from(mesh.edge_distance_km[source_edge]),
                width: endpoints[0].distance(endpoints[1]),
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
        .map(|(face, endpoints)| {
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
            (
                face.cell,
                endpoints.map(canonical_point),
                endpoints[0].distance(endpoints[1]),
                Some([
                    canonical_zero(face.projected_span_start_km),
                    canonical_zero(face.projected_span_end_km),
                ]),
                condition,
            )
        })
        .collect();
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
        cell_area_km2: (0..n)
            .map(|cell| {
                let start = cell_polygon_offsets[cell] as usize;
                let end = cell_polygon_offsets[cell + 1] as usize;
                canonical_zero(planar_polygon_signed_area(
                    &cell_polygon_vertices_km[start..end],
                ))
            })
            .collect(),
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

    let tolerance = config.endpoint_match_abs_km;
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
            if (endpoint_width - graph.edge_shared_width_km[edge]).abs() > tolerance {
                return Err(LandformError::ReciprocalGeometryMismatch { cell, neighbor });
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
            || segment
                .projected_span_km
                .is_some_and(|span| span.into_iter().any(|value| !has_canonical_zero(value)))
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
    Ok(())
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
    Ok(radius_km * a.cross(b).length().atan2(a.dot(b)))
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
/// Structural peak-saddle output for the current partial planar slice.
///
/// This is not yet the complete G0/S0 landform object packet: morphology,
/// measurement/classification records and spherical product adaptation are not
/// represented by this type.
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

    let mut group_start = 0usize;
    while group_start < order.len() {
        let bits = elevation[order[group_start]].to_bits();
        let mut group_end = group_start + 1;
        while group_end < order.len() && elevation[order[group_end]].to_bits() == bits {
            group_end += 1;
        }
        let level_cells = &order[group_start..group_end];
        let mut in_level = vec![false; n];
        for &cell in level_cells {
            in_level[cell] = true;
        }
        let mut visited = vec![false; n];
        let mut components = Vec::<Vec<usize>>::new();
        for &start in level_cells {
            if visited[start] {
                continue;
            }
            visited[start] = true;
            let mut queue = VecDeque::from([start]);
            let mut component = Vec::new();
            while let Some(cell) = queue.pop_front() {
                component.push(cell);
                for edge in edge_range(&graph.edge_offsets, cell) {
                    let neighbor = graph.edge_neighbor[edge] as usize;
                    if in_level[neighbor] && !visited[neighbor] {
                        visited[neighbor] = true;
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
    let mut hierarchy = SurfaceHierarchyV0 {
        schema_version: config.schema_version.to_owned(),
        hash_version: config.hash_version.to_owned(),
        peaks,
        saddles,
        roots,
        cell_peak_owner,
        populations,
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
mod tests {
    use super::*;

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
