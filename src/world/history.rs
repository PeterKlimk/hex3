//! Tectonic boundary episodes and explicit geological-time provenance.
//!
//! T1a uses the present connected boundary topology and constant Euler velocities.
//! Episode duration is a kinematic residence-time bound: one chain length divided by
//! its edge-weighted relative speed, capped by the world's history lookback. This is a
//! derived first clock (small/fast contacts are young), not a full plate reconstruction.

use std::collections::{HashMap, HashSet, VecDeque};
use std::time::Instant;

use super::boundary::{collect_plate_boundaries, BoundaryKind, PlateBoundaryEdge};
use super::{
    Crust, CrustType, Dynamics, OrogenModel, Plates, Tessellation, PLANET_RADIUS_KM,
    TECTONIC_CARRIER_CELLS, TECTONIC_CARRIER_STEP_MYR,
};
use crate::geometry::ConvexHull;
use glam::Quat;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HistoryModel {
    StationaryTopologyConstantVelocity,
    SeedVoronoiBackrotation,
    MaterialRasterBackrotation,
    FixedCarrierBackrotation,
}

/// Runtime carrier resolution for experimental scorecards. Product/default
/// generation retains the constants-backed default; exposing this avoids
/// recompiling three binaries to run a resolution-convergence audit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TectonicCarrierConfig {
    pub cells: usize,
    pub step_myr: f32,
}

impl Default for TectonicCarrierConfig {
    fn default() -> Self {
        Self {
            cells: TECTONIC_CARRIER_CELLS,
            step_myr: TECTONIC_CARRIER_STEP_MYR,
        }
    }
}

/// One reconstructed state on the immutable tectonic carrier. `occupancy`
/// is the pre-fill material ledger: zero entries are raster gaps and entries
/// above one are overlaps. `plate_owner`/`crust_owner` are the deterministic
/// surface state after resolving both, suitable for topology reconstruction.
#[derive(Clone, Debug, PartialEq)]
pub struct CarrierSnapshot {
    pub lookback_myr: f32,
    pub plate_owner: Vec<u16>,
    pub crust_owner: Vec<CrustType>,
    /// Material parcel exposed at each filled carrier cell. Gap cells inherit
    /// the nearest occupied surface parcel; overlap losers remain in the
    /// explicit overlap ledger and retain their own state off-surface.
    pub surface_parcel: Vec<u16>,
    /// Landing cell of every material parcel before overlap/gap resolution.
    pub parcel_cells: Vec<u16>,
    pub occupancy: Vec<u16>,
    pub gap_cell_indices: Vec<u32>,
    /// All material landings in multiply occupied cells, including the
    /// selected surface parcel. A future underthrust solver therefore retains
    /// the identities and crust types hidden below the surface owner.
    pub overlap_landings: Vec<CarrierParcelLanding>,
    pub gap_cells: usize,
    pub overlap_excess: usize,
    pub max_occupancy: u16,
    pub adjacent_pairs: Vec<(u16, u16)>,
    pub convergent_pairs: usize,
    pub divergent_pairs: usize,
    pub transform_pairs: usize,
    pub topology_changes_from_previous: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CarrierParcelLanding {
    pub cell: u32,
    pub parcel: u16,
    pub plate: u16,
    pub crust: CrustType,
}

/// Persistent output of the experimental moving-domain reconstruction.
/// Geometry is constructed once and never rebuilt during replay.
#[derive(Clone, Debug)]
pub struct CarrierReplay {
    pub num_cells: usize,
    pub mean_spacing_km: f32,
    pub step_myr: f32,
    pub snapshots: Vec<CarrierSnapshot>,
    pub build_seconds: f32,
    pub(crate) mesh: CarrierMesh,
}

#[derive(Clone, Debug)]
pub(crate) struct CarrierMesh {
    pub centers: Vec<glam::Vec3>,
    pub areas: Vec<f32>,
    pub neighbors: Vec<Vec<u16>>,
    pub edges: Vec<CarrierMeshEdge>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct CarrierMeshEdge {
    pub a: usize,
    pub b: usize,
    pub face_length: f32,
    pub conductance: f32,
    pub normal_a_to_b: glam::Vec3,
}

#[derive(Clone, Debug)]
pub struct BoundaryEpisode {
    pub id: usize,
    pub plate_a: usize,
    pub plate_b: usize,
    pub kind: BoundaryKind,
    pub subducting_plate: Option<usize>,
    pub edge_count: usize,
    pub length_km: f32,
    pub mean_convergence_km_per_myr: f32,
    pub mean_shear_km_per_myr: f32,
    pub mean_relative_speed_km_per_myr: f32,
    pub duration_myr: f32,
    pub integrated_normal_displacement_km: f32,
    pub integrated_shear_displacement_km: f32,
    pub model: HistoryModel,
}

#[derive(Clone, Debug)]
pub struct TectonicHistory {
    pub lookback_myr: f32,
    pub step_myr: f32,
    pub episodes: Vec<BoundaryEpisode>,
    pub carrier_replay: Option<CarrierReplay>,
    edge_episode_id: HashMap<(usize, usize), usize>,
}

impl TectonicHistory {
    pub fn compute(
        seed: u64,
        tessellation: &Tessellation,
        plates: &Plates,
        crust: &Crust,
        dynamics: &Dynamics,
        orogen_model: OrogenModel,
        carrier_config: TectonicCarrierConfig,
    ) -> Self {
        let boundaries = collect_plate_boundaries(tessellation, plates, crust, dynamics);
        let mut history = Self::from_boundaries(
            &boundaries,
            dynamics.clock.lookback_myr,
            dynamics.clock.step_myr,
        );
        let requested_pairs: HashSet<_> = history
            .episodes
            .iter()
            .map(|episode| (episode.plate_a, episode.plate_b))
            .collect();
        let (contact_ages, carrier_replay, model) = match orogen_model {
            OrogenModel::HistoryMaterial => (
                replay_material_pair_contact_ages(tessellation, plates, dynamics, &requested_pairs),
                None,
                HistoryModel::MaterialRasterBackrotation,
            ),
            OrogenModel::HistoryCarrierThinSheet | OrogenModel::HistoryCarrierEvolved => {
                let (ages, replay) = replay_fixed_carrier(
                    seed,
                    tessellation,
                    plates,
                    crust,
                    dynamics,
                    &requested_pairs,
                    carrier_config.cells,
                    carrier_config.step_myr,
                );
                (ages, Some(replay), HistoryModel::FixedCarrierBackrotation)
            }
            _ => (
                replay_pair_contact_ages(plates, dynamics, &requested_pairs),
                None,
                HistoryModel::SeedVoronoiBackrotation,
            ),
        };
        for episode in &mut history.episodes {
            if let Some(&contact_age) = contact_ages.get(&(episode.plate_a, episode.plate_b)) {
                episode.duration_myr = episode.duration_myr.min(contact_age);
                episode.integrated_normal_displacement_km =
                    episode.mean_convergence_km_per_myr * episode.duration_myr;
                episode.integrated_shear_displacement_km =
                    episode.mean_shear_km_per_myr * episode.duration_myr;
                episode.model = model;
            }
        }
        if let Some(replay) = &carrier_replay {
            history.step_myr = replay.step_myr;
        }
        history.carrier_replay = carrier_replay;
        history
    }

    fn from_boundaries(boundaries: &[PlateBoundaryEdge], lookback_myr: f32, step_myr: f32) -> Self {
        let mut groups: HashMap<(usize, usize, BoundaryKind), Vec<usize>> = HashMap::new();
        for (i, edge) in boundaries.iter().enumerate() {
            let pair = canonical_pair(edge.plate_a, edge.plate_b);
            groups
                .entry((pair.0, pair.1, edge.kind))
                .or_default()
                .push(i);
        }

        let mut keys: Vec<_> = groups.keys().copied().collect();
        keys.sort_by_key(|&(a, b, kind)| (a, b, kind_order(kind)));

        let mut episodes = Vec::new();
        let mut edge_episode_id = HashMap::new();
        for key in keys {
            let edge_ids = &groups[&key];
            let mut by_cell: HashMap<usize, Vec<usize>> = HashMap::new();
            for &edge_id in edge_ids {
                let edge = &boundaries[edge_id];
                by_cell.entry(edge.cell_a).or_default().push(edge_id);
                by_cell.entry(edge.cell_b).or_default().push(edge_id);
            }

            let mut seen = std::collections::HashSet::new();
            let mut starts = edge_ids.clone();
            starts.sort_unstable();
            for start in starts {
                if !seen.insert(start) {
                    continue;
                }
                let mut queue = VecDeque::from([start]);
                let mut component = Vec::new();
                while let Some(edge_id) = queue.pop_front() {
                    component.push(edge_id);
                    let edge = &boundaries[edge_id];
                    for cell in [edge.cell_a, edge.cell_b] {
                        for &next in &by_cell[&cell] {
                            if seen.insert(next) {
                                queue.push_back(next);
                            }
                        }
                    }
                }
                let episode_id = episodes.len();
                episodes.push(measure_episode(
                    episode_id,
                    key.0,
                    key.1,
                    key.2,
                    &component,
                    boundaries,
                    lookback_myr,
                ));
                for &edge_id in &component {
                    let edge = &boundaries[edge_id];
                    edge_episode_id.insert(canonical_pair(edge.cell_a, edge.cell_b), episode_id);
                }
            }
        }

        Self {
            lookback_myr,
            step_myr,
            episodes,
            carrier_replay: None,
            edge_episode_id,
        }
    }

    pub fn active_duration_myr_for_pair(&self, plate_a: usize, plate_b: usize) -> f32 {
        let pair = canonical_pair(plate_a, plate_b);
        self.episodes
            .iter()
            .filter(|e| (e.plate_a, e.plate_b) == pair)
            .map(|e| e.duration_myr)
            .fold(0.0, f32::max)
    }

    pub fn episode_for_edge(&self, cell_a: usize, cell_b: usize) -> Option<&BoundaryEpisode> {
        let id = self.edge_episode_id.get(&canonical_pair(cell_a, cell_b))?;
        self.episodes.get(*id)
    }
}

fn measure_episode(
    id: usize,
    plate_a: usize,
    plate_b: usize,
    kind: BoundaryKind,
    component: &[usize],
    boundaries: &[PlateBoundaryEdge],
    lookback_myr: f32,
) -> BoundaryEpisode {
    let mut length_km = 0.0f32;
    let mut convergence = 0.0f32;
    let mut shear = 0.0f32;
    let mut relative = 0.0f32;
    let mut subduction_votes: HashMap<usize, f32> = HashMap::new();
    for &edge_id in component {
        let edge = &boundaries[edge_id];
        let length = edge.edge_length * PLANET_RADIUS_KM;
        length_km += length;
        convergence += edge.convergence_km_per_myr() * length;
        shear += edge.shear_km_per_myr() * length;
        relative += edge.relative_speed_km_per_myr() * length;
        if let Some(polarity) = edge.subduction {
            let plate = match polarity {
                super::SubductionPolarity::ASubducts => edge.plate_a,
                super::SubductionPolarity::BSubducts => edge.plate_b,
            };
            *subduction_votes.entry(plate).or_default() += length;
        }
    }
    let inv_length = 1.0 / length_km.max(1e-9);
    let mean_convergence = convergence * inv_length;
    let mean_shear = shear * inv_length;
    let mean_relative = relative * inv_length;
    let duration_myr = kinematic_residence_time_myr(length_km, mean_relative, lookback_myr);
    let subducting_plate = subduction_votes
        .into_iter()
        .max_by(|a, b| a.1.total_cmp(&b.1).then_with(|| b.0.cmp(&a.0)))
        .map(|(plate, _)| plate);

    BoundaryEpisode {
        id,
        plate_a,
        plate_b,
        kind,
        subducting_plate,
        edge_count: component.len(),
        length_km,
        mean_convergence_km_per_myr: mean_convergence,
        mean_shear_km_per_myr: mean_shear,
        mean_relative_speed_km_per_myr: mean_relative,
        duration_myr,
        integrated_normal_displacement_km: mean_convergence * duration_myr,
        integrated_shear_displacement_km: mean_shear * duration_myr,
        model: HistoryModel::StationaryTopologyConstantVelocity,
    }
}

fn canonical_pair(a: usize, b: usize) -> (usize, usize) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

fn kind_order(kind: BoundaryKind) -> u8 {
    match kind {
        BoundaryKind::Convergent => 0,
        BoundaryKind::Divergent => 1,
        BoundaryKind::Transform => 2,
    }
}

fn kinematic_residence_time_myr(length_km: f32, speed_km_per_myr: f32, cap_myr: f32) -> f32 {
    if length_km <= 0.0 || cap_myr <= 0.0 {
        return 0.0;
    }
    if speed_km_per_myr <= 1e-6 {
        return cap_myr;
    }
    (length_km / speed_km_per_myr).clamp(0.0, cap_myr)
}

fn replay_pair_contact_ages(
    plates: &Plates,
    dynamics: &Dynamics,
    requested_pairs: &HashSet<(usize, usize)>,
) -> HashMap<(usize, usize), f32> {
    let clock = dynamics.clock;
    let mut ages: HashMap<(usize, usize), f32> = requested_pairs
        .iter()
        .map(|&pair| (pair, clock.lookback_myr))
        .collect();
    let mut unresolved = requested_pairs.clone();
    let steps = (clock.lookback_myr / clock.step_myr).ceil() as usize;
    for step in 1..=steps {
        if unresolved.is_empty() {
            break;
        }
        let t = (step as f32 * clock.step_myr).min(clock.lookback_myr);
        let positions: Vec<_> = plates
            .seed_positions
            .iter()
            .enumerate()
            .map(|(plate, &position)| {
                let pole = dynamics.euler_pole(plate);
                let angle = -pole.angular_velocity_rad_per_myr() * t;
                Quat::from_axis_angle(pole.axis, angle) * position
            })
            .collect();
        let adjacent = seed_adjacency(&positions);
        let ended: Vec<_> = unresolved
            .iter()
            .copied()
            .filter(|pair| !adjacent.contains(pair))
            .collect();
        for pair in ended {
            // Midpoint of the last-present and first-absent samples.
            ages.insert(pair, (t - 0.5 * clock.step_myr).max(0.5 * clock.step_myr));
            unresolved.remove(&pair);
        }
    }
    ages
}

fn seed_adjacency(points: &[glam::Vec3]) -> HashSet<(usize, usize)> {
    let hull = ConvexHull::compute(points);
    let mut pairs = HashSet::new();
    for facet in hull.facets {
        let [a, b, c] = facet.indices;
        pairs.insert(canonical_pair(a, b));
        pairs.insert(canonical_pair(b, c));
        pairs.insert(canonical_pair(c, a));
    }
    pairs
}

#[derive(Default)]
struct CarrierPairMotion {
    length: f32,
    normal: f32,
    shear: f32,
    positive_length: f32,
    negative_length: f32,
}

/// Build one immutable low-resolution carrier, transfer present material to it,
/// and back-rotate those parcels through fixed-time snapshots. This is a domain
/// reconstruction, not a surface-process integration: it supplies changing
/// topology/contact clocks to the existing history solver while retaining every
/// raster overlap and gap for audit.
pub(crate) fn replay_fixed_carrier(
    seed: u64,
    source: &Tessellation,
    plates: &Plates,
    crust: &Crust,
    dynamics: &Dynamics,
    requested_pairs: &HashSet<(usize, usize)>,
    carrier_cells: usize,
    step_myr: f32,
) -> (HashMap<(usize, usize), f32>, CarrierReplay) {
    let started = Instant::now();
    let mut rng = ChaCha8Rng::seed_from_u64(seed.wrapping_add(0x7465_6374_6f6e_6963));
    let carrier = Tessellation::generate(carrier_cells, 0, &mut rng);
    let mean_spacing_km =
        PLANET_RADIUS_KM * (4.0 * std::f32::consts::PI / carrier_cells as f32).sqrt();
    let carrier_mesh = build_carrier_mesh(&carrier);

    // Each carrier site represents one present-day material parcel. Coherent
    // walking makes the one-off 8k -> source transfer linear in practice.
    let mut source_hint = 0usize;
    let mut parcel_plate = Vec::with_capacity(carrier_cells);
    let mut parcel_crust = Vec::with_capacity(carrier_cells);
    for parcel in 0..carrier_cells {
        source_hint = nearest_cell_walk(source, carrier.cell_center(parcel), source_hint);
        parcel_plate.push(plates.cell_plate[source_hint] as usize);
        parcel_crust.push(crust.crust_type(source_hint));
    }

    let steps = (dynamics.clock.lookback_myr / step_myr).ceil() as usize;
    let mut snapshots = Vec::with_capacity(steps + 1);
    let mut landing_hint: Vec<usize> = (0..carrier_cells).collect();
    let mut previous_pairs = HashSet::new();
    for step in 0..=steps {
        let t = (step as f32 * step_myr).min(dynamics.clock.lookback_myr);
        let mut occupancy = vec![0u16; carrier_cells];
        let mut owner = vec![usize::MAX; carrier_cells];
        let mut owner_crust = vec![CrustType::Oceanic; carrier_cells];
        let mut winner_dot = vec![f32::NEG_INFINITY; carrier_cells];
        let mut winner_parcel = vec![usize::MAX; carrier_cells];
        let mut landed_cell = vec![0usize; carrier_cells];

        for parcel in 0..carrier_cells {
            let plate = parcel_plate[parcel];
            let pole = dynamics.euler_pole(plate);
            let position =
                Quat::from_axis_angle(pole.axis, -pole.angular_velocity_rad_per_myr() * t)
                    * carrier.cell_center(parcel);
            let cell = nearest_cell_walk(&carrier, position, landing_hint[parcel]);
            landing_hint[parcel] = cell;
            landed_cell[parcel] = cell;
            occupancy[cell] = occupancy[cell].saturating_add(1);
            let dot = position.dot(carrier.cell_center(cell));
            if dot > winner_dot[cell] || (dot == winner_dot[cell] && parcel < winner_parcel[cell]) {
                winner_dot[cell] = dot;
                winner_parcel[cell] = parcel;
                owner[cell] = plate;
                owner_crust[cell] = parcel_crust[parcel];
            }
        }

        let gap_cell_indices: Vec<_> = occupancy
            .iter()
            .enumerate()
            .filter_map(|(cell, &count)| (count == 0).then_some(cell as u32))
            .collect();
        let gap_cells = gap_cell_indices.len();
        let overlap_excess = occupancy
            .iter()
            .map(|&count| count.saturating_sub(1) as usize)
            .sum();
        let max_occupancy = occupancy.iter().copied().max().unwrap_or(0);
        let overlap_landings = landed_cell
            .iter()
            .enumerate()
            .filter(|&(_, &cell)| occupancy[cell] > 1)
            .map(|(parcel, &cell)| CarrierParcelLanding {
                cell: cell as u32,
                parcel: parcel as u16,
                plate: parcel_plate[parcel] as u16,
                crust: parcel_crust[parcel],
            })
            .collect();

        // Stable multi-source graph fill supplies a surface owner while the
        // untouched occupancy vector continues to record missing material.
        let mut queue = VecDeque::new();
        for (cell, &plate) in owner.iter().enumerate() {
            if plate != usize::MAX {
                queue.push_back(cell);
            }
        }
        while let Some(cell) = queue.pop_front() {
            for &next in carrier.neighbors(cell) {
                if owner[next] == usize::MAX {
                    owner[next] = owner[cell];
                    owner_crust[next] = owner_crust[cell];
                    winner_parcel[next] = winner_parcel[cell];
                    queue.push_back(next);
                }
            }
        }

        let (pairs, kinds) = carrier_pair_topology(&carrier, &owner, dynamics);
        let topology_changes_from_previous = if step == 0 {
            0
        } else {
            pairs.symmetric_difference(&previous_pairs).count()
        };
        let mut adjacent_pairs: Vec<_> = pairs.iter().map(|&(a, b)| (a as u16, b as u16)).collect();
        adjacent_pairs.sort_unstable();
        let count_kind = |kind| kinds.values().filter(|&&value| value == kind).count();
        snapshots.push(CarrierSnapshot {
            lookback_myr: t,
            plate_owner: owner.iter().map(|&plate| plate as u16).collect(),
            crust_owner: owner_crust,
            surface_parcel: winner_parcel.iter().map(|&parcel| parcel as u16).collect(),
            parcel_cells: landed_cell.iter().map(|&cell| cell as u16).collect(),
            occupancy,
            gap_cell_indices,
            overlap_landings,
            gap_cells,
            overlap_excess,
            max_occupancy,
            adjacent_pairs,
            convergent_pairs: count_kind(BoundaryKind::Convergent),
            divergent_pairs: count_kind(BoundaryKind::Divergent),
            transform_pairs: count_kind(BoundaryKind::Transform),
            topology_changes_from_previous,
        });
        previous_pairs = pairs;
    }

    let mut ages: HashMap<_, _> = requested_pairs
        .iter()
        .map(|&pair| (pair, dynamics.clock.lookback_myr))
        .collect();
    let present_pairs: HashSet<_> = snapshots[0]
        .adjacent_pairs
        .iter()
        .map(|&(a, b)| (a as usize, b as usize))
        .collect();
    let mut unresolved = requested_pairs.clone();
    for pair in requested_pairs.difference(&present_pairs) {
        ages.insert(*pair, 0.5 * step_myr);
        unresolved.remove(pair);
    }
    for snapshot in snapshots.iter().skip(1) {
        if unresolved.is_empty() {
            break;
        }
        let pairs: HashSet<_> = snapshot
            .adjacent_pairs
            .iter()
            .map(|&(a, b)| (a as usize, b as usize))
            .collect();
        let ended: Vec<_> = unresolved
            .iter()
            .copied()
            .filter(|pair| !pairs.contains(pair))
            .collect();
        for pair in ended {
            ages.insert(
                pair,
                (snapshot.lookback_myr - 0.5 * step_myr).max(0.5 * step_myr),
            );
            unresolved.remove(&pair);
        }
    }

    let replay = CarrierReplay {
        num_cells: carrier_cells,
        mean_spacing_km,
        step_myr,
        snapshots,
        build_seconds: started.elapsed().as_secs_f32(),
        mesh: carrier_mesh,
    };
    (ages, replay)
}

fn build_carrier_mesh(tessellation: &Tessellation) -> CarrierMesh {
    let n = tessellation.num_cells();
    let centers = (0..n).map(|cell| tessellation.cell_center(cell)).collect();
    let areas = tessellation.cell_areas();
    let neighbors = (0..n)
        .map(|cell| {
            tessellation
                .neighbors(cell)
                .iter()
                .map(|&next| next as u16)
                .collect()
        })
        .collect();
    let mut edges = Vec::with_capacity(tessellation.adjacency.total_neighbor_entries() / 2);
    for a in 0..n {
        for &b in tessellation.neighbors(a) {
            if b <= a {
                continue;
            }
            let center_a = tessellation.cell_center(a);
            let center_b = tessellation.cell_center(b);
            let center_distance = (center_b - center_a).length();
            let face_length = tessellation.shared_edge_length(a, b);
            if center_distance <= 1e-8 || face_length <= 0.0 {
                continue;
            }
            let midpoint = (center_a + center_b).normalize_or_zero();
            let chord = center_b - center_a;
            let normal = (chord - midpoint * midpoint.dot(chord)).normalize_or_zero();
            if normal != glam::Vec3::ZERO {
                edges.push(CarrierMeshEdge {
                    a,
                    b,
                    face_length,
                    conductance: face_length / center_distance,
                    normal_a_to_b: normal,
                });
            }
        }
    }
    CarrierMesh {
        centers,
        areas,
        neighbors,
        edges,
    }
}

fn carrier_pair_topology(
    carrier: &Tessellation,
    owner: &[usize],
    dynamics: &Dynamics,
) -> (
    HashSet<(usize, usize)>,
    HashMap<(usize, usize), BoundaryKind>,
) {
    use super::constants::{
        PLATE_PAIR_MIN_ACTIVE_LENGTH, PLATE_PAIR_MIN_BOUNDARY_LENGTH, TRANSFORM_NORMAL_THRESHOLD,
        TRANSFORM_RATIO,
    };

    let mut stats: HashMap<(usize, usize), CarrierPairMotion> = HashMap::new();
    for cell in 0..carrier.num_cells() {
        let a = owner[cell];
        let pos_a = carrier.cell_center(cell);
        for &next in carrier.neighbors(cell) {
            let b = owner[next];
            if next <= cell || a == b {
                continue;
            }
            let pos_b = carrier.cell_center(next);
            let point = (pos_a + pos_b).normalize();
            let chord = pos_b - pos_a;
            let normal = (chord - point * chord.dot(point)).normalize_or_zero();
            let along = point.cross(normal).normalize_or_zero();
            let relative = dynamics.euler_pole(a).velocity_at(point)
                - dynamics.euler_pole(b).velocity_at(point);
            let convergence = relative.dot(normal);
            let shear = relative.dot(along).abs();
            let length = carrier.shared_edge_length(cell, next);
            let entry = stats.entry(canonical_pair(a, b)).or_default();
            entry.length += length;
            entry.normal += convergence * length;
            entry.shear += shear * length;
            if convergence >= TRANSFORM_NORMAL_THRESHOLD {
                entry.positive_length += length;
            } else if convergence <= -TRANSFORM_NORMAL_THRESHOLD {
                entry.negative_length += length;
            }
        }
    }
    let pairs = stats.keys().copied().collect();
    let kinds = stats
        .into_iter()
        .map(|(pair, stats)| {
            let mean_normal = stats.normal / stats.length.max(1e-12);
            let mean_shear = stats.shear / stats.length.max(1e-12);
            let kind = if stats.length < PLATE_PAIR_MIN_BOUNDARY_LENGTH {
                BoundaryKind::Transform
            } else if mean_shear > mean_normal.abs() * TRANSFORM_RATIO
                && mean_normal.abs() < TRANSFORM_NORMAL_THRESHOLD
            {
                BoundaryKind::Transform
            } else if mean_normal > TRANSFORM_NORMAL_THRESHOLD
                && stats.positive_length >= PLATE_PAIR_MIN_ACTIVE_LENGTH
            {
                BoundaryKind::Convergent
            } else if mean_normal < -TRANSFORM_NORMAL_THRESHOLD
                && stats.negative_length >= PLATE_PAIR_MIN_ACTIVE_LENGTH
            {
                BoundaryKind::Divergent
            } else {
                BoundaryKind::Transform
            };
            (pair, kind)
        })
        .collect();
    (pairs, kinds)
}

/// Back-rotate every present-day plate cell as a material parcel, rasterize the
/// parcels onto the fixed tectonic mesh, and derive plate-pair adjacency from
/// the reconstructed domains. Multiple parcels in one cell are resolved by
/// nearest center; gaps are filled by deterministic graph propagation. This
/// preserves the generated noisy plate domains instead of replacing them with
/// seed Voronoi cells.
fn replay_material_pair_contact_ages(
    tessellation: &Tessellation,
    plates: &Plates,
    dynamics: &Dynamics,
    requested_pairs: &HashSet<(usize, usize)>,
) -> HashMap<(usize, usize), f32> {
    let clock = dynamics.clock;
    let n = tessellation.num_cells();
    let mut ages: HashMap<_, _> = requested_pairs
        .iter()
        .map(|&pair| (pair, clock.lookback_myr))
        .collect();
    let mut unresolved = requested_pairs.clone();
    let mut parcel_cells: Vec<usize> = (0..n).collect();
    let steps = (clock.lookback_myr / clock.step_myr).ceil() as usize;

    for step in 1..=steps {
        if unresolved.is_empty() {
            break;
        }
        let t = (step as f32 * clock.step_myr).min(clock.lookback_myr);
        let mut owner = vec![usize::MAX; n];
        let mut owner_dot = vec![f32::NEG_INFINITY; n];
        for parcel in 0..n {
            let plate = plates.cell_plate[parcel] as usize;
            let pole = dynamics.euler_pole(plate);
            let position =
                Quat::from_axis_angle(pole.axis, -pole.angular_velocity_rad_per_myr() * t)
                    * tessellation.cell_center(parcel);
            let cell = nearest_cell_walk(tessellation, position, parcel_cells[parcel]);
            parcel_cells[parcel] = cell;
            let dot = position.dot(tessellation.cell_center(cell));
            if dot > owner_dot[cell] || (dot == owner_dot[cell] && plate < owner[cell]) {
                owner_dot[cell] = dot;
                owner[cell] = plate;
            }
        }

        // Resolve raster gaps without inventing new topology from the original
        // seeds: nearest occupied material in graph distance wins, with stable
        // cell-order tie breaking from the queue initialization.
        let mut queue = VecDeque::new();
        for (cell, &plate) in owner.iter().enumerate() {
            if plate != usize::MAX {
                queue.push_back(cell);
            }
        }
        while let Some(cell) = queue.pop_front() {
            let plate = owner[cell];
            for &next in tessellation.neighbors(cell) {
                if owner[next] == usize::MAX {
                    owner[next] = plate;
                    queue.push_back(next);
                }
            }
        }

        let mut adjacent = HashSet::new();
        for cell in 0..n {
            for &next in tessellation.neighbors(cell) {
                if owner[cell] != owner[next] {
                    adjacent.insert(canonical_pair(owner[cell], owner[next]));
                }
            }
        }
        let ended: Vec<_> = unresolved
            .iter()
            .copied()
            .filter(|pair| !adjacent.contains(pair))
            .collect();
        for pair in ended {
            ages.insert(pair, (t - 0.5 * clock.step_myr).max(0.5 * clock.step_myr));
            unresolved.remove(&pair);
        }
    }
    ages
}

fn nearest_cell_walk(tessellation: &Tessellation, position: glam::Vec3, start: usize) -> usize {
    let mut cell = start;
    loop {
        let current_dot = position.dot(tessellation.cell_center(cell));
        let mut best = cell;
        let mut best_dot = current_dot;
        for &next in tessellation.neighbors(cell) {
            let dot = position.dot(tessellation.cell_center(next));
            if dot > best_dot {
                best = next;
                best_dot = dot;
            }
        }
        if best == cell {
            return cell;
        }
        cell = best;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn residence_time_is_kinematic_and_capped() {
        assert_eq!(kinematic_residence_time_myr(500.0, 50.0, 100.0), 10.0);
        assert_eq!(kinematic_residence_time_myr(10_000.0, 50.0, 100.0), 100.0);
        assert_eq!(kinematic_residence_time_myr(500.0, 0.0, 100.0), 100.0);
    }

    #[test]
    fn integrated_displacement_uses_physical_time() {
        let duration = kinematic_residence_time_myr(600.0, 60.0, 100.0);
        assert_eq!(duration, 10.0);
        assert_eq!(30.0 * duration, 300.0);
    }

    #[test]
    fn carrier_config_default_preserves_product_constants() {
        let config = TectonicCarrierConfig::default();
        assert_eq!(config.cells, TECTONIC_CARRIER_CELLS);
        assert_eq!(config.step_myr, TECTONIC_CARRIER_STEP_MYR);
    }

    fn carrier_fixture() -> (Tessellation, Plates, Crust, Dynamics) {
        let mut mesh_rng = ChaCha8Rng::seed_from_u64(91);
        let tessellation = Tessellation::generate(512, 0, &mut mesh_rng);
        let mut plate_rng = ChaCha8Rng::seed_from_u64(92);
        let plates = Plates::generate(&tessellation, 6, &mut plate_rng);
        let mut crust_rng = ChaCha8Rng::seed_from_u64(93);
        let crust = Crust::generate(&tessellation, 3, 0.3, &mut crust_rng);
        let mut dynamics_rng = ChaCha8Rng::seed_from_u64(94);
        let mut dynamics = Dynamics::generate(&plates, &mut dynamics_rng);
        dynamics.clock.lookback_myr = 8.0;
        (tessellation, plates, crust, dynamics)
    }

    #[test]
    fn carrier_replay_is_deterministic_and_conserves_parcel_count() {
        let (tessellation, plates, crust, dynamics) = carrier_fixture();
        let requested = HashSet::new();
        let (_, a) = replay_fixed_carrier(
            12345,
            &tessellation,
            &plates,
            &crust,
            &dynamics,
            &requested,
            256,
            2.0,
        );
        let (_, b) = replay_fixed_carrier(
            12345,
            &tessellation,
            &plates,
            &crust,
            &dynamics,
            &requested,
            256,
            2.0,
        );
        assert_eq!(a.snapshots, b.snapshots);
        for snapshot in &a.snapshots {
            assert_eq!(
                snapshot
                    .occupancy
                    .iter()
                    .map(|&n| n as usize)
                    .sum::<usize>(),
                256
            );
            assert_eq!(snapshot.gap_cells, snapshot.overlap_excess);
            assert_eq!(snapshot.gap_cells, snapshot.gap_cell_indices.len());
            assert_eq!(
                snapshot.overlap_landings.len(),
                snapshot
                    .occupancy
                    .iter()
                    .filter(|&&count| count > 1)
                    .map(|&count| count as usize)
                    .sum::<usize>()
            );
        }
        assert_eq!(a.snapshots[0].gap_cells, 0);
        assert_eq!(a.snapshots[0].overlap_excess, 0);
    }

    #[test]
    fn carrier_states_are_invariant_to_time_subdivision() {
        let (tessellation, plates, crust, dynamics) = carrier_fixture();
        let requested = HashSet::new();
        let (_, fine) = replay_fixed_carrier(
            777,
            &tessellation,
            &plates,
            &crust,
            &dynamics,
            &requested,
            256,
            2.0,
        );
        let (_, coarse) = replay_fixed_carrier(
            777,
            &tessellation,
            &plates,
            &crust,
            &dynamics,
            &requested,
            256,
            4.0,
        );
        for (coarse_snapshot, fine_snapshot) in coarse
            .snapshots
            .iter()
            .zip(fine.snapshots.iter().step_by(2))
        {
            assert_eq!(coarse_snapshot.lookback_myr, fine_snapshot.lookback_myr);
            assert_eq!(coarse_snapshot.plate_owner, fine_snapshot.plate_owner);
            assert_eq!(coarse_snapshot.crust_owner, fine_snapshot.crust_owner);
            assert_eq!(coarse_snapshot.occupancy, fine_snapshot.occupancy);
            assert_eq!(coarse_snapshot.adjacent_pairs, fine_snapshot.adjacent_pairs);
        }
    }
}
