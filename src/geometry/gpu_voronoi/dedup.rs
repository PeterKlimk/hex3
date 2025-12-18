//! Vertex deduplication for Voronoi cell construction.

use glam::Vec3;
use rustc_hash::FxHashMap;

use crate::geometry::VoronoiCell;

// 27-cell neighborhood offsets for grid-based proximity merge.
const NEIGHBOR_OFFSETS_27: [(i32, i32, i32); 27] = [
    (-1, -1, -1),
    (-1, -1, 0),
    (-1, -1, 1),
    (-1, 0, -1),
    (-1, 0, 0),
    (-1, 0, 1),
    (-1, 1, -1),
    (-1, 1, 0),
    (-1, 1, 1),
    (0, -1, -1),
    (0, -1, 0),
    (0, -1, 1),
    (0, 0, -1),
    (0, 0, 0),
    (0, 0, 1),
    (0, 1, -1),
    (0, 1, 0),
    (0, 1, 1),
    (1, -1, -1),
    (1, -1, 0),
    (1, -1, 1),
    (1, 0, -1),
    (1, 0, 0),
    (1, 0, 1),
    (1, 1, -1),
    (1, 1, 0),
    (1, 1, 1),
];

#[inline]
pub fn bits_for_indices(max_index: usize) -> u32 {
    let bits = usize::BITS - max_index.leading_zeros();
    bits.max(1)
}

#[inline]
fn pack_triplet_u64(triplet: [usize; 3], bits: u32) -> u64 {
    (triplet[0] as u64) | ((triplet[1] as u64) << bits) | ((triplet[2] as u64) << (2 * bits))
}

#[inline]
pub fn pack_triplet_u128(triplet: [usize; 3], bits: u32) -> u128 {
    (triplet[0] as u128) | ((triplet[1] as u128) << bits) | ((triplet[2] as u128) << (2 * bits))
}

/// Hash-based vertex deduplication.
/// Takes ordered cells with keyed vertices and produces deduplicated vertex list.
pub fn dedup_vertices_hash(
    num_points: usize,
    ordered_cells: Vec<Vec<([usize; 3], Vec3)>>,
) -> (Vec<Vec3>, Vec<VoronoiCell>, Vec<usize>) {
    dedup_vertices_hash_with_degeneracy_edges(num_points, ordered_cells, &[])
}

/// Hash-based vertex deduplication with degeneracy unification.
/// Uses triplet identity for the normal case, and union-find to unify triplets that
/// correspond to the same geometric vertex under 4+ equidistant generator degeneracy.
pub fn dedup_vertices_hash_with_degeneracy_edges(
    num_points: usize,
    ordered_cells: Vec<Vec<([usize; 3], Vec3)>>,
    degenerate_edges: &[([usize; 3], [usize; 3])],
) -> (Vec<Vec3>, Vec<VoronoiCell>, Vec<usize>) {
    dedup_vertices_hash_with_degeneracy_edges_timed(num_points, ordered_cells, degenerate_edges, false)
}

/// Same as above but with optional timing output for profiling.
pub fn dedup_vertices_hash_with_degeneracy_edges_timed(
    num_points: usize,
    ordered_cells: Vec<Vec<([usize; 3], Vec3)>>,
    degenerate_edges: &[([usize; 3], [usize; 3])],
    print_timing: bool,
) -> (Vec<Vec3>, Vec<VoronoiCell>, Vec<usize>) {
    use std::time::Instant;
    let t0 = Instant::now();

    let mut cell_starts: Vec<usize> = Vec::with_capacity(num_points + 1);
    cell_starts.push(0);
    let mut total_indices = 0usize;
    for c in &ordered_cells {
        total_indices += c.len();
        cell_starts.push(total_indices);
    }

    let expected_vertices = num_points * 2;
    let mut all_vertices: Vec<Vec3> = Vec::with_capacity(expected_vertices);

    // Per-cell triplet storage: cell_triplets[a] holds (packed_bc, vertex_idx)
    // for all triplets [a, b, c] where a < b < c
    // Each cell has ~3 entries on average, max ~21
    // Start empty - small vecs don't benefit much from pre-allocation
    let mut cell_triplets: Vec<Vec<(u64, usize)>> = vec![Vec::new(); num_points];

    let mut cells: Vec<VoronoiCell> = Vec::with_capacity(num_points);
    for generator_index in 0..num_points {
        let vertex_start = cell_starts[generator_index];
        let vertex_count = cell_starts[generator_index + 1] - vertex_start;
        cells.push(VoronoiCell::new(generator_index, vertex_start, vertex_count));
    }

    let t1 = Instant::now();

    // Pack b and c into u64: b in low 32 bits, c in high 32 bits
    #[inline(always)]
    fn pack_bc(b: usize, c: usize) -> u64 {
        (b as u64) | ((c as u64) << 32)
    }

    let mut cell_indices: Vec<usize> = vec![0usize; total_indices];
    for (cell_idx, ordered_keyed_verts) in ordered_cells.into_iter().enumerate() {
        let base = cell_starts[cell_idx];
        for (local_i, (triplet, pos)) in ordered_keyed_verts.into_iter().enumerate() {
            let a = triplet[0];
            let bc = pack_bc(triplet[1], triplet[2]);

            // Linear scan of cell a's triplets (~3 entries avg, max ~21)
            let list = &mut cell_triplets[a];
            let mut found_idx = None;
            for &(existing_bc, idx) in list.iter() {
                if existing_bc == bc {
                    found_idx = Some(idx);
                    break;
                }
            }

            let idx = match found_idx {
                Some(idx) => idx,
                None => {
                    let idx = all_vertices.len();
                    all_vertices.push(pos);
                    list.push((bc, idx));
                    idx
                }
            };
            cell_indices[base + local_i] = idx;
        }
    }

    let t2 = Instant::now();

    if degenerate_edges.is_empty() {
        // No degeneracies - skip unification pass entirely
        let (deduped_cells, deduped_indices) = deduplicate_cell_indices(&cells, &cell_indices);
        let t3 = Instant::now();
        if print_timing {
            eprintln!("  [dedup] setup: {:.1}ms, lookup: {:.1}ms, dedup_cells: {:.1}ms",
                (t1 - t0).as_secs_f64() * 1000.0,
                (t2 - t1).as_secs_f64() * 1000.0,
                (t3 - t2).as_secs_f64() * 1000.0);
        }
        return (all_vertices, deduped_cells, deduped_indices);
    }

    #[derive(Clone)]
    struct DisjointSet {
        parent: Vec<usize>,
        size: Vec<u32>,
    }

    impl DisjointSet {
        fn new(n: usize) -> Self {
            Self {
                parent: (0..n).collect(),
                size: vec![1u32; n],
            }
        }

        fn find(&mut self, x: usize) -> usize {
            let mut x = x;
            while self.parent[x] != x {
                self.parent[x] = self.parent[self.parent[x]];
                x = self.parent[x];
            }
            x
        }

        fn union(&mut self, a: usize, b: usize) -> bool {
            let mut a = self.find(a);
            let mut b = self.find(b);
            if a == b {
                return false;
            }
            if self.size[a] < self.size[b] {
                std::mem::swap(&mut a, &mut b);
            }
            self.parent[b] = a;
            self.size[a] += self.size[b];
            true
        }
    }

    // Helper to lookup vertex index from triplet using cell_triplets
    let lookup_triplet = |triplet: &[usize; 3]| -> Option<usize> {
        let a = triplet[0];
        let bc = pack_bc(triplet[1], triplet[2]);
        for &(existing_bc, idx) in &cell_triplets[a] {
            if existing_bc == bc {
                return Some(idx);
            }
        }
        None
    };

    // Turn degeneracy edges into index unions (guarded by a distance check to tolerate false positives).
    let tol = 1e-5;
    let tol_sq = tol * tol;
    let mut pairs: Vec<(usize, usize)> = Vec::with_capacity(degenerate_edges.len());
    for (t1, t2) in degenerate_edges {
        let (Some(i1), Some(i2)) = (lookup_triplet(t1), lookup_triplet(t2)) else {
            continue;
        };
        if i1 == i2 {
            continue;
        }
        let diff = all_vertices[i1] - all_vertices[i2];
        if diff.dot(diff) > tol_sq {
            continue;
        }
        let (a, b) = if i1 < i2 { (i1, i2) } else { (i2, i1) };
        pairs.push((a, b));
    }

    if pairs.is_empty() {
        let (deduped_cells, deduped_indices) = deduplicate_cell_indices(&cells, &cell_indices);
        let t3 = Instant::now();
        if print_timing {
            eprintln!("  [dedup] setup: {:.1}ms, lookup: {:.1}ms, dsu_prep: 0ms, dedup_cells: {:.1}ms (no unions)",
                (t1 - t0).as_secs_f64() * 1000.0,
                (t2 - t1).as_secs_f64() * 1000.0,
                (t3 - t2).as_secs_f64() * 1000.0);
        }
        return (all_vertices, deduped_cells, deduped_indices);
    }

    pairs.sort_unstable();
    pairs.dedup();

    let t3 = Instant::now();

    let mut dsu = DisjointSet::new(all_vertices.len());
    let mut unions = 0usize;
    for (a, b) in pairs {
        unions += dsu.union(a, b) as usize;
    }

    let t4 = Instant::now();

    if unions > 0 {
        // Remap cell indices to DSU roots (no compaction - leaves unused vertices in buffer)
        for idx in cell_indices.iter_mut() {
            *idx = dsu.find(*idx);
        }
    }

    let t5 = Instant::now();

    // Deduplicate within each cell
    let (deduped_cells, deduped_indices) = deduplicate_cell_indices(&cells, &cell_indices);

    let t6 = Instant::now();

    if print_timing {
        eprintln!("  [dedup] setup: {:.1}ms, lookup: {:.1}ms, dsu_prep: {:.1}ms, dsu_union: {:.1}ms, remap: {:.1}ms, dedup_cells: {:.1}ms ({} unions)",
            (t1 - t0).as_secs_f64() * 1000.0,
            (t2 - t1).as_secs_f64() * 1000.0,
            (t3 - t2).as_secs_f64() * 1000.0,
            (t4 - t3).as_secs_f64() * 1000.0,
            (t5 - t4).as_secs_f64() * 1000.0,
            (t6 - t5).as_secs_f64() * 1000.0,
            unions);
    }

    (all_vertices, deduped_cells, deduped_indices)
}

/// Remove duplicate vertex indices within each cell.
fn deduplicate_cell_indices(
    cells: &[VoronoiCell],
    cell_indices: &[usize],
) -> (Vec<VoronoiCell>, Vec<usize>) {
    let mut new_cells: Vec<VoronoiCell> = Vec::with_capacity(cells.len());
    let mut new_indices: Vec<usize> = Vec::with_capacity(cell_indices.len());

    // Reuse a single buffer to avoid per-cell allocations at large scales.
    // Pre-allocate for typical max cell size (~8 vertices)
    let mut seen: Vec<usize> = Vec::with_capacity(8);

    for cell in cells {
        let start = cell.vertex_start();
        let end = cell.vertex_start() + cell.vertex_count();
        let old_indices = &cell_indices[start..end];

        let new_start = new_indices.len();
        seen.clear();
        for &idx in old_indices {
            if !seen.contains(&idx) {
                seen.push(idx);
                new_indices.push(idx);
            }
        }
        let new_count = new_indices.len() - new_start;

        new_cells.push(VoronoiCell::new(
            cell.generator_index,
            new_start,
            new_count,
        ));
    }

    (new_cells, new_indices)
}

/// Merge vertices at the same position (within tolerance).
pub fn merge_coincident_vertices(vertices: &[Vec3], tolerance: f32) -> (Vec<Vec3>, Vec<usize>) {
    if vertices.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let cell_size = tolerance * 2.0;
    let inv_cell_size = 1.0 / cell_size;

    // Pack (i32, i32, i32) into u64: 21 bits each, offset by 1M to handle negatives
    const OFFSET: i32 = 1_000_000;
    const MASK: u64 = (1 << 21) - 1;

    #[inline]
    fn pack_cell(x: i32, y: i32, z: i32) -> u64 {
        let xu = (x + OFFSET) as u64 & MASK;
        let yu = (y + OFFSET) as u64 & MASK;
        let zu = (z + OFFSET) as u64 & MASK;
        xu | (yu << 21) | (zu << 42)
    }

    #[inline]
    fn hash_pos(p: Vec3, inv_cell_size: f32) -> (i32, i32, i32) {
        (
            (p.x * inv_cell_size).floor() as i32,
            (p.y * inv_cell_size).floor() as i32,
            (p.z * inv_cell_size).floor() as i32,
        )
    }

    // Pre-size hash map - expect ~1 vertex per cell on average for unit sphere
    let expected_cells = vertices.len();
    let mut grid: FxHashMap<u64, Vec<usize>> =
        FxHashMap::with_capacity_and_hasher(expected_cells, Default::default());

    let mut merged: Vec<Vec3> = Vec::with_capacity(vertices.len());
    let mut old_to_new: Vec<usize> = vec![0; vertices.len()];

    let tol_sq = tolerance * tolerance;

    for (old_idx, &pos) in vertices.iter().enumerate() {
        let (cx, cy, cz) = hash_pos(pos, inv_cell_size);

        let mut found_match: Option<usize> = None;
        'outer: for &(dx, dy, dz) in &NEIGHBOR_OFFSETS_27 {
            let neighbor_key = pack_cell(cx + dx, cy + dy, cz + dz);
            if let Some(indices) = grid.get(&neighbor_key) {
                for &existing_idx in indices {
                    let diff = merged[existing_idx] - pos;
                    if diff.dot(diff) < tol_sq {
                        found_match = Some(existing_idx);
                        break 'outer;
                    }
                }
            }
        }

        let new_idx = match found_match {
            Some(idx) => idx,
            None => {
                let idx = merged.len();
                merged.push(pos);
                let key = pack_cell(cx, cy, cz);
                grid.entry(key).or_insert_with(|| Vec::with_capacity(2)).push(idx);
                idx
            }
        };

        old_to_new[old_idx] = new_idx;
    }

    (merged, old_to_new)
}

// Degeneracy unification is handled via union-find in `dedup_vertices_hash_with_degeneracy_edges`.
