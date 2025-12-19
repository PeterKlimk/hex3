//! Vertex deduplication for Voronoi cell construction.

use glam::Vec3;
use rustc_hash::FxHashMap;

use crate::geometry::VoronoiCell;
use super::FlatCellsData;

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
pub fn pack_triplet_u128(triplet: [usize; 3], bits: u32) -> u128 {
    (triplet[0] as u128) | ((triplet[1] as u128) << bits) | ((triplet[2] as u128) << (2 * bits))
}

/// Hash-based vertex deduplication for flat chunk data.
/// Iterates through chunks without flattening, avoiding extra copies.
pub fn dedup_vertices_hash_flat(
    flat_data: FlatCellsData,
    print_timing: bool,
) -> (Vec<Vec3>, Vec<VoronoiCell>, Vec<usize>) {
    use std::time::Instant;
    let t0 = Instant::now();

    let num_points = flat_data.num_cells();
    let total_indices: usize = flat_data.chunks.iter()
        .map(|c| c.counts.iter().map(|&count| count as usize).sum::<usize>())
        .sum();
    debug_assert_eq!(
        total_indices,
        flat_data.chunks.iter().map(|c| c.vertices.len()).sum::<usize>(),
        "flat counts do not match vertex storage"
    );

    let expected_vertices = num_points * 2;
    let mut all_vertices: Vec<Vec3> = Vec::with_capacity(expected_vertices);

    // Node pool for triplet lookup (replaces Vec<Vec<(u64, usize)>>)
    const NIL: u32 = u32::MAX;

    #[repr(C)]
    struct TripletNode {
        bc: u64,
        idx: u32,
        next: u32,
    }

    let mut heads: Vec<u32> = vec![NIL; num_points];
    let mut nodes: Vec<TripletNode> = Vec::with_capacity(expected_vertices);

    let t1 = Instant::now();

    #[inline(always)]
    fn pack_bc(b: u32, c: u32) -> u64 {
        (b as u64) | ((c as u64) << 32)
    }

    let mut cell_indices: Vec<usize> = vec![0usize; total_indices];
    let mut cells: Vec<VoronoiCell> = Vec::with_capacity(num_points);
    let mut cell_idx = 0usize;
    let mut write_idx = 0usize;

    // Iterate through chunks in order
    for chunk in &flat_data.chunks {
        let mut chunk_vert_idx = 0usize;
        for &count in &chunk.counts {
            let count = count as usize;
            let base = write_idx;

            for local_i in 0..count {
                let (triplet, pos) = chunk.vertices[chunk_vert_idx + local_i];
                let a = triplet[0] as usize;
                let bc = pack_bc(triplet[1], triplet[2]);

                // Linear scan through linked list for cell `a`
                let mut node_id = heads[a];
                let mut found_idx: Option<u32> = None;
                while node_id != NIL {
                    let node = &nodes[node_id as usize];
                    if node.bc == bc {
                        found_idx = Some(node.idx);
                        break;
                    }
                    node_id = node.next;
                }

                let idx = match found_idx {
                    Some(idx) => idx as usize,
                    None => {
                        let idx = all_vertices.len();
                        all_vertices.push(pos);
                        let new_id = nodes.len() as u32;
                        nodes.push(TripletNode { bc, idx: idx as u32, next: heads[a] });
                        heads[a] = new_id;
                        idx
                    }
                };
                cell_indices[base + local_i] = idx;
            }

            chunk_vert_idx += count;
            cells.push(VoronoiCell::new(cell_idx, base, count));
            cell_idx += 1;
            write_idx += count;
        }
    }
    debug_assert_eq!(cell_idx, num_points);
    debug_assert_eq!(write_idx, total_indices);

    let t2 = Instant::now();

    let degenerate_edges = &flat_data.degenerate_edges;
    if degenerate_edges.is_empty() {
        let (deduped_cells, deduped_indices) = deduplicate_cell_indices(&cells, &cell_indices);
        let t3 = Instant::now();
        if print_timing {
            eprintln!("  [dedup-flat] setup: {:.1}ms, lookup: {:.1}ms, dedup_cells: {:.1}ms",
                (t1 - t0).as_secs_f64() * 1000.0,
                (t2 - t1).as_secs_f64() * 1000.0,
                (t3 - t2).as_secs_f64() * 1000.0);
        }
        return (all_vertices, deduped_cells, deduped_indices);
    }

    // Handle degeneracy unification (same logic as original)
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

    let lookup_triplet = |triplet: &[u32; 3]| -> Option<usize> {
        let a = triplet[0] as usize;
        let bc = pack_bc(triplet[1], triplet[2]);
        let mut node_id = heads[a];
        while node_id != NIL {
            let node = &nodes[node_id as usize];
            if node.bc == bc {
                return Some(node.idx as usize);
            }
            node_id = node.next;
        }
        None
    };

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
            eprintln!("  [dedup-flat] setup: {:.1}ms, lookup: {:.1}ms, dedup_cells: {:.1}ms (no unions)",
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
        for idx in cell_indices.iter_mut() {
            *idx = dsu.find(*idx);
        }
    }

    let t5 = Instant::now();

    let (deduped_cells, deduped_indices) = deduplicate_cell_indices(&cells, &cell_indices);

    let t6 = Instant::now();

    if print_timing {
        eprintln!("  [dedup-flat] setup: {:.1}ms, lookup: {:.1}ms, dsu_prep: {:.1}ms, dsu_union: {:.1}ms, remap: {:.1}ms, dedup_cells: {:.1}ms ({} unions)",
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
