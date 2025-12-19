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

    // Tolerance for merging close vertices detected via orphan edge analysis.
    let tol = 5e-4;

    // DSU for merging vertices
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

    // Initial dedup without any DSU merging
    let (mut deduped_cells, mut deduped_indices) = deduplicate_cell_indices(&cells, &cell_indices);

    let t3 = Instant::now();

    // Check for orphan edges and fix them by merging close vertices
    let orphan_pairs = find_orphan_vertex_pairs(
        &deduped_cells,
        &deduped_indices,
        &all_vertices,
        tol,
    );

    let t4 = Instant::now();

    let orphan_unions = if !orphan_pairs.is_empty() {
        let mut dsu = DisjointSet::new(all_vertices.len());

        // Merge orphan vertex pairs
        let mut unions = 0usize;
        for (a, b) in orphan_pairs {
            unions += dsu.union(a, b) as usize;
        }

        if unions > 0 {
            // Remap cell indices
            for idx in cell_indices.iter_mut() {
                *idx = dsu.find(*idx);
            }
            // Re-deduplicate
            let (new_cells, new_indices) = deduplicate_cell_indices(&cells, &cell_indices);
            deduped_cells = new_cells;
            deduped_indices = new_indices;
        }
        unions
    } else {
        0
    };

    let t5 = Instant::now();

    if print_timing {
        eprintln!("  [dedup-flat] setup: {:.1}ms, lookup: {:.1}ms, dedup_cells: {:.1}ms, orphan_check: {:.1}ms, orphan_fix: {:.1}ms ({} unions)",
            (t1 - t0).as_secs_f64() * 1000.0,
            (t2 - t1).as_secs_f64() * 1000.0,
            (t3 - t2).as_secs_f64() * 1000.0,
            (t4 - t3).as_secs_f64() * 1000.0,
            (t5 - t4).as_secs_f64() * 1000.0,
            orphan_unions);
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

/// Find orphan edges using vertex-indexed neighbor tracking.
/// Each vertex should have exactly 3 neighbors, each appearing exactly twice
/// (once per adjacent cell). Returns pairs of vertices that need merging.
pub fn find_orphan_vertex_pairs(
    cells: &[VoronoiCell],
    cell_indices: &[usize],
    vertices: &[Vec3],
    tolerance: f32,
) -> Vec<(usize, usize)> {
    let num_vertices = vertices.len();
    if num_vertices == 0 {
        return Vec::new();
    }

    const EMPTY: u32 = u32::MAX;
    const MAX_NEIGHBORS: usize = 6; // Allow some overflow to detect issues

    // Each vertex has up to 3 neighbors normally, but duplicates can cause more.
    let mut neighbors: Vec<[u32; MAX_NEIGHBORS]> = vec![[EMPTY; MAX_NEIGHBORS]; num_vertices];
    let mut counts: Vec<[u8; MAX_NEIGHBORS]> = vec![[0; MAX_NEIGHBORS]; num_vertices];
    let mut overflow_vertices: Vec<usize> = Vec::new();

    // Record an edge: a connects to b
    let mut record_edge = |a: usize, b: usize, overflow: &mut Vec<usize>| {
        let b_u32 = b as u32;
        for i in 0..MAX_NEIGHBORS {
            if neighbors[a][i] == b_u32 {
                counts[a][i] += 1;
                return;
            }
            if neighbors[a][i] == EMPTY {
                neighbors[a][i] = b_u32;
                counts[a][i] = 1;
                // Track if we're using more than 3 neighbors (indicates topology issue)
                if i >= 3 && !overflow.contains(&a) {
                    overflow.push(a);
                }
                return;
            }
        }
        // Severe overflow - more than MAX_NEIGHBORS
        if !overflow.contains(&a) {
            overflow.push(a);
        }
    };

    // Process all cell edges
    for cell in cells {
        let start = cell.vertex_start();
        let count = cell.vertex_count();
        if count < 3 {
            continue;
        }
        let indices = &cell_indices[start..start + count];
        for i in 0..count {
            let a = indices[i];
            let b = indices[(i + 1) % count];
            if a != b {
                record_edge(a, b, &mut overflow_vertices);
                record_edge(b, a, &mut overflow_vertices);
            }
        }
    }

    // Find orphan edges (count == 1) and collect the orphan vertices
    let mut orphan_vertices: Vec<usize> = Vec::new();
    for v in 0..num_vertices {
        for i in 0..MAX_NEIGHBORS {
            if neighbors[v][i] != EMPTY && counts[v][i] == 1 {
                orphan_vertices.push(v);
                break;
            }
        }
    }

    // Also include overflow vertices as problematic
    for v in overflow_vertices {
        if !orphan_vertices.contains(&v) {
            orphan_vertices.push(v);
        }
    }

    if orphan_vertices.is_empty() {
        return Vec::new();
    }

    // For each problematic vertex, find spatially close vertices to merge
    let tol_sq = tolerance * tolerance;
    let mut merge_pairs: Vec<(usize, usize)> = Vec::new();

    for &ov in &orphan_vertices {
        let pos = vertices[ov];
        for &other in &orphan_vertices {
            if other <= ov {
                continue; // Only check each pair once
            }
            let diff = vertices[other] - pos;
            if diff.dot(diff) < tol_sq {
                merge_pairs.push((ov, other));
            }
        }
    }

    merge_pairs
}
