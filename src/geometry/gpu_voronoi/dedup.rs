//! Vertex deduplication for Voronoi cell construction.

use glam::Vec3;

use crate::geometry::VoronoiCell;
use super::{FlatCellsData, VERTEX_WELD_FRACTION, mean_generator_spacing_chord};
use rustc_hash::FxHashMap;

#[inline]
pub fn bits_for_indices(max_index: usize) -> u32 {
    let bits = usize::BITS - max_index.leading_zeros();
    bits.max(1)
}

#[inline]
pub fn pack_triplet_u128(triplet: [usize; 3], bits: u32) -> u128 {
    (triplet[0] as u128) | ((triplet[1] as u128) << bits) | ((triplet[2] as u128) << (2 * bits))
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

/// Heuristic weld distance based on mean generator spacing.
#[inline]
fn default_weld_distance(num_points: usize) -> f32 {
    mean_generator_spacing_chord(num_points) * VERTEX_WELD_FRACTION
}

/// Parallel sort-based vertex welding.
/// Uses sorting + binary search instead of hash map for better cache locality
/// when cells are sparse (nearly 1:1 with vertices).
fn weld_vertices_by_distance(
    vertices: &mut [Vec3],
    cell_indices: &mut [usize],
    weld_distance: f32,
    print_timing: bool,
) -> usize {
    use rayon::prelude::*;
    use std::time::Instant;

    if weld_distance <= 0.0 || vertices.len() < 2 {
        return 0;
    }

    let t0 = Instant::now();

    let n = vertices.len();
    let weld_distance_sq = weld_distance * weld_distance;
    let inv_cell_size = 1.0 / weld_distance;

    #[inline]
    fn grid_cell(p: Vec3, inv_cell_size: f32) -> (i32, i32, i32) {
        (
            (p.x * inv_cell_size).floor() as i32,
            (p.y * inv_cell_size).floor() as i32,
            (p.z * inv_cell_size).floor() as i32,
        )
    }

    // Sort-based approach: compute grid cells, sort by cell, then scan for pairs
    // This avoids hash map overhead when cells are sparse (nearly 1:1 with vertices)

    // Phase 1: Compute grid cell for each vertex (parallel)
    let mut sorted_indices: Vec<(u64, u32)> = (0..n as u32)
        .into_par_iter()
        .map(|i| {
            let p = vertices[i as usize];
            let (cx, cy, cz) = grid_cell(p, inv_cell_size);
            // Pack cell coords into u64 for sorting (offset to handle negatives)
            let key = ((cx as i64 + i32::MAX as i64) as u64) << 42
                    | ((cy as i64 + i32::MAX as i64) as u64) << 21
                    | ((cz as i64 + i32::MAX as i64) as u64);
            (key, i)
        })
        .collect();

    let t1 = Instant::now();

    // Phase 2: Sort by grid cell
    sorted_indices.par_sort_unstable_by_key(|&(key, _)| key);

    let t2 = Instant::now();

    // Phase 3: Find pairs by scanning sorted list
    // Vertices in the same cell are adjacent after sorting
    // For cross-cell pairs, we need to check neighbors - use a sliding window approach

    // First, find cell boundaries
    let mut cell_starts: Vec<usize> = Vec::with_capacity(n / 4);
    cell_starts.push(0);
    for i in 1..n {
        if sorted_indices[i].0 != sorted_indices[i - 1].0 {
            cell_starts.push(i);
        }
    }
    cell_starts.push(n); // sentinel
    let num_cells = cell_starts.len() - 1;

    let t3 = Instant::now();

    // For each cell, check pairs within cell and with neighboring cells
    // Use parallel iteration over cells
    let pairs: Vec<(usize, usize)> = (0..num_cells)
        .into_par_iter()
        .flat_map(|cell_idx| {
            let start = cell_starts[cell_idx];
            let end = cell_starts[cell_idx + 1];
            let mut local_pairs = Vec::new();

            // Get the cell key to find neighbors
            let cell_key = sorted_indices[start].0;
            let cx = ((cell_key >> 42) as i64 - i32::MAX as i64) as i32;
            let cy = (((cell_key >> 21) & 0x1FFFFF) as i64 - i32::MAX as i64) as i32;
            let cz = ((cell_key & 0x1FFFFF) as i64 - i32::MAX as i64) as i32;

            // Pairs within this cell
            for i in start..end {
                let vi = sorted_indices[i].1 as usize;
                let p_i = vertices[vi];
                for j in (i + 1)..end {
                    let vj = sorted_indices[j].1 as usize;
                    if (vertices[vj] - p_i).length_squared() <= weld_distance_sq {
                        local_pairs.push((vi, vj));
                    }
                }
            }

            // Check neighboring cells (13 of 26 neighbors to avoid duplicate pairs)
            // Only check cells with "larger" keys
            for dx in 0i32..=1 {
                for dy in -1i32..=1 {
                    for dz in -1i32..=1 {
                        if dx == 0 && dy < 0 {
                            continue;
                        }
                        if dx == 0 && dy == 0 && dz <= 0 {
                            continue;
                        }
                        let ncx = cx + dx;
                        let ncy = cy + dy;
                        let ncz = cz + dz;
                        let neighbor_key = ((ncx as i64 + i32::MAX as i64) as u64) << 42
                                         | ((ncy as i64 + i32::MAX as i64) as u64) << 21
                                         | ((ncz as i64 + i32::MAX as i64) as u64);

                        // Binary search for neighbor cell
                        if let Ok(pos) = sorted_indices.binary_search_by_key(&neighbor_key, |&(k, _)| k) {
                            // Find cell boundaries
                            let mut nstart = pos;
                            while nstart > 0 && sorted_indices[nstart - 1].0 == neighbor_key {
                                nstart -= 1;
                            }
                            let mut nend = pos + 1;
                            while nend < n && sorted_indices[nend].0 == neighbor_key {
                                nend += 1;
                            }

                            // Check all pairs between cells
                            for i in start..end {
                                let vi = sorted_indices[i].1 as usize;
                                let p_i = vertices[vi];
                                for j in nstart..nend {
                                    let vj = sorted_indices[j].1 as usize;
                                    if (vertices[vj] - p_i).length_squared() <= weld_distance_sq {
                                        local_pairs.push((vi, vj));
                                    }
                                }
                            }
                        }
                    }
                }
            }

            local_pairs
        })
        .collect();

    let t4 = Instant::now();

    if pairs.is_empty() {
        if print_timing {
            eprintln!(
                "    [weld] compute_keys: {:.1}ms, sort: {:.1}ms, find_bounds: {:.1}ms, find_pairs: {:.1}ms (0 pairs, {} cells)",
                (t1 - t0).as_secs_f64() * 1000.0,
                (t2 - t1).as_secs_f64() * 1000.0,
                (t3 - t2).as_secs_f64() * 1000.0,
                (t4 - t3).as_secs_f64() * 1000.0,
                num_cells,
            );
        }
        return 0;
    }

    let num_pairs = pairs.len();

    // Phase 4: Sequential union-find (cheap now that we have the pairs)
    let mut dsu = DisjointSet::new(n);
    let mut unions = 0usize;
    for (i, j) in pairs {
        unions += dsu.union(i, j) as usize;
    }

    let t5 = Instant::now();

    if unions == 0 {
        return 0;
    }

    // Average welded vertices to keep results stable and on-sphere.
    let mut sums = vec![Vec3::ZERO; n];
    let mut counts = vec![0u32; n];
    for i in 0..n {
        let root = dsu.find(i);
        sums[root] += vertices[i];
        counts[root] += 1;
    }
    for i in 0..n {
        if counts[i] == 0 {
            continue;
        }
        let avg = sums[i] / counts[i] as f32;
        let len_sq = avg.length_squared();
        if len_sq > 0.0 {
            vertices[i] = avg / len_sq.sqrt();
        }
    }

    let t6 = Instant::now();

    // Precompute all roots (sequential, but DSU find is nearly O(1))
    let roots: Vec<usize> = (0..n).map(|i| dsu.find(i)).collect();

    // Update cell indices in parallel
    cell_indices.par_iter_mut().for_each(|idx| {
        *idx = roots[*idx];
    });

    let t7 = Instant::now();

    if print_timing {
        eprintln!(
            "    [weld] compute_keys: {:.1}ms, sort: {:.1}ms, find_bounds: {:.1}ms, find_pairs: {:.1}ms, union: {:.1}ms, avg+update: {:.1}ms",
            (t1 - t0).as_secs_f64() * 1000.0,
            (t2 - t1).as_secs_f64() * 1000.0,
            (t3 - t2).as_secs_f64() * 1000.0,
            (t4 - t3).as_secs_f64() * 1000.0,
            (t5 - t4).as_secs_f64() * 1000.0,
            (t7 - t5).as_secs_f64() * 1000.0,
        );
        eprintln!(
            "    [weld] n={}, cells={}, pairs={}, unions={}",
            n, num_cells, num_pairs, unions,
        );
    }

    unions
}

/// Hash-based vertex deduplication for flat chunk data.
/// Performs a global vertex weld after triplet-based deduplication to merge
/// near-duplicate vertices.
pub fn dedup_vertices_hash_flat(
    flat_data: FlatCellsData,
    points: &[Vec3],
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

    // Node pool for triplet lookup
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

    let weld_distance = default_weld_distance(points.len());
    let weld_unions = weld_vertices_by_distance(&mut all_vertices, &mut cell_indices, weld_distance, print_timing);
    let t3 = Instant::now();

    let (deduped_cells, deduped_indices) = deduplicate_cell_indices(&cells, &cell_indices);
    let t4 = Instant::now();

    if print_timing {
        let mut pre_lt3 = 0usize;
        let mut post_lt3 = 0usize;
        let mut over_merged = 0usize;
        for (cell, deduped) in cells.iter().zip(deduped_cells.iter()) {
            let pre = cell.vertex_count();
            let post = deduped.vertex_count();
            if pre < 3 {
                pre_lt3 += 1;
            }
            if post < 3 {
                post_lt3 += 1;
            }
            if pre >= 3 && post < 3 {
                over_merged += 1;
            }
        }
        eprintln!(
            "  [dedup-flat] setup: {:.1}ms, lookup: {:.1}ms, weld: {:.1}ms, dedup_cells: {:.1}ms (weld_dist={:.2e}, unions={})",
            (t1 - t0).as_secs_f64() * 1000.0,
            (t2 - t1).as_secs_f64() * 1000.0,
            (t3 - t2).as_secs_f64() * 1000.0,
            (t4 - t3).as_secs_f64() * 1000.0,
            weld_distance,
            weld_unions
        );
        eprintln!(
            "  [dedup-flat] pre_lt3={}, post_lt3={}, over_merged={}",
            pre_lt3,
            post_lt3,
            over_merged
        );
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
