//! Vertex deduplication for Voronoi cell construction.

use glam::Vec3;
use rustc_hash::FxHashMap;

use crate::geometry::VoronoiCell;
use super::{FlatCellsData, VertexKey};

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
/// Uses triplet keys for the common case and a support-list hash map for degeneracies.
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
    let mut support_map: FxHashMap<Vec<u32>, usize> = FxHashMap::default();

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
                let (key, pos) = chunk.vertices[chunk_vert_idx + local_i];
                let idx = match key {
                    VertexKey::Triplet(triplet) => {
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

                        match found_idx {
                            Some(idx) => idx as usize,
                            None => {
                                let idx = all_vertices.len();
                                all_vertices.push(pos);
                                let new_id = nodes.len() as u32;
                                nodes.push(TripletNode { bc, idx: idx as u32, next: heads[a] });
                                heads[a] = new_id;
                                idx
                            }
                        }
                    }
                    VertexKey::Support { start, len } => {
                        let start = start as usize;
                        let len = len as usize;
                        debug_assert!(start + len <= chunk.support_data.len(), "support key out of bounds");
                        let support = &chunk.support_data[start..start + len];
                        if let Some(&idx) = support_map.get(support) {
                            idx
                        } else {
                            let idx = all_vertices.len();
                            all_vertices.push(pos);
                            support_map.insert(support.to_vec(), idx);
                            idx
                        }
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

    // Deduplicate cell indices (removes consecutive duplicates after remapping)
    let (deduped_cells, deduped_indices) = deduplicate_cell_indices(&cells, &cell_indices);
    let t3 = Instant::now();

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
            "  [dedup-flat] setup: {:.1}ms, lookup: {:.1}ms, dedup_cells: {:.1}ms (support_keys={})",
            (t1 - t0).as_secs_f64() * 1000.0,
            (t2 - t1).as_secs_f64() * 1000.0,
            (t3 - t2).as_secs_f64() * 1000.0,
            support_map.len()
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
