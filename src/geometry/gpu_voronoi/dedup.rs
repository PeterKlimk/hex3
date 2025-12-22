//! Vertex deduplication for Voronoi cell construction.

use glam::Vec3;
use rustc_hash::FxHashMap;

use super::timing::{DedupSubPhases, Timer};
use super::{FlatCellsData, VertexKey};
use crate::geometry::VoronoiCell;

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
    _print_timing: bool,
) -> (Vec<Vec3>, Vec<VoronoiCell>, Vec<usize>, DedupSubPhases) {
    let t0 = Timer::start();

    let num_points = flat_data.num_cells();
    let total_indices: usize = flat_data
        .chunks
        .iter()
        .map(|c| c.counts.iter().map(|&count| count as usize).sum::<usize>())
        .sum();
    debug_assert_eq!(
        total_indices,
        flat_data
            .chunks
            .iter()
            .map(|c| c.vertices.len())
            .sum::<usize>(),
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

    // Key counters for timing
    #[cfg(feature = "timing")]
    let mut triplet_keys = 0u64;
    #[cfg(feature = "timing")]
    let mut support_keys = 0u64;

    #[allow(unused_variables)]
    let setup_time = t0.elapsed();
    let t1 = Timer::start();

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
                        #[cfg(feature = "timing")]
                        {
                            triplet_keys += 1;
                        }
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
                                nodes.push(TripletNode {
                                    bc,
                                    idx: idx as u32,
                                    next: heads[a],
                                });
                                heads[a] = new_id;
                                idx
                            }
                        }
                    }
                    VertexKey::Support { start, len } => {
                        #[cfg(feature = "timing")]
                        {
                            support_keys += 1;
                        }
                        let start = start as usize;
                        let len = len as usize;
                        debug_assert!(
                            start + len <= chunk.support_data.len(),
                            "support key out of bounds"
                        );
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

    #[allow(unused_variables)]
    let lookup_time = t1.elapsed();
    let t2 = Timer::start();

    // Deduplicate cell indices (removes consecutive duplicates after remapping)
    let (deduped_cells, deduped_indices, dupes_removed) =
        deduplicate_cell_indices(&cells, &cell_indices, all_vertices.len());

    #[allow(unused_variables)]
    let cell_dedup_time = t2.elapsed();

    // Build timing result
    #[cfg(feature = "timing")]
    let sub_phases = DedupSubPhases {
        setup: setup_time,
        lookup: lookup_time,
        cell_dedup: cell_dedup_time,
        overflow_collect: std::time::Duration::ZERO,
        overflow_flush: std::time::Duration::ZERO,
        concat_vertices: std::time::Duration::ZERO,
        emit_cells: std::time::Duration::ZERO,
        triplet_keys,
        support_keys,
        cell_dupes_removed: dupes_removed,
    };

    #[cfg(not(feature = "timing"))]
    let sub_phases = DedupSubPhases;

    (all_vertices, deduped_cells, deduped_indices, sub_phases)
}

/// Remove duplicate vertex indices within each cell.
/// Returns (new_cells, new_indices, dupes_removed).
pub(super) fn deduplicate_cell_indices(
    cells: &[VoronoiCell],
    cell_indices: &[usize],
    num_vertices: usize,
) -> (Vec<VoronoiCell>, Vec<usize>, u64) {
    let mut new_cells: Vec<VoronoiCell> = Vec::with_capacity(cells.len());
    let mut new_indices: Vec<usize> = Vec::with_capacity(cell_indices.len());
    let mut dupes_removed = 0u64;

    for cell in cells {
        let start = cell.vertex_start();
        let end = cell.vertex_start() + cell.vertex_count();
        let old_indices = &cell_indices[start..end];

        let new_start = new_indices.len();
        // Cells are small (<= MAX_VERTICES), so a cache-friendly linear "seen" set
        // is typically faster than a giant random-access marks array.
        let mut seen: [usize; super::MAX_VERTICES] = [0usize; super::MAX_VERTICES];
        let mut seen_len = 0usize;

        for &idx in old_indices {
            debug_assert!(idx < num_vertices, "vertex index out of bounds");
            let mut duplicate = false;
            for &v in &seen[..seen_len] {
                if v == idx {
                    duplicate = true;
                    break;
                }
            }
            if !duplicate {
                debug_assert!(seen_len < super::MAX_VERTICES);
                seen[seen_len] = idx;
                seen_len += 1;
                new_indices.push(idx);
            } else {
                dupes_removed += 1;
            }
        }
        let new_count = new_indices.len() - new_start;

        new_cells.push(VoronoiCell::new(cell.generator_index, new_start, new_count));
    }

    (new_cells, new_indices, dupes_removed)
}
