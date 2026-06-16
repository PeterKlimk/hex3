//! Numerical measurement of generated-world features.
//!
//! Connected-component analysis over cell masks with physical sizes (km,
//! km² at Earth scale), plus per-component property aggregation. This is the
//! primary diagnostic instrument for judging whether generated artifacts
//! (rift seaways, islands, mountain ranges, collision plateaus...) are
//! physically plausible — spatial statistics catch what summary stats and
//! rendered images both miss.

use super::Tessellation;

/// Earth radius used to express angular sizes in physical units.
pub use super::constants::PLANET_RADIUS_KM as EARTH_RADIUS_KM;

/// Measured geometry of one connected component of a cell mask.
#[derive(Debug, Clone)]
pub struct ComponentStats {
    /// Cell indices in this component.
    pub cells: Vec<usize>,
    /// Surface area (km², Earth scale).
    pub area_km2: f32,
    /// Greatest geodesic extent (km) — max pairwise distance, sampled.
    pub length_km: f32,
    /// Mean width (km): area / length.
    pub width_km: f32,
}

impl ComponentStats {
    /// Elongation ratio (length / mean width). 1 ~ round, >5 ~ linear.
    pub fn elongation(&self) -> f32 {
        if self.width_km > 1e-3 {
            self.length_km / self.width_km
        } else {
            0.0
        }
    }

    /// Mean of a per-cell field over this component.
    pub fn mean_of(&self, field: &[f32]) -> f32 {
        if self.cells.is_empty() {
            return 0.0;
        }
        self.cells.iter().map(|&c| field[c]).sum::<f32>() / self.cells.len() as f32
    }

    /// Min/max of a per-cell field over this component.
    pub fn range_of(&self, field: &[f32]) -> (f32, f32) {
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for &c in &self.cells {
            lo = lo.min(field[c]);
            hi = hi.max(field[c]);
        }
        (lo, hi)
    }

    /// Fraction of component cells satisfying a predicate.
    pub fn fraction_where(&self, pred: impl Fn(usize) -> bool) -> f32 {
        if self.cells.is_empty() {
            return 0.0;
        }
        self.cells.iter().filter(|&&c| pred(c)).count() as f32 / self.cells.len() as f32
    }
}

/// Find connected components of `mask` over the tessellation adjacency and
/// measure each. Sorted by area, largest first.
pub fn measure_components(tessellation: &Tessellation, mask: &[bool]) -> Vec<ComponentStats> {
    let n = tessellation.num_cells();
    let areas = tessellation.cell_areas();
    let mut visited = vec![false; n];
    let mut out = Vec::new();

    for start in 0..n {
        if !mask[start] || visited[start] {
            continue;
        }
        // BFS flood
        let mut cells = Vec::new();
        let mut queue = std::collections::VecDeque::new();
        visited[start] = true;
        queue.push_back(start);
        while let Some(cell) = queue.pop_front() {
            cells.push(cell);
            for &nb in tessellation.neighbors(cell) {
                if mask[nb] && !visited[nb] {
                    visited[nb] = true;
                    queue.push_back(nb);
                }
            }
        }

        let area_sr: f32 = cells.iter().map(|&c| areas[c]).sum();
        let area_km2 = area_sr * EARTH_RADIUS_KM * EARTH_RADIUS_KM;
        let length_km = max_extent_km(tessellation, &cells);
        let width_km = if length_km > 1e-3 {
            area_km2 / length_km
        } else {
            0.0
        };

        out.push(ComponentStats {
            cells,
            area_km2,
            length_km,
            width_km,
        });
    }

    out.sort_by(|a, b| b.area_km2.total_cmp(&a.area_km2));
    out
}

/// Greatest geodesic extent of a cell set (km). Max pairwise arc distance
/// over a deterministic sample (exact for small components).
fn max_extent_km(tessellation: &Tessellation, cells: &[usize]) -> f32 {
    const MAX_SAMPLE: usize = 200;
    let stride = (cells.len() / MAX_SAMPLE).max(1);
    let sample: Vec<usize> = cells.iter().step_by(stride).copied().collect();

    let mut max_arc = 0.0f32;
    for (i, &a) in sample.iter().enumerate() {
        let pa = tessellation.cell_center(a);
        for &b in &sample[i + 1..] {
            let arc = pa.dot(tessellation.cell_center(b)).clamp(-1.0, 1.0).acos();
            max_arc = max_arc.max(arc);
        }
    }
    max_arc * EARTH_RADIUS_KM
}

/// Multi-source graph distance (radians, arc-length weighted) from a seed
/// mask to all cells. Useful for measured gaps (e.g. arc-to-trench).
pub fn distance_from_mask(tessellation: &Tessellation, seeds: &[bool]) -> Vec<f32> {
    use ordered_float::OrderedFloat;
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    let n = tessellation.num_cells();
    let mut dist = vec![f32::INFINITY; n];
    let mut heap: BinaryHeap<Reverse<(OrderedFloat<f32>, usize)>> = BinaryHeap::new();
    for i in 0..n {
        if seeds[i] {
            dist[i] = 0.0;
            heap.push(Reverse((OrderedFloat(0.0), i)));
        }
    }
    while let Some(Reverse((OrderedFloat(d), cell))) = heap.pop() {
        if d > dist[cell] {
            continue;
        }
        let pos = tessellation.cell_center(cell);
        for &nb in tessellation.neighbors(cell) {
            let arc = pos
                .dot(tessellation.cell_center(nb))
                .clamp(-1.0, 1.0)
                .acos();
            let nd = d + arc;
            if nd < dist[nb] {
                dist[nb] = nd;
                heap.push(Reverse((OrderedFloat(nd), nb)));
            }
        }
    }
    dist
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn components_partition_mask_and_measure_sanely() {
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let tessellation = Tessellation::generate(2000, 1, &mut rng);
        let n = tessellation.num_cells();

        // Mask: northern cap (single component, ~known size).
        let mask: Vec<bool> = (0..n)
            .map(|i| tessellation.cell_center(i).y > 0.8)
            .collect();
        let comps = measure_components(&tessellation, &mask);
        assert_eq!(comps.len(), 1, "spherical cap should be one component");

        let comp = &comps[0];
        // Cap area: 2*pi*(1-0.8) sr -> km².
        let expected = 2.0 * std::f32::consts::PI * 0.2 * EARTH_RADIUS_KM * EARTH_RADIUS_KM;
        assert!(
            (comp.area_km2 - expected).abs() / expected < 0.1,
            "cap area {} vs expected {expected}",
            comp.area_km2
        );
        // Cap diameter: 2*acos(0.8) rad along the surface.
        let expected_len = 2.0 * 0.8f32.acos() * EARTH_RADIUS_KM;
        assert!(
            (comp.length_km - expected_len).abs() / expected_len < 0.15,
            "cap extent {} vs expected {expected_len}",
            comp.length_km
        );
        assert_eq!(
            mask.iter().filter(|&&m| m).count(),
            comp.cells.len(),
            "component should cover the whole mask"
        );
    }
}
