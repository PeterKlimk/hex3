//! Shared water-geography primitives used before full hydrology exists.
//!
//! Connected ocean identity is available as soon as elevation and crust setting
//! exist. Climate and hydrology must use the same definition so an inland
//! below-datum basin cannot act as an ocean moisture source and later become a
//! lake or dry basin.

use std::collections::VecDeque;

use super::Tessellation;

/// Minimum connected ocean component area as a fraction of the planet.
const MIN_OCEAN_AREA_FRACTION: f32 = 0.001;

/// Identify connected below-datum regions that touch oceanic crust.
///
/// `continentality < 0.5` denotes oceanic crust. The area threshold rejects
/// tiny below-datum components even if discretization gives them an oceanic
/// cell. The result is the shared Stage-2/Stage-3 ocean identity; lakes and
/// other inland water are derived later by hydrology.
pub(crate) fn connected_ocean_cells(
    tessellation: &Tessellation,
    continentality: &[f32],
    elevation: &[f32],
    areas: &[f32],
) -> Vec<bool> {
    let n = tessellation.num_cells();
    assert_eq!(continentality.len(), n);
    assert_eq!(elevation.len(), n);
    assert_eq!(areas.len(), n);

    let total_area: f32 = areas.iter().sum();
    let min_ocean_area = total_area * MIN_OCEAN_AREA_FRACTION;
    let mut is_ocean = vec![false; n];
    let mut visited = vec![false; n];

    for start in 0..n {
        if visited[start] || elevation[start] >= 0.0 {
            continue;
        }

        let mut component = Vec::new();
        let mut component_area = 0.0f32;
        let mut touches_oceanic = false;
        let mut queue = VecDeque::from([start]);
        visited[start] = true;

        while let Some(cell) = queue.pop_front() {
            component.push(cell);
            component_area += areas[cell];
            touches_oceanic |= continentality[cell] < 0.5;

            for &neighbor in tessellation.neighbors(cell) {
                if !visited[neighbor] && elevation[neighbor] < 0.0 {
                    visited[neighbor] = true;
                    queue.push_back(neighbor);
                }
            }
        }

        if touches_oceanic && component_area >= min_ocean_area {
            for cell in component {
                is_ocean[cell] = true;
            }
        }
    }

    is_ocean
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use super::*;

    #[test]
    fn inland_below_datum_component_is_not_ocean() {
        let mut rng = ChaCha8Rng::seed_from_u64(91);
        let tessellation = Tessellation::generate(2_000, 0, &mut rng);
        let n = tessellation.num_cells();
        let inland = 0;

        // A large connected below-datum ocean surrounds a positive-elevation
        // ring. The isolated center remains below datum on continental crust.
        let mut elevation = vec![-0.2; n];
        elevation[inland] = -0.1;
        for &neighbor in tessellation.neighbors(inland) {
            elevation[neighbor] = 0.1;
        }
        let mut continentality = vec![0.0; n];
        continentality[inland] = 1.0;
        let areas = tessellation.cell_areas();

        let mask = connected_ocean_cells(&tessellation, &continentality, &elevation, &areas);

        assert!(!mask[inland], "isolated continental depression is inland");
        assert!(
            (0..n).any(|cell| mask[cell]),
            "the large oceanic component must remain ocean"
        );
        assert!(
            tessellation
                .neighbors(inland)
                .iter()
                .all(|&cell| !mask[cell]),
            "positive-elevation separating ring is land"
        );
    }
}
