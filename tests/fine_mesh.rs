use hex3::world::{FineWorld, World};

#[test]
fn fine_mesh_pipeline_smoke() {
    let mut world = World::new(2027, 1200, 1);
    world.generate_plates(6);
    world.generate_crust();
    world.generate_dynamics();
    world.generate_features();
    world.generate_elevation();
    world.generate_atmosphere();

    let fine = FineWorld::generate_with_target(
        world.seed,
        &world.tessellation,
        world.crust.as_ref().unwrap(),
        world.features.as_ref().unwrap(),
        world.elevation.as_ref().unwrap(),
        world.atmosphere.as_ref().unwrap(),
        4000,
    );

    let n = fine.tessellation.num_cells();
    // The count is emergent (the criterion sizes cells; the 4000 passed in is a
    // cap), approached within stochastic-rounding tolerance — not a fixed target.
    assert!(n > 1000, "fine mesh implausibly small: {n}");
    assert!(n <= 4400, "fine mesh exceeded cap + tolerance: {n}");
    assert_eq!(fine.coarse_cell.len(), n);
    assert_eq!(fine.elevation.values.len(), n);
    assert_eq!(fine.hydrology.flow_accumulation.len(), n);
    assert!(fine.elevation.values.iter().all(|v| v.is_finite()));
    assert!(fine
        .fields
        .precipitation
        .iter()
        .all(|v| v.is_finite() && *v >= 0.0));

    // Sea level is a fixed datum inherited from the coarse mesh: the land AREA
    // fraction should be land-like, not drowned (the old cell-count solve bug
    // pushed it toward ~0) nor all-land. The band is wide because this smoke mesh
    // is tiny (1200 coarse / ~4000 fine), so coastline interpolation inflates
    // land area; at production scale it lands near LAND_FRACTION (~0.26).
    // (By cell count land is ~90% here, since cells concentrate on the land.)
    let areas = fine.tessellation.cell_areas();
    let total_area: f32 = areas.iter().sum();
    let land_area: f32 = areas
        .iter()
        .zip(fine.elevation.values.iter())
        .filter(|(_, &e)| e >= 0.0)
        .map(|(&a, _)| a)
        .sum();
    let land_fraction = land_area / total_area;
    assert!(
        (0.15..0.45).contains(&land_fraction),
        "land area fraction {land_fraction} outside sane band (LAND_FRACTION {})",
        hex3::world::LAND_FRACTION
    );
    assert!(fine.tessellation.morans_i(&fine.fields.temperature) > 0.85);
    assert!(fine.tessellation.morans_i(&fine.fields.precipitation) > 0.55);
}
