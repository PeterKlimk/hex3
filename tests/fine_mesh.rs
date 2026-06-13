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
    assert_eq!(n, 4000);
    assert_eq!(fine.coarse_cell.len(), n);
    assert_eq!(fine.elevation.values.len(), n);
    assert_eq!(fine.hydrology.flow_accumulation.len(), n);
    assert!(fine.elevation.values.iter().all(|v| v.is_finite()));
    assert!(fine
        .fields
        .precipitation
        .iter()
        .all(|v| v.is_finite() && *v >= 0.0));

    let land_fraction =
        fine.elevation.values.iter().filter(|&&e| e >= 0.0).count() as f32 / n as f32;
    assert!((land_fraction - hex3::world::LAND_FRACTION).abs() < 0.01);
    assert!(fine.tessellation.morans_i(&fine.fields.temperature) > 0.85);
    assert!(fine.tessellation.morans_i(&fine.fields.precipitation) > 0.55);
}
