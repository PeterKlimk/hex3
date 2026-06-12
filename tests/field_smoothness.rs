//! Regression tests for spatial field quality.
//!
//! Fields that are physically smooth at synoptic scales (rain, uplift) must
//! not degenerate into cell-scale speckle (Moran's I near 0). This is the
//! non-visual detector for the class of bug where a per-cell operator's mesh
//! noise leaks into a derived field.

use hex3::world::World;

fn stage2_world() -> World {
    let mut world = World::new(4242, 4000, 1);
    world.generate_plates(8);
    world.generate_crust();
    world.generate_dynamics();
    world.generate_features();
    world.generate_elevation();
    world.generate_atmosphere();
    world
}

#[test]
fn atmosphere_fields_are_spatially_smooth() {
    let world = stage2_world();
    let tess = &world.tessellation;
    let atmosphere = world.atmosphere.as_ref().unwrap();

    let precip_i = tess.morans_i(&atmosphere.precipitation);
    let uplift_i = tess.morans_i(&atmosphere.uplift);
    let temp_i = tess.morans_i(&atmosphere.temperature);

    println!(
        "Moran's I: precipitation={precip_i:.3}, uplift={uplift_i:.3}, temperature={temp_i:.3}"
    );

    // Temperature is latitude+elevation driven: near-perfectly smooth.
    assert!(
        temp_i > 0.95,
        "temperature Moran's I {temp_i} should be > 0.95"
    );
    // Rain and uplift must be coherent fields, not mesh-scale speckle.
    // (Raw flux-divergence uplift scores ~0.1; phi-based scores well above.)
    assert!(uplift_i > 0.6, "uplift Moran's I {uplift_i} too noisy");
    assert!(
        precip_i > 0.6,
        "precipitation Moran's I {precip_i} too noisy"
    );
}
