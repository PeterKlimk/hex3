use hex3::world::{
    BiomeKind, EcologySemantics, ErosionParams, FineCacheMode, FineCacheOutcome, FineDensityParams,
    FineStructureParams, FineWorld, LivingSurfaceSemantics, OrogenFronts, RiverNetwork,
    RiverThresholdPolicy, WaterBodySemantics, World,
};

/// Build a tiny coarse world through stage 2 (atmosphere) — shared setup for the
/// fine-mesh smoke tests.
fn coarse_world() -> World {
    let mut world = World::new(2027, 1200, 1);
    world.generate_plates(6);
    world.generate_crust();
    world.generate_dynamics();
    world.generate_features();
    world.generate_elevation();
    world.generate_atmosphere();
    world
}

fn generate_pre(world: &World) -> FineWorld {
    let fronts = OrogenFronts::build(
        &world.tessellation,
        world.plates.as_ref().unwrap(),
        world.crust.as_ref().unwrap(),
        world.dynamics.as_ref().unwrap(),
    );
    FineWorld::generate_pre(
        world.seed,
        &world.tessellation,
        world.crust.as_ref().unwrap(),
        world.features.as_ref().unwrap(),
        world.elevation.as_ref().unwrap(),
        world.atmosphere.as_ref().unwrap(),
        4000,
        FineCacheMode::Disabled,
        FineDensityParams::default(),
        FineStructureParams::default(),
        &fronts,
    )
}

#[test]
fn fine_mesh_pipeline_smoke() {
    let world = coarse_world();

    // Build the stage-3 base + pre-erosion surface (cache disabled so the smoke
    // test stays hermetic). `base` holds the mesh + transferred fields; `pre`
    // holds the (un-eroded) elevation + hydrology.
    let fine = generate_pre(&world);

    let tess = &fine.base.tessellation;
    let n = tess.num_cells();
    // The count is emergent (the criterion sizes cells; the 4000 passed in is a
    // cap), approached within stochastic-rounding tolerance — not a fixed target.
    assert!(n > 1000, "fine mesh implausibly small: {n}");
    assert!(n <= 4400, "fine mesh exceeded cap + tolerance: {n}");
    assert_eq!(fine.base.coarse_cell.len(), n);
    assert_eq!(fine.pre.elevation.values.len(), n);
    assert_eq!(fine.pre.hydrology.flow_accumulation.len(), n);
    assert_eq!(fine.cache_record.mode, FineCacheMode::Disabled);
    assert_eq!(
        fine.cache_record.outcome,
        FineCacheOutcome::DisabledGenerated
    );
    assert_eq!(fine.cache_record.max_cells, 4000);
    assert_eq!(fine.cache_record.actual_cells, n);
    assert!(fine.pre.elevation.values.iter().all(|v| v.is_finite()));
    assert!(fine
        .base
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
    let areas = tess.cell_areas();
    let total_area: f32 = areas.iter().sum();
    let land_area: f32 = areas
        .iter()
        .zip(fine.pre.elevation.values.iter())
        .filter(|(_, &e)| e >= 0.0)
        .map(|(&a, _)| a)
        .sum();
    let land_fraction = land_area / total_area;
    assert!(
        (0.15..0.45).contains(&land_fraction),
        "land area fraction {land_fraction} outside sane band (LAND_FRACTION {})",
        hex3::world::LAND_FRACTION
    );
    assert!(tess.morans_i(&fine.base.fields.temperature) > 0.85);
    assert!(tess.morans_i(&fine.base.fields.precipitation) > 0.55);

    let water = WaterBodySemantics::build(tess, &fine.pre.hydrology);
    for cell in 0..n {
        if fine.pre.hydrology.is_submerged(cell) {
            assert!(water.cell_body[cell].is_some());
        }
    }
    let mut ids: Vec<_> = water.bodies.iter().map(|body| body.id).collect();
    ids.sort_by_key(|id| (id.basin_id, id.anchor_cell));
    ids.dedup();
    assert_eq!(ids.len(), water.bodies.len());

    let rivers = RiverNetwork::build(
        tess,
        &fine.pre.hydrology,
        &water,
        RiverThresholdPolicy::default(),
    );
    assert!(rivers
        .major_cells
        .iter()
        .zip(&rivers.all_cells)
        .all(|(&major, &all)| !major || all));
    let mut covered = vec![false; n];
    for reach in &rivers.reaches {
        for pair in reach.cells.windows(2) {
            assert_eq!(fine.pre.hydrology.downstream(pair[0]), Some(pair[1]));
        }
        for &cell in &reach.cells {
            covered[cell] = true;
        }
    }
    assert!((0..n).all(|cell| !rivers.all_cells[cell] || covered[cell]));

    let ecology = EcologySemantics::build(
        tess,
        &fine.pre.elevation.values,
        &fine.pre.temperature,
        &fine.pre.precipitation,
        Some(&fine.pre.hydrology),
    );
    assert_eq!(ecology.cells.len(), n);
    assert!(ecology.cells.iter().all(|cell| {
        cell.classification_confidence.is_finite()
            && (0.0..=1.0).contains(&cell.classification_confidence)
            && cell.potentials.vegetation.is_finite()
            && (0.0..=1.0).contains(&cell.potentials.vegetation)
    }));
    for (i, cell) in ecology.cells.iter().enumerate() {
        if fine.pre.hydrology.is_ocean(i) {
            assert_eq!(cell.biome, BiomeKind::Ocean);
        } else if fine.pre.hydrology.is_lake_water(i) {
            assert_eq!(cell.biome, BiomeKind::Lake);
        }
    }

    let living_surface = LivingSurfaceSemantics::build(
        tess,
        &fine.pre.temperature,
        &fine.pre.precipitation,
        &fine.pre.hydrology,
    );
    assert_eq!(living_surface.cells.len(), n);
    for (cell, state) in living_surface.cells.iter().enumerate() {
        for value in [
            state.thermal_opportunity,
            state.relative_water_limitation,
            state.drainage_saturation,
            state.growth_opportunity,
            state.vegetation_cover,
            state.woody_share,
        ] {
            assert!(value.is_finite() && (0.0..=1.0).contains(&value));
        }
        if fine.pre.hydrology.is_submerged(cell) {
            assert_eq!(state.fractions.terrestrial_sum(), 0.0);
        } else {
            assert!((state.fractions.terrestrial_sum() - 1.0).abs() <= 2e-6);
        }
    }
}

/// Stage 4 exercises the whole erosion path: fluvial incision + transport-aware
/// deposition + terminal-lake base levels + the glacial pass. Smoke-check it runs
/// and leaves a finite, sane surface (sea level still a fixed datum).
#[test]
fn fine_mesh_erosion_stage4_smoke() {
    let world = coarse_world();
    let mut fine = generate_pre(&world);

    fine.compute_eroded(world.seed, ErosionParams::default());
    assert!(fine.has_eroded());

    let eroded = fine.surface_for(4);
    let tess = &fine.base.tessellation;
    let n = tess.num_cells();
    assert_eq!(eroded.elevation.values.len(), n);
    assert_eq!(eroded.hydrology.flow_accumulation.len(), n);
    assert!(eroded.elevation.values.iter().all(|v| v.is_finite()));

    // Erosion + glacial only sculpt land; sea level stays the fixed datum, so the
    // land-area fraction stays in the same sane band as the pre-erosion surface.
    let areas = tess.cell_areas();
    let total: f32 = areas.iter().sum();
    let land: f32 = areas
        .iter()
        .zip(eroded.elevation.values.iter())
        .filter(|(_, &e)| e >= 0.0)
        .map(|(&a, _)| a)
        .sum();
    let land_fraction = land / total;
    assert!(
        (0.15..0.45).contains(&land_fraction),
        "post-erosion land fraction {land_fraction} outside sane band"
    );
}
