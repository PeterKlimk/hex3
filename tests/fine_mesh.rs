#[cfg(feature = "research-landscape")]
use glam::Vec3;
#[cfg(feature = "research-landscape")]
use hex3::world::{
    apply_conservative_finite_age_flux_v0, collect_convergent_fronts, collect_plate_boundaries,
    frozen_support_uplift, project_owned_front, FiniteAgeFluxModel,
};
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
        #[cfg(feature = "research-landscape")]
        world.tectonic_history.as_ref().unwrap(),
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

#[cfg(feature = "research-landscape")]
#[test]
fn owned_front_projection_uses_continuous_exact_arc_coordinates() {
    let mut fronts = OrogenFronts::default();
    fronts.seg_a.push(Vec3::X);
    fronts.seg_b.push(Vec3::Y);
    fronts.chain_id.push(17);
    fronts.u_lin.push(2.0);
    fronts.u_dir.push(1.0);

    let angle = std::f32::consts::FRAC_PI_8;
    let on_arc = Vec3::new(angle.cos(), angle.sin(), 0.0);
    let first = project_owned_front(on_arc, 0, &fronts).unwrap();
    let second = project_owned_front(on_arc, 0, &fronts).unwrap();

    assert_eq!(first, second, "projection must be deterministic");
    assert_eq!(first.front_index, 0);
    assert_eq!(first.chain_id, 17);
    assert!((first.u_lin_radians - (2.0 - std::f32::consts::FRAC_PI_8)).abs() < 1e-6);
    assert!(first.arc_distance_radians.abs() < 1e-6);

    let latitude = 0.3f32;
    let off_arc = Vec3::new(
        angle.cos() * latitude.cos(),
        angle.sin() * latitude.cos(),
        latitude.sin(),
    );
    let offset = project_owned_front(off_arc, 0, &fronts).unwrap();
    assert!((offset.u_lin_radians - first.u_lin_radians).abs() < 1e-6);
    assert!((offset.arc_distance_radians - latitude).abs() < 1e-6);

    assert!(project_owned_front(on_arc, u32::MAX, &fronts).is_none());
    assert!(project_owned_front(on_arc, 1, &fronts).is_none());
}

#[cfg(feature = "research-landscape")]
#[test]
fn finite_age_source_retains_front_and_episode_provenance() {
    let world = coarse_world();
    let fronts = OrogenFronts::build(
        &world.tessellation,
        world.plates.as_ref().unwrap(),
        world.crust.as_ref().unwrap(),
        world.dynamics.as_ref().unwrap(),
        world.tectonic_history.as_ref().unwrap(),
    );
    let fine = generate_pre(&world);
    let source = frozen_support_uplift(&fine.base, &fronts);

    let front_count = fronts.points.len();
    for len in [
        fronts.seg_a.len(),
        fronts.seg_b.len(),
        fronts.accept_plate.len(),
        fronts.arc_u.len(),
        fronts.chain_id.len(),
        fronts.u_lin.len(),
        fronts.u_dir.len(),
        fronts.episode_duration_myr.len(),
        fronts.episode_id.len(),
        fronts.edge_id.len(),
        fronts.convergence_km_per_myr.len(),
    ] {
        assert_eq!(len, front_count, "front provenance vector misalignment");
    }
    for ((&episode_id, &duration), edge_id) in fronts
        .episode_id
        .iter()
        .zip(&fronts.episode_duration_myr)
        .zip(&fronts.edge_id)
    {
        let history = world.tectonic_history.as_ref().unwrap();
        assert!(episode_id < history.episodes.len());
        assert_eq!(
            history
                .episode_for_edge(edge_id.cell_a, edge_id.cell_b)
                .unwrap()
                .id,
            episode_id
        );
        assert_eq!(
            duration.to_bits(),
            history.episodes[episode_id].duration_myr.to_bits()
        );
    }

    let fine_count = fine.base.tessellation.num_cells();
    assert_eq!(source.shape.len(), fine_count);
    assert_eq!(source.duration_myr.len(), fine_count);
    assert_eq!(source.owner_front.len(), fine_count);
    assert!(source.owned_cells > 0);
    assert!(source.shape.contains(&0.0));
    assert_eq!(
        source.owned_cells,
        source.shape.iter().filter(|&&rate| rate > 0.0).count()
    );
    for ((&rate, &duration), &owner) in source
        .shape
        .iter()
        .zip(&source.duration_myr)
        .zip(&source.owner_front)
    {
        if rate > 0.0 {
            let owner = owner as usize;
            assert!(owner < front_count, "positive source has invalid owner");
            assert_eq!(
                duration.to_bits(),
                fronts.episode_duration_myr[owner].to_bits()
            );
            assert_eq!(
                duration.to_bits(),
                world.tectonic_history.as_ref().unwrap().episodes[fronts.episode_id[owner]]
                    .duration_myr
                    .to_bits()
            );
        } else {
            assert_eq!(owner, u32::MAX, "zero source retained a front owner");
            assert_eq!(duration.to_bits(), 0.0f32.to_bits());
        }
    }
    for (cell, &owner) in source.owner_front.iter().enumerate() {
        let projection =
            project_owned_front(fine.base.tessellation.cell_center(cell), owner, &fronts);
        if owner == u32::MAX {
            assert!(projection.is_none());
            continue;
        }
        let projection = projection.expect("retained owner must project onto its exact arc");
        let owner = owner as usize;
        assert_eq!(projection.front_index as usize, owner);
        assert_eq!(projection.chain_id, fronts.chain_id[owner]);
        assert!(projection.u_lin_radians.is_finite());
        assert!((0.0..=std::f32::consts::PI).contains(&projection.arc_distance_radians));
        let segment_length = fronts.seg_a[owner]
            .dot(fronts.seg_b[owner])
            .clamp(-1.0, 1.0)
            .acos();
        assert!(
            (projection.u_lin_radians - fronts.u_lin[owner]).abs() <= 0.5 * segment_length + 1e-6,
            "continuous coordinate escaped its retained owner segment"
        );
    }
}

#[cfg(feature = "research-landscape")]
#[test]
fn finite_age_raw_flux_model_preserves_source_bytes() {
    let world = coarse_world();
    let fine = generate_pre(&world);
    let direct_fronts = OrogenFronts::build(
        &world.tessellation,
        world.plates.as_ref().unwrap(),
        world.crust.as_ref().unwrap(),
        world.dynamics.as_ref().unwrap(),
        world.tectonic_history.as_ref().unwrap(),
    );
    let configured_fronts = OrogenFronts::build(
        &world.tessellation,
        world.plates.as_ref().unwrap(),
        world.crust.as_ref().unwrap(),
        world.dynamics.as_ref().unwrap(),
        world.tectonic_history.as_ref().unwrap(),
    );
    let params = ErosionParams::default();
    assert_eq!(
        params.finite_age_flux_model,
        FiniteAgeFluxModel::RawEdgePositiveV0
    );
    assert_eq!(
        serde_json::to_value(params).unwrap()["finite_age_flux_model"],
        "raw-edge-positive-v0"
    );

    // This is the raw branch used by World: no adapter call between the existing
    // front constructor and existing frozen-support builder.
    let direct = frozen_support_uplift(&fine.base, &direct_fronts);
    let configured = frozen_support_uplift(&fine.base, &configured_fronts);
    assert_eq!(
        direct
            .shape
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        configured
            .shape
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        direct
            .duration_myr
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        configured
            .duration_myr
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
    assert_eq!(direct.owner_front, configured.owner_front);
    assert_eq!(direct.owned_cells, configured.owned_cells);
    assert_eq!(direct.distinct_durations, configured.distinct_durations);
}

#[cfg(feature = "research-landscape")]
#[test]
fn conservative_finite_age_flux_installs_closed_candidate_with_same_provenance() {
    let world = coarse_world();
    let mut fronts = OrogenFronts::build(
        &world.tessellation,
        world.plates.as_ref().unwrap(),
        world.crust.as_ref().unwrap(),
        world.dynamics.as_ref().unwrap(),
        world.tectonic_history.as_ref().unwrap(),
    );
    let raw = fronts.clone();
    let audit = apply_conservative_finite_age_flux_v0(
        &mut fronts,
        &world.tessellation,
        world.plates.as_ref().unwrap(),
        world.crust.as_ref().unwrap(),
        world.dynamics.as_ref().unwrap(),
        world.tectonic_history.as_ref().unwrap(),
    )
    .unwrap();

    assert_eq!(
        audit.model,
        FiniteAgeFluxModel::ConservativeSignedOneCollisionWidthV0
    );
    assert_eq!(audit.implicit_substeps, 8);
    assert_eq!(
        audit.sigma_km.to_bits(),
        (f64::from(hex3::world::COLLISION_WIDTH) * f64::from(hex3::world::PLANET_RADIUS_KM))
            .to_bits()
    );
    assert!(audit.processed_segment_count > 0);
    assert!(audit.processed_edge_count > 0);
    assert_eq!(
        audit.processed_edge_count + audit.untouched_omission_edge_count,
        fronts.edge_id.len()
    );
    let signed_scale = audit.input_signed_flux_km2_per_myr.abs().max(1.0);
    assert!(audit.closure_residual_km2_per_myr.abs() <= 1e-10 * signed_scale);
    assert!(audit.installed_f32_cast_residual_km2_per_myr.abs() <= 1e-5 * signed_scale);
    assert!(audit.rectification_excess_reduction_km2_per_myr >= -1e-8 * signed_scale);

    // The candidate changes only aligned source rate: topology, exact owner IDs,
    // age and chain coordinates remain the raw source's provenance.
    assert_eq!(fronts.points, raw.points);
    assert_eq!(fronts.seg_a, raw.seg_a);
    assert_eq!(fronts.seg_b, raw.seg_b);
    assert_eq!(fronts.accept_plate, raw.accept_plate);
    assert_eq!(fronts.arc_u, raw.arc_u);
    assert_eq!(fronts.chain_id, raw.chain_id);
    assert_eq!(fronts.u_lin, raw.u_lin);
    assert_eq!(fronts.u_dir, raw.u_dir);
    assert_eq!(fronts.episode_duration_myr, raw.episode_duration_myr);
    assert_eq!(fronts.episode_id, raw.episode_id);
    assert_eq!(fronts.edge_id, raw.edge_id);
    assert!(fronts
        .convergence_km_per_myr
        .iter()
        .all(|rate| rate.is_finite() && *rate >= 0.0));
    assert!(fronts
        .convergence_km_per_myr
        .iter()
        .zip(&raw.convergence_km_per_myr)
        .any(|(&candidate, &control)| candidate.to_bits() != control.to_bits()));

    let boundaries = collect_plate_boundaries(
        &world.tessellation,
        world.plates.as_ref().unwrap(),
        world.crust.as_ref().unwrap(),
        world.dynamics.as_ref().unwrap(),
    );
    let exact = collect_convergent_fronts(
        &world.tessellation,
        &boundaries,
        world.tectonic_history.as_ref().unwrap(),
    )
    .unwrap();
    let length_by_id: std::collections::BTreeMap<_, _> = exact
        .edges
        .iter()
        .map(|edge| (edge.id, f64::from(edge.length_km)))
        .collect();
    let installed_positive = fronts
        .edge_id
        .iter()
        .zip(&fronts.convergence_km_per_myr)
        .map(|(edge_id, &rate)| length_by_id[edge_id] * f64::from(rate))
        .sum::<f64>();
    assert_eq!(
        installed_positive.to_bits(),
        audit
            .installed_f32_positive_clipped_flux_km2_per_myr
            .to_bits()
    );
    assert_eq!(
        serde_json::to_value(audit).unwrap()["model"],
        "conservative-signed-one-collision-width-v0"
    );
}

#[cfg(feature = "research-landscape")]
#[test]
fn finite_age_frozen_support_candidate_reaches_authoritative_hydrology() {
    let mut world = coarse_world();
    world.fine_cache = FineCacheMode::Disabled;
    world.fine_structure_params.emergent_lambda = 1.0;
    world.fine_structure_params.emergent_structured = 0.0;
    world.fine_structure_params.interior_relief = 0.0;
    world.fine_structure_params.fault_scarp_height = 0.0;
    world.erosion_params.finite_age_uplift = true;
    world.erosion_params.steps = 12;
    world.erosion_params.precip_outer_iters = 1;
    world.erosion_params.hillslope_critical_slope = 200.0;

    world.generate_fine_pre_with_cap(4000);
    let pre = world.fine.as_ref().unwrap().pre.elevation.values.clone();
    world.generate_fine_eroded();

    let fine = world.fine.as_ref().unwrap();
    let candidate = fine.eroded.as_ref().expect("finite-age stage-4 surface");
    assert_eq!(
        candidate.elevation.values.len(),
        fine.base.tessellation.num_cells()
    );
    assert_eq!(
        candidate.hydrology.drainage_dir.len(),
        fine.base.tessellation.num_cells()
    );
    assert!(candidate
        .elevation
        .values
        .iter()
        .all(|value| value.is_finite()));
    assert!(candidate
        .elevation
        .values
        .iter()
        .zip(pre)
        .any(|(&after, before)| after.to_bits() != before.to_bits()));
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
