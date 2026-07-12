use glam::Vec3;
use hex3::world::landscape::{
    apply_effective_areal_denudation, flow_aligned_physical_grade,
    reconstruct_mean_surface_gradient, C0LandscapeParams, C0LandscapeSolver, C0LandscapeState,
    ConservativeHillslopeParams, DeformationFrame, EffectiveArealDenudationParams, FaceFlowCache,
    FlowPartition, LandscapeMesh,
};

const WIDTH_KM: f64 = 64.0;
const HEIGHT_KM: f64 = 48.0;

fn initial_surface(mesh: &LandscapeMesh) -> Vec<f64> {
    mesh.cell_center_km
        .iter()
        .map(|p| {
            let y = 2.0 * p.y / HEIGHT_KM;
            // A smooth divide at y=0 and strictly outward fall on either side.
            0.35 + 0.22 * (1.0 - y * y)
                + 0.012 * (std::f64::consts::PI * p.x / WIDTH_KM).cos() * (1.0 - y * y).max(0.0)
        })
        .collect()
}

fn forcing(mesh: &LandscapeMesh) -> DeformationFrame {
    let rock_vertical_rate_km_myr = mesh
        .cell_center_km
        .iter()
        .map(|p| {
            let r2 = (p.x / 18.0).powi(2) + (p.y / 12.0).powi(2);
            (0.012 * (-r2).exp()) as f32
        })
        .collect();
    DeformationFrame {
        rock_vertical_rate_km_myr,
        horizontal_velocity_km_myr: vec![Vec3::ZERO; mesh.cell_count()],
        dominant_episode: vec![None; mesh.cell_count()],
    }
}

fn params() -> C0LandscapeParams {
    C0LandscapeParams {
        effective_areal_denudation: EffectiveArealDenudationParams {
            // Explicit manufactured-regime value. The product default remains zero.
            // Direct-specific-discharge manufactured law (m=n=1). This K is
            // confined to the fixture and chosen only for a resolvable response.
            k: 1.0e-5,
            discharge_exponent_m: 1.0,
            slope_exponent_n: 1.0,
        },
        runoff_depth_rate_km_myr: 100.0,
        hillslope: ConservativeHillslopeParams {
            // Keep this convergence fixture specific to uplift + C0 denudation;
            // hillslope transport has its own manufactured finite-volume tests.
            diffusivity_km2_myr: 0.0,
            ..ConservativeHillslopeParams::default()
        },
        maximum_uplift_depth_km: 1.0,
        maximum_effective_denudation_depth_km: 1.0,
        ..C0LandscapeParams::default()
    }
}

fn run(spacing_km: f64, requested_dt_myr: f64, end_myr: f64) -> (LandscapeMesh, C0LandscapeState) {
    let mesh = LandscapeMesh::uniform_planar_hex(WIDTH_KM, HEIGHT_KM, spacing_km).unwrap();
    let solver = C0LandscapeSolver::new(params()).unwrap();
    let frame = forcing(&mesh);
    let mut state = C0LandscapeState::new(&mesh, initial_surface(&mesh)).unwrap();
    while state.time_myr < end_myr - 16.0 * f64::EPSILON {
        let dt = requested_dt_myr.min(end_myr - state.time_myr);
        let diagnostic = solver.step(&mesh, &frame, dt, &mut state).unwrap();
        let water_scale = diagnostic.water.total_supply_km3_myr.max(1.0);
        assert!(diagnostic.water.water_balance_error_km3_myr.abs() < 2.0e-12 * water_scale);
        assert!(diagnostic.closure_error_km3.abs() < 2.0e-10);
    }
    let ledger = state.elevation_volume_moment_ledger;
    assert!(ledger.effective_areal_denudation_export_km3 > 0.0);
    assert!(ledger.rock_uplift_moment_km3 > 0.0);
    assert!(ledger.closure_error_km3.abs() < 5.0e-10);
    (mesh, state)
}

fn weighted_rms_difference(mesh: &LandscapeMesh, a: &[f64], b: &[f64]) -> f64 {
    let (weighted_square, weight) = mesh
        .cell_center_km
        .iter()
        .zip(&mesh.cell_area_km2)
        .zip(a.iter().zip(b))
        .map(|((p, area), (a, b))| {
            // Smoothly suppress the portal and closed-side boundary stencils.
            let w = (-(p.x / 22.0).powi(8) - (p.y / 15.0).powi(8)).exp() * area;
            (w * (a - b).powi(2), w)
        })
        .fold((0.0, 0.0), |sum, item| (sum.0 + item.0, sum.1 + item.1));
    (weighted_square / weight).sqrt()
}

#[derive(Debug, Clone, Copy)]
struct Summary {
    mean: f64,
    rms: f64,
    uplift: f64,
    denudation: f64,
}

fn buffered_summary(mesh: &LandscapeMesh, state: &C0LandscapeState) -> Summary {
    let initial = initial_surface(mesh);
    let uplift_rate = forcing(mesh).rock_vertical_rate_km_myr;
    let (weight, first, second, uplift, denudation) = mesh
        .cell_center_km
        .iter()
        .zip(&mesh.cell_area_km2)
        .enumerate()
        .map(|(cell, (p, area))| {
            let z = state.mean_bedrock_elevation_km[cell];
            let w = (-(p.x / 22.0).powi(8) - (p.y / 15.0).powi(8)).exp() * area;
            let uplift_depth = f64::from(uplift_rate[cell]) * state.time_myr;
            (
                w,
                w * z,
                w * z * z,
                w * uplift_depth,
                w * (initial[cell] + uplift_depth - z),
            )
        })
        .fold((0.0, 0.0, 0.0, 0.0, 0.0), |sum, item| {
            (
                sum.0 + item.0,
                sum.1 + item.1,
                sum.2 + item.2,
                sum.3 + item.3,
                sum.4 + item.4,
            )
        });
    Summary {
        mean: first / weight,
        rms: (second / weight).sqrt(),
        uplift,
        denudation,
    }
}

#[test]
fn manufactured_c0_run_is_deterministic_and_closes_ledgers() {
    let (mesh_a, a) = run(4.0, 0.002, 0.012);
    let (mesh_b, b) = run(4.0, 0.002, 0.012);
    assert!(
        a.elevation_volume_moment_ledger
            .effective_areal_denudation_export_km3
            > 1.0e-3
    );
    assert_eq!(mesh_a, mesh_b);
    assert_eq!(a, b);
}

#[test]
fn manufactured_c0_temporal_error_shrinks_under_step_halving() {
    let (mesh_coarse, coarse) = run(4.0, 0.004, 0.016);
    let (mesh_medium, medium) = run(4.0, 0.002, 0.016);
    let (mesh_fine, fine) = run(4.0, 0.001, 0.016);
    assert_eq!(mesh_coarse, mesh_medium);
    assert_eq!(mesh_medium, mesh_fine);

    let coarse_medium = weighted_rms_difference(
        &mesh_coarse,
        &coarse.mean_bedrock_elevation_km,
        &medium.mean_bedrock_elevation_km,
    );
    let medium_fine = weighted_rms_difference(
        &mesh_medium,
        &medium.mean_bedrock_elevation_km,
        &fine.mean_bedrock_elevation_km,
    );
    assert!(
        medium_fine < 0.8 * coarse_medium,
        "temporal refinement did not reduce error: dt-dt/2={coarse_medium:e}, dt/2-dt/4={medium_fine:e}"
    );
}

#[test]
fn manufactured_c0_buffered_summaries_improve_with_spatial_refinement() {
    let (_, reference_state) = run(1.0, 0.001, 0.012);
    let reference_mesh = LandscapeMesh::uniform_planar_hex(WIDTH_KM, HEIGHT_KM, 1.0).unwrap();
    let reference = buffered_summary(&reference_mesh, &reference_state);

    let mut summaries = Vec::new();
    for spacing in [8.0, 4.0, 2.0] {
        let (mesh, state) = run(spacing, 0.001, 0.012);
        summaries.push(buffered_summary(&mesh, &state));
    }
    let errors_for = |field: fn(Summary) -> f64| {
        summaries
            .iter()
            .map(|summary| (field(*summary) - field(reference)).abs())
            .collect::<Vec<_>>()
    };
    for (name, errors) in [
        ("buffered mean elevation", errors_for(|s| s.mean)),
        ("buffered RMS elevation", errors_for(|s| s.rms)),
        ("integrated uplift moment", errors_for(|s| s.uplift)),
        ("integrated denudation export", errors_for(|s| s.denudation)),
    ] {
        assert!(
            errors[1] < errors[0] && errors[2] < errors[1],
            "{name} errors did not improve monotonically (8/4/2 km): {errors:?}"
        );
    }
}

#[test]
fn depression_fill_carries_water_but_cannot_invent_physical_denudation_grade() {
    let mesh = LandscapeMesh::uniform_planar_hex(WIDTH_KM, HEIGHT_KM, 4.0).unwrap();
    // This closed physical bowl can reach the north/south portals only by a
    // derived fill route that initially runs flat or uphill on the real surface.
    let physical: Vec<_> = mesh
        .cell_center_km
        .iter()
        .map(|p| 0.2 + 0.0002 * (p.x * p.x + p.y * p.y))
        .collect();
    let supply: Vec<_> = mesh.cell_area_km2.iter().map(|area| 100.0 * area).collect();
    let flow =
        FaceFlowCache::route_with_depressions(&mesh, &physical, &supply, FlowPartition::MfdSlope)
            .unwrap();
    assert!(flow.total_portal_outflow_km3_myr > 0.0);
    assert!(flow
        .routing_elevation_km
        .iter()
        .zip(&physical)
        .any(|(route, real)| route > real));

    let gradient = reconstruct_mean_surface_gradient(&mesh, &physical).unwrap();
    let grade =
        flow_aligned_physical_grade(&gradient.vector, &flow.specific_discharge_vector_km2_myr)
            .unwrap();
    let uphill_routed: Vec<_> = flow
        .specific_discharge_vector_km2_myr
        .iter()
        .zip(&gradient.vector)
        .enumerate()
        .filter_map(|(cell, (q, grad))| (q.length() > 0.0 && q.dot(*grad) >= 0.0).then_some(cell))
        .collect();
    assert!(uphill_routed.len() > mesh.cell_count() / 3);
    assert!(uphill_routed.iter().all(|&cell| grade[cell] == 0.0));

    let mut evolved = physical.clone();
    let result = apply_effective_areal_denudation(
        params().effective_areal_denudation,
        &mesh,
        &mut evolved,
        &flow.specific_discharge_km2_myr,
        &grade,
        0.01,
    )
    .unwrap();
    assert!(uphill_routed
        .iter()
        .all(|&cell| result.rate_km_myr[cell] == 0.0 && evolved[cell] == physical[cell]));
}
