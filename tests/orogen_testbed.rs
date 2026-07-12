use hex3::world::landscape::{
    linked_scenario, uniform_scenario, BoundaryCondition, DeformationEvaluator, LandscapeMesh,
    LandscapeParams, LandscapeSolver, LandscapeState,
};

fn initial_bedrock(mesh: &LandscapeMesh) -> Vec<f64> {
    let max_y = mesh
        .cell_center_km
        .iter()
        .map(|point| point.y.abs())
        .fold(0.0, f64::max);
    mesh.cell_center_km
        .iter()
        .enumerate()
        .map(|(cell, point)| {
            if matches!(mesh.boundary[cell], BoundaryCondition::OpenBaseLevel { .. }) {
                0.0
            } else {
                let crown = 0.02 * (1.0 - point.y.abs() / max_y).max(0.0);
                let perturbation = 0.002 * (0.071 * point.x + 0.043 * point.y).sin()
                    + 0.0015 * (-0.038 * point.x + 0.063 * point.y).sin();
                crown + perturbation
            }
        })
        .collect()
}

fn run(
    mesh: &LandscapeMesh,
    evaluator: &DeformationEvaluator,
    requested_dt_myr: f64,
    end_myr: f64,
) -> LandscapeState {
    let solver = LandscapeSolver::new(LandscapeParams::default()).unwrap();
    let mut state = LandscapeState::new(mesh, initial_bedrock(mesh)).unwrap();
    solver.refresh_drainage(mesh, &mut state).unwrap();
    while state.time_myr < end_myr - 1.0e-12 {
        let requested = requested_dt_myr.min(end_myr - state.time_myr);
        solver
            .step_with_forcing(mesh, requested, &mut state, |time| evaluator.evaluate(time))
            .unwrap();
    }
    state
}

fn assert_closed(state: &LandscapeState) {
    let scale = state.ledger.rock_uplift_km3.abs().max(1.0);
    assert!(
        state.ledger.closure_error_km3.abs() <= scale * 1.0e-8,
        "landscape ledger failed to close: {:?}",
        state.ledger
    );
    assert!(
        state.drainage.water_balance_error_km3_myr().abs()
            <= state.drainage.total_runoff_km3_myr * 1.0e-12
    );
}

#[test]
fn coupled_uniform_and_linked_cases_are_budget_matched_but_distinct() {
    let mesh = LandscapeMesh::uniform_planar_hex(240.0, 160.0, 8.0).unwrap();
    let uniform = uniform_scenario().compile(&mesh).unwrap();
    let linked = linked_scenario().compile(&mesh).unwrap();
    let uniform_state = run(&mesh, &uniform, 0.01, 0.05);
    let linked_state = run(&mesh, &linked, 0.01, 0.05);

    assert_closed(&uniform_state);
    assert_closed(&linked_state);
    assert!(uniform_state.ledger.rock_uplift_km3 > 0.0);
    assert!(uniform_state.ledger.incision_export_km3 > 0.0);
    assert_ne!(
        uniform_state.bedrock_elevation_km,
        linked_state.bedrock_elevation_km
    );
    let work_scale = uniform_state.ledger.rock_uplift_km3.abs().max(1.0);
    assert!(
        (uniform_state.ledger.rock_uplift_km3 - linked_state.ledger.rock_uplift_km3).abs()
            <= work_scale * 2.0e-6
    );
}

#[test]
fn halving_timestep_preserves_first_coupled_response() {
    let mesh = LandscapeMesh::uniform_planar_hex(160.0, 120.0, 8.0).unwrap();
    let evaluator = linked_scenario().compile(&mesh).unwrap();
    let coarse = run(&mesh, &evaluator, 0.01, 0.04);
    let fine = run(&mesh, &evaluator, 0.005, 0.04);

    assert_closed(&coarse);
    assert_closed(&fine);
    let max_coarse = coarse
        .bedrock_elevation_km
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let max_fine = fine
        .bedrock_elevation_km
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    assert!((max_coarse - max_fine).abs() / max_fine.abs().max(1.0e-9) < 0.05);
    assert!(
        (coarse.ledger.incision_export_km3 - fine.ledger.incision_export_km3).abs()
            / fine.ledger.incision_export_km3.abs().max(1.0e-9)
            < 0.05
    );
}
