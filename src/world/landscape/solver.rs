//! Dimensioned landscape-evolution operators for the bounded orogen testbed.
//!
//! This module deliberately owns only evolving bedrock. Routing is derived from
//! the current surface, and deformation enters as a velocity rather than a
//! prescribed height. All calculations use km, km², km³, and Myr.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fmt;

use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};

use super::{BoundaryCondition, DeformationFrame, LandscapeMesh};

/// Dimensioned parameters for the first coupled landscape-evolution rung.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LandscapeParams {
    /// Stream-power coefficient in the units implied by `incision_m` and
    /// `incision_n` for `E = K Q^m S^n`.
    pub incision_k: f64,
    pub incision_m: f64,
    pub incision_n: f64,
    /// Uniform material diffusivity (km²/Myr).
    pub hillslope_diffusivity_km2_myr: f64,
    /// Material threshold expressed as physical grade.
    pub critical_slope_grade: f64,
    /// Runoff depth rate (km/Myr). One m/yr is 1,000 km/Myr.
    pub runoff_km_myr: f64,
    /// Accuracy limit on vertical rock motion in one step (km).
    pub max_uplift_step_km: f64,
    /// Accuracy limit on implicit incision, expressed as a fraction of receiver
    /// distance. This is not a stability clamp.
    pub max_incision_courant: f64,
    /// Safety multiplier applied to the explicit hillslope limit.
    pub hillslope_safety: f64,
    /// Denominator floor for the threshold-flux singularity. Encountering it is
    /// reported and should be treated as a regime warning, not hidden tuning.
    pub nonlinear_denominator_floor: f64,
    /// Minimum useful step. A lower stable limit is returned as an error.
    pub minimum_dt_myr: f64,
}

impl Default for LandscapeParams {
    fn default() -> Self {
        Self {
            // Provisional dimensioned Q-form regime. This converts a common
            // order-of-magnitude stream-power response into km, km³/Myr and
            // Myr; it has not been fitted to desired morphology.
            incision_k: 0.03,
            incision_m: 0.45,
            incision_n: 1.0,
            hillslope_diffusivity_km2_myr: 0.1,
            critical_slope_grade: 0.7,
            runoff_km_myr: 500.0,
            max_uplift_step_km: 0.02,
            max_incision_courant: 0.25,
            hillslope_safety: 0.4,
            nonlinear_denominator_floor: 1.0e-3,
            minimum_dt_myr: 1.0e-8,
        }
    }
}

impl LandscapeParams {
    fn validate(self) -> Result<(), LandscapeError> {
        let positive = [
            ("critical_slope_grade", self.critical_slope_grade),
            ("max_uplift_step_km", self.max_uplift_step_km),
            ("max_incision_courant", self.max_incision_courant),
            ("hillslope_safety", self.hillslope_safety),
            (
                "nonlinear_denominator_floor",
                self.nonlinear_denominator_floor,
            ),
            ("minimum_dt_myr", self.minimum_dt_myr),
        ];
        for (name, value) in positive {
            if !value.is_finite() || value <= 0.0 {
                return Err(LandscapeError::InvalidParameter(name));
            }
        }
        if !self.incision_k.is_finite()
            || self.incision_k < 0.0
            || !self.incision_m.is_finite()
            || self.incision_m < 0.0
            || !self.incision_n.is_finite()
            || self.incision_n <= 0.0
            || !self.hillslope_diffusivity_km2_myr.is_finite()
            || self.hillslope_diffusivity_km2_myr < 0.0
            || !self.runoff_km_myr.is_finite()
            || self.runoff_km_myr < 0.0
            || self.hillslope_safety > 1.0
            || self.nonlinear_denominator_floor > 1.0
        {
            return Err(LandscapeError::InvalidParameter("solver parameter"));
        }
        Ok(())
    }
}

/// Revision-keyed drainage derived from, but never written into, bedrock.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DrainageCache {
    pub terrain_revision: u64,
    pub receiver: Vec<Option<usize>>,
    /// Upstream water supply including the cell's local runoff (km³/Myr).
    pub discharge_km3_myr: Vec<f64>,
    /// Priority-flood routing surface. Diagnostic only; it is not terrain.
    pub routing_elevation_km: Vec<f64>,
    /// Downstream-to-upstream topological order.
    pub downstream_order: Vec<usize>,
    pub outlet_by_cell: Vec<Option<usize>>,
    pub total_runoff_km3_myr: f64,
    pub outlet_discharge_km3_myr: f64,
    pub closed_sink_storage_km3_myr: f64,
}

impl DrainageCache {
    fn empty(cell_count: usize) -> Self {
        Self {
            terrain_revision: u64::MAX,
            receiver: vec![None; cell_count],
            discharge_km3_myr: vec![0.0; cell_count],
            routing_elevation_km: vec![0.0; cell_count],
            downstream_order: Vec::with_capacity(cell_count),
            outlet_by_cell: vec![None; cell_count],
            total_runoff_km3_myr: 0.0,
            outlet_discharge_km3_myr: 0.0,
            closed_sink_storage_km3_myr: 0.0,
        }
    }

    /// Water balance residual for the instantaneous routing graph.
    pub fn water_balance_error_km3_myr(&self) -> f64 {
        self.total_runoff_km3_myr - self.outlet_discharge_km3_myr - self.closed_sink_storage_km3_myr
    }
}

/// Cumulative solid-volume accounting (km³).
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct LandscapeLedger {
    pub initial_bedrock_volume_km3: f64,
    pub rock_uplift_km3: f64,
    pub incision_export_km3: f64,
    pub hillslope_boundary_export_km3: f64,
    /// Signed volume imposed by fixed open base-level cells.
    pub base_level_adjustment_km3: f64,
    pub final_bedrock_volume_km3: f64,
    pub closure_error_km3: f64,
}

impl LandscapeLedger {
    pub fn expected_volume_km3(&self) -> f64 {
        self.initial_bedrock_volume_km3 + self.rock_uplift_km3
            - self.incision_export_km3
            - self.hillslope_boundary_export_km3
            + self.base_level_adjustment_km3
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LandscapeState {
    pub time_myr: f64,
    pub revision: u64,
    pub bedrock_elevation_km: Vec<f64>,
    pub drainage: DrainageCache,
    pub ledger: LandscapeLedger,
}

impl LandscapeState {
    pub fn new(
        mesh: &LandscapeMesh,
        bedrock_elevation_km: Vec<f64>,
    ) -> Result<Self, LandscapeError> {
        validate_mesh_arrays(mesh)?;
        if bedrock_elevation_km.len() != mesh.cell_area_km2.len() {
            return Err(LandscapeError::LengthMismatch {
                field: "bedrock_elevation_km",
                expected: mesh.cell_area_km2.len(),
                actual: bedrock_elevation_km.len(),
            });
        }
        if bedrock_elevation_km.iter().any(|value| !value.is_finite()) {
            return Err(LandscapeError::NonFinite("bedrock_elevation_km"));
        }
        let initial = volume_km3(mesh, &bedrock_elevation_km);
        Ok(Self {
            time_myr: 0.0,
            revision: 0,
            drainage: DrainageCache::empty(bedrock_elevation_km.len()),
            ledger: LandscapeLedger {
                initial_bedrock_volume_km3: initial,
                final_bedrock_volume_km3: initial,
                ..LandscapeLedger::default()
            },
            bedrock_elevation_km,
        })
    }

    pub fn surface_volume_km3(&self, mesh: &LandscapeMesh) -> f64 {
        volume_km3(mesh, &self.bedrock_elevation_km)
    }

    pub fn snapshot(&self) -> LandscapeSnapshot {
        LandscapeSnapshot {
            time_myr: self.time_myr,
            revision: self.revision,
            bedrock_elevation_km: self.bedrock_elevation_km.clone(),
            discharge_km3_myr: self.drainage.discharge_km3_myr.clone(),
            ledger: self.ledger,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LandscapeSnapshot {
    pub time_myr: f64,
    pub revision: u64,
    pub bedrock_elevation_km: Vec<f64>,
    pub discharge_km3_myr: Vec<f64>,
    pub ledger: LandscapeLedger,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TimestepAudit {
    pub requested_dt_myr: f64,
    pub accepted_dt_myr: f64,
    pub uplift_limit_myr: f64,
    pub incision_limit_myr: f64,
    pub hillslope_limit_myr: f64,
    /// Reserved for the explicitly omitted horizontal-advection rung.
    pub advection_limit_myr: f64,
    pub limiting_operator: TimestepLimiter,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TimestepLimiter {
    Requested,
    Uplift,
    Incision,
    Hillslope,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct StepDiagnostics {
    pub time_start_myr: f64,
    pub time_end_myr: f64,
    pub timestep: TimestepAudit,
    pub uplift_km3: f64,
    pub incision_export_km3: f64,
    pub hillslope_boundary_export_km3: f64,
    pub base_level_adjustment_km3: f64,
    pub volume_change_km3: f64,
    pub closure_error_km3: f64,
    pub maximum_discharge_km3_myr: f64,
    pub maximum_slope_ratio: f64,
    pub nonlinear_regularized_faces: usize,
}

#[derive(Debug, Clone)]
pub struct LandscapeSolver {
    pub params: LandscapeParams,
}

impl LandscapeSolver {
    pub fn new(params: LandscapeParams) -> Result<Self, LandscapeError> {
        params.validate()?;
        Ok(Self { params })
    }

    /// Refresh the derived routing graph without advancing or modifying
    /// bedrock. This is useful for the t=0 checkpoint.
    pub fn refresh_drainage(
        &self,
        mesh: &LandscapeMesh,
        state: &mut LandscapeState,
    ) -> Result<(), LandscapeError> {
        validate_mesh_arrays(mesh)?;
        if state.bedrock_elevation_km.len() != mesh.cell_area_km2.len() {
            return Err(LandscapeError::LengthMismatch {
                field: "bedrock_elevation_km",
                expected: mesh.cell_area_km2.len(),
                actual: state.bedrock_elevation_km.len(),
            });
        }
        state.drainage = route_surface(
            mesh,
            &state.bedrock_elevation_km,
            self.params.runoff_km_myr,
            state.revision,
        )?;
        Ok(())
    }

    /// Advance with a time-dependent forcing evaluator. If an operator shortens
    /// the requested step, forcing is re-evaluated at the accepted midpoint so
    /// a ramp is never sampled at the midpoint of a rejected step.
    pub fn step_with_forcing<F>(
        &self,
        mesh: &LandscapeMesh,
        requested_dt_myr: f64,
        state: &mut LandscapeState,
        mut evaluate: F,
    ) -> Result<StepDiagnostics, LandscapeError>
    where
        F: FnMut(f64) -> DeformationFrame,
    {
        if !requested_dt_myr.is_finite() || requested_dt_myr <= 0.0 {
            return Err(LandscapeError::InvalidTimestep);
        }
        let mut candidate = requested_dt_myr;
        let mut adaptive_limiter = TimestepLimiter::Requested;
        for _ in 0..12 {
            let frame = evaluate(state.time_myr + 0.5 * candidate);
            validate_state(mesh, &frame, candidate, state)?;
            let drainage = route_surface(
                mesh,
                &state.bedrock_elevation_km,
                self.params.runoff_km_myr,
                state.revision,
            )?;
            let local_audit = self.timestep_audit(mesh, &frame, candidate, state, &drainage);
            if local_audit.accepted_dt_myr >= candidate * (1.0 - 8.0 * f64::EPSILON) {
                let mut diagnostics = self.step(mesh, &frame, candidate, state)?;
                diagnostics.timestep.requested_dt_myr = requested_dt_myr;
                diagnostics.timestep.accepted_dt_myr = candidate;
                diagnostics.timestep.limiting_operator = adaptive_limiter;
                return Ok(diagnostics);
            }
            adaptive_limiter = local_audit.limiting_operator;
            candidate = local_audit.accepted_dt_myr;
        }
        Err(LandscapeError::TimestepDidNotConverge)
    }

    /// Advance one adaptive operator-split step. The forcing frame is assumed to
    /// have been evaluated at the caller's desired midpoint.
    pub fn step(
        &self,
        mesh: &LandscapeMesh,
        forcing: &DeformationFrame,
        requested_dt_myr: f64,
        state: &mut LandscapeState,
    ) -> Result<StepDiagnostics, LandscapeError> {
        validate_state(mesh, forcing, requested_dt_myr, state)?;

        // Route before selecting the incision accuracy limit. This cache is
        // replaced after uplift as required by the operator contract.
        let pre_route = route_surface(
            mesh,
            &state.bedrock_elevation_km,
            self.params.runoff_km_myr,
            state.revision,
        )?;
        let timestep = self.timestep_audit(mesh, forcing, requested_dt_myr, state, &pre_route);
        if timestep.accepted_dt_myr < self.params.minimum_dt_myr {
            return Err(LandscapeError::StableStepTooSmall {
                stable_dt_myr: timestep.accepted_dt_myr,
                minimum_dt_myr: self.params.minimum_dt_myr,
            });
        }
        let dt = timestep.accepted_dt_myr;
        let volume_before = state.surface_volume_km3(mesh);
        let time_start = state.time_myr;

        let uplift_km3 = apply_uplift(mesh, forcing, dt, &mut state.bedrock_elevation_km);
        state.revision += 1;
        // Open cells are external fixed-level reservoirs, not uplifted routing
        // nodes. Account for pinning before drainage consumes the surface.
        let mut base_level_adjustment_km3 =
            enforce_open_base_level(mesh, &mut state.bedrock_elevation_km);
        state.drainage = route_surface(
            mesh,
            &state.bedrock_elevation_km,
            self.params.runoff_km_myr,
            state.revision,
        )?;

        let incision_export_km3 = apply_implicit_incision(
            mesh,
            &state.drainage,
            self.params,
            dt,
            &mut state.bedrock_elevation_km,
        );
        state.revision += 1;

        let (hillslope_boundary_export_km3, maximum_slope_ratio, regularized) =
            apply_hillslopes(mesh, self.params, dt, &mut state.bedrock_elevation_km);
        state.revision += 1;

        // Hillslope flux treats open cells as reservoirs and should leave this
        // exactly zero; retaining the check protects that boundary invariant.
        base_level_adjustment_km3 += enforce_open_base_level(mesh, &mut state.bedrock_elevation_km);
        state.revision += 1;
        state.time_myr += dt;
        state.drainage = route_surface(
            mesh,
            &state.bedrock_elevation_km,
            self.params.runoff_km_myr,
            state.revision,
        )?;

        let volume_after = state.surface_volume_km3(mesh);
        let volume_change_km3 = volume_after - volume_before;
        let step_expected = uplift_km3 - incision_export_km3 - hillslope_boundary_export_km3
            + base_level_adjustment_km3;
        let closure_error_km3 = volume_change_km3 - step_expected;

        state.ledger.rock_uplift_km3 += uplift_km3;
        state.ledger.incision_export_km3 += incision_export_km3;
        state.ledger.hillslope_boundary_export_km3 += hillslope_boundary_export_km3;
        state.ledger.base_level_adjustment_km3 += base_level_adjustment_km3;
        state.ledger.final_bedrock_volume_km3 = volume_after;
        state.ledger.closure_error_km3 = volume_after - state.ledger.expected_volume_km3();

        Ok(StepDiagnostics {
            time_start_myr: time_start,
            time_end_myr: state.time_myr,
            timestep,
            uplift_km3,
            incision_export_km3,
            hillslope_boundary_export_km3,
            base_level_adjustment_km3,
            volume_change_km3,
            closure_error_km3,
            maximum_discharge_km3_myr: state
                .drainage
                .discharge_km3_myr
                .iter()
                .copied()
                .fold(0.0, f64::max),
            maximum_slope_ratio,
            nonlinear_regularized_faces: regularized,
        })
    }

    fn timestep_audit(
        &self,
        mesh: &LandscapeMesh,
        forcing: &DeformationFrame,
        requested: f64,
        state: &LandscapeState,
        drainage: &DrainageCache,
    ) -> TimestepAudit {
        let max_uplift_rate = forcing
            .rock_vertical_rate_km_myr
            .iter()
            .map(|value| f64::from(*value).abs())
            .fold(0.0, f64::max);
        let uplift_limit = if max_uplift_rate > 0.0 {
            self.params.max_uplift_step_km / max_uplift_rate
        } else {
            f64::INFINITY
        };

        let mut incision_limit = f64::INFINITY;
        for cell in 0..state.bedrock_elevation_km.len() {
            let Some(receiver) = drainage.receiver[cell] else {
                continue;
            };
            let distance = edge_distance(mesh, cell, receiver).unwrap_or(f64::INFINITY);
            let slope = ((state.bedrock_elevation_km[cell] - state.bedrock_elevation_km[receiver])
                / distance)
                .max(0.0);
            let rate = self.params.incision_k
                * drainage.discharge_km3_myr[cell].powf(self.params.incision_m)
                * slope.powf(self.params.incision_n);
            if rate > 0.0 {
                incision_limit = incision_limit
                    .min(self.params.max_incision_courant * distance * slope.max(1.0e-6) / rate);
            }
        }

        let hillslope_limit = hillslope_dt_limit(mesh, &state.bedrock_elevation_km, self.params);
        let candidates = [
            (requested, TimestepLimiter::Requested),
            (uplift_limit, TimestepLimiter::Uplift),
            (incision_limit, TimestepLimiter::Incision),
            (hillslope_limit, TimestepLimiter::Hillslope),
        ];
        let (accepted, limiter) = candidates
            .into_iter()
            .min_by(|a, b| {
                a.0.total_cmp(&b.0)
                    .then_with(|| limiter_rank(a.1).cmp(&limiter_rank(b.1)))
            })
            .unwrap();
        TimestepAudit {
            requested_dt_myr: requested,
            accepted_dt_myr: accepted,
            uplift_limit_myr: uplift_limit,
            incision_limit_myr: incision_limit,
            hillslope_limit_myr: hillslope_limit,
            advection_limit_myr: f64::INFINITY,
            limiting_operator: limiter,
        }
    }
}

fn limiter_rank(limiter: TimestepLimiter) -> u8 {
    match limiter {
        TimestepLimiter::Requested => 0,
        TimestepLimiter::Uplift => 1,
        TimestepLimiter::Incision => 2,
        TimestepLimiter::Hillslope => 3,
    }
}

fn route_surface(
    mesh: &LandscapeMesh,
    elevation: &[f64],
    runoff_km_myr: f64,
    revision: u64,
) -> Result<DrainageCache, LandscapeError> {
    let count = elevation.len();
    let mut cache = DrainageCache::empty(count);
    cache.terrain_revision = revision;
    let mut visited = vec![false; count];
    let mut heap: BinaryHeap<Reverse<(OrderedFloat<f64>, usize)>> = BinaryHeap::new();

    for cell in 0..count {
        if let BoundaryCondition::OpenBaseLevel { elevation_km } = mesh.boundary[cell] {
            let routing_height = elevation[cell].max(f64::from(elevation_km));
            cache.routing_elevation_km[cell] = routing_height;
            cache.outlet_by_cell[cell] = Some(cell);
            visited[cell] = true;
            heap.push(Reverse((OrderedFloat(routing_height), cell)));
        }
    }

    // A fully closed component is allowed: seed its deterministic lowest cell
    // as explicit storage. Normal U/L patches enter through the open outlets.
    if heap.is_empty() && count > 0 {
        let sink = (0..count)
            .min_by(|&a, &b| {
                elevation[a]
                    .total_cmp(&elevation[b])
                    .then_with(|| a.cmp(&b))
            })
            .unwrap();
        cache.routing_elevation_km[sink] = elevation[sink];
        visited[sink] = true;
        heap.push(Reverse((OrderedFloat(elevation[sink]), sink)));
    }

    while let Some(Reverse((OrderedFloat(spill_height), cell))) = heap.pop() {
        cache.downstream_order.push(cell);
        for edge in edge_range(mesh, cell) {
            let neighbor = mesh.edge_neighbor[edge] as usize;
            if visited[neighbor] {
                continue;
            }
            visited[neighbor] = true;
            let routed_height = elevation[neighbor].max(spill_height);
            cache.routing_elevation_km[neighbor] = routed_height;
            cache.receiver[neighbor] = Some(cell);
            cache.outlet_by_cell[neighbor] = cache.outlet_by_cell[cell];
            heap.push(Reverse((OrderedFloat(routed_height), neighbor)));
        }
    }

    if visited.iter().any(|value| !value) {
        return Err(LandscapeError::DisconnectedMesh);
    }

    for cell in 0..count {
        let local = runoff_km_myr * mesh.cell_area_km2[cell];
        cache.discharge_km3_myr[cell] = local;
        cache.total_runoff_km3_myr += local;
    }
    for &cell in cache.downstream_order.iter().rev() {
        if let Some(receiver) = cache.receiver[cell] {
            cache.discharge_km3_myr[receiver] += cache.discharge_km3_myr[cell];
        }
    }
    for &cell in &cache.downstream_order {
        if cache.receiver[cell].is_none() {
            if cache.outlet_by_cell[cell].is_some() {
                cache.outlet_discharge_km3_myr += cache.discharge_km3_myr[cell];
            } else {
                cache.closed_sink_storage_km3_myr += cache.discharge_km3_myr[cell];
            }
        }
    }
    Ok(cache)
}

fn apply_uplift(
    mesh: &LandscapeMesh,
    forcing: &DeformationFrame,
    dt: f64,
    elevation: &mut [f64],
) -> f64 {
    let mut volume = 0.0;
    for ((height, rate), area) in elevation
        .iter_mut()
        .zip(&forcing.rock_vertical_rate_km_myr)
        .zip(&mesh.cell_area_km2)
    {
        let dz = f64::from(*rate) * dt;
        *height += dz;
        volume += dz * area;
    }
    volume
}

fn apply_implicit_incision(
    mesh: &LandscapeMesh,
    drainage: &DrainageCache,
    params: LandscapeParams,
    dt: f64,
    elevation: &mut [f64],
) -> f64 {
    let before = volume_km3(mesh, elevation);
    // Receivers precede donors in this order, so n=1 sees the receiver's new
    // elevation. Other exponents use a stable local backward-Euler solve.
    for &cell in &drainage.downstream_order {
        let Some(receiver) = drainage.receiver[cell] else {
            continue;
        };
        let distance = edge_distance(mesh, cell, receiver).expect("receiver must be a neighbor");
        let receiver_z = elevation[receiver];
        let old_relief = (elevation[cell] - receiver_z).max(0.0);
        if old_relief == 0.0 || params.incision_k == 0.0 {
            continue;
        }
        let coefficient =
            dt * params.incision_k * drainage.discharge_km3_myr[cell].powf(params.incision_m)
                / distance.powf(params.incision_n);
        let new_relief = implicit_relief(old_relief, coefficient, params.incision_n);
        elevation[cell] = receiver_z + new_relief;
    }
    before - volume_km3(mesh, elevation)
}

fn implicit_relief(old: f64, coefficient: f64, exponent: f64) -> f64 {
    if coefficient == 0.0 {
        return old;
    }
    if (exponent - 1.0).abs() < 1.0e-12 {
        return old / (1.0 + coefficient);
    }
    // Solve x + a*x^n = old by deterministic bisection. The residual is
    // monotone for x>=0 and n>0.
    let mut low = 0.0;
    let mut high = old;
    for _ in 0..48 {
        let middle = 0.5 * (low + high);
        if middle + coefficient * middle.powf(exponent) > old {
            high = middle;
        } else {
            low = middle;
        }
    }
    0.5 * (low + high)
}

fn hillslope_dt_limit(mesh: &LandscapeMesh, elevation: &[f64], params: LandscapeParams) -> f64 {
    if params.hillslope_diffusivity_km2_myr == 0.0 {
        return f64::INFINITY;
    }
    let mut conductance_sum = vec![0.0; elevation.len()];
    for cell in 0..elevation.len() {
        for edge in edge_range(mesh, cell) {
            let neighbor = mesh.edge_neighbor[edge] as usize;
            if neighbor <= cell {
                continue;
            }
            let distance = f64::from(mesh.edge_distance_km[edge]);
            let width = f64::from(mesh.edge_face_width_km[edge]);
            let slope_ratio = ((elevation[cell] - elevation[neighbor]).abs()
                / distance
                / params.critical_slope_grade)
                .min(1.0);
            let denominator =
                (1.0 - slope_ratio * slope_ratio).max(params.nonlinear_denominator_floor);
            // Jacobian of q = D*s/(1-(s/Sc)^2). The derivative, rather than
            // the secant conductance, controls explicit stability near Sc.
            let nonlinear_derivative =
                (1.0 + slope_ratio * slope_ratio) / (denominator * denominator);
            let conductance =
                params.hillslope_diffusivity_km2_myr * width / distance * nonlinear_derivative;
            conductance_sum[cell] += conductance;
            conductance_sum[neighbor] += conductance;
        }
    }
    conductance_sum
        .into_iter()
        .enumerate()
        .filter(|(_, sum)| *sum > 0.0)
        .map(|(cell, sum)| params.hillslope_safety * mesh.cell_area_km2[cell] / sum)
        .fold(f64::INFINITY, f64::min)
}

fn apply_hillslopes(
    mesh: &LandscapeMesh,
    params: LandscapeParams,
    dt: f64,
    elevation: &mut [f64],
) -> (f64, f64, usize) {
    if params.hillslope_diffusivity_km2_myr == 0.0 {
        return (0.0, maximum_slope_ratio(mesh, elevation, params), 0);
    }
    let mut volume_delta = vec![0.0; elevation.len()];
    let mut boundary_export = 0.0;
    let mut maximum_ratio: f64 = 0.0;
    let mut regularized = 0;
    for cell in 0..elevation.len() {
        for edge in edge_range(mesh, cell) {
            let neighbor = mesh.edge_neighbor[edge] as usize;
            if neighbor <= cell {
                continue;
            }
            let distance = f64::from(mesh.edge_distance_km[edge]);
            let width = f64::from(mesh.edge_face_width_km[edge]);
            let signed_slope = (elevation[cell] - elevation[neighbor]) / distance;
            let ratio = signed_slope.abs() / params.critical_slope_grade;
            maximum_ratio = maximum_ratio.max(ratio);
            let raw_denominator = 1.0 - ratio * ratio;
            if raw_denominator < params.nonlinear_denominator_floor {
                regularized += 1;
            }
            let denominator = raw_denominator.max(params.nonlinear_denominator_floor);
            let signed_volume =
                params.hillslope_diffusivity_km2_myr * signed_slope * width / denominator * dt;

            let cell_open = matches!(mesh.boundary[cell], BoundaryCondition::OpenBaseLevel { .. });
            let neighbor_open = matches!(
                mesh.boundary[neighbor],
                BoundaryCondition::OpenBaseLevel { .. }
            );
            match (cell_open, neighbor_open) {
                (false, false) => {
                    volume_delta[cell] -= signed_volume;
                    volume_delta[neighbor] += signed_volume;
                }
                (false, true) if signed_volume > 0.0 => {
                    volume_delta[cell] -= signed_volume;
                    boundary_export += signed_volume;
                }
                (true, false) if signed_volume < 0.0 => {
                    volume_delta[neighbor] += signed_volume;
                    boundary_export -= signed_volume;
                }
                _ => {}
            }
        }
    }
    for cell in 0..elevation.len() {
        elevation[cell] += volume_delta[cell] / mesh.cell_area_km2[cell];
    }
    (boundary_export, maximum_ratio, regularized)
}

fn maximum_slope_ratio(mesh: &LandscapeMesh, elevation: &[f64], params: LandscapeParams) -> f64 {
    let mut maximum: f64 = 0.0;
    for cell in 0..elevation.len() {
        for edge in edge_range(mesh, cell) {
            let neighbor = mesh.edge_neighbor[edge] as usize;
            if neighbor > cell {
                let slope = (elevation[cell] - elevation[neighbor]).abs()
                    / f64::from(mesh.edge_distance_km[edge]);
                maximum = maximum.max(slope / params.critical_slope_grade);
            }
        }
    }
    maximum
}

fn enforce_open_base_level(mesh: &LandscapeMesh, elevation: &mut [f64]) -> f64 {
    let mut adjustment = 0.0;
    for (cell, height) in elevation.iter_mut().enumerate() {
        if let BoundaryCondition::OpenBaseLevel { elevation_km } = mesh.boundary[cell] {
            let target = f64::from(elevation_km);
            adjustment += (target - *height) * mesh.cell_area_km2[cell];
            *height = target;
        }
    }
    adjustment
}

fn volume_km3(mesh: &LandscapeMesh, elevation: &[f64]) -> f64 {
    elevation
        .iter()
        .zip(&mesh.cell_area_km2)
        .map(|(height, area)| height * area)
        .sum()
}

fn edge_range(mesh: &LandscapeMesh, cell: usize) -> std::ops::Range<usize> {
    mesh.edge_offsets[cell] as usize..mesh.edge_offsets[cell + 1] as usize
}

fn edge_distance(mesh: &LandscapeMesh, cell: usize, neighbor: usize) -> Option<f64> {
    edge_range(mesh, cell)
        .find(|&edge| mesh.edge_neighbor[edge] as usize == neighbor)
        .map(|edge| f64::from(mesh.edge_distance_km[edge]))
}

fn validate_mesh_arrays(mesh: &LandscapeMesh) -> Result<(), LandscapeError> {
    let count = mesh.cell_area_km2.len();
    if mesh.boundary.len() != count {
        return Err(LandscapeError::LengthMismatch {
            field: "boundary",
            expected: count,
            actual: mesh.boundary.len(),
        });
    }
    if mesh.edge_offsets.len() != count + 1 {
        return Err(LandscapeError::LengthMismatch {
            field: "edge_offsets",
            expected: count + 1,
            actual: mesh.edge_offsets.len(),
        });
    }
    let edges = mesh.edge_neighbor.len();
    if mesh.edge_distance_km.len() != edges || mesh.edge_face_width_km.len() != edges {
        return Err(LandscapeError::LengthMismatch {
            field: "edge geometry",
            expected: edges,
            actual: mesh
                .edge_distance_km
                .len()
                .min(mesh.edge_face_width_km.len()),
        });
    }
    if mesh
        .cell_area_km2
        .iter()
        .any(|area| !area.is_finite() || *area <= 0.0)
        || mesh
            .edge_distance_km
            .iter()
            .any(|distance| !distance.is_finite() || *distance <= 0.0)
        || mesh
            .edge_face_width_km
            .iter()
            .any(|width| !width.is_finite() || *width <= 0.0)
    {
        return Err(LandscapeError::InvalidMeshGeometry);
    }
    Ok(())
}

fn validate_state(
    mesh: &LandscapeMesh,
    forcing: &DeformationFrame,
    requested_dt_myr: f64,
    state: &LandscapeState,
) -> Result<(), LandscapeError> {
    validate_mesh_arrays(mesh)?;
    let count = mesh.cell_area_km2.len();
    if state.bedrock_elevation_km.len() != count {
        return Err(LandscapeError::LengthMismatch {
            field: "bedrock_elevation_km",
            expected: count,
            actual: state.bedrock_elevation_km.len(),
        });
    }
    if forcing.rock_vertical_rate_km_myr.len() != count {
        return Err(LandscapeError::LengthMismatch {
            field: "rock_vertical_rate_km_myr",
            expected: count,
            actual: forcing.rock_vertical_rate_km_myr.len(),
        });
    }
    if !requested_dt_myr.is_finite() || requested_dt_myr <= 0.0 {
        return Err(LandscapeError::InvalidTimestep);
    }
    if !state.time_myr.is_finite()
        || state
            .bedrock_elevation_km
            .iter()
            .any(|value| !value.is_finite())
        || forcing
            .rock_vertical_rate_km_myr
            .iter()
            .any(|value| !value.is_finite())
    {
        return Err(LandscapeError::NonFinite("state or forcing"));
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
pub enum LandscapeError {
    LengthMismatch {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    InvalidParameter(&'static str),
    InvalidMeshGeometry,
    InvalidTimestep,
    NonFinite(&'static str),
    DisconnectedMesh,
    StableStepTooSmall {
        stable_dt_myr: f64,
        minimum_dt_myr: f64,
    },
    TimestepDidNotConverge,
}

impl fmt::Display for LandscapeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch {
                field,
                expected,
                actual,
            } => write!(formatter, "{field} length {actual}, expected {expected}"),
            Self::InvalidParameter(name) => write!(formatter, "invalid landscape parameter: {name}"),
            Self::InvalidMeshGeometry => write!(formatter, "invalid landscape mesh geometry"),
            Self::InvalidTimestep => write!(formatter, "landscape timestep must be finite and positive"),
            Self::NonFinite(field) => write!(formatter, "non-finite value in {field}"),
            Self::DisconnectedMesh => write!(formatter, "landscape mesh has an unreachable component"),
            Self::StableStepTooSmall {
                stable_dt_myr,
                minimum_dt_myr,
            } => write!(
                formatter,
                "stable landscape timestep {stable_dt_myr} Myr is below minimum {minimum_dt_myr} Myr"
            ),
            Self::TimestepDidNotConverge => {
                write!(formatter, "landscape timestep/forcing iteration did not converge")
            }
        }
    }
}

impl std::error::Error for LandscapeError {}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{DVec3, Vec3};

    fn small_mesh() -> LandscapeMesh {
        LandscapeMesh::uniform_planar_hex(48.0, 40.0, 4.0).unwrap()
    }

    /// A deliberately one-dimensional routing graph for testing the stream-
    /// power pathway mathematics independently of raster routing and cell
    /// support. Cell 0 is the fixed outlet; every other cell drains to i - 1.
    fn receiver_chain_mesh(cell_count: usize, spacing_km: f64) -> LandscapeMesh {
        assert!(cell_count >= 2);
        let mut edge_offsets = Vec::with_capacity(cell_count + 1);
        let mut edge_neighbor = Vec::with_capacity(2 * cell_count - 2);
        let mut edge_distance_km = Vec::with_capacity(2 * cell_count - 2);
        let mut edge_face_width_km = Vec::with_capacity(2 * cell_count - 2);
        let mut edge_outward_tangent = Vec::with_capacity(2 * cell_count - 2);
        for cell in 0..cell_count {
            edge_offsets.push(edge_neighbor.len() as u32);
            for neighbor in [
                cell.checked_sub(1),
                (cell + 1 < cell_count).then_some(cell + 1),
            ]
            .into_iter()
            .flatten()
            {
                edge_neighbor.push(neighbor as u32);
                edge_distance_km.push(spacing_km as f32);
                edge_face_width_km.push(1.0);
                edge_outward_tangent.push(Vec3::new(
                    if neighbor < cell { -1.0 } else { 1.0 },
                    0.0,
                    0.0,
                ));
            }
        }
        edge_offsets.push(edge_neighbor.len() as u32);
        let mesh = LandscapeMesh {
            cell_center_km: (0..cell_count)
                .map(|cell| DVec3::new(cell as f64 * spacing_km, 0.0, 0.0))
                .collect(),
            // Arbitrary here: this test treats elevation as a sampled dominant
            // flow-path profile, not as a cell-mean material volume.
            cell_area_km2: vec![1.0; cell_count],
            edge_offsets,
            edge_neighbor,
            edge_distance_km,
            edge_face_width_km,
            edge_outward_tangent,
            boundary: (0..cell_count)
                .map(|cell| {
                    if cell == 0 {
                        BoundaryCondition::OpenBaseLevel { elevation_km: 0.0 }
                    } else {
                        BoundaryCondition::Interior
                    }
                })
                .collect(),
            boundary_faces: Vec::new(),
            outlet_portals: Vec::new(),
        };
        mesh.validate().unwrap();
        mesh
    }

    #[test]
    fn priority_flood_is_derived_acyclic_and_water_balanced() {
        let mesh = small_mesh();
        let terrain: Vec<f64> = mesh
            .cell_center_km
            .iter()
            .enumerate()
            .map(|(cell, point)| {
                // A closed depression plus a deterministic sub-grid-scale tie
                // perturbation exercises filling without mutating terrain.
                0.001 * point.y.abs() - 0.01 * (-(point.length_squared()) / 64.0).exp()
                    + cell as f64 * 1.0e-12
            })
            .collect();
        let original = terrain.clone();
        let cache = route_surface(&mesh, &terrain, 500.0, 7).unwrap();

        assert_eq!(terrain, original);
        assert_eq!(cache.terrain_revision, 7);
        assert_eq!(cache.downstream_order.len(), terrain.len());
        let mut rank = vec![usize::MAX; terrain.len()];
        for (position, &cell) in cache.downstream_order.iter().enumerate() {
            rank[cell] = position;
        }
        for (cell, receiver) in cache.receiver.iter().enumerate() {
            if let Some(receiver) = receiver {
                assert!(rank[*receiver] < rank[cell]);
                assert!(edge_distance(&mesh, cell, *receiver).is_some());
            }
        }
        assert!(cache.water_balance_error_km3_myr().abs() <= cache.total_runoff_km3_myr * 1.0e-12);
    }

    #[test]
    fn nonlinear_hillslopes_are_internally_conservative() {
        let mut mesh = small_mesh();
        mesh.boundary.fill(BoundaryCondition::Closed);
        let mut terrain: Vec<f64> = mesh
            .cell_center_km
            .iter()
            .map(|point| 0.004 * point.x + 0.002 * point.y)
            .collect();
        let params = LandscapeParams {
            hillslope_diffusivity_km2_myr: 0.2,
            ..LandscapeParams::default()
        };
        let before = volume_km3(&mesh, &terrain);
        let stable_dt = hillslope_dt_limit(&mesh, &terrain, params);
        let (export, _, regularized) =
            apply_hillslopes(&mesh, params, stable_dt * 0.5, &mut terrain);
        let after = volume_km3(&mesh, &terrain);

        assert_eq!(export, 0.0);
        assert_eq!(regularized, 0);
        assert!((after - before).abs() <= before.abs().max(1.0) * 1.0e-12);
    }

    #[test]
    fn implicit_stream_power_matches_linear_closed_form() {
        for old in [0.001, 0.2, 3.0] {
            for coefficient in [1.0e-5, 0.25, 4.0] {
                let actual = implicit_relief(old, coefficient, 1.0);
                assert_eq!(actual, old / (1.0 + coefficient));
                assert!(actual >= 0.0 && actual <= old);
            }
        }
    }

    #[test]
    fn fixed_receiver_chain_preserves_analytic_stream_power_profile() {
        let spacing_km = 2.0;
        let mesh = receiver_chain_mesh(6, spacing_km);
        let uplift_km_myr: f64 = 0.01;
        let dt_myr: f64 = 0.2;
        let params = LandscapeParams {
            incision_k: 0.04,
            incision_m: 0.5,
            incision_n: 1.3,
            ..LandscapeParams::default()
        };
        // Discharge is prescribed, rather than accumulated on a raster, so
        // this isolates the pathway law E = K Q^m S^n. Values increase toward
        // the outlet, as they would on a real receiver chain.
        let discharge_km3_myr: Vec<f64> = vec![64.0, 32.0, 16.0, 8.0, 4.0, 2.0];
        let receiver: Vec<Option<usize>> = (0..mesh.cell_count())
            .map(|cell| cell.checked_sub(1))
            .collect();
        let drainage = DrainageCache {
            terrain_revision: 0,
            receiver,
            discharge_km3_myr: discharge_km3_myr.clone(),
            routing_elevation_km: vec![0.0; mesh.cell_count()],
            downstream_order: (0..mesh.cell_count()).collect(),
            outlet_by_cell: vec![Some(0); mesh.cell_count()],
            total_runoff_km3_myr: discharge_km3_myr[0],
            outlet_discharge_km3_myr: discharge_km3_myr[0],
            closed_sink_storage_km3_myr: 0.0,
        };

        // Integrate the analytic steady slopes upstream from the fixed outlet:
        // S = (U / (K Q^m))^(1/n). This is a channel/path elevation profile;
        // neither its support nor its incision is interpreted as cell volume.
        let mut elevation_km = vec![0.0; mesh.cell_count()];
        for cell in 1..mesh.cell_count() {
            let slope = (uplift_km_myr
                / (params.incision_k * discharge_km3_myr[cell].powf(params.incision_m)))
            .powf(1.0 / params.incision_n);
            elevation_km[cell] = elevation_km[cell - 1] + spacing_km * slope;
        }
        let expected_profile = elevation_km.clone();

        // A backward-Euler stream-power step exactly removes one uniform
        // uplift increment from this equilibrium profile. The returned
        // area-weighted change is intentionally ignored: this control proves
        // pathway dynamics only and makes no solid-volume claim.
        for elevation in elevation_km.iter_mut().skip(1) {
            *elevation += uplift_km_myr * dt_myr;
        }
        let _area_weighted_change =
            apply_implicit_incision(&mesh, &drainage, params, dt_myr, &mut elevation_km);

        for (cell, (actual, expected)) in
            elevation_km.iter().zip(expected_profile.iter()).enumerate()
        {
            assert!(
                (actual - expected).abs() <= 2.0e-13,
                "cell {cell}: expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn state_and_snapshot_preserve_f64_material_volume() {
        let mesh = small_mesh();
        let terrain: Vec<f64> = (0..mesh.cell_count())
            .map(|cell| cell as f64 * 1.0e-9)
            .collect();
        let state = LandscapeState::new(&mesh, terrain.clone()).unwrap();
        let snapshot = state.snapshot();
        assert_eq!(snapshot.bedrock_elevation_km, terrain);
        assert_eq!(
            state.ledger.initial_bedrock_volume_km3,
            state.surface_volume_km3(&mesh)
        );
    }

    #[test]
    fn complete_step_is_bit_deterministic_and_closes_ledger() {
        let mesh = small_mesh();
        let frame = DeformationFrame {
            rock_vertical_rate_km_myr: vec![0.0005; mesh.cell_count()],
            horizontal_velocity_km_myr: vec![Vec3::ZERO; mesh.cell_count()],
            dominant_episode: vec![None; mesh.cell_count()],
        };
        let solver = LandscapeSolver::new(LandscapeParams::default()).unwrap();
        let terrain: Vec<f64> = mesh
            .cell_center_km
            .iter()
            .enumerate()
            .map(|(cell, point)| {
                0.0001 * point.y.abs() + ((cell * 37 % 101) as f64 - 50.0) * 1.0e-7
            })
            .collect();
        let mut first = LandscapeState::new(&mesh, terrain.clone()).unwrap();
        let mut second = LandscapeState::new(&mesh, terrain).unwrap();

        let first_diagnostic = solver.step(&mesh, &frame, 0.01, &mut first).unwrap();
        let second_diagnostic = solver.step(&mesh, &frame, 0.01, &mut second).unwrap();

        assert_eq!(first, second);
        assert_eq!(first_diagnostic, second_diagnostic);
        let scale = first.ledger.rock_uplift_km3.abs().max(1.0);
        assert!(first.ledger.closure_error_km3.abs() <= scale * 1.0e-10);
        assert!(
            first.drainage.water_balance_error_km3_myr().abs()
                <= first.drainage.total_runoff_km3_myr * 1.0e-12
        );
    }

    #[test]
    fn adaptive_step_resamples_forcing_at_accepted_midpoint() {
        let mesh = small_mesh();
        let params = LandscapeParams {
            incision_k: 0.0,
            hillslope_diffusivity_km2_myr: 0.0,
            max_uplift_step_km: 0.02,
            ..LandscapeParams::default()
        };
        let solver = LandscapeSolver::new(params).unwrap();
        let mut state = LandscapeState::new(&mesh, vec![0.0; mesh.cell_count()]).unwrap();
        let mut sample_times = Vec::new();
        let diagnostics = solver
            .step_with_forcing(&mesh, 1.0, &mut state, |time| {
                sample_times.push(time);
                let rate = if time > 0.1 { 1.0 } else { 0.5 };
                DeformationFrame {
                    rock_vertical_rate_km_myr: vec![rate; mesh.cell_count()],
                    horizontal_velocity_km_myr: vec![Vec3::ZERO; mesh.cell_count()],
                    dominant_episode: vec![None; mesh.cell_count()],
                }
            })
            .unwrap();

        assert_eq!(diagnostics.timestep.requested_dt_myr, 1.0);
        assert_eq!(diagnostics.timestep.accepted_dt_myr, 0.02);
        assert_eq!(
            diagnostics.timestep.limiting_operator,
            TimestepLimiter::Uplift
        );
        assert_eq!(sample_times, vec![0.5, 0.01]);
        for (cell, boundary) in mesh.boundary.iter().enumerate() {
            if matches!(boundary, BoundaryCondition::OpenBaseLevel { .. }) {
                assert_eq!(state.bedrock_elevation_km[cell], 0.0);
                assert_eq!(state.drainage.routing_elevation_km[cell], 0.0);
            }
        }
    }
}
