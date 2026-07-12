//! Manufactured minimum dual channel/interfluve (`C1`) representation.
//!
//! This module only owns local geometry and elevation-volume bookkeeping. It
//! deliberately does not infer a network, route water, evolve channel width,
//! or represent sediment. Cell area, routed reach length, and active-channel
//! width are prescribed physical geometry.

use std::fmt;

use serde::{Deserialize, Serialize};

/// Prescribed physical geometry for one dual-representation cell.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct C1CellGeometry {
    pub cell_area_km2: f64,
    pub reach_length_km: f64,
    pub channel_width_km: f64,
}

impl C1CellGeometry {
    /// Validate this geometry as an isolated cell.
    pub fn validate(self) -> Result<(), C1FixtureError> {
        self.validate_at(0)
    }

    /// Occupied active-channel area, `w L` (km²).
    pub fn channel_area_km2(self) -> f64 {
        self.channel_width_km * self.reach_length_km
    }

    /// Physical active-channel fraction, `w L / A`.
    pub fn channel_fraction(self) -> f64 {
        self.channel_area_km2() / self.cell_area_km2
    }

    fn validate_at(self, cell: usize) -> Result<(), C1FixtureError> {
        for (field, value) in [
            ("cell_area_km2", self.cell_area_km2),
            ("reach_length_km", self.reach_length_km),
            ("channel_width_km", self.channel_width_km),
        ] {
            if !value.is_finite() {
                return Err(C1FixtureError::NonFiniteGeometry { field, cell });
            }
        }
        if self.cell_area_km2 <= 0.0 {
            return Err(C1FixtureError::NonPositiveCellArea { cell });
        }
        if self.reach_length_km < 0.0 {
            return Err(C1FixtureError::NegativeGeometry {
                field: "reach_length_km",
                cell,
            });
        }
        if self.channel_width_km < 0.0 {
            return Err(C1FixtureError::NegativeGeometry {
                field: "channel_width_km",
                cell,
            });
        }
        let channel_area_km2 = self.channel_area_km2();
        if !channel_area_km2.is_finite() {
            return Err(C1FixtureError::NonFiniteChannelArea { cell });
        }
        if channel_area_km2 >= self.cell_area_km2 {
            return Err(C1FixtureError::ChannelNotSubgrid { cell });
        }
        Ok(())
    }
}

/// Minimum dynamic C1 state for one cell.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct C1CellState {
    /// Authoritative cell-mean surface elevation (km).
    pub mean_elevation_km: f64,
    /// Active-channel surface elevation (km).
    pub channel_surface_elevation_km: f64,
}

impl C1CellState {
    /// Reconstruct the interfluve mean from the exact compartment mixture.
    ///
    /// With no occupied channel area this returns the authoritative cell mean;
    /// the otherwise-unused channel-surface state has no effect.
    pub fn interfluve_mean_elevation_km(
        self,
        geometry: C1CellGeometry,
    ) -> Result<f64, C1FixtureError> {
        geometry.validate()?;
        self.validate(0)?;
        let channel_fraction = geometry.channel_fraction();
        if channel_fraction == 0.0 {
            return Ok(self.mean_elevation_km);
        }
        let interfluve = (self.mean_elevation_km
            - channel_fraction * self.channel_surface_elevation_km)
            / (1.0 - channel_fraction);
        if !interfluve.is_finite() {
            return Err(C1FixtureError::NonFiniteReconstruction { cell: 0 });
        }
        Ok(interfluve)
    }

    fn validate(self, cell: usize) -> Result<(), C1FixtureError> {
        for (field, value) in [
            ("mean_elevation_km", self.mean_elevation_km),
            (
                "channel_surface_elevation_km",
                self.channel_surface_elevation_km,
            ),
        ] {
            if !value.is_finite() {
                return Err(C1FixtureError::NonFiniteState { field, cell });
            }
        }
        Ok(())
    }
}

/// Exact elevation-volume-moment accounting for channel-only excavation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct C1ExcavationLedger {
    pub initial_elevation_volume_moment_km3: f64,
    /// Signed moment change `sum(A_c dz_c)`; non-positive for excavation.
    pub channel_elevation_volume_moment_change_km3: f64,
    pub exported_solid_volume_km3: f64,
    pub final_elevation_volume_moment_km3: f64,
    pub closure_error_km3: f64,
}

/// Cancelling compartment accounting for an internal volume transfer.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct C1InternalTransferLedger {
    pub interfluve_compartment_moment_change_km3: f64,
    pub channel_compartment_moment_change_km3: f64,
    pub net_elevation_volume_moment_change_km3: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum C1FixtureError {
    LengthMismatch {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    NonFiniteGeometry {
        field: &'static str,
        cell: usize,
    },
    NegativeGeometry {
        field: &'static str,
        cell: usize,
    },
    NonPositiveCellArea {
        cell: usize,
    },
    NonFiniteChannelArea {
        cell: usize,
    },
    /// `A_c >= A`: the channel is not unresolved at this geometry.
    ChannelNotSubgrid {
        cell: usize,
    },
    NonFiniteState {
        field: &'static str,
        cell: usize,
    },
    NonFiniteReconstruction {
        cell: usize,
    },
    InvalidChannelSurfaceChange {
        cell: usize,
    },
    InvalidTransferVolume {
        cell: usize,
    },
    /// A positive compartment transfer cannot enter zero occupied area.
    TransferWithoutChannelArea {
        cell: usize,
    },
    NonFiniteUpdate {
        cell: usize,
    },
    NonFiniteLedger,
}

impl fmt::Display for C1FixtureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch {
                field,
                expected,
                actual,
            } => write!(f, "{field} has length {actual}, expected {expected}"),
            Self::NonFiniteGeometry { field, cell } => {
                write!(f, "{field} is non-finite at cell {cell}")
            }
            Self::NegativeGeometry { field, cell } => {
                write!(f, "{field} is negative at cell {cell}")
            }
            Self::NonPositiveCellArea { cell } => {
                write!(f, "cell_area_km2 is not positive at cell {cell}")
            }
            Self::NonFiniteChannelArea { cell } => {
                write!(f, "w L is non-finite at cell {cell}")
            }
            Self::ChannelNotSubgrid { cell } => {
                write!(
                    f,
                    "channel area must be smaller than cell area at cell {cell}"
                )
            }
            Self::NonFiniteState { field, cell } => {
                write!(f, "{field} is non-finite at cell {cell}")
            }
            Self::NonFiniteReconstruction { cell } => {
                write!(f, "interfluve reconstruction is non-finite at cell {cell}")
            }
            Self::InvalidChannelSurfaceChange { cell } => write!(
                f,
                "channel-surface change must be finite and non-positive at cell {cell}"
            ),
            Self::InvalidTransferVolume { cell } => {
                write!(
                    f,
                    "internal transfer must be finite and non-negative at cell {cell}"
                )
            }
            Self::TransferWithoutChannelArea { cell } => {
                write!(
                    f,
                    "positive internal transfer has no channel area at cell {cell}"
                )
            }
            Self::NonFiniteUpdate { cell } => write!(f, "C1 update is non-finite at cell {cell}"),
            Self::NonFiniteLedger => f.write_str("C1 ledger is non-finite"),
        }
    }
}

impl std::error::Error for C1FixtureError {}

/// Apply prescribed signed channel-surface changes (`dz_c <= 0`).
///
/// The interfluve compartment is invariant. All cells and changes are
/// validated and candidate states are computed before any state is mutated.
/// With `w L = 0`, the operator is inert even when a valid lowering is
/// prescribed.
pub fn apply_channel_only_excavation(
    geometry: &[C1CellGeometry],
    state: &mut [C1CellState],
    channel_surface_change_km: &[f64],
) -> Result<C1ExcavationLedger, C1FixtureError> {
    validate_lengths(
        geometry,
        state,
        "channel_surface_change_km",
        channel_surface_change_km.len(),
    )?;

    let mut candidate = Vec::with_capacity(state.len());
    let mut initial_moment = 0.0;
    let mut channel_moment_change = 0.0;
    for cell in 0..state.len() {
        geometry[cell].validate_at(cell)?;
        state[cell].validate(cell)?;
        let dz_c = channel_surface_change_km[cell];
        if !dz_c.is_finite() || dz_c > 0.0 {
            return Err(C1FixtureError::InvalidChannelSurfaceChange { cell });
        }

        let area = geometry[cell].cell_area_km2;
        let channel_area = geometry[cell].channel_area_km2();
        initial_moment += area * state[cell].mean_elevation_km;
        if channel_area == 0.0 {
            candidate.push(state[cell]);
            continue;
        }

        let channel_fraction = channel_area / area;
        let updated = C1CellState {
            mean_elevation_km: state[cell].mean_elevation_km + channel_fraction * dz_c,
            channel_surface_elevation_km: state[cell].channel_surface_elevation_km + dz_c,
        };
        if !updated.mean_elevation_km.is_finite()
            || !updated.channel_surface_elevation_km.is_finite()
        {
            return Err(C1FixtureError::NonFiniteUpdate { cell });
        }
        channel_moment_change += channel_area * dz_c;
        candidate.push(updated);
    }

    let final_moment = geometry
        .iter()
        .zip(&candidate)
        .map(|(geometry, state)| geometry.cell_area_km2 * state.mean_elevation_km)
        .sum::<f64>();
    let exported = if channel_moment_change == 0.0 {
        0.0
    } else {
        -channel_moment_change
    };
    let closure = final_moment - (initial_moment - exported);
    if [
        initial_moment,
        channel_moment_change,
        exported,
        final_moment,
        closure,
    ]
    .iter()
    .any(|value| !value.is_finite())
    {
        return Err(C1FixtureError::NonFiniteLedger);
    }

    state.copy_from_slice(&candidate);
    Ok(C1ExcavationLedger {
        initial_elevation_volume_moment_km3: initial_moment,
        channel_elevation_volume_moment_change_km3: channel_moment_change,
        exported_solid_volume_km3: exported,
        final_elevation_volume_moment_km3: final_moment,
        closure_error_km3: closure,
    })
}

/// Apply prescribed internal interfluve-to-channel volume transfers.
///
/// Positive volume lowers the reconstructed interfluve mean by
/// `V_t/(A-A_c)` and raises the channel surface by `V_t/A_c`. The authoritative
/// mean is copied bit-for-bit and the two compartment ledgers cancel exactly.
/// This proves algebraic bookkeeping only: without sediment or material-
/// availability state, the fixture cannot decide whether an otherwise finite
/// prescribed transfer is physically admissible.
pub fn apply_internal_interfluve_channel_transfer(
    geometry: &[C1CellGeometry],
    state: &mut [C1CellState],
    transfer_volume_km3: &[f64],
) -> Result<C1InternalTransferLedger, C1FixtureError> {
    validate_lengths(
        geometry,
        state,
        "transfer_volume_km3",
        transfer_volume_km3.len(),
    )?;

    let mut candidate = Vec::with_capacity(state.len());
    let mut transferred = 0.0;
    for cell in 0..state.len() {
        geometry[cell].validate_at(cell)?;
        state[cell].validate(cell)?;
        let volume = transfer_volume_km3[cell];
        if !volume.is_finite() || volume < 0.0 {
            return Err(C1FixtureError::InvalidTransferVolume { cell });
        }
        let channel_area = geometry[cell].channel_area_km2();
        if volume == 0.0 {
            candidate.push(state[cell]);
            continue;
        }
        if channel_area == 0.0 {
            return Err(C1FixtureError::TransferWithoutChannelArea { cell });
        }

        let updated_channel_surface =
            state[cell].channel_surface_elevation_km + volume / channel_area;
        let interfluve_area = geometry[cell].cell_area_km2 - channel_area;
        let old_interfluve = reconstruct_interfluve(state[cell], geometry[cell], cell)?;
        let updated_interfluve = old_interfluve - volume / interfluve_area;
        // Reconstructing the mixture here is an independent finite check. The
        // authoritative mean itself remains the original bits by contract.
        let reconstructed_mean = (channel_area * updated_channel_surface
            + interfluve_area * updated_interfluve)
            / geometry[cell].cell_area_km2;
        if !updated_channel_surface.is_finite()
            || !updated_interfluve.is_finite()
            || !reconstructed_mean.is_finite()
        {
            return Err(C1FixtureError::NonFiniteUpdate { cell });
        }
        candidate.push(C1CellState {
            mean_elevation_km: state[cell].mean_elevation_km,
            channel_surface_elevation_km: updated_channel_surface,
        });
        transferred += volume;
    }
    if !transferred.is_finite() {
        return Err(C1FixtureError::NonFiniteLedger);
    }

    state.copy_from_slice(&candidate);
    let interfluve_change = -transferred;
    let channel_change = transferred;
    Ok(C1InternalTransferLedger {
        interfluve_compartment_moment_change_km3: interfluve_change,
        channel_compartment_moment_change_km3: channel_change,
        net_elevation_volume_moment_change_km3: interfluve_change + channel_change,
    })
}

fn reconstruct_interfluve(
    state: C1CellState,
    geometry: C1CellGeometry,
    cell: usize,
) -> Result<f64, C1FixtureError> {
    let fraction = geometry.channel_fraction();
    if fraction == 0.0 {
        return Ok(state.mean_elevation_km);
    }
    let value = (state.mean_elevation_km - fraction * state.channel_surface_elevation_km)
        / (1.0 - fraction);
    if !value.is_finite() {
        return Err(C1FixtureError::NonFiniteReconstruction { cell });
    }
    Ok(value)
}

fn validate_lengths(
    geometry: &[C1CellGeometry],
    state: &[C1CellState],
    field: &'static str,
    field_len: usize,
) -> Result<(), C1FixtureError> {
    if state.len() != geometry.len() {
        return Err(C1FixtureError::LengthMismatch {
            field: "state",
            expected: geometry.len(),
            actual: state.len(),
        });
    }
    if field_len != geometry.len() {
        return Err(C1FixtureError::LengthMismatch {
            field,
            expected: geometry.len(),
            actual: field_len,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const REACH_KM: f64 = 128.0;
    const WIDTH_KM: f64 = 0.2;
    const SWATH_WIDTH_KM: f64 = 16.0;
    const SPACINGS_KM: [f64; 3] = [8.0, 4.0, 2.0];

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual={actual:.17e}, expected={expected:.17e}, tolerance={tolerance:.3e}"
        );
    }

    fn straight_fixture(spacing_km: f64) -> (Vec<C1CellGeometry>, Vec<C1CellState>) {
        let count = (REACH_KM / spacing_km) as usize;
        let geometry = (0..count)
            .map(|_| C1CellGeometry {
                cell_area_km2: SWATH_WIDTH_KM * spacing_km,
                reach_length_km: spacing_km,
                channel_width_km: WIDTH_KM,
            })
            .collect::<Vec<_>>();
        let fraction = WIDTH_KM / SWATH_WIDTH_KM;
        let channel = 0.6;
        let interfluve = 1.2;
        let mean = fraction * channel + (1.0 - fraction) * interfluve;
        let state = vec![
            C1CellState {
                mean_elevation_km: mean,
                channel_surface_elevation_km: channel,
            };
            count
        ];
        (geometry, state)
    }

    #[test]
    fn registered_straight_reach_geometry_is_resolution_invariant() {
        let expected_channel_area = WIDTH_KM * REACH_KM;
        for spacing in SPACINGS_KM {
            let (geometry, _) = straight_fixture(spacing);
            let length: f64 = geometry.iter().map(|cell| cell.reach_length_km).sum();
            let area: f64 = geometry.iter().map(|cell| cell.channel_area_km2()).sum();
            assert_close(length, REACH_KM, 1.0e-14);
            assert_close(area, expected_channel_area, 4.0e-14);
        }
    }

    #[test]
    fn excavation_history_has_invariant_bed_export_and_volume_closure() {
        let history = [-0.015, -0.025, -0.01, -0.05];
        let expected_export = WIDTH_KM * REACH_KM * 0.1;
        for spacing in SPACINGS_KM {
            let (geometry, mut state) = straight_fixture(spacing);
            let initial_interfluve = state[0].interfluve_mean_elevation_km(geometry[0]).unwrap();
            let initial_moment: f64 = geometry
                .iter()
                .zip(&state)
                .map(|(g, s)| g.cell_area_km2 * s.mean_elevation_km)
                .sum();
            let mut export = 0.0;
            let mut expected_channel = 0.6;
            for dz in history {
                let ledger =
                    apply_channel_only_excavation(&geometry, &mut state, &vec![dz; geometry.len()])
                        .unwrap();
                export += ledger.exported_solid_volume_km3;
                expected_channel += dz;
                for cell_state in &state {
                    assert_close(
                        cell_state.channel_surface_elevation_km,
                        expected_channel,
                        1.0e-15,
                    );
                }
                // Roughly 2,400 km³ is accumulated over as many as 64 cells;
                // this is a few ulps of that integral, not a tuned residual.
                assert_close(ledger.closure_error_km3, 0.0, 1.0e-11);
            }
            assert_close(export, expected_export, 2.0e-14);
            for cell in 0..state.len() {
                assert_close(state[cell].channel_surface_elevation_km, 0.5, 1.0e-15);
                assert_close(
                    state[cell]
                        .interfluve_mean_elevation_km(geometry[cell])
                        .unwrap(),
                    initial_interfluve,
                    5.0e-16,
                );
            }
            let final_moment: f64 = geometry
                .iter()
                .zip(&state)
                .map(|(g, s)| g.cell_area_km2 * s.mean_elevation_km)
                .sum();
            assert_close(final_moment, initial_moment - export, 1.0e-11);
        }
    }

    #[test]
    fn internal_transfer_cancels_and_preserves_authoritative_mean_bits() {
        for spacing in SPACINGS_KM {
            let (geometry, mut state) = straight_fixture(spacing);
            let before = state.clone();
            let old_interfluve = state[0].interfluve_mean_elevation_km(geometry[0]).unwrap();
            let per_cell_volume = 0.002;
            let ledger = apply_internal_interfluve_channel_transfer(
                &geometry,
                &mut state,
                &vec![per_cell_volume; geometry.len()],
            )
            .unwrap();
            assert_eq!(ledger.net_elevation_volume_moment_change_km3.to_bits(), 0);
            assert_eq!(
                ledger.interfluve_compartment_moment_change_km3,
                -ledger.channel_compartment_moment_change_km3
            );
            for cell in 0..state.len() {
                assert_eq!(
                    state[cell].mean_elevation_km.to_bits(),
                    before[cell].mean_elevation_km.to_bits()
                );
                assert_close(
                    state[cell].channel_surface_elevation_km
                        - before[cell].channel_surface_elevation_km,
                    per_cell_volume / geometry[cell].channel_area_km2(),
                    1.0e-15,
                );
                let new_interfluve = state[cell]
                    .interfluve_mean_elevation_km(geometry[cell])
                    .unwrap();
                assert_close(
                    new_interfluve - old_interfluve,
                    -per_cell_volume
                        / (geometry[cell].cell_area_km2 - geometry[cell].channel_area_km2()),
                    2.0e-16,
                );
            }
        }
    }

    #[test]
    fn zero_width_is_exact_c0_reduction() {
        let geometry = vec![C1CellGeometry {
            cell_area_km2: 4.0,
            reach_length_km: 2.0,
            channel_width_km: 0.0,
        }];
        let mut state = vec![C1CellState {
            mean_elevation_km: 1.25,
            channel_surface_elevation_km: -70.0,
        }];
        let before = state.clone();
        assert_eq!(
            state[0]
                .interfluve_mean_elevation_km(geometry[0])
                .unwrap()
                .to_bits(),
            state[0].mean_elevation_km.to_bits()
        );
        let ledger = apply_channel_only_excavation(&geometry, &mut state, &[-2.0]).unwrap();
        assert_eq!(state, before);
        assert_eq!(ledger.exported_solid_volume_km3.to_bits(), 0);
        assert_eq!(ledger.closure_error_km3.to_bits(), 0);

        let transfer =
            apply_internal_interfluve_channel_transfer(&geometry, &mut state, &[0.0]).unwrap();
        assert_eq!(state, before);
        assert_eq!(transfer, C1InternalTransferLedger::default());
    }

    #[test]
    fn invalid_inputs_fail_before_any_mutation() {
        let valid_geometry = C1CellGeometry {
            cell_area_km2: 8.0,
            reach_length_km: 2.0,
            channel_width_km: 0.25,
        };
        let mut state = vec![
            C1CellState {
                mean_elevation_km: 1.0,
                channel_surface_elevation_km: 0.5,
            },
            C1CellState {
                mean_elevation_km: 1.1,
                channel_surface_elevation_km: 0.6,
            },
        ];
        let before = state.clone();
        let geometry = vec![
            valid_geometry,
            C1CellGeometry {
                cell_area_km2: 0.5,
                ..valid_geometry
            },
        ];
        assert_eq!(
            apply_channel_only_excavation(&geometry, &mut state, &[-0.1, -0.1]),
            Err(C1FixtureError::ChannelNotSubgrid { cell: 1 })
        );
        assert_eq!(state, before);

        let geometry = vec![valid_geometry; 2];
        assert_eq!(
            apply_channel_only_excavation(&geometry, &mut state, &[-0.1, f64::NAN]),
            Err(C1FixtureError::InvalidChannelSurfaceChange { cell: 1 })
        );
        assert_eq!(state, before);
        assert_eq!(
            apply_internal_interfluve_channel_transfer(&geometry, &mut state, &[0.01, -0.1]),
            Err(C1FixtureError::InvalidTransferVolume { cell: 1 })
        );
        assert_eq!(state, before);

        let no_channel = vec![
            C1CellGeometry {
                channel_width_km: 0.0,
                ..valid_geometry
            };
            2
        ];
        assert_eq!(
            apply_internal_interfluve_channel_transfer(&no_channel, &mut state, &[0.0, 0.01]),
            Err(C1FixtureError::TransferWithoutChannelArea { cell: 1 })
        );
        assert_eq!(state, before);
    }

    #[test]
    fn nonfinite_geometry_and_state_are_rejected_transactionally() {
        let geometry = vec![C1CellGeometry {
            cell_area_km2: 10.0,
            reach_length_km: f64::INFINITY,
            channel_width_km: 0.1,
        }];
        let mut state = vec![C1CellState {
            mean_elevation_km: 1.0,
            channel_surface_elevation_km: 0.5,
        }];
        let before = state.clone();
        assert_eq!(
            apply_channel_only_excavation(&geometry, &mut state, &[-0.1]),
            Err(C1FixtureError::NonFiniteGeometry {
                field: "reach_length_km",
                cell: 0
            })
        );
        assert_eq!(state, before);

        let geometry = vec![C1CellGeometry {
            cell_area_km2: 10.0,
            reach_length_km: 1.0,
            channel_width_km: 0.1,
        }];
        state[0].mean_elevation_km = f64::NAN;
        let before = state.clone();
        assert_eq!(
            apply_channel_only_excavation(&geometry, &mut state, &[-0.1]),
            Err(C1FixtureError::NonFiniteState {
                field: "mean_elevation_km",
                cell: 0
            })
        );
        assert_eq!(
            state[0].mean_elevation_km.to_bits(),
            before[0].mean_elevation_km.to_bits()
        );
    }

    #[test]
    fn repeated_runs_are_bitwise_deterministic() {
        let (geometry, initial) = straight_fixture(2.0);
        let run = || {
            let mut state = initial.clone();
            let excavation =
                apply_channel_only_excavation(&geometry, &mut state, &vec![-0.075; geometry.len()])
                    .unwrap();
            let transfer = apply_internal_interfluve_channel_transfer(
                &geometry,
                &mut state,
                &vec![0.001; geometry.len()],
            )
            .unwrap();
            (state, excavation, transfer)
        };
        assert_eq!(run(), run());
    }
}
