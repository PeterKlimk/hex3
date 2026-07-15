use super::LandscapeMesh;
use glam::{DVec2, Vec3};
use serde::{Deserialize, Serialize};
use std::{fmt, ops::Range};

pub const REFERENCE_ROCK_VOLUME_RATE_KM3_MYR: f64 = 17_500.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EpisodeId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SegmentId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SegmentGeometry {
    pub start_km: DVec2,
    pub end_km: DVec2,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Taper {
    Flat,
    CosineEnds { end_fraction: f64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SegmentLinkKind {
    Continuation,
    Transfer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SegmentLink {
    pub other: SegmentId,
    pub kind: SegmentLinkKind,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeformationSegment {
    pub id: SegmentId,
    pub geometry: SegmentGeometry,
    pub width_km: f64,
    pub along_strike_taper: Taper,
    pub vergence: Vec3,
    pub links: Vec<SegmentLink>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SegmentShare {
    pub segment_id: SegmentId,
    pub share: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeformationEpisode {
    pub id: EpisodeId,
    pub active_myr: Range<f64>,
    pub ramp_myr: f64,
    pub rock_volume_rate_km3_myr: f64,
    pub segment_shares: Vec<SegmentShare>,
}

/// Area-normalized segment support, in inverse square kilometres.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SupportStencil {
    pub segment_id: SegmentId,
    pub weight_per_km2: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LandscapeScenario {
    pub id: String,
    pub segments: Vec<DeformationSegment>,
    pub episodes: Vec<DeformationEpisode>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeformationFrame {
    pub rock_vertical_rate_km_myr: Vec<f32>,
    pub horizontal_velocity_km_myr: Vec<Vec3>,
    pub dominant_episode: Vec<Option<EpisodeId>>,
}

#[derive(Debug, Clone)]
pub struct DeformationEvaluator {
    cell_area_km2: Vec<f64>,
    episodes: Vec<DeformationEpisode>,
    stencils: Vec<SupportStencil>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForcingError(pub String);

impl fmt::Display for ForcingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for ForcingError {}

impl LandscapeScenario {
    pub fn compile(&self, mesh: &LandscapeMesh) -> Result<DeformationEvaluator, ForcingError> {
        mesh.validate().map_err(|e| ForcingError(e.to_string()))?;
        let mut stencils = Vec::with_capacity(self.segments.len());
        for segment in &self.segments {
            if !segment.width_km.is_finite() || segment.width_km <= 0.0 {
                return Err(ForcingError(format!(
                    "invalid width for segment {:?}",
                    segment.id
                )));
            }
            let axis = segment.geometry.end_km - segment.geometry.start_km;
            let length = axis.length();
            if !length.is_finite() || length <= 0.0 {
                return Err(ForcingError(format!("degenerate segment {:?}", segment.id)));
            }
            let unit = axis / length;
            let mut weights = Vec::with_capacity(mesh.cell_count());
            for center in &mesh.cell_center_km {
                let p = DVec2::new(center.x, center.y) - segment.geometry.start_km;
                let along = p.dot(unit) / length;
                let cross = (p - unit * p.dot(unit)).length();
                let cross_weight = if cross >= 0.5 * segment.width_km {
                    0.0
                } else {
                    0.5 * (1.0 + (std::f64::consts::PI * cross / (0.5 * segment.width_km)).cos())
                };
                let along_weight = if !(0.0..=1.0).contains(&along) {
                    0.0
                } else {
                    match segment.along_strike_taper {
                        Taper::Flat => 1.0,
                        Taper::CosineEnds { end_fraction } => {
                            let f = end_fraction.clamp(1e-6, 0.5);
                            if along < f {
                                0.5 * (1.0 - (std::f64::consts::PI * along / f).cos())
                            } else if along > 1.0 - f {
                                0.5 * (1.0 - (std::f64::consts::PI * (1.0 - along) / f).cos())
                            } else {
                                1.0
                            }
                        }
                    }
                };
                weights.push(cross_weight * along_weight);
            }
            let integral: f64 = weights
                .iter()
                .zip(&mesh.cell_area_km2)
                .map(|(w, a)| w * a)
                .sum();
            if integral <= 0.0 || !integral.is_finite() {
                return Err(ForcingError(format!(
                    "segment {:?} has empty mesh support",
                    segment.id
                )));
            }
            for weight in &mut weights {
                *weight /= integral;
            }
            stencils.push(SupportStencil {
                segment_id: segment.id,
                weight_per_km2: weights,
            });
        }
        for episode in &self.episodes {
            let sum: f64 = episode.segment_shares.iter().map(|s| s.share).sum();
            if (sum - 1.0).abs() > 1e-9 || episode.active_myr.start >= episode.active_myr.end {
                return Err(ForcingError(format!("invalid episode {:?}", episode.id)));
            }
            if episode
                .segment_shares
                .iter()
                .any(|s| !stencils.iter().any(|x| x.segment_id == s.segment_id))
            {
                return Err(ForcingError(format!(
                    "episode {:?} references missing segment",
                    episode.id
                )));
            }
        }
        Ok(DeformationEvaluator {
            cell_area_km2: mesh.cell_area_km2.clone(),
            episodes: self.episodes.clone(),
            stencils,
        })
    }
}

impl DeformationEvaluator {
    /// Read-only access to the exact area-normalized support compiled from the
    /// scenario. This is exposed for immutable research-input manifests; callers
    /// cannot replace the compiler-owned values.
    pub fn support_stencils(&self) -> &[SupportStencil] {
        &self.stencils
    }

    pub fn evaluate(&self, time_myr: f64) -> DeformationFrame {
        let n = self.cell_area_km2.len();
        let mut rate = vec![0.0_f64; n];
        let mut dominant = vec![None; n];
        let mut dominant_rate = vec![0.0_f64; n];
        for episode in &self.episodes {
            let activity = episode_activity(episode, time_myr);
            if activity == 0.0 {
                continue;
            }
            for share in &episode.segment_shares {
                let stencil = self
                    .stencils
                    .iter()
                    .find(|s| s.segment_id == share.segment_id)
                    .expect("compiled segment");
                let scale = activity * episode.rock_volume_rate_km3_myr * share.share;
                for i in 0..n {
                    let contribution = scale * stencil.weight_per_km2[i];
                    rate[i] += contribution;
                    if contribution.abs() > dominant_rate[i] {
                        dominant_rate[i] = contribution.abs();
                        dominant[i] = Some(episode.id);
                    }
                }
            }
        }
        DeformationFrame {
            rock_vertical_rate_km_myr: rate.into_iter().map(|v| v as f32).collect(),
            horizontal_velocity_km_myr: vec![Vec3::ZERO; n],
            dominant_episode: dominant,
        }
    }

    pub fn integrated_rate_km3_myr(&self, frame: &DeformationFrame) -> f64 {
        frame
            .rock_vertical_rate_km_myr
            .iter()
            .zip(&self.cell_area_km2)
            .map(|(u, a)| *u as f64 * a)
            .sum()
    }
}

fn episode_activity(episode: &DeformationEpisode, time: f64) -> f64 {
    if time < episode.active_myr.start || time > episode.active_myr.end {
        return 0.0;
    }
    let ramp = episode
        .ramp_myr
        .max(0.0)
        .min(0.5 * (episode.active_myr.end - episode.active_myr.start));
    if ramp == 0.0 {
        return 1.0;
    }
    let edge = ((time - episode.active_myr.start) / ramp)
        .min((episode.active_myr.end - time) / ramp)
        .clamp(0.0, 1.0);
    edge * edge * (3.0 - 2.0 * edge)
}

pub fn uniform_scenario() -> LandscapeScenario {
    LandscapeScenario {
        id: "U".into(),
        segments: vec![DeformationSegment {
            id: SegmentId(0),
            geometry: SegmentGeometry {
                start_km: DVec2::new(-350.0, -43.0),
                end_km: DVec2::new(350.0, 43.0),
            },
            width_km: 100.0,
            along_strike_taper: Taper::CosineEnds { end_fraction: 0.12 },
            vergence: Vec3::Y,
            links: vec![],
        }],
        episodes: vec![DeformationEpisode {
            id: EpisodeId(0),
            active_myr: 0.0..6.0,
            ramp_myr: 0.25,
            rock_volume_rate_km3_myr: REFERENCE_ROCK_VOLUME_RATE_KM3_MYR,
            segment_shares: vec![SegmentShare {
                segment_id: SegmentId(0),
                share: 1.0,
            }],
        }],
    }
}

pub fn linked_scenario() -> LandscapeScenario {
    let link = |other| {
        vec![SegmentLink {
            other,
            kind: SegmentLinkKind::Transfer,
        }]
    };
    LandscapeScenario {
        id: "L".into(),
        segments: vec![
            DeformationSegment {
                id: SegmentId(0),
                geometry: SegmentGeometry {
                    start_km: DVec2::new(-360.0, -65.0),
                    end_km: DVec2::new(-10.0, -22.0),
                },
                width_km: 100.0,
                along_strike_taper: Taper::CosineEnds { end_fraction: 0.22 },
                vergence: Vec3::Y,
                links: link(SegmentId(1)),
            },
            DeformationSegment {
                id: SegmentId(1),
                geometry: SegmentGeometry {
                    start_km: DVec2::new(-70.0, 22.0),
                    end_km: DVec2::new(280.0, 65.0),
                },
                width_km: 100.0,
                along_strike_taper: Taper::CosineEnds { end_fraction: 0.22 },
                vergence: Vec3::Y,
                links: link(SegmentId(0)),
            },
        ],
        episodes: vec![DeformationEpisode {
            id: EpisodeId(0),
            active_myr: 0.0..6.0,
            ramp_myr: 0.25,
            rock_volume_rate_km3_myr: REFERENCE_ROCK_VOLUME_RATE_KM3_MYR,
            segment_shares: vec![
                SegmentShare {
                    segment_id: SegmentId(0),
                    share: 0.5,
                },
                SegmentShare {
                    segment_id: SegmentId(1),
                    share: 0.5,
                },
            ],
        }],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn u_and_l_are_area_normalized_to_same_work_rate() {
        for spacing in [8.0, 4.0] {
            let mesh = LandscapeMesh::uniform_planar_hex(960.0, 640.0, spacing).unwrap();
            for scenario in [uniform_scenario(), linked_scenario()] {
                let evaluator = scenario.compile(&mesh).unwrap();
                let frame = evaluator.evaluate(3.0);
                let actual = evaluator.integrated_rate_km3_myr(&frame);
                assert!((actual / REFERENCE_ROCK_VOLUME_RATE_KM3_MYR - 1.0).abs() < 2e-7);
                assert!(frame
                    .horizontal_velocity_km_myr
                    .iter()
                    .all(|v| *v == Vec3::ZERO));
            }
        }
    }

    #[test]
    fn forcing_is_inactive_after_episode() {
        let mesh = LandscapeMesh::uniform_planar_hex(200.0, 160.0, 8.0).unwrap();
        let frame = uniform_scenario().compile(&mesh).unwrap().evaluate(7.0);
        assert!(frame.rock_vertical_rate_km_myr.iter().all(|u| *u == 0.0));
    }
}
