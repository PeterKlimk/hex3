use serde::Serialize;

use super::{
    ErosionParams, FineCacheMode, FineDensityParams, FineStructureParams, OrogenModel,
    TectonicCarrierConfig, VoronoiBackend, World, ELEVATION_UNIT_KM, PHYSICAL_RELIEF_SCALE,
    PLANET_RADIUS_KM,
};

#[derive(Debug, Clone, Serialize)]
pub struct BuildProvenance {
    pub git_revision: &'static str,
    pub git_dirty: bool,
}

impl BuildProvenance {
    pub fn current() -> Self {
        Self {
            git_revision: env!("HEX3_GIT_REVISION"),
            git_dirty: env!("HEX3_GIT_DIRTY") == "true",
        }
    }

    pub fn label(&self) -> String {
        format!(
            "{}{}",
            self.git_revision,
            if self.git_dirty { "+dirty" } else { "" }
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum FineCacheOutcome {
    DisabledGenerated,
    Hit,
    MissGenerated,
    Rebuilt,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FineCacheRecord {
    pub mode: FineCacheMode,
    pub version: u32,
    pub key_hex: String,
    pub outcome: FineCacheOutcome,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub write_succeeded: Option<bool>,
    pub max_cells: usize,
    pub actual_cells: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct RunManifest {
    pub build: BuildProvenance,
    pub units: UnitManifest,
    pub seed: u64,
    pub voronoi_backend: VoronoiBackend,
    pub lloyd_iterations: usize,
    pub coarse_cells: usize,
    pub plate_count: Option<usize>,
    pub orogen_model: OrogenModel,
    pub tectonic_carrier: TectonicCarrierConfig,
    pub erosion: ErosionParams,
    pub fine_density: FineDensityParams,
    pub fine_structure: FineStructureParams,
    pub fine_cache_mode: FineCacheMode,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fine_cache: Option<FineCacheRecord>,
    pub computed_stage: u32,
    pub viewed_stage: u32,
    pub active_cells: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct UnitManifest {
    pub contract_version: u32,
    pub elevation_unit_km: f32,
    pub planet_radius_km: f32,
    pub physical_relief_scale: f32,
    pub sea_level_datum: &'static str,
    pub native_slope: &'static str,
}

impl Default for UnitManifest {
    fn default() -> Self {
        Self {
            contract_version: 1,
            elevation_unit_km: ELEVATION_UNIT_KM,
            planet_radius_km: PLANET_RADIUS_KM,
            physical_relief_scale: PHYSICAL_RELIEF_SCALE,
            sea_level_datum: "coarse-area-quantile-zero",
            native_slope: "elevation-per-arc-radian",
        }
    }
}

impl RunManifest {
    pub fn from_world(world: &World) -> Self {
        let computed_stage = world.current_stage();
        let viewed_stage = world.view_stage().min(computed_stage);
        Self {
            build: BuildProvenance::current(),
            units: UnitManifest::default(),
            seed: world.seed,
            voronoi_backend: world.voronoi_backend,
            lloyd_iterations: world.lloyd_iterations,
            coarse_cells: world.tessellation.num_cells(),
            plate_count: world.plates.as_ref().map(|plates| plates.num_plates),
            orogen_model: world.orogen_model,
            tectonic_carrier: world.tectonic_carrier_config,
            erosion: world.erosion_params,
            fine_density: world.fine_density_params,
            fine_structure: world.fine_structure_params,
            fine_cache_mode: world.fine_cache,
            fine_cache: world.fine.as_ref().map(|fine| fine.cache_record.clone()),
            computed_stage,
            viewed_stage,
            active_cells: world.num_cells(),
        }
    }

    pub fn summary(&self) -> String {
        let cache = self
            .fine_cache
            .as_ref()
            .map(|record| format!("{:?}:{}", record.outcome, record.key_hex))
            .unwrap_or_else(|| "not-built".into());
        format!(
            "build={} seed={} backend={} model={} stage={}/{} cells={}/{} cache={}",
            self.build.label(),
            self.seed,
            self.voronoi_backend,
            self.orogen_model,
            self.viewed_stage,
            self.computed_stage,
            self.coarse_cells,
            self.active_cells,
            cache,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_world_manifest_records_generation_identity() {
        let world = World::new_with_options(42, 64, 0, VoronoiBackend::ConvexHull);
        let manifest = world.manifest();
        assert_eq!(manifest.seed, 42);
        assert_eq!(manifest.voronoi_backend, VoronoiBackend::ConvexHull);
        assert_eq!(manifest.lloyd_iterations, 0);
        assert_eq!(manifest.computed_stage, 0);
        assert_eq!(manifest.viewed_stage, 0);
        assert_eq!(manifest.coarse_cells, world.tessellation.num_cells());
    }

    #[test]
    fn cache_key_serializes_as_hex_string() {
        let record = FineCacheRecord {
            mode: FineCacheMode::Enabled,
            version: super::super::fine_cache::FINE_BASE_CACHE_VERSION,
            key_hex: "fedcba9876543210".into(),
            outcome: FineCacheOutcome::Hit,
            write_succeeded: None,
            max_cells: 1_000,
            actual_cells: 900,
        };
        let json = serde_json::to_value(record).unwrap();
        assert_eq!(json["key_hex"], "fedcba9876543210");
        assert_eq!(json["outcome"], "hit");
    }
}
