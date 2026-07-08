//! Disk cache for the expensive [`FineBase`] (staging tooling — see
//! docs/specs/staging.md). The fine mesh is slow to rebuild after a recompile,
//! but most before/afters change erosion-or-later code, which lives in
//! `FineSurface`, NOT `FineBase`. So caching the base lets a recompile of
//! downstream code reload the mesh and jump straight to the new code.
//!
//! The cache is AUTHORITATIVE: fine-mesh generation is not deterministic
//! run-to-run (s2-voronoi parallel welding drifts the cell count), so we store
//! and load the bytes rather than asserting regeneration reproduces them. The
//! key is a content hash of everything the base depends on, so a changed input
//! is a cache miss. Code changes the hash can't see (sampling/relaxation/
//! transfer logic) require bumping [`FINE_BASE_CACHE_VERSION`].

use std::fs::{self, File};
use std::io::{BufReader, BufWriter};
use std::path::PathBuf;

use glam::Vec3;

use super::constants::*;
use super::fine::{FineBase, FineDensityParams, FineStructureParams, OrogenFronts};
use super::{Atmosphere, Crust, Elevation, FeatureFields, Tessellation};

/// Bump when fine-mesh GENERATION CODE changes (sampling / relaxation / field
/// transfer / density logic) in a way the content hash below can't observe.
/// Constant changes ARE caught by the content hash (they move elevation /
/// atmosphere / the generators) and do not need a bump.
/// v2: hydrology became area-weighted; the fine-mesh density prior now reads
/// flow as a count-equivalent (`flow_count_equiv`), shifting the sampled mesh.
/// v3: (a) `atmosphere.wind` is now part of the key (it's transferred into the
/// base and drives orographic precip), so a wind-only change is no longer a
/// false hit; (b) coarse drainage now uses distance-normalized steepest descent,
/// which shifts the flow field feeding the fine-mesh density prior — a
/// generation-logic change the content hash can't observe.
/// v4: the fine base now synthesizes interior structural relief + range-front
/// scarps into `base_elevation` (erosion-v2 P1a) — a code-generated change the
/// pre-v4 hash can't observe; the structural knobs ARE hashed (below) but the
/// generation logic itself needs the bump.
/// v5: the interior relief is now strike-aware near convergent fronts (P1b) — the
/// banded grain is a generation-logic change; `front_strike_weight` + the convergent-
/// front primitives are hashed below, but the logic itself needs the bump.
/// v6: active/passive margin contrast (P1c) modulates the relief amplitude near the
/// coast — a generation-logic change; `margin_contrast` + its shape constants are
/// hashed below, but the logic itself needs the bump.
/// v7: emergent orogens (erosion-v3) demote the orogen peak from the base by
/// `emergent_lambda·(arc+collision)` and add the field to `FineBase` — a serialized-
/// struct + generation-logic change; the knob is hashed below.
/// v8: O0 structured emergent uplift (orogen-structure) adds the `emergent_uplift_shape`
/// field to `FineBase` and the `emergent_structured` knob (hashed below) — serialized-
/// struct + generation-logic change.
/// v9: O0 segmentation now uses real arc-length CHAINING of the fronts (not 3D-noise
/// proxy) + the structured builder gained a per-cell land floor — generation-logic
/// changes the content hash can't observe.
const FINE_BASE_CACHE_VERSION: u32 = 9;

/// How the fine-mesh base should use the on-disk cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FineCacheMode {
    /// Never read or write the cache (always regenerate).
    Disabled,
    /// Load a valid entry if present, otherwise generate and save it.
    #[default]
    Enabled,
    /// Always regenerate and overwrite the cached entry (ignore any existing).
    Rebuild,
}

/// Content hash of everything `FineBase` generation depends on. A change to any
/// input (or the version, or the fine density constants) yields a different key
/// and therefore a cache miss. Hashing a few million f32 is ~ms — negligible
/// next to the mesh build it guards.
#[allow(clippy::too_many_arguments)]
pub fn fine_base_key(
    seed: u64,
    coarse: &Tessellation,
    crust: &Crust,
    features: &FeatureFields,
    coarse_elevation: &Elevation,
    atmosphere: &Atmosphere,
    max_cells: usize,
    density: &FineDensityParams,
    structure: &FineStructureParams,
    fronts: &OrogenFronts,
) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64; // FNV-1a offset basis
    mix_u64(&mut h, FINE_BASE_CACHE_VERSION as u64);
    mix_u64(&mut h, seed);
    mix_u64(&mut h, max_cells as u64);

    // Fine density / sampling knobs that shape the mesh (defaults from the FINE_*
    // consts, so a default run hashes the same values as before; an override is a
    // distinct key -> a cache miss).
    for f in [
        density.plains_km,
        density.mountain_km,
        density.ocean_km,
        density.exponent,
        density.slope_weight,
        density.flow_weight,
        density.activity_weight,
    ] {
        mix_f32(&mut h, f);
    }
    mix_u64(&mut h, FINE_RELAX_PASSES as u64);

    // Structural-relief knobs that shape the pre-erosion base_elevation (P1a;
    // decision A — these are fine-base inputs, so a sweep regenerates the base).
    mix_f32(&mut h, structure.fault_scarp_height);
    mix_f32(&mut h, structure.interior_relief);
    // Active/passive margin contrast (P1c): the knob (the margin SOURCE,
    // crust.signed_margin_distance, is already hashed above). Shape constants below.
    mix_f32(&mut h, structure.margin_contrast);
    // Emergent orogens (erosion-v3): the demotion fraction (the arc+collision source is
    // already hashed via features). Changes the base envelope, so it's a fine-base input.
    mix_f32(&mut h, structure.emergent_lambda);
    // O0 structured emergent uplift (orogen-structure): the structured-shape blend knob.
    mix_f32(&mut h, structure.emergent_structured);
    // Candidate A/A' meso relief: these shape `emergent_uplift_shape` and/or the
    // pre-erosion base elevation.
    mix_f32(&mut h, structure.meso_relief);
    mix_f32(&mut h, structure.meso_base_relief);
    mix_f32(&mut h, structure.meso_wavelength_km);

    // Strike-aware fronts (P1b): the knob + the convergent-front primitives the
    // banded grain consumes (arc endpoints, per-front overriding side, coarse plate
    // map). The arc endpoints are the consumed geometry; `points` is derived from them.
    mix_f32(&mut h, structure.front_strike_weight);
    mix_vec3s(&mut h, &fronts.seg_a);
    mix_vec3s(&mut h, &fronts.seg_b);
    mix_u64(&mut h, fronts.accept_plate.len() as u64);
    for a in &fronts.accept_plate {
        // distinguish None (collision) from Some(plate); +1 avoids a 0/None clash.
        mix_u64(&mut h, a.map_or(0, |p| p as u64 + 1));
    }
    mix_u64(&mut h, fronts.coarse_cell_plate.len() as u64);
    for &p in &fronts.coarse_cell_plate {
        mix_u64(&mut h, p as u64);
    }

    // Structural-relief SHAPE constants (P1a interior + P1b fronts). These are
    // generation constants the FineBase synthesis consumes but no hashed input
    // observes, so a recompile that tweaks one would otherwise false-hit the cache
    // (cf. FINE_RELAX_PASSES above). Hash them so a constant change is a clean miss.
    for f in [
        FINE_INTERIOR_FREQUENCY as f32,
        FINE_INTERIOR_OCTAVES as f32,
        FINE_INTERIOR_MIN_ELEV,
        FINE_INTERIOR_ELEV_BAND,
        FINE_INTERIOR_FORCING_THRESHOLD,
        FINE_INTERIOR_FORCING_BAND,
        FINE_FRONT_BAND_FREQUENCY as f32,
        FINE_FRONT_INFLUENCE_RADIUS,
        FINE_FRONT_WARP,
        FINE_FRONT_WARP_FREQUENCY as f32,
        FINE_FRONT_GATHER_MARGIN,
        FINE_MARGIN_WIDTH,
        FINE_MARGIN_ACTIVE_FACTOR,
        FINE_MARGIN_PASSIVE_FACTOR,
        FINE_MESO_PHASE_FREQUENCY as f32,
        FINE_MESO_SPUR_FREQUENCY as f32,
        FINE_MESO_PHASE_WARP,
        FINE_MESO_WAVELENGTH_JITTER,
        FINE_MESO_AMP_MIN,
        FINE_MESO_ISO_WEIGHT,
        FINE_MESO_ISO_FREQUENCY as f32,
    ] {
        mix_f32(&mut h, f);
    }

    // Coarse-world fingerprint. The generators pin the coarse mesh identity
    // (seed + resolution + Lloyd). Coarse elevation + atmosphere capture every
    // elevation/atmosphere CONSTANT (they're downstream of them). But the fine
    // base also TRANSFERS crust/features fields (crust_thickness, continentality,
    // arc, collision, rift_delta, ...) that the scalar coarse elevation reduces
    // lossily, so hash crust + features directly too — otherwise a change that
    // alters a transferred field without moving coarse elevation would be a
    // false cache hit. (Generation-CODE changes still need a VERSION bump.)
    mix_vec3s(&mut h, &coarse.voronoi.generators);
    mix_f32s(&mut h, &coarse_elevation.values);
    mix_f32s(&mut h, &atmosphere.temperature);
    mix_f32s(&mut h, &atmosphere.precipitation);
    mix_f32s(&mut h, &atmosphere.uplift);
    // Wind is transferred into the fine base (`fine.rs`) and consumed by the
    // orographic-precip feedback, so it must be in the key too.
    mix_vec3s(&mut h, &atmosphere.wind);

    mix_f32s(&mut h, &crust.signed_margin_distance);
    mix_u64(&mut h, crust.cell_craton.len() as u64);
    for &c in &crust.cell_craton {
        mix_u64(&mut h, c as u64);
    }
    for field in [
        &features.trench,
        &features.arc,
        &features.ridge,
        &features.collision,
        &features.activity,
        &features.convergent,
        &features.divergent,
        &features.ridge_distance,
        &features.ridge_age_distance,
        &features.ridge_spreading_rate,
        &features.collision_distance,
        &features.arc_distance,
        &features.arc_shape_noise,
        &features.rift_delta,
    ] {
        mix_f32s(&mut h, field);
    }
    h
}

/// Load a cached base for `key`, or `None` on miss / unreadable file.
pub fn load(key: u64) -> Option<FineBase> {
    let path = cache_path(key);
    let file = File::open(&path).ok()?;
    match bincode::deserialize_from(BufReader::new(file)) {
        Ok(base) => {
            log::info!("fine mesh: loaded FineBase from cache {}", path.display());
            Some(base)
        }
        Err(e) => {
            log::warn!(
                "fine mesh: cache {} unreadable ({e}); regenerating",
                path.display()
            );
            None
        }
    }
}

/// Serialize `base` to the cache under `key` (best effort; logs on failure).
pub fn save(key: u64, base: &FineBase) {
    let dir = cache_dir();
    if let Err(e) = fs::create_dir_all(&dir) {
        log::warn!("fine mesh: can't create cache dir {}: {e}", dir.display());
        return;
    }
    let path = cache_path(key);
    let result = File::create(&path)
        .map_err(|e| e.to_string())
        .and_then(|f| bincode::serialize_into(BufWriter::new(f), base).map_err(|e| e.to_string()));
    match result {
        Ok(()) => log::info!("fine mesh: saved FineBase to cache {}", path.display()),
        Err(e) => log::warn!("fine mesh: failed to save cache {}: {e}", path.display()),
    }
}

fn cache_dir() -> PathBuf {
    PathBuf::from(".cache").join("finebase")
}

fn cache_path(key: u64) -> PathBuf {
    cache_dir().join(format!("{key:016x}.bin"))
}

#[inline]
fn mix_u64(h: &mut u64, x: u64) {
    *h ^= x;
    *h = h.wrapping_mul(0x0000_0100_0000_01b3); // FNV-1a prime
}

#[inline]
fn mix_f32(h: &mut u64, x: f32) {
    // Bit pattern: distinguishes -0.0/0.0 and NaN payloads, which is fine (and
    // desirable) for a cache key.
    mix_u64(h, x.to_bits() as u64);
}

fn mix_f32s(h: &mut u64, xs: &[f32]) {
    mix_u64(h, xs.len() as u64);
    for &x in xs {
        mix_f32(h, x);
    }
}

fn mix_vec3s(h: &mut u64, xs: &[Vec3]) {
    mix_u64(h, xs.len() as u64);
    for v in xs {
        mix_f32(h, v.x);
        mix_f32(h, v.y);
        mix_f32(h, v.z);
    }
}
