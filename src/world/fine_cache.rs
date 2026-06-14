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
use super::fine::FineBase;
use super::{Atmosphere, Elevation, Tessellation};

/// Bump when fine-mesh GENERATION CODE changes (sampling / relaxation / field
/// transfer / density logic) in a way the content hash below can't observe.
/// Constant changes ARE caught by the content hash (they move elevation /
/// atmosphere / the generators) and do not need a bump.
const FINE_BASE_CACHE_VERSION: u32 = 1;

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
pub fn fine_base_key(
    seed: u64,
    coarse: &Tessellation,
    coarse_elevation: &Elevation,
    atmosphere: &Atmosphere,
    max_cells: usize,
) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64; // FNV-1a offset basis
    mix_u64(&mut h, FINE_BASE_CACHE_VERSION as u64);
    mix_u64(&mut h, seed);
    mix_u64(&mut h, max_cells as u64);

    // Fine density / sampling constants that shape the mesh.
    for f in [
        FINE_PLAINS_CELL_KM,
        FINE_MOUNTAIN_CELL_KM,
        FINE_OCEAN_CELL_KM,
        FINE_DENSITY_FEATURE_EXPONENT,
        FINE_SLOPE_DENSITY_WEIGHT,
        FINE_FLOW_DENSITY_WEIGHT,
        FINE_ACTIVITY_DENSITY_WEIGHT,
    ] {
        mix_f32(&mut h, f);
    }
    mix_u64(&mut h, FINE_RELAX_PASSES as u64);

    // Coarse-world fingerprint. The generators pin the coarse mesh identity
    // (seed + resolution + Lloyd); coarse elevation is downstream of every
    // stage-1 tectonic input (plates/crust/features + all elevation constants);
    // atmosphere is downstream of elevation + all atmosphere constants. Together
    // these change whenever anything upstream of the fine base changes.
    mix_vec3s(&mut h, &coarse.voronoi.generators);
    mix_f32s(&mut h, &coarse_elevation.values);
    mix_f32s(&mut h, &atmosphere.temperature);
    mix_f32s(&mut h, &atmosphere.precipitation);
    mix_f32s(&mut h, &atmosphere.uplift);
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
