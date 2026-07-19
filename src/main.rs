mod app;

use std::path::PathBuf;

use clap::{Parser, ValueEnum};
use winit::event_loop::{ControlFlow, EventLoop};

use app::world::{
    advance_to_stage_2, advance_to_stage_3, advance_to_stage_3_with_cap, advance_to_stage_4,
};
use hex3::world::{FineCacheMode, OrogenModel, VoronoiBackend};

const HIDE_RESEARCH_CLI: bool = !cfg!(feature = "research-landscape");

#[derive(Clone, Copy, Debug, ValueEnum)]
enum CliVoronoiBackend {
    #[value(name = "convex-hull")]
    ConvexHull,
    #[value(name = "knn-clipping")]
    KnnClipping,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum CliOrogenModel {
    Legacy,
    #[value(name = "legacy-yield")]
    LegacyYield,
    #[value(name = "conserved-local")]
    ConservedLocal,
    #[value(name = "conserved-feature-footprint")]
    ConservedFeatureFootprint,
    #[value(name = "conserved-isotropic")]
    ConservedIsotropic,
    #[value(name = "history-local")]
    HistoryLocal,
    #[value(name = "history-diffusive")]
    HistoryDiffusive,
    #[value(name = "history-material")]
    HistoryMaterial,
    #[value(name = "history-thin-sheet")]
    HistoryThinSheet,
    #[value(name = "history-carrier-thin-sheet")]
    HistoryCarrierThinSheet,
    #[value(name = "history-carrier-evolved")]
    HistoryCarrierEvolved,
    #[value(name = "history-carrier-lifecycle")]
    HistoryCarrierLifecycle,
    #[value(name = "thin-sheet")]
    ThinSheet,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum CliReliefPreset {
    Flat,
    Physical,
    Authentic,
    Dramatic,
}

impl From<CliReliefPreset> for app::ReliefPreset {
    fn from(value: CliReliefPreset) -> Self {
        match value {
            CliReliefPreset::Flat => Self::Flat,
            CliReliefPreset::Physical => Self::Physical,
            CliReliefPreset::Authentic => Self::Authentic,
            CliReliefPreset::Dramatic => Self::Dramatic,
        }
    }
}

impl From<CliOrogenModel> for OrogenModel {
    fn from(value: CliOrogenModel) -> Self {
        match value {
            CliOrogenModel::Legacy => OrogenModel::Legacy,
            CliOrogenModel::LegacyYield => OrogenModel::LegacyYield,
            CliOrogenModel::ConservedLocal => OrogenModel::ConservedLocal,
            CliOrogenModel::ConservedFeatureFootprint => OrogenModel::ConservedFeatureFootprint,
            CliOrogenModel::ConservedIsotropic => OrogenModel::ConservedIsotropic,
            CliOrogenModel::HistoryLocal => OrogenModel::HistoryLocal,
            CliOrogenModel::HistoryDiffusive => OrogenModel::HistoryDiffusive,
            CliOrogenModel::HistoryMaterial => OrogenModel::HistoryMaterial,
            CliOrogenModel::HistoryThinSheet => OrogenModel::HistoryThinSheet,
            CliOrogenModel::HistoryCarrierThinSheet => OrogenModel::HistoryCarrierThinSheet,
            CliOrogenModel::HistoryCarrierEvolved => OrogenModel::HistoryCarrierEvolved,
            CliOrogenModel::HistoryCarrierLifecycle => OrogenModel::HistoryCarrierLifecycle,
            CliOrogenModel::ThinSheet => OrogenModel::ThinSheet,
        }
    }
}

impl From<CliVoronoiBackend> for VoronoiBackend {
    fn from(value: CliVoronoiBackend) -> Self {
        match value {
            CliVoronoiBackend::ConvexHull => VoronoiBackend::ConvexHull,
            CliVoronoiBackend::KnnClipping => VoronoiBackend::KnnClipping,
        }
    }
}

/// Hex3 - Spherical Voronoi planet generator
#[derive(Parser, Debug)]
#[command(name = "hex3", version, about)]
struct Cli {
    /// Run in headless mode (no window, generate and quit)
    #[arg(long)]
    headless: bool,

    /// Target stage (1-4: Lithosphere, Atmosphere, Hydrosphere, Erosion).
    /// Interactive defaults to 1, headless defaults to max.
    #[arg(long)]
    stage: Option<u32>,

    /// Random seed for world generation
    #[arg(long)]
    seed: Option<u64>,

    /// Coarse Voronoi cell count. Controls global resolution and generation cost
    /// for stages 1-2 (lithosphere/atmosphere).
    #[arg(long, default_value_t = 100_000)]
    cells: usize,

    /// Uniform multiplier on the fine-mesh cell-size targets. >1 coarsens the
    /// fine mesh; <1 refines it. A value other than 1.0 disables the fine-base
    /// disk cache.
    #[arg(long, default_value_t = 1.0)]
    fine_scale: f32,

    /// Named relief presentation. World elevations remain unchanged.
    #[arg(long, value_enum, default_value_t = CliReliefPreset::Authentic)]
    relief_preset: CliReliefPreset,

    /// Raw relief displacement override. Takes precedence over
    /// --relief-preset; 0.00157 is approximately physical 1x.
    #[arg(long, default_value_t = -1.0, allow_hyphen_values = true)]
    relief_scale: f32,

    /// Cartographic river-width multiplier in screen space. River selection
    /// remains controlled by physical catchment thresholds.
    #[arg(long, default_value_t = 1.0)]
    river_width_scale: f32,

    /// Fine-mesh cell guardrail for headless diagnostic exports. Zero uses the
    /// normal product default. Ignored by interactive mode.
    #[arg(long, default_value_t = 0)]
    fine_max: usize,

    /// Export world data to file (supports .json and .json.gz)
    #[arg(long, value_name = "FILE")]
    export: Option<PathBuf>,

    /// Bake the river network into an equirectangular PNG (river-render dev/debug).
    #[arg(long, value_name = "FILE", hide = HIDE_RESEARCH_CLI)]
    river_texture: Option<PathBuf>,

    /// River render density: minimum catchment area (km²) that renders as a
    /// river in 'All' mode (Major outlet/branch scale with it, 75×/12.5×).
    /// Physical when represented by at least four global-mean cells; coarser
    /// meshes raise the effective threshold. Earth-ish map density ~1000-4000;
    /// higher = sparser rivers. Diagnose reports requested/effective values.
    #[arg(long, default_value_t = app::world::RIVER_DEFAULT_MIN_CATCHMENT_KM2)]
    river_min_catchment_km2: f32,

    /// A/B: use the LEGACY count-equivalent river render thresholds (tuned on
    /// the coarse mesh; on the fine mesh they render only the largest trunk
    /// stubs — see diagnose --river-audit).
    #[arg(long, default_value_t = false, hide = HIDE_RESEARCH_CLI)]
    river_legacy: bool,

    /// Compatibility switch for quarantined geometry-backend comparisons.
    #[arg(
        long,
        value_enum,
        default_value_t = CliVoronoiBackend::ConvexHull,
        hide = HIDE_RESEARCH_CLI
    )]
    voronoi_backend: CliVoronoiBackend,

    /// Compatibility switch for quarantined orogen experiments. The ordinary
    /// product path uses Legacy; research binaries own model comparison.
    #[arg(
        long,
        value_enum,
        default_value_t = CliOrogenModel::Legacy,
        hide = HIDE_RESEARCH_CLI
    )]
    orogen_model: CliOrogenModel,

    /// Disable the fine-mesh base disk cache (always regenerate stage 3a)
    #[arg(long)]
    no_fine_cache: bool,

    /// Force-rebuild and overwrite the fine-mesh base disk cache
    #[arg(long)]
    rebuild_fine_cache: bool,

    /// MFD erosion exponent (routing ladder Rung 2/3). <0 = EROSION_MFD_EXPONENT
    /// default (off); 0 = single-flow; ~1 dispersive .. high ≈ single-flow. Set
    /// >0 to visually A/B MFD incision. See docs/archive/specs/erosion-routing-ladder.md.
    #[arg(long, default_value_t = -1.0, hide = HIDE_RESEARCH_CLI)]
    erosion_mfd_exponent: f32,

    /// Barnes convergent flat resolution (Rung 1). -1 = default (on); 0 = off
    /// (old flood_parent wavefront); 1 = on. A/B the spiral-on-flats fix.
    #[arg(long, default_value_t = -1, hide = HIDE_RESEARCH_CLI)]
    erosion_flat_resolution: i8,

    /// Plains alluvial regime gate: channel slope (elev/km) at/above which
    /// incision is full; gentler channels fade to alluvial (floodplains, not
    /// ditches). <0 = EROSION_CONFINEMENT_SLOPE default (off). See
    /// docs/archive/specs/erosion-valleys-not-channels.md.
    #[arg(long, default_value_t = -1.0, hide = HIDE_RESEARCH_CLI)]
    erosion_confinement_slope: f32,

    /// Erosion erodibility K (incision strength; default 4e-2). <0 = default.
    /// LOWER = gentler dissection (less sharp). Visual dissection-texture lever.
    #[arg(long, default_value_t = -1.0, hide = HIDE_RESEARCH_CLI)]
    erosion_k: f32,

    /// Stream-power SLOPE exponent n in E=K·A^m·S^n (default 1). <0 = default.
    /// >1 (≈1.5–2) = sharper valleys/divides ("ranges not bumps"). Newton-solved.
    #[arg(long, default_value_t = -1.0, hide = HIDE_RESEARCH_CLI)]
    erosion_n: f32,

    /// Hillslope diffusivity (smoothing; default 2e-8). <0 = default. HIGHER =
    /// smoother (rounds the sharp dissection). Visual dissection-texture lever.
    #[arg(long, default_value_t = -1.0, hide = HIDE_RESEARCH_CLI)]
    erosion_diffusivity: f32,

    /// Channel-initiation support area (km² at mean land wetness; default 30).
    /// <0 = default. HIGHER = channels start later = LOWER drainage density
    /// (fewer/broader valleys, less "busy"). The primary density lever.
    #[arg(long, default_value_t = -1.0, hide = HIDE_RESEARCH_CLI)]
    erosion_channel_support: f32,

    /// Uplift-FORCING smoothing length (km; escalation #1). Smooths the per-step
    /// tectonic uplift source over a sub-grid orogenic width to remove mountain-top
    /// cell-scale "swiss cheese" without flattening orogens. <0 = default; 0 = off.
    /// See docs/archive/specs/erosion-uplift-smoothing.md.
    #[arg(long, default_value_t = -1.0, hide = HIDE_RESEARCH_CLI)]
    erosion_uplift_smooth: f32,

    /// Roering nonlinear-hillslope critical slope S_c (escalation #2; Δelev/radian,
    /// ~grade·637). Diffusivity blows up toward S_c -> planar slopes + crisp
    /// ridges (vs linear-creep mush). <0 = default; 0 = off. Visual de-prickle
    /// lever; sweep ~150-300. See docs/archive/specs/erosion-escalations.md.
    #[arg(long, default_value_t = -1.0, hide = HIDE_RESEARCH_CLI)]
    erosion_hillslope_crit: f32,

    /// Hillslope-diffusion Jacobi sweeps per step (default 6). 0 = default. LOWER =
    /// cheaper diffuse phase; 6->3 measured metric-equivalent (incision percentiles
    /// byte-identical) — a near-free speedup, visual-confirm it.
    #[arg(long, default_value_t = 0, hide = HIDE_RESEARCH_CLI)]
    erosion_diffusion_iters: usize,

    /// Steps between drainage re-routings (default 6). 0 = default; 1 = re-route
    /// every step. HIGHER = cheaper route phase; 6->12 measured near-equivalent
    /// (~3-5% incision drift). Visual-A/B the speedup.
    #[arg(long, default_value_t = 0, hide = HIDE_RESEARCH_CLI)]
    erosion_reroute_interval: usize,

    /// Fluvial step count per precip pass (default 60). 0 = default. FEWER = cheaper
    /// AND less-mature / less-dissected terrain (erosion is still evolving at 60, not
    /// converged) — a perf x "less busy" fidelity dial, not free.
    #[arg(long, default_value_t = 0, hide = HIDE_RESEARCH_CLI)]
    erosion_steps: usize,

    /// Coupled erode<->precip feedback passes (default 2). 0 = default; 1 = no
    /// feedback (halves total erosion, but +16% incision / pricklier — the 2nd pass
    /// does anti-prickle healing). Fidelity dial, not free.
    #[arg(long, default_value_t = 0, hide = HIDE_RESEARCH_CLI)]
    erosion_precip_iters: usize,

    /// Tectonic uplift scale ("Hold & carve"). <0 = EROSION_UPLIFT_SCALE default;
    /// 0 = relaxation only. NOTE: ignored by the default emergent path.
    #[arg(long, default_value_t = -1.0, hide = HIDE_RESEARCH_CLI)]
    erosion_uplift_scale: f32,

    /// Emergent builder over-rebuild gain (relief-spectrum candidate B). >1 builds
    /// more orogen volume than the coarse target so erosion carves the excess into
    /// relief. <0 = EMERGENT_REBUILD_GAIN default (1.2).
    #[arg(long, default_value_t = -1.0, hide = HIDE_RESEARCH_CLI)]
    erosion_rebuild_gain: f32,

    /// Depositional repose slope (fans/floodplains/deltas). <0 = default; 0 =
    /// sink-fill only.
    #[arg(long, default_value_t = -1.0, hide = HIDE_RESEARCH_CLI)]
    erosion_deposition_slope: f32,

    /// Lithologic erodibility contrast sigma. <0 = default; 0 = uniform K.
    #[arg(long, default_value_t = -1.0, hide = HIDE_RESEARCH_CLI)]
    erosion_litho_sigma: f32,

    /// Structural-grain erodibility strength (fold-belt ridge-and-valley). <0 =
    /// default; 0 = off.
    #[arg(long, default_value_t = -1.0, hide = HIDE_RESEARCH_CLI)]
    erosion_litho_grain: f32,

    /// Orographic precip modulation strength (windward wetter, lee drier). <0 =
    /// default; 0 = coarse precip.
    #[arg(long, default_value_t = -1.0, hide = HIDE_RESEARCH_CLI)]
    erosion_orographic_strength: f32,

    /// Downwind rain-shadow strength (propagate lee dry-anomaly downwind). <0 = default
    /// (0=off); up = stronger progressive lee drying.
    #[arg(long, default_value_t = -1.0, hide = HIDE_RESEARCH_CLI)]
    erosion_downwind_shadow: f32,

    /// Lakes-as-evaporation precip boost. <0 = default; 0 = off.
    #[arg(long, default_value_t = -1.0, hide = HIDE_RESEARCH_CLI)]
    erosion_lake_evap: f32,

    /// Glacial abrasion coefficient (ice-flux over-deepening). <0 = default; 0 =
    /// no glacial pass.
    #[arg(long, default_value_t = -1.0, hide = HIDE_RESEARCH_CLI)]
    glacial_k: f32,

    /// A4 drainage-pulse dial: burn-in erode → trunk/interfluve uplift modifier →
    /// frozen final epoch (meso-a4-drainage-pulse.md). <0 = default (0=off).
    #[arg(long, default_value_t = -1.0, hide = HIDE_RESEARCH_CLI)]
    drainage_pulse: f32,

    /// A4 burn-in epoch steps (drainage self-organization). 0 = default (80).
    #[arg(long, default_value_t = 0, hide = HIDE_RESEARCH_CLI)]
    pulse_burnin_steps: usize,

    /// A4 trunk-proximity Gaussian sigma, km. <0 = default (15).
    #[arg(long, default_value_t = -1.0, hide = HIDE_RESEARCH_CLI)]
    pulse_smooth_km: f32,

    /// Fault range-front scarp relief. <0 = default; 0 = off (smooth fronts).
    #[arg(long, default_value_t = -1.0, hide = HIDE_RESEARCH_CLI)]
    fault_scarp: f32,

    /// Fine interior structural relief amplitude (P1a: mid-band fault/fold grain
    /// that breaks the flat orogen summit). <0 = default; 0 = off (pure interpolant).
    #[arg(long, default_value_t = -1.0, hide = HIDE_RESEARCH_CLI)]
    interior_relief: f32,

    /// Fine strike-band weight (P1b: interior grain aligned to the nearest orogen
    /// front vs isotropic). <0 = default; 0 = isotropic (P1a).
    #[arg(long, default_value_t = -1.0, hide = HIDE_RESEARCH_CLI)]
    front_strike_weight: f32,

    /// Fine margin contrast (P1c: sharpen relief on active coasts, damp passive).
    /// <0 = default; 0 = off (P1b).
    #[arg(long, default_value_t = -1.0, hide = HIDE_RESEARCH_CLI)]
    margin_contrast: f32,

    /// Emergent-orogens demotion fraction (erosion-v3): demote λ·(arc+collision) and
    /// rebuild via active uplift. <0 = default (0=off). Pair with --erosion-uplift-scale.
    #[arg(long, default_value_t = -1.0, hide = HIDE_RESEARCH_CLI)]
    emergent_lambda: f32,

    /// O0 structured emergent uplift (asymmetric+segmented vs uniform rebuild).
    /// <0 = default (0=off); 1 = fully structured. Needs --emergent-lambda + --erosion-n~2.
    #[arg(long, default_value_t = -1.0, hide = HIDE_RESEARCH_CLI)]
    emergent_structured: f32,
    /// Candidate A meso uplift-shape modulation depth. <0 = default (0=off);
    /// 0 = off; 0.3-0.9 = test. Regenerates the fine base.
    #[arg(long, default_value_t = -1.0, hide = HIDE_RESEARCH_CLI)]
    meso_relief: f32,
    /// Fold-train irregularity 0..1 (cross-strike decorrelation + 2nd octave +
    /// crest sharpening). <0 = default (0.7); 0 = plain periodic train.
    #[arg(
        long,
        default_value_t = -1.0,
        allow_hyphen_values = true,
        hide = HIDE_RESEARCH_CLI
    )]
    meso_irregularity: f32,

    /// Meso construction style: 0 = fold train (foreland preset), 1 =
    /// massif-corridor (alpine default). <0 = default (1).
    #[arg(
        long,
        default_value_t = -1,
        allow_hyphen_values = true,
        hide = HIDE_RESEARCH_CLI
    )]
    meso_style: i32,

    /// Candidate A' meso base-elevation relief amplitude. <0 = default (0=off);
    /// elevation units: 0.01 is about 100 m. Regenerates the fine base.
    #[arg(long, default_value_t = -1.0, hide = HIDE_RESEARCH_CLI)]
    meso_base_relief: f32,
    /// Candidate A meso fold-train wavelength in km. <0 = default (25 km).
    /// Regenerates the fine base.
    #[arg(long, default_value_t = -1.0, hide = HIDE_RESEARCH_CLI)]
    meso_wavelength_km: f32,

    /// Sweep mode: erosion knob to vary across columns (enables a headless
    /// render-to-PNG sweep). Knobs: k, diffusivity, channel_support,
    /// hillslope_crit, confinement_slope, uplift_smooth, mfd_exponent,
    /// diffusion_iters, reroute_interval, steps, precip_iters, flat_resolution,
    /// relief_scale and river_width_scale (renderer-only).
    #[arg(long, hide = HIDE_RESEARCH_CLI)]
    sweep: Option<String>,

    /// Cumulative-stack preset (o3a, p1, v3, o0, meso), or the special
    /// `range-ancestry` diagnostic packet (requires exactly three explicit
    /// --sweep-target values and stage 4), `water-geography` (automatically
    /// selects a coast complex and overflowing lake; stage 4), or
    /// `living-surface-preview` (global fractional-physiognomy evidence; stage 4),
    /// or `consequential-geography` (matched aggregate-site probe, ablations and
    /// same-site physical/zero-grade terrestrial routes; stage 4), or
    /// `world-readability-v0` (matched Terrain/Living Surface globe, map and
    /// derived river-mouth relationship views over one Stage-4 world).
    /// Ignores --sweep/--sweep-values.
    #[arg(long, hide = HIDE_RESEARCH_CLI)]
    sweep_stack: Option<String>,

    /// Comma-separated values for --sweep (e.g. "10,30,60,120").
    #[arg(long, default_value = "", hide = HIDE_RESEARCH_CLI)]
    sweep_values: String,

    /// Optional second knob varied across rows (2-D grid).
    #[arg(long, hide = HIDE_RESEARCH_CLI)]
    sweep2: Option<String>,

    /// Comma-separated values for --sweep2.
    #[arg(long, default_value = "", hide = HIDE_RESEARCH_CLI)]
    sweep2_values: String,

    /// Output directory for sweep PNG tiles + montage.png.
    #[arg(
        long,
        default_value = "sweep_out",
        hide = HIDE_RESEARCH_CLI
    )]
    out_dir: PathBuf,

    /// Sweep tile width in pixels.
    #[arg(long, default_value_t = 1024, hide = HIDE_RESEARCH_CLI)]
    sweep_width: u32,

    /// Sweep tile height in pixels.
    #[arg(long, default_value_t = 1024, hide = HIDE_RESEARCH_CLI)]
    sweep_height: u32,

    /// Sweep camera yaw in degrees (globe orbit).
    #[arg(long, default_value_t = 30.0, hide = HIDE_RESEARCH_CLI)]
    sweep_yaw: f32,

    /// Sweep camera pitch in degrees (globe orbit).
    #[arg(long, default_value_t = 25.0, hide = HIDE_RESEARCH_CLI)]
    sweep_pitch: f32,

    /// Sweep camera distance from globe center (overview).
    #[arg(long, default_value_t = 2.2, hide = HIDE_RESEARCH_CLI)]
    sweep_distance: f32,

    /// Zoomed close-up views per tile (auto-aimed at the highest land), in
    /// addition to the globe overview. 0 = overview only.
    #[arg(long, default_value_t = 3, hide = HIDE_RESEARCH_CLI)]
    sweep_zoom_views: usize,

    /// Close-up camera altitude above target (smaller = tighter zoom).
    #[arg(long, default_value_t = 0.3, hide = HIDE_RESEARCH_CLI)]
    sweep_zoom_alt: f32,

    /// Stable close-up target as id:latitude_degrees:longitude_degrees. Repeat
    /// for matched dossier captures. Explicit targets replace auto highlands.
    #[arg(
        long,
        value_name = "ID:LAT_DEG:LON_DEG",
        hide = HIDE_RESEARCH_CLI
    )]
    sweep_target: Vec<app::sweep::SweepTarget>,

    /// Rivers in sweep tiles: off, major, or all.
    #[arg(long, default_value = "major", hide = HIDE_RESEARCH_CLI)]
    sweep_rivers: String,

    /// Research-only globe display A/B: geodesically subdivide each unified
    /// render triangle this many times after sampling the unchanged world.
    /// One level produces four display triangles per physical render triangle.
    #[arg(long, default_value_t = 0, hide = HIDE_RESEARCH_CLI)]
    sweep_display_subdivision: usize,

    /// Legacy flag: equivalent to --stage 2
    #[arg(long, hide = true)]
    stage2: bool,
}

/// Parse a comma-separated list of f64 values, ignoring empty entries.
fn parse_values(s: &str) -> Vec<f64> {
    s.split(',')
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .map(|t| {
            t.parse::<f64>()
                .unwrap_or_else(|_| panic!("invalid sweep value: '{t}'"))
        })
        .collect()
}

#[cfg(test)]
mod cli_contract_tests {
    use super::{Cli, CliOrogenModel};
    use clap::{CommandFactory, Parser};

    #[cfg(not(feature = "research-landscape"))]
    const QUARANTINED_HELP_MARKERS: &[&str] = &[
        "--voronoi-backend",
        "--orogen-model",
        "--river-texture",
        "--river-legacy",
        "--erosion-",
        "--glacial-",
        "--drainage-pulse",
        "--fault-scarp",
        "--interior-relief",
        "--front-strike-weight",
        "--margin-contrast",
        "--emergent-",
        "--meso-",
        "--sweep",
    ];

    #[cfg(feature = "research-landscape")]
    const REPRESENTATIVE_RESEARCH_FLAGS: &[&str] = &[
        "--voronoi-backend",
        "--orogen-model",
        "--river-texture",
        "--river-legacy",
        "--erosion-mfd-exponent",
        "--erosion-k",
        "--glacial-k",
        "--drainage-pulse",
        "--fault-scarp",
        "--emergent-lambda",
        "--meso-relief",
        "--sweep-stack",
        "--sweep-display-subdivision",
    ];

    #[cfg(not(feature = "research-landscape"))]
    #[test]
    fn product_help_hides_quarantined_research_surface() {
        let help = Cli::command().render_long_help().to_string();

        for marker in QUARANTINED_HELP_MARKERS {
            assert!(
                !help.contains(marker),
                "product help exposed research marker {marker}"
            );
        }
        assert!(!help.contains("history-carrier-lifecycle"));
        assert!(!help.contains("thin-sheet"));

        for flag in [
            "--relief-preset",
            "--river-width-scale",
            "--river-min-catchment-km2",
            "--no-fine-cache",
            "--rebuild-fine-cache",
        ] {
            assert!(help.contains(flag), "product help omitted {flag}");
        }
    }

    #[cfg(feature = "research-landscape")]
    #[test]
    fn research_help_exposes_quarantined_research_surface() {
        let help = Cli::command().render_long_help().to_string();

        for flag in REPRESENTATIVE_RESEARCH_FLAGS {
            assert!(help.contains(flag), "research help omitted {flag}");
        }
    }

    #[test]
    fn quarantined_switches_remain_parse_compatible() {
        let cli = Cli::try_parse_from([
            "hex3",
            "--orogen-model",
            "history-carrier-lifecycle",
            "--erosion-k",
            "0.03",
            "--glacial-k",
            "0.0",
            "--meso-relief",
            "0.5",
            "--sweep",
            "k",
            "--sweep-values",
            "0.02,0.03",
            "--sweep-display-subdivision",
            "1",
        ])
        .expect("quarantined compatibility switches should still parse");

        assert!(matches!(
            cli.orogen_model,
            CliOrogenModel::HistoryCarrierLifecycle
        ));
        assert_eq!(cli.erosion_k, 0.03);
        assert_eq!(cli.glacial_k, 0.0);
        assert_eq!(cli.meso_relief, 0.5);
        assert_eq!(cli.sweep.as_deref(), Some("k"));
        assert_eq!(cli.sweep_display_subdivision, 1);
    }

    #[test]
    fn product_default_remains_legacy_orogen() {
        let cli = Cli::try_parse_from(["hex3"]).expect("default CLI should parse");

        assert!(matches!(cli.orogen_model, CliOrogenModel::Legacy));
    }
}

fn main() {
    env_logger::init();
    let cli = Cli::parse();

    // Determine target stage
    let target_stage = if let Some(s) = cli.stage {
        s
    } else if cli.stage2 {
        2 // Legacy flag
    } else if cli.headless {
        4 // Headless defaults to max stage (currently 4 = Erosion)
    } else {
        1 // Interactive defaults to stage 1
    };

    let backend = VoronoiBackend::from(cli.voronoi_backend);
    let orogen_model = OrogenModel::from(cli.orogen_model);
    let named_relief = app::ReliefPreset::from(cli.relief_preset);
    let relief_scale = if cli.relief_scale >= 0.0 {
        cli.relief_scale
    } else {
        named_relief.scale()
    };

    // River render calibration (A/B: --river-legacy vs physical catchment km²).
    app::world::set_river_threshold_mode(if cli.river_legacy {
        app::world::RiverThresholdMode::Legacy
    } else {
        app::world::RiverThresholdMode::CatchmentKm2(cli.river_min_catchment_km2)
    });

    let erosion = app::world::ErosionOverrides {
        relief_scale: Some(relief_scale),
        river_width_scale: Some(cli.river_width_scale.max(0.0)),
        mfd_exponent: (cli.erosion_mfd_exponent >= 0.0).then_some(cli.erosion_mfd_exponent),
        flat_resolution: (cli.erosion_flat_resolution >= 0)
            .then_some(cli.erosion_flat_resolution != 0),
        confinement_slope: (cli.erosion_confinement_slope >= 0.0)
            .then_some(cli.erosion_confinement_slope),
        k: (cli.erosion_k >= 0.0).then_some(cli.erosion_k),
        n: (cli.erosion_n >= 0.0).then_some(cli.erosion_n),
        diffusivity: (cli.erosion_diffusivity >= 0.0).then_some(cli.erosion_diffusivity),
        channel_support_km2: (cli.erosion_channel_support >= 0.0)
            .then_some(cli.erosion_channel_support),
        uplift_smooth_km: (cli.erosion_uplift_smooth >= 0.0).then_some(cli.erosion_uplift_smooth),
        hillslope_critical_slope: (cli.erosion_hillslope_crit >= 0.0)
            .then_some(cli.erosion_hillslope_crit),
        diffusion_iters: (cli.erosion_diffusion_iters > 0).then_some(cli.erosion_diffusion_iters),
        reroute_interval: (cli.erosion_reroute_interval > 0)
            .then_some(cli.erosion_reroute_interval),
        steps: (cli.erosion_steps > 0).then_some(cli.erosion_steps),
        precip_outer_iters: (cli.erosion_precip_iters > 0).then_some(cli.erosion_precip_iters),
        uplift_scale: (cli.erosion_uplift_scale >= 0.0).then_some(cli.erosion_uplift_scale),
        rebuild_gain: (cli.erosion_rebuild_gain >= 0.0).then_some(cli.erosion_rebuild_gain),
        deposition_slope: (cli.erosion_deposition_slope >= 0.0)
            .then_some(cli.erosion_deposition_slope),
        litho_sigma: (cli.erosion_litho_sigma >= 0.0).then_some(cli.erosion_litho_sigma),
        litho_grain_strength: (cli.erosion_litho_grain >= 0.0).then_some(cli.erosion_litho_grain),
        orographic_precip_strength: (cli.erosion_orographic_strength >= 0.0)
            .then_some(cli.erosion_orographic_strength),
        downwind_shadow_strength: (cli.erosion_downwind_shadow >= 0.0)
            .then_some(cli.erosion_downwind_shadow),
        lake_evap_strength: (cli.erosion_lake_evap >= 0.0).then_some(cli.erosion_lake_evap),
        climate_ratio: None, // CLI doesn't expose climate (runtime Up/Down + sweep knob only)
        glacial_k: (cli.glacial_k >= 0.0).then_some(cli.glacial_k),
        drainage_pulse: (cli.drainage_pulse >= 0.0).then_some(cli.drainage_pulse),
        pulse_burnin_steps: (cli.pulse_burnin_steps > 0).then_some(cli.pulse_burnin_steps),
        pulse_smooth_km: (cli.pulse_smooth_km >= 0.0).then_some(cli.pulse_smooth_km),
        orogen_model: Some(orogen_model),
        fault_scarp_height: (cli.fault_scarp >= 0.0).then_some(cli.fault_scarp),
        interior_relief: (cli.interior_relief >= 0.0).then_some(cli.interior_relief),
        front_strike_weight: (cli.front_strike_weight >= 0.0).then_some(cli.front_strike_weight),
        margin_contrast: (cli.margin_contrast >= 0.0).then_some(cli.margin_contrast),
        emergent_lambda: (cli.emergent_lambda >= 0.0).then_some(cli.emergent_lambda),
        emergent_structured: (cli.emergent_structured >= 0.0).then_some(cli.emergent_structured),
        meso_relief: (cli.meso_relief >= 0.0).then_some(cli.meso_relief),
        meso_irregularity: (cli.meso_irregularity >= 0.0).then_some(cli.meso_irregularity),
        meso_style: (cli.meso_style >= 0).then_some(cli.meso_style as usize),
        meso_base_relief: (cli.meso_base_relief >= 0.0).then_some(cli.meso_base_relief),
        meso_wavelength_km: (cli.meso_wavelength_km >= 0.0).then_some(cli.meso_wavelength_km),
    };
    let fine_cache = if cli.no_fine_cache {
        FineCacheMode::Disabled
    } else if cli.rebuild_fine_cache {
        FineCacheMode::Rebuild
    } else {
        FineCacheMode::Enabled
    };

    if cli.sweep.is_some() || cli.sweep_stack.is_some() {
        let river_mode = match cli.sweep_rivers.as_str() {
            "off" => app::RiverMode::Off,
            "major" => app::RiverMode::Major,
            "all" => app::RiverMode::All,
            other => panic!("invalid --sweep-rivers '{other}'; use off, major, or all"),
        };
        let values1 = parse_values(&cli.sweep_values);
        if cli.sweep_stack.is_none() && values1.is_empty() {
            panic!("--sweep requires --sweep-values (e.g. --sweep-values 10,30,60)");
        }
        let opts = app::sweep::SweepOptions {
            seed: cli.seed.unwrap_or_else(rand::random),
            cells: cli.cells,
            fine_scale: cli.fine_scale,
            fine_max: cli.fine_max,
            // Sweeps are about erosion, so default to the erosion stage.
            target_stage: cli.stage.unwrap_or(4),
            voronoi_backend: backend,
            orogen_model,
            fine_cache,
            base_erosion: erosion,
            stack: cli.sweep_stack.clone(),
            knob1: cli.sweep.clone().unwrap_or_default(),
            values1,
            knob2: cli.sweep2.clone(),
            values2: parse_values(&cli.sweep2_values),
            out_dir: cli.out_dir,
            width: cli.sweep_width,
            height: cli.sweep_height,
            yaw_deg: cli.sweep_yaw,
            pitch_deg: cli.sweep_pitch,
            distance: cli.sweep_distance,
            zoom_views: cli.sweep_zoom_views,
            zoom_alt: cli.sweep_zoom_alt,
            targets: cli.sweep_target,
            river_mode,
            river_threshold_policy: if cli.river_legacy {
                "legacy-count-equivalent".to_string()
            } else {
                "catchment-km2".to_string()
            },
            river_min_catchment_km2: (!cli.river_legacy).then_some(cli.river_min_catchment_km2),
            display_subdivision_levels: cli.sweep_display_subdivision,
        };
        app::sweep::run_sweep(opts);
    } else if cli.headless {
        run_headless(
            cli.seed,
            cli.cells,
            cli.fine_scale,
            cli.fine_max,
            target_stage,
            cli.export,
            cli.river_texture,
            backend,
            fine_cache,
            orogen_model,
            erosion,
        );
    } else {
        run_interactive(
            cli.seed,
            target_stage,
            cli.export,
            backend,
            fine_cache,
            orogen_model,
            erosion,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn run_headless(
    seed: Option<u64>,
    num_cells: usize,
    fine_scale: f32,
    fine_max: usize,
    target_stage: u32,
    export_path: Option<PathBuf>,
    river_texture_path: Option<PathBuf>,
    voronoi_backend: VoronoiBackend,
    fine_cache: FineCacheMode,
    orogen_model: OrogenModel,
    erosion: app::world::ErosionOverrides,
) {
    let seed = seed.unwrap_or_else(rand::random);
    println!(
        "Headless mode: seed={}, cells={}, fine_scale={}, target_stage={}, voronoi_backend={}, orogen_model={}",
        seed, num_cells, fine_scale, target_stage, voronoi_backend, orogen_model
    );

    // Generate world
    print!("Generating world... ");
    let start = std::time::Instant::now();
    let mut world = app::world::create_world_with_orogen_model(
        seed,
        num_cells,
        voronoi_backend,
        fine_cache,
        orogen_model,
    );
    erosion.apply(&mut world);
    // Apply the fine-mesh resolution multiplier. A non-default scale changes the
    // sampled mesh, so the disk cache (keyed on the density params) would miss
    // anyway; disable it to avoid writing a base per scale.
    if (fine_scale - 1.0).abs() > f32::EPSILON {
        let dp = &mut world.fine_density_params;
        dp.plains_km *= fine_scale;
        dp.mountain_km *= fine_scale;
        dp.ocean_km *= fine_scale;
        world.fine_cache = FineCacheMode::Disabled;
    }
    println!("{:.1}ms", start.elapsed().as_secs_f64() * 1000.0);

    // Advance to target stage
    if target_stage >= 2 {
        print!("Advancing to stage 2 (Climate)... ");
        let start = std::time::Instant::now();
        advance_to_stage_2(&mut world);
        println!("{:.1}ms", start.elapsed().as_secs_f64() * 1000.0);
    }
    if target_stage >= 3 {
        print!("Advancing to stage 3 (Hydrosphere, pre-erosion)... ");
        let start = std::time::Instant::now();
        if fine_max > 0 {
            advance_to_stage_3_with_cap(&mut world, fine_max);
        } else {
            advance_to_stage_3(&mut world);
        }
        println!("{:.1}ms", start.elapsed().as_secs_f64() * 1000.0);
    }
    if target_stage >= 4 {
        print!("Advancing to stage 4 (Erosion)... ");
        let start = std::time::Instant::now();
        advance_to_stage_4(&mut world);
        println!("{:.1}ms", start.elapsed().as_secs_f64() * 1000.0);
    }

    println!(
        "World complete: {} cells, stage {}",
        world.num_cells(),
        world.current_stage()
    );

    // Export if requested
    if let Some(path) = export_path {
        app::export::export_world(&mut world, seed, &path);
    }

    // Bake the river network to an equirect PNG (river-render dev/debug).
    if let Some(path) = river_texture_path {
        let (w, h) = (4096u32, 2048u32);
        let rgba = app::world::bake_river_texture(&world, w, h);
        let file = std::fs::File::create(&path).expect("create river texture file");
        let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), w, h);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        encoder
            .write_header()
            .expect("png header")
            .write_image_data(&rgba)
            .expect("png write");
        println!("Baked river texture -> {}", path.display());
    }
}

fn run_interactive(
    seed: Option<u64>,
    target_stage: u32,
    export_path: Option<PathBuf>,
    voronoi_backend: VoronoiBackend,
    fine_cache: FineCacheMode,
    orogen_model: OrogenModel,
    erosion: app::world::ErosionOverrides,
) {
    let event_loop = EventLoop::new().expect("Failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Wait);

    if export_path.is_some() {
        eprintln!("Note: --export is ignored in interactive mode; press D to export instead.");
    }

    let config = app::AppConfig {
        seed,
        target_stage,
        voronoi_backend,
        fine_cache,
        orogen_model,
        erosion,
    };

    let mut app = app::App::new(config);
    event_loop
        .run_app(&mut app)
        .expect("Failed to run application");
}
