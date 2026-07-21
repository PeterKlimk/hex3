//! World diagnostics: generate a world and measure its features numerically.
//!
//! The primary instrument for judging physical plausibility of generated
//! artifacts — sizes in km/km² with Earth references alongside. Catches what
//! aggregate stats and rendered images both miss.
//!
//!     cargo run --release --bin diagnose -- --seed 12345
//!     cargo run --release --bin diagnose -- --seed 12345 --cells 40000

use clap::{Parser, ValueEnum};
use hex3::world::diagnostics::{
    distance_from_mask, measure_components, ComponentStats, EARTH_RADIUS_KM,
};
use hex3::world::{
    elevation_to_km, CellWaterState, FineSurface, OrogenModel, RiverSelection,
    RiverThresholdPolicy, World, ELEVATION_UNIT_KM,
};
use kiddo::{KdTree, SquaredEuclidean};

#[cfg(feature = "research-landscape")]
use std::path::{Path, PathBuf};

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

#[derive(Parser, Debug)]
#[command(name = "diagnose", about = "Measure generated world features")]
struct Cli {
    #[arg(long, default_value_t = 12345)]
    seed: u64,
    #[arg(long, default_value_t = 100_000)]
    cells: usize,
    /// Convergent-orogen model to diagnose.
    #[arg(long, value_enum, default_value_t = CliOrogenModel::Legacy)]
    orogen_model: CliOrogenModel,
    /// Max components listed per section.
    #[arg(long, default_value_t = 8)]
    top: usize,
    /// Run the COARSE-vs-FINE drainage audit (endorheic land fraction, lake
    /// fraction, basin counts) and exit. Compares the freshly-computed coarse
    /// hydrology against the fine (eroded stage-4) hydrology.
    #[arg(long, default_value_t = false)]
    drainage_audit: bool,
    /// Run the per-lake FEATURE audit and exit: object-level lake statistics
    /// (size spectrum, shape, placement, outlet plumbing) with Earth references,
    /// plus the climate-dial response curve, for coarse + fine hydrology.
    /// A/B against the pre-integration world with HEX3_NO_DRAINAGE_INTEGRATION=1.
    #[arg(long, default_value_t = false)]
    lake_audit: bool,
    /// Run the per-range MOUNTAIN audit and exit: range objects on the fine
    /// eroded surface (size/elongation, crest-pass spectrum, cross-strike
    /// asymmetry, sampled local relief) with Earth references.
    #[arg(long, default_value_t = false)]
    mountain_audit: bool,
    /// Trace the frozen Legacy collision compiler from attributed convergent
    /// sources to the highest/largest recurrent mountain roofs. Research only;
    /// observational (no replacement model or counterfactual).
    #[cfg(feature = "research-landscape")]
    #[arg(long, default_value_t = false)]
    roof_causal_trace: bool,
    /// Write the roof causal trace's compact deterministic JSON record.
    #[cfg(feature = "research-landscape")]
    #[arg(long, requires = "roof_causal_trace")]
    roof_causal_trace_out: Option<PathBuf>,
    /// Audit finite-age source components against the terrain and drainage they
    /// actually own. Implies the coherent frozen-support Slice A preset.
    #[cfg(feature = "research-landscape")]
    #[arg(long, default_value_t = false)]
    finite_age_component_audit: bool,
    /// Write the finite-age component correspondence as deterministic JSON.
    #[cfg(feature = "research-landscape")]
    #[arg(long, requires = "finite_age_component_audit")]
    finite_age_component_audit_out: Option<PathBuf>,
    /// Trace one finite-age episode along its actual ordered front chains. The
    /// trace is a bounded source-to-final diagnostic, not another terrain knob.
    /// Implies the coherent frozen-support Slice A preset.
    #[cfg(feature = "research-landscape")]
    #[arg(long)]
    finite_age_spatial_episode: Option<usize>,
    /// Write the selected episode's along-strike station record as JSON.
    #[cfg(feature = "research-landscape")]
    #[arg(long, requires = "finite_age_spatial_episode")]
    finite_age_spatial_trace_out: Option<PathBuf>,
    /// Audit the emergent builder per connected orogen: coarse target versus
    /// final peak, plus whether the planet-wide shape normalizer subsidizes or
    /// taxes each component. This is the standing coarse→fine rebuild gate.
    #[arg(long, default_value_t = false)]
    rebuild_fidelity_audit: bool,
    /// Audit how much tectonic support/detail survives each coarse→fine→erosion
    /// stage over the complete process footprint (not only surviving mountains).
    #[arg(long, default_value_t = false)]
    detail_survival_audit: bool,
    /// Audit connected plate-pair boundary episodes using physical km/Myr and
    /// Myr units. This is coarse-only and exits before atmosphere/fine generation.
    #[arg(long, default_value_t = false)]
    tectonic_history_audit: bool,
    /// Write the tectonic-history audit's compact deterministic source-
    /// viability JSON record. Research only.
    #[cfg(feature = "research-landscape")]
    #[arg(long, requires = "tectonic_history_audit")]
    tectonic_history_audit_out: Option<PathBuf>,
    /// Run the RIVER NETWORK audit and exit: the rendered river network's
    /// Strahler/Horton structure, drainage density, mouths, and per-trunk
    /// length/sinuosity/profile table with Earth references.
    #[arg(long, default_value_t = false)]
    river_audit: bool,
    /// Audit seasonless ecological semantics and exit: area-weighted biome mix,
    /// continuous potentials, transition coverage, and coherent regions.
    #[arg(long, default_value_t = false)]
    biome_audit: bool,
    /// Run a bounded cross-resolution pilot-response audit and exit. The pilot
    /// mesh is discarded before the native reference is generated.
    #[arg(long, default_value_t = false)]
    resolution_pilot_audit: bool,
    /// Fine-cell cap for the cheap response pilot.
    #[arg(long, default_value_t = 100_000)]
    pilot_max: usize,
    /// Erosion steps in the short pilot response. The default spans four
    /// re-route intervals at the current default interval of six.
    #[arg(long, default_value_t = 24)]
    pilot_steps: usize,
    /// Fine-mesh cell cap (the emergent count is coarsened to fit). Lower it to
    /// iterate faster on erosion/roughness probes. 0 = use the FINE_MAX_CELLS default.
    #[arg(long, default_value_t = 0)]
    fine_max: usize,
    /// Override erosion erodibility K. <0 = use the EROSION_K default. Lets you
    /// sweep erosion strength without a recompile (FineBase is cached).
    #[arg(long, default_value_t = -1.0)]
    erosion_k: f32,
    /// Override stream-power SLOPE exponent n (E=K·A^m·S^n). <0 = EROSION_N default (1);
    /// >1 (≈1.5–2) = sharper valleys/divides. Newton-solved.
    #[arg(long, default_value_t = -1.0)]
    erosion_n: f32,
    /// Override erosion step count. 0 = use the EROSION_STEPS default.
    #[arg(long, default_value_t = 0)]
    erosion_steps: usize,
    /// Override hillslope diffusivity. <0 = use the EROSION_DIFFUSIVITY default.
    #[arg(long, default_value_t = -1.0)]
    erosion_diffusivity: f32,
    /// Override channel-initiation support area (km² at mean land wetness).
    /// <0 = use the EROSION_CHANNEL_SUPPORT_KM2 default; 0 = disable the
    /// threshold (incise wherever downhill).
    #[arg(long, default_value_t = -1.0)]
    erosion_channel_support: f32,
    /// Override Jacobi sweeps per implicit diffusion solve. 0 = use the
    /// EROSION_DIFFUSION_ITERS default. Sweep to check the finest cells aren't
    /// under-converged (speckle).
    #[arg(long, default_value_t = 0)]
    erosion_diffusion_iters: usize,
    /// Override the drainage re-route interval (steps between re-routings).
    /// 0 = use the EROSION_REROUTE_INTERVAL default; 1 = re-route every step.
    /// Set to 1 to test whether stale routing is driving the spiral/perforation
    /// artifacts (see docs/archive/specs/erosion.md "Roughness counters").
    #[arg(long, default_value_t = 0)]
    erosion_reroute_interval: usize,
    /// Barnes convergent flat resolution (Rung 1). -1 = use the
    /// EROSION_FLAT_RESOLUTION default; 0 = off (old flood_parent wavefront);
    /// 1 = on. A/B the spiral-on-flats fix (docs/archive/specs/erosion-routing-ladder.md).
    #[arg(long, default_value_t = -1)]
    erosion_flat_resolution: i8,
    /// MFD drainage-area exponent (Rung 2). <0 = use EROSION_MFD_EXPONENT; 0 =
    /// single-flow (SFD); higher = sharper (p→∞ ≈ SFD), lower ≈ 1 = dispersive.
    /// Sweep to A/B multi-flow vs single-flow discharge.
    #[arg(long, default_value_t = -1.0)]
    erosion_mfd_exponent: f32,
    /// Plains alluvial regime gate: channel slope (elev/km) at/above which
    /// incision is full; gentler channels fade to alluvial. <0 = use
    /// EROSION_CONFINEMENT_SLOPE; 0 = off. See docs/archive/specs/erosion-valleys-not-channels.md.
    #[arg(long, default_value_t = -1.0)]
    erosion_confinement_slope: f32,
    /// Override lithologic erodibility contrast (exp-amplitude sigma). <0 = use
    /// the EROSION_LITHO_SIGMA default; 0 = uniform K (no lithology).
    #[arg(long, default_value_t = -1.0)]
    erosion_litho_sigma: f32,
    /// Override tectonic uplift scale (Hold & carve: uplift ~balances erosion to
    /// hold orogens while valleys grade). <0 = use the EROSION_UPLIFT_SCALE
    /// default; 0 = relaxation only (no uplift).
    #[arg(long, default_value_t = -1.0)]
    erosion_uplift_scale: f32,
    /// Emergent builder over-rebuild gain (relief-spectrum candidate B): >1 builds
    /// more orogen volume than the coarse target so erosion carves the excess into
    /// relief. <0 = EMERGENT_REBUILD_GAIN default (1.2). Pair with --erosion-k /
    /// --erosion-hillslope-crit for the joint high-relief regime.
    #[arg(long, default_value_t = -1.0)]
    rebuild_gain: f32,
    /// Override uplift-FORCING smoothing length (km). Escalation #1: smooths the
    /// per-step uplift source over a sub-grid orogenic width to kill mountain-top
    /// cell-scale chatter without flattening orogens. <0 = use
    /// EROSION_UPLIFT_SMOOTH_KM default; 0 = off. See
    /// docs/archive/specs/erosion-uplift-smoothing.md.
    #[arg(long, default_value_t = -1.0)]
    erosion_uplift_smooth: f32,
    /// Override Roering nonlinear-hillslope critical slope S_c (escalation #2;
    /// Δelev/radian, ~grade·637). Diffusivity blows up toward S_c -> planar
    /// slopes + crisp ridges. <0 = use EROSION_HILLSLOPE_CRITICAL_SLOPE default;
    /// 0 = off (linear creep). Read curv-rms/peak% as you sweep ~150-300.
    #[arg(long, default_value_t = -1.0)]
    erosion_hillslope_crit: f32,
    /// Override orographic precip modulation strength (climate↔erosion: windward
    /// wetter, lee drier). <0 = use OROGRAPHIC_PRECIP_STRENGTH; 0 = coarse precip.
    #[arg(long, default_value_t = -1.0)]
    erosion_orographic_strength: f32,
    /// Override downwind rain-shadow strength (lee dry-anomaly propagated downwind).
    /// <0 = use DOWNWIND_SHADOW_STRENGTH (0=off).
    #[arg(long, default_value_t = -1.0)]
    erosion_downwind_shadow: f32,
    /// Override coupled erode↔precip feedback passes. 0 = use
    /// EROSION_PRECIP_OUTER_ITERS default; 1 = no erosion feedback.
    #[arg(long, default_value_t = 0)]
    erosion_precip_iters: usize,
    /// Override lakes-as-evaporation precip boost strength. <0 = use
    /// LAKE_EVAP_STRENGTH default; 0 = off.
    #[arg(long, default_value_t = -1.0)]
    erosion_lake_evap: f32,
    /// Override depositional repose slope (en-route aggradation: fans/floodplains/
    /// deltas). <0 = use EROSION_DEPOSITION_SLOPE default; 0 = sink-fill only.
    /// Read the mass ledger (lost-to-ocean should drop) as you raise it.
    #[arg(long, default_value_t = -1.0)]
    erosion_deposition_slope: f32,
    /// Override glacial abrasion coefficient (ice-flux over-deepening). <0 = use
    /// GLACIAL_K default; 0 = no glacial pass. Read the logged glaciated coverage
    /// and abraded volume as you sweep it.
    #[arg(long, default_value_t = -1.0)]
    glacial_k: f32,
    /// Override glacial over-deepening max (reverse-gradient/tarn depth a cell may
    /// carve below its receiver). <0 = use GLACIAL_OVERDEEPEN_MAX default; 0 =
    /// no over-deepening (no closed rock basins). Isolates over-deepening from
    /// SFD ice abrasion as the curv-rms/pit source.
    #[arg(long, default_value_t = -1.0)]
    glacial_overdeepen_max: f32,
    /// Override structural-grain erodibility strength (fold-belt ridge-and-valley).
    /// <0 = use EROSION_LITHO_GRAIN_STRENGTH default; 0 = no grain. Experimental.
    #[arg(long, default_value_t = -1.0)]
    litho_grain_strength: f32,
    /// A4 drainage-pulse dial (meso-a4-drainage-pulse.md): burn-in erode →
    /// trunk/interfluve zero-mean uplift modifier → frozen final epoch. <0 =
    /// default (0=off). Erosion-side only (fine base untouched).
    #[arg(long, default_value_t = -1.0)]
    drainage_pulse: f32,
    /// A4 burn-in epoch steps (drainage topology self-organization). 0 = default (80).
    #[arg(long, default_value_t = 0)]
    pulse_burnin_steps: usize,
    /// A4 trunk-proximity Gaussian sigma, km (meso band; floor ~8 km). <0 = default (15).
    #[arg(long, default_value_t = -1.0)]
    pulse_smooth_km: f32,
    /// Override fault range-front scarp relief (sharpen active orogen margins).
    /// <0 = use FAULT_SCARP_HEIGHT default; 0 = off (smooth fronts).
    #[arg(long, default_value_t = -1.0)]
    fault_scarp: f32,
    /// Override fine interior structural relief amplitude (P1a: mid-band fault/fold
    /// grain that breaks the flat orogen summit). <0 = use FINE_INTERIOR_RELIEF
    /// default; 0 = off (pure interpolant). Regenerates the fine base.
    #[arg(long, default_value_t = -1.0)]
    interior_relief: f32,
    /// Override fine strike-band weight (P1b: how much of the interior grain aligns
    /// to the nearest orogen front vs isotropic). <0 = use FINE_FRONT_STRIKE_WEIGHT
    /// default; 0 = isotropic (P1a). Regenerates the fine base.
    #[arg(long, default_value_t = -1.0)]
    front_strike_weight: f32,
    /// Override fine margin contrast (P1c: sharpen relief on active/convergent coasts,
    /// damp on passive ones). <0 = use FINE_MARGIN_CONTRAST default; 0 = off (P1b).
    /// Regenerates the fine base.
    #[arg(long, default_value_t = -1.0)]
    margin_contrast: f32,
    /// Emergent-orogens demotion fraction (erosion-v3): demote λ·(arc+collision) from
    /// the base and rebuild it by active uplift. <0 = default (0=off); 0.25-0.5 = test.
    /// Pair with a raised --erosion-uplift-scale (~λ/(steps·dt)). NOTE: the painted P1
    /// relief still runs unless you also zero it (--interior-relief 0.005 --fault-scarp 0
    /// etc.) — the `--sweep-stack v3` preset does this for a clean A/B. Regenerates base.
    #[arg(long, default_value_t = -1.0)]
    emergent_lambda: f32,
    /// Slice A: rebuild the fully demoted Legacy convergent budget from finite-age,
    /// frozen exact present-front supports while drainage and hillslopes evolve.
    #[cfg(feature = "research-landscape")]
    #[arg(long, default_value_t = false)]
    finite_age_uplift: bool,
    /// O0 structured emergent uplift (orogen-structure): blend of asymmetric+segmented
    /// uplift shape vs uniform rebuild. <0 = default (0=off); 1 = fully structured.
    /// Needs --emergent-lambda >0 and (for the decisive test) --erosion-n ~2.
    #[arg(long, default_value_t = -1.0)]
    emergent_structured: f32,
    /// Candidate A meso uplift-shape modulation depth. <0 = default (0=off);
    /// 0 = off; 0.3-0.9 = test. Regenerates the fine base.
    #[arg(long, default_value_t = -1.0)]
    meso_relief: f32,
    /// Fold-train irregularity 0..1 (cross-strike decorrelation + 2nd octave +
    /// crest sharpening). <0 = default (0.7); 0 = plain periodic train.
    #[arg(long, default_value_t = -1.0, allow_hyphen_values = true)]
    meso_irregularity: f32,

    /// Meso construction style: 0 = fold train (foreland preset), 1 =
    /// massif-corridor (alpine default). <0 = default (1).
    #[arg(long, default_value_t = -1, allow_hyphen_values = true)]
    meso_style: i32,

    /// Candidate A' meso base-elevation relief amplitude. <0 = default (0=off);
    /// elevation units: 0.01 is about 100 m. Regenerates the fine base.
    #[arg(long, default_value_t = -1.0)]
    meso_base_relief: f32,
    /// Candidate A meso fold-train wavelength in km. <0 = default (25 km).
    /// Regenerates the fine base.
    #[arg(long, default_value_t = -1.0)]
    meso_wavelength_km: f32,
    /// Fine-mesh density knobs (cell-size targets in km / blend). <0 = use the
    /// FINE_* default. Setting any forces fine-base regeneration (no cache). Use
    /// to sweep the ocean/plains/mountain budget: e.g. --fine-plains-km 20.
    #[arg(long, default_value_t = -1.0)]
    fine_plains_km: f32,
    #[arg(long, default_value_t = -1.0)]
    fine_mountain_km: f32,
    #[arg(long, default_value_t = -1.0)]
    fine_ocean_km: f32,
    #[arg(long, default_value_t = -1.0)]
    fine_density_exponent: f32,
    #[arg(long, default_value_t = -1.0)]
    fine_slope_weight: f32,
    #[arg(long, default_value_t = -1.0)]
    fine_flow_weight: f32,
    #[arg(long, default_value_t = -1.0)]
    fine_activity_weight: f32,
    /// Uniform multiplier on the fine-mesh cell-size targets (plains/mountain/
    /// ocean km) — one knob to sweep fine-mesh resolution. >1 coarsens, <1
    /// refines. Forces fine-base regeneration (no cache). Use with the
    /// "Fine-scale local relief" convergence probe.
    #[arg(long, default_value_t = 1.0)]
    fine_scale: f32,
}

fn main() {
    env_logger::init();
    let cli = Cli::parse();

    eprintln!(
        "Generating world (seed={}, cells={}, orogen={:?})...",
        cli.seed, cli.cells, cli.orogen_model
    );
    let mut world = World::new(cli.seed, cli.cells, 1);
    world.orogen_model = match cli.orogen_model {
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
    };
    world.generate_plates(hex3::world::NUM_PLATES_DEFAULT);
    world.generate_crust();
    world.generate_dynamics();
    world.generate_features();
    world.generate_elevation();
    if cli.tectonic_history_audit {
        eprintln!("provenance: {}", world.manifest().summary());
        run_tectonic_history_audit(&world, cli.seed, cli.top);
        #[cfg(feature = "research-landscape")]
        run_source_viability_audit(
            &world,
            cli.seed,
            cli.cells,
            cli.tectonic_history_audit_out.as_deref(),
        );
        return;
    }
    world.generate_atmosphere();
    #[cfg(feature = "research-landscape")]
    if cli.finite_age_uplift
        || cli.finite_age_component_audit
        || cli.finite_age_spatial_episode.is_some()
    {
        world.erosion_params.finite_age_uplift = true;
        world.fine_structure_params.emergent_lambda = 1.0;
        world.fine_structure_params.emergent_structured = 0.0;
        world.fine_structure_params.interior_relief = 0.0;
        world.fine_structure_params.front_strike_weight = 0.0;
        world.fine_structure_params.margin_contrast = 0.0;
        world.fine_structure_params.meso_relief = 0.0;
        world.fine_structure_params.meso_base_relief = 0.0;
        world.fine_structure_params.fault_scarp_height = 0.0;
        world.erosion_params.litho_sigma = 0.0;
        world.erosion_params.litho_grain_strength = 0.0;
        world.erosion_params.drainage_pulse = 0.0;
        world.erosion_params.glacial_k = 0.0;
        world.erosion_params.hillslope_critical_slope = 200.0;
    }
    if cli.erosion_k >= 0.0 {
        world.erosion_params.k = cli.erosion_k;
    }
    if cli.erosion_n >= 0.0 {
        world.erosion_params.n = cli.erosion_n;
    }
    if cli.erosion_steps > 0 {
        world.erosion_params.steps = cli.erosion_steps;
    }
    if cli.erosion_diffusivity >= 0.0 {
        world.erosion_params.diffusivity = cli.erosion_diffusivity;
    }
    if cli.erosion_channel_support >= 0.0 {
        world.erosion_params.channel_support_km2 = cli.erosion_channel_support;
    }
    if cli.erosion_diffusion_iters > 0 {
        world.erosion_params.diffusion_iters = cli.erosion_diffusion_iters;
    }
    if cli.erosion_reroute_interval > 0 {
        world.erosion_params.reroute_interval = cli.erosion_reroute_interval;
    }
    if cli.erosion_flat_resolution >= 0 {
        world.erosion_params.flat_resolution = cli.erosion_flat_resolution != 0;
    }
    if cli.erosion_mfd_exponent >= 0.0 {
        world.erosion_params.mfd_exponent = cli.erosion_mfd_exponent;
    }
    if cli.erosion_confinement_slope >= 0.0 {
        world.erosion_params.confinement_slope = cli.erosion_confinement_slope;
    }
    if cli.erosion_litho_sigma >= 0.0 {
        world.erosion_params.litho_sigma = cli.erosion_litho_sigma;
    }
    if cli.erosion_uplift_scale >= 0.0 {
        world.erosion_params.uplift_scale = cli.erosion_uplift_scale;
    }
    if cli.rebuild_gain >= 0.0 {
        world.erosion_params.rebuild_gain = cli.rebuild_gain;
    }
    if cli.erosion_uplift_smooth >= 0.0 {
        world.erosion_params.uplift_smooth_km = cli.erosion_uplift_smooth;
    }
    if cli.erosion_hillslope_crit >= 0.0 {
        world.erosion_params.hillslope_critical_slope = cli.erosion_hillslope_crit;
    }
    if cli.erosion_orographic_strength >= 0.0 {
        world.erosion_params.orographic_precip_strength = cli.erosion_orographic_strength;
    }
    if cli.erosion_downwind_shadow >= 0.0 {
        world.erosion_params.downwind_shadow_strength = cli.erosion_downwind_shadow;
    }
    if cli.erosion_precip_iters > 0 {
        world.erosion_params.precip_outer_iters = cli.erosion_precip_iters;
    }
    if cli.erosion_lake_evap >= 0.0 {
        world.erosion_params.lake_evap_strength = cli.erosion_lake_evap;
    }
    if cli.erosion_deposition_slope >= 0.0 {
        world.erosion_params.deposition_slope = cli.erosion_deposition_slope;
    }
    if cli.glacial_overdeepen_max >= 0.0 {
        world.erosion_params.glacial_overdeepen_max = cli.glacial_overdeepen_max;
    }
    if cli.glacial_k >= 0.0 {
        world.erosion_params.glacial_k = cli.glacial_k;
    }
    if cli.litho_grain_strength >= 0.0 {
        world.erosion_params.litho_grain_strength = cli.litho_grain_strength;
    }
    if cli.drainage_pulse >= 0.0 {
        world.erosion_params.drainage_pulse = cli.drainage_pulse;
    }
    if cli.pulse_burnin_steps > 0 {
        world.erosion_params.pulse_burnin_steps = cli.pulse_burnin_steps;
    }
    if cli.pulse_smooth_km >= 0.0 {
        world.erosion_params.pulse_smooth_km = cli.pulse_smooth_km;
    }
    if cli.fault_scarp >= 0.0 {
        world.fine_structure_params.fault_scarp_height = cli.fault_scarp;
    }
    if cli.interior_relief >= 0.0 {
        world.fine_structure_params.interior_relief = cli.interior_relief;
    }
    if cli.front_strike_weight >= 0.0 {
        world.fine_structure_params.front_strike_weight = cli.front_strike_weight;
    }
    if cli.margin_contrast >= 0.0 {
        world.fine_structure_params.margin_contrast = cli.margin_contrast;
    }
    if cli.emergent_lambda >= 0.0 {
        world.fine_structure_params.emergent_lambda = cli.emergent_lambda;
    }
    if cli.emergent_structured >= 0.0 {
        world.fine_structure_params.emergent_structured = cli.emergent_structured;
    }
    if cli.meso_relief >= 0.0 {
        world.fine_structure_params.meso_relief = cli.meso_relief;
    }
    if cli.meso_irregularity >= 0.0 {
        world.fine_structure_params.meso_irregularity = cli.meso_irregularity;
    }
    if cli.meso_style >= 0 {
        world.fine_structure_params.meso_style = cli.meso_style as usize;
    }
    if cli.meso_base_relief >= 0.0 {
        world.fine_structure_params.meso_base_relief = cli.meso_base_relief;
    }
    if cli.meso_wavelength_km >= 0.0 {
        world.fine_structure_params.meso_wavelength_km = cli.meso_wavelength_km;
    }
    // Fine-mesh density overrides (force regeneration so the cache can't serve a
    // base built with the default knobs).
    let mut dp = world.fine_density_params;
    let overrides = [
        (&mut dp.plains_km, cli.fine_plains_km),
        (&mut dp.mountain_km, cli.fine_mountain_km),
        (&mut dp.ocean_km, cli.fine_ocean_km),
        (&mut dp.exponent, cli.fine_density_exponent),
        (&mut dp.slope_weight, cli.fine_slope_weight),
        (&mut dp.flow_weight, cli.fine_flow_weight),
        (&mut dp.activity_weight, cli.fine_activity_weight),
    ];
    let mut density_overridden = false;
    for (target, v) in overrides {
        if v >= 0.0 {
            *target = v;
            density_overridden = true;
        }
    }
    // Uniform fine-resolution multiplier, applied after any explicit per-class
    // overrides so it scales whatever budget is in effect.
    if (cli.fine_scale - 1.0).abs() > f32::EPSILON {
        dp.plains_km *= cli.fine_scale;
        dp.mountain_km *= cli.fine_scale;
        dp.ocean_km *= cli.fine_scale;
        density_overridden = true;
    }
    if density_overridden {
        world.fine_density_params = dp;
        world.fine_cache = hex3::world::FineCacheMode::Disabled;
        eprintln!(
            "fine density override: plains={:.1} mountain={:.1} ocean={:.1} km, exp={:.1}, weights slope/flow/activity={:.1}/{:.1}/{:.1}",
            dp.plains_km, dp.mountain_km, dp.ocean_km, dp.exponent, dp.slope_weight, dp.flow_weight, dp.activity_weight
        );
    }

    if cli.resolution_pilot_audit {
        let native_max = if cli.fine_max == 0 {
            1_000_000
        } else {
            cli.fine_max
        };
        if cli.pilot_max == 0 || cli.pilot_steps == 0 {
            eprintln!("--pilot-max and --pilot-steps must both be greater than zero");
            std::process::exit(2);
        }
        if native_max <= cli.pilot_max {
            eprintln!(
                "--fine-max ({native_max}) must be greater than --pilot-max ({}) for --resolution-pilot-audit",
                cli.pilot_max
            );
            std::process::exit(2);
        }
        run_resolution_pilot_audit(
            &mut world,
            cli.seed,
            cli.pilot_max,
            native_max,
            cli.pilot_steps,
        );
        return;
    }

    if cli.fine_max > 0 {
        world.generate_hydrology_with_fine_cap(cli.fine_max);
    } else {
        world.generate_hydrology();
    }
    eprintln!("provenance: {}", world.manifest().summary());

    // Feature audits are combinable (one world generation, several panels).
    let mut audited = false;
    if cli.drainage_audit {
        run_drainage_audit(&world, cli.seed);
        audited = true;
    }
    if cli.lake_audit {
        run_lake_audit(&mut world, cli.seed, cli.top);
        audited = true;
    }
    if cli.mountain_audit {
        run_mountain_audit(&world, cli.seed, cli.top);
        run_roughness_probe(&world);
        audited = true;
    }
    #[cfg(feature = "research-landscape")]
    if cli.roof_causal_trace {
        run_roof_causal_trace(
            &world,
            cli.seed,
            cli.cells,
            cli.roof_causal_trace_out.as_deref(),
        );
        audited = true;
    }
    #[cfg(feature = "research-landscape")]
    if cli.finite_age_component_audit {
        run_finite_age_component_audit(
            &world,
            cli.seed,
            cli.cells,
            cli.top,
            cli.finite_age_component_audit_out.as_deref(),
        );
        audited = true;
    }
    #[cfg(feature = "research-landscape")]
    if let Some(episode_id) = cli.finite_age_spatial_episode {
        run_finite_age_spatial_trace(
            &world,
            cli.seed,
            cli.cells,
            episode_id,
            cli.finite_age_spatial_trace_out.as_deref(),
        );
        audited = true;
    }
    if cli.rebuild_fidelity_audit {
        run_rebuild_fidelity_audit(&world, cli.seed, cli.top);
        audited = true;
    }
    if cli.detail_survival_audit {
        run_detail_survival_audit(&world, cli.seed);
        audited = true;
    }
    if cli.river_audit {
        run_river_audit(&world, cli.seed, cli.top);
        audited = true;
    }
    if cli.biome_audit {
        run_biome_audit(&world, cli.seed);
        audited = true;
    }
    if audited {
        return;
    }

    let tess = world.active_tessellation();
    let n = tess.num_cells();
    let crust = world.crust.as_ref().unwrap();
    let features = world.features.as_ref().unwrap();
    let elevation = &world.active_elevation().unwrap().values;
    let hydrology = world.active_hydrology().unwrap();
    let temperature = world.active_temperature().unwrap();
    let precipitation = world.active_precipitation().unwrap();
    let uplift = world.active_uplift().unwrap();
    let coarse_of = |i: usize| -> usize {
        world
            .fine
            .as_ref()
            .map(|fine| fine.coarse_cell()[i])
            .unwrap_or(i)
    };
    let cont_mask: Vec<bool> = (0..n)
        .map(|i| {
            world
                .fine
                .as_ref()
                .map(|fine| fine.fields().elevation_fields.continentality[i] >= 0.5)
                .unwrap_or_else(|| crust.is_continental(i))
        })
        .collect();
    let margin_distance: Vec<f32> = (0..n)
        .map(|i| {
            world
                .fine
                .as_ref()
                .map(|fine| {
                    let cont = fine.fields().elevation_fields.continentality[i];
                    (cont - 0.5).abs() * 0.1
                })
                .unwrap_or_else(|| crust.margin_distance(i))
        })
        .collect();
    let feature_divergent: Vec<f32> = (0..n).map(|i| features.divergent[coarse_of(i)]).collect();
    let feature_collision: Vec<f32> = (0..n).map(|i| features.collision[coarse_of(i)]).collect();
    let feature_arc: Vec<f32> = (0..n).map(|i| features.arc[coarse_of(i)]).collect();
    let feature_trench: Vec<f32> = world
        .fine
        .as_ref()
        .map(|fine| fine.fields().elevation_fields.trench.clone())
        .unwrap_or_else(|| features.trench.clone());
    let feature_ridge_age: Vec<f32> = world
        .fine
        .as_ref()
        .map(|fine| fine.fields().elevation_fields.ridge_age_distance.clone())
        .unwrap_or_else(|| features.ridge_age_distance.clone());

    let cell_km2 = tess.mean_cell_area() * EARTH_RADIUS_KM * EARTH_RADIUS_KM;
    println!(
        "\n================ WORLD DIAGNOSTICS seed={} cells={} ================",
        cli.seed, n
    );
    println!(
        "resolution: {:.0} km²/cell (~{:.0} km spacing) — features below ~{:.0} km² are unresolvable",
        cell_km2,
        (tess.mean_cell_area()).sqrt() * EARTH_RADIUS_KM,
        cell_km2
    );
    if let Some(fine) = &world.fine {
        println!(
            "fine mesh: coarse {} -> fine {} cells | density ratio {:.1}:1",
            world.tessellation.num_cells(),
            fine.tessellation().num_cells(),
            fine.achieved_density_ratio()
        );
        run_roughness_probe(&world);

        let ftess = fine.tessellation();

        // ---- Carved dissection density (the "too busy" calibration target) ----
        // Incision = pre-erosion base minus eroded (meters). A "carved" cell is
        // incised past a depth threshold (a real valley, not surface noise); the
        // density of carved cells (channel km / km² land) and spacing ~1/Dd index
        // how busy the dissection is — and UNLIKE the hydrology network this
        // responds to channel-support / K. Resolution + threshold dependent, so
        // read vs Earth as a ballpark + a knob-response signal, not a hard number.
        const M_PER_UNIT: f32 = ELEVATION_UNIT_KM * 1_000.0;
        let base = &fine.surface_for(3).elevation.values;
        let eroded = &fine.surface_for(4).elevation.values;
        let fareas = ftess.cell_areas();
        let mut land_area_km2 = 0.0f64;
        let mut depths: Vec<f32> = Vec::new(); // land incision depth (m, clamped >=0)
        let mut carved: Vec<(f32, f32)> = Vec::new(); // (depth_m, width_km) for incised
        for i in 0..ftess.num_cells() {
            if eroded[i] < 0.0 {
                continue; // ocean
            }
            land_area_km2 += (fareas[i] * EARTH_RADIUS_KM * EARTH_RADIUS_KM) as f64;
            let inc = (base[i] - eroded[i]) * M_PER_UNIT;
            depths.push(inc.max(0.0));
            if inc > 0.0 {
                carved.push((inc, fareas[i].sqrt() * EARTH_RADIUS_KM));
            }
        }
        depths.sort_by(f32::total_cmp);
        let nland = depths.len().max(1);
        let dq = |p: f32| depths[(((nland - 1) as f32) * p) as usize];
        println!(
            "\n-- Carved dissection (incision = base - eroded)  [Earth land Dd ~0.5-5 km/km², spacing ~0.2-2 km] --"
        );
        println!(
            "  incision depth (land): p50 {:>4.0} m | p90 {:>4.0} m | p99 {:>5.0} m | max {:>5.0} m",
            dq(0.50),
            dq(0.90),
            dq(0.99),
            dq(1.0),
        );
        for thr in [30.0f32, 100.0, 300.0] {
            let len_km: f64 = carved
                .iter()
                .filter(|&&(d, _)| d >= thr)
                .map(|&(_, w)| w as f64)
                .sum();
            let cnt = carved.iter().filter(|&&(d, _)| d >= thr).count();
            let dd = if land_area_km2 > 0.0 {
                len_km / land_area_km2
            } else {
                0.0
            };
            let spacing = if dd > 0.0 { 1.0 / dd } else { f64::INFINITY };
            println!(
                "  carved >{:>3.0} m: {:>5.1}% of land | Dd ~{:.3} km/km² | spacing ~{:.2} km",
                thr,
                100.0 * cnt as f32 / nland as f32,
                dd,
                spacing,
            );
        }
    }

    // ---- Global elevation structure ----
    let land: Vec<bool> = elevation.iter().map(|&e| e >= 0.0).collect();
    let land_frac = land.iter().filter(|&&l| l).count() as f32 / n as f32;
    let mut sorted: Vec<f32> = elevation.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let pct = |p: f32| sorted[((p * (n - 1) as f32) as usize).min(n - 1)];
    println!("\n-- Global --");
    println!(
        "land {:.1}% | elevation p5 {:+.3} p50 {:+.3} p95 {:+.3} max {:+.3}",
        100.0 * land_frac,
        pct(0.05),
        pct(0.50),
        pct(0.95),
        sorted[n - 1]
    );
    println!(
        "field smoothness (Moran's I): elevation {:.3}, precipitation {:.3}, uplift {:.3}",
        tess.morans_i(elevation),
        tess.morans_i(precipitation),
        tess.morans_i(uplift)
    );
    let land_temps: Vec<f32> = (0..n)
        .filter(|&i| elevation[i] >= 0.0)
        .map(|i| temperature[i])
        .collect();
    let ocean_temps: Vec<f32> = (0..n)
        .filter(|&i| elevation[i] < 0.0)
        .map(|i| temperature[i])
        .collect();
    let mean = |values: &[f32]| {
        if values.is_empty() {
            0.0
        } else {
            values.iter().sum::<f32>() / values.len() as f32
        }
    };
    let land_temp_mean = mean(&land_temps);
    let ocean_temp_mean = mean(&ocean_temps);
    println!(
        "thermal contrast: land mean {:+.3}, ocean mean {:+.3}, delta {:+.3}",
        land_temp_mean,
        ocean_temp_mean,
        land_temp_mean - ocean_temp_mean
    );

    // ---- Continents (connected continental crust) ----
    let continents = measure_components(tess, &cont_mask);
    println!("\n-- Continents (connected continental crust) --   [Earth: Afro-Eurasia 85M km², Americas 42M, Australia 7.7M, Greenland 2.1M]");
    for c in continents.iter().take(cli.top) {
        let submerged = c.fraction_where(|i| elevation[i] < 0.0);
        println!(
            "  {:>10.0} km²  extent {:>5.0} km  submerged {:>4.1}%",
            c.area_km2,
            c.length_km,
            100.0 * submerged
        );
    }
    println!("  ({} total)", continents.len());

    // ---- Interior continental water (rift seaways, inland seas, lakes) ----
    let interior_water: Vec<bool> = (0..n)
        .map(|i| {
            cont_mask[i]
                && margin_distance[i] > 0.03
                && matches!(
                    hydrology.water_state(i),
                    CellWaterState::Ocean | CellWaterState::LakeWater
                )
        })
        .collect();
    let waters = measure_components(tess, &interior_water);
    println!("\n-- Interior continental water (>~190km from margin) --   [Earth: Caspian 371k km², Black Sea 436k, Red Sea 438k (2250x355 km), Tanganyika 33k (670x50), Baikal 32k]");
    for w in waters.iter().take(cli.top) {
        let connected_to_ocean =
            w.fraction_where(|i| hydrology.water_state(i) == CellWaterState::Ocean);
        let mean_div = w.mean_of(&feature_divergent);
        let (min_e, _) = w.range_of(elevation);
        println!(
            "  {:>9.0} km²  {:>5.0} x {:>4.0} km (elong {:>4.1})  depth_max {:>5.2}  divergent {:.2}  {}",
            w.area_km2,
            w.length_km,
            w.width_km,
            w.elongation(),
            -min_e,
            mean_div,
            if connected_to_ocean > 0.5 { "SEAWAY (ocean-connected)" } else { "landlocked" }
        );
    }
    let total_water_km2: f32 = waters.iter().map(|w| w.area_km2).sum();
    println!(
        "  ({} bodies, {:.0} km² total = {:.1}% of continental crust)",
        waters.len(),
        total_water_km2,
        100.0 * total_water_km2 / (continents.iter().map(|c| c.area_km2).sum::<f32>()).max(1.0)
    );

    // ---- Islands (oceanic-crust land) ----
    let island_mask: Vec<bool> = (0..n)
        .map(|i| !cont_mask[i] && elevation[i] >= 0.0)
        .collect();
    let islands = measure_components(tess, &island_mask);
    println!("\n-- Islands (oceanic-crust land) --   [Earth: Greenland 2.1M km² (continental), Honshu 228k, Iceland 103k, Hawaii Big Island 10k]");
    for isl in islands.iter().take(cli.top.min(5)) {
        let (_, max_e) = isl.range_of(elevation);
        println!(
            "  {:>8.0} km²  extent {:>5.0} km  peak {:+.2}",
            isl.area_km2, isl.length_km, max_e
        );
    }
    let single_cell = islands.iter().filter(|i| i.cells.len() == 1).count();
    println!(
        "  ({} islands, {} of them single-cell)",
        islands.len(),
        single_cell
    );

    // ---- Mountain ranges (land above threshold) ----
    const RANGE_ELEV: f32 = 0.15;
    let range_mask: Vec<bool> = (0..n).map(|i| elevation[i] >= RANGE_ELEV).collect();
    let ranges = measure_components(tess, &range_mask);
    println!("\n-- Mountain ranges (elevation > {RANGE_ELEV}) --   [Earth: Andes 7000x300 km, Himalaya+Tibet 2400x1000, Alps 1200x200]");
    for r in ranges.iter().take(cli.top) {
        let (_, peak) = r.range_of(elevation);
        let collision_frac = r.fraction_where(|i| feature_collision[i] > 0.02);
        let arc_frac = r.fraction_where(|i| feature_arc[i] > 0.02);
        println!(
            "  {:>8.0} km²  {:>5.0} x {:>4.0} km  peak {:+.2}  driver: collision {:>3.0}% arc {:>3.0}%",
            r.area_km2,
            r.length_km,
            r.width_km,
            peak,
            100.0 * collision_frac,
            100.0 * arc_frac
        );
    }
    println!("  ({} ranges)", ranges.len());

    // ---- Arc-trench gap ----
    let trench_peak = feature_trench.iter().cloned().fold(0.0f32, f32::max);
    let arc_peak = feature_arc.iter().cloned().fold(0.0f32, f32::max);
    if trench_peak > 0.0 && arc_peak > 0.0 {
        let trench_mask: Vec<bool> = feature_trench
            .iter()
            .map(|&t| t > 0.3 * trench_peak)
            .collect();
        let dist = distance_from_mask(tess, &trench_mask);
        let mut gaps: Vec<f32> = (0..n)
            .filter(|&i| feature_arc[i] > 0.5 * arc_peak && dist[i].is_finite())
            .map(|i| dist[i] * EARTH_RADIUS_KM)
            .collect();
        if !gaps.is_empty() {
            gaps.sort_by(|a, b| a.total_cmp(b));
            println!(
                "\n-- Arc-trench gap (arc crest cells to nearest trench) --   [Earth: 100-250 km]"
            );
            println!(
                "  p25 {:>4.0} km  p50 {:>4.0} km  p75 {:>4.0} km",
                gaps[gaps.len() / 4],
                gaps[gaps.len() / 2],
                gaps[3 * gaps.len() / 4]
            );
        }
    }

    // ---- Flexure profile ----
    let deepest_deflection = feature_trench.iter().cloned().fold(0.0f32, f32::max);
    let strongest_outer_rise = -feature_trench.iter().cloned().fold(0.0f32, f32::min);
    let flexure_ratio = if deepest_deflection > 0.0 {
        strongest_outer_rise / deepest_deflection
    } else {
        0.0
    };
    let outer_rise_cells = feature_trench.iter().filter(|&&t| t < 0.0).count();
    println!(
        "\n-- Flexure profile --   [Earth: outer rise ~200-500 m vs trenches 2-8 km -> ~0.05]"
    );
    println!(
        "  deepest deflection {:.3} | strongest outer rise {:.3} | ratio {:.3} | outer-rise cells {}",
        deepest_deflection, strongest_outer_rise, flexure_ratio, outer_rise_cells
    );
    let mut ridge_age: Vec<f32> = feature_ridge_age
        .iter()
        .copied()
        .filter(|d| d.is_finite())
        .collect();
    if !ridge_age.is_empty() {
        ridge_age.sort_by(|a, b| a.total_cmp(b));
        println!(
            "  ridge age-distance rad: min {:.3} | median {:.3} | max {:.3}",
            ridge_age[0],
            ridge_age[ridge_age.len() / 2],
            ridge_age[ridge_age.len() - 1]
        );
    }

    // ---- Rivers ----
    let max_flow = hydrology
        .flow_accumulation
        .iter()
        .cloned()
        .fold(0.0f32, f32::max);
    let big_rivers = hydrology
        .flow_accumulation
        .iter()
        .zip(land.iter())
        .filter(|(&f, &l)| l && f > 0.01 * max_flow)
        .count();
    println!("\n-- Rivers --");
    println!(
        "  max flow {:.0} cell-equivalents | land cells carrying >1% of max: {}",
        max_flow, big_rivers
    );

    // ---- Climate ----
    // Aridity index AI = P / PET (precip ÷ evaporative demand), the Earth-standard
    // way to define arid — NOT raw precip. PET rises with temperature (warm air
    // evaporates more), so a cold low-precip region (poles) is NOT arid the way a
    // hot low-precip one (subtropics) is. PET proxy: temperature in ~[0,1] mapped
    // to a 0.2..1.0 demand (same shape as the moisture model's carrying_capacity).
    // AI normalized to land-mean 1 so the threshold is a relative water-stress
    // cutoff; report the ZONAL AI pattern too (that's the threshold-robust signal).
    let pet = |i: usize| 0.2 + 0.8 * temperature[i].clamp(0.0, 1.0);
    let ai_raw: Vec<f32> = (0..n)
        .map(|i| precipitation[i] / pet(i).max(1e-6))
        .collect();
    let land_idx: Vec<usize> = (0..n).filter(|&i| land[i]).collect();
    let ai_mean = land_idx.iter().map(|&i| ai_raw[i]).sum::<f32>() / land_idx.len().max(1) as f32;
    let ai: Vec<f32> = ai_raw.iter().map(|&v| v / ai_mean.max(1e-9)).collect();
    let frac = |pred: &dyn Fn(usize) -> bool| -> f32 {
        100.0 * land_idx.iter().filter(|&&i| pred(i)).count() as f32 / land_idx.len().max(1) as f32
    };
    // Raw-precip arid (the OLD metric) for comparison, then the AI-based one.
    let arid_precip = frac(&|i| precipitation[i] < 0.35);
    let arid_ai = frac(&|i| ai[i] < 0.4); // hot+dry water stress (relative threshold)
    let humid_ai = frac(&|i| ai[i] > 2.0);
    println!(
        "\n-- Climate --   [arid by P/PET (aridity index), not raw P — Earth arid+semiarid ~33%]"
    );
    println!(
        "  aridity index (P/PET, land-mean 1): arid(AI<0.4) {:.0}%  humid(AI>2) {:.0}%  | raw-precip<0.35 {:.0}% (old, temp-blind) | lakes {:.2}%",
        arid_ai,
        humid_ai,
        arid_precip,
        100.0
            * hydrology
                .water_bodies
                .iter()
                .map(|wb| wb.cells.len())
                .sum::<usize>() as f32
            / n as f32
    );
    // Zonal aridity index (mean AI per 15° band): deserts are LOW-AI bands; on a
    // physical planet they sit in the hot subtropics, not the cold poles.
    {
        const BW: f32 = 15.0;
        let nb = (180.0 / BW) as usize;
        let (mut s, mut c) = (vec![0.0f64; nb], vec![0u32; nb]);
        for &i in &land_idx {
            let lat = tess.cell_center(i).y.clamp(-1.0, 1.0).asin().to_degrees();
            let b = (((lat + 90.0) / BW) as usize).min(nb - 1);
            s[b] += ai[i] as f64;
            c[b] += 1;
        }
        let mut line = String::new();
        for b in (0..nb).rev() {
            if c[b] == 0 {
                continue;
            }
            let lo = -90.0 + b as f32 * BW;
            line.push_str(&format!(
                "  [{:+.0}] {:.2}",
                lo + BW * 0.5,
                s[b] / c[b] as f64
            ));
        }
        println!("  zonal aridity index (AI, mid-band lat):{line}");
    }
    let mean_evap = if hydrology.basins.is_empty() {
        1.0
    } else {
        hydrology
            .basins
            .iter()
            .map(|b| b.evaporation_factor)
            .sum::<f32>()
            / hydrology.basins.len() as f32
    };
    println!(
        "  basin evaporation: mean factor {:.2} across {} basins",
        mean_evap,
        hydrology.basins.len()
    );

    // ---- Zonal precipitation profile (latitude bands) ----
    // Aggregate arid% hides WHERE the dryness is. A physical climate has wet
    // bands at the equator (ITCZ ascent) and mid-latitudes, and DRY bands in the
    // subtropics (~15-35°, Hadley descent). Mean LAND precip per 15° band shows
    // whether subtropical dry bands exist (the H9 subsidence-suppression test).
    {
        const BW: f32 = 15.0; // band width, degrees
        let nb = (180.0 / BW) as usize;
        let mut sum = vec![0.0f64; nb];
        let mut cnt = vec![0u32; nb];
        for i in 0..n {
            if !land[i] {
                continue;
            }
            let lat = tess.cell_center(i).y.clamp(-1.0, 1.0).asin().to_degrees();
            let b = (((lat + 90.0) / BW) as usize).min(nb - 1);
            sum[b] += precipitation[i] as f64;
            cnt[b] += 1;
        }
        println!("\n-- Zonal land precip (mean per 15° lat band)  [wet: eq + midlat; dry: subtropics ~15-35°] --");
        let mut line = String::new();
        for b in (0..nb).rev() {
            if cnt[b] == 0 {
                continue;
            }
            let lo = -90.0 + b as f32 * BW;
            let mean = (sum[b] / cnt[b] as f64) as f32;
            line.push_str(&format!("  [{:+.0}..{:+.0}] {:.2}", lo, lo + BW, mean));
        }
        println!("{line}");
    }

    // ---- River concavity (population slope-area) — UNRELIABLE, see note ----
    // WARNING (2026-06-15): this population slope-area theta is NOT trustworthy on
    // this adaptive mesh. It returns flat/negative theta (~0) even when the river
    // long profiles are demonstrably concave-up/graded, and swings wildly (+0.8 to
    // -0.2) under small parameter changes. Causes: flow_accumulation routes through
    // filled/non-overflowing lakes (artificial drainage area), single-edge drop/dist
    // is noisy on irregular Voronoi cells, and the channel population mixes regimes.
    // It sent a whole erosion investigation chasing a phantom "under-graded" bug
    // that did not exist (the aggregate long-profile probe below showed rivers ARE
    // graded). KEPT only as a documented negative result; judge grading by the
    // "River grading (aggregate ...)" probe, not this theta. See
    // docs/erosion-validation-2026-06-15.md.
    // Detachment-limited stream power at steady state gives S ~ A^(-theta),
    // theta = m/n (~0.5). Population method: over channel cells (flow > support,
    // draining downhill) take S = drop/dist, A = flow, bin by ln(A), regress
    // median ln(S) per bin -> theta = -slope.
    {
        // Count-equivalent flow (hydrology stores physical precip×area discharge),
        // so the absolute channel-support threshold stays in upstream-cell units.
        let flow: Vec<f32> = (0..n).map(|i| hydrology.flow_count_equiv(i)).collect();
        let drainage = &hydrology.drainage_dir;
        let channel_thresh = 50.0f32;
        // Exclude cells whose flow_accumulation is contaminated by lake routing:
        // lake-water cells, and cells inside a non-overflowing (terminal) basin
        // whose drainage is captured by the lake rather than reaching the ocean.
        // flow_accumulation routes through filled basins, so without this the
        // slope-area population mixes real channels with lake-spill artifacts.
        let in_terminal_basin = |c: usize| -> bool {
            match hydrology.basin_id[c] {
                Some(b) => !hydrology.basins[b].is_overflowing(),
                None => false,
            }
        };
        let is_lake = |c: usize| hydrology.water_state(c) == CellWaterState::LakeWater;
        let mut pts: Vec<(f32, f32)> = Vec::new(); // (ln A, ln S)
        let mut skipped_lake = 0usize;
        for c in 0..n {
            if !land[c] || flow[c] < channel_thresh {
                continue;
            }
            if is_lake(c) || in_terminal_basin(c) {
                skipped_lake += 1;
                continue;
            }
            let Some(d) = drainage[c] else { continue };
            // Receiver in a lake: the "slope" is to a lake surface, not a graded
            // channel — drop it.
            if is_lake(d) {
                skipped_lake += 1;
                continue;
            }
            let dz = elevation[c] - elevation[d];
            let dx = (tess.cell_center(c) - tess.cell_center(d)).length() * EARTH_RADIUS_KM;
            if dz <= 0.0 || dx <= 0.0 {
                continue;
            }
            pts.push((flow[c].ln(), (dz / dx).ln()));
        }
        if skipped_lake > 0 {
            println!(
                "  (excluded {} lake/terminal-basin channel cells from slope-area)",
                skipped_lake
            );
        }
        println!("\n-- River concavity (slope-area)  [stream-power theta=m/n ~0.5, concave-up] --");
        if pts.len() < 50 {
            println!("  too few channel cells ({}) to fit", pts.len());
        } else {
            let (lo, hi) = pts
                .iter()
                .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &(x, _)| {
                    (lo.min(x), hi.max(x))
                });
            const NB: usize = 12;
            let mut bins: Vec<Vec<f32>> = vec![Vec::new(); NB];
            for &(x, y) in &pts {
                let t = ((x - lo) / (hi - lo) * NB as f32).floor() as usize;
                bins[t.min(NB - 1)].push(y);
            }
            // Median ln(S) per bin -> (bin-center ln A, median ln S).
            let (mut sx, mut sy, mut sxx, mut sxy, mut k) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0u32);
            for (bi, b) in bins.iter_mut().enumerate() {
                if b.len() < 5 {
                    continue;
                }
                b.sort_by(f32::total_cmp);
                let med = b[b.len() / 2] as f64;
                let xc = (lo + (bi as f32 + 0.5) / NB as f32 * (hi - lo)) as f64;
                sx += xc;
                sy += med;
                sxx += xc * xc;
                sxy += xc * med;
                k += 1;
            }
            let theta = if k >= 3 {
                -((k as f64 * sxy - sx * sy) / (k as f64 * sxx - sx * sx)) as f32
            } else {
                f32::NAN
            };
            println!(
                "  channel cells {} | bins fitted {} | theta = {:+.2}  (UNRELIABLE on this mesh — judge grading by the aggregate long-profile probe below, not this)",
                pts.len(),
                k,
                theta,
            );
        }
    }

    // ---- River grading: aggregate long-profile concavity (ground truth) ----
    // A SINGLE river's profile is too sensitive to which head gets picked (and the
    // population slope-area theta is noisy on this mesh), so aggregate over the N
    // largest mountain rivers. Trace each (highest unused channel head -> sea),
    // measure its concavity, and report the MEDIAN: a graded landscape has rivers
    // that are concave-up (steep source, gentle mouth) — negative normalized bow,
    // source/mouth slope ratio > 1. This cancels single-river selection noise.
    {
        let cflow: Vec<f32> = (0..n).map(|i| hydrology.flow_count_equiv(i)).collect();
        let mut heads: Vec<usize> = (0..n).filter(|&i| land[i] && cflow[i] > 100.0).collect();
        heads.sort_by(|&a, &b| elevation[b].total_cmp(&elevation[a])); // highest first
        let mut visited = vec![false; n];
        let mut bows: Vec<f32> = Vec::new(); // normalized bow (bow/drop): <0 = concave-up
        let mut ratios: Vec<f32> = Vec::new(); // source-third / mouth-third slope
        let (mut lake_sum, mut stem_sum) = (0usize, 0usize);
        const N_RIVERS: usize = 50;
        const MIN_KM: f32 = 200.0;
        for &head in &heads {
            if bows.len() >= N_RIVERS {
                break;
            }
            if visited[head] {
                continue;
            }
            let mut stem = vec![head];
            visited[head] = true;
            let mut cur = head;
            while let Some(d) = hydrology.drainage_dir[cur] {
                if hydrology.water_state(d) == CellWaterState::Ocean || visited[d] || stem.len() > n
                {
                    break;
                }
                visited[d] = true;
                stem.push(d);
                cur = d;
            }
            if stem.len() < 8 {
                continue;
            }
            let mut dist = vec![0.0f32; stem.len()];
            for k in 1..stem.len() {
                dist[k] = dist[k - 1]
                    + (tess.cell_center(stem[k]) - tess.cell_center(stem[k - 1])).length()
                        * EARTH_RADIUS_KM;
            }
            let total_len = dist[stem.len() - 1];
            let drop = elevation[head] - elevation[*stem.last().unwrap()];
            if total_len < MIN_KM || drop <= 1e-4 {
                continue;
            }
            let mid = total_len * 0.5;
            let mk = (0..stem.len())
                .min_by(|&a, &b| (dist[a] - mid).abs().total_cmp(&(dist[b] - mid).abs()))
                .unwrap();
            let chord =
                elevation[head] + (elevation[*stem.last().unwrap()] - elevation[head]) * 0.5;
            bows.push((elevation[stem[mk]] - chord) / drop);
            let seg = |lo: f32, hi: f32| -> f32 {
                let (mut dz, mut dx) = (0.0f32, 0.0f32);
                for k in 1..stem.len() {
                    if dist[k] > lo * total_len && dist[k] <= hi * total_len {
                        dz += elevation[stem[k - 1]] - elevation[stem[k]];
                        dx += dist[k] - dist[k - 1];
                    }
                }
                if dx > 0.0 {
                    dz / dx
                } else {
                    0.0
                }
            };
            let (ss, sm) = (seg(0.0, 0.333), seg(0.667, 1.0));
            if sm > 1e-9 {
                ratios.push(ss / sm);
            }
            lake_sum += stem
                .iter()
                .filter(|&&c| hydrology.water_state(c) == CellWaterState::LakeWater)
                .count();
            stem_sum += stem.len();
        }
        println!("\n-- River grading (aggregate over largest mountain rivers, source -> sea) --   [graded = concave-up]");
        if bows.is_empty() {
            println!("  (no qualifying rivers)");
        } else {
            bows.sort_by(f32::total_cmp);
            ratios.sort_by(f32::total_cmp);
            let med_bow = bows[bows.len() / 2];
            let pct_concave =
                100.0 * bows.iter().filter(|&&b| b < 0.0).count() as f32 / bows.len() as f32;
            let med_ratio = ratios.get(ratios.len() / 2).copied().unwrap_or(f32::NAN);
            println!(
                "  rivers {} | median norm-bow {:+.3} ({}) | concave {:.0}% | median source/mouth slope ratio {:.2} (graded >1) | lake reaches {:.0}%",
                bows.len(),
                med_bow,
                if med_bow < -0.02 {
                    "concave-up / graded"
                } else if med_bow > 0.02 {
                    "convex (NOT graded)"
                } else {
                    "~uniform"
                },
                pct_concave,
                med_ratio,
                100.0 * lake_sum as f32 / stem_sum.max(1) as f32,
            );
        }
    }

    // ---- Drainage density (wet vs arid) ----
    // Area-weighted accumulation should make wet regions carry a finer channel
    // network than arid ones. Channel = flow above a support threshold; density
    // = channel length / land area (1/km). NOTE: flow is precip-weighted, so wet
    // land has more flow trivially -- read the ratio as directional (wet should
    // be denser), and the upland-restricted line controls somewhat for uplift.
    {
        // Count-equivalent flow (see river-concavity probe) for the absolute
        // channel-support threshold below.
        let flow: Vec<f32> = (0..n).map(|i| hydrology.flow_count_equiv(i)).collect();
        let areas = tess.cell_areas();
        let mut lp: Vec<f32> = (0..n)
            .filter(|&i| land[i])
            .map(|i| precipitation[i])
            .collect();
        lp.sort_by(f32::total_cmp);
        let mut le: Vec<f32> = (0..n).filter(|&i| land[i]).map(|i| elevation[i]).collect();
        le.sort_by(f32::total_cmp);
        let p_med = lp.get(lp.len() / 2).copied().unwrap_or(0.0);
        let e_med = le.get(le.len() / 2).copied().unwrap_or(0.0);
        let channel_thresh = 50.0f32;

        let density = |upland_only: bool| -> (f32, f32) {
            let (mut lw, mut aw, mut la, mut aa) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
            for i in 0..n {
                if !land[i] || (upland_only && elevation[i] < e_med) {
                    continue;
                }
                let a_km2 = areas[i] * EARTH_RADIUS_KM * EARTH_RADIUS_KM;
                let l_km = areas[i].sqrt() * EARTH_RADIUS_KM;
                let wet = precipitation[i] >= p_med;
                if wet {
                    aw += a_km2;
                } else {
                    aa += a_km2;
                }
                if flow[i] >= channel_thresh {
                    if wet {
                        lw += l_km;
                    } else {
                        la += l_km;
                    }
                }
            }
            (
                if aw > 0.0 { lw / aw } else { 0.0 },
                if aa > 0.0 { la / aa } else { 0.0 },
            )
        };

        let (d_wet, d_arid) = density(false);
        let (u_wet, u_arid) = density(true);
        println!(
            "\n-- Drainage density (channel km/km², channel=flow>={})  [wet should dissect more] --",
            channel_thresh
        );
        println!(
            "  all land  : wet(precip>={:.2}) D={:.4} | arid D={:.4} | ratio {:.2}",
            p_med,
            d_wet,
            d_arid,
            d_wet / d_arid.max(1e-9),
        );
        println!(
            "  uplands   : wet D={:.4} | arid D={:.4} | ratio {:.2}  (elev>={:.3}, controls for uplift)",
            u_wet,
            u_arid,
            u_wet / u_arid.max(1e-9),
            e_med,
        );
    }

    // ---- Density allocation audit (fine mesh only) ----
    // Is the fine-cell budget spent where the terrain needs it? The monitor is
    // relief-per-cell Δh_i = max_neighbour |elev_i - elev_j| — the resolved
    // elevation step across a cell. An optimally adaptive mesh EQUIDISTRIBUTES
    // this: classes with tiny Δh/cell (ocean, plains) are over-resolved (cells
    // smaller than their relief needs — wasted budget); classes with large
    // Δh/cell (mountains) carry the representation error. We report, per terrain
    // class: cell budget (count, %), area (%), achieved vs intended cell size,
    // and the Δh/cell distribution — so "coarsen ocean/plains?" and "are
    // mountains starved?" are answered numerically. (Δh in elevation units;
    // cross-class RATIOS are what matter and are unit-free.)
    if let Some(fine) = &world.fine {
        let areas = tess.cell_areas();
        let density = fine.density(); // intended areal density g (cells/steradian)
                                      // Δh per cell: max elevation step to a neighbour.
        let relief: Vec<f32> = (0..n)
            .map(|i| {
                let e = elevation[i];
                tess.neighbors(i)
                    .iter()
                    .map(|&j| (e - elevation[j]).abs())
                    .fold(0.0f32, f32::max)
            })
            .collect();

        // Classes by eroded elevation (independent of the relief monitor, so the
        // equidistribution comparison isn't circular). MOUNTAIN matches the
        // range threshold used above.
        let class_of = |i: usize| -> usize {
            let e = elevation[i];
            if e < 0.0 {
                0 // ocean
            } else if e < 0.03 {
                1 // lowland / plains
            } else if e < RANGE_ELEV {
                2 // upland / hills
            } else {
                3 // mountain
            }
        };
        let names = ["ocean", "lowland", "upland", "mountain"];
        let r2km2 = EARTH_RADIUS_KM * EARTH_RADIUS_KM;
        let _ = density; // intended-size cross-check dropped (cap-scaled + coastal outliers)

        println!("\n-- Density allocation audit (monitor: relief Δh per cell = max neighbour elev step) --");
        println!("   (cell km is cap-dependent: --fine-max coarsens uniformly; relief RATIOS and %cells are cap-robust)");
        println!(
            "{:<9} {:>9} {:>6} {:>9} {:>6} {:>8} {:>9} {:>9} {:>10}",
            "class",
            "cells",
            "%cell",
            "area Mkm²",
            "%area",
            "cell km",
            "Δh/cell p90",
            "awΔh p90",
            "slope p90"
        );
        let total_area: f32 = areas.iter().sum();
        let mut p90_by_class = [0.0f32; 4]; // area-weighted Δh p90 (the steerable monitor)
        let mut slope_p90_by_class = [0.0f32; 4];
        for c in 0..4 {
            let idx: Vec<usize> = (0..n).filter(|&i| class_of(i) == c).collect();
            if idx.is_empty() {
                continue;
            }
            let cnt = idx.len();
            let area_sr: f32 = idx.iter().map(|&i| areas[i]).sum();
            // Median achieved cell size (robust to a few coastal slivers).
            let mut sizes: Vec<f32> = idx
                .iter()
                .map(|&i| areas[i].max(0.0).sqrt() * EARTH_RADIUS_KM)
                .collect();
            sizes.sort_by(f32::total_cmp);
            let cell_km = sizes[cnt / 2];
            // Cell-count Δh p90.
            let mut r: Vec<f32> = idx.iter().map(|&i| relief[i]).collect();
            r.sort_by(f32::total_cmp);
            let cc_p90 = r[(((cnt - 1) as f32) * 0.90) as usize];
            // Area-weighted Δh p90 (each cell weighted by its area, so big cells
            // count proportionally — avoids small mountain cells dominating).
            let mut rw: Vec<(f32, f32)> = idx.iter().map(|&i| (relief[i], areas[i])).collect();
            rw.sort_by(|a, b| a.0.total_cmp(&b.0));
            let aw_p90 = {
                let target = 0.90 * area_sr;
                let mut acc = 0.0f32;
                let mut v = rw.last().map(|x| x.0).unwrap_or(0.0);
                for &(val, a) in &rw {
                    acc += a;
                    if acc >= target {
                        v = val;
                        break;
                    }
                }
                v
            };
            // Slope p90 = Δh per km (separates steepness from cell size: ocean's
            // big Δh is big cells, not steep terrain — this exposes that).
            let mut slope: Vec<f32> = idx
                .iter()
                .map(|&i| relief[i] / (areas[i].max(1e-20).sqrt() * EARTH_RADIUS_KM))
                .collect();
            slope.sort_by(f32::total_cmp);
            let slope_p90 = slope[(((cnt - 1) as f32) * 0.90) as usize];
            p90_by_class[c] = aw_p90;
            slope_p90_by_class[c] = slope_p90;
            println!(
                "{:<9} {:>9} {:>5.1}% {:>9.1} {:>5.1}% {:>8.1} {:>11.4} {:>9.4} {:>10.5}",
                names[c],
                cnt,
                100.0 * cnt as f32 / n as f32,
                area_sr * r2km2 / 1e6,
                100.0 * area_sr / total_area,
                cell_km,
                cc_p90,
                aw_p90,
                slope_p90,
            );
        }
        // Equidistribution ratios on the AREA-WEIGHTED Δh (the quantity the prior
        // should flatten over land). Slope ratios show how much is steepness vs
        // cell size: if the slope ratio ≈ 1 but Δh ratio ≫ 1, it's pure cell-size
        // (over-resolution); if slope ratio is also ≫ 1, the terrain is genuinely
        // steeper there.
        let safe = |x: f32| if x > 1e-9 { x } else { 1e-9 };
        println!(
            "  equidistribution (area-wtd Δh p90): mtn/ocean {:.1}x | mtn/lowland {:.1}x | mtn/upland {:.1}x  (1.0 = balanced)",
            p90_by_class[3] / safe(p90_by_class[0]),
            p90_by_class[3] / safe(p90_by_class[1]),
            p90_by_class[3] / safe(p90_by_class[2]),
        );
        println!(
            "  slope p90 ratio (Δh/km): mtn/ocean {:.1}x | mtn/lowland {:.1}x | mtn/upland {:.1}x  (steepness, cell-size removed)",
            slope_p90_by_class[3] / safe(slope_p90_by_class[0]),
            slope_p90_by_class[3] / safe(slope_p90_by_class[1]),
            slope_p90_by_class[3] / safe(slope_p90_by_class[2]),
        );
        let ocean_cells = (0..n).filter(|&i| class_of(i) == 0).count();
        let ocean_area: f32 = (0..n).filter(|&i| class_of(i) == 0).map(|i| areas[i]).sum();
        println!(
            "  budget: ocean = {:.1}% of cells for {:.1}% of surface; coarsening ocean kx shrinks ocean cells k²-fold",
            100.0 * ocean_cells as f32 / n as f32,
            100.0 * ocean_area / total_area,
        );

        // ---- Fine-scale local relief (scale-controlled convergence probe) ----
        // Fixed PHYSICAL-radius local relief (max-min eroded elevation within R km).
        // Unlike per-cell Δh (which shrinks trivially with cell size), this is
        // independent of resolution, so across a FINE_MOUNTAIN_CELL_KM sweep it
        // reveals whether finer cells keep uncovering dissection (rising ->
        // under-resolved) or it has converged (flat -> dense enough). Sampled over
        // mountain-class cells (the relief-bearing terrain).
        // Mutable KdTree (not ImmutableKdTree): the immutable optimal-layout
        // build panics with "mid > len" on the near-coincident points that
        // appear at multi-million-cell fine meshes (see fine.rs:1167).
        use kiddo::{KdTree, SquaredEuclidean};
        let entries: Vec<[f32; 3]> = (0..n).map(|i| tess.cell_center(i).to_array()).collect();
        let mut tree = KdTree::<f32, 3>::with_capacity(entries.len());
        for (i, e) in entries.iter().enumerate() {
            tree.add(e, i as u64);
        }
        let mtn: Vec<usize> = (0..n).filter(|&i| elevation[i] >= RANGE_ELEV).collect();
        let stride = (mtn.len() / 20_000).max(1);
        let sample: Vec<usize> = mtn.iter().copied().step_by(stride).collect();
        // ~10 km vertical per elev-unit (canonical scale) -> meters for Earth refs.
        const M_PER_UNIT: f32 = ELEVATION_UNIT_KM * 1_000.0;
        println!(
            "\n-- Fine-scale local relief (fixed-radius max-min, meters, {} mountain samples)  [Earth mountains: ~1-2 km within 25 km; hills ~0.3-1 km] --",
            sample.len()
        );
        println!("   ('sharp' = high relief at short scale; scale-controlled: rising p90 across a density sweep => under-resolved, flat => converged)");
        for &rk in &[10.0f32, 25.0f32] {
            let rad = rk / EARTH_RADIUS_KM; // small-angle: chord ≈ arc
            let rsq = rad * rad;
            let mut relief_r: Vec<f32> = sample
                .iter()
                .map(|&i| {
                    let mut lo = f32::INFINITY;
                    let mut hi = f32::NEG_INFINITY;
                    for nb in tree.within_unsorted::<SquaredEuclidean>(&entries[i], rsq) {
                        let e = elevation[nb.item as usize];
                        lo = lo.min(e);
                        hi = hi.max(e);
                    }
                    (hi - lo).max(0.0)
                })
                .collect();
            relief_r.sort_by(f32::total_cmp);
            let m = relief_r.len().max(1);
            let q = |p: f32| relief_r[(((m - 1) as f32) * p) as usize] * M_PER_UNIT;
            println!(
                "  R={:>2.0} km: local relief p50 {:>5.0} m | p90 {:>5.0} m | p99 {:>5.0} m",
                rk,
                q(0.50),
                q(0.90),
                q(0.99)
            );
        }
    }
}

// ===================== TECTONIC-HISTORY AUDIT =====================

#[cfg(feature = "research-landscape")]
#[derive(Debug, serde::Serialize)]
struct SourceViabilityCapability {
    measure: &'static str,
    status: &'static str,
    reason: &'static str,
}

#[cfg(feature = "research-landscape")]
#[derive(Debug, serde::Serialize)]
struct PresentConvergentComponent {
    episode_id: usize,
    plate_pair: [usize; 2],
    support_edges: usize,
    length_km: f64,
    duration_myr: f32,
    mean_convergence_km_per_myr: f32,
    shortening_area_opportunity_km2: f64,
    length_weighted_centroid_unit_xyz: Option<[f32; 3]>,
    continental_continental_length_km: f64,
    continental_oceanic_length_km: f64,
    oceanic_oceanic_length_km: f64,
    episode_majority_subducting_plate: Option<usize>,
    uniform_receiving_plate: Option<usize>,
}

#[cfg(feature = "research-landscape")]
#[derive(Debug, serde::Serialize)]
struct StaticOnsetFrame {
    age_lo_myr: f32,
    age_hi_myr: f32,
    active_episode_ids: Vec<usize>,
    opportunity_support_edges: usize,
    exact_arc_support_length_km: f64,
    shortening_area_rate_km2_per_myr: f64,
    collision_rate_km2_per_myr: f64,
    subduction_rate_km2_per_myr: f64,
    continental_continental_rate_km2_per_myr: f64,
    continental_oceanic_rate_km2_per_myr: f64,
    oceanic_oceanic_rate_km2_per_myr: f64,
    opportunity_weighted_centroid_unit_xyz: Option<[f32; 3]>,
    support_length_jaccard_vs_younger: Option<f64>,
    opportunity_cosine_vs_younger: Option<f64>,
    normalized_opportunity_l1_vs_younger: Option<f64>,
    composition_centroid_shift_vs_younger_km: Option<f64>,
    rank_one_relative_residual_vs_present: Option<f64>,
}

#[cfg(feature = "research-landscape")]
#[derive(Debug, serde::Serialize)]
struct SourceViabilityReport {
    schema: &'static str,
    seed: u64,
    provenance: String,
    requested_coarse_cells: usize,
    actual_coarse_cells: usize,
    history_model: String,
    classification: &'static str,
    static_onset_transient_prerequisite_satisfied: bool,
    moving_or_reversing_support_prerequisite_satisfied: bool,
    within_component_temporal_spatial_rank_one: bool,
    retained_temporal_opportunity_frames: usize,
    derived_frozen_support_onset_frames: usize,
    history_semantics: &'static str,
    length_method: &'static str,
    substantial_composition_threshold_normalized_l1: f64,
    max_adjacent_normalized_opportunity_l1: Option<f64>,
    max_rank_one_relative_residual: Option<f64>,
    max_composition_centroid_shift_km: Option<f64>,
    total_present_shortening_area_opportunity_km2: f64,
    largest_present_component_opportunity_fraction: Option<f64>,
    capabilities: Vec<SourceViabilityCapability>,
    present_convergent_components: Vec<PresentConvergentComponent>,
    static_onset_frames: Vec<StaticOnsetFrame>,
}

#[cfg(all(test, feature = "research-landscape"))]
mod source_viability_tests {
    use super::{
        area_weighted_quantile, normalized_l1, rank_one_residual, spatial_cadence,
        spearman_rank_correlation, FiniteAgeSpatialStation,
    };

    #[test]
    fn distribution_metrics_separate_scaling_from_support_redistribution() {
        let present = [1.0, 2.0, 3.0];
        let scaled = [2.0, 4.0, 6.0];
        assert!(normalized_l1(&present, &scaled).unwrap() < 1.0e-12);
        assert!(rank_one_residual(&scaled, &present).unwrap() < 1.0e-12);

        let nested = [0.0, 2.0, 3.0];
        assert!(normalized_l1(&present, &nested).unwrap() > 0.1);
        assert!(rank_one_residual(&nested, &present).unwrap() > 0.1);
    }

    #[test]
    fn component_statistics_respect_area_and_rank_ties() {
        let samples = vec![(1.0, 9.0), (5.0, 1.0)];
        assert_eq!(area_weighted_quantile(samples.clone(), 0.5), Some(1.0));
        assert_eq!(area_weighted_quantile(samples, 0.95), Some(5.0));

        let increasing = [(1.0, 4.0), (2.0, 5.0), (2.0, 5.0), (3.0, 9.0)];
        let decreasing = [(1.0, 9.0), (2.0, 5.0), (2.0, 5.0), (3.0, 4.0)];
        assert!((spearman_rank_correlation(increasing).unwrap() - 1.0).abs() < 1.0e-12);
        assert!((spearman_rank_correlation(decreasing).unwrap() + 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn spatial_cadence_never_bridges_an_empty_station() {
        let station = |bin: i64, value: Option<f64>| FiniteAgeSpatialStation {
            chain_id: 1,
            bin,
            along_strike_km: bin as f64 * 50.0,
            source_cells: usize::from(value.is_some()),
            source_area_km2: 1.0,
            owner_convergence_mean_km_per_myr: value,
            scheduled_uplift_mean_km: value,
            substrate_elevation_p95_km: value,
            final_crest_p95_km: value,
            positive_response_p95_km: value,
            transverse_trunk_cells: 0,
            transverse_trunk_max_accumulation: None,
        };
        let stations = [
            station(0, Some(1.0)),
            station(1, Some(3.0)),
            station(2, Some(1.0)),
            station(3, None),
            station(4, Some(1.0)),
            station(5, Some(3.0)),
            station(6, Some(1.0)),
        ];
        let cadence = spatial_cadence(&stations, |item| item.final_crest_p95_km, 0.5);
        assert_eq!(cadence.peaks, 2);
        assert_eq!(cadence.spacing_samples, 0);
        assert!(cadence.spacing_mean_km.is_none());
    }
}

#[cfg(feature = "research-landscape")]
fn normalized_l1(left: &[f64], right: &[f64]) -> Option<f64> {
    let left_sum = left.iter().sum::<f64>();
    let right_sum = right.iter().sum::<f64>();
    if left_sum <= 0.0 || right_sum <= 0.0 {
        return None;
    }
    Some(
        left.iter()
            .zip(right)
            .map(|(&a, &b)| (a / left_sum - b / right_sum).abs())
            .sum(),
    )
}

#[cfg(feature = "research-landscape")]
fn cosine_similarity(left: &[f64], right: &[f64]) -> Option<f64> {
    let dot = left.iter().zip(right).map(|(&a, &b)| a * b).sum::<f64>();
    let left_norm = left.iter().map(|value| value * value).sum::<f64>().sqrt();
    let right_norm = right.iter().map(|value| value * value).sum::<f64>().sqrt();
    (left_norm > 0.0 && right_norm > 0.0).then_some(dot / (left_norm * right_norm))
}

#[cfg(feature = "research-landscape")]
fn rank_one_residual(field: &[f64], reference: &[f64]) -> Option<f64> {
    let field_norm2 = field.iter().map(|value| value * value).sum::<f64>();
    let reference_norm2 = reference.iter().map(|value| value * value).sum::<f64>();
    if field_norm2 <= 0.0 || reference_norm2 <= 0.0 {
        return None;
    }
    let scale = field
        .iter()
        .zip(reference)
        .map(|(&a, &b)| a * b)
        .sum::<f64>()
        / reference_norm2;
    let residual2 = field
        .iter()
        .zip(reference)
        .map(|(&value, &basis)| {
            let residual = value - scale * basis;
            residual * residual
        })
        .sum::<f64>();
    Some((residual2 / field_norm2).sqrt())
}

#[cfg(feature = "research-landscape")]
fn weighted_support_jaccard(left: &[f64], right: &[f64], lengths: &[f64]) -> Option<f64> {
    let mut intersection = 0.0;
    let mut union = 0.0;
    for ((&a, &b), &length) in left.iter().zip(right).zip(lengths) {
        if a > 0.0 || b > 0.0 {
            union += length;
            if a > 0.0 && b > 0.0 {
                intersection += length;
            }
        }
    }
    (union > 0.0).then_some(intersection / union)
}

#[cfg(feature = "research-landscape")]
fn weighted_centroid(
    weights: &[f64],
    fronts: &[hex3::world::ConvergentFrontEdge],
) -> Option<glam::Vec3> {
    let sum = weights
        .iter()
        .zip(fronts)
        .fold(glam::DVec3::ZERO, |sum, (&weight, front)| {
            sum + front.midpoint.as_dvec3() * weight
        });
    (sum.length_squared() > 1.0e-24).then(|| sum.normalize().as_vec3())
}

#[cfg(feature = "research-landscape")]
fn composition_shift_km(left: Option<glam::Vec3>, right: Option<glam::Vec3>) -> Option<f64> {
    let (Some(left), Some(right)) = (left, right) else {
        return None;
    };
    Some(f64::from(left.dot(right).clamp(-1.0, 1.0).acos()) * f64::from(EARTH_RADIUS_KM))
}

#[cfg(feature = "research-landscape")]
fn run_source_viability_audit(
    world: &World,
    seed: u64,
    requested_cells: usize,
    out: Option<&Path>,
) {
    use std::collections::BTreeMap;

    use hex3::world::{
        collect_convergent_fronts, collect_plate_boundaries, BoundaryKind, CrustType,
        StructuralRegime,
    };

    let history = world
        .tectonic_history
        .as_ref()
        .expect("tectonic history generated with features");
    let plates = world.plates.as_ref().expect("plates generated");
    let crust = world.crust.as_ref().expect("crust generated");
    let dynamics = world.dynamics.as_ref().expect("dynamics generated");
    let boundaries = collect_plate_boundaries(&world.tessellation, plates, crust, dynamics);
    let fronts = match collect_convergent_fronts(&world.tessellation, &boundaries, history) {
        Ok(fronts) => fronts,
        Err(error) => {
            eprintln!("source viability could not collect exact convergent fronts: {error}");
            std::process::exit(2);
        }
    };
    let mut edges_by_episode = BTreeMap::<usize, Vec<_>>::new();
    for edge in &fronts.edges {
        edges_by_episode
            .entry(edge.episode_id)
            .or_default()
            .push(edge);
    }

    let mut components = Vec::new();
    for episode in history
        .episodes
        .iter()
        .filter(|episode| episode.kind == BoundaryKind::Convergent)
    {
        let edges = edges_by_episode
            .get(&episode.id)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let mut centroid = glam::DVec3::ZERO;
        let mut length_km = 0.0f64;
        let mut opportunity_km2 = 0.0f64;
        let mut cc_km = 0.0f64;
        let mut co_km = 0.0f64;
        let mut oo_km = 0.0f64;
        let mut convergence_length_sum = 0.0f64;
        for edge in edges {
            let edge_km = f64::from(edge.length_km);
            length_km += edge_km;
            convergence_length_sum += edge_km * f64::from(edge.convergence_km_per_myr);
            centroid += edge.midpoint.as_dvec3() * edge_km;
            opportunity_km2 += edge.shortening_area_opportunity_km2;
            match (edge.crust[0], edge.crust[1]) {
                (CrustType::Continental, CrustType::Continental) => cc_km += edge_km,
                (CrustType::Oceanic, CrustType::Oceanic) => oo_km += edge_km,
                _ => co_km += edge_km,
            }
        }
        let centroid_xyz = (centroid.length_squared() > 1.0e-24).then(|| {
            let value = centroid.normalize();
            [value.x as f32, value.y as f32, value.z as f32]
        });
        let receiving_plates: std::collections::BTreeSet<_> = edges
            .iter()
            .filter_map(|edge| edge.receiving_plate)
            .collect();
        let uniform_receiving_plate = (edges.iter().all(|edge| edge.receiving_plate.is_some())
            && receiving_plates.len() == 1)
            .then(|| *receiving_plates.first().unwrap());
        components.push(PresentConvergentComponent {
            episode_id: episode.id,
            plate_pair: [episode.plate_a, episode.plate_b],
            support_edges: edges.len(),
            length_km,
            duration_myr: episode.duration_myr,
            mean_convergence_km_per_myr: if length_km > 0.0 {
                (convergence_length_sum / length_km) as f32
            } else {
                0.0
            },
            shortening_area_opportunity_km2: opportunity_km2,
            length_weighted_centroid_unit_xyz: centroid_xyz,
            continental_continental_length_km: cc_km,
            continental_oceanic_length_km: co_km,
            oceanic_oceanic_length_km: oo_km,
            episode_majority_subducting_plate: episode.subducting_plate,
            uniform_receiving_plate,
        });
    }
    components.sort_by_key(|component| component.episode_id);
    let total_opportunity = components
        .iter()
        .map(|component| component.shortening_area_opportunity_km2)
        .sum::<f64>();
    let largest_fraction = (total_opportunity > 0.0).then(|| {
        components
            .iter()
            .map(|component| component.shortening_area_opportunity_km2)
            .fold(0.0f64, f64::max)
            / total_opportunity
    });

    // The current model does not retain reconstructed boundary geometry, but
    // its duration breakpoints do define the exact frozen-support onset frames
    // consumed by `solve_history_thin_sheet`: a present front is active at age
    // `t` iff `t <= episode.duration_myr`. These frames can diagnose changing
    // composition, but never migration, reversal, or local kinematic change.
    let mut ages = vec![0.0f32];
    ages.extend(
        history
            .episodes
            .iter()
            .filter(|episode| {
                episode.kind == BoundaryKind::Convergent && episode.duration_myr > 0.0
            })
            .map(|episode| episode.duration_myr),
    );
    ages.sort_by(f32::total_cmp);
    ages.dedup_by(|left, right| (*left - *right).abs() < 1.0e-6);
    let edge_lengths: Vec<f64> = fronts
        .edges
        .iter()
        .map(|edge| f64::from(edge.length_km))
        .collect();
    let mut onset_frames = Vec::new();
    let mut younger_weights: Option<Vec<f64>> = None;
    let mut younger_centroid = None;
    let mut present_weights: Option<Vec<f64>> = None;
    for interval in ages.windows(2) {
        let age_lo = interval[0];
        let age_hi = interval[1];
        let age_mid = 0.5 * (age_lo + age_hi);
        let weights: Vec<f64> = fronts
            .edges
            .iter()
            .map(|edge| {
                if edge.episode_duration_myr >= age_mid {
                    f64::from(edge.length_km) * f64::from(edge.convergence_km_per_myr.max(0.0))
                } else {
                    0.0
                }
            })
            .collect();
        if present_weights.is_none() {
            present_weights = Some(weights.clone());
        }
        let active_episode_ids = fronts
            .edges
            .iter()
            .zip(&weights)
            .filter_map(|(edge, &weight)| (weight > 0.0).then_some(edge.episode_id))
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        let support_edges = weights.iter().filter(|&&weight| weight > 0.0).count();
        let support_length = weights
            .iter()
            .zip(&edge_lengths)
            .filter(|&(&weight, _)| weight > 0.0)
            .map(|(_, &length)| length)
            .sum();
        let rate = weights.iter().sum::<f64>();
        let sum_rate = |predicate: &dyn Fn(&hex3::world::ConvergentFrontEdge) -> bool| {
            fronts
                .edges
                .iter()
                .zip(&weights)
                .filter(|(edge, _)| predicate(edge))
                .map(|(_, &weight)| weight)
                .sum::<f64>()
        };
        let centroid = weighted_centroid(&weights, &fronts.edges);
        let centroid_xyz = centroid.map(|value| [value.x, value.y, value.z]);
        let rank_residual = present_weights
            .as_ref()
            .and_then(|present| rank_one_residual(&weights, present));
        onset_frames.push(StaticOnsetFrame {
            age_lo_myr: age_lo,
            age_hi_myr: age_hi,
            active_episode_ids,
            opportunity_support_edges: support_edges,
            exact_arc_support_length_km: support_length,
            shortening_area_rate_km2_per_myr: rate,
            collision_rate_km2_per_myr: sum_rate(&|edge| {
                edge.regime == StructuralRegime::Collision
            }),
            subduction_rate_km2_per_myr: sum_rate(&|edge| {
                edge.regime == StructuralRegime::Subduction
            }),
            continental_continental_rate_km2_per_myr: sum_rate(&|edge| {
                edge.crust == [CrustType::Continental, CrustType::Continental]
            }),
            continental_oceanic_rate_km2_per_myr: sum_rate(&|edge| edge.crust[0] != edge.crust[1]),
            oceanic_oceanic_rate_km2_per_myr: sum_rate(&|edge| {
                edge.crust == [CrustType::Oceanic, CrustType::Oceanic]
            }),
            opportunity_weighted_centroid_unit_xyz: centroid_xyz,
            support_length_jaccard_vs_younger: younger_weights
                .as_ref()
                .and_then(|younger| weighted_support_jaccard(&weights, younger, &edge_lengths)),
            opportunity_cosine_vs_younger: younger_weights
                .as_ref()
                .and_then(|younger| cosine_similarity(&weights, younger)),
            normalized_opportunity_l1_vs_younger: younger_weights
                .as_ref()
                .and_then(|younger| normalized_l1(&weights, younger)),
            composition_centroid_shift_vs_younger_km: composition_shift_km(
                centroid,
                younger_centroid,
            ),
            rank_one_relative_residual_vs_present: rank_residual,
        });
        younger_centroid = centroid;
        younger_weights = Some(weights);
    }
    let max_adjacent_l1 = onset_frames
        .iter()
        .filter_map(|frame| frame.normalized_opportunity_l1_vs_younger)
        .reduce(f64::max);
    let max_rank_residual = onset_frames
        .iter()
        .filter_map(|frame| frame.rank_one_relative_residual_vs_present)
        .reduce(f64::max);
    let max_centroid_shift = onset_frames
        .iter()
        .filter_map(|frame| frame.composition_centroid_shift_vs_younger_km)
        .reduce(f64::max);
    const SUBSTANTIAL_COMPOSITION_L1_THRESHOLD: f64 = 0.10;
    let static_onset_viable =
        max_adjacent_l1.is_some_and(|value| value >= SUBSTANTIAL_COMPOSITION_L1_THRESHOLD);
    let classification = if onset_frames.is_empty() {
        "unsupported_no_positive_duration_convergent_frames"
    } else if max_rank_residual.unwrap_or(0.0) <= 1.0e-9 {
        "amplitude_only_rank_one_frozen_support"
    } else if static_onset_viable {
        "nested_static_support_substantial_composition_change"
    } else {
        "nested_static_support_minor_redistribution"
    };
    let capabilities = vec![
        SourceViabilityCapability {
            measure: "episode support size, length, and shortening-area opportunity",
            status: "supported_present_components_only",
            reason: "BoundaryEpisode partitions connected components of the present boundary; it is not a retained time episode",
        },
        SourceViabilityCapability {
            measure: "spatial support overlap and Jaccard through time",
            status: "supported_derived_frozen_present_support_only",
            reason: "duration breakpoints define nested onset frames over exact present front arcs; this cannot measure support migration",
        },
        SourceViabilityCapability {
            measure: "opportunity-distribution similarity and rank-one residual",
            status: "supported_derived_frozen_present_support_only",
            reason: "local present length*convergence rates can be compared across nested episode-onset frames",
        },
        SourceViabilityCapability {
            measure: "support centroid migration",
            status: "composition_shift_only_not_migration",
            reason: "onset frames can move an opportunity-weighted centroid by changing active components, but every front arc remains fixed at its present location",
        },
        SourceViabilityCapability {
            measure: "activation and deactivation",
            status: "supported_static_episode_onset_only",
            reason: "episodes switch on once when integrating from their inferred contact age toward present; no front deactivation is represented",
        },
        SourceViabilityCapability {
            measure: "regime and receiving-side changes",
            status: "composition_change_only_not_edge_change",
            reason: "onset frames change the mix of fixed present collision/subduction sides; no edge can change regime, polarity, or receiver",
        },
        SourceViabilityCapability {
            measure: "material and crust transitions",
            status: "static_composition_only",
            reason: "onset frames change the fixed present crust-regime mix, but no material state transitions are retained",
        },
    ];
    let history_model = history
        .episodes
        .first()
        .map(|episode| format!("{:?}", episode.model).to_lowercase())
        .unwrap_or_else(|| "none".to_string());
    let report = SourceViabilityReport {
        schema: "hex3.tectonic-source-viability.v1",
        seed,
        provenance: world.manifest().summary(),
        requested_coarse_cells: requested_cells,
        actual_coarse_cells: world.tessellation.num_cells(),
        history_model,
        classification,
        static_onset_transient_prerequisite_satisfied: static_onset_viable,
        moving_or_reversing_support_prerequisite_satisfied: false,
        within_component_temporal_spatial_rank_one: true,
        retained_temporal_opportunity_frames: 0,
        derived_frozen_support_onset_frames: onset_frames.len(),
        history_semantics: "BoundaryEpisode is a present connected plate-pair/kind component whose duration is capped by inferred contact age; it does not retain boundary geometry at multiple times",
        length_method: "exact Voronoi front great-circle arc; opportunity is length_km * positive local convergence_km_per_myr * duration_myr",
        substantial_composition_threshold_normalized_l1:
            SUBSTANTIAL_COMPOSITION_L1_THRESHOLD,
        max_adjacent_normalized_opportunity_l1: max_adjacent_l1,
        max_rank_one_relative_residual: max_rank_residual,
        max_composition_centroid_shift_km: max_centroid_shift,
        total_present_shortening_area_opportunity_km2: total_opportunity,
        largest_present_component_opportunity_fraction: largest_fraction,
        capabilities,
        present_convergent_components: components,
        static_onset_frames: onset_frames,
    };

    println!("\n================ SOURCE VIABILITY seed={seed} ================");
    println!(
        "  classification: {} | static-onset transient gate: {} | moving/reversing-support gate: NO",
        report.classification,
        if report.static_onset_transient_prerequisite_satisfied { "YES" } else { "NO" },
    );
    println!(
        "  temporal source: 0 retained moving-support frames; {} derived nested-onset frames over frozen exact present fronts",
        report.derived_frozen_support_onset_frames,
    );
    println!(
        "  within each connected component: temporal spatial forcing is exactly rank-one (fixed geometry/rate; onset scalar only)"
    );
    println!(
        "  present-only convergent components: {} | shortening-area opportunity {:.3e} km² | largest share {}",
        report.present_convergent_components.len(),
        report.total_present_shortening_area_opportunity_km2,
        report
            .largest_present_component_opportunity_fraction
            .map(|value| format!("{:.1}%", 100.0 * value))
            .unwrap_or_else(|| "unsupported".to_string()),
    );
    println!(
        "  derived composition: max adjacent normalized L1 {} | max rank-one residual {} | max centroid shift {} km",
        report.max_adjacent_normalized_opportunity_l1.map(|value| format!("{value:.3}")).unwrap_or_else(|| "unsupported".to_string()),
        report.max_rank_one_relative_residual.map(|value| format!("{value:.3}")).unwrap_or_else(|| "unsupported".to_string()),
        report.max_composition_centroid_shift_km.map(|value| format!("{value:.0}")).unwrap_or_else(|| "unsupported".to_string()),
    );
    println!("  interpretation: composition changes are episode onset on static present fronts, not migration, reversal, polarity change, or material transition");
    println!(
        "  temporal polarity, receiving-side and material transitions: unsupported by Legacy history"
    );
    if let Some(path) = out {
        let json = serde_json::to_string_pretty(&report).expect("source viability serializes");
        if let Err(error) = std::fs::write(path, format!("{json}\n")) {
            eprintln!(
                "failed to write source viability {}: {error}",
                path.display()
            );
            std::process::exit(2);
        }
        eprintln!("wrote source viability {}", path.display());
    }
}

// ================= FINITE-AGE COMPONENT CORRESPONDENCE =================

#[cfg(feature = "research-landscape")]
const FINITE_AGE_VISIBLE_RESPONSE_KM: f32 = 0.1;

#[cfg(feature = "research-landscape")]
#[derive(Debug, serde::Serialize)]
struct FiniteAgeComponentRecord {
    episode_id: usize,
    duration_myr: f32,
    exact_front_edges: usize,
    exact_front_length_km: f64,
    exact_shortening_area_opportunity_km2: f64,
    exact_opportunity_share: f64,
    source_cells: usize,
    source_support_area_km2: f64,
    raw_fine_integrated_opportunity_km3: f64,
    fine_opportunity_share: f64,
    target_land_source_cells: usize,
    target_land_source_area_km2: f64,
    full_age_builder_uplift_volume_km3: f64,
    full_age_builder_uplift_share: f64,
    finite_age_scheduled_uplift_volume_km3: f64,
    finite_age_scheduled_uplift_share: f64,
    removed_legacy_volume_km3: f64,
    signed_net_response_volume_km3: f64,
    positive_net_response_volume_km3: f64,
    positive_response_share: f64,
    positive_response_to_removed_legacy: Option<f64>,
    effective_positive_response_area_km2: f64,
    effective_area_fraction_of_support: Option<f64>,
    visible_response_threshold_km: f32,
    visible_response_fragments: usize,
    visible_response_area_km2: f64,
    largest_visible_length_km: f32,
    largest_visible_width_km: f32,
    final_land_fraction_of_support: Option<f64>,
    final_peak_km: Option<f32>,
    final_elevation_p90_km: Option<f64>,
    local_relief_25km_p90_km: Option<f64>,
    summit_downhill_grade_p50: Option<f64>,
    summit_downhill_grade_p90: Option<f64>,
    trunk_cells: usize,
    trunk_longitudinal_fraction: Option<f64>,
    trunk_oblique_fraction: Option<f64>,
    trunk_transverse_fraction: Option<f64>,
}

#[cfg(feature = "research-landscape")]
#[derive(Debug, serde::Serialize)]
struct FiniteAgeCorrespondenceSummary {
    exact_components: usize,
    fine_supported_components: usize,
    target_land_builder_supported_components: usize,
    exact_vs_fine_opportunity_share_l1: Option<f64>,
    fine_opportunity_vs_scheduled_uplift_share_l1: Option<f64>,
    static_builder_budget_km3: f64,
    attributed_full_age_builder_budget_km3: f64,
    present_support_retained_fraction: Option<f64>,
    finite_age_scheduled_builder_budget_km3: f64,
    finite_age_retained_fraction_of_attributed_full_age: Option<f64>,
    signed_response_volume_km3: f64,
    signed_response_fraction_of_scheduled_uplift: Option<f64>,
    positive_response_volume_km3: f64,
    positive_response_fraction_of_scheduled_uplift: Option<f64>,
    scheduled_uplift_vs_positive_response_share_l1: Option<f64>,
    exact_opportunity_vs_positive_response_spearman: Option<f64>,
    scheduled_uplift_vs_positive_response_spearman: Option<f64>,
    duration_vs_local_relief_spearman: Option<f64>,
    duration_vs_local_relief_pairs: usize,
    duration_vs_summit_grade_spearman: Option<f64>,
    duration_vs_summit_grade_pairs: usize,
    duration_vs_trunk_transverse_spearman: Option<f64>,
    duration_vs_trunk_transverse_pairs: usize,
}

#[cfg(feature = "research-landscape")]
#[derive(Debug, serde::Serialize)]
struct FiniteAgeComponentReport {
    schema: &'static str,
    seed: u64,
    manifest: hex3::world::RunManifest,
    requested_coarse_cells: usize,
    actual_coarse_cells: usize,
    fine_cells: usize,
    source_semantics: &'static str,
    builder_budget_semantics: &'static str,
    response_semantics: &'static str,
    drainage_semantics: &'static str,
    maturity_correlation_semantics: &'static str,
    visible_response_threshold_km: f32,
    exact_front_identity_crosscheck: &'static str,
    summary: FiniteAgeCorrespondenceSummary,
    components: Vec<FiniteAgeComponentRecord>,
}

#[cfg(feature = "research-landscape")]
#[derive(Default)]
struct ExactFiniteAgeComponent {
    duration_myr: f32,
    front_edges: usize,
    length_km: f64,
    opportunity_km2: f64,
}

#[cfg(feature = "research-landscape")]
fn area_weighted_quantile(mut samples: Vec<(f64, f64)>, probability: f64) -> Option<f64> {
    samples.retain(|(value, weight)| value.is_finite() && weight.is_finite() && *weight > 0.0);
    if samples.is_empty() {
        return None;
    }
    samples.sort_by(|left, right| left.0.total_cmp(&right.0));
    let total = samples.iter().map(|(_, weight)| weight).sum::<f64>();
    if total <= 0.0 {
        return None;
    }
    let target = probability.clamp(0.0, 1.0) * total;
    let mut cumulative = 0.0;
    for (value, weight) in &samples {
        cumulative += weight;
        if cumulative >= target {
            return Some(*value);
        }
    }
    samples.last().map(|(value, _)| *value)
}

#[cfg(feature = "research-landscape")]
fn average_ranks(values: &[f64]) -> Option<Vec<f64>> {
    if values.len() < 3 || values.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&left, &right| values[left].total_cmp(&values[right]));
    let mut ranks = vec![0.0; values.len()];
    let mut start = 0usize;
    while start < order.len() {
        let mut end = start + 1;
        while end < order.len()
            && values[order[start]].total_cmp(&values[order[end]]) == std::cmp::Ordering::Equal
        {
            end += 1;
        }
        let average = 0.5 * ((start + 1) as f64 + end as f64);
        for &index in &order[start..end] {
            ranks[index] = average;
        }
        start = end;
    }
    Some(ranks)
}

#[cfg(feature = "research-landscape")]
fn spearman_rank_correlation(pairs: impl IntoIterator<Item = (f64, f64)>) -> Option<f64> {
    let pairs: Vec<_> = pairs
        .into_iter()
        .filter(|(left, right)| left.is_finite() && right.is_finite())
        .collect();
    let left: Vec<_> = pairs.iter().map(|pair| pair.0).collect();
    let right: Vec<_> = pairs.iter().map(|pair| pair.1).collect();
    let left = average_ranks(&left)?;
    let right = average_ranks(&right)?;
    let left_mean = left.iter().sum::<f64>() / left.len() as f64;
    let right_mean = right.iter().sum::<f64>() / right.len() as f64;
    let covariance = left
        .iter()
        .zip(&right)
        .map(|(&a, &b)| (a - left_mean) * (b - right_mean))
        .sum::<f64>();
    let left_ss = left
        .iter()
        .map(|value| (value - left_mean).powi(2))
        .sum::<f64>();
    let right_ss = right
        .iter()
        .map(|value| (value - right_mean).powi(2))
        .sum::<f64>();
    (left_ss > 0.0 && right_ss > 0.0).then_some(covariance / (left_ss * right_ss).sqrt())
}

#[cfg(feature = "research-landscape")]
fn exact_fronts_match(
    fronts: &hex3::world::OrogenFronts,
    exact: &std::collections::BTreeMap<hex3::world::CellEdgeId, &hex3::world::ConvergentFrontEdge>,
) -> bool {
    if fronts.points.len() != exact.len()
        || fronts.edge_id.len() != fronts.points.len()
        || fronts.episode_id.len() != fronts.points.len()
    {
        return false;
    }
    fronts.edge_id.iter().enumerate().all(|(index, id)| {
        let Some(edge) = exact.get(id) else {
            return false;
        };
        if edge.episode_id != fronts.episode_id[index] {
            return false;
        }
        let [ea, eb] = edge.endpoints;
        let (fa, fb) = (fronts.seg_a[index], fronts.seg_b[index]);
        let direct = fa.dot(ea) > 1.0 - 1.0e-5 && fb.dot(eb) > 1.0 - 1.0e-5;
        let reversed = fa.dot(eb) > 1.0 - 1.0e-5 && fb.dot(ea) > 1.0 - 1.0e-5;
        direct || reversed
    })
}

#[cfg(feature = "research-landscape")]
fn run_finite_age_component_audit(
    world: &World,
    seed: u64,
    requested_cells: usize,
    top: usize,
    out: Option<&Path>,
) {
    use std::collections::BTreeMap;

    use hex3::world::{
        collect_convergent_fronts, collect_plate_boundaries, frozen_support_uplift, OrogenFronts,
    };

    let fine = world
        .fine
        .as_ref()
        .expect("finite-age component audit requires a fine world");
    assert!(
        world.erosion_params.finite_age_uplift,
        "finite-age component audit requires Slice A"
    );
    assert_eq!(
        world.orogen_model,
        OrogenModel::Legacy,
        "finite-age component audit requires the Legacy coarse target"
    );
    assert!(
        (fine.base.emergent_lambda - 1.0).abs() <= f32::EPSILON,
        "finite-age component audit requires full Legacy relief demotion"
    );
    assert!(
        world.erosion_params.uplift_smooth_km == 0.0,
        "finite-age component attribution requires zero uplift smoothing"
    );
    let final_surface = fine
        .eroded
        .as_ref()
        .expect("finite-age component audit requires Stage 4");
    let plates = world.plates.as_ref().expect("plates generated");
    let crust = world.crust.as_ref().expect("crust generated");
    let dynamics = world.dynamics.as_ref().expect("dynamics generated");
    let history = world
        .tectonic_history
        .as_ref()
        .expect("tectonic history generated");
    let boundaries = collect_plate_boundaries(&world.tessellation, plates, crust, dynamics);
    let exact_fronts = collect_convergent_fronts(&world.tessellation, &boundaries, history)
        .unwrap_or_else(|error| panic!("finite-age exact front collection failed: {error}"));
    let exact_by_id: BTreeMap<_, _> = exact_fronts
        .edges
        .iter()
        .map(|edge| (edge.id, edge))
        .collect();
    let fronts = OrogenFronts::build(&world.tessellation, plates, crust, dynamics, history);
    assert!(
        exact_fronts_match(&fronts, &exact_by_id),
        "finite-age owner fronts do not match the exact convergent-front set"
    );
    let source = frozen_support_uplift(&fine.base, &fronts);
    let tess = fine.tessellation();
    let n = tess.num_cells();
    assert_eq!(source.shape.len(), n);
    assert_eq!(source.duration_myr.len(), n);
    assert_eq!(source.owner_front.len(), n);

    let mut exact = BTreeMap::<usize, ExactFiniteAgeComponent>::new();
    for edge in &exact_fronts.edges {
        let component = exact.entry(edge.episode_id).or_default();
        if component.front_edges == 0 {
            component.duration_myr = edge.episode_duration_myr;
        } else {
            assert_eq!(
                component.duration_myr.to_bits(),
                edge.episode_duration_myr.to_bits(),
                "episode duration must be uniform within a source component"
            );
        }
        component.front_edges += 1;
        component.length_km += f64::from(edge.length_km);
        component.opportunity_km2 += edge.shortening_area_opportunity_km2;
    }
    let mut cells_by_episode = BTreeMap::<usize, Vec<usize>>::new();
    for (cell, &owner) in source.owner_front.iter().enumerate() {
        if owner == u32::MAX {
            continue;
        }
        let owner = owner as usize;
        assert!(
            owner < fronts.points.len(),
            "source owner index is in range"
        );
        cells_by_episode
            .entry(fronts.episode_id[owner])
            .or_default()
            .push(cell);
    }
    assert!(
        cells_by_episode
            .keys()
            .all(|episode| exact.contains_key(episode)),
        "every fine source owner maps to an exact component"
    );

    let entries: Vec<[f32; 3]> = (0..n)
        .map(|cell| tess.cell_center(cell).to_array())
        .collect();
    let mut tree = KdTree::<f32, 3>::with_capacity(n);
    for (cell, point) in entries.iter().enumerate() {
        tree.add(point, cell as u64);
    }
    let radius = 25.0f32 / EARTH_RADIUS_KM;
    let relief_radius_sq = (2.0 * (0.5 * radius).sin()).powi(2);
    let areas = tess.cell_areas_ref();
    let area_scale = f64::from(EARTH_RADIUS_KM).powi(2);
    let height_scale = f64::from(ELEVATION_UNIT_KM);
    let final_elevation = &final_surface.elevation.values;
    let final_hydrology = &final_surface.hydrology;

    // Reconstruct the current structured builder exactly in surface-elevation
    // units. The finite-age scheduler multiplies this nominal full-epoch height
    // by ceil(age/lookback * steps)/steps. Only positive-source cells have a
    // retained duration, so this also exposes the separate present-support gate.
    let target = &fine.base.coarse_base_elevation;
    let substrate = &fine.base.base_elevation;
    let mut demoted_volume = 0.0f64;
    let mut shaped_volume = 0.0f64;
    let mut floor_volume = 0.0f64;
    for cell in 0..n {
        if target[cell] < 0.0 {
            continue;
        }
        let area = f64::from(areas[cell]);
        demoted_volume += f64::from((target[cell] - substrate[cell]).max(0.0)) * area;
        shaped_volume += f64::from(source.shape[cell].max(0.0)) * area;
        floor_volume +=
            f64::from((hex3::world::EMERGENT_LAND_FLOOR_MARGIN - substrate[cell]).max(0.0)) * area;
    }
    let excess_volume =
        (f64::from(world.erosion_params.rebuild_gain) * demoted_volume - floor_volume).max(0.0);
    let builder_shape_scale = if shaped_volume > 0.0 {
        (excess_volume / shaped_volume) as f32
    } else {
        0.0f32
    };
    let static_builder_budget_km3 =
        f64::from(world.erosion_params.rebuild_gain) * demoted_volume * area_scale * height_scale;
    let active_fraction = |duration_myr: f32| {
        let steps = world.erosion_params.steps;
        if duration_myr <= 0.0 || steps == 0 {
            0.0
        } else {
            ((((f64::from(duration_myr) / f64::from(history.lookback_myr)) * steps as f64).ceil()
                as usize)
                .clamp(1, steps) as f64)
                / steps as f64
        }
    };

    let total_exact = exact
        .values()
        .map(|component| component.opportunity_km2)
        .sum::<f64>();
    let mut records = Vec::with_capacity(exact.len());
    for (&episode_id, exact_component) in &exact {
        let cells = cells_by_episode
            .get(&episode_id)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let mut support_area_km2 = 0.0f64;
        let mut fine_opportunity_km3 = 0.0f64;
        let mut target_land_source_cells = 0usize;
        let mut target_land_source_area_km2 = 0.0f64;
        let mut full_age_builder_uplift_km3 = 0.0f64;
        let mut finite_age_scheduled_uplift_km3 = 0.0f64;
        let mut removed_legacy_km3 = 0.0f64;
        let mut signed_response_km3 = 0.0f64;
        let mut positive_response_km3 = 0.0f64;
        let mut response_height2_area_km4 = 0.0f64;
        let mut land_area_km2 = 0.0f64;
        let mut final_peak_km = f32::NEG_INFINITY;
        let mut final_samples = Vec::with_capacity(cells.len());
        let mut relief_samples = Vec::with_capacity(cells.len());
        for &cell in cells {
            let area_km2 = f64::from(areas[cell]) * area_scale;
            let duration = f64::from(source.duration_myr[cell]);
            support_area_km2 += area_km2;
            fine_opportunity_km3 += f64::from(source.shape[cell]) * duration * area_km2;
            if target[cell] < 0.0 {
                continue;
            }
            target_land_source_cells += 1;
            target_land_source_area_km2 += area_km2;
            let floor =
                f64::from((hex3::world::EMERGENT_LAND_FLOOR_MARGIN - substrate[cell]).max(0.0));
            let nominal_height =
                floor + f64::from(builder_shape_scale * source.shape[cell].max(0.0));
            let full_age_volume = nominal_height * height_scale * area_km2;
            full_age_builder_uplift_km3 += full_age_volume;
            finite_age_scheduled_uplift_km3 +=
                full_age_volume * active_fraction(source.duration_myr[cell]);
            let removed_height_km = f64::from(
                (fine.base.coarse_base_elevation[cell] - fine.base.base_elevation[cell]).max(0.0),
            ) * height_scale;
            removed_legacy_km3 += removed_height_km * area_km2;
            let final_height_km = f64::from(final_elevation[cell]) * height_scale;
            let response_height_km =
                f64::from(final_elevation[cell] - fine.base.base_elevation[cell]) * height_scale;
            signed_response_km3 += response_height_km * area_km2;
            let positive_height_km = response_height_km.max(0.0);
            positive_response_km3 += positive_height_km * area_km2;
            response_height2_area_km4 += positive_height_km.powi(2) * area_km2;
            if final_elevation[cell] < 0.0 {
                continue;
            }
            land_area_km2 += area_km2;
            final_peak_km = final_peak_km.max(final_height_km as f32);
            final_samples.push((final_height_km, area_km2));

            let mut low = f32::INFINITY;
            let mut high = f32::NEG_INFINITY;
            for neighbor in
                tree.within_unsorted::<SquaredEuclidean>(&entries[cell], relief_radius_sq)
            {
                let elevation = final_elevation[neighbor.item as usize];
                if elevation < 0.0 {
                    continue;
                }
                low = low.min(elevation);
                high = high.max(elevation);
            }
            if low.is_finite() && high.is_finite() {
                relief_samples.push((f64::from((high - low).max(0.0)) * height_scale, area_km2));
            }
        }

        let summit_threshold = area_weighted_quantile(final_samples.clone(), 0.9);
        let mut summit_grade_samples = Vec::new();
        if let Some(threshold) = summit_threshold {
            for &cell in cells {
                if target[cell] < 0.0 || final_elevation[cell] < 0.0 {
                    continue;
                }
                let height_km = f64::from(final_elevation[cell]) * height_scale;
                if height_km < threshold {
                    continue;
                }
                let center = tess.cell_center(cell);
                let mut max_grade = 0.0f64;
                for &neighbor in tess.neighbors(cell) {
                    if final_elevation[neighbor] < 0.0 {
                        continue;
                    }
                    let drop_km =
                        f64::from((final_elevation[cell] - final_elevation[neighbor]).max(0.0))
                            * height_scale;
                    let distance_km = f64::from(
                        center
                            .dot(tess.cell_center(neighbor))
                            .clamp(-1.0, 1.0)
                            .acos()
                            * EARTH_RADIUS_KM,
                    );
                    if distance_km > 0.0 {
                        max_grade = max_grade.max(drop_km / distance_km);
                    }
                }
                summit_grade_samples.push((max_grade, f64::from(areas[cell]) * area_scale));
            }
        }

        let accumulation_samples: Vec<_> = cells
            .iter()
            .filter(|&&cell| target[cell] >= 0.0 && final_elevation[cell] >= 0.0)
            .map(|&cell| {
                (
                    f64::from(final_hydrology.flow_accumulation[cell]),
                    f64::from(areas[cell]) * area_scale,
                )
            })
            .collect();
        let trunk_threshold = area_weighted_quantile(accumulation_samples, 0.9);
        let (mut trunk_cells, mut trunk_longitudinal, mut trunk_oblique, mut trunk_transverse) =
            (0usize, 0.0f64, 0.0f64, 0.0f64);
        if let Some(threshold) = trunk_threshold.filter(|threshold| *threshold > 0.0) {
            for &cell in cells {
                if target[cell] < 0.0 || final_elevation[cell] < 0.0 {
                    continue;
                }
                if f64::from(final_hydrology.flow_accumulation[cell]) < threshold {
                    continue;
                }
                let Some(receiver) = final_hydrology.drainage_dir[cell] else {
                    continue;
                };
                let owner = source.owner_front[cell] as usize;
                let center = tess.cell_center(cell);
                let receiver_center = tess.cell_center(receiver);
                let flow =
                    (receiver_center - center * center.dot(receiver_center)).normalize_or_zero();
                let front_normal = fronts.seg_a[owner]
                    .cross(fronts.seg_b[owner])
                    .normalize_or_zero();
                let strike = front_normal.cross(center).normalize_or_zero();
                if flow.length_squared() == 0.0 || strike.length_squared() == 0.0 {
                    continue;
                }
                let angle = flow.dot(strike).abs().clamp(0.0, 1.0).acos().to_degrees();
                let weight = f64::from(areas[cell]) * area_scale;
                trunk_cells += 1;
                if angle < 30.0 {
                    trunk_longitudinal += weight;
                } else if angle > 60.0 {
                    trunk_transverse += weight;
                } else {
                    trunk_oblique += weight;
                }
            }
        }
        let trunk_total = trunk_longitudinal + trunk_oblique + trunk_transverse;

        let mut visible_mask = vec![false; n];
        for &cell in cells {
            if target[cell] < 0.0 || final_elevation[cell] < 0.0 {
                continue;
            }
            let response_km =
                (final_elevation[cell] - fine.base.base_elevation[cell]) * ELEVATION_UNIT_KM;
            visible_mask[cell] = response_km >= FINITE_AGE_VISIBLE_RESPONSE_KM;
        }
        let visible_components = measure_components(tess, &visible_mask);
        let visible_area_km2 = visible_components
            .iter()
            .map(|component| f64::from(component.area_km2))
            .sum::<f64>();
        let largest_visible = visible_components.first();
        let effective_area_km2 = if response_height2_area_km4 > 0.0 {
            positive_response_km3.powi(2) / response_height2_area_km4
        } else {
            0.0
        };
        records.push(FiniteAgeComponentRecord {
            episode_id,
            duration_myr: exact_component.duration_myr,
            exact_front_edges: exact_component.front_edges,
            exact_front_length_km: exact_component.length_km,
            exact_shortening_area_opportunity_km2: exact_component.opportunity_km2,
            exact_opportunity_share: if total_exact > 0.0 {
                exact_component.opportunity_km2 / total_exact
            } else {
                0.0
            },
            source_cells: cells.len(),
            source_support_area_km2: support_area_km2,
            raw_fine_integrated_opportunity_km3: fine_opportunity_km3,
            fine_opportunity_share: 0.0,
            target_land_source_cells,
            target_land_source_area_km2,
            full_age_builder_uplift_volume_km3: full_age_builder_uplift_km3,
            full_age_builder_uplift_share: 0.0,
            finite_age_scheduled_uplift_volume_km3: finite_age_scheduled_uplift_km3,
            finite_age_scheduled_uplift_share: 0.0,
            removed_legacy_volume_km3: removed_legacy_km3,
            signed_net_response_volume_km3: signed_response_km3,
            positive_net_response_volume_km3: positive_response_km3,
            positive_response_share: 0.0,
            positive_response_to_removed_legacy: (removed_legacy_km3 > 0.0)
                .then_some(positive_response_km3 / removed_legacy_km3),
            effective_positive_response_area_km2: effective_area_km2,
            effective_area_fraction_of_support: (target_land_source_area_km2 > 0.0)
                .then_some(effective_area_km2 / target_land_source_area_km2),
            visible_response_threshold_km: FINITE_AGE_VISIBLE_RESPONSE_KM,
            visible_response_fragments: visible_components.len(),
            visible_response_area_km2: visible_area_km2,
            largest_visible_length_km: largest_visible.map_or(0.0, |component| component.length_km),
            largest_visible_width_km: largest_visible.map_or(0.0, |component| component.width_km),
            final_land_fraction_of_support: (target_land_source_area_km2 > 0.0)
                .then_some(land_area_km2 / target_land_source_area_km2),
            final_peak_km: final_peak_km.is_finite().then_some(final_peak_km),
            final_elevation_p90_km: area_weighted_quantile(final_samples, 0.9),
            local_relief_25km_p90_km: area_weighted_quantile(relief_samples, 0.9),
            summit_downhill_grade_p50: area_weighted_quantile(summit_grade_samples.clone(), 0.5),
            summit_downhill_grade_p90: area_weighted_quantile(summit_grade_samples, 0.9),
            trunk_cells,
            trunk_longitudinal_fraction: (trunk_total > 0.0)
                .then_some(trunk_longitudinal / trunk_total),
            trunk_oblique_fraction: (trunk_total > 0.0).then_some(trunk_oblique / trunk_total),
            trunk_transverse_fraction: (trunk_total > 0.0)
                .then_some(trunk_transverse / trunk_total),
        });
    }

    let total_fine = records
        .iter()
        .map(|record| record.raw_fine_integrated_opportunity_km3)
        .sum::<f64>();
    let total_response = records
        .iter()
        .map(|record| record.positive_net_response_volume_km3)
        .sum::<f64>();
    let total_signed_response = records
        .iter()
        .map(|record| record.signed_net_response_volume_km3)
        .sum::<f64>();
    let total_full_age_builder = records
        .iter()
        .map(|record| record.full_age_builder_uplift_volume_km3)
        .sum::<f64>();
    let total_scheduled_builder = records
        .iter()
        .map(|record| record.finite_age_scheduled_uplift_volume_km3)
        .sum::<f64>();
    for record in &mut records {
        record.fine_opportunity_share = if total_fine > 0.0 {
            record.raw_fine_integrated_opportunity_km3 / total_fine
        } else {
            0.0
        };
        record.positive_response_share = if total_response > 0.0 {
            record.positive_net_response_volume_km3 / total_response
        } else {
            0.0
        };
        record.full_age_builder_uplift_share = if total_full_age_builder > 0.0 {
            record.full_age_builder_uplift_volume_km3 / total_full_age_builder
        } else {
            0.0
        };
        record.finite_age_scheduled_uplift_share = if total_scheduled_builder > 0.0 {
            record.finite_age_scheduled_uplift_volume_km3 / total_scheduled_builder
        } else {
            0.0
        };
    }
    let exact_weights: Vec<_> = records
        .iter()
        .map(|record| record.exact_shortening_area_opportunity_km2)
        .collect();
    let fine_weights: Vec<_> = records
        .iter()
        .map(|record| record.raw_fine_integrated_opportunity_km3)
        .collect();
    let scheduled_weights: Vec<_> = records
        .iter()
        .map(|record| record.finite_age_scheduled_uplift_volume_km3)
        .collect();
    let response_weights: Vec<_> = records
        .iter()
        .map(|record| record.positive_net_response_volume_km3)
        .collect();
    let summary = FiniteAgeCorrespondenceSummary {
        exact_components: records.len(),
        fine_supported_components: records
            .iter()
            .filter(|record| record.source_cells > 0)
            .count(),
        target_land_builder_supported_components: records
            .iter()
            .filter(|record| record.finite_age_scheduled_uplift_volume_km3 > 0.0)
            .count(),
        exact_vs_fine_opportunity_share_l1: normalized_l1(&exact_weights, &fine_weights),
        fine_opportunity_vs_scheduled_uplift_share_l1: normalized_l1(
            &fine_weights,
            &scheduled_weights,
        ),
        static_builder_budget_km3,
        attributed_full_age_builder_budget_km3: total_full_age_builder,
        present_support_retained_fraction: (static_builder_budget_km3 > 0.0)
            .then_some(total_full_age_builder / static_builder_budget_km3),
        finite_age_scheduled_builder_budget_km3: total_scheduled_builder,
        finite_age_retained_fraction_of_attributed_full_age: (total_full_age_builder > 0.0)
            .then_some(total_scheduled_builder / total_full_age_builder),
        signed_response_volume_km3: total_signed_response,
        signed_response_fraction_of_scheduled_uplift: (total_scheduled_builder > 0.0)
            .then_some(total_signed_response / total_scheduled_builder),
        positive_response_volume_km3: total_response,
        positive_response_fraction_of_scheduled_uplift: (total_scheduled_builder > 0.0)
            .then_some(total_response / total_scheduled_builder),
        scheduled_uplift_vs_positive_response_share_l1: normalized_l1(
            &scheduled_weights,
            &response_weights,
        ),
        exact_opportunity_vs_positive_response_spearman: spearman_rank_correlation(
            records.iter().map(|record| {
                (
                    record.exact_shortening_area_opportunity_km2,
                    record.positive_net_response_volume_km3,
                )
            }),
        ),
        scheduled_uplift_vs_positive_response_spearman: spearman_rank_correlation(
            records
                .iter()
                .filter(|record| record.finite_age_scheduled_uplift_volume_km3 > 0.0)
                .map(|record| {
                    (
                        record.finite_age_scheduled_uplift_volume_km3,
                        record.positive_net_response_volume_km3,
                    )
                }),
        ),
        duration_vs_local_relief_spearman: spearman_rank_correlation(records.iter().filter_map(
            |record| {
                record
                    .local_relief_25km_p90_km
                    .map(|relief| (f64::from(record.duration_myr), relief))
            },
        )),
        duration_vs_local_relief_pairs: records
            .iter()
            .filter(|record| record.local_relief_25km_p90_km.is_some())
            .count(),
        duration_vs_summit_grade_spearman: spearman_rank_correlation(records.iter().filter_map(
            |record| {
                record
                    .summit_downhill_grade_p50
                    .map(|grade| (f64::from(record.duration_myr), grade))
            },
        )),
        duration_vs_summit_grade_pairs: records
            .iter()
            .filter(|record| record.summit_downhill_grade_p50.is_some())
            .count(),
        duration_vs_trunk_transverse_spearman: spearman_rank_correlation(
            records.iter().filter_map(|record| {
                record
                    .trunk_transverse_fraction
                    .map(|fraction| (f64::from(record.duration_myr), fraction))
            }),
        ),
        duration_vs_trunk_transverse_pairs: records
            .iter()
            .filter(|record| record.trunk_transverse_fraction.is_some())
            .count(),
    };
    let report = FiniteAgeComponentReport {
        schema: "hex3.finite-age-component-correspondence.v0",
        seed,
        manifest: world.manifest(),
        requested_coarse_cells: requested_cells,
        actual_coarse_cells: world.tessellation.num_cells(),
        fine_cells: n,
        source_semantics: "one nearest exact present-front owner per positive finite-age source cell; episode identity is the present connected BoundaryEpisode; no migration",
        builder_budget_semantics: "surface-equivalent uplift reconstructed from the current unsmoothed land-gated structured-builder floor+shape equation (including its f32 scale) and the exact ceil(age/lookback*steps) suffix schedule; present-support retention compares all-old owned target-land support with the globally calibrated static rebuild budget",
        response_semantics: "net coupled response is final Stage-4 elevation minus the fully demoted non-orogenic FineBase substrate over owned target-land source cells; morphology samples and relief/grade neighbors additionally require final land so bathymetric coastal drops are excluded; response mixes uplift, incision, hillslope transport, deposition and final hydrology integration and is not an incision ledger",
        drainage_semantics: "trunk orientation uses the final authoritative drainage receiver for the area-weighted top accumulation decile on final-land cells within each target-land source footprint, measured against that cell's exact owner-front strike",
        maturity_correlation_semantics: "exploratory unweighted cross-component Spearman correlations over components with final-land support; pair counts are reported, component size/rate/material/setting remain confounders, and these are not age-causal estimates",
        visible_response_threshold_km: FINITE_AGE_VISIBLE_RESPONSE_KM,
        exact_front_identity_crosscheck: "every OrogenFronts owner arc matched collect_convergent_fronts by canonical CellEdgeId, episode id and exact endpoint pair",
        summary,
        components: records,
    };

    println!("\n========== FINITE-AGE COMPONENT CORRESPONDENCE seed={seed} ==========");
    println!(
        "  components {} | share L1 exact→fine {} →scheduled {} →response {}",
        report.components.len(),
        report
            .summary
            .exact_vs_fine_opportunity_share_l1
            .map(|value| format!("{value:.3}"))
            .unwrap_or_else(|| "unsupported".to_string()),
        report
            .summary
            .fine_opportunity_vs_scheduled_uplift_share_l1
            .map(|value| format!("{value:.3}"))
            .unwrap_or_else(|| "unsupported".to_string()),
        report
            .summary
            .scheduled_uplift_vs_positive_response_share_l1
            .map(|value| format!("{value:.3}"))
            .unwrap_or_else(|| "unsupported".to_string()),
    );
    println!(
        "  fine source reaches {}/{} exact components; target-land builder reaches {}",
        report.summary.fine_supported_components,
        report.summary.exact_components,
        report.summary.target_land_builder_supported_components,
    );
    println!(
        "  builder budget: static {:.3e} km³ → present-support {:.1}% → finite-age {:.1}% of supported; in-footprint signed/positive response {:.1}/{:.1}% of scheduled",
        report.summary.static_builder_budget_km3,
        100.0 * report.summary.present_support_retained_fraction.unwrap_or(0.0),
        100.0
            * report
                .summary
                .finite_age_retained_fraction_of_attributed_full_age
                .unwrap_or(0.0),
        100.0
            * report
                .summary
                .signed_response_fraction_of_scheduled_uplift
                .unwrap_or(0.0),
        100.0
            * report
                .summary
                .positive_response_fraction_of_scheduled_uplift
                .unwrap_or(0.0),
    );
    println!(
        "  Spearman exact opportunity/scheduled uplift→positive response: {}/{} | age→relief/summit-grade/trunk-transverse: {}/{}/{}",
        report.summary.exact_opportunity_vs_positive_response_spearman.map(|value| format!("{value:+.3}")).unwrap_or_else(|| "unsupported".to_string()),
        report.summary.scheduled_uplift_vs_positive_response_spearman.map(|value| format!("{value:+.3}")).unwrap_or_else(|| "unsupported".to_string()),
        report.summary.duration_vs_local_relief_spearman.map(|value| format!("{value:+.3}(n={})", report.summary.duration_vs_local_relief_pairs)).unwrap_or_else(|| "unsupported".to_string()),
        report.summary.duration_vs_summit_grade_spearman.map(|value| format!("{value:+.3}(n={})", report.summary.duration_vs_summit_grade_pairs)).unwrap_or_else(|| "unsupported".to_string()),
        report.summary.duration_vs_trunk_transverse_spearman.map(|value| format!("{value:+.3}(n={})", report.summary.duration_vs_trunk_transverse_pairs)).unwrap_or_else(|| "unsupported".to_string()),
    );
    println!("  top exact-opportunity components:");
    println!("    ep   age  exact%  fine% sched%  resp% land-src support-km² visible-width  peak  R25p90 summit-g50 trunk-X");
    let mut order: Vec<_> = report.components.iter().collect();
    order.sort_by(|left, right| {
        right
            .exact_shortening_area_opportunity_km2
            .total_cmp(&left.exact_shortening_area_opportunity_km2)
    });
    for record in order.into_iter().take(top.max(1)) {
        println!(
            "    {:>2} {:>5.1} {:>6.1} {:>6.1} {:>6.1} {:>6.1} {:>8} {:>11.0} {:>13.0} {:>5.2} {:>7} {:>10} {:>7}",
            record.episode_id,
            record.duration_myr,
            100.0 * record.exact_opportunity_share,
            100.0 * record.fine_opportunity_share,
            100.0 * record.finite_age_scheduled_uplift_share,
            100.0 * record.positive_response_share,
            record.target_land_source_cells,
            record.source_support_area_km2,
            record.largest_visible_width_km,
            record.final_peak_km.unwrap_or(0.0),
            record.local_relief_25km_p90_km.map(|value| format!("{value:.2}")).unwrap_or_else(|| "-".to_string()),
            record.summit_downhill_grade_p50.map(|value| format!("{value:.4}")).unwrap_or_else(|| "-".to_string()),
            record.trunk_transverse_fraction.map(|value| format!("{:.0}%", 100.0 * value)).unwrap_or_else(|| "-".to_string()),
        );
    }
    println!("  interpretation: opportunity correlations test source→response correspondence; age correlations are descriptive maturity evidence, not monotonic physical pass thresholds");
    println!("  net response is not incision; exact component denudation remains unretained");

    if let Some(path) = out {
        let json = serde_json::to_string_pretty(&report).expect("component report serializes");
        if let Err(error) = std::fs::write(path, format!("{json}\n")) {
            eprintln!(
                "failed to write finite-age component audit {}: {error}",
                path.display()
            );
            std::process::exit(2);
        }
        eprintln!("wrote finite-age component audit {}", path.display());
    }
}

// ================= FINITE-AGE WITHIN-EPISODE TRACE =================

#[cfg(feature = "research-landscape")]
const FINITE_AGE_SPATIAL_BIN_KM: f64 = 50.0;

#[cfg(feature = "research-landscape")]
#[derive(Debug, serde::Serialize)]
struct FiniteAgeSpatialStation {
    chain_id: u32,
    bin: i64,
    along_strike_km: f64,
    source_cells: usize,
    source_area_km2: f64,
    owner_convergence_mean_km_per_myr: Option<f64>,
    scheduled_uplift_mean_km: Option<f64>,
    substrate_elevation_p95_km: Option<f64>,
    final_crest_p95_km: Option<f64>,
    positive_response_p95_km: Option<f64>,
    transverse_trunk_cells: usize,
    transverse_trunk_max_accumulation: Option<f64>,
}

#[cfg(feature = "research-landscape")]
#[derive(Debug, serde::Serialize)]
struct FiniteAgeSpatialCadence {
    peaks: usize,
    spacing_samples: usize,
    spacing_mean_km: Option<f64>,
    spacing_cv: Option<f64>,
}

#[cfg(feature = "research-landscape")]
#[derive(Debug, serde::Serialize)]
struct FiniteAgeSpatialChain {
    chain_id: u32,
    scheduled_uplift_volume_km3: f64,
    occupied_span_km: f64,
    occupied_stations: usize,
    gap_stations: usize,
    owner_convergence_cadence: FiniteAgeSpatialCadence,
    scheduled_uplift_cadence: FiniteAgeSpatialCadence,
    substrate_crest_cadence: FiniteAgeSpatialCadence,
    final_crest_cadence: FiniteAgeSpatialCadence,
    positive_response_cadence: FiniteAgeSpatialCadence,
    transverse_trunk_cadence: FiniteAgeSpatialCadence,
    owner_convergence_vs_scheduled_uplift_spearman: Option<f64>,
    owner_convergence_vs_final_crest_spearman: Option<f64>,
    scheduled_uplift_vs_final_crest_spearman: Option<f64>,
    substrate_vs_final_crest_spearman: Option<f64>,
    owner_convergence_delta_vs_final_crest_delta_spearman: Option<f64>,
    scheduled_uplift_delta_vs_final_crest_delta_spearman: Option<f64>,
    stations: Vec<FiniteAgeSpatialStation>,
}

#[cfg(feature = "research-landscape")]
#[derive(Debug, serde::Serialize)]
struct FiniteAgeSpatialReport {
    schema: &'static str,
    seed: u64,
    manifest: hex3::world::RunManifest,
    requested_coarse_cells: usize,
    actual_coarse_cells: usize,
    fine_cells: usize,
    episode_id: usize,
    duration_myr: f32,
    station_width_km: f64,
    trunk_threshold_quantile: f64,
    cadence_semantics: &'static str,
    source_semantics: &'static str,
    substrate_semantics: &'static str,
    response_semantics: &'static str,
    drainage_semantics: &'static str,
    dominant_chain_id: Option<u32>,
    chains: Vec<FiniteAgeSpatialChain>,
}

#[cfg(feature = "research-landscape")]
#[derive(Default)]
struct FiniteAgeSpatialBin {
    owner_convergence: Vec<(f64, f64)>,
    scheduled: Vec<(f64, f64)>,
    substrate: Vec<(f64, f64)>,
    final_elevation: Vec<(f64, f64)>,
    positive_response: Vec<(f64, f64)>,
    source_cells: usize,
    source_area_km2: f64,
    scheduled_volume_km3: f64,
    transverse_trunk_cells: usize,
    transverse_trunk_max_accumulation: Option<f64>,
}

#[cfg(feature = "research-landscape")]
fn spatial_cadence(
    stations: &[FiniteAgeSpatialStation],
    value: impl Fn(&FiniteAgeSpatialStation) -> Option<f64>,
    minimum_prominence: f64,
) -> FiniteAgeSpatialCadence {
    let mut peak_indices = Vec::new();
    for index in 1..stations.len().saturating_sub(1) {
        let (Some(left), Some(center), Some(right)) = (
            value(&stations[index - 1]),
            value(&stations[index]),
            value(&stations[index + 1]),
        ) else {
            continue;
        };
        if center > left && center >= right && center - left.min(right) >= minimum_prominence {
            peak_indices.push(index);
        }
    }
    let mut spacings = Vec::new();
    for pair in peak_indices.windows(2) {
        if stations[pair[0]..=pair[1]]
            .iter()
            .all(|station| station.source_cells > 0)
        {
            spacings.push(stations[pair[1]].along_strike_km - stations[pair[0]].along_strike_km);
        }
    }
    let mean = (!spacings.is_empty()).then(|| spacings.iter().sum::<f64>() / spacings.len() as f64);
    let cv = mean.filter(|mean| *mean > 0.0).map(|mean| {
        let variance = spacings
            .iter()
            .map(|spacing| (spacing - mean).powi(2))
            .sum::<f64>()
            / spacings.len() as f64;
        variance.sqrt() / mean
    });
    FiniteAgeSpatialCadence {
        peaks: peak_indices.len(),
        spacing_samples: spacings.len(),
        spacing_mean_km: mean,
        spacing_cv: cv,
    }
}

#[cfg(feature = "research-landscape")]
fn run_finite_age_spatial_trace(
    world: &World,
    seed: u64,
    requested_cells: usize,
    episode_id: usize,
    out: Option<&Path>,
) {
    use std::collections::BTreeMap;

    use hex3::world::{
        frozen_support_uplift, project_owned_front, OrogenFronts, EMERGENT_LAND_FLOOR_MARGIN,
    };

    let fine = world
        .fine
        .as_ref()
        .expect("finite-age spatial trace requires a fine world");
    let final_surface = fine
        .eroded
        .as_ref()
        .expect("finite-age spatial trace requires Stage 4");
    assert!(world.erosion_params.finite_age_uplift);
    assert_eq!(world.orogen_model, OrogenModel::Legacy);
    assert!((fine.base.emergent_lambda - 1.0).abs() <= f32::EPSILON);
    assert_eq!(
        world.erosion_params.uplift_smooth_km.to_bits(),
        0.0f32.to_bits()
    );

    let plates = world.plates.as_ref().expect("plates generated");
    let crust = world.crust.as_ref().expect("crust generated");
    let dynamics = world.dynamics.as_ref().expect("dynamics generated");
    let history = world
        .tectonic_history
        .as_ref()
        .expect("tectonic history generated");
    let fronts = OrogenFronts::build(&world.tessellation, plates, crust, dynamics, history);
    let source = frozen_support_uplift(&fine.base, &fronts);
    let tess = fine.tessellation();
    let areas = tess.cell_areas_ref();
    let area_scale = f64::from(EARTH_RADIUS_KM).powi(2);
    let height_scale = f64::from(ELEVATION_UNIT_KM);
    let target = &fine.base.coarse_base_elevation;
    let substrate = &fine.base.base_elevation;
    let final_elevation = &final_surface.elevation.values;
    let hydrology = &final_surface.hydrology;

    let mut demoted_volume = 0.0f64;
    let mut shaped_volume = 0.0f64;
    let mut floor_volume = 0.0f64;
    for cell in 0..tess.num_cells() {
        if target[cell] < 0.0 {
            continue;
        }
        let area = f64::from(areas[cell]);
        demoted_volume += f64::from((target[cell] - substrate[cell]).max(0.0)) * area;
        shaped_volume += f64::from(source.shape[cell].max(0.0)) * area;
        floor_volume += f64::from((EMERGENT_LAND_FLOOR_MARGIN - substrate[cell]).max(0.0)) * area;
    }
    let excess_volume =
        (f64::from(world.erosion_params.rebuild_gain) * demoted_volume - floor_volume).max(0.0);
    let builder_shape_scale = if shaped_volume > 0.0 {
        (excess_volume / shaped_volume) as f32
    } else {
        0.0
    };
    let active_fraction = |duration_myr: f32| {
        let steps = world.erosion_params.steps;
        if duration_myr <= 0.0 || steps == 0 {
            0.0
        } else {
            ((((f64::from(duration_myr) / f64::from(history.lookback_myr)) * steps as f64).ceil()
                as usize)
                .clamp(1, steps) as f64)
                / steps as f64
        }
    };

    let mut selected = Vec::new();
    for cell in 0..tess.num_cells() {
        let owner = source.owner_front[cell];
        let Some(projection) = project_owned_front(tess.cell_center(cell), owner, &fronts) else {
            continue;
        };
        if fronts.episode_id[projection.front_index as usize] != episode_id || target[cell] < 0.0 {
            continue;
        }
        selected.push((cell, projection));
    }
    assert!(
        !selected.is_empty(),
        "episode {episode_id} has no owned target-land fine source cells"
    );
    let duration_myr = source.duration_myr[selected[0].0];
    assert!(selected
        .iter()
        .all(|(cell, _)| source.duration_myr[*cell].to_bits() == duration_myr.to_bits()));

    let accumulation_samples = selected
        .iter()
        .filter(|(cell, _)| final_elevation[*cell] >= 0.0)
        .map(|(cell, _)| {
            (
                f64::from(hydrology.flow_accumulation[*cell]),
                f64::from(areas[*cell]) * area_scale,
            )
        })
        .collect();
    let trunk_threshold =
        area_weighted_quantile(accumulation_samples, 0.9).filter(|threshold| *threshold > 0.0);

    let mut u_min_by_chain = BTreeMap::<u32, f64>::new();
    for (_, projection) in &selected {
        let u_km = f64::from(projection.u_lin_radians) * f64::from(EARTH_RADIUS_KM);
        u_min_by_chain
            .entry(projection.chain_id)
            .and_modify(|minimum| *minimum = minimum.min(u_km))
            .or_insert(u_km);
    }
    let mut bins = BTreeMap::<(u32, i64), FiniteAgeSpatialBin>::new();
    for (cell, projection) in selected {
        let chain_id = projection.chain_id;
        let u_km = f64::from(projection.u_lin_radians) * f64::from(EARTH_RADIUS_KM);
        let bin_index =
            ((u_km - u_min_by_chain[&chain_id]) / FINITE_AGE_SPATIAL_BIN_KM).floor() as i64;
        let bin = bins.entry((chain_id, bin_index)).or_default();
        let area_km2 = f64::from(areas[cell]) * area_scale;
        let floor = (EMERGENT_LAND_FLOOR_MARGIN - substrate[cell]).max(0.0);
        let nominal = floor + builder_shape_scale * source.shape[cell].max(0.0);
        let scheduled_km =
            f64::from(nominal) * active_fraction(source.duration_myr[cell]) * height_scale;
        let substrate_km = f64::from(substrate[cell]) * height_scale;
        bin.source_cells += 1;
        bin.source_area_km2 += area_km2;
        bin.owner_convergence.push((
            f64::from(fronts.convergence_km_per_myr[projection.front_index as usize]),
            area_km2,
        ));
        bin.scheduled_volume_km3 += scheduled_km * area_km2;
        bin.scheduled.push((scheduled_km, area_km2));
        bin.substrate.push((substrate_km, area_km2));
        if final_elevation[cell] < 0.0 {
            continue;
        }
        let final_km = f64::from(final_elevation[cell]) * height_scale;
        let response_km = (final_km - substrate_km).max(0.0);
        bin.final_elevation.push((final_km, area_km2));
        bin.positive_response.push((response_km, area_km2));

        let Some(threshold) = trunk_threshold else {
            continue;
        };
        let accumulation = f64::from(hydrology.flow_accumulation[cell]);
        if accumulation < threshold {
            continue;
        }
        let Some(receiver) = hydrology.drainage_dir[cell] else {
            continue;
        };
        let center = tess.cell_center(cell);
        let receiver_center = tess.cell_center(receiver);
        let flow = (receiver_center - center * center.dot(receiver_center)).normalize_or_zero();
        let owner = projection.front_index as usize;
        let front_normal = fronts.seg_a[owner]
            .cross(fronts.seg_b[owner])
            .normalize_or_zero();
        let strike = front_normal.cross(center).normalize_or_zero();
        if flow.length_squared() == 0.0 || strike.length_squared() == 0.0 {
            continue;
        }
        let angle = flow.dot(strike).abs().clamp(0.0, 1.0).acos().to_degrees();
        if angle > 60.0 {
            bin.transverse_trunk_cells += 1;
            bin.transverse_trunk_max_accumulation = Some(
                bin.transverse_trunk_max_accumulation
                    .map_or(accumulation, |current| current.max(accumulation)),
            );
        }
    }

    let mut chains = Vec::new();
    for (&chain_id, &u_min) in &u_min_by_chain {
        let max_bin = bins
            .keys()
            .filter_map(|(candidate, bin)| (*candidate == chain_id).then_some(*bin))
            .max()
            .unwrap_or(0);
        let mut stations = Vec::with_capacity(max_bin as usize + 1);
        let mut scheduled_volume_km3 = 0.0;
        for bin_index in 0..=max_bin {
            let bin = bins.remove(&(chain_id, bin_index)).unwrap_or_default();
            scheduled_volume_km3 += bin.scheduled_volume_km3;
            let scheduled_uplift_mean_km = (bin.source_area_km2 > 0.0)
                .then_some(bin.scheduled_volume_km3 / bin.source_area_km2);
            stations.push(FiniteAgeSpatialStation {
                chain_id,
                bin: bin_index,
                along_strike_km: u_min + (bin_index as f64 + 0.5) * FINITE_AGE_SPATIAL_BIN_KM,
                source_cells: bin.source_cells,
                source_area_km2: bin.source_area_km2,
                owner_convergence_mean_km_per_myr: (bin.source_area_km2 > 0.0).then(|| {
                    bin.owner_convergence
                        .iter()
                        .map(|(value, weight)| value * weight)
                        .sum::<f64>()
                        / bin.source_area_km2
                }),
                scheduled_uplift_mean_km,
                substrate_elevation_p95_km: area_weighted_quantile(bin.substrate, 0.95),
                final_crest_p95_km: area_weighted_quantile(bin.final_elevation, 0.95),
                positive_response_p95_km: area_weighted_quantile(bin.positive_response, 0.95),
                transverse_trunk_cells: bin.transverse_trunk_cells,
                transverse_trunk_max_accumulation: bin.transverse_trunk_max_accumulation,
            });
        }
        let occupied_stations = stations
            .iter()
            .filter(|station| station.source_cells > 0)
            .count();
        let shared_source_final = stations.iter().filter_map(|station| {
            Some((
                station.scheduled_uplift_mean_km?,
                station.final_crest_p95_km?,
            ))
        });
        let shared_convergence_scheduled = stations.iter().filter_map(|station| {
            Some((
                station.owner_convergence_mean_km_per_myr?,
                station.scheduled_uplift_mean_km?,
            ))
        });
        let shared_convergence_final = stations.iter().filter_map(|station| {
            Some((
                station.owner_convergence_mean_km_per_myr?,
                station.final_crest_p95_km?,
            ))
        });
        let shared_substrate_final = stations.iter().filter_map(|station| {
            Some((
                station.substrate_elevation_p95_km?,
                station.final_crest_p95_km?,
            ))
        });
        let convergence_final_deltas = stations.windows(2).filter_map(|pair| {
            if pair.iter().any(|station| station.source_cells == 0) {
                return None;
            }
            Some((
                pair[1].owner_convergence_mean_km_per_myr?
                    - pair[0].owner_convergence_mean_km_per_myr?,
                pair[1].final_crest_p95_km? - pair[0].final_crest_p95_km?,
            ))
        });
        let scheduled_final_deltas = stations.windows(2).filter_map(|pair| {
            if pair.iter().any(|station| station.source_cells == 0) {
                return None;
            }
            Some((
                pair[1].scheduled_uplift_mean_km? - pair[0].scheduled_uplift_mean_km?,
                pair[1].final_crest_p95_km? - pair[0].final_crest_p95_km?,
            ))
        });
        chains.push(FiniteAgeSpatialChain {
            chain_id,
            scheduled_uplift_volume_km3: scheduled_volume_km3,
            occupied_span_km: max_bin as f64 * FINITE_AGE_SPATIAL_BIN_KM
                + FINITE_AGE_SPATIAL_BIN_KM,
            occupied_stations,
            gap_stations: stations.len() - occupied_stations,
            owner_convergence_cadence: spatial_cadence(
                &stations,
                |station| station.owner_convergence_mean_km_per_myr,
                1.0,
            ),
            scheduled_uplift_cadence: spatial_cadence(
                &stations,
                |station| station.scheduled_uplift_mean_km,
                0.025,
            ),
            substrate_crest_cadence: spatial_cadence(
                &stations,
                |station| station.substrate_elevation_p95_km,
                0.05,
            ),
            final_crest_cadence: spatial_cadence(
                &stations,
                |station| station.final_crest_p95_km,
                0.10,
            ),
            positive_response_cadence: spatial_cadence(
                &stations,
                |station| station.positive_response_p95_km,
                0.10,
            ),
            transverse_trunk_cadence: spatial_cadence(
                &stations,
                |station| station.transverse_trunk_max_accumulation,
                0.0,
            ),
            owner_convergence_vs_scheduled_uplift_spearman: spearman_rank_correlation(
                shared_convergence_scheduled,
            ),
            owner_convergence_vs_final_crest_spearman: spearman_rank_correlation(
                shared_convergence_final,
            ),
            scheduled_uplift_vs_final_crest_spearman: spearman_rank_correlation(
                shared_source_final,
            ),
            substrate_vs_final_crest_spearman: spearman_rank_correlation(shared_substrate_final),
            owner_convergence_delta_vs_final_crest_delta_spearman: spearman_rank_correlation(
                convergence_final_deltas,
            ),
            scheduled_uplift_delta_vs_final_crest_delta_spearman: spearman_rank_correlation(
                scheduled_final_deltas,
            ),
            stations,
        });
    }
    chains.sort_by(|left, right| {
        right
            .scheduled_uplift_volume_km3
            .total_cmp(&left.scheduled_uplift_volume_km3)
    });
    let report = FiniteAgeSpatialReport {
        schema: "hex3.finite-age-within-episode-trace.v0",
        seed,
        manifest: world.manifest(),
        requested_coarse_cells: requested_cells,
        actual_coarse_cells: world.tessellation.num_cells(),
        fine_cells: tess.num_cells(),
        episode_id,
        duration_myr,
        station_width_km: FINITE_AGE_SPATIAL_BIN_KM,
        trunk_threshold_quantile: 0.9,
        cadence_semantics: "neighbor peaks on contiguous occupied 50 km stations; 1 km/Myr owner-rate, 25 m source, 50 m substrate and 100 m final/response minimum prominence; spacings never bridge empty stations; compact discriminator, not calibrated landform classification",
        source_semantics: "exact retained present-front owner projected onto its actual ordered chain coordinate; scheduled uplift reconstructs the unsmoothed target-land floor+shape builder and ceil(age/lookback*steps) suffix",
        substrate_semantics: "fully demoted pre-erosion FineBase elevation: inherited structural substrate, not an uplift-only or pre-response retained state",
        response_semantics: "final Stage-4 elevation and positive final-minus-substrate response on final land; response mixes uplift, incision, hillslopes, deposition and hydrology integration",
        drainage_semantics: "top area-weighted accumulation decile within this episode, final authoritative receiver, >60 degree flow-to-strike cells; these are transverse-trunk cell proxies, not independent outlets, divides or basins",
        dominant_chain_id: chains.first().map(|chain| chain.chain_id),
        chains,
    };

    println!("\n========== FINITE-AGE SPATIAL TRACE seed={seed} episode={episode_id} ==========");
    println!(
        "  age {:.1} Myr | {} actual chains | dominant chain {:?}",
        report.duration_myr,
        report.chains.len(),
        report.dominant_chain_id
    );
    for chain in report.chains.iter().take(8) {
        println!(
            "  chain {:>3}: uplift {:>9.0} km³, span {:>5.0} km, stations {}/{}; peaks rate/source/substrate/final/response/trunk {}/{}/{}/{}/{}/{}; final spacing {} km CV {}",
            chain.chain_id,
            chain.scheduled_uplift_volume_km3,
            chain.occupied_span_km,
            chain.occupied_stations,
            chain.stations.len(),
            chain.owner_convergence_cadence.peaks,
            chain.scheduled_uplift_cadence.peaks,
            chain.substrate_crest_cadence.peaks,
            chain.final_crest_cadence.peaks,
            chain.positive_response_cadence.peaks,
            chain.transverse_trunk_cadence.peaks,
            chain.final_crest_cadence.spacing_mean_km.map(|value| format!("{value:.0}")).unwrap_or_else(|| "-".to_string()),
            chain.final_crest_cadence.spacing_cv.map(|value| format!("{value:.2}")).unwrap_or_else(|| "-".to_string()),
        );
    }
    println!("  limits: source→final correspondence cannot alone identify which coupled response process created a cadence");

    if let Some(path) = out {
        let json = serde_json::to_string_pretty(&report).expect("spatial trace serializes");
        if let Err(error) = std::fs::write(path, format!("{json}\n")) {
            eprintln!(
                "failed to write finite-age spatial trace {}: {error}",
                path.display()
            );
            std::process::exit(2);
        }
        eprintln!("wrote finite-age spatial trace {}", path.display());
    }
}

fn run_tectonic_history_audit(world: &World, seed: u64, top: usize) {
    let history = world
        .tectonic_history
        .as_ref()
        .expect("tectonic history generated with features");
    let mut episodes: Vec<_> = history.episodes.iter().collect();
    episodes.sort_by(|a, b| {
        b.integrated_normal_displacement_km
            .abs()
            .total_cmp(&a.integrated_normal_displacement_km.abs())
    });

    let mut durations: Vec<f32> = episodes.iter().map(|e| e.duration_myr).collect();
    durations.sort_by(f32::total_cmp);
    if durations.is_empty() {
        println!("\n================ TECTONIC HISTORY seed={seed} ================");
        println!("  no boundary episodes");
        return;
    }
    let q = |p: f32| durations[(((durations.len() - 1) as f32) * p) as usize];
    let count = |kind| episodes.iter().filter(|e| e.kind == kind).count();
    let total_length: f32 = episodes.iter().map(|e| e.length_km).sum();
    let features = world.features.as_ref().expect("features generated");
    let aggregate_work: f64 = features
        .tectonic_crust_work
        .iter()
        .map(|&work| work as f64)
        .sum();
    let episode_work: f64 = features
        .tectonic_episode_work
        .iter()
        .flat_map(|episode| episode.cell_work.iter())
        .map(|&(_, work)| work as f64)
        .sum();
    let material_work: f64 = features
        .tectonic_material_episode_work
        .iter()
        .flat_map(|episode| episode.cell_work.iter())
        .map(|&(_, work)| work as f64)
        .sum();
    let target_area: f64 = features
        .tectonic_material_episode_work
        .iter()
        .map(|episode| episode.target_footprint_area as f64)
        .sum();
    let allocated_area: f64 = features
        .tectonic_material_episode_work
        .iter()
        .map(|episode| episode.allocated_footprint_area as f64)
        .sum();
    let work_residual = episode_work - aggregate_work;
    let max_elevation_km = elevation_to_km(
        world
            .elevation
            .as_ref()
            .expect("elevation generated")
            .values
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max),
    );

    let history_model = format!("{:?}", episodes[0].model).to_lowercase();
    println!(
        "\n================ TECTONIC HISTORY seed={} model={} ================",
        seed, history_model
    );
    println!(
        "  clock: lookback {:.1} Myr, sample step {:.1} Myr | max plate speed {:.1} km/Myr ({:.1} cm/yr)",
        history.lookback_myr,
        history.step_myr,
        hex3::world::MAX_PLATE_SPEED_KM_PER_MYR,
        hex3::world::MAX_PLATE_SPEED_CM_PER_YEAR,
    );
    println!(
        "  episodes {} (conv/div/transform {}/{}/{}) | boundary length {:.0} km",
        episodes.len(),
        count(hex3::world::BoundaryKind::Convergent),
        count(hex3::world::BoundaryKind::Divergent),
        count(hex3::world::BoundaryKind::Transform),
        total_length,
    );
    println!(
        "  residence duration p10/p50/p90 = {:.1}/{:.1}/{:.1} Myr",
        q(0.1),
        q(0.5),
        q(0.9),
    );
    if let Some(carrier) = &history.carrier_replay {
        let snapshot_count = carrier.snapshots.len().max(1);
        let mean_gaps: f32 = carrier
            .snapshots
            .iter()
            .map(|snapshot| snapshot.gap_cells as f32)
            .sum::<f32>()
            / snapshot_count as f32;
        let mean_overlaps: f32 = carrier
            .snapshots
            .iter()
            .map(|snapshot| snapshot.overlap_excess as f32)
            .sum::<f32>()
            / snapshot_count as f32;
        let max_gaps = carrier
            .snapshots
            .iter()
            .map(|snapshot| snapshot.gap_cells)
            .max()
            .unwrap_or(0);
        let max_overlaps = carrier
            .snapshots
            .iter()
            .map(|snapshot| snapshot.overlap_excess)
            .max()
            .unwrap_or(0);
        let topology_changes: usize = carrier
            .snapshots
            .iter()
            .map(|snapshot| snapshot.topology_changes_from_previous)
            .sum();
        let last = carrier.snapshots.last().expect("carrier has t=0 snapshot");
        println!(
            "  carrier: {} cells ({:.0} km spacing), {} snapshots x {:.1} Myr, built {:.3}s",
            carrier.num_cells,
            carrier.mean_spacing_km,
            carrier.snapshots.len(),
            carrier.step_myr,
            carrier.build_seconds,
        );
        println!(
            "  raster ledger: mean/max gaps {:.0}/{} cells, overlap excess {:.0}/{} parcels, max occupancy {} | pair topology changes {}",
            mean_gaps,
            max_gaps,
            mean_overlaps,
            max_overlaps,
            carrier
                .snapshots
                .iter()
                .map(|snapshot| snapshot.max_occupancy)
                .max()
                .unwrap_or(0),
            topology_changes,
        );
        println!(
            "  oldest topology @ {:.0} Myr: {} pairs (conv/div/transform {}/{}/{})",
            last.lookback_myr,
            last.adjacent_pairs.len(),
            last.convergent_pairs,
            last.divergent_pairs,
            last.transform_pairs,
        );
    }
    println!(
        "  crust-work ledger: aggregate {:.6e}, episode-sparse {:.6e}, residual {:+.3e} ({:.3e} relative) | model peak {:.1} km",
        aggregate_work,
        episode_work,
        work_residual,
        work_residual.abs() / aggregate_work.abs().max(1e-30),
        max_elevation_km,
    );
    if !features.tectonic_material_episode_work.is_empty() {
        println!(
            "  material remap: work {:.6e}, residual {:+.3e} | footprint target/allocated {:.6e}/{:.6e} sr ({:.1}% capacity)",
            material_work,
            material_work - aggregate_work,
            target_area,
            allocated_area,
            100.0 * allocated_area / target_area.max(1e-30),
        );
    }
    if matches!(
        world.orogen_model,
        OrogenModel::ThinSheet
            | OrogenModel::HistoryThinSheet
            | OrogenModel::HistoryCarrierThinSheet
            | OrogenModel::HistoryCarrierEvolved
            | OrogenModel::HistoryCarrierLifecycle
    ) {
        if world.orogen_model == OrogenModel::HistoryCarrierLifecycle {
            println!(
                "  lifecycle aggregate mass: created+magma added {:.6e}, ocean consumed {:.6e}, residual {:+.3e}",
                features.thin_sheet_material_added,
                features.thin_sheet_material_removed,
                features.thin_sheet_material_residual,
            );
        } else {
            println!(
                "  thin-sheet mass: magma-added {:.6e}, residual {:+.3e} ({:.3e} relative)",
                features.thin_sheet_material_added,
                features.thin_sheet_material_residual,
                features.thin_sheet_material_residual.abs()
                    / features.thin_sheet_material_added.abs().max(1e-30),
            );
        }
        if world.orogen_model == OrogenModel::HistoryCarrierEvolved {
            let areas = world.tessellation.cell_areas();
            let inherited = &features.thin_sheet_thickness_delta;
            let uplift = &features.tectonic_uplift_rate;
            let total_area: f64 = areas.iter().map(|&area| area as f64).sum();
            let inherited_rms = (inherited
                .iter()
                .zip(areas.iter())
                .map(|(&value, &area)| value as f64 * value as f64 * area as f64)
                .sum::<f64>()
                / total_area.max(1e-30))
            .sqrt();
            let uplift_rms = (uplift
                .iter()
                .zip(areas.iter())
                .map(|(&value, &area)| value as f64 * value as f64 * area as f64)
                .sum::<f64>()
                / total_area.max(1e-30))
            .sqrt();
            let uplift_max = uplift.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let uplift_min = uplift.iter().copied().fold(f32::INFINITY, f32::min);
            println!(
                "  carrier evolution: {:.3}s | inherited thickness RMS {:.3} | present rate RMS {:.4} (min/max {:+.4}/{:+.4}) per Myr",
                features.carrier_evolution_seconds,
                inherited_rms,
                uplift_rms,
                uplift_min,
                uplift_max,
            );
            println!(
                "  moving-vs-stationary forcing: {:.1}% of historical receiver-parcel events lie outside present receiver support",
                100.0 * features.carrier_moving_forcing_fraction,
            );
        }
        if let Some(audit) = &features.lifecycle_audit {
            let elevation = &world
                .elevation
                .as_ref()
                .expect("elevation generated")
                .values;
            let areas = world.tessellation.cell_areas();
            let land_area: f64 = elevation
                .iter()
                .zip(areas.iter())
                .filter(|&(&height, _)| height >= 0.0)
                .map(|(_, &area)| area as f64)
                .sum();
            let mountain_area: f64 = elevation
                .iter()
                .zip(areas.iter())
                .filter(|&(&height, _)| height >= 0.2)
                .map(|(_, &area)| area as f64)
                .sum();
            let total_area: f64 = areas.iter().map(|&area| area as f64).sum();
            let evolved_continental_area: f64 = features
                .lifecycle_final_continental
                .as_ref()
                .expect("lifecycle final crust")
                .iter()
                .zip(areas.iter())
                .filter(|&(&continental, _)| continental)
                .map(|(_, &area)| area as f64)
                .sum();
            println!(
                "  lifecycle {:.3}s: ocean created area/volume {:.4e}/{:.4e}, consumed {:.4e}/{:.4e}",
                audit.runtime_seconds,
                audit.created_ocean_area_sr,
                audit.created_ocean_volume,
                audit.consumed_ocean_area_sr,
                audit.consumed_ocean_volume,
            );
            println!(
                "  collision: underthrust {:.4e}, foundered {:.4e}, under/magma area {:.4e}/{:.4e} sr, max under/buried layer {:.3}/{:.3}, magma foundered {:.4e}, sutures {}, merges/splits {}/{}, plates {} (motion changes {})",
                audit.continental_underthrust_volume,
                audit.foundered_continental_volume,
                audit.underthrust_footprint_area_sr,
                audit.magma_footprint_area_sr,
                audit.max_underthrust_layer_fraction,
                audit.max_buried_layer_fraction,
                audit.foundered_magmatic_volume,
                audit.active_sutures,
                audit.plate_merges,
                audit.plate_splits,
                audit.final_plate_count,
                audit.motion_changes,
            );
            println!(
                "  lifecycle ledger: magma {:.4e}, residual {:+.3e}, continental residual {:+.3e}, ghost overlaps {} | zero-age ocean cells {}",
                audit.magmatic_added_volume,
                audit.material_residual,
                audit.continental_material_residual,
                audit.final_unresolved_overlaps,
                audit.final_zero_age_ocean_cells,
            );
            println!(
                "  lifecycle terrain: peak {:.1} km | mountain footprint {:.1}% of land (coarse elev >=2 km)",
                max_elevation_km,
                100.0 * mountain_area / land_area.max(1e-30),
            );
            println!(
                "  carrier thickness p50/p90/p99/max {:.3}/{:.3}/{:.3}/{:.3}; delta max carrier/projected {:.3}/{:.3}",
                audit.thickness_p50,
                audit.thickness_p90,
                audit.thickness_p99,
                audit.carrier_max_thickness,
                audit.carrier_max_delta,
                audit.projected_max_delta,
            );
            println!(
                "  positive-load decomposition volume underthrust/magma/remap {:.4e}/{:.4e}/{:.4e}; per-cell max thickness {:.3}/{:.3}/{:.3}; max collision deposits {}",
                audit.underthrust_positive_volume,
                audit.magma_positive_volume,
                audit.remap_positive_volume,
                audit.max_underthrust_thickness,
                audit.max_magma_thickness,
                audit.max_remap_thickness,
                audit.max_collision_deposits,
            );
            println!(
                "  evolved continental/land/positive-thickness area {:.1}/{:.1}/{:.1}% planet; mountains occupy {:.1}% planet",
                100.0 * evolved_continental_area / total_area.max(1e-30),
                100.0 * land_area / total_area.max(1e-30),
                100.0 * audit.positive_delta_area_fraction,
                100.0 * mountain_area / total_area.max(1e-30),
            );
            println!(
                "  source-layout trench/arc/ridge/collision remain diagnostics only and are zeroed in lifecycle elevation assembly"
            );
        }
    }
    println!(
        "  id pair kind       edges length_km rate_n rate_s speed duration displacement_n/shear km"
    );
    for e in episodes.into_iter().take(top) {
        println!(
            " {:>3} {:>2}-{:>2} {:<10} {:>5} {:>9.0} {:+6.1} {:+6.1} {:>5.1} {:>7.1} {:+8.0}/{:+8.0}",
            e.id,
            e.plate_a,
            e.plate_b,
            format!("{:?}", e.kind).to_lowercase(),
            e.edge_count,
            e.length_km,
            e.mean_convergence_km_per_myr,
            e.mean_shear_km_per_myr,
            e.mean_relative_speed_km_per_myr,
            e.duration_myr,
            e.integrated_normal_displacement_km,
            e.integrated_shear_displacement_km,
        );
    }
    if world.orogen_model == OrogenModel::HistoryCarrierLifecycle {
        println!(
            "  provenance: generated carrier layout is oldest state; topology-aware conservative pullback evolves forward; Euler motion changes only on continental merge events"
        );
    } else {
        println!(
            "  provenance: duration=min(kinematic residence, back-rotated pair adjacency); present geography is the reconstruction boundary condition"
        );
    }
}

// ===================== REBUILD-FIDELITY AUDIT =====================

/// Standing audit for the structured emergent builder. Components use the
/// exact builder-active mask (`shape > 0 && target >= 0`) and fine-mesh graph,
/// matching the normalization domain in `erosion.rs`.
fn run_rebuild_fidelity_audit(world: &World, seed: u64, top: usize) {
    let Some(fine) = world.fine.as_ref() else {
        println!("rebuild-fidelity audit requires a generated fine world");
        return;
    };
    let Some(shape) = fine.base.emergent_uplift_shape.as_deref() else {
        println!(
            "\n================ REBUILD FIDELITY seed={} model={} ================",
            seed, world.orogen_model
        );
        println!("  no structured emergent uplift shape is active");
        return;
    };

    let tess = fine.tessellation();
    let target = &fine.base.coarse_base_elevation;
    let base = &fine.base.base_elevation;
    let final_elev = &fine.surface_for(u32::MAX).elevation.values;
    let areas = tess.cell_areas();
    let n = tess.num_cells();
    let gain = world.erosion_params.rebuild_gain as f64;
    let floor = |i: usize| (hex3::world::EMERGENT_LAND_FLOOR_MARGIN - base[i]).max(0.0);
    let active = |i: usize| shape[i] > 0.0 && target[i] >= 0.0;

    let mut component = vec![usize::MAX; n];
    let mut components: Vec<Vec<usize>> = Vec::new();
    let mut queue = std::collections::VecDeque::new();
    for i in 0..n {
        if !active(i) || component[i] != usize::MAX {
            continue;
        }
        let id = components.len();
        component[i] = id;
        queue.push_back(i);
        let mut cells = Vec::new();
        while let Some(c) = queue.pop_front() {
            cells.push(c);
            for &nb in tess.neighbors(c) {
                if active(nb) && component[nb] == usize::MAX {
                    component[nb] = id;
                    queue.push_back(nb);
                }
            }
        }
        components.push(cells);
    }

    #[derive(Debug)]
    struct Row {
        id: usize,
        cells: usize,
        area_km2: f64,
        dvol: f64,
        svol: f64,
        fvol: f64,
        local_c: f64,
        transfer_mean_km: f64,
        transfer_pct: f64,
        target_peak_km: f32,
        final_peak_km: f32,
        peak_target_km: f32,
        peak_overshoot_km: f32,
        max_overshoot_km: f32,
    }

    let mut global_dvol = 0.0f64;
    let mut global_svol = 0.0f64;
    let mut global_fvol = 0.0f64;
    for i in 0..n {
        if active(i) {
            let a = areas[i] as f64;
            global_dvol += (target[i] - base[i]).max(0.0) as f64 * a;
            global_svol += shape[i].max(0.0) as f64 * a;
            global_fvol += floor(i) as f64 * a;
        }
    }
    let global_excess = (gain * global_dvol - global_fvol).max(0.0);
    let global_c = if global_svol > 0.0 {
        global_excess / global_svol
    } else {
        0.0
    };

    let radius2 = (EARTH_RADIUS_KM * EARTH_RADIUS_KM) as f64;
    let mut rows = Vec::with_capacity(components.len());
    for (id, cells) in components.iter().enumerate() {
        let mut area = 0.0f64;
        let mut dvol = 0.0f64;
        let mut svol = 0.0f64;
        let mut fvol = 0.0f64;
        let mut target_peak = f32::NEG_INFINITY;
        let mut final_peak = f32::NEG_INFINITY;
        let mut peak_target = 0.0f32;
        let mut max_overshoot = f32::NEG_INFINITY;
        for &i in cells {
            let a = areas[i] as f64;
            area += a;
            dvol += (target[i] - base[i]).max(0.0) as f64 * a;
            svol += shape[i].max(0.0) as f64 * a;
            fvol += floor(i) as f64 * a;
            target_peak = target_peak.max(target[i]);
            max_overshoot = max_overshoot.max(final_elev[i] - target[i]);
            if final_elev[i] > final_peak {
                final_peak = final_elev[i];
                peak_target = target[i];
            }
        }
        let local_excess = (gain * dvol - fvol).max(0.0);
        let local_c = if svol > 0.0 { local_excess / svol } else { 0.0 };
        // Positive means the planet-wide scalar gives this component more
        // rebuild volume than its own demoted-volume budget; negative = tax.
        let transfer = fvol + global_c * svol - gain * dvol;
        let transfer_mean_km = if area > 0.0 {
            10.0 * transfer / area
        } else {
            0.0
        };
        let transfer_pct = if gain * dvol > 0.0 {
            100.0 * transfer / (gain * dvol)
        } else {
            0.0
        };
        rows.push(Row {
            id,
            cells: cells.len(),
            area_km2: area * radius2,
            dvol,
            svol,
            fvol,
            local_c,
            transfer_mean_km,
            transfer_pct,
            target_peak_km: 10.0 * target_peak,
            final_peak_km: 10.0 * final_peak,
            peak_target_km: 10.0 * peak_target,
            peak_overshoot_km: 10.0 * (final_peak - peak_target),
            max_overshoot_km: 10.0 * max_overshoot,
        });
    }
    rows.sort_by(|a, b| b.final_peak_km.total_cmp(&a.final_peak_km));

    let transfer_sum: f64 = rows
        .iter()
        .map(|r| r.fvol + global_c * r.svol - gain * r.dvol)
        .sum();
    let worst = rows
        .iter()
        .map(|r| r.max_overshoot_km)
        .fold(f32::NEG_INFINITY, f32::max);
    let verdict = if worst > 4.0 {
        "FAIL"
    } else if worst > 2.0 {
        "WARN"
    } else {
        "ok"
    };

    println!(
        "\n================ REBUILD FIDELITY seed={} model={} ================",
        seed, world.orogen_model
    );
    println!(
        "  {} active orogens | gain {:.2} | global shape_c {:.5} | transfer ledger {:.3e}",
        rows.len(),
        gain,
        global_c,
        transfer_sum
    );
    println!("  gate [{verdict}]: max(final−target) {worst:.2} km  (warn >2 km, fail >4 km)");
    println!(
        "  transfer: + = globally subsidized, − = globally taxed; mean km is equivalent uplift over component area"
    );
    println!(
        "  id    cells   area_km²   shape_c local/global   transfer km(%)   target→final peak   peak Δ   max Δ"
    );
    for r in rows.iter().take(top) {
        let ratio = if r.local_c > 0.0 {
            global_c / r.local_c
        } else {
            f64::INFINITY
        };
        println!(
            " {:>3}  {:>7}  {:>9.0}   {:>7.4}/{:>7.4} ({:>5.2}x)  {:+6.2} ({:+6.1}%)   {:>5.2}→{:>5.2} ({:>5.2})  {:+5.2}  {:+5.2}",
            r.id,
            r.cells,
            r.area_km2,
            r.local_c,
            global_c,
            ratio,
            r.transfer_mean_km,
            r.transfer_pct,
            r.target_peak_km,
            r.final_peak_km,
            r.peak_target_km,
            r.peak_overshoot_km,
            r.max_overshoot_km,
        );
    }
}

/// Result of auditing one hydrology surface (coarse or fine).
struct DrainageAudit {
    label: String,
    num_cells: usize,
    land_cells: usize,
    land_area: f64,
    endorheic_land_area: f64,
    lake_area: f64,
    num_basins: usize,
    num_endorheic_basins: usize,
    // Lake-capability breakdown (static / climate-independent unless noted):
    lake_capable_basins: usize, // spill - bottom >= MIN_LAKE_DEPTH
    lake_capable_area: f64,     // sum of those basins' total_area (steradians)
    has_water_basins: usize,    // water_level > bottom (at current climate)
    overflowing_basins: usize,  // water_level >= spill (at current climate)
    is_lake_bodies: usize,      // water_bodies with is_lake == true
}

/// Classify a basin chain as sea-reaching or endorheic by walking the
/// `overflow_target` chain. A basin's water escapes to the sea only if every
/// basin along the chain is currently overflowing (full to its spill point) and
/// the chain terminates at a basin whose `overflow_target == None` (drains to
/// ocean). Stops at the first non-overflowing basin (endorheic terminal sink).
/// Cycles (mutually-overflowing closed groups with no external escape) are
/// treated as endorheic. Returns true if sea-reaching.
fn basin_reaches_sea(basins: &[hex3::world::Basin], start: usize) -> bool {
    let mut cur = start;
    let mut visited = std::collections::HashSet::new();
    let cap = basins.len() + 1;
    for _ in 0..cap {
        if !visited.insert(cur) {
            return false; // cycle: closed group, no escape -> endorheic
        }
        let b = &basins[cur];
        if !b.is_overflowing() {
            return false; // water pools here below spill -> endorheic terminal
        }
        match b.overflow_target {
            None => return true, // overflowing AND drains to ocean -> sea-reaching
            Some(next) => cur = next,
        }
    }
    false // ran past the cap without resolving -> treat as endorheic
}

/// Audit one hydrology surface over LAND cells (area-weighted).
fn audit_hydrology(
    label: &str,
    tess: &hex3::world::Tessellation,
    hydro: &hex3::world::Hydrology,
) -> DrainageAudit {
    let n = tess.num_cells();
    let areas = tess.cell_areas();

    // Pre-classify every basin once.
    let basin_sea: Vec<bool> = (0..hydro.basins.len())
        .map(|b| basin_reaches_sea(&hydro.basins, b))
        .collect();

    let mut land_cells = 0usize;
    let mut land_area = 0.0f64;
    let mut endorheic_land_area = 0.0f64;
    let mut lake_area = 0.0f64;

    for i in 0..n {
        if hydro.is_ocean(i) {
            continue;
        }
        // LAND = anything not ocean. Lake/dry-basin cells sit on land surface.
        let a = areas[i] as f64;
        land_cells += 1;
        land_area += a;

        // Lake water on land counts toward the lake fraction.
        if matches!(hydro.water_state(i), hex3::world::CellWaterState::LakeWater) {
            lake_area += a;
        }

        // Trace this cell's drainage to its first basin (capture point), then
        // classify via that basin's overflow chain. If the drainage path reaches
        // ocean without entering any basin, it is sea-reaching.
        let endorheic = {
            let mut cell = i;
            let mut hops = 0usize;
            let cap = n + 1;
            loop {
                if hydro.is_ocean(cell) {
                    break false; // reached the sea directly
                }
                if let Some(bid) = hydro.basin_id[cell] {
                    // First basin entered: its chain decides the fate.
                    break !basin_sea[bid];
                }
                hops += 1;
                if hops > cap {
                    break true; // drainage cycle off-basin: cannot reach sea
                }
                match hydro.downstream(cell) {
                    Some(next) => cell = next,
                    None => break true, // dead-end not in a basin -> inland sink
                }
            }
        };
        if endorheic {
            endorheic_land_area += a;
        }
    }

    let num_endorheic_basins = hydro.basins.iter().filter(|b| !b.is_overflowing()).count();

    // Lake-capability breakdown. lake_capable is static topology (independent of
    // climate); has_water/overflowing/is_lake reflect the CURRENT climate ratio.
    let min_lake_depth = hex3::world::MIN_LAKE_DEPTH;
    let mut lake_capable_basins = 0usize;
    let mut lake_capable_area = 0.0f64;
    let mut has_water_basins = 0usize;
    let mut overflowing_basins = 0usize;
    for b in &hydro.basins {
        if (b.spill_elevation - b.bottom_elevation) >= min_lake_depth {
            lake_capable_basins += 1;
            lake_capable_area += b.total_area as f64;
        }
        if b.has_water() {
            has_water_basins += 1;
        }
        if b.is_overflowing() {
            overflowing_basins += 1;
        }
    }
    let is_lake_bodies = hydro.water_bodies.iter().filter(|w| w.is_lake).count();

    DrainageAudit {
        label: label.to_string(),
        num_cells: n,
        land_cells,
        land_area,
        endorheic_land_area,
        lake_area,
        num_basins: hydro.basins.len(),
        num_endorheic_basins,
        lake_capable_basins,
        lake_capable_area,
        has_water_basins,
        overflowing_basins,
        is_lake_bodies,
    }
}

fn print_audit(a: &DrainageAudit) {
    let land = a.land_area.max(1e-30);
    println!(
        "  [{}] cells={} land_cells={} land_area={:.4} sr\n     endorheic_land_fraction = {:.1}%   lake_fraction_of_land = {:.2}%\n     basins: {} total, {} endorheic (non-overflowing) = {:.1}%",
        a.label,
        a.num_cells,
        a.land_cells,
        a.land_area,
        100.0 * a.endorheic_land_area / land,
        100.0 * a.lake_area / land,
        a.num_basins,
        a.num_endorheic_basins,
        if a.num_basins > 0 {
            100.0 * a.num_endorheic_basins as f64 / a.num_basins as f64
        } else {
            0.0
        },
    );
    println!(
        "     lake-capable basins (depth>=MIN_LAKE_DEPTH): {} ({:.2}% of land area)\n     of those, currently: {} have water, {} overflowing | {} water-bodies classified is_lake",
        a.lake_capable_basins,
        100.0 * a.lake_capable_area / land,
        a.has_water_basins,
        a.overflowing_basins,
        a.is_lake_bodies,
    );
}

/// COARSE-vs-FINE drainage audit. Computes endorheic land fraction, lake
/// fraction, and basin counts for (A) the fine eroded hydrology and (B) a
/// freshly-computed coarse hydrology, using the SAME generator + default
/// climate ratio.
fn run_drainage_audit(world: &World, seed: u64) {
    use hex3::world::{Hydrology, DEFAULT_CLIMATE_RATIO};

    println!(
        "\n================ DRAINAGE AUDIT seed={} (climate_ratio={}) ================",
        seed, DEFAULT_CLIMATE_RATIO
    );
    println!("(land = non-ocean cells; all fractions AREA-weighted in steradians)");

    // (B) COARSE hydrology: same call fine.rs uses for its preview, on the
    // coarse tessellation + coarse elevation + coarse crust/atmosphere fields.
    let crust = world.crust.as_ref().expect("crust");
    let coarse_elev = world.elevation.as_ref().expect("coarse elevation");
    let atmos = world.atmosphere.as_ref().expect("atmosphere");
    let coarse_hydro = Hydrology::generate(
        &world.tessellation,
        crust,
        coarse_elev,
        &atmos.precipitation,
        &atmos.temperature,
    );
    let coarse = audit_hydrology("COARSE", &world.tessellation, &coarse_hydro);

    // (A) FINE hydrology: the eroded stage-4 surface (default view_stage = MAX).
    let fine_audit = match world.fine.as_ref() {
        Some(fine) => {
            let surf = fine.surface_for(u32::MAX); // eroded stage-4 surface
            Some(audit_hydrology(
                "FINE (eroded)",
                fine.tessellation(),
                &surf.hydrology,
            ))
        }
        None => {
            println!("  [FINE] no fine surface present (run without disabling fine mesh)");
            None
        }
    };

    print_audit(&coarse);
    if let Some(f) = &fine_audit {
        print_audit(f);
    }

    println!("\n  VERDICT INPUTS:");
    println!(
        "    COARSE endorheic-land = {:.1}%   FINE endorheic-land = {}",
        100.0 * coarse.endorheic_land_area / coarse.land_area.max(1e-30),
        fine_audit
            .as_ref()
            .map(|f| format!(
                "{:.1}%",
                100.0 * f.endorheic_land_area / f.land_area.max(1e-30)
            ))
            .unwrap_or_else(|| "n/a".into())
    );
}

// ======================== LAKE AUDIT ========================
//
// Object-level lake statistics: each lake is measured as a FEATURE (size,
// shape, depth, placement, plumbing) rather than a per-cell fraction. This is
// the instrument for judging lake QUALITY numerically — aggregate fractions
// and rendered images both miss the failure modes that read as "not great"
// (speckle-sized lakes, perched lakes, lakes without catchments, a dead
// climate dial).

/// One lake (an `is_lake` water body) with object-level measurements.
struct LakeRecord {
    area_km2: f32,
    cells: usize,
    max_depth: f32,
    length_km: f32,
    elongation: f32,
    /// Basin overflows: the lake has an outlet river (exorheic).
    has_outlet: bool,
    /// Direct (first-capture) catchment area / lake area. Earth: ~10-100.
    catchment_ratio: f32,
    /// Area-weighted mean precipitation over the direct catchment.
    mean_catchment_precip: f32,
    /// Land-hypsometry percentile of the lake surface (0 = lowest land).
    hypsometric_pct: f32,
}

struct LakePanel {
    label: String,
    land_area_km2: f64,
    /// Lakes, sorted largest first.
    records: Vec<LakeRecord>,
    /// Water bodies below the lake depth threshold (not counted as lakes).
    ponds: usize,
    climate_ratio: f32,
}

/// Attribute every land cell to the FIRST basin its drainage path enters
/// (path-compressed walk). Cells already inside a depression belong to that
/// basin directly (handled by the caller via `basin_id`); cells whose path
/// reaches the ocean, dead-ends, or cycles off-basin belong to none.
fn first_capture_basin(hydro: &hex3::world::Hydrology, n: usize) -> Vec<Option<usize>> {
    let mut capture: Vec<Option<usize>> = vec![None; n];
    let mut state = vec![0u8; n]; // 0 unvisited, 1 on current path, 2 resolved
    for start in 0..n {
        if hydro.is_ocean(start) || hydro.basin_id[start].is_some() || state[start] == 2 {
            continue;
        }
        let mut path = Vec::new();
        let mut cell = start;
        let result = loop {
            if hydro.is_ocean(cell) {
                break None;
            }
            if let Some(b) = hydro.basin_id[cell] {
                break Some(b);
            }
            match state[cell] {
                2 => break capture[cell],
                1 => break None, // drainage cycle off-basin: no capture
                _ => {}
            }
            state[cell] = 1;
            path.push(cell);
            match hydro.downstream(cell) {
                Some(next) => cell = next,
                None => break None,
            }
        };
        for c in path {
            capture[c] = result;
            state[c] = 2;
        }
    }
    capture
}

fn measure_lakes(
    label: &str,
    tess: &hex3::world::Tessellation,
    hydro: &hex3::world::Hydrology,
    precipitation: &[f32],
) -> LakePanel {
    use hex3::world::{SemanticWaterKind, WaterBodySemantics, WaterOutlet};

    let n = tess.num_cells();
    let areas = tess.cell_areas();
    let r2 = EARTH_RADIUS_KM * EARTH_RADIUS_KM;

    // Land hypsometry (sorted elevation + prefix areas) for placement percentiles.
    let mut land: Vec<(f32, f32)> = (0..n)
        .filter(|&i| !hydro.is_ocean(i))
        .map(|i| (hydro.elevation[i], areas[i]))
        .collect();
    land.sort_by(|a, b| a.0.total_cmp(&b.0));
    let land_area_sr: f64 = land.iter().map(|&(_, a)| a as f64).sum();
    let mut prefix = Vec::with_capacity(land.len());
    let mut acc = 0.0f64;
    for &(_, a) in &land {
        acc += a as f64;
        prefix.push(acc);
    }
    let hyps_pct = |elev: f32| -> f32 {
        let idx = land.partition_point(|&(e, _)| e < elev);
        if idx == 0 {
            0.0
        } else {
            (prefix[idx - 1] / land_area_sr.max(1e-30)) as f32 * 100.0
        }
    };

    // Direct catchment (first-capture) per basin: area + mean precipitation.
    let capture = first_capture_basin(hydro, n);
    let mut catch_area = vec![0.0f64; hydro.basins.len()];
    let mut catch_precip = vec![0.0f64; hydro.basins.len()];
    for i in 0..n {
        if hydro.is_ocean(i) {
            continue;
        }
        if let Some(b) = hydro.basin_id[i].or(capture[i]) {
            catch_area[b] += areas[i] as f64;
            catch_precip[b] += (precipitation[i] * areas[i]) as f64;
        }
    }

    // Lake objects: diagnostics consume the same semantic identities, depth and
    // outlet classification available to rendering/future inspection.
    let semantics = WaterBodySemantics::build(tess, hydro);
    let mask: Vec<bool> = (0..n).map(|i| hydro.is_lake_water(i)).collect();
    let comps = measure_components(tess, &mask);
    let mut records = Vec::with_capacity(comps.len());
    for comp in &comps {
        let Some(body_id) = comp.cells.iter().find_map(|&c| semantics.cell_body[c]) else {
            continue;
        };
        let body = &semantics.bodies[body_id];
        if body.kind != SemanticWaterKind::Lake {
            continue;
        }
        let basin_id = body.id.basin_id.expect("semantic lake has basin");
        let basin = &hydro.basins[basin_id];
        let lake_area_sr = (body.area_km2 / r2) as f64;
        records.push(LakeRecord {
            area_km2: body.area_km2,
            cells: body.cells.len(),
            max_depth: body.max_depth_km,
            length_km: comp.length_km,
            elongation: comp.elongation(),
            has_outlet: body.outlet != WaterOutlet::Terminal,
            catchment_ratio: (catch_area[basin_id] / lake_area_sr.max(1e-30)) as f32,
            mean_catchment_precip: (catch_precip[basin_id] / catch_area[basin_id].max(1e-30))
                as f32,
            hypsometric_pct: hyps_pct(basin.water_level),
        });
    }
    let ponds = semantics
        .bodies
        .iter()
        .filter(|body| body.kind == SemanticWaterKind::Pond)
        .count();

    LakePanel {
        label: label.to_string(),
        land_area_km2: land_area_sr * r2 as f64,
        records,
        ponds,
        climate_ratio: hydro.climate_ratio(),
    }
}

fn print_lake_panel(p: &LakePanel, top: usize) {
    println!("\n  [{}] climate_ratio={:.2}", p.label, p.climate_ratio);
    if p.records.is_empty() {
        println!(
            "     NO LAKES ({} sub-threshold ponds). Earth ref: lakes ≈ 1.8% of land.",
            p.ponds
        );
        return;
    }
    let total_km2: f64 = p.records.iter().map(|r| r.area_km2 as f64).sum();
    let land_pct = 100.0 * total_km2 / p.land_area_km2.max(1e-30);

    // Size spectrum (records are sorted largest-first).
    let largest = &p.records[0];
    let largest_share = 100.0 * largest.area_km2 as f64 / total_km2.max(1e-30);
    let area_at = |q: f64| -> f32 {
        // records sorted DESC; take from the ascending view
        let k = ((p.records.len() - 1) as f64 * (1.0 - q)) as usize;
        p.records[k].area_km2
    };
    let speckle: Vec<&LakeRecord> = p.records.iter().filter(|r| r.cells <= 2).collect();
    let speckle_area: f64 = speckle.iter().map(|r| r.area_km2 as f64).sum();

    println!(
        "     lakes: {} (+{} sub-threshold ponds)   total {:.0} km² = {:.2}% of land (Earth ≈ 1.8%)",
        p.records.len(),
        p.ponds,
        total_km2,
        land_pct
    );
    println!(
        "     size spectrum: largest {:.0} km² = {:.0}% of all lake area (Earth: Caspian ≈ 30%) | p50 {:.0} km², p90 {:.0} km² (Earth is HEAVY-tailed: many small, area in the few big)",
        largest.area_km2,
        largest_share,
        area_at(0.5),
        area_at(0.9)
    );
    println!(
        "     speckle: {} lakes ≤2 cells = {:.0}% of count, {:.1}% of lake area (high count-share + low area-share is fine; high AREA-share reads as noise)",
        speckle.len(),
        100.0 * speckle.len() as f64 / p.records.len() as f64,
        100.0 * speckle_area / total_km2.max(1e-30)
    );

    // Plumbing + placement.
    let outlets = p.records.iter().filter(|r| r.has_outlet).count();
    let mean_precip = |sel: &dyn Fn(&&LakeRecord) -> bool| -> Option<f32> {
        let v: Vec<f32> = p
            .records
            .iter()
            .filter(sel)
            .map(|r| r.mean_catchment_precip)
            .collect();
        if v.is_empty() {
            None
        } else {
            Some(v.iter().sum::<f32>() / v.len() as f32)
        }
    };
    let fmt_opt = |o: Option<f32>| o.map(|v| format!("{v:.2}")).unwrap_or_else(|| "-".into());
    println!(
        "     outlets: {}/{} lakes overflow (have an outlet river) | catchment precip: outlet-lakes {} vs terminal {} (Earth: terminal lakes sit in DRY basins)",
        outlets,
        p.records.len(),
        fmt_opt(mean_precip(&|r| r.has_outlet)),
        fmt_opt(mean_precip(&|r| !r.has_outlet))
    );
    let mut hyps: Vec<f32> = p.records.iter().map(|r| r.hypsometric_pct).collect();
    hyps.sort_by(f32::total_cmp);
    let mut catch: Vec<f32> = p.records.iter().map(|r| r.catchment_ratio).collect();
    catch.sort_by(f32::total_cmp);
    let q = |v: &[f32], f: f64| v[((v.len() - 1) as f64 * f) as usize];
    println!(
        "     placement: lake-surface land-hypsometry pct p10/p50/p90 = {:.0}/{:.0}/{:.0} (Earth: most lake AREA sits low) | catchment/lake ratio p10/p50/p90 = {:.0}/{:.0}/{:.0}× (Earth ≈ 10-100×; ~1× = a puddle with no watershed)",
        q(&hyps, 0.1), q(&hyps, 0.5), q(&hyps, 0.9),
        q(&catch, 0.1), q(&catch, 0.5), q(&catch, 0.9)
    );

    println!(
        "     top lakes:   area_km²  cells  depth_km  len_km  elong  outlet  catch×  precip  hyps%"
    );
    for (i, r) in p.records.iter().take(top).enumerate() {
        println!(
            "       {:>2}. {:>10.0}  {:>5}  {:>5.3}  {:>6.0}  {:>5.1}  {:>6} {:>6.0}×  {:>6.2}  {:>4.0}",
            i + 1,
            r.area_km2,
            r.cells,
            r.max_depth,
            r.length_km,
            r.elongation,
            if r.has_outlet { "yes" } else { "TERM" },
            r.catchment_ratio,
            r.mean_catchment_precip,
            r.hypsometric_pct
        );
    }
}

/// Lake count + lake fraction of land at the CURRENT climate ratio.
fn lake_dial_point(
    tess: &hex3::world::Tessellation,
    hydro: &hex3::world::Hydrology,
) -> (usize, f64) {
    let areas = tess.cell_areas();
    let n = tess.num_cells();
    let mut land = 0.0f64;
    let mut lake = 0.0f64;
    for i in 0..n {
        if hydro.is_ocean(i) {
            continue;
        }
        land += areas[i] as f64;
        if hydro.is_lake_water(i) {
            lake += areas[i] as f64;
        }
    }
    let semantics = hex3::world::WaterBodySemantics::build(tess, hydro);
    let n_lakes = semantics
        .bodies
        .iter()
        .filter(|body| body.kind == hex3::world::SemanticWaterKind::Lake)
        .count();
    (n_lakes, 100.0 * lake / land.max(1e-30))
}

fn run_lake_audit(world: &mut World, seed: u64, top: usize) {
    use hex3::world::Hydrology;

    println!(
        "\n================ LAKE AUDIT seed={} ================",
        seed
    );
    println!("(object-level lake features, Earth refs inline; all fractions AREA-weighted)");
    println!("(pre-integration baseline: rerun with HEX3_NO_DRAINAGE_INTEGRATION=1)");

    // COARSE: freshly generated on the coarse tessellation (same call the fine
    // preview uses), OWNED so the dial sweep below can mutate it.
    let crust = world.crust.as_ref().expect("crust");
    let coarse_elev = world.elevation.as_ref().expect("coarse elevation");
    let atmos = world.atmosphere.as_ref().expect("atmosphere");
    let mut coarse_hydro = Hydrology::generate(
        &world.tessellation,
        crust,
        coarse_elev,
        &atmos.precipitation,
        &atmos.temperature,
    );
    let coarse_panel = measure_lakes(
        "COARSE",
        &world.tessellation,
        &coarse_hydro,
        &atmos.precipitation,
    );
    print_lake_panel(&coarse_panel, top);

    // FINE: the eroded stage-4 surface (falls back to pre-erosion if absent).
    match world.fine.as_ref() {
        Some(fine) => {
            let surf = fine.surface_for(u32::MAX);
            let panel = measure_lakes(
                "FINE (eroded)",
                fine.tessellation(),
                &surf.hydrology,
                &surf.precipitation,
            );
            print_lake_panel(&panel, top);
        }
        None => println!("  [FINE] no fine surface present (run without disabling fine mesh)"),
    }

    // Climate-dial response: the lake system's transfer function. Healthy =
    // smooth + monotonic; a step function = degenerate fill criterion.
    let ratios = [0.05f32, 0.10, 0.15, 0.20, 0.30, 0.50];
    let fine_orig = world
        .fine
        .as_ref()
        .map(|f| f.surface_for(u32::MAX).hydrology.climate_ratio());
    println!("\n  CLIMATE-DIAL RESPONSE (lakes | lake%-of-land):");
    println!("    ratio        COARSE               FINE");
    for &r in &ratios {
        coarse_hydro.set_climate_ratio(&world.tessellation, r);
        let (cn, cf) = lake_dial_point(&world.tessellation, &coarse_hydro);
        let fine_str = match world.fine.as_mut() {
            Some(fine) => {
                fine.set_climate_ratio(4, r);
                let surf = fine.surface_for(u32::MAX);
                let (fnum, ff) = lake_dial_point(fine.tessellation(), &surf.hydrology);
                format!("{fnum:>5} | {ff:5.2}%")
            }
            None => "n/a".into(),
        };
        println!("    {:>5.2}   {:>5} | {:5.2}%      {}", r, cn, cf, fine_str);
    }
    // Restore the world's fine climate ratio (the audit must not mutate state
    // a later consumer would see).
    if let (Some(fine), Some(orig)) = (world.fine.as_mut(), fine_orig) {
        fine.set_climate_ratio(4, orig);
    }
}

// ===================== DETAIL-SURVIVAL AUDIT =====================
//
// Unlike the mountain audit, this panel starts from a model-independent
// tectonic-process footprint and keeps cells in the denominator even when a
// pipeline version lowers or smooths their terrain out of the mountain mask.

fn run_detail_survival_audit(world: &World, seed: u64) {
    let Some(fine) = world.fine.as_ref() else {
        println!("detail-survival audit requires a generated fine world");
        return;
    };
    let tess = fine.tessellation();
    let fields = &fine.base.fields.elevation_fields;
    let coarse = &fine.base.coarse_base_elevation;
    let base = &fine.base.base_elevation;
    let final_elev = &fine.surface_for(u32::MAX).elevation.values;
    let n = tess.num_cells();
    let areas = tess.cell_areas();
    let r2 = EARTH_RADIUS_KM * EARTH_RADIUS_KM;

    let forcing: Vec<f32> = fields
        .arc
        .iter()
        .zip(fields.collision.iter())
        .map(|(&arc, &collision)| (arc + collision).max(0.0))
        .collect();
    let forcing_max = forcing.iter().copied().fold(0.0f32, f32::max);
    if forcing_max <= f32::EPSILON {
        println!(
            "\n================ DETAIL SURVIVAL seed={} model={} ================",
            seed, world.orogen_model
        );
        println!("  no arc/collision process footprint exists for this world");
        return;
    }
    let forcing_gate = 0.05 * forcing_max;
    let footprint: Vec<bool> = forcing.iter().map(|&f| f >= forcing_gate).collect();

    let structural: Vec<f32> = base
        .iter()
        .zip(coarse.iter())
        .map(|(&b, &c)| b - c)
        .collect();
    let erosion: Vec<f32> = final_elev
        .iter()
        .zip(base.iter())
        .map(|(&e, &b)| e - b)
        .collect();

    let local_relief = |elevation: &[f32]| -> Vec<f32> {
        (0..n)
            .map(|i| {
                let mut lo = elevation[i];
                let mut hi = elevation[i];
                for &nb in tess.neighbors(i) {
                    lo = lo.min(elevation[nb]);
                    hi = hi.max(elevation[nb]);
                }
                hi - lo
            })
            .collect()
    };
    let coarse_relief = local_relief(coarse);
    let base_relief = local_relief(base);
    let final_relief = local_relief(final_elev);

    let area_where = |pred: &dyn Fn(usize) -> bool| -> f64 {
        (0..n)
            .filter(|&i| footprint[i] && pred(i))
            .map(|i| areas[i] as f64 * r2 as f64)
            .sum()
    };
    let footprint_area = area_where(&|_| true).max(1e-12);
    let pct = |pred: &dyn Fn(usize) -> bool| 100.0 * area_where(pred) / footprint_area;
    let rms_m = |values: &[f32]| -> f64 {
        let (weighted, area): (f64, f64) =
            (0..n)
                .filter(|&i| footprint[i])
                .fold((0.0, 0.0), |(sum, area), i| {
                    let a = areas[i] as f64;
                    let v = values[i] as f64 * AUDIT_M_PER_UNIT as f64;
                    (sum + a * v * v, area + a)
                });
        (weighted / area.max(1e-20)).sqrt()
    };

    let slope = (hex3::world::CONTINENTAL_BASE - hex3::world::ABYSSAL_DEPTH)
        / (hex3::world::CRUST_THICKNESS_CONTINENTAL - hex3::world::CRUST_THICKNESS_OCEANIC);
    let load_elev: Vec<f32> = fields
        .tectonic_thickening
        .iter()
        .map(|&h| h * slope)
        .collect();
    let load_volume: f64 = fields
        .tectonic_thickening
        .iter()
        .zip(areas.iter())
        .map(|(&height, &area)| height as f64 * area as f64)
        .sum();
    let positive_load_volume: f64 = fields
        .tectonic_thickening
        .iter()
        .zip(areas.iter())
        .map(|(&height, &area)| height.max(0.0) as f64 * area as f64)
        .sum();
    let negative_load_volume: f64 = fields
        .tectonic_thickening
        .iter()
        .zip(areas.iter())
        .map(|(&height, &area)| (-height).max(0.0) as f64 * area as f64)
        .sum();
    let footprint_solid_angle: f64 = (0..n)
        .filter(|&i| footprint[i])
        .map(|i| areas[i] as f64)
        .sum();
    let mean_spacing_km = tess.mean_cell_area().sqrt() * EARTH_RADIUS_KM;

    println!(
        "\n================ DETAIL SURVIVAL seed={} model={} ================",
        seed, world.orogen_model
    );
    println!(
        "  process footprint: {:.2} Mkm² (arc+collision >= 5% of max {:.3}); one-hop scale ~{:.0} km",
        footprint_area / 1.0e6,
        forcing_max,
        mean_spacing_km,
    );
    println!(
        "  tectonic thickness-volume net={:.5}, +{:.5}/-{:.5} (unit-sphere); mean net over footprint={:.3} ({:.2} km isostatic)",
        load_volume,
        positive_load_volume,
        negative_load_volume,
        load_volume / footprint_solid_angle.max(1e-20),
        load_volume / footprint_solid_angle.max(1e-20)
            * slope as f64
            * ELEVATION_UNIT_KM as f64,
    );
    println!("  survival funnel (% of fixed process footprint):");
    println!(
        "    tectonic support >=0.2/0.5/1.0 km: {:5.1}% / {:5.1}% / {:5.1}%",
        pct(&|i| load_elev[i] >= 0.02),
        pct(&|i| load_elev[i] >= 0.05),
        pct(&|i| load_elev[i] >= 0.10),
    );
    println!(
        "    coarse land / fine-base land / final land: {:5.1}% / {:5.1}% / {:5.1}%",
        pct(&|i| coarse[i] >= 0.0),
        pct(&|i| base[i] >= 0.0),
        pct(&|i| final_elev[i] >= 0.0),
    );
    println!(
        "    final elevation >=1.5/3/5 km:            {:5.1}% / {:5.1}% / {:5.1}%",
        pct(&|i| final_elev[i] >= 0.15),
        pct(&|i| final_elev[i] >= 0.30),
        pct(&|i| final_elev[i] >= 0.50),
    );

    println!("  detail support (% footprint with one-hop local relief):");
    for (label, relief) in [
        ("coarse envelope", &coarse_relief),
        ("fine base", &base_relief),
        ("final eroded", &final_relief),
    ] {
        println!(
            "    {label:<15} >=50/150/300/600m: {:5.1}% / {:5.1}% / {:5.1}% / {:5.1}%   rms={:5.0}m",
            pct(&|i| relief[i] >= 0.005),
            pct(&|i| relief[i] >= 0.015),
            pct(&|i| relief[i] >= 0.030),
            pct(&|i| relief[i] >= 0.060),
            rms_m(relief),
        );
    }
    println!(
        "  stage deltas over footprint: structural rms={:.0}m (|Δ|>=50/100m {:4.1}%/{:4.1}%), erosion rms={:.0}m (|Δ|>=50/100m {:4.1}%/{:4.1}%)",
        rms_m(&structural),
        pct(&|i| structural[i].abs() >= 0.005),
        pct(&|i| structural[i].abs() >= 0.010),
        rms_m(&erosion),
        pct(&|i| erosion[i].abs() >= 0.005),
        pct(&|i| erosion[i].abs() >= 0.010),
    );

    // A thresholded local-relief network is intentionally footprint-relative:
    // unlike an absolute mountain mask it retains low-elevation structured belts.
    let relief_network: Vec<bool> = (0..n)
        .map(|i| footprint[i] && final_relief[i] >= 0.015)
        .collect();
    let components = measure_components(tess, &relief_network);
    let mut significant_elongation: Vec<f32> = components
        .iter()
        .filter(|c| c.area_km2 >= AUDIT_SIGNIFICANT_RANGE_KM2)
        .map(ComponentStats::elongation)
        .collect();
    significant_elongation.sort_by(f32::total_cmp);
    let median_elongation = significant_elongation
        .get(significant_elongation.len() / 2)
        .copied()
        .unwrap_or(0.0);
    println!(
        "  relief-network (>=150m one-hop): {} components, {} >=20k km², median elongation {:.1}",
        components.len(),
        significant_elongation.len(),
        median_elongation,
    );
}

// ======================== MOUNTAIN AUDIT ========================
//
// Object-level RANGE statistics on the fine eroded surface. Each range is a
// connected component of high land, measured as a feature: size/shape, crest
// continuity (the pass/gap spectrum along strike), cross-strike asymmetry,
// and sampled local relief. This is where "the mountains look wrong" gets
// localized: blob-shaped ranges (low elongation), unbroken walls (no passes),
// symmetric profiles (lost O0 asymmetry), or low local relief all have
// distinct signatures here. Earth refs inline.

const AUDIT_RANGE_ELEV: f32 = 0.15;
const AUDIT_M_PER_UNIT: f32 = ELEVATION_UNIT_KM * 1_000.0;
const AUDIT_SIGNIFICANT_RANGE_KM2: f32 = 20_000.0;

#[cfg(feature = "research-landscape")]
#[derive(Clone, Debug, serde::Serialize)]
struct RoofTraceStats {
    count: usize,
    min: Option<f64>,
    max: Option<f64>,
    mean: Option<f64>,
    std_dev: Option<f64>,
    cv: Option<f64>,
}

#[cfg(feature = "research-landscape")]
impl RoofTraceStats {
    fn positive(values: impl IntoIterator<Item = f32>) -> Self {
        let values: Vec<f64> = values
            .into_iter()
            .filter(|value| value.is_finite() && *value > 0.0)
            .map(f64::from)
            .collect();
        if values.is_empty() {
            return Self {
                count: 0,
                min: None,
                max: None,
                mean: None,
                std_dev: None,
                cv: None,
            };
        }
        let count = values.len();
        let mean = values.iter().sum::<f64>() / count as f64;
        let variance = values
            .iter()
            .map(|value| (value - mean) * (value - mean))
            .sum::<f64>()
            / count as f64;
        let std_dev = variance.sqrt();
        Self {
            count,
            min: values.iter().copied().reduce(f64::min),
            max: values.iter().copied().reduce(f64::max),
            mean: Some(mean),
            std_dev: Some(std_dev),
            cv: (mean > 0.0).then_some(std_dev / mean),
        }
    }
}

#[cfg(feature = "research-landscape")]
#[derive(Debug, serde::Serialize)]
struct RoofTraceStageRecord {
    stage: &'static str,
    domain: &'static str,
    units: &'static str,
    stats: RoofTraceStats,
}

#[cfg(feature = "research-landscape")]
#[derive(Debug, serde::Serialize)]
struct RoofTraceSurfaceRecord {
    surface: &'static str,
    peak_km: f32,
    cap_500m_km2: f64,
    cap_1000m_km2: f64,
    cap_500m_below_1pct_grade_fraction: f64,
    component_below_1pct_grade_fraction: f64,
}

#[cfg(feature = "research-landscape")]
#[derive(Debug, serde::Serialize)]
struct RoofTraceSourceRecord {
    selected_fronts: usize,
    geometric_fronts: usize,
    co_seed_fronts: usize,
    bridge_fronts: usize,
    diffuse_dependency_fronts: usize,
    collision_fronts: usize,
    subduction_fronts: usize,
    total_length_km: f64,
    episode_count: usize,
    convergence_km_per_myr: RoofTraceStats,
    episode_duration_myr: RoofTraceStats,
    shortening_area_opportunity_km2: f64,
}

#[cfg(feature = "research-landscape")]
#[derive(Debug, serde::Serialize)]
struct RoofTraceRangeRecord {
    roles: Vec<&'static str>,
    component_id: usize,
    fine_cells: usize,
    area_km2: f32,
    final_peak_km: f32,
    direct_coarse_cells: usize,
    interpolation_support_cells: usize,
    source: RoofTraceSourceRecord,
    stages: Vec<RoofTraceStageRecord>,
    amplitude_saturation_fraction: f64,
    reconstruction_max_abs_error: f32,
    nearest_source_work_scale: f32,
    episode_mean_work_scale: f32,
    collision_mean_uplift_km: f64,
    collision_share_of_positive_coarse_interpolant: f64,
    collision_supported_area_fraction: f64,
    surfaces: Vec<RoofTraceSurfaceRecord>,
}

#[cfg(feature = "research-landscape")]
#[derive(Debug, serde::Serialize)]
struct RoofCausalTraceReport {
    schema: &'static str,
    seed: u64,
    provenance: String,
    requested_coarse_cells: usize,
    actual_coarse_cells: usize,
    actual_fine_cells: usize,
    orogen_model: String,
    final_component_threshold_km: f32,
    significant_component_area_km2: f32,
    significant_component_count: usize,
    baseline_reinterpolation_max_abs_error: f32,
    ranges: Vec<RoofTraceRangeRecord>,
}

#[cfg(feature = "research-landscape")]
fn roof_trace_interpolate_coarse_field(
    coarse: &hex3::world::Tessellation,
    fine: &hex3::world::Tessellation,
    coarse_cell: &[usize],
    values: &[f32],
) -> Vec<f32> {
    (0..fine.num_cells())
        .map(|cell| {
            let position = fine.cell_center(cell);
            let nearest = coarse_cell[cell];
            let mut weighted = 0.0f32;
            let mut total_weight = 0.0f32;
            for source in std::iter::once(nearest).chain(coarse.neighbors(nearest).iter().copied())
            {
                let distance = coarse
                    .cell_center(source)
                    .dot(position)
                    .clamp(-1.0, 1.0)
                    .acos();
                let weight = 1.0 / (distance * distance + 1.0e-8);
                weighted += values[source] * weight;
                total_weight += weight;
            }
            weighted / total_weight
        })
        .collect()
}

#[cfg(feature = "research-landscape")]
fn roof_trace_surface_record(
    surface: &'static str,
    tess: &hex3::world::Tessellation,
    component_cells: &[usize],
    elevation: &[f32],
) -> RoofTraceSurfaceRecord {
    let peak = component_cells
        .iter()
        .map(|&cell| elevation[cell])
        .fold(f32::NEG_INFINITY, f32::max);
    let areas = tess.cell_areas_ref();
    let radius2 = f64::from(EARTH_RADIUS_KM) * f64::from(EARTH_RADIUS_KM);
    let mut cap_500m_km2 = 0.0;
    let mut cap_1000m_km2 = 0.0;
    let mut flat_500m_km2 = 0.0;
    let mut component_km2 = 0.0;
    let mut gentle_component_km2 = 0.0;
    for &cell in component_cells {
        let area_km2 = f64::from(areas[cell]) * radius2;
        component_km2 += area_km2;
        if elevation[cell] >= peak - 0.10 {
            cap_1000m_km2 += area_km2;
        }
        let center = tess.cell_center(cell);
        let max_downhill = tess
            .neighbors(cell)
            .iter()
            .map(|&neighbor| {
                let distance_km =
                    (center - tess.cell_center(neighbor)).length().max(1e-9) * EARTH_RADIUS_KM;
                (elevation[cell] - elevation[neighbor]).max(0.0) / distance_km
            })
            .fold(0.0f32, f32::max);
        // Elevation units are 10 km, hence 1e-3 units/km is a 1% grade.
        if max_downhill < 1.0e-3 {
            gentle_component_km2 += area_km2;
        }
        if elevation[cell] >= peak - 0.05 {
            cap_500m_km2 += area_km2;
            if max_downhill < 1.0e-3 {
                flat_500m_km2 += area_km2;
            }
        }
    }
    RoofTraceSurfaceRecord {
        surface,
        peak_km: peak * ELEVATION_UNIT_KM,
        cap_500m_km2,
        cap_1000m_km2,
        cap_500m_below_1pct_grade_fraction: if cap_500m_km2 > 0.0 {
            flat_500m_km2 / cap_500m_km2
        } else {
            0.0
        },
        component_below_1pct_grade_fraction: if component_km2 > 0.0 {
            gentle_component_km2 / component_km2
        } else {
            0.0
        },
    }
}

#[cfg(feature = "research-landscape")]
fn roof_trace_stats_line(stage: &RoofTraceStageRecord) -> String {
    let Some(mean) = stage.stats.mean else {
        return format!(
            "    {:<25} {:<14} {:>7}  no positive values",
            stage.stage, stage.domain, stage.stats.count
        );
    };
    format!(
        "    {:<25} {:<14} {:>7}  {:>10.3e} {:>10.3e} {:>10.3e} {:>10.3e} {:>7.3}",
        stage.stage,
        stage.domain,
        stage.stats.count,
        stage.stats.min.unwrap_or(mean),
        stage.stats.max.unwrap_or(mean),
        mean,
        stage.stats.std_dev.unwrap_or(0.0),
        stage.stats.cv.unwrap_or(0.0),
    )
}

#[cfg(feature = "research-landscape")]
fn run_roof_causal_trace(world: &World, seed: u64, requested_cells: usize, out: Option<&Path>) {
    use std::collections::{BTreeMap, BTreeSet};

    use hex3::world::{
        attribute_legacy_convergent_sources, collect_convergent_fronts, collect_plate_boundaries,
        StructuralRegime,
    };

    if world.orogen_model != OrogenModel::Legacy {
        eprintln!(
            "--roof-causal-trace requires --orogen-model legacy; selected {}",
            world.orogen_model
        );
        std::process::exit(2);
    }

    let Some(fine) = world.fine.as_ref() else {
        println!("\n[ROOF CAUSAL TRACE] no fine surface present");
        return;
    };
    let (Some(plates), Some(crust), Some(dynamics), Some(features), Some(history)) = (
        world.plates.as_ref(),
        world.crust.as_ref(),
        world.dynamics.as_ref(),
        world.features.as_ref(),
        world.tectonic_history.as_ref(),
    ) else {
        println!("\n[ROOF CAUSAL TRACE] incomplete coarse tectonic state");
        return;
    };

    let fine_tess = fine.tessellation();
    let final_surface = fine.surface_for(u32::MAX);
    let final_elevation = &final_surface.elevation.values;
    let mask: Vec<bool> = final_elevation
        .iter()
        .map(|&elevation| elevation >= AUDIT_RANGE_ELEV)
        .collect();
    let components = measure_components(fine_tess, &mask);
    let significant: Vec<&ComponentStats> = components
        .iter()
        .filter(|component| component.area_km2 >= AUDIT_SIGNIFICANT_RANGE_KM2)
        .collect();

    println!(
        "\n================ ROOF CAUSAL TRACE seed={} model={} ================",
        seed, world.orogen_model
    );
    println!(
        "  fixed domain: final fine elevation >=1.5 km; {} significant components >=20,000 km²",
        significant.len()
    );
    if significant.is_empty() {
        println!("  no significant ranges to trace");
        return;
    }

    let boundaries = collect_plate_boundaries(&world.tessellation, plates, crust, dynamics);
    let fronts = match collect_convergent_fronts(&world.tessellation, &boundaries, history) {
        Ok(fronts) => fronts,
        Err(error) => {
            println!("  source collection failed: {error}");
            return;
        }
    };
    let front_by_id: BTreeMap<_, _> = fronts.edges.iter().map(|front| (front.id, front)).collect();

    let highest = significant
        .iter()
        .copied()
        .max_by(|left, right| {
            let peak = |component: &ComponentStats| {
                component
                    .cells
                    .iter()
                    .map(|&cell| final_elevation[cell])
                    .fold(f32::NEG_INFINITY, f32::max)
            };
            peak(left)
                .total_cmp(&peak(right))
                .then_with(|| right.cells[0].cmp(&left.cells[0]))
        })
        .expect("significant checked nonempty");
    // Keep the two roles distinct when possible so one enormous range cannot
    // hide the next-largest ordinary roof in the fixed corpus.
    let largest = significant
        .iter()
        .copied()
        .find(|component| component.cells[0] != highest.cells[0])
        .unwrap_or(highest);
    let mut selected: BTreeMap<usize, (Vec<&'static str>, &ComponentStats)> = BTreeMap::new();
    selected.insert(highest.cells[0], (vec!["highest-peak"], highest));
    selected
        .entry(largest.cells[0])
        .and_modify(|(roles, _)| roles.push("largest-area"))
        .or_insert_with(|| (vec!["largest-area"], largest));

    let raw_eroded_pre_integration: Vec<f32> = (0..fine_tess.num_cells())
        .map(|cell| final_surface.hydrology.pre_integration_elevation(cell))
        .collect();
    let trace = &features.legacy_collision_trace;
    let coarse_elevation = &world.elevation.as_ref().expect("coarse elevation").values;
    let replace_collision = |replacement: &[f32]| -> Vec<f32> {
        coarse_elevation
            .iter()
            .zip(&features.collision)
            .zip(replacement)
            .map(|((&elevation, &legacy_collision), &counterfactual)| {
                elevation - legacy_collision + counterfactual
            })
            .collect()
    };
    let reinterpolated_baseline = roof_trace_interpolate_coarse_field(
        &world.tessellation,
        fine_tess,
        &fine.base.coarse_cell,
        coarse_elevation,
    );
    let baseline_reinterpolation_max_abs_error = reinterpolated_baseline
        .iter()
        .zip(&fine.base.coarse_base_elevation)
        .map(|(&reconstructed, &retained)| (reconstructed - retained).abs())
        .fold(0.0f32, f32::max);
    let nearest_source_fine = roof_trace_interpolate_coarse_field(
        &world.tessellation,
        fine_tess,
        &fine.base.coarse_cell,
        &replace_collision(&trace.nearest_source_matched_response),
    );
    let episode_mean_fine = roof_trace_interpolate_coarse_field(
        &world.tessellation,
        fine_tess,
        &fine.base.coarse_cell,
        &replace_collision(&trace.episode_mean_matched_response),
    );
    println!(
        "  diagnostic coarse->fine reconstruction max error: {:.3e}",
        baseline_reinterpolation_max_abs_error
    );
    let mut range_records = Vec::with_capacity(selected.len());

    for (component_id, (roles, component)) in selected {
        let attribution = match attribute_legacy_convergent_sources(
            &world.tessellation,
            fine_tess,
            &fine.base.coarse_cell,
            plates,
            crust,
            features,
            &boundaries,
            &fronts,
            &component.cells,
        ) {
            Ok(attribution) => attribution,
            Err(error) => {
                println!(
                    "  {:?} component {} attribution failed: {error}",
                    roles, component_id
                );
                continue;
            }
        };
        let selected_fronts: Vec<_> = attribution
            .selected_source_edges
            .iter()
            .filter_map(|id| front_by_id.get(id).copied())
            .collect();
        let source_cells: BTreeSet<usize> = selected_fronts
            .iter()
            .flat_map(|front| front.cells)
            .collect();
        let direct_coarse: BTreeSet<usize> = component
            .cells
            .iter()
            .map(|&cell| fine.base.coarse_cell[cell])
            .collect();
        let roof_support: BTreeSet<usize> = attribution.coarse_read_cells.iter().copied().collect();

        let stage =
            |stage, domain, units, values: &[f32], cells: &BTreeSet<usize>| RoofTraceStageRecord {
                stage,
                domain,
                units,
                stats: RoofTraceStats::positive(cells.iter().map(|&cell| values[cell])),
            };
        let stages = vec![
            stage(
                "raw collision seed",
                "source cells",
                "legacy forcing*radian",
                &trace.raw_collision_seed,
                &source_cells,
            ),
            stage(
                "normalized seed",
                "source cells",
                "legacy forcing",
                &trace.normalized_collision_seed,
                &source_cells,
            ),
            stage(
                "smoothed forcing",
                "roof support",
                "legacy forcing",
                &trace.smoothed_collision_forcing,
                &roof_support,
            ),
            stage(
                "uncapped sqrt amplitude",
                "roof support",
                "elevation units",
                &trace.uncapped_sqrt_amplitude,
                &roof_support,
            ),
            stage(
                "capped amplitude",
                "roof support",
                "elevation units",
                &trace.capped_amplitude,
                &roof_support,
            ),
            stage(
                "Gaussian distance kernel",
                "roof support",
                "dimensionless",
                &trace.gaussian_distance_kernel,
                &roof_support,
            ),
            stage(
                "final collision response",
                "roof support",
                "elevation units",
                &trace.final_collision_response,
                &roof_support,
            ),
            stage(
                "nearest-source forcing",
                "roof support",
                "legacy forcing",
                &trace.nearest_source_forcing,
                &roof_support,
            ),
            stage(
                "nearest-source matched",
                "roof support",
                "elevation units",
                &trace.nearest_source_matched_response,
                &roof_support,
            ),
            stage(
                "episode-mean forcing",
                "roof support",
                "legacy forcing",
                &trace.episode_mean_forcing,
                &roof_support,
            ),
            stage(
                "episode-mean matched",
                "roof support",
                "elevation units",
                &trace.episode_mean_matched_response,
                &roof_support,
            ),
        ];

        let positive_amplitudes: Vec<usize> = roof_support
            .iter()
            .copied()
            .filter(|&cell| trace.uncapped_sqrt_amplitude[cell] > 0.0)
            .collect();
        let saturated = positive_amplitudes
            .iter()
            .filter(|&&cell| {
                trace.uncapped_sqrt_amplitude[cell] > trace.capped_amplitude[cell] + 1.0e-7
            })
            .count();
        let amplitude_saturation_fraction = if positive_amplitudes.is_empty() {
            0.0
        } else {
            saturated as f64 / positive_amplitudes.len() as f64
        };
        let reconstruction_max_abs_error = roof_support
            .iter()
            .map(|&cell| {
                (trace.capped_amplitude[cell] * trace.gaussian_distance_kernel[cell]
                    - trace.final_collision_response[cell])
                    .abs()
            })
            .fold(0.0f32, f32::max);

        let fine_areas = fine_tess.cell_areas_ref();
        let mut component_area = 0.0f64;
        let mut collision_area_integral = 0.0f64;
        let mut positive_coarse_integral = 0.0f64;
        let mut collision_supported_area = 0.0f64;
        for &cell in &component.cells {
            let area = f64::from(fine_areas[cell]);
            let collision = f64::from(fine.base.fields.elevation_fields.collision[cell].max(0.0));
            component_area += area;
            collision_area_integral += collision * area;
            positive_coarse_integral +=
                f64::from(fine.base.coarse_base_elevation[cell].max(0.0)) * area;
            if collision > 0.0 {
                collision_supported_area += area;
            }
        }

        let mut episode_duration = BTreeMap::new();
        for front in &selected_fronts {
            episode_duration
                .entry(front.episode_id)
                .or_insert(front.episode_duration_myr);
        }
        let source_record = RoofTraceSourceRecord {
            selected_fronts: selected_fronts.len(),
            geometric_fronts: attribution.geometric_source_edges.len(),
            co_seed_fronts: attribution.co_seed_source_edges.len(),
            bridge_fronts: attribution.bridge_source_edges.len(),
            diffuse_dependency_fronts: attribution.diffuse_dependency_edges.len(),
            collision_fronts: selected_fronts
                .iter()
                .filter(|front| front.regime == StructuralRegime::Collision)
                .count(),
            subduction_fronts: selected_fronts
                .iter()
                .filter(|front| front.regime == StructuralRegime::Subduction)
                .count(),
            total_length_km: selected_fronts
                .iter()
                .map(|front| f64::from(front.length_km))
                .sum(),
            episode_count: episode_duration.len(),
            convergence_km_per_myr: RoofTraceStats::positive(
                selected_fronts
                    .iter()
                    .map(|front| front.convergence_km_per_myr),
            ),
            episode_duration_myr: RoofTraceStats::positive(episode_duration.values().copied()),
            shortening_area_opportunity_km2: selected_fronts
                .iter()
                .map(|front| front.shortening_area_opportunity_km2)
                .sum(),
        };

        let surfaces = vec![
            roof_trace_surface_record(
                "fine coarse interpolant",
                fine_tess,
                &component.cells,
                &fine.base.coarse_base_elevation,
            ),
            roof_trace_surface_record(
                "fine base",
                fine_tess,
                &component.cells,
                &fine.base.base_elevation,
            ),
            roof_trace_surface_record(
                "nearest-source compiler",
                fine_tess,
                &component.cells,
                &nearest_source_fine,
            ),
            roof_trace_surface_record(
                "episode-mean null compiler",
                fine_tess,
                &component.cells,
                &episode_mean_fine,
            ),
            roof_trace_surface_record(
                "raw eroded pre-integration",
                fine_tess,
                &component.cells,
                &raw_eroded_pre_integration,
            ),
            roof_trace_surface_record("final", fine_tess, &component.cells, final_elevation),
        ];
        let record = RoofTraceRangeRecord {
            roles,
            component_id,
            fine_cells: component.cells.len(),
            area_km2: component.area_km2,
            final_peak_km: component
                .cells
                .iter()
                .map(|&cell| final_elevation[cell])
                .fold(f32::NEG_INFINITY, f32::max)
                * ELEVATION_UNIT_KM,
            direct_coarse_cells: direct_coarse.len(),
            interpolation_support_cells: roof_support.len(),
            source: source_record,
            stages,
            amplitude_saturation_fraction,
            reconstruction_max_abs_error,
            nearest_source_work_scale: trace.nearest_source_work_scale,
            episode_mean_work_scale: trace.episode_mean_work_scale,
            collision_mean_uplift_km: if component_area > 0.0 {
                ELEVATION_UNIT_KM as f64 * collision_area_integral / component_area
            } else {
                0.0
            },
            collision_share_of_positive_coarse_interpolant: if positive_coarse_integral > 0.0 {
                collision_area_integral / positive_coarse_integral
            } else {
                0.0
            },
            collision_supported_area_fraction: if component_area > 0.0 {
                collision_supported_area / component_area
            } else {
                0.0
            },
            surfaces,
        };

        println!(
            "\n  {:?}: component={} cells={} area={:.0} km² final-peak={:.2} km",
            record.roles,
            record.component_id,
            record.fine_cells,
            record.area_km2,
            record.final_peak_km
        );
        println!(
            "    ancestry: selected={} (geometric={} co-seed={} bridge={}; diffuse dependencies={}) | collision/subduction={}/{}",
            record.source.selected_fronts,
            record.source.geometric_fronts,
            record.source.co_seed_fronts,
            record.source.bridge_fronts,
            record.source.diffuse_dependency_fronts,
            record.source.collision_fronts,
            record.source.subduction_fronts,
        );
        println!(
            "    source: length={:.0} km episodes={} convergence min/mean/max={:.2}/{:.2}/{:.2} km/Myr duration min/mean/max={:.1}/{:.1}/{:.1} Myr opportunity={:.3e} km²",
            record.source.total_length_km,
            record.source.episode_count,
            record.source.convergence_km_per_myr.min.unwrap_or(0.0),
            record.source.convergence_km_per_myr.mean.unwrap_or(0.0),
            record.source.convergence_km_per_myr.max.unwrap_or(0.0),
            record.source.episode_duration_myr.min.unwrap_or(0.0),
            record.source.episode_duration_myr.mean.unwrap_or(0.0),
            record.source.episode_duration_myr.max.unwrap_or(0.0),
            record.source.shortening_area_opportunity_km2,
        );
        println!(
            "    transform stages (positive values only):                 count         min        max       mean        std      CV"
        );
        for stage in &record.stages {
            println!("{}", roof_trace_stats_line(stage));
        }
        println!(
            "    amplitude: saturated={:.1}% reconstruction max|capped*kernel-response|={:.3e}",
            100.0 * record.amplitude_saturation_fraction,
            record.reconstruction_max_abs_error
        );
        println!(
            "    compiler controls: nearest-source work scale={:.5}; episode-mean work scale={:.5}",
            record.nearest_source_work_scale, record.episode_mean_work_scale
        );
        println!(
            "    collision contribution: mean={:.2} km; {:.1}% of positive coarse-interpolant elevation-area; supported area={:.1}%",
            record.collision_mean_uplift_km,
            100.0 * record.collision_share_of_positive_coarse_interpolant,
            100.0 * record.collision_supported_area_fraction,
        );
        println!(
            "    fixed-mask surfaces:                         peak km    cap500 km²  cap1000 km²  cap500<1%   all<1%"
        );
        for surface in &record.surfaces {
            println!(
                "      {:<36} {:>7.2} {:>12.0} {:>12.0} {:>10.1}% {:>8.1}%",
                surface.surface,
                surface.peak_km,
                surface.cap_500m_km2,
                surface.cap_1000m_km2,
                100.0 * surface.cap_500m_below_1pct_grade_fraction,
                100.0 * surface.component_below_1pct_grade_fraction,
            );
        }
        range_records.push(record);
    }

    let report = RoofCausalTraceReport {
        schema: "hex3.roof-causal-trace.v1",
        seed,
        provenance: world.manifest().summary(),
        requested_coarse_cells: requested_cells,
        actual_coarse_cells: world.tessellation.num_cells(),
        actual_fine_cells: fine_tess.num_cells(),
        orogen_model: world.orogen_model.to_string(),
        final_component_threshold_km: AUDIT_RANGE_ELEV * ELEVATION_UNIT_KM,
        significant_component_area_km2: AUDIT_SIGNIFICANT_RANGE_KM2,
        significant_component_count: significant.len(),
        baseline_reinterpolation_max_abs_error,
        ranges: range_records,
    };
    if let Some(path) = out {
        let json = serde_json::to_string_pretty(&report).expect("roof trace serializes");
        if let Err(error) = std::fs::write(path, format!("{json}\n")) {
            eprintln!(
                "failed to write roof causal trace {}: {error}",
                path.display()
            );
            std::process::exit(2);
        }
        eprintln!("wrote roof causal trace {}", path.display());
    }
}

/// The (sampled) farthest-apart pair of cells in a component — the range's
/// principal axis endpoints.
fn audit_extremal_pair(tess: &hex3::world::Tessellation, cells: &[usize]) -> (usize, usize) {
    const MAX_SAMPLE: usize = 200;
    let stride = (cells.len() / MAX_SAMPLE).max(1);
    let sample: Vec<usize> = cells.iter().step_by(stride).copied().collect();
    let (mut best, mut pair) = (-1.0f32, (cells[0], cells[0]));
    for (i, &a) in sample.iter().enumerate() {
        let pa = tess.cell_center(a);
        for &b in &sample[i + 1..] {
            let d = (pa - tess.cell_center(b)).length_squared();
            if d > best {
                best = d;
                pair = (a, b);
            }
        }
    }
    pair
}

struct RangeAudit {
    area_km2: f32,
    length_km: f32,
    width_km: f32,
    peak_m: f32,
    /// Crest bins along the principal axis that contain range cells.
    crest_bins: usize,
    /// Distinct passes: contiguous crest-profile lows ≥100 m below both
    /// flanking crest maxima.
    passes: usize,
    /// Median pass depth (m below the lower flanking crest max).
    median_pass_depth_m: f32,
    /// Lowest crest-profile point (m) — the range's deepest through-gap.
    crest_floor_m: f32,
    /// Crest offset across strike: 0 = crest centered, ±1 = crest hugs an
    /// edge (asymmetric range profile, the O0 signature).
    crest_offset: f32,
}

fn audit_range(
    tess: &hex3::world::Tessellation,
    comp: &ComponentStats,
    elev: &[f32],
) -> RangeAudit {
    let (pa, pb) = audit_extremal_pair(tess, &comp.cells);
    let a = tess.cell_center(pa);
    let b = tess.cell_center(pb);
    let nrm = a.cross(b).normalize_or_zero();
    let c_axis = nrm.cross(a).normalize_or_zero();
    let arc = a.dot(b).clamp(-1.0, 1.0).acos();

    // Along-strike parameter t (radians along the great-circle axis) and
    // signed cross-strike parameter s (radians off the axis) per cell.
    let param = |i: usize| -> (f32, f32) {
        let p = tess.cell_center(i);
        let t = p.dot(c_axis).atan2(p.dot(a));
        let s = p.dot(nrm).clamp(-1.0, 1.0).asin();
        (t, s)
    };

    // Crest profile: max elevation per ~50 km bin along strike.
    let bin_w = 50.0 / EARTH_RADIUS_KM;
    let nbins = ((arc / bin_w).ceil() as usize).clamp(1, 4096);
    let mut crest = vec![f32::NEG_INFINITY; nbins];
    let mut ss: Vec<f32> = Vec::with_capacity(comp.cells.len());
    let mut peak = f32::NEG_INFINITY;
    for &i in &comp.cells {
        let (t, s) = param(i);
        let bin = ((t / arc.max(1e-9)) * nbins as f32) as usize;
        let bin = bin.min(nbins - 1);
        crest[bin] = crest[bin].max(elev[i]);
        ss.push(s);
        peak = peak.max(elev[i]);
    }
    let filled: Vec<(usize, f32)> = crest
        .iter()
        .enumerate()
        .filter(|(_, &e)| e.is_finite())
        .map(|(b, &e)| (b, e))
        .collect();

    // Pass detection on the filled crest profile: prefix/suffix maxima.
    let m = filled.len();
    let mut passes = 0usize;
    let mut pass_depths: Vec<f32> = Vec::new();
    let mut crest_floor = peak;
    if m >= 3 {
        let mut pre = vec![f32::NEG_INFINITY; m];
        let mut suf = vec![f32::NEG_INFINITY; m];
        for k in 1..m {
            pre[k] = pre[k - 1].max(filled[k - 1].1);
        }
        for k in (0..m - 1).rev() {
            suf[k] = suf[k + 1].max(filled[k + 1].1);
        }
        const PASS_MIN_DEPTH: f32 = 0.01; // 100 m
        let mut in_pass = false;
        let mut cur_depth = 0.0f32;
        for k in 1..m - 1 {
            let depth = pre[k].min(suf[k]) - filled[k].1;
            crest_floor = crest_floor.min(filled[k].1);
            if depth >= PASS_MIN_DEPTH {
                in_pass = true;
                cur_depth = cur_depth.max(depth);
            } else if in_pass {
                passes += 1;
                pass_depths.push(cur_depth);
                in_pass = false;
                cur_depth = 0.0;
            }
        }
        if in_pass {
            passes += 1;
            pass_depths.push(cur_depth);
        }
    }
    pass_depths.sort_by(f32::total_cmp);
    let median_pass_depth = pass_depths
        .get(pass_depths.len() / 2)
        .copied()
        .unwrap_or(0.0);

    // Cross-strike asymmetry: where does the crest sit inside the range's
    // s-envelope? Crest cells = top 20% of the elevation band.
    ss.sort_by(f32::total_cmp);
    let sq = |q: f32| ss[(((ss.len() - 1) as f32) * q) as usize];
    let (s_lo, s_hi) = (sq(0.05), sq(0.95));
    let half = ((s_hi - s_lo) * 0.5).max(1e-9);
    let center = (s_hi + s_lo) * 0.5;
    let crest_thr = peak - 0.2 * (peak - AUDIT_RANGE_ELEV);
    let mut s_crest: Vec<f32> = comp
        .cells
        .iter()
        .filter(|&&i| elev[i] >= crest_thr)
        .map(|&i| param(i).1)
        .collect();
    s_crest.sort_by(f32::total_cmp);
    let s_crest_med = s_crest.get(s_crest.len() / 2).copied().unwrap_or(center);
    let crest_offset = ((s_crest_med - center) / half).clamp(-1.5, 1.5);

    RangeAudit {
        area_km2: comp.area_km2,
        length_km: comp.length_km,
        width_km: comp.width_km,
        peak_m: peak * AUDIT_M_PER_UNIT,
        crest_bins: m,
        passes,
        median_pass_depth_m: median_pass_depth * AUDIT_M_PER_UNIT,
        crest_floor_m: crest_floor * AUDIT_M_PER_UNIT,
        crest_offset,
    }
}

/// Crest-train stats for one range: ridge spacings along cross-strike transects
/// (does the meso band read as a metronome, a quasi-periodic fold belt, or
/// drainage-organized?) + flow-orientation split vs strike. Metronomic train:
/// spacing CV ~0.1-0.2 and longitudinal-dominant flow; natural fold belts are
/// quasi-periodic (CV ~0.4+); drainage-organized alpine has no dominant spacing
/// and transverse-dominant flow.
struct CrestTrainStats {
    /// Crest-to-crest spacings (km) pooled over transects.
    spacings_km: Vec<f32>,
    /// Ridge crests per transect (prominence-filtered).
    ridges_per_transect: Vec<f32>,
    /// Mountain cells whose steepest-descent direction is within 30° of strike.
    longitudinal_cells: usize,
    /// ... within 30° of the cross-strike normal.
    transverse_cells: usize,
    total_cells: usize,
    /// Same split restricted to TRUNK cells (top-decile flow accumulation within
    /// the component) — hillslope cells are near-isotropic in every terrain class,
    /// so the all-cell split can't see drainage grammar; trunks can.
    trunk_longitudinal: f64,
    trunk_transverse: f64,
    trunk_total: f64,
}

fn crest_train_stats(
    tess: &hex3::world::Tessellation,
    comp: &ComponentStats,
    elev: &[f32],
    accum: &[f32],
) -> CrestTrainStats {
    let (pa, pb) = audit_extremal_pair(tess, &comp.cells);
    let a = tess.cell_center(pa);
    let b = tess.cell_center(pb);
    let nrm = a.cross(b).normalize_or_zero();
    let c_axis = nrm.cross(a).normalize_or_zero();
    let arc = a.dot(b).clamp(-1.0, 1.0).acos().max(1e-9);
    let param = |i: usize| -> (f32, f32) {
        let p = tess.cell_center(i);
        let t = p.dot(c_axis).atan2(p.dot(a));
        let s = p.dot(nrm).clamp(-1.0, 1.0).asin();
        (t, s)
    };

    // Transect slabs along strike; crest profile across strike per slab.
    const SLAB_KM: f32 = 15.0; // along-strike slab width
    const SBIN_KM: f32 = 4.0; // cross-strike profile resolution (~mesh scale)
    const PROMINENCE: f32 = 0.015; // 150 m — a ridge, not surface noise
    let slab_w = SLAB_KM / EARTH_RADIUS_KM;
    let sbin_w = SBIN_KM / EARTH_RADIUS_KM;
    let nslabs = ((arc / slab_w).ceil() as usize).clamp(1, 4096);
    // (slab, s-bin) -> max elevation. s ranges over +-width; offset bins by s_min.
    let mut s_min = f32::INFINITY;
    let mut s_max = f32::NEG_INFINITY;
    let params: Vec<(usize, f32, f32)> = comp
        .cells
        .iter()
        .map(|&i| {
            let (t, s) = param(i);
            s_min = s_min.min(s);
            s_max = s_max.max(s);
            (i, t, s)
        })
        .collect();
    let nsbins = (((s_max - s_min) / sbin_w).ceil() as usize).clamp(1, 4096);
    let mut profiles = vec![f32::NEG_INFINITY; nslabs * nsbins];
    for &(i, t, s) in &params {
        let slab = (((t / arc) * nslabs as f32) as usize).min(nslabs - 1);
        let sb = ((((s - s_min) / sbin_w) as usize).max(0)).min(nsbins - 1);
        let e = &mut profiles[slab * nsbins + sb];
        *e = e.max(elev[i]);
    }

    let mut spacings_km = Vec::new();
    let mut ridges_per_transect = Vec::new();
    for slab in 0..nslabs {
        let prof: Vec<(usize, f32)> = (0..nsbins)
            .filter(|&sb| profiles[slab * nsbins + sb].is_finite())
            .map(|sb| (sb, profiles[slab * nsbins + sb]))
            .collect();
        if prof.len() < 8 {
            continue; // too narrow here to read a train
        }
        // Local maxima, then prominence filter: drop a peak if the saddle toward
        // every higher peak is shallower than PROMINENCE (O(n²) on tiny profiles).
        let mut peaks: Vec<usize> = (1..prof.len() - 1)
            .filter(|&k| prof[k].1 > prof[k - 1].1 && prof[k].1 >= prof[k + 1].1)
            .collect();
        peaks = peaks
            .iter()
            .copied()
            .filter(|&k| {
                let mut prom = f32::INFINITY;
                for (dir, range) in [(-1i32, 0..k), (1, k + 1..prof.len())] {
                    let _ = dir;
                    let mut saddle = prof[k].1;
                    let mut bounded = false;
                    let it: Box<dyn Iterator<Item = usize>> = if range.start == 0 {
                        Box::new(range.rev())
                    } else {
                        Box::new(range)
                    };
                    for j in it {
                        saddle = saddle.min(prof[j].1);
                        if prof[j].1 > prof[k].1 {
                            prom = prom.min(prof[k].1 - saddle);
                            bounded = true;
                            break;
                        }
                    }
                    if !bounded {
                        // edge side: prominence vs the profile minimum on that side
                        prom = prom.min(prof[k].1 - saddle);
                    }
                }
                prom >= PROMINENCE
            })
            .collect();
        ridges_per_transect.push(peaks.len() as f32);
        for w in peaks.windows(2) {
            let d_bins = (prof[w[1]].0 - prof[w[0]].0) as f32;
            spacings_km.push(d_bins * SBIN_KM);
        }
    }

    // Flow orientation vs strike: steepest-descent neighbor direction, classified
    // by angle to the local along-strike tangent.
    let mut acc_sorted: Vec<f32> = params.iter().map(|&(i, _, _)| accum[i]).collect();
    acc_sorted.sort_by(f32::total_cmp);
    let trunk_thr = acc_sorted[(((acc_sorted.len() - 1) as f32) * 0.9) as usize];
    let (mut longitudinal, mut transverse, mut total) = (0usize, 0usize, 0usize);
    let (mut t_lon, mut t_tra, mut t_tot) = (0f64, 0f64, 0f64);
    for &(i, _, _) in &params {
        let p = tess.cell_center(i);
        let mut best = (elev[i], i);
        for &nb in tess.neighbors(i) {
            if elev[nb] < best.0 {
                best = (elev[nb], nb);
            }
        }
        if best.1 == i {
            continue; // pit
        }
        let e = (tess.cell_center(best.1) - p).normalize_or_zero();
        let strike_dir = nrm.cross(p).normalize_or_zero();
        let cross_dir = (nrm - p * nrm.dot(p)).normalize_or_zero();
        let (cs, cc) = (e.dot(strike_dir).abs(), e.dot(cross_dir).abs());
        let angle = cc.atan2(cs).to_degrees(); // 0 = along strike, 90 = across
        total += 1;
        if angle < 30.0 {
            longitudinal += 1;
        } else if angle > 60.0 {
            transverse += 1;
        }
        if accum[i] >= trunk_thr {
            t_tot += 1.0;
            if angle < 30.0 {
                t_lon += 1.0;
            } else if angle > 60.0 {
                t_tra += 1.0;
            }
        }
    }

    CrestTrainStats {
        spacings_km,
        ridges_per_transect,
        longitudinal_cells: longitudinal,
        transverse_cells: transverse,
        total_cells: total,
        trunk_longitudinal: t_lon,
        trunk_transverse: t_tra,
        trunk_total: t_tot,
    }
}

/// Erosion roughness counters + mountain-top plateau probe — the artifact gates
/// (pit%/checker%/curv-rms + summit cottage-cheese). Shared by the default fine
/// diagnostics and `--mountain-audit` so gate runs emit them without the full panel.
fn run_roughness_probe(world: &World) {
    let Some(fine) = world.fine.as_ref() else {
        return;
    };
    // ---- Erosion roughness counters (artifact vs. genuine dissection) ----
    // pit% is the swiss-cheese meter (a drained surface is ~0); checkerboard%
    // the SFD-groove banding; aspect R/entropy catch GLOBAL anisotropy / mesh-
    // axis locking only (a local spiral with balanced azimuths stays hidden —
    // judge spirals on the map). Pre-erosion (stage 3) vs eroded (stage 4) on
    // the SAME fine mesh. NOTE: the land mask is recomputed per surface, so the
    // `land` column shifts if erosion moves cells across sea level — a falling
    // pit% with a falling `land` may be submergence, not artifact removal.
    // See docs/archive/specs/erosion.md.
    let ftess = fine.tessellation();
    let pre = hex3::world::roughness_counters(ftess, &fine.surface_for(3).elevation.values);
    let ero = hex3::world::roughness_counters(ftess, &fine.surface_for(4).elevation.values);
    println!(
            "\n-- Erosion roughness counters (reroute-interval {})  [pit% is the swiss-cheese meter; lower better] --",
            world.erosion_params.reroute_interval
        );
    println!(
        "             {:>9} {:>10} {:>10} {:>10} {:>12} {:>9} {:>9}",
        "land", "pit%", "peak%", "checker%", "curv-rms", "aspectR", "entropy"
    );
    let row = |label: &str, c: &hex3::world::RoughnessCounters| {
        println!(
            "  {:<8}   {:>9} {:>10.3} {:>10.3} {:>10.2} {:>12.3e} {:>9.3} {:>9.3}",
            label,
            c.land,
            c.pit_pct,
            c.peak_pct,
            c.checkerboard_pct,
            c.curv_rms,
            c.aspect_r,
            c.aspect_entropy
        );
    };
    row("pre", &pre);
    row("eroded", &ero);
    println!(
        "  {:<8}   {:>+9} {:>+10.3} {:>+10.3} {:>+10.2} {:>12} {:>+9.3} {:>+9.3}",
        "delta",
        ero.land as i64 - pre.land as i64,
        ero.pit_pct - pre.pit_pct,
        ero.peak_pct - pre.peak_pct,
        ero.checkerboard_pct - pre.checkerboard_pct,
        "",
        ero.aspect_r - pre.aspect_r,
        ero.aspect_entropy - pre.aspect_entropy,
    );

    // ---- Mountain-top plateau probe (the localized "cottage cheese on flat
    // summits" artifact). Global counters are blind to a summit-only pattern,
    // so restrict to the highest land (top elevation decile) and compare
    // pre-erosion vs eroded THERE. Also report summit max-downhill slope: low
    // slope = genuinely flat-topped (plateau). If the texture is already in
    // `pre`, it's the fine-base synthesis (interp + noise), not erosion/routing.
    {
        let pre_e = &fine.surface_for(3).elevation.values;
        let ero_e = &fine.surface_for(4).elevation.values;
        let mut land_el: Vec<f32> = ero_e.iter().copied().filter(|&e| e >= 0.0).collect();
        land_el.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let thr = land_el[((land_el.len() as f32 * 0.90) as usize).min(land_el.len() - 1)];
        let mask = |src: &[f32]| -> Vec<f32> {
            (0..ftess.num_cells())
                .map(|i| if ero_e[i] >= thr { src[i] } else { -1.0 })
                .collect()
        };
        let pre_top = hex3::world::roughness_counters(ftess, &mask(pre_e));
        let ero_top = hex3::world::roughness_counters(ftess, &mask(ero_e));
        // Summit max-downhill slope (elev/km), pre and eroded, for flatness.
        let summit_slopes = |elev: &[f32]| -> Vec<f32> {
            let mut s = Vec::new();
            for i in 0..ftess.num_cells() {
                if ero_e[i] < thr {
                    continue;
                }
                let ci = ftess.cell_center(i);
                let mut g = 0.0f32;
                for &nb in ftess.neighbors(i) {
                    let d = (ci - ftess.cell_center(nb)).length().max(1e-9) * EARTH_RADIUS_KM;
                    g = g.max((elev[i] - elev[nb]) / d);
                }
                s.push(g);
            }
            s.sort_by(|a, b| a.partial_cmp(b).unwrap());
            s
        };
        let pct = |v: &[f32], p: f32| v[(((v.len() as f32) * p) as usize).min(v.len() - 1)];
        let (sp, se) = (summit_slopes(pre_e), summit_slopes(ero_e));
        println!(
            "\n-- Mountain-top plateau probe (top elev decile, elev>={:.3}) --",
            thr
        );
        row("top pre", &pre_top);
        row("top eroded", &ero_top);
        println!(
                "  summit max-downhill slope (elev/km): pre p50 {:.3e} p90 {:.3e} | eroded p50 {:.3e} p90 {:.3e}  (global land slope p50 ~3.8e-4; << = flat plateau)",
                pct(&sp, 0.5),
                pct(&sp, 0.9),
                pct(&se, 0.5),
                pct(&se, 0.9),
            );
    }
}

fn run_mountain_audit(world: &World, seed: u64, top: usize) {
    let Some(fine) = world.fine.as_ref() else {
        println!("\n[MOUNTAIN AUDIT] no fine surface present");
        return;
    };
    let tess = fine.tessellation();
    let surf = fine.surface_for(u32::MAX);
    let elev = &surf.elevation.values;
    let n = tess.num_cells();
    let areas = tess.cell_areas();

    println!(
        "\n================ MOUNTAIN AUDIT seed={} (fine eroded, elev>={}) ================",
        seed, AUDIT_RANGE_ELEV
    );

    let mask: Vec<bool> = (0..n).map(|i| elev[i] >= AUDIT_RANGE_ELEV).collect();
    let comps = measure_components(tess, &mask);
    let land_km2: f64 = (0..n)
        .filter(|&i| elev[i] >= 0.0)
        .map(|i| (areas[i] * EARTH_RADIUS_KM * EARTH_RADIUS_KM) as f64)
        .sum();
    let mtn_km2: f64 = comps.iter().map(|c| c.area_km2 as f64).sum();
    let significant: Vec<&ComponentStats> = comps
        .iter()
        .filter(|c| c.area_km2 >= AUDIT_SIGNIFICANT_RANGE_KM2)
        .collect();
    println!(
        "  mountain land: {:.1}% ({} components, {} significant >=20k km²)  [Earth high-mountain land ~10-12%]",
        100.0 * mtn_km2 / land_km2.max(1e-30),
        comps.len(),
        significant.len()
    );
    if significant.is_empty() {
        println!("  NO significant ranges — nothing to audit at this threshold");
        return;
    }
    let mut elong: Vec<f32> = significant.iter().map(|c| c.elongation()).collect();
    elong.sort_by(f32::total_cmp);
    let eq = |q: f32| elong[(((elong.len() - 1) as f32) * q) as usize];
    println!(
        "  elongation (significant ranges) p10/p50/p90 = {:.1}/{:.1}/{:.1}  [Earth belts 5-20x; <2 = blob]",
        eq(0.1),
        eq(0.5),
        eq(0.9)
    );

    // Per-range plateau geometry. A broad cap can disappear in a global
    // top-decile aggregate when a structured builder fragments one massif into
    // several components. Measure each significant range independently:
    // physical area within 0.5/1.0 km of its own summit, plus the fraction of
    // the 0.5-km cap whose steepest downhill edge is below a 1% grade. These
    // are terrain-field measurements; relief scale, lighting, and coloring do
    // not enter them.
    let r2 = EARTH_RADIUS_KM * EARTH_RADIUS_KM;
    let mut cap500_areas = Vec::with_capacity(significant.len());
    let mut cap1000_areas = Vec::with_capacity(significant.len());
    let mut cap500_flat_pct = Vec::with_capacity(significant.len());
    for comp in &significant {
        let peak = comp
            .cells
            .iter()
            .map(|&i| elev[i])
            .fold(f32::NEG_INFINITY, f32::max);
        let mut cap500_area = 0.0f64;
        let mut cap1000_area = 0.0f64;
        let mut flat500_area = 0.0f64;
        for &i in &comp.cells {
            let area_km2 = areas[i] as f64 * r2 as f64;
            if elev[i] >= peak - 0.10 {
                cap1000_area += area_km2;
            }
            if elev[i] < peak - 0.05 {
                continue;
            }
            cap500_area += area_km2;
            let ci = tess.cell_center(i);
            let mut max_downhill = 0.0f32;
            for &nb in tess.neighbors(i) {
                let d_km = (ci - tess.cell_center(nb)).length().max(1e-9) * EARTH_RADIUS_KM;
                max_downhill = max_downhill.max((elev[i] - elev[nb]).max(0.0) / d_km);
            }
            // Elevation unit = 10 km, so 1e-3 elevation/km = 1% grade.
            if max_downhill < 1.0e-3 {
                flat500_area += area_km2;
            }
        }
        cap500_areas.push(cap500_area as f32);
        cap1000_areas.push(cap1000_area as f32);
        cap500_flat_pct.push(if cap500_area > 0.0 {
            (100.0 * flat500_area / cap500_area) as f32
        } else {
            0.0
        });
    }
    for values in [&mut cap500_areas, &mut cap1000_areas, &mut cap500_flat_pct] {
        values.sort_by(f32::total_cmp);
    }
    let q = |values: &[f32], p: f32| values[(((values.len() - 1) as f32) * p) as usize];
    println!(
        "  per-range summit caps p50/p90: within 0.5km {:>7.0}/{:>7.0} km² | within 1.0km {:>7.0}/{:>7.0} km² | 0.5km cap below 1% grade {:>5.1}/{:>5.1}%",
        q(&cap500_areas, 0.5),
        q(&cap500_areas, 0.9),
        q(&cap1000_areas, 0.5),
        q(&cap1000_areas, 0.9),
        q(&cap500_flat_pct, 0.5),
        q(&cap500_flat_pct, 0.9),
    );

    println!(
        "  per-range:   area_km²  len×wid km   peak_m  crest: bins passes med_pass_m floor_m  offset"
    );
    println!(
        "               [Andes 7000x300, Himalaya 2400x1000, Alps 1200x200 | passes: real belts are BREACHED — Alps ~10 major; floor<<peak | offset: 0 = symmetric, ±1 = one-sided (O0)]"
    );
    for (k, comp) in significant.iter().take(top).enumerate() {
        let r = audit_range(tess, comp, elev);
        println!(
            "    {:>2}. {:>9.0}  {:>5.0}x{:>4.0}  {:>7.0}   {:>4} {:>6} {:>10.0} {:>7.0}  {:>+.2}",
            k + 1,
            r.area_km2,
            r.length_km,
            r.width_km,
            r.peak_m,
            r.crest_bins,
            r.passes,
            r.median_pass_depth_m,
            r.crest_floor_m,
            r.crest_offset
        );
    }

    // PLAUSIBILITY SELF-GATE (2026-07-10 lesson: 18-km peaks reached the user's
    // eyes because absolute bounds weren't self-enforced). Numbers Claude checks
    // BEFORE any visual hand-off; user-calibrated: ~10 km fine, 12 km borderline,
    // 14+ reads absurd. Earth max 8.8 km.
    {
        let max_peak_m = significant
            .iter()
            .map(|c| audit_range(tess, c, elev).peak_m)
            .fold(f32::NEG_INFINITY, f32::max);
        let verdict = if max_peak_m > 14_000.0 {
            "FLAG-ABSURD (do not send to visual)"
        } else if max_peak_m > 12_000.0 {
            "FLAG-borderline"
        } else {
            "ok"
        };
        println!(
            "  plausibility: max range peak {:.1} km [{}]  (Earth max 8.8 km; user calibration 2026-07-10: <=~12 ok, 14+ absurd)",
            max_peak_m / 1000.0,
            verdict
        );
    }

    // SPIRE / ISOLATION PROBE (2026-07-10 "pillar" verdict): peak HEIGHT alone
    // passes needles that are cliffs on all sides. Earth summits connect to
    // massifs — the HIGHEST terrain in a ring around a big peak sits close below
    // it (Everest→Lhotse −0.3 km at 3 km; even the most isolated volcanoes:
    // Kilimanjaro ring-10 max drop ~1.4 km, ring-25 ~2.5 with Mawenzi). The
    // drop from summit to ring-MAX measures how alone a summit stands; a needle
    // drops many km in EVERY direction (the gentlest exit is still a cliff).
    {
        let mut by_elev: Vec<usize> = (0..n).filter(|&i| mask[i]).collect();
        by_elev.sort_by(|&a, &b| elev[b].total_cmp(&elev[a]));
        let mut summits: Vec<usize> = Vec::new();
        for &i in &by_elev {
            if summits.len() >= 3 {
                break;
            }
            let pos = tess.cell_center(i);
            let min_sep = 50.0 / EARTH_RADIUS_KM;
            if summits
                .iter()
                .all(|&s| (tess.cell_center(s) - pos).length() > min_sep)
            {
                summits.push(i);
            }
        }
        // Two scales of aloneness. SUMMIT scale (needle): rings ≤50 km — Earth
        // connected 8-km peaks drop ≲1 km to ring-10 max, isolated volcanoes ≲2.
        // RANGE scale (island block / "pillar"): rings 50-250 km — Earth belts
        // continue along strike, so ring-250 max around ANY big summit stays
        // within ~2-3 km even for Kilimanjaro (Meru) / Denali (Alaska Range). A
        // 9-km summit whose 100-250 km surroundings are lowland is a freestanding
        // pillar no vertical-exaggeration excuse survives.
        println!("  spire probe (summit drop to HIGHEST terrain per ring; Earth: ring-10 ≲1-2, ring-25 ≲3, ring-100 ≲3, ring-250 ≲3 km):");
        let rings_km = [
            (2.0f32, 5.0f32),
            (5.0, 10.0),
            (10.0, 25.0),
            (25.0, 50.0),
            (50.0, 100.0),
            (100.0, 250.0),
        ];
        // ring-MAX asks "does ANY comparable terrain exist in the ring" (the
        // Himalaya scores ~0 out to 250 km — Kangchenjunga). The pillar signature
        // is the TYPICAL surrounding height: ring-p90 drop. Everest's 10-25 km
        // annulus is dense 5-7 km terrain (p90 drop ~2 km); Kilimanjaro — Earth's
        // most pillar-like big mountain — drops ~2 at ring-10 p90, ~4.4 at
        // ring-25. A summit whose ring-10 p90 drop exceeds ~5 km is a cliff on
        // (nearly) all sides at a scale Earth does not produce.
        let (mut worst10, mut worst25) = (0.0f32, 0.0f32);
        let mut worst_block = 0.0f32;
        let mut worst_render_wall_deg = 0.0f32;
        for &s in &summits {
            let sp = tess.cell_center(s);
            let mut ring_vals: [Vec<f32>; 6] = Default::default();
            for i in 0..n {
                let d_km = (tess.cell_center(i) - sp).length() * EARTH_RADIUS_KM;
                for (k, &(lo, hi)) in rings_km.iter().enumerate() {
                    if d_km >= lo && d_km < hi {
                        ring_vals[k].push(elev[i].max(0.0));
                    }
                }
            }
            let stat = |k: usize, q: f32| -> f32 {
                let v = &mut ring_vals[k].clone();
                if v.is_empty() {
                    return f32::NAN;
                }
                v.sort_by(f32::total_cmp);
                let x = v[(((v.len() - 1) as f32) * q) as usize];
                (elev[s] - x).max(0.0) * AUDIT_M_PER_UNIT / 1000.0
            };
            worst10 = worst10.max(stat(1, 0.9));
            worst25 = worst25.max(stat(2, 0.9));
            worst_block = worst_block.max(stat(5, 0.9));
            let block_p50 = stat(5, 0.5);
            let block_p10 = stat(5, 0.1);
            // What the current relief renderer does to the TYPICAL outer
            // flank. Elevation units are 10 km; compare displayed radial rise
            // with the midpoint radius of the 100..250 km annulus.
            let displayed_rise = (block_p50 / 10.0) * hex3::world::RELIEF_SCALE;
            let horizontal = 175.0 / EARTH_RADIUS_KM;
            let render_wall_deg = displayed_rise.atan2(horizontal).to_degrees();
            worst_render_wall_deg = worst_render_wall_deg.max(render_wall_deg);
            println!(
                "    summit {:>5.1} km: drop to ring-p90 (max)  5-10km {:>4.1} ({:>4.1}) | 10-25 {:>4.1} ({:>4.1}) | 25-50 {:>4.1} ({:>4.1}) | 50-100 {:>4.1} ({:>4.1}) | 100-250 {:>4.1} ({:>4.1}) km",
                elev[s] * AUDIT_M_PER_UNIT / 1000.0,
                stat(1, 0.9),
                stat(1, 1.0),
                stat(2, 0.9),
                stat(2, 1.0),
                stat(3, 0.9),
                stat(3, 1.0),
                stat(4, 0.9),
                stat(4, 1.0),
                stat(5, 0.9),
                stat(5, 1.0),
            );
            println!(
                "      visual surround: 100-250km drop p50/p10 {:>4.1}/{:>4.1} km | apparent median wall @ relief {:.2}: {:>4.0}°",
                block_p50,
                block_p10,
                hex3::world::RELIEF_SCALE,
                render_wall_deg,
            );
        }
        // Two failure shapes. NEEDLE (summit scale): big p90 drops at 10-25 km.
        // MESA/PILLAR (range scale, the 2026-07-10 verdict): flat top out to
        // ~100 km then a multi-km cliff to the typical 100-250 km surroundings —
        // an isolated orogen block with no foothill taper. Earth's ceiling at the
        // 100-250 ring is ~4 km (Everest over the Gangetic plain, WITH a
        // monotonic taper through the intermediate rings; our mesa is flat-then-
        // cliff). Baseline inherits this from the coarse macro envelope; meso/
        // pulse raise the top and texture it but do not create the cliff.
        let verdict = if worst10 > 5.0 || worst25 > 6.5 || worst_block > 6.0 {
            "FLAG-PILLAR-ABSURD (do not send to visual)"
        } else if worst10 > 3.5 || worst25 > 5.0 || worst_block > 4.5 {
            "FLAG-pillar"
        } else {
            "ok"
        };
        println!(
            "    spire gate [{verdict}]  (p90 drop: flag ring-10 >3.5 | ring-25 >5 | ring-250 >4.5 km; absurd >5 / >6.5 / >6. Earth: Everest ~2/2.3/~4-taper, Kilimanjaro ~2/4.4)"
        );
        let render_verdict = if worst_render_wall_deg > 60.0 {
            "FLAG-TOWER"
        } else if worst_render_wall_deg > 45.0 {
            "FLAG-steep"
        } else {
            "ok"
        };
        println!(
            "    render-tower gate [{render_verdict}]  (worst median outer-flank apparent angle {:.0}°; flag >45°, tower >60°)",
            worst_render_wall_deg
        );
    }

    // CREST-TRAIN: is the meso band a metronome, a quasi-periodic fold belt, or
    // drainage-organized? Ridge spacings along cross-strike transects (pooled over
    // the significant ranges) + flow-orientation split vs strike.
    {
        // SFD accumulation (cell counts) over the whole fine mesh: descending-
        // elevation order, each cell adds its count to its steepest receiver.
        let mut accum = vec![1.0f32; n];
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| elev[b].total_cmp(&elev[a]));
        for &i in &order {
            let mut best = (elev[i], i);
            for &nb in tess.neighbors(i) {
                if elev[nb] < best.0 {
                    best = (elev[nb], nb);
                }
            }
            if best.1 != i {
                accum[best.1] += accum[i];
            }
        }
        let mut spacings: Vec<f32> = Vec::new();
        let mut ridges: Vec<f32> = Vec::new();
        let (mut lon, mut tra, mut tot) = (0usize, 0usize, 0usize);
        let (mut tl, mut tt, mut ttot) = (0f64, 0f64, 0f64);
        for comp in significant.iter().take(top) {
            let ct = crest_train_stats(tess, comp, elev, &accum);
            spacings.extend(ct.spacings_km);
            ridges.extend(ct.ridges_per_transect);
            lon += ct.longitudinal_cells;
            tra += ct.transverse_cells;
            tot += ct.total_cells;
            tl += ct.trunk_longitudinal;
            tt += ct.trunk_transverse;
            ttot += ct.trunk_total;
        }
        if !spacings.is_empty() && tot > 0 {
            spacings.sort_by(f32::total_cmp);
            let sq = |q: f32| spacings[(((spacings.len() - 1) as f32) * q) as usize];
            let mean: f32 = spacings.iter().sum::<f32>() / spacings.len() as f32;
            let var: f32 = spacings
                .iter()
                .map(|s| (s - mean) * (s - mean))
                .sum::<f32>()
                / spacings.len() as f32;
            let cv = var.sqrt() / mean.max(1e-6);
            ridges.sort_by(f32::total_cmp);
            let r_med = ridges[ridges.len() / 2];
            println!(
                "  crest-train ({} spacings, 150 m prominence): spacing p25/p50/p75 = {:.0}/{:.0}/{:.0} km, CV {:.2}, ridges/transect med {:.0}  [CV: metronome ~0.1-0.2, quasi-periodic fold belt higher, drainage-organized no dominant spacing]",
                spacings.len(),
                sq(0.25),
                sq(0.5),
                sq(0.75),
                cv,
                r_med
            );
            println!(
                "  flow orientation vs strike (mountain cells): longitudinal {:.0}% / oblique {:.0}% / transverse {:.0}%  [fold-train terrain drains ALONG strike valleys; alpine dissection is transverse-dominant]",
                100.0 * lon as f32 / tot as f32,
                100.0 * (tot - lon - tra) as f32 / tot as f32,
                100.0 * tra as f32 / tot as f32
            );
            if ttot > 0.0 {
                println!(
                    "  trunk flow orientation (top-decile accumulation): longitudinal {:.0}% / oblique {:.0}% / transverse {:.0}%  [the grammar carrier: trellis = strike-trunks + cross gaps; alpine = transverse/oblique trunks]",
                    100.0 * tl / ttot,
                    100.0 * (ttot - tl - tt) / ttot,
                    100.0 * tt / ttot
                );
            }
        }
    }

    // RELIEF SPECTRUM: local relief (max-min elevation) at nested window sizes
    // over mountain cells, on the PRE-erosion base and the ERODED surface. This
    // is the wavelength-ownership instrument: which synthesis layer produces
    // relief at which scale, and what erosion adds/removes per band. Chord
    // distances (f32 acos collapses below ~3 km on the fine mesh). One BFS per
    // sample at the largest radius; inner windows read from the same expansion.
    const WINDOWS_KM: [f32; 5] = [5.0, 10.0, 25.0, 50.0, 100.0]; // diameters
    let pre_elev = &fine.surface_for(3).elevation.values;
    let mtn_cells: Vec<usize> = (0..n).filter(|&i| mask[i]).collect();
    let stride = (mtn_cells.len() / 4_000).max(1);
    let max_radius = WINDOWS_KM[WINDOWS_KM.len() - 1] * 0.5 / EARTH_RADIUS_KM;
    let radii: Vec<f32> = WINDOWS_KM
        .iter()
        .map(|w| w * 0.5 / EARTH_RADIUS_KM)
        .collect();
    // reliefs[metric][surface][window] -> samples. metric 0 = max-min (reference),
    // metric 1 = p95-p05 within the window (robust: one spike/pit can't inflate it
    // — the B-regime GATE metric per the relief-spectrum spec).
    let mut reliefs = vec![vec![vec![Vec::<f32>::new(); WINDOWS_KM.len()]; 2]; 2];
    let mut visited_mark = vec![u32::MAX; n];
    let mut window_cells: Vec<(f32, f32, f32)> = Vec::new(); // (dist, pre, eroded)
    for (si, &start) in mtn_cells.iter().step_by(stride).enumerate() {
        let mark = si as u32;
        let c0 = tess.cell_center(start);
        window_cells.clear();
        let mut queue = std::collections::VecDeque::new();
        visited_mark[start] = mark;
        queue.push_back((start, 0.0f32));
        while let Some((cell, dist)) = queue.pop_front() {
            window_cells.push((dist, pre_elev[cell], elev[cell]));
            for &nb in tess.neighbors(cell) {
                if visited_mark[nb] != mark {
                    let d = (tess.cell_center(nb) - c0).length();
                    if d <= max_radius {
                        visited_mark[nb] = mark;
                        queue.push_back((nb, d));
                    }
                }
            }
        }
        for (w, &r) in radii.iter().enumerate() {
            for s in 0..2 {
                let mut vals: Vec<f32> = window_cells
                    .iter()
                    .filter(|&&(d, _, _)| d <= r)
                    .map(|&(_, pre, ero)| if s == 0 { pre } else { ero })
                    .collect();
                if vals.is_empty() {
                    continue;
                }
                vals.sort_by(f32::total_cmp);
                let q = |f: f32| vals[(((vals.len() - 1) as f32) * f) as usize];
                reliefs[0][s][w].push((q(1.0) - q(0.0)) * AUDIT_M_PER_UNIT);
                reliefs[1][s][w].push((q(0.95) - q(0.05)) * AUDIT_M_PER_UNIT);
            }
        }
    }
    if !reliefs[0][1][0].is_empty() {
        println!(
            "  relief spectrum ({} samples; p50 m, [p90]):  [Earth alpine ballpark: 10 km ~1000-1800, 25 km ~1500-3000, 50 km ~2000-3500, then saturates. p95-p05 is the GATE metric; max-min is reference]",
            reliefs[0][1][0].len()
        );
        for (mlabel, m) in [("max-min", 0usize), ("p95-p05", 1usize)] {
            for (label, s) in [("pre", 0usize), ("eroded", 1usize)] {
                let mut line = format!("    {mlabel:<8} {label:<7}");
                for (w, wkm) in WINDOWS_KM.iter().enumerate() {
                    let v = &mut reliefs[m][s][w];
                    if v.is_empty() {
                        continue;
                    }
                    v.sort_by(f32::total_cmp);
                    let q = |f: f32| v[(((v.len() - 1) as f32) * f) as usize];
                    line.push_str(&format!(
                        "  {:>3.0}km {:>4.0} [{:>4.0}]",
                        wkm,
                        q(0.5),
                        q(0.9)
                    ));
                }
                println!("{line}");
            }
        }
    }
}

// ======================== RIVER AUDIT ========================
//
// The RENDERED river network as an object: same thresholds as the renderer
// (All = 0.00005·N count-equivalents; Major = 0.004/0.0006·N outlet/branch),
// so these numbers describe the rivers the user sees. Structure (Strahler/
// Horton), coverage (drainage density, mouth census), and the top trunks
// (length, sinuosity, long-profile shape). Earth refs inline.

fn run_biome_audit(world: &World, seed: u64) {
    use hex3::world::{BiomeKind, EcologySemantics};

    let tess = world.active_tessellation();
    let ecology = EcologySemantics::build(
        tess,
        &world.active_elevation().expect("elevation").values,
        world.active_temperature().expect("temperature"),
        world.active_precipitation().expect("precipitation"),
        world.active_hydrology(),
    );
    let areas = tess.cell_areas();
    let is_land = |kind: BiomeKind| !matches!(kind, BiomeKind::Ocean | BiomeKind::Lake);
    let land_area: f64 = ecology
        .cells
        .iter()
        .enumerate()
        .filter(|(_, cell)| is_land(cell.biome))
        .map(|(i, _)| areas[i] as f64)
        .sum();
    let mut sums = [0.0f64; 5];
    let mut transition_area = 0.0f64;
    for (i, cell) in ecology
        .cells
        .iter()
        .enumerate()
        .filter(|(_, cell)| is_land(cell.biome))
    {
        let area = areas[i] as f64;
        sums[0] += area * cell.potentials.heat as f64;
        sums[1] += area * cell.potentials.moisture as f64;
        sums[2] += area * cell.potentials.vegetation as f64;
        sums[3] += area * cell.potentials.tree as f64;
        sums[4] += area * cell.potentials.wetland as f64;
        if cell.classification_confidence < 0.20 {
            transition_area += area;
        }
    }

    println!("\n================ BIOME AUDIT seed={seed} ================");
    println!("(seasonless ecological potentials; calibrated proxies, not Köppen classes)");
    println!(
        "  land-mean raw P/demand normalization: {:.3}",
        ecology.land_mean_raw_aridity
    );
    println!(
        "  mean potentials heat/moisture/vegetation/tree/wetland: {:.2} / {:.2} / {:.2} / {:.2} / {:.2}",
        sums[0] / land_area.max(1e-30),
        sums[1] / land_area.max(1e-30),
        sums[2] / land_area.max(1e-30),
        sums[3] / land_area.max(1e-30),
        sums[4] / land_area.max(1e-30),
    );
    println!(
        "  transition area (confidence <0.20): {:.1}% of land",
        100.0 * transition_area / land_area.max(1e-30)
    );
    println!("\n  biome                 land%  regions  largest_km²  largest_extent_km");
    for kind in BiomeKind::LAND_KINDS {
        let mask: Vec<bool> = ecology
            .cells
            .iter()
            .map(|cell| cell.biome == kind)
            .collect();
        let area: f64 = mask
            .iter()
            .enumerate()
            .filter(|(_, included)| **included)
            .map(|(i, _)| areas[i] as f64)
            .sum();
        let components = measure_components(tess, &mask);
        let largest = components.first();
        println!(
            "  {:<20?} {:6.1}  {:7}  {:11.0}  {:17.0}",
            kind,
            (100.0 * area / land_area.max(1e-30)).max(0.0),
            components.len(),
            largest.map(|c| c.area_km2).unwrap_or(0.0),
            largest.map(|c| c.length_km).unwrap_or(0.0),
        );
    }
}

fn run_river_audit(world: &World, seed: u64, top: usize) {
    use hex3::world::{RiverNetwork, RiverThresholdPolicy, WaterBodySemantics};
    let Some(fine) = world.fine.as_ref() else {
        println!("\n[RIVER AUDIT] no fine surface present");
        return;
    };
    let tess = fine.tessellation();
    let surf = fine.surface_for(u32::MAX);
    let hydro = &surf.hydrology;
    let elev = &surf.elevation.values;
    let n = tess.num_cells();
    let areas = tess.cell_areas();
    let r_km = EARTH_RADIUS_KM;

    println!(
        "\n================ RIVER AUDIT seed={} (fine eroded) ================",
        seed
    );

    let land_km2: f64 = (0..n)
        .filter(|&i| !hydro.is_submerged(i) && elev[i] >= 0.0)
        .map(|i| (areas[i] * r_km * r_km) as f64)
        .sum();
    // Continent scale reference for river lengths (computed once).
    let land_mask: Vec<bool> = (0..n)
        .map(|i| !hydro.is_submerged(i) && elev[i] >= 0.0)
        .collect();
    let continents = measure_components(tess, &land_mask);
    let cont_len = continents.first().map(|c| c.length_km).unwrap_or(0.0);

    // A/B the two semantic policies. These are the same shared definitions the
    // renderer consumes; this audit no longer reconstructs its own network.
    let min_catchment = 2000.0f32;
    let modes = [
        (
            "LEGACY (count-equivalent)".to_string(),
            RiverThresholdPolicy::legacy(),
        ),
        (
            format!("CATCHMENT ({min_catchment:.0} km² min)"),
            RiverThresholdPolicy::catchment(min_catchment),
        ),
    ];
    river_scale_panel(tess, hydro, &areas, land_km2);
    let water = WaterBodySemantics::build(tess, hydro);
    for (label, policy) in modes {
        let network = RiverNetwork::build(tess, hydro, &water, policy);
        river_mode_panel(
            &label, tess, hydro, &network, elev, &areas, land_km2, cont_len, top,
        );
    }
}

#[derive(Clone, Copy)]
struct RiverScaleStats {
    cells: usize,
    heads: usize,
    junctions: usize,
    max_order: u8,
    cell_width_p50_km: f32,
    reach_length_p50_km: f32,
    length_km: f64,
}

/// Show which physical catchment scales the current mesh can actually express
/// as semantic river geometry. This is deliberately part of the existing river
/// audit rather than another experimental harness.
fn river_scale_panel(
    tess: &hex3::world::Tessellation,
    hydro: &hex3::world::Hydrology,
    areas: &[f32],
    land_km2: f64,
) {
    use hex3::world::{Hydrology, RiverThresholdPolicy};

    const REQUESTED_KM2: [f32; 6] = [4.0, 100.0, 500.0, 2_000.0, 10_000.0, 50_000.0];
    const PLANET_VIEW_WIDTH_PX: f32 = 1_920.0;

    let n = tess.num_cells();
    let mean_cell_km2 = 4.0 * std::f32::consts::PI * EARTH_RADIUS_KM.powi(2) / n.max(1) as f32;
    let mut land_widths: Vec<f32> = (0..n)
        .filter(|&cell| !hydro.is_submerged(cell))
        .map(|cell| areas[cell].sqrt() * EARTH_RADIUS_KM)
        .collect();
    land_widths.sort_by(f32::total_cmp);
    let quantile = |values: &[f32], fraction: f32| {
        values
            .get((((values.len().saturating_sub(1)) as f32 * fraction) as usize).min(values.len()))
            .copied()
            .unwrap_or(0.0)
    };

    println!("\n  [CATCHMENT-SCALE LADDER]");
    println!(
        "    mesh: {n} cells; global mean {:.0} km²; land cell width p10/p50/p90 {:.1}/{:.1}/{:.1} km",
        mean_cell_km2,
        quantile(&land_widths, 0.10),
        quantile(&land_widths, 0.50),
        quantile(&land_widths, 0.90),
    );
    println!(
        "    requested effective span_km px@1920   cells  heads  joins ord cell_km reach_km  Dd_km/km²  status"
    );

    let circumference_km = 2.0 * std::f32::consts::PI * EARTH_RADIUS_KM;
    let mut previous: Option<(f32, RiverScaleStats)> = None;
    for requested in REQUESTED_KM2 {
        let policy = RiverThresholdPolicy::catchment(requested);
        let effective = policy
            .effective_all_minimum_km2(n)
            .expect("catchment policy has physical scale");
        let same_as_previous = previous
            .as_ref()
            .is_some_and(|(prior, _)| prior.to_bits() == effective.to_bits());
        let stats = if let Some((_, stats)) =
            previous.filter(|(prior, _)| prior.to_bits() == effective.to_bits())
        {
            stats
        } else {
            let selection = RiverSelection::build(hydro, policy);
            river_scale_stats(tess, hydro, areas, &selection.all_cells)
        };
        previous = Some((effective, stats));

        let span_km = 2.0 * (effective / std::f32::consts::PI).sqrt();
        let pixels = span_km / circumference_km * PLANET_VIEW_WIDTH_PX;
        let status = if same_as_previous {
            "same floor"
        } else if effective > requested * 1.001 {
            "mesh floor"
        } else {
            "physical"
        };
        println!(
            "    {:>9.0} {:>9.0} {:>7.0} {:>7.2} {:>7} {:>6} {:>6} {:>3} {:>7.1} {:>8.1} {:>10.4}  {}",
            requested,
            effective,
            span_km,
            pixels,
            stats.cells,
            stats.heads,
            stats.junctions,
            stats.max_order,
            stats.cell_width_p50_km,
            stats.reach_length_p50_km,
            stats.length_km / land_km2.max(1e-30),
            status,
        );
    }
    println!(
        "    span_km is equal-area basin diameter; px@1920 is its equatorial span in a full-width planet map, not rendered river width"
    );

    println!("\n    [UNFLOORED PHYSICAL-SELECTION PROBE]");
    println!("    requested   cells  heads  joins ord cell_km reach_km  Dd_km/km² local4%");
    for requested in [4.0f32, 100.0, 500.0, 2_000.0] {
        let threshold = Hydrology::flow_for_catchment_km2(requested);
        let mask: Vec<bool> = (0..n)
            .map(|cell| hydro.flow_accumulation[cell] >= threshold && !hydro.is_submerged(cell))
            .collect();
        let stats = river_scale_stats(tess, hydro, areas, &mask);
        let resolved_length_km: f64 = (0..n)
            .filter(|&cell| {
                mask[cell] && 4.0 * areas[cell] * EARTH_RADIUS_KM * EARTH_RADIUS_KM <= requested
            })
            .map(|cell| (areas[cell].sqrt() * EARTH_RADIUS_KM) as f64)
            .sum();
        println!(
            "    {:>9.0} {:>7} {:>6} {:>6} {:>3} {:>7.1} {:>8.1} {:>10.4} {:>6.1}%",
            requested,
            stats.cells,
            stats.heads,
            stats.junctions,
            stats.max_order,
            stats.cell_width_p50_km,
            stats.reach_length_p50_km,
            stats.length_km / land_km2.max(1e-30),
            (100.0 * resolved_length_km / stats.length_km.max(f64::EPSILON)).max(0.0),
        );
    }
    println!(
        "    local4% is network length on cells for which the requested basin spans at least four local cell areas"
    );
}

fn river_scale_stats(
    tess: &hex3::world::Tessellation,
    hydro: &hex3::world::Hydrology,
    areas: &[f32],
    mask: &[bool],
) -> RiverScaleStats {
    let n = tess.num_cells();
    let mut channel_cells: Vec<usize> = (0..n).filter(|&cell| mask[cell]).collect();
    let mut upstream_count = vec![0u8; n];
    for &cell in &channel_cells {
        if let Some(downstream) = hydro.downstream(cell).filter(|&next| mask[next]) {
            upstream_count[downstream] = upstream_count[downstream].saturating_add(1);
        }
    }

    channel_cells.sort_by(|&a, &b| {
        hydro.flow_accumulation[a]
            .total_cmp(&hydro.flow_accumulation[b])
            .then_with(|| a.cmp(&b))
    });
    let mut order = vec![0u8; n];
    let mut best_upstream = vec![0u8; n];
    let mut best_ties = vec![0u8; n];
    for &cell in &channel_cells {
        order[cell] = match best_upstream[cell] {
            0 => 1,
            best if best_ties[cell] >= 2 => best.saturating_add(1),
            best => best,
        };
        if let Some(downstream) = hydro.downstream(cell).filter(|&next| mask[next]) {
            if order[cell] > best_upstream[downstream] {
                best_upstream[downstream] = order[cell];
                best_ties[downstream] = 1;
            } else if order[cell] == best_upstream[downstream] {
                best_ties[downstream] = best_ties[downstream].saturating_add(1);
            }
        }
    }

    let mut reach_lengths = Vec::new();
    for &start in &channel_cells {
        if upstream_count[start] == 1 {
            continue;
        }
        let mut current = start;
        let mut length = 0.0f32;
        while let Some(next) = hydro.downstream(current).filter(|&cell| mask[cell]) {
            length +=
                (tess.cell_center(current) - tess.cell_center(next)).length() * EARTH_RADIUS_KM;
            current = next;
            if upstream_count[current] != 1 {
                break;
            }
        }
        if length > 0.0 {
            reach_lengths.push(length);
        }
    }
    reach_lengths.sort_by(f32::total_cmp);
    let mut cell_widths: Vec<f32> = channel_cells
        .iter()
        .map(|&cell| areas[cell].sqrt() * EARTH_RADIUS_KM)
        .collect();
    cell_widths.sort_by(f32::total_cmp);
    let median = |values: &[f32]| values.get(values.len() / 2).copied().unwrap_or(0.0);

    RiverScaleStats {
        cells: channel_cells.len(),
        heads: channel_cells
            .iter()
            .filter(|&&cell| upstream_count[cell] == 0)
            .count(),
        junctions: channel_cells
            .iter()
            .filter(|&&cell| upstream_count[cell] >= 2)
            .count(),
        max_order: channel_cells
            .iter()
            .map(|&cell| order[cell])
            .max()
            .unwrap_or(0),
        cell_width_p50_km: median(&cell_widths),
        reach_length_p50_km: median(&reach_lengths),
        length_km: channel_cells
            .iter()
            .map(|&cell| (areas[cell].sqrt() * EARTH_RADIUS_KM) as f64)
            .sum(),
    }
}

// ===================== RESOLUTION-PILOT AUDIT =====================

#[derive(Clone, Copy, Default)]
struct NativePilotBin {
    land_area: f64,
    slope_mass: f64,
    neighbor_relief_mass: f64,
    min_elevation: f32,
    max_elevation: f32,
    channel_heads: f64,
    channel_junctions: f64,
}

impl NativePilotBin {
    fn empty() -> Self {
        Self {
            min_elevation: f32::INFINITY,
            max_elevation: f32::NEG_INFINITY,
            ..Self::default()
        }
    }
}

/// Ask whether a short solved surface adds spatial information beyond the
/// existing coarse slope/flow/activity density prior. This deliberately stops
/// before remeshing: a response field that cannot rank where native refinement
/// pays does not justify transfer and mesh-lifecycle machinery.
fn run_resolution_pilot_audit(
    world: &mut World,
    seed: u64,
    pilot_max: usize,
    native_max: usize,
    pilot_steps: usize,
) {
    let audit_start = std::time::Instant::now();
    eprintln!(
        "resolution pilot: generating capped pilot (cap={pilot_max}, response_steps={pilot_steps})"
    );
    world.generate_fine_pre_with_cap(pilot_max);

    let short_start = std::time::Instant::now();
    let mut short_params = world.erosion_params;
    short_params.steps = pilot_steps;
    short_params.precip_outer_iters = 1;
    short_params.drainage_pulse = 0.0;
    short_params.glacial_k = 0.0;
    let short = {
        let fine = world.fine.as_ref().expect("pilot fine world");
        FineSurface::generate(seed, &fine.base, &fine.pre.hydrology, short_params)
    };
    let short_seconds = short_start.elapsed().as_secs_f64();

    let (signed_response, abs_response, response_variation, routing_change, specific_area_slope) = {
        let fine = world.fine.as_ref().expect("pilot fine world");
        let tess = fine.tessellation();
        let n = tess.num_cells();
        let signed: Vec<f32> = (0..n)
            .map(|i| {
                // Keep drainage-integration repair out of the pilot signal: the
                // response is the terrain supplied to hydrology, not its sparse
                // outlet-path correction.
                short.hydrology.pre_integration_elevation(i) - fine.base.base_elevation[i]
            })
            .collect();
        let absolute: Vec<f32> = signed.iter().map(|value| value.abs()).collect();
        let variation: Vec<f32> = (0..n)
            .map(|i| {
                tess.neighbors(i)
                    .iter()
                    .map(|&neighbor| (signed[i] - signed[neighbor]).abs())
                    .fold(0.0f32, f32::max)
            })
            .collect();
        let routing: Vec<f32> = (0..n)
            .map(|i| {
                (fine.pre.hydrology.downstream(i) != short.hydrology.downstream(i)) as u8 as f32
            })
            .collect();
        // A compact, physically motivated channel baseline: specific catchment
        // area supplies water while local grade supplies stream power. The
        // exponent is a diagnostic ranking choice, not a calibrated erosion law.
        let area_slope: Vec<f32> = (0..n)
            .map(|i| {
                if short.elevation.values[i] < 0.0 {
                    return f32::NEG_INFINITY;
                }
                let catchment_km2 = short.hydrology.flow_accumulation[i].max(0.0)
                    * EARTH_RADIUS_KM
                    * EARTH_RADIUS_KM;
                let contour_width_km = tess.cell_areas_ref()[i].max(1e-20).sqrt() * EARTH_RADIUS_KM;
                let specific_area_km = catchment_km2 / contour_width_km.max(1e-6);
                let slope = short.elevation.physical_grade(tess, i);
                specific_area_km.max(1e-6).ln() + 2.0 * slope.max(1e-6).ln()
            })
            .collect();
        (signed, absolute, variation, routing, area_slope)
    };
    drop(short);

    let full_pilot_start = std::time::Instant::now();
    world.generate_fine_eroded();
    let full_pilot_seconds = full_pilot_start.elapsed().as_secs_f64();

    // Retain only compact predictors/reference summaries. `world.fine.take()` is
    // essential: generate_fine_pre_with_cap constructs its replacement before
    // assignment, which would otherwise overlap both heavy FineWorlds in RAM.
    let pilot = world.fine.take().expect("completed pilot fine world");
    let pilot_tess = pilot.tessellation();
    let pilot_surface = pilot.eroded.as_ref().expect("full pilot erosion");
    let pilot_n = pilot_tess.num_cells();
    let centers: Vec<[f32; 3]> = (0..pilot_n)
        .map(|i| pilot_tess.cell_center(i).to_array())
        .collect();
    let pilot_areas = pilot_tess.cell_areas();
    let pilot_land: Vec<bool> = pilot_surface
        .elevation
        .values
        .iter()
        .map(|&value| value >= 0.0)
        .collect();
    let density = pilot.density().to_vec();
    let low_slope_energy: Vec<f64> = (0..pilot_n)
        .map(|i| {
            let slope = pilot_surface.elevation.physical_grade(pilot_tess, i) as f64;
            slope * slope
        })
        .collect();
    let low_neighbor_relief_energy: Vec<f64> = (0..pilot_n)
        .map(|i| {
            let elevation = pilot_surface.elevation.values[i];
            let delta = pilot_tess
                .neighbors(i)
                .iter()
                .map(|&neighbor| (elevation - pilot_surface.elevation.values[neighbor]).abs())
                .fold(0.0f32, f32::max) as f64;
            delta * delta
        })
        .collect();
    let normalized_abs = robust_unit_interval(&abs_response, &pilot_land);
    let normalized_variation = robust_unit_interval(&response_variation, &pilot_land);
    let combined: Vec<f32> = (0..pilot_n)
        .map(|i| {
            (normalized_abs[i] + normalized_variation[i] + routing_change[i].clamp(0.0, 1.0)) / 3.0
        })
        .collect();
    let pilot_actual = pilot_n;
    drop(pilot);
    drop(signed_response);

    let mut tree = KdTree::<f32, 3>::with_capacity(centers.len());
    for (index, center) in centers.iter().enumerate() {
        tree.add(center, index as u64);
    }

    eprintln!(
        "resolution pilot: pilot heavy state dropped; generating native reference (cap={native_max})"
    );
    let native_start = std::time::Instant::now();
    world.generate_fine_pre_with_cap(native_max);
    world.generate_fine_eroded();
    let native_seconds = native_start.elapsed().as_secs_f64();

    let native = world.fine.as_ref().expect("native fine world");
    let native_tess = native.tessellation();
    let native_surface = native.eroded.as_ref().expect("native erosion");
    let native_elevation = &native_surface.elevation;
    let native_hydrology = &native_surface.hydrology;
    let native_areas = native_tess.cell_areas();
    let native_rivers = RiverSelection::build(native_hydrology, RiverThresholdPolicy::default());
    let mut bins = vec![NativePilotBin::empty(); pilot_n];
    let mut native_land_area = 0.0f64;

    for i in 0..native_tess.num_cells() {
        if native_elevation.values[i] < 0.0 {
            continue;
        }
        let area = native_areas[i] as f64;
        native_land_area += area;
        let center = native_tess.cell_center(i).to_array();
        let owner = tree.nearest_one::<SquaredEuclidean>(&center).item as usize;
        if !pilot_land[owner] {
            continue;
        }
        let slope = native_elevation.physical_grade(native_tess, i) as f64;
        let neighbor_delta = native_tess
            .neighbors(i)
            .iter()
            .map(|&neighbor| (native_elevation.values[i] - native_elevation.values[neighbor]).abs())
            .fold(0.0f32, f32::max) as f64;
        let bin = &mut bins[owner];
        bin.land_area += area;
        bin.slope_mass += area * slope * slope;
        bin.neighbor_relief_mass += area * neighbor_delta * neighbor_delta;
        bin.min_elevation = bin.min_elevation.min(native_elevation.values[i]);
        bin.max_elevation = bin.max_elevation.max(native_elevation.values[i]);

        if native_rivers.all_cells[i] {
            let upstream = native_tess
                .neighbors(i)
                .iter()
                .filter(|&&neighbor| {
                    native_rivers.all_cells[neighbor]
                        && native_hydrology.downstream(neighbor) == Some(i)
                })
                .count();
            if upstream == 0 {
                bin.channel_heads += 1.0;
            } else if upstream >= 2 {
                bin.channel_junctions += 1.0;
            }
        }
    }

    let mut native_slope_mass = vec![0.0f64; pilot_n];
    let mut slope_gain_mass = vec![0.0f64; pilot_n];
    let mut native_neighbor_mass = vec![0.0f64; pilot_n];
    let mut neighbor_gain_mass = vec![0.0f64; pilot_n];
    let mut within_owner_relief_mass = vec![0.0f64; pilot_n];
    let mut channel_head_mass = vec![0.0f64; pilot_n];
    let mut channel_junction_mass = vec![0.0f64; pilot_n];
    for i in 0..pilot_n {
        let bin = bins[i];
        if bin.land_area <= 0.0 {
            continue;
        }
        native_slope_mass[i] = bin.slope_mass;
        native_neighbor_mass[i] = bin.neighbor_relief_mass;
        let native_slope_mean = bin.slope_mass / bin.land_area;
        let native_neighbor_mean = bin.neighbor_relief_mass / bin.land_area;
        slope_gain_mass[i] =
            (native_slope_mean - low_slope_energy[i]).max(0.0) * pilot_areas[i] as f64;
        neighbor_gain_mass[i] =
            (native_neighbor_mean - low_neighbor_relief_energy[i]).max(0.0) * pilot_areas[i] as f64;
        if bin.min_elevation.is_finite() && bin.max_elevation.is_finite() {
            let range = (bin.max_elevation - bin.min_elevation) as f64;
            within_owner_relief_mass[i] = bin.land_area * range * range;
        }
        channel_head_mass[i] = bin.channel_heads;
        channel_junction_mass[i] = bin.channel_junctions;
    }

    let channel_event_mass: Vec<f64> = channel_head_mass
        .iter()
        .zip(&channel_junction_mass)
        .map(|(&heads, &junctions)| heads + junctions)
        .collect();

    let predictors: [(&str, &[f32]); 6] = [
        ("density-control", &density),
        ("abs-response", &abs_response),
        ("response-variation", &response_variation),
        ("routing-change", &routing_change),
        ("combined-response", &combined),
        ("specific-area-slope", &specific_area_slope),
    ];
    let targets: [(&str, &[f64]); 8] = [
        ("native-slope-energy", &native_slope_mass),
        ("fine-only-slope-gain", &slope_gain_mass),
        ("native-neighbor-relief", &native_neighbor_mass),
        ("fine-only-neighbor-gain", &neighbor_gain_mass),
        ("within-pilot-cell-relief", &within_owner_relief_mass),
        ("selected-channel-heads", &channel_head_mass),
        ("selected-channel-junctions", &channel_junction_mass),
        ("selected-channel-heads+junctions", &channel_event_mass),
    ];

    println!("\n================ RESOLUTION PILOT seed={seed} ================");
    println!(
        "pilot cap/actual {pilot_max}/{pilot_actual}; native cap/actual {native_max}/{}; pilot steps {pilot_steps}/{}",
        native_tess.num_cells(),
        world.erosion_params.steps
    );
    println!(
        "timing: short response {:.2}s, full pilot {:.2}s, native {:.2}s, total {:.2}s",
        short_seconds,
        full_pilot_seconds,
        native_seconds,
        audit_start.elapsed().as_secs_f64()
    );
    let mapped_land_area: f64 = bins.iter().map(|bin| bin.land_area).sum();
    println!(
        "ownership coverage: {:.2}% of native land area maps to pilot-final land",
        100.0 * mapped_land_area / native_land_area.max(f64::EPSILON)
    );
    println!(
        "signal: integration cuts excluded; conditional columns rank within ten equal-land-area density bands"
    );
    println!(
        "entries are captured target mass / lift over equal-area selection; 10% and 20% land-area budgets"
    );
    for (target_name, target_mass) in targets {
        let total: f64 = target_mass.iter().sum();
        println!("\n-- {target_name} (mass={total:.6e}) --");
        println!(
            "  predictor                 global-10       global-20       conditional-10  conditional-20"
        );
        for (predictor_name, predictor) in predictors {
            let g10 = area_budget_capture(predictor, target_mass, &pilot_areas, &pilot_land, 0.10);
            let g20 = area_budget_capture(predictor, target_mass, &pilot_areas, &pilot_land, 0.20);
            let c10 = density_stratified_capture(
                predictor,
                &density,
                target_mass,
                &pilot_areas,
                &pilot_land,
                0.10,
            );
            let c20 = density_stratified_capture(
                predictor,
                &density,
                target_mass,
                &pilot_areas,
                &pilot_land,
                0.20,
            );
            println!(
                "  {predictor_name:<25} {:>6.2}%/{:>4.2}x  {:>6.2}%/{:>4.2}x  {:>6.2}%/{:>4.2}x  {:>6.2}%/{:>4.2}x",
                100.0 * g10,
                g10 / 0.10,
                100.0 * g20,
                g20 / 0.20,
                100.0 * c10,
                c10 / 0.10,
                100.0 * c20,
                c20 / 0.20,
            );
        }
    }
    println!(
        "\nInterpretation: judge relief and channel allocators separately; each must add conditional lift over the density bands on its own target. The native reference remains density-prior sampled, so failure is stronger evidence than success."
    );
}

fn robust_unit_interval(values: &[f32], mask: &[bool]) -> Vec<f32> {
    let mut finite: Vec<f32> = values
        .iter()
        .zip(mask)
        .filter_map(|(&value, &included)| (included && value.is_finite()).then_some(value))
        .collect();
    finite.sort_by(|a, b| a.total_cmp(b));
    if finite.is_empty() {
        return vec![0.0; values.len()];
    }
    let at = |q: f32| {
        let index = ((finite.len() - 1) as f32 * q).round() as usize;
        finite[index]
    };
    let lo = at(0.05);
    let mut hi = at(0.95);
    if hi <= lo {
        hi = *finite.last().unwrap();
    }
    let span = hi - lo;
    values
        .iter()
        .map(|&value| {
            if value.is_finite() && span > f32::EPSILON {
                ((value - lo) / span).clamp(0.0, 1.0)
            } else {
                0.0
            }
        })
        .collect()
}

fn area_budget_capture(
    predictor: &[f32],
    target_mass: &[f64],
    areas: &[f32],
    land: &[bool],
    fraction: f64,
) -> f64 {
    let indices: Vec<usize> = (0..predictor.len()).filter(|&i| land[i]).collect();
    capture_over_indices(&indices, predictor, target_mass, areas, fraction)
}

fn density_stratified_capture(
    predictor: &[f32],
    density: &[f32],
    target_mass: &[f64],
    areas: &[f32],
    land: &[bool],
    fraction: f64,
) -> f64 {
    let mut by_density: Vec<usize> = (0..density.len()).filter(|&i| land[i]).collect();
    by_density.sort_by(|&a, &b| density[a].total_cmp(&density[b]));
    let total_area: f64 = by_density.iter().map(|&i| areas[i] as f64).sum();
    let total_target: f64 = target_mass.iter().sum();
    if total_area <= 0.0 || total_target <= 0.0 {
        return 0.0;
    }
    let mut strata: [Vec<usize>; 10] = std::array::from_fn(|_| Vec::new());
    let mut cumulative_area = 0.0;
    for index in by_density {
        let midpoint = cumulative_area + 0.5 * areas[index] as f64;
        let stratum = ((10.0 * midpoint / total_area).floor() as usize).min(9);
        strata[stratum].push(index);
        cumulative_area += areas[index] as f64;
    }
    let captured: f64 = strata
        .iter()
        .map(|indices| capture_mass_over_indices(indices, predictor, target_mass, areas, fraction))
        .sum();
    captured / total_target
}

fn capture_over_indices(
    indices: &[usize],
    predictor: &[f32],
    target_mass: &[f64],
    areas: &[f32],
    fraction: f64,
) -> f64 {
    let total_target: f64 = target_mass.iter().sum();
    if total_target <= 0.0 {
        return 0.0;
    }
    capture_mass_over_indices(indices, predictor, target_mass, areas, fraction) / total_target
}

fn capture_mass_over_indices(
    indices: &[usize],
    predictor: &[f32],
    target_mass: &[f64],
    areas: &[f32],
    fraction: f64,
) -> f64 {
    let mut ranked = indices.to_vec();
    ranked.sort_by(|&a, &b| {
        predictor[b]
            .total_cmp(&predictor[a])
            .then_with(|| a.cmp(&b))
    });
    let total_area: f64 = ranked.iter().map(|&i| areas[i] as f64).sum();
    let mut remaining = total_area * fraction.clamp(0.0, 1.0);
    let mut captured = 0.0;
    for index in ranked {
        if remaining <= 0.0 {
            break;
        }
        let area = areas[index] as f64;
        if area <= 0.0 {
            continue;
        }
        let used = remaining.min(area);
        captured += target_mass[index] * used / area;
        remaining -= used;
    }
    captured
}

#[cfg(test)]
mod resolution_pilot_tests {
    use super::{area_budget_capture, density_stratified_capture, robust_unit_interval};

    #[test]
    fn robust_normalization_clips_outlier_without_nan() {
        let normalized = robust_unit_interval(&[0.0, 1.0, 2.0, 3.0, 1000.0], &[true; 5]);
        assert_eq!(normalized[0], 0.0);
        assert_eq!(normalized[4], 1.0);
        assert!(normalized.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn area_budget_capture_respects_fractional_boundary_cell() {
        let capture =
            area_budget_capture(&[2.0, 1.0], &[8.0, 2.0], &[1.0, 1.0], &[true, true], 0.25);
        assert!((capture - 0.4).abs() < 1e-12);
    }

    #[test]
    fn conditional_capture_can_find_signal_inside_density_bands() {
        let predictor: Vec<f32> = (0..20).map(|i| (i % 2 == 0) as u8 as f32).collect();
        let density: Vec<f32> = (0..20).map(|i| (i / 2) as f32).collect();
        let target: Vec<f64> = predictor.iter().map(|&value| value as f64).collect();
        let capture =
            density_stratified_capture(&predictor, &density, &target, &[1.0; 20], &[true; 20], 0.5);
        assert!((capture - 1.0).abs() < 1e-12);
    }
}

/// One river-network panel (density, Strahler/Horton, mouths, top trunks) for
/// a given 'All' flow threshold + Major outlet/branch count-equivalents.
#[allow(clippy::too_many_arguments)]
fn river_mode_panel(
    label: &str,
    tess: &hex3::world::Tessellation,
    hydro: &hex3::world::Hydrology,
    network: &hex3::world::RiverNetwork,
    elev: &[f32],
    areas: &[f32],
    land_km2: f64,
    cont_len: f32,
    top: usize,
) {
    let n = tess.num_cells();
    let r_km = EARTH_RADIUS_KM;
    let is_channel = &network.all_cells;
    let is_major = &network.major_cells;
    let net_len = |m: &dyn Fn(usize) -> bool| -> f64 {
        (0..n)
            .filter(|&i| m(i))
            .map(|i| (areas[i].sqrt() * r_km) as f64)
            .sum()
    };
    let all_len = net_len(&|i| is_channel[i]);
    let major_len = net_len(&|i| is_major[i]);
    println!("\n  [{label}]");
    println!(
        "    network ('All'):   {:>7} cells, ~{:>7.0} km total, Dd {:.4} km/km²   [Earth perennial-river Dd at map scale ~0.01-0.1]",
        is_channel.iter().filter(|&&c| c).count(),
        all_len,
        all_len / land_km2.max(1e-30)
    );
    println!(
        "    network ('Major'): {:>7} cells, ~{:>7.0} km total, Dd {:.4} km/km²",
        is_major.iter().filter(|&&c| c).count(),
        major_len,
        major_len / land_km2.max(1e-30)
    );

    let ups = &network.upstream;
    let order = &network.strahler_order;
    let chan: Vec<usize> = (0..n).filter(|&cell| is_channel[cell]).collect();

    // Horton counts derive from the shared Strahler field.
    let max_order = chan.iter().map(|&cell| order[cell]).max().unwrap_or(0);
    let mut streams = vec![0u32; max_order as usize + 1];
    for &c in &chan {
        let o = order[c];
        // A stream of order o starts where no upstream child has order o.
        if !ups[c].iter().any(|&u| order[u] == o) {
            streams[o as usize] += 1;
        }
    }
    let mut rbs: Vec<f64> = Vec::new();
    for o in 1..max_order as usize {
        if streams[o + 1] > 0 {
            rbs.push(streams[o] as f64 / streams[o + 1] as f64);
        }
    }
    let rb = if rbs.is_empty() {
        0.0
    } else {
        (rbs.iter().map(|r| r.ln()).sum::<f64>() / rbs.len() as f64).exp()
    };
    let stream_counts: Vec<String> = (1..=max_order as usize)
        .map(|o| format!("N{}={}", o, streams[o]))
        .collect();
    println!(
        "    Strahler: max order {} | {} | Horton Rb = {:.1}   [Earth Rb 3-5]",
        max_order,
        stream_counts.join(" "),
        rb
    );

    // Mouth census: where do rivers end?
    let (mut ocean_m, mut lake_m, mut inland_m) = (0u32, 0u32, 0u32);
    let mut mouths = network.mouths.clone();
    for &c in &mouths {
        match hydro.downstream(c) {
            Some(d) if hydro.is_submerged(d) => {
                if hydro.is_ocean(d) {
                    ocean_m += 1;
                } else {
                    lake_m += 1;
                }
            }
            None => {
                inland_m += 1;
            }
            _ => {}
        }
    }
    println!(
        "    mouths: {} to ocean, {} to lakes, {} inland dead-ends   [post-integration a major river should NOT dead-end inland]",
        ocean_m, lake_m, inland_m
    );

    // Top trunks by mouth discharge: walk upstream along the max-flow child.
    mouths.sort_by(|&a, &b| hydro.flow_accumulation[b].total_cmp(&hydro.flow_accumulation[a]));
    println!(
        "    top rivers:    len_km  straight  sinuos  basin_km²  src_m  %drop-upper-half  mouth"
    );
    println!(
        "                   [largest continent extent {:.0} km; Earth: longest ~0.5-0.9x continent; graded profile drops 60-85% in the upper half]",
        cont_len
    );
    for (k, &mouth) in mouths.iter().take(top).enumerate() {
        // Trace the trunk upstream.
        let mut path = vec![mouth];
        let mut cur = mouth;
        loop {
            let next = ups[cur]
                .iter()
                .copied()
                .max_by(|&a, &b| hydro.flow_accumulation[a].total_cmp(&hydro.flow_accumulation[b]));
            match next {
                Some(u) => {
                    cur = u;
                    path.push(cur);
                }
                None => break,
            }
        }
        // Chord-sum length (arc ≈ chord at cell scale; f32 acos is unusable here).
        let mut len_km = 0.0f64;
        for w in path.windows(2) {
            len_km += ((tess.cell_center(w[0]) - tess.cell_center(w[1])).length() * r_km) as f64;
        }
        let straight_km =
            (tess.cell_center(mouth) - tess.cell_center(*path.last().unwrap())).length() * r_km;
        let sinuosity = len_km / (straight_km as f64).max(1e-9);
        // Basin area: BFS the full upstream tree (all cells, not just channels).
        let mut basin_km2 = 0.0f64;
        {
            let mut stack = vec![mouth as u32];
            let mut seen = std::collections::HashSet::new();
            seen.insert(mouth as u32);
            while let Some(c) = stack.pop() {
                let c = c as usize;
                basin_km2 += (areas[c] * r_km * r_km) as f64;
                for &nb in tess.neighbors(c) {
                    if hydro.downstream(nb) == Some(c) && seen.insert(nb as u32) {
                        stack.push(nb as u32);
                    }
                }
            }
        }
        // Long-profile shape: % of total drop achieved in the upper half of length.
        let src = *path.last().unwrap();
        let drop = elev[src] - elev[mouth];
        let mut acc = 0.0f64;
        let mut mid_elev = elev[mouth];
        for w in path.windows(2) {
            acc += ((tess.cell_center(w[0]) - tess.cell_center(w[1])).length() * r_km) as f64;
            if acc >= len_km * 0.5 {
                mid_elev = elev[w[1]];
                break;
            }
        }
        // Meaningless on stub trunks (<2 cells) or near-zero total drop (<50 m).
        let upper_frac = if path.len() < 3 || drop < 0.005 {
            f32::NAN
        } else {
            100.0 * (elev[src] - mid_elev) / drop
        };
        let mouth_kind = match hydro.downstream(mouth) {
            Some(d) if hydro.is_ocean(d) => "ocean",
            Some(_) => "lake",
            None => "INLAND",
        };
        let upper_str = if upper_frac.is_nan() {
            "     -".to_string()
        } else {
            format!("{upper_frac:>5.0}%")
        };
        println!(
            "      {:>2}. {:>8.0}  {:>8.0}  {:>6.2}  {:>9.0}  {:>5.0}  {:>16}  {}",
            k + 1,
            len_km,
            straight_km,
            sinuosity,
            basin_km2,
            elev[src] * AUDIT_M_PER_UNIT,
            upper_str,
            mouth_kind
        );
    }
}
