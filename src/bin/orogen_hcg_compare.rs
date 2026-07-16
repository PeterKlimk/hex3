//! Run the disposable 4 km H/C/G comparison and emit one JSON plus one PNG.

use clap::Parser;
use hex3::world::landscape::build_linked_shared_input_bundle_v0;
use hex3::world::landscape::organization_comparison::build_thin_hcg_common_evidence_v0;
use hex3::world::landscape::organization_output::{
    build_thin_hcg_numerical_output_v0, ThinHcgNumericalOutputV0,
};
use hex3::world::landscape::organization_owner::run_thin_g_4km_v0;
use hex3::world::landscape::organization_owner_c::run_thin_c_4km_v0;
use hex3::world::landscape::organization_owner_h::run_thin_h_4km_v0;
use hex3::world::landscape::organization_render::{
    write_thin_hcg_matched_png_v0, ThinHcgRenderConfigV0, ThinHcgRenderMetadataV0,
};
use serde::Serialize;
use std::fs::{self, File};
use std::io::BufWriter;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Debug, Parser)]
#[command(about = "Run the disposable CPU-only 4 km H/C/G comparison")]
struct Cli {
    /// A new directory which will receive comparison.json and matched.png.
    #[arg(long)]
    output_dir: PathBuf,
}

#[derive(Debug, Serialize)]
struct TimingsV0 {
    linked_input_seconds: f64,
    h_seconds: f64,
    c_seconds: f64,
    g_seconds: f64,
    common_evidence_seconds: f64,
    numerical_summary_seconds: f64,
    render_seconds: f64,
    total_seconds: f64,
}

#[derive(Debug, Serialize)]
struct ComparisonFileV0 {
    schema_version: &'static str,
    warning: &'static str,
    source_revision: &'static str,
    source_dirty: bool,
    timings: TimingsV0,
    render: ThinHcgRenderMetadataV0,
    numerical: ThinHcgNumericalOutputV0,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    fs::create_dir(&cli.output_dir)?;
    let total_started = Instant::now();

    let started = Instant::now();
    let bundle = build_linked_shared_input_bundle_v0()?;
    let linked_input_seconds = started.elapsed().as_secs_f64();

    let started = Instant::now();
    let h = run_thin_h_4km_v0(&bundle)?;
    let h_seconds = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let c = run_thin_c_4km_v0(&bundle)?;
    let c_seconds = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let g = run_thin_g_4km_v0(&bundle)?;
    let g_seconds = started.elapsed().as_secs_f64();

    let started = Instant::now();
    let evidence = build_thin_hcg_common_evidence_v0(&bundle, &h, &c, &g)?;
    let common_evidence_seconds = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let numerical = build_thin_hcg_numerical_output_v0(&bundle, &h, &c, &g, &evidence)?;
    let numerical_summary_seconds = started.elapsed().as_secs_f64();

    let input = bundle
        .resolutions
        .iter()
        .find(|value| value.nominal_spacing_km.to_bits() == 4.0_f64.to_bits())
        .ok_or("accepted bundle has no exact 4 km input")?;
    let started = Instant::now();
    let render = write_thin_hcg_matched_png_v0(
        cli.output_dir.join("matched.png"),
        input,
        &h.final_elevation_km,
        &c.final_elevation_km,
        &g.final_elevation_km,
        ThinHcgRenderConfigV0::default(),
    )?;
    let render_seconds = started.elapsed().as_secs_f64();
    let timings = TimingsV0 {
        linked_input_seconds,
        h_seconds,
        c_seconds,
        g_seconds,
        common_evidence_seconds,
        numerical_summary_seconds,
        render_seconds,
        total_seconds: total_started.elapsed().as_secs_f64(),
    };
    let output = ComparisonFileV0 {
        schema_version: "orogen-owner-thin-hcg-comparison-file-v0",
        warning: "DISPOSABLE ENGINEERING COMPARISON: not a promotion result",
        source_revision: env!("HEX3_GIT_REVISION"),
        source_dirty: env!("HEX3_GIT_DIRTY") == "true",
        timings,
        render,
        numerical,
    };
    let file = File::create(cli.output_dir.join("comparison.json"))?;
    serde_json::to_writer_pretty(BufWriter::new(file), &output)?;
    Ok(())
}
