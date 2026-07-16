//! Generate a compact spatial dossier packet without initializing a GPU.

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use clap::Parser;
use hex3::world::{
    DossierOptions, DossierPacket, FineCacheMode, VoronoiBackend, World, NUM_PLATES_DEFAULT,
};

#[derive(Debug, Parser)]
#[command(
    name = "dossier",
    about = "Generate a CPU-only Hex3 spatial dossier packet"
)]
struct Cli {
    #[arg(long, default_value_t = 12345)]
    seed: u64,
    #[arg(long, default_value_t = 100_000)]
    cells: usize,
    #[arg(long, default_value_t = 1)]
    lloyd_iterations: usize,
    /// Fine-mesh guardrail. Zero uses the product default.
    #[arg(long, default_value_t = 0)]
    fine_max: usize,
    /// Output JSON. Defaults to artifacts/dossiers/seed-<seed>.json.
    #[arg(long)]
    output: Option<PathBuf>,
    /// Include per-cell product/null fields for external spatial mapping.
    #[arg(long)]
    include_climatology_spatial: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let cli = Cli::parse();
    let mut world = World::new_with_options(
        cli.seed,
        cli.cells,
        cli.lloyd_iterations,
        VoronoiBackend::ConvexHull,
    );
    world.fine_cache = FineCacheMode::Disabled;
    world.generate_all(NUM_PLATES_DEFAULT);
    world.generate_atmosphere();
    if cli.fine_max == 0 {
        world.generate_hydrology();
    } else {
        world.generate_hydrology_with_fine_cap(cli.fine_max);
    }
    let packet = DossierPacket::build_with_options(
        &world,
        DossierOptions {
            include_climatology_spatial_evidence: cli.include_climatology_spatial,
        },
    )?;
    let output = cli
        .output
        .unwrap_or_else(|| PathBuf::from(format!("artifacts/dossiers/seed-{}.json", cli.seed)));
    write_json_atomic(&output, &packet)?;
    println!("wrote {}", output.display());
    Ok(())
}

fn write_json_atomic(path: &Path, value: &impl serde::Serialize) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temp = path.with_extension(format!("json.tmp-{}", std::process::id()));
    let mut writer = BufWriter::new(File::create(&temp)?);
    serde_json::to_writer_pretty(&mut writer, value)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    drop(writer);
    fs::rename(temp, path)
}
