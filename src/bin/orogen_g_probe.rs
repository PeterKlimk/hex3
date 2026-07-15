//! Run the non-authoritative 4 km G engineering probe.

use clap::Parser;
use hex3::world::landscape::decode_linked_shared_input_bundle_stored_v0;
use hex3::world::landscape::organization_owner::{
    run_repeated_thin_g_4km_v0, ThinG4KmObservationV0, THIN_OWNER_PROFILE_V0,
};
use serde::Serialize;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Component, Path, PathBuf};
use std::time::Instant;

const ENVELOPE_SCHEMA_V0: &str = "orogen-owner-thin-g-probe-envelope-v0";

#[derive(Debug, Parser)]
#[command(
    name = "orogen_g_probe",
    about = "Run the disposable, non-authoritative 4 km G engineering probe"
)]
struct Cli {
    /// Accepted linked-input `shared-input.bin`.
    #[arg(long)]
    input: PathBuf,

    /// New JSON output file outside canonical organization-owner paths.
    #[arg(long)]
    output: PathBuf,
}

#[derive(Debug, Serialize)]
struct ProbeCostV0 {
    wall_seconds_including_input_validation: f64,
    peak_rss_kib: Option<u64>,
    deterministic_json_bytes: u64,
}

#[derive(Debug, Serialize)]
struct ProbeEnvelopeV0 {
    schema_version: &'static str,
    profile: &'static str,
    warning: &'static str,
    source_revision: &'static str,
    source_dirty: bool,
    cost: ProbeCostV0,
    deterministic: ThinG4KmObservationV0,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    validate_output_path(&cli.output)?;
    let started = Instant::now();
    let input_bytes = fs::read(&cli.input)?;
    // Stored decode rejects malformed/corrupt bytes; the runner then performs
    // one full executable replay validation before two identical owner runs.
    let bundle = decode_linked_shared_input_bundle_stored_v0(&input_bytes)?;
    let deterministic = run_repeated_thin_g_4km_v0(&bundle)?;
    let deterministic_json_bytes = u64::try_from(serde_json::to_vec(&deterministic)?.len())?;
    let envelope = ProbeEnvelopeV0 {
        schema_version: ENVELOPE_SCHEMA_V0,
        profile: THIN_OWNER_PROFILE_V0,
        warning: "DISPOSABLE ENGINEERING PROBE: not a campaign artifact or promotion result",
        source_revision: env!("HEX3_GIT_REVISION"),
        source_dirty: env!("HEX3_GIT_DIRTY") == "true",
        cost: ProbeCostV0 {
            wall_seconds_including_input_validation: started.elapsed().as_secs_f64(),
            peak_rss_kib: peak_rss_kib(),
            deterministic_json_bytes,
        },
        deterministic,
    };
    // Reserve the exact inode immediately before writing so computation
    // failures leave no empty result and later path changes cannot redirect it.
    let mut output_file = reserve_new_output(&cli.output)?;
    write_pretty_json(&mut output_file, &envelope)?;
    Ok(())
}

fn validate_output_path(path: &Path) -> Result<(), String> {
    if path.as_os_str().is_empty() {
        return Err("output path is empty".into());
    }
    if path.components().any(
        |component| matches!(component, Component::Normal(value) if value == "orogen-owner-v0"),
    ) {
        return Err(
            "engineering probes cannot write inside canonical orogen-owner-v0 paths".into(),
        );
    }
    for ancestor in path.ancestors() {
        if fs::symlink_metadata(ancestor).is_ok_and(|metadata| metadata.file_type().is_symlink()) {
            return Err("probe output cannot traverse an existing symbolic link".into());
        }
    }
    if path.exists() {
        return Err("probe output already exists".into());
    }
    Ok(())
}

fn reserve_new_output(path: &Path) -> Result<File, Box<dyn std::error::Error>> {
    validate_output_path(path)?;
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    validate_output_path(path)?;
    Ok(OpenOptions::new().create_new(true).write(true).open(path)?)
}

fn write_pretty_json<T: Serialize>(file: &mut File, value: &T) -> Result<(), std::io::Error> {
    let mut bytes = serde_json::to_vec_pretty(value).map_err(std::io::Error::other)?;
    bytes.push(b'\n');
    file.write_all(&bytes)?;
    file.sync_all()
}

fn peak_rss_kib() -> Option<u64> {
    fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|text| {
            text.lines().find_map(|line| {
                line.strip_prefix("VmHWM:")?
                    .split_whitespace()
                    .next()?
                    .parse()
                    .ok()
            })
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn cli_has_only_input_and_output_semantic_arguments() {
        Cli::command().debug_assert();
        assert!(Cli::try_parse_from(["orogen_g_probe"]).is_err());
        assert!(Cli::try_parse_from([
            "orogen_g_probe",
            "--input",
            "shared-input.bin",
            "--output",
            "probe.json",
            "--amplitude",
            "0.01",
        ])
        .is_err());
    }

    #[test]
    fn canonical_campaign_output_path_is_rejected() {
        assert!(validate_output_path(Path::new("artifacts/orogen-owner-v0/runs/g.json")).is_err());
        assert!(
            validate_output_path(Path::new("artifacts/orogen-owner-engineering/g.json")).is_ok()
        );
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_output_parent_is_rejected() {
        use std::os::unix::fs::symlink;

        let root = std::env::temp_dir().join(format!(
            "hex3-orogen-g-probe-path-test-{}",
            std::process::id()
        ));
        let canonical = root.join("orogen-owner-v0");
        let alias = root.join("engineering-alias");
        fs::create_dir_all(&canonical).unwrap();
        symlink(&canonical, &alias).unwrap();

        assert!(validate_output_path(&alias.join("g.json")).is_err());
        fs::remove_dir_all(&root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn reserved_output_cannot_be_redirected_after_the_run_starts() {
        use std::os::unix::fs::symlink;

        let root = std::env::temp_dir().join(format!(
            "hex3-orogen-g-probe-reservation-test-{}",
            std::process::id()
        ));
        let original_parent = root.join("engineering");
        let moved_parent = root.join("engineering-reserved");
        let canonical = root.join("orogen-owner-v0");
        fs::create_dir_all(&original_parent).unwrap();
        fs::create_dir_all(&canonical).unwrap();
        let output = original_parent.join("g.json");
        let mut file = reserve_new_output(&output).unwrap();

        fs::rename(&original_parent, &moved_parent).unwrap();
        symlink(&canonical, &original_parent).unwrap();
        write_pretty_json(&mut file, &[1u32, 2, 3]).unwrap();
        drop(file);

        assert_eq!(
            fs::read(moved_parent.join("g.json")).unwrap(),
            b"[\n  1,\n  2,\n  3\n]\n"
        );
        assert!(!canonical.join("g.json").exists());
        fs::remove_dir_all(&root).unwrap();
    }
}
