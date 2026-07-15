//! Materialize the preregistered arm-neutral linked-orogen input identity.

use clap::Parser;
use hex3::world::landscape::materialize_linked_shared_input_v0;
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(
    name = "orogen_linked_input",
    about = "Materialize the preregistered linked-orogen shared-input V0 bundle"
)]
struct Cli {
    /// Required new directory to publish atomically. It must not already exist.
    #[arg(long, value_name = "PATH")]
    output_dir: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    materialize_linked_shared_input_v0(&cli.output_dir)?;
    eprintln!("wrote {}", cli.output_dir.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_directory_is_required_and_no_extra_semantic_arguments_exist() {
        assert!(Cli::try_parse_from(["orogen_linked_input"]).is_err());
        let parsed = Cli::try_parse_from([
            "orogen_linked_input",
            "--output-dir",
            "artifacts/orogen-linked-input-v0",
        ])
        .unwrap();
        assert_eq!(
            parsed.output_dir,
            PathBuf::from("artifacts/orogen-linked-input-v0")
        );
        assert!(Cli::try_parse_from([
            "orogen_linked_input",
            "--output-dir",
            "somewhere",
            "--arm",
            "C"
        ])
        .is_err());
    }
}
