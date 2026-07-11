use std::process::Command;

fn git_output(args: &[&str]) -> Option<String> {
    let output = Command::new("git").args(args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn main() {
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=.git/index");
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=Cargo.toml");
    println!("cargo:rerun-if-changed=build.rs");

    let revision = git_output(&["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".into());
    let dirty = git_output(&["status", "--porcelain", "--untracked-files=normal"])
        .is_some_and(|status| !status.is_empty());

    println!("cargo:rustc-env=HEX3_GIT_REVISION={revision}");
    println!("cargo:rustc-env=HEX3_GIT_DIRTY={dirty}");
}
