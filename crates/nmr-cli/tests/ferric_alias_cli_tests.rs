//! Integration tests verifying `v0.15.0` binary branding (`ferric` only; scheduled removal of
//! the `v0.14.x` `nmrassign` compatibility alias) and `--version` output format.
//!
//! Provenance: Clean-room CLI integration test verifying `PHASE18_FERRIC_RENAME_BRIEF.md`
//! §2 (`v0.15.0` alias removal) and `PHASE19_W1_CLAUDE_REVIEW_SUMMARY.md` guidance.

use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn ferric_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_ferric"))
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[test]
fn test_nmrassign_alias_binary_removed_in_v0_15() {
    assert!(
        option_env!("CARGO_BIN_EXE_nmrassign").is_none(),
        "CARGO_BIN_EXE_nmrassign must not be built in v0.15.0+"
    );

    let cli_cargo = fs::read_to_string(workspace_root().join("crates/nmr-cli/Cargo.toml"))
        .expect("read crates/nmr-cli/Cargo.toml");
    let bin_headers: Vec<&str> = cli_cargo
        .lines()
        .filter(|l| l.trim() == "[[bin]]")
        .collect();
    assert_eq!(
        bin_headers.len(),
        1,
        "crates/nmr-cli/Cargo.toml must define strictly one [[bin]] target in v0.15.0"
    );
    assert!(
        cli_cargo.contains("name = \"ferric\""),
        "single [[bin]] target must be named ferric"
    );
    assert!(
        !cli_cargo.contains("name = \"nmrassign\""),
        "nmrassign [[bin]] alias must be removed in v0.15.0"
    );
}

#[test]
fn test_ferric_version_output_exact_format_and_empty_stderr() {
    let out = Command::new(ferric_bin())
        .arg("--version")
        .output()
        .expect("spawn ferric --version");
    assert!(out.status.success(), "ferric --version failed");
    let stderr_str = String::from_utf8(out.stderr).expect("utf-8 stderr");
    assert!(
        stderr_str.is_empty(),
        "expected empty stderr on ferric --version, got {stderr_str:?}"
    );
    let stdout_str = String::from_utf8(out.stdout).expect("utf-8 stdout");
    let line = stdout_str.trim_end_matches('\n');
    let expected_prefix = format!("ferric {} (", env!("CARGO_PKG_VERSION"));
    assert!(
        line.starts_with(&expected_prefix) && line.ends_with(") \u{2014} Rust + WebGPU"),
        "unexpected --version output: {line:?}"
    );
    assert!(line.contains("0.15.0"), "expected v0.15.0 in {line:?}");
}
