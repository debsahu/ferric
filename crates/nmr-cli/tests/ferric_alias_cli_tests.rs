//! Integration tests for Slice `18.1` (`ferric` primary binary, `nmrassign` deprecated alias,
//! `--version` format, and software provenance).
//!
//! Provenance: Clean-room CLI integration test verifying `PHASE18_FERRIC_RENAME_BRIEF.md`
//! §2 and §4.1 requirements.

use std::fs;
use std::path::PathBuf;
use std::process::Command;

const EXPECTED_DEPRECATION_LINE: &str =
    "warning: 'nmrassign' is deprecated; use 'ferric' (alias removed in v0.15.0)\n";

fn ferric_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_ferric"))
}

fn nmrassign_alias_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_nmrassign"))
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[test]
fn test_deprecated_nmrassign_alias_emits_single_stderr_warning_and_exits_zero() {
    let out = Command::new(nmrassign_alias_bin())
        .output()
        .expect("spawn nmrassign deprecated alias");
    assert!(
        out.status.success(),
        "nmrassign alias exited non-zero: {}",
        out.status
    );
    let stderr_str = String::from_utf8(out.stderr).expect("utf-8 stderr");
    assert_eq!(stderr_str, EXPECTED_DEPRECATION_LINE);
}

#[test]
fn test_nmrassign_alias_stdout_is_byte_identical_to_ferric() {
    let ws = workspace_root();
    let fixture_star = ws.join("data/fixtures/synthetic_ubiquitin.str");
    let fixture_dir = ws.join("data/fixtures");
    let real_manifest = ws.join("data/benchmark/real_manifest.tsv");

    let cases: Vec<Vec<String>> = vec![
        vec!["--help".to_string()],
        vec![
            "mcp-config".to_string(),
            "--client".to_string(),
            "antigravity".to_string(),
            "--root".to_string(),
            "/data/nmr_project".to_string(),
        ],
        vec![
            "score".to_string(),
            "--pred".to_string(),
            fixture_star.to_string_lossy().into_owned(),
            "--truth".to_string(),
            fixture_star.to_string_lossy().into_owned(),
        ],
        vec![
            "real-benchmark".to_string(),
            "--manifest".to_string(),
            real_manifest.to_string_lossy().into_owned(),
            "--split".to_string(),
            "dev".to_string(),
            "--discover-only".to_string(),
        ],
    ];

    for args in cases {
        let ferric_out = Command::new(ferric_bin())
            .args(&args)
            .output()
            .expect("spawn ferric");
        let alias_out = Command::new(nmrassign_alias_bin())
            .args(&args)
            .output()
            .expect("spawn nmrassign alias");

        assert!(
            ferric_out.status.success(),
            "ferric {:?} failed: {}",
            args,
            String::from_utf8_lossy(&ferric_out.stderr)
        );
        assert!(
            alias_out.status.success(),
            "nmrassign {:?} failed: {}",
            args,
            String::from_utf8_lossy(&alias_out.stderr)
        );
        assert_eq!(
            alias_out.stdout, ferric_out.stdout,
            "stdout mismatch between ferric and nmrassign for args {args:?}"
        );
        assert!(
            ferric_out.stderr.is_empty(),
            "ferric {:?} should not emit deprecation warning on stderr, got: {}",
            args,
            String::from_utf8_lossy(&ferric_out.stderr)
        );
        assert_eq!(
            String::from_utf8(alias_out.stderr).expect("utf-8 stderr"),
            EXPECTED_DEPRECATION_LINE,
            "nmrassign {:?} must emit exact single-line deprecation warning on stderr",
            args
        );
    }

    // Also verify `inspect --json` emits identical structured payload (normalizing wall-clock `elapsed_ms`)
    let inspect_args = ["inspect", fixture_dir.to_str().unwrap(), "--json"];
    let ferric_insp = Command::new(ferric_bin())
        .args(inspect_args)
        .output()
        .expect("spawn ferric inspect");
    let alias_insp = Command::new(nmrassign_alias_bin())
        .args(inspect_args)
        .output()
        .expect("spawn nmrassign inspect");
    assert!(ferric_insp.status.success());
    assert!(alias_insp.status.success());
    assert!(ferric_insp.stderr.is_empty());
    assert_eq!(
        String::from_utf8(alias_insp.stderr).expect("utf-8 stderr"),
        EXPECTED_DEPRECATION_LINE
    );
    let mut ferric_json: serde_json::Value =
        serde_json::from_slice(&ferric_insp.stdout).expect("ferric inspect json");
    let mut alias_json: serde_json::Value =
        serde_json::from_slice(&alias_insp.stdout).expect("alias inspect json");
    ferric_json
        .as_object_mut()
        .expect("json object")
        .remove("elapsed_ms");
    alias_json
        .as_object_mut()
        .expect("json object")
        .remove("elapsed_ms");
    assert_eq!(alias_json, ferric_json);
}

#[test]
fn test_ferric_version_output_exact_format() {
    for flag in ["--version", "-V", "version"] {
        let out = Command::new(ferric_bin())
            .arg(flag)
            .output()
            .expect("run ferric --version");
        assert!(out.status.success(), "ferric {flag} failed");
        assert!(
            out.stderr.is_empty(),
            "ferric {flag} emitted unexpected stderr: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let stdout = String::from_utf8(out.stdout).expect("utf-8 stdout");
        let line = stdout
            .strip_suffix('\n')
            .unwrap_or(&stdout)
            .trim_end_matches('\r');
        // Exact format: `ferric 0.14.0 (<sha>) — Rust + WebGPU`
        let prefix = format!("ferric {} (", env!("CARGO_PKG_VERSION"));
        assert!(
            line.starts_with(&prefix),
            "expected '{prefix}...' for `{flag}`, got: {line:?}"
        );
        let rest = &line[prefix.len()..];
        let (sha, suffix) = rest
            .split_once(") — Rust + WebGPU")
            .unwrap_or_else(|| panic!("expected ') — Rust + WebGPU' suffix in {line:?}"));
        assert!(suffix.is_empty(), "trailing chars after suffix in {line:?}");
        assert!(
            !sha.is_empty() && sha.chars().all(|c| c.is_ascii_hexdigit()),
            "expected lowercase hex git SHA in {line:?}, got {sha:?}"
        );
    }
}

#[test]
fn test_ferric_software_provenance_in_nef_and_html_report() {
    let tmp = std::env::temp_dir().join(format!("ferric_prov_test_{}", std::process::id()));
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).expect("create tmp dir");

    let fixture_star = workspace_root().join("data/fixtures/synthetic_ubiquitin.str");
    let nef_out = tmp.join("ubq.nef");
    let html_out = tmp.join("ubq_report.html");

    let nef_res = Command::new(ferric_bin())
        .args([
            "nef-export",
            "--star",
            fixture_star.to_str().unwrap(),
            "--out",
            nef_out.to_str().unwrap(),
        ])
        .output()
        .expect("run ferric nef-export");
    assert!(
        nef_res.status.success(),
        "nef-export failed: {}",
        String::from_utf8_lossy(&nef_res.stderr)
    );
    let nef_txt = fs::read_to_string(&nef_out).expect("read exported nef");
    assert!(
        nef_txt.contains("_nef_nmr_meta_data.program_name          FERRIC"),
        "expected program_name FERRIC in NEF:\n{}",
        &nef_txt[..nef_txt.len().min(600)]
    );

    let rep_res = Command::new(ferric_bin())
        .args([
            "report",
            "--star",
            fixture_star.to_str().unwrap(),
            "--out",
            html_out.to_str().unwrap(),
        ])
        .output()
        .expect("run ferric report");
    assert!(
        rep_res.status.success(),
        "report failed: {}",
        String::from_utf8_lossy(&rep_res.stderr)
    );
    let html_txt = fs::read_to_string(&html_out).expect("read exported report");
    assert!(
        html_txt.contains("Generated by <code>FERRIC</code> v0.14.0"),
        "expected FERRIC v0.14.0 footer in HTML report"
    );

    let _ = fs::remove_dir_all(&tmp);
}
