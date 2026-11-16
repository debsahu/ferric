//! End-to-end CLI integration tests for `ferric export-tour` (`T18.1a` + `T18.1b`), verifying:
//! (a) byte-for-byte deterministic JSON (`schema_version = "ferric-tour-v1"`) and `1280x720`
//!     RFC 2083 PNG keyframes across independent invocations (`CANONICAL_TOUR_SEED = 20261801`)
//!     for all 8 modules (`01..08`),
//! (b) scientific candor disclosures (empirical/cold range on `02`, `F8` `1KHM` holdout on `03`,
//!     `SYNTHETIC FORWARD-MODEL SHOWCASE` on `04..08`, `PROTOCOL SHOWCASE` on `08`) and zero
//!     `NaN`/`Inf`/host-path leaks,
//! (c) adversarial failure paths including odd `--width`, unknown `--module`, and fail-closed
//!     missing-`ffmpeg` handling under empty `PATH`.
//!
//! # Literature & Clean-Room Provenance
//! - Original clean-room 2D anti-aliased rasterizer, RFC 2083 PNG encoder, and deterministic
//!   `nmr-core` forward-model tour orchestrator (`seed = 20261801`).

use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::Value;

static NEXT_TMP_ID: AtomicU64 = AtomicU64::new(1);

const EXPECTED_ALL_SLUGS: &[&str] = &[
    "01_fid_pulprog_nus",
    "02_hsqc_strips_solvers",
    "03_structure_noe_rdc",
    "04_relax_modelfree",
    "05_exchange_cest_cpmg_csp",
    "06_nmr1d_multiplet_qnmr_octet",
    "07_dosy_diffusion",
    "08_metabolomics_mcp",
];

struct ScopedTempDir {
    path: PathBuf,
}

impl ScopedTempDir {
    fn new(prefix: &str) -> Self {
        let id = NEXT_TMP_ID.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "ferric_p19_export_tour_{prefix}_{}_{}",
            std::process::id(),
            id
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("create temp dir for export-tour test");
        Self { path: dir }
    }
}

impl Drop for ScopedTempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn ferric_bin() -> &'static str {
    env!("CARGO_BIN_EXE_ferric")
}

fn run_ferric(args: &[&str]) -> Result<String, String> {
    let out = Command::new(ferric_bin())
        .args(args)
        .output()
        .map_err(|e| format!("failed to spawn ferric: {e}"))?;
    if out.status.success() {
        Ok(String::from_utf8_lossy(&out.stdout).into_owned())
    } else {
        Err(format!(
            "exit={:?} stderr={}",
            out.status.code(),
            String::from_utf8_lossy(&out.stderr)
        ))
    }
}

fn assert_valid_png_1280x720_rgb(bytes: &[u8], label: &str) {
    assert!(
        bytes.starts_with(b"\x89PNG\r\n\x1a\n"),
        "{label}: missing PNG magic header"
    );
    assert!(
        bytes.len() > 64,
        "{label}: PNG file unexpectedly tiny ({} bytes)",
        bytes.len()
    );
    assert_eq!(&bytes[12..16], b"IHDR", "{label}: first chunk must be IHDR");
    let w = u32::from_be_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
    let h = u32::from_be_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]);
    assert_eq!((w, h), (1280, 720), "{label}: expected 1280x720 PNG");
    assert_eq!(bytes[24], 8, "{label}: expected 8-bit depth");
    assert_eq!(bytes[25], 2, "{label}: expected RGB color type 2");
    assert!(
        bytes.ends_with(b"IEND\xaeB`\x82"),
        "{label}: missing IEND trailer"
    );
}

#[test]
fn test_export_tour_keyframes_deterministic_and_schema_valid() {
    let run1 = ScopedTempDir::new("det_run1");
    let run2 = ScopedTempDir::new("det_run2");

    let r1_str = run1.path.to_string_lossy().into_owned();
    let r2_str = run2.path.to_string_lossy().into_owned();

    run_ferric(&[
        "export-tour",
        "--out-dir",
        &r1_str,
        "--module",
        "all",
        "--keyframes-only",
    ])
    .expect("first export-tour --keyframes-only run must succeed");

    run_ferric(&[
        "export-tour",
        "--out-dir",
        &r2_str,
        "--module",
        "all",
        "--keyframes-only",
    ])
    .expect("second export-tour --keyframes-only run must succeed");

    for (idx, slug) in EXPECTED_ALL_SLUGS.iter().enumerate() {
        let j1_path = run1.path.join("data").join(format!("{slug}.tour.json"));
        let j2_path = run2.path.join("data").join(format!("{slug}.tour.json"));
        let j1 = fs::read_to_string(&j1_path).expect("read run1 tour json");
        let j2 = fs::read_to_string(&j2_path).expect("read run2 tour json");
        assert_eq!(
            j1, j2,
            "{slug}: tour JSON must be byte-identical across runs"
        );
        assert!(
            !j1.contains("NaN") && !j1.contains("Infinity"),
            "{slug}: non-finite JSON"
        );
        assert!(
            !j1.contains("/Users/") && !j1.contains("/home/"),
            "{slug}: host path leak"
        );

        let parsed: Value = serde_json::from_str(&j1).expect("parse tour json");
        assert_eq!(parsed["schema_version"], "ferric-tour-v1");
        assert_eq!(parsed["module_index"], (idx + 1) as u64);
        assert_eq!(parsed["slug"], *slug);
        assert_eq!(parsed["duration_sec"], 16.0);
        assert_eq!(parsed["fps"], 30);
        let chapters = parsed["chapters"].as_array().expect("chapters array");
        assert_eq!(chapters.len(), 4, "{slug}: expected 4 chapters");

        // Scientific candor checks
        let badge_json = parsed["validation_badge"].to_string();
        match idx + 1 {
            2 => assert!(
                badge_json.contains("88.25")
                    && badge_json.contains("95.30")
                    && badge_json.contains("CISD3")
                    && badge_json.contains("FAIL")
                    && badge_json.contains("withdrawn"),
                "module 02 must disclose evo-graph cold BMRB range (88.25-95.30% FAIL/withdrawn) and CISD3"
            ),
            3 => assert!(
                badge_json.contains("F8") && badge_json.contains("1KHM"),
                "module 03 must disclose F8 1KHM unseen holdout"
            ),
            4 => assert!(
                badge_json.contains("SYNTHETIC FORWARD-MODEL SHOWCASE")
                    && badge_json.contains("F10/F11"),
                "module 04 must disclose SYNTHETIC FORWARD-MODEL SHOWCASE and F10/F11"
            ),
            5 => assert!(
                badge_json.contains("SYNTHETIC FORWARD-MODEL SHOWCASE")
                    && badge_json.contains("F13"),
                "module 05 must disclose SYNTHETIC FORWARD-MODEL SHOWCASE and F13"
            ),
            6 => assert!(
                badge_json.contains("SYNTHETIC FORWARD-MODEL SHOWCASE")
                    && badge_json.contains("F14"),
                "module 06 must disclose SYNTHETIC FORWARD-MODEL SHOWCASE and F14"
            ),
            7 => assert!(
                badge_json.contains("SYNTHETIC FORWARD-MODEL SHOWCASE")
                    && badge_json.contains("F16"),
                "module 07 must disclose SYNTHETIC FORWARD-MODEL SHOWCASE and F16"
            ),
            8 => assert!(
                badge_json.contains("SYNTHETIC") && badge_json.contains("PROTOCOL"),
                "module 08 must disclose SYNTHETIC & PROTOCOL SHOWCASE"
            ),
            _ => {}
        }

        let mut prev_png: Option<Vec<u8>> = None;
        for ch in 1..=4 {
            let p1 = run1
                .path
                .join("keyframes")
                .join(format!("{slug}_ch{ch}.png"));
            let p2 = run2
                .path
                .join("keyframes")
                .join(format!("{slug}_ch{ch}.png"));
            let b1 = fs::read(&p1).expect("read run1 png");
            let b2 = fs::read(&p2).expect("read run2 png");
            assert_valid_png_1280x720_rgb(&b1, &format!("{slug}_ch{ch}"));
            assert_eq!(b1, b2, "{slug}_ch{ch}.png must be deterministic");
            if let Some(prev) = prev_png {
                assert_ne!(prev, b1, "{slug} ch{ch} must differ from previous chapter");
            }
            prev_png = Some(b1);
        }
    }
}

#[test]
fn test_export_tour_single_module_and_adversarial_failures() {
    let dir = ScopedTempDir::new("single_and_adv");
    let dir_str = dir.path.to_string_lossy().into_owned();

    run_ferric(&[
        "export-tour",
        "--out-dir",
        &dir_str,
        "--module",
        "03_structure_noe_rdc",
        "--keyframes-only",
    ])
    .expect("single module export-tour must succeed");

    assert!(dir
        .path
        .join("data/03_structure_noe_rdc.tour.json")
        .is_file());
    assert!(!dir.path.join("data/01_fid_pulprog_nus.tour.json").exists());

    // Adversarial failure checks
    assert!(run_ferric(&["export-tour"]).is_err());
    assert!(run_ferric(&["export-tour", "--out-dir", ""]).is_err());
    assert!(run_ferric(&["export-tour", "--out-dir", &dir_str, "--module", "bogus"]).is_err());
    assert!(run_ferric(&["export-tour", "--out-dir", &dir_str, "--width", "1281"]).is_err());
    assert!(run_ferric(&["export-tour", "--out-dir", &dir_str, "--fps", "0"]).is_err());

    let regular_file = dir.path.join("not_a_dir.txt");
    fs::write(&regular_file, "x").expect("write regular file");
    let rf_str = regular_file.to_string_lossy().into_owned();
    assert!(run_ferric(&["export-tour", "--out-dir", &rf_str]).is_err());

    // Verify fail-closed diagnostic when ffmpeg is missing from PATH
    let out = Command::new(ferric_bin())
        .env("PATH", "")
        .args([
            "export-tour",
            "--out-dir",
            &dir_str,
            "--module",
            "01",
            "--render-videos",
        ])
        .output()
        .expect("spawn ferric with empty PATH");
    assert!(
        !out.status.success(),
        "missing ffmpeg under empty PATH must fail closed"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("ffmpeg"),
        "expected diagnostic mentioning ffmpeg, got: {stderr}"
    );
}

#[test]
fn test_module_01_and_02_validation_badges_candor() {
    let out_dir = ScopedTempDir::new("badge_candor");
    let out_str = out_dir.path.to_string_lossy().into_owned();
    run_ferric(&[
        "export-tour",
        "--out-dir",
        &out_str,
        "--module",
        "01",
        "--keyframes-only",
    ])
    .expect("export-tour module 01");
    run_ferric(&[
        "export-tour",
        "--out-dir",
        &out_str,
        "--module",
        "02",
        "--keyframes-only",
    ])
    .expect("export-tour module 02");

    let m1_json: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(out_dir.path.join("data/01_fid_pulprog_nus.tour.json"))
            .expect("read m1 json"),
    )
    .expect("parse m1 json");
    let m1_kind = m1_json["validation_badge"]["kind"]
        .as_str()
        .unwrap_or_default();
    assert_ne!(m1_kind, "empirical_calibrated");
    assert_eq!(m1_kind, "fid_ingest_synthetic_nus");
    let m1_detail = m1_json["validation_badge"]["detail"]
        .as_str()
        .unwrap_or_default();
    assert!(
        m1_detail.to_ascii_lowercase().contains("synthetic"),
        "Module 01 badge detail must disclose synthetic NUS demo: {m1_detail}"
    );

    let m2_json: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(out_dir.path.join("data/02_hsqc_strips_solvers.tour.json"))
            .expect("read m2 json"),
    )
    .expect("parse m2 json");
    let m2_kind = m2_json["validation_badge"]["kind"]
        .as_str()
        .unwrap_or_default();
    assert_ne!(m2_kind, "empirical_calibrated");
    assert_eq!(m2_kind, "cold_gate_failed_withdrawn");
    let m2_label = m2_json["validation_badge"]["label"]
        .as_str()
        .unwrap_or_default();
    let m2_detail = m2_json["validation_badge"]["detail"]
        .as_str()
        .unwrap_or_default();
    let combined = format!("{m2_label} {m2_detail}");
    assert!(
        combined.to_ascii_lowercase().contains("synthetic"),
        "Module 02 badge must label Ubiquitin as synthetic: {combined}"
    );
    assert!(
        combined.to_ascii_lowercase().contains("dev"),
        "Module 02 badge must label CISD3 as dev: {combined}"
    );
    assert!(
        combined.contains("FAIL") && combined.to_ascii_lowercase().contains("withdrawn"),
        "Module 02 badge must disclose cold gate FAIL and 13.1:P1-2 withdrawal: {combined}"
    );
}
