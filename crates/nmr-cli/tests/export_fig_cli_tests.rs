//! End-to-end CLI integration tests for `nmrassign export-fig` (`T16.1`), verifying:
//! (a) byte-for-byte deterministic SVG generation (`0`-byte diff across runs and vs committed `docs/assets/*.svg`),
//! (b) valid XML/SVG structure (`<svg` ... `</svg>`, non-empty geometric primitives & `<text`, `0` `NaN`/`Inf`, `0` host-path leaks),
//! (c) adversarial `is_err()` failure paths (unknown `--panel`, missing/empty `--out-dir`, regular-file `--out-dir`).
//!
//! # Literature & Clean-Room Provenance
//! - Lorensen, W. E. & Cline, H. E. (*ACM SIGGRAPH Comput. Graph.* **1987**, 21, 163–169) — bilinear marching-squares contour extraction.
//! - Driven end-to-end by deterministic `nmr-core` forward models & solvers (`seed = 42`).

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_TMP_ID: AtomicU64 = AtomicU64::new(1);

const EXPECTED_PANELS: &[(&str, &str)] = &[
    ("hsqc", "01_hsqc_contour_overlay.svg"),
    ("strips", "02_sequential_strip_walk.svg"),
    ("pdb-s2", "03_pdb_s2_dynamics_putty.svg"),
    ("nmr1d-stack", "04_nmr1d_multiplet_stack.svg"),
    ("dosy", "05_dosy_2d_diffusion_map.svg"),
    ("metabolomics", "06_metabolomics_oplsda_stocsy.svg"),
    ("exchange", "07_cest_cpmg_csp_exchange.svg"),
    ("octet", "08_octet_structure_link.svg"),
];

struct ScopedTempDir {
    path: PathBuf,
}

impl ScopedTempDir {
    fn new(prefix: &str) -> Self {
        let id = NEXT_TMP_ID.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "nmrassign_p16w1_export_fig_{prefix}_{}_{}",
            std::process::id(),
            id
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("create temp dir for export-fig test");
        Self { path: dir }
    }
}

impl Drop for ScopedTempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn nmrassign_bin() -> &'static str {
    env!("CARGO_BIN_EXE_nmrassign")
}

fn repo_assets_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../docs/assets")
}

fn run_nmrassign(args: &[&str]) -> Result<String, String> {
    let out = Command::new(nmrassign_bin())
        .args(args)
        .output()
        .map_err(|e| format!("failed to spawn nmrassign: {e}"))?;
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

/// Lightweight strict XML/SVG structural validator checking root `<svg>...</svg>`, balanced
/// element stack, well-quoted attributes, valid XML character entities, required SVG primitives,
/// `0` `NaN`/`Inf` numeric tokens, and `0` local filesystem user-path leaks.
fn assert_valid_svg_document(filename: &str, svg: &str) {
    let trimmed = svg.trim();
    let body = if let Some(rest) = trimmed.strip_prefix("<?xml") {
        let end_pi = rest
            .find("?>")
            .unwrap_or_else(|| panic!("{filename}: unclosed XML declaration"));
        rest[end_pi + 2..].trim_start()
    } else {
        trimmed
    };
    assert!(
        body.starts_with("<svg ") || body.starts_with("<svg>"),
        "{filename}: must open with <svg>"
    );
    assert!(
        body.ends_with("</svg>"),
        "{filename}: must close with </svg>"
    );
    assert!(
        svg.contains("xmlns=\"http://www.w3.org/2000/svg\""),
        "{filename}: missing SVG namespace"
    );

    // Must contain real plotted vector primitives and labels
    let has_geom = svg.contains("<path ") || svg.contains("<circle ") || svg.contains("<line ");
    assert!(
        has_geom,
        "{filename}: expected at least one <path, <circle, or <line element"
    );
    assert!(
        svg.contains("<text "),
        "{filename}: expected non-empty <text annotations"
    );

    // Check zero NaN / Inf tokens anywhere in attributes or text
    for forbidden in [
        "NaN", "nan\"", "\"nan", " nan ", "Inf\"", "-Inf", "inf\"", "-inf\"", "infinity",
        "Infinity",
    ] {
        assert!(
            !svg.contains(forbidden),
            "{filename}: contains non-finite token {forbidden:?}"
        );
    }

    // Check zero host path leaks
    let lower = svg.to_ascii_lowercase();
    let mac_prefix = concat!("/", "us", "ers/");
    let nix_prefix = concat!("/", "ho", "me/");
    assert!(
        !lower.contains(mac_prefix) && !lower.contains(nix_prefix),
        "{filename}: leaked local user directory path"
    );

    // Verify balanced XML element stack and valid entities outside `<style>...</style>`
    verify_balanced_xml_tags(filename, body);
}

fn verify_balanced_xml_tags(filename: &str, xml: &str) {
    let bytes = xml.as_bytes();
    let n = bytes.len();
    let mut i = 0usize;
    let mut stack: Vec<String> = Vec::new();

    while i < n {
        if bytes[i] == b'&' {
            let semi_rel = xml[i..]
                .find(';')
                .unwrap_or_else(|| panic!("{filename}: unescaped '&' near byte {i}"));
            let ent = &xml[i + 1..i + semi_rel];
            let valid_ent = matches!(ent, "amp" | "lt" | "gt" | "quot" | "apos")
                || (ent.starts_with("#x")
                    && ent.len() > 2
                    && ent[2..].chars().all(|c| c.is_ascii_hexdigit()))
                || (ent.starts_with('#')
                    && ent.len() > 1
                    && ent[1..].chars().all(|c| c.is_ascii_digit()));
            assert!(
                valid_ent,
                "{filename}: invalid XML entity &{ent}; near byte {i}"
            );
            i += semi_rel + 1;
            continue;
        }
        if bytes[i] != b'<' {
            i += 1;
            continue;
        }
        if xml[i..].starts_with("<!--") {
            let end_c = xml[i + 4..]
                .find("-->")
                .unwrap_or_else(|| panic!("{filename}: unclosed comment"));
            i += 4 + end_c + 3;
            continue;
        }
        if xml[i..].starts_with("<![CDATA[") {
            let end_cd = xml[i + 9..]
                .find("]]>")
                .unwrap_or_else(|| panic!("{filename}: unclosed CDATA"));
            i += 9 + end_cd + 3;
            continue;
        }

        // Parse tag up to `>` respecting quoted attribute strings
        let mut j = i + 1;
        let mut in_quote: Option<u8> = None;
        while j < n {
            let b = bytes[j];
            if let Some(q) = in_quote {
                if b == q {
                    in_quote = None;
                }
            } else if b == b'"' || b == b'\'' {
                in_quote = Some(b);
            } else if b == b'>' {
                break;
            } else if b == b'<' {
                panic!("{filename}: unexpected '<' inside tag near byte {j}");
            }
            j += 1;
        }
        assert!(
            j < n && in_quote.is_none(),
            "{filename}: unclosed tag or attribute quote near byte {i}"
        );

        let inside = xml[i + 1..j].trim();
        assert!(!inside.is_empty(), "{filename}: empty <> tag");

        if let Some(close_name) = inside.strip_prefix('/') {
            let close_tag = close_name.trim();
            let opened = stack
                .pop()
                .unwrap_or_else(|| panic!("{filename}: closing </{close_tag}> on empty stack"));
            assert_eq!(
                opened, close_tag,
                "{filename}: mismatched closing tag </{close_tag}> for <{opened}>"
            );
        } else {
            let self_closing = inside.ends_with('/');
            let core = if self_closing {
                inside[..inside.len() - 1].trim_end()
            } else {
                inside
            };
            let tag_name = core
                .split(|c: char| c.is_ascii_whitespace())
                .next()
                .unwrap_or("");
            assert!(
                !tag_name.is_empty(),
                "{filename}: missing element name near byte {i}"
            );
            if !self_closing {
                if tag_name == "style" {
                    // Advance past raw CSS text until `</style>`
                    let rest = &xml[j + 1..];
                    let close_rel = rest
                        .find("</style>")
                        .unwrap_or_else(|| panic!("{filename}: unclosed <style> block"));
                    i = j + 1 + close_rel + "</style>".len();
                    continue;
                }
                stack.push(tag_name.to_string());
            }
        }
        i = j + 1;
    }

    assert!(
        stack.is_empty(),
        "{filename}: unclosed XML tags remaining: {stack:?}"
    );
}

#[test]
fn test_export_fig_all_deterministic_zero_diff_vs_committed_assets() {
    let dir_a = ScopedTempDir::new("all_a");
    let dir_b = ScopedTempDir::new("all_b");

    let dir_a_str = dir_a.path.to_str().expect("utf8 temp path");
    let dir_b_str = dir_b.path.to_str().expect("utf8 temp path");

    run_nmrassign(&["export-fig", "--out-dir", dir_a_str])
        .expect("nmrassign export-fig default all should succeed");
    run_nmrassign(&["export-fig", "--out-dir", dir_b_str, "--panel", "all"])
        .expect("nmrassign export-fig --panel all should succeed");

    let committed_dir = repo_assets_dir();
    for &(_slug, fname) in EXPECTED_PANELS {
        let p_a = dir_a.path.join(fname);
        let p_b = dir_b.path.join(fname);
        let p_committed = committed_dir.join(fname);

        assert!(p_a.is_file(), "missing generated panel {}", p_a.display());
        assert!(p_b.is_file(), "missing generated panel {}", p_b.display());
        assert!(
            p_committed.is_file(),
            "missing committed asset {}",
            p_committed.display()
        );

        let bytes_a = fs::read(&p_a).expect("read panel run A");
        let bytes_b = fs::read(&p_b).expect("read panel run B");
        let bytes_committed = fs::read(&p_committed).expect("read committed panel");

        assert_eq!(
            bytes_a, bytes_b,
            "{fname}: repeated export-fig runs must have 0-byte diff"
        );
        assert_eq!(
            bytes_a, bytes_committed,
            "{fname}: fresh export-fig output must have 0-byte diff vs committed docs/assets/{fname}"
        );

        let svg_text = std::str::from_utf8(&bytes_a).expect("UTF-8 SVG output");
        assert_valid_svg_document(fname, svg_text);
    }
}

#[test]
fn test_export_fig_individual_panels_match_committed_bytes() {
    let committed_dir = repo_assets_dir();
    for &(slug, fname) in EXPECTED_PANELS {
        let single_dir = ScopedTempDir::new(&format!("single_{slug}"));
        let out_str = single_dir.path.to_str().expect("utf8 path");
        run_nmrassign(&["export-fig", "--out-dir", out_str, "--panel", slug])
            .unwrap_or_else(|e| panic!("single panel --panel {slug} failed: {e}"));

        let mut written: Vec<String> = fs::read_dir(&single_dir.path)
            .expect("read single panel dir")
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        written.sort();
        assert_eq!(
            written,
            vec![fname.to_string()],
            "--panel {slug} should write exclusively {fname}"
        );

        let single_bytes = fs::read(single_dir.path.join(fname)).expect("read single panel svg");
        let committed_bytes = fs::read(committed_dir.join(fname)).expect("read committed svg");
        assert_eq!(
            single_bytes, committed_bytes,
            "--panel {slug} must match committed {fname} byte-for-byte"
        );
    }
}

#[test]
fn test_export_fig_adversarial_cli_errors_is_err() {
    let tmp = ScopedTempDir::new("adversarial");
    let tmp_str = tmp.path.to_str().expect("utf8 path");

    // 1. Unknown `--panel` value
    let err_panel = run_nmrassign(&[
        "export-fig",
        "--out-dir",
        tmp_str,
        "--panel",
        "nonexistent-panel",
    ]);
    assert!(err_panel.is_err(), "unknown --panel must fail");

    // 2. Missing `--out-dir`
    let err_missing_out = run_nmrassign(&["export-fig", "--panel", "hsqc"]);
    assert!(err_missing_out.is_err(), "missing --out-dir must fail");

    // 3. Empty `--out-dir ""`
    let err_empty_out = run_nmrassign(&["export-fig", "--out-dir", "", "--panel", "hsqc"]);
    assert!(err_empty_out.is_err(), "empty --out-dir must fail");

    // 4. Unknown flag
    let err_flag = run_nmrassign(&["export-fig", "--out-dir", tmp_str, "--not-a-flag"]);
    assert!(err_flag.is_err(), "unknown flag must fail");

    // 5. `--out-dir` pointing to an existing regular file (and child under regular file)
    let blocker_file: PathBuf = Path::new(&tmp.path).join("regular_file_not_a_dir.txt");
    fs::write(&blocker_file, b"not a directory").expect("write regular file");
    let blocker_str = blocker_file.to_str().expect("utf8 file path");
    let err_file_out = run_nmrassign(&["export-fig", "--out-dir", blocker_str, "--panel", "octet"]);
    assert!(
        err_file_out.is_err(),
        "--out-dir pointing to an existing regular file must fail"
    );

    let child_under_file = blocker_file.join("nested_assets");
    let child_str = child_under_file.to_str().expect("utf8 child path");
    let err_nested_file_out =
        run_nmrassign(&["export-fig", "--out-dir", child_str, "--panel", "octet"]);
    assert!(
        err_nested_file_out.is_err(),
        "--out-dir nested beneath a regular file must fail"
    );
}
