//! CLI integration tests for `ferric assign --af-structure` (`21.W2:P1-2`, `P1-3`, `P2-4`, `P2-6`).

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use nmr_core::nmrstar::read_star;
use nmr_core::peaks::write_peaks;
use nmr_core::simulate::{simulate, SimConfig};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn run_ferric(root: &Path, args: &[&str]) -> Output {
    let bin = PathBuf::from(env!("CARGO_BIN_EXE_ferric"));
    Command::new(&bin)
        .current_dir(root)
        .args(args)
        .output()
        .expect("run ferric binary")
}

#[test]
fn test_cli_af_structure_fail_closed_no_auto_ref_and_calibration_warning() {
    let root = workspace_root();
    let tmp_dir = std::env::temp_dir().join(format!("ferric_af_cli_p2_6_{}", std::process::id()));
    let _ = fs::remove_dir_all(&tmp_dir);
    fs::create_dir_all(&tmp_dir).expect("create temp dir");

    let star_path = root.join("data/benchmark/bmr17769.str");
    let ref_data = read_star(&star_path, None).unwrap();
    let (peaks, _) = simulate(
        &ref_data,
        &SimConfig {
            seed: 0,
            experiments: vec!["HSQC".to_string(), "HNCA".to_string()],
            ..SimConfig::default()
        },
    );
    let peaks_path = tmp_dir.join("peaks.tsv");
    write_peaks(&peaks_path, &peaks).unwrap();

    let af_cif = root.join("crates/nmr-core/tests/fixtures/AF-P0CG48-F1-model_v6.cif");
    let af_pae =
        root.join("crates/nmr-core/tests/fixtures/AF-P0CG48-F1-predicted_aligned_error_v6.json");
    let nmr_1d3z = root.join("data/pdb_refs/1d3z.pdb");
    let xray_1ubq = root.join("data/fixtures/ubiquitin_1ubq_xray.pdb");
    let headerless_pdb = tmp_dir.join("headerless.pdb");
    fs::write(
        &headerless_pdb,
        "ATOM      1  N   MET A   1      10.000  10.000  10.000  1.00 90.00           N\n\
         ATOM      2  CA  MET A   1      11.000  10.500  10.200  1.00 90.00           C\n\
         ATOM      3  C   MET A   1      12.000  11.000  10.400  1.00 90.00           C\n",
    )
    .unwrap();

    // 1. P1-3: Fail-closed rejection of experimental / headerless structures without --treat-as-model
    for (label, bad_path) in [
        ("1d3z", &nmr_1d3z),
        ("1ubq", &xray_1ubq),
        ("headerless", &headerless_pdb),
    ] {
        let out = run_ferric(
            &root,
            &[
                "assign",
                "--peaks",
                peaks_path.to_str().unwrap(),
                "--star",
                star_path.to_str().unwrap(),
                "--af-structure",
                bad_path.to_str().unwrap(),
                "--out",
                tmp_dir.join(format!("fail_{label}")).to_str().unwrap(),
            ],
        );
        assert!(
            !out.status.success(),
            "experimental/headerless {label} without --treat-as-model must fail closed"
        );
    }

    // 2. P1-3: Explicit --treat-as-model allows experimental structure opt-in
    let ok_treat = run_ferric(
        &root,
        &[
            "assign",
            "--peaks",
            peaks_path.to_str().unwrap(),
            "--star",
            star_path.to_str().unwrap(),
            "--af-structure",
            nmr_1d3z.to_str().unwrap(),
            "--treat-as-model",
            "--runs",
            "8",
            "--steps",
            "100",
            "--out",
            tmp_dir.join("ok_treat").to_str().unwrap(),
        ],
    );
    assert!(
        ok_treat.status.success(),
        "--treat-as-model must allow explicit override: stderr={}",
        String::from_utf8_lossy(&ok_treat.stderr)
    );

    // 3. P2-4 & P1-2: Real AF-P0CG48-F1 v6 with --no-auto-ref must report exact zero AF offsets
    //    AND emit uncalibrated_af_prior=true / confidence_calibrated=false
    let json_no_ref = tmp_dir.join("af_no_ref.json");
    let out_no_ref = tmp_dir.join("af_no_ref");
    let ok_no_ref = run_ferric(
        &root,
        &[
            "assign",
            "--peaks",
            peaks_path.to_str().unwrap(),
            "--star",
            star_path.to_str().unwrap(),
            "--af-structure",
            af_cif.to_str().unwrap(),
            "--af-pae",
            af_pae.to_str().unwrap(),
            "--no-auto-ref",
            "--noe-weight",
            "0.5",
            "--runs",
            "8",
            "--steps",
            "100",
            "--out",
            out_no_ref.to_str().unwrap(),
            "--json-out",
            json_no_ref.to_str().unwrap(),
        ],
    );
    assert!(
        ok_no_ref.status.success(),
        "AF v6 with --no-auto-ref failed: stderr={}",
        String::from_utf8_lossy(&ok_no_ref.stderr)
    );
    let json_raw = fs::read_to_string(&json_no_ref).unwrap();
    assert!(
        json_raw.contains("\"confidence_calibrated\": false"),
        "expected confidence_calibrated: false in JSON output"
    );
    let json_val: serde_json::Value = serde_json::from_str(&json_raw).unwrap();
    assert_eq!(
        json_val["af_priors"]["uncalibrated_af_prior"],
        serde_json::json!(true)
    );
    let offsets = &json_val["af_priors"]["af_offsets"];
    for k in ["h", "n", "ca", "cb", "c"] {
        assert_eq!(
            offsets[k].as_f64().unwrap(),
            0.0,
            "--no-auto-ref must report exact 0.0 for {k}"
        );
    }

    // 4. Out-of-range --af-weight must be rejected cleanly
    let bad_weight = run_ferric(
        &root,
        &[
            "assign",
            "--peaks",
            peaks_path.to_str().unwrap(),
            "--star",
            star_path.to_str().unwrap(),
            "--af-structure",
            af_cif.to_str().unwrap(),
            "--af-weight",
            "1.5",
            "--out",
            tmp_dir.join("bad_weight").to_str().unwrap(),
        ],
    );
    assert!(
        !bad_weight.status.success(),
        "--af-weight 1.5 outside [0, 1] must be rejected"
    );

    // 5. Full E2E with --noesy-15n, --nef-out, firewall flags, and --noesy-15n-without-af rejection
    let noesy_path = tmp_dir.join("noesy_15n.tsv");
    fs::write(
        &noesy_path,
        "experiment\th\tn\th2\tsign\theight\tvolume\nNOESY_N15\t8.2500\t120.5000\t8.7500\t1\t100000.0\t100000.0\n",
    )
    .unwrap();
    let e2e_prefix = tmp_dir.join("assigned_af");
    let e2e_json = tmp_dir.join("assigned_af.json");
    let e2e_nef = tmp_dir.join("assigned_af.nef");
    let ok_e2e = run_ferric(
        &root,
        &[
            "assign",
            "--peaks",
            peaks_path.to_str().unwrap(),
            "--star",
            star_path.to_str().unwrap(),
            "--af-structure",
            af_cif.to_str().unwrap(),
            "--af-pae",
            af_pae.to_str().unwrap(),
            "--noesy-15n",
            noesy_path.to_str().unwrap(),
            "--exclude-pdb",
            "1D3Z,1UBQ,1G6J",
            "--exclude-bmrb",
            "6457,17769,5387",
            "--exclude-topology",
            "2.60.98",
            "--runs",
            "8",
            "--steps",
            "100",
            "--out",
            e2e_prefix.to_str().unwrap(),
            "--json-out",
            e2e_json.to_str().unwrap(),
            "--nef-out",
            e2e_nef.to_str().unwrap(),
        ],
    );
    assert!(
        ok_e2e.status.success(),
        "E2E ferric assign --af-structure failed: stderr={}",
        String::from_utf8_lossy(&ok_e2e.stderr)
    );
    let e2e_json_txt = fs::read_to_string(&e2e_json).unwrap();
    assert!(e2e_json_txt.contains("\"exclusion_list_sha256\""));
    assert!(e2e_json_txt.contains("\"admitted_contacts_total\""));
    assert!(e2e_json_txt.contains("\"uncalibrated_af_prior\": true"));
    assert!(e2e_json_txt.contains("\"confidence_calibrated\": false"));
    let str_txt = fs::read_to_string(e2e_prefix.with_extension("str")).unwrap();
    assert!(str_txt.contains("af_priors("));
    assert!(str_txt.contains("uncalibrated_af_prior=true"));
    assert!(e2e_nef.exists());

    let no_af_noesy = run_ferric(
        &root,
        &[
            "assign",
            "--peaks",
            peaks_path.to_str().unwrap(),
            "--star",
            star_path.to_str().unwrap(),
            "--noesy-15n",
            noesy_path.to_str().unwrap(),
            "--out",
            tmp_dir.join("should_fail").to_str().unwrap(),
        ],
    );
    assert!(
        !no_af_noesy.status.success(),
        "--noesy-15n without --af-structure must be rejected"
    );

    let _ = fs::remove_dir_all(&tmp_dir);
}
