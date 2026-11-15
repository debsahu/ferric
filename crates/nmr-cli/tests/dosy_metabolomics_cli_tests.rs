//! End-to-end CLI integration tests for `nmrassign dosy` (`fit`, `ilt`, `mw`, `T14.4`) and
//! `nmrassign metabolomics` (`bin`, `pca`, `oplsda`, `stocsy`, `T14.5`).
//!
//! # Literature & Clean-Room Provenance
//! - Stejskal, E. O. & Tanner, J. E. (*J. Chem. Phys.* **1965**, 42, 288–292).
//! - Wu, D. H., Chen, A. D. & Johnson, C. S. Jr. (*J. Magn. Reson. A* **1995**, 115, 260–264).
//! - Provencher, S. W. (*Comput. Phys. Commun.* **1982**, 27, 213–227 & 229–242).
//! - Golub, G. H., Heath, M. & Wahba, G. (*Technometrics* **1979**, 21, 215–223).
//! - Evans, R. et al. (*Angew. Chem. Int. Ed.* **2013**, 52, 3199–3202).
//! - Savorani, F., Tomasi, G. & Engelsen, S. B. (*J. Magn. Reson.* **2010**, 202, 190–202).
//! - De Meyer, T. et al. (*Anal. Chem.* **2008**, 80, 3783–3790).
//! - Dieterle, F., Ross, A., Schlotterbeck, G. & Senn, H. (*Anal. Chem.* **2006**, 78, 4281–4290).
//! - Trygg, J. & Wold, S. (*J. Chemom.* **2002**, 16, 119–128); Westerhuis, J. A. et al.
//!   (*Metabolomics* **2008**, 4, 81–89); Wiklund, S. et al. (*Anal. Chem.* **2008**, 80, 115–122).
//! - Cloarec, O. et al. (*Anal. Chem.* **2005**, 77, 1282–1289).

use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use nmr_core::dosy::{
    compute_b_values_s_m2, DosyNucleus, DosySequenceType, GradientShape, ILT_ARTIFACT_CAVEAT,
};
use nmr_core::metabolomics::generate_demo_biofluid_cohort;
use nmr_core::rng::Rng;
use nmr_gpu::GpuEngine;
use serde_json::Value;

static NEXT_TMP_ID: AtomicU64 = AtomicU64::new(1);

struct ScopedTempDir {
    path: PathBuf,
}

impl ScopedTempDir {
    fn new(prefix: &str) -> Self {
        let id = NEXT_TMP_ID.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "nmrassign_p14w2_cli_{prefix}_{}_{}",
            std::process::id(),
            id
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("create temp test dir");
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

#[test]
fn test_dosy_and_metabolomics_cli_help_and_error_handling() {
    let top = Command::new(nmrassign_bin())
        .output()
        .expect("run top-level nmrassign");
    let top_stdout = String::from_utf8_lossy(&top.stdout);
    assert!(top_stdout.contains("nmrassign dosy"));
    assert!(top_stdout.contains("nmrassign metabolomics"));

    for args in [
        vec!["dosy", "--help"],
        vec!["dosy", "fit", "--help"],
        vec!["dosy", "ilt", "--help"],
        vec!["dosy", "mw", "--help"],
        vec!["metabolomics", "--help"],
        vec!["metabolomics", "bin", "--help"],
        vec!["metabolomics", "pca", "--help"],
        vec!["metabolomics", "oplsda", "--help"],
        vec!["metabolomics", "stocsy", "--help"],
    ] {
        let out = Command::new(nmrassign_bin())
            .args(&args)
            .output()
            .expect("run --help");
        assert!(out.status.success(), "--help failed for {args:?}");
    }

    for bad_args in [
        vec!["dosy", "unknown-sub"],
        vec!["dosy", "fit", "--engine", "quantum"],
        vec!["dosy", "ilt", "--demo", "--strategy", "discrepancy"],
        vec!["dosy", "mw"],
        vec!["metabolomics", "unknown-sub"],
        vec!["metabolomics", "bin", "--method", "bogus"],
        vec!["metabolomics", "stocsy", "--demo"],
    ] {
        let out = Command::new(nmrassign_bin())
            .args(&bad_args)
            .output()
            .expect("run invalid args");
        assert!(
            !out.status.success(),
            "expected non-zero exit for {bad_args:?}"
        );
    }
}

#[test]
fn test_dosy_fit_cpu_gpu_parity_and_bruker_varian_ingestion() {
    let tmp = ScopedTempDir::new("dosy_fit");

    // 1. Construct synthetic Bruker BPP-LED experiment directory (`difflist` + `acqus`)
    let bruker_dir = tmp.path.join("bruker_bppled");
    fs::create_dir_all(&bruker_dir).unwrap();
    let n_grad = 24usize;
    let grads_g_cm: Vec<f64> = (0..n_grad)
        .map(|m| 2.0 + 53.0 * (m as f64) / ((n_grad - 1) as f64))
        .collect();
    let difflist_txt: String = grads_g_cm.iter().map(|g| format!("{g:.4}\n")).collect();
    fs::write(bruker_dir.join("difflist"), difflist_txt).unwrap();
    let acqus_txt = "\
##TITLE= Synthetic BPP-LED DOSY
##$PULPROG= <ledbpgp2s>
##$NUC1= <1H>
##$D= (0..63)
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.0002 0 0 0 0.1000
##$P= (0..63)
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1200.0
##$GPNAM= (0..9)
<> <> <> <> <> <> <SINE.100> <> <> <>
##END=
";
    fs::write(bruker_dir.join("acqus"), acqus_txt).unwrap();

    let b_vals = compute_b_values_s_m2(
        &grads_g_cm,
        DosyNucleus::H1,
        DosySequenceType::BppLed,
        GradientShape::SineBell,
        0.1000,
        0.0024,
        0.0002,
    )
    .expect("compute b values");

    // Build wide decay TSV with planted K=1, K=2, and K=3 peaks
    let d1_true = 2.10e-10_f64;
    let d2_a = 1.85e-10_f64;
    let d2_b = 9.20e-10_f64;
    let d3_a = 1.15e-10_f64;
    let d3_b = 4.60e-10_f64;
    let d3_c = 1.95e-9_f64;

    let mut rng = Rng::new(20261404);
    let mut decay_tsv = String::from("label\tppm");
    for m in 0..n_grad {
        decay_tsv.push_str(&format!("\tI_{m}"));
    }
    decay_tsv.push('\n');

    let mut row_k1 = String::from("Mono_7.20\t7.200");
    let mut row_k2 = String::from("BiMix_3.65\t3.650");
    let mut row_k3 = String::from("TriMix_1.40\t1.400");
    for &b in &b_vals {
        let y1 = 100.0 * (-b * d1_true).exp() + rng.normal(0.0, 0.04);
        let y2 = 55.0 * (-b * d2_a).exp() + 45.0 * (-b * d2_b).exp() + rng.normal(0.0, 0.03);
        let y3 = 38.0 * (-b * d3_a).exp()
            + 34.0 * (-b * d3_b).exp()
            + 28.0 * (-b * d3_c).exp()
            + rng.normal(0.0, 0.015);
        row_k1.push_str(&format!("\t{y1:.6}"));
        row_k2.push_str(&format!("\t{y2:.6}"));
        row_k3.push_str(&format!("\t{y3:.6}"));
    }
    decay_tsv.push_str(&row_k1);
    decay_tsv.push('\n');
    decay_tsv.push_str(&row_k2);
    decay_tsv.push('\n');
    decay_tsv.push_str(&row_k3);
    decay_tsv.push('\n');

    let decay_path = tmp.path.join("mixture_decays.tsv");
    fs::write(&decay_path, &decay_tsv).unwrap();

    let cpu_json_path = tmp.path.join("dosy_cpu.json");
    let cpu_tsv_path = tmp.path.join("dosy_cpu.tsv");
    let out_cpu = Command::new(nmrassign_bin())
        .args([
            "dosy",
            "fit",
            "--in",
            decay_path.to_str().unwrap(),
            "--exp-dir",
            bruker_dir.to_str().unwrap(),
            "--engine",
            "cpu",
            "--max-components",
            "3",
            "--bootstrap",
            "160",
            "-o",
            cpu_tsv_path.to_str().unwrap(),
            "--json-out",
            cpu_json_path.to_str().unwrap(),
        ])
        .output()
        .expect("run dosy fit cpu");
    assert!(
        out_cpu.status.success(),
        "dosy fit cpu failed: {}",
        String::from_utf8_lossy(&out_cpu.stderr)
    );

    let cpu_val: Value =
        serde_json::from_str(&fs::read_to_string(&cpu_json_path).unwrap()).unwrap();
    assert_eq!(cpu_val["engine_used"].as_str(), Some("cpu"));
    let peaks = cpu_val["peaks"].as_array().unwrap();
    assert_eq!(peaks.len(), 3);

    // K=1 mono peak check
    assert_eq!(peaks[0]["selected_k"].as_u64(), Some(1));
    let d1_fit = peaks[0]["components"][0]["d_m2_s"].as_f64().unwrap();
    let d1_rel_err = (d1_fit - d1_true).abs() / d1_true;
    assert!(d1_rel_err < 0.015, "K=1 rel err {d1_rel_err}");
    let ci_lo = peaks[0]["components"][0]["ci95_d_m2_s"][0]
        .as_f64()
        .unwrap();
    let ci_hi = peaks[0]["components"][0]["ci95_d_m2_s"][1]
        .as_f64()
        .unwrap();
    assert!(ci_lo < d1_fit && d1_fit < ci_hi);

    // K=2 bi-exponential peak check
    assert_eq!(peaks[1]["selected_k"].as_u64(), Some(2));
    let mut k2_ds: Vec<f64> = peaks[1]["components"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["d_m2_s"].as_f64().unwrap())
        .collect();
    k2_ds.sort_by(|a, b| a.total_cmp(b));
    let k2_err_a = (k2_ds[0] - d2_a).abs() / d2_a;
    let k2_err_b = (k2_ds[1] - d2_b).abs() / d2_b;
    assert!(k2_err_a < 0.05 && k2_err_b < 0.05);

    // K=3 tri-exponential peak check
    assert_eq!(peaks[2]["selected_k"].as_u64(), Some(3));
    let mut k3_ds: Vec<f64> = peaks[2]["components"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["d_m2_s"].as_f64().unwrap())
        .collect();
    k3_ds.sort_by(|a, b| a.total_cmp(b));
    assert!((k3_ds[0] - d3_a).abs() / d3_a < 0.12);
    assert!((k3_ds[1] - d3_b).abs() / d3_b < 0.15);
    assert!((k3_ds[2] - d3_c).abs() / d3_c < 0.12);

    // 2. WebGPU parity check on `--demo` (K=1 & K=2 columns)
    let demo_cpu_json = tmp.path.join("demo_cpu.json");
    let demo_gpu_json = tmp.path.join("demo_gpu.json");
    assert!(Command::new(nmrassign_bin())
        .args([
            "dosy",
            "fit",
            "--demo",
            "--engine",
            "cpu",
            "--bootstrap",
            "64",
            "--json-out",
            demo_cpu_json.to_str().unwrap(),
        ])
        .status()
        .unwrap()
        .success());

    let mut max_gpu_dex_diff = 0.0_f64;
    let gpu_available = GpuEngine::new_headless().is_ok();
    if gpu_available {
        let out_gpu = Command::new(nmrassign_bin())
            .args([
                "dosy",
                "fit",
                "--demo",
                "--engine",
                "gpu",
                "--bootstrap",
                "64",
                "--json-out",
                demo_gpu_json.to_str().unwrap(),
            ])
            .output()
            .expect("run dosy fit gpu");
        assert!(
            out_gpu.status.success(),
            "dosy fit gpu failed: {}",
            String::from_utf8_lossy(&out_gpu.stderr)
        );
        let c_v: Value =
            serde_json::from_str(&fs::read_to_string(&demo_cpu_json).unwrap()).unwrap();
        let g_v: Value =
            serde_json::from_str(&fs::read_to_string(&demo_gpu_json).unwrap()).unwrap();
        assert_eq!(g_v["engine_used"].as_str(), Some("gpu"));
        for (cp, gp) in c_v["peaks"]
            .as_array()
            .unwrap()
            .iter()
            .zip(g_v["peaks"].as_array().unwrap().iter())
        {
            assert_eq!(cp["selected_k"], gp["selected_k"]);
            for (cc, gc) in cp["components"]
                .as_array()
                .unwrap()
                .iter()
                .zip(gp["components"].as_array().unwrap().iter())
            {
                let diff =
                    (cc["log10_d"].as_f64().unwrap() - gc["log10_d"].as_f64().unwrap()).abs();
                max_gpu_dex_diff = max_gpu_dex_diff.max(diff);
            }
        }
        assert!(
            max_gpu_dex_diff < 0.015,
            "CPU vs WebGPU log10(D) diff {max_gpu_dex_diff}"
        );
    }

    // 3. Varian `procpar` directory ingestion check
    let varian_dir = tmp.path.join("varian_ste");
    fs::create_dir_all(&varian_dir).unwrap();
    let procpar_txt = format!(
        "seqfil 2 2 8 0 0 2 1 0 1 64\n1 \"Dste\"\n0\ntn 2 2 8 0 0 2 1 0 1 64\n1 \"H1\"\n0\ngshape 2 2 8 0 0 2 1 0 1 64\n1 \"r\"\n0\ngt1 1 1 1 0 0 2 1 0 1 64\n1 0.0020\n0\ndel 1 1 1 0 0 2 1 0 1 64\n1 0.0800\n0\ngcal 1 1 1 0 0 2 1 0 1 64\n1 0.0020\n0\ngzlvl1 1 1 32767 -32768 0 2 1 0 1 64\n{} {}\n0\n",
        n_grad,
        (0..n_grad)
            .map(|m| format!("{}", 1000 + m * 1000))
            .collect::<Vec<_>>()
            .join(" ")
    );
    fs::write(varian_dir.join("procpar"), procpar_txt).unwrap();
    let var_json = tmp.path.join("varian_fit.json");
    let out_var = Command::new(nmrassign_bin())
        .args([
            "dosy",
            "fit",
            "--in",
            decay_path.to_str().unwrap(),
            "--exp-dir",
            varian_dir.to_str().unwrap(),
            "--seq-type",
            "ste",
            "--shape",
            "rect",
            "--bootstrap",
            "32",
            "--json-out",
            var_json.to_str().unwrap(),
        ])
        .output()
        .expect("run dosy fit varian");
    assert!(
        out_var.status.success(),
        "varian dosy fit failed: {}",
        String::from_utf8_lossy(&out_var.stderr)
    );

    println!(
        "W2_METRIC_JSON: {}",
        serde_json::json!({
            "kind": "dosy_fit_cli",
            "k1_rel_err": d1_rel_err,
            "k2_rel_err_max": k2_err_a.max(k2_err_b),
            "gpu_available": gpu_available,
            "max_gpu_dex_diff": max_gpu_dex_diff,
            "passed": true,
        })
    );
}

#[test]
fn test_dosy_ilt_regularization_strategies_and_mw_calibration() {
    let tmp = ScopedTempDir::new("dosy_ilt_mw");

    let ilt_tsv = tmp.path.join("ilt_gcv.tsv");
    let ilt_json = tmp.path.join("ilt_gcv.json");
    let sweep_tsv = tmp.path.join("ilt_sweep.tsv");

    let out_gcv = Command::new(nmrassign_bin())
        .args([
            "dosy",
            "ilt",
            "--demo",
            "--strategy",
            "gcv",
            "--n-grid",
            "72",
            "--alpha-max",
            "1.0",
            "-o",
            ilt_tsv.to_str().unwrap(),
            "--json-out",
            ilt_json.to_str().unwrap(),
            "--alpha-sweep-out",
            sweep_tsv.to_str().unwrap(),
        ])
        .output()
        .expect("run dosy ilt gcv");
    assert!(
        out_gcv.status.success(),
        "dosy ilt gcv failed: {}",
        String::from_utf8_lossy(&out_gcv.stderr)
    );

    let tsv_content = fs::read_to_string(&ilt_tsv).unwrap();
    assert!(tsv_content.contains(ILT_ARTIFACT_CAVEAT));
    let sweep_lines = fs::read_to_string(&sweep_tsv).unwrap();
    assert!(sweep_lines.lines().count() > 20);

    let gcv_val: Value = serde_json::from_str(&fs::read_to_string(&ilt_json).unwrap()).unwrap();
    assert_eq!(gcv_val["caveat"].as_str(), Some(ILT_ARTIFACT_CAVEAT));
    let mix_modes = gcv_val["peaks"][1]["modes"].as_array().unwrap();
    assert!(
        mix_modes.len() >= 2,
        "expected bimodal ILT distribution on overlapping 3.68 ppm peak, got {mix_modes:?}"
    );

    // Shorthand `--fit contin-ilt` + `lcurve` and `discrepancy`
    let lc_json = tmp.path.join("ilt_lcurve.json");
    assert!(Command::new(nmrassign_bin())
        .args([
            "dosy",
            "--demo",
            "--fit",
            "contin-ilt",
            "--strategy",
            "lcurve",
            "--json-out",
            lc_json.to_str().unwrap(),
        ])
        .status()
        .unwrap()
        .success());

    let disc_json = tmp.path.join("ilt_disc.json");
    assert!(Command::new(nmrassign_bin())
        .args([
            "dosy",
            "ilt",
            "--demo",
            "--strategy",
            "discrepancy",
            "--noise-sigma",
            "0.40",
            "--json-out",
            disc_json.to_str().unwrap(),
        ])
        .status()
        .unwrap()
        .success());

    // Stokes-Einstein Rh & empirical MW (`dosy mw`)
    let mw_json = tmp.path.join("mw_protein.json");
    let out_mw = Command::new(nmrassign_bin())
        .args([
            "dosy",
            "mw",
            "--d",
            "1.11e-10,5.20e-10",
            "--solvent",
            "D2O",
            "--temp-k",
            "298.15",
            "--class",
            "protein",
            "--json-out",
            mw_json.to_str().unwrap(),
        ])
        .output()
        .expect("run dosy mw");
    assert!(out_mw.status.success());
    let mw_val: Value = serde_json::from_str(&fs::read_to_string(&mw_json).unwrap()).unwrap();
    let rh_lyso = mw_val["estimates"][0]["r_h_angstrom"].as_f64().unwrap();
    assert!(
        (17.5..=22.5).contains(&rh_lyso),
        "expected Lysozyme Rh near ~19.8 A in D2O, got {rh_lyso}"
    );

    // Standards-calibrated power-law MW
    let cal_json = tmp.path.join("mw_cal.json");
    assert!(Command::new(nmrassign_bin())
        .args([
            "dosy",
            "mw",
            "--d",
            "3.40e-10",
            "--standard",
            "180.16:5.20e-10",
            "--standard",
            "342.30:4.00e-10",
            "--standard",
            "1153.0:2.20e-10",
            "--json-out",
            cal_json.to_str().unwrap(),
        ])
        .status()
        .unwrap()
        .success());
    let cal_val: Value = serde_json::from_str(&fs::read_to_string(&cal_json).unwrap()).unwrap();
    assert_eq!(cal_val["calibrated_from_standards"].as_bool(), Some(true));
}

#[test]
fn test_metabolomics_bin_pca_oplsda_and_stocsy_end_to_end() {
    let tmp = ScopedTempDir::new("metabolomics_e2e");

    // 1. `nmrassign metabolomics bin` on 24-sample biofluid cohort (AIB + PQN + Pareto + Urine mask)
    let buckets_csv = tmp.path.join("buckets.csv");
    let study_json = tmp.path.join("study.json");
    let out_bin = Command::new(nmrassign_bin())
        .args([
            "metabolomics",
            "bin",
            "--demo",
            "--seed",
            "20261115",
            "--mask",
            "urine",
            "--method",
            "aib",
            "--norm",
            "pqn",
            "--scale",
            "pareto",
            "-o",
            buckets_csv.to_str().unwrap(),
            "--study-out",
            study_json.to_str().unwrap(),
        ])
        .output()
        .expect("run metabolomics bin");
    assert!(
        out_bin.status.success(),
        "metabolomics bin failed: {}",
        String::from_utf8_lossy(&out_bin.stderr)
    );
    let bin_stdout = String::from_utf8_lossy(&out_bin.stdout);
    assert!(bin_stdout.contains("samples=24"));

    // Verify water (4.65..5.05 ppm) and urea (5.55..6.05 ppm) exclusion zones have 0 buckets
    let study_val: Value = serde_json::from_str(&fs::read_to_string(&study_json).unwrap()).unwrap();
    let buckets_arr = study_val["buckets"].as_array().unwrap();
    assert!(buckets_arr.len() >= 15);
    for b in buckets_arr {
        let c = b["center_ppm"].as_f64().unwrap();
        assert!(
            !(4.65..=5.05).contains(&c),
            "water region bucket leaked: {c}"
        );
        assert!(
            !(5.55..=6.05).contains(&c),
            "urea region bucket leaked: {c}"
        );
    }

    // Also test `--manifest` ingestion of 2-column ASCII spectra + `--method uniform`
    let demo_samples = generate_demo_biofluid_cohort(20261115);
    let spec_dir = tmp.path.join("ascii_cohort");
    fs::create_dir_all(&spec_dir).unwrap();
    let mut manifest_txt = String::from("sample_id\tgroup\tpath\n");
    for s in demo_samples.iter().take(6) {
        let sp_file = format!("{}.tsv", s.id);
        let mut body = format!("# obs_mhz={:.2}\nppm\tintensity\n", s.spectrum.axis.obs_mhz);
        for k in (0..s.spectrum.len()).step_by(2) {
            body.push_str(&format!(
                "{:.5}\t{:.6e}\n",
                s.spectrum.ppm_at(k),
                s.spectrum.real[k]
            ));
        }
        fs::write(spec_dir.join(&sp_file), body).unwrap();
        manifest_txt.push_str(&format!("{}\t{}\t{}\n", s.id, s.group, sp_file));
    }
    let manifest_path = spec_dir.join("manifest.tsv");
    fs::write(&manifest_path, manifest_txt).unwrap();
    let unif_json = tmp.path.join("uniform_bin.json");
    assert!(Command::new(nmrassign_bin())
        .args([
            "metabolomics",
            "bin",
            "--manifest",
            manifest_path.to_str().unwrap(),
            "--method",
            "uniform",
            "--bin-width",
            "0.04",
            "--norm",
            "tsa",
            "-o",
            unif_json.to_str().unwrap(),
        ])
        .status()
        .unwrap()
        .success());

    // 2. `nmrassign metabolomics pca` (both from `study.json` and from `buckets.csv`)
    let pca_json = tmp.path.join("pca.json");
    let pca_csv_tsv = tmp.path.join("pca_from_csv.tsv");
    assert!(Command::new(nmrassign_bin())
        .args([
            "metabolomics",
            "pca",
            "--in",
            study_json.to_str().unwrap(),
            "--components",
            "5",
            "-o",
            pca_json.to_str().unwrap(),
        ])
        .status()
        .unwrap()
        .success());
    assert!(Command::new(nmrassign_bin())
        .args([
            "metabolomics",
            "pca",
            "--in",
            buckets_csv.to_str().unwrap(),
            "--components",
            "5",
            "-o",
            pca_csv_tsv.to_str().unwrap(),
        ])
        .status()
        .unwrap()
        .success());

    let pca_val: Value = serde_json::from_str(&fs::read_to_string(&pca_json).unwrap()).unwrap();
    let pc1_var = pca_val["explained_variance_ratio"][0].as_f64().unwrap();
    let cum_var = pca_val["cumulative_variance_ratio"]
        .as_array()
        .unwrap()
        .last()
        .unwrap()
        .as_f64()
        .unwrap();
    assert!(pc1_var > 0.15 && cum_var <= 1.000_001);
    assert!(pca_val["hotelling_t2_limit_95"].as_f64().unwrap() > 1.0);
    assert!(pca_val["q_residual_limit_95"].as_f64().unwrap() > 0.0);

    // 3. `nmrassign metabolomics oplsda`
    let opls_json = tmp.path.join("oplsda.json");
    let bm_csv = tmp.path.join("biomarkers.csv");
    let out_opls = Command::new(nmrassign_bin())
        .args([
            "metabolomics",
            "oplsda",
            "--in",
            study_json.to_str().unwrap(),
            "--ortho",
            "1",
            "--cv-folds",
            "7",
            "--permutations",
            "100",
            "--seed",
            "20261115",
            "--biomarkers-out",
            bm_csv.to_str().unwrap(),
            "-o",
            opls_json.to_str().unwrap(),
        ])
        .output()
        .expect("run metabolomics oplsda");
    assert!(
        out_opls.status.success(),
        "metabolomics oplsda failed: {}",
        String::from_utf8_lossy(&out_opls.stderr)
    );
    let opls_val: Value = serde_json::from_str(&fs::read_to_string(&opls_json).unwrap()).unwrap();
    let q2 = opls_val["q2"].as_f64().unwrap();
    let r2_y = opls_val["r2_y"].as_f64().unwrap();
    let perm_p = opls_val["permutation_p_value_q2"].as_f64().unwrap();
    assert!(q2 > 0.85, "expected OPLS-DA Q2 > 0.85, got {q2}");
    assert!(perm_p <= 0.02, "expected perm_p <= 0.02, got {perm_p}");
    assert_eq!(opls_val["is_validated"].as_bool(), Some(true));
    assert!(opls_val["overfit_warning"].is_null());

    let top_bms = opls_val["top_biomarkers"].as_array().unwrap();
    assert!(top_bms[0]["vip"].as_f64().unwrap() > 1.2);
    assert!(top_bms[0]["s_plot_corr"].as_f64().unwrap().abs() > 0.75);

    // 4. `nmrassign metabolomics stocsy --driver-ppm 1.330` (L-Lactate doublet -> 4.11 ppm quartet)
    let stocsy_json = tmp.path.join("stocsy_lactate.json");
    let mat2d_json = tmp.path.join("stocsy_2d.json");
    let out_stocsy = Command::new(nmrassign_bin())
        .args([
            "metabolomics",
            "stocsy",
            "--in",
            study_json.to_str().unwrap(),
            "--driver-ppm",
            "1.330",
            "--min-r2",
            "0.50",
            "--matrix-2d-out",
            mat2d_json.to_str().unwrap(),
            "-o",
            stocsy_json.to_str().unwrap(),
        ])
        .output()
        .expect("run metabolomics stocsy");
    assert!(
        out_stocsy.status.success(),
        "metabolomics stocsy failed: {}",
        String::from_utf8_lossy(&out_stocsy.stderr)
    );

    let st_val: Value = serde_json::from_str(&fs::read_to_string(&stocsy_json).unwrap()).unwrap();
    let cov_hits = st_val["covarying_buckets"].as_array().unwrap();
    let has_quartet = cov_hits.iter().any(|h| {
        let ppm = h["center_ppm"].as_f64().unwrap();
        let r = h["correlation"].as_f64().unwrap();
        (ppm - 4.110).abs() <= 0.05 && r > 0.85
    });
    assert!(
        has_quartet,
        "expected STOCSY from 1.330 ppm driver to recover L-Lactate 4.110 ppm quartet: {cov_hits:?}"
    );
    let top_match_id = st_val["metabolite_matches"][0]["bmrb_id"].as_str().unwrap();
    assert_eq!(top_match_id, "bmse000036");

    let mat2d_val: Value = serde_json::from_str(&fs::read_to_string(&mat2d_json).unwrap()).unwrap();
    let p_dim = mat2d_val["bucket_centers_ppm"].as_array().unwrap().len();
    assert_eq!(p_dim, buckets_arr.len());
    let diag0 = mat2d_val["correlation_matrix"][0][0].as_f64().unwrap();
    assert!((diag0 - 1.0).abs() < 1e-9);

    println!(
        "W2_METRIC_JSON: {}",
        serde_json::json!({
            "kind": "metabolomics_e2e_cli",
            "n_samples": 24,
            "n_buckets": buckets_arr.len(),
            "pca_pc1_var": pc1_var,
            "opls_r2_y": r2_y,
            "opls_q2": q2,
            "opls_perm_p": perm_p,
            "stocsy_top_bmrb_id": top_match_id,
            "passed": true,
        })
    );
}

#[test]
fn test_metabolomics_oplsda_null_label_anti_overfitting_guard() {
    let tmp = ScopedTempDir::new("metabolomics_null");
    let null_json = tmp.path.join("oplsda_null.json");

    let out = Command::new(nmrassign_bin())
        .args([
            "metabolomics",
            "oplsda",
            "--demo",
            "--seed",
            "20261115",
            "--permute-cohort-labels",
            "--permutations",
            "60",
            "-o",
            null_json.to_str().unwrap(),
        ])
        .output()
        .expect("run null-label oplsda");
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("OVERFIT WARNING"));

    let val: Value = serde_json::from_str(&fs::read_to_string(&null_json).unwrap()).unwrap();
    let q2_null = val["q2"].as_f64().unwrap();
    let perm_p_null = val["permutation_p_value_q2"].as_f64().unwrap();
    assert_eq!(val["is_validated"].as_bool(), Some(false));
    assert!(val["overfit_warning"].is_string());
    assert!(
        q2_null < 0.40 || perm_p_null > 0.05,
        "null-label OPLS-DA unexpectedly validated: Q2={q2_null}, p={perm_p_null}"
    );

    println!(
        "W2_METRIC_JSON: {}",
        serde_json::json!({
            "kind": "metabolomics_null_guard_cli",
            "null_q2": q2_null,
            "null_perm_p": perm_p_null,
            "is_validated": false,
            "passed": true,
        })
    );
}

#[test]
fn test_adversarial_error_paths_dosy_and_metabolomics_cli() {
    let tmp = ScopedTempDir::new("adversarial_w2_cli");

    let expect_cli_err = |args: &[&str]| {
        let out = Command::new(nmrassign_bin())
            .args(args)
            .output()
            .expect("spawn nmrassign");
        assert!(
            !out.status.success(),
            "expected CLI failure for args {args:?}, but exited 0"
        );
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            !stderr.trim().is_empty(),
            "expected diagnostic stderr for {args:?}"
        );
        assert!(
            !stderr.contains("panicked at"),
            "CLI panicked on {args:?}: {stderr}"
        );
    };

    // 1. DOSY adversarial inputs: NaN intensity TSV, < 3 b-steps, unphysical Delta <= delta/3, NaN shape, bad MW
    let nan_dosy_tsv = tmp.path.join("nan_dosy.tsv");
    fs::write(
        &nan_dosy_tsv,
        "b_s_m2\tpk_7.20\n1.0e8\t100.0\n5.0e8\tNaN\n1.0e9\t40.0\n2.0e9\t15.0\n",
    )
    .unwrap();
    let short_dosy_tsv = tmp.path.join("short_dosy.tsv");
    fs::write(
        &short_dosy_tsv,
        "b_s_m2\tpk_7.20\n1.0e8\t100.0\n5.0e8\t60.0\n",
    )
    .unwrap();
    let bad_json = tmp.path.join("corrupt.json");
    fs::write(&bad_json, "{corrupt_json").unwrap();

    expect_cli_err(&["dosy", "fit", "--in", nan_dosy_tsv.to_str().unwrap()]);
    expect_cli_err(&["dosy", "ilt", "--in", nan_dosy_tsv.to_str().unwrap()]);
    expect_cli_err(&["dosy", "fit", "--in", short_dosy_tsv.to_str().unwrap()]);
    expect_cli_err(&["dosy", "fit", "--in", bad_json.to_str().unwrap()]);
    expect_cli_err(&[
        "dosy",
        "fit",
        "--in",
        nan_dosy_tsv.to_str().unwrap(),
        "--shape",
        "NaN",
    ]);
    expect_cli_err(&["dosy", "mw", "--d", "-1.0e-10"]);
    expect_cli_err(&["dosy", "mw", "--d", "NaN"]);

    // 2. Metabolomics adversarial inputs: NaN bucket CSV, single-class OPLS-DA, min-ppm >= max-ppm, missing driver
    let nan_buckets = tmp.path.join("nan_buckets.csv");
    fs::write(
        &nan_buckets,
        "sample_id,group,dilution_factor,B0_1.330ppm,B1_4.110ppm\nS1,Control,1.0,10.0,NaN\nS2,Case,1.0,20.0,6.5\n",
    )
    .unwrap();
    let one_group_buckets = tmp.path.join("one_group.csv");
    fs::write(
        &one_group_buckets,
        "sample_id,group,dilution_factor,B0_1.330ppm,B1_4.110ppm\nS1,Control,1.0,10.0,3.2\nS2,Control,1.0,11.0,3.5\nS3,Control,1.0,9.5,3.1\n",
    )
    .unwrap();

    expect_cli_err(&["metabolomics", "pca", "--in", nan_buckets.to_str().unwrap()]);
    expect_cli_err(&[
        "metabolomics",
        "oplsda",
        "--in",
        one_group_buckets.to_str().unwrap(),
    ]);
    expect_cli_err(&[
        "metabolomics",
        "bin",
        "--demo",
        "--min-ppm",
        "9.0",
        "--max-ppm",
        "1.0",
    ]);
    expect_cli_err(&["metabolomics", "stocsy", "--demo"]);
    expect_cli_err(&["metabolomics", "pca", "--in", bad_json.to_str().unwrap()]);
}
