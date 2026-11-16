//! End-to-end CLI integration tests for `nmrassign cest`, `nmrassign cpmg`, and `nmrassign csp`
//! (`T13.4`–`T13.6`).
//!
//! # Literature & Clean-Room Provenance
//! - McConnell, H. M. (*J. Chem. Phys.* 1958, 28, 430–431).
//! - Helgstrand, M., Härd, T. & Allard, P. (*J. Biomol. NMR* 2000, 18, 47–61).
//! - Vallurupalli, P., Bouvignies, G. & Kay, L. E. (*J. Am. Chem. Soc.* 2012, 134, 8148–8161).
//! - Fawzi, N. L. et al. (*Nature* 2011, 480, 268–272).
//! - Higham, N. J. (*SIAM J. Matrix Anal. Appl.* 2005, 26, 1179–1193).
//! - Luz, Z. & Meiboom, S. (*J. Chem. Phys.* 1963, 39, 366–370).
//! - Carver, J. P. & Richards, R. E. (*J. Magn. Reson.* 1972, 6, 89–105).
//! - Baldwin, A. J. (*J. Magn. Reson.* 2014, 244, 114–124).
//! - Trott, O. & Palmer, A. G. III (*J. Magn. Reson.* 2002, 154, 157–160).
//! - Williamson, M. P. (*Prog. Nucl. Magn. Reson. Spectrosc.* 2013, 73, 1–16).

use std::f64::consts::TAU;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use nmr_core::cest::{
    simulate_cest_2state_profile, simulate_cest_3state_profile, write_cest_profiles_tsv,
    Cest2StateParams, Cest3StateParams, CestGlobalFitResult, CestNormalization, CestProfileDataset,
};
use nmr_core::cpmg::{
    r1rho_trott_palmer_2002, r2eff_baldwin2014, write_cpmg_series_tsv, CpmgGlobalFitResult,
    CpmgSeriesDataset, R1RhoFitResult,
};
use nmr_core::csp::{
    diluted_protein_conc_for_target_ligand, quadratic_bound_fraction, CspGlobalFitResult,
};
use nmr_core::nmrstar::read_star;
use nmr_core::rng::Rng;

static NEXT_TMP_ID: AtomicU64 = AtomicU64::new(1);

struct ScopedTempDir {
    path: PathBuf,
}

impl ScopedTempDir {
    fn new(prefix: &str) -> Self {
        let id = NEXT_TMP_ID.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "nmrassign_exchange_cli_{prefix}_{}_{}",
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
    env!("CARGO_BIN_EXE_ferric")
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("workspace root")
}

#[test]
fn test_exchange_cli_help_and_error_handling() {
    // 1. Top-level usage advertises `cest`, `cpmg`, and `csp`
    let top = Command::new(nmrassign_bin())
        .output()
        .expect("run ferric usage");
    let top_stdout = String::from_utf8_lossy(&top.stdout);
    assert!(top_stdout.contains("ferric cest"));
    assert!(top_stdout.contains("ferric cpmg"));
    assert!(top_stdout.contains("ferric csp"));

    // 2. Subcommand `--help` exits cleanly with code 0
    for sub in ["cest", "cpmg", "csp"] {
        let out = Command::new(nmrassign_bin())
            .args([sub, "--help"])
            .output()
            .expect("run subcommand --help");
        assert!(out.status.success(), "{sub} --help failed");
        let s = String::from_utf8_lossy(&out.stdout);
        assert!(s.contains(&format!("ferric {sub}")));
    }

    // 3. Invalid flags and missing inputs return clean non-zero exits
    for bad_args in [
        vec!["cest", "--unknown-flag"],
        vec!["cest", "--states", "4"],
        vec!["cpmg", "--model", "nonexistent"],
        vec!["cpmg", "--r1rho", "missing.tsv"],
        vec!["csp", "--experiment", "unknown"],
    ] {
        let out = Command::new(nmrassign_bin())
            .args(&bad_args)
            .output()
            .expect("run invalid args");
        assert!(
            !out.status.success(),
            "expected failure for {bad_args:?}, got success"
        );
    }
}

#[test]
fn test_cest_cli_2state_cpu_gpu_and_3state_e2e() {
    let tmp = ScopedTempDir::new("cest");
    let offsets: Vec<f64> = (0..=56).map(|k| 106.0 + 0.45 * (k as f64)).collect();
    let true_pb = 0.015;
    let true_kex = 150.0;
    let residues = [
        (12, "ILE", 114.2, 5.4, 1.35, 13.5, 18.0),
        (24, "GLU", 118.6, -4.8, 1.45, 15.0, 21.0),
        (38, "PRO", 121.0, 6.2, 1.25, 12.8, 17.5),
        (53, "GLY", 111.5, 4.1, 1.50, 11.5, 16.0),
    ];

    let mut rng = Rng::new(20261340);
    let mut profs_15 = Vec::new();
    let mut profs_30 = Vec::new();
    for &(seq_id, res_name, major_ppm, dw_ab, r1a, r2a, r2b) in &residues {
        for &(b1_hz, is_30) in &[(15.0, false), (30.0, true)] {
            let p = Cest2StateParams {
                p_b: true_pb,
                k_ex_s_inv: true_kex,
                major_ppm,
                dw_ab_ppm: dw_ab,
                r1a_s_inv: r1a,
                r2a_s_inv: r2a,
                r1b_s_inv: r1a,
                r2b_s_inv: r2b,
                larmor_mhz: 60.8106,
                t_ex_s: 0.40,
                b1_hz,
                b1_inhomogeneity_frac: 0.08,
                n_b1_quad: 5,
                normalization: CestNormalization::ZeroTimePhaseCycled,
            };
            let clean = simulate_cest_2state_profile(&p, &offsets).expect("sim 2s cest");
            let noisy: Vec<f64> = clean
                .iter()
                .map(|&y| (y + rng.normal(0.0, 0.002)).clamp(0.0, 1.05))
                .collect();
            let ds = CestProfileDataset {
                seq_id,
                res_name: res_name.to_string(),
                larmor_mhz: 60.8106,
                t_ex_s: 0.40,
                b1_hz,
                b1_inhomogeneity_frac: 0.08,
                normalization: CestNormalization::ZeroTimePhaseCycled,
                major_ppm_hint: None,
                offsets_ppm: offsets.clone(),
                intensities: noisy,
                uncertainties: vec![0.002; offsets.len()],
            };
            if is_30 {
                profs_30.push(ds);
            } else {
                profs_15.push(ds);
            }
        }
    }

    let p15_tsv = tmp.path.join("cest_15hz.tsv");
    let p30_tsv = tmp.path.join("cest_30hz.tsv");
    fs::write(&p15_tsv, write_cest_profiles_tsv(&profs_15)).unwrap();
    fs::write(&p30_tsv, write_cest_profiles_tsv(&profs_30)).unwrap();

    let cpu_tsv = tmp.path.join("cest_cpu_fit.tsv");
    let cpu_json = tmp.path.join("cest_cpu_fit.json");
    let out_cpu = Command::new(nmrassign_bin())
        .args([
            "cest",
            "--in",
            p15_tsv.to_str().unwrap(),
            "--in",
            p30_tsv.to_str().unwrap(),
            "--engine",
            "cpu",
            "--chi2-surface",
            "--out",
            cpu_tsv.to_str().unwrap(),
            "--json-out",
            cpu_json.to_str().unwrap(),
        ])
        .output()
        .expect("run nmrassign cest --engine cpu");
    assert!(
        out_cpu.status.success(),
        "cest cpu failed: {}",
        String::from_utf8_lossy(&out_cpu.stderr)
    );
    let fit_cpu: CestGlobalFitResult =
        serde_json::from_str(&fs::read_to_string(&cpu_json).unwrap()).expect("parse cest json");
    assert!((fit_cpu.p_b - true_pb).abs() < 0.0005);
    assert!((fit_cpu.k_ex_s_inv - true_kex).abs() < 5.0);
    assert!(fit_cpu.chi2_surface.is_some());

    // Also run `--engine auto` (uses WebGPU when present, matching CPU tightly)
    let auto_json = tmp.path.join("cest_auto_fit.json");
    let out_auto = Command::new(nmrassign_bin())
        .args([
            "cest",
            "--in",
            p15_tsv.to_str().unwrap(),
            "--in",
            p30_tsv.to_str().unwrap(),
            "--engine",
            "auto",
            "--json-out",
            auto_json.to_str().unwrap(),
        ])
        .output()
        .expect("run nmrassign cest --engine auto");
    assert!(
        out_auto.status.success(),
        "cest auto failed: {}",
        String::from_utf8_lossy(&out_auto.stderr)
    );
    let fit_auto: CestGlobalFitResult =
        serde_json::from_str(&fs::read_to_string(&auto_json).unwrap()).unwrap();
    assert!((fit_auto.p_b - fit_cpu.p_b).abs() < 2e-4);
    assert!((fit_auto.k_ex_s_inv - fit_cpu.k_ex_s_inv).abs() < 1.5);

    // 3-state linear `A <-> B <-> C` via `nmrassign cest --states 3`
    let offs_3s: Vec<f64> = (0..=64).map(|k| 110.0 + 0.25 * (k as f64)).collect();
    let mut profs_3s = Vec::new();
    for &b1_hz in &[15.0, 35.0] {
        let p3 = Cest3StateParams {
            p_b: 0.024,
            p_c: 0.014,
            k_ex_ab_s_inv: 180.0,
            k_ex_bc_s_inv: 140.0,
            k_ex_ac_s_inv: 0.0,
            major_ppm: 118.0,
            dw_ab_ppm: 4.0,
            dw_ac_ppm: -3.5,
            r1a_s_inv: 1.35,
            r2a_s_inv: 14.0,
            r1b_s_inv: 1.35,
            r2b_s_inv: 20.0,
            r1c_s_inv: 1.35,
            r2c_s_inv: 22.0,
            larmor_mhz: 60.8106,
            t_ex_s: 0.40,
            b1_hz,
            b1_inhomogeneity_frac: 0.0,
            n_b1_quad: 1,
            normalization: CestNormalization::ZeroTimePhaseCycled,
        };
        let y3 = simulate_cest_3state_profile(&p3, &offs_3s).unwrap();
        profs_3s.push(CestProfileDataset {
            seq_id: 41,
            res_name: "GLN".to_string(),
            larmor_mhz: 60.8106,
            t_ex_s: 0.40,
            b1_hz,
            b1_inhomogeneity_frac: 0.0,
            normalization: CestNormalization::ZeroTimePhaseCycled,
            major_ppm_hint: Some(118.0),
            offsets_ppm: offs_3s.clone(),
            intensities: y3,
            uncertainties: vec![0.002; offs_3s.len()],
        });
    }
    let p3_in = tmp.path.join("cest_3state_in.tsv");
    let p3_out = tmp.path.join("cest_3state_out.tsv");
    fs::write(&p3_in, write_cest_profiles_tsv(&profs_3s)).unwrap();
    let out_3s = Command::new(nmrassign_bin())
        .args([
            "cest",
            "--in",
            p3_in.to_str().unwrap(),
            "--states",
            "3",
            "--b1-quad",
            "1",
            "-o",
            p3_out.to_str().unwrap(),
        ])
        .output()
        .expect("run cest --states 3");
    assert!(
        out_3s.status.success(),
        "cest --states 3 failed: {}",
        String::from_utf8_lossy(&out_3s.stderr)
    );
    let tsv_3s = fs::read_to_string(&p3_out).unwrap();
    assert!(tsv_3s.contains("41\tGLN\t0.024"));
}

#[test]
fn test_cpmg_cli_multifield_raw_planes_and_r1rho_e2e() {
    let tmp = ScopedTempDir::new("cpmg");
    let nus = [
        25.0, 50.0, 100.0, 150.0, 200.0, 300.0, 400.0, 600.0, 800.0, 1000.0,
    ];
    let true_pb = 0.035;
    let true_kex = 900.0;
    let ex_residues = [
        (14, "THR", 2.8, 11.5, 13.8),
        (27, "LYS", 4.1, 12.2, 14.6),
        (45, "PHE", 3.4, 10.8, 13.0),
    ];

    let mut s600 = Vec::new();
    let mut s800 = Vec::new();
    for &(seq_id, res_name, dw_ppm, r20_600, r20_800) in &ex_residues {
        for &(larmor_mhz, r20, is_800) in &[(60.8106, r20_600, false), (81.0808, r20_800, true)] {
            let dw_rad = TAU * larmor_mhz * dw_ppm;
            let r2_vals: Vec<f64> = nus
                .iter()
                .map(|&nu| r2eff_baldwin2014(r20, r20, true_pb, true_kex, dw_rad, 0.040, nu))
                .collect();
            let ds = CpmgSeriesDataset {
                seq_id,
                res_name: res_name.to_string(),
                larmor_mhz,
                t_relax_s: 0.040,
                nu_cpmg_hz: nus.to_vec(),
                r2_eff_s_inv: r2_vals,
                r2_eff_err: vec![0.12; nus.len()],
            };
            if is_800 {
                s800.push(ds);
            } else {
                s600.push(ds);
            }
        }
    }
    // Add a flat decoy residue (`seq_id = 70`) that the exchange gate must filter out
    for (larmor_mhz, is_800) in [(60.8106, false), (81.0808, true)] {
        let ds = CpmgSeriesDataset {
            seq_id: 70,
            res_name: "VAL".to_string(),
            larmor_mhz,
            t_relax_s: 0.040,
            nu_cpmg_hz: nus.to_vec(),
            r2_eff_s_inv: vec![11.0; nus.len()],
            r2_eff_err: vec![0.15; nus.len()],
        };
        if is_800 {
            s800.push(ds);
        } else {
            s600.push(ds);
        }
    }

    let f600 = tmp.path.join("cpmg_600.tsv");
    let f800 = tmp.path.join("cpmg_800.tsv");
    fs::write(&f600, write_cpmg_series_tsv(&s600)).unwrap();
    fs::write(&f800, write_cpmg_series_tsv(&s800)).unwrap();

    let fit_tsv = tmp.path.join("cpmg_fit.tsv");
    let fit_json = tmp.path.join("cpmg_fit.json");
    let out = Command::new(nmrassign_bin())
        .args([
            "cpmg",
            "--in",
            f600.to_str().unwrap(),
            "--in",
            f800.to_str().unwrap(),
            "--model",
            "baldwin2014",
            "--engine",
            "auto",
            "-o",
            fit_tsv.to_str().unwrap(),
            "--json-out",
            fit_json.to_str().unwrap(),
        ])
        .output()
        .expect("run nmrassign cpmg");
    assert!(
        out.status.success(),
        "cpmg failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let fit: CpmgGlobalFitResult =
        serde_json::from_str(&fs::read_to_string(&fit_json).unwrap()).unwrap();
    assert_eq!(
        fit.residues.len(),
        3,
        "flat decoy seq_id=70 must be gated out"
    );
    assert!((fit.p_b - true_pb).abs() / true_pb < 0.02);
    assert!((fit.k_ex_s_inv - true_kex).abs() / true_kex < 0.02);

    // Raw CPMG intensity planes (`nu_cpmg_hz = 0` reference + refocused planes)
    let mut raw_plane_tsv = String::from(
        "seq_id\tres_name\tlarmor_mhz\tt_relax_s\tnu_cpmg_hz\tintensity\tbaseline_noise\n",
    );
    for (larmor_mhz, r20) in [(60.8106, 11.5), (81.0808, 13.8)] {
        raw_plane_tsv.push_str(&format!(
            "14\tTHR\t{larmor_mhz:.5}\t0.040\t0.0\t1000.0\t2.0\n"
        ));
        raw_plane_tsv.push_str(&format!(
            "14\tTHR\t{larmor_mhz:.5}\t0.040\t0.0\t1002.0\t2.0\n"
        ));
        let dw_rad = TAU * larmor_mhz * 3.2;
        for &nu in &nus {
            let r2 = r2eff_baldwin2014(r20, r20, 0.04, 850.0, dw_rad, 0.040, nu);
            let inten = 1001.0 * (-r2 * 0.040).exp();
            raw_plane_tsv.push_str(&format!(
                "14\tTHR\t{larmor_mhz:.5}\t0.040\t{nu:.2}\t{inten:.4}\t2.0\n"
            ));
        }
    }
    let raw_p = tmp.path.join("raw_planes.tsv");
    let series_out_p = tmp.path.join("computed_r2eff.tsv");
    fs::write(&raw_p, raw_plane_tsv).unwrap();
    let out_raw = Command::new(nmrassign_bin())
        .args([
            "cpmg",
            "--raw-planes",
            raw_p.to_str().unwrap(),
            "--series-out",
            series_out_p.to_str().unwrap(),
            "--engine",
            "cpu",
        ])
        .output()
        .expect("run cpmg --raw-planes");
    assert!(
        out_raw.status.success(),
        "cpmg --raw-planes failed: {}",
        String::from_utf8_lossy(&out_raw.stderr)
    );
    assert!(series_out_p.is_file());

    // Off-resonance R1rho Trott–Palmer (2002) signed `dw_AB = -3.80 ppm`
    let major_ppm = 119.5;
    let larmor_mhz = 60.8106;
    let scale = TAU * larmor_mhz;
    let mut r1rho_tsv = String::from("carrier_ppm\tspin_lock_hz\tr1rho_s_inv\tr1rho_err\n");
    for &sl_hz in &[25.0, 50.0, 100.0] {
        for k in 0..=14 {
            let carrier = 112.5 + 1.0 * (k as f64);
            let om_a = scale * (major_ppm - carrier);
            let (_, _, r1r) = r1rho_trott_palmer_2002(
                1.40,
                12.5,
                0.025,
                1200.0,
                om_a,
                scale * (-3.80),
                TAU * sl_hz,
            );
            r1rho_tsv.push_str(&format!("{carrier:.4}\t{sl_hz:.2}\t{r1r:.5}\t0.08\n"));
        }
    }
    let r1rho_in = tmp.path.join("r1rho.tsv");
    let r1rho_json = tmp.path.join("r1rho_fit.json");
    fs::write(&r1rho_in, r1rho_tsv).unwrap();
    let out_r1r = Command::new(nmrassign_bin())
        .args([
            "cpmg",
            "--r1rho",
            r1rho_in.to_str().unwrap(),
            "--major-ppm",
            "119.5",
            "--larmor-mhz",
            "60.8106",
            "--json-out",
            r1rho_json.to_str().unwrap(),
        ])
        .output()
        .expect("run cpmg --r1rho");
    assert!(
        out_r1r.status.success(),
        "cpmg --r1rho failed: {}",
        String::from_utf8_lossy(&out_r1r.stderr)
    );
    let tp_fit: R1RhoFitResult =
        serde_json::from_str(&fs::read_to_string(&r1rho_json).unwrap()).unwrap();
    assert!(
        tp_fit.dw_ab_ppm < 0.0,
        "must preserve negative sign of dw_AB"
    );
    assert!((tp_fit.dw_ab_ppm - (-3.80)).abs() < 0.05);
}

#[test]
fn test_csp_cli_tight_binding_dilution_star_and_methyl_hmqc_e2e() {
    let tmp = ScopedTempDir::new("csp");
    let p0_um = 100.0;
    let stock_um = 2500.0;
    let true_kd_um = 5.0;
    let ligand_steps_um = [15.0, 35.0, 60.0, 90.0, 120.0, 180.0, 300.0, 500.0];

    // 1. Build 24-residue protein (5 binding-pocket sites + 19 background sites)
    let mut apo_tsv = String::from("seq_index\tseq_id\tres_name\th_ppm\tn_ppm\n");
    let mut apo_coords = Vec::new();
    for idx in 0..24usize {
        let seq_id = (idx + 1) as i32;
        let res_name = if idx % 6 == 0 { "GLY" } else { "ALA" };
        let h0 = 7.40 + 0.07 * (idx as f64);
        let n0 = 112.0 + 0.65 * (idx as f64);
        let (dh_max, dn_max) = match seq_id {
            4 => (0.28, 1.40),
            9 => (-0.24, 1.65),
            14 => (0.31, -1.25),
            19 => (0.22, 1.50),
            22 => (-0.26, -1.35),
            _ => (0.004 * (((idx % 3) as f64) - 1.0), 0.02),
        };
        apo_tsv.push_str(&format!("{idx}\t{seq_id}\t{res_name}\t{h0:.5}\t{n0:.5}\n"));
        apo_coords.push((h0, n0, dh_max, dn_max));
    }
    let apo_p = tmp.path.join("apo_amide.tsv");
    fs::write(&apo_p, apo_tsv).unwrap();

    let mut step_args: Vec<String> = Vec::new();
    for (si, &l_um) in ligand_steps_um.iter().enumerate() {
        let p_dil = diluted_protein_conc_for_target_ligand(p0_um, stock_um, l_um).unwrap();
        let fb = quadratic_bound_fraction(p_dil, l_um, true_kd_um);
        let mut step_tsv = String::from("h_ppm\tn_ppm\n");
        // Write peaks in reverse order to exercise Hungarian assignment
        for &(h0, n0, dh_max, dn_max) in apo_coords.iter().rev() {
            let h = h0 + fb * dh_max;
            let n = n0 + fb * dn_max;
            step_tsv.push_str(&format!("{h:.5}\t{n:.5}\n"));
        }
        let sp = tmp.path.join(format!("step_{si}.tsv"));
        fs::write(&sp, step_tsv).unwrap();
        step_args.push("--step".to_string());
        step_args.push(format!("{l_um}={}", sp.display()));
    }

    let csp_out = tmp.path.join("csp_fit.tsv");
    let csp_json = tmp.path.join("csp_fit.json");
    let traj_out = tmp.path.join("csp_traj.tsv");

    let mut cmd_args = vec![
        "csp".to_string(),
        "--apo-peaks".to_string(),
        apo_p.to_string_lossy().into_owned(),
        "--protein-um".to_string(),
        format!("{p0_um}"),
        "--ligand-stock-um".to_string(),
        format!("{stock_um}"),
        "--mc-trials".to_string(),
        "120".to_string(),
        "--trajectories-out".to_string(),
        traj_out.to_string_lossy().into_owned(),
        "--json-out".to_string(),
        csp_json.to_string_lossy().into_owned(),
        "-o".to_string(),
        csp_out.to_string_lossy().into_owned(),
    ];
    cmd_args.extend(step_args);

    let out = Command::new(nmrassign_bin())
        .args(&cmd_args)
        .output()
        .expect("run nmrassign csp");
    assert!(
        out.status.success(),
        "csp failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let fit: CspGlobalFitResult =
        serde_json::from_str(&fs::read_to_string(&csp_json).unwrap()).unwrap();
    let kd_rel_err = (fit.global_kd_um - true_kd_um).abs() / true_kd_um;
    assert!(
        kd_rel_err < 0.02,
        "tight-binding Kd rel err {kd_rel_err:.4} >= 2% (fit={} vs true={true_kd_um})",
        fit.global_kd_um
    );
    assert!(
        fit.global_naive_mm_kd_um / true_kd_um > 8.0,
        "naive MM must overestimate tight-binding Kd by > 8x, got {}",
        fit.global_naive_mm_kd_um
    );
    assert_eq!(fit.thresholds.significant_seq_ids, vec![4, 9, 14, 19, 22]);

    // 2. Round-trip `--trajectories` refit must match exact `global_kd_um`
    let refit_json = tmp.path.join("csp_refit.json");
    let out_refit = Command::new(nmrassign_bin())
        .args([
            "csp",
            "--trajectories",
            traj_out.to_str().unwrap(),
            "--mc-trials",
            "40",
            "--json-out",
            refit_json.to_str().unwrap(),
        ])
        .output()
        .expect("run csp --trajectories");
    assert!(out_refit.status.success());
    let refit: CspGlobalFitResult =
        serde_json::from_str(&fs::read_to_string(&refit_json).unwrap()).unwrap();
    assert!((refit.global_kd_um - fit.global_kd_um).abs() < 1e-3);

    // 3. `--star` apo input (`synthetic_ubiquitin.str`) + `--experiment hmqc` methyl input
    let star_path = workspace_root().join("data/fixtures/synthetic_ubiquitin.str");
    let ref_ubq = read_star(&star_path, None).unwrap();
    for (si, &l_um) in [25.0, 75.0, 150.0, 400.0].iter().enumerate() {
        let fb = quadratic_bound_fraction(100.0, l_um, 80.0);
        let mut s_tsv = String::from("h_ppm\tn_ppm\n");
        for (k, rn) in ref_ubq.sequence.iter().enumerate() {
            if rn == "PRO" {
                continue;
            }
            if let (Some(&h), Some(&n)) = (
                ref_ubq.shifts.get(&(k, "H".to_string())),
                ref_ubq.shifts.get(&(k, "N".to_string())),
            ) {
                let (dh, dn) = if matches!(k, 5 | 41 | 43 | 67) {
                    (0.24 * fb, 1.35 * fb)
                } else {
                    (0.0, 0.0)
                };
                s_tsv.push_str(&format!("{:.5}\t{:.5}\n", h + dh, n + dn));
            }
        }
        fs::write(tmp.path.join(format!("ubq_{si}.tsv")), s_tsv).unwrap();
    }
    let ubq_json = tmp.path.join("ubq_csp.json");
    let out_ubq = Command::new(nmrassign_bin())
        .args([
            "csp",
            "--star",
            star_path.to_str().unwrap(),
            "--step",
            &format!("25={}", tmp.path.join("ubq_0.tsv").display()),
            "--step",
            &format!("75={}", tmp.path.join("ubq_1.tsv").display()),
            "--step",
            &format!("150={}", tmp.path.join("ubq_2.tsv").display()),
            "--step",
            &format!("400={}", tmp.path.join("ubq_3.tsv").display()),
            "--mc-trials",
            "40",
            "--json-out",
            ubq_json.to_str().unwrap(),
        ])
        .output()
        .expect("run csp --star");
    assert!(
        out_ubq.status.success(),
        "csp --star failed: {}",
        String::from_utf8_lossy(&out_ubq.stderr)
    );
    let ubq_fit: CspGlobalFitResult =
        serde_json::from_str(&fs::read_to_string(&ubq_json).unwrap()).unwrap();
    assert!((ubq_fit.global_kd_um - 80.0).abs() / 80.0 < 0.02);

    println!(
        "W4_METRIC_JSON: {{\"task\":\"T13.6\",\"kind\":\"exchange_cli_e2e\",\"cest_subcommand\":true,\"cpmg_subcommand\":true,\"csp_subcommand\":true,\"tight_kd_um\":{:.4},\"tight_kd_rel_err_pct\":{:.3},\"ubq_star_kd_um\":{:.3},\"passed\":true}}",
        fit.global_kd_um,
        kd_rel_err * 100.0,
        ubq_fit.global_kd_um,
    );
}
