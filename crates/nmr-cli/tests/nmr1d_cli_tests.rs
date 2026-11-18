//! End-to-end CLI integration tests for `ferric nmr1d process`, `ferric nmr1d multiplet`,
//! and `ferric nmr1d qnmr` (`T14.1`–`T14.2`).
//!
//! # Clean-Room Provenance & Literature Citations
//! - Bartholdi, E. & Ernst, R. R. (*J. Magn. Reson.* **1973**, 11, 9–19).
//! - Chen, L., Weng, Z., Goh, L. & Garland, M. (*J. Magn. Reson.* **2002**, 158, 164–168).
//! - Eilers, P. H. C. (*Anal. Chem.* **2003**, 75, 3631–3636);
//!   Cobas, J. C. et al. (*J. Magn. Reson.* **2006**, 183, 145–151).
//! - Barkhuijsen, H., de Beer, R. & van Ormondt, D. (*J. Magn. Reson.* **1987**, 73, 553–557).
//! - Fulmer, G. R. et al. (*Organometallics* **2010**, 29, 2176–2179).
//! - Thompson, P., Cox, D. E. & Hastings, J. B. (*J. Appl. Crystallogr.* **1987**, 20, 79–83).
//! - Hoye, T. R., Hanson, P. R. & Vyvyan, J. R. (*J. Org. Chem.* **1994**, 59, 4096–4103);
//!   Hoye, T. R. & Zhao, H. (*J. Org. Chem.* **2002**, 67, 4014–4016).
//! - Pauli, G. F. et al. (*J. Med. Chem.* **2014**, 57, 9220–9231);
//!   Malz, F. & Jancke, H. (*J. Pharm. Biomed. Anal.* **2005**, 38, 813–823).

use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::Value;

use nmr_core::multiplet::{
    parse_journal_nmr_report, synthesize_multiplet_peaks, synthesize_multiplet_spectrum,
    MultipletClass, QnmrAnalyteSignal, QnmrAnalyteSpec, QnmrPurityReport, QnmrStandardSpec,
};
use nmr_core::nmr1d::{synthesize_fid_1d, Experiment1DKind, Spectrum1D, SyntheticLine1D};
use nmr_core::octet_analyt::jcamp_spec1d::{
    write_jcamp_nmr, JcampCompression, Spectrum1d, SpectrumKind,
};
use nmr_core::octet_mol::{clean_2d, parse_smiles, write_mol_v2000};
use nmr_core::spectrum::AxisPpm;

static NEXT_TMP_ID: AtomicU64 = AtomicU64::new(1);

struct ScopedTempDir {
    path: PathBuf,
}

impl ScopedTempDir {
    fn new(prefix: &str) -> Self {
        let id = NEXT_TMP_ID.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "nmrassign_nmr1d_cli_{prefix}_{}_{}",
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

fn nmrassign_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_ferric"))
}

#[test]
fn cli_nmr1d_process_fid_hsvd_autophase_whittaker_solvent_voigt() {
    let tmp = ScopedTempDir::new("process_fid");

    // Planted systematic reference offset `-0.035 ppm`: CDCl3 residual appears at `7.225 ppm`
    // (true `7.260 ppm`), trace water in CDCl3 at `1.525 ppm` (true `1.560 ppm`), huge HDO spike
    // at `4.760 ppm` to be removed by HSVD, and an overlapping pair at `3.365` / `3.381 ppm`
    // (true `3.400` / `3.416 ppm` after calibration, area ratio `2.0 : 1.0`).
    let off = -0.035_f64;
    let phi0 = 36.0_f64;
    let mut lines = vec![
        SyntheticLine1D {
            ppm: 4.795 + off,
            amplitude: 250.0,
            fwhm_hz: 3.2,
            gauss_fwhm_hz: 0.0,
            phase_deg: phi0,
        },
        SyntheticLine1D {
            ppm: 7.260 + off,
            amplitude: 8.0,
            fwhm_hz: 1.1,
            gauss_fwhm_hz: 0.0,
            phase_deg: phi0,
        },
        SyntheticLine1D {
            ppm: 1.560 + off,
            amplitude: 3.5,
            fwhm_hz: 1.2,
            gauss_fwhm_hz: 0.0,
            phase_deg: phi0,
        },
        SyntheticLine1D {
            ppm: 3.400 + off,
            amplitude: 20.0,
            fwhm_hz: 1.4,
            gauss_fwhm_hz: 0.4,
            phase_deg: phi0,
        },
        SyntheticLine1D {
            ppm: 3.416 + off,
            amplitude: 10.0,
            fwhm_hz: 1.4,
            gauss_fwhm_hz: 0.4,
            phase_deg: phi0,
        },
    ];
    // Apply linear first-order phase `PHC1 = -28 deg` across the 12 ppm (6000 Hz at 500 MHz) window.
    let obs_mhz = 500.0;
    let sw_hz = 6000.0;
    let carrier_ppm = 4.50;
    let sw_ppm = sw_hz / obs_mhz;
    let high_ppm = carrier_ppm + 0.5 * sw_ppm;
    for ln in &mut lines {
        let u = (high_ppm - ln.ppm) / sw_ppm;
        ln.phase_deg += -28.0 * (u - 0.5);
    }

    let td = synthesize_fid_1d(
        "synth_1h_fid",
        Experiment1DKind::Proton1D,
        4096,
        sw_hz,
        obs_mhz,
        carrier_ppm,
        &lines,
    );
    let fid_json = tmp.path.join("input_fid.json");
    fs::write(&fid_json, serde_json::to_string(&td).unwrap()).unwrap();

    let out_tsv = tmp.path.join("processed.tsv");
    let peaks_tsv = tmp.path.join("voigt_peaks.tsv");
    let summary_json = tmp.path.join("process_summary.json");

    let output = Command::new(nmrassign_bin())
        .args([
            "nmr1d",
            "process",
            "--in",
            fid_json.to_str().unwrap(),
            "--out",
            out_tsv.to_str().unwrap(),
            "--hsvd",
            "4.60:4.90",
            "--apod",
            "em",
            "--lb",
            "0.2",
            "--zf",
            "2",
            "--auto-phase",
            "--baseline",
            "whittaker",
            "--lambda",
            "1e6",
            "--calibrate-solvent",
            "CDCl3",
            "--match-impurities",
            "--deconvolve",
            "--peaks-out",
            peaks_tsv.to_str().unwrap(),
            "--json-out",
            summary_json.to_str().unwrap(),
        ])
        .output()
        .expect("run nmrassign nmr1d process");

    assert!(
        output.status.success(),
        "nmr1d process failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(out_tsv.is_file());
    assert!(peaks_tsv.is_file());

    let summary: Value = serde_json::from_str(&fs::read_to_string(&summary_json).unwrap()).unwrap();
    let hsvd_sub = summary["hsvd_poles_subtracted"].as_u64().unwrap();
    assert!(hsvd_sub >= 1, "expected HSVD to subtract HDO solvent pole");

    let cal_delta = summary["solvent_calibration_delta_ppm"].as_f64().unwrap();
    assert!(
        (cal_delta - 0.035).abs() < 0.004,
        "expected Fulmer CDCl3 calibration shift ~+0.035 ppm, got {cal_delta:.5}"
    );

    let impurities = summary["impurity_matches"].as_array().unwrap();
    assert!(
        impurities.iter().any(|m| {
            let c = m["compound"].as_str().unwrap_or("").to_ascii_lowercase();
            c.contains("water") || c == "h2o"
        }),
        "expected trace Water/H2O impurity at 1.56 ppm in CDCl3, got {impurities:?}"
    );

    let vpeaks = summary["deconvolved_peaks"].as_array().unwrap();
    let pair: Vec<&Value> = vpeaks
        .iter()
        .filter(|p| {
            let c = p["center_ppm"].as_f64().unwrap_or(0.0);
            (3.37..=3.45).contains(&c)
        })
        .collect();
    assert_eq!(
        pair.len(),
        2,
        "expected 2 deconvolved Voigt peaks in 3.37..3.45 ppm, got {pair:?}"
    );
    let a_hi = pair[0]["area_ppm"].as_f64().unwrap();
    let a_lo = pair[1]["area_ppm"].as_f64().unwrap();
    // `pair` is sorted high-to-low ppm: `3.416 ppm` (`amplitude = 10`) then `3.400 ppm` (`amplitude = 20`).
    let ratio = a_lo / a_hi.max(1e-12);
    let ratio_rel_err_pct = 100.0 * (ratio - 2.0).abs() / 2.0;
    assert!(
        ratio_rel_err_pct < 2.5,
        "expected deconvolved area ratio ~2.00, got {ratio:.4} ({ratio_rel_err_pct:.2}% error)"
    );

    println!(
        "W1_METRIC_JSON: {{\"gate\":\"CLI.nmr1d_process_fid\",\"hsvd_poles_subtracted\":{hsvd_sub},\"cal_delta_err_ppm\":{:.6},\"voigt_ratio\":{ratio:.5},\"voigt_ratio_err_pct\":{ratio_rel_err_pct:.3}}}",
        (cal_delta - 0.035).abs()
    );
}

#[test]
fn cli_nmr1d_process_jcamp_and_bruker_pdata_ingest() {
    let tmp = ScopedTempDir::new("process_jcamp_bruker");
    let n = 2048usize;
    let axis = AxisPpm::from_bounds("1H", n, 9.0, 0.0, 400.0);
    let ppm_grid: Vec<f64> = (0..n).map(|i| axis.index_to_ppm(i as f64)).collect();
    let pks = synthesize_multiplet_peaks(7.35, 400.0, &[(1, 8.0, None)], 1.2);
    let mut y = synthesize_multiplet_spectrum(&ppm_grid, &pks, 400.0, 1.2, 0.1);
    // Add gentle quadratic baseline drift
    for (idx, val) in y.iter_mut().enumerate() {
        let u = (idx as f64) / ((n - 1) as f64);
        *val += 0.25 + 0.4 * (u - 0.5).powi(2);
    }

    // 1. Write JCAMP-DX `.jdx`
    let jspec = Spectrum1d {
        title: "jcamp_doublet".to_string(),
        kind: SpectrumKind::Nmr,
        nucleus: Some("1H".to_string()),
        mhz: Some(400.0),
        solvent: Some("CDCl3".to_string()),
        x_unit: "PPM".to_string(),
        y_unit: "ARBITRARY UNITS".to_string(),
        first_x: 9.0,
        last_x: 0.0,
        y: y.clone(),
        xs: None,
        sticks: false,
        peaks: None,
        integral: None,
    };
    let jdx_path = tmp.path.join("doublet.jdx");
    fs::write(&jdx_path, write_jcamp_nmr(&jspec, JcampCompression::Affn)).unwrap();

    let jdx_out = tmp.path.join("from_jdx.json");
    let res_jdx = Command::new(nmrassign_bin())
        .args([
            "nmr1d",
            "process",
            "--in",
            jdx_path.to_str().unwrap(),
            "--out",
            jdx_out.to_str().unwrap(),
            "--baseline",
            "bernstein",
            "--bernstein-degree",
            "4",
            "--deconvolve",
        ])
        .output()
        .expect("run nmr1d process on jdx");
    assert!(
        res_jdx.status.success(),
        "JCAMP process failed: {}",
        String::from_utf8_lossy(&res_jdx.stderr)
    );
    let out_spec: Spectrum1D =
        serde_json::from_str(&fs::read_to_string(&jdx_out).unwrap()).unwrap();
    assert_eq!(out_spec.len(), n);

    // 2. Write synthetic Bruker `pdata/1/{procs,1r}` directory (little-endian int32)
    let pdir = tmp.path.join("bruker_exp").join("pdata").join("1");
    fs::create_dir_all(&pdir).unwrap();
    let procs_txt = format!(
        "##TITLE= Parameter file\n##$SI= {n}\n##$OFFSET= 9.0\n##$SW_p= 3600.0\n##$SF= 400.0\n##$NC_proc= -6\n##$BYTORDP= 0\n##$DTYPP= 0\n##END=\n"
    );
    fs::write(pdir.join("procs"), procs_txt).unwrap();
    let mut bytes_1r = Vec::with_capacity(n * 4);
    for &val in &y {
        let scaled = (val * 1.0e5).round() as i32;
        bytes_1r.extend_from_slice(&scaled.to_le_bytes());
    }
    fs::write(pdir.join("1r"), bytes_1r).unwrap();

    let bruker_out = tmp.path.join("from_bruker.tsv");
    let bruker_json = tmp.path.join("from_bruker_summary.json");
    let res_brk = Command::new(nmrassign_bin())
        .args([
            "nmr1d",
            "process",
            "--in",
            tmp.path.join("bruker_exp").to_str().unwrap(),
            "--out",
            bruker_out.to_str().unwrap(),
            "--baseline",
            "bernstein",
            "--deconvolve",
            "--json-out",
            bruker_json.to_str().unwrap(),
        ])
        .output()
        .expect("run nmr1d process on bruker pdata");
    assert!(
        res_brk.status.success(),
        "Bruker pdata process failed: {}",
        String::from_utf8_lossy(&res_brk.stderr)
    );
    let bsum: Value = serde_json::from_str(&fs::read_to_string(&bruker_json).unwrap()).unwrap();
    let b_vpeaks = bsum["deconvolved_peaks"].as_array().unwrap().len();
    assert_eq!(
        b_vpeaks, 2,
        "expected 2 deconvolved doublet lines from Bruker 1r"
    );

    println!(
        "W1_METRIC_JSON: {{\"gate\":\"CLI.nmr1d_process_jcamp_bruker\",\"points\":{n},\"deconvolved_peaks\":{b_vpeaks}}}"
    );
}

#[test]
fn cli_nmr1d_multiplet_hoye_tree_journal_styles_and_octet_atom_links() {
    let tmp = ScopedTempDir::new("multiplet_ethyl_acetate");
    let mhz = 500.0;
    let n = 16384usize;
    let axis = AxisPpm::from_bounds("1H", n, 8.5, 0.0, mhz);
    let ppm_grid: Vec<f64> = (0..n).map(|i| axis.index_to_ppm(i as f64)).collect();

    // Ethyl acetate (`CCOC(=O)C`, 8H) + an aromatic `dd` internal test peak (`1H` at 7.42 ppm, J = 8.4, 1.8 Hz)
    // First test pure ethyl acetate with Octet `.mol` & SMILES atom linking!
    let mut y_ea = vec![0.0_f64; n];
    let q_pks = synthesize_multiplet_peaks(4.12, mhz, &[(3, 7.15, None)], 0.9);
    let s_pks = synthesize_multiplet_peaks(2.05, mhz, &[], 0.9);
    let t_pks = synthesize_multiplet_peaks(1.26, mhz, &[(2, 7.15, None)], 0.9);
    for (scale_h, pks) in [(2.0, q_pks), (3.0, s_pks), (3.0, t_pks)] {
        let trace = synthesize_multiplet_spectrum(&ppm_grid, &pks, mhz, 0.9, 0.0);
        for (dst, src) in y_ea.iter_mut().zip(trace.iter()) {
            *dst += scale_h * src;
        }
    }
    let mut spec_ea = Spectrum1D::new(
        "ethyl_acetate_1h",
        Experiment1DKind::Proton1D,
        axis.clone(),
        y_ea,
        None,
    );
    spec_ea.solvent = Some("CDCl3".to_string());
    let spec_ea_json = tmp.path.join("ethyl_acetate.json");
    fs::write(&spec_ea_json, serde_json::to_string(&spec_ea).unwrap()).unwrap();

    // Also write an MDL V2000 `.mol` file for ethyl acetate via `nmr_core::octet_mol`
    let mut mol = parse_smiles("[CH3:1][CH2:2]O[C:3](=O)[CH3:4]").unwrap();
    clean_2d(&mut mol);
    let mol_path = tmp.path.join("ethyl_acetate.mol");
    fs::write(&mol_path, write_mol_v2000(&mol).unwrap()).unwrap();

    let mult_tsv = tmp.path.join("ea_multiplets.tsv");
    let links_tsv = tmp.path.join("ea_links.tsv");
    let mult_json = tmp.path.join("ea_multiplets.json");

    let out_ea = Command::new(nmrassign_bin())
        .args([
            "nmr1d",
            "multiplet",
            "--in",
            spec_ea_json.to_str().unwrap(),
            "--regions",
            "4.25:3.98:OCH2,2.15:1.95:COCH3,1.38:1.14:CH2CH3",
            "--smiles",
            "[CH3:1][CH2:2]O[C:3](=O)[CH3:4]",
            "--style",
            "acs",
            "--encoding",
            "unicode",
            "--out",
            mult_tsv.to_str().unwrap(),
            "--links-out",
            links_tsv.to_str().unwrap(),
            "--json-out",
            mult_json.to_str().unwrap(),
        ])
        .output()
        .expect("run nmr1d multiplet on ethyl acetate");

    assert!(
        out_ea.status.success(),
        "nmr1d multiplet failed:\n{}",
        String::from_utf8_lossy(&out_ea.stderr)
    );
    assert!(mult_tsv.is_file());
    assert!(links_tsv.is_file());

    let rep: Value = serde_json::from_str(&fs::read_to_string(&mult_json).unwrap()).unwrap();
    let mults = rep["multiplets"].as_array().unwrap();
    assert_eq!(mults.len(), 3);
    assert_eq!(mults[0]["label"].as_str().unwrap(), "q");
    assert_eq!(mults[1]["label"].as_str().unwrap(), "s");
    assert_eq!(mults[2]["label"].as_str().unwrap(), "t");

    let q_j = mults[0]["couplings"][0]["j_hz"].as_f64().unwrap();
    let t_j = mults[2]["couplings"][0]["j_hz"].as_f64().unwrap();
    assert!((q_j - 7.15).abs() < 0.08, "quartet J={q_j}");
    assert!((t_j - 7.15).abs() < 0.08, "triplet J={t_j}");

    // Verify all 4 journal styles (`acs`, `joc`, `rsc`, `nature`) round-trip through `parse_journal_nmr_report`
    for st in ["acs", "joc", "rsc", "nature"] {
        let line = rep["formatted_styles"][st].as_str().unwrap();
        let parsed = parse_journal_nmr_report(line)
            .unwrap_or_else(|e| panic!("failed to round-trip `{st}` journal line {line:?}: {e}"));
        assert_eq!(parsed.signals.len(), 3);
        assert_eq!(
            parsed.signals[0].multiplicity,
            Some(MultipletClass::Quartet)
        );
        assert_eq!(
            parsed.signals[1].multiplicity,
            Some(MultipletClass::Singlet)
        );
        assert_eq!(
            parsed.signals[2].multiplicity,
            Some(MultipletClass::Triplet)
        );
    }

    // Verify Octet 0-based <-> SDF 1-based <-> SMILES atom-map synchronization
    let links = rep["assignment_table"]["links"].as_array().unwrap();
    assert_eq!(
        links.len(),
        3,
        "expected all 3 ethyl acetate multiplets assigned to Octet atoms"
    );
    for lk in links {
        let oct_idx = lk["octet_atom_indices"][0].as_u64().unwrap();
        let sdf_num = lk["sdf_atom_numbers"][0].as_u64().unwrap();
        assert_eq!(sdf_num, oct_idx + 1);
        assert!(
            !lk["smiles_atom_maps"].as_array().unwrap().is_empty(),
            "expected SMILES atom map preserved on assigned atom"
        );
    }

    // Also verify `.mol` file input + complex `dd`, `ddd`, `dt`, `td` multiplet tree classification!
    let mut y_trees = vec![0.0_f64; n];
    let tree_defs = [
        (7.60, vec![(1usize, 8.4_f64, None), (1, 1.8, None)], "dd"),
        (
            6.90,
            vec![(1, 11.2, None), (1, 6.8, None), (1, 2.1, None)],
            "ddd",
        ),
        (5.80, vec![(1, 15.6, None), (2, 6.4, None)], "dt"),
        (3.20, vec![(2, 7.6, None), (1, 2.4, None)], "td"),
    ];
    for (c_ppm, ref cpls, _) in &tree_defs {
        let pks = synthesize_multiplet_peaks(*c_ppm, mhz, cpls, 0.8);
        let tr = synthesize_multiplet_spectrum(&ppm_grid, &pks, mhz, 0.8, 0.0);
        for (d, s) in y_trees.iter_mut().zip(tr.iter()) {
            *d += s;
        }
    }
    let spec_trees = Spectrum1D::new(
        "hoye_trees_1h",
        Experiment1DKind::Proton1D,
        axis,
        y_trees,
        None,
    );
    let trees_in = tmp.path.join("hoye_trees.json");
    let trees_json = tmp.path.join("hoye_trees_out.json");
    let mol_links_json = tmp.path.join("mol_links.json");
    fs::write(&trees_in, serde_json::to_string(&spec_trees).unwrap()).unwrap();

    let out_trees = Command::new(nmrassign_bin())
        .args([
            "nmr1d",
            "multiplet",
            "--in",
            trees_in.to_str().unwrap(),
            "--regions",
            "7.70:7.50,7.02:6.78,5.92:5.68,3.30:3.10",
            "--total-protons",
            "4",
            "--style",
            "rsc",
            "--mol",
            mol_path.to_str().unwrap(),
            "--links-out",
            mol_links_json.to_str().unwrap(),
            "--json-out",
            trees_json.to_str().unwrap(),
        ])
        .output()
        .expect("run nmr1d multiplet on dd/ddd/dt/td");
    assert!(out_trees.status.success());

    let trep: Value = serde_json::from_str(&fs::read_to_string(&trees_json).unwrap()).unwrap();
    let tmults = trep["multiplets"].as_array().unwrap();
    let labels: Vec<&str> = tmults
        .iter()
        .map(|m| m["label"].as_str().unwrap())
        .collect();
    assert_eq!(labels, vec!["dd", "ddd", "dt", "td"]);

    println!(
        "W1_METRIC_JSON: {{\"gate\":\"CLI.nmr1d_multiplet\",\"ea_links\":{},\"max_j_err_hz\":{:.5},\"tree_classes\":\"{}\"}}",
        links.len(),
        (q_j - 7.15).abs().max((t_j - 7.15).abs()),
        labels.join(",")
    );
}

#[test]
fn cli_nmr1d_qnmr_internal_standard_purity_uncertainty_and_t1_guard() {
    let tmp = ScopedTempDir::new("qnmr_purity");
    let mhz = 600.0;
    let n = 16384usize;
    let axis = AxisPpm::from_bounds("1H", n, 9.0, 0.0, mhz);
    let ppm_grid: Vec<f64> = (0..n).map(|i| axis.index_to_ppm(i as f64)).collect();

    // Internal Standard: TMB (`MW = 168.190 g/mol`, `N_std = 3` at `6.09 ppm`, `m_std = 5.200 mg`, `P_std = 0.9995`)
    // Analyte: `MW_x = 242.270 g/mol`, `m_x = 12.500 mg`, true purity `P_x = 0.9840` (`98.40% w/w`)
    let mw_std = 168.190_f64;
    let m_std = 5.200_f64;
    let p_std = 0.9995_f64;
    let n_std = 3.0_f64;

    let mw_x = 242.270_f64;
    let m_x = 12.500_f64;
    let true_px = 0.9840_f64;

    let mol_std = (m_std * p_std) / mw_std;
    let mol_x = (m_x * true_px) / mw_x;

    // Area proportional to `moles * N_protons`
    let area_std = 100.0 * mol_std * n_std;
    let area_x_2h = 100.0 * mol_x * 2.0;
    let area_x_3h = 100.0 * mol_x * 3.0;

    let mut y = vec![0.0_f64; n];
    for (area, ppm, cpls) in [
        (area_std, 6.09, vec![]),
        (area_x_2h, 7.35, vec![(1usize, 8.2_f64, None)]),
        (area_x_3h, 3.85, vec![]),
    ] {
        let pks = synthesize_multiplet_peaks(ppm, mhz, &cpls, 1.0);
        let tr = synthesize_multiplet_spectrum(&ppm_grid, &pks, mhz, 1.0, 0.0);
        for (d, s) in y.iter_mut().zip(tr.iter()) {
            *d += area * s;
        }
    }

    let spec = Spectrum1D::new("qnmr_tube1", Experiment1DKind::Proton1D, axis, y, None);
    let spec_path = tmp.path.join("qnmr_tube1.json");
    fs::write(&spec_path, serde_json::to_string(&spec).unwrap()).unwrap();

    let out_tsv = tmp.path.join("qnmr_out.tsv");
    let out_json = tmp.path.join("qnmr_out.json");

    // 1. Quantitative recycle delay (`D1 = 22.0 s`, `AQ = 3.0 s` -> `T_rep = 25.0 s >= 5 * 3.6 s`)
    let res_ok = Command::new(nmrassign_bin())
        .args([
            "nmr1d",
            "qnmr",
            "--in",
            spec_path.to_str().unwrap(),
            "--std-crm",
            "TMB",
            "--std-mass-mg",
            "5.200",
            "--std-mass-sigma-mg",
            "0.005",
            "--std-purity-pct",
            "99.95",
            "--std-purity-sigma-pct",
            "0.05",
            "--std-region",
            "6.22:5.96",
            "--analyte-name",
            "CandidateAPI",
            "--analyte-mw",
            "242.270",
            "--analyte-mw-sigma",
            "0.005",
            "--analyte-mass-mg",
            "12.500",
            "--analyte-mass-sigma-mg",
            "0.008",
            "--volume-ml",
            "0.600",
            "--analyte-region",
            "7.48:7.22:2:ArH:2.4",
            "--analyte-region",
            "3.98:3.72:3:OCH3:1.9",
            "--d1-s",
            "22.0",
            "--aq-s",
            "3.0",
            "--out",
            out_tsv.to_str().unwrap(),
            "--json-out",
            out_json.to_str().unwrap(),
        ])
        .output()
        .expect("run nmr1d qnmr");

    assert!(
        res_ok.status.success(),
        "nmr1d qnmr failed:\n{}",
        String::from_utf8_lossy(&res_ok.stderr)
    );
    assert!(out_tsv.is_file());

    let qrep: QnmrPurityReport =
        serde_json::from_str(&fs::read_to_string(&out_json).unwrap()).unwrap();
    let abs_purity_err_pct = (qrep.purity_percent - 98.40).abs();
    assert!(
        abs_purity_err_pct < 0.25,
        "expected recovered purity ~98.40%, got {:.4}%",
        qrep.purity_percent
    );
    assert!(!qrep.t1_recycle_warning);
    assert!(
        qrep.expanded_uncertainty_k2_percent > 0.05 && qrep.expanded_uncertainty_k2_percent < 0.50
    );
    assert!(qrep.molar_concentration_mm.unwrap() > 80.0);

    // 2. Short recycle delay (`D1 = 2.0 s`, `AQ = 2.0 s` < `5 * 3.6 s = 18 s`) triggers `t1_recycle_warning`!
    let warn_json = tmp.path.join("qnmr_warn.json");
    let res_warn = Command::new(nmrassign_bin())
        .args([
            "nmr1d",
            "qnmr",
            "--std-crm",
            "TMB",
            "--std-mass-mg",
            "5.200",
            "--std-purity-pct",
            "99.95",
            "--std-integral",
            &format!("{area_std:.8}"),
            "--analyte-name",
            "CandidateAPI",
            "--analyte-mw",
            "242.270",
            "--analyte-mass-mg",
            "12.500",
            "--signal",
            &format!("2:{area_x_2h:.8}:0.01:ArH:2.4"),
            "--d1-s",
            "2.0",
            "--aq-s",
            "2.0",
            "--json-out",
            warn_json.to_str().unwrap(),
        ])
        .output()
        .expect("run nmr1d qnmr short d1");
    assert!(res_warn.status.success());
    let wrep: QnmrPurityReport =
        serde_json::from_str(&fs::read_to_string(&warn_json).unwrap()).unwrap();
    assert!(
        wrep.t1_recycle_warning,
        "expected 5*T1 incomplete relaxation warning when D1+AQ=4s < 18s"
    );

    // 3. Multi-tube replicate JSON input
    let rep_tubes = serde_json::json!([
        {
            "analyte": QnmrAnalyteSpec {
                name: "CandidateAPI".to_string(),
                mw_g_mol: mw_x,
                mw_sigma_g_mol: 0.005,
                mass_mg: m_x,
                mass_sigma_mg: 0.008,
                signals: vec![QnmrAnalyteSignal::new(2, area_x_2h * 1.0005, 0.01)],
                solution_volume_ml: Some(0.600),
                solution_volume_sigma_ml: Some(0.002),
            },
            "standard": QnmrStandardSpec::new("TMB", mw_std, m_std, p_std, 3, area_std)
        },
        {
            "analyte": QnmrAnalyteSpec {
                name: "CandidateAPI".to_string(),
                mw_g_mol: mw_x,
                mw_sigma_g_mol: 0.005,
                mass_mg: m_x,
                mass_sigma_mg: 0.008,
                signals: vec![QnmrAnalyteSignal::new(2, area_x_2h * 0.9995, 0.01)],
                solution_volume_ml: Some(0.600),
                solution_volume_sigma_ml: Some(0.002),
            },
            "standard": QnmrStandardSpec::new("TMB", mw_std, m_std, p_std, 3, area_std)
        }
    ]);
    let rep_in = tmp.path.join("tubes.json");
    let rep_out = tmp.path.join("tubes_out.json");
    fs::write(&rep_in, serde_json::to_string_pretty(&rep_tubes).unwrap()).unwrap();
    let res_rep = Command::new(nmrassign_bin())
        .args([
            "nmr1d",
            "qnmr",
            "--replicates-json",
            rep_in.to_str().unwrap(),
            "--json-out",
            rep_out.to_str().unwrap(),
        ])
        .output()
        .expect("run nmr1d qnmr replicates");
    assert!(res_rep.status.success());
    let rrep: QnmrPurityReport =
        serde_json::from_str(&fs::read_to_string(&rep_out).unwrap()).unwrap();
    assert_eq!(rrep.replicate_count, 2);
    assert!(rrep.replicate_rsd_percent.unwrap() < 0.20);

    println!(
        "W1_METRIC_JSON: {{\"gate\":\"CLI.nmr1d_qnmr\",\"recovered_purity_pct\":{:.4},\"abs_err_pct\":{abs_purity_err_pct:.5},\"expanded_u_k2_pct\":{:.4},\"t1_guard_triggered\":{}}}",
        qrep.purity_percent,
        qrep.expanded_uncertainty_k2_percent,
        wrep.t1_recycle_warning
    );
}

#[test]
fn cli_f15_adversarial_1d_ingest_error_paths() {
    let tmp = ScopedTempDir::new("f15_adversarial");

    // 1. Truncated Bruker `1r` (`SI=64` expects 256 bytes, only 48 bytes present)
    let trunc_dir = tmp.path.join("bruker_trunc");
    let trunc_pdata = trunc_dir.join("pdata").join("1");
    fs::create_dir_all(&trunc_pdata).unwrap();
    fs::write(
        trunc_dir.join("acqus"),
        "##$PULPROG= <zg30>\n##$NUC1= <1H>\n",
    )
    .unwrap();
    fs::write(
        trunc_pdata.join("procs"),
        "##$SI= 64\n##$SF= 500.13\n##$SW_p= 5000.0\n##$OFFSET= 10.0\n##$DTYPP= 0\n",
    )
    .unwrap();
    fs::write(trunc_pdata.join("1r"), vec![0u8; 48]).unwrap();

    // 2. Oversized declared `SI` (`SI=500000000` on tiny 16-byte `1r`)
    let huge_dir = tmp.path.join("bruker_huge_si");
    let huge_pdata = huge_dir.join("pdata").join("1");
    fs::create_dir_all(&huge_pdata).unwrap();
    fs::write(
        huge_pdata.join("procs"),
        "##$SI= 500000000\n##$SF= 500.13\n##$SW_p= 5000.0\n##$OFFSET= 10.0\n##$DTYPP= 0\n",
    )
    .unwrap();
    fs::write(huge_pdata.join("1r"), vec![0u8; 16]).unwrap();

    // 3. `DTYPP=2` (`float64`) Bruker `1r` containing `NaN`
    let nan_dir = tmp.path.join("bruker_nan_1r");
    let nan_pdata = nan_dir.join("pdata").join("1");
    fs::create_dir_all(&nan_pdata).unwrap();
    fs::write(
        nan_pdata.join("procs"),
        "##$SI= 4\n##$SF= 500.13\n##$SW_p= 5000.0\n##$OFFSET= 10.0\n##$DTYPP= 2\n",
    )
    .unwrap();
    let mut nan_bytes = Vec::new();
    for v in [1.0_f64, f64::NAN, 2.0, 3.0] {
        nan_bytes.extend_from_slice(&v.to_le_bytes());
    }
    fs::write(nan_pdata.join("1r"), nan_bytes).unwrap();

    // 4. Adversarial TSV FID with `NaN` and JSON with mismatched `axis.size`
    let nan_tsv = tmp.path.join("nan_fid.tsv");
    fs::write(
        &nan_tsv,
        "# domain=fid; sw_hz=5000; obs_mhz=500\n0.0\t1.0\t0.0\n0.0002\tNaN\t0.1\n",
    )
    .unwrap();

    let bad_json = tmp.path.join("mismatch.json");
    fs::write(
        &bad_json,
        r#"{"name":"bad","experiment":"Proton1D","axis":{"label":"1H","obs_mhz":500.0,"size":32,"first_ppm":10.0,"step_ppm":-0.1},"real":[1.0,2.0,3.0,4.0]}"#,
    )
    .unwrap();

    for bad_in in [&trunc_dir, &huge_dir, &nan_dir, &nan_tsv, &bad_json] {
        let out_proc = Command::new(nmrassign_bin())
            .args([
                "nmr1d",
                "process",
                "--in",
                bad_in.to_str().unwrap(),
                "--out",
                tmp.path.join("never_written.tsv").to_str().unwrap(),
            ])
            .output()
            .expect("invoke nmr1d process on adversarial input");
        assert!(
            !out_proc.status.success(),
            "expected failure for {}, but succeeded",
            bad_in.display()
        );
        let stderr = String::from_utf8_lossy(&out_proc.stderr);
        assert!(!stderr.trim().is_empty());
        assert!(
            !stderr.contains("panicked at"),
            "unexpected panic: {stderr}"
        );
    }

    // Also verify `nmr1d multiplet` and `nmr1d qnmr` fail cleanly on truncated `1r`
    let out_mult = Command::new(nmrassign_bin())
        .args([
            "nmr1d",
            "multiplet",
            "--in",
            trunc_dir.to_str().unwrap(),
            "--regions",
            "7.5:7.0",
        ])
        .output()
        .expect("invoke nmr1d multiplet on truncated 1r");
    assert!(!out_mult.status.success());

    let out_qnmr = Command::new(nmrassign_bin())
        .args([
            "nmr1d",
            "qnmr",
            "--in",
            bad_json.to_str().unwrap(),
            "--std-crm",
            "TMB",
            "--std-mass-mg",
            "5.0",
            "--std-region",
            "6.2:6.0",
            "--analyte-mw",
            "200.0",
            "--analyte-mass-mg",
            "10.0",
            "--analyte-region",
            "7.5:7.2:2",
        ])
        .output()
        .expect("invoke nmr1d qnmr on mismatched json");
    assert!(!out_qnmr.status.success());
}

#[test]
fn cli_f29_nmr1d_process_c13_dss_ref_and_jeol_jdf_ingest() {
    let tmp = ScopedTempDir::new("f29_c13_jdf");

    // 1. Verify `--c13-ref dss` shifts TMS-referenced 13C peaks by exact IUPAC Xi +2.665 ppm onto BMRB DSS scale (`20.W1:P0-2`)
    let lines_c13 = [
        SyntheticLine1D::lorentzian(96.568, 10.0, 1.5),
        SyntheticLine1D::lorentzian(15.853, 12.0, 1.8),
    ];
    let td_c13 = synthesize_fid_1d(
        "sample_13c",
        Experiment1DKind::Carbon13Decoupled,
        4096,
        22_000.0,
        100.612_729,
        100.0,
        &lines_c13,
    );
    let c13_in = tmp.path.join("c13_fid.json");
    fs::write(&c13_in, serde_json::to_string(&td_c13).unwrap()).unwrap();
    let c13_out = tmp.path.join("c13_dss.tsv");
    let c13_sum = tmp.path.join("c13_dss_summary.json");

    let res_c13 = Command::new(nmrassign_bin())
        .args([
            "nmr1d",
            "process",
            "--in",
            c13_in.to_str().unwrap(),
            "--out",
            c13_out.to_str().unwrap(),
            "--c13-ref",
            "dss",
            "--json-out",
            c13_sum.to_str().unwrap(),
        ])
        .output()
        .expect("run nmr1d process --c13-ref dss");
    assert!(
        res_c13.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&res_c13.stderr)
    );
    let rep: Value = serde_json::from_str(&fs::read_to_string(&c13_sum).unwrap()).unwrap();
    assert_eq!(rep["c13_reference_standard"].as_str(), Some("dss"));
    let applied_ref = rep["applied_reference_shift_ppm"]
        .as_f64()
        .expect("applied_reference_shift_ppm present");
    assert!((applied_ref - 2.6648).abs() < 0.005);
    let mut pks: Vec<&Value> = rep["picked_peaks"].as_array().unwrap().iter().collect();
    pks.sort_by(|a, b| {
        b["height"]
            .as_f64()
            .unwrap()
            .total_cmp(&a["height"].as_f64().unwrap())
    });
    let mut top2_ppms = [
        pks[0]["ppm"].as_f64().unwrap(),
        pks[1]["ppm"].as_f64().unwrap(),
    ];
    top2_ppms.sort_by(|a, b| b.total_cmp(a));
    assert!((top2_ppms[0] - (96.568 + 2.6648)).abs() < 0.02);
    assert!((top2_ppms[1] - (15.853 + 2.6646)).abs() < 0.02);

    // 2. Verify CLI `nmr1d process` ingests `.jdf` cleanly and rejects corrupted `.jdf` without panic
    let mut jdf = vec![0u8; 1296 + 256 * 16];
    jdf[0..8].copy_from_slice(b"JEOL.NMR");
    jdf[8] = 1; // LE
    jdf[9] = 1;
    jdf[12] = 1; // 1D
    jdf[14] = 1; // f64
    jdf[24] = 3; // split complex
    jdf[32] = 1;
    jdf[33] = 26; // PPM
    jdf[176..180].copy_from_slice(&256u32.to_be_bytes());
    jdf[208..212].copy_from_slice(&0u32.to_be_bytes());
    jdf[240..244].copy_from_slice(&255u32.to_be_bytes());
    jdf[272..280].copy_from_slice(&10.0f64.to_be_bytes());
    jdf[336..344].copy_from_slice(&0.0f64.to_be_bytes());
    jdf[1064..1072].copy_from_slice(&500.0f64.to_be_bytes());
    jdf[1284..1288].copy_from_slice(&1296u32.to_be_bytes());
    jdf[1292..1296].copy_from_slice(&(256u32 * 16).to_be_bytes());
    for k in 0..256usize {
        let ppm = 10.0 - 10.0 * (k as f64) / 255.0;
        let v = 30.0 / (1.0 + ((ppm - 3.65) / 0.08).powi(2));
        let off = 1296 + k * 8;
        jdf[off..off + 8].copy_from_slice(&v.to_le_bytes());
    }
    let jdf_path = tmp.path.join("sample.jdf");
    fs::write(&jdf_path, &jdf).unwrap();
    let jdf_out = tmp.path.join("jdf_out.tsv");
    let jdf_sum = tmp.path.join("jdf_sum.json");
    let out_jdf = Command::new(nmrassign_bin())
        .args([
            "nmr1d",
            "process",
            "--in",
            jdf_path.to_str().unwrap(),
            "--out",
            jdf_out.to_str().unwrap(),
            "--json-out",
            jdf_sum.to_str().unwrap(),
        ])
        .output()
        .expect("run nmr1d process on .jdf");
    assert!(
        out_jdf.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out_jdf.stderr)
    );
    let jrep: Value = serde_json::from_str(&fs::read_to_string(&jdf_sum).unwrap()).unwrap();
    let jpks = jrep["picked_peaks"].as_array().unwrap();
    let jmain = jpks
        .iter()
        .max_by(|a, b| {
            a["height"]
                .as_f64()
                .unwrap()
                .total_cmp(&b["height"].as_f64().unwrap())
        })
        .unwrap();
    assert!((jmain["ppm"].as_f64().unwrap() - 3.65).abs() < 0.05);

    // Truncated `.jdf` must fail cleanly without panic
    let trunc_jdf = tmp.path.join("trunc.jdf");
    fs::write(&trunc_jdf, &jdf[..512]).unwrap();
    let bad_out = Command::new(nmrassign_bin())
        .args([
            "nmr1d",
            "process",
            "--in",
            trunc_jdf.to_str().unwrap(),
            "--out",
            tmp.path.join("unused.tsv").to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!bad_out.status.success());
    assert!(!String::from_utf8_lossy(&bad_out.stderr).contains("panicked at"));

    // Real public JEOL Delta `.jdf` (`rutin_q1h_400mhz_dmsod6.jdf`, `20.W1:P2-2`) end-to-end CLI check
    let real_jdf = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../data/fixtures/jeol_public/rutin_q1h_400mhz_dmsod6.jdf");
    let real_out = tmp.path.join("rutin_out.tsv");
    let real_sum = tmp.path.join("rutin_sum.json");
    let res_real = Command::new(nmrassign_bin())
        .args([
            "nmr1d",
            "process",
            "--in",
            real_jdf.to_str().unwrap(),
            "--out",
            real_out.to_str().unwrap(),
            "--auto-phase",
            "--baseline",
            "whittaker",
            "--json-out",
            real_sum.to_str().unwrap(),
        ])
        .output()
        .expect("run nmr1d process on real public JEOL .jdf");
    assert!(
        res_real.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&res_real.stderr)
    );
    let rrep: Value = serde_json::from_str(&fs::read_to_string(&real_sum).unwrap()).unwrap();
    assert!(rrep["applied_reference_shift_ppm"].as_f64().is_some());
}

#[test]
fn cli_v0150_p1_3_shared_pipeline_fid_auto_ref_dept90_and_lineshape_guard() {
    let tmp = ScopedTempDir::new("v0150_p1_3_cli");

    // 1. Raw FID carrying `solvent = Some("CDCl3")` auto-calibrates via the shared 1D pipeline even
    //    WITHOUT explicit `--calibrate-solvent`, recording `applied_reference_shift_ppm` in JSON & TSV!
    let lines_cdcl3 = [
        SyntheticLine1D::lorentzian(7.215, 10.0, 1.2), // uncalibrated CHCl3 (-0.045 ppm from 7.260)
        SyntheticLine1D::lorentzian(3.655, 22.0, 1.3),
    ];
    let mut fid_cdcl3 = synthesize_fid_1d(
        "auto_ref_fid",
        Experiment1DKind::Proton1D,
        4096,
        6000.0,
        500.0,
        4.70,
        &lines_cdcl3,
    );
    fid_cdcl3.solvent = Some("CDCl3".to_string());
    let fid_in = tmp.path.join("cdcl3_fid.json");
    fs::write(&fid_in, serde_json::to_string(&fid_cdcl3).unwrap()).unwrap();
    let fid_out = tmp.path.join("cdcl3_out.tsv");
    let fid_sum = tmp.path.join("cdcl3_sum.json");
    let res = Command::new(nmrassign_bin())
        .args([
            "nmr1d",
            "process",
            "--in",
            fid_in.to_str().unwrap(),
            "--out",
            fid_out.to_str().unwrap(),
            "--json-out",
            fid_sum.to_str().unwrap(),
        ])
        .output()
        .expect("nmr1d process auto-ref on FID");
    assert!(
        res.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&res.stderr)
    );
    let rep: Value = serde_json::from_str(&fs::read_to_string(&fid_sum).unwrap()).unwrap();
    let shift = rep["applied_reference_shift_ppm"]
        .as_f64()
        .expect("applied_reference_shift_ppm recorded by CLI");
    assert!((shift - 0.045).abs() < 0.004, "got shift={shift}");
    let tsv_hdr = fs::read_to_string(&fid_out)
        .unwrap()
        .lines()
        .next()
        .unwrap()
        .to_string();
    assert!(tsv_hdr.contains("applied_reference_shift_ppm="));

    // 2. Explicit `--calibrate-solvent CDCl3` fails closed (`!status.success()`) when only a sharp
    //    solute J-doublet sits near 7.26 ppm (`v0150:P1-3` lineshape guard).
    let dbl_lines = [
        SyntheticLine1D::lorentzian(7.242, 14.0, 1.1),
        SyntheticLine1D::lorentzian(7.258, 14.0, 1.1),
    ];
    let fid_dbl = synthesize_fid_1d(
        "dbl_fid",
        Experiment1DKind::Proton1D,
        8192,
        6000.0,
        500.0,
        4.70,
        &dbl_lines,
    );
    let dbl_in = tmp.path.join("dbl_fid.json");
    fs::write(&dbl_in, serde_json::to_string(&fid_dbl).unwrap()).unwrap();
    let res_dbl = Command::new(nmrassign_bin())
        .args([
            "nmr1d",
            "process",
            "--in",
            dbl_in.to_str().unwrap(),
            "--out",
            tmp.path.join("dbl_out.tsv").to_str().unwrap(),
            "--calibrate-solvent",
            "CDCl3",
        ])
        .output()
        .expect("nmr1d process --calibrate-solvent on doublet");
    assert!(
        !res_dbl.status.success(),
        "explicit --calibrate-solvent must fail closed on solute J-doublet"
    );

    // 3. `--dept90` disambiguates an inverted C-alpha (54 ppm CH) + PEG (66 ppm CH2) DEPT-135 (`20.W1:P1-2`).
    let d135_lines = [
        SyntheticLine1D {
            ppm: 54.0,
            amplitude: 1.2,
            fwhm_hz: 1.8,
            gauss_fwhm_hz: 0.0,
            phase_deg: 150.0, // inverted (> 90 deg) without DEPT-90!
        },
        SyntheticLine1D {
            ppm: 66.0,
            amplitude: 1.6,
            fwhm_hz: 1.8,
            gauss_fwhm_hz: 0.0,
            phase_deg: -30.0, // 180 deg opposite to CH
        },
    ];
    let fid_135 = synthesize_fid_1d(
        "d135",
        Experiment1DKind::Dept135,
        4096,
        20_000.0,
        125.76,
        60.0,
        &d135_lines,
    );
    let fid_90 = synthesize_fid_1d(
        "d90",
        Experiment1DKind::Dept90,
        4096,
        20_000.0,
        125.76,
        60.0,
        &[SyntheticLine1D::lorentzian(54.0, 1.2, 1.8)],
    );
    let d135_in = tmp.path.join("d135.json");
    let d90_in = tmp.path.join("d90.json");
    let d135_out = tmp.path.join("d135_phased.json");
    let d135_sum = tmp.path.join("d135_sum.json");
    fs::write(&d135_in, serde_json::to_string(&fid_135).unwrap()).unwrap();
    fs::write(&d90_in, serde_json::to_string(&fid_90).unwrap()).unwrap();
    let res_d135 = Command::new(nmrassign_bin())
        .args([
            "nmr1d",
            "process",
            "--in",
            d135_in.to_str().unwrap(),
            "--out",
            d135_out.to_str().unwrap(),
            "--auto-phase",
            "--dept90",
            d90_in.to_str().unwrap(),
            "--json-out",
            d135_sum.to_str().unwrap(),
        ])
        .output()
        .expect("nmr1d process --dept90");
    assert!(
        res_d135.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&res_d135.stderr)
    );
    let drep: Value = serde_json::from_str(&fs::read_to_string(&d135_sum).unwrap()).unwrap();
    assert_eq!(
        drep["auto_phase"]["polarity_resolution"].as_str(),
        Some("MultiplicityReference")
    );
    let sp_phased: nmr_core::nmr1d::Spectrum1D =
        serde_json::from_str(&fs::read_to_string(&d135_out).unwrap()).unwrap();
    let idx_54 = sp_phased.axis.ppm_to_index(54.0).round() as usize;
    let idx_66 = sp_phased.axis.ppm_to_index(66.0).round() as usize;
    assert!(sp_phased.real[idx_54] > 0.0 && sp_phased.real[idx_66] < 0.0);
}
