//! End-to-end integration tests for `nmrassign relax` and `nmrassign modelfree` (`T13.1`–`T13.3`).
//!
//! # Literature & Clean-Room Provenance
//! - Kay, L. E., Torchia, D. A. & Bax, A. (1989). *Biochemistry* 28(23), 8972–8979.
//! - Farrow, N. A. et al. (1994). *Biochemistry* 33(19), 5984–6003; (1995) *J. Biomol. NMR* 6, 153–162.
//! - Lipari, G. & Szabo, A. (1982a, 1982b). *J. Am. Chem. Soc.* 104(17), 4546–4559 & 4559–4570.
//! - Clore, G. M. et al. (1990). *J. Am. Chem. Soc.* 112(12), 4989–4991.
//! - Mandel, A. M., Akke, M. & Palmer, A. G. III (1995). *J. Mol. Biol.* 246(1), 144–163.
//! - Woessner, D. E. (1962). *J. Chem. Phys.* 37(3), 647–654.
//! - d'Auvergne, E. J. & Gooley, P. R. (2003). *J. Biomol. NMR* 25(1), 25–39.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use nmr_core::modelfree::{
    extract_nh_unit_vectors_from_pdb, predict_r1_r2_noe, DiffusionTensor, LocalMfParams,
};
use nmr_core::relax_fit::RelaxPhysicsConfig;
use nmr_core::relax_io::{
    write_relax_series_tsv_str, LockedPeakFootprint, RelaxExperimentKind, RelaxSchedule,
    RelaxSeriesDataset, ResidueRelaxTrace,
};
use nmr_core::spectrum::{write_ucsf_2d_tiled, AxisPpm, Spectrum2D};

static NEXT_TMP_ID: AtomicU64 = AtomicU64::new(1);

struct ScopedTempDir {
    path: PathBuf,
}

impl ScopedTempDir {
    fn new(prefix: &str) -> Self {
        let id = NEXT_TMP_ID.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "nmrassign_relax_cli_{prefix}_{}_{}",
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

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("workspace root")
}

fn nmrassign_bin() -> &'static str {
    env!("CARGO_BIN_EXE_nmrassign")
}

fn build_synthetic_series_tsv(
    name: &str,
    kind: RelaxExperimentKind,
    field_mhz: f64,
    delays_sec: &[f64],
    rows: &[(i32, &str, Vec<f64>, Vec<f64>)],
) -> String {
    let n_planes = delays_sec.len();
    let traces: Vec<ResidueRelaxTrace> = rows
        .iter()
        .map(|(seq_id, res_name, intensities, sigmas)| {
            let fp = LockedPeakFootprint {
                seq_id: *seq_id,
                res_name: (*res_name).to_string(),
                atom_h: "H".to_string(),
                atom_x: "N".to_string(),
                delta_h_ppm: 8.20,
                delta_x_ppm: 120.0,
                sigma_h_ppm: 0.022,
                sigma_x_ppm: 0.20,
                eta_h: 0.0,
                eta_x: 0.0,
                overlap_cluster_id: (*seq_id as usize).saturating_sub(1),
            };
            let vol_factor = fp.unit_volume_factor_ppm2();
            let volumes = intensities.iter().map(|&a| a * vol_factor).collect();
            ResidueRelaxTrace {
                footprint: fp,
                intensities: intensities.clone(),
                sigmas: sigmas.clone(),
                volumes,
            }
        })
        .collect();

    let ds = RelaxSeriesDataset {
        name: name.to_string(),
        experiment_kind: kind,
        field_mhz,
        schedule: RelaxSchedule::VariableDelaysSec {
            delays_sec: delays_sec.to_vec(),
        },
        plane_noise_mad: vec![1.0; n_planes],
        plane_is_duplicate: vec![false; n_planes],
        duplicate_variance_scale: 1.0,
        duplicate_dof: 0,
        traces,
    };
    write_relax_series_tsv_str(&ds)
}

#[test]
fn cli_relax_and_modelfree_help_and_validation_guards() {
    let out_relax = Command::new(nmrassign_bin())
        .args(["relax", "--help"])
        .output()
        .expect("spawn nmrassign relax --help");
    assert!(out_relax.status.success(), "relax --help must exit 0");
    let stdout_relax = String::from_utf8_lossy(&out_relax.stdout);
    assert!(stdout_relax.contains("nmrassign relax"));
    assert!(stdout_relax.contains("--mc-trials"));

    let out_mf = Command::new(nmrassign_bin())
        .args(["modelfree", "--help"])
        .output()
        .expect("spawn nmrassign modelfree --help");
    assert!(out_mf.status.success(), "modelfree --help must exit 0");
    let stdout_mf = String::from_utf8_lossy(&out_mf.stdout);
    assert!(stdout_mf.contains("nmrassign modelfree"));
    assert!(stdout_mf.contains("--criterion"));

    let bad_relax = Command::new(nmrassign_bin())
        .args(["relax", "--unknown-flag"])
        .output()
        .expect("spawn nmrassign relax invalid");
    assert!(!bad_relax.status.success());

    let bad_mf = Command::new(nmrassign_bin())
        .args(["modelfree", "--out", "unused.tsv"])
        .output()
        .expect("spawn nmrassign modelfree missing inputs");
    assert!(!bad_mf.status.success());
}

#[test]
fn cli_relax_2d_ucsf_folder_integration_and_nef_roundtrip() {
    let tmp = ScopedTempDir::new("ucsf_folder");
    let series_dir = tmp.path.join("planes_2d");
    fs::create_dir_all(&series_dir).unwrap();

    let delays = [0.01_f64, 0.08, 0.18, 0.35, 0.65, 1.05];
    let vd_txt = delays
        .iter()
        .map(|d| format!("{:.0}ms", d * 1000.0))
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(series_dir.join("vdlist"), format!("{vd_txt}\n")).unwrap();

    // Synthesize 6 clean 2D UCSF planes containing 5 well-resolved crosspeaks with known R1 rates
    let nx = 96usize;
    let ny = 64usize;
    let axis_x = AxisPpm::from_bounds("1H", nx, 9.4, 7.0, 600.13);
    let axis_y = AxisPpm::from_bounds("15N", ny, 128.0, 108.0, 60.81);

    let targets = [
        (1, 9.05, 125.0, 1.15_f64),
        (2, 8.65, 121.5, 1.35_f64),
        (3, 8.25, 118.0, 1.55_f64),
        (4, 7.85, 114.5, 1.80_f64),
        (5, 7.45, 111.0, 2.05_f64),
    ];

    let mut peaks_tsv = String::from("# exp\th\tn\tc\tsign\theight\n");
    for &(_seq_id, dh, dn, _r1) in &targets {
        peaks_tsv.push_str(&format!("HSQC\t{dh:.4}\t{dn:.4}\t.\t1\t1000.0\n"));
    }
    let peaks_path = tmp.path.join("seeds.tsv");
    fs::write(&peaks_path, peaks_tsv).unwrap();

    let sig_h = 0.032_f64;
    let sig_n = 0.38_f64;
    for (p_idx, &t_sec) in delays.iter().enumerate() {
        let mut data = vec![0.0_f32; nx * ny];
        for iy in 0..ny {
            let n_ppm = axis_y.first_ppm + (iy as f64) * axis_y.step_ppm;
            for ix in 0..nx {
                let h_ppm = axis_x.first_ppm + (ix as f64) * axis_x.step_ppm;
                let mut val = 0.0_f64;
                for &(_seq_id, dh, dn, r1) in &targets {
                    let amp = 1000.0 * (-r1 * t_sec).exp();
                    let gh = (-0.5 * ((h_ppm - dh) / sig_h).powi(2)).exp();
                    let gn = (-0.5 * ((n_ppm - dn) / sig_n).powi(2)).exp();
                    val += amp * gh * gn;
                }
                // Small deterministic background ripple (avoids zero-MAD degenerate noise)
                let pseudo_noise =
                    0.15 * (((ix * 31 + iy * 17 + p_idx * 13) % 19) as f64 - 9.0) / 9.0;
                data[iy * nx + ix] = (val + pseudo_noise) as f32;
            }
        }
        let spec = Spectrum2D {
            name: format!("plane_{p_idx:02}"),
            axis_x: axis_x.clone(),
            axis_y: axis_y.clone(),
            data,
        };
        let ucsf_path = series_dir.join(format!("plane_{p_idx:02}.ucsf"));
        write_ucsf_2d_tiled(&ucsf_path, &spec, 16, 16).unwrap();
    }

    let series_nef = tmp.path.join("integrated.nef");
    let rates_from_dir = tmp.path.join("r1_from_dir.tsv");
    let out1 = Command::new(nmrassign_bin())
        .args([
            "relax",
            "--series",
            series_dir.to_str().unwrap(),
            "--peaks",
            peaks_path.to_str().unwrap(),
            "--kind",
            "r1",
            "--engine",
            "cpu",
            "--mc-trials",
            "150",
            "--series-out",
            series_nef.to_str().unwrap(),
            "--out",
            rates_from_dir.to_str().unwrap(),
        ])
        .output()
        .expect("run nmrassign relax on 2D UCSF folder");
    assert!(
        out1.status.success(),
        "nmrassign relax folder failed: {}",
        String::from_utf8_lossy(&out1.stderr)
    );
    let nef_body = fs::read_to_string(&series_nef).unwrap();
    assert!(nef_body.contains("save_nmrassign_relaxation_series"));

    let rates_from_nef = tmp.path.join("r1_from_nef.tsv");
    let out2 = Command::new(nmrassign_bin())
        .args([
            "relax",
            "--series",
            series_nef.to_str().unwrap(),
            "--engine",
            "auto",
            "--mc-trials",
            "150",
            "--out",
            rates_from_nef.to_str().unwrap(),
        ])
        .output()
        .expect("run nmrassign relax on exported NEF");
    assert!(
        out2.status.success(),
        "nmrassign relax NEF failed: {}",
        String::from_utf8_lossy(&out2.stderr)
    );

    let tsv_txt = fs::read_to_string(&rates_from_nef).unwrap();
    let mut fitted_rates: BTreeMap<i32, f64> = BTreeMap::new();
    for line in tsv_txt.lines() {
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') || t.starts_with("seq_id") {
            continue;
        }
        let cols: Vec<&str> = t.split('\t').collect();
        let seq_id: i32 = cols[0].parse().unwrap();
        let rate: f64 = cols[4].parse().unwrap();
        fitted_rates.insert(seq_id, rate);
    }
    assert_eq!(fitted_rates.len(), targets.len());
    for &(seq_id, _dh, _dn, true_r1) in &targets {
        let fit_r1 = fitted_rates[&seq_id];
        assert!(
            (fit_r1 - true_r1).abs() < 0.015,
            "seq_id={seq_id}: fitted R1={fit_r1:.5} vs true={true_r1:.5}"
        );
    }
}

#[test]
fn cli_relax_to_modelfree_multifield_1ubq_end_to_end() {
    let tmp = ScopedTempDir::new("e2e_1ubq_mf");
    let pdb_path = workspace_root().join("data/fixtures/ubiquitin_1ubq_xray.pdb");
    assert!(pdb_path.is_file(), "1UBQ fixture must exist");

    let nh_vecs = extract_nh_unit_vectors_from_pdb(&pdb_path).expect("extract 1UBQ N-H vectors");
    assert!(
        nh_vecs.len() >= 70,
        "expected >= 70 reconstructed N-H vectors on 1UBQ, got {}",
        nh_vecs.len()
    );

    // Ground-truth prolate axial diffusion tensor (`tau_m ~ 4.10 ns`, `D_par / D_perp = 1.48`)
    let d_perp = 3.50e7_f64;
    let d_par = 5.18e7_f64;
    let true_tensor = DiffusionTensor::AxialSymmetric {
        d_perp_s_inv: d_perp,
        d_par_s_inv: d_par,
        theta_rad: 0.62,
        phi_rad: 1.05,
    };
    let true_tau_m_ns = true_tensor.effective_tau_m_sec() * 1.0e9;
    let phys = RelaxPhysicsConfig::default();

    // Assign realistic globular core (M1/M2) + loop/exchange (M3/M4/M5) ground-truth dynamics across 1UBQ
    let mut truth_s2: BTreeMap<i32, f64> = BTreeMap::new();
    let mut truth_params: Vec<(i32, &'static str, [f64; 3], LocalMfParams)> = Vec::new();
    for (idx, &(seq_id, u_vec)) in nh_vecs.iter().enumerate() {
        let local = if idx % 9 == 7 {
            LocalMfParams::m3(0.85, 2.8)
        } else if idx % 13 == 5 {
            LocalMfParams::m4(0.80, 50.0e-12, 2.9)
        } else if idx % 11 == 9 {
            LocalMfParams::m5(0.88, 0.72, 0.90e-9)
        } else if idx % 4 == 1 {
            LocalMfParams::m2(0.82, 45.0e-12)
        } else {
            LocalMfParams::m1(0.85 + ((idx % 5) as f64) * 0.015)
        };
        truth_s2.insert(seq_id, local.s2);
        truth_params.push((seq_id, "ALA", u_vec, local));
    }

    let r1_delays = [0.01_f64, 0.05, 0.12, 0.25, 0.45, 0.70, 1.00, 1.40];
    let r2_delays = [0.008_f64, 0.024, 0.048, 0.080, 0.120, 0.176, 0.240, 0.320];
    let noe_delays = [0.0_f64, 5.0];
    let fields = [600.0_f64, 800.0_f64];

    let allow_no_gpu = std::env::var("NMR_ALLOW_NO_GPU")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    let mut r1_out_files = Vec::new();
    let mut r2_out_files = Vec::new();
    let mut noe_out_files = Vec::new();

    for (f_idx, &fm) in fields.iter().enumerate() {
        let mut r1_rows = Vec::new();
        let mut r2_rows = Vec::new();
        let mut noe_rows = Vec::new();

        for &(seq_id, res_name, u_vec, ref local) in &truth_params {
            let comps = true_tensor.rotational_components(Some(u_vec));
            let (r1, r2, noe) = predict_r1_r2_noe(&comps, local, fm, 600.0, &phys);
            let i0 = 1000.0_f64;
            let r1_int: Vec<f64> = r1_delays.iter().map(|&t| i0 * (-r1 * t).exp()).collect();
            let r1_sig = vec![2.5_f64; r1_delays.len()];
            r1_rows.push((seq_id, res_name, r1_int, r1_sig));

            let r2_int: Vec<f64> = r2_delays.iter().map(|&t| i0 * (-r2 * t).exp()).collect();
            let r2_sig = vec![2.5_f64; r2_delays.len()];
            r2_rows.push((seq_id, res_name, r2_int, r2_sig));

            let noe_int = vec![i0, i0 * noe];
            let noe_sig = vec![3.0_f64, 3.0_f64];
            noe_rows.push((seq_id, res_name, noe_int, noe_sig));
        }

        let r1_ser = tmp.path.join(format!("series_r1_{fm:.0}.tsv"));
        let r2_ser = tmp.path.join(format!("series_r2_{fm:.0}.tsv"));
        let noe_ser = tmp.path.join(format!("series_noe_{fm:.0}.tsv"));

        fs::write(
            &r1_ser,
            build_synthetic_series_tsv(
                &format!("ubq_r1_{fm:.0}"),
                RelaxExperimentKind::R1,
                fm,
                &r1_delays,
                &r1_rows,
            ),
        )
        .unwrap();
        fs::write(
            &r2_ser,
            build_synthetic_series_tsv(
                &format!("ubq_r2_{fm:.0}"),
                RelaxExperimentKind::R2,
                fm,
                &r2_delays,
                &r2_rows,
            ),
        )
        .unwrap();
        fs::write(
            &noe_ser,
            build_synthetic_series_tsv(
                &format!("ubq_noe_{fm:.0}"),
                RelaxExperimentKind::HetNoe,
                fm,
                &noe_delays,
                &noe_rows,
            ),
        )
        .unwrap();

        let r1_rates = tmp.path.join(format!("rates_r1_{fm:.0}.tsv"));
        let r2_rates = tmp.path.join(format!("rates_r2_{fm:.0}.tsv"));
        let noe_rates = tmp.path.join(format!("rates_noe_{fm:.0}.tsv"));

        // Exercise `--engine gpu` on the first R1 dataset (unless NMR_ALLOW_NO_GPU=1) and `--engine cpu` on the second.
        let r1_engine = if f_idx == 0 && !allow_no_gpu {
            "gpu"
        } else if f_idx == 0 {
            "auto"
        } else {
            "cpu"
        };

        for (ser, out, kind, eng) in [
            (&r1_ser, &r1_rates, "r1", r1_engine),
            (&r2_ser, &r2_rates, "r2", "cpu"),
            (&noe_ser, &noe_rates, "hetnoe", "cpu"),
        ] {
            let status = Command::new(nmrassign_bin())
                .args([
                    "relax",
                    "--series",
                    ser.to_str().unwrap(),
                    "--kind",
                    kind,
                    "--engine",
                    eng,
                    "--mc-trials",
                    "200",
                    "--out",
                    out.to_str().unwrap(),
                ])
                .output()
                .expect("run nmrassign relax");
            assert!(
                status.status.success(),
                "nmrassign relax ({kind}, {fm} MHz, {eng}) failed: {}",
                String::from_utf8_lossy(&status.stderr)
            );
        }

        r1_out_files.push(r1_rates);
        r2_out_files.push(r2_rates);
        noe_out_files.push(noe_rates);
    }

    let mf_out = tmp.path.join("modelfree_1ubq_aic.tsv");
    let rsd_out = tmp.path.join("rsd_1ubq.tsv");
    let mut mf_cmd = Command::new(nmrassign_bin());
    mf_cmd.arg("modelfree");
    for p in &r1_out_files {
        mf_cmd.args(["--r1", p.to_str().unwrap()]);
    }
    for p in &r2_out_files {
        mf_cmd.args(["--r2", p.to_str().unwrap()]);
    }
    for p in &noe_out_files {
        mf_cmd.args(["--hetnoe", p.to_str().unwrap()]);
    }
    mf_cmd.args([
        "--pdb",
        pdb_path.to_str().unwrap(),
        "--diffusion",
        "auto",
        "--criterion",
        "aic",
        "--rsd-out",
        rsd_out.to_str().unwrap(),
        "--out",
        mf_out.to_str().unwrap(),
    ]);

    let mf_status = mf_cmd.output().expect("run nmrassign modelfree");
    assert!(
        mf_status.status.success(),
        "nmrassign modelfree failed: {}",
        String::from_utf8_lossy(&mf_status.stderr)
    );
    let mf_stdout = String::from_utf8_lossy(&mf_status.stdout);
    assert!(
        mf_stdout.contains("AxialSymmetric"),
        "expected Auto diffusion selection to pick AxialSymmetric on prolate 1UBQ, stdout={mf_stdout}"
    );

    // Validate Farrow (1995) RSD output
    let rsd_txt = fs::read_to_string(&rsd_out).unwrap();
    assert!(rsd_txt.contains("trimmed_tau_m_ns="));
    let rsd_rows = rsd_txt
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with('#') && !l.starts_with("seq_id"))
        .count();
    assert_eq!(rsd_rows, nh_vecs.len() * 2);

    // Validate Lipari-Szabo M1-M5 output table and S^2 / tau_m accuracy
    let mf_txt = fs::read_to_string(&mf_out).unwrap();
    let mut parsed_s2: BTreeMap<i32, f64> = BTreeMap::new();
    let mut s2_col_idx: Option<usize> = None;
    let mut fitted_tau_m_ns: Option<f64> = None;

    for line in mf_txt.lines() {
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        if let Some(comment) = t.strip_prefix('#') {
            for tok in comment.split_whitespace() {
                if let Some(v) = tok.strip_prefix("effective_tau_m_ns=") {
                    fitted_tau_m_ns = v.parse::<f64>().ok();
                }
            }
            continue;
        }
        let cols: Vec<&str> = t.split('\t').collect();
        if cols[0] == "seq_id" {
            s2_col_idx = cols.iter().position(|&c| c == "s2");
            continue;
        }
        let seq_id: i32 = cols[0].parse().unwrap();
        let s2_idx = s2_col_idx.expect("s2 column in modelfree.tsv");
        let s2_val: f64 = cols[s2_idx].parse().unwrap();
        parsed_s2.insert(seq_id, s2_val);
    }

    assert_eq!(parsed_s2.len(), nh_vecs.len());
    let s2_mae: f64 = parsed_s2
        .iter()
        .map(|(seq_id, &fit_s2)| (fit_s2 - truth_s2[seq_id]).abs())
        .sum::<f64>()
        / (parsed_s2.len() as f64);
    assert!(
        s2_mae < 0.005,
        "end-to-end CLI `relax -> modelfree` S^2 MAE = {s2_mae:.6} (must be < 0.005)"
    );

    if let Some(tm_ns) = fitted_tau_m_ns {
        let rel_err = ((tm_ns - true_tau_m_ns) / true_tau_m_ns).abs();
        assert!(
            rel_err < 0.005,
            "fitted effective tau_m = {tm_ns:.4} ns vs true {true_tau_m_ns:.4} ns (rel_err={rel_err:.5})"
        );
    }
}
