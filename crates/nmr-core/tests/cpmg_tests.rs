//! Quantitative verification suite for `nmr-core::cpmg` (`T13.5`):
//! - `Baldwin2014` closed-form `R2,eff` agreement `< 0.02 s^-1` against exact numerical `2x2`
//!   spin-echo matrix propagation across `k_ex = 50..5000 s^-1`, `p_B = 0.5%..25%`, and
//!   `Delta_R2 = R2B - R2A != 0`.
//! - Replicate `nu_CPMG` intensity error propagation and `AIC` / nested `F`-test statistical
//!   exchange gating against the `Flat` null model.
//! - Multi-field (`600 + 800 MHz`) global `(p_B, k_ex)` + per-residue `|Delta_omega|` recovery
//!   within `< 2%` error.
//! - Trott–Palmer (2002) asymmetric off-resonance `R1rho(omega_eff, theta)` recovering signed
//!   `Delta_varpi_AB`.
//!
//! Provenance: Clean-room verification citing Carver & Richards (*JMR* 1972), Luz & Meiboom
//! (*J. Chem. Phys.* 1963), Baldwin (*JMR* 2014), and Trott & Palmer (*JMR* 2002).

use std::f64::consts::TAU;

use nmr_core::cpmg::{
    compute_r2eff_from_intensities, evaluate_exchange_gate, fit_cpmg_global_cpu,
    fit_r1rho_trott_palmer_cpu, format_cpmg_fit_json, format_cpmg_fit_tsv, r1rho_trott_palmer_2002,
    r2eff_baldwin2014, r2eff_cr72, r2eff_lm63, r2eff_numerical_2x2, read_cpmg_series_tsv,
    write_cpmg_series_tsv, CpmgFitConfig, CpmgModel, CpmgSeriesDataset, ExchangeGateConfig,
    R1RhoOffResPoint, RawCpmgIntensityPlane,
};
use nmr_core::rng::Rng;
use serde_json::json;

#[test]
fn test_baldwin2014_matches_numerical_2x2_propagator_under_0p02_across_all_regimes() {
    let t_relax_s = 0.040; // 40 ms constant-time CPMG
                           // Integer cycles `n_cyc` in 40 ms -> `nu_CPMG = n_cyc / 0.040 = 25, 50, 100, ..., 1000 Hz`
    let nu_list = [25.0, 50.0, 100.0, 200.0, 400.0, 650.0, 1000.0];
    let kex_list = [50.0, 150.0, 450.0, 1200.0, 2800.0, 5000.0];
    let pb_list = [0.005, 0.02, 0.06, 0.15, 0.25];
    let delta_r2_list = [-8.0, 0.0, 15.0, 55.0];
    let dw_rad_list = [120.0, 450.0, 1400.0, 3200.0];

    let mut max_err = 0.0f64;
    let mut max_cr72_gap_nonzero_dr2 = 0.0f64;

    for &kex in &kex_list {
        for &pb in &pb_list {
            for &dr2 in &delta_r2_list {
                let r2a: f64 = 14.0;
                let r2b = (r2a + dr2).max(1.0);
                for &dw_rad in &dw_rad_list {
                    for &nu in &nu_list {
                        let b14 = r2eff_baldwin2014(r2a, r2b, pb, kex, dw_rad, t_relax_s, nu);
                        let num = r2eff_numerical_2x2(r2a, r2b, pb, kex, dw_rad, t_relax_s, nu);
                        let err = (b14 - num).abs();
                        max_err = max_err.max(err);
                        assert!(
                            err < 0.02,
                            "Baldwin2014 vs Numerical2x2 diff {err:.4e} >= 0.02 at kex={kex}, pb={pb}, dr2={dr2}, dw={dw_rad}, nu={nu}"
                        );

                        if dr2.abs() > 20.0 {
                            let cr = r2eff_cr72(r2a, r2b, pb, kex, dw_rad, nu);
                            max_cr72_gap_nonzero_dr2 =
                                max_cr72_gap_nonzero_dr2.max((cr - num).abs());
                        }
                    }
                }
            }
        }
    }

    println!(
        "[T13.5 CPU Baldwin2014] max_err_vs_num2x2={max_err:.4e} s^-1, max_cr72_gap_dr2={max_cr72_gap_nonzero_dr2:.4} s^-1"
    );
    assert!(
        max_err < 0.02,
        "max Baldwin2014 error {max_err:.3e} >= 0.02 s^-1"
    );
    assert!(
        max_cr72_gap_nonzero_dr2 > 0.10,
        "Baldwin2014 finite-T projection term must visibly improve over CR72 when Delta_R2 != 0"
    );

    // Also verify fast-exchange LM63 approaches Baldwin2014 when dw << k_ex and dR2 == 0
    let dw_fast = 120.0;
    let kex_fast = 5000.0;
    let pb_fast = 0.03;
    let lm = r2eff_lm63(
        12.0,
        (1.0 - pb_fast) * pb_fast * dw_fast * dw_fast,
        kex_fast,
        100.0,
    );
    let b14_fast = r2eff_baldwin2014(12.0, 12.0, pb_fast, kex_fast, dw_fast, 0.04, 100.0);
    let lm_diff = (lm - b14_fast).abs();
    assert!(lm_diff < 0.02);

    let n_grid =
        kex_list.len() * pb_list.len() * delta_r2_list.len() * dw_rad_list.len() * nu_list.len();
    println!(
        "W4_METRIC_JSON: {}",
        json!({
            "task": "T13.5",
            "kind": "cpmg_baldwin_vs_num2x2",
            "n_grid_points": n_grid,
            "max_err_s_inv": max_err,
            "gate_s_inv": 0.02,
            "max_cr72_gap_dr2_s_inv": max_cr72_gap_nonzero_dr2,
            "lm63_fast_diff_s_inv": lm_diff,
            "passed": max_err < 0.02 && lm_diff < 0.02,
        })
    );
}

#[test]
fn test_replicate_plane_r2eff_propagation_and_exchange_gate() {
    let mut rng = Rng::new(42421305);
    let t_relax = 0.040;
    let i0_true = 10_000.0;
    let noise_sd = 45.0;
    let nu_grid = [
        25.0, 50.0, 100.0, 150.0, 200.0, 300.0, 400.0, 600.0, 800.0, 1000.0,
    ];

    // 1. Exchanging residue (replicate planes at 0, 50, 200, 800 Hz)
    let mut ex_planes = vec![
        RawCpmgIntensityPlane {
            nu_cpmg_hz: 0.0,
            intensity: i0_true + rng.normal(0.0, noise_sd),
            baseline_noise: Some(noise_sd),
        },
        RawCpmgIntensityPlane {
            nu_cpmg_hz: 0.0,
            intensity: i0_true + rng.normal(0.0, noise_sd),
            baseline_noise: Some(noise_sd),
        },
    ];
    for &nu in &nu_grid {
        let r2 = r2eff_baldwin2014(12.5, 12.5, 0.035, 900.0, TAU * 60.8106 * 3.8, t_relax, nu);
        let reps = if nu == 50.0 || nu == 200.0 || nu == 800.0 {
            2
        } else {
            1
        };
        for _ in 0..reps {
            let inten = i0_true * (-r2 * t_relax).exp() + rng.normal(0.0, noise_sd);
            ex_planes.push(RawCpmgIntensityPlane {
                nu_cpmg_hz: nu,
                intensity: inten,
                baseline_noise: Some(noise_sd),
            });
        }
    }
    let ex_series =
        compute_r2eff_from_intensities(24, "THR", 60.8106, t_relax, &ex_planes, 0.05).unwrap();
    assert_eq!(ex_series.nu_cpmg_hz.len(), nu_grid.len());
    // Replicate 50 Hz point should have smaller uncertainty than single-plane 100 Hz neighbor
    assert!(ex_series.r2_eff_err[1] < ex_series.r2_eff_err[2]);

    let gate_cfg = ExchangeGateConfig::default();
    let ex_verdict = evaluate_exchange_gate(&[&ex_series], &gate_cfg);
    assert!(
        ex_verdict.significant_exchange,
        "exchanging residue must pass gate"
    );
    assert!(ex_verdict.observed_rex_s_inv > 8.0);
    assert!(ex_verdict.f_pvalue < 1e-4);

    // 2. Non-exchanging flat residue (`R2 = 13.0 s^-1`)
    let mut flat_planes = vec![
        RawCpmgIntensityPlane {
            nu_cpmg_hz: 0.0,
            intensity: i0_true + rng.normal(0.0, noise_sd),
            baseline_noise: Some(noise_sd),
        },
        RawCpmgIntensityPlane {
            nu_cpmg_hz: 0.0,
            intensity: i0_true + rng.normal(0.0, noise_sd),
            baseline_noise: Some(noise_sd),
        },
    ];
    for &nu in &nu_grid {
        let inten = i0_true * (-13.0 * t_relax).exp() + rng.normal(0.0, noise_sd);
        flat_planes.push(RawCpmgIntensityPlane {
            nu_cpmg_hz: nu,
            intensity: inten,
            baseline_noise: Some(noise_sd),
        });
    }
    let flat_series =
        compute_r2eff_from_intensities(25, "ALA", 60.8106, t_relax, &flat_planes, 0.10).unwrap();
    let flat_verdict = evaluate_exchange_gate(&[&flat_series], &gate_cfg);
    assert!(
        !flat_verdict.significant_exchange,
        "flat residue must be rejected by exchange gate"
    );
    println!(
        "W4_METRIC_JSON: {}",
        json!({
            "task": "T13.5",
            "kind": "cpmg_exchange_gate",
            "ex_rex_s_inv": ex_verdict.observed_rex_s_inv,
            "ex_delta_aic": ex_verdict.delta_aic,
            "ex_f_pvalue": ex_verdict.f_pvalue,
            "ex_significant": ex_verdict.significant_exchange,
            "flat_rex_s_inv": flat_verdict.observed_rex_s_inv,
            "flat_f_pvalue": flat_verdict.f_pvalue,
            "flat_significant": flat_verdict.significant_exchange,
            "passed": ex_verdict.significant_exchange && !flat_verdict.significant_exchange,
        })
    );
}

#[test]
fn test_multifield_600_800_mhz_global_cpmg_fit_under_2pct_and_trott_palmer_r1rho() {
    let true_pb = 0.032;
    let true_kex = 850.0;
    let nu_grid: Vec<f64> = vec![
        25.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0, 400.0, 550.0, 700.0, 850.0, 1000.0,
    ];
    let t_relax = 0.040;
    let fields = [(60.8106, 0.0), (81.0808, 2.4)]; // 600 MHz & 800 MHz 15N
    let exchanging = [
        (15, "LYS", 3.6, 11.5),
        (23, "ILE", 4.8, 13.0),
        (31, "GLN", 2.9, 10.8),
        (44, "VAL", 5.4, 14.2),
        (56, "LEU", 4.1, 12.1),
        (68, "HIS", 3.2, 11.9),
    ];

    let mut rng = Rng::new(20261350);
    let err_s_inv = 0.12;

    for &(model, model_label, seed_m) in &[
        (CpmgModel::Baldwin2014, "Baldwin2014", 20261350_u64),
        (CpmgModel::Cr72, "CR72", 20261351_u64),
    ] {
        let mut rng_m = Rng::new(seed_m);
        let mut model_series = Vec::new();
        for &(seq_id, res_name, dw_ppm, base_r2_600) in &exchanging {
            for &(larmor_mhz, r2_inc) in &fields {
                let r20 = base_r2_600 + r2_inc;
                let dw_rad = TAU * larmor_mhz * dw_ppm;
                let r2_vals: Vec<f64> = nu_grid
                    .iter()
                    .map(|&nu| {
                        let clean = match model {
                            CpmgModel::Cr72 => r2eff_cr72(r20, r20, true_pb, true_kex, dw_rad, nu),
                            _ => {
                                r2eff_baldwin2014(r20, r20, true_pb, true_kex, dw_rad, t_relax, nu)
                            }
                        };
                        clean + rng_m.normal(0.0, err_s_inv)
                    })
                    .collect();
                model_series.push(CpmgSeriesDataset {
                    seq_id,
                    res_name: res_name.to_string(),
                    larmor_mhz,
                    t_relax_s: t_relax,
                    nu_cpmg_hz: nu_grid.clone(),
                    r2_eff_s_inv: r2_vals,
                    r2_eff_err: vec![err_s_inv; nu_grid.len()],
                });
            }
        }
        for &(larmor_mhz, r2_inc) in &fields {
            let r2_vals: Vec<f64> = nu_grid
                .iter()
                .map(|_| 11.0 + r2_inc + rng_m.normal(0.0, err_s_inv))
                .collect();
            model_series.push(CpmgSeriesDataset {
                seq_id: 99,
                res_name: "SER".to_string(),
                larmor_mhz,
                t_relax_s: t_relax,
                nu_cpmg_hz: nu_grid.clone(),
                r2_eff_s_inv: r2_vals,
                r2_eff_err: vec![err_s_inv; nu_grid.len()],
            });
        }

        let tsv = write_cpmg_series_tsv(&model_series);
        let parsed = read_cpmg_series_tsv(&tsv).expect("CPMG TSV roundtrip");
        assert_eq!(parsed.len(), model_series.len());

        let cfg = CpmgFitConfig {
            model,
            ..CpmgFitConfig::default()
        };
        let fit = fit_cpmg_global_cpu(&parsed, &cfg).expect("global CPMG fit");
        assert_eq!(
            fit.residues.len(),
            exchanging.len(),
            "flat decoy 99 must be gated out ({model_label})"
        );

        let rel_pb = (fit.p_b - true_pb).abs() / true_pb;
        let rel_kex = (fit.k_ex_s_inv - true_kex).abs() / true_kex;
        assert!(
            rel_pb < 0.02,
            "{model_label} p_B relative error {rel_pb:.4} >= 2% (fit={}, true={true_pb})",
            fit.p_b
        );
        assert!(
            rel_kex < 0.02,
            "{model_label} k_ex relative error {rel_kex:.4} >= 2% (fit={}, true={true_kex})",
            fit.k_ex_s_inv
        );
        let mut max_rel_dw = 0.0f64;
        for (rf, &(_, _, true_dw, _)) in fit.residues.iter().zip(exchanging.iter()) {
            let rel_dw = (rf.dw_ppm - true_dw).abs() / true_dw;
            max_rel_dw = max_rel_dw.max(rel_dw);
            assert!(
                rel_dw < 0.02,
                "{model_label} res {} |dw| relative error {rel_dw:.4} >= 2% (fit={}, true={true_dw})",
                rf.seq_id,
                rf.dw_ppm
            );
        }

        println!(
            "W4_METRIC_JSON: {}",
            json!({
                "task": "T13.5",
                "kind": "cpmg_cpu_multifield_fit",
                "model": model_label,
                "fields_1h_mhz": [600, 800],
                "n_exchanging": fit.residues.len(),
                "n_gated_out": 1,
                "true_pb": true_pb,
                "fit_pb": fit.p_b,
                "rel_err_pb_pct": rel_pb * 100.0,
                "true_kex_s_inv": true_kex,
                "fit_kex_s_inv": fit.k_ex_s_inv,
                "rel_err_kex_pct": rel_kex * 100.0,
                "max_rel_dw_pct": max_rel_dw * 100.0,
                "reduced_chi2": fit.reduced_chi2,
                "dof": fit.dof,
                "passed": true,
            })
        );

        if model == CpmgModel::Baldwin2014 {
            let out_tsv = format_cpmg_fit_tsv(&fit);
            let out_json = format_cpmg_fit_json(&fit).unwrap();
            assert!(out_tsv.contains("baldwin2014") && out_json.contains("\"k_ex_s_inv\""));
        }
    }

    // Verify Trott–Palmer (2002) asymmetric off-resonance R1rho recovers SIGNED dw_ab_ppm!
    let major_ppm = 120.0;
    let larmor = 60.8106;
    let true_signed_dw = -4.2; // negative shift difference!
    let mut r1rho_pts = Vec::new();
    for &sl_hz in &[400.0, 800.0, 1400.0] {
        for step in -10..=10 {
            let carrier = major_ppm + 0.8 * (step as f64);
            let om_a = TAU * larmor * (major_ppm - carrier);
            let dw_rad = TAU * larmor * true_signed_dw;
            let (weff, theta, r1r) =
                r1rho_trott_palmer_2002(1.4, 13.0, 0.025, 1400.0, om_a, dw_rad, TAU * sl_hz);
            r1rho_pts.push(R1RhoOffResPoint {
                carrier_ppm: carrier,
                spin_lock_hz: sl_hz,
                omega_eff_rad_s: weff,
                theta_rad: theta,
                r1rho_s_inv: r1r + rng.normal(0.0, 0.05),
                r1rho_err: 0.05,
            });
        }
    }
    let tp_fit = fit_r1rho_trott_palmer_cpu(major_ppm, larmor, &r1rho_pts).expect("R1rho fit");
    assert!(
        (tp_fit.dw_ab_ppm - true_signed_dw).abs() < 0.12 && tp_fit.dw_ab_ppm < 0.0,
        "Trott-Palmer R1rho must recover negative signed dw_AB = -4.2 ppm, got {:.3}",
        tp_fit.dw_ab_ppm
    );
    println!(
        "W4_METRIC_JSON: {}",
        json!({
            "task": "T13.5",
            "kind": "r1rho_trott_palmer_fit",
            "true_pb": 0.025,
            "fit_pb": tp_fit.p_b,
            "true_kex_s_inv": 1400.0,
            "fit_kex_s_inv": tp_fit.k_ex_s_inv,
            "true_dw_ab_ppm": true_signed_dw,
            "fit_dw_ab_ppm": tp_fit.dw_ab_ppm,
            "dw_abs_err_ppm": (tp_fit.dw_ab_ppm - true_signed_dw).abs(),
            "reduced_chi2": tp_fit.chi2 / ((r1rho_pts.len().saturating_sub(5).max(1)) as f64),
            "passed": true,
        })
    );
}

#[test]
fn test_exchange_gate_empty_slice_small_sample_aicc_and_zero_dof_guards() {
    let default_gate = ExchangeGateConfig::default();

    // 1. Empty slice does not panic and returns non-significant verdict
    let empty_verdict = evaluate_exchange_gate(&[], &default_gate);
    assert!(!empty_verdict.significant_exchange);
    assert_eq!(empty_verdict.f_pvalue, 1.0);

    // 2. Mismatched array lengths do not panic and return non-significant verdict
    let mismatched = CpmgSeriesDataset {
        seq_id: 7,
        res_name: "ALA".to_string(),
        larmor_mhz: 60.8106,
        t_relax_s: 0.040,
        nu_cpmg_hz: vec![50.0, 100.0, 200.0],
        r2_eff_s_inv: vec![18.0, 15.0],
        r2_eff_err: vec![0.1, 0.1, 0.1],
    };
    assert!(!evaluate_exchange_gate(&[&mismatched], &default_gate).significant_exchange);

    // 3. Small-sample Hurvich-Tsai AICc (`N <= k_ex_params + 1` -> `N = 3, 4` for single-field
    // `k_ex_params = 3`) rejects exchange even when `max_f_pvalue = 1.0` (AIC-only gate)!
    let aic_only_gate = ExchangeGateConfig {
        min_delta_r2_eff_s_inv: 1.0,
        min_delta_aic: 2.0,
        max_f_pvalue: 1.0,
    };
    for n_pts in [3usize, 4usize] {
        let tiny = CpmgSeriesDataset {
            seq_id: 11,
            res_name: "VAL".to_string(),
            larmor_mhz: 60.8106,
            t_relax_s: 0.040,
            nu_cpmg_hz: (0..n_pts).map(|i| 50.0 * ((i + 1) as f64)).collect(),
            r2_eff_s_inv: (0..n_pts).map(|i| 22.0 - 3.5 * (i as f64)).collect(),
            r2_eff_err: vec![0.05; n_pts],
        };
        let v = evaluate_exchange_gate(&[&tiny], &aic_only_gate);
        assert!(
            !v.significant_exchange,
            "small-sample N={n_pts} <= k_ex+1=4 must be rejected even under AIC-only gate"
        );
    }

    // 4. True `dof = 0` when `total_pts <= n_params` (with `filter_by_exchange_gate = false`)
    let underdetermined = CpmgSeriesDataset {
        seq_id: 12,
        res_name: "LEU".to_string(),
        larmor_mhz: 60.8106,
        t_relax_s: 0.040,
        nu_cpmg_hz: vec![50.0, 200.0, 800.0],
        r2_eff_s_inv: vec![16.0, 14.0, 12.5],
        r2_eff_err: vec![0.1, 0.1, 0.1],
    };
    let cfg_no_gate = CpmgFitConfig {
        filter_by_exchange_gate: false,
        ..CpmgFitConfig::default()
    };
    let fit_zero_dof = fit_cpmg_global_cpu(&[underdetermined], &cfg_no_gate).unwrap();
    assert_eq!(fit_zero_dof.dof, 0);
    assert!(fit_zero_dof.reduced_chi2.is_finite());
}
