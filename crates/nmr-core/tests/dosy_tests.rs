//! Comprehensive integration tests for `nmr_core::dosy` (`T14.4`):
//! Bruker `difflist`/`acqus` (`STE` & `BPP-LED`) and Varian `procpar` (`gzlvl1`) ingestion,
//! mono-/bi-/tri-exponential Stejskal–Tanner Levenberg–Marquardt fitting with `AIC`/`BIC`
//! model-order selection and bootstrap 95% confidence intervals, Tikhonov `ILT` bimodal recovery,
//! pseudo-2D `(ppm, log10(D))` synthesis, and Stokes–Einstein hydrodynamics.

use std::f64::consts::PI;
use std::fs;

use nmr_core::dosy::{
    compute_b_values_s_m2, effective_diffusion_time_s, estimate_hydrodynamics,
    fit_stejskal_tanner_k, fit_stejskal_tanner_multi, parse_bruker_difflist, parse_bruker_dosy,
    parse_varian_procpar_dosy, read_bruker_dosy_dir, read_varian_dosy_dir, solve_tikhonov_ilt,
    stokes_einstein_radius_m, synthesize_dosy_grid_from_columns, synthesize_dosy_grid_from_peaks,
    BootstrapMethod, BrukerDosyOverrides, BrukerP30Mode, DosyFitConfig, DosyGridConfig,
    DosyNucleus, DosyPeakEntry, DosySequenceType, DosySolvent, GradientShape, IltConfig,
    IltLambdaStrategy, ModelSelectionCriterion, PowerLawCalibration, SoluteCompactnessClass,
    TikhonovRegularizer, VarianDosyOverrides, BOLTZMANN_J_PER_K, GAMMA_H1_RAD_S_T,
    GAUSS_PER_CM_TO_TESLA_PER_M,
};
use nmr_core::rng::Rng;

#[test]
fn ste_and_bppled_attenuation_factors_and_shapes_exact() {
    let grads = vec![5.0, 25.0, 50.0]; // G/cm
    let big_delta = 0.100; // 100 ms
    let little_delta = 0.0024; // 2.4 ms
    let tau = 0.0002; // 0.2 ms

    // 1. Rectangular STE: Delta_eff = Delta - delta / 3
    let b_ste_rect = compute_b_values_s_m2(
        &grads,
        DosyNucleus::H1,
        DosySequenceType::Ste,
        GradientShape::Rectangular,
        big_delta,
        little_delta,
        0.0,
    )
    .expect("STE rect b");
    let expected_deff_ste = big_delta - little_delta / 3.0;
    assert!(
        (effective_diffusion_time_s(DosySequenceType::Ste, big_delta, little_delta, 0.0).unwrap()
            - expected_deff_ste)
            .abs()
            < 1.0e-15
    );
    for (g, b) in grads.iter().zip(b_ste_rect.iter()) {
        let g_tm = g * GAUSS_PER_CM_TO_TESLA_PER_M;
        let b_ref = (GAMMA_H1_RAD_S_T * g_tm * little_delta).powi(2) * expected_deff_ste;
        assert!((b - b_ref).abs() / b_ref < 1.0e-14);
    }

    // 2. Sine-bell BPP-LED: g_eff = (2/pi) * g, Delta_eff = Delta - delta/3 - tau/2
    let b_bpp_sine = compute_b_values_s_m2(
        &grads,
        DosyNucleus::H1,
        DosySequenceType::BppLed,
        GradientShape::SineBell,
        big_delta,
        little_delta,
        tau,
    )
    .expect("BPP-LED sine b");
    let expected_deff_bpp = big_delta - little_delta / 3.0 - 0.5 * tau;
    for (g, b) in grads.iter().zip(b_bpp_sine.iter()) {
        let g_eff = (2.0 / PI) * (g * GAUSS_PER_CM_TO_TESLA_PER_M);
        let b_ref = (GAMMA_H1_RAD_S_T * g_eff * little_delta).powi(2) * expected_deff_bpp;
        assert!((b - b_ref).abs() / b_ref < 1.0e-14);
    }
}

#[test]
fn bruker_difflist_and_acqus_bppled_and_ste_parsing() {
    let difflist = "\
# Bruker difflist diffusion ramp (G/cm)
  2.675
 10.700
 21.400
 37.450
 50.825
";
    let parsed_list = parse_bruker_difflist(difflist).expect("difflist parse");
    assert_eq!(parsed_list, vec![2.675, 10.700, 21.400, 37.450, 50.825]);

    // Build realistic Bruker JCAMP `acqus` with D[16]=0.0002 s, D[20]=0.12 s, P[30]=1100 us (bipolar lobe -> delta=2.2 ms), GPNAM[6]=<SINE.100>
    let mut d_arr = ["0.0"; 32];
    d_arr[16] = "0.0002";
    d_arr[20] = "0.12";
    let mut p_arr = ["0.0"; 32];
    p_arr[30] = "1100.0";
    let mut gpnam_arr = ["<>"; 10];
    gpnam_arr[6] = "<SINE.100>";
    gpnam_arr[7] = "<SMSQ10.100>";

    let acqus = format!(
        "##TITLE= Parameter file\n\
         ##$PULPROG= <ledbpgp2s>\n\
         ##$NUC1= <1H>\n\
         ##$D= (0..31)\n{}\n\
         ##$P= (0..31)\n{}\n\
         ##$GPNAM= (0..9)\n{}\n\
         ##END=\n",
        d_arr.join(" "),
        p_arr.join(" "),
        gpnam_arr.join(" "),
    );

    let params = parse_bruker_dosy(&acqus, difflist, &BrukerDosyOverrides::default())
        .expect("parse Bruker BPP-LED");
    assert_eq!(params.sequence_type, DosySequenceType::BppLed);
    assert_eq!(params.gradient_shape, GradientShape::SineBell);
    assert!((params.big_delta_s - 0.12).abs() < 1.0e-12);
    // ledbpgp2s is bipolar (`bpgp`), so P30 = 1100 us => total little_delta = 2 * 1.1 ms = 2.2 ms
    assert!((params.little_delta_s - 0.0022).abs() < 1.0e-12);
    assert!((params.tau_s - 0.0002).abs() < 1.0e-12);
    assert_eq!(params.b_values_s_m2.len(), 5);

    // Also verify monopolar STE override (`P30` = full delta, rectangular) and on-disk directory reader
    let tmp_dir = std::env::temp_dir().join("nmr_core_dosy_bruker_test_dir");
    let _ = fs::remove_dir_all(&tmp_dir);
    fs::create_dir_all(&tmp_dir).expect("mkdir");
    fs::write(tmp_dir.join("difflist"), difflist).expect("write difflist");
    fs::write(tmp_dir.join("acqus"), &acqus).expect("write acqus");

    let ste_ovr = BrukerDosyOverrides {
        sequence_type: Some(DosySequenceType::Ste),
        p30_mode: Some(BrukerP30Mode::FullDeltaMonopolar),
        gradient_shape: Some(GradientShape::Rectangular),
        ..Default::default()
    };
    let dir_params = read_bruker_dosy_dir(&tmp_dir, &ste_ovr).expect("read_bruker_dosy_dir");
    let _ = fs::remove_dir_all(&tmp_dir);
    assert_eq!(dir_params.sequence_type, DosySequenceType::Ste);
    assert!((dir_params.little_delta_s - 0.0011).abs() < 1.0e-12);
    assert_eq!(dir_params.tau_s, 0.0);
}

#[test]
fn varian_procpar_gzlvl1_and_dac_calibration_parsing() {
    let procpar = "\
seqfil 2 2 8 0 0 2 1 0 1 64
1 \"Dbppste\"
0
tn 2 2 8 0 0 2 1 0 1 64
1 \"H1\"
0
gshape 2 2 8 0 0 2 1 0 1 64
1 \"s\"
0
gt1 1 1 1 0 0 2 1 0 1 64
1 0.0020
0
del 1 1 1 0 0 2 1 0 1 64
1 0.080
0
tau 1 1 1 0 0 2 1 0 1 64
1 0.00015
0
gcal 1 1 1 0 0 2 1 0 1 64
1 0.0020
0
gzlvl1 1 1 32767 -32768 0 2 1 0 1 64
6 1000 5000 10000 15000 22000 30000
0
";

    let params = parse_varian_procpar_dosy(procpar, &VarianDosyOverrides::default())
        .expect("parse Varian procpar");
    assert_eq!(params.sequence_type, DosySequenceType::BppLed);
    assert_eq!(params.gradient_shape, GradientShape::SineBell);
    assert!((params.little_delta_s - 0.0020).abs() < 1.0e-12);
    assert!((params.big_delta_s - 0.080).abs() < 1.0e-12);
    assert!((params.tau_s - 0.00015).abs() < 1.0e-12);
    assert_eq!(
        params.gradients_g_per_cm,
        vec![2.0, 10.0, 20.0, 30.0, 44.0, 60.0]
    );

    // Verify explicit DAC-to-G/cm override + directory reader
    let tmp_dir = std::env::temp_dir().join("nmr_core_dosy_varian_test_dir");
    let _ = fs::remove_dir_all(&tmp_dir);
    fs::create_dir_all(&tmp_dir).expect("mkdir");
    fs::write(tmp_dir.join("procpar"), procpar).expect("write procpar");

    let ovr = VarianDosyOverrides {
        dac_to_g_per_cm: Some(0.0015),
        ..Default::default()
    };
    let p2 = read_varian_dosy_dir(&tmp_dir, &ovr).expect("read_varian_dosy_dir");
    let _ = fs::remove_dir_all(&tmp_dir);
    assert!((p2.gradients_g_per_cm[5] - 45.0).abs() < 1.0e-12);
}

#[test]
fn mono_bi_tri_stejskal_tanner_bic_selection_and_bootstrap_cis() {
    // 24-point BPP-LED linear gradient ramp covering 2 to 60 G/cm
    let grads: Vec<f64> = (0..24).map(|i| 2.0 + (i as f64) * (58.0 / 23.0)).collect();
    let b_vals = compute_b_values_s_m2(
        &grads,
        DosyNucleus::H1,
        DosySequenceType::BppLed,
        GradientShape::Rectangular,
        0.100,
        0.0020,
        0.0002,
    )
    .expect("b_vals");

    let d1_true = 2.0e-10_f64;
    let d2_true = 8.5e-10_f64;

    // 1. Noiseless pure & mixed peaks: verify BIC never splits pure peaks on float precision
    let y_pure_slow: Vec<f64> = b_vals
        .iter()
        .map(|&b| 100.0 * (-b * d1_true).exp())
        .collect();
    let y_pure_fast: Vec<f64> = b_vals
        .iter()
        .map(|&b| 100.0 * (-b * d2_true).exp())
        .collect();
    let y_mixed: Vec<f64> = b_vals
        .iter()
        .map(|&b| 55.0 * (-b * d1_true).exp() + 45.0 * (-b * d2_true).exp())
        .collect();

    let cfg_fast = DosyFitConfig {
        bootstrap_samples: 0,
        ..DosyFitConfig::default()
    };
    let s1 = fit_stejskal_tanner_multi(&b_vals, &y_pure_slow, &cfg_fast).unwrap();
    let s2 = fit_stejskal_tanner_multi(&b_vals, &y_pure_fast, &cfg_fast).unwrap();
    let sm = fit_stejskal_tanner_multi(&b_vals, &y_mixed, &cfg_fast).unwrap();
    assert_eq!(s1.selected_k, 1);
    assert_eq!(s2.selected_k, 1);
    assert_eq!(sm.selected_k, 2);
    assert!((sm.best_fit.components[0].d_m2_s - d1_true).abs() / d1_true < 1.0e-5);
    assert!((sm.best_fit.components[1].d_m2_s - d2_true).abs() / d2_true < 1.0e-5);

    // 2. Multi-seed noisy evaluation (SNR = 80 on mixed, SNR = 40 on pure): >= 95% BIC order accuracy
    let mut pure_k1_correct = 0usize;
    let mut mixed_k2_correct = 0usize;
    let mut sum_err_pure = 0.0_f64;
    let mut sum_err_d1 = 0.0_f64;
    let mut sum_err_d2 = 0.0_f64;
    let n_trials = 40usize;
    for seed in 0..n_trials {
        let mut rng = Rng::new(10_000 + seed as u64);
        let noisy_pure: Vec<f64> = y_pure_slow
            .iter()
            .map(|&y| y + rng.normal(0.0, 2.5)) // SNR = 40
            .collect();
        let noisy_mix: Vec<f64> = y_mixed
            .iter()
            .map(|&y| y + rng.normal(0.0, 0.8)) // per-component SNR ~ 56-68
            .collect();

        let fit_p = fit_stejskal_tanner_multi(&b_vals, &noisy_pure, &cfg_fast).unwrap();
        if fit_p.selected_k == 1 {
            pure_k1_correct += 1;
            let rel_err = (fit_p.best_fit.components[0].d_m2_s - d1_true).abs() / d1_true;
            assert!(rel_err < 0.08, "noisy pure D error {rel_err}");
            sum_err_pure += rel_err;
        }

        let fit_m = fit_stejskal_tanner_multi(&b_vals, &noisy_mix, &cfg_fast).unwrap();
        if fit_m.selected_k == 2 {
            mixed_k2_correct += 1;
            let err1 = (fit_m.best_fit.components[0].d_m2_s - d1_true).abs() / d1_true;
            let err2 = (fit_m.best_fit.components[1].d_m2_s - d2_true).abs() / d2_true;
            assert!(
                err1 < 0.20 && err2 < 0.22,
                "mixed D errors ({err1}, {err2})"
            );
            sum_err_d1 += err1;
            sum_err_d2 += err2;
        }
    }
    assert!(
        pure_k1_correct >= 38,
        "pure single-species BIC K=1 rate {pure_k1_correct}/{n_trials} < 95%"
    );
    assert!(
        mixed_k2_correct >= 38,
        "mixed bi-exponential BIC K=2 rate {mixed_k2_correct}/{n_trials} < 95%"
    );
    assert!(sum_err_pure / (pure_k1_correct as f64) < 0.025);
    assert!(sum_err_d1 / (mixed_k2_correct as f64) < 0.045);
    assert!(sum_err_d2 / (mixed_k2_correct as f64) < 0.080);

    // 3. Bootstrap 95% confidence intervals (both Residual and Parametric) on mixed peak
    let mut rng_ci = Rng::new(424242);
    let y_mix_ci: Vec<f64> = y_mixed
        .iter()
        .map(|&y| y + rng_ci.normal(0.0, 0.4))
        .collect();
    for method in [BootstrapMethod::Residual, BootstrapMethod::Parametric] {
        let cfg_ci = DosyFitConfig {
            bootstrap_samples: 250,
            bootstrap_method: method,
            rng_seed: 20261114,
            ..DosyFitConfig::default()
        };
        let sel_ci = fit_stejskal_tanner_multi(&b_vals, &y_mix_ci, &cfg_ci).unwrap();
        assert_eq!(sel_ci.selected_k, 2);
        let c0 = &sel_ci.best_fit.components[0];
        let c1 = &sel_ci.best_fit.components[1];
        assert!(
            c0.ci95_d_m2_s.0 < c0.ci95_d_m2_s.1
                && c0.ci95_d_m2_s.0 <= d1_true * 1.05
                && c0.ci95_d_m2_s.1 >= d1_true * 0.95,
            "D1 CI {:?} failed to bracket {d1_true:e}",
            c0.ci95_d_m2_s
        );
        assert!(
            c1.ci95_d_m2_s.0 < c1.ci95_d_m2_s.1
                && c1.ci95_d_m2_s.0 <= d2_true * 1.05
                && c1.ci95_d_m2_s.1 >= d2_true * 0.95,
            "D2 CI {:?} failed to bracket {d2_true:e}",
            c1.ci95_d_m2_s
        );
    }

    // 4. Tri-exponential (K = 3) recovery on wide-dynamic-range b grid
    let b_wide: Vec<f64> = (0..32)
        .map(|i| 2.0e7 * (450.0_f64).powf((i as f64) / 31.0))
        .collect();
    let d_tri = [1.2e-10, 5.0e-10, 2.2e-9];
    let y_tri: Vec<f64> = b_wide
        .iter()
        .map(|&b| {
            35.0 * (-b * d_tri[0]).exp()
                + 35.0 * (-b * d_tri[1]).exp()
                + 30.0 * (-b * d_tri[2]).exp()
        })
        .collect();
    let tri_fit = fit_stejskal_tanner_k(&b_wide, &y_tri, 3, &cfg_fast, false).unwrap();
    for (k, &dt) in d_tri.iter().enumerate() {
        let rel = (tri_fit.components[k].d_m2_s - dt).abs() / dt;
        assert!(rel < 0.02, "tri-exponential component {k} rel err {rel}");
    }
    let tri_sel = fit_stejskal_tanner_multi(
        &b_wide,
        &y_tri,
        &DosyFitConfig {
            criterion: ModelSelectionCriterion::Bic,
            bootstrap_samples: 0,
            ..DosyFitConfig::default()
        },
    )
    .unwrap();
    assert_eq!(tri_sel.selected_k, 3);
    let max_tri_rel_err = d_tri
        .iter()
        .enumerate()
        .map(|(k, &dt)| (tri_fit.components[k].d_m2_s - dt).abs() / dt)
        .fold(0.0_f64, f64::max);
    println!(
        "W2_METRIC_JSON: {}",
        serde_json::json!({
            "kind": "dosy_cpu_discrete",
            "n_trials": n_trials,
            "pure_k1_correct": pure_k1_correct,
            "mixed_k2_correct": mixed_k2_correct,
            "mean_pure_rel_err": sum_err_pure / (pure_k1_correct as f64),
            "mean_mix_d1_rel_err": sum_err_d1 / (mixed_k2_correct as f64),
            "mean_mix_d2_rel_err": sum_err_d2 / (mixed_k2_correct as f64),
            "tri_selected_k": tri_sel.selected_k,
            "max_tri_rel_err": max_tri_rel_err,
            "passed": true,
        })
    );
}

#[test]
fn tikhonov_ilt_bimodal_recovery_gcv_and_lcurve() {
    let b_vals: Vec<f64> = (0..28)
        .map(|i| 3.0e7 * (300.0_f64).powf((i as f64) / 27.0))
        .collect();
    let d1 = 2.0e-10_f64;
    let d2 = 8.5e-10_f64;
    let mut rng = Rng::new(778899);
    let y: Vec<f64> = b_vals
        .iter()
        .map(|&b| 50.0 * (-b * d1).exp() + 50.0 * (-b * d2).exp() + rng.normal(0.0, 0.15))
        .collect();

    for strat in [IltLambdaStrategy::Gcv, IltLambdaStrategy::LCurve] {
        let cfg = IltConfig {
            n_grid: 72,
            d_min_m2_s: 3.0e-11,
            d_max_m2_s: 4.0e-9,
            regularizer: TikhonovRegularizer::SecondDerivative,
            lambda_strategy: strat,
            lambda_min: 1.0e-5,
            lambda_max: 1.0,
            n_lambda_sweep: 24,
        };
        let res = solve_tikhonov_ilt(&b_vals, &y, &cfg).expect("solve_tikhonov_ilt");
        assert!(res.spectrum.iter().all(|&v| v >= 0.0));
        assert!(res.caveat.contains("ill-posed"));
        assert_eq!(res.lambda_sweep.len(), 24);
        assert!(
            res.modes.len() >= 2,
            "expected bimodal ILT recovery under {strat:?}, got {:?}",
            res.modes
        );

        // Sort top 2 modes by area and check proximity to D1 = 2.0e-10 and D2 = 8.5e-10
        let mut top2 = res.modes.clone();
        top2.sort_by(|a, b| b.area_fraction.total_cmp(&a.area_fraction));
        top2.truncate(2);
        top2.sort_by(|a, b| a.d_m2_s.total_cmp(&b.d_m2_s));

        let log_err_1 = (top2[0].log10_d - d1.log10()).abs();
        let log_err_2 = (top2[1].log10_d - d2.log10()).abs();
        assert!(
            log_err_1 < 0.08 && log_err_2 < 0.08,
            "ILT mode log10(D) errors ({log_err_1:.3}, {log_err_2:.3}) too large under {strat:?}"
        );
        assert!((top2[0].area_fraction - 0.50).abs() < 0.12);
        assert!((top2[1].area_fraction - 0.50).abs() < 0.12);
        let strat_label = match strat {
            IltLambdaStrategy::Gcv => "GCV",
            IltLambdaStrategy::LCurve => "LCurve",
            _ => "other",
        };
        println!(
            "W2_METRIC_JSON: {}",
            serde_json::json!({
                "kind": "dosy_ilt_bimodal",
                "strategy": strat_label,
                "selected_lambda": res.selected_lambda,
                "n_modes": res.modes.len(),
                "log10_err_slow": log_err_1,
                "log10_err_fast": log_err_2,
                "area_slow": top2[0].area_fraction,
                "area_fast": top2[1].area_fraction,
                "passed": true,
            })
        );
    }
}

#[test]
fn pseudo2d_dosy_grid_synthesis_alignment() {
    let ppm_axis: Vec<f64> = (0..101).map(|i| (i as f64) * 0.1).collect(); // 0.0 .. 10.0 ppm
    let b_vals: Vec<f64> = (1..=16).map(|i| (i as f64) * 4.0e8).collect();
    let y_mix: Vec<f64> = b_vals
        .iter()
        .map(|&b| 60.0 * (-b * 2.0e-10).exp() + 40.0 * (-b * 8.5e-10).exp())
        .collect();

    let fit_cfg = DosyFitConfig {
        bootstrap_samples: 40,
        ..DosyFitConfig::default()
    };
    let sel = fit_stejskal_tanner_multi(&b_vals, &y_mix, &fit_cfg).unwrap();
    let ilt = solve_tikhonov_ilt(&b_vals, &y_mix, &IltConfig::default()).unwrap();

    let mut top_1d = vec![0.0; ppm_axis.len()];
    top_1d[35] = 100.0; // 3.5 ppm

    let pk = DosyPeakEntry {
        label: "3.5 ppm".to_string(),
        center_ppm: 3.5,
        fwhm_ppm: 0.12,
        intensities: y_mix,
        multi_exp: sel.clone(),
        ilt: Some(ilt),
    };

    let grid_cfg = DosyGridConfig {
        n_log10_d: 101,
        log10_d_min: -10.5,
        log10_d_max: -8.5,
        min_sigma_log10_d: 0.03,
        max_sigma_log10_d: 0.20,
    };

    let grid_disc = synthesize_dosy_grid_from_peaks(
        &ppm_axis,
        &top_1d,
        std::slice::from_ref(&pk),
        &grid_cfg,
        false,
    )
    .unwrap();
    let grid_ilt =
        synthesize_dosy_grid_from_peaks(&ppm_axis, &top_1d, &[pk], &grid_cfg, true).unwrap();

    // Check slice at 3.5 ppm (col 35) has clear non-zero intensity near both log10(2e-10) = -9.70 and log10(8.5e-10) = -9.07
    let row_slow = grid_disc
        .log10_d_axis
        .iter()
        .enumerate()
        .min_by(|a, b| (a.1 - (-9.699)).abs().total_cmp(&(b.1 - (-9.699)).abs()))
        .unwrap()
        .0;
    let row_fast = grid_disc
        .log10_d_axis
        .iter()
        .enumerate()
        .min_by(|a, b| (a.1 - (-9.071)).abs().total_cmp(&(b.1 - (-9.071)).abs()))
        .unwrap()
        .0;
    let row_mid = (row_slow + row_fast) / 2;

    assert!(grid_disc.get(row_slow, 35) > grid_disc.get(row_mid, 35) * 2.0);
    assert!(grid_disc.get(row_fast, 35) > grid_disc.get(row_mid, 35) * 2.0);
    assert!(grid_ilt.get(row_slow, 35) > 0.0 && grid_ilt.get(row_fast, 35) > 0.0);

    // Also test column-wise synthesizer
    let mut col_fits = vec![None; ppm_axis.len()];
    col_fits[35] = Some(sel.best_fit);
    let grid_cols =
        synthesize_dosy_grid_from_columns(&ppm_axis, &top_1d, &col_fits, &grid_cfg).unwrap();
    assert!(grid_cols.get(row_slow, 35) > 0.0);
    assert_eq!(grid_cols.get(row_slow, 10), 0.0);
}

#[test]
fn stokes_einstein_hydrodynamics_lysozyme_small_molecules_and_solvents() {
    // 1. All 6 solvents have positive decreasing viscosity with increasing temperature
    for solv in DosySolvent::ALL {
        let eta_293 = solv.viscosity_pa_s(293.15).unwrap();
        let eta_298 = solv.viscosity_pa_s(298.15).unwrap();
        let eta_310 = solv.viscosity_pa_s(310.15).unwrap();
        assert!(
            eta_293 > eta_298 && eta_298 > eta_310 && eta_310 > 0.0,
            "non-monotonic viscosity for {:?}",
            solv
        );
        assert_eq!(DosySolvent::from_name(solv.label()), Some(solv));
    }

    // 2. Exact Stokes-Einstein identity check: R_h = k_B T / (6 pi eta D)
    let eta_h2o = DosySolvent::H2O.viscosity_pa_s(298.15).unwrap();
    let d_test = 5.0e-10;
    let rh = stokes_einstein_radius_m(d_test, 298.15, eta_h2o).unwrap();
    let rh_ref = (BOLTZMANN_J_PER_K * 298.15) / (6.0 * PI * eta_h2o * d_test);
    assert!((rh - rh_ref).abs() / rh_ref < 1.0e-15);

    // 3. Hen egg-white Lysozyme (MW = 14.3 kDa, D = 1.03e-10 m^2/s in D2O at 298.15 K):
    // Literature R_h = 19.0 - 20.5 Å
    let lyso = estimate_hydrodynamics(
        1.03e-10,
        DosySolvent::D2O,
        298.15,
        SoluteCompactnessClass::GlobularProtein,
    )
    .expect("lysozyme hydrodynamics");
    assert!(
        (18.8..=20.2).contains(&lyso.r_h_angstrom),
        "lysozyme R_h = {:.2} Å out of expected [18.8, 20.2] Å",
        lyso.r_h_angstrom
    );
    assert!(
        (lyso.estimated_mw_da - 14_300.0).abs() / 14_300.0 < 0.08,
        "lysozyme estimated MW = {:.0} Da",
        lyso.estimated_mw_da
    );

    // 4. Sucrose (MW = 342.3 Da, D = 4.35e-10 m^2/s in D2O at 298.15 K): R_h ~ 4.5 - 4.7 Å
    let sucr = estimate_hydrodynamics(
        4.35e-10,
        DosySolvent::D2O,
        298.15,
        SoluteCompactnessClass::SmallMolecule,
    )
    .expect("sucrose hydrodynamics");
    assert!(
        (4.4..=4.8).contains(&sucr.r_h_angstrom),
        "sucrose R_h = {:.2} Å",
        sucr.r_h_angstrom
    );
    assert!(
        (sucr.estimated_mw_da - 342.3).abs() / 342.3 < 0.10,
        "sucrose estimated MW = {:.1} Da",
        sucr.estimated_mw_da
    );

    // 5. Internal-standard log-log power-law calibration recovery
    let alpha_true = 1.45e-8_f64;
    let beta_true = 0.54_f64;
    let stds: Vec<(f64, f64)> = [150.0_f64, 300.0, 600.0, 1200.0]
        .into_iter()
        .map(|mw| (mw, alpha_true * mw.powf(-beta_true)))
        .collect();
    let cal = PowerLawCalibration::fit_from_standards(&stds).expect("fit_from_standards");
    assert!((cal.alpha - alpha_true).abs() / alpha_true < 1.0e-12);
    assert!((cal.beta - beta_true).abs() / beta_true < 1.0e-12);
    assert!(
        (cal.estimate_mw_da(cal.predict_d_m2_s(450.0).unwrap())
            .unwrap()
            - 450.0)
            .abs()
            < 1.0e-9
    );
    println!(
        "W2_METRIC_JSON: {}",
        serde_json::json!({
            "kind": "dosy_hydrodynamics",
            "lysozyme_rh_a": lyso.r_h_angstrom,
            "lysozyme_mw_da": lyso.estimated_mw_da,
            "sucrose_rh_a": sucr.r_h_angstrom,
            "sucrose_mw_da": sucr.estimated_mw_da,
            "passed": true,
        })
    );
}

#[test]
fn adversarial_dosy_error_paths_reject_corrupted_inputs() {
    // 1. Empty gradients, negative/NaN gradients, and unphysical timing (`Delta <= delta / 3`)
    assert!(compute_b_values_s_m2(
        &[],
        DosyNucleus::H1,
        DosySequenceType::BppLed,
        GradientShape::SineBell,
        0.10,
        0.002,
        0.0002
    )
    .is_err());
    assert!(compute_b_values_s_m2(
        &[2.0, f64::NAN, 10.0],
        DosyNucleus::H1,
        DosySequenceType::Ste,
        GradientShape::Rectangular,
        0.10,
        0.002,
        0.0
    )
    .is_err());
    assert!(effective_diffusion_time_s(DosySequenceType::Ste, 0.0005, 0.0030, 0.0).is_err());

    // 2. Malformed Bruker `difflist` and Varian `procpar`
    assert!(parse_bruker_difflist("").is_err());
    assert!(parse_bruker_difflist("2.0\nnot_a_float\n10.0\n").is_err());
    assert!(parse_varian_procpar_dosy("empty procpar", &VarianDosyOverrides::default()).is_err());

    // 3. Stejskal–Tanner & Tikhonov ILT dimension mismatches, too-few points, and NaN/Inf decays
    let b_ok = vec![1e8, 3e8, 6e8, 1e9, 2e9, 4e9];
    assert!(
        fit_stejskal_tanner_multi(&b_ok[..2], &[100.0, 80.0], &DosyFitConfig::default()).is_err()
    );
    assert!(
        fit_stejskal_tanner_multi(&b_ok, &[100.0, 80.0, 60.0], &DosyFitConfig::default()).is_err()
    );
    assert!(fit_stejskal_tanner_multi(
        &b_ok,
        &[100.0, 80.0, f64::NAN, 40.0, 20.0, 10.0],
        &DosyFitConfig::default()
    )
    .is_err());
    assert!(solve_tikhonov_ilt(
        &b_ok,
        &[100.0, f64::INFINITY, 60.0, 40.0, 20.0, 10.0],
        &IltConfig::default()
    )
    .is_err());

    // 4. Hydrodynamics & power-law calibration invalid parameters
    assert!(stokes_einstein_radius_m(-1.0e-10, 298.15, 1.0e-3).is_err());
    assert!(stokes_einstein_radius_m(5.0e-10, 0.0, 1.0e-3).is_err());
    assert!(DosySolvent::D2O.viscosity_pa_s(-10.0).is_err());
    assert!(PowerLawCalibration::fit_from_standards(&[(300.0, 4.0e-10)]).is_err());
}

#[test]
fn f29_bruker_oneshot_b_factor_water_skirt_mask_and_published_d_ref_corpus() {
    use nmr_core::dosy::{
        effective_diffusion_time_oneshot_s, is_in_dosy_d2o_water_skirt,
        lookup_published_dosy_d_ref, pick_dosy_peaks_1d, published_dosy_d_ref_corpus,
        DosyPeakPickConfig, DEFAULT_DOSY_D2O_WATER_EXCLUSION_PPM,
    };
    use nmr_core::nmr1d::{
        apply_apodization_1d, fft_1d, synthesize_fid_1d, zero_fill_pow2_1d, Apodization1D,
        Experiment1DKind, SyntheticLine1D,
    };

    // 1. Bruker Pelta et al. (2002) `<gdp_dosyoneshot_tlg>` Oneshot recognition (`2 * P30` bipolar + alpha = 0.2)
    let difflist = "5.35\n15.0\n28.5\n42.8\n53.5\n";
    let mut d_arr = ["0.0"; 32];
    d_arr[16] = "0.0002";
    d_arr[20] = "0.08";
    let mut p_arr = ["0.0"; 32];
    p_arr[1] = "8.58";
    p_arr[30] = "1100.0"; // half-bipolar lobe -> little_delta = 2.2 ms (NOT monopolar 1.1 ms!)
    let mut cnst_arr = ["1.0"; 32]; // Bruker fills unused CNST slots with 1.0!
    cnst_arr[14] = "0.2";
    cnst_arr[15] = "1.2";
    cnst_arr[16] = "0.8";
    cnst_arr[17] = "0.00021716";
    let mut gpnam_arr = ["<SINE.100>"; 8];
    gpnam_arr[0] = "<>";

    let acqus_oneshot = format!(
        "##TITLE= Oneshot DOSY\n\
         ##$PULPROG= <gdp_dosyoneshot_tlg>\n\
         ##$NUC1= <1H>\n\
         ##$CNST= (0..31)\n{}\n\
         ##$D= (0..31)\n{}\n\
         ##$P= (0..31)\n{}\n\
         ##$GPNAM= (0..7)\n{}\n\
         ##END=\n",
        cnst_arr.join(" "),
        d_arr.join(" "),
        p_arr.join(" "),
        gpnam_arr.join(" "),
    );

    let os_params = parse_bruker_dosy(&acqus_oneshot, difflist, &BrukerDosyOverrides::default())
        .expect("parse Oneshot DOSY");
    assert_eq!(os_params.sequence_type, DosySequenceType::BppLed);
    assert_eq!(os_params.gradient_shape, GradientShape::SineBell);
    assert!(
        (os_params.little_delta_s - 0.0022).abs() < 1e-12,
        "Oneshot must use bipolar 2 * P30 = 2.2 ms, got {}",
        os_params.little_delta_s
    );
    assert!((os_params.oneshot_alpha - 0.2).abs() < 1e-12);
    assert!((os_params.tau_s - 0.000_217_16).abs() < 1e-10);

    // Compare against the buggy monopolar STE b-values (`4x` under-scaled!)
    let monopolar_b = compute_b_values_s_m2(
        &os_params.gradients_g_per_cm,
        DosyNucleus::H1,
        DosySequenceType::Ste,
        GradientShape::SineBell,
        0.08,
        0.0011,
        0.0,
    )
    .unwrap();
    let ratio = os_params.b_values_s_m2[4] / monopolar_b[4];
    assert!(
        (ratio - 3.98).abs() < 0.05,
        "Oneshot bipolar b-factor must fix ~4x monopolar under-scaling, got ratio {ratio:.4}"
    );

    // Also check Bruker pulse-program `CNST[19] = 0.2` Oneshot variant and `alpha = 0` fallback
    let mut cnst19_arr = ["1.0"; 32];
    cnst19_arr[19] = "0.2";
    let acqus_c19 = format!(
        "##$PULPROG= <stebpgp1s19_oneshot>\n##$NUC1= <1H>\n##$CNST= (0..31)\n{}\n##$D= (0..31)\n{}\n##$P= (0..31)\n{}\n##$GPNAM= (0..7)\n{}\n",
        cnst19_arr.join(" "),
        d_arr.join(" "),
        p_arr.join(" "),
        gpnam_arr.join(" ")
    );
    let c19_params =
        parse_bruker_dosy(&acqus_c19, difflist, &BrukerDosyOverrides::default()).unwrap();
    assert!((c19_params.oneshot_alpha - 0.2).abs() < 1e-12);

    let alpha0_deff =
        effective_diffusion_time_oneshot_s(GradientShape::SineBell, 0.08, 0.0022, 0.0002, 0.0)
            .unwrap();
    let bppled_deff =
        effective_diffusion_time_s(DosySequenceType::BppLed, 0.08, 0.0022, 0.0002).unwrap();
    assert!((alpha0_deff - bppled_deff).abs() < 1e-15);
    assert!(
        effective_diffusion_time_oneshot_s(GradientShape::SineBell, 0.08, 0.0022, 0.0002, 1.0)
            .is_err()
    );

    // 2. Widened `[4.50, 4.98] ppm` D2O water-skirt exclusion suppresses `4.56..4.59 ppm` ripples
    assert_eq!(DEFAULT_DOSY_D2O_WATER_EXCLUSION_PPM, (4.50, 4.98));
    assert!(is_in_dosy_d2o_water_skirt(4.57));
    assert!(is_in_dosy_d2o_water_skirt(4.96));
    assert!(!is_in_dosy_d2o_water_skirt(4.45));

    let plane_lines = [
        SyntheticLine1D::lorentzian(4.79, 80.0, 2.5), // HDO center
        SyntheticLine1D::lorentzian(4.57, 4.0, 1.5),  // 4.57 ppm water-skirt ripple!
        SyntheticLine1D::lorentzian(3.65, 12.0, 1.4), // Solute peak A
        SyntheticLine1D::lorentzian(1.48, 15.0, 1.4), // Solute peak B
    ];
    let mut fid_plane = synthesize_fid_1d(
        "dosy_plane0",
        Experiment1DKind::Proton1D,
        4096,
        5000.0,
        500.0,
        4.70,
        &plane_lines,
    );
    apply_apodization_1d(
        &mut fid_plane,
        Apodization1D::Exponential { lb_hz: 1.0 },
        0.5,
    );
    zero_fill_pow2_1d(&mut fid_plane, 2);
    let spec_plane = fft_1d(&fid_plane);
    let picked = pick_dosy_peaks_1d(&spec_plane, &DosyPeakPickConfig::default());
    assert_eq!(
        picked.len(),
        2,
        "expected only 2 solute peaks outside [4.50, 4.98], got {picked:?}"
    );
    assert!(picked.iter().all(|p| !(4.50..=4.98).contains(&p.ppm)));

    // 3. Clean-room published literature `D_ref` corpus check
    let corpus = published_dosy_d_ref_corpus();
    assert!(corpus.len() >= 16);
    let hdo = lookup_published_dosy_d_ref("HDO", Some(DosySolvent::D2O)).unwrap();
    let glc = lookup_published_dosy_d_ref("D-Glucose", Some(DosySolvent::D2O)).unwrap();
    let tms = lookup_published_dosy_d_ref("TMS", Some(DosySolvent::CDCl3)).unwrap();
    let quin = lookup_published_dosy_d_ref("Quinine", Some(DosySolvent::DmsoD6)).unwrap();
    assert!((hdo.d_ref_m2_s - 1.902e-9).abs() < 1e-12);
    assert!((glc.d_ref_m2_s - 5.45e-10).abs() < 1e-12);
    assert!((tms.d_ref_m2_s - 2.25e-9).abs() < 1e-12);
    assert!((quin.d_ref_m2_s - 2.28e-10).abs() < 1e-12);
}
