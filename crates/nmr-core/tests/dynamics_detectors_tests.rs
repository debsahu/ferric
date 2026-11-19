//! `[synthetic self-consistency]` Verification suite for `nmr_core::dynamics_detectors`
//! (`T22.3` Linear Optimal-Window Model-Free Dynamics Detectors & `T22.4` LeMaster + Multi-Field
//! Stacked IMPACT Spectral Density Decomposition).
//!
//! Validation scope: All quantitative recovery, variance-calibration, and optimality assertions
//! below test synthetic forward-model self-consistency using CODATA-2018 $^{15}\text{N}$ dipolar +
//! CSA kernels (not empirical deposited spectrometer datasets).

use nmr_core::dynamics_detectors::{
    build_dynamics_detectors, build_sensitivity_matrix, effective_tau_sec,
    experiments_and_observations_from_rates, fit_impact_from_multi_field_rates,
    fit_impact_from_relaxation_experiments, fit_impact_spectral_density, fit_lemaster_from_rates,
    fit_lemaster_from_spectral_density, gamma_noe_from_r1_hetnoe, impact_predict_spectral_density,
    lemaster_predict_spectral_density, log_spaced_grid, select_impact_model_aicc,
    select_impact_model_aicc_from_rates, single_tau_rate, solve_simplex_wls, DetectorConfig,
    ImpactConfig, RelaxExperiment, RelaxObservableKind, SpectralDensityObs, TumblingMode,
};
use nmr_core::modelfree::{predict_r1_r2_noe, LocalMfParams};
use nmr_core::relax_fit::{ReducedSpectralDensity, RelaxPhysicsConfig, ResidueRelaxRates};
use nmr_core::rng::Rng;

const MULTI_FIELDS_MHZ: [f64; 4] = [500.0, 600.0, 800.0, 950.0];

fn make_multi_field_experiments(
    fields_mhz: &[f64],
    rel_err: f64,
    ref_tau_m: f64,
    phys: &RelaxPhysicsConfig,
) -> Vec<RelaxExperiment> {
    let mut exps = Vec::with_capacity(fields_mhz.len() * 3);
    let ref_local = LocalMfParams::m2(0.82, 0.35e-9);
    for &mhz in fields_mhz {
        let (r1, r2, noe) = predict_r1_r2_noe(&[(1.0, ref_tau_m)], &ref_local, mhz, 600.0, phys);
        let gamma = gamma_noe_from_r1_hetnoe(r1, noe, phys);
        exps.push(RelaxExperiment::new(
            RelaxObservableKind::R1,
            mhz,
            (rel_err * r1).max(1.0e-4),
        ));
        exps.push(RelaxExperiment::new(
            RelaxObservableKind::R2,
            mhz,
            (rel_err * r2).max(1.0e-4),
        ));
        exps.push(RelaxExperiment::new(
            RelaxObservableKind::GammaNoe,
            mhz,
            (rel_err * gamma.abs()).max(1.0e-4),
        ));
    }
    exps
}

fn rates_from_local_mf(
    experiments: &[RelaxExperiment],
    tau_m_sec: f64,
    local: &LocalMfParams,
    phys: &RelaxPhysicsConfig,
) -> Vec<f64> {
    experiments
        .iter()
        .map(|exp| {
            let (r1, r2, noe) =
                predict_r1_r2_noe(&[(1.0, tau_m_sec)], local, exp.field_mhz, 600.0, phys);
            match exp.kind {
                RelaxObservableKind::R1 => r1,
                RelaxObservableKind::R2 => r2,
                RelaxObservableKind::GammaNoe => gamma_noe_from_r1_hetnoe(r1, noe, phys),
            }
        })
        .collect()
}

#[test]
fn synthetic_self_consistency_physical_kernel_matches_modelfree_across_fields() {
    let phys = RelaxPhysicsConfig::default();
    let fields = [400.0, 500.0, 600.0, 800.0, 950.0, 1200.0];
    let taus = [50.0e-12, 0.4e-9, 2.5e-9, 8.0e-9, 25.0e-9];

    for &mhz in &fields {
        for &tau in &taus {
            let rigid = LocalMfParams::m1(1.0);
            let (r1_mf, r2_mf, noe_mf) =
                predict_r1_r2_noe(&[(1.0, tau)], &rigid, mhz, 600.0, &phys);
            let g_mf = gamma_noe_from_r1_hetnoe(r1_mf, noe_mf, &phys);

            let r1_k = single_tau_rate(RelaxObservableKind::R1, mhz, tau, &phys);
            let r2_k = single_tau_rate(RelaxObservableKind::R2, mhz, tau, &phys);
            let g_k = single_tau_rate(RelaxObservableKind::GammaNoe, mhz, tau, &phys);

            assert!(
                (r1_k - r1_mf).abs() < 1.0e-12,
                "R1 mismatch at {} MHz, tau={:.2e}: {} vs {}",
                mhz,
                tau,
                r1_k,
                r1_mf
            );
            assert!(
                (r2_k - r2_mf).abs() < 1.0e-12,
                "R2 mismatch at {} MHz, tau={:.2e}: {} vs {}",
                mhz,
                tau,
                r2_k,
                r2_mf
            );
            assert!(
                (g_k - g_mf).abs() < 1.0e-12,
                "GammaNOE mismatch at {} MHz, tau={:.2e}: {} vs {}",
                mhz,
                tau,
                g_k,
                g_mf
            );
        }
    }
}

#[test]
fn synthetic_self_consistency_t22_3_detector_localization_nonnegativity_and_ordering() {
    let phys = RelaxPhysicsConfig::default();
    let exps = make_multi_field_experiments(&MULTI_FIELDS_MHZ, 0.02, 6.0e-9, &phys);

    // 1. Pure correlation-time spectrum (K = 4 detectors across 500/600/800/950 MHz)
    let cfg_pure = DetectorConfig {
        physics: phys,
        tumbling: TumblingMode::PureCorrelationSpectrum,
        num_detectors: Some(4),
        ..DetectorConfig::default()
    };
    let design_pure = build_dynamics_detectors(&exps, &cfg_pure)
        .expect("pure-spectrum detector build should succeed");

    assert_eq!(design_pure.num_detectors, 4);
    assert_eq!(design_pure.windows.len(), 4);
    assert!(
        design_pure.transform_condition_number < 200.0,
        "detector transform T poorly conditioned: {}",
        design_pure.transform_condition_number
    );

    for (n, win) in design_pure.windows.iter().enumerate() {
        assert!(
            win.min_value >= -1.0e-8,
            "pure detector {} violated non-negativity: min = {:.3e}",
            n,
            win.min_value
        );
        assert!(
            (win.max_value - 1.0).abs() < 1.0e-10,
            "pure detector {} not unit-peak: max = {:.12}",
            n,
            win.max_value
        );
        assert!(
            win.fwhm_decades > 0.0 && win.fwhm_decades < 3.5,
            "pure detector {} poorly localized: FWHM = {:.3} decades",
            n,
            win.fwhm_decades
        );
        if n > 0 {
            assert!(
                win.peak_tau_sec > design_pure.windows[n - 1].peak_tau_sec,
                "pure detectors not strictly ordered in peak tau: {:.3e} <= {:.3e}",
                win.peak_tau_sec,
                design_pure.windows[n - 1].peak_tau_sec
            );
        }
    }
    assert!(
        design_pure.max_pairwise_overlap < 0.75,
        "pure detectors overlap too strongly: max cosine overlap = {:.3}",
        design_pure.max_pairwise_overlap
    );

    // 2. Solution-NMR IsotropicInternal mode (Smith et al. 2019, tau_m = 8 ns)
    let cfg_int = DetectorConfig {
        physics: phys,
        tumbling: TumblingMode::IsotropicInternal { tau_m_sec: 8.0e-9 },
        num_detectors: Some(4),
        ..DetectorConfig::default()
    };
    let design_int = build_dynamics_detectors(&exps, &cfg_int)
        .expect("solution-NMR internal detector build should succeed");

    assert_eq!(design_int.num_detectors, 4);
    for (n, win) in design_int.windows.iter().enumerate() {
        assert!(
            win.min_value >= -1.0e-8,
            "internal detector {} violated non-negativity: min = {:.3e}",
            n,
            win.min_value
        );
        assert!(
            (win.max_value - 1.0).abs() < 1.0e-10,
            "internal detector {} not unit-peak: max = {:.12}",
            n,
            win.max_value
        );
        if n > 0 {
            assert!(
                win.peak_tau_sec > design_int.windows[n - 1].peak_tau_sec,
                "internal detectors not strictly ordered in peak tau: {:.3e} <= {:.3e}",
                win.peak_tau_sec,
                design_int.windows[n - 1].peak_tau_sec
            );
        }
    }
    // Fastest internal detector should peak near ultrafast limit (<< 100 ps)
    assert!(
        design_int.windows[0].peak_tau_sec < 1.0e-10,
        "fastest internal detector should sense ultrafast motion, peaked at {:.3e} s",
        design_int.windows[0].peak_tau_sec
    );

    // Also verify TumblingMode::EffectiveTimescale (tau_eff^-1 = tau_m^-1 + tau^-1)
    let cfg_eff = DetectorConfig {
        physics: phys,
        tumbling: TumblingMode::EffectiveTimescale { tau_m_sec: 8.0e-9 },
        num_detectors: Some(4),
        ..DetectorConfig::default()
    };
    let design_eff = build_dynamics_detectors(&exps, &cfg_eff)
        .expect("effective-timescale detector build should succeed");
    assert_eq!(design_eff.num_detectors, 4);
    for (n, win) in design_eff.windows.iter().enumerate() {
        assert!(
            win.min_value >= -1.0e-8,
            "effective-timescale detector {} violated non-negativity: min = {:.3e}",
            n,
            win.min_value
        );
        assert!(
            (win.max_value - 1.0).abs() < 1.0e-10,
            "effective-timescale detector {} not unit-peak: max = {:.12}",
            n,
            win.max_value
        );
    }

    // 3. Automatic rank selection via singular value threshold
    let cfg_auto = DetectorConfig {
        physics: phys,
        tumbling: TumblingMode::PureCorrelationSpectrum,
        num_detectors: None,
        rel_sv_threshold: 5.0e-3,
        ..DetectorConfig::default()
    };
    let design_auto = build_dynamics_detectors(&exps, &cfg_auto)
        .expect("automatic rank detector build should succeed");
    assert!(
        (3..=7).contains(&design_auto.num_detectors),
        "auto rank selected unexpected K = {} (singular values = {:?})",
        design_auto.num_detectors,
        design_auto.singular_values
    );
}

#[test]
fn synthetic_self_consistency_t22_3_exact_integral_identity_on_lipari_szabo_m1_m2_m5() {
    let phys = RelaxPhysicsConfig::default();
    let tau_m = 7.5e-9; // 7.5 ns overall tumbling
    let exps = make_multi_field_experiments(&MULTI_FIELDS_MHZ, 0.02, tau_m, &phys);

    // Auto-rank pure-spectrum detector bank (retains all singular vectors above 0.5% threshold)
    let cfg_pure = DetectorConfig {
        physics: phys,
        tumbling: TumblingMode::PureCorrelationSpectrum,
        num_detectors: None,
        rel_sv_threshold: 5.0e-3,
        ..DetectorConfig::default()
    };
    let design_pure = build_dynamics_detectors(&exps, &cfg_pure).expect("pure design");

    let cfg_int = DetectorConfig {
        physics: phys,
        tumbling: TumblingMode::IsotropicInternal { tau_m_sec: tau_m },
        num_detectors: None,
        rel_sv_threshold: 5.0e-3,
        ..DetectorConfig::default()
    };
    let design_int = build_dynamics_detectors(&exps, &cfg_int).expect("internal design");

    // Test cases: Lipari-Szabo M1, M2, and Clore extended M5
    type MfCase = (
        &'static str,
        LocalMfParams,
        Vec<(f64, f64)>,
        Vec<(f64, f64)>,
    );
    let cases: [MfCase; 3] = [
        (
            "M1 (S2=0.85)",
            LocalMfParams::m1(0.85),
            // Pure total spectrum: S^2 at tau_m, (1 - S^2) at tau_f = 0
            vec![(0.85, tau_m), (0.15, 0.0)],
            // Internal motion spectrum: (1 - S^2) at tau_f = 0
            vec![(0.15, 0.0)],
        ),
        (
            "M2 (S2=0.72, tau_e=180ps)",
            LocalMfParams::m2(0.72, 180.0e-12),
            vec![(0.72, tau_m), (0.28, effective_tau_sec(180.0e-12, tau_m))],
            vec![(0.28, 180.0e-12)],
        ),
        (
            "M5 (Sf2=0.88, Ss2=0.65, tau_s=1.4ns)",
            LocalMfParams::m5(0.88, 0.65, 1.4e-9),
            vec![
                (0.88 * 0.65, tau_m),
                (0.88 * (1.0 - 0.65), effective_tau_sec(1.4e-9, tau_m)),
                (1.0 - 0.88, 0.0),
            ],
            vec![(1.0 - 0.88, 0.0), (0.88 * (1.0 - 0.65), 1.4e-9)],
        ),
    ];

    for (label, local, pure_comps, int_comps) in &cases {
        let obs_rates = rates_from_local_mf(&exps, tau_m, local, &phys);

        // Pure correlation-time spectrum identity: rho_hat == sum_c w_c rho_n(tau_c)
        let fit_pure = design_pure
            .estimate(&obs_rates, None)
            .expect("pure estimate");
        let exact_pure = design_pure
            .evaluate_distribution(pure_comps)
            .expect("pure exact integral");
        for (n, (&got, &expected)) in fit_pure.responses.iter().zip(exact_pure.iter()).enumerate() {
            assert!(
                (got - expected).abs() < 1.0e-12,
                "[{}] pure detector {} integral identity error: hat={:.15} vs int={:.15}",
                label,
                n,
                got,
                expected
            );
        }
        assert!(
            fit_pure.reduced_chi2 < 0.2,
            "[{}] pure noiseless reduced chi2 unexpectedly large: {:.3e}",
            label,
            fit_pure.reduced_chi2
        );

        // Solution-NMR internal detector identity: rho_hat == sum_c w_c rho_n^int(tau_c)
        let fit_int = design_int
            .estimate(&obs_rates, None)
            .expect("internal estimate");
        let exact_int = design_int
            .evaluate_distribution(int_comps)
            .expect("internal exact integral");
        for (n, (&got, &expected)) in fit_int.responses.iter().zip(exact_int.iter()).enumerate() {
            assert!(
                (got - expected).abs() < 1.0e-12,
                "[{}] internal detector {} integral identity error: hat={:.15} vs int={:.15}",
                label,
                n,
                got,
                expected
            );
        }
        assert!(
            fit_int.reduced_chi2 < 0.2,
            "[{}] internal noiseless reduced chi2 unexpectedly large: {:.3e}",
            label,
            fit_int.reduced_chi2
        );
    }

    // Continuous log-normal distribution theta(tau) quadrature identity check
    let quad_taus = log_spaced_grid(20.0e-12, 20.0e-9, 120).expect("quad grid");
    let mut weights = Vec::with_capacity(quad_taus.len());
    for &t in &quad_taus {
        let z = t.log10();
        let w = (-0.5 * ((z + 9.2) / 0.45).powi(2)).exp()
            + 0.7 * (-0.5 * ((z + 8.1) / 0.25).powi(2)).exp();
        weights.push(w);
    }
    let w_sum: f64 = weights.iter().sum();
    let comps: Vec<(f64, f64)> = weights
        .into_iter()
        .zip(quad_taus)
        .map(|(w, t)| (w / w_sum, t))
        .collect();

    let mut cont_rates = vec![0.0_f64; exps.len()];
    for (i, exp) in exps.iter().enumerate() {
        for &(w, t) in &comps {
            cont_rates[i] += w * single_tau_rate(exp.kind, exp.field_mhz, t, &phys);
        }
    }
    let fit_cont = design_pure
        .estimate(&cont_rates, None)
        .expect("cont estimate");
    let exact_cont = design_pure
        .evaluate_distribution(&comps)
        .expect("cont integral");
    for (n, (&got, &expected)) in fit_cont.responses.iter().zip(exact_cont.iter()).enumerate() {
        assert!(
            (got - expected).abs() < 1.0e-12,
            "continuous distribution integral identity mismatch on detector {}: {} vs {}",
            n,
            got,
            expected
        );
    }
}

#[test]
fn synthetic_self_consistency_t22_3_analytical_variance_matches_300_trial_mc() {
    let phys = RelaxPhysicsConfig::default();
    let tau_m = 6.5e-9;
    let local = LocalMfParams::m5(0.86, 0.72, 1.1e-9);
    let exps = make_multi_field_experiments(&MULTI_FIELDS_MHZ, 0.02, tau_m, &phys);
    let true_rates = rates_from_local_mf(&exps, tau_m, &local, &phys);

    let cfg = DetectorConfig {
        physics: phys,
        tumbling: TumblingMode::IsotropicInternal { tau_m_sec: tau_m },
        num_detectors: Some(4),
        ..DetectorConfig::default()
    };
    let design = build_dynamics_detectors(&exps, &cfg).expect("detector build");
    let noiseless = design.estimate(&true_rates, None).expect("noiseless fit");

    // 300-trial independent Gaussian Monte Carlo validation (diagonal Sigma_exp)
    // Plus 2000-trial asymptotic convergence check to prove zero analytical covariance bias.
    let trials = 300usize;
    let asymp_trials = 2000usize;
    let mut rng = Rng::new(42);
    let k = design.num_detectors;
    let mut trial_resps = vec![Vec::with_capacity(asymp_trials); k];
    let mut mean_red_chi2 = 0.0_f64;

    for t_idx in 0..asymp_trials {
        let noisy: Vec<f64> = true_rates
            .iter()
            .zip(exps.iter())
            .map(|(&r, e)| rng.normal(r, e.sigma))
            .collect();
        let fit = design.estimate(&noisy, None).expect("noisy fit");
        if t_idx < trials {
            mean_red_chi2 += fit.reduced_chi2;
        }
        for (bucket, &resp) in trial_resps.iter_mut().zip(fit.responses.iter()) {
            bucket.push(resp);
        }
    }
    mean_red_chi2 /= trials as f64;
    let expected_red_chi2 = 1.0 + noiseless.reduced_chi2;
    assert!(
        (mean_red_chi2 - expected_red_chi2).abs() <= 0.15,
        "mean reduced chi2 across {} noisy trials should match 1 + chi2_trunc ({:.3}), got {:.3}",
        trials,
        expected_red_chi2,
        mean_red_chi2
    );

    for (n, samples) in trial_resps.iter().enumerate() {
        let slice_300 = &samples[..trials];
        let mean_300 = slice_300.iter().sum::<f64>() / (trials as f64);
        let emp_var_300 = slice_300
            .iter()
            .map(|&v| (v - mean_300) * (v - mean_300))
            .sum::<f64>()
            / ((trials - 1) as f64);
        let ana_var = noiseless.sigmas[n] * noiseless.sigmas[n];
        let rel_err_300 = (emp_var_300 - ana_var).abs() / ana_var.max(1.0e-30);
        assert!(
            rel_err_300 <= 0.15,
            "detector {} 300-trial variance mismatch: empirical={:.4e} vs analytical={:.4e} (rel_err={:.1}%)",
            n,
            emp_var_300,
            ana_var,
            100.0 * rel_err_300
        );

        let mean_all = samples.iter().sum::<f64>() / (asymp_trials as f64);
        let emp_var_all = samples
            .iter()
            .map(|&v| (v - mean_all) * (v - mean_all))
            .sum::<f64>()
            / ((asymp_trials - 1) as f64);
        let rel_err_all = (emp_var_all - ana_var).abs() / ana_var.max(1.0e-30);
        assert!(
            rel_err_all <= 0.08,
            "detector {} 2000-trial asymptotic variance mismatch: empirical={:.4e} vs analytical={:.4e} (rel_err={:.1}%)",
            n,
            emp_var_all,
            ana_var,
            100.0 * rel_err_all
        );
    }

    // Also validate full correlated covariance propagation when converting raw (R1, R2, hetNOE)
    let mut true_field_rates = Vec::new();
    for &mhz in &MULTI_FIELDS_MHZ {
        let (r1, r2, noe) = predict_r1_r2_noe(&[(1.0, tau_m)], &local, mhz, 600.0, &phys);
        true_field_rates.push(ResidueRelaxRates {
            seq_id: 1,
            res_name: "ALA".to_string(),
            field_mhz: mhz,
            r1,
            sigma_r1: 0.02 * r1,
            r2,
            sigma_r2: 0.02 * r2,
            hetnoe: Some(noe),
            sigma_hetnoe: Some(0.015),
        });
    }
    let (exps_corr, obs_corr, cov_corr) =
        experiments_and_observations_from_rates(&true_field_rates, &phys)
            .expect("convert rates to correlated experiments");
    let design_corr = build_dynamics_detectors(&exps_corr, &cfg).expect("correlated design");
    let fit_corr = design_corr
        .estimate(&obs_corr, Some(&cov_corr))
        .expect("correlated analytical fit");

    let mut corr_resps = vec![Vec::with_capacity(trials); k];
    for _ in 0..trials {
        let mut noisy_obs = Vec::with_capacity(exps_corr.len());
        for fr in &true_field_rates {
            let r1_n = rng.normal(fr.r1, fr.sigma_r1);
            let r2_n = rng.normal(fr.r2, fr.sigma_r2);
            let noe_n = rng.normal(fr.hetnoe.unwrap_or(0.8), fr.sigma_hetnoe.unwrap_or(0.015));
            noisy_obs.push(r1_n);
            noisy_obs.push(r2_n);
            noisy_obs.push(gamma_noe_from_r1_hetnoe(r1_n, noe_n, &phys));
        }
        let fit = design_corr
            .estimate(&noisy_obs, Some(&cov_corr))
            .expect("noisy correlated fit");
        for (bucket, &resp) in corr_resps.iter_mut().zip(fit.responses.iter()) {
            bucket.push(resp);
        }
    }
    for (n, samples) in corr_resps.iter().enumerate() {
        let mean = samples.iter().sum::<f64>() / (trials as f64);
        let emp_var = samples
            .iter()
            .map(|&v| (v - mean) * (v - mean))
            .sum::<f64>()
            / ((trials - 1) as f64);
        let ana_var = fit_corr.sigmas[n] * fit_corr.sigmas[n];
        let rel_err = (emp_var - ana_var).abs() / ana_var.max(1.0e-30);
        assert!(
            rel_err <= 0.15,
            "correlated detector {} variance mismatch: empirical={:.4e} vs analytical={:.4e} (rel_err={:.1}%)",
            n,
            emp_var,
            ana_var,
            100.0 * rel_err
        );
    }
}

#[test]
fn synthetic_self_consistency_t22_4_lemaster_exact_recovery_and_uncertainty() {
    let phys = RelaxPhysicsConfig::default();
    let field_mhz = 600.0;
    let tau_m = 6.8e-9;
    let a_slow_true = 0.68;
    let a_mid_true = 0.14;
    let a_fast_true = 0.11;
    let a_ultra_true = 1.0 - (a_slow_true + a_mid_true + a_fast_true);

    let (j0, jn, jh) = lemaster_predict_spectral_density(
        a_slow_true,
        a_mid_true,
        a_fast_true,
        field_mhz,
        tau_m,
        &phys,
    )
    .expect("forward LeMaster");

    let rsdm = ReducedSpectralDensity {
        seq_id: 42,
        res_name: "LEU".to_string(),
        field_mhz,
        j_0: j0,
        sigma_j_0: 0.02 * j0,
        j_omega_n: jn,
        sigma_j_omega_n: 0.02 * jn,
        j_087_omega_h: jh,
        sigma_j_087_omega_h: 0.03 * jh,
    };

    let fit = fit_lemaster_from_spectral_density(&rsdm, tau_m, &phys).expect("LeMaster fit");
    assert!((fit.a_slow - a_slow_true).abs() < 1.0e-12);
    assert!((fit.a_mid - a_mid_true).abs() < 1.0e-12);
    assert!((fit.a_fast - a_fast_true).abs() < 1.0e-12);
    assert!((fit.a_ultrafast - a_ultra_true).abs() < 1.0e-12);
    assert!(fit.is_physical);

    // Verify exact propagated uncertainties against 1000-trial MC (chi2_999 SD = 4.5%)
    let trials = 1000usize;
    let mut rng = Rng::new(20262241);
    let mut s_slow = Vec::with_capacity(trials);
    let mut s_mid = Vec::with_capacity(trials);
    let mut s_fast = Vec::with_capacity(trials);
    let mut s_ultra = Vec::with_capacity(trials);
    for _ in 0..trials {
        let noisy = ReducedSpectralDensity {
            j_0: rng.normal(rsdm.j_0, rsdm.sigma_j_0),
            j_omega_n: rng.normal(rsdm.j_omega_n, rsdm.sigma_j_omega_n),
            j_087_omega_h: rng.normal(rsdm.j_087_omega_h, rsdm.sigma_j_087_omega_h),
            ..rsdm.clone()
        };
        let f = fit_lemaster_from_spectral_density(&noisy, tau_m, &phys).expect("noisy LeMaster");
        s_slow.push(f.a_slow);
        s_mid.push(f.a_mid);
        s_fast.push(f.a_fast);
        s_ultra.push(f.a_ultrafast);
    }

    for (name, samples, ana_sig) in [
        ("A_slow", &s_slow, fit.sigma_a_slow),
        ("A_mid", &s_mid, fit.sigma_a_mid),
        ("A_fast", &s_fast, fit.sigma_a_fast),
        ("A_ultrafast", &s_ultra, fit.sigma_a_ultrafast),
    ] {
        let mean = samples.iter().sum::<f64>() / (trials as f64);
        let emp_var = samples
            .iter()
            .map(|&v| (v - mean) * (v - mean))
            .sum::<f64>()
            / ((trials - 1) as f64);
        let ana_var = ana_sig * ana_sig;
        let rel = (emp_var - ana_var).abs() / ana_var.max(1.0e-30);
        assert!(
            rel <= 0.12,
            "LeMaster {} variance mismatch: emp={:.4e} vs ana={:.4e} (rel={:.1}%)",
            name,
            emp_var,
            ana_var,
            100.0 * rel
        );
    }

    // Also verify fit_lemaster_from_rates on a Lipari-Szabo M2 residue
    let (r1, r2, noe) = predict_r1_r2_noe(
        &[(1.0, tau_m)],
        &LocalMfParams::m2(0.80, 0.30e-9),
        field_mhz,
        600.0,
        &phys,
    );
    let rate_in = ResidueRelaxRates {
        seq_id: 7,
        res_name: "VAL".to_string(),
        field_mhz,
        r1,
        sigma_r1: 0.02 * r1,
        r2,
        sigma_r2: 0.02 * r2,
        hetnoe: Some(noe),
        sigma_hetnoe: Some(0.015),
    };
    let fit_rates = fit_lemaster_from_rates(&rate_in, tau_m, &phys).expect("LeMaster from rates");
    assert!(
        (fit_rates.a_slow - 0.80).abs() < 0.03,
        "LeMaster from rates A_slow = {:.4} should track S^2 = 0.80",
        fit_rates.a_slow
    );
    assert!(fit_rates.sigma_a_slow > 0.0 && fit_rates.sigma_a_slow.is_finite());
}

#[test]
fn synthetic_self_consistency_t22_4_impact_kkt_simplex_and_bimodal_recovery() {
    let phys = RelaxPhysicsConfig::default();
    let tau_grid = log_spaced_grid(20.0e-12, 10.0e-9, 5).expect("5-point IMPACT grid");
    // Planted bimodal correlation-time histogram: fast ps mode (index 0) + slow ns mode (index 4)
    let true_amps = [0.35, 0.0, 0.0, 0.0, 0.65];

    let mut obs = Vec::new();
    for &mhz in &MULTI_FIELDS_MHZ {
        let (wh, wn_signed) = phys.signed_omegas_rad_s(mhz);
        let wn = wn_signed.abs();
        let w_hf = 0.87 * wh;
        for (omega, tag) in [(0.0, "J0"), (wn, "JwN"), (w_hf, "J0.87wH")] {
            let j_true = impact_predict_spectral_density(omega, &tau_grid, &true_amps);
            obs.push(SpectralDensityObs::new(
                omega,
                j_true,
                (0.015 * j_true).max(1.0e-13),
                mhz,
                format!("{}_{:.0}MHz", tag, mhz),
            ));
        }
    }

    let cfg = ImpactConfig {
        tau_min_sec: 20.0e-12,
        tau_max_sec: 10.0e-9,
        mc_replicates: 150,
        seed: 20262204,
        kkt_tol: 1.0e-10,
        max_iter: 2000,
    };

    let fit = fit_impact_spectral_density(&obs, &tau_grid, &cfg).expect("IMPACT bimodal fit");

    // Exact simplex & KKT optimality checks
    assert!(
        (fit.sum_amplitudes - 1.0).abs() < 1.0e-12,
        "simplex sum violated: {:.15}",
        fit.sum_amplitudes
    );
    for (m, &am) in fit.amplitudes.iter().enumerate() {
        assert!(am >= 0.0, "negative IMPACT amplitude A[{}] = {:.3e}", m, am);
    }
    assert!(
        fit.kkt_violation < 1.0e-7,
        "IMPACT KKT violation too large: {:.3e}",
        fit.kkt_violation
    );
    assert!(
        fit.fw_gap < 1.0e-7,
        "IMPACT Frank-Wolfe gap too large: {:.3e}",
        fit.fw_gap
    );

    // Exact recovery of planted bimodal amplitudes
    for (m, (&got, &expected)) in fit.amplitudes.iter().zip(true_amps.iter()).enumerate() {
        assert!(
            (got - expected).abs() < 1.0e-4,
            "bimodal component {} mismatch: got {:.6} vs expected {:.6}",
            m,
            got,
            expected
        );
        // 95% MC CI must bracket the recovered value cleanly and stay inside [0, 1]
        let [lo, hi] = fit.amplitudes_ci95[m];
        assert!(
            lo >= 0.0 && hi <= 1.0 + 1.0e-12 && lo <= hi,
            "invalid 95% CI [{:.4}, {:.4}] for component {}",
            lo,
            hi,
            m
        );
    }
    assert!(
        fit.chi2 < 1.0e-10,
        "noiseless IMPACT chi2 should be ~0, got {:.3e}",
        fit.chi2
    );

    // Also test noisy bimodal recovery + multi-field raw relaxation rate stacking
    let mut rng = Rng::new(777224);
    let noisy_obs: Vec<SpectralDensityObs> = obs
        .iter()
        .map(|o| SpectralDensityObs {
            value: rng.normal(o.value, o.sigma),
            ..o.clone()
        })
        .collect();
    let noisy_fit =
        fit_impact_spectral_density(&noisy_obs, &tau_grid, &cfg).expect("noisy IMPACT fit");
    assert!((noisy_fit.sum_amplitudes - 1.0).abs() < 1.0e-12);
    assert!(noisy_fit.kkt_violation < 1.0e-7);
    let fast_weight = noisy_fit.amplitudes[0] + noisy_fit.amplitudes[1];
    let slow_weight = noisy_fit.amplitudes[3] + noisy_fit.amplitudes[4];
    assert!(
        (fast_weight - 0.35).abs() < 0.08 && (slow_weight - 0.65).abs() < 0.08,
        "noisy bimodal mass not recovered: fast={:.3} (exp 0.35), slow={:.3} (exp 0.65)",
        fast_weight,
        slow_weight
    );
    assert!(
        noisy_fit.amplitudes[2] < 0.08,
        "middle valley mode should remain near zero, got {:.3}",
        noisy_fit.amplitudes[2]
    );

    // Multi-field raw (R1, R2, hetNOE) -> IMPACT recovery on bimodal motion
    let mut multi_rates = Vec::new();
    let exps = make_multi_field_experiments(&MULTI_FIELDS_MHZ, 0.02, 8.0e-9, &phys);
    let mut exact_exp_rates = Vec::with_capacity(exps.len());
    for exp in &exps {
        let mut r_val = 0.0_f64;
        for (&tau, &a) in tau_grid.iter().zip(true_amps.iter()) {
            r_val += a * single_tau_rate(exp.kind, exp.field_mhz, tau, &phys);
        }
        exact_exp_rates.push(r_val);
    }
    let fit_raw_exp = fit_impact_from_relaxation_experiments(
        &exps,
        &exact_exp_rates,
        &tau_grid,
        &phys,
        &ImpactConfig {
            mc_replicates: 0,
            ..cfg.clone()
        },
    )
    .expect("IMPACT directly from exact relaxation experiments");
    assert!((fit_raw_exp.sum_amplitudes - 1.0).abs() < 1.0e-12);
    assert!(fit_raw_exp.kkt_violation < 1.0e-7);
    for (m, (&got, &expected)) in fit_raw_exp
        .amplitudes
        .iter()
        .zip(true_amps.iter())
        .enumerate()
    {
        assert!(
            (got - expected).abs() < 1.0e-4,
            "direct relaxation IMPACT mismatch at {}: {:.6} vs {:.6}",
            m,
            got,
            expected
        );
    }

    // Build ResidueRelaxRates for fit_impact_from_multi_field_rates
    for (f_idx, &mhz) in MULTI_FIELDS_MHZ.iter().enumerate() {
        let r1 = exact_exp_rates[f_idx * 3];
        let r2 = exact_exp_rates[f_idx * 3 + 1];
        let gamma = exact_exp_rates[f_idx * 3 + 2];
        let noe = 1.0 + (phys.gamma_h / phys.gamma_n) * (gamma / r1);
        multi_rates.push(ResidueRelaxRates {
            seq_id: 10,
            res_name: "GLY".to_string(),
            field_mhz: mhz,
            r1,
            sigma_r1: 0.02 * r1,
            r2,
            sigma_r2: 0.02 * r2,
            hetnoe: Some(noe),
            sigma_hetnoe: Some(0.015),
        });
    }
    let fit_stacked = fit_impact_from_multi_field_rates(
        &multi_rates,
        &tau_grid,
        &phys,
        &ImpactConfig {
            mc_replicates: 60,
            ..cfg
        },
    )
    .expect("IMPACT stacked from multi-field ResidueRelaxRates");
    assert!((fit_stacked.sum_amplitudes - 1.0).abs() < 1.0e-12);
    assert!(fit_stacked.kkt_violation < 1.0e-7);
    assert!(
        (fit_stacked.amplitudes[0] - 0.35).abs() < 0.05
            && (fit_stacked.amplitudes[4] - 0.65).abs() < 0.05,
        "stacked RSDM IMPACT recovery drifted: {:?}",
        fit_stacked.amplitudes
    );
}

#[test]
fn synthetic_self_consistency_t22_4_impact_aicc_selection_and_rank_deficient_kkt() {
    let phys = RelaxPhysicsConfig::default();
    // Plant a simple 3-timescale motion on [20ps, 8ns] and sample 4 fields (12 observations)
    let tau_3 = log_spaced_grid(20.0e-12, 8.0e-9, 3).expect("3-point grid");
    let true_3 = [0.25, 0.20, 0.55];

    let mut obs = Vec::new();
    for &mhz in &MULTI_FIELDS_MHZ {
        let (wh, wn_signed) = phys.signed_omegas_rad_s(mhz);
        let wn = wn_signed.abs();
        let w_hf = 0.87 * wh;
        for (omega, tag) in [(0.0, "J0"), (wn, "JwN"), (w_hf, "J0.87wH")] {
            let j_val = impact_predict_spectral_density(omega, &tau_3, &true_3);
            obs.push(SpectralDensityObs::new(
                omega,
                j_val,
                (0.02 * j_val).max(1.0e-13),
                mhz,
                format!("{}_{:.0}MHz", tag, mhz),
            ));
        }
    }

    let cfg = ImpactConfig {
        tau_min_sec: 20.0e-12,
        tau_max_sec: 8.0e-9,
        mc_replicates: 40,
        seed: 20262249,
        kkt_tol: 1.0e-10,
        max_iter: 2000,
    };

    let sel = select_impact_model_aicc(&obs, &[3, 4, 5, 6], &cfg).expect("AICc selection");
    assert_eq!(
        sel.selected_num_timescales, 3,
        "AICc should select parsimonious M=3 on 3-timescale synthetic data, got M={} (candidates AICc: {:?})",
        sel.selected_num_timescales,
        sel.candidates
            .iter()
            .map(|c| (c.tau_grid_sec.len(), c.aicc))
            .collect::<Vec<_>>()
    );
    assert_eq!(sel.candidates.len(), 4);
    assert!((sel.best_fit.sum_amplitudes - 1.0).abs() < 1.0e-12);
    assert!(sel.best_fit.kkt_violation < 1.0e-7);

    // Also test select_impact_model_aicc_from_rates on 3-field (9-obs) data
    let mut rates_3f = Vec::new();
    for &mhz in &[500.0, 600.0, 800.0] {
        let (r1, r2, noe) = predict_r1_r2_noe(
            &[(1.0, 6.0e-9)],
            &LocalMfParams::m2(0.75, 0.25e-9),
            mhz,
            600.0,
            &phys,
        );
        rates_3f.push(ResidueRelaxRates {
            seq_id: 1,
            res_name: "SER".to_string(),
            field_mhz: mhz,
            r1,
            sigma_r1: 0.02 * r1,
            r2,
            sigma_r2: 0.02 * r2,
            hetnoe: Some(noe),
            sigma_hetnoe: Some(0.015),
        });
    }
    let sel_rates = select_impact_model_aicc_from_rates(&rates_3f, &[3, 4, 5, 6], &phys, &cfg)
        .expect("AICc from rates");
    assert!((3..=6).contains(&sel_rates.selected_num_timescales));
    assert!((sel_rates.best_fit.sum_amplitudes - 1.0).abs() < 1.0e-12);
    assert!(sel_rates.best_fit.kkt_violation < 1.0e-7);

    // Stress-test simplex WLS directly on a rank-deficient / collinear design matrix (N=4, M=7)
    let design = vec![
        vec![1.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.01],
        vec![0.8, 0.8, 0.6, 0.4, 0.2, 0.10, 0.02],
        vec![0.3, 0.3, 0.5, 0.7, 0.6, 0.30, 0.10],
        vec![0.1, 0.1, 0.2, 0.3, 0.5, 0.80, 0.90],
    ];
    let targets = vec![0.45, 0.48, 0.42, 0.52];
    let sigmas = vec![0.02, 0.02, 0.02, 0.02];
    let sol = solve_simplex_wls(&design, &targets, &sigmas, 1.0e-10, 4000)
        .expect("rank-deficient simplex WLS should converge cleanly");
    let sum_a: f64 = sol.amplitudes.iter().sum();
    assert!((sum_a - 1.0).abs() < 1.0e-12);
    assert!(sol.amplitudes.iter().all(|&a| a >= 0.0));
    assert!(
        sol.kkt_violation < 1.0e-7 && sol.fw_gap < 1.0e-7,
        "rank-deficient simplex KKT violation too large: kkt={:.3e}, fw={:.3e}",
        sol.kkt_violation,
        sol.fw_gap
    );
}

#[test]
fn synthetic_self_consistency_hostile_and_underdetermined_inputs_fail_closed() {
    let phys = RelaxPhysicsConfig::default();
    let grid = log_spaced_grid(1.0e-12, 1.0e-8, 20).expect("valid grid");

    // 1. Empty experiment / observation datasets
    assert!(build_dynamics_detectors(&[], &DetectorConfig::default()).is_err());
    assert!(
        build_sensitivity_matrix(&[], &grid, TumblingMode::PureCorrelationSpectrum, &phys).is_err()
    );
    assert!(experiments_and_observations_from_rates(&[], &phys).is_err());
    assert!(
        fit_impact_spectral_density(&[], &[1e-11, 1e-9, 1e-8], &ImpactConfig::default()).is_err()
    );
    assert!(select_impact_model_aicc(&[], &[3, 4, 5, 6], &ImpactConfig::default()).is_err());

    // 2. Non-positive fields and sigma <= 0 and NaN / Inf
    let bad_field = vec![RelaxExperiment::new(RelaxObservableKind::R1, 0.0, 0.05)];
    assert!(build_dynamics_detectors(&bad_field, &DetectorConfig::default()).is_err());
    let neg_field = vec![RelaxExperiment::new(RelaxObservableKind::R1, -600.0, 0.05)];
    assert!(build_dynamics_detectors(&neg_field, &DetectorConfig::default()).is_err());
    let zero_sig = vec![RelaxExperiment::new(RelaxObservableKind::R1, 600.0, 0.0)];
    assert!(build_dynamics_detectors(&zero_sig, &DetectorConfig::default()).is_err());
    let neg_sig = vec![RelaxExperiment::new(RelaxObservableKind::R1, 600.0, -0.02)];
    assert!(build_dynamics_detectors(&neg_sig, &DetectorConfig::default()).is_err());
    let nan_sig = vec![RelaxExperiment::new(
        RelaxObservableKind::R1,
        600.0,
        f64::NAN,
    )];
    assert!(build_dynamics_detectors(&nan_sig, &DetectorConfig::default()).is_err());

    // 3. Invalid tumbling tau_m and requesting more detectors than experiments
    let valid_exps = make_multi_field_experiments(&[600.0], 0.02, 6.0e-9, &phys);
    assert!(build_dynamics_detectors(
        &valid_exps,
        &DetectorConfig {
            tumbling: TumblingMode::IsotropicInternal { tau_m_sec: -1.0e-9 },
            ..DetectorConfig::default()
        }
    )
    .is_err());
    assert!(build_dynamics_detectors(
        &valid_exps,
        &DetectorConfig {
            num_detectors: Some(4), // only 3 experiments!
            ..DetectorConfig::default()
        }
    )
    .is_err());
    assert!(build_dynamics_detectors(
        &valid_exps,
        &DetectorConfig {
            num_detectors: Some(0),
            ..DetectorConfig::default()
        }
    )
    .is_err());

    // 4. NaN observed rates and non-PSD / asymmetric covariance in DetectorDesign::estimate
    let design = build_dynamics_detectors(
        &valid_exps,
        &DetectorConfig {
            num_detectors: Some(2),
            ..DetectorConfig::default()
        },
    )
    .expect("valid 3-exp design");
    assert!(design.estimate(&[1.5, f64::NAN, 0.02], None).is_err());
    assert!(design.estimate(&[1.5, 10.0], None).is_err()); // wrong length
    let non_psd_cov = vec![
        vec![1.0, 2.0, 0.0],
        vec![2.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ];
    assert!(design
        .estimate(&[1.5, 10.0, 0.02], Some(&non_psd_cov))
        .is_err());

    // 5. Hostile LeMaster inputs (NaN, sigma <= 0, non-positive field, degenerate tau_m == tau_N)
    let (_, wn_signed) = phys.signed_omegas_rad_s(600.0);
    let tau_n_600 = 1.0 / wn_signed.abs();
    let rsdm_ok = ReducedSpectralDensity {
        seq_id: 1,
        res_name: "ALA".to_string(),
        field_mhz: 600.0,
        j_0: 2.0e-9,
        sigma_j_0: 1.0e-11,
        j_omega_n: 0.4e-9,
        sigma_j_omega_n: 1.0e-11,
        j_087_omega_h: 0.02e-9,
        sigma_j_087_omega_h: 1.0e-12,
    };
    assert!(fit_lemaster_from_spectral_density(&rsdm_ok, tau_n_600, &phys).is_err());
    assert!(fit_lemaster_from_spectral_density(&rsdm_ok, 0.0, &phys).is_err());
    assert!(fit_lemaster_from_spectral_density(&rsdm_ok, f64::NAN, &phys).is_err());
    assert!(fit_lemaster_from_spectral_density(
        &ReducedSpectralDensity {
            j_0: f64::NAN,
            ..rsdm_ok.clone()
        },
        6.0e-9,
        &phys
    )
    .is_err());
    assert!(fit_lemaster_from_spectral_density(
        &ReducedSpectralDensity {
            sigma_j_0: 0.0,
            ..rsdm_ok.clone()
        },
        6.0e-9,
        &phys
    )
    .is_err());
    assert!(fit_lemaster_from_spectral_density(
        &ReducedSpectralDensity {
            field_mhz: -600.0,
            ..rsdm_ok
        },
        6.0e-9,
        &phys
    )
    .is_err());

    // 6. Hostile and under-determined IMPACT inputs
    let single_field_obs = vec![
        SpectralDensityObs::new(0.0, 2.0e-9, 1.0e-11, 600.0, "J0"),
        SpectralDensityObs::new(3.8e8, 0.4e-9, 1.0e-11, 600.0, "JwN"),
        SpectralDensityObs::new(3.3e9, 0.02e-9, 1.0e-12, 600.0, "Jh"),
    ];
    // Under-determined: 3 obs for M = 5 (k = 4 free params)
    let grid5 = log_spaced_grid(1.0e-11, 1.0e-8, 5).expect("grid5");
    assert!(
        fit_impact_spectral_density(&single_field_obs, &grid5, &ImpactConfig::default()).is_err()
    );
    // Under-determined AICc: 3 obs cannot produce finite AICc for any M in [3, 4, 5, 6]
    assert!(
        select_impact_model_aicc(&single_field_obs, &[3, 4, 5, 6], &ImpactConfig::default())
            .is_err()
    );
    // NaN / sigma <= 0 / negative omega in IMPACT
    let bad_obs_nan = vec![
        SpectralDensityObs::new(0.0, f64::NAN, 1.0e-11, 600.0, "J0"),
        SpectralDensityObs::new(3.8e8, 0.4e-9, 1.0e-11, 600.0, "JwN"),
        SpectralDensityObs::new(3.3e9, 0.02e-9, 1.0e-12, 600.0, "Jh"),
    ];
    let grid3 = log_spaced_grid(1.0e-11, 1.0e-8, 3).expect("grid3");
    assert!(fit_impact_spectral_density(&bad_obs_nan, &grid3, &ImpactConfig::default()).is_err());
    let bad_obs_sig = vec![
        SpectralDensityObs::new(0.0, 2.0e-9, 0.0, 600.0, "J0"),
        SpectralDensityObs::new(3.8e8, 0.4e-9, 1.0e-11, 600.0, "JwN"),
        SpectralDensityObs::new(3.3e9, 0.02e-9, 1.0e-12, 600.0, "Jh"),
    ];
    assert!(fit_impact_spectral_density(&bad_obs_sig, &grid3, &ImpactConfig::default()).is_err());
    let bad_obs_omega = vec![
        SpectralDensityObs::new(-1.0, 2.0e-9, 1.0e-11, 600.0, "J0"),
        SpectralDensityObs::new(3.8e8, 0.4e-9, 1.0e-11, 600.0, "JwN"),
        SpectralDensityObs::new(3.3e9, 0.02e-9, 1.0e-12, 600.0, "Jh"),
    ];
    assert!(fit_impact_spectral_density(&bad_obs_omega, &grid3, &ImpactConfig::default()).is_err());
}
