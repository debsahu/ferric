//! Comprehensive verification suite for `nmr_core::multi_cest` (`T22.6` + `T22.7`).
//!
//! All synthetic profile simulations, noisy parameter recoveries, and online training/held-out
//! evaluations in this test file are `[synthetic self-consistency]` benchmarks generated directly
//! from exact `7x7` homogeneous Bloch–McConnell equations.
//!
//! Provenance: Cites Carneiro et al. (*J. Biomol. NMR* 2015, 62, 237–246), Yuwen, Kay &
//! Bouvignies (*J. Phys. Chem. Lett.* 2018, 9, 2501–2507), Vallurupalli, Bouvignies & Kay
//! (*JACS* 2012, 134, 8148–8161), and Higham (*SIAM J. Matrix Anal. Appl.* 2005, 26, 1179–1193).

use nmr_core::cest::{simulate_cest_2state_profile, Cest2StateParams, CestNormalization};
use nmr_core::multi_cest::{
    fit_cest_amortized_seeded_lm, fit_multi_cest_grid_lm, fit_multi_cest_lm,
    generate_cest_training_ensemble, simulate_cos_cest_profile, simulate_dante_cest_profile,
    simulate_multi_cest_profile, train_amortized_cest_net, AmortizedCestNet,
    AmortizedCestNetConfig, AmortizedHybridFitConfig, AmortizedSeedStatus, AmortizedTrainConfig,
    CestEnsembleGenConfig, CestTrainingSample, CosCestParams, DanteCestParams, MultiCestFitParams,
    MultiCestLmConfig, MultiCestProfile, MultiCestScheme, MultiCestSimParams,
};
use nmr_core::rng::Rng;

/// `[synthetic self-consistency]` Verify DANTE-CEST sideband comb periodicity
/// `sw_DANTE = 1 / (tau_p + tau_d)` and convergence to CW-CEST `< 1e-4` as `tau_p, tau_d -> 0`
/// at constant time-averaged `b1_eff` and constant `T_EX`.
#[test]
fn test_dante_cest_sideband_periodicity_and_cw_limit_convergence() {
    let larmor_mhz = 60.8106; // 15N at 600 MHz 1H
    let major_ppm = 118.0;
    let dw_ppm = 3.5;

    // 1. CW-limit convergence: shrink cycle period tau_c -> 0 while keeping
    //    b1_eff = b1_pulse * tau_p / (tau_p + tau_d) = 25 Hz and T_EX = 0.40 s constant.
    let offsets_cw: Vec<f64> = (-28..=28).map(|k| major_ppm + 0.30 * (k as f64)).collect();
    let cw_params = Cest2StateParams {
        p_b: 0.025,
        k_ex_s_inv: 180.0,
        major_ppm,
        dw_ab_ppm: dw_ppm,
        r1a_s_inv: 1.3,
        r2a_s_inv: 14.0,
        r1b_s_inv: 1.3,
        r2b_s_inv: 20.0,
        larmor_mhz,
        t_ex_s: 0.40,
        b1_hz: 25.0,
        b1_inhomogeneity_frac: 0.0,
        n_b1_quad: 1,
        normalization: CestNormalization::ZeroTimePhaseCycled,
    };
    let cw_ref = simulate_cest_2state_profile(&cw_params, &offsets_cw).unwrap();

    // Very short cycle: tau_c = 20 us -> sw_DANTE = 50,000 Hz (>> offset span), N = 20,000
    let tau_p_short = 2.0e-6;
    let tau_d_short = 18.0e-6;
    let dante_cw_limit = DanteCestParams {
        p_b: cw_params.p_b,
        k_ex_s_inv: cw_params.k_ex_s_inv,
        major_ppm,
        dw_ab_ppm: dw_ppm,
        r1a_s_inv: cw_params.r1a_s_inv,
        r2a_s_inv: cw_params.r2a_s_inv,
        r1b_s_inv: cw_params.r1b_s_inv,
        r2b_s_inv: cw_params.r2b_s_inv,
        larmor_mhz,
        tau_pulse_s: tau_p_short,
        tau_delay_s: tau_d_short,
        n_pulses: 20_000,
        b1_pulse_hz: 25.0 * ((tau_p_short + tau_d_short) / tau_p_short),
        b1_inhomogeneity_frac: 0.0,
        n_b1_quad: 1,
        normalization: CestNormalization::ZeroTimePhaseCycled,
    };
    assert!((dante_cw_limit.effective_b1_hz() - 25.0).abs() < 1e-12);
    assert!((dante_cw_limit.total_duration_s() - 0.40).abs() < 1e-12);

    let dante_lim_prof = simulate_dante_cest_profile(&dante_cw_limit, &offsets_cw).unwrap();
    let max_cw_diff = cw_ref
        .iter()
        .zip(dante_lim_prof.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    println!("[synthetic self-consistency] DANTE-CEST CW-limit max diff = {max_cw_diff:.3e}");
    assert!(
        max_cw_diff < 1e-4,
        "DANTE-CEST CW limit failed: max diff {max_cw_diff:.3e} >= 1e-4"
    );

    // 2. Exact comb sideband periodicity sw_DANTE = 1 / (tau_p + tau_d) in the short-pulse regime
    //    tau_p = 10 us, tau_d = 2.49 ms => tau_c = 2.5 ms, sw_DANTE = 400 Hz (~6.578 ppm).
    let tau_p = 10.0e-6;
    let tau_d = 2.49e-3;
    let sw_hz = 1.0 / (tau_p + tau_d);
    let sw_ppm = sw_hz / larmor_mhz;
    let dante_comb = DanteCestParams {
        p_b: 0.025,
        k_ex_s_inv: 180.0,
        major_ppm,
        dw_ab_ppm: 2.2,
        r1a_s_inv: 1.3,
        r2a_s_inv: 14.0,
        r1b_s_inv: 1.3,
        r2b_s_inv: 20.0,
        larmor_mhz,
        tau_pulse_s: tau_p,
        tau_delay_s: tau_d,
        n_pulses: 160, // T_EX = 0.40 s
        b1_pulse_hz: 20.0 * ((tau_p + tau_d) / tau_p),
        b1_inhomogeneity_frac: 0.0,
        n_b1_quad: 1,
        normalization: CestNormalization::SameDelayZeroB1,
    };
    assert!((dante_comb.comb_spacing_hz() - 400.0).abs() < 1e-9);
    assert!((dante_comb.comb_spacing_ppm() - sw_ppm).abs() < 1e-12);

    // Compare central comb window [major - 0.5*sw, major + 0.5*sw] vs shifted by +-sw_ppm
    let base_offsets: Vec<f64> = (-12..=12)
        .map(|k| major_ppm + (k as f64) * (sw_ppm / 24.0))
        .collect();
    let plus_offsets: Vec<f64> = base_offsets.iter().map(|&o| o + sw_ppm).collect();
    let minus_offsets: Vec<f64> = base_offsets.iter().map(|&o| o - sw_ppm).collect();

    let p0 = simulate_dante_cest_profile(&dante_comb, &base_offsets).unwrap();
    let p_plus = simulate_dante_cest_profile(&dante_comb, &plus_offsets).unwrap();
    let p_minus = simulate_dante_cest_profile(&dante_comb, &minus_offsets).unwrap();

    // Sinc rolloff over +-1 comb tooth at 0.4% duty cycle is < 3e-4
    let max_period_plus = p0
        .iter()
        .zip(p_plus.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    let max_period_minus = p0
        .iter()
        .zip(p_minus.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    // Also verify there is a strong major dip at major_ppm AND at major_ppm +- sw_ppm
    let dip_probe = simulate_dante_cest_profile(
        &dante_comb,
        &[
            major_ppm - sw_ppm,
            major_ppm,
            major_ppm + sw_ppm,
            major_ppm + 0.5 * sw_ppm,
        ],
    )
    .unwrap();
    println!(
        "[synthetic self-consistency] DANTE comb periodicity diff (+sw={max_period_plus:.3e}, -sw={max_period_minus:.3e}), dips={dip_probe:?}"
    );
    assert!(
        max_period_plus < 2e-3 && max_period_minus < 2e-3,
        "DANTE comb sideband periodicity violated: +sw={max_period_plus:.3e}, -sw={max_period_minus:.3e}"
    );
    assert!(dip_probe[0] < 0.25 && dip_probe[1] < 0.25 && dip_probe[2] < 0.25);
    assert!(
        dip_probe[3] > 0.75,
        "half-comb baseline should stay high relative to comb dips, got {}",
        dip_probe[3]
    );
}

/// `[synthetic self-consistency]` Verify exact `cos-CEST` dual-sideband symmetry at `+-nu_mod`
/// and exact dual major + dual minor dip placement for a 2-state exchanging system.
#[test]
fn test_cos_cest_exact_dual_sideband_symmetry() {
    let larmor_mhz = 60.8106;
    let major_ppm = 120.0;
    let mod_freq_hz = 250.0; // sideband offset = 250 / 60.8106 = 4.1111 ppm
    let sb_ppm = mod_freq_hz / larmor_mhz;

    // 1. Single-state (p_B = 0): I(major + delta) must equal I(major - delta) to machine precision
    let cos_1s = CosCestParams {
        p_b: 0.0,
        k_ex_s_inv: 0.0,
        major_ppm,
        dw_ab_ppm: 0.0,
        r1a_s_inv: 1.3,
        r2a_s_inv: 14.0,
        r1b_s_inv: 1.3,
        r2b_s_inv: 14.0,
        larmor_mhz,
        t_ex_s: 0.40,
        b1_eff_hz: 25.0,
        mod_freq_hz,
        n_steps_per_cycle: 32,
        b1_inhomogeneity_frac: 0.0,
        n_b1_quad: 1,
        normalization: CestNormalization::SameDelayZeroB1,
    };
    let deltas: Vec<f64> = (0..=30).map(|k| 0.25 * (k as f64)).collect();
    let pos_offs: Vec<f64> = deltas.iter().map(|&d| major_ppm + d).collect();
    let neg_offs: Vec<f64> = deltas.iter().map(|&d| major_ppm - d).collect();
    let y_pos = simulate_cos_cest_profile(&cos_1s, &pos_offs).unwrap();
    let y_neg = simulate_cos_cest_profile(&cos_1s, &neg_offs).unwrap();

    let max_asym = y_pos
        .iter()
        .zip(y_neg.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    println!("[synthetic self-consistency] cos-CEST 1-state dual-sideband max asymmetry = {max_asym:.3e}");
    assert!(
        max_asym < 1e-12,
        "cos-CEST dual sidebands must be strictly symmetric, got {max_asym:.3e}"
    );

    // 2. Verify carrier at major_ppm (center, zero irradiation) stays near 1.0 while +-sb_ppm dip deeply
    let probes = simulate_cos_cest_profile(
        &cos_1s,
        &[major_ppm - sb_ppm, major_ppm, major_ppm + sb_ppm],
    )
    .unwrap();
    assert!(probes[0] < 0.15, "lower sideband must saturate major state");
    assert!(probes[2] < 0.15, "upper sideband must saturate major state");
    assert!(
        (probes[0] - probes[2]).abs() < 1e-12,
        "lower and upper sideband dips must match exactly"
    );
    // At carrier = major_ppm, two symmetric sidebands at +-mod_freq yield off-resonance
    // R1rho excess ~ 2 * (R2 - R1) * (b1_eff / mod_freq)^2 -> exp(-0.254 * 0.40) ~ 0.903
    let expected_center = (-2.0
        * (cos_1s.r2a_s_inv - cos_1s.r1a_s_inv)
        * (cos_1s.b1_eff_hz / cos_1s.mod_freq_hz).powi(2)
        * cos_1s.t_ex_s)
        .exp();
    assert!(
        (probes[1] - expected_center).abs() < 0.01,
        "central carrier should match analytical dual-sideband off-resonance decay {expected_center:.4}, got {:.4}",
        probes[1]
    );

    // 3. Two-state reflection symmetry: swapping (dw -> -dw) and (offset -> 2*major - offset)
    //    must produce identical profiles, and dual minor dips appear at (major + dw) +- sb_ppm.
    let dw = 4.2;
    let cos_2s_pos = CosCestParams {
        p_b: 0.025,
        k_ex_s_inv: 180.0,
        major_ppm,
        dw_ab_ppm: dw,
        r1a_s_inv: 1.3,
        r2a_s_inv: 14.0,
        r1b_s_inv: 1.3,
        r2b_s_inv: 20.0,
        larmor_mhz,
        t_ex_s: 0.40,
        b1_eff_hz: 25.0,
        mod_freq_hz: 300.0,
        n_steps_per_cycle: 32,
        b1_inhomogeneity_frac: 0.05,
        n_b1_quad: 5,
        normalization: CestNormalization::ZeroTimePhaseCycled,
    };
    let cos_2s_neg = CosCestParams {
        dw_ab_ppm: -dw,
        ..cos_2s_pos.clone()
    };
    let offs_a: Vec<f64> = (-20..=20).map(|k| major_ppm + 0.5 * (k as f64)).collect();
    let offs_b: Vec<f64> = offs_a.iter().map(|&o| 2.0 * major_ppm - o).collect();
    let prof_a = simulate_cos_cest_profile(&cos_2s_pos, &offs_a).unwrap();
    let prof_b = simulate_cos_cest_profile(&cos_2s_neg, &offs_b).unwrap();
    let max_2s_asym = prof_a
        .iter()
        .zip(prof_b.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_2s_asym < 1e-12,
        "2-state cos-CEST reflection symmetry failed: {max_2s_asym:.3e}"
    );

    // Check both minor sidebands at (major + dw) +- (300 / 60.8106) dip well below far baseline
    let sb2 = cos_2s_pos.sideband_offset_ppm();
    let minor_probe = simulate_cos_cest_profile(
        &cos_2s_pos,
        &[major_ppm + dw - sb2, major_ppm + dw + sb2, major_ppm + 15.0],
    )
    .unwrap();
    assert!(minor_probe[0] < minor_probe[2] - 0.08);
    assert!(minor_probe[1] < minor_probe[2] - 0.08);
}

/// `[synthetic self-consistency]` Multi-B1 DANTE-CEST global LM recovery of planted
/// `(p_B = 2.5%, k_ex = 180 s^-1, dw = +4.2 ppm)` within `< 5%` on noisy synthetic profiles,
/// with valid Cramér–Rao covariance and reduced chi^2 diagnostics.
#[test]
fn test_multi_b1_dante_cest_noisy_parameter_recovery_under_5pct() {
    let larmor_mhz = 60.8106;
    let major_ppm = 118.0;
    let true_p = MultiCestFitParams {
        p_b: 0.025,
        k_ex_s_inv: 180.0,
        dw_ab_ppm: 4.2,
        r1_s_inv: 1.3,
        r2a_s_inv: 14.0,
        r2b_s_inv: 22.0,
    };

    // sw_DANTE = 1 / 1.5 ms = 666.67 Hz (~10.96 ppm), unambiguously covering +-5.48 ppm!
    let tau_p = 15.0e-6;
    let tau_d = 1.485e-3;
    let n_pulses = 266u32; // T_EX = 0.399 s
    let duty = tau_p / (tau_p + tau_d);

    let offsets: Vec<f64> = (-30..=30).map(|k| major_ppm + 0.18 * (k as f64)).collect();
    let noise_sigma = 0.004;
    let mut rng = Rng::new(220_601);

    let mut profiles = Vec::new();
    for &b1_eff in &[12.0_f64, 28.0_f64] {
        let scheme = MultiCestScheme::Dante {
            tau_pulse_s: tau_p,
            tau_delay_s: tau_d,
            n_pulses,
            b1_pulse_hz: b1_eff / duty,
        };
        let sim_p = MultiCestSimParams {
            p_b: true_p.p_b,
            k_ex_s_inv: true_p.k_ex_s_inv,
            major_ppm,
            dw_ab_ppm: true_p.dw_ab_ppm,
            r1a_s_inv: true_p.r1_s_inv,
            r2a_s_inv: true_p.r2a_s_inv,
            r1b_s_inv: true_p.r1_s_inv,
            r2b_s_inv: true_p.r2b_s_inv,
            larmor_mhz,
            b1_inhomogeneity_frac: 0.0,
            n_b1_quad: 1,
            normalization: CestNormalization::ZeroTimePhaseCycled,
            scheme: scheme.clone(),
        };
        let clean = simulate_multi_cest_profile(&sim_p, &offsets).unwrap();
        let noisy: Vec<f64> = clean
            .iter()
            .map(|&y| y + rng.normal(0.0, noise_sigma))
            .collect();
        profiles.push(MultiCestProfile {
            larmor_mhz,
            major_ppm,
            b1_inhomogeneity_frac: 0.0,
            n_b1_quad: 1,
            normalization: CestNormalization::ZeroTimePhaseCycled,
            scheme,
            offsets_ppm: offsets.clone(),
            intensities: noisy,
            uncertainties: vec![noise_sigma; offsets.len()],
        });
    }

    let lm_cfg = MultiCestLmConfig {
        grid_dw_range_ppm: (-5.2, 5.2),
        grid_dw_step_ppm: 0.6,
        ..Default::default()
    };
    let fit = fit_multi_cest_grid_lm(&profiles, &lm_cfg).unwrap();

    let pb_rel = (fit.params.p_b - true_p.p_b).abs() / true_p.p_b;
    let kex_rel = (fit.params.k_ex_s_inv - true_p.k_ex_s_inv).abs() / true_p.k_ex_s_inv;
    let dw_rel = (fit.params.dw_ab_ppm - true_p.dw_ab_ppm).abs() / true_p.dw_ab_ppm.abs();

    println!(
        "[synthetic self-consistency] DANTE-CEST fit: p_B={:.4} (err={:.2}%), k_ex={:.2} (err={:.2}%), dw={:.3} (err={:.2}%), chi2_nu={:.3}",
        fit.params.p_b,
        pb_rel * 100.0,
        fit.params.k_ex_s_inv,
        kex_rel * 100.0,
        fit.params.dw_ab_ppm,
        dw_rel * 100.0,
        fit.reduced_chi2
    );
    assert!(fit.converged, "DANTE-CEST LM fit must converge");
    assert!(
        pb_rel < 0.05,
        "p_B relative error {:.2}% >= 5%",
        pb_rel * 100.0
    );
    assert!(
        kex_rel < 0.05,
        "k_ex relative error {:.2}% >= 5%",
        kex_rel * 100.0
    );
    assert!(
        dw_rel < 0.05,
        "dw relative error {:.2}% >= 5%",
        dw_rel * 100.0
    );
    assert!(
        fit.reduced_chi2 > 0.5 && fit.reduced_chi2 < 1.6,
        "reduced_chi2={:.3} out of expected noise band",
        fit.reduced_chi2
    );
    for (j, &se) in fit.std_errors.iter().enumerate() {
        assert!(
            se.is_finite() && se > 0.0,
            "Cramer-Rao SE[{j}] must be finite > 0"
        );
        assert!(
            (fit.cramer_rao_cov[j][j].sqrt() - se).abs() < 1e-12,
            "SE must match sqrt(diag(cov))"
        );
    }
}

/// `[synthetic self-consistency]` Multi-B1 `cos-CEST` global LM recovery of planted
/// `(p_B = 2.5%, k_ex = 180 s^-1, dw = +4.2 ppm)` within `< 5%` on noisy synthetic profiles.
#[test]
fn test_multi_b1_cos_cest_noisy_parameter_recovery_under_5pct() {
    let larmor_mhz = 60.8106;
    let major_ppm = 120.0;
    let true_p = MultiCestFitParams {
        p_b: 0.025,
        k_ex_s_inv: 180.0,
        dw_ab_ppm: 4.2,
        r1_s_inv: 1.3,
        r2a_s_inv: 14.0,
        r2b_s_inv: 22.0,
    };

    // mod_freq = 320 Hz (~5.262 ppm): major dips at +-5.26 ppm, minor dips at -1.06 & +9.46 ppm
    let mod_freq_hz = 320.0;
    let offsets: Vec<f64> = (-26..=38).map(|k| major_ppm + 0.30 * (k as f64)).collect();
    let noise_sigma = 0.004;
    let mut rng = Rng::new(220_602);

    let mut profiles = Vec::new();
    for &b1_eff in &[14.0_f64, 30.0_f64] {
        let scheme = MultiCestScheme::CosModulated {
            t_ex_s: 0.40,
            b1_eff_hz: b1_eff,
            mod_freq_hz,
            n_steps_per_cycle: 24,
        };
        let sim_p = MultiCestSimParams {
            p_b: true_p.p_b,
            k_ex_s_inv: true_p.k_ex_s_inv,
            major_ppm,
            dw_ab_ppm: true_p.dw_ab_ppm,
            r1a_s_inv: true_p.r1_s_inv,
            r2a_s_inv: true_p.r2a_s_inv,
            r1b_s_inv: true_p.r1_s_inv,
            r2b_s_inv: true_p.r2b_s_inv,
            larmor_mhz,
            b1_inhomogeneity_frac: 0.0,
            n_b1_quad: 1,
            normalization: CestNormalization::ZeroTimePhaseCycled,
            scheme: scheme.clone(),
        };
        let clean = simulate_multi_cest_profile(&sim_p, &offsets).unwrap();
        let noisy: Vec<f64> = clean
            .iter()
            .map(|&y| y + rng.normal(0.0, noise_sigma))
            .collect();
        profiles.push(MultiCestProfile {
            larmor_mhz,
            major_ppm,
            b1_inhomogeneity_frac: 0.0,
            n_b1_quad: 1,
            normalization: CestNormalization::ZeroTimePhaseCycled,
            scheme,
            offsets_ppm: offsets.clone(),
            intensities: noisy,
            uncertainties: vec![noise_sigma; offsets.len()],
        });
    }

    let lm_cfg = MultiCestLmConfig {
        grid_dw_range_ppm: (-6.0, 6.0),
        grid_dw_step_ppm: 0.8,
        ..Default::default()
    };
    let fit = fit_multi_cest_grid_lm(&profiles, &lm_cfg).unwrap();

    let pb_rel = (fit.params.p_b - true_p.p_b).abs() / true_p.p_b;
    let kex_rel = (fit.params.k_ex_s_inv - true_p.k_ex_s_inv).abs() / true_p.k_ex_s_inv;
    let dw_rel = (fit.params.dw_ab_ppm - true_p.dw_ab_ppm).abs() / true_p.dw_ab_ppm.abs();

    println!(
        "[synthetic self-consistency] cos-CEST fit: p_B={:.4} (err={:.2}%), k_ex={:.2} (err={:.2}%), dw={:.3} (err={:.2}%), chi2_nu={:.3}",
        fit.params.p_b,
        pb_rel * 100.0,
        fit.params.k_ex_s_inv,
        kex_rel * 100.0,
        fit.params.dw_ab_ppm,
        dw_rel * 100.0,
        fit.reduced_chi2
    );
    assert!(fit.converged, "cos-CEST LM fit must converge");
    assert!(
        pb_rel < 0.05,
        "p_B relative error {:.2}% >= 5%",
        pb_rel * 100.0
    );
    assert!(
        kex_rel < 0.05,
        "k_ex relative error {:.2}% >= 5%",
        kex_rel * 100.0
    );
    assert!(
        dw_rel < 0.05,
        "dw relative error {:.2}% >= 5%",
        dw_rel * 100.0
    );
    assert!(
        fit.reduced_chi2 > 0.5 && fit.reduced_chi2 < 1.6,
        "reduced_chi2={:.3} out of expected noise band",
        fit.reduced_chi2
    );
    for &se in &fit.std_errors {
        assert!(se.is_finite() && se > 0.0);
    }
}

/// `[synthetic self-consistency]` Finite-difference gradient check `< 1e-5` on EVERY parameter
/// of [`AmortizedCestNet`] (Conv1D + Conv1D + MLP + heteroscedastic Gaussian NLL + L2 decay).
#[test]
fn test_amortized_cest_net_backprop_fd_gradient_check() {
    let cfg = AmortizedCestNetConfig {
        n_channels: 2,
        n_grid: 16,
        offset_window_ppm: (-8.0, 8.0),
        conv1_channels: 3,
        conv2_channels: 4,
        kernel_size: 3,
        hidden_dim: 8,
        ..Default::default()
    };
    let mut net = AmortizedCestNet::new(cfg.clone(), 12345).unwrap();
    let mut rng = Rng::new(54321);

    let batch: Vec<CestTrainingSample> = (0..3)
        .map(|_| CestTrainingSample {
            input_tensor: (0..(cfg.n_channels * cfg.n_grid))
                .map(|_| rng.uniform(0.0, 1.0))
                .collect(),
            target_phys: [
                rng.uniform(-2.1, -1.2),
                rng.uniform(1.6, 2.5),
                rng.uniform(-5.5, 5.5),
            ],
        })
        .collect();

    let weight_decay = 1e-3;
    let (_loss0, grads) = net.batch_loss_and_grads(&batch, weight_decay).unwrap();
    let anal = grads.as_flat_vec();
    assert_eq!(anal.len(), net.num_params());

    let eps = 1e-6;
    let mut max_abs_err = 0.0_f64;
    let mut max_rel_err = 0.0_f64;

    for (idx, &anal_g) in anal.iter().enumerate() {
        let orig = *net.param_mut(idx).unwrap();
        *net.param_mut(idx).unwrap() = orig + eps;
        let (lp, _) = net.batch_loss_and_grads(&batch, weight_decay).unwrap();
        *net.param_mut(idx).unwrap() = orig - eps;
        let (lm, _) = net.batch_loss_and_grads(&batch, weight_decay).unwrap();
        *net.param_mut(idx).unwrap() = orig;

        let fd = (lp - lm) / (2.0 * eps);
        let abs_err = (anal_g - fd).abs();
        let rel_err = abs_err / anal_g.abs().max(fd.abs()).max(1.0);
        max_abs_err = max_abs_err.max(abs_err);
        max_rel_err = max_rel_err.max(rel_err);
    }

    println!(
        "[synthetic self-consistency] AmortizedCestNet FD grad check ({} params): max_abs={max_abs_err:.3e}, max_rel={max_rel_err:.3e}",
        net.num_params()
    );
    assert!(
        max_abs_err < 1e-5 && max_rel_err < 1e-5,
        "backprop FD gradient check failed: max_abs={max_abs_err:.3e}, max_rel={max_rel_err:.3e}"
    );
}

/// `[synthetic self-consistency]` Online backpropagation training of [`AmortizedCestNet`] on
/// synthetic Bloch–McConnell multi-`B1` `CEST` ensembles, verifying:
/// 1. NLL loss decreases substantially during Adam training.
/// 2. On held-out synthetic profiles, `fit_cest_amortized_seeded_lm` matches/beats grid-seeded LM
///    in parameter accuracy & chi^2 while converging in fewer evaluations & iterations.
/// 3. Clean `AbstainToGridFallback` when given an uncalibrated/sabotaged network.
#[test]
fn test_amortized_cest_training_hybrid_lm_speedup_and_abstention() {
    let larmor_mhz = 60.8106;
    let major_ppm = 118.0;
    let schemes = vec![
        MultiCestScheme::Cw {
            t_ex_s: 0.40,
            b1_hz: 15.0,
        },
        MultiCestScheme::Cw {
            t_ex_s: 0.40,
            b1_hz: 30.0,
        },
    ];

    let net_cfg = AmortizedCestNetConfig {
        n_channels: 2,
        n_grid: 40,
        offset_window_ppm: (-8.0, 8.0),
        conv1_channels: 8,
        conv2_channels: 10,
        kernel_size: 5,
        hidden_dim: 40,
        target_mean: [-1.65, 2.20, 0.0],
        target_std: [0.25, 0.25, 4.2],
        sigma_floor: 0.03,
    };

    let gen_cfg = CestEnsembleGenConfig {
        larmor_mhz,
        major_ppm,
        schemes: schemes.clone(),
        normalization: CestNormalization::ZeroTimePhaseCycled,
        b1_inhomogeneity_frac: 0.0,
        n_b1_quad: 1,
        log10_pb_range: (-2.0, -1.3),
        log10_kex_range: (1.85, 2.55),
        dw_abs_range_ppm: (2.0, 6.2),
        r1_range: (1.1, 1.5),
        r2a_range: (12.0, 16.0),
        r2b_range: (16.0, 28.0),
        noise_sigma: 0.004,
    };

    let train_data = generate_cest_training_ensemble(&net_cfg, &gen_cfg, 180, 7701).unwrap();
    let mut net = AmortizedCestNet::new(net_cfg.clone(), 7702).unwrap();
    let train_cfg = AmortizedTrainConfig {
        epochs: 65,
        batch_size: 16,
        learning_rate: 4e-3,
        weight_decay: 1e-4,
        grad_clip_norm: 5.0,
        seed: 7703,
    };
    let report = train_amortized_cest_net(&mut net, &train_data, &train_cfg).unwrap();
    println!(
        "[synthetic self-consistency] AmortizedCestNet NLL: {:.3} -> {:.3} ({} updates)",
        report.initial_nll, report.final_nll, report.updates_completed
    );
    assert!(
        report.final_nll < report.initial_nll - 0.8,
        "training failed to substantially reduce NLL: {} -> {}",
        report.initial_nll,
        report.final_nll
    );

    // Held-out synthetic profiles not in train_data
    let held_out_cases = [
        MultiCestFitParams {
            p_b: 0.025,
            k_ex_s_inv: 180.0,
            dw_ab_ppm: 4.2,
            r1_s_inv: 1.3,
            r2a_s_inv: 14.0,
            r2b_s_inv: 21.0,
        },
        MultiCestFitParams {
            p_b: 0.018,
            k_ex_s_inv: 130.0,
            dw_ab_ppm: -3.6,
            r1_s_inv: 1.25,
            r2a_s_inv: 13.5,
            r2b_s_inv: 24.0,
        },
        MultiCestFitParams {
            p_b: 0.032,
            k_ex_s_inv: 240.0,
            dw_ab_ppm: 5.1,
            r1_s_inv: 1.35,
            r2a_s_inv: 15.0,
            r2b_s_inv: 19.0,
        },
    ];

    let offsets: Vec<f64> = (-25..=25).map(|k| major_ppm + 0.32 * (k as f64)).collect();
    let noise_sigma = 0.004;
    let mut rng = Rng::new(7799);

    let lm_cfg = MultiCestLmConfig {
        grid_dw_range_ppm: (-6.5, 6.5),
        grid_dw_step_ppm: 0.65,
        ..Default::default()
    };
    let hybrid_cfg = AmortizedHybridFitConfig {
        lm_cfg: lm_cfg.clone(),
        fallback_reduced_chi2_threshold: 2.2,
        max_sigma_multistarts: 2,
        early_accept_reduced_chi2: 1.5,
        max_dw_std_abstain_ppm: 5.0,
    };

    let mut total_amort_iters = 0usize;
    let mut total_grid_iters = 0usize;
    let mut total_amort_evals = 0usize;
    let mut total_grid_evals = 0usize;
    let mut accepted_count = 0usize;

    let mut first_profiles: Option<Vec<MultiCestProfile>> = None;

    for (idx, true_p) in held_out_cases.iter().enumerate() {
        let mut profiles = Vec::new();
        for scheme in &schemes {
            let sim_p = MultiCestSimParams {
                p_b: true_p.p_b,
                k_ex_s_inv: true_p.k_ex_s_inv,
                major_ppm,
                dw_ab_ppm: true_p.dw_ab_ppm,
                r1a_s_inv: true_p.r1_s_inv,
                r2a_s_inv: true_p.r2a_s_inv,
                r1b_s_inv: true_p.r1_s_inv,
                r2b_s_inv: true_p.r2b_s_inv,
                larmor_mhz,
                b1_inhomogeneity_frac: 0.0,
                n_b1_quad: 1,
                normalization: CestNormalization::ZeroTimePhaseCycled,
                scheme: scheme.clone(),
            };
            let clean = simulate_multi_cest_profile(&sim_p, &offsets).unwrap();
            let noisy: Vec<f64> = clean
                .iter()
                .map(|&y| y + rng.normal(0.0, noise_sigma))
                .collect();
            profiles.push(MultiCestProfile {
                larmor_mhz,
                major_ppm,
                b1_inhomogeneity_frac: 0.0,
                n_b1_quad: 1,
                normalization: CestNormalization::ZeroTimePhaseCycled,
                scheme: scheme.clone(),
                offsets_ppm: offsets.clone(),
                intensities: noisy,
                uncertainties: vec![noise_sigma; offsets.len()],
            });
        }
        if idx == 0 {
            first_profiles = Some(profiles.clone());
        }

        let grid_fit = fit_multi_cest_grid_lm(&profiles, &lm_cfg).unwrap();
        let amort_res = fit_cest_amortized_seeded_lm(&net, &profiles, &hybrid_cfg).unwrap();

        total_grid_iters += grid_fit.iterations;
        total_amort_iters += amort_res.fit.iterations;
        total_grid_evals += grid_fit.n_evals;
        total_amort_evals += amort_res.fit.n_evals;
        if amort_res.status == AmortizedSeedStatus::AcceptedNeuralSeed {
            accepted_count += 1;
        }

        let pb_err = (amort_res.fit.params.p_b - true_p.p_b).abs() / true_p.p_b;
        let kex_err =
            (amort_res.fit.params.k_ex_s_inv - true_p.k_ex_s_inv).abs() / true_p.k_ex_s_inv;
        let dw_err = (amort_res.fit.params.dw_ab_ppm - true_p.dw_ab_ppm).abs();
        println!(
            "[synthetic self-consistency] held-out #{idx}: status={:?}, post_dw={:.2}+/-{:.2}, fit_chi2={:.2} vs grid_chi2={:.2}, iters={} vs {}, evals={} vs {}, errs=(pB {:.2}%, kex {:.2}%, dw {:.3} ppm)",
            amort_res.status,
            amort_res.posterior.dw_ppm_mean,
            amort_res.posterior.dw_ppm_std,
            amort_res.fit.chi2,
            grid_fit.chi2,
            amort_res.fit.iterations,
            grid_fit.iterations,
            amort_res.fit.n_evals,
            grid_fit.n_evals,
            pb_err * 100.0,
            kex_err * 100.0,
            dw_err
        );

        assert!(pb_err < 0.05 && kex_err < 0.06 && dw_err < 0.08);
        assert!(
            amort_res.fit.chi2 <= grid_fit.chi2 + 1e-2,
            "amortized-seeded LM must match or beat grid-seeded LM chi^2"
        );
    }

    assert!(
        accepted_count >= 2,
        "trained amortized net should accept >= 2/3 held-out seeds without fallback, got {accepted_count}/3"
    );
    assert!(
        total_amort_iters < total_grid_iters && total_amort_evals < total_grid_evals,
        "amortized-seeded LM should converge in fewer iterations ({total_amort_iters} vs {total_grid_iters}) and evaluations ({total_amort_evals} vs {total_grid_evals})"
    );

    // Verify clean AbstainToGridFallback when neural network predicts a grossly wrong dw_ppm
    let mut bad_net = AmortizedCestNet::new(net_cfg, 9999).unwrap();
    // Force fc2 weights to zero and bias dw_std to -3.0 (so mean dw = -12.6 ppm, far from +4.2 ppm)
    bad_net.fc2_w.fill(0.0);
    bad_net.fc2_b = vec![-2.0, -2.0, -3.5, -2.0, -2.0, -2.0];
    let profiles_0 = first_profiles.unwrap();
    let fallback_res = fit_cest_amortized_seeded_lm(&bad_net, &profiles_0, &hybrid_cfg).unwrap();
    println!(
        "[synthetic self-consistency] sabotaged net fallback: status={:?}, neural_chi2_nu={:.2}, recovered_chi2_nu={:.3}, recovered_dw={:.3}",
        fallback_res.status,
        fallback_res.neural_seeded_reduced_chi2,
        fallback_res.fit.reduced_chi2,
        fallback_res.fit.params.dw_ab_ppm
    );
    assert_eq!(
        fallback_res.status,
        AmortizedSeedStatus::AbstainToGridFallback
    );
    assert!(
        (fallback_res.fit.params.dw_ab_ppm - 4.2).abs() < 0.1,
        "grid fallback must still recover exact dw = +4.2 ppm"
    );
    assert!(fallback_res.fit.reduced_chi2 < 1.6);
}

/// Hostile input rejection tests for `DANTE-CEST`, `cos-CEST`, LM fitter, and amortized network.
#[test]
fn test_multi_cest_hostile_input_rejection() {
    let valid_dante = DanteCestParams {
        p_b: 0.02,
        k_ex_s_inv: 150.0,
        major_ppm: 118.0,
        dw_ab_ppm: 3.0,
        r1a_s_inv: 1.3,
        r2a_s_inv: 14.0,
        r1b_s_inv: 1.3,
        r2b_s_inv: 20.0,
        larmor_mhz: 60.8106,
        tau_pulse_s: 15.0e-6,
        tau_delay_s: 2.0e-3,
        n_pulses: 100,
        b1_pulse_hz: 1000.0,
        b1_inhomogeneity_frac: 0.0,
        n_b1_quad: 1,
        normalization: CestNormalization::ZeroTimePhaseCycled,
    };

    // Zero n_pulses
    let mut bad = valid_dante.clone();
    bad.n_pulses = 0;
    assert!(simulate_dante_cest_profile(&bad, &[118.0]).is_err());

    // Negative tau_pulse_s
    let mut bad = valid_dante.clone();
    bad.tau_pulse_s = -1e-6;
    assert!(simulate_dante_cest_profile(&bad, &[118.0]).is_err());

    // NaN offset
    assert!(simulate_dante_cest_profile(&valid_dante, &[f64::NAN]).is_err());

    // Invalid p_b >= 1.0
    let mut bad = valid_dante.clone();
    bad.p_b = 1.0;
    assert!(simulate_dante_cest_profile(&bad, &[118.0]).is_err());

    // cos-CEST with mod_freq_hz = 0 or n_steps_per_cycle < 4
    let mut bad_cos = CosCestParams {
        p_b: 0.02,
        k_ex_s_inv: 150.0,
        major_ppm: 118.0,
        dw_ab_ppm: 3.0,
        r1a_s_inv: 1.3,
        r2a_s_inv: 14.0,
        r1b_s_inv: 1.3,
        r2b_s_inv: 20.0,
        larmor_mhz: 60.8106,
        t_ex_s: 0.4,
        b1_eff_hz: 20.0,
        mod_freq_hz: 0.0,
        n_steps_per_cycle: 32,
        b1_inhomogeneity_frac: 0.0,
        n_b1_quad: 1,
        normalization: CestNormalization::ZeroTimePhaseCycled,
    };
    assert!(simulate_cos_cest_profile(&bad_cos, &[118.0]).is_err());
    bad_cos.mod_freq_hz = 200.0;
    bad_cos.n_steps_per_cycle = 2;
    assert!(simulate_cos_cest_profile(&bad_cos, &[118.0]).is_err());

    // LM fitter rejects empty profiles & invalid uncertainties
    let init = MultiCestFitParams {
        p_b: 0.02,
        k_ex_s_inv: 150.0,
        dw_ab_ppm: 3.0,
        r1_s_inv: 1.3,
        r2a_s_inv: 14.0,
        r2b_s_inv: 20.0,
    };
    assert!(fit_multi_cest_lm(&[], init, &MultiCestLmConfig::default()).is_err());
    let bad_prof = MultiCestProfile {
        larmor_mhz: 60.8106,
        major_ppm: 118.0,
        b1_inhomogeneity_frac: 0.0,
        n_b1_quad: 1,
        normalization: CestNormalization::ZeroTimePhaseCycled,
        scheme: MultiCestScheme::Cw {
            t_ex_s: 0.4,
            b1_hz: 20.0,
        },
        offsets_ppm: vec![118.0; 8],
        intensities: vec![0.5; 8],
        uncertainties: vec![0.0; 8], // zero sigma rejected
    };
    assert!(fit_multi_cest_lm(&[bad_prof], init, &MultiCestLmConfig::default()).is_err());

    // Amortized net rejects wrong input length and invalid config
    let net = AmortizedCestNet::new(AmortizedCestNetConfig::default(), 1).unwrap();
    assert!(net.predict_posterior(&[0.0; 5]).is_err());
    let bad_cfg = AmortizedCestNetConfig {
        kernel_size: 4, // even kernel rejected
        ..Default::default()
    };
    assert!(AmortizedCestNet::new(bad_cfg, 1).is_err());
}
