//! `[synthetic self-consistency]` tests for `nmr_core::ccr_relax` (`T22.5`):
//! - Goldman (1984) / Tjandra, Szabo & Bax (1996) / Lee et al. (2006) dipolar–CSA CCR prefactors
//!   and $\eta_{xy}/\eta_z$ invariance.
//! - `TRACT` $\alpha/\beta$ decay fitting and $R_{\text{ex}}$-free $\tau_m$ inversion across
//!   $\tau_m \in \{4, 10, 25, 60\}\text{ ns}$ vs biased naive $R_2 / R_1$.
//! - Kroenke–Palmer (1998) $R_{\text{ex}}$ separation from paired $(R_2, \eta_{xy})$.
//! - Tjandra (1996) site-specific $^{15}\text{N}$ CSA determination & physical-range diagnostics.
//! - Hostile / invalid input fail-closed behavior.

use nmr_core::ccr_relax::{
    compute_ccr_rates_isotropic, compute_full_relax_and_ccr_isotropic, fit_site_specific_csa,
    fit_site_specific_csa_batch, fit_tract_batch, fit_tract_decay_pair, invert_eta_ratio_to_tau_m,
    invert_tract_eta_xy_to_tau_m, separate_rex_kroenke_palmer, CcrPhysicsConfig, CsaRangeStatus,
    KroenkeConfig, KroenkeResidueInput, SiteCsaConfig, SiteCsaFieldObs, SiteCsaInput, TractConfig,
    TractDecayInput, CSA_N_PHYSICAL_MAX_PPM, CSA_N_PHYSICAL_MIN_PPM,
    ETA_RATIO_EXTREME_NARROWING_LIMIT,
};
use nmr_core::relax_fit::{
    invert_r2_over_r1_to_tau_c, GAMMA_15N_RAD_S_T, GAMMA_1H_RAD_S_T, HBAR_J_S, MU0_OVER_4PI,
};
use nmr_core::rng::Rng;

#[test]
fn synthetic_self_consistency_goldman_tjandra_lee_prefactors_and_cauchy_schwarz() {
    let phys = CcrPhysicsConfig::default();
    let p2 = phys.p2_cos_theta();
    let expected_p2 = 0.5 * (3.0 * 17.0_f64.to_radians().cos().powi(2) - 1.0);
    assert!(
        (p2 - expected_p2).abs() < 1.0e-15,
        "P2(cos 17 deg) mismatch"
    );

    for &field_mhz in &[500.0_f64, 600.0, 800.0, 900.0] {
        // 1. Lee et al. (2006) form: -2 * p * delta_N * P2(cos theta)
        let pref_lee = -2.0 * phys.dipolar_p_rad_s() * phys.csa_delta_n_rad_s(field_mhz) * p2;

        // 2. Tjandra (1996) / Kroenke (1998) form: -(sqrt(3)/6) * c * d_signed * P2(cos theta)
        let pref_tjandra = -(3.0_f64.sqrt() / 6.0)
            * phys.csa_c_rad_s(field_mhz)
            * phys.dipolar_d_signed_rad_s()
            * p2;

        // 3. Fundamental CODATA-2018 form: -(mu0 * hbar * gH * gN^2 * B0 * dSigma / (24 pi r^3)) * P2
        let r_m = phys.r_nh_angstrom * 1.0e-10;
        let b0 = phys.b0_tesla(field_mhz);
        let pref_fund = -(MU0_OVER_4PI
            * HBAR_J_S
            * GAMMA_1H_RAD_S_T
            * GAMMA_15N_RAD_S_T
            * GAMMA_15N_RAD_S_T
            * b0
            * (phys.csa_n_ppm * 1.0e-6)
            / (6.0 * r_m * r_m * r_m))
            * p2;

        let pref_api = phys.ccr_prefactor(field_mhz);
        assert!(pref_api > 0.0, "CCR prefactor must be > 0 for 15N");
        assert!(
            ((pref_lee - pref_tjandra) / pref_api).abs() < 1.0e-14,
            "Lee vs Tjandra prefactor mismatch at {field_mhz} MHz"
        );
        assert!(
            ((pref_lee - pref_fund) / pref_api).abs() < 1.0e-14,
            "Lee vs fundamental prefactor mismatch at {field_mhz} MHz"
        );

        let ccr = compute_ccr_rates_isotropic(10.0e-9, 1.0, 0.0, field_mhz, &phys)
            .expect("valid CCR calculation");
        assert!(ccr.eta_xy > 0.0);
        assert!(ccr.eta_z > 0.0);
        assert!((ccr.r_beta_minus_r_alpha - 2.0 * ccr.eta_xy).abs() < 1.0e-14);
    }

    // 4. Exact Cauchy–Schwarz TROSY perfection check at theta = 0 deg and the magic field where |p| == |delta_N|
    let phys_collinear = CcrPhysicsConfig {
        theta_deg: 0.0,
        ..CcrPhysicsConfig::default()
    };
    // Magic field where |p| = |delta_N| => d / (2 sqrt(2)) = |omega_N Delta_sigma| / (3 sqrt(2))
    let d = phys_collinear.dipolar_d_rad_s();
    let omega_n_magic = 1.5 * d / (phys_collinear.csa_n_ppm.abs() * 1.0e-6);
    let omega_h_magic = omega_n_magic * (phys_collinear.gamma_h / phys_collinear.gamma_n.abs());
    let field_magic_mhz = omega_h_magic / (2.0 * std::f64::consts::PI * 1.0e6);

    let p_abs = phys_collinear.dipolar_p_rad_s().abs();
    let delta_abs = phys_collinear.csa_delta_n_rad_s(field_magic_mhz).abs();
    assert!(((p_abs - delta_abs) / p_abs).abs() < 1.0e-14);

    let obs_magic = compute_full_relax_and_ccr_isotropic(
        20.0e-9,
        1.0,
        0.0,
        0.0,
        field_magic_mhz,
        &phys_collinear,
    )
    .expect("magic field observables");
    let c = phys_collinear.csa_c_rad_s(field_magic_mhz);
    // Secular transverse cancellation: (d^2/8 + c^2/6) - eta_xy/(4 J(0) + 3 J(wN)) == 0
    let sec_residual =
        (0.125 * d * d + (c * c) / 6.0) - phys_collinear.ccr_prefactor(field_magic_mhz);
    assert!(
        sec_residual.abs() / (0.125 * d * d) < 1.0e-14,
        "secular DD+CSA-CCR cancellation must vanish at magic field (got {sec_residual})"
    );
    // Longitudinal N-frequency cancellation: (3/4 d^2 + c^2) - 6 * prefactor == 0
    let long_residual =
        (0.75 * d * d + c * c) - 6.0 * phys_collinear.ccr_prefactor(field_magic_mhz);
    assert!(
        long_residual.abs() / (0.75 * d * d) < 1.0e-14,
        "longitudinal DD+CSA-CCR cancellation must vanish at magic field (got {long_residual})"
    );
    // R_alpha remains strictly positive (non-secular proton DD terms remain)
    assert!(obs_magic.r_alpha > 0.0 && obs_magic.r_beta > obs_magic.r_alpha);
}

#[test]
fn synthetic_self_consistency_eta_ratio_invariance_and_exact_roundtrip() {
    let field_mhz = 600.0;
    let tau_m = 12.5e-9;
    let base_phys = CcrPhysicsConfig::default();
    let base_ccr =
        compute_ccr_rates_isotropic(tau_m, 1.0, 0.0, field_mhz, &base_phys).expect("base CCR");

    // Vary S^2, Delta_sigma_N, r_NH, and theta simultaneously: eta_xy / eta_z must stay invariant to < 1e-12!
    let configs = [
        (0.35, -135.0, 1.00, 5.0),
        (0.70, -160.0, 1.02, 17.0),
        (0.85, -172.0, 1.04, 24.0),
        (1.00, -215.0, 1.01, 32.0),
    ];
    for &(s2, csa_ppm, r_nh, theta_deg) in &configs {
        let p = CcrPhysicsConfig {
            r_nh_angstrom: r_nh,
            csa_n_ppm: csa_ppm,
            theta_deg,
            ..CcrPhysicsConfig::default()
        };
        let ccr = compute_ccr_rates_isotropic(tau_m, s2, 0.0, field_mhz, &p).expect("varied CCR");
        assert!(
            (ccr.ratio_xy_over_z - base_ccr.ratio_xy_over_z).abs() < 1.0e-12,
            "eta_xy/eta_z violated invariance for (S2={s2}, csa={csa_ppm}, r={r_nh}, theta={theta_deg})"
        );

        let inv = invert_eta_ratio_to_tau_m(ccr.eta_xy, 0.0, ccr.eta_z, 0.0, field_mhz, &p)
            .expect("ratio inversion");
        assert!(
            ((inv.tau_m_sec - tau_m) / tau_m).abs() < 1.0e-12,
            "eta_xy/eta_z inversion error too large"
        );
        assert!(
            (inv.implied_s2 - s2).abs() < 1.0e-12,
            "implied S^2 mismatch: got {}, expected {}",
            inv.implied_s2,
            s2
        );
    }

    // Exact round-trip of invert_tract_eta_xy_to_tau_m across 0.5 ns .. 120 ns (rigid & Lipari–Szabo)
    for &tm_ns in &[0.5_f64, 4.0, 10.0, 25.0, 60.0, 120.0] {
        let tm = tm_ns * 1.0e-9;
        let rigid = compute_ccr_rates_isotropic(tm, 1.0, 0.0, field_mhz, &base_phys)
            .expect("rigid forward");
        let inv_rigid =
            invert_tract_eta_xy_to_tau_m(rigid.eta_xy, 0.0, field_mhz, 1.0, 0.0, &base_phys)
                .expect("rigid invert");
        assert!(
            ((inv_rigid.tau_m_sec - tm) / tm).abs() < 1.0e-12,
            "rigid TRACT round-trip failed at {tm_ns} ns"
        );

        let mf = compute_ccr_rates_isotropic(tm, 0.85, 45.0e-12, field_mhz, &base_phys)
            .expect("model-free forward");
        let inv_mf =
            invert_tract_eta_xy_to_tau_m(mf.eta_xy, 0.0, field_mhz, 0.85, 45.0e-12, &base_phys)
                .expect("model-free invert");
        assert!(
            ((inv_mf.tau_m_sec - tm) / tm).abs() < 1.0e-12,
            "S^2 + tau_e TRACT round-trip failed at {tm_ns} ns"
        );
    }
}

#[test]
fn synthetic_self_consistency_tract_rex_invariance_vs_biased_r2_r1() {
    let field_mhz = 600.0;
    let phys = CcrPhysicsConfig::default();
    let relax_phys = phys.to_relax_physics();
    let planted_rex = 15.0_f64;
    let mut rng = Rng::new(2_026_220_501);

    for &tau_m_ns in &[4.0_f64, 10.0, 25.0, 60.0] {
        let tau_m_true = tau_m_ns * 1.0e-9;
        let obs = compute_full_relax_and_ccr_isotropic(
            tau_m_true,
            1.0,
            0.0,
            planted_rex,
            field_mhz,
            &phys,
        )
        .expect("synthetic forward with R_ex");

        // Design realistic delay schedules matched to alpha (TROSY) and beta (anti-TROSY) decay rates
        let n_pts = 10usize;
        let t_max_a = (2.2 / obs.r_alpha).min(0.25);
        let t_max_b = (2.2 / obs.r_beta).min(0.15);
        let i0_a = 1000.0;
        let i0_b = 1000.0;
        let noise_sd = 1.5; // SNR ~ 667 at t=0, typical for 1D TRACT bulk envelope

        let mut ta = Vec::with_capacity(n_pts);
        let mut ia = Vec::with_capacity(n_pts);
        let mut sa = Vec::with_capacity(n_pts);
        let mut tb = Vec::with_capacity(n_pts);
        let mut ib = Vec::with_capacity(n_pts);
        let mut sb = Vec::with_capacity(n_pts);

        for k in 0..n_pts {
            let frac = (k as f64) / ((n_pts - 1) as f64);
            let t_ak = frac * t_max_a;
            let t_bk = frac * t_max_b;
            ta.push(t_ak);
            ia.push(i0_a * (-obs.r_alpha * t_ak).exp() + rng.normal(0.0, noise_sd));
            sa.push(noise_sd);
            tb.push(t_bk);
            ib.push(i0_b * (-obs.r_beta * t_bk).exp() + rng.normal(0.0, noise_sd));
            sb.push(noise_sd);
        }

        let input = TractDecayInput {
            label: format!("1D-TRACT-{tau_m_ns:.0}ns"),
            seq_id: None,
            field_mhz,
            alpha_times_sec: ta,
            alpha_intensities: ia,
            alpha_sigmas: sa,
            beta_times_sec: tb,
            beta_intensities: ib,
            beta_sigmas: sb,
            eta_z: Some(obs.eta_z),
            sigma_eta_z: Some(0.002 * obs.eta_z),
        };
        let cfg = TractConfig {
            physics: phys,
            order_parameter_s2: None,
            tau_e_sec: 0.0,
            max_rate_s_inv: 500.0,
            mc_replicates: 0,
            seed: 42,
        };

        let fit = fit_tract_decay_pair(&input, &cfg).expect("TRACT fit");
        let rel_err_tract = ((fit.tau_m_rigid.tau_m_sec - tau_m_true) / tau_m_true).abs();
        assert!(
            rel_err_tract < 0.01,
            "TRACT tau_m at {tau_m_ns} ns with R_ex=15 s^-1 drifted by {:.3}% (got {:.3} ns)",
            100.0 * rel_err_tract,
            fit.tau_m_rigid.tau_m_sec * 1.0e9
        );

        let ratio_est = fit.tau_m_from_eta_ratio.expect("eta_xy/eta_z tau_m");
        let rel_err_ratio = ((ratio_est.tau_m_sec - tau_m_true) / tau_m_true).abs();
        assert!(
            rel_err_ratio < 0.01,
            "eta_xy/eta_z tau_m at {tau_m_ns} ns drifted by {:.3}%",
            100.0 * rel_err_ratio
        );

        // Compare with naive R2/R1 inversion on the exact same R_ex-contaminated rates: severely biased!
        // Note: in the slow-tumbling limit R2/R1 ~ tau_m^2, so d(tau_m)/tau_m ~ 0.5 * R_ex / R2^0
        // (>80% at 4 ns, >40% at 10 ns, >18% at 25 ns, and ~8.4% = +5 ns even at 60 ns).
        let naive_tau_m = invert_r2_over_r1_to_tau_c(obs.r2 / obs.r1, field_mhz, &relax_phys);
        let rel_err_naive = (naive_tau_m - tau_m_true) / tau_m_true;
        assert!(
            rel_err_naive > 0.08 && rel_err_naive > 10.0 * rel_err_tract,
            "naive R2/R1 at {tau_m_ns} ns with R_ex=15 s^-1 should be severely overestimated vs TRACT (naive {:+.1}%, TRACT {:.3}%)",
            100.0 * rel_err_naive,
            100.0 * rel_err_tract
        );
        if tau_m_ns <= 25.0 {
            assert!(
                rel_err_naive > 0.18,
                "naive R2/R1 at {tau_m_ns} ns with R_ex=15 s^-1 should exceed +18% bias (got {:+.1}%)",
                100.0 * rel_err_naive
            );
        }
    }

    // Order-parameter-corrected 2D TRACT batch + parametric MC bootstrap test
    let s2_true = 0.85;
    let tau_m_true = 15.0e-9;
    let mut batch_inputs = Vec::new();
    for (idx, &rex) in [0.0_f64, 12.0, 25.0].iter().enumerate() {
        let obs =
            compute_full_relax_and_ccr_isotropic(tau_m_true, s2_true, 0.0, rex, field_mhz, &phys)
                .expect("2D TRACT forward");
        let times_a: Vec<f64> = (0..8).map(|k| (k as f64) * 0.015).collect();
        let times_b: Vec<f64> = (0..8).map(|k| (k as f64) * 0.005).collect();
        let ia: Vec<f64> = times_a
            .iter()
            .map(|&t| 500.0 * (-obs.r_alpha * t).exp())
            .collect();
        let ib: Vec<f64> = times_b
            .iter()
            .map(|&t| 500.0 * (-obs.r_beta * t).exp())
            .collect();
        batch_inputs.push(TractDecayInput {
            label: format!("RES-{}", idx + 10),
            seq_id: Some((idx + 10) as i32),
            field_mhz,
            alpha_times_sec: times_a,
            alpha_intensities: ia,
            alpha_sigmas: vec![1.0; 8],
            beta_times_sec: times_b,
            beta_intensities: ib,
            beta_sigmas: vec![1.0; 8],
            eta_z: None,
            sigma_eta_z: None,
        });
    }
    let cfg_s2 = TractConfig {
        physics: phys,
        order_parameter_s2: Some(s2_true),
        tau_e_sec: 0.0,
        max_rate_s_inv: 500.0,
        mc_replicates: 40,
        seed: 777,
    };
    let batch_res = fit_tract_batch(&batch_inputs, &cfg_s2).expect("2D TRACT batch");
    assert_eq!(batch_res.len(), 3);
    for r in &batch_res {
        // Rigid lower bound underestimates true tau_m (~ s2 * tau_m)
        assert!(r.tau_m_rigid.tau_m_sec < tau_m_true);
        // S^2-corrected TRACT recovers exact tau_m across all R_ex values!
        let corr = r.tau_m_s2_corrected.expect("S^2-corrected tau_m");
        assert!(
            ((corr.tau_m_sec - tau_m_true) / tau_m_true).abs() < 1.0e-5,
            "S^2-corrected TRACT failed: got {:.4} ns",
            corr.tau_m_sec * 1.0e9
        );
        assert!(corr.sigma_tau_m_sec > 0.0 && corr.sigma_tau_m_sec.is_finite());
    }
}

#[test]
fn synthetic_self_consistency_kroenke_palmer_rex_separation_zero_false_positives() {
    let field_mhz = 600.0;
    let phys = CcrPhysicsConfig::default();
    let tau_m = 10.0e-9;
    let mut rng = Rng::new(2_026_220_502);

    // 20-residue protein: 18 rigid residues with heterogeneous S^2 + 2 planted exchanging sites (8 & 20 s^-1)
    let mut inputs_pair_only = Vec::new();
    let mut inputs_with_hf = Vec::new();

    for i in 1..=20 {
        let s2 = 0.76 + 0.01 * ((i % 15) as f64);
        let tau_e = 25.0e-12;
        let planted_rex = match i {
            7 => 8.0,
            14 => 20.0,
            _ => 0.0,
        };
        let obs =
            compute_full_relax_and_ccr_isotropic(tau_m, s2, tau_e, planted_rex, field_mhz, &phys)
                .expect("synthetic Kroenke forward");

        let sig_r2 = 0.18;
        let sig_eta = 0.15;
        let sig_r1 = 0.02;
        let sig_noe = 0.015;

        let r2_obs = obs.r2 + rng.normal(0.0, sig_r2);
        let eta_obs = obs.eta_xy + rng.normal(0.0, sig_eta);
        let r1_obs = obs.r1 + rng.normal(0.0, sig_r1);
        let noe_obs = obs.hetnoe + rng.normal(0.0, sig_noe);

        inputs_pair_only.push(KroenkeResidueInput {
            seq_id: i,
            res_name: format!("R{i:02}"),
            r2: r2_obs,
            sigma_r2: sig_r2,
            eta_xy: eta_obs,
            sigma_eta_xy: sig_eta,
            r1: None,
            sigma_r1: None,
            hetnoe: None,
            sigma_hetnoe: None,
        });
        inputs_with_hf.push(KroenkeResidueInput {
            seq_id: i,
            res_name: format!("R{i:02}"),
            r2: r2_obs,
            sigma_r2: sig_r2,
            eta_xy: eta_obs,
            sigma_eta_xy: sig_eta,
            r1: Some(r1_obs),
            sigma_r1: Some(sig_r1),
            hetnoe: Some(noe_obs),
            sigma_hetnoe: Some(sig_noe),
        });
    }

    let cfg = KroenkeConfig {
        field_mhz,
        physics: phys,
        ..KroenkeConfig::default()
    };

    for (mode, data) in [
        ("paired (R2, eta_xy)", &inputs_pair_only),
        ("HF-corrected (R2, eta_xy, R1, hetNOE)", &inputs_with_hf),
    ] {
        let rep = separate_rex_kroenke_palmer(data, &cfg).expect("Kroenke-Palmer separation");
        assert_eq!(
            rep.exchanging_seq_ids,
            vec![7, 14],
            "{mode}: expected exactly residues 7 and 14 exchanging (0 false positives), got {:?}",
            rep.exchanging_seq_ids
        );
        assert!(
            (rep.kappa0 - rep.theoretical_kappa0).abs() < 0.05,
            "{mode}: empirical kappa0 {:.4} diverged from theory {:.4}",
            rep.kappa0,
            rep.theoretical_kappa0
        );

        for res in &rep.residues {
            match res.seq_id {
                7 => {
                    assert!(res.has_significant_rex);
                    assert!(
                        (res.r_ex - 8.0).abs() < 0.8,
                        "{mode}: res 7 R_ex = {:.2}, expected ~8.0",
                        res.r_ex
                    );
                    assert!(!res.used_in_rigid_baseline);
                }
                14 => {
                    assert!(res.has_significant_rex);
                    assert!(
                        (res.r_ex - 20.0).abs() < 0.8,
                        "{mode}: res 14 R_ex = {:.2}, expected ~20.0",
                        res.r_ex
                    );
                    assert!(!res.used_in_rigid_baseline);
                }
                _ => {
                    assert!(
                        !res.has_significant_rex,
                        "{mode}: false positive on rigid residue {} (R_ex={:.2}, z={:.2})",
                        res.seq_id, res.r_ex, res.z_score
                    );
                }
            }
        }
    }
}

#[test]
fn synthetic_self_consistency_tjandra_site_specific_csa_recovery_and_diagnostics() {
    let base_phys = CcrPhysicsConfig::default();
    let cfg = SiteCsaConfig {
        physics: base_phys,
        ..SiteCsaConfig::default()
    };
    let mut rng = Rng::new(2_026_220_503);

    // 1. Multi-field (500 + 800 MHz) & Single-field (600 MHz) recovery of -155, -172, -190 ppm within < 2 ppm
    let targets = [
        (10, "ALA", -155.0_f64),
        (20, "LEU", -172.0),
        (30, "VAL", -190.0),
    ];

    let mut multi_inputs = Vec::new();
    let mut single_inputs = Vec::new();

    for &(seq_id, name, true_csa) in &targets {
        let p_true = CcrPhysicsConfig {
            csa_n_ppm: true_csa,
            ..base_phys
        };

        let make_field_obs = |field_mhz: f64, r_ex: f64, r: &mut Rng| -> SiteCsaFieldObs {
            let obs = compute_full_relax_and_ccr_isotropic(
                8.0e-9, 0.85, 50.0e-12, r_ex, field_mhz, &p_true,
            )
            .expect("synthetic CSA forward");
            let sr1 = 0.005 * obs.r1;
            let sr2 = 0.005 * obs.r2;
            let snoe = 0.005;
            let sexy = 0.005 * obs.eta_xy;
            let sez = 0.005 * obs.eta_z;
            SiteCsaFieldObs {
                field_mhz,
                r1: Some(obs.r1 + r.normal(0.0, sr1)),
                sigma_r1: Some(sr1),
                r2: Some(obs.r2 + r.normal(0.0, sr2)),
                sigma_r2: Some(sr2),
                hetnoe: Some(obs.hetnoe + r.normal(0.0, snoe)),
                sigma_hetnoe: Some(snoe),
                eta_xy: Some(obs.eta_xy + r.normal(0.0, sexy)),
                sigma_eta_xy: Some(sexy),
                eta_z: Some(obs.eta_z + r.normal(0.0, sez)),
                sigma_eta_z: Some(sez),
            }
        };

        multi_inputs.push(SiteCsaInput {
            seq_id,
            res_name: name.to_string(),
            fields: vec![
                make_field_obs(500.0, 0.0, &mut rng),
                make_field_obs(800.0, 0.0, &mut rng),
            ],
        });
        single_inputs.push(SiteCsaInput {
            seq_id,
            res_name: name.to_string(),
            fields: vec![make_field_obs(600.0, 0.0, &mut rng)],
        });
    }

    let multi_res = fit_site_specific_csa_batch(&multi_inputs, &cfg).expect("multi-field CSA");
    let single_res = fit_site_specific_csa_batch(&single_inputs, &cfg).expect("single-field CSA");

    for (idx, &(_, _, true_csa)) in targets.iter().enumerate() {
        let rm = &multi_res[idx];
        assert!(
            (rm.csa_n_ppm - true_csa).abs() < 2.0,
            "multi-field CSA recovery failed for {true_csa} ppm: got {:.2} +/- {:.2} ppm",
            rm.csa_n_ppm,
            rm.sigma_csa_n_ppm
        );
        assert_eq!(rm.range_status, CsaRangeStatus::InPhysicalRange);
        assert!(rm.in_physical_range);
        assert!(!rm.rex_detected_on_r2);
        assert_eq!(rm.n_observables_used, 4);

        let rs = &single_res[idx];
        assert!(
            (rs.csa_n_ppm - true_csa).abs() < 2.0,
            "single-field CSA recovery failed for {true_csa} ppm: got {:.2} +/- {:.2} ppm",
            rs.csa_n_ppm,
            rs.sigma_csa_n_ppm
        );
        assert_eq!(rs.range_status, CsaRangeStatus::InPhysicalRange);
        assert!(rs.in_physical_range);
        assert!(rs.sigma_csa_n_ppm > 0.0 && rs.sigma_csa_n_ppm.is_finite());
    }

    // 2. Automatic R_ex contamination exclusion on R2: R_ex = 12 s^-1 must NOT corrupt Delta_sigma_N!
    let p_rex = CcrPhysicsConfig {
        csa_n_ppm: -165.0,
        ..base_phys
    };
    let obs_rex = compute_full_relax_and_ccr_isotropic(8.0e-9, 0.85, 30.0e-12, 12.0, 600.0, &p_rex)
        .expect("R_ex-contaminated CSA forward");
    let rex_inp = SiteCsaInput {
        seq_id: 42,
        res_name: "GLY".to_string(),
        fields: vec![SiteCsaFieldObs {
            field_mhz: 600.0,
            r1: Some(obs_rex.r1),
            sigma_r1: Some(0.01 * obs_rex.r1),
            r2: Some(obs_rex.r2),
            sigma_r2: Some(0.01 * obs_rex.r2),
            hetnoe: Some(obs_rex.hetnoe),
            sigma_hetnoe: Some(0.01),
            eta_xy: Some(obs_rex.eta_xy),
            sigma_eta_xy: Some(0.01 * obs_rex.eta_xy),
            eta_z: Some(obs_rex.eta_z),
            sigma_eta_z: Some(0.01 * obs_rex.eta_z),
        }],
    };
    let rex_res = fit_site_specific_csa(&rex_inp, &cfg).expect("R_ex-protected CSA fit");
    assert!(
        rex_res.rex_detected_on_r2,
        "expected automatic R_ex detection on R2'/eta_xy"
    );
    assert_eq!(rex_res.n_observables_used, 1);
    assert!(
        (rex_res.csa_n_ppm - (-165.0)).abs() < 2.0,
        "R_ex corrupted CSA fit: got {:.2} ppm",
        rex_res.csa_n_ppm
    );

    // 3. Out-of-physical-range diagnostics (< -250 ppm and > -100 ppm)
    for &(anomalous_csa, expected_status) in &[
        (-275.0_f64, CsaRangeStatus::BelowPhysicalRange),
        (-80.0_f64, CsaRangeStatus::AbovePhysicalRange),
    ] {
        let p_anom = CcrPhysicsConfig {
            csa_n_ppm: anomalous_csa,
            ..base_phys
        };
        let obs500 = compute_full_relax_and_ccr_isotropic(8.0e-9, 0.9, 0.0, 0.0, 500.0, &p_anom)
            .expect("anom 500");
        let obs800 = compute_full_relax_and_ccr_isotropic(8.0e-9, 0.9, 0.0, 0.0, 800.0, &p_anom)
            .expect("anom 800");
        let inp = SiteCsaInput {
            seq_id: 99,
            res_name: "TRP".to_string(),
            fields: vec![
                SiteCsaFieldObs {
                    field_mhz: 500.0,
                    r1: Some(obs500.r1),
                    sigma_r1: Some(0.005 * obs500.r1),
                    r2: Some(obs500.r2),
                    sigma_r2: Some(0.005 * obs500.r2),
                    hetnoe: Some(obs500.hetnoe),
                    sigma_hetnoe: Some(0.005),
                    eta_xy: Some(obs500.eta_xy),
                    sigma_eta_xy: Some(0.005 * obs500.eta_xy),
                    eta_z: Some(obs500.eta_z),
                    sigma_eta_z: Some(0.005 * obs500.eta_z),
                },
                SiteCsaFieldObs {
                    field_mhz: 800.0,
                    r1: Some(obs800.r1),
                    sigma_r1: Some(0.005 * obs800.r1),
                    r2: Some(obs800.r2),
                    sigma_r2: Some(0.005 * obs800.r2),
                    hetnoe: Some(obs800.hetnoe),
                    sigma_hetnoe: Some(0.005),
                    eta_xy: Some(obs800.eta_xy),
                    sigma_eta_xy: Some(0.005 * obs800.eta_xy),
                    eta_z: Some(obs800.eta_z),
                    sigma_eta_z: Some(0.005 * obs800.eta_z),
                },
            ],
        };
        let res = fit_site_specific_csa(&inp, &cfg).expect("out-of-range CSA fit");
        assert_eq!(res.range_status, expected_status);
        assert!(!res.in_physical_range);
        assert!(res.csa_n_ppm < CSA_N_PHYSICAL_MIN_PPM || res.csa_n_ppm > CSA_N_PHYSICAL_MAX_PPM);
        assert!(!res.warnings.is_empty());
    }
}

#[test]
fn synthetic_self_consistency_hostile_and_invalid_inputs_fail_closed() {
    let phys = CcrPhysicsConfig::default();
    let tract_cfg = TractConfig::default();

    // 1. Non-positive / NaN / Inf field and dynamics parameters
    assert!(compute_ccr_rates_isotropic(10.0e-9, 1.0, 0.0, 0.0, &phys).is_err());
    assert!(compute_ccr_rates_isotropic(10.0e-9, 1.0, 0.0, -600.0, &phys).is_err());
    assert!(compute_ccr_rates_isotropic(10.0e-9, 1.0, 0.0, f64::NAN, &phys).is_err());
    assert!(compute_ccr_rates_isotropic(f64::INFINITY, 1.0, 0.0, 600.0, &phys).is_err());
    assert!(compute_ccr_rates_isotropic(10.0e-9, 0.0, 0.0, 600.0, &phys).is_err());
    assert!(compute_ccr_rates_isotropic(10.0e-9, 1.2, 0.0, 600.0, &phys).is_err());
    assert!(compute_ccr_rates_isotropic(10.0e-9, 0.85, -1.0e-12, 600.0, &phys).is_err());

    // 2. TRACT inversions: non-positive / NaN eta_xy, magic angle, zero CSA
    assert!(invert_tract_eta_xy_to_tau_m(0.0, 0.1, 600.0, 1.0, 0.0, &phys).is_err());
    assert!(invert_tract_eta_xy_to_tau_m(-5.0, 0.1, 600.0, 1.0, 0.0, &phys).is_err());
    assert!(invert_tract_eta_xy_to_tau_m(f64::NAN, 0.1, 600.0, 1.0, 0.0, &phys).is_err());
    assert!(invert_tract_eta_xy_to_tau_m(5.0, -0.1, 600.0, 1.0, 0.0, &phys).is_err());
    let zero_csa = CcrPhysicsConfig {
        csa_n_ppm: 0.0,
        ..phys
    };
    assert!(invert_tract_eta_xy_to_tau_m(5.0, 0.1, 600.0, 1.0, 0.0, &zero_csa).is_err());

    // 3. eta_xy / eta_z below or equal to extreme-narrowing limit 7/6
    assert!(invert_eta_ratio_to_tau_m(
        ETA_RATIO_EXTREME_NARROWING_LIMIT,
        0.01,
        1.0,
        0.01,
        600.0,
        &phys
    )
    .is_err());
    assert!(invert_eta_ratio_to_tau_m(1.0, 0.01, 1.0, 0.01, 600.0, &phys).is_err());
    assert!(invert_eta_ratio_to_tau_m(5.0, 0.01, 0.0, 0.01, 600.0, &phys).is_err());

    // 4. TRACT decay fitter: R_beta <= R_alpha MUST fail closed
    let times = vec![0.0, 0.01, 0.02, 0.04, 0.08];
    let make_curve =
        |rate: f64| -> Vec<f64> { times.iter().map(|&t| 100.0 * (-rate * t).exp()).collect() };
    let bad_inverted = TractDecayInput {
        label: "inverted".to_string(),
        seq_id: Some(1),
        field_mhz: 600.0,
        alpha_times_sec: times.clone(),
        alpha_intensities: make_curve(25.0),
        alpha_sigmas: vec![1.0; 5],
        beta_times_sec: times.clone(),
        beta_intensities: make_curve(10.0), // R_beta < R_alpha!
        beta_sigmas: vec![1.0; 5],
        eta_z: None,
        sigma_eta_z: None,
    };
    assert!(fit_tract_decay_pair(&bad_inverted, &tract_cfg).is_err());

    let bad_equal = TractDecayInput {
        beta_intensities: make_curve(25.0), // R_beta == R_alpha!
        ..bad_inverted.clone()
    };
    assert!(fit_tract_decay_pair(&bad_equal, &tract_cfg).is_err());

    // 5. TRACT hostile inputs: negative delay, NaN intensity, zero sigma, flat delays, zero signal
    let mut bad_neg_delay = bad_inverted.clone();
    bad_neg_delay.beta_intensities = make_curve(40.0);
    bad_neg_delay.alpha_times_sec[0] = -0.005;
    assert!(fit_tract_decay_pair(&bad_neg_delay, &tract_cfg).is_err());

    let mut bad_nan = bad_inverted.clone();
    bad_nan.beta_intensities = make_curve(40.0);
    bad_nan.alpha_intensities[2] = f64::NAN;
    assert!(fit_tract_decay_pair(&bad_nan, &tract_cfg).is_err());

    let mut bad_sigma = bad_inverted.clone();
    bad_sigma.beta_intensities = make_curve(40.0);
    bad_sigma.beta_sigmas[1] = 0.0;
    assert!(fit_tract_decay_pair(&bad_sigma, &tract_cfg).is_err());

    let mut bad_flat_t = bad_inverted.clone();
    bad_flat_t.beta_intensities = make_curve(40.0);
    bad_flat_t.alpha_times_sec = vec![0.01; 5];
    assert!(fit_tract_decay_pair(&bad_flat_t, &tract_cfg).is_err());

    let mut bad_zero_sig = bad_inverted.clone();
    bad_zero_sig.beta_intensities = make_curve(40.0);
    bad_zero_sig.alpha_intensities = vec![0.0; 5];
    assert!(fit_tract_decay_pair(&bad_zero_sig, &tract_cfg).is_err());

    assert!(fit_tract_batch(&[], &tract_cfg).is_err());

    // 6. Kroenke–Palmer hostile inputs
    let k_cfg = KroenkeConfig::default();
    assert!(separate_rex_kroenke_palmer(&[], &k_cfg).is_err());
    let bad_kroenke = vec![KroenkeResidueInput {
        seq_id: 1,
        res_name: "ALA".to_string(),
        r2: f64::NAN,
        sigma_r2: 0.2,
        eta_xy: 5.0,
        sigma_eta_xy: 0.2,
        r1: None,
        sigma_r1: None,
        hetnoe: None,
        sigma_hetnoe: None,
    }];
    assert!(separate_rex_kroenke_palmer(&bad_kroenke, &k_cfg).is_err());

    // 7. Site-specific CSA hostile inputs: empty fields, missing sigmas, NaN, empty batch
    let csa_cfg = SiteCsaConfig::default();
    assert!(fit_site_specific_csa_batch(&[], &csa_cfg).is_err());
    let empty_csa = SiteCsaInput {
        seq_id: 1,
        res_name: "ALA".to_string(),
        fields: vec![],
    };
    assert!(fit_site_specific_csa(&empty_csa, &csa_cfg).is_err());

    let missing_obs_csa = SiteCsaInput {
        seq_id: 1,
        res_name: "ALA".to_string(),
        fields: vec![SiteCsaFieldObs {
            field_mhz: 600.0,
            r1: Some(1.5),
            sigma_r1: Some(0.05),
            r2: None,
            sigma_r2: None,
            hetnoe: Some(0.8),
            sigma_hetnoe: Some(0.02),
            eta_xy: None,
            sigma_eta_xy: None,
            eta_z: None, // missing both eta_z and eta_xy
            sigma_eta_z: None,
        }],
    };
    assert!(fit_site_specific_csa(&missing_obs_csa, &csa_cfg).is_err());

    let missing_sigma_csa = SiteCsaInput {
        seq_id: 1,
        res_name: "ALA".to_string(),
        fields: vec![SiteCsaFieldObs {
            field_mhz: 600.0,
            r1: Some(1.5),
            sigma_r1: Some(0.05),
            r2: None,
            sigma_r2: None,
            hetnoe: Some(0.8),
            sigma_hetnoe: Some(0.02),
            eta_xy: None,
            sigma_eta_xy: None,
            eta_z: Some(1.2),
            sigma_eta_z: None, // missing sigma!
        }],
    };
    assert!(fit_site_specific_csa(&missing_sigma_csa, &csa_cfg).is_err());

    // 8. Serde JSON round-trip check
    let json = serde_json::to_string(&phys).expect("serialize CcrPhysicsConfig");
    let back: CcrPhysicsConfig = serde_json::from_str(&json).expect("deserialize CcrPhysicsConfig");
    assert_eq!(phys, back);
}
