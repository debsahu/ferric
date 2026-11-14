//! Comprehensive integration & statistical verification suite for `T13.2`:
//! Batched CPU $R_1, R_2$ exponential curve fitting (2-param & AIC/BIC-gated 3-param), Fisher /
//! Cramér–Rao vs Monte Carlo uncertainty bounds (`+-3%`), steady-state $\{^1\text{H}\}\text{–}^{15}\text{N}$
//! hetNOE error propagation, Kay–Torchia–Bax (1989) outlier-trimmed $R_2/R_1$ $\tau_m$ & Woessner
//! axial $D_\parallel / D_\perp$ estimation, and Farrow et al. (1995) reduced spectral density mapping.
//!
//! Provenance: Clean-room verification citing Farrow et al. (*Biochemistry* 1994; *J. Biomol. NMR*
//! 1995), Kay, Torchia & Bax (*Biochemistry* 1989), Lipari & Szabo (*JACS* 1982a,b), Clore et al.
//! (*JACS* 1990), Mandel, Akke & Palmer (*J. Mol. Biol.* 1995), Woessner (*J. Chem. Phys.* 1962),
//! and d'Auvergne & Gooley (*J. Biomol. NMR* 2003).

use std::f64::consts::PI;
use std::path::PathBuf;

use nmr_core::noesy::{parse_pdb_first_model, v_norm, v_scale, v_sub};
use nmr_core::relax_fit::{
    compute_hetnoe_batch, estimate_initial_axial_diffusion_from_r2_r1, estimate_trimmed_tau_m,
    fisher_crlb_two_param, fit_decay_batch_cpu, fit_decay_curve_cpu, map_reduced_spectral_density,
    rigid_r1_r2_from_rot_components, woessner_axial_components, DecayCurveInput, ExpFitConfig,
    HetNoeInput, OffsetModelGate, RelaxPhysicsConfig, ResidueRelaxRates, TrimmedTauMConfig,
};
use nmr_core::rng::Rng;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[test]
fn two_param_and_aic_bic_three_param_decay_fits_and_crlb_mc_agreement() {
    let delays = vec![
        0.010, 0.020, 0.040, 0.060, 0.080, 0.100, 0.140, 0.180, 0.240, 0.320, 0.040, 0.140,
    ];
    let n_res = 150usize;
    let mut rng = Rng::new(2026_1302);
    let mut curves = Vec::with_capacity(n_res);
    let mut true_rates = Vec::with_capacity(n_res);

    for r in 0..n_res {
        let i0_true = 1000.0 + (r as f64) * 5.0;
        let rate_true = 1.4 + ((r % 30) as f64) * 0.38; // spans R1 (~1.4 s^-1) through R2 (~12.4 s^-1)
        let sigma_pt = i0_true * 0.012; // SNR ~ 83 at t=0
        let intensities: Vec<f64> = delays
            .iter()
            .map(|&t| i0_true * (-rate_true * t).exp() + rng.normal(0.0, sigma_pt))
            .collect();
        curves.push(DecayCurveInput {
            seq_id: (r as i32) + 1,
            res_name: "ALA".to_string(),
            times_sec: delays.clone(),
            intensities,
            sigmas: vec![sigma_pt; delays.len()],
        });
        true_rates.push(rate_true);
    }

    let cfg = ExpFitConfig {
        offset_gate: OffsetModelGate::Disabled,
        mc_replicates: 2000,
        seed: 99_1302,
        max_rate_s_inv: 200.0,
    };
    let fits = fit_decay_batch_cpu(&curves, &cfg).expect("CPU batch decay fit");
    assert_eq!(fits.len(), n_res);

    let mut sum_mc_to_fisher = 0.0_f64;
    let mut within_2sigma = 0usize;
    for (idx, (f, &r_true)) in fits.iter().zip(true_rates.iter()).enumerate() {
        assert!(!f.used_three_param);
        let (_, crlb_r) = fisher_crlb_two_param(&delays, &curves[idx].sigmas, f.i0, f.rate);
        assert!((f.sigma_rate_fisher - crlb_r).abs() < 1.0e-12);
        sum_mc_to_fisher += f.sigma_rate_mc / f.sigma_rate_fisher;
        if (f.rate - r_true).abs() <= 2.0 * f.sigma_rate_fisher {
            within_2sigma += 1;
        }
        assert!(f.rate_ci68[0] < f.rate && f.rate < f.rate_ci68[1]);
        assert!(f.rate_ci95[0] < f.rate_ci68[0] && f.rate_ci68[1] < f.rate_ci95[1]);
    }

    let mean_ratio = sum_mc_to_fisher / (n_res as f64);
    assert!(
        (mean_ratio - 1.0).abs() <= 0.03,
        "ensemble MC/Fisher 1-sigma ratio {mean_ratio:.5} must lie within +-3% of 1.0"
    );
    println!(
        "[T13.2 CPU Fisher/MC Gate] 150-curve 1000-MC/Fisher 1-sigma ratio = {mean_ratio:.5} | 2-sigma coverage = {within_2sigma}/{n_res}"
    );
    assert!(
        within_2sigma >= 135,
        "at least 90% of fitted rates must fall within 2*sigma_Fisher of truth, got {within_2sigma}/{n_res}"
    );

    // Verify AIC & BIC 3-parameter offset gating:
    // 1. Pure zero-asymptote curve -> AIC/BIC must reject 3-param model (`used_three_param == false`)
    let pure_curve = DecayCurveInput {
        seq_id: 501,
        res_name: "GLY".to_string(),
        times_sec: delays.clone(),
        intensities: delays.iter().map(|&t| 500.0 * (-4.2 * t).exp()).collect(),
        sigmas: vec![4.0; delays.len()],
    };
    let fit_pure = fit_decay_curve_cpu(
        &pure_curve,
        &ExpFitConfig {
            offset_gate: OffsetModelGate::Aic,
            mc_replicates: 200,
            seed: 7,
            max_rate_s_inv: 200.0,
        },
    )
    .unwrap();
    assert!(!fit_pure.used_three_param);
    assert!((fit_pure.rate - 4.2).abs() < 1.0e-6);

    // 2. Curve with genuine non-zero baseline offset C = +45.0 -> AIC & BIC must select 3-param and recover R & C!
    let offset_curve = DecayCurveInput {
        seq_id: 502,
        res_name: "SER".to_string(),
        times_sec: vec![0.0, 0.05, 0.12, 0.25, 0.45, 0.75, 1.10, 1.60],
        intensities: [0.0, 0.05, 0.12, 0.25, 0.45, 0.75, 1.10, 1.60]
            .iter()
            .map(|&t: &f64| 400.0 * (-3.1_f64 * t).exp() + 45.0)
            .collect(),
        sigmas: vec![2.5; 8],
    };
    for gate in [
        OffsetModelGate::Aic,
        OffsetModelGate::Aicc,
        OffsetModelGate::Bic,
    ] {
        let fit_off = fit_decay_curve_cpu(
            &offset_curve,
            &ExpFitConfig {
                offset_gate: gate,
                mc_replicates: 300,
                seed: 19,
                max_rate_s_inv: 200.0,
            },
        )
        .unwrap();
        assert!(
            fit_off.used_three_param,
            "gate {gate:?} must select 3-param model"
        );
        assert!((fit_off.rate - 3.1).abs() < 1.0e-4);
        assert!((fit_off.offset_c - 45.0).abs() < 1.0e-2);
    }

    // 3. Small-sample Hurvich–Tsai `AICc` guard (`n <= k + 1`): at `n = 4`, 3-param (`k = 3`, `n - k - 1 = 0`)
    // has `AICc = +inf` and MUST be rejected even when a large baseline offset `C = +80.0` is present!
    let n4_offset_curve = DecayCurveInput {
        seq_id: 503,
        res_name: "THR".to_string(),
        times_sec: vec![0.0, 0.2, 0.6, 1.5],
        intensities: [0.0, 0.2, 0.6, 1.5]
            .iter()
            .map(|&t: &f64| 400.0 * (-3.0_f64 * t).exp() + 80.0)
            .collect(),
        sigmas: vec![2.0; 4],
    };
    let fit_n4_aicc = fit_decay_curve_cpu(
        &n4_offset_curve,
        &ExpFitConfig {
            offset_gate: OffsetModelGate::Aicc,
            mc_replicates: 0,
            seed: 23,
            max_rate_s_inv: 200.0,
        },
    )
    .unwrap();
    assert!(
        !fit_n4_aicc.used_three_param,
        "AICc must reject 3-parameter model at n = 4 (n <= k + 1)"
    );
}

#[test]
fn steady_state_hetnoe_positive_and_negative_error_propagation_and_mc_cis() {
    let inputs = vec![
        // Rigid core beta-sheet residue (+0.82)
        HetNoeInput {
            seq_id: 4,
            res_name: "PHE".to_string(),
            i_sat: 820.0,
            sigma_sat: 11.0,
            i_unsat: 1000.0,
            sigma_unsat: 11.0,
        },
        // Disordered C-terminal tail residue with inverted negative steady-state NOE (-0.46)
        HetNoeInput {
            seq_id: 76,
            res_name: "GLY".to_string(),
            i_sat: -230.0,
            sigma_sat: 7.5,
            i_unsat: 500.0,
            sigma_unsat: 7.5,
        },
    ];

    let results = compute_hetnoe_batch(&inputs, 4000, 202677).expect("hetNOE batch");
    assert_eq!(results.len(), 2);

    assert!((results[0].noe - 0.82).abs() < 1.0e-12);
    let expected_sig0 = (11.0_f64.hypot(0.82 * 11.0)) / 1000.0;
    assert!((results[0].sigma_noe - expected_sig0).abs() < 1.0e-12);
    assert!(
        ((results[0].sigma_noe_mc / results[0].sigma_noe) - 1.0).abs() <= 0.03,
        "MC hetNOE SD {:.5} vs analytical {:.5} must agree within +-3%",
        results[0].sigma_noe_mc,
        results[0].sigma_noe
    );

    assert!((results[1].noe - (-0.46)).abs() < 1.0e-12);
    let expected_sig1 = (7.5_f64.hypot(-0.46 * 7.5)) / 500.0;
    assert!((results[1].sigma_noe - expected_sig1).abs() < 1.0e-12);
    assert!(
        ((results[1].sigma_noe_mc / results[1].sigma_noe) - 1.0).abs() <= 0.03,
        "negative hetNOE MC SD {:.5} vs analytical {:.5} must agree within +-3%",
        results[1].sigma_noe_mc,
        results[1].sigma_noe
    );
    assert!(results[1].noe_ci95[0] < results[1].noe && results[1].noe < results[1].noe_ci95[1]);
}

#[test]
fn kay_torchia_bax_outlier_trimmed_tau_m_and_axial_diffusion_recovery() {
    let phys = RelaxPhysicsConfig::default();
    let field_mhz = 600.0;
    let tau_true = 4.35e-9; // 4.35 ns Ubiquitin-like global tumbling

    let (r1_core, r2_core) = rigid_r1_r2_from_rot_components(&[(1.0, tau_true)], field_mhz, &phys);
    let mut rng = Rng::new(1989_0600);
    let mut iso_rates = Vec::new();

    for seq_id in 1..=72 {
        if seq_id >= 68 {
            // Flexible C-terminus (low hetNOE < 0.65, depressed R2)
            iso_rates.push(ResidueRelaxRates {
                seq_id,
                res_name: "GLY".to_string(),
                field_mhz,
                r1: 1.55,
                sigma_r1: 0.03,
                r2: 2.40,
                sigma_r2: 0.05,
                hetnoe: Some(0.22),
                sigma_hetnoe: Some(0.03),
            });
        } else if seq_id == 23 || seq_id == 25 || seq_id == 53 {
            // Conformational exchange Rex outliers (+5.5 s^-1 elevated R2)
            iso_rates.push(ResidueRelaxRates {
                seq_id,
                res_name: "ASN".to_string(),
                field_mhz,
                r1: r1_core,
                sigma_r1: 0.03,
                r2: r2_core + 5.5,
                sigma_r2: 0.12,
                hetnoe: Some(0.79),
                sigma_hetnoe: Some(0.02),
            });
        } else {
            iso_rates.push(ResidueRelaxRates {
                seq_id,
                res_name: "VAL".to_string(),
                field_mhz,
                r1: r1_core * (1.0 + rng.normal(0.0, 0.008)),
                sigma_r1: 0.02 * r1_core,
                r2: r2_core * (1.0 + rng.normal(0.0, 0.008)),
                sigma_r2: 0.02 * r2_core,
                hetnoe: Some(0.78),
                sigma_hetnoe: Some(0.02),
            });
        }
    }

    let est = estimate_trimmed_tau_m(&iso_rates, &TrimmedTauMConfig::default())
        .expect("trimmed tau_m estimate");
    let rel_err = ((est.tau_m_sec - tau_true) / tau_true).abs();
    assert!(
        rel_err < 0.005,
        "Kay-Torchia-Bax trimmed tau_m {:.4} ns vs true 4.3500 ns error {:.3}% must be < 0.5%",
        est.tau_m_sec * 1.0e9,
        rel_err * 100.0
    );
    for rex_sid in [23, 25, 53, 68, 69, 70, 71, 72] {
        assert!(
            est.excluded_seq_ids.contains(&rex_sid),
            "outlier residue {rex_sid} must be excluded from trimmed core set"
        );
    }

    // Axial diffusion recovery on real 1UBQ N-H bond vectors!
    let pdb_path = workspace_root().join("data/fixtures/ubiquitin_1ubq_xray.pdb");
    let chain = parse_pdb_first_model(&pdb_path).expect("parse 1UBQ");

    let mut nh_vecs = Vec::new();
    for res in &chain.residues {
        if let Some(h) = res.h {
            let d = v_sub(h, res.n);
            let nrm = v_norm(d);
            if nrm > 0.5 {
                nh_vecs.push((res.seq_id, v_scale(d, 1.0 / nrm)));
            }
        }
    }
    assert!(nh_vecs.len() >= 65);

    let d_perp_true = 3.40e7_f64;
    let ratio_true = 1.45_f64;
    let d_par_true = d_perp_true * ratio_true;
    let theta_true = 0.68_f64;
    let phi_true = -0.95_f64;
    let axis_true = [
        theta_true.sin() * phi_true.cos(),
        theta_true.sin() * phi_true.sin(),
        theta_true.cos(),
    ];

    let mut axial_rates = Vec::new();
    for &(sid, u) in &nh_vecs {
        let comps = woessner_axial_components(u, axis_true, d_perp_true, d_par_true);
        let (r1, r2) = rigid_r1_r2_from_rot_components(&comps, 800.0, &phys);
        axial_rates.push(ResidueRelaxRates {
            seq_id: sid,
            res_name: "ALA".to_string(),
            field_mhz: 800.0,
            r1,
            sigma_r1: 0.015 * r1,
            r2,
            sigma_r2: 0.015 * r2,
            hetnoe: Some(0.83),
            sigma_hetnoe: Some(0.02),
        });
    }

    let ax_est = estimate_initial_axial_diffusion_from_r2_r1(
        &axial_rates,
        &nh_vecs,
        &TrimmedTauMConfig {
            trim_sd: 3.5, // keep wide orientation spread of true axial distribution!
            ..TrimmedTauMConfig::default()
        },
    )
    .expect("axial R2/R1 diffusion estimate");

    assert!(
        ((ax_est.anisotropy_ratio - ratio_true) / ratio_true).abs() < 0.02,
        "recovered D_par/D_perp {:.4} vs {:.4}",
        ax_est.anisotropy_ratio,
        ratio_true
    );
    let axis_fit = [
        ax_est.theta_rad.sin() * ax_est.phi_rad.cos(),
        ax_est.theta_rad.sin() * ax_est.phi_rad.sin(),
        ax_est.theta_rad.cos(),
    ];
    let cos_ang =
        (axis_fit[0] * axis_true[0] + axis_fit[1] * axis_true[1] + axis_fit[2] * axis_true[2])
            .abs()
            .clamp(0.0, 1.0);
    let ang_deg = cos_ang.acos() * 180.0 / PI;
    assert!(
        ang_deg < 1.5,
        "axial symmetry axis angle error {ang_deg:.3} deg must be < 1.5 deg"
    );
}

#[test]
fn farrow_1995_reduced_spectral_density_mapping_recovers_exact_j0_jwn_j087wh() {
    let phys = RelaxPhysicsConfig::default();
    let field_mhz = 600.0;
    let (wh, wn) = phys.signed_omegas_rad_s(field_mhz);
    let d = phys.dipolar_d_rad_s();
    let d2 = d * d;
    let c = phys.csa_c_rad_s(field_mhz);
    let c2 = c * c;

    // Synthesize exact Lipari-Szabo observables (S^2 = 0.85, tau_m = 6.0 ns, tau_e = 40 ps)
    let s2 = 0.85_f64;
    let tm = 6.0e-9_f64;
    let te = 40.0e-12_f64;
    let tp = (tm * te) / (tm + te);
    let j_exact = |w: f64| -> f64 {
        0.4 * (s2 * tm / (1.0 + (w * tm).powi(2)) + (1.0 - s2) * tp / (1.0 + (w * tp).powi(2)))
    };

    let r1 = 0.25 * d2 * (j_exact(wh - wn) + 3.0 * j_exact(wn) + 6.0 * j_exact(wh + wn))
        + c2 * j_exact(wn);
    let r2 = 0.125
        * d2
        * (4.0 * j_exact(0.0)
            + j_exact(wh - wn)
            + 3.0 * j_exact(wn)
            + 6.0 * j_exact(wh)
            + 6.0 * j_exact(wh + wn))
        + (c2 / 6.0) * (4.0 * j_exact(0.0) + 3.0 * j_exact(wn));
    let sigma_nh = 0.25 * d2 * (6.0 * j_exact(wh + wn) - j_exact(wh - wn));
    let noe = 1.0 + (phys.gamma_h / phys.gamma_n) * (sigma_nh / r1);

    let obs = ResidueRelaxRates {
        seq_id: 42,
        res_name: "ARG".to_string(),
        field_mhz,
        r1,
        sigma_r1: 0.02 * r1,
        r2,
        sigma_r2: 0.02 * r2,
        hetnoe: Some(noe),
        sigma_hetnoe: Some(0.015),
    };
    let rsd = map_reduced_spectral_density(&obs, &phys).expect("map reduced spectral density");

    let j0_true = j_exact(0.0);
    let jn_true = j_exact(wn);
    let jh_true = j_exact(0.870_04 * wh);

    assert!(
        ((rsd.j_0 - j0_true) / j0_true).abs() < 0.005,
        "J(0) {:.4e} vs exact {:.4e}",
        rsd.j_0,
        j0_true
    );
    assert!(
        ((rsd.j_omega_n - jn_true) / jn_true).abs() < 0.005,
        "J(omega_N) {:.4e} vs exact {:.4e}",
        rsd.j_omega_n,
        jn_true
    );
    assert!(
        ((rsd.j_087_omega_h - jh_true) / jh_true).abs() < 0.015,
        "J(0.87*omega_H) {:.4e} vs exact {:.4e}",
        rsd.j_087_omega_h,
        jh_true
    );

    // Verify analytical error propagation matches 5,000-trial MC perturbation within +-3%
    let mut rng = Rng::new(1995087);
    let n_mc = 5000usize;
    let mut v0 = Vec::with_capacity(n_mc);
    let mut vn = Vec::with_capacity(n_mc);
    let mut vh = Vec::with_capacity(n_mc);
    for _ in 0..n_mc {
        let mut p_obs = obs.clone();
        p_obs.r1 += rng.normal(0.0, obs.sigma_r1);
        p_obs.r2 += rng.normal(0.0, obs.sigma_r2);
        p_obs.hetnoe = Some(noe + rng.normal(0.0, 0.015));
        let m = map_reduced_spectral_density(&p_obs, &phys).unwrap();
        v0.push(m.j_0);
        vn.push(m.j_omega_n);
        vh.push(m.j_087_omega_h);
    }
    let sd_of = |xs: &[f64]| -> f64 {
        let mu = xs.iter().sum::<f64>() / (xs.len() as f64);
        (xs.iter().map(|&x| (x - mu).powi(2)).sum::<f64>() / ((xs.len() - 1) as f64)).sqrt()
    };
    assert!(((sd_of(&v0) / rsd.sigma_j_0) - 1.0).abs() <= 0.03);
    assert!(((sd_of(&vn) / rsd.sigma_j_omega_n) - 1.0).abs() <= 0.03);
    assert!(((sd_of(&vh) / rsd.sigma_j_087_omega_h) - 1.0).abs() <= 0.03);
}
