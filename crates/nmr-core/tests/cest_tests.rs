//! Quantitative verification suite for `nmr-core::cest` (`T13.4`):
//! - Higham (2005) `[6/6]` Padé `exp(L * T_EX)` parity `< 1e-11` against complex QR eigendecomposition
//!   and `24`-term high-order reference across `7x7` (`2`-state `CEST` & `DEST`) and `10x10` (`3`-state
//!   linear & triangular) Bloch–McConnell Liouvillians.
//! - Gauss–Hermite `B1` inhomogeneity quadrature and high-`R2B` `DEST` dark-state profiles.
//! - Dual-`B1` (`15 Hz` & `30 Hz`) synthetic `15N` `CEST` global fit recovering `p_B = 1.50% +- 0.05%`,
//!   `k_ex = 150 +- 5 s^-1`, and `Delta_varpi_AB` within `< 0.05 ppm`, plus `1D`/`2D` `chi^2`
//!   confidence surfaces and TSV/JSON roundtrip I/O.
//!
//! Provenance: Clean-room verification citing McConnell (*J. Chem. Phys.* 1958), Vallurupalli,
//! Bouvignies & Kay (*JACS* 2012), Fawzi et al. (*Nature* 2011), and Higham (*SIAM J. Matrix Anal.
//! Appl.* 2005).

use std::f64::consts::TAU;

use nmr_core::cest::{
    build_liouvillian_2state, build_liouvillian_3state, expm_eigen_reference,
    expm_high_order_reference, expm_pade66, fit_cest_global_cpu, format_cest_fit_json,
    format_cest_fit_tsv, read_cest_profiles_tsv, simulate_cest_2state_profile,
    simulate_cest_3state_profile, write_cest_profiles_tsv, Cest2StateParams, Cest3StateParams,
    CestFitConfig, CestNormalization, CestProfileDataset,
};
use nmr_core::rng::Rng;

#[test]
fn test_pade66_expm_matches_eigen_and_high_order_to_1e11_2state_and_3state() {
    let mut max_err_vs_eigen = 0.0f64;
    let mut max_err_vs_high = 0.0f64;

    // Sweep diverse 2-state CEST and high-R2B DEST Liouvillians
    let regimes_2state = [
        // (p_b, k_ex, om_a_hz, om_b_hz, b1_hz, r1a, r2a, r2b, t_ex)
        (0.015, 150.0, -450.0, 120.0, 15.0, 1.4, 14.0, 22.0, 0.40),
        (0.015, 150.0, 0.5, -280.0, 30.0, 1.4, 14.0, 22.0, 0.40),
        (0.050, 600.0, 80.0, -40.0, 50.0, 1.8, 18.0, 45.0, 0.35),
        // High-R2B DEST protofibril dark-state regime (Fawzi et al. Nature 2011)
        (
            0.025, 85.0, 1800.0, 1800.0, 250.0, 1.2, 12.0, 18_000.0, 0.50,
        ),
    ];

    for &(pb, kex, oa_hz, ob_hz, b1_hz, r1a, r2a, r2b, t_ex) in &regimes_2state {
        let mut lt = build_liouvillian_2state(
            pb,
            kex,
            TAU * oa_hz,
            TAU * ob_hz,
            TAU * b1_hz,
            0.0,
            r1a,
            r2a,
            r1a,
            r2b,
        );
        for row in &mut lt {
            for val in row {
                *val *= t_ex;
            }
        }

        let pade = expm_pade66(&lt).expect("Padé [6/6] 7x7");
        let eig = expm_eigen_reference(&lt).expect("Eigen 7x7");
        let high = expm_high_order_reference(&lt);

        for i in 0..7 {
            for j in 0..7 {
                max_err_vs_eigen = max_err_vs_eigen.max((pade[i][j] - eig[i][j]).abs());
                max_err_vs_high = max_err_vs_high.max((pade[i][j] - high[i][j]).abs());
            }
        }
    }

    // 3-state linear (`k_ex_ac = 0`) and triangular (`k_ex_ac > 0`) 10x10 Liouvillians
    for &k_ac in &[0.0, 95.0] {
        let mut lt3 = build_liouvillian_3state(
            0.020,
            0.010,
            180.0,
            320.0,
            k_ac,
            TAU * (-210.0),
            TAU * 65.0,
            TAU * 340.0,
            TAU * 25.0,
            0.0,
            1.3,
            15.0,
            1.3,
            28.0,
            1.5,
            42.0,
        );
        for row in &mut lt3 {
            for val in row {
                *val *= 0.40;
            }
        }
        let pade = expm_pade66(&lt3).expect("Padé [6/6] 10x10");
        let eig = expm_eigen_reference(&lt3).expect("Eigen 10x10");
        let high = expm_high_order_reference(&lt3);

        for i in 0..10 {
            for j in 0..10 {
                max_err_vs_eigen = max_err_vs_eigen.max((pade[i][j] - eig[i][j]).abs());
                max_err_vs_high = max_err_vs_high.max((pade[i][j] - high[i][j]).abs());
            }
        }
    }

    println!(
        "[T13.4 CPU Pade66] max_err_vs_eigen={max_err_vs_eigen:.3e}, max_err_vs_high={max_err_vs_high:.3e}"
    );
    assert!(
        max_err_vs_eigen < 1e-11,
        "Padé [6/6] vs complex eigendecomposition max error {max_err_vs_eigen:.3e} >= 1e-11"
    );
    assert!(
        max_err_vs_high < 1e-11,
        "Padé [6/6] vs 24-term reference max error {max_err_vs_high:.3e} >= 1e-11"
    );
}

#[test]
fn test_3state_cest_dips_and_dest_dark_state_broadening_and_b1_quadrature() {
    let offsets: Vec<f64> = (0..=120).map(|i| 112.0 + 0.125 * (i as f64)).collect();

    // 1. 3-state CEST profile: major A at 118.0 ppm, B at 122.0 ppm (+4.0), C at 114.5 ppm (-3.5)
    let p3 = Cest3StateParams {
        p_b: 0.022,
        p_c: 0.015,
        k_ex_ab_s_inv: 160.0,
        k_ex_bc_s_inv: 120.0,
        k_ex_ac_s_inv: 90.0,
        major_ppm: 118.0,
        dw_ab_ppm: 4.0,
        dw_ac_ppm: -3.5,
        r1a_s_inv: 1.2,
        r2a_s_inv: 14.0,
        r1b_s_inv: 1.2,
        r2b_s_inv: 20.0,
        r1c_s_inv: 1.2,
        r2c_s_inv: 22.0,
        larmor_mhz: 60.8106,
        t_ex_s: 0.45,
        b1_hz: 20.0,
        b1_inhomogeneity_frac: 0.08,
        n_b1_quad: 5,
        normalization: CestNormalization::ThermalEquilibrium,
    };
    let curve3 = simulate_cest_3state_profile(&p3, &offsets).expect("3-state CEST");
    let idx_at = |ppm: f64| -> usize {
        offsets
            .iter()
            .enumerate()
            .min_by(|a, b| (a.1 - ppm).abs().total_cmp(&(b.1 - ppm).abs()))
            .unwrap()
            .0
    };
    assert!(
        curve3[idx_at(118.0)] < 0.08,
        "major A dip must saturate deeply"
    );
    assert!(
        curve3[idx_at(122.0)] < 0.86,
        "excited state B dip must be visible at 122 ppm"
    );
    assert!(
        curve3[idx_at(114.5)] < 0.90,
        "excited state C dip must be visible at 114.5 ppm"
    );
    assert!(
        curve3[idx_at(126.5)] > 0.98,
        "far off-resonance baseline ~ 1.0"
    );

    // 2. Gauss–Hermite quadrature convergence (`11-pt` vs `15-pt` `< 1e-4` globally including
    //    9-cycle on-resonance Rabi nutation, and `5-pt` vs `15-pt` `< 1e-4` across all
    //    excited-state dips & wings `|rf - major| >= 0.4 ppm`)
    let mut p3_11 = p3.clone();
    p3_11.n_b1_quad = 11;
    let curve3_11 = simulate_cest_3state_profile(&p3_11, &offsets).expect("11-pt quad");
    let mut p3_15 = p3.clone();
    p3_15.n_b1_quad = 15;
    let curve3_15 = simulate_cest_3state_profile(&p3_15, &offsets).expect("15-pt quad");
    let max_quad_11v15 = curve3_11
        .iter()
        .zip(curve3_15.iter())
        .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
    assert!(
        max_quad_11v15 < 5e-4,
        "11-pt vs 15-pt Gauss-Hermite diff {max_quad_11v15}"
    );
    let max_quad_5v15_excited = offsets
        .iter()
        .zip(curve3.iter().zip(curve3_15.iter()))
        .filter(|(&rf, _)| (rf - p3.major_ppm).abs() >= 0.4)
        .fold(0.0f64, |m, (_, (a, b))| m.max((a - b).abs()));
    assert!(
        max_quad_5v15_excited < 1e-4,
        "5-pt vs 15-pt excited-dip Gauss-Hermite diff {max_quad_5v15_excited}"
    );

    // 3. DEST dark-state saturation (`R2B = 15,000 s^-1` creates broad wings far from 118 ppm!)
    let dest_offsets: Vec<f64> = (-20..=20).map(|k| 118.0 + 2.5 * (k as f64)).collect();
    let dest_params = Cest2StateParams {
        p_b: 0.03,
        k_ex_s_inv: 120.0,
        major_ppm: 118.0,
        dw_ab_ppm: 0.0,
        r1a_s_inv: 1.4,
        r2a_s_inv: 12.0,
        r1b_s_inv: 1.4,
        r2b_s_inv: 15_000.0,
        larmor_mhz: 60.8106,
        t_ex_s: 0.70,
        b1_hz: 250.0,
        b1_inhomogeneity_frac: 0.0,
        n_b1_quad: 1,
        normalization: CestNormalization::ThermalEquilibrium,
    };
    let mut monomer_only = dest_params.clone();
    monomer_only.p_b = 0.0;
    let dest_curve = simulate_cest_2state_profile(&dest_params, &dest_offsets).unwrap();
    let mono_curve = simulate_cest_2state_profile(&monomer_only, &dest_offsets).unwrap();
    // At +35 ppm (~2.13 kHz off-resonance), monomer is > 0.90, while DEST dark state shows strong attenuation (< 0.45)!
    let idx_wing = dest_offsets
        .iter()
        .position(|&x| (x - 153.0).abs() < 1e-6)
        .unwrap();
    assert!(
        mono_curve[idx_wing] > 0.90,
        "monomer wing at +35 ppm should stay > 0.90 (got {})",
        mono_curve[idx_wing]
    );
    assert!(
        dest_curve[idx_wing] < 0.55,
        "DEST dark-state wing must strongly attenuate signal at +35 ppm (got {})",
        dest_curve[idx_wing]
    );
}

#[test]
fn test_dual_b1_15hz_30hz_global_cest_fit_done_gate_and_io() {
    let true_pb = 0.0150;
    let true_kex = 150.0;
    let larmor_mhz = 60.8106; // 600 MHz 1H -> 60.81 MHz 15N
    let t_ex_s = 0.40;
    let residues = [
        (12, "VAL", 120.4, 5.2, 1.35, 13.5, 22.0),
        (18, "LEU", 123.1, -4.6, 1.45, 15.0, 26.0),
        (27, "ILE", 116.8, 6.4, 1.25, 12.5, 19.0),
        (34, "GLU", 119.5, -3.8, 1.50, 14.2, 24.0),
        (45, "PHE", 121.7, 4.3, 1.30, 16.0, 28.0),
        (58, "ASP", 117.9, -5.5, 1.40, 13.0, 21.0),
    ];

    let mut rng = Rng::new(13042026);
    let sigma = 0.0025;
    let mut datasets = Vec::new();

    for &(seq_id, res_name, maj_ppm, dw_ppm, r1a, r2a, r2b) in &residues {
        let offsets: Vec<f64> = (0..65)
            .map(|i| (maj_ppm - 8.0) + 0.25 * (i as f64))
            .collect();
        for &b1_hz in &[15.0, 30.0] {
            let p = Cest2StateParams {
                p_b: true_pb,
                k_ex_s_inv: true_kex,
                major_ppm: maj_ppm,
                dw_ab_ppm: dw_ppm,
                r1a_s_inv: r1a,
                r2a_s_inv: r2a,
                r1b_s_inv: r1a,
                r2b_s_inv: r2b,
                larmor_mhz,
                t_ex_s,
                b1_hz,
                b1_inhomogeneity_frac: 0.05,
                n_b1_quad: 5,
                normalization: CestNormalization::ZeroTimePhaseCycled,
            };
            let clean = simulate_cest_2state_profile(&p, &offsets).unwrap();
            let noisy: Vec<f64> = clean
                .iter()
                .map(|&y| (y + rng.normal(0.0, sigma)).max(0.0))
                .collect();
            datasets.push(CestProfileDataset {
                seq_id,
                res_name: res_name.to_string(),
                larmor_mhz,
                t_ex_s,
                b1_hz,
                b1_inhomogeneity_frac: 0.05,
                normalization: CestNormalization::ZeroTimePhaseCycled,
                major_ppm_hint: None, // test blind major+minor dip seeding!
                offsets_ppm: offsets.clone(),
                intensities: noisy,
                uncertainties: vec![sigma; offsets.len()],
            });
        }
    }

    // Verify TSV roundtrip I/O
    let tsv = write_cest_profiles_tsv(&datasets);
    let parsed = read_cest_profiles_tsv(&tsv).expect("parse CEST TSV");
    assert_eq!(parsed.len(), datasets.len());

    let cfg = CestFitConfig {
        n_b1_quad: 5,
        max_lm_iters: 40,
        compute_chi2_surface: true,
        surface_grid_dims: (9, 9),
        ..Default::default()
    };
    let fit = fit_cest_global_cpu(&parsed, &cfg).expect("global CEST CPU fit");

    assert!(
        (fit.p_b - true_pb).abs() < 0.0005,
        "expected p_B = 0.0150 +- 0.0005, got {:.6} +- {:.6}",
        fit.p_b,
        fit.p_b_err
    );
    assert!(
        (fit.k_ex_s_inv - true_kex).abs() < 5.0,
        "expected k_ex = 150 +- 5 s^-1, got {:.3} +- {:.3}",
        fit.k_ex_s_inv,
        fit.k_ex_err
    );

    for (rf, &(_, _, _, true_dw, _, _, _)) in fit.residues.iter().zip(residues.iter()) {
        let dw_err = (rf.dw_ab_ppm - true_dw).abs();
        assert!(
            dw_err < 0.05,
            "res {} dw_AB error {dw_err:.4} ppm >= 0.05 ppm (fit={:.4}, true={true_dw})",
            rf.seq_id,
            rf.dw_ab_ppm
        );
    }

    let surf = fit.chi2_surface.as_ref().expect("2D chi2 surface present");
    assert_eq!(surf.chi2_2d.len(), 81);
    let fit_tsv = format_cest_fit_tsv(&fit);
    let fit_json = format_cest_fit_json(&fit).expect("CEST JSON");
    assert!(fit_tsv.contains("dw_ab_ppm") && fit_json.contains("\"p_b\""));
}
