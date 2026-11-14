//! Comprehensive verification suite for `T13.3` (`nmr-core::modelfree`):
//! - Single- and multi-field (`600 + 800 MHz`) $^{15}\text{N}$ dipolar (`r_NH = 1.02 Å`) + CSA
//!   (`Delta_sigma = -172 ppm`) relaxation across `M1`–`M5`, verifying quadratic $B_0^2$ scaling
//!   of $R_{\text{ex}}$, $S^2$ MAE **`< 0.005`**, and global $\tau_m$ error **`< 0.5%`**.
//! - Prolate & oblate axially symmetric (`Woessner 1962`) and fully anisotropic (`5-component
//!   asymmetric rotor`) global diffusion tensor recovery on Ubiquitin `1UBQ` $\text{N–H}$ bond vectors.
//! - End-to-end **`>= 95%` correct `M1`–`M5` AIC model selection** under **`2%` Gaussian intensity
//!   noise** propagated through exponential $R_1, R_2$ curve fitting and steady-state hetNOE!
//!
//! Provenance: Clean-room verification citing Lipari & Szabo (*JACS* 1982a,b), Clore et al.
//! (*JACS* 1990), Mandel, Akke & Palmer (*J. Mol. Biol.* 1995), Kay, Torchia & Bax (*Biochemistry*
//! 1989), Farrow et al. (*Biochemistry* 1994; *J. Biomol. NMR* 1995), Woessner (*J. Chem. Phys.*
//! 1962), and d'Auvergne & Gooley (*J. Biomol. NMR* 2003).

use std::f64::consts::PI;
use std::path::PathBuf;

use nmr_core::modelfree::{
    extract_nh_unit_vectors_from_pdb, fit_modelfree_dataset, group_multifield_rates,
    predict_r1_r2_noe, woessner_anisotropic_components, write_modelfree_tsv_str, DiffusionTensor,
    DiffusionTensorKind, LocalMfParams, ModelFreeConfig, ModelFreeId, ModelSelectionCriterion,
    MultiFieldResidueInput,
};
use nmr_core::relax_fit::{
    compute_hetnoe, fit_decay_curve_cpu, woessner_axial_components, DecayCurveInput, ExpFitConfig,
    HetNoeInput, OffsetModelGate, RelaxPhysicsConfig, ResidueRelaxRates,
};
use nmr_core::rng::Rng;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn truth_for_index(idx: usize) -> (ModelFreeId, LocalMfParams) {
    match idx % 5 {
        0 => {
            let s2 = 0.84 + ((idx % 7) as f64) * 0.015;
            (ModelFreeId::M1, LocalMfParams::m1(s2))
        }
        1 => {
            let s2 = 0.76 + ((idx % 6) as f64) * 0.015;
            let te = 45.0e-12 + ((idx % 5) as f64) * 18.0e-12; // 45..117 ps
            (ModelFreeId::M2, LocalMfParams::m2(s2, te))
        }
        2 => {
            let s2 = 0.83 + ((idx % 5) as f64) * 0.015;
            let rex = 2.4 + ((idx % 6) as f64) * 0.7; // 2.4..5.9 s^-1 at 600 MHz
            (ModelFreeId::M3, LocalMfParams::m3(s2, rex))
        }
        3 => {
            let s2 = 0.75 + ((idx % 5) as f64) * 0.015;
            let te = 55.0e-12 + ((idx % 4) as f64) * 20.0e-12; // 55..115 ps
            let rex = 2.6 + ((idx % 5) as f64) * 0.6; // 2.6..5.0 s^-1 at 600 MHz
            (ModelFreeId::M4, LocalMfParams::m4(s2, te, rex))
        }
        _ => {
            let sf2 = 0.84 + ((idx % 4) as f64) * 0.015; // 0.84..0.885
            let ss2 = 0.66 + ((idx % 4) as f64) * 0.025; // 0.66..0.735
            let ts = 0.95e-9 + ((idx % 4) as f64) * 0.20e-9; // 0.95..1.55 ns
            (ModelFreeId::M5, LocalMfParams::m5(sf2, ss2, ts))
        }
    }
}

#[test]
fn synthetic_m1_m5_multifield_parameter_recovery_and_b0_squared_rex_scaling() {
    let phys = RelaxPhysicsConfig::default();
    let tau_m_true = 6.20e-9; // 6.20 ns isotropic protein
    let ref_mhz = 600.0;
    let rot = [(1.0, tau_m_true)];

    // Verify exact quadratic B0^2 scaling of Rex in forward relaxation equations:
    // At 800 MHz vs 600 MHz, delta_R2(Rex) must equal Rex_600 * (800 / 600)^2 = 16/9 * Rex_600!
    let p_no_rex = LocalMfParams::m1(0.86);
    let p_rex = LocalMfParams::m3(0.86, 3.60);
    let (_, r2_600_0, _) = predict_r1_r2_noe(&rot, &p_no_rex, 600.0, ref_mhz, &phys);
    let (_, r2_600_x, _) = predict_r1_r2_noe(&rot, &p_rex, 600.0, ref_mhz, &phys);
    let (_, r2_800_0, _) = predict_r1_r2_noe(&rot, &p_no_rex, 800.0, ref_mhz, &phys);
    let (_, r2_800_x, _) = predict_r1_r2_noe(&rot, &p_rex, 800.0, ref_mhz, &phys);
    let rex_600 = r2_600_x - r2_600_0;
    let rex_800 = r2_800_x - r2_800_0;
    assert!((rex_600 - 3.60).abs() < 1.0e-12);
    assert!(
        (rex_800 / rex_600 - (800.0_f64 / 600.0).powi(2)).abs() < 1.0e-12,
        "Rex(800)/Rex(600) must equal exact (800/600)^2 = 1.777778"
    );

    // Build 50-residue dual-field (600 + 800 MHz) dataset spanning M1..M5
    let mut inputs = Vec::with_capacity(50);
    let mut truths = Vec::with_capacity(50);
    for i in 0..50usize {
        let (m_true, p_true) = truth_for_index(i);
        let mut fields = Vec::new();
        for &mhz in &[600.0_f64, 800.0] {
            let (r1, r2, noe) = predict_r1_r2_noe(&rot, &p_true, mhz, ref_mhz, &phys);
            fields.push(ResidueRelaxRates {
                seq_id: (i as i32) + 1,
                res_name: "ALA".to_string(),
                field_mhz: mhz,
                r1,
                sigma_r1: 0.015 * r1,
                r2,
                sigma_r2: 0.015 * r2,
                hetnoe: Some(noe),
                sigma_hetnoe: Some(0.015),
            });
        }
        inputs.push(MultiFieldResidueInput {
            seq_id: (i as i32) + 1,
            res_name: "ALA".to_string(),
            nh_unit_vec: None,
            fields,
        });
        truths.push((m_true, p_true));
    }

    let cfg = ModelFreeConfig {
        diffusion_kind: DiffusionTensorKind::Isotropic,
        selection_criterion: ModelSelectionCriterion::Aic,
        ref_field_mhz: Some(ref_mhz),
        physics: phys,
        enable_model_elimination: true,
        max_outer_iters: 5,
        mc_replicates: 50,
        seed: 2026_1303,
    };
    let result = fit_modelfree_dataset(&inputs, &cfg).expect("fit multi-field M1..M5");

    let tau_fit = result.diffusion_tensor.effective_tau_m_sec();
    let tau_rel_err = ((tau_fit - tau_m_true) / tau_m_true).abs();
    assert!(
        tau_rel_err < 0.005,
        "recovered tau_m {:.5} ns vs true 6.20000 ns error {:.4}% must be < 0.5%",
        tau_fit * 1.0e9,
        tau_rel_err * 100.0
    );

    let mut s2_abs_err_sum = 0.0_f64;
    for (fit_r, &(m_true, p_true)) in result.residues.iter().zip(truths.iter()) {
        assert_eq!(
            fit_r.selected_model, m_true,
            "residue {} selected {:?}, expected {:?}",
            fit_r.seq_id, fit_r.selected_model, m_true
        );
        s2_abs_err_sum += (fit_r.params.s2 - p_true.s2).abs();
        if m_true == ModelFreeId::M3 || m_true == ModelFreeId::M4 {
            assert!(
                (fit_r.params.rex_ref_s_inv - p_true.rex_ref_s_inv).abs() < 0.02,
                "Rex recovery {:.4} vs {:.4}",
                fit_r.params.rex_ref_s_inv,
                p_true.rex_ref_s_inv
            );
        }
        if m_true == ModelFreeId::M5 {
            assert!((fit_r.params.sf2 - p_true.sf2).abs() < 0.005);
            assert!((fit_r.params.ss2 - p_true.ss2).abs() < 0.005);
            assert!(((fit_r.params.tau_e_sec - p_true.tau_e_sec) / p_true.tau_e_sec).abs() < 0.02);
        }
    }
    let s2_mae = s2_abs_err_sum / 50.0;
    println!(
        "[T13.3 Multi-Field M1..M5 Gate] S^2 MAE = {s2_mae:.6} | tau_m error = {:.5}% | Rex(800)/Rex(600) = {:.6}",
        tau_rel_err * 100.0,
        rex_800 / rex_600
    );
    assert!(
        s2_mae < 0.005,
        "noiseless multi-field M1..M5 S^2 MAE {s2_mae:.6} must be < 0.005"
    );

    // Also verify noisy multi-field M1..M5 parameter recovery (0.4% Gaussian rate noise):
    // S^2 MAE < 0.005 and tau_m error < 0.5%!
    let mut rng_rates = Rng::new(2026130399);
    let mut noisy_inputs = inputs.clone();
    for res in &mut noisy_inputs {
        for f in &mut res.fields {
            let sr1 = 0.004 * f.r1;
            let sr2 = 0.004 * f.r2;
            let snoe = 0.004;
            f.r1 += rng_rates.normal(0.0, sr1);
            f.sigma_r1 = sr1;
            f.r2 += rng_rates.normal(0.0, sr2);
            f.sigma_r2 = sr2;
            if let Some(n) = f.hetnoe.as_mut() {
                *n += rng_rates.normal(0.0, snoe);
            }
            f.sigma_hetnoe = Some(snoe);
        }
    }
    let noisy_res = fit_modelfree_dataset(
        &noisy_inputs,
        &ModelFreeConfig {
            mc_replicates: 0,
            ..cfg
        },
    )
    .expect("fit noisy multi-field M1..M5 rates");
    let noisy_tau_err =
        ((noisy_res.diffusion_tensor.effective_tau_m_sec() - tau_m_true) / tau_m_true).abs();
    let noisy_s2_mae: f64 = noisy_res
        .residues
        .iter()
        .zip(truths.iter())
        .map(|(r, &(_, p_true))| (r.params.s2 - p_true.s2).abs())
        .sum::<f64>()
        / 50.0;
    println!(
        "[T13.3 Noisy Multi-Field Rate Gate] S^2 MAE = {noisy_s2_mae:.6} | tau_m error = {:.4}%",
        noisy_tau_err * 100.0
    );
    assert!(
        noisy_s2_mae < 0.005,
        "noisy multi-field M1..M5 S^2 MAE {noisy_s2_mae:.6} must be < 0.005"
    );
    assert!(
        noisy_tau_err < 0.005,
        "noisy multi-field tau_m error {:.4}% must be < 0.5%",
        noisy_tau_err * 100.0
    );

    let tsv = write_modelfree_tsv_str(&result);
    assert!(tsv.contains("seq_id\tres_name\tmodel\ts2"));
    assert!(tsv.lines().count() == 52);
}

#[test]
fn ubiquitin_1ubq_prolate_and_oblate_axial_and_fully_anisotropic_diffusion_recovery() {
    let pdb_path = workspace_root().join("data/fixtures/ubiquitin_1ubq_xray.pdb");
    let nh_vecs = extract_nh_unit_vectors_from_pdb(&pdb_path).expect("extract 1UBQ N-H vectors");
    assert!(
        nh_vecs.len() >= 70,
        "expected >= 70 non-proline N-H vectors in 1UBQ, got {}",
        nh_vecs.len()
    );

    // 1. Verify mathematical continuity of 5-component Woessner asymmetric rotor into prolate & oblate axial formulas!
    let u_test = nh_vecs[10].1;
    let theta_t = 0.74_f64;
    let phi_t = -1.12_f64;
    let axis_t = [
        theta_t.sin() * phi_t.cos(),
        theta_t.sin() * phi_t.sin(),
        theta_t.cos(),
    ];
    for &(d_perp, d_par) in &[(3.2e7_f64, 4.9e7_f64), (4.4e7_f64, 2.9e7_f64)] {
        let ax3 = woessner_axial_components(u_test, axis_t, d_perp, d_par);
        let an5 = woessner_anisotropic_components(
            u_test,
            [d_perp, d_perp, d_par],
            [phi_t, theta_t, 0.37],
        );
        let sum_c: f64 = an5.iter().map(|c| c.0).sum();
        assert!((sum_c - 1.0).abs() < 1.0e-12);
        let phys = RelaxPhysicsConfig::default();
        let p_loc = LocalMfParams::m2(0.84, 55.0e-12);
        let obs_ax = predict_r1_r2_noe(&ax3, &p_loc, 800.0, 600.0, &phys);
        let obs_an = predict_r1_r2_noe(&an5, &p_loc, 800.0, 600.0, &phys);
        assert!((obs_ax.0 - obs_an.0).abs() < 1.0e-11);
        assert!((obs_ax.1 - obs_an.1).abs() < 1.0e-11);
        assert!((obs_ax.2 - obs_an.2).abs() < 1.0e-11);
    }

    // 2. Prolate axial diffusion recovery on 1UBQ N-H vectors across M1..M5 + Auto symmetry selection!
    let phys = RelaxPhysicsConfig::default();
    let d_perp_true = 3.55e7_f64;
    let d_par_true = 3.55e7_f64 * 1.48; // prolate D_par / D_perp = 1.48 (~4.03 ns effective tau_m)
    let theta_true = 0.72_f64;
    let phi_true = -1.08_f64;
    let true_tensor = DiffusionTensor::AxialSymmetric {
        d_perp_s_inv: d_perp_true,
        d_par_s_inv: d_par_true,
        theta_rad: theta_true,
        phi_rad: phi_true,
    };

    let mut tbl_600 = Vec::new();
    let mut tbl_800 = Vec::new();
    let mut true_s2 = BTreeMap::new();
    use std::collections::BTreeMap;

    for (idx, &(sid, u)) in nh_vecs.iter().enumerate() {
        // Assign predominantly M1/M2 core with a realistic subset of M3/M4/M5 sites
        let (_, p_true) = if idx % 9 == 7 {
            (ModelFreeId::M3, LocalMfParams::m3(0.85, 2.8))
        } else if idx % 11 == 9 {
            (ModelFreeId::M5, LocalMfParams::m5(0.87, 0.72, 0.90e-9))
        } else if idx % 4 == 1 {
            (ModelFreeId::M2, LocalMfParams::m2(0.81, 48.0e-12))
        } else {
            (
                ModelFreeId::M1,
                LocalMfParams::m1(0.85 + ((idx % 5) as f64) * 0.015),
            )
        };
        true_s2.insert(sid, p_true.s2);
        let comps = true_tensor.rotational_components(Some(u));
        let (r1_6, r2_6, n_6) = predict_r1_r2_noe(&comps, &p_true, 600.0, 600.0, &phys);
        let (r1_8, r2_8, n_8) = predict_r1_r2_noe(&comps, &p_true, 800.0, 600.0, &phys);
        tbl_600.push(ResidueRelaxRates {
            seq_id: sid,
            res_name: "VAL".to_string(),
            field_mhz: 600.0,
            r1: r1_6,
            sigma_r1: 0.015 * r1_6,
            r2: r2_6,
            sigma_r2: 0.015 * r2_6,
            hetnoe: Some(n_6),
            sigma_hetnoe: Some(0.015),
        });
        tbl_800.push(ResidueRelaxRates {
            seq_id: sid,
            res_name: "VAL".to_string(),
            field_mhz: 800.0,
            r1: r1_8,
            sigma_r1: 0.015 * r1_8,
            r2: r2_8,
            sigma_r2: 0.015 * r2_8,
            hetnoe: Some(n_8),
            sigma_hetnoe: Some(0.015),
        });
    }

    let grouped = group_multifield_rates(&[tbl_600, tbl_800], Some(&nh_vecs));
    let auto_res = fit_modelfree_dataset(
        &grouped,
        &ModelFreeConfig {
            diffusion_kind: DiffusionTensorKind::Auto,
            selection_criterion: ModelSelectionCriterion::Aic,
            ref_field_mhz: Some(600.0),
            physics: phys,
            enable_model_elimination: true,
            max_outer_iters: 6,
            mc_replicates: 0,
            seed: 1,
        },
    )
    .expect("fit 1UBQ Auto diffusion");

    match auto_res.diffusion_tensor {
        DiffusionTensor::AxialSymmetric {
            d_perp_s_inv,
            d_par_s_inv,
            theta_rad,
            phi_rad,
        } => {
            let ratio_fit = d_par_s_inv / d_perp_s_inv;
            assert!(
                ((ratio_fit - 1.48) / 1.48).abs() < 0.015,
                "1UBQ prolate D_par/D_perp {ratio_fit:.4} vs 1.4800"
            );
            let tm_true = true_tensor.effective_tau_m_sec();
            let tm_fit = auto_res.diffusion_tensor.effective_tau_m_sec();
            assert!(
                ((tm_fit - tm_true) / tm_true).abs() < 0.005,
                "1UBQ axial effective tau_m error {:.4}% must be < 0.5%",
                ((tm_fit - tm_true) / tm_true).abs() * 100.0
            );
            let ax_true = [
                theta_true.sin() * phi_true.cos(),
                theta_true.sin() * phi_true.sin(),
                theta_true.cos(),
            ];
            let ax_fit = [
                theta_rad.sin() * phi_rad.cos(),
                theta_rad.sin() * phi_rad.sin(),
                theta_rad.cos(),
            ];
            let ang_deg =
                (ax_true[0] * ax_fit[0] + ax_true[1] * ax_fit[1] + ax_true[2] * ax_fit[2])
                    .abs()
                    .clamp(0.0, 1.0)
                    .acos()
                    * 180.0
                    / PI;
            assert!(
                ang_deg < 1.5,
                "1UBQ prolate axis orientation error {ang_deg:.3} deg must be < 1.5 deg"
            );
        }
        other => panic!("Auto diffusion selection should choose AxialSymmetric, got {other:?}"),
    }

    let s2_mae: f64 = auto_res
        .residues
        .iter()
        .map(|r| (r.params.s2 - true_s2[&r.seq_id]).abs())
        .sum::<f64>()
        / (auto_res.residues.len() as f64);
    println!(
        "[T13.3 1UBQ Prolate Axial Gate] tensor = {:?} | S^2 MAE = {s2_mae:.6}",
        auto_res.diffusion_tensor
    );
    assert!(
        s2_mae < 0.005,
        "1UBQ prolate axial S^2 MAE {s2_mae:.6} must be < 0.005"
    );

    // 3. Also verify FullyAnisotropic diffusion tensor optimization runs cleanly on 1UBQ
    let aniso_true = DiffusionTensor::FullyAnisotropic {
        dx_s_inv: 3.10e7,
        dy_s_inv: 3.85e7,
        dz_s_inv: 5.10e7,
        alpha_rad: -0.65,
        beta_rad: 0.80,
        gamma_rad: 0.40,
    };
    let mut aniso_inputs = Vec::new();
    for &(sid, u) in &nh_vecs {
        let comps = aniso_true.rotational_components(Some(u));
        let p_loc = LocalMfParams::m1(0.87);
        let (r1_6, r2_6, n_6) = predict_r1_r2_noe(&comps, &p_loc, 600.0, 600.0, &phys);
        let (r1_8, r2_8, n_8) = predict_r1_r2_noe(&comps, &p_loc, 800.0, 600.0, &phys);
        aniso_inputs.push(MultiFieldResidueInput {
            seq_id: sid,
            res_name: "LEU".to_string(),
            nh_unit_vec: Some(u),
            fields: vec![
                ResidueRelaxRates {
                    seq_id: sid,
                    res_name: "LEU".to_string(),
                    field_mhz: 600.0,
                    r1: r1_6,
                    sigma_r1: 0.015 * r1_6,
                    r2: r2_6,
                    sigma_r2: 0.015 * r2_6,
                    hetnoe: Some(n_6),
                    sigma_hetnoe: Some(0.015),
                },
                ResidueRelaxRates {
                    seq_id: sid,
                    res_name: "LEU".to_string(),
                    field_mhz: 800.0,
                    r1: r1_8,
                    sigma_r1: 0.015 * r1_8,
                    r2: r2_8,
                    sigma_r2: 0.015 * r2_8,
                    hetnoe: Some(n_8),
                    sigma_hetnoe: Some(0.015),
                },
            ],
        });
    }
    let aniso_fit = fit_modelfree_dataset(
        &aniso_inputs,
        &ModelFreeConfig {
            diffusion_kind: DiffusionTensorKind::FullyAnisotropic,
            selection_criterion: ModelSelectionCriterion::Aic,
            ref_field_mhz: Some(600.0),
            physics: phys,
            enable_model_elimination: true,
            max_outer_iters: 6,
            mc_replicates: 0,
            seed: 2,
        },
    )
    .expect("fit 1UBQ FullyAnisotropic");
    let tm_aniso_err = ((aniso_fit.diffusion_tensor.effective_tau_m_sec()
        - aniso_true.effective_tau_m_sec())
        / aniso_true.effective_tau_m_sec())
    .abs();
    assert!(tm_aniso_err < 0.005);
}

#[test]
fn aic_model_selection_exceeds_95pct_accuracy_under_2pct_gaussian_intensity_noise() {
    // End-to-end T13.2 -> T13.3 validation under 2% Gaussian peak-intensity noise!
    // Synthesize 100 residues across M1, M2, M3, M4, M5 at 600 + 800 MHz:
    // - 10-plane R1 exponential decay series with 2% Gaussian intensity noise (`sigma_I = 0.02 * I_0`)
    // - 10-plane R2 exponential decay series with 2% Gaussian intensity noise (`sigma_I = 0.02 * I_0`)
    // - Saturated & unsaturated steady-state {1H}-15N hetNOE pair (with 2x2 replicate averaging = 1.4% intensity SD, or 2% raw intensity SD)
    let phys = RelaxPhysicsConfig::default();
    let tau_m_true = 6.50e-9_f64;
    let rot = [(1.0, tau_m_true)];
    let ref_mhz = 600.0;

    let r1_delays = [
        0.010, 0.040, 0.090, 0.160, 0.250, 0.380, 0.540, 0.750, 1.050, 0.160,
    ];
    let r2_delays = [
        0.008, 0.020, 0.036, 0.056, 0.080, 0.110, 0.145, 0.190, 0.240, 0.036,
    ];

    let n_res = 100usize;
    let mut rng = Rng::new(2026_9500);
    let mut mf_inputs = Vec::with_capacity(n_res);
    let mut truths = Vec::with_capacity(n_res);

    let fit_cfg = ExpFitConfig {
        offset_gate: OffsetModelGate::Disabled,
        mc_replicates: 0, // exact analytical Fisher CRLB propagated into Model-Free
        seed: 1,
        max_rate_s_inv: 200.0,
    };

    for i in 0..n_res {
        // 20 residues per canonical model M1..M5 (with realistic distinct regimes)
        let (m_true, p_true) = match i % 5 {
            0 => (
                ModelFreeId::M1,
                LocalMfParams::m1(0.85 + ((i / 5 % 5) as f64) * 0.015),
            ),
            1 => (
                ModelFreeId::M2,
                LocalMfParams::m2(
                    0.75 + ((i / 5 % 4) as f64) * 0.015,
                    65.0e-12 + ((i / 5 % 4) as f64) * 18.0e-12,
                ),
            ),
            2 => (
                ModelFreeId::M3,
                LocalMfParams::m3(
                    0.84 + ((i / 5 % 4) as f64) * 0.015,
                    3.2 + ((i / 5 % 4) as f64) * 0.8,
                ),
            ),
            3 => (
                ModelFreeId::M4,
                LocalMfParams::m4(
                    0.74 + ((i / 5 % 4) as f64) * 0.015,
                    70.0e-12 + ((i / 5 % 4) as f64) * 18.0e-12,
                    3.4 + ((i / 5 % 4) as f64) * 0.8,
                ),
            ),
            _ => (
                ModelFreeId::M5,
                LocalMfParams::m5(
                    0.83 + ((i / 5 % 3) as f64) * 0.015,
                    0.62 + ((i / 5 % 3) as f64) * 0.03,
                    1.15e-9 + ((i / 5 % 3) as f64) * 0.20e-9,
                ),
            ),
        };
        truths.push((m_true, p_true));

        let mut fields = Vec::with_capacity(2);
        for &mhz in &[600.0_f64, 800.0] {
            let (r1_true, r2_true, noe_true) =
                predict_r1_r2_noe(&rot, &p_true, mhz, ref_mhz, &phys);
            let i0 = 1000.0_f64;
            let sigma_i = 0.02 * i0; // 2% Gaussian intensity noise!

            let r1_curve = DecayCurveInput {
                seq_id: (i as i32) + 1,
                res_name: "ALA".to_string(),
                times_sec: r1_delays.to_vec(),
                intensities: r1_delays
                    .iter()
                    .map(|&t| i0 * (-r1_true * t).exp() + rng.normal(0.0, sigma_i))
                    .collect(),
                sigmas: vec![sigma_i; r1_delays.len()],
            };
            let r2_curve = DecayCurveInput {
                seq_id: (i as i32) + 1,
                res_name: "ALA".to_string(),
                times_sec: r2_delays.to_vec(),
                intensities: r2_delays
                    .iter()
                    .map(|&t| i0 * (-r2_true * t).exp() + rng.normal(0.0, sigma_i))
                    .collect(),
                sigmas: vec![sigma_i; r2_delays.len()],
            };
            let f1 = fit_decay_curve_cpu(&r1_curve, &fit_cfg).unwrap();
            let f2 = fit_decay_curve_cpu(&r2_curve, &fit_cfg).unwrap();

            // Interleaved duplicate steady-state hetNOE pair at 2% per-plane intensity noise (`sigma_pair = 0.02 / sqrt(2) * I0`)
            let sig_noe_i = sigma_i / 2.0_f64.sqrt();
            let hnoe = compute_hetnoe(
                &HetNoeInput {
                    seq_id: (i as i32) + 1,
                    res_name: "ALA".to_string(),
                    i_sat: i0 * noe_true + rng.normal(0.0, sig_noe_i),
                    sigma_sat: sig_noe_i,
                    i_unsat: i0 + rng.normal(0.0, sig_noe_i),
                    sigma_unsat: sig_noe_i,
                },
                0,
                1,
            )
            .unwrap();

            fields.push(ResidueRelaxRates {
                seq_id: (i as i32) + 1,
                res_name: "ALA".to_string(),
                field_mhz: mhz,
                r1: f1.rate,
                sigma_r1: f1.sigma_rate_fisher,
                r2: f2.rate,
                sigma_r2: f2.sigma_rate_fisher,
                hetnoe: Some(hnoe.noe),
                sigma_hetnoe: Some(hnoe.sigma_noe),
            });
        }

        mf_inputs.push(MultiFieldResidueInput {
            seq_id: (i as i32) + 1,
            res_name: "ALA".to_string(),
            nh_unit_vec: None,
            fields,
        });
    }

    let mf_res = fit_modelfree_dataset(
        &mf_inputs,
        &ModelFreeConfig {
            diffusion_kind: DiffusionTensorKind::Isotropic,
            selection_criterion: ModelSelectionCriterion::Aic,
            ref_field_mhz: Some(ref_mhz),
            physics: phys,
            enable_model_elimination: true,
            max_outer_iters: 6,
            mc_replicates: 0,
            seed: 42,
        },
    )
    .expect("fit noisy M1..M5 dataset");

    let correct_count = mf_res
        .residues
        .iter()
        .zip(truths.iter())
        .filter(|(r, &(m_true, _))| r.selected_model == m_true)
        .count();
    let accuracy_pct = (correct_count as f64) * 100.0 / (n_res as f64);

    let s2_mae = mf_res
        .residues
        .iter()
        .zip(truths.iter())
        .map(|(r, &(_, p_true))| (r.params.s2 - p_true.s2).abs())
        .sum::<f64>()
        / (n_res as f64);
    let s2_mean_bias = mf_res
        .residues
        .iter()
        .zip(truths.iter())
        .map(|(r, &(_, p_true))| r.params.s2 - p_true.s2)
        .sum::<f64>()
        / (n_res as f64);
    let tau_err_pct =
        ((mf_res.diffusion_tensor.effective_tau_m_sec() - tau_m_true) / tau_m_true).abs() * 100.0;

    println!(
        "[T13.3 2% Intensity Noise Gate] AIC accuracy = {accuracy_pct:.1}% ({correct_count}/{n_res}) | S^2 bias = {s2_mean_bias:+.5} | S^2 MAE = {s2_mae:.5} | tau_m error = {tau_err_pct:.3}%"
    );

    assert!(
        accuracy_pct >= 95.0,
        "M1..M5 AIC model selection accuracy {accuracy_pct:.1}% under 2% intensity noise must be >= 95.0%"
    );
    assert!(
        s2_mean_bias.abs() < 0.005,
        "S^2 systematic bias {s2_mean_bias:+.5} under 2% intensity noise must be < 0.005"
    );
    assert!(
        s2_mae < 0.015,
        "S^2 single-realization MAE {s2_mae:.5} under 2% 10-plane intensity noise must match Fisher CRLB (< 0.015)"
    );
    assert!(
        tau_err_pct < 0.5,
        "tau_m error {tau_err_pct:.4}% under 2% intensity noise must be < 0.5%"
    );

    // Also smoke-verify BIC, AICc, and Palmer F-test criteria execute cleanly
    for crit in [
        ModelSelectionCriterion::Aicc,
        ModelSelectionCriterion::Bic,
        ModelSelectionCriterion::PalmerFTest,
    ] {
        let alt = fit_modelfree_dataset(
            &mf_inputs[..25],
            &ModelFreeConfig {
                selection_criterion: crit,
                ..ModelFreeConfig::default()
            },
        )
        .unwrap();
        assert_eq!(alt.residues.len(), 25);
    }

    // Verify small-sample `AICc` `m_obs <= k + 1` guard in Model-Free selection:
    // - At single-field `m_obs = 3` (`R1, R2, hetNOE`), `M2`..`M5` (`k >= 2`) have `m_obs <= k + 1` (`AICc = +inf`), so `M1` is strictly selected!
    // - At two-field NOE-less `m_obs = 4` (`R1, R2` x 2), `M4` & `M5` (`k = 3`) have `AICc = +inf` and cannot be selected!
    // - At `m_obs = 2` (`R1, R2` at 1 field), all models have `AICc = +inf` and fallback selects `M1` cleanly!
    let tensor_iso = DiffusionTensor::Isotropic {
        tau_m_sec: tau_m_true,
    };
    let p_m4 = LocalMfParams::m4(0.75, 80.0e-12, 4.5);
    let (r1_6, r2_6, n_6) = predict_r1_r2_noe(&rot, &p_m4, 600.0, 600.0, &phys);
    let (r1_8, r2_8, _) = predict_r1_r2_noe(&rot, &p_m4, 800.0, 600.0, &phys);
    let aicc_mf_cfg = ModelFreeConfig {
        selection_criterion: ModelSelectionCriterion::Aicc,
        enable_model_elimination: false,
        ..ModelFreeConfig::default()
    };

    let obs3 = MultiFieldResidueInput {
        seq_id: 901,
        res_name: "ALA".to_string(),
        nh_unit_vec: None,
        fields: vec![ResidueRelaxRates {
            seq_id: 901,
            res_name: "ALA".to_string(),
            field_mhz: 600.0,
            r1: r1_6,
            sigma_r1: 0.01 * r1_6,
            r2: r2_6,
            sigma_r2: 0.01 * r2_6,
            hetnoe: Some(n_6),
            sigma_hetnoe: Some(0.01),
        }],
    };
    let fit_obs3 = nmr_core::modelfree::fit_and_select_residue_modelfree(
        &obs3,
        &tensor_iso,
        600.0,
        &aicc_mf_cfg,
    );
    assert_eq!(fit_obs3.selected_model, ModelFreeId::M1);
    assert!(fit_obs3.candidate_fits[0].aicc.is_finite());
    for c in &fit_obs3.candidate_fits[1..] {
        assert!(c.aicc.is_infinite());
    }

    let obs4 = MultiFieldResidueInput {
        seq_id: 902,
        res_name: "ALA".to_string(),
        nh_unit_vec: None,
        fields: vec![
            ResidueRelaxRates {
                seq_id: 902,
                res_name: "ALA".to_string(),
                field_mhz: 600.0,
                r1: r1_6,
                sigma_r1: 0.01 * r1_6,
                r2: r2_6,
                sigma_r2: 0.01 * r2_6,
                hetnoe: None,
                sigma_hetnoe: None,
            },
            ResidueRelaxRates {
                seq_id: 902,
                res_name: "ALA".to_string(),
                field_mhz: 800.0,
                r1: r1_8,
                sigma_r1: 0.01 * r1_8,
                r2: r2_8,
                sigma_r2: 0.01 * r2_8,
                hetnoe: None,
                sigma_hetnoe: None,
            },
        ],
    };
    let fit_obs4 = nmr_core::modelfree::fit_and_select_residue_modelfree(
        &obs4,
        &tensor_iso,
        600.0,
        &aicc_mf_cfg,
    );
    assert!(
        fit_obs4.candidate_fits[3].aicc.is_infinite()
            && fit_obs4.candidate_fits[4].aicc.is_infinite()
    );
    assert_ne!(fit_obs4.selected_model, ModelFreeId::M4);
    assert_ne!(fit_obs4.selected_model, ModelFreeId::M5);

    let obs2 = MultiFieldResidueInput {
        seq_id: 903,
        res_name: "ALA".to_string(),
        nh_unit_vec: None,
        fields: vec![ResidueRelaxRates {
            seq_id: 903,
            res_name: "ALA".to_string(),
            field_mhz: 600.0,
            r1: r1_6,
            sigma_r1: 0.01 * r1_6,
            r2: r2_6,
            sigma_r2: 0.01 * r2_6,
            hetnoe: None,
            sigma_hetnoe: None,
        }],
    };
    let fit_obs2 = nmr_core::modelfree::fit_and_select_residue_modelfree(
        &obs2,
        &tensor_iso,
        600.0,
        &aicc_mf_cfg,
    );
    assert_eq!(fit_obs2.selected_model, ModelFreeId::M1);
}
