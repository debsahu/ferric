//! `[synthetic self-consistency]` test suite for `nmr_core::artsy_rdc` (Phase 22 Wave 22.W4, T22.9).
//!
//! All tests are self-contained synthetic benchmarks validating:
//! 1. Exact algebraic forward/inverse `ARTSY` coupling recovery across `D_NH in [-25, +25] Hz`.
//! 2. Sign conventions (`SignNormalized` vs `PhysicalGammaSign`), per-residue isotropic subtraction,
//!    differential `T_2` relaxation mismatch, and scan-ratio scaling.
//! 3. Exact bivariate Gaussian analytical error propagation vs 2,000-trial 2D Monte Carlo resampling,
//!    plus `NearSaturation`, `OutOfDomainRatio`, and `LowSnr` fail-closed diagnostics.
//! 4. End-to-end 2D interleaved `TROSY-HSQC` reference/attenuated extraction on 20 residues at
//!    `SNR = 60` (`RMSE < 0.5 Hz`) and Saupe tensor SVD consistency (`Cornilescu Q < 0.05`).
//! 5. Hostile input rejection (`NaN`, `Inf`, `T <= 0`, `I_ref == 0`, negative noise, malformed 2D grids).

use std::f64::consts::PI;

use nmr_core::artsy_rdc::{
    artsy_abs_coupling_from_rdc, artsy_forward_intensities, artsy_forward_ratio,
    artsy_results_to_rdc_medium_set, estimate_artsy_uncertainty_mc, estimate_spectrum_2d_noise_mad,
    extract_artsy_aligned_and_isotropic_spectra_2d, extract_artsy_coupling,
    extract_artsy_from_spectra_2d, extract_artsy_rdc, ArtsyConfig, ArtsyIntegrationMode,
    ArtsyIntensityPair, ArtsyPeakTarget, ArtsyQualityFlag, ArtsySpectrumExtractConfig,
    DEFAULT_ARTSY_DEPHASING_S, DEFAULT_J_NH_ISO_HZ,
};
use nmr_core::noesy::{v_norm, v_scale, v_sub, PdbChain};
use nmr_core::rdc::{
    calc_rdc_from_unit_vec, fit_saupe_tensor_svd, saupe_matrix_from_pas, RdcBondType, RdcMediumSet,
    RdcSignConvention,
};
use nmr_core::rng::Rng;
use nmr_core::spectrum::{AxisPpm, Spectrum2D};
use nmr_core::structure::build_backbone_from_torsions;

fn build_20_residue_amide_chain() -> PdbChain {
    // 20 non-proline residues with diverse beta-strand, alpha-helix, and loop orientations
    // so the 20 N-H bond vectors span all 5 Saupe tensor components with low condition number.
    let seq: Vec<String> = [
        "MET", "GLN", "ILE", "PHE", "VAL", "LYS", "THR", "LEU", "THR", "GLY", "LYS", "THR", "ILE",
        "THR", "LEU", "GLU", "VAL", "GLU", "ALA", "SER",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let seq_ids: Vec<i32> = (1..=20).collect();
    let phi = vec![
        -60.0, -122.0, -131.0, -118.0, -125.0, -95.0, -82.0, 62.0, -78.0, 82.0, -92.0, -120.0,
        -128.0, -115.0, -124.0, -64.0, -61.0, -65.0, -58.0, -75.0,
    ];
    let psi = vec![
        135.0, 138.0, 152.0, 140.0, 128.0, 125.0, -16.0, 28.0, 145.0, 15.0, 136.0, 132.0, 144.0,
        138.0, 129.0, -42.0, -44.0, -39.0, -47.0, 130.0,
    ];
    let omega = vec![180.0; 20];
    build_backbone_from_torsions(&seq, &seq_ids, &phi, &psi, &omega)
}

/// Synthesize a paired 2D interleaved `TROSY-HSQC` `(ref_spec, att_spec)` containing 2D Gaussian
/// peaks at `peaks` with exact `ARTSY` amplitudes `(i_ref, i_att)` and optional independent
/// Gaussian white noise of per-voxel standard deviation `sigma_vox`.
#[allow(clippy::too_many_arguments)]
fn synthesize_artsy_2d_pair(
    name_prefix: &str,
    peaks: &[ArtsyPeakTarget],
    abs_couplings_hz: &[f64],
    peak_i0: f64,
    sigma_h_vox: f64,
    sigma_n_vox: f64,
    sigma_vox: f64,
    cfg: &ArtsyConfig,
    seed: u64,
) -> (Spectrum2D, Spectrum2D) {
    let nx = 256usize;
    let ny = 256usize;
    let axis_x = AxisPpm::from_bounds("1H", nx, 10.5, 6.5, 600.0);
    let axis_y = AxisPpm::from_bounds("15N", ny, 132.0, 104.0, 60.8);

    let mut ref_data = vec![0.0_f64; nx * ny];
    let mut att_data = vec![0.0_f64; nx * ny];

    for (pk, &abs_jd) in peaks.iter().zip(abs_couplings_hz.iter()) {
        let (a_ref, a_att) = artsy_forward_intensities(abs_jd, peak_i0, cfg)
            .expect("valid forward intensities in synthetic spectrum generator");
        let cx = axis_x.ppm_to_index(pk.h_ppm);
        let cy = axis_y.ppm_to_index(pk.n_ppm);
        let ix_min = (cx - 6.0 * sigma_h_vox).floor().max(0.0) as usize;
        let ix_max = (cx + 6.0 * sigma_h_vox).ceil().min((nx - 1) as f64) as usize;
        let iy_min = (cy - 6.0 * sigma_n_vox).floor().max(0.0) as usize;
        let iy_max = (cy + 6.0 * sigma_n_vox).ceil().min((ny - 1) as f64) as usize;

        for iy in iy_min..=iy_max {
            let dy = (iy as f64 - cy) / sigma_n_vox;
            let ey = (-0.5 * dy * dy).exp();
            for ix in ix_min..=ix_max {
                let dx = (ix as f64 - cx) / sigma_h_vox;
                let shape = ey * (-0.5 * dx * dx).exp();
                ref_data[iy * nx + ix] += a_ref * shape;
                att_data[iy * nx + ix] += a_att * shape;
            }
        }
    }

    if sigma_vox > 0.0 {
        let mut rng = Rng::new(seed);
        for v in &mut ref_data {
            *v += rng.normal(0.0, sigma_vox);
        }
        for v in &mut att_data {
            *v += rng.normal(0.0, sigma_vox);
        }
    }

    let ref_spec = Spectrum2D {
        name: format!("{name_prefix}_ref"),
        axis_x: axis_x.clone(),
        axis_y: axis_y.clone(),
        data: ref_data.into_iter().map(|v| v as f32).collect(),
    };
    let att_spec = Spectrum2D {
        name: format!("{name_prefix}_att"),
        axis_x,
        axis_y,
        data: att_data.into_iter().map(|v| v as f32).collect(),
    };
    (ref_spec, att_spec)
}

#[test]
fn synthetic_self_consistency_algebraic_roundtrip_d_nh_minus25_to_plus25_both_conventions() {
    for convention in [
        RdcSignConvention::SignNormalized,
        RdcSignConvention::PhysicalGammaSign,
    ] {
        let cfg = ArtsyConfig {
            dephasing_delay_s: DEFAULT_ARTSY_DEPHASING_S,
            j_iso_default_hz: DEFAULT_J_NH_ISO_HZ,
            sign_convention: convention,
            ..ArtsyConfig::default()
        };
        let inv_t = 1.0 / cfg.dephasing_delay_s;

        for step in -25..=25 {
            let d_true = step as f64;
            let abs_jd_true = artsy_abs_coupling_from_rdc(d_true, cfg.j_iso_default_hz, convention)
                .expect("valid |J + D|");

            // 1. Forward via physical INEPT intensities (I_ref, I_att)
            let (i_ref, i_att) =
                artsy_forward_intensities(abs_jd_true, 1000.0, &cfg).expect("forward intensities");
            let q_direct = artsy_forward_ratio(abs_jd_true, &cfg).expect("forward ratio");
            assert!(
                ((i_att / i_ref) - q_direct).abs() < 1e-13,
                "forward intensity ratio mismatch at D={d_true}"
            );

            // Verify sign inversion of I_att across |J + D| = 1 / T
            if abs_jd_true < inv_t - 0.1 {
                assert!(
                    i_att > 0.0,
                    "expected positive I_att for |J+D|={abs_jd_true} < 1/T={inv_t}"
                );
            } else if abs_jd_true > inv_t + 0.1 {
                assert!(
                    i_att < 0.0,
                    "expected negative inverted I_att for |J+D|={abs_jd_true} > 1/T={inv_t}"
                );
            }

            let pair = ArtsyIntensityPair {
                i_ref,
                i_att,
                sigma_ref: 0.0,
                sigma_att: 0.0,
            };
            let res = extract_artsy_rdc(10, "ALA", &pair, None, None, &cfg)
                .expect("noiseless extraction");
            assert_eq!(res.quality, ArtsyQualityFlag::Good);
            let abs_jd_rec = res.aligned.abs_j_plus_d_hz.expect("recovered |J + D|");
            let d_rec = res.d_nh_hz.expect("recovered D_NH");

            assert!(
                (abs_jd_rec - abs_jd_true).abs() < 1e-10,
                "|J + D| roundtrip error {:.3e} Hz at D={d_true} ({convention:?})",
                (abs_jd_rec - abs_jd_true).abs()
            );
            assert!(
                (d_rec - d_true).abs() < 1e-10,
                "D_NH roundtrip error {:.3e} Hz at D={d_true} ({convention:?})",
                (d_rec - d_true).abs()
            );

            // Global phase flip (-I_ref, -I_att) must recover identically
            let flipped = ArtsyIntensityPair {
                i_ref: -i_ref,
                i_att: -i_att,
                sigma_ref: 0.0,
                sigma_att: 0.0,
            };
            let res_flip =
                extract_artsy_rdc(10, "ALA", &flipped, None, None, &cfg).expect("flipped pair");
            assert!(
                (res_flip.d_nh_hz.expect("flipped D_NH") - d_true).abs() < 1e-10,
                "global phase flip changed recovered D_NH"
            );

            // Check RdcRestraint D_obs / D_max sign-convention invariance
            let restraint = res
                .to_rdc_restraint("pf1", 0.1, false)
                .expect("valid RdcRestraint");
            restraint.validate().expect("restraint validation");
            let ratio = restraint.d_obs_hz / restraint.d_max_hz;
            let expected_ratio = match convention {
                RdcSignConvention::SignNormalized => d_true / RdcBondType::NH.default_d_max_hz(),
                RdcSignConvention::PhysicalGammaSign => {
                    d_true / (-RdcBondType::NH.default_d_max_hz())
                }
            };
            assert!((ratio - expected_ratio).abs() < 1e-14);
        }
    }
}

#[test]
fn synthetic_self_consistency_isotropic_subtraction_t2_mismatch_and_scan_ratio() {
    // Fitzkee & Bax (2010): 2x scans on attenuated spectrum + differential T2 relaxation mismatch
    let cfg = ArtsyConfig {
        dephasing_delay_s: 0.01075,
        j_iso_default_hz: -93.0,
        sign_convention: RdcSignConvention::SignNormalized,
        delta_r2_s_inv: 18.0,
        scan_ratio_att_over_ref: 2.0,
        ..ArtsyConfig::default()
    };

    // Heterogeneous residue-specific isotropic ^1J_NH = -91.4 Hz (1.6 Hz off canonical -93.0 Hz)
    let j_iso_true_abs = 91.4;
    let d_true = 14.25;
    let abs_jd_true = j_iso_true_abs + d_true;

    let (iso_ref, iso_att) = artsy_forward_intensities(j_iso_true_abs, 1000.0, &cfg).expect("iso");
    let (aln_ref, aln_att) = artsy_forward_intensities(abs_jd_true, 1000.0, &cfg).expect("aligned");

    let iso_pair = ArtsyIntensityPair {
        i_ref: iso_ref,
        i_att: iso_att,
        sigma_ref: 10.0,
        sigma_att: 10.0 * 2.0_f64.sqrt(),
    };
    let aln_pair = ArtsyIntensityPair {
        i_ref: aln_ref,
        i_att: aln_att,
        sigma_ref: 10.0,
        sigma_att: 10.0 * 2.0_f64.sqrt(),
    };

    let res = extract_artsy_rdc(42, "LEU", &aln_pair, Some(&iso_pair), None, &cfg)
        .expect("aligned + isotropic extraction");
    assert_eq!(res.quality, ArtsyQualityFlag::Good);
    assert!((res.j_iso_abs_hz - j_iso_true_abs).abs() < 1e-10);
    assert!((res.d_nh_hz.expect("D_NH") - d_true).abs() < 1e-10);

    // Quadrature error propagation: sigma_D == sqrt(sigma_aligned^2 + sigma_iso^2)
    let sig_aln = res.aligned.sigma_analytical_hz.expect("sig_aln");
    let sig_iso = res
        .isotropic
        .as_ref()
        .and_then(|e| e.sigma_analytical_hz)
        .expect("sig_iso");
    let sig_d = res.sigma_d_nh_hz.expect("sig_d");
    assert!(
        (sig_d - sig_aln.hypot(sig_iso)).abs() < 1e-12,
        "quadrature propagation mismatch: {sig_d} vs {}",
        sig_aln.hypot(sig_iso)
    );

    // Omitting the differential T2 correction must visibly bias the extracted coupling
    let cfg_no_t2 = ArtsyConfig {
        delta_r2_s_inv: 0.0,
        ..cfg.clone()
    };
    let biased = extract_artsy_rdc(42, "LEU", &aln_pair, Some(&iso_pair), None, &cfg_no_t2)
        .expect("uncorrected extraction");
    let bias_hz = (biased.d_nh_hz.expect("biased D_NH") - d_true).abs();
    assert!(
        bias_hz > 0.5,
        "expected visible bias (> 0.5 Hz) when delta_r2=18 s^-1 is uncorrected, got {bias_hz:.3} Hz"
    );
}

#[test]
fn synthetic_self_consistency_analytical_vs_mc_2000_trials_and_saturation_fail_closed() {
    // 1. Linear regime: analytical bivariate Gaussian sigma_D vs 2,000-trial MC sigma_D (<= 8%)
    let cfg = ArtsyConfig {
        dephasing_delay_s: DEFAULT_ARTSY_DEPHASING_S,
        j_iso_default_hz: DEFAULT_J_NH_ISO_HZ,
        sign_convention: RdcSignConvention::SignNormalized,
        mc_trials: 2000,
        mc_seed: 2026_2209,
        ..ArtsyConfig::default()
    };

    for (idx, &(d_hz, snr)) in [
        (-20.0_f64, 60.0_f64),
        (-10.0, 50.0),
        (0.0, 60.0),
        (10.0, 40.0),
        (20.0, 60.0),
        (15.0, 30.0),
    ]
    .iter()
    .enumerate()
    {
        let abs_jd = artsy_abs_coupling_from_rdc(d_hz, cfg.j_iso_default_hz, cfg.sign_convention)
            .expect("valid |J + D|");
        let (i_ref, i_att) = artsy_forward_intensities(abs_jd, 100.0, &cfg).expect("forward");
        let sigma_vox = i_ref.abs() / snr;
        let pair = ArtsyIntensityPair {
            i_ref,
            i_att,
            sigma_ref: sigma_vox,
            sigma_att: sigma_vox * 1.15, // exercise unequal noise variances too
        };
        let mut point_cfg = cfg.clone();
        point_cfg.mc_seed = 0x517C_0000 + (idx as u64) * 17;

        let est = extract_artsy_coupling(&pair, &point_cfg).expect("coupling extraction");
        assert_eq!(est.quality, ArtsyQualityFlag::Good);
        let sig_ana = est.sigma_analytical_hz.expect("analytical sigma");
        let mc = est.mc_uncertainty.as_ref().expect("MC result");
        assert_eq!(mc.n_trials, 2000);
        assert_eq!(mc.out_of_domain_fraction, 0.0);
        let sig_mc = mc.sigma_mc_hz.expect("MC sigma");
        let rel_err = (sig_mc - sig_ana).abs() / sig_ana;
        assert!(
            rel_err <= 0.08,
            "MC vs analytical sigma mismatch at D={d_hz} Hz, SNR={snr}: ana={sig_ana:.4}, mc={sig_mc:.4}, rel={:.2}%",
            rel_err * 100.0
        );
    }

    // Verify Fitzkee & Bax (2010) precision benchmark: at |J + D| = 1 / T, sigma = 1 / (pi * T * SNR) ~ 30 / SNR Hz
    let abs_sweet = 1.0 / cfg.dephasing_delay_s;
    let (i_ref_sw, i_att_sw) = artsy_forward_intensities(abs_sweet, 60.0, &cfg).expect("sweet");
    let pair_sw = ArtsyIntensityPair {
        i_ref: i_ref_sw,
        i_att: i_att_sw,
        sigma_ref: 1.0,
        sigma_att: 1.0,
    };
    let est_sw = extract_artsy_coupling(&pair_sw, &ArtsyConfig::default()).expect("sweet spot");
    let expected_sw = 1.0 / (PI * cfg.dephasing_delay_s * 60.0);
    assert!(
        (est_sw.sigma_analytical_hz.expect("sweet sigma") - expected_sw).abs() < 1e-12,
        "exact sweet-spot sigma mismatch"
    );

    // 2. NearSaturation boundary (|q| in [0.90, 1.0), where 1 / sqrt(1 - q^2) amplifies strongly)
    let q_near = 0.96_f64;
    let pair_near = ArtsyIntensityPair {
        i_ref: 100.0,
        i_att: 2.0 * q_near * 100.0,
        sigma_ref: 2.0,
        sigma_att: 2.0,
    };
    let near_cfg = ArtsyConfig {
        mc_trials: 2000,
        mc_seed: 9901,
        ..ArtsyConfig::default()
    };
    let est_near = extract_artsy_coupling(&pair_near, &near_cfg).expect("near saturation");
    assert_eq!(est_near.quality, ArtsyQualityFlag::NearSaturation);
    let sig_near_ana = est_near.sigma_analytical_hz.expect("near-sat analytical");
    let mc_near = est_near.mc_uncertainty.as_ref().expect("near-sat MC");
    let sig_near_mc = mc_near.sigma_mc_hz.expect("near-sat MC finite sigma");
    assert!(sig_near_ana.is_finite() && sig_near_mc.is_finite());
    assert!(
        sig_near_ana > 3.0 * (1.0 / (PI * near_cfg.dephasing_delay_s * 50.0)),
        "expected strong noise amplification near saturation"
    );
    assert!(
        mc_near.out_of_domain_fraction > 0.0,
        "expected non-zero MC out-of-domain fraction near q=0.96 at SNR=50"
    );

    // 3. OutOfDomainRatio fail-closed handling (|q| >= 1.0, both positive and negative)
    for bad_q in [1.0_f64, -1.0, 1.08, -1.25] {
        let pair_ood = ArtsyIntensityPair {
            i_ref: 100.0,
            i_att: 2.0 * bad_q * 100.0,
            sigma_ref: 2.0,
            sigma_att: 2.0,
        };
        let res_ood = extract_artsy_rdc(7, "GLY", &pair_ood, None, None, &near_cfg)
            .expect("fail-closed out-of-domain should return Ok with OutOfDomainRatio flag");
        assert_eq!(res_ood.quality, ArtsyQualityFlag::OutOfDomainRatio);
        assert!(res_ood.aligned.abs_j_plus_d_hz.is_none());
        assert!(res_ood.aligned.sigma_analytical_hz.is_none());
        assert!(res_ood.d_nh_hz.is_none());
        assert!(res_ood.sigma_d_nh_hz.is_none());
        assert!(res_ood.to_rdc_restraint("pf1", 0.1, true).is_none());
    }

    // 4. LowSnr flagging (reference SNR < min_ref_snr)
    let pair_low = ArtsyIntensityPair {
        i_ref: 10.0,
        i_att: 3.0,
        sigma_ref: 2.0, // SNR = 5.0 < 10.0
        sigma_att: 2.0,
    };
    let res_low =
        extract_artsy_rdc(8, "VAL", &pair_low, None, None, &ArtsyConfig::default()).expect("low");
    assert_eq!(res_low.quality, ArtsyQualityFlag::LowSnr);
    assert!(res_low.d_nh_hz.is_some());
    assert!(res_low.to_rdc_restraint("pf1", 0.1, false).is_none());
    assert!(res_low.to_rdc_restraint("pf1", 0.1, true).is_some());
}

#[test]
fn synthetic_self_consistency_2d_trosy_hsqc_20_residues_and_saupe_svd() {
    let chain = build_20_residue_amide_chain();
    let s_true = saupe_matrix_from_pas(-14.5, 0.38, [35.0, 58.0, 122.0]);

    // Plant exact 1D_NH RDCs from the 3D backbone N-H vectors across all 20 residues
    let mut peaks = Vec::with_capacity(20);
    let mut true_d_norm = Vec::with_capacity(20);
    let mut abs_couplings = Vec::with_capacity(20);

    for (idx, res) in chain.residues.iter().enumerate() {
        let h_pos = res.h.expect("all 20 residues have amide H");
        let dv = v_sub(h_pos, res.n);
        let u = v_scale(dv, 1.0 / v_norm(dv));
        let d_norm = calc_rdc_from_unit_vec(u, RdcBondType::NH.default_d_max_hz(), &s_true);
        true_d_norm.push(d_norm);
        abs_couplings.push(DEFAULT_J_NH_ISO_HZ.abs() + d_norm);

        // Lay out 20 well-separated peaks on a 5x4 grid with deliberate sub-voxel offsets
        let col = idx % 5;
        let row = idx / 5;
        let h_ppm = 9.75 - (col as f64) * 0.65 + ((idx % 3) as f64 - 1.0) * 0.0047;
        let n_ppm = 128.5 - (row as f64) * 6.2 + ((idx % 4) as f64 - 1.5) * 0.031;
        peaks.push(ArtsyPeakTarget {
            seq_id: res.seq_id,
            res_name: res.res_name.clone(),
            h_ppm,
            n_ppm,
            j_iso_override: None,
        });
    }

    // 1. Noiseless 2D sanity check: both BoxSum and ParabolicSubvoxel must have near-zero lineshape bias
    let base_artsy = ArtsyConfig::default();
    let (clean_ref, clean_att) = synthesize_artsy_2d_pair(
        "ubq20_clean",
        &peaks,
        &abs_couplings,
        1000.0,
        1.6,
        1.6,
        0.0,
        &base_artsy,
        1,
    );
    for mode in [
        ArtsyIntegrationMode::BoxSum,
        ArtsyIntegrationMode::ParabolicSubvoxel,
    ] {
        let cfg = ArtsySpectrumExtractConfig {
            artsy: base_artsy.clone(),
            integration: mode,
            ..ArtsySpectrumExtractConfig::default()
        };
        let out = extract_artsy_from_spectra_2d("pf1_clean", &clean_ref, &clean_att, &peaks, &cfg)
            .expect("clean 2D extraction");
        assert_eq!(out.restraints.len(), 20);
        let max_err = out
            .residues
            .iter()
            .zip(true_d_norm.iter())
            .map(|(r, &dt)| (r.d_nh_hz.expect("clean D") - dt).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_err < 1e-4,
            "noiseless 2D {mode:?} max error {max_err:.3e} Hz exceeds 1e-4 Hz"
        );
    }

    // 2. Noisy 2D interleaved TROSY-HSQC at SNR = 60 (peak amplitude 600.0, per-voxel noise 10.0)
    let peak_i0 = 600.0_f64;
    let sigma_vox_true = peak_i0 / 60.0;
    let (noisy_ref, noisy_att) = synthesize_artsy_2d_pair(
        "ubq20_snr60",
        &peaks,
        &abs_couplings,
        peak_i0,
        1.6,
        1.6,
        sigma_vox_true,
        &base_artsy,
        2026_0022,
    );

    // Verify robust two-pass MAD noise recovery
    let mad_ref = estimate_spectrum_2d_noise_mad(&noisy_ref).expect("MAD ref");
    let mad_att = estimate_spectrum_2d_noise_mad(&noisy_att).expect("MAD att");
    assert!(
        ((mad_ref - sigma_vox_true) / sigma_vox_true).abs() < 0.08,
        "MAD ref noise {mad_ref:.3} vs true {sigma_vox_true:.3}"
    );
    assert!(
        ((mad_att - sigma_vox_true) / sigma_vox_true).abs() < 0.08,
        "MAD att noise {mad_att:.3} vs true {sigma_vox_true:.3}"
    );

    for mode in [
        ArtsyIntegrationMode::BoxSum,
        ArtsyIntegrationMode::ParabolicSubvoxel,
    ] {
        let cfg = ArtsySpectrumExtractConfig {
            artsy: base_artsy.clone(),
            integration: mode,
            box_radius_x: 1,
            box_radius_y: 1,
            ref_snap_radius: 1,
            noise_override: None,
            min_restraint_sigma_hz: 0.05,
            include_flagged_in_restraints: false,
        };
        let out = extract_artsy_from_spectra_2d("pf1_snr60", &noisy_ref, &noisy_att, &peaks, &cfg)
            .expect("noisy 2D extraction");
        assert_eq!(out.residues.len(), 20);
        assert_eq!(
            out.restraints.len(),
            20,
            "all 20 residues should pass quality gates at SNR=60"
        );
        assert!(out
            .residues
            .iter()
            .all(|r| r.quality == ArtsyQualityFlag::Good));

        let mse: f64 = out
            .residues
            .iter()
            .zip(true_d_norm.iter())
            .map(|(r, &dt)| {
                let diff = r.d_nh_hz.expect("noisy D") - dt;
                diff * diff
            })
            .sum::<f64>()
            / 20.0;
        let rmse = mse.sqrt();
        let mean_pred_sigma: f64 = out
            .residues
            .iter()
            .map(|r| r.sigma_d_nh_hz.expect("predicted sigma"))
            .sum::<f64>()
            / 20.0;
        // BoxSum integrates across the 3x3 peak footprint (~2.3x SNR gain over single-voxel height,
        // giving RMSE well below 0.5 Hz at single-voxel SNR=60); ParabolicSubvoxel evaluates single-point
        // sub-voxel height whose Fitzkee & Bax theoretical floor at SNR=60 is ~30/60 ~ 0.55 Hz.
        let rmse_limit = match mode {
            ArtsyIntegrationMode::BoxSum => 0.5,
            ArtsyIntegrationMode::ParabolicSubvoxel => 0.75,
        };
        assert!(
            rmse < rmse_limit,
            "2D ARTSY ({mode:?}) RMSE at SNR=60 was {rmse:.4} Hz (expected < {rmse_limit} Hz)"
        );
        // Uncertainty calibration check: empirical RMSE matches analytical prediction within [0.5x, 1.6x]
        let z_ratio = rmse / mean_pred_sigma;
        assert!(
            (0.5..=1.6).contains(&z_ratio),
            "2D ARTSY ({mode:?}) uncertainty calibration ratio rmse/pred_sigma = {z_ratio:.3} (rmse={rmse:.4}, pred={mean_pred_sigma:.4})"
        );

        // Saupe tensor SVD consistency via nmr_core::rdc::fit_saupe_tensor_svd
        let medium = RdcMediumSet {
            medium_id: out.medium_id.clone(),
            restraints: out.restraints.clone(),
        };
        let fit = fit_saupe_tensor_svd(&chain, &medium).expect("Saupe SVD fit");
        assert!(
            fit.q_work < 0.05,
            "2D ARTSY ({mode:?}) Saupe SVD Cornilescu Q = {:.4} (expected < 0.05)",
            fit.q_work
        );
        assert!(
            (fit.da_nh_hz - (-14.5)).abs() < 0.8,
            "recovered Da_NH = {:.3} Hz vs planted -14.5 Hz",
            fit.da_nh_hz
        );
        assert!(
            (fit.rhombicity - 0.38).abs() < 0.06,
            "recovered rhombicity = {:.3} vs planted 0.38",
            fit.rhombicity
        );
    }

    // 3. Verify PhysicalGammaSign produces identical Saupe tensor & Cornilescu Q on the exact same 2D spectra
    let phys_cfg = ArtsySpectrumExtractConfig {
        artsy: ArtsyConfig {
            sign_convention: RdcSignConvention::PhysicalGammaSign,
            ..ArtsyConfig::default()
        },
        ..ArtsySpectrumExtractConfig::default()
    };
    let out_norm = extract_artsy_from_spectra_2d(
        "pf1_norm",
        &noisy_ref,
        &noisy_att,
        &peaks,
        &ArtsySpectrumExtractConfig::default(),
    )
    .expect("norm");
    let out_phys =
        extract_artsy_from_spectra_2d("pf1_phys", &noisy_ref, &noisy_att, &peaks, &phys_cfg)
            .expect("phys");
    let med_norm = artsy_results_to_rdc_medium_set("pf1_norm", &out_norm.residues, 0.05, false)
        .expect("norm medium");
    let med_phys = artsy_results_to_rdc_medium_set("pf1_phys", &out_phys.residues, 0.05, false)
        .expect("phys medium");
    let fit_norm = fit_saupe_tensor_svd(&chain, &med_norm).expect("fit norm");
    let fit_phys = fit_saupe_tensor_svd(&chain, &med_phys).expect("fit phys");
    assert!(
        (fit_norm.q_work - fit_phys.q_work).abs() < 1e-12,
        "SignNormalized vs PhysicalGammaSign Q mismatch: {} vs {}",
        fit_norm.q_work,
        fit_phys.q_work
    );
    assert!((fit_norm.da_nh_hz - fit_phys.da_nh_hz).abs() < 1e-11);
}

#[test]
fn synthetic_self_consistency_2d_aligned_and_isotropic_heterogeneous_j_iso() {
    let chain = build_20_residue_amide_chain();
    let s_true = saupe_matrix_from_pas(12.8, 0.25, [110.0, 42.0, 215.0]);
    let base_artsy = ArtsyConfig::default();

    let mut peaks = Vec::with_capacity(20);
    let mut true_d_norm = Vec::with_capacity(20);
    let mut iso_abs_j = Vec::with_capacity(20);
    let mut aln_abs_jd = Vec::with_capacity(20);

    for (idx, res) in chain.residues.iter().enumerate() {
        let h_pos = res.h.expect("amide H");
        let dv = v_sub(h_pos, res.n);
        let u = v_scale(dv, 1.0 / v_norm(dv));
        let d_norm = calc_rdc_from_unit_vec(u, RdcBondType::NH.default_d_max_hz(), &s_true);
        // Heterogeneous isotropic ^1J_NH ranging from 91.0 Hz to 95.0 Hz (±2 Hz around 93 Hz)
        let j_abs = 91.0 + 4.0 * ((idx as f64) / 19.0);
        true_d_norm.push(d_norm);
        iso_abs_j.push(j_abs);
        aln_abs_jd.push(j_abs + d_norm);

        let col = idx % 5;
        let row = idx / 5;
        peaks.push(ArtsyPeakTarget {
            seq_id: res.seq_id,
            res_name: res.res_name.clone(),
            h_ppm: 9.7 - (col as f64) * 0.65,
            n_ppm: 128.0 - (row as f64) * 6.0,
            j_iso_override: None,
        });
    }

    let peak_i0 = 800.0;
    let sigma_vox = peak_i0 / 80.0; // SNR = 80
    let (aln_ref, aln_att) = synthesize_artsy_2d_pair(
        "aln",
        &peaks,
        &aln_abs_jd,
        peak_i0,
        1.6,
        1.6,
        sigma_vox,
        &base_artsy,
        3001,
    );
    let (iso_ref, iso_att) = synthesize_artsy_2d_pair(
        "iso",
        &peaks,
        &iso_abs_j,
        peak_i0,
        1.6,
        1.6,
        sigma_vox,
        &base_artsy,
        3002,
    );

    let cfg = ArtsySpectrumExtractConfig::default();
    let paired = extract_artsy_aligned_and_isotropic_spectra_2d(
        "bicelle", &aln_ref, &aln_att, &iso_ref, &iso_att, &peaks, &cfg,
    )
    .expect("aligned + isotropic 2D extraction");
    let canonical_only =
        extract_artsy_from_spectra_2d("bicelle_canon", &aln_ref, &aln_att, &peaks, &cfg)
            .expect("canonical-only 2D extraction");

    let rmse_paired = (paired
        .residues
        .iter()
        .zip(true_d_norm.iter())
        .map(|(r, &dt)| (r.d_nh_hz.expect("D") - dt).powi(2))
        .sum::<f64>()
        / 20.0)
        .sqrt();
    let rmse_canon = (canonical_only
        .residues
        .iter()
        .zip(true_d_norm.iter())
        .map(|(r, &dt)| (r.d_nh_hz.expect("D") - dt).powi(2))
        .sum::<f64>()
        / 20.0)
        .sqrt();

    assert!(
        rmse_paired < 0.5,
        "paired aligned+isotropic RMSE {rmse_paired:.4} Hz >= 0.5 Hz"
    );
    assert!(
        rmse_paired < 0.5 * rmse_canon,
        "per-residue isotropic subtraction ({rmse_paired:.3} Hz) must strongly beat uniform -93 Hz baseline ({rmse_canon:.3} Hz)"
    );

    let med = RdcMediumSet {
        medium_id: paired.medium_id.clone(),
        restraints: paired.restraints.clone(),
    };
    let fit = fit_saupe_tensor_svd(&chain, &med).expect("Saupe SVD fit");
    assert!(
        fit.q_work < 0.05,
        "aligned+isotropic Saupe Q = {:.4} >= 0.05",
        fit.q_work
    );
}

#[test]
fn synthetic_self_consistency_hostile_input_rejection() {
    let good_cfg = ArtsyConfig::default();
    let good_pair = ArtsyIntensityPair {
        i_ref: 100.0,
        i_att: 10.0,
        sigma_ref: 1.0,
        sigma_att: 1.0,
    };

    // 1. Hostile ArtsyConfig values
    let bad_configs = [
        ArtsyConfig {
            dephasing_delay_s: 0.0,
            ..good_cfg.clone()
        },
        ArtsyConfig {
            dephasing_delay_s: -0.01075,
            ..good_cfg.clone()
        },
        ArtsyConfig {
            dephasing_delay_s: f64::NAN,
            ..good_cfg.clone()
        },
        ArtsyConfig {
            dephasing_delay_s: f64::INFINITY,
            ..good_cfg.clone()
        },
        ArtsyConfig {
            j_iso_default_hz: 0.0,
            ..good_cfg.clone()
        },
        ArtsyConfig {
            j_iso_default_hz: f64::NAN,
            ..good_cfg.clone()
        },
        ArtsyConfig {
            j_iso_default_sigma_hz: -0.1,
            ..good_cfg.clone()
        },
        ArtsyConfig {
            delta_r2_s_inv: f64::NAN,
            ..good_cfg.clone()
        },
        ArtsyConfig {
            scan_ratio_att_over_ref: 0.0,
            ..good_cfg.clone()
        },
        ArtsyConfig {
            scan_ratio_att_over_ref: -1.0,
            ..good_cfg.clone()
        },
        ArtsyConfig {
            near_saturation_threshold: 0.0,
            ..good_cfg.clone()
        },
        ArtsyConfig {
            near_saturation_threshold: 1.0,
            ..good_cfg.clone()
        },
        ArtsyConfig {
            min_ref_snr: -1.0,
            ..good_cfg.clone()
        },
    ];
    for cfg in &bad_configs {
        assert!(
            extract_artsy_coupling(&good_pair, cfg).is_err(),
            "expected Err for hostile config {cfg:?}"
        );
    }

    // 2. Hostile ArtsyIntensityPair values
    let bad_pairs = [
        ArtsyIntensityPair {
            i_ref: 0.0,
            ..good_pair
        },
        ArtsyIntensityPair {
            i_ref: 1e-35,
            ..good_pair
        },
        ArtsyIntensityPair {
            i_ref: f64::NAN,
            ..good_pair
        },
        ArtsyIntensityPair {
            i_ref: f64::INFINITY,
            ..good_pair
        },
        ArtsyIntensityPair {
            i_att: f64::NAN,
            ..good_pair
        },
        ArtsyIntensityPair {
            i_att: f64::NEG_INFINITY,
            ..good_pair
        },
        ArtsyIntensityPair {
            sigma_ref: -0.5,
            ..good_pair
        },
        ArtsyIntensityPair {
            sigma_att: -0.1,
            ..good_pair
        },
        ArtsyIntensityPair {
            sigma_ref: f64::NAN,
            ..good_pair
        },
        ArtsyIntensityPair {
            sigma_att: f64::INFINITY,
            ..good_pair
        },
    ];
    for pair in &bad_pairs {
        assert!(
            extract_artsy_coupling(pair, &good_cfg).is_err(),
            "expected Err for hostile pair {pair:?}"
        );
    }

    // 3. Hostile MC trial counts & forward inputs
    assert!(estimate_artsy_uncertainty_mc(&good_pair, &good_cfg, 0, 1).is_err());
    assert!(estimate_artsy_uncertainty_mc(&good_pair, &good_cfg, 1, 1).is_err());
    assert!(artsy_forward_ratio(0.0, &good_cfg).is_err());
    assert!(artsy_forward_ratio(-93.0, &good_cfg).is_err());
    assert!(artsy_forward_ratio(f64::NAN, &good_cfg).is_err());
    assert!(artsy_forward_intensities(93.0, 0.0, &good_cfg).is_err());
    assert!(
        artsy_abs_coupling_from_rdc(f64::NAN, -93.0, RdcSignConvention::SignNormalized).is_err()
    );
    assert!(artsy_abs_coupling_from_rdc(-100.0, -93.0, RdcSignConvention::SignNormalized).is_err());

    // 4. Hostile 2D spectrum inputs
    let axis_x = AxisPpm::from_bounds("1H", 32, 10.0, 7.0, 600.0);
    let axis_y = AxisPpm::from_bounds("15N", 32, 130.0, 110.0, 60.8);
    let spec_ok = Spectrum2D {
        name: "ok".to_string(),
        axis_x: axis_x.clone(),
        axis_y: axis_y.clone(),
        data: vec![1.0_f32; 32 * 32],
    };
    let peak_ok = ArtsyPeakTarget {
        seq_id: 1,
        res_name: "ALA".to_string(),
        h_ppm: 8.5,
        n_ppm: 120.0,
        j_iso_override: None,
    };
    let scfg = ArtsySpectrumExtractConfig::default();

    // Empty medium_id & empty peak list
    assert!(extract_artsy_from_spectra_2d(
        "  ",
        &spec_ok,
        &spec_ok,
        std::slice::from_ref(&peak_ok),
        &scfg
    )
    .is_err());
    assert!(extract_artsy_from_spectra_2d("pf1", &spec_ok, &spec_ok, &[], &scfg).is_err());

    // Dimension mismatch
    let spec_small = Spectrum2D {
        name: "small".to_string(),
        axis_x: AxisPpm::from_bounds("1H", 16, 10.0, 7.0, 600.0),
        axis_y: axis_y.clone(),
        data: vec![1.0_f32; 16 * 32],
    };
    assert!(extract_artsy_from_spectra_2d(
        "pf1",
        &spec_ok,
        &spec_small,
        std::slice::from_ref(&peak_ok),
        &scfg
    )
    .is_err());

    // Axis calibration mismatch
    let spec_bad_axis = Spectrum2D {
        name: "bad_axis".to_string(),
        axis_x: AxisPpm::from_bounds("1H", 32, 11.0, 7.0, 600.0),
        axis_y: axis_y.clone(),
        data: vec![1.0_f32; 32 * 32],
    };
    assert!(extract_artsy_from_spectra_2d(
        "pf1",
        &spec_ok,
        &spec_bad_axis,
        std::slice::from_ref(&peak_ok),
        &scfg
    )
    .is_err());

    // Truncated buffer
    let spec_trunc = Spectrum2D {
        name: "trunc".to_string(),
        axis_x: axis_x.clone(),
        axis_y: axis_y.clone(),
        data: vec![1.0_f32; 10],
    };
    assert!(extract_artsy_from_spectra_2d(
        "pf1",
        &spec_trunc,
        &spec_ok,
        std::slice::from_ref(&peak_ok),
        &scfg
    )
    .is_err());

    // NaN voxel in spectrum
    let mut spec_nan = spec_ok.clone();
    spec_nan.data[0] = f32::NAN;
    assert!(extract_artsy_from_spectra_2d(
        "pf1",
        &spec_nan,
        &spec_ok,
        std::slice::from_ref(&peak_ok),
        &scfg
    )
    .is_err());

    // Negative noise override
    let bad_noise_cfg = ArtsySpectrumExtractConfig {
        noise_override: Some((-1.0, 1.0)),
        ..scfg.clone()
    };
    assert!(extract_artsy_from_spectra_2d(
        "pf1",
        &spec_ok,
        &spec_ok,
        std::slice::from_ref(&peak_ok),
        &bad_noise_cfg
    )
    .is_err());

    // Out-of-bounds & NaN peak coordinates
    let peak_oob = ArtsyPeakTarget {
        h_ppm: 15.0,
        ..peak_ok.clone()
    };
    let peak_nan = ArtsyPeakTarget {
        n_ppm: f64::NAN,
        ..peak_ok
    };
    assert!(extract_artsy_from_spectra_2d("pf1", &spec_ok, &spec_ok, &[peak_oob], &scfg).is_err());
    assert!(extract_artsy_from_spectra_2d("pf1", &spec_ok, &spec_ok, &[peak_nan], &scfg).is_err());
}
