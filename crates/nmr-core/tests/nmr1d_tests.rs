//! Comprehensive physics & numerical verification suite for `nmr_core::nmr1d` (`T14.1`).
//!
//! Provenance: Clean-room synthetic spin-physics validation of `docs/specs/17-nmr1d-and-dosy.md` §1
//! (Chen et al. 2002; Eilers 2003; Cobas et al. 2006; Barkhuijsen et al. 1987; Fulmer et al. 2010;
//! Thompson, Cox & Hastings 1987).

use nmr_core::nmr1d::{
    all_deuterated_solvents, apply_apodization_1d, apply_phase_1d, auto_calibrate_solvent_1d,
    auto_phase_1d, auto_phase_and_apply_1d, classify_dept_multiplicity, convert_phase_pivot_1d,
    correct_baseline_1d, deconvolve_region_1d, deconvolve_spectrum_1d, dept_transfer_amplitude,
    eval_pseudo_voigt_1d, fft_1d, hilbert_reconstruct_imaginary_1d, lookup_deuterated_solvent,
    match_trace_impurities, phase_diff_deg, pick_peaks_1d, solvent_residual_shifts,
    subtract_solvent_hsvd_1d, suppress_solvent_notch_1d, synthesize_fid_1d, zero_fill_pow2_1d,
    Apodization1D, AutoPhase1DConfig, BaselineConfig1D, BaselineMethod1D, CarbonDeptMultiplicity,
    DeconvolveConfig1D, Experiment1DKind, HsvdConfig, NotchConfig1D, Nucleus1D, PeakPickConfig1D,
    Pivot1D, Polarity1D, Spectrum1D, SyntheticLine1D,
};
use nmr_core::rng::Rng;
use nmr_core::spectrum::AxisPpm;

fn add_complex_gaussian_noise(fid: &mut nmr_core::nmr1d::TimeDomain1D, sigma: f64, seed: u64) {
    let mut rng = Rng::new(seed);
    for c in &mut fid.data {
        c[0] += rng.normal(0.0, sigma);
        c[1] += rng.normal(0.0, sigma);
    }
}

fn add_real_gaussian_noise(y: &mut [f64], sigma: f64, seed: u64) {
    let mut rng = Rng::new(seed);
    for v in y.iter_mut() {
        *v += rng.normal(0.0, sigma);
    }
}

#[test]
fn multinuclear_apodization_zerofill_fft_hilbert_and_dept_physics() {
    // 1. Verify multi-nuclear Xi ratios and experiment modalities (1H, 13C, DEPT-90/135, 19F, 31P).
    assert_eq!(Nucleus1D::parse("1H"), Nucleus1D::H1);
    assert_eq!(Nucleus1D::parse("13C{1H}"), Nucleus1D::C13);
    assert_eq!(Nucleus1D::parse("19F"), Nucleus1D::F19);
    assert_eq!(Nucleus1D::parse("31P"), Nucleus1D::P31);
    assert!((Nucleus1D::C13.xi_ratio() - 0.251_450_20).abs() < 1e-8);
    assert!((Nucleus1D::F19.xi_ratio() - 0.940_940_11).abs() < 1e-8);
    assert!((Nucleus1D::P31.xi_ratio() - 0.404_807_42).abs() < 1e-8);

    assert_eq!(
        Experiment1DKind::from_pulse_program("dept135", "13C"),
        Experiment1DKind::Dept135
    );
    assert_eq!(
        Experiment1DKind::from_pulse_program("dept90", "13C"),
        Experiment1DKind::Dept90
    );
    assert_eq!(
        Experiment1DKind::Dept135.default_polarity(),
        Polarity1D::Biphasic
    );

    // Verify Doddrell et al. (1982) DEPT-90 and DEPT-135 polarization-transfer amplitudes.
    assert!(dept_transfer_amplitude(0, 90.0).abs() < 1e-12);
    assert!((dept_transfer_amplitude(1, 90.0) - 1.0).abs() < 1e-12);
    assert!(dept_transfer_amplitude(2, 90.0).abs() < 1e-12);
    assert!(dept_transfer_amplitude(3, 90.0).abs() < 1e-12);

    let ch_135 = dept_transfer_amplitude(1, 135.0);
    let ch2_135 = dept_transfer_amplitude(2, 135.0);
    let ch3_135 = dept_transfer_amplitude(3, 135.0);
    assert!(ch_135 > 0.70);
    assert!((ch2_135 - (-1.0)).abs() < 1e-12);
    assert!(ch3_135 > 1.05);

    assert_eq!(
        classify_dept_multiplicity(0.0, 0.0, 0.05),
        CarbonDeptMultiplicity::QuaternaryC
    );
    assert_eq!(
        classify_dept_multiplicity(1.0, ch_135, 0.05),
        CarbonDeptMultiplicity::Ch
    );
    assert_eq!(
        classify_dept_multiplicity(0.0, ch2_135, 0.05),
        CarbonDeptMultiplicity::Ch2
    );
    assert_eq!(
        classify_dept_multiplicity(0.0, ch3_135, 0.05),
        CarbonDeptMultiplicity::Ch3
    );

    // 2. Verify 19F and 31P processing + Parseval/Riemann area preservation and Hilbert causality.
    let lines_f19 = [
        SyntheticLine1D::lorentzian(-62.50, 3.0, 2.5), // CF3 group
        SyntheticLine1D::lorentzian(-114.20, 1.0, 2.5), // Ar-F
    ];
    let mut fid_f19 = synthesize_fid_1d(
        "f19_test",
        Experiment1DKind::Fluorine19,
        4096,
        80_000.0,
        470.59,
        -100.0,
        &lines_f19,
    );
    apply_apodization_1d(&mut fid_f19, Apodization1D::Exponential { lb_hz: 1.0 }, 0.5);
    zero_fill_pow2_1d(&mut fid_f19, 2);
    let spec_f19 = fft_1d(&fid_f19);
    assert_eq!(spec_f19.len(), 8192);

    // Total Riemann integral in Hz must equal total FID amplitude `3.0 + 1.0 = 4.0`.
    let total_area_hz: f64 = spec_f19.real.iter().sum::<f64>() * spec_f19.bin_width_hz();
    assert!(
        (total_area_hz - 4.0).abs() < 1e-3,
        "expected total area 4.0, got {total_area_hz:.6}"
    );

    // Discard imaginary channel and verify exact Hilbert reconstruction on descending-ppm axis!
    let orig_imag = spec_f19.imag.clone().expect("complex FFT imag");
    let mut real_only = spec_f19.clone();
    real_only.imag = None;
    hilbert_reconstruct_imaginary_1d(&mut real_only);
    let rec_imag = real_only.imag.as_ref().expect("reconstructed imag");
    let max_imag = orig_imag.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let max_err = orig_imag
        .iter()
        .zip(rec_imag.iter())
        .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
    assert!(
        max_err / max_imag < 1e-6,
        "Hilbert reconstruction relative error too high: {:.3e}",
        max_err / max_imag
    );

    // 3. Verify 31P processing + Lorentz-to-Gauss doublet valley resolution enhancement
    // and CosineBell apodization + solvent shift referencing.
    let j_hz = 2.4;
    let obs_p31 = 202.46;
    let doublet_p31 = [
        SyntheticLine1D::lorentzian(18.50 + 0.5 * j_hz / obs_p31, 1.0, 2.2),
        SyntheticLine1D::lorentzian(18.50 - 0.5 * j_hz / obs_p31, 1.0, 2.2),
    ];
    let mut fid_exp = synthesize_fid_1d(
        "p31_exp",
        Experiment1DKind::Phosphorus31,
        4096,
        12_000.0,
        obs_p31,
        10.0,
        &doublet_p31,
    );
    let mut fid_lg = fid_exp.clone();
    let mut fid_cos = fid_exp.clone();

    apply_apodization_1d(&mut fid_exp, Apodization1D::Exponential { lb_hz: 0.5 }, 0.5);
    zero_fill_pow2_1d(&mut fid_exp, 2);
    let spec_exp = fft_1d(&fid_exp);

    apply_apodization_1d(
        &mut fid_lg,
        Apodization1D::LorentzToGauss {
            narrow_lb_hz: 1.8,
            gb_hz: 1.2,
            center_frac: 0.0,
        },
        0.5,
    );
    zero_fill_pow2_1d(&mut fid_lg, 2);
    let spec_lg = fft_1d(&fid_lg);

    apply_apodization_1d(&mut fid_cos, Apodization1D::CosineBell { power: 2.0 }, 0.5);
    zero_fill_pow2_1d(&mut fid_cos, 2);
    let spec_cos = fft_1d(&fid_cos);
    assert_eq!(spec_cos.len(), 8192);

    // Measure valley-to-peak ratio at doublet midpoint `18.50 ppm`: Lorentz-to-Gauss must deepen the valley!
    let idx_mid = spec_exp.axis.ppm_to_index(18.50).round() as usize;
    let idx_pk = spec_exp
        .axis
        .ppm_to_index(18.50 + 0.5 * j_hz / obs_p31)
        .round() as usize;
    let ratio_exp = spec_exp.real[idx_mid] / spec_exp.real[idx_pk];
    let ratio_lg = spec_lg.real[idx_mid] / spec_lg.real[idx_pk];
    assert!(
        ratio_lg < ratio_exp - 0.15,
        "Lorentz-to-Gauss should resolve doublet valley better than Exp: ratio_lg={ratio_lg:.3}, ratio_exp={ratio_exp:.3}"
    );

    // 4. Verify automatic CDCl3 solvent referencing (`7.26 ppm`).
    let misref_offset = -0.137; // observed at 7.123 ppm instead of 7.260 ppm
    let lines_h1 = [
        SyntheticLine1D::lorentzian(7.26 + misref_offset, 0.5, 1.0),
        SyntheticLine1D::lorentzian(3.65 + misref_offset, 3.0, 1.2),
    ];
    let mut fid_h1 = synthesize_fid_1d(
        "h1_cdcl3",
        Experiment1DKind::Proton1D,
        8192,
        6000.0,
        500.0,
        4.5,
        &lines_h1,
    );
    apply_apodization_1d(&mut fid_h1, Apodization1D::Exponential { lb_hz: 0.3 }, 0.5);
    zero_fill_pow2_1d(&mut fid_h1, 2);
    let mut spec_h1 = fft_1d(&fid_h1);
    let applied_shift =
        auto_calibrate_solvent_1d(&mut spec_h1, "chloroform-d", 0.25).expect("calibrate CDCl3");
    assert!(
        (applied_shift - (-misref_offset)).abs() < 1e-3,
        "expected shift +{:.4}, got {:.4}",
        -misref_offset,
        applied_shift
    );
    let picks = pick_peaks_1d(&spec_h1, &PeakPickConfig1D::default());
    assert!((picks[0].ppm - 7.26).abs() < 1e-3);
    assert!((picks[1].ppm - 3.65).abs() < 1e-3);
    println!(
        "W1_METRIC_JSON:{}",
        serde_json::json!({
            "task": "T14.1",
            "kind": "t14_1_fft_hilbert_solvent_cal",
            "f19_area_err": (total_area_hz - 4.0).abs(),
            "hilbert_rel_err": max_err / max_imag,
            "lg_valley_ratio": ratio_lg,
            "exp_valley_ratio": ratio_exp,
            "solvent_cal_err_ppm": (applied_shift - (-misref_offset)).abs()
        })
    );
}

#[test]
fn chen_minimum_entropy_autophase_recovers_phc0_and_phc1_across_pivots() {
    let lines = [
        SyntheticLine1D::lorentzian(8.72, 1.0, 1.4),
        SyntheticLine1D::lorentzian(7.45, 2.0, 1.5),
        SyntheticLine1D::lorentzian(7.43, 2.0, 1.5),
        SyntheticLine1D::lorentzian(5.31, 1.0, 1.3),
        SyntheticLine1D::lorentzian(3.84, 3.0, 1.2),
        SyntheticLine1D::lorentzian(2.18, 2.0, 1.4),
        SyntheticLine1D::lorentzian(1.25, 6.0, 1.5),
        SyntheticLine1D::lorentzian(0.89, 3.0, 1.4),
    ];

    let mut fid = synthesize_fid_1d(
        "h1_phase_test",
        Experiment1DKind::Proton1D,
        4096,
        6000.0,
        500.0,
        4.7,
        &lines,
    );
    add_complex_gaussian_noise(&mut fid, 2e-3, 20261114);
    apply_apodization_1d(&mut fid, Apodization1D::Exponential { lb_hz: 0.3 }, 0.5);
    zero_fill_pow2_1d(&mut fid, 2);
    let clean_spec = fft_1d(&fid);

    // Case A: Arbitrary pivot at 4.70 ppm, strong zero- and first-order phase corruption.
    let pivot_ppm = Pivot1D::Ppm(4.70);
    let true_corr_phc0 = 43.5_f64;
    let true_corr_phc1 = -68.0_f64;

    let mut corrupted = clean_spec.clone();
    apply_phase_1d(&mut corrupted, -true_corr_phc0, -true_corr_phc1, pivot_ppm);

    // Verify pivot conversion algebra first.
    let (p0_edge, p1_edge) = convert_phase_pivot_1d(
        &corrupted,
        true_corr_phc0,
        true_corr_phc1,
        pivot_ppm,
        Pivot1D::Fraction(0.0),
    );
    let (p0_back, p1_back) = convert_phase_pivot_1d(
        &corrupted,
        p0_edge,
        p1_edge,
        Pivot1D::Fraction(0.0),
        pivot_ppm,
    );
    assert!(phase_diff_deg(p0_back, true_corr_phc0).abs() < 1e-9);
    assert!((p1_back - true_corr_phc1).abs() < 1e-9);

    let cfg = AutoPhase1DConfig {
        pivot: pivot_ppm,
        ..AutoPhase1DConfig::default()
    };
    let sol = auto_phase_and_apply_1d(&mut corrupted, &cfg);
    let err_p0 = phase_diff_deg(sol.phc0_deg, true_corr_phc0).abs();
    let err_p1 = (sol.phc1_deg - true_corr_phc1).abs();
    assert!(
        err_p0 < 2.0,
        "Case A |dPHC0| = {err_p0:.3} deg exceeds 2.0 deg (got {:.2}, expected {:.2})",
        sol.phc0_deg,
        true_corr_phc0
    );
    assert!(
        err_p1 < 5.0,
        "Case A |dPHC1| = {err_p1:.3} deg exceeds 5.0 deg (got {:.2}, expected {:.2})",
        sol.phc1_deg,
        true_corr_phc1
    );

    // Case B: Large inverted phase (-124 deg, +92 deg) on real-only spectrum (Hilbert-reconstructed)!
    let true_b_p0 = -124.0_f64;
    let true_b_p1 = 92.0_f64;
    let mut corrupted_b = clean_spec.clone();
    apply_phase_1d(
        &mut corrupted_b,
        -true_b_p0,
        -true_b_p1,
        Pivot1D::Fraction(0.5),
    );
    corrupted_b.imag = None; // Force Hilbert transform path!

    let sol_b = auto_phase_1d(
        &corrupted_b,
        &AutoPhase1DConfig {
            pivot: Pivot1D::Fraction(0.5),
            ..AutoPhase1DConfig::default()
        },
    );
    let err_b_p0 = phase_diff_deg(sol_b.phc0_deg, true_b_p0).abs();
    let err_b_p1 = (sol_b.phc1_deg - true_b_p1).abs();
    assert!(
        err_b_p0 < 2.0,
        "Case B |dPHC0| = {err_b_p0:.3} deg exceeds 2.0 deg (got {:.2}, expected {:.2})",
        sol_b.phc0_deg,
        true_b_p0
    );
    assert!(
        err_b_p1 < 5.0,
        "Case B |dPHC1| = {err_b_p1:.3} deg exceeds 5.0 deg (got {:.2}, expected {:.2})",
        sol_b.phc1_deg,
        true_b_p1
    );

    // Case C: DEPT-135 biphasic spectrum (positive CH/CH3 and negative CH2 resonances).
    let dept_lines = [
        SyntheticLine1D {
            ppm: 135.2,
            amplitude: 1.2,
            fwhm_hz: 2.0,
            gauss_fwhm_hz: 0.0,
            phase_deg: 0.0,
        },
        SyntheticLine1D {
            ppm: 72.4,
            amplitude: 1.0,
            fwhm_hz: 2.0,
            gauss_fwhm_hz: 0.0,
            phase_deg: 0.0,
        },
        SyntheticLine1D {
            ppm: 41.8,
            amplitude: 1.5,
            fwhm_hz: 2.0,
            gauss_fwhm_hz: 0.0,
            phase_deg: 180.0, // CH2 inverted
        },
        SyntheticLine1D {
            ppm: 28.6,
            amplitude: 1.4,
            fwhm_hz: 2.0,
            gauss_fwhm_hz: 0.0,
            phase_deg: 180.0, // CH2 inverted
        },
        SyntheticLine1D {
            ppm: 14.2,
            amplitude: 1.8,
            fwhm_hz: 2.0,
            gauss_fwhm_hz: 0.0,
            phase_deg: 0.0, // CH3 positive
        },
    ];
    let mut fid_dept = synthesize_fid_1d(
        "dept135",
        Experiment1DKind::Dept135,
        4096,
        25_000.0,
        125.76,
        90.0,
        &dept_lines,
    );
    apply_apodization_1d(
        &mut fid_dept,
        Apodization1D::Exponential { lb_hz: 1.0 },
        0.5,
    );
    zero_fill_pow2_1d(&mut fid_dept, 2);
    let mut spec_dept = fft_1d(&fid_dept);
    apply_phase_1d(&mut spec_dept, -31.0, 44.0, Pivot1D::Fraction(0.5));

    let sol_dept = auto_phase_and_apply_1d(
        &mut spec_dept,
        &AutoPhase1DConfig {
            pivot: Pivot1D::Fraction(0.5),
            polarity: Some(Polarity1D::Biphasic),
            biphasic_positive_ppm: Some((120.0, 150.0)),
            ..AutoPhase1DConfig::default()
        },
    );
    assert!(
        phase_diff_deg(sol_dept.phc0_deg, 31.0).abs() < 2.0,
        "DEPT-135 |dPHC0| got {:.3}, expected 31.0 (phc1 got {:.3}, expected -44.0)",
        sol_dept.phc0_deg,
        sol_dept.phc1_deg
    );
    assert!(
        (sol_dept.phc1_deg - (-44.0)).abs() < 5.0,
        "DEPT-135 |dPHC1| got {:.3}, expected -44.0",
        sol_dept.phc1_deg
    );
    println!(
        "W1_METRIC_JSON:{}",
        serde_json::json!({
            "task": "T14.1",
            "kind": "t14_1_autophase",
            "case_a_phc0_err_deg": err_p0,
            "case_a_phc1_err_deg": err_p1,
            "hilbert_phc0_err_deg": err_b_p0,
            "hilbert_phc1_err_deg": err_b_p1,
            "dept135_phc0_err_deg": phase_diff_deg(sol_dept.phc0_deg, 31.0).abs(),
            "dept135_phc1_err_deg": (sol_dept.phc1_deg - (-44.0)).abs()
        })
    );
}

#[test]
fn whittaker_and_bernstein_baseline_correction_flatten_curved_roll() {
    let lines = [
        SyntheticLine1D::lorentzian(8.12, 2.0, 1.5),
        SyntheticLine1D::lorentzian(7.26, 0.8, 1.2),
        SyntheticLine1D::lorentzian(4.15, 2.0, 1.6),
        SyntheticLine1D::lorentzian(4.136, 6.0, 1.6),
        SyntheticLine1D::lorentzian(4.122, 6.0, 1.6),
        SyntheticLine1D::lorentzian(4.108, 2.0, 1.6),
        SyntheticLine1D::lorentzian(2.05, 3.0, 1.4),
        SyntheticLine1D::lorentzian(1.26, 3.0, 1.5),
    ];

    let mut fid = synthesize_fid_1d(
        "baseline_test",
        Experiment1DKind::Proton1D,
        4096,
        6000.0,
        500.0,
        4.5,
        &lines,
    );
    apply_apodization_1d(&mut fid, Apodization1D::Exponential { lb_hz: 0.3 }, 0.5);
    zero_fill_pow2_1d(&mut fid, 2);
    let pure_spec = fft_1d(&fid);
    let n = pure_spec.len();
    let peak_max = pure_spec.real.iter().copied().fold(0.0_f64, f64::max);

    // Construct a severe curved baseline roll (~25% of peak height) + thermal noise.
    let mut true_baseline = vec![0.0_f64; n];
    for (i, b) in true_baseline.iter_mut().enumerate() {
        let u = 2.0 * (i as f64) / ((n - 1) as f64) - 1.0;
        *b = peak_max
            * (0.12 * u * u * u - 0.18 * u * u
                + 0.09 * u
                + 0.06 * (1.7 * std::f64::consts::PI * u).sin()
                + 0.08);
    }

    let mut distorted_whittaker = pure_spec.clone();
    for (y, &b) in distorted_whittaker
        .real
        .iter_mut()
        .zip(true_baseline.iter())
    {
        *y += b;
    }
    add_real_gaussian_noise(&mut distorted_whittaker.real, 1e-4 * peak_max, 991122);
    let mut distorted_bernstein = distorted_whittaker.clone();

    // Pre-correction baseline RMS error on true signal-free regions (> 0.25 ppm away from peaks).
    let is_far_from_peaks = |ppm: f64| -> bool { lines.iter().all(|l| (l.ppm - ppm).abs() > 0.25) };
    let raw_rms: f64 = {
        let vals: Vec<f64> = (0..n)
            .filter(|&i| is_far_from_peaks(pure_spec.ppm_at(i)))
            .map(|i| distorted_whittaker.real[i] - pure_spec.real[i])
            .collect();
        (vals.iter().map(|v| v * v).sum::<f64>() / (vals.len() as f64)).sqrt()
    };

    // 1. Whittaker penalized least-squares baseline correction (Eilers 2003 + Cobas 2006 mask).
    let cfg_w = BaselineConfig1D {
        method: BaselineMethod1D::Whittaker { lambda: 5.0e5 },
        wavelet_sigma_thresh: 3.2,
        wavelet_scale_pts: 8,
        peak_dilation_pts: 28,
        refine_iters: 3,
    };
    let res_w = correct_baseline_1d(&mut distorted_whittaker, &cfg_w);
    let post_w_rms: f64 = {
        let vals: Vec<f64> = (0..n)
            .filter(|&i| is_far_from_peaks(pure_spec.ppm_at(i)))
            .map(|i| distorted_whittaker.real[i] - pure_spec.real[i])
            .collect();
        (vals.iter().map(|v| v * v).sum::<f64>() / (vals.len() as f64)).sqrt()
    };
    assert!(
        post_w_rms < raw_rms / 40.0,
        "Whittaker baseline RMS did not flatten enough: raw={raw_rms:.5}, post={post_w_rms:.5}"
    );
    // Verify peaks were masked (`is_baseline == false` at quartet center `4.13 ppm`).
    let idx_q = pure_spec.axis.ppm_to_index(4.13).round() as usize;
    assert!(!res_w.is_baseline[idx_q]);

    // Verify quartet integral over [3.95, 4.30] ppm is preserved within < 1.5%.
    let integrate_window = |s: &Spectrum1D, lo: f64, hi: f64| -> f64 {
        s.real
            .iter()
            .enumerate()
            .filter(|&(j, _)| {
                let p = s.ppm_at(j);
                p >= lo && p <= hi
            })
            .map(|(_, &y)| y * s.bin_width_hz())
            .sum()
    };
    let area_true = integrate_window(&pure_spec, 3.95, 4.30);
    let area_w = integrate_window(&distorted_whittaker, 3.95, 4.30);
    assert!(
        ((area_w - area_true) / area_true).abs() < 0.015,
        "Whittaker quartet integral error {:.3}% >= 1.5%",
        100.0 * ((area_w - area_true) / area_true).abs()
    );

    // 2. Bernstein polynomial baseline correction (`degree = 8`).
    let cfg_b = BaselineConfig1D {
        method: BaselineMethod1D::Bernstein { degree: 8 },
        wavelet_sigma_thresh: 3.2,
        wavelet_scale_pts: 8,
        peak_dilation_pts: 28,
        refine_iters: 3,
    };
    let _ = correct_baseline_1d(&mut distorted_bernstein, &cfg_b);
    let post_b_rms: f64 = {
        let vals: Vec<f64> = (0..n)
            .filter(|&i| is_far_from_peaks(pure_spec.ppm_at(i)))
            .map(|i| distorted_bernstein.real[i] - pure_spec.real[i])
            .collect();
        (vals.iter().map(|v| v * v).sum::<f64>() / (vals.len() as f64)).sqrt()
    };
    assert!(
        post_b_rms < raw_rms / 35.0,
        "Bernstein baseline RMS did not flatten enough: raw={raw_rms:.5}, post={post_b_rms:.5}"
    );
    println!(
        "W1_METRIC_JSON:{}",
        serde_json::json!({
            "task": "T14.1",
            "kind": "t14_1_baseline",
            "whittaker_suppression_ratio": raw_rms / post_w_rms,
            "bernstein_suppression_ratio": raw_rms / post_b_rms,
            "whittaker_quartet_area_rel_err_pct": 100.0 * ((area_w - area_true) / area_true).abs()
        })
    );
}

#[test]
fn hsvd_water_subtraction_and_frequency_notch_preserve_nearby_solute_peaks() {
    // Solute resonances including an anomeric/Halpha proton at 4.91 ppm (only 0.12 ppm = 60 Hz
    // downfield of a 500x-stronger broad HDO solvent peak at 4.79 ppm!).
    let solute_lines = [
        SyntheticLine1D::lorentzian(7.82, 1.0, 1.5),
        SyntheticLine1D::lorentzian(4.91, 1.0, 1.4), // adjacent to water!
        SyntheticLine1D::lorentzian(4.63, 1.0, 1.4), // upfield adjacent to water!
        SyntheticLine1D::lorentzian(1.42, 3.0, 1.5),
    ];
    let mut with_water_lines = solute_lines.to_vec();
    with_water_lines.push(SyntheticLine1D {
        ppm: 4.79,
        amplitude: 500.0,
        fwhm_hz: 8.5,
        gauss_fwhm_hz: 0.0,
        phase_deg: 27.0, // out-of-phase radiation-damped water!
    });

    let mut fid_ref = synthesize_fid_1d(
        "solute_only",
        Experiment1DKind::Proton1D,
        4096,
        6000.0,
        500.0,
        4.79,
        &solute_lines,
    );
    let mut fid_wet = synthesize_fid_1d(
        "solute_plus_water",
        Experiment1DKind::Proton1D,
        4096,
        6000.0,
        500.0,
        4.79,
        &with_water_lines,
    );
    add_complex_gaussian_noise(&mut fid_wet, 1.5e-3, 77714);

    // Measure raw water height before HSVD.
    let mut fid_raw_check = fid_wet.clone();
    apply_apodization_1d(
        &mut fid_raw_check,
        Apodization1D::Exponential { lb_hz: 0.3 },
        0.5,
    );
    zero_fill_pow2_1d(&mut fid_raw_check, 2);
    let spec_raw = fft_1d(&fid_raw_check);
    let idx_water = spec_raw.axis.ppm_to_index(4.79).round() as usize;
    let raw_water_mag = spec_raw.real[idx_water].hypot(spec_raw.imag.as_ref().unwrap()[idx_water]);

    // Run time-domain HSVD water subtraction across `[4.72, 4.85]` ppm (excluding 4.91 and 4.63!).
    let hsvd_cfg = HsvdConfig {
        subspace_rank: 16,
        max_hankel_points: 1024,
        solvent_ppm_windows: vec![(4.72, 4.85)],
    };
    let hsvd_res = subtract_solvent_hsvd_1d(&mut fid_wet, &hsvd_cfg);
    assert!(
        !hsvd_res.subtracted_poles.is_empty(),
        "HSVD should identify and subtract the 4.79 ppm water pole"
    );

    apply_apodization_1d(&mut fid_ref, Apodization1D::Exponential { lb_hz: 0.3 }, 0.5);
    zero_fill_pow2_1d(&mut fid_ref, 2);
    let spec_ref = fft_1d(&fid_ref);

    apply_apodization_1d(&mut fid_wet, Apodization1D::Exponential { lb_hz: 0.3 }, 0.5);
    zero_fill_pow2_1d(&mut fid_wet, 2);
    let mut spec_clean = fft_1d(&fid_wet);

    let post_water_mag =
        spec_clean.real[idx_water].hypot(spec_clean.imag.as_ref().unwrap()[idx_water]);
    assert!(
        post_water_mag < 1e-3 * raw_water_mag,
        "HSVD should suppress >99.9% of water peak: before={raw_water_mag:.3}, after={post_water_mag:.5}"
    );

    // Nearby solute peaks at 4.91 ppm and 4.63 ppm must retain their exact heights & integrals (< 1.5% diff)!
    let mut max_solute_rel_err = 0.0_f64;
    for &sol_ppm in &[4.91, 4.63, 7.82, 1.42] {
        let idx = spec_ref.axis.ppm_to_index(sol_ppm).round() as usize;
        let h_ref = spec_ref.real[idx];
        let h_clean = spec_clean.real[idx];
        let rel_err = ((h_clean - h_ref) / h_ref).abs();
        max_solute_rel_err = max_solute_rel_err.max(rel_err);
        assert!(
            rel_err < 0.015,
            "Solute peak at {sol_ppm:.2} ppm perturbed by {:.3}% after HSVD",
            100.0 * rel_err
        );
    }

    // Also verify frequency-domain polynomial notch suppression on `[4.74, 4.84]` ppm.
    spec_clean.real[idx_water] = 50.0; // inject synthetic frequency spike
    suppress_solvent_notch_1d(&mut spec_clean, &NotchConfig1D::new(4.74, 4.84));
    assert!(spec_clean.real[idx_water].abs() < 0.02);
    println!(
        "W1_METRIC_JSON:{}",
        serde_json::json!({
            "task": "T14.1",
            "kind": "t14_1_hsvd_solvent",
            "water_suppression_pct": 100.0 * (1.0 - post_water_mag / raw_water_mag),
            "max_solute_height_err_pct": 100.0 * max_solute_rel_err,
            "subtracted_poles": hsvd_res.subtracted_poles.len()
        })
    );
}

#[test]
fn fulmer_2010_deuterated_solvent_and_impurity_lookup_table_accuracy() {
    let required = [
        ("CDCl3", Some(7.26), Some(77.16), 1.56),
        ("DMSO-d6", Some(2.50), Some(39.52), 3.33),
        ("D2O", Some(4.79), None, 4.79),
        ("CD3OD", Some(3.31), Some(49.00), 4.87),
        ("CD3CN", Some(1.94), Some(1.32), 2.13),
        ("acetone-d6", Some(2.05), Some(29.84), 2.84),
        ("THF-d8", Some(3.58), Some(67.21), 2.46),
        ("C6D6", Some(7.16), Some(128.06), 0.40),
        ("CD2Cl2", Some(5.32), Some(53.84), 1.52),
        ("toluene-d8", Some(2.08), Some(20.43), 0.43),
    ];

    assert_eq!(all_deuterated_solvents().len(), 10);

    for (name, exp_h1, exp_c13, exp_h2o) in required {
        let entry = lookup_deuterated_solvent(name)
            .unwrap_or_else(|| panic!("missing solvent entry for {name}"));
        assert_eq!(entry.canonical_name, name);
        assert!(
            (entry.h2o_h1_ppm - exp_h2o).abs() < 1e-6,
            "{name} H2O shift mismatch"
        );

        let h1 = solvent_residual_shifts(name, &Nucleus1D::H1).expect("1H slice");
        if let Some(target_h1) = exp_h1 {
            assert!(
                (h1[0].ppm - target_h1).abs() < 1e-6,
                "{name} 1H shift got {}, expected {target_h1}",
                h1[0].ppm
            );
        }

        let c13 = solvent_residual_shifts(name, &Nucleus1D::C13).expect("13C slice");
        match exp_c13 {
            Some(target_c13) => assert!(
                (c13[0].ppm - target_c13).abs() < 1e-6,
                "{name} 13C shift got {}, expected {target_c13}",
                c13[0].ppm
            ),
            None => assert!(c13.is_empty(), "D2O must have empty 13C residual list"),
        }
    }

    // Verify aliases and trace laboratory impurity matching (Fulmer et al. 2010).
    assert_eq!(
        lookup_deuterated_solvent("MeOD").unwrap().canonical_name,
        "CD3OD"
    );
    assert_eq!(
        lookup_deuterated_solvent("(CD3)2SO")
            .unwrap()
            .canonical_name,
        "DMSO-d6"
    );
    assert_eq!(
        lookup_deuterated_solvent("DCM-d2").unwrap().canonical_name,
        "CD2Cl2"
    );

    let etoac_hits = match_trace_impurities("CDCl3", &Nucleus1D::H1, 4.122, 0.02);
    assert!(!etoac_hits.is_empty());
    assert_eq!(etoac_hits[0].entry.compound, "ethyl acetate");

    let dmf_hits = match_trace_impurities("DMSO-d6", &Nucleus1D::H1, 7.95, 0.02);
    assert!(!dmf_hits.is_empty());
    assert_eq!(dmf_hits[0].entry.compound, "DMF");
}

#[test]
fn global_pseudo_voigt_lm_deconvolution_recovers_overlapping_multiplet_areas() {
    // Part 1: Tightly overlapping 500 MHz 1H region containing:
    // - A 1:2:1 triplet at 3.720, 3.706, 3.692 ppm (`J = 7.0 Hz`, `FWHM = 4.2 Hz` = 0.0084 ppm, `eta = 0.65`,
    //   total area = 2.00 -> areas `0.50, 1.00, 0.50`), AND
    // - An overlapping 1:1 doublet at 3.683, 3.669 ppm (`J = 7.0 Hz`, `FWHM = 4.0 Hz` = 0.0080 ppm, `eta = 0.75`,
    //   total area = 1.00 -> areas `0.50, 0.50`) whose downfield wing at 3.683 ppm merges directly into
    //   the upfield line of the triplet at 3.692 ppm (`Delta = 0.009 ppm ~ 1.07 * FWHM`!).
    let obs_mhz = 500.0;
    let axis = AxisPpm::from_bounds("1H", 8192, 4.20, 3.20, obs_mhz);

    let truth = [
        // Triplet (0.50 : 1.00 : 0.50)
        (3.720_f64, 0.0084_f64, 0.65_f64, 0.50_f64),
        (3.706_f64, 0.0084_f64, 0.65_f64, 1.00_f64),
        (3.692_f64, 0.0084_f64, 0.65_f64, 0.50_f64),
        // Overlapping doublet (0.50 : 0.50)
        (3.683_f64, 0.0080_f64, 0.75_f64, 0.50_f64),
        (3.669_f64, 0.0080_f64, 0.75_f64, 0.50_f64),
        // Separate singlet downfield (`eta = 0.35`, area `1.50`)
        (4.020_f64, 0.0060_f64, 0.35_f64, 1.50_f64),
    ];

    let mut real = vec![0.0_f64; axis.size];
    for (j, dst) in real.iter_mut().enumerate() {
        let x = axis.index_to_ppm(j as f64);
        for &(mu, g, eta, area) in &truth {
            *dst += eval_pseudo_voigt_1d(x, mu, g, eta, area);
        }
    }
    add_real_gaussian_noise(&mut real, 0.05, 314159);

    let spec = Spectrum1D::new(
        "voigt_overlap",
        Experiment1DKind::Proton1D,
        axis,
        real,
        None,
    );
    let dec = deconvolve_spectrum_1d(&spec, &DeconvolveConfig1D::default());

    assert_eq!(
        dec.peaks.len(),
        truth.len(),
        "expected {} deconvolved Voigt components, got {}",
        truth.len(),
        dec.peaks.len()
    );
    assert!(
        dec.residual_rmsd < 0.08,
        "residual RMSD {:.4} should sit near thermal noise 0.05",
        dec.residual_rmsd
    );

    // Match each ground-truth component to its nearest fitted component and assert < 2% area error!
    let mut max_voigt_area_err_pct = 0.0_f64;
    let mut max_voigt_pos_err_hz = 0.0_f64;
    for &(true_mu, true_g, true_eta, true_area) in &truth {
        let best = dec
            .peaks
            .iter()
            .min_by(|a, b| {
                (a.center_ppm - true_mu)
                    .abs()
                    .total_cmp(&(b.center_ppm - true_mu).abs())
            })
            .expect("matched peak");
        let pos_err_hz = (best.center_ppm - true_mu).abs() * obs_mhz;
        let area_rel_err = ((best.area_ppm - true_area) / true_area).abs();
        let fwhm_rel_err = ((best.fwhm_ppm - true_g) / true_g).abs();
        let eta_err = (best.lorentz_fraction - true_eta).abs();
        max_voigt_area_err_pct = max_voigt_area_err_pct.max(100.0 * area_rel_err);
        max_voigt_pos_err_hz = max_voigt_pos_err_hz.max(pos_err_hz);

        assert!(
            pos_err_hz < 0.15,
            "peak at {true_mu:.3} ppm position error {pos_err_hz:.4} Hz >= 0.15 Hz"
        );
        assert!(
            area_rel_err < 0.02,
            "peak at {true_mu:.3} ppm area relative error {:.3}% >= 2.0% (got {:.4}, expected {:.4})",
            100.0 * area_rel_err,
            best.area_ppm,
            true_area
        );
        assert!(
            fwhm_rel_err < 0.04,
            "peak at {true_mu:.3} ppm FWHM relative error {:.3}%",
            100.0 * fwhm_rel_err
        );
        assert!(
            eta_err < 0.08,
            "peak at {true_mu:.3} ppm Lorentz fraction error {eta_err:.3}"
        );
    }

    // Part 2: End-to-end time-domain FID -> FFT -> Voigt deconvolution on overlapping Lorentzian-Gaussian
    // doublet + triplet verifying < 2% area recovery in Hz (`area_hz` vs time-domain `amplitude`)!
    let fid_lines = [
        SyntheticLine1D {
            ppm: 2.420,
            amplitude: 1.0,
            fwhm_hz: 2.2,
            gauss_fwhm_hz: 1.0,
            phase_deg: 0.0,
        },
        SyntheticLine1D {
            ppm: 2.411,
            amplitude: 2.0,
            fwhm_hz: 2.2,
            gauss_fwhm_hz: 1.0,
            phase_deg: 0.0,
        },
        SyntheticLine1D {
            ppm: 2.402,
            amplitude: 1.0,
            fwhm_hz: 2.2,
            gauss_fwhm_hz: 1.0,
            phase_deg: 0.0,
        },
        SyntheticLine1D {
            ppm: 2.395,
            amplitude: 1.5,
            fwhm_hz: 2.2,
            gauss_fwhm_hz: 1.0,
            phase_deg: 0.0,
        },
        SyntheticLine1D {
            ppm: 2.386,
            amplitude: 1.5,
            fwhm_hz: 2.2,
            gauss_fwhm_hz: 1.0,
            phase_deg: 0.0,
        },
    ];
    let mut fid = synthesize_fid_1d(
        "fid_voigt_e2e",
        Experiment1DKind::Proton1D,
        8192,
        4000.0,
        500.0,
        2.40,
        &fid_lines,
    );
    apply_apodization_1d(&mut fid, Apodization1D::Exponential { lb_hz: 0.3 }, 0.5);
    zero_fill_pow2_1d(&mut fid, 4);
    let spec_fid = fft_1d(&fid);

    let cluster = deconvolve_region_1d(&spec_fid, 2.34, 2.46, None, &DeconvolveConfig1D::default());
    assert_eq!(cluster.peaks.len(), 5);
    let mut max_fid_voigt_err_pct = 0.0_f64;
    for line in &fid_lines {
        let fit = cluster
            .peaks
            .iter()
            .min_by(|a, b| {
                (a.center_ppm - line.ppm)
                    .abs()
                    .total_cmp(&(b.center_ppm - line.ppm).abs())
            })
            .unwrap();
        let rel_err = ((fit.area_hz - line.amplitude) / line.amplitude).abs();
        max_fid_voigt_err_pct = max_fid_voigt_err_pct.max(100.0 * rel_err);
        assert!(
            rel_err < 0.02,
            "FID peak at {:.3} ppm area_hz error {:.3}% >= 2.0% (got {:.4}, expected {:.4})",
            line.ppm,
            100.0 * rel_err,
            fit.area_hz,
            line.amplitude
        );
    }
    println!(
        "W1_METRIC_JSON:{}",
        serde_json::json!({
            "task": "T14.1",
            "kind": "t14_1_voigt_deconv",
            "n_components": dec.peaks.len(),
            "residual_rmsd": dec.residual_rmsd,
            "max_voigt_area_err_pct": max_voigt_area_err_pct,
            "max_voigt_pos_err_hz": max_voigt_pos_err_hz,
            "max_fid_e2e_area_err_pct": max_fid_voigt_err_pct
        })
    );
}

#[test]
fn dept135_biphasic_deconvolution_group_delay_serde_and_edge_cases() {
    // 1. DEPT-135 biphasic Voigt deconvolution (positive CH/CH3 and negative CH2 areas < 1.5% error).
    let obs_c13 = 125.76;
    let axis_dept = AxisPpm::from_bounds("13C", 4096, 80.0, 10.0, obs_c13);
    let dept_truth = [
        (68.40_f64, 0.030_f64, 0.80_f64, 1.00_f64),  // CH  (> 0)
        (42.15_f64, 0.030_f64, 0.80_f64, -2.00_f64), // CH2 (< 0)
        (19.80_f64, 0.030_f64, 0.80_f64, 3.00_f64),  // CH3 (> 0)
    ];
    let mut dept_real = vec![0.0_f64; axis_dept.size];
    for (j, dst) in dept_real.iter_mut().enumerate() {
        let x = axis_dept.index_to_ppm(j as f64);
        for &(mu, g, eta, area) in &dept_truth {
            *dst += eval_pseudo_voigt_1d(x, mu, g, eta, area);
        }
    }
    let spec_dept = Spectrum1D::new(
        "dept135_voigt",
        Experiment1DKind::Dept135,
        axis_dept,
        dept_real,
        None,
    );
    let dec_dept = deconvolve_spectrum_1d(&spec_dept, &DeconvolveConfig1D::default());
    assert_eq!(dec_dept.peaks.len(), 3);
    for (fit, &(_, _, _, exp_area)) in dec_dept.peaks.iter().zip(dept_truth.iter()) {
        assert_eq!(fit.area_ppm.signum(), exp_area.signum());
        assert!(((fit.area_ppm - exp_area) / exp_area).abs() < 0.015);
    }

    // 2. Fractional digital-filter group delay (`67.68` complex samples, Moskau 2002) compensation
    //    before apodization + sub-sample linear phase in `fft_1d`.
    let gd_exact = 67.68_f64;
    let dt = 1.0 / 6000.0;
    let obs = 500.0;
    let carrier = 4.7;
    let gd_lines = [
        (8.45_f64, 1.5_f64, 1.8_f64),
        (4.70_f64, 2.0_f64, 1.8_f64),
        (1.12_f64, 3.0_f64, 1.8_f64),
    ];
    let n_raw = 4096 + 68;
    let mut delayed_data = vec![[0.0, 0.0]; n_raw];
    for (k, dst) in delayed_data.iter_mut().enumerate() {
        let tau = ((k as f64) - gd_exact) * dt;
        if tau >= 0.0 {
            for &(ppm, amp, fwhm) in &gd_lines {
                let df = (ppm - carrier) * obs;
                let env = amp * (-std::f64::consts::PI * fwhm * tau).exp();
                let (s, c) = (2.0 * std::f64::consts::PI * df * tau).sin_cos();
                dst[0] += env * c;
                dst[1] += env * s;
            }
        }
    }
    let mut td_gd = nmr_core::nmr1d::TimeDomain1D::new(
        "gd_test",
        Experiment1DKind::Proton1D,
        6000.0,
        obs,
        carrier,
        delayed_data,
    );
    td_gd.group_delay_pts = gd_exact;
    apply_apodization_1d(&mut td_gd, Apodization1D::Exponential { lb_hz: 0.3 }, 0.5);
    zero_fill_pow2_1d(&mut td_gd, 2);
    let spec_gd = fft_1d(&td_gd);
    let residual_phase = auto_phase_1d(&spec_gd, &AutoPhase1DConfig::default());
    assert!(
        residual_phase.phc0_deg.abs() < 1.0 && residual_phase.phc1_deg.abs() < 2.0,
        "group delay compensation left residual phase ({:.3}, {:.3})",
        residual_phase.phc0_deg,
        residual_phase.phc1_deg
    );

    // 3. AsymmetricWhittaker baseline correction + Spectrum1D JSON serde round-trip.
    let mut spec_asym = spec_gd.clone();
    let n_asym = spec_asym.len();
    for (j, y) in spec_asym.real.iter_mut().enumerate() {
        let u = (j as f64) / ((n_asym - 1) as f64);
        *y += 0.25 * (std::f64::consts::PI * u).sin() + 0.10;
    }
    let _ = correct_baseline_1d(
        &mut spec_asym,
        &BaselineConfig1D {
            method: BaselineMethod1D::AsymmetricWhittaker {
                lambda: 5.0e5,
                asymmetry_p: 0.005,
            },
            ..BaselineConfig1D::default()
        },
    );
    let quiet_idx = spec_asym.axis.ppm_to_index(6.50).round() as usize;
    assert!(spec_asym.real[quiet_idx].abs() < 0.01);

    let json = serde_json::to_string(&spec_gd).expect("serialize Spectrum1D");
    let roundtrip: Spectrum1D = serde_json::from_str(&json).expect("deserialize Spectrum1D");
    assert_eq!(spec_gd.name, roundtrip.name);
    assert_eq!(spec_gd.experiment, roundtrip.experiment);
    assert_eq!(spec_gd.len(), roundtrip.len());
    for (&a, &b) in spec_gd.real.iter().zip(roundtrip.real.iter()) {
        assert!((a - b).abs() < 1e-14);
    }

    // 4. Coalesced shoulder doublet (`Delta = 0.72 * FWHM`, single local maximum!) rescued by
    //    residual-peak insertion in `deconvolve_region_1d`.
    let sh_axis = AxisPpm::from_bounds("1H", 4096, 3.60, 3.40, 500.0);
    let mut sh_real = vec![0.0_f64; sh_axis.size];
    for (j, dst) in sh_real.iter_mut().enumerate() {
        let x = sh_axis.index_to_ppm(j as f64);
        *dst += eval_pseudo_voigt_1d(x, 3.5050, 0.0100, 0.60, 1.00);
        *dst += eval_pseudo_voigt_1d(x, 3.4978, 0.0100, 0.60, 0.60);
    }
    let sh_spec = Spectrum1D::new(
        "shoulder",
        Experiment1DKind::Proton1D,
        sh_axis,
        sh_real,
        None,
    );
    let sh_res = deconvolve_region_1d(&sh_spec, 3.44, 3.56, None, &DeconvolveConfig1D::default());
    assert_eq!(sh_res.peaks.len(), 2);
    let total_sh_area: f64 = sh_res.peaks.iter().map(|p| p.area_ppm).sum();
    assert!(((total_sh_area - 1.60) / 1.60).abs() < 0.01);

    // 5. Degenerate all-zero & tiny spectra must never panic or produce NaN.
    let zero_ax = AxisPpm::from_bounds("1H", 16, 10.0, 0.0, 500.0);
    let mut zero_spec = Spectrum1D::new(
        "zeros",
        Experiment1DKind::Proton1D,
        zero_ax,
        vec![0.0; 16],
        None,
    );
    let z_ph = auto_phase_and_apply_1d(&mut zero_spec, &AutoPhase1DConfig::default());
    assert!(z_ph.phc0_deg.is_finite() && z_ph.phc1_deg.is_finite());
    let _ = correct_baseline_1d(&mut zero_spec, &BaselineConfig1D::default());
    assert!(zero_spec.real.iter().all(|v| v.is_finite()));
    assert!(pick_peaks_1d(&zero_spec, &PeakPickConfig1D::default()).is_empty());
    assert!(
        deconvolve_spectrum_1d(&zero_spec, &DeconvolveConfig1D::default())
            .peaks
            .is_empty()
    );
}

#[test]
fn test_f15_adversarial_1d_ingest_error_paths() {
    use nmr_core::nmr1d::{
        load_1d_input, load_bruker_pdata_1d_dir, load_processed_spectrum_1d, parse_json_1d_str,
        parse_tsv_1d_str, MAX_1D_INGEST_POINTS,
    };
    use std::fs;

    let tmp_root = std::env::temp_dir().join(format!(
        "nmr_core_f15_adversarial_1d_{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&tmp_root);
    fs::create_dir_all(&tmp_root).expect("create temp dir");

    let write_pdata = |sub: &str, procs_txt: &str, bytes_1r: &[u8], bytes_1i: Option<&[u8]>| {
        let exp_dir = tmp_root.join(sub);
        let pdata_dir = exp_dir.join("pdata").join("1");
        fs::create_dir_all(&pdata_dir).expect("create pdata/1");
        fs::write(
            exp_dir.join("acqus"),
            "##TITLE= f15_test\n##$PULPROG= <zg30>\n##$NUC1= <1H>\n##$SOLVENT= <CDCl3>\n",
        )
        .expect("write acqus");
        fs::write(pdata_dir.join("procs"), procs_txt).expect("write procs");
        fs::write(pdata_dir.join("1r"), bytes_1r).expect("write 1r");
        if let Some(im) = bytes_1i {
            fs::write(pdata_dir.join("1i"), im).expect("write 1i");
        }
        exp_dir
    };

    // Baseline valid Bruker pdata (`SI=16`, `DTYPP=0` i32 LE, 64 bytes)
    let mut valid_1r = Vec::with_capacity(64);
    for i in 0..16i32 {
        valid_1r.extend_from_slice(&(i * 100).to_le_bytes());
    }
    let valid_procs = "##$SI= 16\n##$SF= 500.13\n##$SW_p= 5000.0\n##$OFFSET= 10.0\n##$NC_proc= 0\n##$BYTORDP= 0\n##$DTYPP= 0\n";
    let ok_dir = write_pdata("ok_pdata", valid_procs, &valid_1r, None);
    let ok_spec = load_processed_spectrum_1d(&ok_dir, None).expect("valid pdata must load");
    assert_eq!(ok_spec.len(), 16);

    // 1. Truncated Bruker `1r` (`SI=64` expects 256 bytes, file has 100 bytes)
    let trunc_procs =
        "##$SI= 64\n##$SF= 500.13\n##$SW_p= 5000.0\n##$OFFSET= 10.0\n##$BYTORDP= 0\n##$DTYPP= 0\n";
    let trunc_dir = write_pdata("trunc_1r", trunc_procs, &[0u8; 100], None);
    let err_trunc =
        load_processed_spectrum_1d(&trunc_dir, None).expect_err("truncated 1r must Err");
    assert!(err_trunc.contains("truncated Bruker"), "got: {err_trunc}");

    // 2. `SI` disagreeing with `1r` length (`SI=16` expects 64 bytes, file has 128 bytes)
    let mismatch_dir = write_pdata("si_mismatch", valid_procs, &[0u8; 128], None);
    let err_mismatch =
        load_processed_spectrum_1d(&mismatch_dir, None).expect_err("SI != 1r length must Err");
    assert!(
        err_mismatch.contains("disagrees with"),
        "got: {err_mismatch}"
    );

    // 3. Truncated imaginary `1i` (`SI=16`, `1r`=64 bytes, `1i`=32 bytes)
    let bad_1i_dir = write_pdata("trunc_1i", valid_procs, &valid_1r, Some(&[0u8; 32]));
    let err_1i = load_bruker_pdata_1d_dir(&bad_1i_dir, None).expect_err("truncated 1i must Err");
    assert!(err_1i.contains("1i"), "got: {err_1i}");

    // 4. Oversized declared `SI` (`SI = 1_073_741_824` on tiny 16-byte `1r`) rejected BEFORE allocation
    let huge_si_procs =
        "##$SI= 1073741824\n##$SF= 500.13\n##$SW_p= 5000.0\n##$OFFSET= 10.0\n##$DTYPP= 0\n";
    let huge_dir = write_pdata("huge_si", huge_si_procs, &[0u8; 16], None);
    let err_huge =
        load_processed_spectrum_1d(&huge_dir, None).expect_err("oversized SI must Err pre-alloc");
    assert!(
        err_huge.contains(&MAX_1D_INGEST_POINTS.to_string()),
        "got: {err_huge}"
    );

    // 5. `NaN` and `+Inf` in Bruker `1r` (`DTYPP=2` IEEE-754 f64) and in `procs`
    let f64_procs =
        "##$SI= 8\n##$SF= 500.13\n##$SW_p= 5000.0\n##$OFFSET= 10.0\n##$BYTORDP= 0\n##$DTYPP= 2\n";
    let mut nan_1r = Vec::with_capacity(64);
    for k in 0..8 {
        let v = if k == 3 { f64::NAN } else { 1.0 };
        nan_1r.extend_from_slice(&v.to_le_bytes());
    }
    let nan_dir = write_pdata("nan_1r", f64_procs, &nan_1r, None);
    assert!(load_processed_spectrum_1d(&nan_dir, None).is_err());

    let mut inf_1r = Vec::with_capacity(64);
    for k in 0..8 {
        let v = if k == 5 { f64::INFINITY } else { 1.0 };
        inf_1r.extend_from_slice(&v.to_le_bytes());
    }
    let inf_dir = write_pdata("inf_1r", f64_procs, &inf_1r, None);
    assert!(load_processed_spectrum_1d(&inf_dir, None).is_err());

    let nan_sf_procs =
        "##$SI= 16\n##$SF= NaN\n##$SW_p= 5000.0\n##$OFFSET= 10.0\n##$BYTORDP= 0\n##$DTYPP= 0\n";
    let nan_sf_dir = write_pdata("nan_sf", nan_sf_procs, &valid_1r, None);
    assert!(load_processed_spectrum_1d(&nan_sf_dir, None).is_err());

    // 6. Adversarial TSV 1D FID & spectrum inputs
    let bad_tsvs = [
        "",
        "ppm\treal\n7.26\t1.0\n",
        "ppm\treal\n7.26\tNaN\n1.50\t2.0\n",
        "ppm\treal\ninf\t1.0\n1.50\t2.0\n",
        "ppm\treal\n7.26\t1.0\n5.00\t2.0\t0.5\n",
        "5.00\t1.0\n5.00\t2.0\n",
        "# domain=fid; sw_hz=-2000\n0.0\t1.0\t0.0\n0.001\t0.5\t0.1\n",
        "# domain=fid\nt_s\tre\tim\n0.002\t1.0\t0.0\n0.001\t0.5\t0.1\n",
        "# obs_mhz=NaN\n8.0\t1.0\n1.0\t2.0\n",
    ];
    for bad in bad_tsvs {
        assert!(
            parse_tsv_1d_str(bad, "bad_tsv", None).is_err(),
            "expected Err on adversarial TSV {bad:?}"
        );
    }

    // 7. Adversarial JSON 1D FID & spectrum inputs
    let bad_jsons = [
        "{not valid json",
        "[]",
        r#"{"foo": 123}"#,
        // Oversized axis.size rejected pre-deserialization
        r#"{"name":"x","experiment":"Proton1D","axis":{"label":"1H","obs_mhz":500.0,"size":99999999,"first_ppm":10.0,"step_ppm":-0.001},"real":[1.0,2.0]}"#,
        // axis.size != real.len()
        r#"{"name":"x","experiment":"Proton1D","axis":{"label":"1H","obs_mhz":500.0,"size":8,"first_ppm":10.0,"step_ppm":-0.1},"real":[1.0,2.0,3.0,4.0]}"#,
        // imag.len() != real.len()
        r#"{"name":"x","experiment":"Proton1D","axis":{"label":"1H","obs_mhz":500.0,"size":4,"first_ppm":10.0,"step_ppm":-0.1},"real":[1.0,2.0,3.0,4.0],"imag":[0.0,0.0]}"#,
        // TimeDomain1D active_td > data.len()
        r#"{"name":"td","experiment":"Proton1D","sw_hz":5000.0,"obs_mhz":500.0,"carrier_ppm":4.7,"data":[[1.0,0.0],[0.5,0.1]],"active_td":16,"group_delay_pts":0.0,"conjugate_for_fft":false}"#,
        // TimeDomain1D non-positive sw_hz
        r#"{"name":"td","experiment":"Proton1D","sw_hz":0.0,"obs_mhz":500.0,"carrier_ppm":4.7,"data":[[1.0,0.0],[0.5,0.1]],"active_td":2,"group_delay_pts":0.0,"conjugate_for_fft":false}"#,
    ];
    for bad in bad_jsons {
        assert!(
            parse_json_1d_str(bad, None).is_err(),
            "expected Err on adversarial JSON {bad}"
        );
    }

    // Empty directory must also return Err cleanly
    let empty_dir = tmp_root.join("empty_dir");
    fs::create_dir_all(&empty_dir).expect("mkdir empty");
    assert!(load_1d_input(&empty_dir, None, false).is_err());

    let _ = fs::remove_dir_all(&tmp_root);
}

#[test]
fn f29_dept135_ch2_free_vs_ch2_containing_hdo_mask_and_iupac_c13_dss_tms() {
    use nmr_core::nmr1d::{
        convert_c13_ppm_reference, convert_c13_ppm_xi_ratio, convert_c13_spectrum_reference_1d,
        ReferenceStandard, C13_DSS_VS_TMS_OFFSET_PPM, XI_C13_DSS, XI_C13_TMS,
    };

    // 1. CH2-free DEPT-135 (L-fucose equilibrium pattern: all positive CH at 96.6, 92.5, 73.3..66.7 ppm
    //    and CH3 at 15.9 ppm, zero CH2 carbons, no `biphasic_positive_ppm` hint!)
    let fucose_lines = [
        SyntheticLine1D::lorentzian(96.57, 1.0, 1.4),
        SyntheticLine1D::lorentzian(92.55, 0.5, 1.4),
        SyntheticLine1D::lorentzian(73.26, 1.0, 1.4),
        SyntheticLine1D::lorentzian(72.23, 0.5, 1.4),
        SyntheticLine1D::lorentzian(71.99, 1.0, 1.4),
        SyntheticLine1D::lorentzian(71.78, 1.0, 1.4),
        SyntheticLine1D::lorentzian(71.22, 1.0, 1.5),
        SyntheticLine1D::lorentzian(69.60, 0.5, 1.4),
        SyntheticLine1D::lorentzian(68.41, 0.5, 1.5),
        SyntheticLine1D::lorentzian(66.66, 0.5, 1.5),
        SyntheticLine1D::lorentzian(15.85, 1.2, 2.0),
    ];
    let mut fid_fuc = synthesize_fid_1d(
        "fucose_dept135",
        Experiment1DKind::Dept135,
        8192,
        15_151.5,
        100.612_729,
        75.0,
        &fucose_lines,
    );
    add_complex_gaussian_noise(&mut fid_fuc, 1.5e-3, 20261117);
    apply_apodization_1d(&mut fid_fuc, Apodization1D::Exponential { lb_hz: 1.0 }, 0.5);
    zero_fill_pow2_1d(&mut fid_fuc, 2);
    let mut spec_fuc = fft_1d(&fid_fuc);
    // Corrupt with vendor-like phase (-57.15 deg, +18.20 deg correction needed)
    apply_phase_1d(&mut spec_fuc, 57.15, -18.20, Pivot1D::Fraction(0.5));

    let sol_fuc = auto_phase_and_apply_1d(&mut spec_fuc, &AutoPhase1DConfig::default());
    assert!(
        sol_fuc.phc1_deg.abs() < 40.0,
        "CH2-free DEPT-135 must NOT trap at PHC1 = +-180 deg, got {:.2}",
        sol_fuc.phc1_deg
    );
    assert!(
        phase_diff_deg(sol_fuc.phc0_deg, -57.15).abs() < 3.0,
        "CH2-free DEPT-135 |dPHC0| too large: got {:.2}, expected -57.15",
        sol_fuc.phc0_deg
    );
    assert!(
        (sol_fuc.phc1_deg - 18.20).abs() < 5.0,
        "CH2-free DEPT-135 |dPHC1| too large: got {:.2}, expected +18.20",
        sol_fuc.phc1_deg
    );
    let picked_fuc = pick_peaks_1d(&spec_fuc, &PeakPickConfig1D::default());
    assert!(picked_fuc.len() >= 10);
    assert!(
        picked_fuc.iter().all(|p| p.height > 0.0),
        "CH2-free DEPT-135 must have 0 inverted CH2 peaks, got {picked_fuc:?}"
    );

    // 2. Genuine CH2-containing DEPT-135 WITHOUT `biphasic_positive_ppm` must still keep CH/CH3 > 0 and CH2 < 0!
    let mixed_dept_lines = [
        SyntheticLine1D::lorentzian(129.4, 1.4, 1.8), // aromatic CH (+)
        SyntheticLine1D::lorentzian(71.8, 1.1, 1.8),  // CH-OH (+)
        SyntheticLine1D {
            ppm: 43.2,
            amplitude: 1.5,
            fwhm_hz: 1.8,
            gauss_fwhm_hz: 0.0,
            phase_deg: 180.0, // CH2 (-)
        },
        SyntheticLine1D {
            ppm: 31.5,
            amplitude: 1.4,
            fwhm_hz: 1.8,
            gauss_fwhm_hz: 0.0,
            phase_deg: 180.0, // CH2 (-)
        },
        SyntheticLine1D::lorentzian(14.1, 1.6, 1.8), // CH3 (+)
    ];
    let mut fid_mix = synthesize_fid_1d(
        "mixed_dept135",
        Experiment1DKind::Dept135,
        4096,
        24_000.0,
        125.76,
        90.0,
        &mixed_dept_lines,
    );
    apply_apodization_1d(&mut fid_mix, Apodization1D::Exponential { lb_hz: 1.0 }, 0.5);
    zero_fill_pow2_1d(&mut fid_mix, 2);
    let mut spec_mix = fft_1d(&fid_mix);
    apply_phase_1d(&mut spec_mix, 42.0, -35.0, Pivot1D::Fraction(0.5));
    let sol_mix = auto_phase_and_apply_1d(&mut spec_mix, &AutoPhase1DConfig::default());
    assert!(phase_diff_deg(sol_mix.phc0_deg, -42.0).abs() < 3.0);
    assert!((sol_mix.phc1_deg - 35.0).abs() < 5.0);
    // Verify CH/CH3 positive and CH2 negative
    let idx_ch = spec_mix.axis.ppm_to_index(129.4).round() as usize;
    let idx_ch2 = spec_mix.axis.ppm_to_index(43.2).round() as usize;
    let idx_ch3 = spec_mix.axis.ppm_to_index(14.1).round() as usize;
    assert!(spec_mix.real[idx_ch] > 0.0 && spec_mix.real[idx_ch3] > 0.0);
    assert!(spec_mix.real[idx_ch2] < 0.0);

    // 3. Aqueous 1H HDO mask (`4.60..4.90 ppm`): massive out-of-phase radiation-damped HDO singlet
    //    at `4.76 ppm` (`50x` solute height, twisted by `+55 deg` radiation damping) must NOT corrupt
    //    solute autophasing even when `solvent` is mislabeled `<CDCl3>`!
    let h1_aq_lines = [
        SyntheticLine1D {
            ppm: 4.76,
            amplitude: 120.0,
            fwhm_hz: 4.5,
            gauss_fwhm_hz: 0.0,
            phase_deg: 55.0, // radiation-damped HDO phase anomaly!
        },
        SyntheticLine1D::lorentzian(5.20, 1.5, 1.3),
        SyntheticLine1D::lorentzian(4.55, 1.2, 1.3), // beta-anomeric H1 right at 4.55 ppm outside [4.60, 4.90]!
        SyntheticLine1D::lorentzian(3.80, 3.0, 1.3),
        SyntheticLine1D::lorentzian(1.22, 4.5, 1.4),
    ];
    let mut fid_aq = synthesize_fid_1d(
        "aq_1h",
        Experiment1DKind::Proton1D,
        4096,
        4800.0,
        400.13,
        4.70,
        &h1_aq_lines,
    );
    add_complex_gaussian_noise(&mut fid_aq, 1.0e-3, 9911);
    apply_apodization_1d(&mut fid_aq, Apodization1D::Exponential { lb_hz: 0.3 }, 0.5);
    zero_fill_pow2_1d(&mut fid_aq, 2);
    let mut spec_aq = fft_1d(&fid_aq);
    spec_aq.solvent = Some("CDCl3".to_string()); // deliberately mislabeled solvent!
    apply_phase_1d(&mut spec_aq, -33.0, 24.0, Pivot1D::Fraction(0.5));
    let sol_aq = auto_phase_and_apply_1d(&mut spec_aq, &AutoPhase1DConfig::default());
    assert!(
        phase_diff_deg(sol_aq.phc0_deg, 33.0).abs() < 3.0,
        "Solute 1H PHC0 corrupted by radiation-damped HDO: got {:.2}, expected 33.0",
        sol_aq.phc0_deg
    );
    assert!(
        (sol_aq.phc1_deg - (-24.0)).abs() < 5.0,
        "Solute 1H PHC1 corrupted by radiation-damped HDO: got {:.2}, expected -24.0",
        sol_aq.phc1_deg
    );

    // 4. IUPAC 2001/2008 13C DSS-vs-TMS ReferenceStandard exact first-principles Xi derivation (`20.W1:P0-2`)
    let expected_dss_in_tms_ppm =
        ((0.251_449_53_f64 - 0.251_450_20_f64) / 0.251_450_20_f64) * 1.0e6;
    assert!((XI_C13_TMS - 0.251_450_20).abs() < 1e-10);
    assert!((XI_C13_DSS - 0.251_449_53).abs() < 1e-10);
    assert!(
        (C13_DSS_VS_TMS_OFFSET_PPM - expected_dss_in_tms_ppm).abs() < 1e-12,
        "C13_DSS_VS_TMS_OFFSET_PPM must equal ((XI_C13_DSS - XI_C13_TMS) / XI_C13_TMS) * 1e6 = {expected_dss_in_tms_ppm:.9}, got {C13_DSS_VS_TMS_OFFSET_PPM:.9}"
    );
    assert!(
        ((0.251_449_53_f64 / 0.251_450_20_f64 - 1.0) * 1.0e6 - C13_DSS_VS_TMS_OFFSET_PPM).abs()
            < 1e-9
    );
    assert!((C13_DSS_VS_TMS_OFFSET_PPM - (-2.664_543_5)).abs() < 1e-6);
    assert!(
        (C13_DSS_VS_TMS_OFFSET_PPM - (-2.371)).abs() > 0.29,
        "regression guard: fucose-fitted -2.371 ppm literal must not return"
    );
    let dss_zero_in_tms =
        convert_c13_ppm_reference(0.0, ReferenceStandard::Dss, ReferenceStandard::Tms);
    assert!((dss_zero_in_tms - expected_dss_in_tms_ppm).abs() < 1e-12);
    let tms_zero_in_dss =
        convert_c13_ppm_reference(0.0, ReferenceStandard::Tms, ReferenceStandard::Dss);
    let xi_zero_in_dss =
        convert_c13_ppm_xi_ratio(0.0, ReferenceStandard::Tms, ReferenceStandard::Dss);
    assert!((tms_zero_in_dss - xi_zero_in_dss).abs() < 1e-12);
    assert!((tms_zero_in_dss - 2.664_550_6).abs() < 1e-6);
    let sample_dss =
        convert_c13_ppm_reference(96.5678, ReferenceStandard::Tms, ReferenceStandard::Dss);
    assert!((sample_dss - (96.5678 + 2.664_808)).abs() < 1e-5);
    let roundtrip =
        convert_c13_ppm_reference(sample_dss, ReferenceStandard::Dss, ReferenceStandard::Tms);
    assert!((roundtrip - 96.5678).abs() < 1e-11);
    let xi_rt = convert_c13_ppm_xi_ratio(
        convert_c13_ppm_xi_ratio(100.0, ReferenceStandard::Tms, ReferenceStandard::Dss),
        ReferenceStandard::Dss,
        ReferenceStandard::Tms,
    );
    assert!((xi_rt - 100.0).abs() < 1e-11);

    let delta_spec = convert_c13_spectrum_reference_1d(
        &mut spec_fuc,
        ReferenceStandard::Tms,
        ReferenceStandard::Dss,
    )
    .expect("13C TMS -> DSS");
    assert!((delta_spec - 2.6647).abs() < 0.005);
    assert!(
        convert_c13_spectrum_reference_1d(
            &mut spec_aq,
            ReferenceStandard::Tms,
            ReferenceStandard::Dss
        )
        .is_err(),
        "1H spectrum must reject 13C reference conversion"
    );
}

#[test]
fn f29_clean_room_jeol_jdf_1d_big_and_little_endian_multinuclear_and_f15_adversarial() {
    use nmr_core::nmr1d::{
        compute_jeol_fir_group_delay, load_1d_from_bytes, load_1d_from_virtual_entries,
        load_1d_input, parse_jeol_jdf_1d_bytes, prepare_loaded_1d_for_studio, Loaded1DData,
        MAX_1D_INGEST_POINTS,
    };

    // Verify exact multi-stage FIR decimation group-delay algebra
    let gd_4_2 = compute_jeol_fir_group_delay("2 28 73", "4 2").expect("FIR group delay");
    assert!((gd_4_2 - 19.6875).abs() < 1e-12);
    let gd_6_2 = compute_jeol_fir_group_delay("2 41 74", "6 2").expect("FIR group delay");
    assert!((gd_6_2 - (40.0 / 24.0 + 73.0 / 4.0)).abs() < 1e-12);

    // Helper to synthesize a clean-room binary JEOL `.jdf` 1D buffer (`1296`-byte BE header + param table + data)
    #[allow(clippy::too_many_arguments)]
    fn build_synth_jdf(
        little_endian: bool,
        data_format: u8,
        axis_type: u8,
        axis_unit_base: u8,
        n_pts: u32,
        off_start: u32,
        off_stop: u32,
        ax_start: f64,
        ax_stop: f64,
        base_freq_mhz: f64,
        params: &[(&str, u32, f64, &str, u8)],
        re: &[f64],
        im: &[f64],
    ) -> Vec<u8> {
        let mut buf = vec![0u8; 1296];
        buf[0..8].copy_from_slice(b"JEOL.NMR");
        buf[8] = if little_endian { 1 } else { 0 };
        buf[9] = 1; // Major_Version = 1
        buf[12] = 1; // Data_Dimension_Number = 1
        buf[13] = 0x01;
        buf[14] = data_format;
        buf[24] = axis_type;
        buf[32] = 1; // unit prefix nibble 0 -> 10^0
        buf[33] = axis_unit_base;
        buf[176..180].copy_from_slice(&n_pts.to_be_bytes());
        buf[208..212].copy_from_slice(&off_start.to_be_bytes());
        buf[240..244].copy_from_slice(&off_stop.to_be_bytes());
        buf[272..280].copy_from_slice(&ax_start.to_be_bytes());
        buf[336..344].copy_from_slice(&ax_stop.to_be_bytes());
        buf[1064..1072].copy_from_slice(&base_freq_mhz.to_be_bytes());

        let param_start = 1296u32;
        let put_u32 = |dst: &mut [u8], v: u32| {
            let b = if little_endian {
                v.to_le_bytes()
            } else {
                v.to_be_bytes()
            };
            dst[0..4].copy_from_slice(&b);
        };
        let put_f64 = |dst: &mut [u8], v: f64| {
            let b = if little_endian {
                v.to_le_bytes()
            } else {
                v.to_be_bytes()
            };
            dst[0..8].copy_from_slice(&b);
        };
        let put_i32 = |dst: &mut [u8], v: i32| {
            let b = if little_endian {
                v.to_le_bytes()
            } else {
                v.to_be_bytes()
            };
            dst[0..4].copy_from_slice(&b);
        };

        let mut psec = vec![0u8; 16 + params.len() * 64];
        put_u32(&mut psec[0..4], 64);
        put_u32(&mut psec[4..8], 0);
        put_u32(&mut psec[8..12], params.len().saturating_sub(1) as u32);
        put_u32(&mut psec[12..16], params.len() as u32);
        for (idx, &(pname, vtype, fval, sval, ubase)) in params.iter().enumerate() {
            let rec = &mut psec[16 + idx * 64..16 + (idx + 1) * 64];
            rec[6] = if ubase == 0 { 0 } else { 1 };
            rec[7] = ubase;
            put_u32(&mut rec[32..36], vtype);
            match vtype {
                0 => {
                    let sb = sval.as_bytes();
                    let l = sb.len().min(16);
                    rec[16..16 + l].copy_from_slice(&sb[..l]);
                }
                1 => put_i32(&mut rec[16..20], fval as i32),
                2 => put_f64(&mut rec[16..24], fval),
                _ => {}
            }
            let nb = pname.as_bytes();
            let nl = nb.len().min(27);
            rec[36..36 + nl].copy_from_slice(&nb[..nl]);
        }

        let param_len = psec.len() as u32;
        let data_start = param_start + param_len;
        let push_scalar = |out: &mut Vec<u8>, v: f64| match data_format {
            1 => {
                let b = if little_endian {
                    v.to_le_bytes()
                } else {
                    v.to_be_bytes()
                };
                out.extend_from_slice(&b);
            }
            2 => {
                let vf = v as f32;
                let b = if little_endian {
                    vf.to_le_bytes()
                } else {
                    vf.to_be_bytes()
                };
                out.extend_from_slice(&b);
            }
            _ => {
                let vi = v.round() as i32;
                let b = if little_endian {
                    vi.to_le_bytes()
                } else {
                    vi.to_be_bytes()
                };
                out.extend_from_slice(&b);
            }
        };

        let mut dsec = Vec::new();
        if axis_type == 1 {
            for &r in re {
                push_scalar(&mut dsec, r);
            }
        } else if axis_type == 4 {
            for (&r, &m) in re.iter().zip(im.iter()) {
                push_scalar(&mut dsec, r);
                push_scalar(&mut dsec, m);
            }
        } else {
            for &r in re {
                push_scalar(&mut dsec, r);
            }
            for &m in im {
                push_scalar(&mut dsec, m);
            }
        }
        let data_len = dsec.len() as u32;
        buf[1212..1216].copy_from_slice(&param_start.to_be_bytes());
        buf[1216..1220].copy_from_slice(&param_len.to_be_bytes());
        buf[1284..1288].copy_from_slice(&data_start.to_be_bytes());
        buf[1292..1296].copy_from_slice(&data_len.to_be_bytes());
        buf.extend_from_slice(&psec);
        buf.extend_from_slice(&dsec);
        buf
    }

    // 1. Little-endian `f64` split-complex `19F` processed spectrum `.jdf` (peak at -62.5 ppm)
    let n_f19 = 516usize;
    let (os, oe) = (2usize, 513usize); // 512 active points
    let mut re_f19 = vec![0.0; n_f19];
    let im_f19 = vec![0.0; n_f19];
    for (idx, slot) in re_f19[os..=oe].iter_mut().enumerate() {
        let u = (idx as f64) / ((oe - os) as f64);
        let ppm = 0.0 - 200.0 * u;
        let d = (ppm - (-62.5)) / 0.25;
        *slot = 10.0 / (1.0 + d * d);
    }
    let jdf_f19 = build_synth_jdf(
        true,
        1,
        3,
        26, // PPM
        n_f19 as u32,
        os as u32,
        oe as u32,
        0.0,
        -200.0,
        564.726,
        &[
            ("X_DOMAIN", 0, 0.0, "Fluorine19", 0),
            ("X_OFFSET", 2, -100.0, "", 26),
            ("solvent", 0, 0.0, "CHLOROFORM-D", 0),
        ],
        &re_f19,
        &im_f19,
    );
    let loaded_f19 =
        load_1d_from_bytes("cf3_sample.jdf", &jdf_f19, None, false).expect("parse LE f64 19F JDF");
    match loaded_f19 {
        Loaded1DData::FrequencyDomain(sp) => {
            assert_eq!(sp.experiment, Experiment1DKind::Fluorine19);
            assert_eq!(sp.len(), 512);
            assert_eq!(sp.solvent.as_deref(), Some("CHLOROFORM-D"));
            let pks = pick_peaks_1d(&sp, &PeakPickConfig1D::default());
            assert_eq!(pks.len(), 1);
            assert!((pks[0].ppm - (-62.5)).abs() < 0.15);
        }
        other => panic!("expected FrequencyDomain, got {other:?}"),
    }

    // 2. Big-endian `f32` interleaved-complex `31P` processed spectrum `.jdf` (peak at +18.4 ppm)
    let n_p31 = 256usize;
    let mut re_p31 = vec![0.0; n_p31];
    let im_p31 = vec![0.0; n_p31];
    for (k, slot) in re_p31.iter_mut().enumerate() {
        let ppm = 80.0 - 120.0 * (k as f64) / ((n_p31 - 1) as f64);
        let d = (ppm - 18.4) / 0.30;
        *slot = 25.0 / (1.0 + d * d);
    }
    let jdf_p31 = build_synth_jdf(
        false, // BIG-ENDIAN!
        2,     // f32
        4,     // interleaved Real_Complex
        26,    // PPM
        n_p31 as u32,
        0,
        (n_p31 - 1) as u32,
        80.0,
        -40.0,
        242.95,
        &[
            ("X_DOMAIN", 0, 0.0, "Phosphorus31", 0),
            ("solvent", 0, 0.0, "DMSO-D6", 0),
        ],
        &re_p31,
        &im_p31,
    );
    let virt = load_1d_from_virtual_entries(&[("ar_p31.jdf", jdf_p31.as_slice())], None, false)
        .expect("virtual entry BE f32 31P JDF");
    assert_eq!(virt.len(), 1);
    let sp_p31 = prepare_loaded_1d_for_studio(virt.into_iter().next().unwrap()).unwrap();
    assert_eq!(sp_p31.experiment, Experiment1DKind::Phosphorus31);
    let pks_p31 = pick_peaks_1d(&sp_p31, &PeakPickConfig1D::default());
    assert!((pks_p31[0].ppm - 18.4).abs() < 0.25);

    // 3. Big-endian `f64` time-domain `1H` FID `.jdf` (`Sec` = 28) with FIR `orders`/`factors` group delay
    //    written to disk and loaded via `load_1d_input`!
    let n_fid = 1024usize;
    let sw_hz = 6000.0;
    let obs_h = 500.0;
    let carrier = 5.0;
    let target_ppm = 7.26; // CHCl3 downfield of 5.0 ppm carrier
    let df_hz = (target_ppm - carrier) * obs_h;
    let gd_samples = 4usize; // orders="1 9", factors="1" -> gd = (9 - 1) / (2 * 1) = 4.0 samples!
    let mut re_h = vec![0.0; n_fid];
    let mut im_h = vec![0.0; n_fid];
    for (idx, (rh, ih)) in re_h[gd_samples..]
        .iter_mut()
        .zip(im_h[gd_samples..].iter_mut())
        .enumerate()
    {
        let t = (idx as f64) / sw_hz;
        let env = 20.0 * (-std::f64::consts::PI * 1.5 * t).exp();
        let ang = 2.0 * std::f64::consts::PI * df_hz * t;
        // JEOL clockwise quadrature (`re - i*im` conjugated during FFT)
        *rh = env * ang.cos();
        *ih = -env * ang.sin();
    }
    let jdf_h_fid = build_synth_jdf(
        false,
        1,
        3,
        28, // Sec -> TimeDomain FID!
        n_fid as u32,
        0,
        (n_fid - 1) as u32,
        0.0,
        ((n_fid - 1) as f64) / sw_hz,
        obs_h,
        &[
            ("X_DOMAIN", 0, 0.0, "Proton", 0),
            ("X_SWEEP", 2, sw_hz, "", 13),
            ("X_OFFSET", 2, carrier, "", 26),
            ("DIGITAL_FILTER", 0, 0.0, "TRUE", 0),
            ("orders", 0, 0.0, "1 9", 0),
            ("factors", 0, 0.0, "1", 0),
            ("solvent", 0, 0.0, "CDCl3", 0),
        ],
        &re_h,
        &im_h,
    );
    let tmp_jdf_path = std::env::temp_dir().join("f29_test_proton_fid.jdf");
    std::fs::write(&tmp_jdf_path, &jdf_h_fid).expect("write temp jdf");
    let loaded_h = load_1d_input(&tmp_jdf_path, None, true).expect("load_1d_input JDF FID");
    let _ = std::fs::remove_file(&tmp_jdf_path);
    assert!(matches!(loaded_h, Loaded1DData::TimeDomain(_)));
    let sp_h = prepare_loaded_1d_for_studio(loaded_h).expect("studio process JDF FID");
    let pks_h = pick_peaks_1d(&sp_h, &PeakPickConfig1D::default());
    let main_pk_h = pks_h
        .iter()
        .max_by(|a, b| a.height.total_cmp(&b.height))
        .expect("picked CHCl3 peak from JEOL FID");
    assert!(
        (main_pk_h.ppm - 7.26).abs() < 0.02,
        "expected 7.26 ppm CHCl3 from JEOL FID, got {:.4}",
        main_pk_h.ppm
    );

    // 4. Adversarial `F15` `.jdf` inputs (truncated header, bad magic, 2D dim=2, oversized n_pts,
    //    truncated data body, and `NaN` scalar) must all return `Err` cleanly with zero panics!
    assert!(parse_jeol_jdf_1d_bytes("short", &jdf_f19[..200], None).is_err());

    let mut bad_magic = jdf_f19.clone();
    bad_magic[0..8].copy_from_slice(b"NOT_JEOL");
    assert!(parse_jeol_jdf_1d_bytes("bad_magic", &bad_magic, None).is_err());

    let mut bad_2d = jdf_f19.clone();
    bad_2d[12] = 2; // 2D JDF
    assert!(parse_jeol_jdf_1d_bytes("dim2", &bad_2d, None).is_err());

    let mut huge_pts = jdf_f19.clone();
    let over_pts = (MAX_1D_INGEST_POINTS as u32) + 100;
    huge_pts[176..180].copy_from_slice(&over_pts.to_be_bytes());
    assert!(parse_jeol_jdf_1d_bytes("oversized", &huge_pts, None).is_err());

    assert!(parse_jeol_jdf_1d_bytes("trunc_body", &jdf_f19[..jdf_f19.len() - 32], None).is_err());

    let mut nan_jdf = jdf_f19.clone();
    let d_off = u32::from_be_bytes(nan_jdf[1284..1288].try_into().unwrap()) as usize + 8 * 10;
    nan_jdf[d_off..d_off + 8].copy_from_slice(&f64::NAN.to_le_bytes());
    assert!(parse_jeol_jdf_1d_bytes("nan_body", &nan_jdf, None).is_err());
}

#[test]
fn p20_w1_p1_2_dept135_calpha_54ppm_ch_and_66ppm_ch2_no_fucose_band_inversion() {
    use nmr_core::nmr1d::{auto_phase_dept135_with_dept90, PolarityResolution1D};

    // Amino-acid / glycopeptide-like motif that defeats any hardcoded chemical-shift-band sign table:
    // - C-alpha methine `CH` at `54.0 ppm` (positive in DEPT-135, inside 50-62 ppm!)
    // - Hydroxymethyl / PEG `CH2`s at `66.0 ppm` and `70.5 ppm` (negative in DEPT-135, inside 62-72 ppm!)
    let dept135_lines = [
        SyntheticLine1D::lorentzian(54.0, 1.2, 1.8), // C-alpha CH (+)
        SyntheticLine1D {
            ppm: 66.0,
            amplitude: 1.5,
            fwhm_hz: 1.8,
            gauss_fwhm_hz: 0.0,
            phase_deg: 180.0, // CH2 (-)
        },
        SyntheticLine1D {
            ppm: 70.5,
            amplitude: 1.5,
            fwhm_hz: 1.8,
            gauss_fwhm_hz: 0.0,
            phase_deg: 180.0, // CH2 (-)
        },
    ];
    let mut fid_135 = synthesize_fid_1d(
        "calpha_peg_dept135",
        Experiment1DKind::Dept135,
        4096,
        20_000.0,
        125.76,
        62.0,
        &dept135_lines,
    );
    apply_apodization_1d(&mut fid_135, Apodization1D::Exponential { lb_hz: 1.0 }, 0.5);
    zero_fill_pow2_1d(&mut fid_135, 2);
    let base_135 = fft_1d(&fid_135);

    // Case A: Without DEPT-90 or `biphasic_positive_ppm`, under mild receiver phase error (+38 deg, -24 deg),
    // autophase MUST NOT flip C-alpha CH(54 ppm) negative and CH2(66, 70.5 ppm) positive, and MUST emit
    // `AmbiguousGlobalPolarity` warning because global 0 vs 180 deg sign cannot be proven from chemical shift alone.
    let mut spec_a = base_135.clone();
    apply_phase_1d(&mut spec_a, 38.0, -24.0, Pivot1D::Fraction(0.5));
    let sol_a = auto_phase_and_apply_1d(&mut spec_a, &AutoPhase1DConfig::default());
    let i_ca = spec_a.axis.ppm_to_index(54.0).round() as usize;
    let i_ch2_66 = spec_a.axis.ppm_to_index(66.0).round() as usize;
    let i_ch2_70 = spec_a.axis.ppm_to_index(70.5).round() as usize;
    assert!(
        spec_a.real[i_ca] > 0.0 && spec_a.real[i_ch2_66] < 0.0 && spec_a.real[i_ch2_70] < 0.0,
        "DEPT-135 C-alpha CH(54 ppm)={:.3} must stay > 0 and CH2(66, 70.5 ppm)=({:.3}, {:.3}) must stay < 0",
        spec_a.real[i_ca],
        spec_a.real[i_ch2_66],
        spec_a.real[i_ch2_70]
    );
    assert_eq!(
        sol_a.polarity_resolution,
        PolarityResolution1D::AmbiguousGlobalPolarity
    );
    assert!(
        spec_a
            .processing_warnings
            .iter()
            .any(|w| w.contains("AmbiguousGlobalPolarity")),
        "expected AmbiguousGlobalPolarity warning on unassisted DEPT-135, got {:?}",
        spec_a.processing_warnings
    );

    // Case B: Even under a 180-deg-inverted receiver phase (-142 deg = +38 - 180 deg), an accompanying
    // `DEPT-90` spectrum (which excites ONLY methine CH at 54.0 ppm) disambiguates global polarity cleanly!
    let dept90_lines = [SyntheticLine1D::lorentzian(54.0, 1.2, 1.8)];
    let mut fid_90 = synthesize_fid_1d(
        "calpha_peg_dept90",
        Experiment1DKind::Dept90,
        4096,
        20_000.0,
        125.76,
        62.0,
        &dept90_lines,
    );
    apply_apodization_1d(&mut fid_90, Apodization1D::Exponential { lb_hz: 1.0 }, 0.5);
    zero_fill_pow2_1d(&mut fid_90, 2);
    let spec_90 = fft_1d(&fid_90);

    let mut spec_b = base_135;
    apply_phase_1d(&mut spec_b, -142.0, -24.0, Pivot1D::Fraction(0.5));
    let sol_b =
        auto_phase_dept135_with_dept90(&mut spec_b, &spec_90, &AutoPhase1DConfig::default())
            .expect("DEPT-90 disambiguation");
    assert_eq!(
        sol_b.polarity_resolution,
        PolarityResolution1D::MultiplicityReference
    );
    assert!(
        spec_b.processing_warnings.is_empty(),
        "DEPT-90 disambiguated DEPT-135 must not emit ambiguity warning: {:?}",
        spec_b.processing_warnings
    );
    assert!(
        spec_b.real[i_ca] > 0.0 && spec_b.real[i_ch2_66] < 0.0 && spec_b.real[i_ch2_70] < 0.0,
        "DEPT-90 disambiguation failed: CH(54)={:.3}, CH2(66)={:.3}, CH2(70.5)={:.3}",
        spec_b.real[i_ca],
        spec_b.real[i_ch2_66],
        spec_b.real[i_ch2_70]
    );
}

#[test]
fn p20_w1_p2_2_real_public_jeol_jdf_fixture_rutin_400mhz_dmsod6() {
    use nmr_core::nmr1d::{
        load_processed_spectrum_1d, parse_jeol_jdf_1d_bytes, prepare_loaded_1d_for_studio,
        Loaded1DData,
    };
    use std::path::Path;

    let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../data/fixtures/jeol_public/rutin_q1h_400mhz_dmsod6.jdf");
    assert!(
        fixture.is_file(),
        "missing real public JEOL .jdf fixture at {}",
        fixture.display()
    );
    let bytes = std::fs::read(&fixture).expect("read public JEOL .jdf fixture");
    assert_eq!(bytes.len(), 555_392);
    assert_eq!(&bytes[0..8], b"JEOL.NMR");

    let parsed = parse_jeol_jdf_1d_bytes("rutin_q1h_400mhz_dmsod6", &bytes, None)
        .expect("parse real public JEOL Delta 1D .jdf");
    match &parsed {
        Loaded1DData::TimeDomain(td) => {
            assert_eq!(td.experiment, Experiment1DKind::Proton1D);
            // `Data_Points = 32768` with `Data_Offset = [20..=32767]` (`32_748` active samples)
            assert_eq!(td.data.len(), 32_748);
            assert!((td.obs_mhz - 399.782_198).abs() < 1e-3);
            assert_eq!(td.solvent.as_deref(), Some("DMSO-D6"));
        }
        Loaded1DData::FrequencyDomain(_) => {
            panic!("expected time-domain qHNMR FID in rutin_q1h_400mhz_dmsod6.jdf")
        }
    }

    let studio_spec =
        prepare_loaded_1d_for_studio(parsed).expect("studio process real JEOL .jdf FID");
    let path_spec = load_processed_spectrum_1d(&fixture, None)
        .expect("load_processed_spectrum_1d real JEOL .jdf");
    assert_eq!(studio_spec.len(), path_spec.len());
    assert!(
        (studio_spec.axis.first_ppm - path_spec.axis.first_ppm).abs() < 1e-12,
        "byte/studio and path 1D pipelines must agree identically on real JEOL .jdf"
    );
    assert!(
        studio_spec.applied_reference_shift_ppm.is_some(),
        "DMSO-d6 residual quintet at 2.50 ppm must calibrate and record applied_reference_shift_ppm"
    );
    let pks = pick_peaks_1d(&studio_spec, &PeakPickConfig1D::default());
    assert!(
        pks.iter().any(|p| (p.ppm - 2.50).abs() < 0.02),
        "expected calibrated DMSO-d5 residual peak at 2.50 ppm, got top peaks {:?}",
        &pks[..pks.len().min(8)]
    );
    // Rutin flavonol H-6 / H-8 / B-ring aromatic protons (6.15 - 7.60 ppm) & H-1'' anomeric (~5.10 ppm)
    assert!(
        pks.iter().any(|p| (6.10..=7.65).contains(&p.ppm)),
        "expected Rutin aromatic 1H resonances in 6.10..7.65 ppm"
    );
}

#[test]
fn v0150_p1_3_and_p2_shared_1d_pipeline_solvent_lineshape_guard_and_sibling_isolation() {
    use nmr_core::nmr1d::{
        load_1d_from_virtual_entries, load_1d_from_virtual_entries_isolated,
        load_processed_spectrum_1d, prepare_loaded_1d_for_studio, process_loaded_1d, Loaded1DData,
        Process1DOptions, WASM32_MAX_1D_ARCHIVE_INFLATE_BYTES,
    };

    // 1. `wasm32` archive expansion cap must be <= 128 MiB (`v0150` P2).
    const {
        assert!(WASM32_MAX_1D_ARCHIVE_INFLATE_BYTES <= 128 * 1024 * 1024);
    }

    // 2. Solvent lineshape guard (`v0150:P1-3`):
    //    (a) A sharp solute J-doublet (`J = 8.0 Hz` = `0.016 ppm` at 500 MHz, equal-height twin lines at
    //        `7.242` and `7.258 ppm`, no CHCl3 singlet) MUST be rejected by `auto_calibrate_solvent_1d`!
    let doublet_lines = [
        SyntheticLine1D::lorentzian(7.242, 12.0, 1.1),
        SyntheticLine1D::lorentzian(7.258, 12.0, 1.1),
        SyntheticLine1D::lorentzian(2.150, 18.0, 1.2),
    ];
    let mut fid_dbl = synthesize_fid_1d(
        "solute_doublet_near_cdcl3",
        Experiment1DKind::Proton1D,
        8192,
        6000.0,
        500.0,
        4.70,
        &doublet_lines,
    );
    add_complex_gaussian_noise(&mut fid_dbl, 5.0e-4, 4242);
    apply_apodization_1d(&mut fid_dbl, Apodization1D::Exponential { lb_hz: 0.2 }, 0.5);
    zero_fill_pow2_1d(&mut fid_dbl, 2);
    let mut spec_dbl = fft_1d(&fid_dbl);
    let first_before = spec_dbl.axis.first_ppm;
    let err_dbl = auto_calibrate_solvent_1d(&mut spec_dbl, "CDCl3", 0.25)
        .expect_err("sharp solute J-doublet near 7.26 ppm must NOT be mistaken for CHCl3 singlet");
    assert!(
        err_dbl.contains("multiplet")
            || err_dbl.contains("doublet")
            || err_dbl.contains("lineshape")
            || err_dbl.contains("companion"),
        "unexpected J-doublet rejection message: {err_dbl}"
    );
    assert!((spec_dbl.axis.first_ppm - first_before).abs() < 1e-12);

    //    (b) A broad baseline hump (`FWHM = 55 Hz` at `7.24 ppm`) MUST also be rejected!
    let hump_lines = [
        SyntheticLine1D::lorentzian(7.24, 15.0, 55.0),
        SyntheticLine1D::lorentzian(1.25, 10.0, 1.2),
    ];
    let mut fid_hump = synthesize_fid_1d(
        "broad_hump_cdcl3",
        Experiment1DKind::Proton1D,
        8192,
        6000.0,
        500.0,
        4.70,
        &hump_lines,
    );
    add_complex_gaussian_noise(&mut fid_hump, 5.0e-4, 5353);
    let mut spec_hump = fft_1d(&fid_hump);
    assert!(
        auto_calibrate_solvent_1d(&mut spec_hump, "CDCl3", 0.25).is_err(),
        "broad 55 Hz hump at 7.24 ppm must NOT be accepted as residual CHCl3"
    );

    //    (c) Genuine residual CHCl3 singlet shifted by `-0.042 ppm` (`7.218 ppm` -> `7.260 ppm`)
    //        calibrates identically across `process_loaded_1d`, `prepare_loaded_1d_for_studio`,
    //        and `load_processed_spectrum_1d`, recording `applied_reference_shift_ppm ≈ +0.042 ppm`.
    let chcl3_lines = [
        SyntheticLine1D::lorentzian(7.218, 10.0, 1.2),
        SyntheticLine1D::lorentzian(3.620, 25.0, 1.3),
    ];
    let mut fid_ok = synthesize_fid_1d(
        "valid_cdcl3_fid",
        Experiment1DKind::Proton1D,
        4096,
        6000.0,
        500.0,
        4.70,
        &chcl3_lines,
    );
    fid_ok.solvent = Some("CDCl3".to_string());
    add_complex_gaussian_noise(&mut fid_ok, 5.0e-4, 6464);

    let tmp_fid_json = std::env::temp_dir().join("v0150_p1_3_shared_cdcl3_fid.json");
    std::fs::write(&tmp_fid_json, serde_json::to_string(&fid_ok).unwrap()).unwrap();
    let sp_from_path =
        load_processed_spectrum_1d(&tmp_fid_json, None).expect("path pipeline on FID");
    let _ = std::fs::remove_file(&tmp_fid_json);
    let sp_from_studio = prepare_loaded_1d_for_studio(Loaded1DData::TimeDomain(fid_ok.clone()))
        .expect("studio pipeline on FID");
    let proc_direct = process_loaded_1d(
        Loaded1DData::TimeDomain(fid_ok),
        &Process1DOptions::interactive_defaults(),
    )
    .expect("direct shared pipeline on FID");

    let shift = sp_from_path
        .applied_reference_shift_ppm
        .expect("applied_reference_shift_ppm recorded");
    assert!(
        (shift - 0.042).abs() < 0.004,
        "expected ~+0.042 ppm calibration shift onto 7.26 ppm CHCl3, got {shift:+.5}"
    );
    assert_eq!(
        sp_from_path.applied_reference_shift_ppm,
        sp_from_studio.applied_reference_shift_ppm
    );
    assert_eq!(
        sp_from_path.applied_reference_shift_ppm,
        proc_direct.applied_reference_shift_ppm
    );
    assert!((sp_from_path.axis.first_ppm - sp_from_studio.axis.first_ppm).abs() < 1e-12);

    // 3. Multi-experiment virtual drop error isolation (`v0150` P2):
    //    `study/1` (valid Bruker pdata) + `study/2` (corrupt/truncated `1r`) + `study/3` (valid Bruker pdata)
    //    must recover BOTH valid siblings (`1` and `3`) and surface the corrupt `study/2` error as a warning!
    let procs_txt = b"##TITLE= pdata\n##$SI= 4\n##$OFFSET= 10.0\n##$SW_p= 4000.0\n##$SF= 400.0\n##$NC_proc= 0\n##$BYTORDP= 0\n##$DTYPP= 0\n##END=\n".to_vec();
    let mut good_1r = Vec::with_capacity(16);
    for v in [10i32, 500, 20, 5] {
        good_1r.extend_from_slice(&v.to_le_bytes());
    }
    let bad_trunc_1r = vec![0u8; 8]; // truncated: 8 bytes instead of 16 for SI=4!

    let entries: Vec<(&str, &[u8])> = vec![
        ("study/1/pdata/1/procs", procs_txt.as_slice()),
        ("study/1/pdata/1/1r", good_1r.as_slice()),
        ("study/2/pdata/1/procs", procs_txt.as_slice()),
        ("study/2/pdata/1/1r", bad_trunc_1r.as_slice()),
        ("study/3/pdata/1/procs", procs_txt.as_slice()),
        ("study/3/pdata/1/1r", good_1r.as_slice()),
    ];
    let outcome = load_1d_from_virtual_entries_isolated(&entries, None, true)
        .expect("isolated multi-expno virtual load");
    assert_eq!(
        outcome.loaded.len(),
        2,
        "expected 2 valid sibling expnos to survive corrupt expno 2, got {}",
        outcome.loaded.len()
    );
    assert_eq!(outcome.dataset_errors.len(), 1);
    assert!(outcome.dataset_errors[0].contains("study/2"));

    let surviving = load_1d_from_virtual_entries(&entries, None, true)
        .expect("multi-expno drop with 2 valid siblings must succeed rather than aborting");
    assert_eq!(surviving.len(), 2);
    let sp0 = prepare_loaded_1d_for_studio(surviving.into_iter().next().unwrap()).unwrap();
    assert!(
        sp0.processing_warnings
            .iter()
            .any(|w| w.contains("study/2")),
        "sibling load error must propagate into processing_warnings, got {:?}",
        sp0.processing_warnings
    );
}
