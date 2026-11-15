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
