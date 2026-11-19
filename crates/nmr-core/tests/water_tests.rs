use nmr_core::nmr1d::{fft_1d, synthesize_fid_1d, Experiment1DKind, SyntheticLine1D};
use nmr_core::water::{detect_water_ppm, suppress_water_spectrum_1d, WaterSuppressionConfig};

#[test]
fn test_b6_dry_d2o_beta_glucose_4p64_ppm_preserved_without_dominant_water() {
    // Realistic 600 MHz 1D 1H beta-glucose / D2O solute spectrum (no dominant residual water):
    // - beta-glucose anomeric H1 at 4.64 ppm (inside 4.60..=5.00 ppm water search band!)
    // - alpha-glucose anomeric H1 at 5.23 ppm
    // - ring protons at 3.24, 3.40, 3.48, 3.72, 3.89 ppm
    let lines_glucose = [
        SyntheticLine1D::lorentzian(4.640, 1.0, 2.5),
        SyntheticLine1D::lorentzian(5.230, 0.6, 2.5),
        SyntheticLine1D::lorentzian(3.890, 1.0, 3.0),
        SyntheticLine1D::lorentzian(3.720, 1.2, 3.0),
        SyntheticLine1D::lorentzian(3.480, 1.5, 3.0),
        SyntheticLine1D::lorentzian(3.400, 1.1, 3.0),
        SyntheticLine1D::lorentzian(3.240, 1.0, 2.5),
    ];
    let mut td = synthesize_fid_1d(
        "beta_glucose_d2o",
        Experiment1DKind::Proton1D,
        4096,
        7200.0,
        600.0,
        4.70,
        &lines_glucose,
    );
    td.data.resize(8192, [0.0, 0.0]);
    let spec_ref = fft_1d(&td);

    let idx_beta = spec_ref.axis.ppm_to_index(4.640).round() as usize;
    let idx_base = spec_ref.axis.ppm_to_index(1.000).round() as usize;
    let ref_beta_amp = spec_ref.real[idx_beta] - spec_ref.real[idx_base];
    assert!(ref_beta_amp > 0.0);

    // Must preserve >= 99% of the 4.64 ppm beta-glucose anomeric proton both un-hinted
    // (dry / metabolomics sample) and with an explicit D2O solvent header when no dominant
    // water peak is present.
    for solvent_opt in [None, Some("D2O".to_string())] {
        let mut spec = spec_ref.clone();
        spec.imag = None; // real-only 1r
        spec.solvent = solvent_opt.clone();

        // `detect_water_ppm` must NOT lock onto the 4.64 ppm solute peak; it falls back to 4.70.
        let det = detect_water_ppm(&spec.axis, &spec.real, spec.solvent.as_deref());
        assert!(
            (det - 4.70).abs() < 1e-6,
            "dry/D2O beta-glucose spectrum (solvent={solvent_opt:?}) must not lock onto 4.64 ppm solute, got {det}"
        );

        let _ = suppress_water_spectrum_1d(&mut spec, &WaterSuppressionConfig::default());
        let post_beta_amp = spec.real[idx_beta] - spec.real[idx_base];
        let retention = post_beta_amp / ref_beta_amp;
        assert!(
            (0.99..=1.01).contains(&retention),
            "B6 regression: 4.64 ppm beta-glucose anomeric proton must retain >=99% amplitude \
             when no dominant water peak is present (solvent={solvent_opt:?}), got {:.4}%",
            retention * 100.0
        );
    }
}

#[test]
fn test_b6_genuine_dominant_water_still_suppressed_with_beta_glucose_present() {
    // Same beta-glucose solute peaks PLUS a genuine dominant 200x residual H2O peak at 4.720 ppm
    // (0.080 ppm away from beta-glucose H1 at 4.640 ppm): water must still be detected and
    // suppressed >1000x while the 4.640 ppm solute peak retains >= 95%.
    let lines_ref = [
        SyntheticLine1D::lorentzian(4.640, 1.0, 2.5),
        SyntheticLine1D::lorentzian(5.230, 0.6, 2.5),
        SyntheticLine1D::lorentzian(3.480, 1.5, 3.0),
    ];
    let mut td_ref = synthesize_fid_1d(
        "glucose_ref",
        Experiment1DKind::Proton1D,
        4096,
        7200.0,
        600.0,
        4.70,
        &lines_ref,
    );
    td_ref.data.resize(8192, [0.0, 0.0]);
    let spec_ref = fft_1d(&td_ref);

    let lines_wet = [
        SyntheticLine1D {
            ppm: 4.720,
            amplitude: 200.0,
            fwhm_hz: 0.5,
            gauss_fwhm_hz: 6.0,
            phase_deg: 0.0,
        },
        SyntheticLine1D::lorentzian(4.640, 1.0, 2.5),
        SyntheticLine1D::lorentzian(5.230, 0.6, 2.5),
        SyntheticLine1D::lorentzian(3.480, 1.5, 3.0),
    ];
    let mut td_wet = synthesize_fid_1d(
        "glucose_wet",
        Experiment1DKind::Proton1D,
        4096,
        7200.0,
        600.0,
        4.70,
        &lines_wet,
    );
    td_wet.data.resize(8192, [0.0, 0.0]);
    let mut spec_wet = fft_1d(&td_wet);
    spec_wet.imag = None;

    let idx_w = spec_wet.axis.ppm_to_index(4.720).round() as usize;
    let idx_beta = spec_wet.axis.ppm_to_index(4.640).round() as usize;
    let idx_base = spec_wet.axis.ppm_to_index(1.000).round() as usize;

    let raw_w = spec_wet.real[idx_w - 6..=idx_w + 6]
        .iter()
        .fold(0.0_f64, |m, &v| m.max(v.abs()));
    let detected = detect_water_ppm(&spec_wet.axis, &spec_wet.real, None);
    assert!(
        (detected - 4.720).abs() < 0.01,
        "expected detected water near 4.720 ppm, got {detected}"
    );

    let used = suppress_water_spectrum_1d(&mut spec_wet, &WaterSuppressionConfig::default());
    assert!((used - 4.720).abs() < 0.01, "used={used}");

    let far_base = spec_wet.real[idx_base];
    let post_w_dev = spec_wet.real[idx_w - 4..=idx_w + 4]
        .iter()
        .fold(0.0_f64, |m, &v| m.max((v - far_base).abs()));
    assert!(
        post_w_dev < 1e-3 * raw_w,
        "dominant water must still be suppressed >1000x: raw={raw_w}, post_dev={post_w_dev}"
    );

    let ref_beta = spec_ref.real[idx_beta] - spec_ref.real[idx_base];
    let post_beta = spec_wet.real[idx_beta] - far_base;
    let retention = post_beta / ref_beta;
    assert!(
        (0.95..=1.05).contains(&retention),
        "4.64 ppm beta-glucose peak 0.08 ppm from water must retain >=95%, got {:.2}%",
        retention * 100.0
    );
}
