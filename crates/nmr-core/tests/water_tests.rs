use nmr_core::nmr1d::Spectrum1D;
use nmr_core::nmr1d::{fft_1d, synthesize_fid_1d, Experiment1DKind, SyntheticLine1D};
use nmr_core::spectrum::{AxisPpm, Spectrum2D, Spectrum3D};
use nmr_core::water::{
    detect_dominant_water, detect_water_ppm, residual_water_window, suppress_water_spectrum_1d,
    suppress_water_spectrum_2d, suppress_water_spectrum_3d, WaterSuppressionConfig,
    WaterSuppressionOutcome,
};

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

        let outcome = suppress_water_spectrum_1d(&mut spec, &WaterSuppressionConfig::default());
        assert!(
            !outcome.is_suppressed(),
            "no dominant water: outcome must be NotDetected, got {outcome:?}"
        );
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

    let used = suppress_water_spectrum_1d(&mut spec_wet, &WaterSuppressionConfig::default())
        .center_ppm()
        .expect("dominant 200x water must be suppressed");
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

// -------------------------------------------------------------------------------------------
// FIX3 21F:B6 — explicit outcome, solvent-aware windows, realistic detection, 2D/3D guard.
// -------------------------------------------------------------------------------------------

/// 600 MHz, 7200 Hz SW, 4096 TD zero-filled to 8192, carrier 4.70 ppm, real-only `1r`.
fn spectrum_1d(lines: &[SyntheticLine1D], solvent: Option<&str>) -> Spectrum1D {
    let mut td = synthesize_fid_1d(
        "b6",
        Experiment1DKind::Proton1D,
        4096,
        7200.0,
        600.0,
        4.70,
        lines,
    );
    td.data.resize(8192, [0.0, 0.0]);
    let mut spec = fft_1d(&td);
    spec.imag = None;
    spec.solvent = solvent.map(str::to_string);
    spec
}

/// Deterministic Gaussian noise (xorshift64* + Box–Muller).
fn add_noise(vals: &mut [f64], sd: f64, seed: u64) {
    let mut s = seed.max(1);
    let mut next = || {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        ((s.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    for v in vals.iter_mut() {
        let (u1, u2) = (next(), next());
        *v += sd * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
    }
}

fn glucose_lines() -> Vec<SyntheticLine1D> {
    vec![
        SyntheticLine1D::lorentzian(4.640, 1.0, 2.5),
        SyntheticLine1D::lorentzian(5.230, 0.6, 2.5),
        SyntheticLine1D::lorentzian(3.890, 1.0, 3.0),
        SyntheticLine1D::lorentzian(3.720, 1.2, 3.0),
        SyntheticLine1D::lorentzian(3.480, 1.5, 3.0),
        SyntheticLine1D::lorentzian(3.400, 1.1, 3.0),
        SyntheticLine1D::lorentzian(3.240, 1.0, 2.5),
    ]
}

/// Chloroform-d solute: residual CHCl3 7.26, OCH3 3.70, CH2 1.25, CH3 0.88.
fn cdcl3_solute_lines() -> Vec<SyntheticLine1D> {
    vec![
        SyntheticLine1D::lorentzian(7.26, 1.0, 1.5),
        SyntheticLine1D::lorentzian(3.70, 2.0, 1.5),
        SyntheticLine1D::lorentzian(1.25, 6.0, 3.0),
        SyntheticLine1D::lorentzian(0.88, 3.0, 3.0),
    ]
}

fn height_at(spec: &Spectrum1D, ppm: f64) -> f64 {
    let i = spec.axis.ppm_to_index(ppm).round() as usize;
    let base = spec.real[spec.axis.ppm_to_index(-0.80).round() as usize];
    spec.real[i.saturating_sub(2)..=i + 2]
        .iter()
        .fold(0.0_f64, |m, &v| m.max((v - base).abs()))
}

fn bits_equal(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

#[test]
fn test_b6_residual_water_window_table_matches_gottlieb_fulmer() {
    let cases = [
        ("CDCl3", 1.56),
        ("<CDCl3>", 1.56),
        ("DMSO", 3.33),
        ("DMSO-d6", 3.33),
        ("MeOD", 4.87),
        ("CD3OD", 4.87),
        ("acetone-d6", 2.84),
        ("Acetone", 2.84),
        ("CD3CN", 2.13),
        ("C6D6", 0.40),
        ("CD2Cl2", 1.52),
        ("THF-d8", 2.46),
        ("toluene-d8", 0.43),
    ];
    for (hint, ppm) in cases {
        let w = residual_water_window(Some(hint));
        assert!(!w.unknown_solvent && !w.aqueous_window, "{hint}: {w:?}");
        assert_eq!(w.literature_ppm, Some(ppm), "{hint}");
        assert!((w.lo_ppm - (ppm - 0.15)).abs() < 1e-12 && (w.hi_ppm - (ppm + 0.15)).abs() < 1e-12);
    }
    for hint in ["D2O", "H2O+D2O", "90% H2O+10% D2O", "H2O"] {
        let w = residual_water_window(Some(hint));
        assert!(w.aqueous_window && !w.unknown_solvent, "{hint}: {w:?}");
        assert_eq!((w.lo_ppm, w.hi_ppm), (4.55, 5.05));
    }
    for hint in [None, Some(""), Some("mystery-solvent")] {
        let w = residual_water_window(hint);
        assert!(w.unknown_solvent && w.aqueous_window, "{hint:?}: {w:?}");
        assert_eq!((w.lo_ppm, w.hi_ppm), (4.55, 5.05));
    }
}

#[test]
fn test_b6_not_detected_outcome_reports_window_and_solvent_and_leaves_data_untouched() {
    for (solvent, label) in [(None, "unknown"), (Some("D2O"), "D2O")] {
        let mut spec = spectrum_1d(&glucose_lines(), solvent);
        let before = spec.clone();
        let outcome = suppress_water_spectrum_1d(&mut spec, &WaterSuppressionConfig::default());
        let WaterSuppressionOutcome::NotDetected(nd) = &outcome else {
            panic!("glucose without water must be NotDetected, got {outcome:?}");
        };
        assert_eq!(nd.solvent_label, label);
        assert_eq!((nd.window_lo_ppm, nd.window_hi_ppm), (4.55, 5.05));
        assert!(outcome.center_ppm().is_none());
        assert!(nd
            .to_string()
            .starts_with("No dominant water peak detected in 4.55–5.05 ppm"));
        assert!(
            bits_equal(&spec.real, &before.real),
            "data must be bit-identical"
        );
        assert!(
            spec.imag.is_none(),
            "real-only input must stay real-only on NotDetected"
        );
    }
}

#[test]
fn test_b6_cdcl3_without_water_is_not_detected_noop() {
    let mut spec = spectrum_1d(&cdcl3_solute_lines(), Some("CDCl3"));
    let sd = 0.002 * height_at(&spec, 1.25);
    add_noise(&mut spec.real, sd, 11);
    let before = spec.real.clone();
    let outcome = suppress_water_spectrum_1d(&mut spec, &WaterSuppressionConfig::default());
    let WaterSuppressionOutcome::NotDetected(nd) = &outcome else {
        panic!("dry CDCl3 must be NotDetected, got {outcome:?}");
    };
    assert_eq!(nd.solvent_label, "CDCl3");
    assert!((nd.window_lo_ppm - 1.41).abs() < 1e-9 && (nd.window_hi_ppm - 1.71).abs() < 1e-9);
    assert!(bits_equal(&spec.real, &before));
}

#[test]
fn test_b6_cdcl3_water_at_1p56_is_suppressed_and_neighbours_kept() {
    let mut lines = cdcl3_solute_lines();
    lines.push(SyntheticLine1D::lorentzian(1.56, 40.0, 4.0)); // ~5x the CH2 at 1.25 ppm
    let reference = spectrum_1d(&cdcl3_solute_lines(), Some("CDCl3"));
    let mut spec = spectrum_1d(&lines, Some("CDCl3"));
    let raw_w = height_at(&spec, 1.56);
    let outcome = suppress_water_spectrum_1d(&mut spec, &WaterSuppressionConfig::default());
    let c = outcome
        .center_ppm()
        .expect("CDCl3 water at 1.56 must be suppressed");
    assert!((c - 1.56).abs() < 0.01, "center {c}");
    assert!(
        height_at(&spec, 1.56) < 0.02 * raw_w,
        "water must drop >50x"
    );
    // The 4.70 aqueous window must not be touched in CDCl3.
    for ppm in [7.26, 3.70, 1.25, 0.88] {
        let r = height_at(&spec, ppm) / height_at(&reference, ppm);
        assert!((0.97..=1.03).contains(&r), "{ppm} ppm retention {r:.4}");
    }
}

#[test]
fn test_b6_cdcl3_hint_preserves_dominant_in_band_4p80_singlet() {
    // A dominant solute singlet at 4.80 ppm (e.g. an OCH2/vinyl singlet), 15x the tallest
    // other peak: with a CDCl3 hint the aqueous window is not searched, so it must survive.
    let mut lines = cdcl3_solute_lines();
    lines.push(SyntheticLine1D::lorentzian(4.80, 60.0, 2.0));
    let mut spec = spectrum_1d(&lines, Some("CDCl3"));
    let before = spec.real.clone();
    let outcome = suppress_water_spectrum_1d(&mut spec, &WaterSuppressionConfig::default());
    assert!(!outcome.is_suppressed(), "got {outcome:?}");
    assert!(
        bits_equal(&spec.real, &before),
        "4.80 ppm singlet must be untouched"
    );

    // Manual override still reaches CDCl3 water at 1.56 without touching 4.80.
    lines.push(SyntheticLine1D::lorentzian(1.56, 8.0, 4.0));
    let mut spec = spectrum_1d(&lines, Some("CDCl3"));
    let h480 = height_at(&spec, 4.80);
    let raw_w = height_at(&spec, 1.56);
    let cfg = WaterSuppressionConfig {
        water_ppm: Some(1.56),
        ..WaterSuppressionConfig::default()
    };
    let outcome = suppress_water_spectrum_1d(&mut spec, &cfg);
    assert_eq!(
        outcome,
        WaterSuppressionOutcome::Suppressed {
            center_ppm: 1.56,
            auto_detected: false
        }
    );
    assert!(height_at(&spec, 1.56) < 0.05 * raw_w);
    assert!((height_at(&spec, 4.80) / h480 - 1.0).abs() < 1e-3);
}

#[test]
fn test_b6_presat_residual_4x_tallest_solute_is_detected() {
    // Realistic presaturation residual: HDO at 4.79 ppm only ~4x the tallest solute peak
    // (glucose H at 3.48 ppm: area 1.5 over 3 Hz), 8 Hz wide, with noise.
    let tallest_height_per_area = 1.5 / 3.0;
    let mut lines = glucose_lines();
    lines.push(SyntheticLine1D::lorentzian(
        4.79,
        4.0 * tallest_height_per_area * 8.0,
        8.0,
    ));
    for solvent in [None, Some("D2O"), Some("H2O+D2O")] {
        let mut spec = spectrum_1d(&lines, solvent);
        let sd = 0.002 * height_at(&spec, 3.48);
        add_noise(&mut spec.real, sd, 7);
        let ratio = height_at(&spec, 4.79) / height_at(&spec, 3.48);
        assert!((3.5..4.5).contains(&ratio), "fixture ratio {ratio}");
        let raw_w = height_at(&spec, 4.79);
        let outcome = suppress_water_spectrum_1d(&mut spec, &WaterSuppressionConfig::default());
        let c = outcome.center_ppm().unwrap_or_else(|| {
            panic!("presat 4x residual must be detected ({solvent:?}): {outcome:?}")
        });
        assert!((c - 4.79).abs() < 0.01, "center {c}");
        assert!(
            height_at(&spec, 4.79) < 0.05 * raw_w,
            "residual must drop >20x"
        );
    }
}

#[test]
fn test_b6_broad_radiation_damped_water_100_150_hz_is_detected() {
    for fwhm in [100.0, 120.0, 150.0] {
        let mut lines = glucose_lines();
        lines.push(SyntheticLine1D::lorentzian(4.79, 1000.0, fwhm));
        let mut spec = spectrum_1d(&lines, None);
        let sd = 0.002 * height_at(&spec, 3.48);
        add_noise(&mut spec.real, sd, 3);
        let raw_w = height_at(&spec, 4.79);
        let det = detect_dominant_water(&spec.axis, &spec.real, None);
        let c = det.unwrap_or_else(|e| panic!("broad {fwhm} Hz water must be detected: {e}"));
        assert!((c - 4.79).abs() < 0.02, "{fwhm} Hz center {c}");
        let outcome = suppress_water_spectrum_1d(&mut spec, &WaterSuppressionConfig::default());
        assert!(outcome.is_suppressed(), "{fwhm} Hz: {outcome:?}");
        let post = height_at(&spec, 4.79);
        eprintln!("broad water {fwhm} Hz: centre height {raw_w:.2} -> {post:.3}");
        // Detection is the B6 fix; suppression quality is limited by the fixed 35 Hz default
        // stopband, which removes only the core of a 100-150 Hz line (measured ~10x / 7x / 5x
        // centre reduction at 100 / 120 / 150 Hz). Assert a real reduction (>4x), not the
        // >1000x achieved on narrow water; widen `half_width_hz` for radiation-damped water.
        assert!(
            post < 0.25 * raw_w,
            "{fwhm} Hz centre must drop >4x: {raw_w} -> {post}"
        );
    }
}

fn hsqc_like_2d(with_noise: bool) -> Spectrum2D {
    let ax = AxisPpm::from_bounds("1H", 512, 11.0, 0.0, 600.0);
    let ay = AxisPpm::from_bounds("15N", 64, 135.0, 100.0, 60.8);
    let (nx, ny) = (ax.size, ay.size);
    let mut data = vec![0.0_f32; nx * ny];
    // Amide cross-peaks plus one solute cross-peak exactly at F2 = 4.70 ppm (no water ridge).
    for &(h, n) in &[(8.3, 120.0), (7.9, 115.0), (8.6, 125.0), (4.70, 118.0)] {
        for iy in 0..ny {
            for ix in 0..nx {
                let dx = (ax.index_to_ppm(ix as f64) - h) / 0.02;
                let dy = (ay.index_to_ppm(iy as f64) - n) / 0.5;
                data[iy * nx + ix] += (100.0 * (-0.5 * (dx * dx + dy * dy)).exp()) as f32;
            }
        }
    }
    if with_noise {
        let mut noise = vec![0.0_f64; nx * ny];
        add_noise(&mut noise, 0.5, 5);
        for (d, n) in data.iter_mut().zip(noise) {
            *d += n as f32;
        }
    }
    Spectrum2D {
        name: "hsqc_no_water".into(),
        axis_x: ax,
        axis_y: ay,
        data,
    }
}

#[test]
fn test_b6_2d_without_dominant_water_is_not_detected_noop() {
    for noise in [false, true] {
        let mut s = hsqc_like_2d(noise);
        let before = s.data.clone();
        let outcome = suppress_water_spectrum_2d(&mut s, &WaterSuppressionConfig::default());
        assert!(!outcome.is_suppressed(), "noise={noise}: {outcome:?}");
        assert!(
            s.data
                .iter()
                .zip(&before)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "noise={noise}: 4.70 ppm cross-peak must be untouched"
        );
    }
    // Adding a genuine water ridge (all rows) at 4.70 ppm flips the outcome to Suppressed.
    let mut s = hsqc_like_2d(true);
    let nx = s.axis_x.size;
    for iy in 0..s.axis_y.size {
        for ix in 0..nx {
            let dx = (s.axis_x.index_to_ppm(ix as f64) - 4.70) / 0.01;
            s.data[iy * nx + ix] += (2000.0 / (1.0 + dx * dx)) as f32;
        }
    }
    let outcome = suppress_water_spectrum_2d(&mut s, &WaterSuppressionConfig::default());
    let c = outcome
        .center_ppm()
        .expect("2D water ridge must be detected");
    assert!((c - 4.70).abs() < 0.03, "center {c}");
}

#[test]
fn test_b6_3d_without_dominant_water_is_not_detected_noop() {
    let ax = AxisPpm::from_bounds("1H", 256, 11.0, 0.0, 600.0);
    let ay = AxisPpm::from_bounds("15N", 16, 135.0, 100.0, 60.8);
    let az = AxisPpm::from_bounds("13C", 16, 70.0, 40.0, 150.9);
    let (nx, ny, nz) = (ax.size, ay.size, az.size);
    let mut data = vec![0.0_f32; nx * ny * nz];
    for &(h, n, c) in &[(8.3, 120.0, 56.0), (7.9, 115.0, 60.0), (4.70, 118.0, 52.0)] {
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let dx = (ax.index_to_ppm(ix as f64) - h) / 0.03;
                    let dy = (ay.index_to_ppm(iy as f64) - n) / 1.0;
                    let dz = (az.index_to_ppm(iz as f64) - c) / 1.0;
                    data[(iz * ny + iy) * nx + ix] +=
                        (100.0 * (-0.5 * (dx * dx + dy * dy + dz * dz)).exp()) as f32;
                }
            }
        }
    }
    let mut cube = Spectrum3D {
        name: "hnca_no_water".into(),
        exp: "HNCA".into(),
        axis_x: ax,
        axis_y: ay,
        axis_z: az,
        data,
    };
    let before = cube.data.clone();
    let outcome = suppress_water_spectrum_3d(&mut cube, &WaterSuppressionConfig::default());
    assert!(!outcome.is_suppressed(), "{outcome:?}");
    assert_eq!(cube.data, before);
}

/// Height of the line at `ppm` above a local baseline interpolated from `ppm ± 0.02` (12 Hz at
/// 600 MHz, ~10 HWHM of a 2.5 Hz line), i.e. the line itself without any water-wing pedestal.
fn local_height_at(spec: &Spectrum1D, ppm: f64) -> f64 {
    let at = |p: f64| spec.real[spec.axis.ppm_to_index(p).round() as usize];
    let base = 0.5 * (at(ppm + 0.02) + at(ppm - 0.02));
    let i = spec.axis.ppm_to_index(ppm).round() as usize;
    spec.real[i - 2..=i + 2]
        .iter()
        .fold(f64::MIN, |m, &v| m.max(v - base))
}

/// Selectivity limit (disclosed, not hidden): beta-glucose H1 at 4.64 ppm next to HDO at
/// 4.79 ppm (delta = 0.15 ppm = 90 Hz at 600 MHz) at 150x and 1500x HDO area (HDO 2.5 Hz
/// Lorentzian, absorptive and 90-degree dispersive).
///
/// The Marion–Ikura–Bax stopband (35 Hz default) removes the water core only; the HDO wings at
/// 90 Hz are left in place. The beta-H1 *line* is preserved (local-baseline retention asserted
/// within ±5%), but its height measured from the far baseline is biased by the remaining wing
/// (printed as "apparent"; see the FIX3 report for the numbers).
#[test]
fn test_b6_selectivity_limit_beta_h1_next_to_hdo_4p79() {
    let reference = spectrum_1d(&glucose_lines(), Some("D2O"));
    let ref_far = height_at(&reference, 4.64);
    let ref_local = local_height_at(&reference, 4.64);
    for phase_deg in [0.0, 90.0] {
        for scale in [150.0, 1500.0] {
            let mut lines = glucose_lines();
            lines.push(SyntheticLine1D {
                ppm: 4.79,
                amplitude: scale,
                fwhm_hz: 2.5,
                gauss_fwhm_hz: 0.0,
                phase_deg,
            });
            let mut spec = spectrum_1d(&lines, Some("D2O"));
            let outcome = suppress_water_spectrum_1d(&mut spec, &WaterSuppressionConfig::default());
            let c = outcome.center_ppm().expect("HDO must be detected");
            assert!((c - 4.79).abs() < 0.01, "center {c}");
            let local = local_height_at(&spec, 4.64) / ref_local;
            let i = spec.axis.ppm_to_index(4.64).round() as usize;
            let base = spec.real[spec.axis.ppm_to_index(-0.80).round() as usize];
            let apparent = (spec.real[i - 2..=i + 2]
                .iter()
                .fold(f64::MIN, |m, &v| m.max(v - base)))
                / ref_far;
            eprintln!(
                "selectivity: HDO {scale}x phase {phase_deg} deg -> beta-H1 4.64 local-baseline \
                 retention {:.1}%, apparent (far-baseline) {:.1}%",
                local * 100.0,
                apparent * 100.0
            );
            assert!(
                (0.95..=1.05).contains(&local),
                "beta-H1 line itself must be preserved: {:.1}%",
                local * 100.0
            );
        }
    }
}
