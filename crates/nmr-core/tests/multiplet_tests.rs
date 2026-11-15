//! Comprehensive unit and physics validation tests for `nmr_core::multiplet` (`T14.2`):
//! 1. 1D region integration with slope/bias baseline correction, reference normalization
//!    (`1.00 H`, `2.00 H`, `3.00 H`), and spectral noise RMS uncertainty propagation.
//! 2. Recursive Hoye multiplet `J`-coupling tree classification across `s`, `br s`, `d`, `t`,
//!    `q`, `quint`, `sext`, `sept`, `dd`, `ddd`, `dt`, `td`, `dq`, `qd`, `tt`, and `m`,
//!    including coalesced topologies and strong second-order roofing (`Delta nu / J = 3.0` & `2.5`)
//!    both analytically and via exact quantum-mechanical spin-1/2 Hamiltonian diagonalization.
//! 3. Publication journal NMR characterization formatting and lossless round-trip parsing across
//!    `ACS`, `JOC`, `RSC`, and `Nature` styles in Unicode (`\u{03B4}`, superscripts), ASCII, and LaTeX.
//! 4. NIST-traceable internal-standard `qNMR` mass-fraction purity (`< 0.1%` numerical error),
//!    molar concentration, GUM uncertainty propagation, replicate RSD, and `5 * T1` warning guard.
//! 5. `Octet`-compatible 2D chemical-structure <-> multiplet assignment tables, `SDF` 1-based &
//!    `SMILES` `[atom:map]` cross-indexing, and `RenderScene` cross-highlighting.

use nmr_core::multiplet::{
    analyze_multiplet_peaks, analyze_multiplet_region, auto_assign_1h_multiplets,
    calculate_molar_concentration_from_standard, calculate_qnmr_purity, calculate_qnmr_replicates,
    estimate_spectral_noise_rms, format_journal_nmr_report, integrate_regions,
    integrate_spectrum1d_regions, parse_journal_nmr_report, qnmr_certified_standards,
    simulate_exact_spin_half_multiplet, solve_ab_quartet, synthesize_multiplet_peaks,
    synthesize_multiplet_spectrum, BaselineMode, IntegralRegionSpec, JournalNmrReport,
    LeaningDirection, MultipletAssignmentTable, MultipletClass, MultipletConfig, MultipletSubPeak,
    NormalizationMode, PublicationStyle, QnmrAcquisitionParams, QnmrAnalyteSignal, QnmrAnalyteSpec,
    QnmrStandardSpec, ReportNucleus, ReportedSignal, StructureAtomMap, TextEncoding,
    VoigtPeakComponent,
};
use nmr_core::octet_analyt::jcamp_spec1d::{Spectrum1d, SpectrumKind};
use nmr_core::octet_mol::{parse_smiles, write_mol_v2000};
use nmr_core::octet_render::styles::RgbaColor;
use nmr_core::octet_render::RenderScene;
use nmr_core::rng::Rng;

type CouplingSpec = (usize, f64, Option<f64>);

fn make_descending_ppm_grid(ppm_high: f64, ppm_low: f64, n_points: usize) -> Vec<f64> {
    (0..n_points)
        .map(|i| ppm_high - (ppm_high - ppm_low) * (i as f64) / ((n_points - 1) as f64))
        .collect()
}

// ============================================================================
// 1. Region Integration, Slope/Bias Baseline Correction & Noise Uncertainty
// ============================================================================

#[test]
fn region_integration_recovers_exact_areas_under_tilted_linear_baseline_and_renormalizes() {
    let mhz = 500.0;
    let ppm = make_descending_ppm_grid(9.0, 0.5, 34_001); // 0.00025 ppm (~0.125 Hz) spacing

    // Synthesize 3 multiplets with exact areas 2.0 (Ar-H dd at 7.50), 3.0 (OCH3 s at 3.85), 1.0 (CH q at 2.20)
    let mut peaks = Vec::new();
    for mut p in synthesize_multiplet_peaks(7.50, mhz, &[(1, 8.4, None), (1, 1.8, None)], 1.0) {
        p.intensity *= 2.0;
        peaks.push(p);
    }
    for mut p in synthesize_multiplet_peaks(3.85, mhz, &[], 1.0) {
        p.intensity *= 3.0;
        peaks.push(p);
    }
    for mut p in synthesize_multiplet_peaks(2.20, mhz, &[(3, 7.0, None)], 1.0) {
        p.intensity *= 1.0;
        peaks.push(p);
    }

    let pure_y = synthesize_multiplet_spectrum(&ppm, &peaks, mhz, 1.0, 0.25);

    // Superimpose a substantial global linear baseline tilt `B(delta) = 42.0 - 6.5 * delta`
    let tilted_y: Vec<f64> = ppm
        .iter()
        .zip(pure_y.iter())
        .map(|(&x, &y)| y + (42.0 - 6.5 * x))
        .collect();

    let specs = vec![
        IntegralRegionSpec::new(7.85, 7.15).with_label("Ar-H"),
        IntegralRegionSpec::new(4.15, 3.55).with_label("OCH3"),
        IntegralRegionSpec::new(2.55, 1.85).with_label("CH"),
    ];

    // AutoLinearEdges baseline correction + normalize CH (region 2) = 1.00 H
    let mut report = integrate_regions(
        &ppm,
        &tilted_y,
        &specs,
        NormalizationMode::ReferenceRegion {
            region_index: 2,
            target_protons: 1.0,
        },
        Some(0.01),
        None,
    )
    .expect("integration should succeed");

    assert_eq!(report.regions.len(), 3);
    assert!((report.regions[0].norm_integral - 2.0).abs() < 0.015);
    assert!((report.regions[1].norm_integral - 3.0).abs() < 0.015);
    assert!((report.regions[2].norm_integral - 1.0).abs() < 1e-12);
    assert_eq!(report.regions[0].nearest_integer_h, 2);
    assert_eq!(report.regions[1].nearest_integer_h, 3);
    assert_eq!(report.regions[2].nearest_integer_h, 1);

    // Cumulative integral trace starts at 0.0 downfield and ends at `norm_integral` upfield
    for r in &report.regions {
        assert!((r.cumulative_integral[0]).abs() < 1e-12);
        let end_val = *r.cumulative_integral.last().unwrap();
        assert!((end_val - r.norm_integral).abs() < 1e-9);
    }

    // Interactive re-normalization: set Ar-H = 2.00 H and OCH3 = 3.00 H
    report
        .renormalize(NormalizationMode::ReferenceRegion {
            region_index: 0,
            target_protons: 2.0,
        })
        .unwrap();
    assert!((report.regions[0].norm_integral - 2.0).abs() < 1e-12);
    assert!((report.regions[1].norm_integral - 3.0).abs() < 0.015);

    report
        .renormalize(NormalizationMode::ReferenceRegion {
            region_index: 1,
            target_protons: 3.0,
        })
        .unwrap();
    assert!((report.regions[1].norm_integral - 3.0).abs() < 1e-12);
    assert!((report.regions[2].norm_integral - 1.0).abs() < 0.01);

    // Explicit bias/slope subtraction recovers exact area to machine/Lorentzian-wing precision
    let lo_1 = 3.55_f64;
    let explicit_bias_at_lo = 42.0 - 6.5 * lo_1;
    let explicit_spec = vec![
        IntegralRegionSpec::new(4.15, 3.55).with_explicit_bias_slope(explicit_bias_at_lo, -6.5)
    ];
    let exact_rep = integrate_regions(
        &ppm,
        &tilted_y,
        &explicit_spec,
        NormalizationMode::None,
        Some(0.0),
        None,
    )
    .unwrap();
    let pure_rep = integrate_regions(
        &ppm,
        &pure_y,
        &[IntegralRegionSpec::new(4.15, 3.55).with_baseline(BaselineMode::None)],
        NormalizationMode::None,
        Some(0.0),
        None,
    )
    .unwrap();
    assert!(
        (exact_rep.regions[0].abs_integral - pure_rep.regions[0].abs_integral).abs() < 1e-11,
        "explicit bias/slope should cancel exact linear baseline to < 1e-11"
    );
}

#[test]
fn region_integration_noise_uncertainty_matches_monte_carlo_and_spectrum1d_wrapper() {
    let ppm = make_descending_ppm_grid(5.0, 1.0, 2_001); // dx = 0.002 ppm
    let peaks = synthesize_multiplet_peaks(3.00, 400.0, &[(2, 7.2, None)], 1.2);
    let clean_y = synthesize_multiplet_spectrum(&ppm, &peaks, 400.0, 1.2, 0.5);
    let true_sigma = 0.35;

    let spec_none = vec![IntegralRegionSpec::new(3.35, 2.65).with_baseline(BaselineMode::None)];
    let spec_auto = vec![IntegralRegionSpec::new(3.35, 2.65)];

    let rep_none = integrate_regions(
        &ppm,
        &clean_y,
        &spec_none,
        NormalizationMode::None,
        Some(true_sigma),
        Some(&[VoigtPeakComponent {
            center_ppm: 3.00,
            fwhm_hz: 1.2,
            area: 1.0,
            area_sigma: 0.005,
        }]),
    )
    .unwrap();
    assert_eq!(rep_none.regions[0].deconvolved_area, Some(1.0));
    assert_eq!(rep_none.regions[0].deconvolved_sigma, Some(0.005));

    let rep_auto = integrate_regions(
        &ppm,
        &clean_y,
        &spec_auto,
        NormalizationMode::None,
        Some(true_sigma),
        None,
    )
    .unwrap();

    // Monte Carlo check of analytical uncertainty propagation (`400` independent noisy realizations)
    let mut rng = Rng::new(20261114);
    let n_trials = 400;
    let mut vals_none = Vec::with_capacity(n_trials);
    let mut vals_auto = Vec::with_capacity(n_trials);
    let mut last_noisy = Vec::new();

    for _ in 0..n_trials {
        let noisy: Vec<f64> = clean_y
            .iter()
            .map(|&y| y + rng.normal(0.0, true_sigma))
            .collect();
        let r0 = integrate_regions(
            &ppm,
            &noisy,
            &spec_none,
            NormalizationMode::None,
            Some(true_sigma),
            None,
        )
        .unwrap();
        let r1 = integrate_regions(
            &ppm,
            &noisy,
            &spec_auto,
            NormalizationMode::None,
            Some(true_sigma),
            None,
        )
        .unwrap();
        vals_none.push(r0.regions[0].abs_integral);
        vals_auto.push(r1.regions[0].abs_integral);
        last_noisy = noisy;
    }

    let emp_sd = |xs: &[f64]| -> f64 {
        let mean = xs.iter().sum::<f64>() / (xs.len() as f64);
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / ((xs.len() - 1) as f64);
        var.sqrt()
    };
    let mc_sd_none = emp_sd(&vals_none);
    let mc_sd_auto = emp_sd(&vals_auto);

    let rel_err_none = (rep_none.regions[0].abs_sigma - mc_sd_none).abs() / mc_sd_none;
    let rel_err_auto = (rep_auto.regions[0].abs_sigma - mc_sd_auto).abs() / mc_sd_auto;
    assert!(
        rel_err_none < 0.10,
        "analytical quadrature sigma ({:.6}) vs MC ({:.6}) differed by {:.2}%",
        rep_none.regions[0].abs_sigma,
        mc_sd_none,
        100.0 * rel_err_none
    );
    assert!(
        rel_err_auto < 0.10,
        "analytical edge-baseline quadrature sigma ({:.6}) vs MC ({:.6}) differed by {:.2}%",
        rep_auto.regions[0].abs_sigma,
        mc_sd_auto,
        100.0 * rel_err_auto
    );

    // Check automatic baseline noise RMS estimator and `Spectrum1d` wrapper
    let est_rms = estimate_spectral_noise_rms(&ppm, &last_noisy, Some((4.8, 4.0))).unwrap();
    assert!((est_rms - true_sigma).abs() / true_sigma < 0.12);

    let spec1d = Spectrum1d {
        title: "Synthetic 1H".to_string(),
        kind: SpectrumKind::Nmr,
        nucleus: Some("1H".to_string()),
        mhz: Some(400.0),
        solvent: Some("CDCl3".to_string()),
        x_unit: "PPM".to_string(),
        y_unit: "ARBITRARY".to_string(),
        first_x: ppm[0],
        last_x: *ppm.last().unwrap(),
        y: clean_y,
        xs: None,
        sticks: false,
        peaks: None,
        integral: None,
    };
    let wrapped = integrate_spectrum1d_regions(
        &spec1d,
        &spec_auto,
        NormalizationMode::TotalProtons { total_protons: 2.0 },
        Some(0.01),
    )
    .unwrap();
    assert!((wrapped.regions[0].norm_integral - 2.0).abs() < 1e-12);
}

// ============================================================================
// 2. Recursive Multiplet Tree Classification & Strong Roofing (`Delta nu / J = 3`)
// ============================================================================

#[test]
fn first_order_multiplet_tree_classifies_all_16_canonical_patterns_and_coalesced_cases() {
    let cfg = MultipletConfig::default();
    let mhz = 500.0;

    // Sharp singlet `s` vs broad singlet `br s` on continuous 1D spectra
    let ppm_s = make_descending_ppm_grid(4.2, 3.4, 4_001);
    let sharp_y = synthesize_multiplet_spectrum(
        &ppm_s,
        &[MultipletSubPeak::new(3.80, 1.0).with_fwhm_hz(1.1)],
        mhz,
        1.1,
        0.0,
    );
    let broad_y = synthesize_multiplet_spectrum(
        &ppm_s,
        &[MultipletSubPeak::new(3.80, 1.0).with_fwhm_hz(9.5)],
        mhz,
        9.5,
        0.0,
    );
    let res_s = analyze_multiplet_region(&ppm_s, &sharp_y, 4.1, 3.5, mhz, &cfg).unwrap();
    let res_brs = analyze_multiplet_region(&ppm_s, &broad_y, 4.1, 3.5, mhz, &cfg).unwrap();
    assert_eq!(res_s.class, MultipletClass::Singlet);
    assert_eq!(res_s.label, "s");
    assert_eq!(res_brs.class, MultipletClass::BroadSinglet);
    assert_eq!(res_brs.label, "br s");

    // 13 structured splitting patterns evaluated directly from continuous Lorentzian/Voigt 1D traces
    let cases: Vec<(MultipletClass, Vec<CouplingSpec>, f64)> = vec![
        (MultipletClass::Doublet, vec![(1, 7.6, None)], 400.0),
        (MultipletClass::Triplet, vec![(2, 7.2, None)], 500.0),
        (MultipletClass::Quartet, vec![(3, 7.1, None)], 500.0),
        (MultipletClass::Quintet, vec![(4, 6.8, None)], 500.0),
        (MultipletClass::Sextet, vec![(5, 6.9, None)], 600.0),
        (MultipletClass::Septet, vec![(6, 6.8, None)], 600.0),
        (
            MultipletClass::DoubletOfDoublets,
            vec![(1, 8.6, None), (1, 2.1, None)],
            500.0,
        ),
        // Equally spaced 4-line `dd` (`J_1 = 2 * J_2`: 8.0 & 4.0 Hz, `1:1:1:1` != `q` `1:3:3:1`)
        (
            MultipletClass::DoubletOfDoublets,
            vec![(1, 8.0, None), (1, 4.0, None)],
            500.0,
        ),
        // Standard 8-line `ddd`
        (
            MultipletClass::DoubletOfDoubletOfDoublets,
            vec![(1, 11.4, None), (1, 7.2, None), (1, 2.4, None)],
            500.0,
        ),
        // Classic 7-line coalesced `ddd` (`J_1 == J_2 + J_3`: 10.0 = 6.2 + 3.8 Hz!)
        (
            MultipletClass::DoubletOfDoubletOfDoublets,
            vec![(1, 10.0, None), (1, 6.2, None), (1, 3.8, None)],
            500.0,
        ),
        // `dt` (`J_d > J_t`) including coalesced 5-line `J_d = 2 * J_t`
        (
            MultipletClass::DoubletOfTriplets,
            vec![(1, 9.2, None), (2, 3.1, None)],
            500.0,
        ),
        (
            MultipletClass::DoubletOfTriplets,
            vec![(1, 8.0, None), (2, 4.0, None)],
            500.0,
        ),
        // `td` (`J_t > J_d`)
        (
            MultipletClass::TripletOfDoublets,
            vec![(2, 8.4, None), (1, 2.3, None)],
            500.0,
        ),
        // `dq`, `qd`, `tt`
        (
            MultipletClass::DoubletOfQuartets,
            vec![(1, 11.5, None), (3, 6.7, None)],
            600.0,
        ),
        (
            MultipletClass::QuartetOfDoublets,
            vec![(3, 7.1, None), (1, 2.0, None)],
            600.0,
        ),
        (
            MultipletClass::TripletOfTriplets,
            vec![(2, 11.2, None), (2, 3.6, None)],
            600.0,
        ),
    ];

    for (expected_cls, couplings_spec, spec_mhz) in &cases {
        let center_ppm = 5.25;
        let pks = synthesize_multiplet_peaks(center_ppm, *spec_mhz, couplings_spec, 0.85);
        let grid = make_descending_ppm_grid(5.55, 4.95, 12_001);
        let trace = synthesize_multiplet_spectrum(&grid, &pks, *spec_mhz, 0.85, 0.2);

        let res = analyze_multiplet_region(&grid, &trace, 5.50, 5.00, *spec_mhz, &cfg)
            .expect("multiplet region analysis should succeed");

        assert_eq!(
            res.class,
            *expected_cls,
            "expected {} for {:?}, got {} (Js = {:?})",
            expected_cls,
            couplings_spec,
            res.class,
            res.j_values_hz()
        );
        assert!((res.center_ppm - center_ppm).abs() < 0.0005);
        assert!(!res.leaning.is_roofed);

        let got_js = res.j_values_hz();
        assert_eq!(got_js.len(), couplings_spec.len());
        for (got_j, &(_, true_j, _)) in got_js.iter().zip(couplings_spec.iter()) {
            assert!(
                (got_j - true_j).abs() < 0.05,
                "class {}: expected J = {:.2} Hz, got {:.4} Hz",
                expected_cls,
                true_j,
                got_j
            );
        }
    }

    // Complex non-first-order irregular cluster falls back cleanly to `m`
    let irregular_peaks = vec![
        MultipletSubPeak::new(7.341, 0.9),
        MultipletSubPeak::new(7.333, 2.4),
        MultipletSubPeak::new(7.319, 1.7),
        MultipletSubPeak::new(7.311, 3.1),
        MultipletSubPeak::new(7.294, 1.2),
    ];
    let fallback = analyze_multiplet_peaks(&irregular_peaks, 500.0, &cfg).unwrap();
    assert_eq!(fallback.class, MultipletClass::Multiplet);
    assert_eq!(fallback.label, "m");
    assert!(fallback.couplings.is_empty());
}

#[test]
fn strong_roofing_delta_nu_over_j_3_recovers_classes_and_couplings_analytical_and_exact_qm() {
    let cfg = MultipletConfig::default();
    let mhz = 500.0;

    // Part A: Analytical roofed multiplets at strong second-order leaning `Delta nu / J = +-3.0` and `2.5`
    // across `d`, `t`, `q`, `dd`, `ddd`, `dt`, `td` rendered onto continuous 1D spectra!
    let roofed_cases: Vec<(MultipletClass, Vec<CouplingSpec>, LeaningDirection)> = vec![
        (
            MultipletClass::Doublet,
            vec![(1, 8.8, Some(3.0))],
            LeaningDirection::Downfield,
        ),
        (
            MultipletClass::Doublet,
            vec![(1, 8.8, Some(-2.5))],
            LeaningDirection::Upfield,
        ),
        (
            MultipletClass::Triplet,
            vec![(2, 7.4, Some(3.0))],
            LeaningDirection::Downfield,
        ),
        (
            MultipletClass::Quartet,
            vec![(3, 7.0, Some(-3.0))],
            LeaningDirection::Upfield,
        ),
        (
            MultipletClass::DoubletOfDoublets,
            vec![(1, 9.2, Some(3.0)), (1, 2.4, None)],
            LeaningDirection::Downfield,
        ),
        (
            MultipletClass::DoubletOfDoublets,
            vec![(1, 8.5, None), (1, 3.2, Some(-3.0))],
            LeaningDirection::Upfield,
        ),
        (
            MultipletClass::DoubletOfDoubletOfDoublets,
            vec![(1, 11.6, Some(3.0)), (1, 7.0, Some(-4.0)), (1, 2.2, None)],
            LeaningDirection::Downfield,
        ),
        (
            MultipletClass::DoubletOfTriplets,
            vec![(1, 9.6, Some(-3.0)), (2, 3.4, None)],
            LeaningDirection::Upfield,
        ),
        (
            MultipletClass::TripletOfDoublets,
            vec![(2, 8.2, Some(3.0)), (1, 2.1, None)],
            LeaningDirection::Downfield,
        ),
    ];

    let grid = make_descending_ppm_grid(6.60, 5.90, 14_001);
    for (expected_cls, spec, expected_dir) in &roofed_cases {
        let pks = synthesize_multiplet_peaks(6.25, mhz, spec, 0.85);
        let trace = synthesize_multiplet_spectrum(&grid, &pks, mhz, 0.85, 0.15);
        let res = analyze_multiplet_region(&grid, &trace, 6.55, 5.95, mhz, &cfg).unwrap();

        assert_eq!(
            res.class, *expected_cls,
            "roofed {:?} misclassified as {}",
            spec, res.class
        );
        assert!(
            res.leaning.is_roofed,
            "expected is_roofed=true for {:?}",
            spec
        );
        assert_eq!(res.leaning.dominant_direction, *expected_dir);
        let est_ratio = res
            .leaning
            .estimated_delta_nu_over_j
            .expect("should report estimated Delta nu / J");
        assert!(
            (2.2..=3.4).contains(&est_ratio),
            "expected Delta nu / J near 2.5..3.0, got {est_ratio:.3}"
        );

        // Tree center is invariant to intensity leaning (whereas raw centroid is pulled toward partner!)
        assert!((res.center_ppm - 6.25).abs() < 0.0005);

        for (got_j, &(_, true_j, _)) in res.j_values_hz().iter().zip(spec.iter()) {
            assert!(
                (got_j - true_j).abs() < 0.06,
                "roofed {}: expected J = {true_j}, got {got_j}",
                expected_cls
            );
        }
    }

    // Part B: Exact Quantum-Mechanical Spin-1/2 Secular Hamiltonian Diagonalization at `Delta nu / J = 3.0`!
    // 1. Exact 2-spin AB system (`delta_B = 7.000 ppm`, `delta_A = 7.054 ppm` -> `Delta nu = 27.0 Hz`, `J_AB = 9.0 Hz`, `Delta nu / J = 3.0`)
    let shifts_ab = [7.054, 7.000];
    let j_ab = vec![vec![0.0, 9.0], vec![9.0, 0.0]];
    let qm_a = simulate_exact_spin_half_multiplet(&shifts_ab, &j_ab, 0, mhz, 0.01).unwrap();
    let qm_b = simulate_exact_spin_half_multiplet(&shifts_ab, &j_ab, 1, mhz, 0.01).unwrap();
    assert_eq!(qm_a.len(), 2);
    assert_eq!(qm_b.len(), 2);

    let ana_a = analyze_multiplet_peaks(&qm_a, mhz, &cfg).unwrap();
    assert_eq!(ana_a.class, MultipletClass::Doublet);
    assert!((ana_a.j_values_hz()[0] - 9.0).abs() < 1e-6);
    assert_eq!(ana_a.leaning.dominant_direction, LeaningDirection::Upfield); // leans toward spin B at 7.000 ppm!
                                                                             // Exact AB Zeeman shift correction from single-doublet leaning recovers `7.054000 ppm`!
    assert!(
        (ana_a.ab_zeeman_ppm - 7.054).abs() < 1e-5,
        "AB Zeeman-corrected shift {:.6} should match exact 7.054000 ppm (uncorrected midpoint = {:.6})",
        ana_a.ab_zeeman_ppm,
        ana_a.center_ppm
    );

    // Hoye & Zhao (2002) closed-form 4-line AB solver recovers both shifts and J to machine precision
    let four_lines = [qm_a[0].ppm, qm_a[1].ppm, qm_b[0].ppm, qm_b[1].ppm];
    let ab_sol = solve_ab_quartet(four_lines, Some(mhz)).unwrap();
    assert!((ab_sol.j_ab_hz - 9.0).abs() < 1e-9);
    assert!((ab_sol.delta_nu_over_j - 3.0).abs() < 1e-9);
    assert!((ab_sol.delta_a_ppm.unwrap() - 7.054).abs() < 1e-9);
    assert!((ab_sol.delta_b_ppm.unwrap() - 7.000).abs() < 1e-9);

    // 2. Exact QM 3-spin ABX (`dd` with `Delta nu_AB / J_AB = 3.0`, `J_AB = 9.0 Hz`, `J_AX = 2.6 Hz`)
    // By Pople, Schneider & Bernstein (1959) exact ABX theory, the two AB sub-quartets split by
    // `0.5 * (J_AX + J_BX) + (C_+ - C_-) = 1.3 + 0.5 * (sqrt(28.3^2 + 81) - sqrt(25.7^2 + 81)) = 2.5332 Hz`.
    let shifts_abx = [7.254, 7.200, 3.500];
    let j_abx = vec![
        vec![0.0, 9.0, 2.6],
        vec![9.0, 0.0, 0.0],
        vec![2.6, 0.0, 0.0],
    ];
    let qm_dd = simulate_exact_spin_half_multiplet(&shifts_abx, &j_abx, 0, mhz, 0.01).unwrap();
    let ana_dd = analyze_multiplet_peaks(&qm_dd, mhz, &cfg).unwrap();
    assert_eq!(ana_dd.class, MultipletClass::DoubletOfDoublets);
    let exact_abx_j2 =
        1.3 + 0.5 * ((28.3_f64.powi(2) + 81.0).sqrt() - (25.7_f64.powi(2) + 81.0).sqrt());
    assert!((ana_dd.j_values_hz()[0] - 9.0).abs() < 1e-4);
    assert!((ana_dd.j_values_hz()[1] - exact_abx_j2).abs() < 1e-3);
    assert!((ana_dd.j_values_hz()[1] - 2.6).abs() < 0.20);

    // 3. Exact QM 4-spin ABXY (`ddd` with `Delta nu_AB / J_AB = 3.0`, `J = 11.0, 6.4, 2.2 Hz`)
    let shifts_abxy = [7.466, 7.400, 4.100, 2.100];
    let j_abxy = vec![
        vec![0.0, 11.0, 6.4, 2.2],
        vec![11.0, 0.0, 0.0, 0.0],
        vec![6.4, 0.0, 0.0, 0.0],
        vec![2.2, 0.0, 0.0, 0.0],
    ];
    let qm_ddd = simulate_exact_spin_half_multiplet(&shifts_abxy, &j_abxy, 0, mhz, 0.01).unwrap();
    let ana_ddd = analyze_multiplet_peaks(&qm_ddd, mhz, &cfg).unwrap();
    assert_eq!(ana_ddd.class, MultipletClass::DoubletOfDoubletOfDoublets);
    assert!((ana_ddd.j_values_hz()[0] - 11.0).abs() < 0.05);
    assert!((ana_ddd.j_values_hz()[1] - 6.4).abs() < 0.20);
    assert!((ana_ddd.j_values_hz()[2] - 2.2).abs() < 0.10);

    // 4. Exact QM 4-spin ABX2 (`dt` with `J_AB = 10.0 Hz` at `Delta nu / J = 3.0` & `J_AX = 3.2 Hz`;
    //    and `td` on A2BX where `J_AX = 9.4 Hz` (`Delta nu_AX >> J`) & `J_AB = 3.4 Hz` at `Delta nu_AB / J_AB = 3.0`)
    let shifts_dt = [6.860, 6.800, 2.500, 2.500];
    let j_dt = vec![
        vec![0.0, 10.0, 3.2, 3.2],
        vec![10.0, 0.0, 0.0, 0.0],
        vec![3.2, 0.0, 0.0, 0.0],
        vec![3.2, 0.0, 0.0, 0.0],
    ];
    let qm_dt = simulate_exact_spin_half_multiplet(&shifts_dt, &j_dt, 0, mhz, 0.01).unwrap();
    let ana_dt = analyze_multiplet_peaks(&qm_dt, mhz, &cfg).unwrap();
    assert_eq!(ana_dt.class, MultipletClass::DoubletOfTriplets);
    assert!((ana_dt.j_values_hz()[0] - 10.0).abs() < 0.05);
    assert!((ana_dt.j_values_hz()[1] - 3.2).abs() < 0.15);

    let shifts_td = [6.8204, 6.8000, 2.500, 2.500]; // Delta nu_AB = 10.2 Hz = 3.0 * 3.4 Hz
    let j_td = vec![
        vec![0.0, 3.4, 9.4, 9.4],
        vec![3.4, 0.0, 7.2, 7.2],
        vec![9.4, 7.2, 0.0, 0.0],
        vec![9.4, 7.2, 0.0, 0.0],
    ];
    let qm_td = simulate_exact_spin_half_multiplet(&shifts_td, &j_td, 0, mhz, 0.01).unwrap();
    let ana_td = analyze_multiplet_peaks(&qm_td, mhz, &cfg).unwrap();
    assert_eq!(ana_td.class, MultipletClass::TripletOfDoublets);
    assert!((ana_td.j_values_hz()[0] - 9.4).abs() < 0.15);
    assert!((ana_td.j_values_hz()[1] - 3.4).abs() < 0.10);
}

// ============================================================================
// 3. Publication Journal Report Formatting & Lossless Round-Trip Parsing
// ============================================================================

#[test]
fn journal_report_formatter_and_parser_roundtrip_all_styles_and_encodings() {
    let report_1h = JournalNmrReport {
        nucleus: ReportNucleus::H1,
        proton_decoupled: false,
        mhz: 500.0,
        solvent: "CDCl3".to_string(),
        signals: vec![
            ReportedSignal {
                shift_ppm: 7.28,
                shift_range_ppm: None,
                multiplicity: Some(MultipletClass::DoubletOfDoublets),
                couplings_hz: vec![8.4, 1.8],
                proton_count: Some(2),
                assignment: Some("ArH".to_string()),
            },
            ReportedSignal {
                shift_ppm: 5.92,
                shift_range_ppm: None,
                multiplicity: Some(MultipletClass::DoubletOfDoubletOfDoublets),
                couplings_hz: vec![17.2, 10.4, 6.8],
                proton_count: Some(1),
                assignment: None,
            },
            ReportedSignal {
                shift_ppm: 4.18,
                shift_range_ppm: None,
                multiplicity: Some(MultipletClass::Quartet),
                couplings_hz: vec![7.1],
                proton_count: Some(2),
                assignment: None,
            },
            ReportedSignal {
                shift_ppm: 3.65,
                shift_range_ppm: None,
                multiplicity: Some(MultipletClass::BroadSinglet),
                couplings_hz: Vec::new(),
                proton_count: Some(1),
                assignment: Some("OH".to_string()),
            },
            ReportedSignal {
                shift_ppm: 2.365,
                shift_range_ppm: Some((2.41, 2.32)),
                multiplicity: Some(MultipletClass::Multiplet),
                couplings_hz: Vec::new(),
                proton_count: Some(2),
                assignment: None,
            },
            ReportedSignal {
                shift_ppm: 1.29,
                shift_range_ppm: None,
                multiplicity: Some(MultipletClass::Triplet),
                couplings_hz: vec![7.1],
                proton_count: Some(3),
                assignment: None,
            },
        ],
    };

    // Exact ASCII ACS string verification
    let acs_ascii =
        format_journal_nmr_report(&report_1h, PublicationStyle::Acs, TextEncoding::Ascii);
    assert_eq!(
        acs_ascii,
        "1H NMR (500 MHz, CDCl3) delta 7.28 (dd, J = 8.4, 1.8 Hz, 2H, ArH), \
         5.92 (ddd, J = 17.2, 10.4, 6.8 Hz, 1H), 4.18 (q, J = 7.1 Hz, 2H), \
         3.65 (br s, 1H, OH), 2.41-2.32 (m, 2H), 1.29 (t, J = 7.1 Hz, 3H)."
    );

    // Unicode ACS string verification (`¹H`, `CDCl₃`, `\u{03B4}`, `\u{2013}`)
    let acs_unicode =
        format_journal_nmr_report(&report_1h, PublicationStyle::Acs, TextEncoding::Unicode);
    assert!(acs_unicode
        .starts_with("¹H NMR (500 MHz, CDCl₃) \u{03B4} 7.28 (dd, J = 8.4, 1.8 Hz, 2H, ArH)"));
    assert!(acs_unicode.contains("2.41\u{2013}2.32 (m, 2H)"));

    // RSC style puts integration count first (`2H, dd, J = 8.4, 1.8 Hz`) and `\delta H`
    let rsc_unicode =
        format_journal_nmr_report(&report_1h, PublicationStyle::Rsc, TextEncoding::Unicode);
    assert!(rsc_unicode
        .starts_with("¹H NMR (500 MHz, CDCl₃) \u{03B4}H 7.28 (2H, dd, J = 8.4, 1.8 Hz, ArH)"));

    // Round-trip all 4 styles x 3 encodings for 1H
    for style in [
        PublicationStyle::Acs,
        PublicationStyle::Joc,
        PublicationStyle::Rsc,
        PublicationStyle::Nature,
    ] {
        for enc in [
            TextEncoding::Unicode,
            TextEncoding::Ascii,
            TextEncoding::Latex,
        ] {
            let formatted = format_journal_nmr_report(&report_1h, style, enc);
            let parsed = parse_journal_nmr_report(&formatted).unwrap_or_else(|e| {
                panic!("failed parsing ({style:?}, {enc:?}) '{formatted}': {e}")
            });
            assert_eq!(parsed.nucleus, ReportNucleus::H1);
            assert!((parsed.mhz - 500.0).abs() < 1e-9);
            assert_eq!(parsed.solvent, "CDCl3");
            assert_eq!(parsed.signals.len(), report_1h.signals.len());

            for (orig, rec) in report_1h.signals.iter().zip(parsed.signals.iter()) {
                assert!((orig.shift_ppm - rec.shift_ppm).abs() < 0.01);
                assert_eq!(orig.shift_range_ppm, rec.shift_range_ppm);
                assert_eq!(orig.multiplicity, rec.multiplicity);
                assert_eq!(orig.couplings_hz, rec.couplings_hz);
                assert_eq!(orig.proton_count, rec.proton_count);
                assert_eq!(orig.assignment, rec.assignment);
            }
        }
    }

    // Round-trip 13C{1H} NMR (including a C-F doublet `162.4 (d, J = 245.2 Hz)`) across all 12 combinations
    let report_13c = JournalNmrReport {
        nucleus: ReportNucleus::C13,
        proton_decoupled: true,
        mhz: 126.0,
        solvent: "DMSO-d6".to_string(),
        signals: vec![
            ReportedSignal::new_decoupled(170.4),
            ReportedSignal {
                shift_ppm: 162.4,
                shift_range_ppm: None,
                multiplicity: Some(MultipletClass::Doublet),
                couplings_hz: vec![245.2],
                proton_count: None,
                assignment: None,
            },
            ReportedSignal::new_decoupled(128.6),
            ReportedSignal::new_decoupled(52.1),
        ],
    };

    for style in [
        PublicationStyle::Acs,
        PublicationStyle::Joc,
        PublicationStyle::Rsc,
        PublicationStyle::Nature,
    ] {
        for enc in [
            TextEncoding::Unicode,
            TextEncoding::Ascii,
            TextEncoding::Latex,
        ] {
            let txt = report_13c.format(style, enc);
            let rec = JournalNmrReport::parse(&txt).unwrap();
            assert_eq!(rec.nucleus, ReportNucleus::C13);
            assert!(rec.proton_decoupled);
            assert!((rec.mhz - 126.0).abs() < 1e-9);
            assert_eq!(rec.solvent, "DMSO-d6");
            assert_eq!(rec.signals.len(), 4);
            assert_eq!(rec.signals[1].multiplicity, Some(MultipletClass::Doublet));
            assert_eq!(rec.signals[1].couplings_hz, vec![245.2]);
        }
    }
}

// ============================================================================
// 4. NIST-Traceable Internal-Standard `qNMR` Purity, Replicate RSD & `5 * T1`
// ============================================================================

#[test]
fn qnmr_end_to_end_nist_internal_standard_purity_uncertainty_replicates_and_t1_guard() {
    // Verify built-in NIST/USP primary calibrants table
    let crms = qnmr_certified_standards();
    let dmso2 = crms
        .iter()
        .find(|c| c.abbreviation == "DMSO2")
        .expect("DMSO2 standard present");
    assert!((dmso2.mw_g_mol - 94.133).abs() < 1e-6);
    assert_eq!(dmso2.protons_n, 6);

    // End-to-end synthetic 1D `1H qNMR` assay in DMSO-d6 at 600 MHz:
    // - Internal standard: Dimethyl sulfone (`MW_std = 94.133 g/mol`, `m_std = 4.70665 mg`, `P_std = 0.9995`, `N_std = 6`)
    //   -> `n_std = (4.70665e-3 * 0.9995) / 94.133 = 4.997500e-5 mol` (`6 * n_std = 2.998500e-4 mol H`).
    // - Analyte: Caffeine (`MW_x = 194.190 g/mol`, `m_x = 12.5000 mg`, true purity `P_true = 0.9840` = `98.40% w/w`)
    //   -> `n_x = (12.5000e-3 * 0.9840) / 194.190 = 6.334003e-5 mol` in `V = 0.600 mL` (`C_x = 0.1055667 mol/L = 105.567 mM`).
    //   Reporter signals: `1H` purine singlet at `8.01 ppm` and `3H` N-CH3 singlet at `3.88 ppm`.
    let mhz = 600.0;
    let mw_x = 194.190;
    let m_x = 12.5000;
    let p_true = 0.9840;
    let m_std = 4.70665;
    let p_std = 0.9995;

    let mol_std = (m_std * 1e-3 * p_std) / dmso2.mw_g_mol;
    let mol_x = (m_x * 1e-3 * p_true) / mw_x;

    // Synthesize full continuous 1D spectrum (including linear baseline drift) and integrate!
    let grid = make_descending_ppm_grid(9.0, 2.0, 56_001); // 0.000125 ppm spacing
    let peaks = vec![
        MultipletSubPeak::new(8.01, 1.0 * mol_x * 1e5).with_fwhm_hz(1.1),
        MultipletSubPeak::new(3.88, 3.0 * mol_x * 1e5).with_fwhm_hz(1.1),
        MultipletSubPeak::new(3.00, 6.0 * mol_std * 1e5).with_fwhm_hz(1.1),
    ];
    let spec_y: Vec<f64> = synthesize_multiplet_spectrum(&grid, &peaks, mhz, 1.1, 0.0)
        .into_iter()
        .enumerate()
        .map(|(i, y)| y + 0.25 - 0.04 * grid[i])
        .collect();

    let int_specs = vec![
        IntegralRegionSpec::new(8.35, 7.67).with_label("Caffeine H-8 (1H)"),
        IntegralRegionSpec::new(4.22, 3.54).with_label("Caffeine N-Me (3H)"),
        IntegralRegionSpec::new(3.34, 2.66).with_label("DMSO2 Std (6H)"),
    ];
    let int_rep = integrate_regions(
        &grid,
        &spec_y,
        &int_specs,
        NormalizationMode::None,
        Some(1e-4),
        None,
    )
    .unwrap();

    let std_spec = QnmrStandardSpec::new(
        "Dimethyl sulfone (NIST SRM)",
        dmso2.mw_g_mol,
        m_std,
        p_std,
        6,
        int_rep.regions[2].abs_integral,
    )
    .with_purity_percent(99.95, 0.05)
    .with_uncertainties(0.002, 0.002, int_rep.regions[2].abs_sigma)
    .with_t1_s(2.6);

    let analyte_spec = QnmrAnalyteSpec {
        name: "Caffeine".to_string(),
        mw_g_mol: mw_x,
        mw_sigma_g_mol: 0.004,
        mass_mg: m_x,
        mass_sigma_mg: 0.003,
        signals: vec![
            QnmrAnalyteSignal::new(
                1,
                int_rep.regions[0].abs_integral,
                int_rep.regions[0].abs_sigma,
            )
            .with_t1_s(3.8),
            QnmrAnalyteSignal::new(
                3,
                int_rep.regions[1].abs_integral,
                int_rep.regions[1].abs_sigma,
            )
            .with_t1_s(2.1),
        ],
        solution_volume_ml: Some(0.600),
        solution_volume_sigma_ml: Some(0.001),
    };

    // Quantitative acquisition (`D1 = 22.0 s, AQ = 3.0 s` -> `T_rep = 25.0 s >= 5 * 3.8 = 19.0 s`)
    let acq_ok = QnmrAcquisitionParams::new(22.0, 3.0);
    let qnmr_ok = calculate_qnmr_purity(&analyte_spec, &std_spec, Some(&acq_ok)).unwrap();

    let abs_pct_err = (qnmr_ok.purity_percent - 98.40).abs();
    assert!(
        abs_pct_err < 0.05,
        "NIST qNMR purity recovery {:.5}% w/w deviated from 98.40% by {:.5}% (must be < 0.1%)",
        qnmr_ok.purity_percent,
        abs_pct_err
    );
    assert!(!qnmr_ok.t1_recycle_warning);
    assert_eq!(qnmr_ok.required_5x_t1_s, Some(19.0));
    assert!(
        qnmr_ok.combined_uncertainty_percent > 0.0 && qnmr_ok.combined_uncertainty_percent < 0.15
    );
    assert!(
        (qnmr_ok.expanded_uncertainty_k2_percent - 2.0 * qnmr_ok.combined_uncertainty_percent)
            .abs()
            < 1e-12
    );

    // Check molar concentration (`0.105567 mol/L` = `105.567 mM`)
    let expected_molar = mol_x / 0.600e-3;
    let got_molar = qnmr_ok.molar_concentration_mol_l.unwrap();
    assert!((got_molar - expected_molar).abs() / expected_molar < 5e-4);
    assert!((qnmr_ok.molar_concentration_mm.unwrap() - 1000.0 * expected_molar).abs() < 0.05);

    let (direct_c, direct_sig) = calculate_molar_concentration_from_standard(
        int_rep.regions[1].abs_integral,
        int_rep.regions[1].abs_sigma,
        3,
        int_rep.regions[2].abs_integral,
        int_rep.regions[2].abs_sigma,
        6,
        qnmr_ok.standard_concentration_mol_l.unwrap(),
        1e-5,
    )
    .unwrap();
    assert!((direct_c - expected_molar).abs() / expected_molar < 5e-4);
    assert!(direct_sig > 0.0);

    // Short recycle delay (`D1 = 2.0 s, AQ = 2.0 s` -> `T_rep = 4.0 s < 19.0 s`) MUST trigger `5 * T1` warning!
    let acq_short = QnmrAcquisitionParams::new(2.0, 2.0);
    let qnmr_warn = calculate_qnmr_purity(&analyte_spec, &std_spec, Some(&acq_short)).unwrap();
    assert!(qnmr_warn.t1_recycle_warning);
    assert!(!qnmr_warn.warnings.is_empty());
    // Since T1_x (3.8 s) > T1_std (2.6 s), analyte saturates MORE than standard -> negative saturation bias!
    assert!(qnmr_warn.estimated_saturation_bias_percent.unwrap() < -5.0);

    // Multi-tube replicate RSD calculation (`n = 4` independent weighings)
    let rep_scales = [0.9985, 1.0010, 0.9992, 1.0013];
    let replicates: Vec<(QnmrAnalyteSpec, QnmrStandardSpec)> = rep_scales
        .iter()
        .map(|&s| {
            let mut an = analyte_spec.clone();
            for sig in &mut an.signals {
                sig.integral *= s;
            }
            (an, std_spec.clone())
        })
        .collect();
    let rep_summary = calculate_qnmr_replicates(&replicates, Some(&acq_ok)).unwrap();
    assert_eq!(rep_summary.replicate_count, 4);
    assert!((rep_summary.purity_percent - 98.40).abs() < 0.05);
    let rsd = rep_summary.replicate_rsd_percent.unwrap();
    assert!(
        rsd > 0.05 && rsd < 0.25,
        "unexpected replicate RSD: {rsd:.4}%"
    );
}

// ============================================================================
// 5. `Octet` 2D Structure <-> Multiplet Assignment Table & SDF/SMILES Mapping
// ============================================================================

#[test]
fn octet_structure_atom_mapping_sdf_smiles_and_scene_cross_highlighting() {
    // Parse atom-mapped SMILES for ethyl acetate: `[CH3:10][CH2:11][O:12][C:13](=[O:14])[CH3:15]`
    let mapped_smiles = "[CH3:10][CH2:11][O:12][C:13](=[O:14])[CH3:15]";
    let (mol, atom_map) = StructureAtomMap::from_smiles(mapped_smiles).unwrap();
    assert_eq!(atom_map.entries.len(), 6);

    // Verify 0-based Octet <-> 1-based SDF <-> SMILES `[atom:map]` correspondence
    assert_eq!(atom_map.by_octet_index(0).unwrap().sdf_atom_number, 1);
    assert_eq!(atom_map.by_sdf_number(2).unwrap().octet_atom_idx, 1);
    assert_eq!(atom_map.octet_indices_for_smiles_map(11), vec![1]);
    assert_eq!(atom_map.by_octet_index(1).unwrap().attached_hydrogens, 2); // CH2
    assert_eq!(atom_map.by_octet_index(0).unwrap().attached_hydrogens, 3); // terminal CH3

    // Round-trip through MDL V2000 `.mol` block
    let mol_block = write_mol_v2000(&mol).unwrap();
    let (_mol_v2k, sdf_map) = StructureAtomMap::from_mol_block(&mol_block).unwrap();
    assert_eq!(sdf_map.entries.len(), 6);
    assert_eq!(sdf_map.by_sdf_number(6).unwrap().attached_hydrogens, 3);

    // Synthesize & analyze the 3 experimental 1H multiplets of ethyl acetate:
    // 1. O-CH2 quartet at 4.12 ppm (J = 7.1 Hz, 2H)
    // 2. C(=O)CH3 singlet at 2.04 ppm (3H)
    // 3. CH2-CH3 triplet at 1.26 ppm (J = 7.1 Hz, 3H)
    let cfg = MultipletConfig::default();
    let mhz = 500.0;
    let mut m_ch2 = analyze_multiplet_peaks(
        &synthesize_multiplet_peaks(4.12, mhz, &[(3, 7.1, None)], 1.0),
        mhz,
        &cfg,
    )
    .unwrap();
    m_ch2.proton_count = Some(2);

    let mut m_ac = analyze_multiplet_peaks(&[MultipletSubPeak::new(2.04, 3.0)], mhz, &cfg).unwrap();
    m_ac.proton_count = Some(3);

    let mut m_et = analyze_multiplet_peaks(
        &synthesize_multiplet_peaks(1.26, mhz, &[(2, 7.1, None)], 1.0),
        mhz,
        &cfg,
    )
    .unwrap();
    m_et.proton_count = Some(3);

    let multiplets = vec![m_ch2, m_ac, m_et];

    // Auto-assign against `mol` (with SMILES maps preserved!)
    let mut table = auto_assign_1h_multiplets(&mol, &multiplets, Some("CDCl3"));
    assert_eq!(table.links.len(), 3);
    assert!(
        table.unassigned_proton_bearing_atoms().is_empty(),
        "all 3 proton-bearing carbons should be assigned"
    );

    // Multiplet 0 (`4.12 ppm, q, 2H`) -> `CH2` (`octet_idx = 1`, `sdf = 2`, `smiles_map = 11`)
    assert_eq!(table.atoms_for_multiplet(0), &[1]);
    assert_eq!(table.sdf_numbers_for_multiplet(0), &[2]);
    assert_eq!(table.multiplets_for_octet_atom(1), vec![0]);
    assert_eq!(table.multiplets_for_sdf_number(2), vec![0]);
    assert_eq!(table.multiplets_for_smiles_map(11), vec![0]);

    // Multiplet 2 (`1.26 ppm, t, 3H`) -> `CH3` next to CH2 (`octet_idx = 0`, `sdf = 1`, `smiles_map = 10`)
    assert_eq!(table.atoms_for_multiplet(2), &[0]);
    assert_eq!(table.multiplets_for_smiles_map(10), vec![2]);

    // Verify symmetry peer grouping on para-disubstituted benzene (`c1cc(Br)ccc1Cl`)
    let sym_mol = parse_smiles("Brc1ccc(Cl)cc1").unwrap();
    let sym_map = StructureAtomMap::from_molecule(&sym_mol);
    // Carbon index 2 has a 2-fold symmetric aromatic CH peer
    assert_eq!(sym_map.symmetry_peers(2).len(), 2);

    // Cross-highlight onto an `Octet` `RenderScene` and verify atom + bond halo propagation
    let mut scene = RenderScene::from(&mol);
    let halo = RgbaColor::rgba(255, 190, 40, 140);
    // Assign multiplet 0 across bonded ethyl carbons `0` and `1` (SDF `1, 2`) to test bond highlighting too
    table
        .assign_multiplet_by_sdf_numbers(0, ReportNucleus::H1, &multiplets[0], &[1, 2], true)
        .unwrap();
    table.highlight_multiplet_on_scene(&mut scene, 0, halo);
    assert_eq!(scene.atoms[0].highlight, Some(halo));
    assert_eq!(scene.atoms[1].highlight, Some(halo));
    assert_eq!(scene.atoms[2].highlight, None);
    let bond_01 = scene
        .bonds
        .iter()
        .find(|b| (b.a == 0 && b.b == 1) || (b.a == 1 && b.b == 0))
        .unwrap();
    assert_eq!(bond_01.highlight, Some(halo));

    MultipletAssignmentTable::clear_scene_highlights(&mut scene);
    assert!(scene.atoms.iter().all(|a| a.highlight.is_none()));
    assert!(scene.bonds.iter().all(|b| b.highlight.is_none()));

    // Verify TSV and SDF property tag export + round-trip reconstruction
    let tsv = table.to_tsv();
    assert!(tsv.contains("multiplet_id\tnucleus\tshift_ppm"));
    assert!(tsv.contains("10,11"));
    let sdf_tag = table.to_sdf_property_tag();
    assert!(sdf_tag.contains("0|4.120|q|2|1,2"));

    let restored =
        MultipletAssignmentTable::from_sdf_property_tag(&mol, ReportNucleus::H1, &sdf_tag).unwrap();
    assert_eq!(restored.links.len(), table.links.len());
    assert_eq!(restored.atoms_for_multiplet(0), &[0, 1]);
    assert_eq!(restored.sdf_numbers_for_multiplet(0), &[1, 2]);
    assert_eq!(restored.multiplets_for_smiles_map(10), vec![0, 2]);
}

// ============================================================================
// 6. 28-Multiplet Noisy Benchmark Sweep, 19F/31P Negative Shifts & NIST Maleic Acid
// ============================================================================

#[test]
fn noisy_28_multiplet_benchmark_sweep_recovers_all_classes_and_sub_020_hz_j() {
    let mhz = 500.0;
    let cfg = MultipletConfig::default();

    // (center_ppm, expected_class, couplings, fwhm_hz)
    let cases: Vec<(f64, MultipletClass, Vec<CouplingSpec>, f64)> = vec![
        (2.15, MultipletClass::Singlet, vec![], 1.1),
        (9.45, MultipletClass::BroadSinglet, vec![], 8.2),
        (7.42, MultipletClass::Doublet, vec![(1, 8.2, None)], 1.0),
        (
            7.88,
            MultipletClass::Doublet,
            vec![(1, 8.6, Some(2.5))],
            1.0,
        ),
        (
            6.95,
            MultipletClass::Doublet,
            vec![(1, 15.8, Some(-3.0))],
            1.0,
        ),
        (1.24, MultipletClass::Triplet, vec![(2, 7.2, None)], 1.0),
        (
            2.78,
            MultipletClass::Triplet,
            vec![(2, 6.8, Some(3.0))],
            1.0,
        ),
        (4.14, MultipletClass::Quartet, vec![(3, 7.1, None)], 1.0),
        (
            3.62,
            MultipletClass::Quartet,
            vec![(3, 6.9, Some(-2.8))],
            1.0,
        ),
        (1.68, MultipletClass::Quintet, vec![(4, 7.0, None)], 0.9),
        (1.52, MultipletClass::Sextet, vec![(5, 7.1, None)], 0.9),
        (2.92, MultipletClass::Septet, vec![(6, 6.8, None)], 0.85),
        (
            7.28,
            MultipletClass::DoubletOfDoublets,
            vec![(1, 8.4, None), (1, 1.9, None)],
            0.9,
        ),
        (
            7.54,
            MultipletClass::DoubletOfDoublets,
            vec![(1, 8.8, Some(2.6)), (1, 2.2, None)],
            0.9,
        ),
        (
            6.62,
            MultipletClass::DoubletOfDoublets,
            vec![(1, 16.2, Some(-3.0)), (1, 6.4, Some(3.2))],
            0.95,
        ),
        (
            5.86,
            MultipletClass::DoubletOfDoubletOfDoublets,
            vec![(1, 17.1, None), (1, 10.3, None), (1, 6.2, None)],
            0.9,
        ),
        (
            6.12,
            MultipletClass::DoubletOfDoubletOfDoublets,
            vec![(1, 11.4, Some(2.8)), (1, 7.2, None), (1, 2.7, None)],
            0.85,
        ),
        (
            6.40,
            MultipletClass::DoubletOfTriplets,
            vec![(1, 15.6, None), (2, 6.7, None)],
            0.95,
        ),
        (
            6.48,
            MultipletClass::DoubletOfTriplets,
            vec![(1, 15.2, Some(2.7)), (2, 6.5, None)],
            0.95,
        ),
        (
            7.14,
            MultipletClass::TripletOfDoublets,
            vec![(2, 7.8, None), (1, 1.8, None)],
            0.85,
        ),
        (
            7.22,
            MultipletClass::TripletOfDoublets,
            vec![(2, 8.0, Some(-2.9)), (1, 2.1, None)],
            0.85,
        ),
        (
            5.74,
            MultipletClass::DoubletOfQuartets,
            vec![(1, 15.4, None), (3, 6.5, None)],
            0.9,
        ),
        (
            5.80,
            MultipletClass::DoubletOfQuartets,
            vec![(1, 14.8, Some(2.8)), (3, 6.4, None)],
            0.9,
        ),
        (
            4.52,
            MultipletClass::QuartetOfDoublets,
            vec![(3, 6.8, None), (1, 2.5, None)],
            0.85,
        ),
        (
            4.60,
            MultipletClass::QuartetOfDoublets,
            vec![(3, 7.0, Some(3.0)), (1, 2.6, None)],
            0.85,
        ),
        (
            3.95,
            MultipletClass::TripletOfTriplets,
            vec![(2, 11.4, None), (2, 4.0, None)],
            0.9,
        ),
        (
            4.05,
            MultipletClass::TripletOfTriplets,
            vec![(2, 11.8, Some(-2.8)), (2, 4.2, None)],
            0.9,
        ),
        (3.31, MultipletClass::Singlet, vec![], 1.0),
    ];

    assert_eq!(cases.len(), 28);
    let mut rng = Rng::new(0x517C_C1B7_2722_0A95);
    let mut max_j_err_hz = 0.0_f64;

    for (idx, (center_ppm, exp_class, couplings, fwhm_hz)) in cases.iter().enumerate() {
        let grid = make_descending_ppm_grid(center_ppm + 0.20, center_ppm - 0.20, 4_001);
        let peaks = synthesize_multiplet_peaks(*center_ppm, mhz, couplings, *fwhm_hz);
        let clean = synthesize_multiplet_spectrum(&grid, &peaks, mhz, *fwhm_hz, 0.0);
        let max_y = clean.iter().copied().fold(0.0_f64, f64::max);
        // Realistic high-resolution 1D thermal noise (`SNR ~ 120` relative to tallest sub-line)
        let noise_sigma = max_y / 120.0;
        let noisy: Vec<f64> = clean
            .into_iter()
            .map(|y| y + rng.normal(0.0, noise_sigma))
            .collect();

        let ana = analyze_multiplet_region(
            &grid,
            &noisy,
            center_ppm + 0.16,
            center_ppm - 0.16,
            mhz,
            &cfg,
        )
        .unwrap();

        assert_eq!(
            ana.class, *exp_class,
            "case #{idx} ({exp_class}) misclassified as {} (Js = {:?})",
            ana.class, ana.couplings
        );

        let mut true_js: Vec<f64> = couplings.iter().map(|c| c.1).collect();
        true_js.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let got_js = ana.j_values_hz();
        assert_eq!(got_js.len(), true_js.len(), "case #{idx} J count mismatch");
        for (gj, tj) in got_js.iter().zip(true_js.iter()) {
            let err = (gj - tj).abs();
            max_j_err_hz = max_j_err_hz.max(err);
            assert!(
                err < 0.20,
                "case #{idx} ({exp_class}): |J_est ({gj:.3}) - J_true ({tj:.3})| = {err:.3} Hz >= 0.20 Hz"
            );
        }
    }

    assert!(max_j_err_hz < 0.20);
}

#[test]
fn heteronuclear_19f_and_31p_negative_shift_journal_formatting_and_parsing_roundtrip() {
    let report_19f = JournalNmrReport {
        nucleus: ReportNucleus::F19,
        mhz: 471.0,
        solvent: "CDCl3".to_string(),
        proton_decoupled: false,
        signals: vec![
            ReportedSignal {
                shift_ppm: -62.84,
                shift_range_ppm: None,
                multiplicity: Some(MultipletClass::Singlet),
                couplings_hz: vec![],
                proton_count: Some(3),
                assignment: Some("CF3".to_string()),
            },
            ReportedSignal {
                shift_ppm: -115.42,
                shift_range_ppm: None,
                multiplicity: Some(MultipletClass::TripletOfTriplets),
                couplings_hz: vec![9.2, 5.6],
                proton_count: Some(1),
                assignment: Some("ArF".to_string()),
            },
            ReportedSignal {
                shift_ppm: -128.5,
                shift_range_ppm: Some((-128.2, -128.9)),
                multiplicity: Some(MultipletClass::Multiplet),
                couplings_hz: vec![],
                proton_count: Some(2),
                assignment: None,
            },
        ],
    };

    for style in [
        PublicationStyle::Acs,
        PublicationStyle::Joc,
        PublicationStyle::Rsc,
        PublicationStyle::Nature,
    ] {
        for enc in [
            TextEncoding::Unicode,
            TextEncoding::Ascii,
            TextEncoding::Latex,
        ] {
            let formatted = format_journal_nmr_report(&report_19f, style, enc);
            let parsed = parse_journal_nmr_report(&formatted).unwrap();
            assert_eq!(parsed.nucleus, ReportNucleus::F19);
            assert!((parsed.mhz - 471.0).abs() < 1e-9);
            assert_eq!(parsed.signals.len(), 3);
            assert!((parsed.signals[0].shift_ppm - (-62.84)).abs() < 1e-6);
            assert_eq!(parsed.signals[0].proton_count, Some(3));
            assert!((parsed.signals[1].shift_ppm - (-115.42)).abs() < 1e-6);
            assert_eq!(
                parsed.signals[1].multiplicity,
                Some(MultipletClass::TripletOfTriplets)
            );
            assert_eq!(parsed.signals[1].couplings_hz, vec![9.2, 5.6]);
            let (hi, lo) = parsed.signals[2].shift_range_ppm.unwrap();
            assert!((hi - (-128.2)).abs() < 0.02);
            assert!((lo - (-128.9)).abs() < 0.02);
        }
    }

    let report_31p = JournalNmrReport {
        nucleus: ReportNucleus::P31,
        mhz: 202.0,
        solvent: "CD3CN".to_string(),
        proton_decoupled: true,
        signals: vec![ReportedSignal {
            shift_ppm: -144.6,
            shift_range_ppm: None,
            multiplicity: Some(MultipletClass::Septet),
            couplings_hz: vec![706.4],
            proton_count: Some(1),
            assignment: Some("PF6".to_string()),
        }],
    };
    let txt_31p = report_31p.format(PublicationStyle::Acs, TextEncoding::Unicode);
    let rec_31p = JournalNmrReport::parse(&txt_31p).unwrap();
    assert_eq!(rec_31p.nucleus, ReportNucleus::P31);
    assert!(rec_31p.proton_decoupled);
    assert!((rec_31p.signals[0].shift_ppm - (-144.6)).abs() < 1e-6);
    assert_eq!(rec_31p.signals[0].couplings_hz, vec![706.4]);
}

#[test]
fn qnmr_second_nist_maleic_acid_standard_and_input_validation_guards() {
    let crms = qnmr_certified_standards();
    let ma = crms
        .iter()
        .find(|c| c.abbreviation == "MA")
        .expect("Maleic acid standard present");
    assert_eq!(ma.protons_n, 2);

    // Sodium acetate (`MW = 82.034 g/mol`, `N_x = 3`, `m_x = 15.000 mg`, true purity `99.12%`)
    // assayed in `D2O` against Maleic acid (`MW = 116.072 g/mol`, `N_std = 2`, `m_std = 8.500 mg`, `P_std = 0.9994`)
    let mw_x = 82.034;
    let m_x = 15.000;
    let p_true = 0.9912;
    let m_std = 8.500;
    let p_std = 0.9994;

    let mol_std = (m_std * 1e-3 * p_std) / ma.mw_g_mol;
    let mol_x = (m_x * 1e-3 * p_true) / mw_x;

    let std_spec = QnmrStandardSpec::new(
        "Maleic acid",
        ma.mw_g_mol,
        m_std,
        p_std,
        ma.protons_n,
        2.0 * mol_std * 1e6,
    );
    let an_spec = QnmrAnalyteSpec {
        name: "Sodium acetate".to_string(),
        mw_g_mol: mw_x,
        mw_sigma_g_mol: 0.002,
        mass_mg: m_x,
        mass_sigma_mg: 0.003,
        signals: vec![QnmrAnalyteSignal::new(3, 3.0 * mol_x * 1e6, 5e-4)],
        solution_volume_ml: Some(0.550),
        solution_volume_sigma_ml: None,
    };

    let rep = calculate_qnmr_purity(&an_spec, &std_spec, None).unwrap();
    assert!(
        (rep.purity_percent - 99.12).abs() < 1e-6,
        "exact qNMR equation recovery error: {}",
        (rep.purity_percent - 99.12).abs()
    );

    // Invalid inputs must return structured `MultipletError::InvalidInput`, never panic or NaN
    let bad_std = QnmrStandardSpec::new("Bad", 100.0, 0.0, 0.999, 2, 100.0);
    assert!(calculate_qnmr_purity(&an_spec, &bad_std, None).is_err());

    let mut bad_an = an_spec.clone();
    bad_an.signals.clear();
    assert!(calculate_qnmr_purity(&bad_an, &std_spec, None).is_err());
}
