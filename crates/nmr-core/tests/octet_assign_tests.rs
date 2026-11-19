//! Integration tests for `nmr_core::octet_assign` (1D experimental-to-structure Hungarian
//! auto-assignment, manual swap/lock overrides, raw trace peak extraction, and ACS formatting).
//!
//! Provenance: Clean-room test suite for FERRIC (`nmr-core`).

use nmr_core::octet_assign::{
    auto_assign_1d_to_structure, auto_assign_1d_to_structure_with_locks,
    extract_experimental_peaks_from_spectrum, format_acs_with_atom_labels,
    predicted_groups_from_signals, swap_or_lock_assignment, ExperimentalPeak1D, PredictedGroup1D,
};
use nmr_core::octet_render::{SceneSpectrum1D, SpectrumSignal1D};

fn aspirin_predicted_1h() -> Vec<PredictedGroup1D> {
    vec![
        PredictedGroup1D {
            group_idx: 0,
            atom_indices: vec![8],
            label: "COOH".to_string(),
            shift_ppm: 11.40,
            sd_ppm: 0.45,
            expected_integral: 1.0,
            multiplicity: "br s".to_string(),
            couplings_hz: vec![],
        },
        PredictedGroup1D {
            group_idx: 1,
            atom_indices: vec![5],
            label: "H-6".to_string(),
            shift_ppm: 8.10,
            sd_ppm: 0.35,
            expected_integral: 1.0,
            multiplicity: "dd".to_string(),
            couplings_hz: vec![7.8, 1.6],
        },
        PredictedGroup1D {
            group_idx: 2,
            atom_indices: vec![3],
            label: "H-4".to_string(),
            shift_ppm: 7.60,
            sd_ppm: 0.35,
            expected_integral: 1.0,
            multiplicity: "td".to_string(),
            couplings_hz: vec![7.7, 1.6],
        },
        PredictedGroup1D {
            group_idx: 3,
            atom_indices: vec![4],
            label: "H-5".to_string(),
            shift_ppm: 7.34,
            sd_ppm: 0.35,
            expected_integral: 1.0,
            multiplicity: "td".to_string(),
            couplings_hz: vec![7.6, 1.1],
        },
        PredictedGroup1D {
            group_idx: 4,
            atom_indices: vec![2],
            label: "H-3".to_string(),
            shift_ppm: 7.11,
            sd_ppm: 0.35,
            expected_integral: 1.0,
            multiplicity: "dd".to_string(),
            couplings_hz: vec![8.1, 1.1],
        },
        PredictedGroup1D {
            group_idx: 5,
            atom_indices: vec![12],
            label: "OAc CH3".to_string(),
            shift_ppm: 2.32,
            sd_ppm: 0.35,
            expected_integral: 3.0,
            multiplicity: "s".to_string(),
            couplings_hz: vec![],
        },
    ]
}

#[test]
fn aspirin_1h_auto_assign_recovers_all_six_groups_and_rejects_solvent_peak() {
    let pred = aspirin_predicted_1h();
    // Shuffled experimental peaks with realistic 0.02-0.08 ppm shift offsets + extra CDCl3 peak at 7.26
    let exp = vec![
        ExperimentalPeak1D {
            peak_idx: 10,
            ppm: 2.35,
            integral: Some(3.02),
            multiplicity: Some("s".to_string()),
            couplings_hz: vec![],
            intensity: 3.0,
        },
        ExperimentalPeak1D {
            peak_idx: 11,
            ppm: 7.26, // residual CHCl3 solvent peak!
            integral: Some(0.18),
            multiplicity: Some("s".to_string()),
            couplings_hz: vec![],
            intensity: 0.3,
        },
        ExperimentalPeak1D {
            peak_idx: 12,
            ppm: 8.13,
            integral: Some(0.98),
            multiplicity: Some("dd".to_string()),
            couplings_hz: vec![7.9, 1.7],
            intensity: 1.0,
        },
        ExperimentalPeak1D {
            peak_idx: 13,
            ppm: 7.12,
            integral: Some(1.01),
            multiplicity: Some("dd".to_string()),
            couplings_hz: vec![8.1, 1.2],
            intensity: 1.0,
        },
        ExperimentalPeak1D {
            peak_idx: 14,
            ppm: 11.46,
            integral: Some(0.94),
            multiplicity: Some("br s".to_string()),
            couplings_hz: vec![],
            intensity: 0.8,
        },
        ExperimentalPeak1D {
            peak_idx: 15,
            ppm: 7.62,
            integral: Some(1.00),
            multiplicity: Some("td".to_string()),
            couplings_hz: vec![7.8, 1.6],
            intensity: 1.0,
        },
        ExperimentalPeak1D {
            peak_idx: 16,
            ppm: 7.36,
            integral: Some(0.99),
            multiplicity: Some("td".to_string()),
            couplings_hz: vec![7.6, 1.1],
            intensity: 1.0,
        },
    ];

    let res = auto_assign_1d_to_structure(&pred, &exp, "1H", 500.0, "CDCl3");
    assert_eq!(res.nucleus, "1H");
    assert_eq!(res.matched_count, 6);
    assert_eq!(
        res.unassigned_peaks,
        vec![11],
        "CHCl3 solvent peak must remain unassigned"
    );
    assert!(
        res.rmsd_ppm < 0.08,
        "Expected tight RMSD < 0.08 ppm, got {:.4}",
        res.rmsd_ppm
    );
    assert!(
        res.mean_confidence > 0.70 && res.mean_confidence <= 1.0,
        "Expected high mean confidence, got {:.3}",
        res.mean_confidence
    );

    // Check exact 1-to-1 assignments
    assert_eq!(res.assignments[0].peak_idx, Some(14)); // COOH
    assert_eq!(res.assignments[1].peak_idx, Some(12)); // H-6
    assert_eq!(res.assignments[2].peak_idx, Some(15)); // H-4
    assert_eq!(res.assignments[3].peak_idx, Some(16)); // H-5
    assert_eq!(res.assignments[4].peak_idx, Some(13)); // H-3
    assert_eq!(res.assignments[5].peak_idx, Some(10)); // OAc CH3

    // Verify atom & peak bidirectional lookup
    assert_eq!(
        res.assignment_for_atom(12).and_then(|a| a.peak_idx),
        Some(10)
    );
    assert_eq!(
        res.assignment_for_peak(10).map(|a| a.label.as_str()),
        Some("OAc CH3")
    );

    // Verify ACS string formatting
    let acs = format_acs_with_atom_labels(&res);
    assert!(acs.starts_with("¹H NMR (500 MHz, CDCl3) δ 11.46"));
    assert!(acs.contains("COOH"));
    assert!(acs.contains("H-6"));
    assert!(acs.contains("J = 7.9, 1.7 Hz"));
    assert!(acs.contains("2.35 (s, 3H, OAc CH3)"));
}

#[test]
fn integral_and_multiplicity_disambiguate_crossed_close_shifts() {
    // OMe (3H singlet) predicted at 3.72 ppm vs methine CH (1H quartet) predicted at 3.76 ppm,
    // while in experiment OMe sits slightly downfield at 3.79 and CH sits at 3.70.
    // Shift alone would cross-assign them, but integral (3H vs 1H) + multiplicity (s vs q) resolves cleanly!
    let pred = vec![
        PredictedGroup1D {
            group_idx: 0,
            atom_indices: vec![1],
            label: "OCH3".to_string(),
            shift_ppm: 3.72,
            sd_ppm: 0.35,
            expected_integral: 3.0,
            multiplicity: "s".to_string(),
            couplings_hz: vec![],
        },
        PredictedGroup1D {
            group_idx: 1,
            atom_indices: vec![2],
            label: "CH".to_string(),
            shift_ppm: 3.76,
            sd_ppm: 0.35,
            expected_integral: 1.0,
            multiplicity: "q".to_string(),
            couplings_hz: vec![6.8],
        },
    ];
    let exp = vec![
        ExperimentalPeak1D {
            peak_idx: 0,
            ppm: 3.79,
            integral: Some(3.0),
            multiplicity: Some("s".to_string()),
            couplings_hz: vec![],
            intensity: 3.0,
        },
        ExperimentalPeak1D {
            peak_idx: 1,
            ppm: 3.70,
            integral: Some(1.0),
            multiplicity: Some("q".to_string()),
            couplings_hz: vec![6.9],
            intensity: 1.0,
        },
    ];

    let res = auto_assign_1d_to_structure(&pred, &exp, "1H", 400.0, "CDCl3");
    assert_eq!(
        res.assignments[0].peak_idx,
        Some(0),
        "OCH3 3H singlet must match peak 0"
    );
    assert_eq!(
        res.assignments[1].peak_idx,
        Some(1),
        "CH 1H quartet must match peak 1"
    );
}

#[test]
fn carbon_13c_auto_assign_and_acs_format() {
    let pred_signals = vec![
        SpectrumSignal1D {
            ppm: 171.2,
            multiplicity: "s".to_string(),
            couplings_hz: vec![],
            integral: 1.0,
            atom_indices: vec![0],
            assignment_label: Some("C=O".to_string()),
        },
        SpectrumSignal1D {
            ppm: 60.4,
            multiplicity: "s".to_string(),
            couplings_hz: vec![],
            integral: 1.0,
            atom_indices: vec![2],
            assignment_label: Some("OCH2".to_string()),
        },
        SpectrumSignal1D {
            ppm: 21.0,
            multiplicity: "s".to_string(),
            couplings_hz: vec![],
            integral: 1.0,
            atom_indices: vec![1],
            assignment_label: Some("Ac CH3".to_string()),
        },
        SpectrumSignal1D {
            ppm: 14.2,
            multiplicity: "s".to_string(),
            couplings_hz: vec![],
            integral: 1.0,
            atom_indices: vec![3],
            assignment_label: Some("Et CH3".to_string()),
        },
    ];
    let pred = predicted_groups_from_signals(&pred_signals, "13C");
    let exp = vec![
        ExperimentalPeak1D {
            peak_idx: 0,
            ppm: 14.1,
            integral: None,
            multiplicity: None,
            couplings_hz: vec![],
            intensity: 0.9,
        },
        ExperimentalPeak1D {
            peak_idx: 1,
            ppm: 20.8,
            integral: None,
            multiplicity: None,
            couplings_hz: vec![],
            intensity: 0.85,
        },
        ExperimentalPeak1D {
            peak_idx: 2,
            ppm: 60.5,
            integral: None,
            multiplicity: None,
            couplings_hz: vec![],
            intensity: 1.0,
        },
        ExperimentalPeak1D {
            peak_idx: 3,
            ppm: 171.0,
            integral: None,
            multiplicity: None,
            couplings_hz: vec![],
            intensity: 0.4,
        },
    ];

    let res = auto_assign_1d_to_structure(&pred, &exp, "13C", 125.76, "CDCl3");
    assert_eq!(res.nucleus, "13C");
    assert_eq!(res.matched_count, 4);
    assert_eq!(res.assignments[0].peak_idx, Some(3));
    assert_eq!(res.assignments[1].peak_idx, Some(2));
    assert_eq!(res.assignments[2].peak_idx, Some(1));
    assert_eq!(res.assignments[3].peak_idx, Some(0));

    let acs = format_acs_with_atom_labels(&res);
    assert_eq!(
        acs,
        "¹³C NMR (126 MHz, CDCl3) δ 171.0 (C=O), 60.5 (OCH2), 20.8 (Ac CH3), 14.1 (Et CH3)."
    );
}

#[test]
fn missing_peak_and_severe_outlier_remain_unassigned() {
    let pred = aspirin_predicted_1h();
    // Omit COOH (11.40) and add a severe outlier at 0.00 (TMS)
    let exp = vec![
        ExperimentalPeak1D {
            peak_idx: 0,
            ppm: 8.11,
            integral: Some(1.0),
            multiplicity: Some("dd".to_string()),
            couplings_hz: vec![],
            intensity: 1.0,
        },
        ExperimentalPeak1D {
            peak_idx: 1,
            ppm: 7.59,
            integral: Some(1.0),
            multiplicity: Some("td".to_string()),
            couplings_hz: vec![],
            intensity: 1.0,
        },
        ExperimentalPeak1D {
            peak_idx: 2,
            ppm: 7.35,
            integral: Some(1.0),
            multiplicity: Some("td".to_string()),
            couplings_hz: vec![],
            intensity: 1.0,
        },
        ExperimentalPeak1D {
            peak_idx: 3,
            ppm: 7.10,
            integral: Some(1.0),
            multiplicity: Some("dd".to_string()),
            couplings_hz: vec![],
            intensity: 1.0,
        },
        ExperimentalPeak1D {
            peak_idx: 4,
            ppm: 2.33,
            integral: Some(3.0),
            multiplicity: Some("s".to_string()),
            couplings_hz: vec![],
            intensity: 3.0,
        },
        ExperimentalPeak1D {
            peak_idx: 5,
            ppm: 0.00, // TMS reference, > 20 sigma away from COOH!
            integral: Some(0.5),
            multiplicity: Some("s".to_string()),
            couplings_hz: vec![],
            intensity: 0.5,
        },
    ];

    let res = auto_assign_1d_to_structure(&pred, &exp, "1H", 500.0, "CDCl3");
    assert_eq!(res.matched_count, 5);
    assert_eq!(
        res.assignments[0].peak_idx, None,
        "Broadened COOH should be unassigned"
    );
    assert_eq!(res.assignments[0].confidence, 0.0);
    assert_eq!(
        res.unassigned_peaks,
        vec![5],
        "TMS outlier must remain unassigned"
    );
}

#[test]
fn swap_or_lock_assignment_performs_atomic_swap_and_respects_locks_on_refit() {
    let pred = aspirin_predicted_1h();
    let exp = vec![
        ExperimentalPeak1D {
            peak_idx: 0,
            ppm: 11.42,
            integral: Some(1.0),
            multiplicity: Some("br s".to_string()),
            couplings_hz: vec![],
            intensity: 1.0,
        },
        ExperimentalPeak1D {
            peak_idx: 1,
            ppm: 8.10,
            integral: Some(1.0),
            multiplicity: Some("dd".to_string()),
            couplings_hz: vec![],
            intensity: 1.0,
        },
        ExperimentalPeak1D {
            peak_idx: 2,
            ppm: 7.58,
            integral: Some(1.0),
            multiplicity: Some("td".to_string()),
            couplings_hz: vec![],
            intensity: 1.0,
        },
        ExperimentalPeak1D {
            peak_idx: 3,
            ppm: 7.36,
            integral: Some(1.0),
            multiplicity: Some("td".to_string()),
            couplings_hz: vec![],
            intensity: 1.0,
        },
        ExperimentalPeak1D {
            peak_idx: 4,
            ppm: 7.12,
            integral: Some(1.0),
            multiplicity: Some("dd".to_string()),
            couplings_hz: vec![],
            intensity: 1.0,
        },
        ExperimentalPeak1D {
            peak_idx: 5,
            ppm: 2.34,
            integral: Some(3.0),
            multiplicity: Some("s".to_string()),
            couplings_hz: vec![],
            intensity: 3.0,
        },
    ];

    let mut res = auto_assign_1d_to_structure(&pred, &exp, "1H", 500.0, "CDCl3");
    assert_eq!(res.assignments[2].peak_idx, Some(2));
    assert_eq!(res.assignments[3].peak_idx, Some(3));

    // Swap group 2 (H-4) onto peak 3 and lock it -> group 3 (H-5) automatically receives peak 2!
    swap_or_lock_assignment(&mut res, 2, Some(3), true).expect("swap succeeds");
    assert_eq!(res.assignments[2].peak_idx, Some(3));
    assert!(res.assignments[2].locked);
    assert_eq!(res.assignments[2].confidence, 1.0);
    assert_eq!(res.assignments[3].peak_idx, Some(2));
    assert_eq!(res.matched_count, 6);
    assert!(res.unassigned_peaks.is_empty());

    // Re-running auto-assign with locks strictly preserves the locked (2 -> 3) override
    let locks = res.locked_pairs();
    assert_eq!(locks, vec![(2, 3)]);
    let refit = auto_assign_1d_to_structure_with_locks(&pred, &exp, "1H", 500.0, "CDCl3", &locks);
    assert_eq!(refit.assignments[2].peak_idx, Some(3));
    assert!(refit.assignments[2].locked);
    assert_eq!(refit.assignments[3].peak_idx, Some(2));

    // Unassigning group 0 releases peak 0 to unassigned_peaks
    swap_or_lock_assignment(&mut res, 0, None, false).expect("unassign succeeds");
    assert_eq!(res.assignments[0].peak_idx, None);
    assert_eq!(res.matched_count, 5);
    assert_eq!(res.unassigned_peaks, vec![0]);

    // Invalid group / peak return clean Err
    assert!(swap_or_lock_assignment(&mut res, 99, Some(0), true).is_err());
    assert!(swap_or_lock_assignment(&mut res, 0, Some(99), true).is_err());
}

#[test]
fn degenerate_tie_reports_honest_ambiguous_confidence_and_handles_empty_inputs() {
    let pred = vec![
        PredictedGroup1D {
            group_idx: 0,
            atom_indices: vec![0],
            label: "H-a".to_string(),
            shift_ppm: 7.20,
            sd_ppm: 0.35,
            expected_integral: 1.0,
            multiplicity: "d".to_string(),
            couplings_hz: vec![8.0],
        },
        PredictedGroup1D {
            group_idx: 1,
            atom_indices: vec![1],
            label: "H-b".to_string(),
            shift_ppm: 7.20,
            sd_ppm: 0.35,
            expected_integral: 1.0,
            multiplicity: "d".to_string(),
            couplings_hz: vec![8.0],
        },
    ];
    let exp = vec![
        ExperimentalPeak1D {
            peak_idx: 0,
            ppm: 7.20,
            integral: Some(1.0),
            multiplicity: Some("d".to_string()),
            couplings_hz: vec![8.0],
            intensity: 1.0,
        },
        ExperimentalPeak1D {
            peak_idx: 1,
            ppm: 7.20,
            integral: Some(1.0),
            multiplicity: Some("d".to_string()),
            couplings_hz: vec![8.0],
            intensity: 1.0,
        },
    ];
    let res = auto_assign_1d_to_structure(&pred, &exp, "1H", 500.0, "CDCl3");
    assert_eq!(res.matched_count, 2);
    for a in &res.assignments {
        assert!(
            (a.confidence - 0.5).abs() < 0.05,
            "Exact symmetric tie should yield ~0.50 confidence, got {:.3}",
            a.confidence
        );
    }

    // Empty inputs
    let empty = auto_assign_1d_to_structure(&[], &[], "1H", 500.0, "CDCl3");
    assert_eq!(empty.matched_count, 0);
    assert_eq!(empty.rmsd_ppm, 0.0);
    assert!(format_acs_with_atom_labels(&empty).contains("no assigned peaks"));
}

#[test]
fn extract_experimental_peaks_from_raw_custom_trace() {
    // Synthesize a clean 3-peak Lorentzian 1H spectrum trace at 7.40, 3.85, 1.25 ppm
    let true_peaks = [(7.40, 1.0), (3.85, 0.7), (1.25, 1.4)];
    let gamma = 0.015_f64;
    let n = 1000_usize;
    let trace: Vec<[f64; 2]> = (0..n)
        .map(|k| {
            let ppm = 10.0 - (k as f64 / (n - 1) as f64) * 10.0;
            let y: f64 = true_peaks
                .iter()
                .map(|&(p, amp)| {
                    let d = (ppm - p) / gamma;
                    amp / (1.0 + d * d)
                })
                .sum();
            [ppm, y]
        })
        .collect();

    let spec = SceneSpectrum1D {
        origin: [0.0, 0.0],
        width_mol: 10.0,
        height_mol: 3.0,
        nucleus: "1H".to_string(),
        spectrometer_mhz: 500.0,
        ppm_min: 0.0,
        ppm_max: 10.0,
        linewidth_hz: 1.5,
        show_integrals: true,
        show_peak_labels: true,
        title: Some("Raw Experimental 1H".to_string()),
        signals: Vec::new(),
        custom_trace: trace,
    };

    let peaks = extract_experimental_peaks_from_spectrum(&spec);
    assert_eq!(peaks.len(), 3, "Expected 3 picked peaks, got {peaks:?}");
    assert!((peaks[0].ppm - 7.40).abs() < 0.015);
    assert!((peaks[1].ppm - 3.85).abs() < 0.015);
    assert!((peaks[2].ppm - 1.25).abs() < 0.015);
}
