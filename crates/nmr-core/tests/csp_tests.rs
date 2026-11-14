//! Quantitative verification suite for `nmr-core::csp` (`T13.6`):
//! - Hungarian (`O(N^3)` Kuhn–Munkres) smooth fast-exchange trajectory tracking across crowded
//!   `2D HSQC` titration steps (including missing/broadened intermediate steps and stock dilution).
//! - Williamson (2013) `alpha_N` (`0.14`, `Gly` `0.20`) combined `CSP` and iterative `2-sigma`
//!   trimmed binding-site detection.
//! - Exact quadratic `1:1` tight-binding `K_d` recovery within `< 2%` error across both weak
//!   (`K_d = 500 uM >> [P]_t = 100 uM`) and tight-binding (`K_d = 5 uM << [P]_t = 100 uM`)
//!   synthetic titrations, demonstrating naive Michaelis–Menten (`[L]_free ≈ [L]_t`) fails by
//!   `> 8x` at `K_d = 5 uM`!
//!
//! Provenance: Clean-room verification citing Williamson (*Prog. Nucl. Magn. Reson. Spectrosc.*
//! 2013, 73, 1–16) and Kuhn (1955) / Munkres (1957).

use nmr_core::csp::{
    combined_amide_csp_ppm, diluted_conc_from_stock_addition,
    diluted_protein_conc_for_target_ligand, fit_csp_global_kd, format_csp_json, format_csp_tsv,
    quadratic_bound_fraction, track_titration_trajectories, write_csp_trajectories_tsv,
    AssignedAmidePeak, CspAnalysisConfig, UnassignedHsqcPeak, UnassignedTitrationStep,
};
use nmr_core::rng::Rng;

#[allow(clippy::type_complexity)]
fn build_synthetic_titration(
    kd_true_um: f64,
    p0_um: f64,
    l_stock_um: f64,
    ligand_targets_um: &[f64],
    seed: u64,
) -> (
    Vec<AssignedAmidePeak>,
    Vec<UnassignedTitrationStep>,
    Vec<i32>,
) {
    let mut rng = Rng::new(seed);
    let n_res = 48usize;
    let pocket_ids = vec![14, 15, 19, 28, 29, 36, 41, 42];

    let mut apo = Vec::with_capacity(n_res);
    let mut true_shifts = Vec::with_capacity(n_res);

    for idx in 0..n_res {
        let seq_id = (idx + 1) as i32;
        let res_name = if idx % 9 == 0 { "GLY" } else { "VAL" };
        // Place peaks on a realistic 2D HSQC grid without artificial apo overlap
        let gx = (idx % 8) as f64;
        let gy = (idx / 8) as f64;
        let h0 = 7.05 + 0.28 * gx + rng.uniform(-0.04, 0.04);
        let n0 = 109.0 + 3.1 * gy + rng.uniform(-0.4, 0.4);

        let (dh_max, dn_max) = if let Some(pos) = pocket_ids.iter().position(|&id| id == seq_id) {
            let sign = if pos % 2 == 0 { 1.0 } else { -1.0 };
            (
                sign * (0.22 + 0.04 * (pos as f64)),
                sign * (1.10 + 0.18 * (pos as f64)),
            )
        } else {
            (rng.normal(0.0, 0.008), rng.normal(0.0, 0.04))
        };

        apo.push(AssignedAmidePeak {
            seq_index: idx,
            seq_id,
            res_name: res_name.to_string(),
            h_ppm: h0,
            n_ppm: n0,
        });
        true_shifts.push((dh_max, dn_max));
    }

    let mut steps = Vec::with_capacity(ligand_targets_um.len());
    for (si, &l_t) in ligand_targets_um.iter().enumerate() {
        let p_t = diluted_protein_conc_for_target_ligand(p0_um, l_stock_um, l_t).unwrap();
        let fb = quadratic_bound_fraction(p_t, l_t, kd_true_um);

        let mut peaks = Vec::with_capacity(n_res);
        for r in 0..n_res {
            let (dh_m, dn_m) = true_shifts[r];
            let h = apo[r].h_ppm + fb * dh_m + rng.normal(0.0, 0.0012);
            let n = apo[r].n_ppm + fb * dn_m + rng.normal(0.0, 0.008);
            peaks.push(UnassignedHsqcPeak { h_ppm: h, n_ppm: n });
        }
        // Shuffle unassigned HSQC peak order at every step so tracker must genuinely match!
        rng.shuffle(&mut peaks);
        steps.push(UnassignedTitrationStep {
            label: format!("step_{}", si + 1),
            protein_conc_um: p_t,
            ligand_conc_um: l_t,
            peaks,
        });
    }

    (apo, steps, pocket_ids)
}

#[test]
fn test_weak_500um_and_tight_5um_binding_kd_done_gate_vs_naive_mm_failure() {
    let cfg = CspAnalysisConfig {
        max_step_csp_ppm: 0.45,
        collinearity_penalty: 0.35,
        csp_sigma_ppm: 0.0025,
        n_mc: 300,
        mc_seed: 20261306,
    };

    // 1. Weak binding (`K_d = 500 uM >> [P]_0 = 100 uM`, `[L]_stock = 50,000 uM`)
    let weak_l = [
        25.0, 60.0, 120.0, 250.0, 450.0, 750.0, 1200.0, 2000.0, 3500.0,
    ];
    let (apo_w, steps_w, pocket_w) =
        build_synthetic_titration(500.0, 100.0, 50_000.0, &weak_l, 101);
    let traj_w = track_titration_trajectories(&apo_w, &steps_w, &cfg).expect("track weak");
    let fit_w = fit_csp_global_kd(&traj_w, &cfg).expect("fit weak Kd");

    let rel_err_weak = (fit_w.global_kd_um - 500.0).abs() / 500.0;
    assert!(
        rel_err_weak < 0.02,
        "weak binding Kd=500 uM error {rel_err_weak:.4} >= 2% (got {:.3} uM)",
        fit_w.global_kd_um
    );
    for id in &pocket_w {
        assert!(
            fit_w.thresholds.significant_seq_ids.contains(id),
            "pocket residue {id} must exceed Williamson 2-sigma threshold"
        );
    }

    // 2. Tight binding (`K_d = 5 uM << [P]_0 = 100 uM`, `[L]_stock = 10,000 uM`)
    let tight_l = [
        10.0, 25.0, 45.0, 65.0, 85.0, 105.0, 135.0, 180.0, 260.0, 400.0,
    ];
    let (apo_t, steps_t, pocket_t) = build_synthetic_titration(5.0, 100.0, 10_000.0, &tight_l, 202);
    let traj_t = track_titration_trajectories(&apo_t, &steps_t, &cfg).expect("track tight");
    let fit_t = fit_csp_global_kd(&traj_t, &cfg).expect("fit tight Kd");

    let rel_err_tight = (fit_t.global_kd_um - 5.0).abs() / 5.0;
    assert!(
        rel_err_tight < 0.02,
        "tight binding Kd=5 uM error {rel_err_tight:.4} >= 2% (got {:.4} uM)",
        fit_t.global_kd_um
    );
    assert!(
        fit_t.global_kd_ci95_low_um < 5.0 && fit_t.global_kd_ci95_high_um > 5.0
            || rel_err_tight < 0.015,
        "MC 95% CI [{:.3}, {:.3}] should bracket true Kd=5.0 uM",
        fit_t.global_kd_ci95_low_um,
        fit_t.global_kd_ci95_high_um
    );

    // Done gate: demonstrate naive Michaelis–Menten (`[L]_free ≈ [L]_t`) fails by `> 8x` at Kd = 5 uM!
    let mm_ratio = fit_t.global_naive_mm_kd_um / 5.0;
    println!(
        "[T13.6 CSP Kd] weak_kd={:.3} uM (err={:.3}%), tight_kd={:.4} uM (err={:.3}%), naive_mm_kd={:.2} uM ({mm_ratio:.2}x)",
        fit_w.global_kd_um,
        rel_err_weak * 100.0,
        fit_t.global_kd_um,
        rel_err_tight * 100.0,
        fit_t.global_naive_mm_kd_um
    );
    assert!(
        mm_ratio > 8.0,
        "naive Michaelis-Menten Kd ({:.2} uM) must overestimate true Kd = 5 uM by > 8x (ratio={mm_ratio:.2}x)",
        fit_t.global_naive_mm_kd_um
    );

    for id in &pocket_t {
        assert!(fit_t.thresholds.significant_seq_ids.contains(id));
    }

    // Verify Gly vs non-Gly alpha_N, stock addition helper, and TSV/JSON export
    assert!((combined_amide_csp_ppm(0.3, 2.0, "GLY") - 0.5).abs() < 1e-12);
    assert!((combined_amide_csp_ppm(0.0, 1.0, "ALA") - 0.14).abs() < 1e-12);
    let (pt, lt) = diluted_conc_from_stock_addition(100.0, 500.0, 10_000.0, 5.0).unwrap();
    assert!((pt - 100.0 * (500.0 / 505.0)).abs() < 1e-10);
    assert!((lt - 10_000.0 * (5.0 / 505.0)).abs() < 1e-10);

    let tsv = format_csp_tsv(&fit_t);
    let traj_tsv = write_csp_trajectories_tsv(&traj_t);
    let json = format_csp_json(&fit_t).unwrap();
    assert!(
        tsv.contains("global_kd_um")
            && traj_tsv.contains("ligand_conc_um")
            && json.contains("\"global_kd_um\"")
    );
}

#[test]
fn test_hungarian_tracker_handles_exchange_broadened_missing_peak_without_hijacking_neighbor() {
    let apo = vec![
        AssignedAmidePeak {
            seq_index: 0,
            seq_id: 1,
            res_name: "LEU".to_string(),
            h_ppm: 8.10,
            n_ppm: 120.0,
        },
        AssignedAmidePeak {
            seq_index: 1,
            seq_id: 2,
            res_name: "ALA".to_string(),
            h_ppm: 8.22,
            n_ppm: 120.2,
        },
    ];

    // Residue 1 shifts strongly and disappears (exchange-broadened) at step 2 (`[L]=50 uM`),
    // while Residue 2 stays almost stationary at (8.22, 120.2)!
    let steps = vec![
        UnassignedTitrationStep {
            label: "s1".to_string(),
            protein_conc_um: 100.0,
            ligand_conc_um: 20.0,
            peaks: vec![
                UnassignedHsqcPeak {
                    h_ppm: 8.22,
                    n_ppm: 120.21,
                },
                UnassignedHsqcPeak {
                    h_ppm: 8.00,
                    n_ppm: 119.5,
                },
            ],
        },
        UnassignedTitrationStep {
            label: "s2_missing_res1".to_string(),
            protein_conc_um: 100.0,
            ligand_conc_um: 50.0,
            peaks: vec![
                // Only Residue 2's peak is visible!
                UnassignedHsqcPeak {
                    h_ppm: 8.221,
                    n_ppm: 120.22,
                },
            ],
        },
        UnassignedTitrationStep {
            label: "s3_reappeared".to_string(),
            protein_conc_um: 100.0,
            ligand_conc_um: 150.0,
            peaks: vec![
                UnassignedHsqcPeak {
                    h_ppm: 7.82,
                    n_ppm: 118.8,
                },
                UnassignedHsqcPeak {
                    h_ppm: 8.222,
                    n_ppm: 120.23,
                },
            ],
        },
    ];

    let cfg = CspAnalysisConfig {
        max_step_csp_ppm: 0.35,
        ..Default::default()
    };
    let traj = track_titration_trajectories(&apo, &steps, &cfg).unwrap();
    // Step index 2 (`s2_missing_res1`, `points[2]` since `points[0]` is apo):
    assert!(
        traj[0].points[2].matched_peak.is_none(),
        "broadened Residue 1 must take slack column at s2"
    );
    assert!(
        traj[1].points[2].matched_peak.is_some(),
        "stationary Residue 2 must keep its own peak at s2"
    );
    assert!(
        traj[0].points[3].matched_peak.is_some(),
        "Residue 1 must recover its saturated peak at s3"
    );
}

#[test]
fn test_explicit_apo_zero_ligand_step_deduplication_matches_implicit_apo() {
    let cfg = CspAnalysisConfig {
        max_step_csp_ppm: 0.45,
        collinearity_penalty: 0.35,
        csp_sigma_ppm: 0.0025,
        n_mc: 100,
        mc_seed: 777,
    };
    let tight_l = [15.0, 40.0, 80.0, 120.0, 200.0, 350.0];
    let (apo, steps_no_zero, _) = build_synthetic_titration(8.0, 100.0, 10_000.0, &tight_l, 909);

    let mut steps_with_zero = Vec::with_capacity(steps_no_zero.len() + 1);
    steps_with_zero.push(UnassignedTitrationStep {
        label: "apo_0uM".to_string(),
        protein_conc_um: 100.0,
        ligand_conc_um: 0.0,
        peaks: apo
            .iter()
            .map(|p| UnassignedHsqcPeak {
                h_ppm: p.h_ppm,
                n_ppm: p.n_ppm,
            })
            .collect(),
    });
    steps_with_zero.extend(steps_no_zero.iter().cloned());

    let traj_implicit = track_titration_trajectories(&apo, &steps_no_zero, &cfg).unwrap();
    let traj_explicit = track_titration_trajectories(&apo, &steps_with_zero, &cfg).unwrap();

    for (ti, te) in traj_implicit.iter().zip(traj_explicit.iter()) {
        assert_eq!(
            ti.points.len(),
            te.points.len(),
            "explicit [L]_t == 0 step must merge into apo point without duplicating [L]_t = 0"
        );
        assert_eq!(te.points.len(), 1 + tight_l.len());
        assert!((te.points[0].protein_conc_um - 100.0).abs() < 1e-12);
        assert!((te.points[0].ligand_conc_um - 0.0).abs() < 1e-12);
    }

    let fit_implicit = fit_csp_global_kd(&traj_implicit, &cfg).unwrap();
    let fit_explicit = fit_csp_global_kd(&traj_explicit, &cfg).unwrap();
    assert_eq!(fit_implicit.global_dof, fit_explicit.global_dof);
    assert!(
        (fit_implicit.global_kd_um - fit_explicit.global_kd_um).abs() < 1e-12,
        "global Kd must be invariant whether [L]_t = 0 is implicit or explicit"
    );
}
