//! `[synthetic self-consistency]` integration tests for `nmr_core::titration_2d` (`T22.8`):
//! cross-series peak trajectory tracking across crossing paths & intermediate-exchange dropouts,
//! per-titration-step exchange-regime verification, exact 2x2 eigenmode vs numerical time-domain
//! 2D States-TPPI FFT lineshape parity, global multi-residue 2D Bloch–McConnell recovery of
//! `(K_d, k_off)` in intermediate exchange, and hostile-input fail-closed validation.

use std::f64::consts::TAU;

use nmr_core::rng::Rng;
use nmr_core::titration_2d::{
    analytical_2d_integrated_volume, binding_equilibrium_rates, classify_alpha,
    classify_exchange_regime, fit_titration_2d_lineshape, simulate_2d_lineshape_analytical,
    simulate_2d_lineshape_eigenmodes, simulate_2d_lineshape_time_domain_fft,
    track_titration_trajectories, AmbiguityReason, Bloch2dKineticState, Bloch2dResidueParams,
    ExchangeRegime, ExchangeRegimeInput, Peak2d, ResidueTitration2dSeries, Roi2dGrid, Roi2dPatch,
    SeriesPeakList, Titration2dError, Titration2dFitConfig, TitrationConcentrationPoint,
    TrackStepStatus, TrajectoryTrackConfig, GAMMA_N15_OVER_H1,
};

fn pearson_correlation(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let n = a.len() as f64;
    let ma = a.iter().sum::<f64>() / n;
    let mb = b.iter().sum::<f64>() / n;
    let mut num = 0.0;
    let mut da = 0.0;
    let mut db = 0.0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let vx = x - ma;
        let vy = y - mb;
        num += vx * vy;
        da += vx * vx;
        db += vy * vy;
    }
    num / (da * db).sqrt()
}

fn normalized_l2_residual(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut num = 0.0;
    let mut den = 0.0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let d = x - y;
        num += d * d;
        den += x * x;
    }
    (num / den).sqrt()
}

/// Simple naive nearest-neighbor tracker (no velocity extrapolation) to prove that crossing
/// trajectories in our benchmark genuinely break nearest-neighbor matching.
fn track_naive_nearest_neighbor(steps: &[SeriesPeakList], alpha_n: f64) -> Vec<Vec<Option<usize>>> {
    let n_tracks = steps[0].peaks.len();
    let mut ids: Vec<Vec<Option<usize>>> = vec![Vec::with_capacity(steps.len()); n_tracks];
    let mut last_pos: Vec<(f64, f64)> = steps[0].peaks.iter().map(|p| (p.h_ppm, p.n_ppm)).collect();
    for (r, pk) in steps[0].peaks.iter().enumerate() {
        ids[r].push(Some(pk.peak_id));
    }

    for step in steps.iter().skip(1) {
        let mut used = vec![false; step.peaks.len()];
        for r in 0..n_tracks {
            let (lh, ln) = last_pos[r];
            let mut best_c: Option<usize> = None;
            let mut best_d = f64::INFINITY;
            for (c, pk) in step.peaks.iter().enumerate() {
                if used[c] {
                    continue;
                }
                let d = (pk.h_ppm - lh).hypot(alpha_n * (pk.n_ppm - ln));
                if d < best_d && d < 0.45 {
                    best_d = d;
                    best_c = Some(c);
                }
            }
            if let Some(c) = best_c {
                used[c] = true;
                last_pos[r] = (step.peaks[c].h_ppm, step.peaks[c].n_ppm);
                ids[r].push(Some(step.peaks[c].peak_id));
            } else {
                ids[r].push(None);
            }
        }
    }
    ids
}

// ============================================================================
// 1. Cross-Series Peak Trajectory Tracking Tests [synthetic self-consistency]
// ============================================================================

#[test]
fn synthetic_crossing_trajectories_and_midpoint_dropout_100pct_recovery() {
    // 8-point ligand titration with:
    // - Crossing Pair 1 (tracks 0 & 1): cross directly through each other between steps 3 and 4
    //   at an acute angle so naive nearest-neighbor swaps them!
    // - Crossing Pair 2 (tracks 2 & 3): cross directly through each other between steps 2 and 3
    //   in the opposite quadrant so naive nearest-neighbor also swaps them!
    // - Dropout Track (track 4): moving steadily, broadens out / drops at midpoint step 3 and
    //   reappears at step 4 along its extrapolated trajectory!
    // - Spectator Tracks (tracks 5 & 6): stationary reference peaks.
    let coords = [0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0];
    let alpha_n = 0.14;

    // Define exact ground-truth trajectories `(h0, n0, dh_per_step, dn_per_step)`
    // Note: in normalized `(h, 0.14*n)` space, Crossing Pair 1 crosses at t = 3.5:
    //   Track 0: h(t) = 7.80 + 0.060*t, n(t) = 118.0 + 0.40*t  (dn_norm = +0.056/step)
    //   Track 1: h(t) = 7.80 + 0.060*t, n(t) = 120.8 - 0.40*t  (dn_norm = -0.056/step)
    // Check t=3 -> t=4:
    //   t=3: T0 = (7.98, 119.2), T1 = (7.98, 119.6)
    //   t=4: T0 = (8.04, 119.6), T1 = (8.04, 119.2)
    //   Naive NN distance from T0(3) to T1(4) = 0.060 < distance to T0(4) = hypot(0.060, 0.056) = 0.082!
    //   So naive NN ALWAYS swaps T0 and T1!
    // Crossing Pair 2 crosses at t = 2.5:
    //   Track 2: h(t) = 8.90 - 0.050*t, n(t) = 124.0 + 0.45*t
    //   Track 3: h(t) = 8.90 - 0.050*t, n(t) = 126.25 - 0.45*t
    //   At t=2: T2=(8.80, 124.90), T3=(8.80, 125.35); at t=3: T2=(8.75, 125.35), T3=(8.75, 124.90)!
    //   Again, naive NN ALWAYS swaps T2 and T3!
    let truth: [(&str, &str, f64, f64, f64, f64); 7] = [
        ("A10", "ALA", 7.80, 118.00, 0.060, 0.40),
        ("L11", "LEU", 7.80, 120.80, 0.060, -0.40),
        ("V20", "VAL", 8.90, 124.00, -0.050, 0.45),
        ("K21", "LYS", 8.90, 126.25, -0.050, -0.45),
        ("D30", "ASP", 8.20, 115.00, 0.045, 0.35), // drops out at step 3!
        ("S40", "SER", 7.40, 113.00, 0.000, 0.00),
        ("E50", "GLU", 9.30, 128.00, 0.002, -0.02),
    ];

    let mut steps = Vec::with_capacity(coords.len());
    for (t, &c) in coords.iter().enumerate() {
        let mut peaks = Vec::new();
        for (id, &(lbl, res, h0, n0, dh, dn)) in truth.iter().enumerate() {
            // Simulate intermediate-exchange line-broadening dropout of track 4 at midpoint t = 3.
            if id == 4 && t == 3 {
                continue;
            }
            let tf = t as f64;
            peaks.push(Peak2d {
                peak_id: id,
                h_ppm: h0 + dh * tf,
                n_ppm: n0 + dn * tf,
                intensity: Some(1.0),
                label: if t == 0 { Some(lbl.to_string()) } else { None },
                res_name: if t == 0 { Some(res.to_string()) } else { None },
            });
        }
        // Reverse peak ordering on odd steps to ensure tracker cannot rely on input index order!
        if t % 2 == 1 {
            peaks.reverse();
        }
        steps.push(SeriesPeakList {
            label: format!("step_{t}"),
            series_coord: c,
            protein_conc_um: Some(50.0),
            ligand_conc_um: Some(c),
            peaks,
        });
    }

    // 1. Verify naive nearest-neighbor FAILS by swapping both crossing pairs!
    let nn_ids = track_naive_nearest_neighbor(&steps, alpha_n);
    assert_eq!(
        nn_ids[0][7],
        Some(1),
        "naive NN must swap crossing pair 1 (track 0 -> peak 1)"
    );
    assert_eq!(
        nn_ids[1][7],
        Some(0),
        "naive NN must swap crossing pair 1 (track 1 -> peak 0)"
    );
    assert_eq!(
        nn_ids[2][7],
        Some(3),
        "naive NN must swap crossing pair 2 (track 2 -> peak 3)"
    );
    assert_eq!(
        nn_ids[3][7],
        Some(2),
        "naive NN must swap crossing pair 2 (track 3 -> peak 2)"
    );

    // 2. Verify velocity-extrapolated Kuhn–Munkres recovers 100% of identities + dropout!
    let cfg = TrajectoryTrackConfig::default();
    let res = track_titration_trajectories(&steps, &cfg).expect("tracking must succeed");
    assert_eq!(res.trajectories.len(), truth.len());

    let mut total_present = 0usize;
    let mut total_correct = 0usize;
    for (id, traj) in res.trajectories.iter().enumerate() {
        assert_eq!(traj.track_id, id);
        assert_eq!(traj.observations.len(), coords.len());
        for (t, obs) in traj.observations.iter().enumerate() {
            if id == 4 && t == 3 {
                match obs.status {
                    TrackStepStatus::DroppedOut {
                        consecutive_misses, ..
                    } => assert_eq!(consecutive_misses, 1),
                    ref other => panic!("track 4 at step 3 should be DroppedOut, got {other:?}"),
                }
            } else {
                total_present += 1;
                match obs.status {
                    TrackStepStatus::Matched {
                        peak_id,
                        reappeared_after_gap,
                        ..
                    } => {
                        if peak_id == id {
                            total_correct += 1;
                        }
                        if id == 4 && t == 4 {
                            assert_eq!(
                                reappeared_after_gap, 1,
                                "track 4 must record reappearance after 1-step dropout"
                            );
                        }
                    }
                    ref other => panic!("track {id} at step {t} should be Matched, got {other:?}"),
                }
            }
        }
    }
    assert_eq!(
        total_correct, total_present,
        "velocity-extrapolated Kuhn-Munkres must recover 100% ({total_present}/{total_present}) of identities"
    );
    assert_eq!(res.trajectories[4].n_dropouts, 1);
    assert_eq!(res.trajectories[4].n_matched, 7);
}

#[test]
fn synthetic_near_tie_crossing_emits_ambiguity_flag_and_supports_cooling_series() {
    // Two peaks converging to nearly the same spot at step 2 in a descending temperature series
    // (310 K -> 290 K).
    let temps = [310.0, 305.0, 300.0, 295.0, 290.0];
    let mut steps = Vec::new();
    for (t, &temp) in temps.iter().enumerate() {
        let tf = t as f64;
        // At t = 2: peak 0 is at (8.200, 120.000), peak 1 is at (8.204, 120.010) -> sep ~ 0.0042 ppm!
        let p0 = Peak2d {
            peak_id: 0,
            h_ppm: 8.100 + 0.050 * tf,
            n_ppm: 119.60 + 0.20 * tf,
            intensity: None,
            label: if t == 0 { Some("R1".to_string()) } else { None },
            res_name: Some("ALA".to_string()),
        };
        let p1 = Peak2d {
            peak_id: 1,
            h_ppm: 8.104 + 0.050 * tf,
            n_ppm: 120.41 - 0.20 * tf,
            intensity: None,
            label: if t == 0 { Some("R2".to_string()) } else { None },
            res_name: Some("ALA".to_string()),
        };
        steps.push(SeriesPeakList {
            label: format!("{temp}K"),
            series_coord: temp,
            protein_conc_um: None,
            ligand_conc_um: None,
            peaks: vec![p0, p1],
        });
    }

    let cfg = TrajectoryTrackConfig::default();
    let res = track_titration_trajectories(&steps, &cfg).expect("cooling series must succeed");
    assert!(
        res.ambiguity_flags.iter().any(|f| {
            f.step_index == 2
                && matches!(
                    f.reason,
                    AmbiguityReason::NearTieCrossingSwap | AmbiguityReason::CloseCrossingProximity
                )
        }),
        "near-coincident crossing at step 2 must emit TrajectoryAmbiguityFlag; got {:?}",
        res.ambiguity_flags
    );
}

// ============================================================================
// 2. Per-Titration-Step Exchange-Regime Verification [synthetic self-consistency]
// ============================================================================

#[test]
fn synthetic_exchange_regime_classification_and_csp_warnings() {
    // Threshold boundaries
    assert_eq!(classify_alpha(0.0), ExchangeRegime::Fast);
    assert_eq!(classify_alpha(0.2999), ExchangeRegime::Fast);
    assert_eq!(classify_alpha(0.3), ExchangeRegime::Intermediate);
    assert_eq!(classify_alpha(1.5), ExchangeRegime::Intermediate);
    assert_eq!(classify_alpha(3.0), ExchangeRegime::Intermediate);
    assert_eq!(classify_alpha(3.0001), ExchangeRegime::Slow);

    let pts: Vec<TitrationConcentrationPoint> = [0.0, 10.0, 25.0, 50.0, 100.0, 250.0, 600.0]
        .iter()
        .enumerate()
        .map(|(i, &l)| TitrationConcentrationPoint {
            step_index: i,
            label: format!("{l} uM"),
            protein_conc_um: 50.0,
            ligand_conc_um: l,
        })
        .collect();

    // Case A: Pure fast exchange on both 1H and 15N (k_off = 3000 s^-1, small shifts)
    let fast_rep = classify_exchange_regime(
        &ExchangeRegimeInput {
            label: "G15".to_string(),
            kd_um: 50.0,
            k_off_s_inv: 3000.0,
            dw_h_ppm: 0.04,
            dw_n_ppm: 0.25,
            h_larmor_mhz: 600.0,
            n_larmor_mhz: None,
        },
        &pts,
    )
    .expect("fast regime classification must succeed");
    assert!(fast_rep.fast_exchange_csp_valid);
    assert!(fast_rep.warnings.is_empty());
    assert_eq!(fast_rep.overall_worst_regime, ExchangeRegime::Fast);

    // Check exact detailed balance p_A * k_AB == p_B * k_BA at every titration step!
    for s in &fast_rep.steps {
        let lhs = s.p_a * s.k_ab_s_inv;
        let rhs = s.p_b * s.k_ba_s_inv;
        assert!(
            (lhs - rhs).abs() < 1e-10 * (1.0 + rhs),
            "detailed balance violated at step {}: {lhs} vs {rhs}",
            s.step_index
        );
        assert!((s.k_ex_s_inv - (s.k_ab_s_inv + s.k_ba_s_inv)).abs() < 1e-12);
    }

    // Case B: Slow -> Intermediate -> Fast transition across titration!
    // At [L]=0: k_ex = k_off = 120 s^-1.
    // dw_H = 0.12 ppm @ 600 MHz -> |dw_H| = 452.4 rad/s -> alpha_H(0) = 3.77 (Slow!)
    // At [L]=600 uM, K_d=25 uM: [L]_free ~ 552 uM -> k_ex ~ 2770 s^-1 -> alpha_H(6) = 0.16 (Fast!)
    // dw_N = 0.45 ppm @ 60.8 MHz -> |dw_N| = 171.9 rad/s -> alpha_N(0) = 1.43 (Intermediate!)
    let mixed_rep = classify_exchange_regime(
        &ExchangeRegimeInput {
            label: "I44".to_string(),
            kd_um: 25.0,
            k_off_s_inv: 120.0,
            dw_h_ppm: 0.12,
            dw_n_ppm: 0.45,
            h_larmor_mhz: 600.0,
            n_larmor_mhz: None,
        },
        &pts,
    )
    .expect("mixed regime classification must succeed");

    assert_eq!(mixed_rep.steps[0].regime_h, ExchangeRegime::Slow);
    assert_eq!(mixed_rep.steps[0].regime_n, ExchangeRegime::Intermediate);
    assert_eq!(mixed_rep.steps[0].overall_regime, ExchangeRegime::Slow);
    // Mid-titration step should be Intermediate on 1H
    assert_eq!(mixed_rep.steps[3].regime_h, ExchangeRegime::Intermediate);
    // Saturating step transitions into Fast exchange on both axes!
    assert_eq!(mixed_rep.steps[6].regime_h, ExchangeRegime::Fast);
    assert_eq!(mixed_rep.steps[6].regime_n, ExchangeRegime::Fast);
    assert!(!mixed_rep.fast_exchange_csp_valid);
    assert_eq!(mixed_rep.warnings.len(), 1);
    assert!(
        mixed_rep.warnings[0]
            .message
            .contains("fast-exchange quadratic CSP"),
        "warning must explicitly flag fast-exchange CSP invalidity"
    );
}

// ============================================================================
// 3. Exact 2x2 Eigenmode Analytical vs Numerical Time-Domain 2D FFT Verification
// ============================================================================

#[test]
fn synthetic_2d_lineshape_analytical_vs_time_domain_fft_across_all_regimes() {
    let h_mhz = 600.0;
    let n_mhz = h_mhz * GAMMA_N15_OVER_H1;
    let carrier_n = 120.0;
    let carrier_h = 8.20;

    // Fine sampling + long acquisition so discretization & truncation error are << 1e-4:
    // N1 = N2 = 1024, SW1 = SW2 = 6400 Hz -> dt = 0.15625 ms, Tmax = 0.16 s.
    // With R2 >= 65 s^-1, exp(-65 * 0.16) = 3.0e-5 (negligible truncation & Nyquist aliasing).
    let n1 = 1024usize;
    let n2 = 1024usize;
    let sw1_hz = 6400.0;
    let sw2_hz = 6400.0;

    // Three regimes:
    // 1. Fast exchange: k_ex = 1600 s^-1 (alpha_H ~ 0.12 < 0.3, alpha_N ~ 0.09 < 0.3)
    // 2. Intermediate exchange RIGHT AT THE EXACT EXCEPTIONAL POINT in F2 (1H)!
    //    p_A = p_B = 0.5, k_AB = k_BA = 150 s^-1, R2A = R2B = 65 s^-1,
    //    |dw_H| = 300 rad/s = 2 * k_AB -> exact coalescence exceptional point delta_2 = 0!
    // 3. Slow exchange: k_ex = 20 s^-1 (alpha_H ~ 18.8, alpha_N ~ 11.5)
    let dw_h_exact_ep_ppm = 300.0 / (TAU * h_mhz);
    let dw_n_inter_ppm = 220.0 / (TAU * n_mhz);

    let cases: [(&str, Bloch2dResidueParams, Bloch2dKineticState); 3] = [
        (
            "fast_exchange",
            Bloch2dResidueParams {
                delta_a_h_ppm: carrier_h - 0.025,
                delta_b_h_ppm: carrier_h + 0.025,
                delta_a_n_ppm: carrier_n - 0.20,
                delta_b_n_ppm: carrier_n + 0.20,
                r2a_h_s_inv: 65.0,
                r2b_h_s_inv: 75.0,
                r2a_n_s_inv: 65.0,
                r2b_n_s_inv: 72.0,
                amplitude: 1000.0,
            },
            Bloch2dKineticState {
                p_a: 0.4,
                p_b: 0.6,
                k_ab_s_inv: 960.0,
                k_ba_s_inv: 640.0,
            },
        ),
        (
            "intermediate_coalescence_exceptional_point",
            Bloch2dResidueParams {
                delta_a_h_ppm: carrier_h - 0.5 * dw_h_exact_ep_ppm,
                delta_b_h_ppm: carrier_h + 0.5 * dw_h_exact_ep_ppm,
                delta_a_n_ppm: carrier_n - 0.5 * dw_n_inter_ppm,
                delta_b_n_ppm: carrier_n + 0.5 * dw_n_inter_ppm,
                r2a_h_s_inv: 65.0,
                r2b_h_s_inv: 65.0,
                r2a_n_s_inv: 65.0,
                r2b_n_s_inv: 65.0,
                amplitude: 1000.0,
            },
            Bloch2dKineticState {
                p_a: 0.5,
                p_b: 0.5,
                k_ab_s_inv: 150.0,
                k_ba_s_inv: 150.0,
            },
        ),
        (
            "slow_exchange_doublet",
            Bloch2dResidueParams {
                delta_a_h_ppm: carrier_h - 0.06,
                delta_b_h_ppm: carrier_h + 0.06,
                delta_a_n_ppm: carrier_n - 0.45,
                delta_b_n_ppm: carrier_n + 0.45,
                r2a_h_s_inv: 65.0,
                r2b_h_s_inv: 78.0,
                r2a_n_s_inv: 65.0,
                r2b_n_s_inv: 72.0,
                amplitude: 1000.0,
            },
            Bloch2dKineticState {
                p_a: 0.65,
                p_b: 0.35,
                k_ab_s_inv: 7.0,
                k_ba_s_inv: 13.0,
            },
        ),
    ];

    for (name, params, state) in &cases {
        let (full_grid, fft_spec) = simulate_2d_lineshape_time_domain_fft(
            n1, n2, sw1_hz, sw2_hz, carrier_n, carrier_h, h_mhz, n_mhz, params, state,
        )
        .expect("time-domain FFT must succeed");

        // Extract central ROI (+- 48 bins around carrier, covering +-150 Hz around both states).
        let i_lo = n1 / 2 - 48;
        let i_hi = n1 / 2 + 48;
        let j_lo = n2 / 2 - 48;
        let j_hi = n2 / 2 + 48;
        let roi_grid = Roi2dGrid {
            n_ppm_axis: full_grid.n_ppm_axis[i_lo..i_hi].to_vec(),
            h_ppm_axis: full_grid.h_ppm_axis[j_lo..j_hi].to_vec(),
            h_larmor_mhz: h_mhz,
            n_larmor_mhz: n_mhz,
        };
        let mut fft_roi = Vec::with_capacity((i_hi - i_lo) * (j_hi - j_lo));
        for i in i_lo..i_hi {
            for j in j_lo..j_hi {
                fft_roi.push(fft_spec[i * n2 + j]);
            }
        }

        let ana_roi = simulate_2d_lineshape_analytical(&roi_grid, params, state)
            .expect("analytical simulation must succeed");
        let eig_roi = simulate_2d_lineshape_eigenmodes(&roi_grid, params, state)
            .expect("eigenmode simulation must succeed");

        // 1. Exact agreement between resolvent analytical and eigenmode double-sum (< 1e-12)
        let eig_vs_ana = normalized_l2_residual(&ana_roi, &eig_roi);
        assert!(
            eig_vs_ana < 1e-12,
            "{name}: eigenmode vs analytical residual too large: {eig_vs_ana:.3e}"
        );

        // 2. Agreement between analytical 2x2 eigenmode and numerical expm(L1 t1)+expm(L2 t2) FFT (< 1e-4)
        let fft_vs_ana = normalized_l2_residual(&ana_roi, &fft_roi);
        let corr = pearson_correlation(&ana_roi, &fft_roi);
        assert!(
            fft_vs_ana < 1e-4,
            "{name}: analytical vs time-domain FFT normalized L2 residual = {fft_vs_ana:.3e} (must be < 1e-4)"
        );
        assert!(
            corr > 0.999_99,
            "{name}: analytical vs time-domain FFT correlation = {corr:.8} (must be > 0.99999)"
        );

        // 3. Verify slow-exchange produces diagonal peaks only (zero false off-diagonal peaks!)
        if *name == "slow_exchange_doublet" {
            let probe_grid = Roi2dGrid {
                n_ppm_axis: vec![params.delta_a_n_ppm, params.delta_b_n_ppm],
                h_ppm_axis: vec![params.delta_a_h_ppm, params.delta_b_h_ppm],
                h_larmor_mhz: h_mhz,
                n_larmor_mhz: n_mhz,
            };
            let p2x2 = simulate_2d_lineshape_analytical(&probe_grid, params, state)
                .expect("probe grid must succeed");
            let diag_a = p2x2[0]; // (wA1, wA2)
            let off_ab = p2x2[1]; // (wA1, wB2) - must be tiny (exchange tail only)
            let off_ba = p2x2[2]; // (wB1, wA2) - must be tiny
            let diag_b = p2x2[3]; // (wB1, wB2)
            assert!(diag_a > 10.0 * off_ab && diag_a > 10.0 * off_ba);
            assert!(diag_b > 5.0 * off_ab && diag_b > 5.0 * off_ba);
        }

        // 4. Verify analytical 2D volume helper matches Riemann sum over the full FFT grid (~1%).
        let d_omega1 = TAU * sw1_hz / (n1 as f64);
        let d_omega2 = TAU * sw2_hz / (n2 as f64);
        let num_vol: f64 = fft_spec.iter().sum::<f64>() * d_omega1 * d_omega2;
        let exact_vol = analytical_2d_integrated_volume(params.amplitude);
        let vol_rel = (num_vol - exact_vol).abs() / exact_vol;
        assert!(
            vol_rel < 0.05,
            "{name}: 2D integrated volume mismatch: num={num_vol:.2}, exact={exact_vol:.2} (rel={vol_rel:.3})"
        );
    }
}

// ============================================================================
// 4. Global 2D Bloch–McConnell Recovery of (K_d = 25 uM, k_off = 120 s^-1)
// ============================================================================

#[test]
fn synthetic_global_2d_bloch_mcconnell_recovers_intermediate_exchange_kd_and_koff() {
    let true_kd_um = 25.0;
    let true_koff_s_inv = 120.0;
    let h_mhz = 600.0;
    let n_mhz = h_mhz * GAMMA_N15_OVER_H1;
    let p_tot_um = 50.0;
    let ligand_concs = [0.0, 10.0, 25.0, 45.0, 75.0, 120.0, 200.0, 350.0];

    // 4 residues spanning intermediate exchange (alpha ~ 0.6 .. 2.8 at mid-titration)
    let true_residues: [(&str, &str, Bloch2dResidueParams); 4] = [
        (
            "L18",
            "LEU",
            Bloch2dResidueParams {
                delta_a_h_ppm: 8.12,
                delta_b_h_ppm: 8.21, // dw_H = +0.09 ppm (~339 rad/s)
                delta_a_n_ppm: 121.20,
                delta_b_n_ppm: 121.85, // dw_N = +0.65 ppm (~248 rad/s)
                r2a_h_s_inv: 24.0,
                r2b_h_s_inv: 32.0,
                r2a_n_s_inv: 18.0,
                r2b_n_s_inv: 25.0,
                amplitude: 5000.0,
            },
        ),
        (
            "G29",
            "GLY",
            Bloch2dResidueParams {
                delta_a_h_ppm: 8.55,
                delta_b_h_ppm: 8.48, // dw_H = -0.07 ppm (~264 rad/s)
                delta_a_n_ppm: 109.40,
                delta_b_n_ppm: 108.60, // dw_N = -0.80 ppm (~306 rad/s)
                r2a_h_s_inv: 22.0,
                r2b_h_s_inv: 29.0,
                r2a_n_s_inv: 16.0,
                r2b_n_s_inv: 22.0,
                amplitude: 4200.0,
            },
        ),
        (
            "V42",
            "VAL",
            Bloch2dResidueParams {
                delta_a_h_ppm: 7.72,
                delta_b_h_ppm: 7.83, // dw_H = +0.11 ppm (~415 rad/s)
                delta_a_n_ppm: 118.80,
                delta_b_n_ppm: 119.30, // dw_N = +0.50 ppm (~191 rad/s)
                r2a_h_s_inv: 26.0,
                r2b_h_s_inv: 34.0,
                r2a_n_s_inv: 20.0,
                r2b_n_s_inv: 26.0,
                amplitude: 4800.0,
            },
        ),
        (
            "D61",
            "ASP",
            Bloch2dResidueParams {
                delta_a_h_ppm: 8.92,
                delta_b_h_ppm: 8.86, // dw_H = -0.06 ppm (~226 rad/s)
                delta_a_n_ppm: 123.60,
                delta_b_n_ppm: 124.30, // dw_N = +0.70 ppm (~267 rad/s)
                r2a_h_s_inv: 20.0,
                r2b_h_s_inv: 27.0,
                r2a_n_s_inv: 17.0,
                r2b_n_s_inv: 23.0,
                amplitude: 3900.0,
            },
        ),
    ];

    let mut rng = Rng::new(2026_2208);
    let mut series_vec = Vec::with_capacity(true_residues.len());

    for (label, res_name, true_p) in &true_residues {
        let n_min = true_p.delta_a_n_ppm.min(true_p.delta_b_n_ppm) - 0.55;
        let n_max = true_p.delta_a_n_ppm.max(true_p.delta_b_n_ppm) + 0.55;
        let h_min = true_p.delta_a_h_ppm.min(true_p.delta_b_h_ppm) - 0.08;
        let h_max = true_p.delta_a_h_ppm.max(true_p.delta_b_h_ppm) + 0.08;
        let grid = Roi2dGrid::uniform(n_min, n_max, 21, h_min, h_max, 21, h_mhz)
            .expect("ROI grid must build");

        // Apo peak height is amplitude / (r2a_n * r2a_h); set SNR ~ 50 on apo peak.
        let apo_peak = true_p.amplitude / (true_p.r2a_n_s_inv * true_p.r2a_h_s_inv);
        let noise_sigma = apo_peak / 50.0;

        let mut patches = Vec::with_capacity(ligand_concs.len());
        for (step_idx, &l_um) in ligand_concs.iter().enumerate() {
            let eq = binding_equilibrium_rates(p_tot_um, l_um, true_kd_um, true_koff_s_inv)
                .expect("equilibrium must succeed");
            let state = Bloch2dKineticState::from(eq);
            let mut intensities = simulate_2d_lineshape_analytical(&grid, true_p, &state)
                .expect("simulation must succeed");
            for v in &mut intensities {
                *v += rng.normal(0.0, noise_sigma);
            }
            patches.push(Roi2dPatch {
                step_index: step_idx,
                protein_conc_um: p_tot_um,
                ligand_conc_um: l_um,
                grid: grid.clone(),
                intensities,
                noise_sigma,
                spectrum_scale: 1.0,
            });
        }

        series_vec.push(ResidueTitration2dSeries {
            label: (*label).to_string(),
            res_name: (*res_name).to_string(),
            patches,
            initial_guess: None, // Zero initial guess: must converge globally from ROI data alone!
        });
    }

    let cfg = Titration2dFitConfig::default();
    let fit = fit_titration_2d_lineshape(&series_vec, &cfg)
        .expect("global 2D Bloch-McConnell fit must succeed");

    let kd_rel_err = (fit.kd_um - true_kd_um).abs() / true_kd_um;
    let koff_rel_err = (fit.k_off_s_inv - true_koff_s_inv).abs() / true_koff_s_inv;

    assert!(
        fit.converged,
        "global LM fitter must converge (iters={})",
        fit.lm_iterations
    );
    assert!(
        kd_rel_err < 0.05,
        "global K_d recovery error {:.2}% exceeds 5% (fit={:.3} uM, true={true_kd_um} uM, sd={:.3})",
        kd_rel_err * 100.0,
        fit.kd_um,
        fit.kd_sd_um
    );
    assert!(
        koff_rel_err < 0.05,
        "global k_off recovery error {:.2}% exceeds 5% (fit={:.3} s^-1, true={true_koff_s_inv} s^-1, sd={:.3})",
        koff_rel_err * 100.0,
        fit.k_off_s_inv,
        fit.k_off_sd_s_inv
    );
    assert!(
        fit.reduced_chi2 > 0.85 && fit.reduced_chi2 < 1.15,
        "reduced chi^2 should be near 1.0 for Gaussian noise, got {:.4}",
        fit.reduced_chi2
    );
    assert!(fit.kd_sd_um.is_finite() && fit.kd_sd_um > 0.0);
    assert!(fit.k_off_sd_s_inv.is_finite() && fit.k_off_sd_s_inv > 0.0);
    assert!(fit.k_on_sd_um_inv_s_inv.is_finite() && fit.k_on_sd_um_inv_s_inv > 0.0);

    // Check per-residue parameter recovery
    for (fitted, (_, _, true_p)) in fit.residues.iter().zip(true_residues.iter()) {
        assert!((fitted.params.delta_a_h_ppm - true_p.delta_a_h_ppm).abs() < 0.002);
        assert!((fitted.params.delta_b_h_ppm - true_p.delta_b_h_ppm).abs() < 0.003);
        assert!((fitted.params.delta_a_n_ppm - true_p.delta_a_n_ppm).abs() < 0.015);
        assert!((fitted.params.delta_b_n_ppm - true_p.delta_b_n_ppm).abs() < 0.025);
        assert!(
            (fitted.params.r2a_h_s_inv - true_p.r2a_h_s_inv).abs()
                <= 3.5 * fitted.uncertainties.r2a_h_s_inv.max(0.5),
            "{}: r2a_h fit={:.3} (sd={:.3}) vs true={:.3}",
            fitted.label,
            fitted.params.r2a_h_s_inv,
            fitted.uncertainties.r2a_h_s_inv,
            true_p.r2a_h_s_inv
        );
        assert!(
            (fitted.params.r2b_h_s_inv - true_p.r2b_h_s_inv).abs()
                <= 3.5 * fitted.uncertainties.r2b_h_s_inv.max(0.5),
            "{}: r2b_h fit={:.3} (sd={:.3}) vs true={:.3}",
            fitted.label,
            fitted.params.r2b_h_s_inv,
            fitted.uncertainties.r2b_h_s_inv,
            true_p.r2b_h_s_inv
        );
        assert!(
            (fitted.params.r2a_n_s_inv - true_p.r2a_n_s_inv).abs()
                <= 3.5 * fitted.uncertainties.r2a_n_s_inv.max(0.5),
            "{}: r2a_n fit={:.3} (sd={:.3}) vs true={:.3}",
            fitted.label,
            fitted.params.r2a_n_s_inv,
            fitted.uncertainties.r2a_n_s_inv,
            true_p.r2a_n_s_inv
        );
        assert!(
            (fitted.params.r2b_n_s_inv - true_p.r2b_n_s_inv).abs()
                <= 3.5 * fitted.uncertainties.r2b_n_s_inv.max(0.5),
            "{}: r2b_n fit={:.3} (sd={:.3}) vs true={:.3}",
            fitted.label,
            fitted.params.r2b_n_s_inv,
            fitted.uncertainties.r2b_n_s_inv,
            true_p.r2b_n_s_inv
        );
        assert!(fitted.dw_h_sd_ppm.is_finite() && fitted.dw_h_sd_ppm > 0.0);
        assert!(fitted.dw_n_sd_ppm.is_finite() && fitted.dw_n_sd_ppm > 0.0);
        assert!((fitted.dw_h_rad_s - TAU * fitted.dw_h_ppm * h_mhz).abs() < 1e-9);
        assert!((fitted.dw_n_rad_s - TAU * fitted.dw_n_ppm * n_mhz).abs() < 1e-9);
    }

    // Must flag intermediate exchange and warn against relying on fast-exchange CSP fitting!
    assert!(!fit.fast_exchange_comparison.fast_exchange_assumption_valid);
    assert!(
        !fit.fast_exchange_comparison.warnings.is_empty(),
        "intermediate-exchange fit must emit fast-exchange CSP warnings"
    );

    // Serde JSON round-trip check
    let json = serde_json::to_string(&fit).expect("fit result must serialize to JSON");
    let decoded: nmr_core::titration_2d::Titration2dFitResult =
        serde_json::from_str(&json).expect("fit result must deserialize from JSON");
    assert_eq!(decoded.residues.len(), fit.residues.len());
    assert!((decoded.kd_um - fit.kd_um).abs() < 1e-12);
}

#[test]
fn synthetic_fast_exchange_2d_fit_agrees_with_quadratic_csp_without_warnings() {
    let true_kd_um = 40.0;
    let true_koff_s_inv = 4000.0;
    let h_mhz = 600.0;
    let p_tot_um = 50.0;
    let ligand_concs = [0.0, 20.0, 50.0, 100.0, 200.0, 400.0];

    let true_p = Bloch2dResidueParams {
        delta_a_h_ppm: 8.10,
        delta_b_h_ppm: 8.15, // dw_H = 0.05 ppm -> 188 rad/s << 4000 s^-1 (alpha < 0.05)
        delta_a_n_ppm: 120.0,
        delta_b_n_ppm: 120.30, // dw_N = 0.30 ppm -> 115 rad/s << 4000 s^-1 (alpha < 0.03)
        r2a_h_s_inv: 22.0,
        r2b_h_s_inv: 24.0,
        r2a_n_s_inv: 18.0,
        r2b_n_s_inv: 19.0,
        amplitude: 4000.0,
    };

    let grid = Roi2dGrid::uniform(119.5, 120.8, 19, 8.03, 8.22, 19, h_mhz).expect("grid ok");
    let apo_peak = true_p.amplitude / (true_p.r2a_n_s_inv * true_p.r2a_h_s_inv);
    let noise_sigma = apo_peak / 80.0;
    let mut rng = Rng::new(2026_9911);

    let mut patches = Vec::new();
    for (step_idx, &l_um) in ligand_concs.iter().enumerate() {
        let eq =
            binding_equilibrium_rates(p_tot_um, l_um, true_kd_um, true_koff_s_inv).expect("eq ok");
        let mut intensities =
            simulate_2d_lineshape_analytical(&grid, &true_p, &Bloch2dKineticState::from(eq))
                .expect("sim ok");
        for v in &mut intensities {
            *v += rng.normal(0.0, noise_sigma);
        }
        patches.push(Roi2dPatch {
            step_index: step_idx,
            protein_conc_um: p_tot_um,
            ligand_conc_um: l_um,
            grid: grid.clone(),
            intensities,
            noise_sigma,
            spectrum_scale: 1.0,
        });
    }

    let series = vec![ResidueTitration2dSeries {
        label: "A12".to_string(),
        res_name: "ALA".to_string(),
        patches,
        initial_guess: None,
    }];
    let fit = fit_titration_2d_lineshape(&series, &Titration2dFitConfig::default())
        .expect("fast-exchange fit must succeed");

    assert!((fit.kd_um - true_kd_um).abs() / true_kd_um < 0.05);
    assert!(
        fit.fast_exchange_comparison.fast_exchange_assumption_valid,
        "pure fast-exchange dataset should pass fast-exchange regime validation"
    );
    assert!(fit.fast_exchange_comparison.warnings.is_empty());
    assert!(
        fit.fast_exchange_comparison.kd_relative_discrepancy < 0.10,
        "in fast exchange, quadratic CSP K_d ({:.2}) should agree with 2D Bloch-McConnell K_d ({:.2})",
        fit.fast_exchange_comparison.csp_kd_um,
        fit.kd_um
    );
}

#[test]
fn synthetic_noiseless_global_2d_exact_recovery_all_parameters() {
    let true_kd_um = 25.0;
    let true_koff_s_inv = 120.0;
    let h_mhz = 600.0;
    let p_tot_um = 50.0;
    let ligand_concs = [0.0, 15.0, 40.0, 80.0, 160.0, 320.0];

    let true_residues: [(&str, &str, Bloch2dResidueParams); 2] = [
        (
            "K14",
            "LYS",
            Bloch2dResidueParams {
                delta_a_h_ppm: 8.05,
                delta_b_h_ppm: 8.14,
                delta_a_n_ppm: 120.5,
                delta_b_n_ppm: 121.2,
                r2a_h_s_inv: 24.0,
                r2b_h_s_inv: 31.0,
                r2a_n_s_inv: 18.0,
                r2b_n_s_inv: 24.0,
                amplitude: 4500.0,
            },
        ),
        (
            "E33",
            "GLU",
            Bloch2dResidueParams {
                delta_a_h_ppm: 8.62,
                delta_b_h_ppm: 8.54,
                delta_a_n_ppm: 117.2,
                delta_b_n_ppm: 116.6,
                r2a_h_s_inv: 21.0,
                r2b_h_s_inv: 28.0,
                r2a_n_s_inv: 17.0,
                r2b_n_s_inv: 22.0,
                amplitude: 3800.0,
            },
        ),
    ];

    let mut series_vec = Vec::new();
    for (lbl, res_name, true_p) in &true_residues {
        let grid = Roi2dGrid::uniform(
            true_p.delta_a_n_ppm.min(true_p.delta_b_n_ppm) - 0.5,
            true_p.delta_a_n_ppm.max(true_p.delta_b_n_ppm) + 0.5,
            17,
            true_p.delta_a_h_ppm.min(true_p.delta_b_h_ppm) - 0.08,
            true_p.delta_a_h_ppm.max(true_p.delta_b_h_ppm) + 0.08,
            17,
            h_mhz,
        )
        .expect("grid ok");
        let mut patches = Vec::new();
        for (step_idx, &l_um) in ligand_concs.iter().enumerate() {
            let eq = binding_equilibrium_rates(p_tot_um, l_um, true_kd_um, true_koff_s_inv)
                .expect("eq ok");
            let intensities =
                simulate_2d_lineshape_analytical(&grid, true_p, &Bloch2dKineticState::from(eq))
                    .expect("sim ok");
            patches.push(Roi2dPatch {
                step_index: step_idx,
                protein_conc_um: p_tot_um,
                ligand_conc_um: l_um,
                grid: grid.clone(),
                intensities,
                noise_sigma: 1e-2,
                spectrum_scale: 1.0,
            });
        }
        series_vec.push(ResidueTitration2dSeries {
            label: (*lbl).to_string(),
            res_name: (*res_name).to_string(),
            patches,
            initial_guess: None,
        });
    }

    let fit = fit_titration_2d_lineshape(&series_vec, &Titration2dFitConfig::default())
        .expect("noiseless fit must succeed");
    assert!((fit.kd_um - true_kd_um).abs() / true_kd_um < 1e-4);
    assert!((fit.k_off_s_inv - true_koff_s_inv).abs() / true_koff_s_inv < 1e-4);
    for (f, (_, _, tp)) in fit.residues.iter().zip(true_residues.iter()) {
        assert!((f.params.delta_a_h_ppm - tp.delta_a_h_ppm).abs() < 1e-5);
        assert!((f.params.delta_b_h_ppm - tp.delta_b_h_ppm).abs() < 1e-5);
        assert!((f.params.delta_a_n_ppm - tp.delta_a_n_ppm).abs() < 1e-4);
        assert!((f.params.delta_b_n_ppm - tp.delta_b_n_ppm).abs() < 1e-4);
        assert!((f.params.r2a_h_s_inv - tp.r2a_h_s_inv).abs() / tp.r2a_h_s_inv < 1e-4);
        assert!((f.params.r2b_h_s_inv - tp.r2b_h_s_inv).abs() / tp.r2b_h_s_inv < 1e-4);
        assert!((f.params.r2a_n_s_inv - tp.r2a_n_s_inv).abs() / tp.r2a_n_s_inv < 1e-4);
        assert!((f.params.r2b_n_s_inv - tp.r2b_n_s_inv).abs() / tp.r2b_n_s_inv < 1e-4);
        assert!((f.params.amplitude - tp.amplitude).abs() / tp.amplitude < 1e-4);
    }
}

// ============================================================================
// 5. Hostile-Input Fail-Closed Validation Tests [synthetic self-consistency]
// ============================================================================

#[test]
fn hostile_inputs_fail_closed_across_all_entry_points() {
    let cfg = TrajectoryTrackConfig::default();

    // 1. Tracker: < 2 steps
    let one_step = vec![SeriesPeakList {
        label: "s0".to_string(),
        series_coord: 0.0,
        protein_conc_um: Some(50.0),
        ligand_conc_um: Some(0.0),
        peaks: vec![Peak2d {
            peak_id: 0,
            h_ppm: 8.0,
            n_ppm: 120.0,
            intensity: None,
            label: None,
            res_name: None,
        }],
    }];
    assert!(matches!(
        track_titration_trajectories(&one_step, &cfg),
        Err(Titration2dError::EmptyInput(_))
    ));

    // 2. Tracker: non-monotonic series_coord
    let bad_order = vec![
        one_step[0].clone(),
        SeriesPeakList {
            series_coord: 20.0,
            ..one_step[0].clone()
        },
        SeriesPeakList {
            series_coord: 10.0,
            ..one_step[0].clone()
        },
    ];
    assert!(matches!(
        track_titration_trajectories(&bad_order, &cfg),
        Err(Titration2dError::InvalidSeriesOrder { step_index: 2, .. })
    ));

    // 3. Tracker: NaN peak coordinate
    let mut nan_steps = vec![
        one_step[0].clone(),
        SeriesPeakList {
            series_coord: 10.0,
            ..one_step[0].clone()
        },
    ];
    nan_steps[1].peaks[0].h_ppm = f64::NAN;
    assert!(matches!(
        track_titration_trajectories(&nan_steps, &cfg),
        Err(Titration2dError::InvalidParameter { .. })
    ));

    // 4. Tracker: invalid velocity_smoothing
    let mut bad_cfg = cfg.clone();
    bad_cfg.velocity_smoothing = 0.0;
    assert!(matches!(
        track_titration_trajectories(&[one_step[0].clone(), one_step[0].clone()], &bad_cfg),
        Err(Titration2dError::InvalidParameter { .. })
    ));

    // 5. Exchange regime: empty points / non-positive Kd / negative ligand
    let valid_ex = ExchangeRegimeInput {
        label: "X1".to_string(),
        kd_um: 25.0,
        k_off_s_inv: 100.0,
        dw_h_ppm: 0.1,
        dw_n_ppm: 0.5,
        h_larmor_mhz: 600.0,
        n_larmor_mhz: None,
    };
    assert!(matches!(
        classify_exchange_regime(&valid_ex, &[]),
        Err(Titration2dError::EmptyInput(_))
    ));
    let mut neg_kd = valid_ex.clone();
    neg_kd.kd_um = -5.0;
    let valid_pt = vec![TitrationConcentrationPoint {
        step_index: 0,
        label: "0".to_string(),
        protein_conc_um: 50.0,
        ligand_conc_um: 0.0,
    }];
    assert!(matches!(
        classify_exchange_regime(&neg_kd, &valid_pt),
        Err(Titration2dError::InvalidParameter { .. })
    ));
    assert!(matches!(
        binding_equilibrium_rates(50.0, -1.0, 25.0, 100.0),
        Err(Titration2dError::InvalidParameter { .. })
    ));

    // 6. Simulator: non-positive R2, zero population, non-power-of-two FFT size
    let grid = Roi2dGrid::uniform(119.0, 121.0, 5, 8.0, 8.2, 5, 600.0).expect("grid ok");
    let mut bad_params = Bloch2dResidueParams {
        delta_a_h_ppm: 8.1,
        delta_b_h_ppm: 8.15,
        delta_a_n_ppm: 120.0,
        delta_b_n_ppm: 120.5,
        r2a_h_s_inv: 0.0, // invalid!
        r2b_h_s_inv: 20.0,
        r2a_n_s_inv: 20.0,
        r2b_n_s_inv: 20.0,
        amplitude: 100.0,
    };
    let good_state = Bloch2dKineticState {
        p_a: 0.5,
        p_b: 0.5,
        k_ab_s_inv: 50.0,
        k_ba_s_inv: 50.0,
    };
    assert!(matches!(
        simulate_2d_lineshape_analytical(&grid, &bad_params, &good_state),
        Err(Titration2dError::InvalidParameter { .. })
    ));
    bad_params.r2a_h_s_inv = 20.0;
    let zero_pop = Bloch2dKineticState {
        p_a: 0.0,
        p_b: 0.0,
        k_ab_s_inv: 0.0,
        k_ba_s_inv: 0.0,
    };
    assert!(matches!(
        simulate_2d_lineshape_analytical(&grid, &bad_params, &zero_pop),
        Err(Titration2dError::InvalidParameter { .. })
    ));
    assert!(matches!(
        simulate_2d_lineshape_time_domain_fft(
            30,
            32,
            1000.0,
            1000.0,
            120.0,
            8.1,
            600.0,
            60.8,
            &bad_params,
            &good_state
        ),
        Err(Titration2dError::InvalidParameter { .. })
    ));

    // 7. Fitter: empty residues, dimension mismatch, NaN intensities, constant ligand conc
    let fit_cfg = Titration2dFitConfig::default();
    assert!(matches!(
        fit_titration_2d_lineshape(&[], &fit_cfg),
        Err(Titration2dError::EmptyInput(_))
    ));
    let bad_patch = Roi2dPatch {
        step_index: 0,
        protein_conc_um: 50.0,
        ligand_conc_um: 0.0,
        grid: grid.clone(),
        intensities: vec![0.0; 10], // expected 25
        noise_sigma: 1.0,
        spectrum_scale: 1.0,
    };
    let bad_series = vec![ResidueTitration2dSeries {
        label: "R1".to_string(),
        res_name: "ALA".to_string(),
        patches: vec![bad_patch.clone(), bad_patch.clone(), bad_patch],
        initial_guess: None,
    }];
    assert!(matches!(
        fit_titration_2d_lineshape(&bad_series, &fit_cfg),
        Err(Titration2dError::DimensionMismatch { .. })
    ));
}
