//! `[synthetic self-consistency]` test suite for clean-room Convex Accelerated Maximum-Entropy
//! (`MaxEnt`) 1D, 2D hypercomplex, and 3D indirect-plane NUS reconstruction (`nmr_core::maxent_nus`).
//!
//! All tests are strictly deterministic, self-contained synthetic self-consistency checks
//! (zero external unpublished data, zero wall-clock timing assertions).

use std::f64::consts::PI;

use nmr_core::fid::{Complex64, QuadMode};
use nmr_core::maxent_nus::{
    dual_to_primal_complex, dual_to_primal_hypercomplex, dual_to_primal_spectrum_1d,
    hoch_hore_entropy_1d, hoch_hore_entropy_2d_hypercomplex, kkt_residual_inf_1d,
    kkt_residual_inf_2d, maxent_1d_workspace_bytes, maxent_2d_workspace_bytes, poisson_gap_mask_1d,
    poisson_gap_mask_2d, reconstruct_maxent_1d, reconstruct_maxent_2d, reconstruct_maxent_3d,
    MaxEntConfig, MaxEntLambdaMode, MAX_MAXENT_3D_TOTAL_VOXELS, MAX_MAXENT_ITERATIONS,
    MAX_MAXENT_OUTER_ITERATIONS, MAX_MAXENT_POINTS, MAX_MAXENT_WORKSPACE_BYTES,
};
use nmr_core::process::{fft_in_place, reconstruct_nus_2d_hypercomplex_ist, NusIstConfig};
use nmr_core::rng::Rng;

// ============================================================================
// Synthetic NMR FID Generators & Evaluation Helpers
// ============================================================================

/// Synthesize a 1D damped complex exponential FID `sum_p A_p exp((-1/T2_p + i 2 pi f_p) k)`.
fn synth_fid_1d(n: usize, peaks: &[(f64, f64, f64)]) -> Vec<Complex64> {
    let mut fid = vec![[0.0, 0.0]; n];
    for &(freq, decay, amp) in peaks {
        for (k, slot) in fid.iter_mut().enumerate() {
            let env = amp * (-(k as f64) / decay).exp();
            let ph = 2.0 * PI * freq * (k as f64);
            slot[0] += env * ph.cos();
            slot[1] += env * ph.sin();
        }
    }
    fid
}

/// Synthesize a 2D canonical `States` hypercomplex FID `[cc, cs, sc, ss]` of shape `(n2, n1)`.
/// Each peak is `(f2, f1, decay2, decay1, amp)` with frequencies in cycles/sample `[0, 1)`.
fn synth_fid_2d_states(n2: usize, n1: usize, peaks: &[(f64, f64, f64, f64, f64)]) -> Vec<[f64; 4]> {
    let mut fid = vec![[0.0; 4]; n2 * n1];
    for &(f2, f1, d2, d1, amp) in peaks {
        for k2 in 0..n2 {
            let e2 = (-(k2 as f64) / d2).exp();
            let p2 = 2.0 * PI * f2 * (k2 as f64);
            let (s2, c2) = (e2 * p2.sin(), e2 * p2.cos());
            for k1 in 0..n1 {
                let e1 = amp * (-(k1 as f64) / d1).exp();
                let p1 = 2.0 * PI * f1 * (k1 as f64);
                let (s1, c1) = (e1 * p1.sin(), e1 * p1.cos());
                let q = &mut fid[k2 * n1 + k1];
                q[0] += c2 * c1;
                q[1] += c2 * s1;
                q[2] += s2 * c1;
                q[3] += s2 * s1;
            }
        }
    }
    fid
}

/// Apply `States` or `StatesTppi` encoding onto a canonical `States` 2D FID.
fn encode_2d_quad(
    states_fid: &[[f64; 4]],
    n2: usize,
    n1: usize,
    quad_modes: [QuadMode; 2],
) -> Vec<[f64; 4]> {
    let mut out = vec![[0.0; 4]; n2 * n1];
    for k2 in 0..n2 {
        for k1 in 0..n1 {
            let mut q = states_fid[k2 * n1 + k1];
            // Encode axis 1 first, then axis 2 (exact orthogonal inverse of quad_to_states)
            if matches!(quad_modes[1], QuadMode::StatesTppi) && k1 % 2 == 1 {
                q = [-q[0], -q[1], -q[2], -q[3]];
            }
            if matches!(quad_modes[0], QuadMode::StatesTppi) && k2 % 2 == 1 {
                q = [-q[0], -q[1], -q[2], -q[3]];
            }
            out[k2 * n1 + k1] = q;
        }
    }
    out
}

/// Compute the 2D hypercomplex absorption spectrum `RR` from a canonical `States` 2D FID via
/// unitary bicomplex 2D FFT (using `nmr_core::process::fft_in_place`).
fn states_fid_to_rr_spectrum(
    fid: &[[f64; 4]],
    n2: usize,
    n1: usize,
    f2: usize,
    f1: usize,
) -> Vec<f64> {
    let s = std::f64::consts::FRAC_1_SQRT_2;
    let n_vox = f2 * f1;
    let mut up = vec![[0.0, 0.0]; n_vox];
    let mut um = vec![[0.0, 0.0]; n_vox];
    for k2 in 0..n2 {
        for k1 in 0..n1 {
            let [cc, cs, sc, ss] = fid[k2 * n1 + k1];
            let g = k2 * f1 + k1;
            up[g] = [s * (cc - ss), s * (sc + cs)];
            um[g] = [s * (cc + ss), s * (sc - cs)];
        }
    }
    let mut col = vec![[0.0, 0.0]; f2];
    for buf in [&mut up, &mut um] {
        for r in 0..f2 {
            fft_in_place(&mut buf[r * f1..(r + 1) * f1], false);
        }
        for c in 0..f1 {
            for r in 0..f2 {
                col[r] = buf[r * f1 + c];
            }
            fft_in_place(&mut col, false);
            for r in 0..f2 {
                buf[r * f1 + c] = col[r];
            }
        }
        let scale = 1.0 / (n_vox as f64).sqrt();
        for v in buf.iter_mut() {
            v[0] *= scale;
            v[1] *= scale;
        }
    }
    let mut rr = vec![0.0_f64; n_vox];
    for m2 in 0..f2 {
        for m1 in 0..f1 {
            let m1_c = (f1 - m1) % f1;
            rr[m2 * f1 + m1] = s * (up[m2 * f1 + m1][0] + um[m2 * f1 + m1_c][0]);
        }
    }
    rr
}

fn pearson_corr(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let n = a.len() as f64;
    let ma = a.iter().sum::<f64>() / n;
    let mb = b.iter().sum::<f64>() / n;
    let mut num = 0.0_f64;
    let mut da = 0.0_f64;
    let mut db = 0.0_f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let dx = x - ma;
        let dy = y - mb;
        num += dx * dy;
        da += dx * dx;
        db += dy * dy;
    }
    num / (da.sqrt() * db.sqrt()).max(1e-30)
}

/// Exact unscaled ground-truth normalized root-mean-square error: `||est - gt||_2 / ||gt||_2`.
fn nrmse_unscaled(est: &[f64], gt: &[f64]) -> f64 {
    assert_eq!(est.len(), gt.len());
    let num_sq: f64 = est
        .iter()
        .zip(gt.iter())
        .map(|(&e, &g)| (e - g) * (e - g))
        .sum();
    let den_sq: f64 = gt.iter().map(|&g| g * g).sum();
    (num_sq / den_sq.max(1e-30)).sqrt()
}

/// Exact optimal-scale ground-truth NRMSE: `min_a ||a * est - gt||_2 / ||gt||_2`.
fn nrmse_scaled(est: &[f64], gt: &[f64]) -> f64 {
    assert_eq!(est.len(), gt.len());
    let dot: f64 = est.iter().zip(gt.iter()).map(|(&e, &g)| e * g).sum();
    let est_sq: f64 = est.iter().map(|&e| e * e).sum();
    let scale = if est_sq > 1e-30 { dot / est_sq } else { 0.0 };
    let num_sq: f64 = est
        .iter()
        .zip(gt.iter())
        .map(|(&e, &g)| (scale * e - g) * (scale * e - g))
        .sum();
    let den_sq: f64 = gt.iter().map(|&g| g * g).sum();
    (num_sq / den_sq.max(1e-30)).sqrt()
}

// ============================================================================
// 1. Convex Optimality (KKT Stationarity & Strong Duality) — 1D & 2D
// ============================================================================

#[test]
fn synthetic_self_consistency_kkt_stationarity_and_convex_duality_1d() {
    let n1 = 64usize;
    let peaks = [
        (0.1875, 40.0, 10.0),
        (0.421875, 36.0, 3.5),
        (0.734375, 32.0, 1.0),
    ];
    let full_fid = synth_fid_1d(n1, &peaks);
    let mask = poisson_gap_mask_1d(n1, 16, 2.0, 2026).expect("1D 25% Poisson-gap mask");
    assert_eq!(mask.iter().filter(|&&m| m).count(), 16);

    let mut obs = vec![[0.0, 0.0]; n1];
    for k in 0..n1 {
        if mask[k] {
            obs[k] = full_fid[k];
        }
    }

    let def = 0.25;
    let lambda = 4.0;
    let mut cfg = MaxEntConfig::fixed(def, lambda);
    cfg.kkt_tol = 1e-7;
    cfg.max_iterations = 2000;
    cfg.record_objective_trace = true;

    let res = reconstruct_maxent_1d(&obs, &mask, &cfg).expect("1D MaxEnt solve");
    assert!(
        res.converged,
        "1D MaxEnt should converge (kkt={:.3e}, iters={})",
        res.kkt_residual_inf, res.iterations
    );
    assert!(
        res.kkt_residual_inf < 1e-5,
        "1D KKT residual must be < 1e-5, got {:.3e}",
        res.kkt_residual_inf
    );
    assert!(
        res.duality_gap < 1e-6,
        "1D primal-dual optimality gap must be < 1e-6, got {:.3e}",
        res.duality_gap
    );
    assert!(
        res.linearity_note.contains("nonlinear"),
        "linearity_note must disclose MaxEnt amplitude nonlinearity"
    );

    // Independent KKT verification from scratch using nmr_core::process::fft_in_place
    let n_fft = res.spectrum.len();
    assert_eq!(n_fft, 2 * n1);
    let mut td = res.spectrum.clone();
    let sqrt_n = (n_fft as f64).sqrt();
    fft_in_place(&mut td, true);
    for c in &mut td {
        c[0] *= sqrt_n;
        c[1] *= sqrt_n;
    }
    let mut resid_td = vec![[0.0, 0.0]; n_fft];
    for k in 0..n1 {
        if mask[k] {
            resid_td[k] = [td[k][0] - obs[k][0], td[k][1] - obs[k][1]];
        }
    }
    fft_in_place(&mut resid_td, false);
    for c in &mut resid_td {
        c[0] /= sqrt_n;
        c[1] /= sqrt_n;
    }
    let mut indep_kkt_inf = 0.0_f64;
    let mut indep_dual_spec = vec![[0.0, 0.0]; n_fft];
    for m in 0..n_fft {
        let f = res.spectrum[m];
        let r = f[0].hypot(f[1]);
        let g_ent = if r > 1e-15 {
            let s = (r / (2.0 * def)).asinh() / (def * r);
            [f[0] * s, f[1] * s]
        } else {
            let s = 0.5 / (def * def);
            [f[0] * s, f[1] * s]
        };
        let kr = (g_ent[0] + lambda * resid_td[m][0]).abs();
        let ki = (g_ent[1] + lambda * resid_td[m][1]).abs();
        indep_kkt_inf = indep_kkt_inf.max(kr.max(ki));
        // y_k = A^*(d - A f) = -resid_td[m]
        indep_dual_spec[m] = [-resid_td[m][0], -resid_td[m][1]];
    }
    assert!(
        indep_kkt_inf < 1e-5,
        "Independent 1D KKT check failed: {indep_kkt_inf:.3e}"
    );
    let helper_kkt =
        kkt_residual_inf_1d(&res.spectrum, &obs, &mask, def, lambda).expect("kkt_residual_inf_1d");
    assert!((helper_kkt - indep_kkt_inf).abs() < 1e-12);

    // Verify closed-form convex-dual mapping f_k(y) = 2*def*sinh(|y_k| / sigma_def) * (y_k/|y_k|)
    // matches primal optimum with sigma_def = 1 / (lambda * def)
    let sigma_def = 1.0 / (lambda * def);
    let dual_primal =
        dual_to_primal_spectrum_1d(&indep_dual_spec, def, sigma_def).expect("dual_to_primal");
    let max_dual_primal_diff = res
        .spectrum
        .iter()
        .zip(dual_primal.iter())
        .map(|(a, b)| (a[0] - b[0]).hypot(a[1] - b[1]))
        .fold(0.0_f64, f64::max);
    assert!(
        max_dual_primal_diff < 1e-4,
        "Primal optimum must match closed-form convex-dual mapping: max_diff={max_dual_primal_diff:.3e}"
    );

    // Verify Hoch-Hore entropy matches direct formula
    let s_check = hoch_hore_entropy_1d(&res.spectrum, def).expect("entropy");
    assert!((s_check - res.entropy).abs() < 1e-10);
}

#[test]
fn synthetic_self_consistency_kkt_stationarity_2d_all_quad_modes() {
    let n2 = 16usize;
    let n1 = 16usize;
    let peaks = [
        (0.1875, 0.25, 14.0, 14.0, 10.0),
        (0.6875, 0.75, 12.0, 12.0, 2.0),
    ];
    let full_states = synth_fid_2d_states(n2, n1, &peaks);
    let mask = poisson_gap_mask_2d(n2, n1, 64, 2.0, 777).expect("2D 25% mask");
    assert_eq!(mask.iter().filter(|&&m| m).count(), 64);

    let def = 0.2;
    let lambda = 3.0;
    let mut cfg = MaxEntConfig::fixed(def, lambda);
    cfg.kkt_tol = 1e-6;
    cfg.max_iterations = 2000;

    let mode_cases = [
        [QuadMode::States, QuadMode::States],
        [QuadMode::StatesTppi, QuadMode::States],
        [QuadMode::States, QuadMode::StatesTppi],
        [QuadMode::StatesTppi, QuadMode::StatesTppi],
    ];

    let mut ref_spectrum: Option<Vec<[f64; 4]>> = None;
    for modes in mode_cases {
        let encoded = encode_2d_quad(&full_states, n2, n1, modes);
        let mut obs = vec![[0.0; 4]; n2 * n1];
        for idx in 0..(n2 * n1) {
            if mask[idx] {
                obs[idx] = encoded[idx];
            }
        }
        let res = reconstruct_maxent_2d(&obs, n2, n1, &mask, modes, &cfg)
            .expect("2D hypercomplex MaxEnt");
        assert!(
            res.converged && res.kkt_residual_inf < 1e-5,
            "2D MaxEnt ({modes:?}) KKT failed: converged={}, kkt={:.3e}",
            res.converged,
            res.kkt_residual_inf
        );
        assert!(
            res.duality_gap < 1e-5,
            "2D MaxEnt ({modes:?}) duality gap too large: {:.3e}",
            res.duality_gap
        );
        let indep_kkt = kkt_residual_inf_2d(
            &res.spectrum,
            res.spectrum_shape[0],
            res.spectrum_shape[1],
            &obs,
            n2,
            n1,
            &mask,
            modes,
            def,
            lambda,
        )
        .expect("kkt_residual_inf_2d");
        assert!(
            indep_kkt < 1e-5,
            "Independent 2D KKT ({modes:?}) failed: {indep_kkt:.3e}"
        );
        let ent = hoch_hore_entropy_2d_hypercomplex(&res.spectrum, def).expect("2D entropy");
        assert!((ent - res.entropy).abs() < 1e-9);

        // Canonical hypercomplex spectra across States / States-TPPI must match!
        if let Some(ref base) = ref_spectrum {
            let max_diff = base
                .iter()
                .zip(res.spectrum.iter())
                .flat_map(|(a, b)| (0..4).map(move |c| (a[c] - b[c]).abs()))
                .fold(0.0_f64, f64::max);
            assert!(
                max_diff < 1e-4,
                "Quadrature mode {modes:?} canonical spectrum diverged from States: max_diff={max_diff:.3e}"
            );
        } else {
            ref_spectrum = Some(res.spectrum);
        }
    }
}

#[test]
fn synthetic_self_consistency_echo_antiecho_rejected_fail_closed() {
    // Regression test for 22.13:P2-2: Echo-AntiEcho recombination acts along the directly
    // detected dimension (in `nmr_core::fid`), not an indirect hypercomplex partner axis.
    // MaxEnt must fail closed on `QuadMode::EchoAntiecho` across all 2D/3D entry points
    // and instruct callers to recombine to States/States-TPPI via `nmr_core::fid` first.
    let (n2, n1) = (8usize, 8usize);
    let obs_2d = vec![[1.0, 0.0, 0.0, 0.0]; n2 * n1];
    let obs_3d = vec![[1.0, 0.0, 0.0, 0.0]; 2 * n2 * n1];
    let spec_2d = vec![[0.0; 4]; (2 * n2) * (2 * n1)];
    let mask = vec![true; n2 * n1];
    let cfg = MaxEntConfig::fixed(0.1, 1.0);

    let rejected_cases = [
        [QuadMode::EchoAntiecho, QuadMode::States],
        [QuadMode::States, QuadMode::EchoAntiecho],
        [QuadMode::EchoAntiecho, QuadMode::StatesTppi],
        [QuadMode::StatesTppi, QuadMode::EchoAntiecho],
        [QuadMode::EchoAntiecho, QuadMode::EchoAntiecho],
    ];

    for modes in rejected_cases {
        let err_2d = reconstruct_maxent_2d(&obs_2d, n2, n1, &mask, modes, &cfg)
            .expect_err("reconstruct_maxent_2d must reject EchoAntiecho fail-closed");
        assert!(
            err_2d.contains("EchoAntiecho")
                && err_2d.contains("States")
                && err_2d.contains("nmr_core::fid"),
            "2D EchoAntiecho error for {modes:?} must direct callers to nmr_core::fid States recombination, got: {err_2d}"
        );

        let err_kkt = kkt_residual_inf_2d(
            &spec_2d,
            2 * n2,
            2 * n1,
            &obs_2d,
            n2,
            n1,
            &mask,
            modes,
            0.1,
            1.0,
        )
        .expect_err("kkt_residual_inf_2d must reject EchoAntiecho fail-closed");
        assert!(
            err_kkt.contains("EchoAntiecho")
                && err_kkt.contains("States")
                && err_kkt.contains("nmr_core::fid"),
            "2D KKT EchoAntiecho error for {modes:?} must direct callers to nmr_core::fid States recombination, got: {err_kkt}"
        );

        let err_3d = reconstruct_maxent_3d(&obs_3d, 2, n2, n1, &mask, modes, &cfg)
            .expect_err("reconstruct_maxent_3d must reject EchoAntiecho fail-closed");
        assert!(
            err_3d.contains("EchoAntiecho")
                && err_3d.contains("States")
                && err_3d.contains("nmr_core::fid"),
            "3D EchoAntiecho error for {modes:?} must direct callers to nmr_core::fid States recombination, got: {err_3d}"
        );
    }
}

// ============================================================================
// 2. Monotonic Convergence, O'Donoghue-Candes Restart & 10:1 Dynamic Range
// ============================================================================

#[test]
fn synthetic_self_consistency_monotonic_convergence_restart_and_10_to_1_peaks_1d() {
    let n1 = 128usize;
    let peaks = [
        (0.21875, 32.0, 10.0), // Strong peak (bin 56 in 256-pt spectrum; ~4*T2 decay, minimal truncation)
        (0.375, 32.0, 1.0),    // 10:1 weak peak (bin 96 in 256-pt spectrum)
        (0.703125, 32.0, 5.0), // Medium peak (bin 180 in 256-pt spectrum)
    ];
    let full_fid = synth_fid_1d(n1, &peaks);
    let mask = poisson_gap_mask_1d(n1, 32, 2.0, 42).expect("25% 1D Poisson-gap mask");
    let mut obs = vec![[0.0, 0.0]; n1];
    for k in 0..n1 {
        if mask[k] {
            obs[k] = full_fid[k];
        }
    }

    let mut cfg_accel = MaxEntConfig::fixed(0.02, 800.0);
    cfg_accel.kkt_tol = 1e-6;
    cfg_accel.max_iterations = 2500;
    cfg_accel.record_objective_trace = true;

    let res_accel = reconstruct_maxent_1d(&obs, &mask, &cfg_accel).expect("accelerated 1D MaxEnt");
    assert!(
        res_accel.converged && res_accel.kkt_residual_inf < 1e-5,
        "accelerated 1D MaxEnt failed to converge: kkt={:.3e}, iters={}",
        res_accel.kkt_residual_inf,
        res_accel.iterations
    );

    // Strict monotonic non-increasing primal objective across all iterations
    assert!(res_accel.objective_trace.len() > 2);
    for pair in res_accel.objective_trace.windows(2) {
        let tol = 1e-10 * pair[0].abs().max(1.0);
        assert!(
            pair[1] <= pair[0] + tol,
            "Primal objective increased: {:.12} -> {:.12}",
            pair[0],
            pair[1]
        );
    }

    // Compare against unaccelerated ISTA (same problem, accelerate = false, fixed step L = 1)
    let mut cfg_plain = cfg_accel.clone();
    cfg_plain.accelerate = false;
    cfg_plain.adaptive_step_size = false;
    cfg_plain.record_objective_trace = false;
    let res_plain =
        reconstruct_maxent_1d(&obs, &mask, &cfg_plain).expect("unaccelerated 1D MaxEnt");
    assert!(
        res_accel.iterations < res_plain.iterations || res_accel.kkt_residual_inf < res_plain.kkt_residual_inf,
        "Accelerated FISTA+restart ({} iters, kkt={:.2e}) should outperform unaccelerated ISTA ({} iters, kkt={:.2e})",
        res_accel.iterations,
        res_accel.kkt_residual_inf,
        res_plain.iterations,
        res_plain.kkt_residual_inf
    );

    // Apply standard NMR post-NUS processing (FCOR = 0.5 at k=0 per Otting et al. 1986,
    // zero-fill to n_fft, forward unitary FFT) to ground-truth, dirty NUS, and MaxEnt mock FID
    let n_fft = res_accel.spectrum.len();
    let inv_sqrt_n = 1.0 / (n_fft as f64).sqrt();
    let fid_to_abs_spec = |fid: &[Complex64]| -> Vec<f64> {
        let mut buf = vec![[0.0, 0.0]; n_fft];
        for (k, &v) in fid.iter().enumerate().take(n1) {
            let scale = if k == 0 { 0.5 } else { 1.0 };
            buf[k] = [v[0] * scale, v[1] * scale];
        }
        fft_in_place(&mut buf, false);
        buf.iter().map(|c| c[0] * inv_sqrt_n).collect()
    };
    let re_gt = fid_to_abs_spec(&full_fid);
    let re_dirty = fid_to_abs_spec(&obs);
    let re_maxent = fid_to_abs_spec(&res_accel.fid);

    let bin_strong = (0.21875 * n_fft as f64).round() as usize;
    let bin_weak = (0.375 * n_fft as f64).round() as usize;
    let strong_h = re_maxent[bin_strong];
    let weak_h = re_maxent[bin_weak];

    // Off-peak ground-truth error RMS away from all 3 Lorentzian peaks (never counting true
    // Lorentzian tails as artifact!) plus full-spectrum ground-truth NRMSE.
    let peak_bins = [
        bin_strong,
        bin_weak,
        (0.703125 * n_fft as f64).round() as usize,
    ];
    let off_peak_err_rms = |est: &[f64], gt: &[f64]| -> f64 {
        let vals: Vec<f64> = (0..n_fft)
            .filter(|&b| peak_bins.iter().all(|&pb| b.abs_diff(pb) > 12))
            .map(|b| est[b] - gt[b])
            .collect();
        (vals.iter().map(|v| v * v).sum::<f64>() / vals.len() as f64).sqrt()
    };
    let artifact_rms_maxent = off_peak_err_rms(&re_maxent, &re_gt);
    let artifact_rms_dirty = off_peak_err_rms(&re_dirty, &re_gt);
    let nrmse_maxent = nrmse_unscaled(&re_maxent, &re_gt);
    let nrmse_dirty = nrmse_unscaled(&re_dirty, &re_gt);
    let r_maxent = pearson_corr(&re_gt, &re_maxent);
    let r_dirty = pearson_corr(&re_gt, &re_dirty);

    assert!(
        weak_h > 10.0 * artifact_rms_maxent.max(1e-9),
        "1D 25% NUS MaxEnt failed to recover 10:1 weak peak cleanly above GT error: strong={strong_h:.4}, weak={weak_h:.4}, artifact_rms_maxent={artifact_rms_maxent:.4}, artifact_rms_dirty={artifact_rms_dirty:.4}"
    );
    assert!(
        artifact_rms_maxent < 0.25 * artifact_rms_dirty && nrmse_maxent < 0.35 * nrmse_dirty,
        "1D MaxEnt must strongly suppress NUS error vs dirty FT: off_peak_err {artifact_rms_maxent:.4} vs {artifact_rms_dirty:.4}, NRMSE {nrmse_maxent:.4} vs {nrmse_dirty:.4}"
    );
    assert!(
        r_maxent > 0.95 && r_maxent > r_dirty + 0.15,
        "1D MaxEnt correlation r={r_maxent:.4} must exceed 0.95 and beat dirty r={r_dirty:.4}"
    );
    assert!(
        strong_h > weak_h * 5.0,
        "Strong peak must remain dominant over 10:1 weak peak: strong={strong_h:.4}, weak={weak_h:.4}"
    );
}

#[test]
fn synthetic_self_consistency_bluestein_non_pow2_grid_1d_and_2d() {
    // Non-power-of-two lengths exercise nmr_core::process::fft_in_place Bluestein path
    let n1 = 45usize;
    let peaks = [(0.2, 30.0, 5.0), (0.6, 28.0, 1.5)];
    let full_fid = synth_fid_1d(n1, &peaks);
    let mask = poisson_gap_mask_1d(n1, 15, 2.0, 99).expect("non-pow2 1D mask");
    let mut obs = vec![[0.0, 0.0]; n1];
    for k in 0..n1 {
        if mask[k] {
            obs[k] = full_fid[k];
        }
    }
    let mut cfg = MaxEntConfig::fixed(0.2, 3.0);
    cfg.zero_fill_factor = 1; // exact n_fft = 45 (non-power-of-two!)
    cfg.kkt_tol = 1e-6;
    let res = reconstruct_maxent_1d(&obs, &mask, &cfg).expect("Bluestein 1D MaxEnt");
    assert_eq!(res.spectrum.len(), 45);
    assert!(
        res.converged && res.kkt_residual_inf < 1e-5,
        "Bluestein 1D KKT failed: {:.3e}",
        res.kkt_residual_inf
    );

    // 2D non-power-of-two (15 x 13) with zero_fill_factor = 1
    let (n2_b, n1_b) = (15usize, 13usize);
    let full_2d = synth_fid_2d_states(n2_b, n1_b, &[(0.2, 0.3, 12.0, 10.0, 4.0)]);
    let mask_2d = poisson_gap_mask_2d(n2_b, n1_b, 65, 2.0, 101).expect("non-pow2 2D mask");
    let mut obs_2d = vec![[0.0; 4]; n2_b * n1_b];
    for i in 0..(n2_b * n1_b) {
        if mask_2d[i] {
            obs_2d[i] = full_2d[i];
        }
    }
    let res2 = reconstruct_maxent_2d(
        &obs_2d,
        n2_b,
        n1_b,
        &mask_2d,
        [QuadMode::States, QuadMode::States],
        &cfg,
    )
    .expect("Bluestein 2D MaxEnt");
    assert_eq!(res2.spectrum_shape, vec![15, 13]);
    assert!(
        res2.converged && res2.kkt_residual_inf < 1e-5,
        "Bluestein 2D KKT failed: {:.3e}",
        res2.kkt_residual_inf
    );
}

// ============================================================================
// 3. Discrepancy-Principle Lambda Selection (chi2_nu in [0.85, 1.15])
// ============================================================================

#[test]
fn synthetic_self_consistency_discrepancy_principle_1d_and_2d() {
    // 1D noisy 25% NUS FID across 3 deterministic noise seeds
    let n1 = 128usize;
    let peaks_1d = [
        (0.1875, 50.0, 8.0),
        (0.453125, 45.0, 3.0),
        (0.765625, 40.0, 1.2),
    ];
    let clean_1d = synth_fid_1d(n1, &peaks_1d);
    let mask_1d = poisson_gap_mask_1d(n1, 32, 2.0, 314).expect("1D mask");
    let sigma_1d = 0.15;

    for seed in [1001u64, 2002, 3003] {
        let mut rng = Rng::new(seed);
        let mut obs = vec![[0.0, 0.0]; n1];
        for k in 0..n1 {
            if mask_1d[k] {
                obs[k] = [
                    clean_1d[k][0] + rng.normal(0.0, sigma_1d),
                    clean_1d[k][1] + rng.normal(0.0, sigma_1d),
                ];
            }
        }
        let mut cfg = MaxEntConfig::discrepancy(sigma_1d, sigma_1d);
        cfg.kkt_tol = 1e-6;
        cfg.max_iterations = 1500;

        let res = reconstruct_maxent_1d(&obs, &mask_1d, &cfg).expect("1D discrepancy MaxEnt");
        let chi2_nu = res.chi2_reduced.expect("chi2_reduced must be Some");
        assert!(
            res.converged && res.kkt_residual_inf < 1e-5,
            "1D discrepancy seed={seed} KKT failed: {:.3e}",
            res.kkt_residual_inf
        );
        assert!(
            (0.85..=1.15).contains(&chi2_nu),
            "1D discrepancy seed={seed} chi2_nu={chi2_nu:.4} outside [0.85, 1.15] (lambda={:.4e}, outer={})",
            res.final_lambda,
            res.outer_iterations
        );
    }

    // 2D hypercomplex noisy 25% NUS plane
    let (n2, n1_2d) = (24usize, 24usize);
    let peaks_2d = [
        (0.208333, 0.291667, 20.0, 20.0, 10.0),
        (0.625, 0.708333, 18.0, 18.0, 2.5),
        (0.375, 0.791667, 18.0, 18.0, 1.0),
    ];
    let clean_2d = synth_fid_2d_states(n2, n1_2d, &peaks_2d);
    let mask_2d = poisson_gap_mask_2d(n2, n1_2d, 144, 2.0, 515).expect("2D 25% mask");
    let sigma_2d = 0.08;
    let mut rng = Rng::new(7777);
    let mut obs_2d = vec![[0.0; 4]; n2 * n1_2d];
    for idx in 0..(n2 * n1_2d) {
        if mask_2d[idx] {
            obs_2d[idx] = [
                clean_2d[idx][0] + rng.normal(0.0, sigma_2d),
                clean_2d[idx][1] + rng.normal(0.0, sigma_2d),
                clean_2d[idx][2] + rng.normal(0.0, sigma_2d),
                clean_2d[idx][3] + rng.normal(0.0, sigma_2d),
            ];
        }
    }

    let mut cfg_2d = MaxEntConfig::discrepancy(sigma_2d, sigma_2d);
    cfg_2d.kkt_tol = 1e-6;
    cfg_2d.max_iterations = 1500;
    let res_2d = reconstruct_maxent_2d(
        &obs_2d,
        n2,
        n1_2d,
        &mask_2d,
        [QuadMode::States, QuadMode::States],
        &cfg_2d,
    )
    .expect("2D discrepancy MaxEnt");
    let chi2_2d = res_2d.chi2_reduced.expect("2D chi2_reduced");
    assert!(
        res_2d.converged && res_2d.kkt_residual_inf < 1e-5,
        "2D discrepancy KKT failed: {:.3e}",
        res_2d.kkt_residual_inf
    );
    assert!(
        (0.85..=1.15).contains(&chi2_2d),
        "2D discrepancy chi2_nu={chi2_2d:.4} outside [0.85, 1.15] (lambda={:.4e})",
        res_2d.final_lambda
    );
}

// ============================================================================
// 4. Head-to-Head [synthetic self-consistency] Benchmark: MaxEnt vs IST/FISTA
// ============================================================================

#[test]
fn synthetic_self_consistency_head_to_head_maxent_vs_ist_25pct_nus_2d() {
    let (n2, n1) = (32usize, 32usize);
    let (f2, f1) = (64usize, 64usize);
    let peaks = [
        (0.1875, 0.25, 24.0, 24.0, 10.0), // Strong peak (10:1)
        (0.4375, 0.6875, 24.0, 24.0, 6.0),
        (0.71875, 0.34375, 24.0, 24.0, 3.0),
        (0.28125, 0.78125, 24.0, 24.0, 1.0), // 10:1 weak peak!
    ];
    let full_fid = synth_fid_2d_states(n2, n1, &peaks);
    let gt_rr = states_fid_to_rr_spectrum(&full_fid, n2, n1, f2, f1);

    let mask = poisson_gap_mask_2d(n2, n1, 256, 2.0, 20260).expect("25% 2D Poisson-gap mask");
    assert_eq!(mask.iter().filter(|&&m| m).count(), 256);

    let sigma = 0.03;
    let mut rng = Rng::new(8888);
    let mut obs = vec![[0.0; 4]; n2 * n1];
    for idx in 0..(n2 * n1) {
        if mask[idx] {
            obs[idx] = [
                full_fid[idx][0] + rng.normal(0.0, sigma),
                full_fid[idx][1] + rng.normal(0.0, sigma),
                full_fid[idx][2] + rng.normal(0.0, sigma),
                full_fid[idx][3] + rng.normal(0.0, sigma),
            ];
        }
    }

    // 1. Dirty zero-filled NUS baseline (for artifact-suppression reference)
    let dirty_rr = states_fid_to_rr_spectrum(&obs, n2, n1, f2, f1);

    // 2. Clean-room Convex Accelerated MaxEnt (Discrepancy Principle)
    let mut maxent_cfg = MaxEntConfig::discrepancy(sigma, sigma);
    maxent_cfg.kkt_tol = 1e-6;
    maxent_cfg.max_iterations = 1500;
    maxent_cfg.record_objective_trace = true;
    let maxent_res = reconstruct_maxent_2d(
        &obs,
        n2,
        n1,
        &mask,
        [QuadMode::States, QuadMode::States],
        &maxent_cfg,
    )
    .expect("2D MaxEnt benchmark");
    assert!(
        maxent_res.converged && maxent_res.kkt_residual_inf < 1e-5,
        "2D MaxEnt benchmark did not converge: kkt={:.3e}",
        maxent_res.kkt_residual_inf
    );
    // Verify monotonic convergence on the 2D benchmark solve too
    for pair in maxent_res.objective_trace.windows(2) {
        let tol = 1e-10 * pair[0].abs().max(1.0);
        assert!(pair[1] <= pair[0] + tol);
    }
    // Direct MaxEnt frequency-domain spectrum (zero-filled 64x64 with time-domain extrapolation)
    // and cropped 32x32 mock-FID spectrum transformed identically to ground truth & IST.
    let maxent_rr: Vec<f64> = maxent_res.spectrum.iter().map(|q| q[0]).collect();
    let maxent_fid_rr = states_fid_to_rr_spectrum(&maxent_res.fid, n2, n1, f2, f1);

    // 3. Existing nmr_core::process IST/FISTA baseline
    let ist_cfg = NusIstConfig::default();
    let ist_fid = reconstruct_nus_2d_hypercomplex_ist(&obs, n2, n1, &mask, &ist_cfg);
    let ist_rr = states_fid_to_rr_spectrum(&ist_fid, n2, n1, f2, f1);

    // Honest ground-truth fidelity metrics (never counting true Lorentzian/truncation tails as artifact):
    // (a) Pearson correlation and exact ground-truth NRMSE (unscaled & optimal-scale)
    let r_dirty = pearson_corr(&gt_rr, &dirty_rr);
    let r_ist = pearson_corr(&gt_rr, &ist_rr);
    let r_maxent_spec = pearson_corr(&gt_rr, &maxent_rr);
    let r_maxent_fid = pearson_corr(&gt_rr, &maxent_fid_rr);

    let dirty_nrmse_unscaled = nrmse_unscaled(&dirty_rr, &gt_rr);
    let dirty_nrmse_scaled = nrmse_scaled(&dirty_rr, &gt_rr);
    let ist_nrmse_unscaled = nrmse_unscaled(&ist_rr, &gt_rr);
    let ist_nrmse_scaled = nrmse_scaled(&ist_rr, &gt_rr);
    let maxent_spec_nrmse_unscaled = nrmse_unscaled(&maxent_rr, &gt_rr);
    let maxent_spec_nrmse_scaled = nrmse_scaled(&maxent_rr, &gt_rr);
    let maxent_fid_nrmse_unscaled = nrmse_unscaled(&maxent_fid_rr, &gt_rr);
    let maxent_fid_nrmse_scaled = nrmse_scaled(&maxent_fid_rr, &gt_rr);

    // (b) Peak-height fidelity vs ground truth & weak-to-strong ratio (nonlinearity disclosure)
    let peak_coords: Vec<(usize, usize)> = peaks
        .iter()
        .map(|&(p2, p1, _, _, _)| {
            (
                (p2 * f2 as f64).round() as usize % f2,
                (p1 * f1 as f64).round() as usize % f1,
            )
        })
        .collect();
    let strong_coord = peak_coords[0];
    let weak_coord = peak_coords[3];

    let local_peak = |spec: &[f64], (c2, c1): (usize, usize)| -> f64 {
        let mut best = f64::NEG_INFINITY;
        for d2 in -1isize..=1 {
            for d1 in -1isize..=1 {
                let r = ((c2 as isize + d2).rem_euclid(f2 as isize)) as usize;
                let c = ((c1 as isize + d1).rem_euclid(f1 as isize)) as usize;
                best = best.max(spec[r * f1 + c]);
            }
        }
        best
    };

    let off_peak_err_rms = |est: &[f64], gt: &[f64]| -> f64 {
        let mut sum_sq = 0.0_f64;
        let mut cnt = 0usize;
        for r in 0..f2 {
            for c in 0..f1 {
                let near_peak = peak_coords.iter().any(|&(p2, p1)| {
                    let dr = r.abs_diff(p2).min(f2 - r.abs_diff(p2));
                    let dc = c.abs_diff(p1).min(f1 - c.abs_diff(p1));
                    dr <= 4 && dc <= 4
                });
                if !near_peak {
                    let d = est[r * f1 + c] - gt[r * f1 + c];
                    sum_sq += d * d;
                    cnt += 1;
                }
            }
        }
        (sum_sq / cnt.max(1) as f64).sqrt()
    };

    let peak_height_ratio_range = |spec: &[f64]| -> (f64, f64) {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for &coord in &peak_coords {
            let ratio = local_peak(spec, coord) / local_peak(&gt_rr, coord);
            lo = lo.min(ratio);
            hi = hi.max(ratio);
        }
        (lo, hi)
    };

    let dirty_err_rms = off_peak_err_rms(&dirty_rr, &gt_rr);
    let ist_err_rms = off_peak_err_rms(&ist_rr, &gt_rr);
    let maxent_spec_err_rms = off_peak_err_rms(&maxent_rr, &gt_rr);
    let maxent_fid_err_rms = off_peak_err_rms(&maxent_fid_rr, &gt_rr);

    let (ist_h_lo, ist_h_hi) = peak_height_ratio_range(&ist_rr);
    let (maxent_spec_h_lo, maxent_spec_h_hi) = peak_height_ratio_range(&maxent_rr);
    let (maxent_fid_h_lo, maxent_fid_h_hi) = peak_height_ratio_range(&maxent_fid_rr);

    let gt_ws = local_peak(&gt_rr, weak_coord) / local_peak(&gt_rr, strong_coord);
    let ist_ws = local_peak(&ist_rr, weak_coord) / local_peak(&ist_rr, strong_coord);
    let maxent_spec_ws = local_peak(&maxent_rr, weak_coord) / local_peak(&maxent_rr, strong_coord);
    let maxent_fid_ws =
        local_peak(&maxent_fid_rr, weak_coord) / local_peak(&maxent_fid_rr, strong_coord);
    let compression_pct = (1.0 - maxent_spec_ws / gt_ws) * 100.0;

    eprintln!(
        "[synthetic self-consistency] 2D 25% NUS ground-truth comparison (32x32 -> 64x64, 10:1 peaks, sigma={sigma}):\n\
         - Dirty NUS:             r={r_dirty:.4}, NRMSE(unscaled/scaled)={dirty_nrmse_unscaled:.4}/{dirty_nrmse_scaled:.4}, off_peak_err_rms={dirty_err_rms:.4}\n\
         - IST/FISTA (linear):    r={r_ist:.4}, NRMSE(unscaled/scaled)={ist_nrmse_unscaled:.4}/{ist_nrmse_scaled:.4}, off_peak_err_rms={ist_err_rms:.4}, peak_vs_GT={ist_h_lo:.3}..{ist_h_hi:.3}x, weak/strong={ist_ws:.4}\n\
         - MaxEnt (64x64 spec):   r={r_maxent_spec:.4}, NRMSE(unscaled/scaled)={maxent_spec_nrmse_unscaled:.4}/{maxent_spec_nrmse_scaled:.4}, off_peak_err_rms={maxent_spec_err_rms:.4}, peak_vs_GT={maxent_spec_h_lo:.3}..{maxent_spec_h_hi:.3}x, weak/strong={maxent_spec_ws:.4} (GT={gt_ws:.4}, compressed {compression_pct:.1}%)\n\
         - MaxEnt (32x32 mockFID): r={r_maxent_fid:.4}, NRMSE(unscaled/scaled)={maxent_fid_nrmse_unscaled:.4}/{maxent_fid_nrmse_scaled:.4}, off_peak_err_rms={maxent_fid_err_rms:.4}, peak_vs_GT={maxent_fid_h_lo:.3}..{maxent_fid_h_hi:.3}x, weak/strong={maxent_fid_ws:.4}\n\
         - MaxEnt diagnostics:    chi2_nu={:.4}, lambda={:.4}, iters={} (total {}), restarts={}",
        maxent_res.chi2_reduced.unwrap_or(f64::NAN),
        maxent_res.final_lambda,
        maxent_res.iterations,
        maxent_res.total_iterations,
        maxent_res.restarts,
    );

    // Honest linear-fidelity ordering: IST achieves much lower ground-truth NRMSE & near-linear heights
    assert!(
        ist_nrmse_scaled < 0.05 && ist_nrmse_unscaled < 0.05,
        "IST/FISTA linear ground-truth NRMSE expected < 0.05, got unscaled={ist_nrmse_unscaled:.4}, scaled={ist_nrmse_scaled:.4}"
    );
    assert!(
        ist_h_lo > 0.85 && ist_h_hi < 1.05,
        "IST/FISTA peak heights should stay near-linear vs GT: {ist_h_lo:.3}..{ist_h_hi:.3}x"
    );
    assert!(
        ist_nrmse_scaled < maxent_spec_nrmse_scaled,
        "Honest check: IST linear NRMSE ({ist_nrmse_scaled:.4}) is lower than nonlinear extrapolated MaxEnt ({maxent_spec_nrmse_scaled:.4})"
    );

    // MaxEnt strongly improves over dirty NUS and cleanly recovers all 4 peaks including the 10:1 weak peak
    assert!(
        r_maxent_spec > 0.96 && r_maxent_spec > r_dirty + 0.15,
        "MaxEnt 2D 25% NUS correlation r={r_maxent_spec:.4} must exceed 0.96 and beat dirty r={r_dirty:.4}"
    );
    assert!(
        maxent_spec_nrmse_scaled < dirty_nrmse_scaled && maxent_fid_nrmse_unscaled < dirty_nrmse_unscaled,
        "MaxEnt must strongly reduce ground-truth NRMSE vs dirty NUS: spec_scaled={maxent_spec_nrmse_scaled:.4} vs dirty={dirty_nrmse_scaled:.4}, fid_unscaled={maxent_fid_nrmse_unscaled:.4} vs dirty={dirty_nrmse_unscaled:.4}"
    );
    assert!(
        maxent_fid_err_rms < 0.35 * dirty_err_rms,
        "MaxEnt mock-FID off-peak GT error RMS ({maxent_fid_err_rms:.4}) must strongly suppress dirty error ({dirty_err_rms:.4})"
    );
    assert!(
        local_peak(&maxent_rr, weak_coord) > 10.0 * maxent_spec_err_rms.max(1e-12),
        "MaxEnt 10:1 weak peak must sit cleanly above off-peak GT error"
    );
    // Verify nonlinearity & extrapolation disclosure is present on every result
    assert!(
        maxent_res.linearity_note.contains("nonlinear")
            && maxent_res.linearity_note.contains("extrapolat")
            && maxent_res.linearity_note.contains("compress"),
        "linearity_note must disclose nonlinearity, weak-peak compression, and extrapolation"
    );
}

// ============================================================================
// 5. 3D Hypercomplex Indirect-Plane NUS Reconstruction
// ============================================================================

#[test]
fn synthetic_self_consistency_3d_hypercomplex_indirect_planes() {
    let (nx, n2, n1) = (6usize, 16usize, 16usize);
    let mask_2d = poisson_gap_mask_2d(n2, n1, 64, 2.0, 909).expect("3D 25% 2D mask");
    let quad_modes = [QuadMode::StatesTppi, QuadMode::States];

    let mut obs_planes = Vec::with_capacity(nx * n2 * n1);
    let slice_peak_coords = [
        (0.25, 0.25),
        (0.25, 0.75),
        (0.50, 0.25),
        (0.50, 0.75),
        (0.75, 0.25),
        (0.75, 0.75),
    ];
    for &(p2, p1) in &slice_peak_coords {
        let states = synth_fid_2d_states(n2, n1, &[(p2, p1, 14.0, 14.0, 8.0)]);
        let encoded = encode_2d_quad(&states, n2, n1, quad_modes);
        for idx in 0..(n2 * n1) {
            if mask_2d[idx] {
                obs_planes.push(encoded[idx]);
            } else {
                obs_planes.push([0.0; 4]);
            }
        }
    }

    let mut cfg = MaxEntConfig::fixed(0.2, 4.0);
    cfg.kkt_tol = 1e-6;
    cfg.max_iterations = 1500;

    let res3 = reconstruct_maxent_3d(&obs_planes, nx, n2, n1, &mask_2d, quad_modes, &cfg)
        .expect("3D MaxEnt reconstruction");
    assert!(
        res3.converged && res3.max_kkt_residual_inf < 1e-5,
        "3D MaxEnt convergence failed: converged={}, max_kkt={:.3e}",
        res3.converged,
        res3.max_kkt_residual_inf
    );
    assert_eq!(res3.plane_diagnostics.len(), nx);
    assert_eq!(res3.f2, 32);
    assert_eq!(res3.f1, 32);
    assert_eq!(res3.spectra_planes.len(), nx * 32 * 32);
    assert_eq!(res3.fid_planes.len(), nx * n2 * n1);

    // Every direct slice's maximum absorption peak must land at its exact synthetic coordinate
    for (ix, &(p2, p1)) in slice_peak_coords.iter().enumerate() {
        let plane = &res3.spectra_planes[ix * 32 * 32..(ix + 1) * 32 * 32];
        let (best_idx, _) = plane
            .iter()
            .enumerate()
            .max_by(|a, b| a.1[0].total_cmp(&b.1[0]))
            .expect("non-empty plane");
        let got_m2 = best_idx / 32;
        let got_m1 = best_idx % 32;
        let exp_m2 = (p2 * 32.0).round() as usize % 32;
        let exp_m1 = (p1 * 32.0).round() as usize % 32;
        assert_eq!(
            (got_m2, got_m1),
            (exp_m2, exp_m1),
            "3D slice {ix} peak landed at ({got_m2}, {got_m1}), expected ({exp_m2}, {exp_m1})"
        );
    }
}

// ============================================================================
// 6. Hostile / Adversarial Input Rejection & Overflow Guards
// ============================================================================

#[test]
fn synthetic_self_consistency_hostile_inputs_and_overflow_guards() {
    let valid_cfg = MaxEntConfig::fixed(0.1, 1.0);

    // Empty input / empty mask
    assert!(reconstruct_maxent_1d(&[], &[], &valid_cfg).is_err());

    // Mismatched mask length
    assert!(reconstruct_maxent_1d(&[[1.0, 0.0]; 4], &[true; 3], &valid_cfg).is_err());

    // All-false NUS mask (fail-closed)
    assert!(reconstruct_maxent_1d(&[[1.0, 0.0]; 8], &[false; 8], &valid_cfg).is_err());
    assert!(reconstruct_maxent_2d(
        &[[1.0, 0.0, 0.0, 0.0]; 16],
        4,
        4,
        &[false; 16],
        [QuadMode::States, QuadMode::States],
        &valid_cfg
    )
    .is_err());
    assert!(reconstruct_maxent_3d(
        &[[1.0, 0.0, 0.0, 0.0]; 32],
        2,
        4,
        4,
        &[false; 16],
        [QuadMode::States, QuadMode::States],
        &valid_cfg
    )
    .is_err());

    // Non-finite samples (NaN / Inf)
    let mut nan_fid = vec![[1.0, 0.0]; 8];
    nan_fid[3] = [f64::NAN, 0.0];
    assert!(reconstruct_maxent_1d(&nan_fid, &[true; 8], &valid_cfg).is_err());
    nan_fid[3] = [0.0, f64::INFINITY];
    assert!(reconstruct_maxent_1d(&nan_fid, &[true; 8], &valid_cfg).is_err());

    let mut nan_2d = vec![[0.0; 4]; 16];
    nan_2d[5][2] = f64::NEG_INFINITY;
    assert!(reconstruct_maxent_2d(
        &nan_2d,
        4,
        4,
        &[true; 16],
        [QuadMode::States, QuadMode::States],
        &valid_cfg
    )
    .is_err());

    // Non-positive or non-finite def, lambda, noise_sigma
    for bad_def in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let cfg = MaxEntConfig::fixed(bad_def, 1.0);
        assert!(reconstruct_maxent_1d(&[[1.0, 0.0]; 8], &[true; 8], &cfg).is_err());
    }
    for bad_lam in [0.0, -0.5, f64::NAN, f64::INFINITY] {
        let cfg = MaxEntConfig::fixed(0.1, bad_lam);
        assert!(reconstruct_maxent_1d(&[[1.0, 0.0]; 8], &[true; 8], &cfg).is_err());
    }
    for bad_sig in [0.0, -0.1, f64::NAN, f64::INFINITY] {
        let cfg = MaxEntConfig::discrepancy(0.1, bad_sig);
        assert!(reconstruct_maxent_1d(&[[1.0, 0.0]; 8], &[true; 8], &cfg).is_err());
    }

    // Invalid zero_fill_factor / max_iterations / oversized grid (> MAX_MAXENT_POINTS)
    let mut bad_zf = valid_cfg.clone();
    bad_zf.zero_fill_factor = 0;
    assert!(reconstruct_maxent_1d(&[[1.0, 0.0]; 8], &[true; 8], &bad_zf).is_err());

    let mut bad_iters = valid_cfg.clone();
    bad_iters.max_iterations = 0;
    assert!(reconstruct_maxent_1d(&[[1.0, 0.0]; 8], &[true; 8], &bad_iters).is_err());

    let mut huge_grid = valid_cfg.clone();
    huge_grid.zero_fill_factor = MAX_MAXENT_POINTS / 4 + 1;
    assert!(reconstruct_maxent_1d(&[[1.0, 0.0]; 8], &[true; 8], &huge_grid).is_err());

    // Invalid QuadMode (Qf / Dqd) on 2D and 3D hypercomplex
    assert!(reconstruct_maxent_2d(
        &[[1.0, 0.0, 0.0, 0.0]; 16],
        4,
        4,
        &[true; 16],
        [QuadMode::Qf, QuadMode::States],
        &valid_cfg
    )
    .is_err());
    assert!(reconstruct_maxent_3d(
        &[[1.0, 0.0, 0.0, 0.0]; 32],
        2,
        4,
        4,
        &[true; 16],
        [QuadMode::States, QuadMode::Dqd],
        &valid_cfg
    )
    .is_err());

    // Overflow-guarded closed-form convex-dual mapping: huge dual inputs MUST stay finite (no Inf/NaN)
    for huge in [1e4, 1e10, 1e100] {
        let f1 = dual_to_primal_complex([huge, -0.5 * huge], 1.0, 0.01)
            .expect("overflow-safe 1D dual_to_primal");
        assert!(
            f1[0].is_finite() && f1[1].is_finite(),
            "dual_to_primal_complex overflowed on {huge}: {f1:?}"
        );
        let f4 = dual_to_primal_hypercomplex([huge, 0.0, -huge, 1.0], 1.0, 0.01)
            .expect("overflow-safe 2D dual_to_primal");
        assert!(
            f4.iter().all(|v| v.is_finite()),
            "dual_to_primal_hypercomplex overflowed on {huge}: {f4:?}"
        );
    }

    // Small-argument smoothness around y = 0 (no 0/0 NaN)
    let f_zero = dual_to_primal_complex([0.0, 0.0], 0.5, 0.25).expect("zero dual");
    assert_eq!(f_zero, [0.0, 0.0]);
    let f_tiny = dual_to_primal_complex([1e-18, -2e-18], 0.5, 0.25).expect("tiny dual");
    assert!((f_tiny[0] - 4e-18).abs() < 1e-30);
    assert!((f_tiny[1] + 8e-18).abs() < 1e-30);

    // Extreme amplitude Hoch-Hore entropy evaluation (no u*u overflow)
    let huge_ent = hoch_hore_entropy_1d(&[[1e200, 0.0]], 1.0).expect("huge entropy");
    assert!(huge_ent.is_finite(), "entropy overflowed: {huge_ent}");

    // Hostile subnormal-square parameters (def = 1e-200 would underflow def*def to 0.0)
    assert!(reconstruct_maxent_1d(
        &[[1.0, 0.0]; 8],
        &[true; 8],
        &MaxEntConfig::fixed(1e-200, 1.0)
    )
    .is_err());
    assert!(reconstruct_maxent_1d(
        &[[1.0, 0.0]; 8],
        &[true; 8],
        &MaxEntConfig::discrepancy(1.0, 1e-200)
    )
    .is_err());

    // Zero dual input with extreme def / sigma_def ratio must not produce 0 * Inf = NaN
    let zero_extreme = dual_to_primal_complex([0.0, 0.0], 1e200, 1e-200).expect("zero extreme");
    assert_eq!(zero_extreme, [0.0, 0.0]);

    // KKT helpers must reject NaN / Inf inputs (never silently return Ok(0.0))
    assert!(kkt_residual_inf_1d(
        &[[f64::NAN, 0.0]; 4],
        &[[0.0, 0.0]; 4],
        &[true; 4],
        1.0,
        1.0
    )
    .is_err());
    assert!(kkt_residual_inf_2d(
        &[[f64::NAN, 0.0, 0.0, 0.0]; 4],
        2,
        2,
        &[[0.0; 4]; 4],
        2,
        2,
        &[true; 4],
        [QuadMode::States, QuadMode::States],
        1.0,
        1.0
    )
    .is_err());

    // Zero-signal FID handled cleanly in both Fixed and DiscrepancyPrinciple modes
    let zero_cfg = MaxEntConfig {
        def: 0.1,
        lambda_mode: MaxEntLambdaMode::DiscrepancyPrinciple {
            noise_sigma: 0.1,
            target_chi2_nu: 1.0,
            chi2_tol: 0.02,
            max_outer_iterations: 16,
        },
        noise_sigma: Some(0.1),
        ..MaxEntConfig::default()
    };
    let json_cfg = serde_json::to_string(&zero_cfg).expect("serialize MaxEntConfig");
    let roundtrip_cfg: MaxEntConfig =
        serde_json::from_str(&json_cfg).expect("deserialize MaxEntConfig");
    assert_eq!(roundtrip_cfg, zero_cfg);

    let zero_res =
        reconstruct_maxent_1d(&[[0.0, 0.0]; 16], &[true; 16], &zero_cfg).expect("zero FID");
    assert!(zero_res.converged);
    assert!(zero_res.spectrum.iter().all(|c| c[0] == 0.0 && c[1] == 0.0));
}

#[test]
fn synthetic_self_consistency_hostile_3d_memory_and_iteration_caps() {
    // Regression test for 22.13:P1-2 (unbounded 3D nx * spec_plane_len & per-thread workspace OOM)
    // and 22.13:P2-3 (hard MAX_MAXENT_ITERATIONS / MAX_MAXENT_OUTER_ITERATIONS caps).
    // Every hostile input below MUST fail closed BEFORE allocating with zero OOM.

    // 1. Hostile 3D output explosion: nx = 16_384, n2 = 1, n1 = 1, zf = 1024.
    //    Time-domain voxels = 16_384 (<= MAX_MAXENT_3D_TOTAL_POINTS), plane FFT = 1_048_576
    //    (<= MAX_MAXENT_POINTS), obs_planes matches nx*n2*n1, YET nx * spec_plane_len = 17.18B
    //    hypercomplex voxels (~550 GB) > MAX_MAXENT_3D_TOTAL_VOXELS!
    let nx_hostile = 16_384usize;
    let obs_small = vec![[1.0, 0.0, 0.0, 0.0]; nx_hostile];
    let mut cfg_zf = MaxEntConfig::fixed(0.1, 1.0);
    cfg_zf.zero_fill_factor = 1024;
    let err_vox = reconstruct_maxent_3d(
        &obs_small,
        nx_hostile,
        1,
        1,
        &[true],
        [QuadMode::States, QuadMode::States],
        &cfg_zf,
    )
    .expect_err("hostile nx * spec_plane_len > MAX_MAXENT_3D_TOTAL_VOXELS must fail closed");
    assert!(
        err_vox.contains("MAX_MAXENT_3D_TOTAL_VOXELS"),
        "expected MAX_MAXENT_3D_TOTAL_VOXELS error, got: {err_vox}"
    );

    // 2. Hostile per-thread workspace explosion: nx = 2, n2 = 1, n1 = 1, zf = 1024.
    //    total_spec_voxels = 2_097_152 (<= MAX_MAXENT_3D_TOTAL_VOXELS), spec_plane_len = 1_048_576
    //    (<= MAX_MAXENT_POINTS), YET Solve2DWorkspace + export = 268.4 MiB > MAX_MAXENT_WORKSPACE_BYTES!
    let ws_bytes = maxent_2d_workspace_bytes(1, 1, 1024, 1024).expect("workspace bytes");
    assert!(
        ws_bytes > MAX_MAXENT_WORKSPACE_BYTES,
        "expected 1024x1024 2D workspace ({ws_bytes} B) > MAX_MAXENT_WORKSPACE_BYTES ({MAX_MAXENT_WORKSPACE_BYTES} B)"
    );
    let err_ws_3d = reconstruct_maxent_3d(
        &[[1.0, 0.0, 0.0, 0.0]; 2],
        2,
        1,
        1,
        &[true],
        [QuadMode::States, QuadMode::States],
        &cfg_zf,
    )
    .expect_err("hostile 3D per-thread workspace > MAX_MAXENT_WORKSPACE_BYTES must fail closed");
    assert!(
        err_ws_3d.contains("MAX_MAXENT_WORKSPACE_BYTES"),
        "expected MAX_MAXENT_WORKSPACE_BYTES error, got: {err_ws_3d}"
    );
    let err_ws_2d = reconstruct_maxent_2d(
        &[[1.0, 0.0, 0.0, 0.0]],
        1,
        1,
        &[true],
        [QuadMode::States, QuadMode::States],
        &cfg_zf,
    )
    .expect_err("hostile 2D workspace > MAX_MAXENT_WORKSPACE_BYTES must fail closed");
    assert!(
        err_ws_2d.contains("MAX_MAXENT_WORKSPACE_BYTES"),
        "expected MAX_MAXENT_WORKSPACE_BYTES error, got: {err_ws_2d}"
    );

    // 3. Checked-mul overflow cases (usize::MAX dimensions / zero_fill_factor)
    let mut cfg_overflow = MaxEntConfig::fixed(0.1, 1.0);
    cfg_overflow.zero_fill_factor = usize::MAX;
    assert!(reconstruct_maxent_3d(
        &[[1.0, 0.0, 0.0, 0.0]],
        1,
        2,
        2,
        &[true; 4],
        [QuadMode::States, QuadMode::States],
        &cfg_overflow,
    )
    .is_err());
    assert!(reconstruct_maxent_3d(
        &[],
        usize::MAX,
        2,
        2,
        &[true; 4],
        [QuadMode::States, QuadMode::States],
        &MaxEntConfig::fixed(0.1, 1.0),
    )
    .is_err());
    assert!(maxent_1d_workspace_bytes(usize::MAX, usize::MAX).is_err());
    assert!(maxent_2d_workspace_bytes(usize::MAX, 2, 2, 2).is_err());

    // 4. Verify constants & exact boundary check for MAX_MAXENT_3D_TOTAL_VOXELS
    let mut cfg_border = MaxEntConfig::fixed(0.1, 1.0);
    cfg_border.zero_fill_factor = 64; // spec_plane_len = 4096 (workspace ~1.05 MiB, valid)
    let nx_over = (MAX_MAXENT_3D_TOTAL_VOXELS / 4096) + 1;
    let err_border = reconstruct_maxent_3d(
        &[],
        nx_over,
        1,
        1,
        &[true],
        [QuadMode::States, QuadMode::States],
        &cfg_border,
    )
    .expect_err("nx * spec_plane_len just over MAX_MAXENT_3D_TOTAL_VOXELS must fail closed");
    assert!(
        err_border.contains("MAX_MAXENT_3D_TOTAL_VOXELS"),
        "expected MAX_MAXENT_3D_TOTAL_VOXELS error, got: {err_border}"
    );

    // 5. Hard iteration caps (22.13:P2-3): max_iterations and max_outer_iterations
    let mut over_iters = MaxEntConfig::fixed(0.1, 1.0);
    over_iters.max_iterations = MAX_MAXENT_ITERATIONS + 1;
    assert!(over_iters.validate().is_err());
    assert!(reconstruct_maxent_1d(&[[1.0, 0.0]; 4], &[true; 4], &over_iters).is_err());
    assert!(reconstruct_maxent_2d(
        &[[1.0, 0.0, 0.0, 0.0]; 4],
        2,
        2,
        &[true; 4],
        [QuadMode::States, QuadMode::States],
        &over_iters,
    )
    .is_err());
    assert!(reconstruct_maxent_3d(
        &[[1.0, 0.0, 0.0, 0.0]; 4],
        1,
        2,
        2,
        &[true; 4],
        [QuadMode::States, QuadMode::States],
        &over_iters,
    )
    .is_err());

    let mut huge_iters = MaxEntConfig::fixed(0.1, 1.0);
    huge_iters.max_iterations = usize::MAX;
    assert!(huge_iters.validate().is_err());

    let over_outer = MaxEntConfig {
        def: 0.1,
        lambda_mode: MaxEntLambdaMode::DiscrepancyPrinciple {
            noise_sigma: 0.1,
            target_chi2_nu: 1.0,
            chi2_tol: 0.02,
            max_outer_iterations: MAX_MAXENT_OUTER_ITERATIONS + 1,
        },
        noise_sigma: Some(0.1),
        ..MaxEntConfig::default()
    };
    assert!(over_outer.validate().is_err());
    assert!(reconstruct_maxent_1d(&[[1.0, 0.0]; 4], &[true; 4], &over_outer).is_err());
}
