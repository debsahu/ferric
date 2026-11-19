//! Comprehensive `[synthetic self-consistency]` verification suite for `T22.1`:
//! Coupled 2D/3D multi-peak Voigt/Lorentzian/Gaussian lineshape deconvolution via separable
//! Variable Projection (`VarPro`) nonlinear least squares and exact Cramér–Rao amplitude
//! covariance.
//!
//! Provenance: Clean-room verification citing Golub & Pereyra (*SIAM J. Numer. Anal.* 1973),
//! Kaufman (*BIT* 1975), Thompson, Cox & Hastings (*J. Appl. Crystallogr.* 1987), Delaglio et al.
//! (*J. Biomol. NMR* 1995), and Cramér (1946) / Rao (1945).

use nmr_core::deconv_nd::{
    deconvolve_plane_series_2d, deconvolve_roi_2d, deconvolve_roi_3d, eval_lineshape_1d,
    eval_varpro_residual_and_jacobian_2d, eval_varpro_residual_and_jacobian_3d, lineshape_area_1d,
    synthesize_roi_2d, synthesize_roi_3d, thompson_cox_hastings_voigt, DeconvConfig,
    DeconvDiagnostic, DeconvError, LineshapeKind, LinewidthSharing, PeakModel2D, PeakModel3D,
    PeakSeed2D, PeakSeed3D, PlaneSeries2D, Roi2D, Roi3D, MAX_DECONV_PEAKS_PER_CLUSTER,
    MAX_DECONV_PLANES, MAX_DECONV_VOXELS,
};
use nmr_core::relax_fit::{fit_decay_curve_cpu, DecayCurveInput, ExpFitConfig, OffsetModelGate};
use nmr_core::rng::Rng;
use nmr_core::spectrum::{AxisPpm, Spectrum2D, Spectrum3D};

fn linspace(start: f64, end: f64, n: usize) -> Vec<f64> {
    if n <= 1 {
        return vec![start];
    }
    let step = (end - start) / ((n - 1) as f64);
    (0..n).map(|i| start + (i as f64) * step).collect()
}

fn rel_l2_err(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let num = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt();
    let den = b.iter().map(|&y| y * y).sum::<f64>().sqrt().max(1.0e-15);
    num / den
}

/// `[synthetic self-consistency]` Verify closed-form 1D lineshape derivatives (`d_mu`, `d_fwhm`,
/// `d_eta`) and analytical 1D integrals against central finite differences and numerical quadrature
/// for Lorentzian, Gaussian, and PseudoVoigt.
#[test]
fn lineshape_1d_derivatives_area_and_tch_voigt() {
    let eps = 1.0e-6;
    let xs = [-4.2, -1.5, -0.3, 0.0, 0.45, 1.8, 5.1];
    let mu = 0.25;
    let fwhm = 2.4;
    let eta = 0.42;

    for kind in [
        LineshapeKind::Lorentzian,
        LineshapeKind::Gaussian,
        LineshapeKind::PseudoVoigt,
    ] {
        for &x in &xs {
            let e = eval_lineshape_1d(kind, x, mu, fwhm, eta);
            let fd_mu = (eval_lineshape_1d(kind, x, mu + eps, fwhm, eta).value
                - eval_lineshape_1d(kind, x, mu - eps, fwhm, eta).value)
                / (2.0 * eps);
            let fd_fwhm = (eval_lineshape_1d(kind, x, mu, fwhm + eps, eta).value
                - eval_lineshape_1d(kind, x, mu, fwhm - eps, eta).value)
                / (2.0 * eps);
            let fd_eta = (eval_lineshape_1d(kind, x, mu, fwhm, eta + eps).value
                - eval_lineshape_1d(kind, x, mu, fwhm, eta - eps).value)
                / (2.0 * eps);

            assert!(
                (e.d_mu - fd_mu).abs() < 1.0e-7 * fd_mu.abs().max(1.0),
                "{kind:?} d_mu mismatch at x={x}: ana={}, fd={fd_mu}",
                e.d_mu
            );
            assert!(
                (e.d_fwhm - fd_fwhm).abs() < 1.0e-7 * fd_fwhm.abs().max(1.0),
                "{kind:?} d_fwhm mismatch at x={x}: ana={}, fd={fd_fwhm}",
                e.d_fwhm
            );
            assert!(
                (e.d_eta - fd_eta).abs() < 1.0e-7 * fd_eta.abs().max(1.0),
                "{kind:?} d_eta mismatch at x={x}: ana={}, fd={fd_eta}",
                e.d_eta
            );
        }

        // Verify half-maximum property f(mu +/- fwhm/2) == 0.5 and f(mu) == 1.0
        assert!((eval_lineshape_1d(kind, mu, mu, fwhm, eta).value - 1.0).abs() < 1.0e-14);
        assert!(
            (eval_lineshape_1d(kind, mu + 0.5 * fwhm, mu, fwhm, eta).value - 0.5).abs() < 1.0e-14
        );
        assert!(
            (eval_lineshape_1d(kind, mu - 0.5 * fwhm, mu, fwhm, eta).value - 0.5).abs() < 1.0e-14
        );

        // Verify analytical 1D area against wide Riemann sum (Gaussian & PseudoVoigt/Lorentzian)
        let area = lineshape_area_1d(kind, fwhm, eta);
        let dx = 0.01;
        let half_n = 40_000_i64;
        let num_area: f64 = (-half_n..=half_n)
            .map(|i| eval_lineshape_1d(kind, mu + (i as f64) * dx, mu, fwhm, eta).value * dx)
            .sum();
        let tol = if kind == LineshapeKind::Gaussian {
            1.0e-8
        } else {
            5.0e-3 // Lorentzian slow 1/x^2 tails beyond +-400
        };
        assert!(
            (area - num_area).abs() / area < tol,
            "{kind:?} area mismatch: ana={area}, num={num_area}"
        );
    }

    // Thompson–Cox–Hastings Voigt boundary & intermediate sanity
    let (fv_g, eta_g) = thompson_cox_hastings_voigt(3.0, 0.0);
    assert!((fv_g - 3.0).abs() < 1.0e-12 && eta_g.abs() < 1.0e-12);
    let (fv_l, eta_l) = thompson_cox_hastings_voigt(0.0, 3.0);
    assert!((fv_l - 3.0).abs() < 1.0e-12 && (eta_l - 1.0).abs() < 1.0e-12);
    let (fv_m, eta_m) = thompson_cox_hastings_voigt(2.0, 2.0);
    assert!(fv_m > 3.0 && fv_m < 3.6);
    assert!(eta_m > 0.5 && eta_m < 0.75);
}

/// `[synthetic self-consistency]` Verify exact Golub–Pereyra VarPro Jacobian against central
/// finite differences (`< 1e-6` relative error) for Lorentzian, Gaussian, and PseudoVoigt in 2D
/// (both `SharedAcrossPlanes` and `SharedPerCluster`) at strictly non-zero residual so both the
/// Kaufman term and the Golub–Pereyra correction term are rigorously verified.
#[test]
fn varpro_jacobian_matches_finite_differences_2d() {
    let c1 = linspace(0.0, 14.0, 15);
    let c2 = linspace(0.0, 16.0, 17);
    let true_peaks = [
        PeakModel2D {
            amplitude: 12.0,
            center_f1: 5.3,
            center_f2: 6.4,
            fwhm_f1: 2.6,
            fwhm_f2: 2.2,
            eta_f1: 0.65,
            eta_f2: 0.35,
        },
        PeakModel2D {
            amplitude: 8.5,
            center_f1: 7.4,
            center_f2: 8.1,
            fwhm_f1: 2.6,
            fwhm_f2: 2.8,
            eta_f1: 0.65,
            eta_f2: 0.55,
        },
    ];

    // Perturbed evaluation point so residual r != 0 (activates Golub-Pereyra 2nd term!)
    let base_seeds = [
        PeakSeed2D {
            center_f1: 5.55,
            center_f2: 6.15,
            fwhm_f1: 2.35,
            fwhm_f2: 2.45,
            eta_f1: 0.52,
            eta_f2: 0.44,
        },
        PeakSeed2D {
            center_f1: 7.15,
            center_f2: 8.35,
            fwhm_f1: 2.85,
            fwhm_f2: 2.55,
            eta_f1: 0.58,
            eta_f2: 0.48,
        },
    ];

    let eps = 1.0e-6;
    let mut rng = Rng::new(2026_2201);

    for kind in [
        LineshapeKind::Lorentzian,
        LineshapeKind::Gaussian,
        LineshapeKind::PseudoVoigt,
    ] {
        let mut roi = synthesize_roi_2d(&c1, &c2, kind, &true_peaks, 0.35).expect("synth 2D");
        // Add non-zero deterministic noise so residual is substantial.
        for v in &mut roi.values {
            *v += rng.normal(0.0, 0.4);
        }

        for sharing in [
            LinewidthSharing::SharedAcrossPlanes,
            LinewidthSharing::SharedPerCluster,
        ] {
            let cfg = DeconvConfig {
                lineshape: kind,
                linewidth_sharing: sharing,
                fit_mixing: kind == LineshapeKind::PseudoVoigt,
                fit_baseline: true,
                ..DeconvConfig::default()
            };

            let seeds = if sharing == LinewidthSharing::SharedPerCluster {
                // Make F1 seed widths and etas match the shared mean used by ParamLayout
                let mut s = base_seeds;
                let mf1 = 0.5 * (s[0].fwhm_f1 + s[1].fwhm_f1);
                let me1 = 0.5 * (s[0].eta_f1 + s[1].eta_f1);
                s[0].fwhm_f1 = mf1;
                s[1].fwhm_f1 = mf1;
                s[0].eta_f1 = me1;
                s[1].eta_f1 = me1;
                s
            } else {
                base_seeds
            };

            let (_yhat0, resid0, jac, _kappa) =
                eval_varpro_residual_and_jacobian_2d(&roi, &seeds, &cfg).expect("eval 2D jac");
            let rss0: f64 = resid0.iter().map(|r| r * r).sum();
            assert!(rss0 > 1.0, "residual must be non-zero to test GP term");

            // Perturb each nonlinear parameter in exact ParamLayout order:
            // [centers (k=0..2, d=0..2), fwhms, etas]
            type Pert2D = Box<dyn Fn(&mut [PeakSeed2D], f64)>;
            let mut perturbations: Vec<Pert2D> = vec![
                Box::new(|s, d| s[0].center_f1 += d),
                Box::new(|s, d| s[0].center_f2 += d),
                Box::new(|s, d| s[1].center_f1 += d),
                Box::new(|s, d| s[1].center_f2 += d),
            ];

            match sharing {
                LinewidthSharing::SharedPerCluster => {
                    perturbations.push(Box::new(|s, d| {
                        s[0].fwhm_f1 += d;
                        s[1].fwhm_f1 += d;
                    }));
                    perturbations.push(Box::new(|s, d| s[0].fwhm_f2 += d));
                    perturbations.push(Box::new(|s, d| s[1].fwhm_f2 += d));
                }
                _ => {
                    perturbations.push(Box::new(|s, d| s[0].fwhm_f1 += d));
                    perturbations.push(Box::new(|s, d| s[0].fwhm_f2 += d));
                    perturbations.push(Box::new(|s, d| s[1].fwhm_f1 += d));
                    perturbations.push(Box::new(|s, d| s[1].fwhm_f2 += d));
                }
            }

            if kind == LineshapeKind::PseudoVoigt {
                match sharing {
                    LinewidthSharing::SharedPerCluster => {
                        perturbations.push(Box::new(|s, d| {
                            s[0].eta_f1 += d;
                            s[1].eta_f1 += d;
                        }));
                        perturbations.push(Box::new(|s, d| s[0].eta_f2 += d));
                        perturbations.push(Box::new(|s, d| s[1].eta_f2 += d));
                    }
                    _ => {
                        perturbations.push(Box::new(|s, d| s[0].eta_f1 += d));
                        perturbations.push(Box::new(|s, d| s[0].eta_f2 += d));
                        perturbations.push(Box::new(|s, d| s[1].eta_f1 += d));
                        perturbations.push(Box::new(|s, d| s[1].eta_f2 += d));
                    }
                }
            }

            assert_eq!(jac.len(), perturbations.len());
            for (j, pert) in perturbations.iter().enumerate() {
                let mut sp = seeds;
                let mut sm = seeds;
                pert(&mut sp, eps);
                pert(&mut sm, -eps);
                let (yp, _, _, _) =
                    eval_varpro_residual_and_jacobian_2d(&roi, &sp, &cfg).expect("fd +");
                let (ym, _, _, _) =
                    eval_varpro_residual_and_jacobian_2d(&roi, &sm, &cfg).expect("fd -");
                let fd_col: Vec<f64> = yp
                    .iter()
                    .zip(ym.iter())
                    .map(|(&a, &b)| (a - b) / (2.0 * eps))
                    .collect();
                let err = rel_l2_err(&jac[j], &fd_col);
                assert!(
                    err < 1.0e-6,
                    "2D VarPro Jacobian mismatch ({kind:?}, {sharing:?}) col {j}: rel_l2={err:.3e}"
                );
            }
        }
    }
}

/// `[synthetic self-consistency]` Verify exact Golub–Pereyra VarPro Jacobian against central
/// finite differences (`< 1e-6` relative error) for Lorentzian, Gaussian, and PseudoVoigt in 3D
/// at strictly non-zero residual.
#[test]
fn varpro_jacobian_matches_finite_differences_3d() {
    let c1 = linspace(0.0, 8.0, 9);
    let c2 = linspace(0.0, 8.0, 9);
    let c3 = linspace(0.0, 8.0, 9);
    let true_peaks = [
        PeakModel3D {
            amplitude: 15.0,
            center_f1: 3.1,
            center_f2: 3.4,
            center_f3: 3.6,
            fwhm_f1: 2.2,
            fwhm_f2: 2.0,
            fwhm_f3: 1.9,
            eta_f1: 0.4,
            eta_f2: 0.6,
            eta_f3: 0.5,
        },
        PeakModel3D {
            amplitude: -9.5,
            center_f1: 5.0,
            center_f2: 4.8,
            center_f3: 4.9,
            fwhm_f1: 2.2,
            fwhm_f2: 2.0,
            fwhm_f3: 2.3,
            eta_f1: 0.4,
            eta_f2: 0.6,
            eta_f3: 0.3,
        },
    ];
    let seeds = [
        PeakSeed3D {
            center_f1: 3.3,
            center_f2: 3.2,
            center_f3: 3.8,
            fwhm_f1: 2.4,
            fwhm_f2: 2.2,
            fwhm_f3: 2.1,
            eta_f1: 0.45,
            eta_f2: 0.55,
            eta_f3: 0.42,
        },
        PeakSeed3D {
            center_f1: 4.8,
            center_f2: 5.0,
            center_f3: 4.7,
            fwhm_f1: 2.1,
            fwhm_f2: 2.3,
            fwhm_f3: 2.0,
            eta_f1: 0.35,
            eta_f2: 0.50,
            eta_f3: 0.38,
        },
    ];

    let eps = 1.0e-6;
    let mut rng = Rng::new(2026_2202);

    for kind in [
        LineshapeKind::Lorentzian,
        LineshapeKind::Gaussian,
        LineshapeKind::PseudoVoigt,
    ] {
        let mut roi = synthesize_roi_3d(&c1, &c2, &c3, kind, &true_peaks, 0.0).expect("synth 3D");
        for v in &mut roi.values {
            *v += rng.normal(0.0, 0.3);
        }

        let cfg = DeconvConfig {
            lineshape: kind,
            linewidth_sharing: LinewidthSharing::SharedAcrossPlanes,
            fit_mixing: kind == LineshapeKind::PseudoVoigt,
            ..DeconvConfig::default()
        };

        let (_yhat0, resid0, jac, _kappa) =
            eval_varpro_residual_and_jacobian_3d(&roi, &seeds, &cfg).expect("eval 3D jac");
        let rss0: f64 = resid0.iter().map(|r| r * r).sum();
        assert!(rss0 > 1.0);

        type Pert3D = Box<dyn Fn(&mut [PeakSeed3D], f64)>;
        let mut perturbations: Vec<Pert3D> = Vec::new();
        for k in 0..2 {
            perturbations.push(Box::new(move |s, d| s[k].center_f1 += d));
            perturbations.push(Box::new(move |s, d| s[k].center_f2 += d));
            perturbations.push(Box::new(move |s, d| s[k].center_f3 += d));
        }
        for k in 0..2 {
            perturbations.push(Box::new(move |s, d| s[k].fwhm_f1 += d));
            perturbations.push(Box::new(move |s, d| s[k].fwhm_f2 += d));
            perturbations.push(Box::new(move |s, d| s[k].fwhm_f3 += d));
        }
        if kind == LineshapeKind::PseudoVoigt {
            for k in 0..2 {
                perturbations.push(Box::new(move |s, d| s[k].eta_f1 += d));
                perturbations.push(Box::new(move |s, d| s[k].eta_f2 += d));
                perturbations.push(Box::new(move |s, d| s[k].eta_f3 += d));
            }
        }

        assert_eq!(jac.len(), perturbations.len());
        for (j, pert) in perturbations.iter().enumerate() {
            let mut sp = seeds;
            let mut sm = seeds;
            pert(&mut sp, eps);
            pert(&mut sm, -eps);
            let (yp, _, _, _) =
                eval_varpro_residual_and_jacobian_3d(&roi, &sp, &cfg).expect("3D fd +");
            let (ym, _, _, _) =
                eval_varpro_residual_and_jacobian_3d(&roi, &sm, &cfg).expect("3D fd -");
            let fd_col: Vec<f64> = yp
                .iter()
                .zip(ym.iter())
                .map(|(&a, &b)| (a - b) / (2.0 * eps))
                .collect();
            let err = rel_l2_err(&jac[j], &fd_col);
            assert!(
                err < 1.0e-6,
                "3D VarPro Jacobian mismatch ({kind:?}) col {j}: rel_l2={err:.3e}"
            );
        }
    }
}

/// `[synthetic self-consistency]` Crowded overlapping 2D doublet (`0.70 x FWHM` separation) and
/// triplet (`0.75 x FWHM` separation): verify `< 2%` amplitude error, sub-voxel center recovery,
/// strict variance inflation over naive fixed-shape $(\Phi^T \Phi)^{-1} \sigma^2$, and Monte
/// Carlo empirical covariance matching the analytical full Cramér–Rao covariance within `<= 15%`.
#[test]
fn crowded_2d_doublet_triplet_recovery_and_monte_carlo_crlb_covariance() {
    let c1 = linspace(0.0, 20.0, 21);
    let c2 = linspace(0.0, 20.0, 21);
    let fwhm = 3.2;
    // Normalized separation = sqrt((1.6/3.2)^2 + (1.6/3.2)^2) = 0.707 * FWHM (strongly overlapping!)
    let true_doublet = [
        PeakModel2D {
            amplitude: 100.0,
            center_f1: 9.2,
            center_f2: 9.2,
            fwhm_f1: fwhm,
            fwhm_f2: fwhm,
            eta_f1: 0.5,
            eta_f2: 0.5,
        },
        PeakModel2D {
            amplitude: 75.0,
            center_f1: 10.8,
            center_f2: 10.8,
            fwhm_f1: fwhm,
            fwhm_f2: fwhm,
            eta_f1: 0.5,
            eta_f2: 0.5,
        },
    ];
    let seeds_doublet = [
        PeakSeed2D {
            center_f1: 8.9,
            center_f2: 9.5,
            fwhm_f1: 2.8,
            fwhm_f2: 3.6,
            eta_f1: 0.5,
            eta_f2: 0.5,
        },
        PeakSeed2D {
            center_f1: 11.1,
            center_f2: 10.5,
            fwhm_f1: 3.6,
            fwhm_f2: 2.8,
            eta_f1: 0.5,
            eta_f2: 0.5,
        },
    ];

    let clean_doublet = synthesize_roi_2d(&c1, &c2, LineshapeKind::Lorentzian, &true_doublet, 0.0)
        .expect("synth doublet");
    let sigma = 0.25;
    let cfg = DeconvConfig {
        lineshape: LineshapeKind::Lorentzian,
        linewidth_sharing: LinewidthSharing::SharedAcrossPlanes,
        noise_sigma: Some(sigma),
        ..DeconvConfig::default()
    };

    // Single noisy realization: check < 2% amplitude & sub-pixel center recovery
    let mut rng = Rng::new(2026_2203);
    let mut noisy = clean_doublet.clone();
    for v in &mut noisy.values {
        *v += rng.normal(0.0, sigma);
    }
    let res = deconvolve_roi_2d(&noisy, &seeds_doublet, &cfg).expect("deconv doublet");
    assert_eq!(res.diagnostic, DeconvDiagnostic::Converged);
    assert!(res.converged);
    assert!((res.reduced_chi2 - 1.0).abs() < 0.35);

    for (k, td) in true_doublet.iter().enumerate() {
        let rel_amp_err = (res.peaks[k].amplitude - td.amplitude).abs() / td.amplitude;
        assert!(
            rel_amp_err < 0.02,
            "doublet peak {k} amplitude error {:.4}% >= 2%",
            rel_amp_err * 100.0
        );
        assert!((res.peaks[k].center_f1 - td.center_f1).abs() < 0.08);
        assert!((res.peaks[k].center_f2 - td.center_f2).abs() < 0.08);
    }

    // Verify full Cramér–Rao amplitude variance is strictly larger than naive fixed-shape (Phi^T Phi)^-1 sigma^2!
    let m = clean_doublet.values.len();
    let mut phi = vec![vec![0.0; m]; 2];
    for (k, phi_k) in phi.iter_mut().enumerate() {
        let pk = res.peaks[k];
        for (i1, &x1) in c1.iter().enumerate() {
            let v1 =
                eval_lineshape_1d(LineshapeKind::Lorentzian, x1, pk.center_f1, pk.fwhm_f1, 1.0)
                    .value;
            for (i2, &x2) in c2.iter().enumerate() {
                let v2 =
                    eval_lineshape_1d(LineshapeKind::Lorentzian, x2, pk.center_f2, pk.fwhm_f2, 1.0)
                        .value;
                phi_k[i1 * c2.len() + i2] = v1 * v2;
            }
        }
    }
    let g00: f64 = phi[0].iter().map(|x| x * x).sum();
    let g11: f64 = phi[1].iter().map(|x| x * x).sum();
    let g01: f64 = phi[0].iter().zip(phi[1].iter()).map(|(a, b)| a * b).sum();
    let det = g00 * g11 - g01 * g01;
    let naive_var0 = sigma * sigma * g11 / det;
    let naive_var1 = sigma * sigma * g00 / det;
    assert!(
        res.amplitude_covariance[0][0] > naive_var0 * 1.5,
        "full CRLB var0 ({}) should reflect nonlinear overlap inflation over naive fixed-shape ({})",
        res.amplitude_covariance[0][0],
        naive_var0
    );
    assert!(
        res.amplitude_covariance[1][1] > naive_var1 * 1.5,
        "full CRLB var1 ({}) should reflect nonlinear overlap inflation over naive fixed-shape ({})",
        res.amplitude_covariance[1][1],
        naive_var1
    );

    // Monte Carlo empirical covariance vs analytical full Cramér–Rao covariance (<= 15%)
    let n_mc = 240usize;
    let mut mc_a0 = Vec::with_capacity(n_mc);
    let mut mc_a1 = Vec::with_capacity(n_mc);
    let true_seeds = [
        PeakSeed2D {
            center_f1: true_doublet[0].center_f1,
            center_f2: true_doublet[0].center_f2,
            fwhm_f1: fwhm,
            fwhm_f2: fwhm,
            eta_f1: 1.0,
            eta_f2: 1.0,
        },
        PeakSeed2D {
            center_f1: true_doublet[1].center_f1,
            center_f2: true_doublet[1].center_f2,
            fwhm_f1: fwhm,
            fwhm_f2: fwhm,
            eta_f1: 1.0,
            eta_f2: 1.0,
        },
    ];
    // Evaluate exact theoretical CRLB at ground truth
    let crlb_ref = deconvolve_roi_2d(&clean_doublet, &true_seeds, &cfg).expect("crlb ref");
    let crlb_cov = &crlb_ref.amplitude_covariance;

    for _ in 0..n_mc {
        let mut trial = clean_doublet.clone();
        for v in &mut trial.values {
            *v += rng.normal(0.0, sigma);
        }
        let fit = deconvolve_roi_2d(&trial, &true_seeds, &cfg).expect("mc fit");
        assert!(fit.converged);
        mc_a0.push(fit.peaks[0].amplitude);
        mc_a1.push(fit.peaks[1].amplitude);
    }

    let mean0 = mc_a0.iter().sum::<f64>() / (n_mc as f64);
    let mean1 = mc_a1.iter().sum::<f64>() / (n_mc as f64);
    let emp_var0 = mc_a0.iter().map(|&a| (a - mean0).powi(2)).sum::<f64>() / ((n_mc - 1) as f64);
    let emp_var1 = mc_a1.iter().map(|&a| (a - mean1).powi(2)).sum::<f64>() / ((n_mc - 1) as f64);
    let emp_cov01 = mc_a0
        .iter()
        .zip(mc_a1.iter())
        .map(|(&a, &b)| (a - mean0) * (b - mean1))
        .sum::<f64>()
        / ((n_mc - 1) as f64);

    let se_rel0 = (emp_var0.sqrt() - crlb_cov[0][0].sqrt()).abs() / crlb_cov[0][0].sqrt();
    let se_rel1 = (emp_var1.sqrt() - crlb_cov[1][1].sqrt()).abs() / crlb_cov[1][1].sqrt();
    let cov_rel01 = (emp_cov01 - crlb_cov[0][1]).abs() / crlb_cov[0][1].abs();
    assert!(
        se_rel0 <= 0.15,
        "MC SE0 ({:.4}) vs CRLB SE0 ({:.4}): rel_err={:.3}",
        emp_var0.sqrt(),
        crlb_cov[0][0].sqrt(),
        se_rel0
    );
    assert!(
        se_rel1 <= 0.15,
        "MC SE1 ({:.4}) vs CRLB SE1 ({:.4}): rel_err={:.3}",
        emp_var1.sqrt(),
        crlb_cov[1][1].sqrt(),
        se_rel1
    );
    assert!(
        cov_rel01 <= 0.15,
        "MC Cov01 ({:.4}) vs CRLB Cov01 ({:.4}): rel_err={:.3}",
        emp_cov01,
        crlb_cov[0][1],
        cov_rel01
    );
    // Overlapping doublet amplitudes are strongly negatively correlated
    assert!(
        crlb_ref.amplitude_correlation[0][1] < -0.5,
        "expected strong negative overlap correlation, got {}",
        crlb_ref.amplitude_correlation[0][1]
    );

    // Crowded overlapping 2D triplet (~0.75 * FWHM pairwise separation, PseudoVoigt)
    let fwhm_t = 3.0;
    let true_triplet = [
        PeakModel2D {
            amplitude: 90.0,
            center_f1: 8.5,
            center_f2: 8.5,
            fwhm_f1: fwhm_t,
            fwhm_f2: fwhm_t,
            eta_f1: 0.4,
            eta_f2: 0.4,
        },
        PeakModel2D {
            amplitude: 115.0,
            center_f1: 10.75,
            center_f2: 8.5,
            fwhm_f1: fwhm_t,
            fwhm_f2: fwhm_t,
            eta_f1: 0.4,
            eta_f2: 0.4,
        },
        PeakModel2D {
            amplitude: 65.0,
            center_f1: 9.6,
            center_f2: 10.6,
            fwhm_f1: fwhm_t,
            fwhm_f2: fwhm_t,
            eta_f1: 0.4,
            eta_f2: 0.4,
        },
    ];
    let seeds_triplet = [
        PeakSeed2D {
            center_f1: 8.25,
            center_f2: 8.75,
            fwhm_f1: 2.7,
            fwhm_f2: 3.3,
            eta_f1: 0.5,
            eta_f2: 0.5,
        },
        PeakSeed2D {
            center_f1: 11.0,
            center_f2: 8.3,
            fwhm_f1: 3.3,
            fwhm_f2: 2.7,
            eta_f1: 0.5,
            eta_f2: 0.5,
        },
        PeakSeed2D {
            center_f1: 9.4,
            center_f2: 10.85,
            fwhm_f1: 2.8,
            fwhm_f2: 3.2,
            eta_f1: 0.5,
            eta_f2: 0.5,
        },
    ];
    let mut triplet_roi =
        synthesize_roi_2d(&c1, &c2, LineshapeKind::PseudoVoigt, &true_triplet, 0.0)
            .expect("synth triplet");
    for v in &mut triplet_roi.values {
        *v += rng.normal(0.0, 0.12);
    }
    let cfg_triplet = DeconvConfig {
        lineshape: LineshapeKind::PseudoVoigt,
        linewidth_sharing: LinewidthSharing::SharedPerCluster,
        fit_mixing: true,
        noise_sigma: Some(0.12),
        ..DeconvConfig::default()
    };
    let res_t =
        deconvolve_roi_2d(&triplet_roi, &seeds_triplet, &cfg_triplet).expect("deconv triplet");
    assert_eq!(res_t.diagnostic, DeconvDiagnostic::Converged);
    for (k, tt) in true_triplet.iter().enumerate() {
        let rel_err = (res_t.peaks[k].amplitude - tt.amplitude).abs() / tt.amplitude;
        assert!(
            rel_err < 0.02,
            "triplet peak {k} amplitude error {:.4}% >= 2%",
            rel_err * 100.0
        );
        assert!((res_t.peaks[k].center_f1 - tt.center_f1).abs() < 0.06);
        assert!((res_t.peaks[k].center_f2 - tt.center_f2).abs() < 0.06);
    }
    // Under SharedPerCluster, all 3 peaks share the exact same indirect F1 FWHM
    assert!((res_t.peaks[0].fwhm_f1 - res_t.peaks[1].fwhm_f1).abs() < 1.0e-12);
    assert!((res_t.peaks[0].fwhm_f1 - res_t.peaks[2].fwhm_f1).abs() < 1.0e-12);
}

/// `[synthetic self-consistency]` Coupled 8-plane pseudo-3D $R_2$ relaxation series with two
/// strongly overlapping peaks (`R_2 = 8.0 s^-1` vs `22.0 s^-1`): verify coupled VarPro accurately
/// recovers both relaxation decay rates via `fit_decay_curve_cpu`, whereas naive peak-height
/// picking suffers severe cross-talk bias.
#[test]
fn coupled_8plane_r2_relaxation_series_beats_naive_peak_picking() {
    let delays = vec![0.010, 0.020, 0.040, 0.060, 0.080, 0.110, 0.150, 0.200];
    let r2_slow = 8.0_f64;
    let r2_fast = 22.0_f64;
    let i0_slow = 100.0_f64;
    let i0_fast = 95.0_f64;

    let c1 = linspace(0.0, 20.0, 21);
    let c2 = linspace(0.0, 20.0, 21);
    // Two strongly overlapping peaks (~0.62 x FWHM separation) with distinct intrinsic linewidths
    let p_slow_c1 = 9.0;
    let p_slow_c2 = 9.0;
    let p_fast_c1 = 10.4;
    let p_fast_c2 = 10.4;
    let fwhm_slow = 3.2;
    let fwhm_fast = 3.4;

    let mut rng = Rng::new(2026_2204);
    let noise_sigma = 0.25;
    let mut plane_values = Vec::with_capacity(delays.len());
    let mut naive_slow_heights = Vec::with_capacity(delays.len());
    let mut naive_fast_heights = Vec::with_capacity(delays.len());

    // Nearest grid indices to the two peak centers for naive height picking
    let idx_slow = 9 * c2.len() + 9;
    let idx_fast = 10 * c2.len() + 10;

    for &t in &delays {
        let a_slow = i0_slow * (-r2_slow * t).exp();
        let a_fast = i0_fast * (-r2_fast * t).exp();
        let peaks = [
            PeakModel2D {
                amplitude: a_slow,
                center_f1: p_slow_c1,
                center_f2: p_slow_c2,
                fwhm_f1: fwhm_slow,
                fwhm_f2: fwhm_slow,
                eta_f1: 0.6,
                eta_f2: 0.6,
            },
            PeakModel2D {
                amplitude: a_fast,
                center_f1: p_fast_c1,
                center_f2: p_fast_c2,
                fwhm_f1: fwhm_fast,
                fwhm_f2: fwhm_fast,
                eta_f1: 0.6,
                eta_f2: 0.6,
            },
        ];
        let mut roi =
            synthesize_roi_2d(&c1, &c2, LineshapeKind::PseudoVoigt, &peaks, 0.0).expect("plane");
        for v in &mut roi.values {
            *v += rng.normal(0.0, noise_sigma);
        }
        naive_slow_heights.push(roi.values[idx_slow]);
        naive_fast_heights.push(roi.values[idx_fast]);
        plane_values.push(roi.values);
    }

    let series = PlaneSeries2D {
        coords_f1: c1,
        coords_f2: c2,
        plane_values,
        plane_coords: delays.clone(),
    };
    let seeds = [
        PeakSeed2D {
            center_f1: 8.75,
            center_f2: 9.25,
            fwhm_f1: 2.7,
            fwhm_f2: 3.3,
            eta_f1: 0.5,
            eta_f2: 0.5,
        },
        PeakSeed2D {
            center_f1: 10.85,
            center_f2: 10.35,
            fwhm_f1: 3.7,
            fwhm_f2: 3.1,
            eta_f1: 0.5,
            eta_f2: 0.5,
        },
    ];
    let cfg = DeconvConfig {
        lineshape: LineshapeKind::PseudoVoigt,
        linewidth_sharing: LinewidthSharing::SharedAcrossPlanes,
        fit_mixing: true,
        noise_sigma: Some(noise_sigma),
        ..DeconvConfig::default()
    };

    let res = deconvolve_plane_series_2d(&series, &seeds, &cfg).expect("plane series deconv");
    assert_eq!(res.diagnostic, DeconvDiagnostic::Converged);
    assert!(res.converged);
    assert!((res.reduced_chi2 - 1.0).abs() < 0.25);

    // Fit exponential decay rates to VarPro deconvolved amplitudes vs naive peak heights
    let exp_cfg = ExpFitConfig {
        offset_gate: OffsetModelGate::Disabled,
        mc_replicates: 0,
        seed: 42,
        max_rate_s_inv: 200.0,
    };

    let fit_vp_slow = fit_decay_curve_cpu(
        &DecayCurveInput {
            seq_id: 1,
            res_name: "ALA".to_string(),
            times_sec: delays.clone(),
            intensities: res.peaks[0].amplitudes.clone(),
            sigmas: res.peaks[0].amplitude_se.clone(),
        },
        &exp_cfg,
    )
    .expect("fit vp slow");
    let fit_vp_fast = fit_decay_curve_cpu(
        &DecayCurveInput {
            seq_id: 2,
            res_name: "GLY".to_string(),
            times_sec: delays.clone(),
            intensities: res.peaks[1].amplitudes.clone(),
            sigmas: res.peaks[1].amplitude_se.clone(),
        },
        &exp_cfg,
    )
    .expect("fit vp fast");

    let fit_naive_fast = fit_decay_curve_cpu(
        &DecayCurveInput {
            seq_id: 2,
            res_name: "GLY".to_string(),
            times_sec: delays.clone(),
            intensities: naive_fast_heights,
            sigmas: vec![noise_sigma; delays.len()],
        },
        &exp_cfg,
    )
    .expect("fit naive fast");

    let err_vp_slow = (fit_vp_slow.rate - r2_slow).abs() / r2_slow;
    let err_vp_fast = (fit_vp_fast.rate - r2_fast).abs() / r2_fast;
    let err_naive_fast = (fit_naive_fast.rate - r2_fast).abs() / r2_fast;

    assert!(
        err_vp_slow < 0.02,
        "VarPro slow R2 = {:.3} (true {r2_slow}), rel_err = {:.3}%",
        fit_vp_slow.rate,
        err_vp_slow * 100.0
    );
    assert!(
        err_vp_fast < 0.03,
        "VarPro fast R2 = {:.3} (true {r2_fast}), rel_err = {:.3}%",
        fit_vp_fast.rate,
        err_vp_fast * 100.0
    );
    assert!(
        err_naive_fast > 0.20,
        "expected naive peak height fast R2 ({:.3}) to be severely biased (>20%), got {:.3}%",
        fit_naive_fast.rate,
        err_naive_fast * 100.0
    );
}

/// `[synthetic self-consistency]` Test `LinewidthSharing::Independent` across a 3-plane
/// exchange-broadening series where peak linewidths change between planes while centers stay locked.
#[test]
fn plane_series_independent_linewidths_exchange_broadening() {
    let c1 = linspace(0.0, 16.0, 17);
    let c2 = linspace(0.0, 16.0, 17);
    let plane_fwhms = [(2.2, 2.0), (2.8, 2.5), (3.6, 3.1)];
    let mut plane_values = Vec::new();
    let mut rng = Rng::new(2026_2205);

    for &(fw1, fw2) in &plane_fwhms {
        let pks = [
            PeakModel2D {
                amplitude: 80.0,
                center_f1: 6.5,
                center_f2: 7.0,
                fwhm_f1: fw1,
                fwhm_f2: fw2,
                eta_f1: 1.0,
                eta_f2: 1.0,
            },
            PeakModel2D {
                amplitude: 60.0,
                center_f1: 9.8,
                center_f2: 9.5,
                fwhm_f1: 2.4,
                fwhm_f2: 2.4,
                eta_f1: 1.0,
                eta_f2: 1.0,
            },
        ];
        let mut roi =
            synthesize_roi_2d(&c1, &c2, LineshapeKind::Lorentzian, &pks, 0.0).expect("plane");
        for v in &mut roi.values {
            *v += rng.normal(0.0, 0.08);
        }
        plane_values.push(roi.values);
    }

    let series = PlaneSeries2D {
        coords_f1: c1,
        coords_f2: c2,
        plane_values,
        plane_coords: vec![0.0, 1.0, 2.0],
    };
    let seeds = [
        PeakSeed2D {
            center_f1: 6.3,
            center_f2: 7.2,
            fwhm_f1: 2.6,
            fwhm_f2: 2.5,
            eta_f1: 1.0,
            eta_f2: 1.0,
        },
        PeakSeed2D {
            center_f1: 10.0,
            center_f2: 9.3,
            fwhm_f1: 2.5,
            fwhm_f2: 2.5,
            eta_f1: 1.0,
            eta_f2: 1.0,
        },
    ];
    let cfg = DeconvConfig {
        lineshape: LineshapeKind::Lorentzian,
        linewidth_sharing: LinewidthSharing::Independent,
        noise_sigma: Some(0.08),
        ..DeconvConfig::default()
    };

    let res = deconvolve_plane_series_2d(&series, &seeds, &cfg).expect("indep plane deconv");
    assert_eq!(res.diagnostic, DeconvDiagnostic::Converged);
    for (p, &(fw1_true, fw2_true)) in plane_fwhms.iter().enumerate() {
        assert!(
            (res.peaks[0].fwhm_f1[p] - fw1_true).abs() < 0.05,
            "plane {p} F1 FWHM mismatch: got {}, expected {fw1_true}",
            res.peaks[0].fwhm_f1[p]
        );
        assert!(
            (res.peaks[0].fwhm_f2[p] - fw2_true).abs() < 0.05,
            "plane {p} F2 FWHM mismatch: got {}, expected {fw2_true}",
            res.peaks[0].fwhm_f2[p]
        );
    }
}

/// `[synthetic self-consistency]` 3D overlapping peak cluster (including opposite-sign HNCACB-like
/// CB peak) with shared indirect (`F1`, `F2`) linewidths and distinct direct (`F3`) linewidths,
/// plus descending-ppm `Spectrum2D` / `Spectrum3D` window extraction verification.
#[test]
fn deconv_3d_overlapping_cluster_shared_indirect_linewidths_and_spectrum_windows() {
    let c1 = linspace(0.0, 10.0, 11);
    let c2 = linspace(0.0, 10.0, 11);
    let c3 = linspace(0.0, 12.0, 13);

    let shared_fwhm_f1 = 2.6;
    let shared_fwhm_f2 = 2.4;
    let true_3d = [
        PeakModel3D {
            amplitude: 120.0,
            center_f1: 4.2,
            center_f2: 4.3,
            center_f3: 5.0,
            fwhm_f1: shared_fwhm_f1,
            fwhm_f2: shared_fwhm_f2,
            fwhm_f3: 2.1,
            eta_f1: 0.35,
            eta_f2: 0.45,
            eta_f3: 0.5,
        },
        PeakModel3D {
            amplitude: -85.0, // negative-phase CB peak
            center_f1: 5.7,
            center_f2: 5.5,
            center_f3: 6.4,
            fwhm_f1: shared_fwhm_f1,
            fwhm_f2: shared_fwhm_f2,
            fwhm_f3: 2.9,
            eta_f1: 0.35,
            eta_f2: 0.45,
            eta_f3: 0.3,
        },
    ];
    let seeds_3d = [
        PeakSeed3D {
            center_f1: 4.0,
            center_f2: 4.5,
            center_f3: 4.8,
            fwhm_f1: 2.3,
            fwhm_f2: 2.7,
            fwhm_f3: 2.4,
            eta_f1: 0.5,
            eta_f2: 0.5,
            eta_f3: 0.5,
        },
        PeakSeed3D {
            center_f1: 5.9,
            center_f2: 5.3,
            center_f3: 6.6,
            fwhm_f1: 2.9,
            fwhm_f2: 2.1,
            fwhm_f3: 2.5,
            eta_f1: 0.5,
            eta_f2: 0.5,
            eta_f3: 0.5,
        },
    ];

    let mut roi = synthesize_roi_3d(&c1, &c2, &c3, LineshapeKind::PseudoVoigt, &true_3d, 0.0)
        .expect("synth 3D");
    let mut rng = Rng::new(2026_2206);
    for v in &mut roi.values {
        *v += rng.normal(0.0, 0.15);
    }

    let cfg = DeconvConfig {
        lineshape: LineshapeKind::PseudoVoigt,
        linewidth_sharing: LinewidthSharing::SharedPerCluster,
        fit_mixing: true,
        noise_sigma: Some(0.15),
        ..DeconvConfig::default()
    };
    let res = deconvolve_roi_3d(&roi, &seeds_3d, &cfg).expect("deconv 3D");
    assert_eq!(res.diagnostic, DeconvDiagnostic::Converged);
    assert!(res.converged);

    for (k, t3) in true_3d.iter().enumerate() {
        let rel_amp = (res.peaks[k].amplitude - t3.amplitude).abs() / t3.amplitude.abs();
        assert!(
            rel_amp < 0.015,
            "3D peak {k} amplitude error {:.3}% >= 1.5%",
            rel_amp * 100.0
        );
        assert!((res.peaks[k].center_f1 - t3.center_f1).abs() < 0.04);
        assert!((res.peaks[k].center_f2 - t3.center_f2).abs() < 0.04);
        assert!((res.peaks[k].center_f3 - t3.center_f3).abs() < 0.04);
        assert!((res.peaks[k].fwhm_f1 - shared_fwhm_f1).abs() < 0.05);
        assert!((res.peaks[k].fwhm_f2 - shared_fwhm_f2).abs() < 0.05);
        assert!((res.peaks[k].fwhm_f3 - t3.fwhm_f3).abs() < 0.06);
    }
    // SharedPerCluster enforces identical indirect F1 & F2 linewidths across peaks
    assert!((res.peaks[0].fwhm_f1 - res.peaks[1].fwhm_f1).abs() < 1.0e-12);
    assert!((res.peaks[0].fwhm_f2 - res.peaks[1].fwhm_f2).abs() < 1.0e-12);
    // Direct F3 linewidths remain distinct
    assert!((res.peaks[1].fwhm_f3 - res.peaks[0].fwhm_f3) > 0.6);

    // Test Spectrum2D and Spectrum3D descending-ppm window extraction + deconvolution
    let axis_h = AxisPpm::from_bounds("1H", 32, 9.0, 7.5, 600.0);
    let axis_n = AxisPpm::from_bounds("15N", 32, 125.0, 115.0, 60.8);
    let mut data_2d = vec![0.0_f32; 32 * 32];
    let true_n = 120.2;
    let true_h = 8.25;
    let fw_n = 0.9;
    let fw_h = 0.14;
    for iy in 0..32 {
        let pn = axis_n.index_to_ppm(iy as f64);
        let vn = eval_lineshape_1d(LineshapeKind::Gaussian, pn, true_n, fw_n, 0.0).value;
        for ix in 0..32 {
            let ph = axis_h.index_to_ppm(ix as f64);
            let vh = eval_lineshape_1d(LineshapeKind::Gaussian, ph, true_h, fw_h, 0.0).value;
            data_2d[iy * 32 + ix] = (50.0 * vn * vh) as f32;
        }
    }
    let spec2d = Spectrum2D {
        name: "synthetic_hsqc".to_string(),
        axis_x: axis_h.clone(),
        axis_y: axis_n.clone(),
        data: data_2d,
    };
    let win2d = Roi2D::from_spectrum_window(&spec2d, (118.0, 122.5), (7.9, 8.6)).expect("win2d");
    let cfg_g = DeconvConfig {
        lineshape: LineshapeKind::Gaussian,
        ..DeconvConfig::default()
    };
    let fit2d = deconvolve_roi_2d(
        &win2d,
        &[PeakSeed2D {
            center_f1: 120.0,
            center_f2: 8.20,
            fwhm_f1: 1.1,
            fwhm_f2: 0.18,
            eta_f1: 0.0,
            eta_f2: 0.0,
        }],
        &cfg_g,
    )
    .expect("fit2d ppm");
    assert_eq!(fit2d.diagnostic, DeconvDiagnostic::Converged);
    assert!((fit2d.peaks[0].center_f1 - true_n).abs() < 1.0e-3);
    assert!((fit2d.peaks[0].center_f2 - true_h).abs() < 1.0e-3);
    assert!((fit2d.peaks[0].amplitude - 50.0).abs() < 0.05);

    let axis_c = AxisPpm::from_bounds("13C", 16, 62.0, 52.0, 150.9);
    let mut data_3d = vec![0.0_f32; 16 * 16 * 16];
    let ax_n3 = AxisPpm::from_bounds("15N", 16, 123.0, 117.0, 60.8);
    let ax_h3 = AxisPpm::from_bounds("1H", 16, 8.6, 7.8, 600.0);
    for iz in 0..16 {
        let pc = axis_c.index_to_ppm(iz as f64);
        let vc = eval_lineshape_1d(LineshapeKind::Gaussian, pc, 57.0, 2.0, 0.0).value;
        for iy in 0..16 {
            let pn = ax_n3.index_to_ppm(iy as f64);
            let vn = eval_lineshape_1d(LineshapeKind::Gaussian, pn, 120.0, 1.2, 0.0).value;
            for ix in 0..16 {
                let ph = ax_h3.index_to_ppm(ix as f64);
                let vh = eval_lineshape_1d(LineshapeKind::Gaussian, ph, 8.2, 0.16, 0.0).value;
                data_3d[(iz * 16 + iy) * 16 + ix] = (40.0 * vc * vn * vh) as f32;
            }
        }
    }
    let spec3d = Spectrum3D {
        name: "synthetic_hnca".to_string(),
        exp: "HNCA".to_string(),
        axis_x: ax_h3,
        axis_y: ax_n3,
        axis_z: axis_c,
        data: data_3d,
    };
    let win3d = Roi3D::from_spectrum_window(&spec3d, (54.0, 60.0), (118.0, 122.0), (7.9, 8.5))
        .expect("win3d");
    let fit3d = deconvolve_roi_3d(
        &win3d,
        &[PeakSeed3D {
            center_f1: 56.7,
            center_f2: 120.2,
            center_f3: 8.15,
            fwhm_f1: 2.3,
            fwhm_f2: 1.4,
            fwhm_f3: 0.2,
            eta_f1: 0.0,
            eta_f2: 0.0,
            eta_f3: 0.0,
        }],
        &cfg_g,
    )
    .expect("fit3d ppm");
    assert_eq!(fit3d.diagnostic, DeconvDiagnostic::Converged);
    assert!((fit3d.peaks[0].center_f1 - 57.0).abs() < 2.0e-3);
    assert!((fit3d.peaks[0].center_f2 - 120.0).abs() < 2.0e-3);
    assert!((fit3d.peaks[0].center_f3 - 8.2).abs() < 2.0e-3);
}

/// `[synthetic self-consistency]` Verify that the block-Schur QR/SVD covariance matches a direct
/// dense $(J_{\text{full}}^T J_{\text{full}})^{-1} \sigma^2$ reference computation to `< 1e-9`.
#[test]
fn block_schur_covariance_matches_dense_fisher_inverse() {
    let c1 = linspace(0.0, 12.0, 13);
    let c2 = linspace(0.0, 12.0, 13);
    let pks = [
        PeakModel2D {
            amplitude: 50.0,
            center_f1: 5.2,
            center_f2: 5.5,
            fwhm_f1: 2.5,
            fwhm_f2: 2.3,
            eta_f1: 0.5,
            eta_f2: 0.5,
        },
        PeakModel2D {
            amplitude: 35.0,
            center_f1: 7.0,
            center_f2: 6.8,
            fwhm_f1: 2.8,
            fwhm_f2: 2.4,
            eta_f1: 0.5,
            eta_f2: 0.5,
        },
    ];
    let roi = synthesize_roi_2d(&c1, &c2, LineshapeKind::Gaussian, &pks, 0.0).expect("synth");
    let sigma = 0.2;
    let cfg = DeconvConfig {
        lineshape: LineshapeKind::Gaussian,
        linewidth_sharing: LinewidthSharing::SharedAcrossPlanes,
        noise_sigma: Some(sigma),
        ..DeconvConfig::default()
    };
    let seeds = [
        PeakSeed2D {
            center_f1: pks[0].center_f1,
            center_f2: pks[0].center_f2,
            fwhm_f1: pks[0].fwhm_f1,
            fwhm_f2: pks[0].fwhm_f2,
            eta_f1: 0.0,
            eta_f2: 0.0,
        },
        PeakSeed2D {
            center_f1: pks[1].center_f1,
            center_f2: pks[1].center_f2,
            fwhm_f1: pks[1].fwhm_f1,
            fwhm_f2: pks[1].fwhm_f2,
            eta_f1: 0.0,
            eta_f2: 0.0,
        },
    ];
    let res = deconvolve_roi_2d(&roi, &seeds, &cfg).expect("fit");

    // Build explicit dense full Jacobian J_full (M x 10): [a0, a1, c0_1, c0_2, c1_1, c1_2, f0_1, f0_2, f1_1, f1_2]
    let m = roi.values.len();
    let p_tot = 10usize;
    let mut j_cols = vec![vec![0.0; m]; p_tot];
    for (k, pk) in res.peaks.iter().enumerate() {
        for (i1, &x1) in c1.iter().enumerate() {
            let e1 = eval_lineshape_1d(LineshapeKind::Gaussian, x1, pk.center_f1, pk.fwhm_f1, 0.0);
            for (i2, &x2) in c2.iter().enumerate() {
                let e2 =
                    eval_lineshape_1d(LineshapeKind::Gaussian, x2, pk.center_f2, pk.fwhm_f2, 0.0);
                let idx = i1 * c2.len() + i2;
                j_cols[k][idx] = e1.value * e2.value;
                j_cols[2 + 2 * k][idx] = pk.amplitude * e1.d_mu * e2.value;
                j_cols[2 + 2 * k + 1][idx] = pk.amplitude * e1.value * e2.d_mu;
                j_cols[6 + 2 * k][idx] = pk.amplitude * e1.d_fwhm * e2.value;
                j_cols[6 + 2 * k + 1][idx] = pk.amplitude * e1.value * e2.d_fwhm;
            }
        }
    }

    // Form H = J^T J and check H * full_cov == sigma^2 * I
    let mut h = vec![vec![0.0; p_tot]; p_tot];
    for i in 0..p_tot {
        for j in 0..p_tot {
            h[i][j] = j_cols[i]
                .iter()
                .zip(j_cols[j].iter())
                .map(|(&a, &b)| a * b)
                .sum();
        }
    }

    assert_eq!(res.full_covariance.len(), p_tot);
    for (i, h_row) in h.iter().enumerate() {
        for j in 0..p_tot {
            let prod: f64 = (0..p_tot)
                .map(|k| h_row[k] * res.full_covariance[k][j])
                .sum();
            let target = if i == j { sigma * sigma } else { 0.0 };
            assert!(
                (prod - target).abs() < 1.0e-9,
                "(H * full_cov)[{i},{j}] = {prod}, expected {target}"
            );
        }
    }
}

/// `[synthetic self-consistency]` Adversarial & hostile inputs + fail-closed diagnostics:
/// `NaN`/`Inf`, oversized ROI, too many peaks/planes, constant/zero patch, insufficient DOF,
/// nearly coincident peaks (`DegeneratePeakPair` & `IllConditionedCluster`), ill-conditioned
/// non-degenerate cluster (`IllConditionedCluster`), and `NonConverged`.
#[test]
fn adversarial_hostile_inputs_and_fail_closed_diagnostics() {
    let cfg = DeconvConfig::default();
    let seed = PeakSeed2D {
        center_f1: 4.0,
        center_f2: 4.0,
        fwhm_f1: 2.0,
        fwhm_f2: 2.0,
        eta_f1: 0.5,
        eta_f2: 0.5,
    };

    // 1. NaN voxel
    let mut vals = vec![1.0; 100];
    vals[17] = f64::NAN;
    let nan_roi = Roi2D::from_grid(10, 10, vals).expect("grid ok");
    assert!(matches!(
        deconvolve_roi_2d(&nan_roi, &[seed], &cfg),
        Err(DeconvError::NonFiniteInput(_))
    ));

    // 2. Inf coordinate
    let mut inf_roi = Roi2D::from_grid(10, 10, (0..100).map(|i| i as f64).collect()).expect("grid");
    inf_roi.coords_f1[3] = f64::INFINITY;
    assert!(matches!(
        deconvolve_roi_2d(&inf_roi, &[seed], &cfg),
        Err(DeconvError::NonFiniteInput(_))
    ));

    // 3. Oversized ROI rejected before allocation / optimization
    let oversized_err = Roi2D::from_grid(300, 300, Vec::new());
    assert!(matches!(
        oversized_err,
        Err(DeconvError::OversizedRoi {
            voxels: 90_000,
            max_voxels: MAX_DECONV_VOXELS
        })
    ));
    let oversized_3d = Roi3D::from_grid(50, 50, 50, Vec::new());
    assert!(matches!(
        oversized_3d,
        Err(DeconvError::OversizedRoi {
            voxels: 125_000,
            max_voxels: MAX_DECONV_VOXELS
        })
    ));

    // 4. Too many peaks
    let ok_roi =
        Roi2D::from_grid(10, 10, (0..100).map(|i| (i as f64).sin()).collect()).expect("ok");
    let many_seeds = vec![seed; MAX_DECONV_PEAKS_PER_CLUSTER + 1];
    assert!(matches!(
        deconvolve_roi_2d(&ok_roi, &many_seeds, &cfg),
        Err(DeconvError::TooManyPeaks { .. })
    ));

    // 5. Too many planes
    let many_planes = PlaneSeries2D {
        coords_f1: linspace(0.0, 9.0, 10),
        coords_f2: linspace(0.0, 9.0, 10),
        plane_values: vec![vec![1.0; 100]; MAX_DECONV_PLANES + 1],
        plane_coords: Vec::new(),
    };
    assert!(matches!(
        deconvolve_plane_series_2d(&many_planes, &[seed], &cfg),
        Err(DeconvError::TooManyPlanes { .. })
    ));

    // 6. Constant / zero patch
    let zero_roi = Roi2D::from_grid(10, 10, vec![0.0; 100]).expect("zero");
    assert_eq!(
        deconvolve_roi_2d(&zero_roi, &[seed], &cfg),
        Err(DeconvError::ConstantOrZeroPatch)
    );
    let const_roi = Roi2D::from_grid(10, 10, vec![42.0; 100]).expect("const");
    assert_eq!(
        deconvolve_roi_2d(&const_roi, &[seed], &cfg),
        Err(DeconvError::ConstantOrZeroPatch)
    );

    // 7. Insufficient DOF (2x2 = 4 voxels, 1 peak = 7 params for PseudoVoigt)
    let tiny_roi = Roi2D::from_grid(2, 2, vec![0.0, 1.0, 2.0, 3.0]).expect("tiny");
    assert!(matches!(
        deconvolve_roi_2d(&tiny_roi, &[seed], &cfg),
        Err(DeconvError::InsufficientDegreesOfFreedom { .. })
    ));

    // 8. Nearly coincident peaks -> DegeneratePeakPair & IllConditionedCluster
    let c1 = linspace(0.0, 14.0, 15);
    let c2 = linspace(0.0, 14.0, 15);
    let single_peak = [PeakModel2D {
        amplitude: 50.0,
        center_f1: 7.0,
        center_f2: 7.0,
        fwhm_f1: 3.0,
        fwhm_f2: 3.0,
        eta_f1: 0.5,
        eta_f2: 0.5,
    }];
    let roi_deg =
        synthesize_roi_2d(&c1, &c2, LineshapeKind::Gaussian, &single_peak, 0.0).expect("deg roi");
    let coincident_seeds = [
        PeakSeed2D {
            center_f1: 6.98,
            center_f2: 7.0,
            fwhm_f1: 3.0,
            fwhm_f2: 3.0,
            eta_f1: 0.0,
            eta_f2: 0.0,
        },
        PeakSeed2D {
            center_f1: 7.02,
            center_f2: 7.0,
            fwhm_f1: 3.0,
            fwhm_f2: 3.0,
            eta_f1: 0.0,
            eta_f2: 0.0,
        },
    ];
    let cfg_deg = DeconvConfig {
        lineshape: LineshapeKind::Gaussian,
        ..DeconvConfig::default()
    };
    let res_deg = deconvolve_roi_2d(&roi_deg, &coincident_seeds, &cfg_deg).expect("deg fit");
    assert_eq!(res_deg.diagnostic, DeconvDiagnostic::DegeneratePeakPair);
    assert!(!res_deg.degenerate_pairs.is_empty());
    assert!(res_deg.min_pair_separation_fwhm < cfg_deg.degenerate_separation_frac);
    assert!(res_deg
        .all_diagnostics
        .contains(&DeconvDiagnostic::DegeneratePeakPair));

    // 9. Ill-conditioned cluster (3 closely spaced collinear peaks separated by 0.30 * FWHM,
    //    above the 0.25 degenerate cutoff, yielding high kappa(Phi) -> triggers IllConditionedCluster)
    let close_peaks = [
        PeakModel2D {
            amplitude: 50.0,
            center_f1: 6.1,
            center_f2: 7.0,
            fwhm_f1: 3.0,
            fwhm_f2: 3.0,
            eta_f1: 0.0,
            eta_f2: 0.0,
        },
        PeakModel2D {
            amplitude: 50.0,
            center_f1: 7.0,
            center_f2: 7.0,
            fwhm_f1: 3.0,
            fwhm_f2: 3.0,
            eta_f1: 0.0,
            eta_f2: 0.0,
        },
        PeakModel2D {
            amplitude: 50.0,
            center_f1: 7.9,
            center_f2: 7.0,
            fwhm_f1: 3.0,
            fwhm_f2: 3.0,
            eta_f1: 0.0,
            eta_f2: 0.0,
        },
    ];
    let roi_ill =
        synthesize_roi_2d(&c1, &c2, LineshapeKind::Gaussian, &close_peaks, 0.0).expect("ill roi");
    let ill_seeds = [
        PeakSeed2D {
            center_f1: 6.1,
            center_f2: 7.0,
            fwhm_f1: 3.0,
            fwhm_f2: 3.0,
            eta_f1: 0.0,
            eta_f2: 0.0,
        },
        PeakSeed2D {
            center_f1: 7.0,
            center_f2: 7.0,
            fwhm_f1: 3.0,
            fwhm_f2: 3.0,
            eta_f1: 0.0,
            eta_f2: 0.0,
        },
        PeakSeed2D {
            center_f1: 7.9,
            center_f2: 7.0,
            fwhm_f1: 3.0,
            fwhm_f2: 3.0,
            eta_f1: 0.0,
            eta_f2: 0.0,
        },
    ];
    let cfg_ill = DeconvConfig {
        lineshape: LineshapeKind::Gaussian,
        ill_conditioned_kappa: 10.0,
        ..DeconvConfig::default()
    };
    let res_ill = deconvolve_roi_2d(&roi_ill, &ill_seeds, &cfg_ill).expect("ill fit");
    assert_eq!(res_ill.diagnostic, DeconvDiagnostic::IllConditionedCluster);
    assert!(res_ill.basis_condition_number > 10.0);

    // 10. NonConverged when max_iterations == 1 far from optimum
    let cfg_nc = DeconvConfig {
        lineshape: LineshapeKind::Gaussian,
        max_iterations: 1,
        ..DeconvConfig::default()
    };
    let far_seed = [PeakSeed2D {
        center_f1: 3.5,
        center_f2: 3.5,
        fwhm_f1: 1.5,
        fwhm_f2: 1.5,
        eta_f1: 0.0,
        eta_f2: 0.0,
    }];
    let res_nc = deconvolve_roi_2d(&roi_deg, &far_seed, &cfg_nc).expect("nc fit");
    assert_eq!(res_nc.diagnostic, DeconvDiagnostic::NonConverged);
    assert!(!res_nc.converged);
}

/// `[synthetic self-consistency]` Verify multi-plane (`P = 3`) coupled block-Schur Cramér–Rao
/// covariance matches empirical Monte Carlo covariance across planes within `<= 15%` and scales
/// properly when `noise_sigma` is estimated from residuals (`noise_sigma: None`).
#[test]
fn multi_plane_coupled_crlb_matches_monte_carlo_and_estimated_sigma() {
    let c1 = linspace(0.0, 16.0, 17);
    let c2 = linspace(0.0, 16.0, 17);
    let plane_amps = [(90.0, 80.0), (60.0, 35.0), (40.0, 15.0)];
    let sigma = 0.20;

    let mut clean_planes = Vec::new();
    for &(a0, a1) in &plane_amps {
        let pks = [
            PeakModel2D {
                amplitude: a0,
                center_f1: 7.2,
                center_f2: 7.2,
                fwhm_f1: 3.0,
                fwhm_f2: 3.0,
                eta_f1: 1.0,
                eta_f2: 1.0,
            },
            PeakModel2D {
                amplitude: a1,
                center_f1: 8.8,
                center_f2: 8.8,
                fwhm_f1: 3.0,
                fwhm_f2: 3.0,
                eta_f1: 1.0,
                eta_f2: 1.0,
            },
        ];
        clean_planes.push(
            synthesize_roi_2d(&c1, &c2, LineshapeKind::Lorentzian, &pks, 0.0)
                .expect("plane")
                .values,
        );
    }

    let seeds = [
        PeakSeed2D {
            center_f1: 7.2,
            center_f2: 7.2,
            fwhm_f1: 3.0,
            fwhm_f2: 3.0,
            eta_f1: 1.0,
            eta_f2: 1.0,
        },
        PeakSeed2D {
            center_f1: 8.8,
            center_f2: 8.8,
            fwhm_f1: 3.0,
            fwhm_f2: 3.0,
            eta_f1: 1.0,
            eta_f2: 1.0,
        },
    ];

    let cfg_known = DeconvConfig {
        lineshape: LineshapeKind::Lorentzian,
        linewidth_sharing: LinewidthSharing::SharedAcrossPlanes,
        noise_sigma: Some(sigma),
        ..DeconvConfig::default()
    };
    let clean_series = PlaneSeries2D {
        coords_f1: c1.clone(),
        coords_f2: c2.clone(),
        plane_values: clean_planes.clone(),
        plane_coords: vec![0.0, 0.05, 0.10],
    };
    let ref_res = deconvolve_plane_series_2d(&clean_series, &seeds, &cfg_known).expect("ref");

    let n_mc = 180usize;
    let mut rng = Rng::new(2026_2209);
    let mut samples: Vec<Vec<Vec<f64>>> = (0..3)
        .map(|_| (0..2).map(|_| Vec::with_capacity(n_mc)).collect())
        .collect();

    for _ in 0..n_mc {
        let noisy_planes: Vec<Vec<f64>> = clean_planes
            .iter()
            .map(|pl| pl.iter().map(|&v| v + rng.normal(0.0, sigma)).collect())
            .collect();
        let series = PlaneSeries2D {
            coords_f1: c1.clone(),
            coords_f2: c2.clone(),
            plane_values: noisy_planes,
            plane_coords: vec![0.0, 0.05, 0.10],
        };
        let fit = deconvolve_plane_series_2d(&series, &seeds, &cfg_known).expect("mc plane fit");
        assert!(fit.converged);
        for (p, plane_samp) in samples.iter_mut().enumerate() {
            plane_samp[0].push(fit.peaks[0].amplitudes[p]);
            plane_samp[1].push(fit.peaks[1].amplitudes[p]);
        }
    }

    for (p, plane_samp) in samples.iter().enumerate() {
        for (k, vals) in plane_samp.iter().enumerate() {
            let mean = vals.iter().sum::<f64>() / (n_mc as f64);
            let emp_se = (vals.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                / ((n_mc - 1) as f64))
                .sqrt();
            let crlb_se = ref_res.peaks[k].amplitude_se[p];
            let rel = (emp_se - crlb_se).abs() / crlb_se;
            assert!(
                rel <= 0.15,
                "plane {p} peak {k}: MC SE ({emp_se:.4}) vs CRLB SE ({crlb_se:.4}), rel={rel:.3}"
            );
        }
    }

    // Also test unknown noise_sigma (None): residual_sigma should match injected sigma within 15%
    let noisy_planes: Vec<Vec<f64>> = clean_planes
        .iter()
        .map(|pl| pl.iter().map(|&v| v + rng.normal(0.0, sigma)).collect())
        .collect();
    let cfg_est = DeconvConfig {
        lineshape: LineshapeKind::Lorentzian,
        linewidth_sharing: LinewidthSharing::SharedAcrossPlanes,
        noise_sigma: None,
        ..DeconvConfig::default()
    };
    let est_res = deconvolve_plane_series_2d(
        &PlaneSeries2D {
            coords_f1: c1,
            coords_f2: c2,
            plane_values: noisy_planes,
            plane_coords: vec![0.0, 0.05, 0.10],
        },
        &seeds,
        &cfg_est,
    )
    .expect("est sigma fit");
    assert_eq!(est_res.diagnostic, DeconvDiagnostic::Converged);
    assert!((est_res.residual_sigma - sigma).abs() / sigma < 0.15);
    assert!((est_res.reduced_chi2 - 1.0).abs() < 1.0e-12);
}
