//! Clean-room Convex Accelerated Maximum-Entropy (`MaxEnt`) 1D, 2D hypercomplex, and 3D
//! indirect-plane non-uniform sampling (NUS) reconstruction.
//!
//! Provenance: Clean-room original implementation strictly from peer-reviewed convex optimization
//! and biomolecular NMR literature (zero third-party GPL source code referenced):
//! - Skilling, J. & Bryan, R. K. (*Mon. Not. R. Astron. Soc.* **1984**, 211, 111–124) —
//!   Maximum-entropy image reconstruction and $\chi^2$ constraint formulation.
//! - Hoch, J. C. & Stern, A. S. (*NMR Data Processing*, Wiley-Liss, **1996**, Ch. 6) —
//!   Hoch–Hore spin-$1/2$ complex / hypercomplex entropy functional for phase-sensitive NMR
//!   spectra, default scale parameter `def`, and amplitude nonlinearity analysis.
//! - Beck, A. & Teboulle, M. (*SIAM J. Imaging Sci.* **2009**, 2, 183–202) —
//!   Fast Iterative Shrinkage-Thresholding Algorithm (`FISTA`) with Lipschitz backtracking.
//! - Scheinberg, K., Goldfarb, D. & Bai, X. (*Math. Program.* **2014**, 146, 333–359) —
//!   Adaptive Lipschitz step-size continuation for accelerated proximal-gradient methods.
//! - O'Donoghue, B. & Candès, E. (*Found. Comput. Math.* **2015**, 15, 715–732) —
//!   Adaptive gradient restart for accelerated first-order optimization.
//! - Worley, B. (*J. Magn. Reson.* **2016**, 265, 90–98) —
//!   Convex accelerated maximum-entropy reconstruction via primal-dual / proximal FISTA.
//! - Morozov, V. A. (*Soviet Math. Dokl.* **1966**, 7, 414–417) —
//!   Discrepancy principle for selecting the regularization parameter $\lambda$ such that
//!   $\chi^2_\nu = \|\mathcal{M}\mathcal{F}^{-1}\mathbf{f} - \mathbf{d}\|_2^2 / (N_{\text{obs}} \sigma^2) \approx 1$.
//! - States, D. J., Haberkorn, R. A. & Ruben, D. J. (*J. Magn. Reson.* **1982**, 48, 286–292);
//!   Marion, D., Ikura, M., Tschudin, R. & Bax, A. (*J. Magn. Reson.* **1989**, 85, 393–399);
//!   Kay, L. E., Keifer, P. & Saarinen, T. (*J. Am. Chem. Soc.* **1992**, 114, 10663–10665) —
//!   Hypercomplex `States`, `States-TPPI`, and `Echo-Antiecho` indirect quadrature encoding.
//! - Hyberts, S. G., Takeuchi, K. & Wagner, G. (*J. Am. Chem. Soc.* **2010**, 132, 2145–2147) —
//!   Sinusoidally weighted Poisson-gap non-uniform sampling schedules.
//!
//! ## Scientific Scope & Nonlinearity Disclosure
//! Maximum-entropy (`MaxEnt`) reconstruction is a **nonlinear, entropy-regularized complement** to
//! linear / sparsity-based NUS reconstruction (such as `nmr_core::process::reconstruct_nus_2d_hypercomplex_ist`,
//! which preserves near-linear peak heights and achieves lower linear ground-truth NRMSE on
//! synthetic benchmarks). The Hoch–Hore spin-$1/2$ entropy penalty compresses low-SNR peak
//! amplitudes more strongly than high-SNR peaks (a nonlinear response depending on `def` and
//! `lambda`), and zero-filled frequency-domain MaxEnt (`zero_fill_factor > 1`) extrapolates the
//! time-domain FID beyond the acquired `n` grid, altering linewidths and peak heights relative to a
//! truncated zero-filled Fourier transform. Quantitative amplitude analyses (relaxation, NOE,
//! titration volumes) should either use linear/IST reconstruction or apply synthetic-signal /
//! standard-curve nonlinearity calibration (Hoch & Stern 1996).

use std::f64::consts::PI;

use serde::{Deserialize, Serialize};

use crate::fid::{Complex64, QuadMode};
use crate::process::fft_in_place;
use crate::rng::Rng;

// ============================================================================
// Safety Bounds & Constants
// ============================================================================

/// Maximum frequency-domain grid points permitted in a single 1D or 2D MaxEnt solve
/// (`2^20 = 1,048,576`) to guard against hostile allocation / DoS inputs.
pub const MAX_MAXENT_POINTS: usize = 1 << 20;

/// Maximum total hypercomplex time-domain voxels permitted in a 3D MaxEnt volume
/// (`2^24 = 16,777,216`).
pub const MAX_MAXENT_3D_TOTAL_POINTS: usize = 1 << 24;

/// Maximum total 3D frequency-domain voxels (`nx * f2 * f1`) permitted in [`reconstruct_maxent_3d`]
/// (`2^24 = 16,777,216` hypercomplex voxels, bounding the reconstructed 3D output volume).
pub const MAX_MAXENT_3D_TOTAL_VOXELS: usize = 1 << 24;

/// Maximum solver workspace memory budget in bytes (`256 MiB`), enforced per solve (1D/2D) and
/// across concurrent worker threads in [`reconstruct_maxent_3d`] before any allocation.
pub const MAX_MAXENT_WORKSPACE_BYTES: usize = 256 * 1024 * 1024;

/// Hard upper bound on inner FISTA iterations per `lambda` solve (`10,000`) to prevent
/// unbounded CPU work from hostile configurations.
pub const MAX_MAXENT_ITERATIONS: usize = 10_000;

/// Hard upper bound on outer Morozov discrepancy-principle search iterations (`64`).
pub const MAX_MAXENT_OUTER_ITERATIONS: usize = 64;

/// Maximum frequency-domain magnitude permitted in overflow-guarded `sinh` dual-to-primal mapping
/// (`1e150`, leaving headroom so squared norms never overflow `f64::MAX`).
const MAX_SAFE_DUAL_PRIMAL_MAG: f64 = 1e150;

/// Maximum safe `sinh` argument cap (well below `ln(f64::MAX) ~ 709.78`).
const MAX_SAFE_SINH_ARG: f64 = 345.0;

/// Maximum Newton iterations for the strictly concave scalar MaxEnt proximal equation.
const MAX_PROX_NEWTON_ITERS: usize = 24;

/// Minimum adaptive local Lipschitz constant (`tau_max = 1 / MIN_ADAPTIVE_L = 20.0`).
const MIN_ADAPTIVE_L: f64 = 0.05;

/// Standard scientific disclosure attached to every MaxEnt reconstruction result.
pub const MAXENT_LINEARITY_DISCLOSURE: &str = "MaxEnt nonlinearity & extrapolation disclosure (Hoch & Stern 1996; Worley 2016): MaxEnt is a nonlinear, entropy-regularized complement to linear/sparsity NUS reconstruction (such as IST/FISTA, which achieves lower linear ground-truth NRMSE and near-linear peak heights). The Hoch-Hore spin-1/2 entropy penalty compresses low-SNR peaks more strongly than high-SNR peaks as a function of `def` and `lambda` (e.g. ~33% weak/strong ratio compression at 10:1 dynamic range), and zero-filled frequency-domain reconstruction (`zero_fill_factor > 1`) extrapolates the FID beyond the acquired window, altering linewidths and peak heights relative to a truncated zero-filled FT. Use reconstructed intensities qualitatively or apply synthetic-signal / standard-curve calibration before quantitative relaxation, NOE, or titration volume analysis.";

// ============================================================================
// Public Configuration & Result Types
// ============================================================================

/// Regularization parameter (`lambda`) selection policy for convex accelerated MaxEnt.
///
/// Minimizes the strictly convex primal objective:
/// $$J(\mathbf{f}) = -S(\mathbf{f}; \delta) + \frac{\lambda}{2} \|\mathcal{M}\mathcal{F}^{-1}\mathbf{f} - \mathbf{d}\|_2^2$$
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum MaxEntLambdaMode {
    /// Fixed Lagrange multiplier `lambda > 0`.
    Fixed { lambda: f64 },
    /// Morozov discrepancy principle: automatically select `lambda > 0` so the reduced
    /// chi-squared $\chi^2_\nu = \|\mathcal{M}\mathcal{F}^{-1}\mathbf{f} - \mathbf{d}\|_2^2 / (N_{\text{obs}} \sigma_{\text{noise}}^2)$
    /// matches `target_chi2_nu` within relative tolerance `chi2_tol`.
    DiscrepancyPrinciple {
        /// Per-real-channel additive noise standard deviation (`> 0`).
        /// For 1D complex FIDs, $N_{\text{obs}} = 2 M$; for 2D hypercomplex planes,
        /// $N_{\text{obs}} = 4 M$, where $M$ is the number of sampled grid points.
        noise_sigma: f64,
        /// Target reduced chi-squared (default `1.0`).
        target_chi2_nu: f64,
        /// Relative tolerance on reduced chi-squared (default `0.02`, i.e. $\pm 2\%$).
        chi2_tol: f64,
        /// Maximum outer bracketing + secant/bisection iterations (default `28`).
        max_outer_iterations: usize,
    },
}

/// Configuration for clean-room Convex Accelerated Maximum-Entropy (`MaxEnt`) NUS reconstruction.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MaxEntConfig {
    /// Hoch–Hore spin-$1/2$ entropy default scale parameter $\delta$ (`def > 0`).
    /// Sets the transition scale between near-quadratic (small $|f_k| \ll \delta$) and
    /// near-logarithmic (large $|f_k| \gg \delta$) regularization.
    pub def: f64,
    /// Regularization parameter policy (`Fixed` or `DiscrepancyPrinciple`).
    pub lambda_mode: MaxEntLambdaMode,
    /// Optional per-channel noise standard deviation used to compute `chi2_reduced` in `Fixed` mode.
    /// In `DiscrepancyPrinciple` mode, the `noise_sigma` inside `lambda_mode` takes precedence.
    pub noise_sigma: Option<f64>,
    /// Maximum FISTA iterations per `lambda` solve (default `1000`).
    pub max_iterations: usize,
    /// Infinity-norm KKT stationarity tolerance:
    /// $\|\nabla_{\mathbf{f}} (-S(\mathbf{f}; \delta) + \lambda C(\mathbf{f}))\|_\infty \le \text{kkt\_tol}$
    /// (default `1e-6`).
    pub kkt_tol: f64,
    /// Enable Beck–Teboulle FISTA momentum acceleration (default `true`).
    pub accelerate: bool,
    /// Enable O'Donoghue–Candès (2015) adaptive gradient restart and monotone descent safeguard (default `true`).
    pub adaptive_restart: bool,
    /// Enable exact Lipschitz backtracking step-size adaptation (default `true`).
    pub adaptive_step_size: bool,
    /// Frequency-domain zero-fill factor along each reconstructed axis (`>= 1`, default `2`).
    pub zero_fill_factor: usize,
    /// Record per-iteration primal objective values in `result.objective_trace` (default `false`).
    pub record_objective_trace: bool,
}

impl Default for MaxEntConfig {
    fn default() -> Self {
        Self {
            def: 1.0,
            lambda_mode: MaxEntLambdaMode::Fixed { lambda: 1.0 },
            noise_sigma: None,
            max_iterations: 1000,
            kkt_tol: 1e-6,
            accelerate: true,
            adaptive_restart: true,
            adaptive_step_size: true,
            zero_fill_factor: 2,
            record_objective_trace: false,
        }
    }
}

impl MaxEntConfig {
    /// Construct a fixed-$\lambda$ MaxEnt configuration with sensible defaults.
    pub fn fixed(def: f64, lambda: f64) -> Self {
        Self {
            def,
            lambda_mode: MaxEntLambdaMode::Fixed { lambda },
            ..Self::default()
        }
    }

    /// Construct a discrepancy-principle MaxEnt configuration targeting $\chi^2_\nu \approx 1.0$.
    pub fn discrepancy(def: f64, noise_sigma: f64) -> Self {
        Self {
            def,
            lambda_mode: MaxEntLambdaMode::DiscrepancyPrinciple {
                noise_sigma,
                target_chi2_nu: 1.0,
                chi2_tol: 0.02,
                max_outer_iterations: 28,
            },
            noise_sigma: Some(noise_sigma),
            ..Self::default()
        }
    }

    /// Validate all numerical parameters and hard iteration caps (fail-closed).
    pub fn validate(&self) -> Result<(), String> {
        validate_config(self)
    }
}

/// Rich reconstruction output and convex optimization diagnostics for 1D (`T = Complex64`) and
/// 2D hypercomplex (`T = [f64; 4]`) MaxEnt NUS reconstruction.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MaxEntResult<T> {
    /// Reconstructed frequency-domain spectrum $\mathbf{f}^*$ (exact primal minimizer of
    /// $-S(\mathbf{f}; \delta) + \lambda C(\mathbf{f})$).
    /// For 2D hypercomplex solves, each element is `[RR, RI, IR, II]` in canonical `States` order.
    pub spectrum: Vec<T>,
    /// Reconstructed time-domain mock FID $\mathcal{F}^{-1}\mathbf{f}^*$ cropped to `fid_shape`.
    /// For 2D hypercomplex solves with non-`States` input `quad_modes`, transformed back to match
    /// the input quadrature encoding.
    pub fid: Vec<T>,
    /// Frequency-domain grid shape (`[n_fft]` for 1D, `[f2, f1]` for 2D).
    pub spectrum_shape: Vec<usize>,
    /// Time-domain grid shape (`[n1]` for 1D, `[n2, n1]` for 2D).
    pub fid_shape: Vec<usize>,
    /// Inner FISTA iterations executed in the final converged `lambda` solve.
    pub iterations: usize,
    /// Total inner FISTA iterations accumulated across all outer discrepancy-principle solves.
    pub total_iterations: usize,
    /// Number of outer discrepancy-principle evaluations (`0` in `Fixed` mode).
    pub outer_iterations: usize,
    /// Number of O'Donoghue–Candès adaptive gradient restarts triggered in the final solve.
    pub restarts: usize,
    /// Whether `kkt_residual_inf <= cfg.kkt_tol` was achieved.
    pub converged: bool,
    /// Exact infinity norm of the primal KKT stationarity gradient:
    /// $\|\nabla_{\mathbf{f}} (-S(\mathbf{f}; \delta) + \lambda C(\mathbf{f}))\|_\infty$.
    pub kkt_residual_inf: f64,
    /// Final primal objective $J(\mathbf{f}) = -S(\mathbf{f}; \delta) + \frac{\lambda}{2}\|\mathcal{M}\mathcal{F}^{-1}\mathbf{f} - \mathbf{d}\|_2^2$.
    pub primal_objective: f64,
    /// Non-negative primal-dual optimality gap $J(\mathbf{f}) + D(\mathbf{w})$ at
    /// $\mathbf{w} = \lambda (\mathbf{d} - \mathcal{M}\mathcal{F}^{-1}\mathbf{f})$.
    pub duality_gap: f64,
    /// Observed time-domain squared residual $\|\mathcal{M}\mathcal{F}^{-1}\mathbf{f} - \mathbf{d}\|_2^2$.
    pub data_residual_sq: f64,
    /// Reduced chi-squared $\chi^2_\nu = \|\mathcal{M}\mathcal{F}^{-1}\mathbf{f} - \mathbf{d}\|_2^2 / (N_{\text{obs}} \sigma^2)$
    /// when a noise standard deviation is available (`None` otherwise).
    pub chi2_reduced: Option<f64>,
    /// Hoch–Hore spin-$1/2$ entropy $S(\mathbf{f}; \delta)$.
    pub entropy: f64,
    /// Final Lagrange multiplier $\lambda > 0$.
    pub final_lambda: f64,
    /// Number of real scalar observations $N_{\text{obs}}$ (`2 * M` for 1D, `4 * M` for 2D).
    pub n_obs_scalars: usize,
    /// Optional per-iteration primal objective trajectory (populated when `cfg.record_objective_trace` is `true`).
    pub objective_trace: Vec<f64>,
    /// Disclosure of MaxEnt's inherent amplitude nonlinearity and low-SNR intensity compression.
    pub linearity_note: String,
}

/// Per-plane diagnostic summary returned inside [`MaxEntResult3D`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MaxEntPlaneDiagnostics {
    pub direct_index: usize,
    pub iterations: usize,
    pub total_iterations: usize,
    pub outer_iterations: usize,
    pub restarts: usize,
    pub converged: bool,
    pub kkt_residual_inf: f64,
    pub primal_objective: f64,
    pub duality_gap: f64,
    pub data_residual_sq: f64,
    pub chi2_reduced: Option<f64>,
    pub entropy: f64,
    pub final_lambda: f64,
}

/// Result of 3D indirect-plane hypercomplex MaxEnt NUS reconstruction across `nx` direct slices.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MaxEntResult3D {
    /// Reconstructed 3D indirect-plane frequency spectra in `(ix, m2, m1)` row-major order
    /// (`len == nx * f2 * f1`), each voxel `[RR, RI, IR, II]` in canonical `States` order.
    pub spectra_planes: Vec<[f64; 4]>,
    /// Reconstructed 3D indirect-plane time-domain mock FIDs in `(ix, k2, k1)` row-major order
    /// (`len == nx * n2 * n1`), in the caller's input `quad_modes` encoding.
    pub fid_planes: Vec<[f64; 4]>,
    /// Number of direct-dimension frequency slices (`nx`).
    pub nx: usize,
    /// Indirect axis-2 time-domain length (`n2`).
    pub n2: usize,
    /// Indirect axis-1 time-domain length (`n1`).
    pub n1: usize,
    /// Indirect axis-2 frequency-domain length (`f2`).
    pub f2: usize,
    /// Indirect axis-1 frequency-domain length (`f1`).
    pub f1: usize,
    /// Per-plane optimization diagnostics (`len == nx`).
    pub plane_diagnostics: Vec<MaxEntPlaneDiagnostics>,
    /// Sum of final-solve inner iterations across all `nx` planes.
    pub iterations: usize,
    /// True iff every indirect plane satisfied `kkt_residual_inf <= cfg.kkt_tol`.
    pub converged: bool,
    /// Worst-case KKT infinity residual across all `nx` planes.
    pub max_kkt_residual_inf: f64,
    /// Mean primal objective across all `nx` planes.
    pub primal_objective: f64,
    /// Mean reduced chi-squared across all `nx` planes (`None` if no `noise_sigma`).
    pub chi2_reduced: Option<f64>,
    /// Mean Hoch–Hore entropy across all `nx` planes.
    pub entropy: f64,
    /// Mean final Lagrange multiplier `lambda` across all `nx` planes.
    pub final_lambda: f64,
    /// Disclosure of MaxEnt's inherent amplitude nonlinearity and low-SNR intensity compression.
    pub linearity_note: String,
}

// ============================================================================
// Hoch-Hore Entropy, Convex Conjugate & Proximal Building Blocks
// ============================================================================

/// Evaluate the negative Hoch–Hore spin-$1/2$ entropy scalar kernel:
/// $$\psi(r; \delta) = \frac{r}{\delta} \operatorname{arsinh}\!\left(\frac{r}{2\delta}\right) - \sqrt{4 + \frac{r^2}{\delta^2}}$$
/// using `hypot(2.0, r / delta)` so large amplitudes never overflow intermediate squaring.
#[inline]
fn neg_entropy_scalar(r: f64, def: f64) -> f64 {
    if r <= 0.0 {
        return -2.0;
    }
    let u = r / def;
    if !u.is_finite() {
        return f64::MAX * 0.25;
    }
    let asinh_term = if u > 1e150 {
        r.ln() - def.ln()
    } else {
        (0.5 * u).asinh()
    };
    let val = u * asinh_term - 2.0_f64.hypot(u);
    if val.is_finite() {
        val
    } else {
        f64::MAX * 0.25
    }
}

/// Evaluate the radial derivative factor $\frac{\psi'(r; \delta)}{r} = \frac{\operatorname{arsinh}(r / (2\delta))}{\delta r}$
/// smoothly at $r = 0$ (limit $\frac{1}{2\delta^2}$) without `0/0`.
#[inline]
fn neg_entropy_grad_scale(r: f64, def: f64) -> f64 {
    let u = r / def;
    if u <= 1e-7 {
        // Taylor expansion: asinh(u/2) / u = 1/2 - u^2 / 48 + 3 u^4 / 1280
        let inv_2d2 = 0.5 / (def * def);
        return inv_2d2 * (1.0 - (u * u) / 24.0);
    }
    let asinh_term = if u > 1e150 || !u.is_finite() {
        r.ln() - def.ln()
    } else {
        (0.5 * u).asinh()
    };
    asinh_term / (def * r)
}

/// Evaluate the convex conjugate $h^*(z; \delta) = 2 \cosh(\delta |z|)$ with overflow guarding.
#[inline]
fn dual_conj_scalar(z_mag: f64, def: f64) -> f64 {
    let arg = (def * z_mag).min(MAX_SAFE_SINH_ARG);
    2.0 * arg.cosh()
}

/// Compute the Hoch–Hore spin-$1/2$ entropy $S(\mathbf{f}; \delta)$ of a 1D complex spectrum.
pub fn hoch_hore_entropy_1d(spectrum: &[Complex64], def: f64) -> Result<f64, String> {
    validate_positive_finite(def, "def")?;
    let mut neg_s = 0.0_f64;
    for (idx, c) in spectrum.iter().enumerate() {
        if !c[0].is_finite() || !c[1].is_finite() {
            return Err(format!("non-finite spectrum value at index {idx}"));
        }
        let r = c[0].hypot(c[1]);
        neg_s += neg_entropy_scalar(r, def);
    }
    Ok(-neg_s)
}

/// Compute the Hoch–Hore spin-$1/2$ entropy $S(\mathbf{f}; \delta)$ of a 2D hypercomplex spectrum
/// where each voxel is `[RR, RI, IR, II]`.
pub fn hoch_hore_entropy_2d_hypercomplex(spectrum: &[[f64; 4]], def: f64) -> Result<f64, String> {
    validate_positive_finite(def, "def")?;
    let mut neg_s = 0.0_f64;
    for (idx, q) in spectrum.iter().enumerate() {
        if q.iter().any(|v| !v.is_finite()) {
            return Err(format!(
                "non-finite hypercomplex spectrum value at index {idx}"
            ));
        }
        let r = norm4(*q);
        neg_s += neg_entropy_scalar(r, def);
    }
    Ok(-neg_s)
}

/// Overflow-safe closed-form convex-dual mapping for a single complex frequency bin:
/// $$f_k(y_k) = 2\delta \sinh\!\left(\frac{|y_k|}{\sigma_{\text{def}}}\right) \frac{y_k}{|y_k|}$$
///
/// Smoothly evaluates the exact small-argument limit $f_k \approx \frac{2\delta}{\sigma_{\text{def}}} y_k$
/// near $y_k = 0$ and clamps the hyperbolic argument so extreme dual iterates never overflow to `Inf`/`NaN`.
pub fn dual_to_primal_complex(y: Complex64, def: f64, sigma_def: f64) -> Result<Complex64, String> {
    validate_positive_finite(def, "def")?;
    validate_positive_finite(sigma_def, "sigma_def")?;
    if !y[0].is_finite() || !y[1].is_finite() {
        return Err("non-finite dual value in dual_to_primal_complex".to_string());
    }
    let y_mag = y[0].hypot(y[1]);
    let scale = safe_dual_primal_scale(y_mag, def, sigma_def);
    Ok([y[0] * scale, y[1] * scale])
}

/// Overflow-safe closed-form convex-dual mapping over a full 1D complex dual spectrum.
pub fn dual_to_primal_spectrum_1d(
    dual_spec: &[Complex64],
    def: f64,
    sigma_def: f64,
) -> Result<Vec<Complex64>, String> {
    validate_positive_finite(def, "def")?;
    validate_positive_finite(sigma_def, "sigma_def")?;
    let mut out = Vec::with_capacity(dual_spec.len());
    for (idx, &y) in dual_spec.iter().enumerate() {
        if !y[0].is_finite() || !y[1].is_finite() {
            return Err(format!("non-finite dual spectrum entry at index {idx}"));
        }
        let y_mag = y[0].hypot(y[1]);
        let scale = safe_dual_primal_scale(y_mag, def, sigma_def);
        out.push([y[0] * scale, y[1] * scale]);
    }
    Ok(out)
}

/// Overflow-safe closed-form convex-dual mapping for a single 2D hypercomplex frequency voxel:
/// $$f_k(y_k) = 2\delta \sinh\!\left(\frac{\|y_k\|_2}{\sigma_{\text{def}}}\right) \frac{y_k}{\|y_k\|_2}$$
pub fn dual_to_primal_hypercomplex(
    y: [f64; 4],
    def: f64,
    sigma_def: f64,
) -> Result<[f64; 4], String> {
    validate_positive_finite(def, "def")?;
    validate_positive_finite(sigma_def, "sigma_def")?;
    if y.iter().any(|v| !v.is_finite()) {
        return Err("non-finite dual hypercomplex entry".to_string());
    }
    let y_mag = norm4(y);
    let scale = safe_dual_primal_scale(y_mag, def, sigma_def);
    Ok([y[0] * scale, y[1] * scale, y[2] * scale, y[3] * scale])
}

#[inline]
fn safe_dual_primal_scale(y_mag: f64, def: f64, sigma_def: f64) -> f64 {
    if y_mag <= 0.0 {
        return 0.0;
    }
    let x = y_mag / sigma_def;
    if x <= 1e-7 {
        let base = (2.0 * def) / sigma_def;
        let s = base * (1.0 + (x * x) / 6.0);
        if s.is_finite() {
            return s.min(MAX_SAFE_DUAL_PRIMAL_MAG / y_mag);
        }
        return MAX_SAFE_DUAL_PRIMAL_MAG / y_mag;
    }
    let max_arg_for_def = (MAX_SAFE_DUAL_PRIMAL_MAG / (2.0 * def))
        .max(0.0)
        .asinh()
        .min(MAX_SAFE_SINH_ARG);
    let x_clamped = x.min(max_arg_for_def);
    let f_mag = (2.0 * def * x_clamped.sinh()).min(MAX_SAFE_DUAL_PRIMAL_MAG);
    let s = f_mag / y_mag;
    if s.is_finite() {
        s
    } else {
        0.0
    }
}

/// Exact radial shrinkage factor $s = r^* / m \in [0, 1]$ for the Hoch–Hore proximal operator:
/// $$r^* = \arg\min_{r \ge 0} \left\{ \alpha \left(\frac{r}{\delta}\operatorname{arsinh}\!\left(\frac{r}{2\delta}\right) - \sqrt{4 + \frac{r^2}{\delta^2}}\right) + \frac{1}{2}(r - m)^2 \right\}$$
///
/// Solves the dimensionless strictly increasing concave equation
/// $F(u) = u + c \operatorname{arsinh}(u/2) - \tilde{m} = 0$ (where $u = r/\delta$, $\tilde{m} = m/\delta$,
/// $c = \alpha/\delta^2$) from a guaranteed lower bound $u_0 \le u^*$ via monotone quadratic Newton steps.
#[inline]
fn maxent_prox_shrinkage(m: f64, alpha: f64, def: f64) -> f64 {
    if m <= 0.0 {
        return 0.0;
    }
    let c = alpha / (def * def);
    if !c.is_finite() {
        return 0.0;
    }
    let m_tilde = m / def;
    if !m_tilde.is_finite() {
        return 1.0;
    }
    if m_tilde <= 1e-12 {
        return 1.0 / (1.0 + 0.5 * c);
    }
    // Lower bound 1: linear secant from u = 0
    let u_lin = m_tilde / (1.0 + 0.5 * c);
    // Lower bound 2: tangent from u = m_tilde (valid because F is strictly concave)
    let asinh_m = if m_tilde > 1e150 {
        m.ln() - def.ln()
    } else {
        (0.5 * m_tilde).asinh()
    };
    let f_m = c * asinh_m;
    let fp_m = 1.0 + c / 2.0_f64.hypot(m_tilde);
    let u_tan = if f_m.is_finite() && fp_m.is_finite() {
        (m_tilde - f_m / fp_m).max(0.0)
    } else {
        0.0
    };
    let mut u = u_lin.max(u_tan).min(m_tilde);

    for _ in 0..MAX_PROX_NEWTON_ITERS {
        let asinh_u = if u > 1e150 {
            (u * def).ln() - def.ln()
        } else {
            (0.5 * u).asinh()
        };
        let f_val = u + c * asinh_u - m_tilde;
        let fp_val = 1.0 + c / 2.0_f64.hypot(u);
        if !f_val.is_finite() || !fp_val.is_finite() || fp_val <= 0.0 {
            break;
        }
        let step = f_val / fp_val;
        let u_next = (u - step).clamp(0.0, m_tilde);
        if (u_next - u).abs() <= 1e-15 * (1.0 + u_next) {
            u = u_next;
            break;
        }
        u = u_next;
    }
    (u / m_tilde).clamp(0.0, 1.0)
}

#[inline]
fn norm4(q: [f64; 4]) -> f64 {
    let a = q[0].hypot(q[1]);
    let b = q[2].hypot(q[3]);
    a.hypot(b)
}

// ============================================================================
// Validation Helpers
// ============================================================================

fn validate_positive_finite(val: f64, name: &str) -> Result<(), String> {
    if !val.is_finite() || val <= 0.0 {
        return Err(format!(
            "{name} must be finite and strictly positive, got {val}"
        ));
    }
    Ok(())
}

/// Validate that `val` is positive, finite, and its square neither underflows to `0.0` nor overflows.
fn validate_positive_squarable(val: f64, name: &str) -> Result<(), String> {
    validate_positive_finite(val, name)?;
    let sq = val * val;
    if !sq.is_finite() || sq <= 0.0 {
        return Err(format!(
            "{name} ({val:e}) out of numerically safe squarable range"
        ));
    }
    Ok(())
}

fn validate_config(cfg: &MaxEntConfig) -> Result<(), String> {
    validate_positive_squarable(cfg.def, "cfg.def")?;
    if cfg.max_iterations == 0 || cfg.max_iterations > MAX_MAXENT_ITERATIONS {
        return Err(format!(
            "cfg.max_iterations ({}) must be in 1..={MAX_MAXENT_ITERATIONS}",
            cfg.max_iterations
        ));
    }
    validate_positive_finite(cfg.kkt_tol, "cfg.kkt_tol")?;
    if cfg.zero_fill_factor == 0 {
        return Err("cfg.zero_fill_factor must be >= 1".to_string());
    }
    if let Some(sigma) = cfg.noise_sigma {
        validate_positive_squarable(sigma, "cfg.noise_sigma")?;
    }
    match cfg.lambda_mode {
        MaxEntLambdaMode::Fixed { lambda } => {
            validate_positive_squarable(lambda, "cfg.lambda_mode.lambda")?;
        }
        MaxEntLambdaMode::DiscrepancyPrinciple {
            noise_sigma,
            target_chi2_nu,
            chi2_tol,
            max_outer_iterations,
        } => {
            validate_positive_squarable(noise_sigma, "cfg.lambda_mode.noise_sigma")?;
            validate_positive_finite(target_chi2_nu, "cfg.lambda_mode.target_chi2_nu")?;
            validate_positive_finite(chi2_tol, "cfg.lambda_mode.chi2_tol")?;
            if max_outer_iterations == 0 || max_outer_iterations > MAX_MAXENT_OUTER_ITERATIONS {
                return Err(format!(
                    "cfg.lambda_mode.max_outer_iterations ({max_outer_iterations}) must be in 1..={MAX_MAXENT_OUTER_ITERATIONS}"
                ));
            }
        }
    }
    Ok(())
}

/// Compute exact peak workspace + output allocation bytes for a 1D MaxEnt solve using
/// overflow-checked arithmetic.
pub fn maxent_1d_workspace_bytes(n1: usize, n_fft: usize) -> Result<usize, String> {
    let c_sz = std::mem::size_of::<Complex64>();
    let b_sz = std::mem::size_of::<bool>();
    // Solve1DWorkspace: 7 Vec<Complex64> of len n_fft + 4 Vec<Complex64> of len n1 + 1 Vec<bool> of len n1
    // Plus exported fid Vec<Complex64> of len n1 (spectrum moves ws.f_cur).
    let fft_bytes = n_fft
        .checked_mul(7)
        .and_then(|v| v.checked_mul(c_sz))
        .ok_or_else(|| "1D MaxEnt workspace byte size overflow".to_string())?;
    let td_bytes = n1
        .checked_mul(5)
        .and_then(|v| v.checked_mul(c_sz))
        .ok_or_else(|| "1D MaxEnt time-domain workspace byte size overflow".to_string())?;
    let mask_bytes = n1
        .checked_mul(b_sz)
        .ok_or_else(|| "1D MaxEnt mask byte size overflow".to_string())?;
    fft_bytes
        .checked_add(td_bytes)
        .and_then(|v| v.checked_add(mask_bytes))
        .ok_or_else(|| "1D MaxEnt total workspace byte size overflow".to_string())
}

/// Compute exact peak per-thread workspace + per-plane export bytes for a 2D hypercomplex
/// MaxEnt solve using overflow-checked arithmetic.
pub fn maxent_2d_workspace_bytes(
    n2: usize,
    n1: usize,
    f2: usize,
    f1: usize,
) -> Result<usize, String> {
    let c_sz = std::mem::size_of::<Complex64>();
    let q_sz = std::mem::size_of::<[f64; 4]>();
    let b_sz = std::mem::size_of::<bool>();
    let total = n2
        .checked_mul(n1)
        .ok_or_else(|| "2D MaxEnt time-domain grid size overflow".to_string())?;
    let n_vox = f2
        .checked_mul(f1)
        .ok_or_else(|| "2D MaxEnt FFT grid size overflow".to_string())?;
    // Solve2DWorkspace:
    // - 14 Vec<Complex64> of len n_vox (work_p/m, fp/fm cur/next, wp/wm, gcp/gcm cur/next/w)
    // - 8 Vec<Complex64> of len total (dp/dm, afp/afm cur/next, awp/awm)
    // - 1 Vec<bool> of len total (mask)
    // - 1 Vec<Complex64> of len f2 (col_buf)
    // Plus per-plane export buffers: 1 Vec<[f64; 4]> of len n_vox + 1 Vec<[f64; 4]> of len total.
    let fft_ws = n_vox
        .checked_mul(14)
        .and_then(|v| v.checked_mul(c_sz))
        .ok_or_else(|| "2D MaxEnt FFT workspace byte size overflow".to_string())?;
    let td_ws = total
        .checked_mul(8)
        .and_then(|v| v.checked_mul(c_sz))
        .ok_or_else(|| "2D MaxEnt time-domain workspace byte size overflow".to_string())?;
    let mask_ws = total
        .checked_mul(b_sz)
        .ok_or_else(|| "2D MaxEnt mask byte size overflow".to_string())?;
    let col_ws = f2
        .checked_mul(c_sz)
        .ok_or_else(|| "2D MaxEnt column scratch byte size overflow".to_string())?;
    let spec_out = n_vox
        .checked_mul(q_sz)
        .ok_or_else(|| "2D MaxEnt spectrum export byte size overflow".to_string())?;
    let fid_out = total
        .checked_mul(q_sz)
        .ok_or_else(|| "2D MaxEnt FID export byte size overflow".to_string())?;
    fft_ws
        .checked_add(td_ws)
        .and_then(|v| v.checked_add(mask_ws))
        .and_then(|v| v.checked_add(col_ws))
        .and_then(|v| v.checked_add(spec_out))
        .and_then(|v| v.checked_add(fid_out))
        .ok_or_else(|| "2D MaxEnt total workspace byte size overflow".to_string())
}

fn validate_hypercomplex_quad_mode(mode: QuadMode, axis_desc: &str) -> Result<(), String> {
    match mode {
        QuadMode::States | QuadMode::StatesTppi | QuadMode::EchoAntiecho => Ok(()),
        other => Err(format!(
            "unsupported hypercomplex QuadMode {other:?} on {axis_desc}; expected States, StatesTppi, or EchoAntiecho"
        )),
    }
}

fn effective_noise_sigma(cfg: &MaxEntConfig) -> Option<f64> {
    match cfg.lambda_mode {
        MaxEntLambdaMode::DiscrepancyPrinciple { noise_sigma, .. } => Some(noise_sigma),
        MaxEntLambdaMode::Fixed { .. } => cfg.noise_sigma,
    }
}

/// Shared Morozov discrepancy-principle outer search over strictly monotone decreasing
/// $\chi^2_\nu(\lambda)$ using exponential bracketing followed by safeguarded log-secant
/// / bisection refinement. Returns `(chosen_lambda, outer_evals, total_inner_iters)`.
#[allow(clippy::too_many_arguments)]
fn select_discrepancy_lambda<F>(
    def: f64,
    noise_sigma: f64,
    target_chi2_nu: f64,
    chi2_tol: f64,
    max_outer_iterations: usize,
    data_energy: f64,
    denom: f64,
    mut eval_chi2: F,
) -> (f64, usize, usize)
where
    F: FnMut(f64) -> (f64, usize),
{
    let target_res_sq = target_chi2_nu * denom;
    if data_energy <= target_res_sq * (1.0 + chi2_tol) {
        let tiny_lam = (1e-6 / (def * noise_sigma)).max(1e-12);
        return (tiny_lam, 0, 0);
    }

    let mut outer_evals = 0usize;
    let mut total_inner = 0usize;

    let lam0 = (1.0 / (def * noise_sigma)).clamp(1e-12, 1e12);
    let (c0, it0) = eval_chi2(lam0);
    outer_evals += 1;
    total_inner += it0;

    let mut best_lam = lam0;
    let mut best_err = (c0 - target_chi2_nu).abs() / target_chi2_nu;
    if best_err <= chi2_tol {
        return (best_lam, outer_evals, total_inner);
    }

    let mut lam_lo = lam0;
    let mut c_lo = c0;
    let mut lam_hi = lam0;
    let mut c_hi = c0;

    if c0 > target_chi2_nu {
        while outer_evals < max_outer_iterations && c_hi > target_chi2_nu && lam_hi < 1e15 {
            lam_lo = lam_hi;
            c_lo = c_hi;
            lam_hi = (lam_hi * 4.0).min(1e15);
            let (c_val, iters) = eval_chi2(lam_hi);
            outer_evals += 1;
            total_inner += iters;
            c_hi = c_val;
            let err = (c_hi - target_chi2_nu).abs() / target_chi2_nu;
            if err < best_err {
                best_err = err;
                best_lam = lam_hi;
            }
            if err <= chi2_tol {
                break;
            }
        }
    } else {
        while outer_evals < max_outer_iterations && c_lo < target_chi2_nu && lam_lo > 1e-15 {
            lam_hi = lam_lo;
            c_hi = c_lo;
            lam_lo = (lam_lo * 0.25).max(1e-15);
            let (c_val, iters) = eval_chi2(lam_lo);
            outer_evals += 1;
            total_inner += iters;
            c_lo = c_val;
            let err = (c_lo - target_chi2_nu).abs() / target_chi2_nu;
            if err < best_err {
                best_err = err;
                best_lam = lam_lo;
            }
            if err <= chi2_tol {
                break;
            }
        }
    }

    while outer_evals < max_outer_iterations
        && best_err > chi2_tol
        && c_lo > target_chi2_nu
        && c_hi < target_chi2_nu
        && (lam_hi / lam_lo).ln() > 1e-6
    {
        let ln_lo = c_lo.max(1e-30).ln();
        let ln_hi = c_hi.max(1e-30).ln();
        let ln_t = target_chi2_nu.ln();
        let w = if (ln_lo - ln_hi).abs() > 1e-12 {
            ((ln_lo - ln_t) / (ln_lo - ln_hi)).clamp(0.15, 0.85)
        } else {
            0.5
        };
        let lam_mid = ((1.0 - w) * lam_lo.ln() + w * lam_hi.ln()).exp();
        let (c_mid, iters) = eval_chi2(lam_mid);
        outer_evals += 1;
        total_inner += iters;
        let err = (c_mid - target_chi2_nu).abs() / target_chi2_nu;
        if err < best_err {
            best_err = err;
            best_lam = lam_mid;
        }
        if c_mid > target_chi2_nu {
            lam_lo = lam_mid;
            c_lo = c_mid;
        } else {
            lam_hi = lam_mid;
            c_hi = c_mid;
        }
    }

    (best_lam, outer_evals, total_inner)
}

// ============================================================================
// Unitary FFT & Hypercomplex Quadrature Operators (Reusing `process::fft_in_place`)
// ============================================================================

#[inline]
fn fft_1d_unitary(buf: &mut [Complex64], inverse: bool) {
    let n = buf.len();
    if n <= 1 {
        return;
    }
    fft_in_place(buf, inverse);
    let scale = if inverse {
        (n as f64).sqrt()
    } else {
        1.0 / (n as f64).sqrt()
    };
    for c in buf.iter_mut() {
        c[0] *= scale;
        c[1] *= scale;
    }
}

fn fft_2d_unitary(
    grid: &mut [Complex64],
    f2: usize,
    f1: usize,
    col_buf: &mut [Complex64],
    inverse: bool,
) {
    for r in 0..f2 {
        fft_in_place(&mut grid[r * f1..(r + 1) * f1], inverse);
    }
    for c in 0..f1 {
        for r in 0..f2 {
            col_buf[r] = grid[r * f1 + c];
        }
        fft_in_place(col_buf, inverse);
        for r in 0..f2 {
            grid[r * f1 + c] = col_buf[r];
        }
    }
    let n_vox = (f2 * f1) as f64;
    let scale = if inverse {
        n_vox.sqrt()
    } else {
        1.0 / n_vox.sqrt()
    };
    for v in grid.iter_mut() {
        v[0] *= scale;
        v[1] *= scale;
    }
}

/// Isometric quadrature conversion of a single 2D hypercomplex sample `[q00, q01, q10, q11]` at
/// grid index `(k2, k1)` from `quad_modes = [mode2, mode1]` into canonical `States` `[cc, cs, sc, ss]`.
///
/// ## Echo-Antiecho Phase Convention (Aligned with `fid.rs::recombine_quad_pair`)
/// For a Rance–Kay sensitivity-enhanced Echo-Antiecho pair `(z0, z1)` where `z0 = (re0, im0)`
/// and `z1 = (re1, im1)`, this applies the exact unitary (`1 / sqrt(2)`-scaled) Bruker TopSpin
/// `MC2 = 5` recombination matching `crate::fid::recombine_quad_pair`:
/// - `out_cos = (1 / sqrt(2)) * [-(im0 + im1), re0 + re1]` (`-i` rotation in Bruker conjugate rep / `+i` in `(re, im)`)
/// - `out_sin = (1 / sqrt(2)) * [re1 - re0, im1 - im0]`
///
/// This removes the 90° sensitivity-enhancement phase offset on real Bruker Echo-Antiecho data
/// while remaining strictly orthogonal (`M^T M = I`), preserving Euclidean noise variance and
/// Hoch–Hore entropy invariance.
#[inline]
fn quad_to_states(mut q: [f64; 4], k2: usize, k1: usize, quad_modes: [QuadMode; 2]) -> [f64; 4] {
    let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
    // Axis 2 (outer)
    match quad_modes[0] {
        QuadMode::States => {}
        QuadMode::StatesTppi => {
            if k2 % 2 == 1 {
                q = [-q[0], -q[1], -q[2], -q[3]];
            }
        }
        QuadMode::EchoAntiecho => {
            // z0 = [q00, q01], z1 = [q10, q11] -> out_cos = [cc, cs], out_sin = [sc, ss]
            // Unitary Bruker MC2 = 5 / fid.rs::recombine_quad_pair with -i rotation
            q = [
                inv_sqrt2 * -(q[1] + q[3]),
                inv_sqrt2 * (q[0] + q[2]),
                inv_sqrt2 * (q[2] - q[0]),
                inv_sqrt2 * (q[3] - q[1]),
            ];
        }
        QuadMode::Qf | QuadMode::Dqd => {}
    }
    // Axis 1 (inner)
    match quad_modes[1] {
        QuadMode::States => {}
        QuadMode::StatesTppi => {
            if k1 % 2 == 1 {
                q = [-q[0], -q[1], -q[2], -q[3]];
            }
        }
        QuadMode::EchoAntiecho => {
            // z0 = [q00, q10], z1 = [q01, q11] -> out_cos = [cc, sc], out_sin = [cs, ss]
            // Unitary Bruker MC2 = 5 / fid.rs::recombine_quad_pair with -i rotation
            q = [
                inv_sqrt2 * -(q[2] + q[3]),
                inv_sqrt2 * (q[1] - q[0]),
                inv_sqrt2 * (q[0] + q[1]),
                inv_sqrt2 * (q[3] - q[2]),
            ];
        }
        QuadMode::Qf | QuadMode::Dqd => {}
    }
    q
}

/// Exact orthogonal inverse (`M^T = M^{-1}`) of [`quad_to_states`]: converts canonical `States`
/// `[cc, cs, sc, ss]` back to `quad_modes = [mode2, mode1]`.
#[inline]
fn states_to_quad(mut q: [f64; 4], k2: usize, k1: usize, quad_modes: [QuadMode; 2]) -> [f64; 4] {
    let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
    // Invert Axis 1 first (reverse order of forward transform)
    match quad_modes[1] {
        QuadMode::States => {}
        QuadMode::StatesTppi => {
            if k1 % 2 == 1 {
                q = [-q[0], -q[1], -q[2], -q[3]];
            }
        }
        QuadMode::EchoAntiecho => {
            let [cc, cs, sc, ss] = q;
            q = [
                inv_sqrt2 * (sc - cs),
                inv_sqrt2 * (sc + cs),
                inv_sqrt2 * (-cc - ss),
                inv_sqrt2 * (-cc + ss),
            ];
        }
        QuadMode::Qf | QuadMode::Dqd => {}
    }
    // Invert Axis 2
    match quad_modes[0] {
        QuadMode::States => {}
        QuadMode::StatesTppi => {
            if k2 % 2 == 1 {
                q = [-q[0], -q[1], -q[2], -q[3]];
            }
        }
        QuadMode::EchoAntiecho => {
            let [cc, cs, sc, ss] = q;
            q = [
                inv_sqrt2 * (cs - sc),
                inv_sqrt2 * (-cc - ss),
                inv_sqrt2 * (cs + sc),
                inv_sqrt2 * (-cc + ss),
            ];
        }
        QuadMode::Qf | QuadMode::Dqd => {}
    }
    q
}

/// Unitary conversion from canonical `States` `[cc, cs, sc, ss]` to bicomplex `(u_+, u_-)`.
#[inline]
fn states_to_bicomplex(q: [f64; 4]) -> (Complex64, Complex64) {
    let s = std::f64::consts::FRAC_1_SQRT_2;
    let [cc, cs, sc, ss] = q;
    (
        [s * (cc - ss), s * (sc + cs)],
        [s * (cc + ss), s * (sc - cs)],
    )
}

/// Exact inverse of [`states_to_bicomplex`]: bicomplex `(u_+, u_-)` to canonical `States` `[cc, cs, sc, ss]`.
#[inline]
fn bicomplex_to_states(up: Complex64, um: Complex64) -> [f64; 4] {
    let s = std::f64::consts::FRAC_1_SQRT_2;
    [
        s * (up[0] + um[0]),
        s * (up[1] - um[1]),
        s * (up[1] + um[1]),
        s * (um[0] - up[0]),
    ]
}

// ============================================================================
// 1D Complex MaxEnt Solver
// ============================================================================

struct Solve1DWorkspace {
    n1: usize,
    n_fft: usize,
    d_masked: Vec<Complex64>,
    mask: Vec<bool>,
    work_td: Vec<Complex64>,
    f_cur: Vec<Complex64>,
    f_next: Vec<Complex64>,
    w_cur: Vec<Complex64>,
    af_cur: Vec<Complex64>,
    af_next: Vec<Complex64>,
    aw_cur: Vec<Complex64>,
    gc_cur: Vec<Complex64>,
    gc_next: Vec<Complex64>,
    gc_w: Vec<Complex64>,
}

struct InnerSolveStats {
    iterations: usize,
    restarts: usize,
    converged: bool,
    kkt_residual_inf: f64,
    primal_objective: f64,
    duality_gap: f64,
    data_residual_sq: f64,
    entropy: f64,
    objective_trace: Vec<f64>,
}

impl Solve1DWorkspace {
    fn new(obs: &[Complex64], mask: &[bool], n_fft: usize) -> Self {
        let n1 = obs.len();
        let mut d_masked = vec![[0.0, 0.0]; n1];
        for k in 0..n1 {
            if mask[k] {
                d_masked[k] = obs[k];
            }
        }
        Self {
            n1,
            n_fft,
            d_masked,
            mask: mask.to_vec(),
            work_td: vec![[0.0, 0.0]; n_fft],
            f_cur: vec![[0.0, 0.0]; n_fft],
            f_next: vec![[0.0, 0.0]; n_fft],
            w_cur: vec![[0.0, 0.0]; n_fft],
            af_cur: vec![[0.0, 0.0]; n1],
            af_next: vec![[0.0, 0.0]; n1],
            aw_cur: vec![[0.0, 0.0]; n1],
            gc_cur: vec![[0.0, 0.0]; n_fft],
            gc_next: vec![[0.0, 0.0]; n_fft],
            gc_w: vec![[0.0, 0.0]; n_fft],
        }
    }

    fn apply_forward_a(
        n1: usize,
        mask: &[bool],
        spec: &[Complex64],
        work_td: &mut [Complex64],
        out_af: &mut [Complex64],
    ) {
        work_td.copy_from_slice(spec);
        fft_1d_unitary(work_td, true);
        for k in 0..n1 {
            out_af[k] = if mask[k] { work_td[k] } else { [0.0, 0.0] };
        }
    }

    fn apply_adjoint_residual(
        n1: usize,
        mask: &[bool],
        af: &[Complex64],
        d_masked: &[Complex64],
        work_td: &mut [Complex64],
        out_gc: &mut [Complex64],
    ) -> f64 {
        work_td.fill([0.0, 0.0]);
        let mut res_sq = 0.0_f64;
        for k in 0..n1 {
            if mask[k] {
                let dr = af[k][0] - d_masked[k][0];
                let di = af[k][1] - d_masked[k][1];
                work_td[k] = [dr, di];
                res_sq += dr * dr + di * di;
            }
        }
        fft_1d_unitary(work_td, false);
        out_gc.copy_from_slice(work_td);
        res_sq
    }

    fn eval_entropy_and_kkt(
        spec: &[Complex64],
        gc: &[Complex64],
        def: f64,
        lambda: f64,
    ) -> (f64, f64) {
        let mut neg_s = 0.0_f64;
        let mut kkt_inf = 0.0_f64;
        for (f, g) in spec.iter().zip(gc.iter()) {
            let r = f[0].hypot(f[1]);
            neg_s += neg_entropy_scalar(r, def);
            let gs = neg_entropy_grad_scale(r, def);
            let kr = (f[0] * gs + lambda * g[0]).abs();
            let ki = (f[1] * gs + lambda * g[1]).abs();
            let kmax = kr.max(ki);
            if kmax > kkt_inf {
                kkt_inf = kmax;
            }
        }
        (-neg_s, kkt_inf)
    }

    fn eval_duality_gap(
        primal_obj: f64,
        af: &[Complex64],
        d_masked: &[Complex64],
        mask: &[bool],
        gc: &[Complex64],
        def: f64,
        lambda: f64,
    ) -> f64 {
        // Dual variable w = lambda * (d - A f); notice A^* w = -lambda * gc!
        let mut h_star = 0.0_f64;
        for g in gc {
            let z_mag = lambda * g[0].hypot(g[1]);
            h_star += dual_conj_scalar(z_mag, def);
        }
        let mut w_norm_sq = 0.0_f64;
        let mut w_dot_d = 0.0_f64;
        for k in 0..af.len() {
            if mask[k] {
                let wr = lambda * (d_masked[k][0] - af[k][0]);
                let wi = lambda * (d_masked[k][1] - af[k][1]);
                w_norm_sq += wr * wr + wi * wi;
                w_dot_d += wr * d_masked[k][0] + wi * d_masked[k][1];
            }
        }
        let dual_obj = h_star + (0.5 / lambda) * w_norm_sq - w_dot_d;
        (primal_obj + dual_obj).max(0.0)
    }

    #[allow(clippy::too_many_arguments)]
    fn solve_fixed_lambda(
        &mut self,
        def: f64,
        lambda: f64,
        max_iters: usize,
        kkt_tol: f64,
        accelerate: bool,
        adaptive_restart: bool,
        adaptive_step_size: bool,
        record_trace: bool,
    ) -> InnerSolveStats {
        let mu = 1.0 / lambda;
        let mut trace = if record_trace {
            Vec::with_capacity(max_iters.min(2048))
        } else {
            Vec::new()
        };

        // Sync initial A f_cur and grad C(f_cur)
        Self::apply_forward_a(
            self.n1,
            &self.mask,
            &self.f_cur,
            &mut self.work_td,
            &mut self.af_cur,
        );
        let mut res_sq = Self::apply_adjoint_residual(
            self.n1,
            &self.mask,
            &self.af_cur,
            &self.d_masked,
            &mut self.work_td,
            &mut self.gc_cur,
        );
        let (mut entropy, mut kkt_inf) =
            Self::eval_entropy_and_kkt(&self.f_cur, &self.gc_cur, def, lambda);
        let mut primal_obj = -entropy + 0.5 * lambda * res_sq;

        self.w_cur.copy_from_slice(&self.f_cur);
        self.aw_cur.copy_from_slice(&self.af_cur);
        self.gc_w.copy_from_slice(&self.gc_cur);

        if record_trace {
            trace.push(primal_obj);
        }

        if kkt_inf <= kkt_tol {
            let duality_gap = Self::eval_duality_gap(
                primal_obj,
                &self.af_cur,
                &self.d_masked,
                &self.mask,
                &self.gc_cur,
                def,
                lambda,
            );
            return InnerSolveStats {
                iterations: 0,
                restarts: 0,
                converged: true,
                kkt_residual_inf: kkt_inf,
                primal_objective: primal_obj,
                duality_gap,
                data_residual_sq: res_sq,
                entropy,
                objective_trace: trace,
            };
        }

        let mut t_fista = 1.0_f64;
        let mut l_lip = 1.0_f64;
        let mut l_prev = 1.0_f64;
        let mut restarts = 0usize;
        let mut iters_done = 0usize;
        let mut momentum_active = false;

        for iter in 0..max_iters {
            iters_done = iter + 1;
            let mut l_try = if adaptive_step_size {
                (0.9 * l_lip).clamp(MIN_ADAPTIVE_L, 1.0)
            } else {
                1.0
            };

            // Backtracking loop on exact quadratic majorization: ||A (f_next - w)||^2 <= L ||f_next - w||^2
            let mut res_next;
            let mut ent_next;
            let mut obj_next;
            let mut kkt_next;

            loop {
                let tau = 1.0 / l_try;
                let alpha = tau * mu;
                let mut diff_sq = 0.0_f64;
                for m in 0..self.n_fft {
                    let vr = self.w_cur[m][0] - tau * self.gc_w[m][0];
                    let vi = self.w_cur[m][1] - tau * self.gc_w[m][1];
                    let vmag = vr.hypot(vi);
                    let shrink = maxent_prox_shrinkage(vmag, alpha, def);
                    let fn_r = vr * shrink;
                    let fn_i = vi * shrink;
                    self.f_next[m] = [fn_r, fn_i];
                    let dr = fn_r - self.w_cur[m][0];
                    let di = fn_i - self.w_cur[m][1];
                    diff_sq += dr * dr + di * di;
                }

                Self::apply_forward_a(
                    self.n1,
                    &self.mask,
                    &self.f_next,
                    &mut self.work_td,
                    &mut self.af_next,
                );

                if l_try < 1.0 && diff_sq > 0.0 {
                    let mut a_diff_sq = 0.0_f64;
                    for k in 0..self.n1 {
                        if self.mask[k] {
                            let dr = self.af_next[k][0] - self.aw_cur[k][0];
                            let di = self.af_next[k][1] - self.aw_cur[k][1];
                            a_diff_sq += dr * dr + di * di;
                        }
                    }
                    if a_diff_sq > l_try * diff_sq * (1.0 + 1e-12) {
                        let l_est = (a_diff_sq / diff_sq) * 1.05;
                        l_try = (2.0 * l_try).max(l_est).min(1.0);
                        continue;
                    }
                }

                res_next = Self::apply_adjoint_residual(
                    self.n1,
                    &self.mask,
                    &self.af_next,
                    &self.d_masked,
                    &mut self.work_td,
                    &mut self.gc_next,
                );
                let (e_n, k_n) =
                    Self::eval_entropy_and_kkt(&self.f_next, &self.gc_next, def, lambda);
                ent_next = e_n;
                kkt_next = k_n;
                obj_next = -ent_next + 0.5 * lambda * res_next;
                break;
            }

            // Monotone objective safeguard: if momentum overshot the primal objective,
            // restart at f_cur and take a guaranteed-descent proximal-gradient step.
            if adaptive_restart
                && momentum_active
                && obj_next > primal_obj + 1e-13 * primal_obj.abs().max(1.0)
            {
                restarts += 1;
                t_fista = 1.0;
                l_try = 1.0;
                l_prev = 1.0;
                momentum_active = false;
                self.w_cur.copy_from_slice(&self.f_cur);
                self.aw_cur.copy_from_slice(&self.af_cur);
                self.gc_w.copy_from_slice(&self.gc_cur);

                let alpha = mu;
                for m in 0..self.n_fft {
                    let vr = self.f_cur[m][0] - self.gc_cur[m][0];
                    let vi = self.f_cur[m][1] - self.gc_cur[m][1];
                    let vmag = vr.hypot(vi);
                    let shrink = maxent_prox_shrinkage(vmag, alpha, def);
                    self.f_next[m] = [vr * shrink, vi * shrink];
                }
                Self::apply_forward_a(
                    self.n1,
                    &self.mask,
                    &self.f_next,
                    &mut self.work_td,
                    &mut self.af_next,
                );
                res_next = Self::apply_adjoint_residual(
                    self.n1,
                    &self.mask,
                    &self.af_next,
                    &self.d_masked,
                    &mut self.work_td,
                    &mut self.gc_next,
                );
                let (e_n, k_n) =
                    Self::eval_entropy_and_kkt(&self.f_next, &self.gc_next, def, lambda);
                ent_next = e_n;
                kkt_next = k_n;
                obj_next = -ent_next + 0.5 * lambda * res_next;
            }

            // O'Donoghue-Candes (2015) adaptive gradient restart check:
            // <w_k - f_{k+1}, f_{k+1} - f_k> > 0
            let mut grad_restart = false;
            if accelerate && adaptive_restart && momentum_active {
                let mut dot = 0.0_f64;
                for m in 0..self.n_fft {
                    let w_minus_fn_r = self.w_cur[m][0] - self.f_next[m][0];
                    let w_minus_fn_i = self.w_cur[m][1] - self.f_next[m][1];
                    let fn_minus_f_r = self.f_next[m][0] - self.f_cur[m][0];
                    let fn_minus_f_i = self.f_next[m][1] - self.f_cur[m][1];
                    dot += w_minus_fn_r * fn_minus_f_r + w_minus_fn_i * fn_minus_f_i;
                }
                if dot > 0.0 {
                    grad_restart = true;
                    restarts += 1;
                }
            }

            // Compute next FISTA extrapolation point
            if !accelerate || grad_restart {
                t_fista = 1.0;
                l_lip = l_try;
                l_prev = l_try;
                momentum_active = false;
                self.f_cur.copy_from_slice(&self.f_next);
                self.af_cur.copy_from_slice(&self.af_next);
                self.gc_cur.copy_from_slice(&self.gc_next);
                self.w_cur.copy_from_slice(&self.f_cur);
                self.aw_cur.copy_from_slice(&self.af_cur);
                self.gc_w.copy_from_slice(&self.gc_cur);
            } else {
                let ratio = (l_prev / l_try).max(1e-6);
                let t_next = 0.5 * (1.0 + (1.0 + 4.0 * ratio * t_fista * t_fista).sqrt());
                let beta = (t_fista - 1.0) / t_next;
                for m in 0..self.n_fft {
                    self.w_cur[m] = [
                        self.f_next[m][0] + beta * (self.f_next[m][0] - self.f_cur[m][0]),
                        self.f_next[m][1] + beta * (self.f_next[m][1] - self.f_cur[m][1]),
                    ];
                    self.gc_w[m] = [
                        self.gc_next[m][0] + beta * (self.gc_next[m][0] - self.gc_cur[m][0]),
                        self.gc_next[m][1] + beta * (self.gc_next[m][1] - self.gc_cur[m][1]),
                    ];
                }
                for k in 0..self.n1 {
                    if self.mask[k] {
                        self.aw_cur[k] = [
                            self.af_next[k][0] + beta * (self.af_next[k][0] - self.af_cur[k][0]),
                            self.af_next[k][1] + beta * (self.af_next[k][1] - self.af_cur[k][1]),
                        ];
                    } else {
                        self.aw_cur[k] = [0.0, 0.0];
                    }
                }
                self.f_cur.copy_from_slice(&self.f_next);
                self.af_cur.copy_from_slice(&self.af_next);
                self.gc_cur.copy_from_slice(&self.gc_next);
                t_fista = t_next;
                l_prev = l_try;
                l_lip = l_try;
                momentum_active = beta > 0.0;
            }

            res_sq = res_next;
            entropy = ent_next;
            kkt_inf = kkt_next;
            primal_obj = obj_next;

            if record_trace {
                trace.push(primal_obj);
            }

            if kkt_inf <= kkt_tol {
                break;
            }
        }

        let duality_gap = Self::eval_duality_gap(
            primal_obj,
            &self.af_cur,
            &self.d_masked,
            &self.mask,
            &self.gc_cur,
            def,
            lambda,
        );

        InnerSolveStats {
            iterations: iters_done,
            restarts,
            converged: kkt_inf <= kkt_tol,
            kkt_residual_inf: kkt_inf,
            primal_objective: primal_obj,
            duality_gap,
            data_residual_sq: res_sq,
            entropy,
            objective_trace: trace,
        }
    }
}

/// Reconstruct a 1D complex non-uniformly sampled FID `obs[0..n1]` sampled on `mask[0..n1]`
/// using clean-room Convex Accelerated Maximum-Entropy (`MaxEnt`) optimization.
///
/// Minimizes:
/// $$J(\mathbf{f}) = -S(\mathbf{f}; \delta) + \frac{\lambda}{2} \|\mathcal{M}\mathcal{F}^{-1}\mathbf{f} - \mathbf{d}\|_2^2$$
/// via accelerated proximal-gradient FISTA with O'Donoghue–Candès adaptive gradient restart,
/// exact quadratic-majorization backtracking, and optional Morozov discrepancy-principle
/// $\lambda$ selection ($\chi^2_\nu \approx 1$).
pub fn reconstruct_maxent_1d(
    obs: &[Complex64],
    mask: &[bool],
    cfg: &MaxEntConfig,
) -> Result<MaxEntResult<Complex64>, String> {
    validate_config(cfg)?;
    let n1 = obs.len();
    if n1 == 0 {
        return Err("1D MaxEnt input must be non-empty".to_string());
    }
    if mask.len() != n1 {
        return Err(format!(
            "1D MaxEnt mask length ({}) does not match obs length ({n1})",
            mask.len()
        ));
    }
    let n_fft = n1
        .checked_mul(cfg.zero_fill_factor)
        .ok_or_else(|| "1D MaxEnt grid size overflow".to_string())?;
    if n_fft > MAX_MAXENT_POINTS {
        return Err(format!(
            "1D MaxEnt FFT grid size {n_fft} exceeds MAX_MAXENT_POINTS ({MAX_MAXENT_POINTS})"
        ));
    }
    let ws_bytes = maxent_1d_workspace_bytes(n1, n_fft)?;
    if ws_bytes > MAX_MAXENT_WORKSPACE_BYTES {
        return Err(format!(
            "1D MaxEnt workspace size ({ws_bytes} bytes) exceeds MAX_MAXENT_WORKSPACE_BYTES ({MAX_MAXENT_WORKSPACE_BYTES})"
        ));
    }
    let mut n_sampled = 0usize;
    let mut data_energy = 0.0_f64;
    for (idx, (&sample, &m)) in obs.iter().zip(mask.iter()).enumerate() {
        if !sample[0].is_finite() || !sample[1].is_finite() {
            return Err(format!("non-finite 1D FID sample at index {idx}"));
        }
        if m {
            n_sampled += 1;
            data_energy += sample[0] * sample[0] + sample[1] * sample[1];
        }
    }
    if n_sampled == 0 {
        return Err("1D MaxEnt NUS mask has zero sampled points (fail-closed)".to_string());
    }
    let n_obs_scalars = 2 * n_sampled;
    let mut ws = Solve1DWorkspace::new(obs, mask, n_fft);

    let (final_lambda, outer_iters, total_iters, stats) = match cfg.lambda_mode {
        MaxEntLambdaMode::Fixed { lambda } => {
            let st = ws.solve_fixed_lambda(
                cfg.def,
                lambda,
                cfg.max_iterations,
                cfg.kkt_tol,
                cfg.accelerate,
                cfg.adaptive_restart,
                cfg.adaptive_step_size,
                cfg.record_objective_trace,
            );
            (lambda, 0, st.iterations, st)
        }
        MaxEntLambdaMode::DiscrepancyPrinciple {
            noise_sigma,
            target_chi2_nu,
            chi2_tol,
            max_outer_iterations,
        } => {
            let denom = (n_obs_scalars as f64) * noise_sigma * noise_sigma;
            let search_tol = (cfg.kkt_tol * 5.0).max(cfg.kkt_tol).min(1e-5);
            let (chosen_lambda, outer_evals, mut total_inner) = select_discrepancy_lambda(
                cfg.def,
                noise_sigma,
                target_chi2_nu,
                chi2_tol,
                max_outer_iterations,
                data_energy,
                denom,
                |lam| {
                    let st = ws.solve_fixed_lambda(
                        cfg.def,
                        lam,
                        cfg.max_iterations,
                        search_tol,
                        cfg.accelerate,
                        cfg.adaptive_restart,
                        cfg.adaptive_step_size,
                        false,
                    );
                    (st.data_residual_sq / denom, st.iterations)
                },
            );

            // Final polish at chosen_lambda to full cfg.kkt_tol
            if cfg.record_objective_trace {
                // Reset to zero so trace captures the full trajectory at chosen_lambda
                ws.f_cur.fill([0.0, 0.0]);
            }
            let st = ws.solve_fixed_lambda(
                cfg.def,
                chosen_lambda,
                cfg.max_iterations,
                cfg.kkt_tol,
                cfg.accelerate,
                cfg.adaptive_restart,
                cfg.adaptive_step_size,
                cfg.record_objective_trace,
            );
            total_inner += st.iterations;
            (chosen_lambda, outer_evals, total_inner, st)
        }
    };

    // Reconstruct full time-domain mock FID F^{-1}(f*) cropped to 0..n1
    ws.work_td.copy_from_slice(&ws.f_cur);
    fft_1d_unitary(&mut ws.work_td, true);
    let fid = ws.work_td[..n1].to_vec();

    let chi2_reduced = effective_noise_sigma(cfg)
        .map(|sigma| stats.data_residual_sq / ((n_obs_scalars as f64) * sigma * sigma));

    Ok(MaxEntResult {
        spectrum: ws.f_cur,
        fid,
        spectrum_shape: vec![n_fft],
        fid_shape: vec![n1],
        iterations: stats.iterations,
        total_iterations: total_iters,
        outer_iterations: outer_iters,
        restarts: stats.restarts,
        converged: stats.converged,
        kkt_residual_inf: stats.kkt_residual_inf,
        primal_objective: stats.primal_objective,
        duality_gap: stats.duality_gap,
        data_residual_sq: stats.data_residual_sq,
        chi2_reduced,
        entropy: stats.entropy,
        final_lambda,
        n_obs_scalars,
        objective_trace: stats.objective_trace,
        linearity_note: MAXENT_LINEARITY_DISCLOSURE.to_string(),
    })
}

/// Independently evaluate the exact 1D KKT stationarity infinity norm:
/// $\|\nabla_{\mathbf{f}} (-S(\mathbf{f}; \delta) + \lambda C(\mathbf{f}))\|_\infty$.
pub fn kkt_residual_inf_1d(
    spectrum: &[Complex64],
    obs: &[Complex64],
    mask: &[bool],
    def: f64,
    lambda: f64,
) -> Result<f64, String> {
    validate_positive_squarable(def, "def")?;
    validate_positive_squarable(lambda, "lambda")?;
    let n1 = obs.len();
    let n_fft = spectrum.len();
    if n1 == 0 || n_fft < n1 || mask.len() != n1 {
        return Err("invalid dimensions for kkt_residual_inf_1d".to_string());
    }
    if n_fft > MAX_MAXENT_POINTS {
        return Err(format!(
            "kkt_residual_inf_1d spectrum length {n_fft} exceeds MAX_MAXENT_POINTS ({MAX_MAXENT_POINTS})"
        ));
    }
    if spectrum
        .iter()
        .any(|c| !c[0].is_finite() || !c[1].is_finite())
    {
        return Err("non-finite spectrum value in kkt_residual_inf_1d".to_string());
    }
    if obs.iter().any(|c| !c[0].is_finite() || !c[1].is_finite()) {
        return Err("non-finite observation value in kkt_residual_inf_1d".to_string());
    }
    let mut td = spectrum.to_vec();
    fft_1d_unitary(&mut td, true);
    for k in 0..n_fft {
        if k < n1 && mask[k] {
            td[k] = [td[k][0] - obs[k][0], td[k][1] - obs[k][1]];
        } else {
            td[k] = [0.0, 0.0];
        }
    }
    fft_1d_unitary(&mut td, false);
    let (_, kkt_inf) = Solve1DWorkspace::eval_entropy_and_kkt(spectrum, &td, def, lambda);
    Ok(kkt_inf)
}

// ============================================================================
// 2D Hypercomplex MaxEnt Solver (States / States-TPPI / Echo-Antiecho)
// ============================================================================

struct Solve2DWorkspace {
    n2: usize,
    n1: usize,
    f2: usize,
    f1: usize,
    n_vox: usize,
    mask: Vec<bool>,
    // Masked target in unitary bicomplex time domain (`len = n2 * n1`)
    dp_masked: Vec<Complex64>,
    dm_masked: Vec<Complex64>,
    // FFT scratch buffers (`len = f2 * f1`)
    work_p: Vec<Complex64>,
    work_m: Vec<Complex64>,
    col_buf: Vec<Complex64>,
    // Primal frequency-domain iterates in bicomplex form (`len = f2 * f1`)
    fp_cur: Vec<Complex64>,
    fm_cur: Vec<Complex64>,
    fp_next: Vec<Complex64>,
    fm_next: Vec<Complex64>,
    wp_cur: Vec<Complex64>,
    wm_cur: Vec<Complex64>,
    // Forward masked time-domain predictions (`len = n2 * n1`)
    afp_cur: Vec<Complex64>,
    afm_cur: Vec<Complex64>,
    afp_next: Vec<Complex64>,
    afm_next: Vec<Complex64>,
    awp_cur: Vec<Complex64>,
    awm_cur: Vec<Complex64>,
    // Adjoint gradients A^*(A f - d) in bicomplex frequency domain (`len = f2 * f1`)
    gcp_cur: Vec<Complex64>,
    gcm_cur: Vec<Complex64>,
    gcp_next: Vec<Complex64>,
    gcm_next: Vec<Complex64>,
    gcp_w: Vec<Complex64>,
    gcm_w: Vec<Complex64>,
}

impl Solve2DWorkspace {
    fn new_empty(n2: usize, n1: usize, f2: usize, f1: usize, mask: &[bool]) -> Self {
        let total = n2 * n1;
        let n_vox = f2 * f1;
        Self {
            n2,
            n1,
            f2,
            f1,
            n_vox,
            mask: mask.to_vec(),
            dp_masked: vec![[0.0, 0.0]; total],
            dm_masked: vec![[0.0, 0.0]; total],
            work_p: vec![[0.0, 0.0]; n_vox],
            work_m: vec![[0.0, 0.0]; n_vox],
            col_buf: vec![[0.0, 0.0]; f2],
            fp_cur: vec![[0.0, 0.0]; n_vox],
            fm_cur: vec![[0.0, 0.0]; n_vox],
            fp_next: vec![[0.0, 0.0]; n_vox],
            fm_next: vec![[0.0, 0.0]; n_vox],
            wp_cur: vec![[0.0, 0.0]; n_vox],
            wm_cur: vec![[0.0, 0.0]; n_vox],
            afp_cur: vec![[0.0, 0.0]; total],
            afm_cur: vec![[0.0, 0.0]; total],
            afp_next: vec![[0.0, 0.0]; total],
            afm_next: vec![[0.0, 0.0]; total],
            awp_cur: vec![[0.0, 0.0]; total],
            awm_cur: vec![[0.0, 0.0]; total],
            gcp_cur: vec![[0.0, 0.0]; n_vox],
            gcm_cur: vec![[0.0, 0.0]; n_vox],
            gcp_next: vec![[0.0, 0.0]; n_vox],
            gcm_next: vec![[0.0, 0.0]; n_vox],
            gcp_w: vec![[0.0, 0.0]; n_vox],
            gcm_w: vec![[0.0, 0.0]; n_vox],
        }
    }

    fn new(
        obs_quad: &[[f64; 4]],
        n2: usize,
        n1: usize,
        f2: usize,
        f1: usize,
        mask: &[bool],
        quad_modes: [QuadMode; 2],
    ) -> Self {
        let mut ws = Self::new_empty(n2, n1, f2, f1, mask);
        let _ = ws.load_plane(obs_quad, quad_modes);
        ws
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_forward_a(
        n2: usize,
        n1: usize,
        f2: usize,
        f1: usize,
        mask: &[bool],
        fp: &[Complex64],
        fm: &[Complex64],
        work_p: &mut [Complex64],
        work_m: &mut [Complex64],
        col_buf: &mut [Complex64],
        out_afp: &mut [Complex64],
        out_afm: &mut [Complex64],
    ) {
        work_p.copy_from_slice(fp);
        work_m.copy_from_slice(fm);
        fft_2d_unitary(work_p, f2, f1, col_buf, true);
        fft_2d_unitary(work_m, f2, f1, col_buf, true);
        for k2 in 0..n2 {
            for k1 in 0..n1 {
                let idx = k2 * n1 + k1;
                if mask[idx] {
                    let g = k2 * f1 + k1;
                    out_afp[idx] = work_p[g];
                    out_afm[idx] = work_m[g];
                } else {
                    out_afp[idx] = [0.0, 0.0];
                    out_afm[idx] = [0.0, 0.0];
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_adjoint_residual(
        n2: usize,
        n1: usize,
        f2: usize,
        f1: usize,
        mask: &[bool],
        afp: &[Complex64],
        afm: &[Complex64],
        dp_masked: &[Complex64],
        dm_masked: &[Complex64],
        work_p: &mut [Complex64],
        work_m: &mut [Complex64],
        col_buf: &mut [Complex64],
        out_gcp: &mut [Complex64],
        out_gcm: &mut [Complex64],
    ) -> f64 {
        work_p.fill([0.0, 0.0]);
        work_m.fill([0.0, 0.0]);
        let mut res_sq = 0.0_f64;
        for k2 in 0..n2 {
            for k1 in 0..n1 {
                let idx = k2 * n1 + k1;
                if mask[idx] {
                    let g = k2 * f1 + k1;
                    let dpr = afp[idx][0] - dp_masked[idx][0];
                    let dpi = afp[idx][1] - dp_masked[idx][1];
                    let dmr = afm[idx][0] - dm_masked[idx][0];
                    let dmi = afm[idx][1] - dm_masked[idx][1];
                    work_p[g] = [dpr, dpi];
                    work_m[g] = [dmr, dmi];
                    res_sq += dpr * dpr + dpi * dpi + dmr * dmr + dmi * dmi;
                }
            }
        }
        fft_2d_unitary(work_p, f2, f1, col_buf, false);
        fft_2d_unitary(work_m, f2, f1, col_buf, false);
        out_gcp.copy_from_slice(work_p);
        out_gcm.copy_from_slice(work_m);
        res_sq
    }

    /// Evaluate Hoch–Hore entropy and exact KKT infinity norm in canonical hypercomplex
    /// `[RR, RI, IR, II]` coordinates (isometric to bicomplex `(U_+[m2, m1], U_-[m2, (f1-m1)%f1])`).
    #[allow(clippy::too_many_arguments)]
    fn eval_entropy_and_kkt(
        f2: usize,
        f1: usize,
        fp: &[Complex64],
        fm: &[Complex64],
        gcp: &[Complex64],
        gcm: &[Complex64],
        def: f64,
        lambda: f64,
    ) -> (f64, f64) {
        let mut neg_s = 0.0_f64;
        let mut kkt_inf = 0.0_f64;
        for m2 in 0..f2 {
            let row = m2 * f1;
            for m1 in 0..f1 {
                let m1_c = (f1 - m1) % f1;
                let idx_p = row + m1;
                let idx_m = row + m1_c;
                let up = fp[idx_p];
                let um = fm[idx_m];
                let f_hc = bicomplex_to_states(up, um);
                let g_hc = bicomplex_to_states(gcp[idx_p], gcm[idx_m]);
                let r = norm4(f_hc);
                neg_s += neg_entropy_scalar(r, def);
                let gs = neg_entropy_grad_scale(r, def);
                for ch in 0..4 {
                    let k = (f_hc[ch] * gs + lambda * g_hc[ch]).abs();
                    if k > kkt_inf {
                        kkt_inf = k;
                    }
                }
            }
        }
        (-neg_s, kkt_inf)
    }

    #[allow(clippy::too_many_arguments)]
    fn eval_duality_gap(
        primal_obj: f64,
        f2: usize,
        f1: usize,
        afp: &[Complex64],
        afm: &[Complex64],
        dp_masked: &[Complex64],
        dm_masked: &[Complex64],
        mask: &[bool],
        gcp: &[Complex64],
        gcm: &[Complex64],
        def: f64,
        lambda: f64,
    ) -> f64 {
        let mut h_star = 0.0_f64;
        for m2 in 0..f2 {
            let row = m2 * f1;
            for m1 in 0..f1 {
                let m1_c = (f1 - m1) % f1;
                let gp = gcp[row + m1];
                let gm = gcm[row + m1_c];
                let z_mag = lambda * gp[0].hypot(gp[1]).hypot(gm[0].hypot(gm[1]));
                h_star += dual_conj_scalar(z_mag, def);
            }
        }
        let mut w_norm_sq = 0.0_f64;
        let mut w_dot_d = 0.0_f64;
        for idx in 0..mask.len() {
            if mask[idx] {
                let wpr = lambda * (dp_masked[idx][0] - afp[idx][0]);
                let wpi = lambda * (dp_masked[idx][1] - afp[idx][1]);
                let wmr = lambda * (dm_masked[idx][0] - afm[idx][0]);
                let wmi = lambda * (dm_masked[idx][1] - afm[idx][1]);
                w_norm_sq += wpr * wpr + wpi * wpi + wmr * wmr + wmi * wmi;
                w_dot_d += wpr * dp_masked[idx][0]
                    + wpi * dp_masked[idx][1]
                    + wmr * dm_masked[idx][0]
                    + wmi * dm_masked[idx][1];
            }
        }
        let dual_obj = h_star + (0.5 / lambda) * w_norm_sq - w_dot_d;
        (primal_obj + dual_obj).max(0.0)
    }

    #[allow(clippy::too_many_arguments)]
    fn solve_fixed_lambda(
        &mut self,
        def: f64,
        lambda: f64,
        max_iters: usize,
        kkt_tol: f64,
        accelerate: bool,
        adaptive_restart: bool,
        adaptive_step_size: bool,
        record_trace: bool,
    ) -> InnerSolveStats {
        let mu = 1.0 / lambda;
        let total = self.n2 * self.n1;
        let mut trace = if record_trace {
            Vec::with_capacity(max_iters.min(2048))
        } else {
            Vec::new()
        };

        Self::apply_forward_a(
            self.n2,
            self.n1,
            self.f2,
            self.f1,
            &self.mask,
            &self.fp_cur,
            &self.fm_cur,
            &mut self.work_p,
            &mut self.work_m,
            &mut self.col_buf,
            &mut self.afp_cur,
            &mut self.afm_cur,
        );
        let mut res_sq = Self::apply_adjoint_residual(
            self.n2,
            self.n1,
            self.f2,
            self.f1,
            &self.mask,
            &self.afp_cur,
            &self.afm_cur,
            &self.dp_masked,
            &self.dm_masked,
            &mut self.work_p,
            &mut self.work_m,
            &mut self.col_buf,
            &mut self.gcp_cur,
            &mut self.gcm_cur,
        );
        let (mut entropy, mut kkt_inf) = Self::eval_entropy_and_kkt(
            self.f2,
            self.f1,
            &self.fp_cur,
            &self.fm_cur,
            &self.gcp_cur,
            &self.gcm_cur,
            def,
            lambda,
        );
        let mut primal_obj = -entropy + 0.5 * lambda * res_sq;

        self.wp_cur.copy_from_slice(&self.fp_cur);
        self.wm_cur.copy_from_slice(&self.fm_cur);
        self.awp_cur.copy_from_slice(&self.afp_cur);
        self.awm_cur.copy_from_slice(&self.afm_cur);
        self.gcp_w.copy_from_slice(&self.gcp_cur);
        self.gcm_w.copy_from_slice(&self.gcm_cur);

        if record_trace {
            trace.push(primal_obj);
        }

        if kkt_inf <= kkt_tol {
            let duality_gap = Self::eval_duality_gap(
                primal_obj,
                self.f2,
                self.f1,
                &self.afp_cur,
                &self.afm_cur,
                &self.dp_masked,
                &self.dm_masked,
                &self.mask,
                &self.gcp_cur,
                &self.gcm_cur,
                def,
                lambda,
            );
            return InnerSolveStats {
                iterations: 0,
                restarts: 0,
                converged: true,
                kkt_residual_inf: kkt_inf,
                primal_objective: primal_obj,
                duality_gap,
                data_residual_sq: res_sq,
                entropy,
                objective_trace: trace,
            };
        }

        let mut t_fista = 1.0_f64;
        let mut l_lip = 1.0_f64;
        let mut l_prev = 1.0_f64;
        let mut restarts = 0usize;
        let mut iters_done = 0usize;
        let mut momentum_active = false;

        for iter in 0..max_iters {
            iters_done = iter + 1;
            let mut l_try = if adaptive_step_size {
                (0.9 * l_lip).clamp(MIN_ADAPTIVE_L, 1.0)
            } else {
                1.0
            };

            let mut res_next;
            let mut ent_next;
            let mut obj_next;
            let mut kkt_next;

            loop {
                let tau = 1.0 / l_try;
                let alpha = tau * mu;
                let mut diff_sq = 0.0_f64;
                for m2 in 0..self.f2 {
                    let row = m2 * self.f1;
                    for m1 in 0..self.f1 {
                        let m1_c = (self.f1 - m1) % self.f1;
                        let idx_p = row + m1;
                        let idx_m = row + m1_c;
                        let vpr = self.wp_cur[idx_p][0] - tau * self.gcp_w[idx_p][0];
                        let vpi = self.wp_cur[idx_p][1] - tau * self.gcp_w[idx_p][1];
                        let vmr = self.wm_cur[idx_m][0] - tau * self.gcm_w[idx_m][0];
                        let vmi = self.wm_cur[idx_m][1] - tau * self.gcm_w[idx_m][1];
                        let vmag = vpr.hypot(vpi).hypot(vmr.hypot(vmi));
                        let shrink = maxent_prox_shrinkage(vmag, alpha, def);
                        let npr = vpr * shrink;
                        let npi = vpi * shrink;
                        let nmr = vmr * shrink;
                        let nmi = vmi * shrink;
                        self.fp_next[idx_p] = [npr, npi];
                        self.fm_next[idx_m] = [nmr, nmi];
                        let dpr = npr - self.wp_cur[idx_p][0];
                        let dpi = npi - self.wp_cur[idx_p][1];
                        let dmr = nmr - self.wm_cur[idx_m][0];
                        let dmi = nmi - self.wm_cur[idx_m][1];
                        diff_sq += dpr * dpr + dpi * dpi + dmr * dmr + dmi * dmi;
                    }
                }

                Self::apply_forward_a(
                    self.n2,
                    self.n1,
                    self.f2,
                    self.f1,
                    &self.mask,
                    &self.fp_next,
                    &self.fm_next,
                    &mut self.work_p,
                    &mut self.work_m,
                    &mut self.col_buf,
                    &mut self.afp_next,
                    &mut self.afm_next,
                );

                if l_try < 1.0 && diff_sq > 0.0 {
                    let mut a_diff_sq = 0.0_f64;
                    for idx in 0..total {
                        if self.mask[idx] {
                            let dpr = self.afp_next[idx][0] - self.awp_cur[idx][0];
                            let dpi = self.afp_next[idx][1] - self.awp_cur[idx][1];
                            let dmr = self.afm_next[idx][0] - self.awm_cur[idx][0];
                            let dmi = self.afm_next[idx][1] - self.awm_cur[idx][1];
                            a_diff_sq += dpr * dpr + dpi * dpi + dmr * dmr + dmi * dmi;
                        }
                    }
                    if a_diff_sq > l_try * diff_sq * (1.0 + 1e-12) {
                        let l_est = (a_diff_sq / diff_sq) * 1.05;
                        l_try = (2.0 * l_try).max(l_est).min(1.0);
                        continue;
                    }
                }

                res_next = Self::apply_adjoint_residual(
                    self.n2,
                    self.n1,
                    self.f2,
                    self.f1,
                    &self.mask,
                    &self.afp_next,
                    &self.afm_next,
                    &self.dp_masked,
                    &self.dm_masked,
                    &mut self.work_p,
                    &mut self.work_m,
                    &mut self.col_buf,
                    &mut self.gcp_next,
                    &mut self.gcm_next,
                );
                let (e_n, k_n) = Self::eval_entropy_and_kkt(
                    self.f2,
                    self.f1,
                    &self.fp_next,
                    &self.fm_next,
                    &self.gcp_next,
                    &self.gcm_next,
                    def,
                    lambda,
                );
                ent_next = e_n;
                kkt_next = k_n;
                obj_next = -ent_next + 0.5 * lambda * res_next;
                break;
            }

            // Monotone objective safeguard
            if adaptive_restart
                && momentum_active
                && obj_next > primal_obj + 1e-13 * primal_obj.abs().max(1.0)
            {
                restarts += 1;
                t_fista = 1.0;
                l_try = 1.0;
                l_prev = 1.0;
                momentum_active = false;
                self.wp_cur.copy_from_slice(&self.fp_cur);
                self.wm_cur.copy_from_slice(&self.fm_cur);
                self.awp_cur.copy_from_slice(&self.afp_cur);
                self.awm_cur.copy_from_slice(&self.afm_cur);
                self.gcp_w.copy_from_slice(&self.gcp_cur);
                self.gcm_w.copy_from_slice(&self.gcm_cur);

                let alpha = mu;
                for m2 in 0..self.f2 {
                    let row = m2 * self.f1;
                    for m1 in 0..self.f1 {
                        let m1_c = (self.f1 - m1) % self.f1;
                        let idx_p = row + m1;
                        let idx_m = row + m1_c;
                        let vpr = self.fp_cur[idx_p][0] - self.gcp_cur[idx_p][0];
                        let vpi = self.fp_cur[idx_p][1] - self.gcp_cur[idx_p][1];
                        let vmr = self.fm_cur[idx_m][0] - self.gcm_cur[idx_m][0];
                        let vmi = self.fm_cur[idx_m][1] - self.gcm_cur[idx_m][1];
                        let vmag = vpr.hypot(vpi).hypot(vmr.hypot(vmi));
                        let shrink = maxent_prox_shrinkage(vmag, alpha, def);
                        self.fp_next[idx_p] = [vpr * shrink, vpi * shrink];
                        self.fm_next[idx_m] = [vmr * shrink, vmi * shrink];
                    }
                }
                Self::apply_forward_a(
                    self.n2,
                    self.n1,
                    self.f2,
                    self.f1,
                    &self.mask,
                    &self.fp_next,
                    &self.fm_next,
                    &mut self.work_p,
                    &mut self.work_m,
                    &mut self.col_buf,
                    &mut self.afp_next,
                    &mut self.afm_next,
                );
                res_next = Self::apply_adjoint_residual(
                    self.n2,
                    self.n1,
                    self.f2,
                    self.f1,
                    &self.mask,
                    &self.afp_next,
                    &self.afm_next,
                    &self.dp_masked,
                    &self.dm_masked,
                    &mut self.work_p,
                    &mut self.work_m,
                    &mut self.col_buf,
                    &mut self.gcp_next,
                    &mut self.gcm_next,
                );
                let (e_n, k_n) = Self::eval_entropy_and_kkt(
                    self.f2,
                    self.f1,
                    &self.fp_next,
                    &self.fm_next,
                    &self.gcp_next,
                    &self.gcm_next,
                    def,
                    lambda,
                );
                ent_next = e_n;
                kkt_next = k_n;
                obj_next = -ent_next + 0.5 * lambda * res_next;
            }

            // O'Donoghue-Candes adaptive gradient restart
            let mut grad_restart = false;
            if accelerate && adaptive_restart && momentum_active {
                let mut dot = 0.0_f64;
                for i in 0..self.n_vox {
                    dot += (self.wp_cur[i][0] - self.fp_next[i][0])
                        * (self.fp_next[i][0] - self.fp_cur[i][0])
                        + (self.wp_cur[i][1] - self.fp_next[i][1])
                            * (self.fp_next[i][1] - self.fp_cur[i][1])
                        + (self.wm_cur[i][0] - self.fm_next[i][0])
                            * (self.fm_next[i][0] - self.fm_cur[i][0])
                        + (self.wm_cur[i][1] - self.fm_next[i][1])
                            * (self.fm_next[i][1] - self.fm_cur[i][1]);
                }
                if dot > 0.0 {
                    grad_restart = true;
                    restarts += 1;
                }
            }

            if !accelerate || grad_restart {
                t_fista = 1.0;
                l_lip = l_try;
                l_prev = l_try;
                momentum_active = false;
                self.fp_cur.copy_from_slice(&self.fp_next);
                self.fm_cur.copy_from_slice(&self.fm_next);
                self.afp_cur.copy_from_slice(&self.afp_next);
                self.afm_cur.copy_from_slice(&self.afm_next);
                self.gcp_cur.copy_from_slice(&self.gcp_next);
                self.gcm_cur.copy_from_slice(&self.gcm_next);
                self.wp_cur.copy_from_slice(&self.fp_cur);
                self.wm_cur.copy_from_slice(&self.fm_cur);
                self.awp_cur.copy_from_slice(&self.afp_cur);
                self.awm_cur.copy_from_slice(&self.afm_cur);
                self.gcp_w.copy_from_slice(&self.gcp_cur);
                self.gcm_w.copy_from_slice(&self.gcm_cur);
            } else {
                let ratio = (l_prev / l_try).max(1e-6);
                let t_next = 0.5 * (1.0 + (1.0 + 4.0 * ratio * t_fista * t_fista).sqrt());
                let beta = (t_fista - 1.0) / t_next;
                for i in 0..self.n_vox {
                    self.wp_cur[i] = [
                        self.fp_next[i][0] + beta * (self.fp_next[i][0] - self.fp_cur[i][0]),
                        self.fp_next[i][1] + beta * (self.fp_next[i][1] - self.fp_cur[i][1]),
                    ];
                    self.wm_cur[i] = [
                        self.fm_next[i][0] + beta * (self.fm_next[i][0] - self.fm_cur[i][0]),
                        self.fm_next[i][1] + beta * (self.fm_next[i][1] - self.fm_cur[i][1]),
                    ];
                    self.gcp_w[i] = [
                        self.gcp_next[i][0] + beta * (self.gcp_next[i][0] - self.gcp_cur[i][0]),
                        self.gcp_next[i][1] + beta * (self.gcp_next[i][1] - self.gcp_cur[i][1]),
                    ];
                    self.gcm_w[i] = [
                        self.gcm_next[i][0] + beta * (self.gcm_next[i][0] - self.gcm_cur[i][0]),
                        self.gcm_next[i][1] + beta * (self.gcm_next[i][1] - self.gcm_cur[i][1]),
                    ];
                }
                for idx in 0..total {
                    if self.mask[idx] {
                        self.awp_cur[idx] = [
                            self.afp_next[idx][0]
                                + beta * (self.afp_next[idx][0] - self.afp_cur[idx][0]),
                            self.afp_next[idx][1]
                                + beta * (self.afp_next[idx][1] - self.afp_cur[idx][1]),
                        ];
                        self.awm_cur[idx] = [
                            self.afm_next[idx][0]
                                + beta * (self.afm_next[idx][0] - self.afm_cur[idx][0]),
                            self.afm_next[idx][1]
                                + beta * (self.afm_next[idx][1] - self.afm_cur[idx][1]),
                        ];
                    } else {
                        self.awp_cur[idx] = [0.0, 0.0];
                        self.awm_cur[idx] = [0.0, 0.0];
                    }
                }
                self.fp_cur.copy_from_slice(&self.fp_next);
                self.fm_cur.copy_from_slice(&self.fm_next);
                self.afp_cur.copy_from_slice(&self.afp_next);
                self.afm_cur.copy_from_slice(&self.afm_next);
                self.gcp_cur.copy_from_slice(&self.gcp_next);
                self.gcm_cur.copy_from_slice(&self.gcm_next);
                t_fista = t_next;
                l_prev = l_try;
                l_lip = l_try;
                momentum_active = beta > 0.0;
            }

            res_sq = res_next;
            entropy = ent_next;
            kkt_inf = kkt_next;
            primal_obj = obj_next;

            if record_trace {
                trace.push(primal_obj);
            }

            if kkt_inf <= kkt_tol {
                break;
            }
        }

        let duality_gap = Self::eval_duality_gap(
            primal_obj,
            self.f2,
            self.f1,
            &self.afp_cur,
            &self.afm_cur,
            &self.dp_masked,
            &self.dm_masked,
            &self.mask,
            &self.gcp_cur,
            &self.gcm_cur,
            def,
            lambda,
        );

        InnerSolveStats {
            iterations: iters_done,
            restarts,
            converged: kkt_inf <= kkt_tol,
            kkt_residual_inf: kkt_inf,
            primal_objective: primal_obj,
            duality_gap,
            data_residual_sq: res_sq,
            entropy,
            objective_trace: trace,
        }
    }

    fn export_spectrum(&self) -> Vec<[f64; 4]> {
        let mut spec = vec![[0.0; 4]; self.n_vox];
        self.export_spectrum_into(&mut spec);
        spec
    }

    fn export_spectrum_into(&self, out: &mut [[f64; 4]]) {
        for m2 in 0..self.f2 {
            let row = m2 * self.f1;
            for m1 in 0..self.f1 {
                let m1_c = (self.f1 - m1) % self.f1;
                out[row + m1] = bicomplex_to_states(self.fp_cur[row + m1], self.fm_cur[row + m1_c]);
            }
        }
    }

    fn export_fid(&mut self, quad_modes: [QuadMode; 2]) -> Vec<[f64; 4]> {
        let mut fid = vec![[0.0; 4]; self.n2 * self.n1];
        self.export_fid_into(quad_modes, &mut fid);
        fid
    }

    fn export_fid_into(&mut self, quad_modes: [QuadMode; 2], out: &mut [[f64; 4]]) {
        self.work_p.copy_from_slice(&self.fp_cur);
        self.work_m.copy_from_slice(&self.fm_cur);
        fft_2d_unitary(&mut self.work_p, self.f2, self.f1, &mut self.col_buf, true);
        fft_2d_unitary(&mut self.work_m, self.f2, self.f1, &mut self.col_buf, true);
        for k2 in 0..self.n2 {
            for k1 in 0..self.n1 {
                let g = k2 * self.f1 + k1;
                let states = bicomplex_to_states(self.work_p[g], self.work_m[g]);
                out[k2 * self.n1 + k1] = states_to_quad(states, k2, k1, quad_modes);
            }
        }
    }

    fn load_plane(&mut self, obs_quad: &[[f64; 4]], quad_modes: [QuadMode; 2]) -> f64 {
        let mut data_energy = 0.0_f64;
        for k2 in 0..self.n2 {
            for k1 in 0..self.n1 {
                let idx = k2 * self.n1 + k1;
                if self.mask[idx] {
                    let q = obs_quad[idx];
                    data_energy += q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
                    let states = quad_to_states(q, k2, k1, quad_modes);
                    let (up, um) = states_to_bicomplex(states);
                    self.dp_masked[idx] = up;
                    self.dm_masked[idx] = um;
                } else {
                    self.dp_masked[idx] = [0.0, 0.0];
                    self.dm_masked[idx] = [0.0, 0.0];
                }
            }
        }
        self.fp_cur.fill([0.0, 0.0]);
        self.fm_cur.fill([0.0, 0.0]);
        data_energy
    }

    fn solve_configured(
        &mut self,
        cfg: &MaxEntConfig,
        n_obs_scalars: usize,
        data_energy: f64,
    ) -> (f64, usize, usize, InnerSolveStats) {
        match cfg.lambda_mode {
            MaxEntLambdaMode::Fixed { lambda } => {
                let st = self.solve_fixed_lambda(
                    cfg.def,
                    lambda,
                    cfg.max_iterations,
                    cfg.kkt_tol,
                    cfg.accelerate,
                    cfg.adaptive_restart,
                    cfg.adaptive_step_size,
                    cfg.record_objective_trace,
                );
                (lambda, 0, st.iterations, st)
            }
            MaxEntLambdaMode::DiscrepancyPrinciple {
                noise_sigma,
                target_chi2_nu,
                chi2_tol,
                max_outer_iterations,
            } => {
                let denom = (n_obs_scalars as f64) * noise_sigma * noise_sigma;
                let search_tol = (cfg.kkt_tol * 5.0).max(cfg.kkt_tol).min(1e-5);
                let (chosen_lambda, outer_evals, mut total_inner) = select_discrepancy_lambda(
                    cfg.def,
                    noise_sigma,
                    target_chi2_nu,
                    chi2_tol,
                    max_outer_iterations,
                    data_energy,
                    denom,
                    |lam| {
                        let st = self.solve_fixed_lambda(
                            cfg.def,
                            lam,
                            cfg.max_iterations,
                            search_tol,
                            cfg.accelerate,
                            cfg.adaptive_restart,
                            cfg.adaptive_step_size,
                            false,
                        );
                        (st.data_residual_sq / denom, st.iterations)
                    },
                );

                if cfg.record_objective_trace {
                    self.fp_cur.fill([0.0, 0.0]);
                    self.fm_cur.fill([0.0, 0.0]);
                }
                let st = self.solve_fixed_lambda(
                    cfg.def,
                    chosen_lambda,
                    cfg.max_iterations,
                    cfg.kkt_tol,
                    cfg.accelerate,
                    cfg.adaptive_restart,
                    cfg.adaptive_step_size,
                    cfg.record_objective_trace,
                );
                total_inner += st.iterations;
                (chosen_lambda, outer_evals, total_inner, st)
            }
        }
    }
}

/// Reconstruct a 2D hypercomplex NUS indirect plane `obs_quad[k2 * n1 + k1] = [q00, q01, q10, q11]`
/// sampled on `mask[k2 * n1 + k1]` with quadrature modes `quad_modes = [axis2_mode, axis1_mode]`
/// (`States`, `StatesTppi`, or `EchoAntiecho`) using clean-room Convex Accelerated MaxEnt.
///
/// The solver converts `obs_quad` isometrically to canonical `States` `(cc, cs, sc, ss)` (for
/// `EchoAntiecho`, using the unitary Bruker `MC2 = 5` / `fid.rs::recombine_quad_pair` convention
/// including its `-i` sensitivity-enhancement rotation), solves the exact quaternion/bicomplex
/// Hoch–Hore entropy convex program, returns the pure-absorption canonical 2D hypercomplex
/// spectrum `[RR, RI, IR, II]` in `result.spectrum`, and returns the reconstructed time-domain
/// mock FID in the caller's input `quad_modes` encoding in `result.fid`.
pub fn reconstruct_maxent_2d(
    obs_quad: &[[f64; 4]],
    n2: usize,
    n1: usize,
    mask: &[bool],
    quad_modes: [QuadMode; 2],
    cfg: &MaxEntConfig,
) -> Result<MaxEntResult<[f64; 4]>, String> {
    validate_config(cfg)?;
    validate_hypercomplex_quad_mode(quad_modes[0], "axis 2 (outer)")?;
    validate_hypercomplex_quad_mode(quad_modes[1], "axis 1 (inner)")?;
    if n2 == 0 || n1 == 0 {
        return Err("2D MaxEnt dimensions (n2, n1) must both be >= 1".to_string());
    }
    let total = n2
        .checked_mul(n1)
        .ok_or_else(|| "2D MaxEnt grid size overflow".to_string())?;
    let f2 = n2
        .checked_mul(cfg.zero_fill_factor)
        .ok_or_else(|| "2D MaxEnt f2 overflow".to_string())?;
    let f1 = n1
        .checked_mul(cfg.zero_fill_factor)
        .ok_or_else(|| "2D MaxEnt f1 overflow".to_string())?;
    let n_vox = f2
        .checked_mul(f1)
        .ok_or_else(|| "2D MaxEnt FFT grid size overflow".to_string())?;
    if n_vox > MAX_MAXENT_POINTS {
        return Err(format!(
            "2D MaxEnt FFT grid size {n_vox} ({f2}x{f1}) exceeds MAX_MAXENT_POINTS ({MAX_MAXENT_POINTS})"
        ));
    }
    let ws_bytes = maxent_2d_workspace_bytes(n2, n1, f2, f1)?;
    if ws_bytes > MAX_MAXENT_WORKSPACE_BYTES {
        return Err(format!(
            "2D MaxEnt workspace size ({ws_bytes} bytes for {n2}x{n1} -> {f2}x{f1}) exceeds MAX_MAXENT_WORKSPACE_BYTES ({MAX_MAXENT_WORKSPACE_BYTES})"
        ));
    }
    if obs_quad.len() != total {
        return Err(format!(
            "2D MaxEnt obs_quad length ({}) does not match n2*n1 ({total})",
            obs_quad.len()
        ));
    }
    if mask.len() != total {
        return Err(format!(
            "2D MaxEnt mask length ({}) does not match n2*n1 ({total})",
            mask.len()
        ));
    }

    let mut n_sampled = 0usize;
    let mut data_energy = 0.0_f64;
    for (idx, (q, &m)) in obs_quad.iter().zip(mask.iter()).enumerate() {
        if q.iter().any(|v| !v.is_finite()) {
            return Err(format!("non-finite 2D hypercomplex sample at index {idx}"));
        }
        if m {
            n_sampled += 1;
            data_energy += q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
        }
    }
    if n_sampled == 0 {
        return Err("2D MaxEnt NUS mask has zero sampled points (fail-closed)".to_string());
    }
    let n_obs_scalars = 4 * n_sampled;
    let mut ws = Solve2DWorkspace::new(obs_quad, n2, n1, f2, f1, mask, quad_modes);
    let (final_lambda, outer_iters, total_iters, stats) =
        ws.solve_configured(cfg, n_obs_scalars, data_energy);

    let spectrum = ws.export_spectrum();
    let fid = ws.export_fid(quad_modes);
    let chi2_reduced = effective_noise_sigma(cfg)
        .map(|sigma| stats.data_residual_sq / ((n_obs_scalars as f64) * sigma * sigma));

    Ok(MaxEntResult {
        spectrum,
        fid,
        spectrum_shape: vec![f2, f1],
        fid_shape: vec![n2, n1],
        iterations: stats.iterations,
        total_iterations: total_iters,
        outer_iterations: outer_iters,
        restarts: stats.restarts,
        converged: stats.converged,
        kkt_residual_inf: stats.kkt_residual_inf,
        primal_objective: stats.primal_objective,
        duality_gap: stats.duality_gap,
        data_residual_sq: stats.data_residual_sq,
        chi2_reduced,
        entropy: stats.entropy,
        final_lambda,
        n_obs_scalars,
        objective_trace: stats.objective_trace,
        linearity_note: MAXENT_LINEARITY_DISCLOSURE.to_string(),
    })
}

/// Independently evaluate the exact 2D hypercomplex KKT stationarity infinity norm:
/// $\|\nabla_{\mathbf{f}} (-S(\mathbf{f}; \delta) + \lambda C(\mathbf{f}))\|_\infty$.
#[allow(clippy::too_many_arguments)]
pub fn kkt_residual_inf_2d(
    spectrum: &[[f64; 4]],
    f2: usize,
    f1: usize,
    obs_quad: &[[f64; 4]],
    n2: usize,
    n1: usize,
    mask: &[bool],
    quad_modes: [QuadMode; 2],
    def: f64,
    lambda: f64,
) -> Result<f64, String> {
    validate_positive_squarable(def, "def")?;
    validate_positive_squarable(lambda, "lambda")?;
    validate_hypercomplex_quad_mode(quad_modes[0], "axis 2")?;
    validate_hypercomplex_quad_mode(quad_modes[1], "axis 1")?;
    let total = n2
        .checked_mul(n1)
        .ok_or_else(|| "kkt_residual_inf_2d grid overflow".to_string())?;
    let n_vox = f2
        .checked_mul(f1)
        .ok_or_else(|| "kkt_residual_inf_2d spectrum overflow".to_string())?;
    if n2 == 0
        || n1 == 0
        || f2 < n2
        || f1 < n1
        || spectrum.len() != n_vox
        || obs_quad.len() != total
        || mask.len() != total
    {
        return Err("invalid dimensions for kkt_residual_inf_2d".to_string());
    }
    if n_vox > MAX_MAXENT_POINTS {
        return Err(format!(
            "kkt_residual_inf_2d spectrum size {n_vox} exceeds MAX_MAXENT_POINTS ({MAX_MAXENT_POINTS})"
        ));
    }
    if spectrum.iter().any(|q| q.iter().any(|v| !v.is_finite())) {
        return Err("non-finite spectrum value in kkt_residual_inf_2d".to_string());
    }
    if obs_quad.iter().any(|q| q.iter().any(|v| !v.is_finite())) {
        return Err("non-finite observation value in kkt_residual_inf_2d".to_string());
    }

    let mut fp = vec![[0.0, 0.0]; n_vox];
    let mut fm = vec![[0.0, 0.0]; n_vox];
    for m2 in 0..f2 {
        let row = m2 * f1;
        for m1 in 0..f1 {
            let m1_c = (f1 - m1) % f1;
            let (up, um) = states_to_bicomplex(spectrum[row + m1]);
            fp[row + m1] = up;
            fm[row + m1_c] = um;
        }
    }

    let mut work_p = fp.clone();
    let mut work_m = fm.clone();
    let mut col_buf = vec![[0.0, 0.0]; f2];
    fft_2d_unitary(&mut work_p, f2, f1, &mut col_buf, true);
    fft_2d_unitary(&mut work_m, f2, f1, &mut col_buf, true);

    for k2 in 0..f2 {
        for k1 in 0..f1 {
            let g = k2 * f1 + k1;
            if k2 < n2 && k1 < n1 && mask[k2 * n1 + k1] {
                let idx = k2 * n1 + k1;
                let states = quad_to_states(obs_quad[idx], k2, k1, quad_modes);
                let (dp, dm) = states_to_bicomplex(states);
                work_p[g] = [work_p[g][0] - dp[0], work_p[g][1] - dp[1]];
                work_m[g] = [work_m[g][0] - dm[0], work_m[g][1] - dm[1]];
            } else {
                work_p[g] = [0.0, 0.0];
                work_m[g] = [0.0, 0.0];
            }
        }
    }
    fft_2d_unitary(&mut work_p, f2, f1, &mut col_buf, false);
    fft_2d_unitary(&mut work_m, f2, f1, &mut col_buf, false);

    let (_, kkt_inf) =
        Solve2DWorkspace::eval_entropy_and_kkt(f2, f1, &fp, &fm, &work_p, &work_m, def, lambda);
    Ok(kkt_inf)
}

// ============================================================================
// 3D Hypercomplex Indirect-Plane MaxEnt Reconstruction
// ============================================================================

/// Reconstruct a 3D NMR volume of `nx` direct-dimension frequency slices, where each slice
/// `ix in 0..nx` is a 2D hypercomplex indirect plane `(n2, n1)` of `[q00, q01, q10, q11]`
/// sampled on the shared 2D NUS mask `mask_2d[k2 * n1 + k1]`.
///
/// Layout of `obs_planes`: row-major `(ix, k2, k1)` with `obs_planes.len() == nx * n2 * n1`.
/// Runs slices in parallel via `std::thread::scope` on native targets and sequentially on `wasm32`.
pub fn reconstruct_maxent_3d(
    obs_planes: &[[f64; 4]],
    nx: usize,
    n2: usize,
    n1: usize,
    mask_2d: &[bool],
    quad_modes: [QuadMode; 2],
    cfg: &MaxEntConfig,
) -> Result<MaxEntResult3D, String> {
    validate_config(cfg)?;
    validate_hypercomplex_quad_mode(quad_modes[0], "axis 2 (outer indirect)")?;
    validate_hypercomplex_quad_mode(quad_modes[1], "axis 1 (inner indirect)")?;
    if nx == 0 || n2 == 0 || n1 == 0 {
        return Err("3D MaxEnt dimensions (nx, n2, n1) must all be >= 1".to_string());
    }
    let plane_len = n2
        .checked_mul(n1)
        .ok_or_else(|| "3D MaxEnt plane size overflow".to_string())?;
    let total_voxels = nx
        .checked_mul(plane_len)
        .ok_or_else(|| "3D MaxEnt volume size overflow".to_string())?;
    if total_voxels > MAX_MAXENT_3D_TOTAL_POINTS {
        return Err(format!(
            "3D MaxEnt volume size {total_voxels} exceeds MAX_MAXENT_3D_TOTAL_POINTS ({MAX_MAXENT_3D_TOTAL_POINTS})"
        ));
    }
    let f2 = n2
        .checked_mul(cfg.zero_fill_factor)
        .ok_or_else(|| "3D MaxEnt f2 overflow".to_string())?;
    let f1 = n1
        .checked_mul(cfg.zero_fill_factor)
        .ok_or_else(|| "3D MaxEnt f1 overflow".to_string())?;
    let spec_plane_len = f2
        .checked_mul(f1)
        .ok_or_else(|| "3D MaxEnt plane FFT size overflow".to_string())?;
    if spec_plane_len > MAX_MAXENT_POINTS {
        return Err(format!(
            "3D MaxEnt plane FFT size {spec_plane_len} exceeds MAX_MAXENT_POINTS ({MAX_MAXENT_POINTS})"
        ));
    }
    let total_spec_voxels = nx
        .checked_mul(spec_plane_len)
        .ok_or_else(|| "3D MaxEnt total spectrum voxel count overflow".to_string())?;
    if total_spec_voxels > MAX_MAXENT_3D_TOTAL_VOXELS {
        return Err(format!(
            "3D MaxEnt total spectrum voxels {total_spec_voxels} ({nx} x {f2} x {f1}) exceeds MAX_MAXENT_3D_TOTAL_VOXELS ({MAX_MAXENT_3D_TOTAL_VOXELS})"
        ));
    }
    let ws_bytes = maxent_2d_workspace_bytes(n2, n1, f2, f1)?;
    if ws_bytes > MAX_MAXENT_WORKSPACE_BYTES {
        return Err(format!(
            "3D MaxEnt per-thread workspace size ({ws_bytes} bytes for {n2}x{n1} -> {f2}x{f1}) exceeds MAX_MAXENT_WORKSPACE_BYTES ({MAX_MAXENT_WORKSPACE_BYTES})"
        ));
    }
    if obs_planes.len() != total_voxels {
        return Err(format!(
            "3D MaxEnt obs_planes length ({}) does not match nx*n2*n1 ({total_voxels})",
            obs_planes.len()
        ));
    }
    if mask_2d.len() != plane_len {
        return Err(format!(
            "3D MaxEnt mask_2d length ({}) does not match n2*n1 ({plane_len})",
            mask_2d.len()
        ));
    }
    let n_sampled = mask_2d.iter().filter(|&&m| m).count();
    if n_sampled == 0 {
        return Err("3D MaxEnt 2D NUS mask has zero sampled points (fail-closed)".to_string());
    }
    for (idx, q) in obs_planes.iter().enumerate() {
        if q.iter().any(|v| !v.is_finite()) {
            return Err(format!(
                "non-finite 3D hypercomplex sample at flat index {idx}"
            ));
        }
    }

    let n_obs_scalars = 4 * n_sampled;
    let sigma_eff = effective_noise_sigma(cfg);
    let mut plane_cfg = cfg.clone();
    plane_cfg.record_objective_trace = false;

    // Pre-allocate exact bounded output buffers once and reuse one Solve2DWorkspace per thread
    let mut spectra_planes = vec![[0.0; 4]; total_spec_voxels];
    let mut fid_planes = vec![[0.0; 4]; total_voxels];
    let mut diag_slots: Vec<Option<MaxEntPlaneDiagnostics>> = vec![None; nx];

    let solve_chunk = |base_ix: usize,
                       spec_chunk: &mut [[f64; 4]],
                       fid_chunk: &mut [[f64; 4]],
                       diag_chunk: &mut [Option<MaxEntPlaneDiagnostics>]| {
        let mut ws = Solve2DWorkspace::new_empty(n2, n1, f2, f1, mask_2d);
        for (offset, diag_slot) in diag_chunk.iter_mut().enumerate() {
            let ix = base_ix + offset;
            let obs_slice = &obs_planes[ix * plane_len..(ix + 1) * plane_len];
            let data_energy = ws.load_plane(obs_slice, quad_modes);
            let (final_lambda, outer_iters, total_iters, stats) =
                ws.solve_configured(&plane_cfg, n_obs_scalars, data_energy);
            let spec_out = &mut spec_chunk[offset * spec_plane_len..(offset + 1) * spec_plane_len];
            let fid_out = &mut fid_chunk[offset * plane_len..(offset + 1) * plane_len];
            ws.export_spectrum_into(spec_out);
            ws.export_fid_into(quad_modes, fid_out);
            let chi2_reduced = sigma_eff
                .map(|sigma| stats.data_residual_sq / ((n_obs_scalars as f64) * sigma * sigma));
            *diag_slot = Some(MaxEntPlaneDiagnostics {
                direct_index: ix,
                iterations: stats.iterations,
                total_iterations: total_iters,
                outer_iterations: outer_iters,
                restarts: stats.restarts,
                converged: stats.converged,
                kkt_residual_inf: stats.kkt_residual_inf,
                primal_objective: stats.primal_objective,
                duality_gap: stats.duality_gap,
                data_residual_sq: stats.data_residual_sq,
                chi2_reduced,
                entropy: stats.entropy,
                final_lambda,
            });
        }
    };

    #[cfg(not(target_arch = "wasm32"))]
    {
        // Cap worker concurrency so total simultaneous workspace memory stays <= MAX_MAXENT_WORKSPACE_BYTES
        let max_threads_by_mem = (MAX_MAXENT_WORKSPACE_BYTES / ws_bytes.max(1)).max(1);
        let n_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .min(nx)
            .min(max_threads_by_mem);
        if n_threads <= 1 {
            solve_chunk(0, &mut spectra_planes, &mut fid_planes, &mut diag_slots);
        } else {
            let chunk_size = nx.div_ceil(n_threads);
            let spec_chunk_len = chunk_size * spec_plane_len;
            let fid_chunk_len = chunk_size * plane_len;
            std::thread::scope(|s| {
                for (chunk_idx, ((spec_chunk, fid_chunk), diag_chunk)) in spectra_planes
                    .chunks_mut(spec_chunk_len)
                    .zip(fid_planes.chunks_mut(fid_chunk_len))
                    .zip(diag_slots.chunks_mut(chunk_size))
                    .enumerate()
                {
                    let base_ix = chunk_idx * chunk_size;
                    s.spawn(move || {
                        solve_chunk(base_ix, spec_chunk, fid_chunk, diag_chunk);
                    });
                }
            });
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        solve_chunk(0, &mut spectra_planes, &mut fid_planes, &mut diag_slots);
    }

    let mut plane_diagnostics = Vec::with_capacity(nx);
    let mut total_iters = 0usize;
    let mut all_converged = true;
    let mut max_kkt = 0.0_f64;
    let mut sum_obj = 0.0_f64;
    let mut sum_ent = 0.0_f64;
    let mut sum_lam = 0.0_f64;
    let mut sum_chi2 = 0.0_f64;
    let mut chi2_count = 0usize;

    for (ix, maybe_diag) in diag_slots.into_iter().enumerate() {
        let diag = maybe_diag
            .ok_or_else(|| format!("missing 3D MaxEnt diagnostics for direct slice {ix}"))?;
        total_iters += diag.iterations;
        all_converged &= diag.converged;
        if diag.kkt_residual_inf > max_kkt {
            max_kkt = diag.kkt_residual_inf;
        }
        sum_obj += diag.primal_objective;
        sum_ent += diag.entropy;
        sum_lam += diag.final_lambda;
        if let Some(c2) = diag.chi2_reduced {
            sum_chi2 += c2;
            chi2_count += 1;
        }
        plane_diagnostics.push(diag);
    }

    let nx_f = nx as f64;
    Ok(MaxEntResult3D {
        spectra_planes,
        fid_planes,
        nx,
        n2,
        n1,
        f2,
        f1,
        plane_diagnostics,
        iterations: total_iters,
        converged: all_converged,
        max_kkt_residual_inf: max_kkt,
        primal_objective: sum_obj / nx_f,
        chi2_reduced: (chi2_count == nx).then_some(sum_chi2 / nx_f),
        entropy: sum_ent / nx_f,
        final_lambda: sum_lam / nx_f,
        linearity_note: MAXENT_LINEARITY_DISCLOSURE.to_string(),
    })
}

// ============================================================================
// Deterministic Sinusoidally Weighted Poisson-Gap NUS Mask Generators
// (Hyberts, Takeuchi & Wagner, J. Am. Chem. Soc. 2010, 132, 2145-2147)
// ============================================================================

/// Generate a deterministic 1D sinusoidally weighted Poisson-gap NUS sampling mask of length `n`
/// containing exactly `target_samples` sampled points (`1 <= target_samples <= n`), always
/// including `t = 0`.
pub fn poisson_gap_mask_1d(
    n: usize,
    target_samples: usize,
    sine_weight_power: f64,
    seed: u64,
) -> Result<Vec<bool>, String> {
    if n == 0 || n > MAX_MAXENT_POINTS {
        return Err(format!("invalid 1D Poisson-gap length {n}"));
    }
    if target_samples == 0 || target_samples > n {
        return Err(format!(
            "target_samples ({target_samples}) must be in 1..={n}"
        ));
    }
    if !sine_weight_power.is_finite() || !(0.0..=8.0).contains(&sine_weight_power) {
        return Err(format!(
            "sine_weight_power must be finite in [0, 8], got {sine_weight_power}"
        ));
    }
    if target_samples == n {
        return Ok(vec![true; n]);
    }

    // Binary search the Poisson gap rate scale `ld` to match `target_samples`
    let mut lo = 0.05_f64;
    let mut hi = (n as f64) * 2.0;
    let mut best_mask = vec![false; n];
    let mut best_diff = usize::MAX;

    for attempt in 0..32 {
        let ld = 0.5 * (lo + hi);
        let mut rng = Rng::new(seed ^ ((attempt as u64).wrapping_mul(0x9e3779b97f4a7c15)));
        let mut mask = vec![false; n];
        let mut idx = 0usize;
        let mut count = 0usize;
        while idx < n {
            mask[idx] = true;
            count += 1;
            let frac = (idx as f64) / ((n.saturating_sub(1)).max(1) as f64);
            let w = (0.5 * PI * frac).sin().max(0.0).powf(sine_weight_power);
            let gap = rng.poisson((ld * w).max(0.0));
            idx = idx.saturating_add(1).saturating_add(gap);
        }
        let diff = count.abs_diff(target_samples);
        if diff < best_diff {
            best_diff = diff;
            best_mask = mask;
            if diff == 0 {
                return Ok(best_mask);
            }
        }
        if count > target_samples {
            lo = ld;
        } else {
            hi = ld;
        }
    }

    // Deterministically adjust remaining discrepancy while keeping t = 0 sampled
    let mut rng = Rng::new(seed ^ 0x517cc1b727220a95);
    let mut cur_count = best_mask.iter().filter(|&&m| m).count();
    while cur_count < target_samples {
        let unsampled: Vec<usize> = (1..n).filter(|&i| !best_mask[i]).collect();
        if unsampled.is_empty() {
            break;
        }
        let pick = unsampled[rng.usize(unsampled.len())];
        best_mask[pick] = true;
        cur_count += 1;
    }
    while cur_count > target_samples {
        let sampled: Vec<usize> = (1..n).filter(|&i| best_mask[i]).collect();
        if sampled.is_empty() {
            break;
        }
        let pick = sampled[rng.usize(sampled.len())];
        best_mask[pick] = false;
        cur_count -= 1;
    }
    Ok(best_mask)
}

/// Generate a deterministic 2D sinusoidally weighted Poisson-gap NUS sampling mask of shape
/// `(n2, n1)` containing exactly `target_samples` sampled points (`1 <= target_samples <= n2 * n1`),
/// always including `(0, 0)`.
pub fn poisson_gap_mask_2d(
    n2: usize,
    n1: usize,
    target_samples: usize,
    sine_weight_power: f64,
    seed: u64,
) -> Result<Vec<bool>, String> {
    if n2 == 0 || n1 == 0 {
        return Err("2D Poisson-gap dimensions must be >= 1".to_string());
    }
    let total = n2
        .checked_mul(n1)
        .ok_or_else(|| "2D Poisson-gap grid overflow".to_string())?;
    if total > MAX_MAXENT_POINTS {
        return Err(format!(
            "2D Poisson-gap grid size {total} exceeds MAX_MAXENT_POINTS ({MAX_MAXENT_POINTS})"
        ));
    }
    if target_samples == 0 || target_samples > total {
        return Err(format!(
            "target_samples ({target_samples}) must be in 1..={total}"
        ));
    }
    if !sine_weight_power.is_finite() || !(0.0..=8.0).contains(&sine_weight_power) {
        return Err(format!(
            "sine_weight_power must be finite in [0, 8], got {sine_weight_power}"
        ));
    }
    if target_samples == total {
        return Ok(vec![true; total]);
    }

    // Build radial quarter-sine weighted probabilities favoring early evolution times while
    // traversing via Poisson gaps along serpentine scan order for gap-size uniformity.
    let mut order = Vec::with_capacity(total);
    for k2 in 0..n2 {
        if k2 % 2 == 0 {
            for k1 in 0..n1 {
                order.push((k2, k1));
            }
        } else {
            for k1 in (0..n1).rev() {
                order.push((k2, k1));
            }
        }
    }

    let mut lo = 0.05_f64;
    let mut hi = (total as f64) * 2.0;
    let mut best_mask = vec![false; total];
    let mut best_diff = usize::MAX;

    for attempt in 0..32 {
        let ld = 0.5 * (lo + hi);
        let mut rng = Rng::new(seed ^ ((attempt as u64).wrapping_mul(0x9e3779b97f4a7c15)));
        let mut mask = vec![false; total];
        let mut pos = 0usize;
        let mut count = 0usize;
        while pos < total {
            let (k2, k1) = order[pos];
            mask[k2 * n1 + k1] = true;
            count += 1;
            let f2 = (k2 as f64) / ((n2.saturating_sub(1)).max(1) as f64);
            let f1 = (k1 as f64) / ((n1.saturating_sub(1)).max(1) as f64);
            let r = (0.5 * (f2 * f2 + f1 * f1)).sqrt().min(1.0);
            let w = (0.5 * PI * r).sin().max(0.0).powf(sine_weight_power);
            let gap = rng.poisson((ld * w).max(0.0));
            pos = pos.saturating_add(1).saturating_add(gap);
        }
        let diff = count.abs_diff(target_samples);
        if diff < best_diff {
            best_diff = diff;
            best_mask = mask;
            if diff == 0 {
                return Ok(best_mask);
            }
        }
        if count > target_samples {
            lo = ld;
        } else {
            hi = ld;
        }
    }

    // Deterministically adjust remaining discrepancy while keeping (0, 0) sampled
    let mut rng = Rng::new(seed ^ 0x517cc1b727220a95);
    let mut cur_count = best_mask.iter().filter(|&&m| m).count();
    while cur_count < target_samples {
        let unsampled: Vec<usize> = (1..total).filter(|&i| !best_mask[i]).collect();
        if unsampled.is_empty() {
            break;
        }
        let pick = unsampled[rng.usize(unsampled.len())];
        best_mask[pick] = true;
        cur_count += 1;
    }
    while cur_count > target_samples {
        let sampled: Vec<usize> = (1..total).filter(|&i| best_mask[i]).collect();
        if sampled.is_empty() {
            break;
        }
        let pick = sampled[rng.usize(sampled.len())];
        best_mask[pick] = false;
        cur_count -= 1;
    }
    Ok(best_mask)
}
