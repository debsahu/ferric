//! Coupled 2D/3D multi-peak Voigt/Lorentzian/Gaussian lineshape deconvolution via separable
//! Variable Projection (`VarPro`) nonlinear least squares and exact Cramér–Rao amplitude
//! covariance (`T22.1`).
//!
//! Provenance: Clean-room original implementation derived strictly from peer-reviewed numerical
//! optimization and NMR spectroscopy literature (zero external GPL or proprietary code consulted):
//! - Golub, G. H. & Pereyra, V. (*SIAM J. Numer. Anal.* **1973**, 10(2), 413–432) — exact Fréchet
//!   derivative and separable Variable Projection (`VarPro`) for nonlinear least-squares problems
//!   whose model is linear in peak amplitudes and nonlinear in lineshape parameters.
//! - Kaufman, L. (*BIT Numerical Mathematics* **1975**, 15, 49–57) — reduced-cost Variable
//!   Projection Jacobian decomposition and block-projected linear-nonlinear coupling.
//! - Levenberg, K. (*Quart. Appl. Math.* **1944**, 2, 164–168); Marquardt, D. W. (*J. Soc. Ind.
//!   Appl. Math.* **1963**, 11, 431–441); Nielsen, H. B. (*IMM Technical Report 1999-05*) —
//!   damped Gauss–Newton / Levenberg–Marquardt trust-region updates.
//! - Thompson, P., Cox, D. E. & Hastings, J. B. (*J. Appl. Crystallogr.* **1987**, 20, 79–83);
//!   Wertheim, G. K., Butler, M. A., West, K. W. & Buchanan, D. N. E. (*Rev. Sci. Instrum.*
//!   **1974**, 45, 1369–1371) — pseudo-Voigt lineshape approximation and mixing parameterization.
//! - Delaglio, F., Grzesiek, S., Vuister, G. W., Zhu, G., Pfeifer, J. & Bax, A. (*J. Biomol. NMR*
//!   **1995**, 6, 277–293) — multidimensional NMR spectral tensor representation and separable
//!   lineshape modeling.
//! - Demmel, J. & Veselić, K. (*SIAM J. Matrix Anal. Appl.* **1992**, 13(4), 1204–1245) —
//!   high-relative-accuracy one-sided Jacobi singular value decomposition.
//! - Cramér, H. (*Mathematical Methods of Statistics*, Princeton Univ. Press, **1946**); Rao, C. R.
//!   (*Bull. Calcutta Math. Soc.* **1945**, 37, 81–91) — full joint linear-plus-nonlinear
//!   Fisher Information and Cramér–Rao covariance bounds.

use std::f64::consts::{LN_2, PI};
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::spectrum::{AxisPpm, Spectrum2D, Spectrum3D};

// ============================================================================
// Safety Guardrail Constants
// ============================================================================

/// Maximum voxel count allowed in any single 2D or 3D deconvolution ROI patch (safe on native
/// and `wasm32-unknown-unknown` targets).
pub const MAX_DECONV_VOXELS: usize = 65_536;

/// Maximum number of peaks allowed in a single coupled deconvolution cluster.
pub const MAX_DECONV_PEAKS_PER_CLUSTER: usize = 32;

/// Maximum number of planes allowed in a coupled pseudo-3D relaxation/titration plane series.
pub const MAX_DECONV_PLANES: usize = 256;

/// Maximum element count (`n_nonlinear * total_obs`) allowed in the VarPro Jacobian
/// (`8_000_000` `f64` elements = `64 MiB`).
pub const MAX_DECONV_JACOBIAN_ELEMS: usize = 8_000_000;

/// Maximum element count (`n_nonlinear * n_nonlinear`) allowed in the Levenberg–Marquardt
/// normal-equation Hessian (`1_000_000` `f64` elements = `8 MiB`).
pub const MAX_DECONV_HESSIAN_ELEMS: usize = 1_000_000;

/// Default upper-tail significance level ($\alpha = 10^{-3}$, i.e., 99.9% quantile) for the
/// known-$\sigma$ reduced $\chi^2_\nu$ model-misspecification gate.
pub const DEFAULT_CHI2_TAIL_PROB: f64 = 1.0e-3;

/// Default condition-number threshold on the basis matrix $\Phi$ (or full Schur Jacobian) above
/// which [`DeconvDiagnostic::IllConditionedCluster`] is raised.
pub const DEFAULT_ILL_CONDITIONED_KAPPA: f64 = 1.0e4;

/// Default normalized inter-peak separation threshold (in units of mean pairwise FWHM) below
/// which [`DeconvDiagnostic::DegeneratePeakPair`] is raised.
pub const DEFAULT_DEGENERATE_SEPARATION_FRAC: f64 = 0.25;

// ============================================================================
// Public Enums, Errors & Configuration
// ============================================================================

/// Separable 1D lineshape profile used along each spectral axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LineshapeKind {
    /// Pure Lorentzian absorption lineshape $L(x) = [1 + (2(x-\mu)/\gamma)^2]^{-1}$.
    Lorentzian,
    /// Pure Gaussian absorption lineshape $G(x) = \exp[-\ln(2)(2(x-\mu)/\gamma)^2]$.
    Gaussian,
    /// Pseudo-Voigt lineshape $V(x) = \eta L(x) + (1 - \eta) G(x)$ with $\eta \in [0, 1]$
    /// (Thompson, Cox & Hastings, *J. Appl. Crystallogr.* 1987).
    PseudoVoigt,
}

/// Linewidth (FWHM) sharing strategy across peaks and pseudo-3D planes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LinewidthSharing {
    /// Each peak owns independent linewidths on every axis; in multi-plane series, each plane
    /// also owns independent linewidths while peak centers remain coupled across planes.
    Independent,
    /// All peaks in the ROI cluster share the indirect-axis linewidth(s) (`F1` in 2D; `F1` and
    /// `F2` in 3D) across the cluster (and across all planes), while each peak retains its own
    /// direct-axis linewidth.
    SharedPerCluster,
    /// Each peak owns its own linewidths on every axis, shared identically across all planes of
    /// a pseudo-3D series (recommended default for $R_1$/$R_2$/CPMG/CEST relaxation series; on a
    /// single plane this assigns one linewidth per peak per axis).
    SharedAcrossPlanes,
}

/// Which side of a box constraint was hit by a nonlinear parameter.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundSide {
    Lower,
    Upper,
}

/// Record of a nonlinear parameter that converged onto an active box bound.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ActiveBound {
    /// Index in the nonlinear parameter vector $\boldsymbol{\theta}$.
    pub param_index: usize,
    /// Human-readable parameter name (e.g. `"peak[0].center_f1"`, `"peak[0].fwhm_f2"`,
    /// `"cluster.fwhm_f1"`, `"peak[0].eta_f1"`).
    pub param_name: String,
    /// Which box bound was reached (`Lower` or `Upper`).
    pub side: BoundSide,
    /// Final parameter value at the bound.
    pub value: f64,
    /// Box bound limit value.
    pub limit: f64,
}

/// Fail-closed diagnostic status returned by deconvolution solvers.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeconvDiagnostic {
    /// Optimizer converged cleanly with well-conditioned basis, interior parameters,
    /// distinct peak centers, and acceptable goodness-of-fit.
    ///
    /// Note: goodness-of-fit ([`DeconvDiagnostic::ModelMisspecified`]) can only be tested when
    /// `DeconvConfig::noise_sigma` is `Some(sigma)`. When `noise_sigma` is `None` (unknown $\sigma$),
    /// $\hat{\sigma}^2 = \mathrm{RSS}/\nu$ is estimated from the residuals so `reduced_chi2` is
    /// identically `1.0`, and an interior, well-conditioned, converged fit to a misspecified model
    /// will still report `Converged`.
    Converged,
    /// Basis matrix $\Phi$ or full joint Jacobian exceeded the condition-number guardrail.
    IllConditionedCluster,
    /// Optimizer reached `max_iterations` without meeting convergence tolerances.
    NonConverged,
    /// Two or more peaks lie closer than `degenerate_separation_frac` times their mean FWHM.
    DegeneratePeakPair,
    /// One or more nonlinear parameters converged onto an active box bound (e.g.
    /// `max_center_shift_fwhm`, min/max FWHM, ROI boundary, or pseudo-Voigt $\eta \in \{0, 1\}$).
    /// Standard unconstrained Cramér–Rao regularity does not hold on a parameter boundary.
    ActiveBoundHit,
    /// Known-$\sigma$ reduced chi-squared $\chi^2_\nu$ exceeded the goodness-of-fit threshold
    /// (default $\chi^2_{0.999}(\nu)/\nu$), indicating model misspecification.
    ///
    /// Only emitted when `DeconvConfig::noise_sigma` is `Some(sigma)`; with unknown $\sigma$
    /// (`noise_sigma == None`), `reduced_chi2` is `1.0` by construction and this gate cannot fire.
    ModelMisspecified { reduced_chi2: f64 },
}

/// Validation and numerical guardrail errors for N-D lineshape deconvolution.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeconvError {
    EmptyRoi,
    EmptyPeaks,
    EmptyPlanes,
    DimensionMismatch(String),
    OversizedRoi {
        voxels: usize,
        max_voxels: usize,
    },
    TooManyPeaks {
        peaks: usize,
        max_peaks: usize,
    },
    TooManyPlanes {
        planes: usize,
        max_planes: usize,
    },
    MemoryBudgetExceeded {
        matrix: String,
        elements: usize,
        max_elements: usize,
    },
    NonFiniteInput(String),
    InvalidParameter(String),
    ConstantOrZeroPatch,
    InsufficientDegreesOfFreedom {
        observations: usize,
        parameters: usize,
    },
    DegenerateAxis(String),
}

impl fmt::Display for DeconvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyRoi => write!(f, "deconvolution ROI is empty"),
            Self::EmptyPeaks => write!(f, "peak seed list is empty"),
            Self::EmptyPlanes => write!(f, "plane series contains zero planes"),
            Self::DimensionMismatch(msg) => write!(f, "ROI dimension mismatch: {msg}"),
            Self::OversizedRoi { voxels, max_voxels } => {
                write!(f, "ROI has {voxels} voxels, exceeding cap {max_voxels}")
            }
            Self::TooManyPeaks { peaks, max_peaks } => {
                write!(f, "cluster has {peaks} peaks, exceeding cap {max_peaks}")
            }
            Self::TooManyPlanes { planes, max_planes } => {
                write!(f, "series has {planes} planes, exceeding cap {max_planes}")
            }
            Self::MemoryBudgetExceeded {
                matrix,
                elements,
                max_elements,
            } => write!(
                f,
                "deconvolution {matrix} requires {elements} elements, exceeding memory budget cap {max_elements}"
            ),
            Self::NonFiniteInput(msg) => write!(f, "non-finite input: {msg}"),
            Self::InvalidParameter(msg) => write!(f, "invalid parameter: {msg}"),
            Self::ConstantOrZeroPatch => {
                write!(f, "ROI patch is constant or identically zero")
            }
            Self::InsufficientDegreesOfFreedom {
                observations,
                parameters,
            } => write!(
                f,
                "insufficient degrees of freedom: {observations} observations for {parameters} parameters"
            ),
            Self::DegenerateAxis(msg) => write!(f, "degenerate axis coordinates: {msg}"),
        }
    }
}

impl std::error::Error for DeconvError {}

/// Configuration for coupled 2D/3D Variable Projection deconvolution.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DeconvConfig {
    /// Separable lineshape model (`Lorentzian`, `Gaussian`, or `PseudoVoigt`).
    pub lineshape: LineshapeKind,
    /// Linewidth sharing policy across peaks and planes.
    pub linewidth_sharing: LinewidthSharing,
    /// Fit the pseudo-Voigt mixing parameter $\eta \in [0, 1]$ (ignored for pure Lorentzian/Gaussian).
    pub fit_mixing: bool,
    /// Include an additive constant offset per plane (flat scalar baseline, not a sloped/linear
    /// baseline) in the VarPro linear projection.
    pub fit_baseline: bool,
    /// Maximum Levenberg–Marquardt iterations.
    pub max_iterations: usize,
    /// Relative parameter step convergence tolerance.
    pub param_tol: f64,
    /// Relative residual sum-of-squares change convergence tolerance.
    pub cost_tol: f64,
    /// Scaled projected gradient norm convergence tolerance.
    pub grad_tol: f64,
    /// Initial Marquardt damping factor $\lambda_0$.
    pub initial_lambda: f64,
    /// Relative singular value cutoff for SVD pseudoinverse regularization.
    pub svd_rcond: f64,
    /// Condition-number threshold for [`DeconvDiagnostic::IllConditionedCluster`].
    pub ill_conditioned_kappa: f64,
    /// Normalized pairwise center separation threshold (in units of mean FWHM) for
    /// [`DeconvDiagnostic::DegeneratePeakPair`].
    pub degenerate_separation_frac: f64,
    /// Optional known voxel noise standard deviation $\sigma$. When `Some(sigma)`, used to compute
    /// exact theoretical Cramér–Rao covariance $\sigma^2 (J_{\text{full}}^T J_{\text{full}})^{-1}$
    /// and reduced $\chi^2_\nu$ (enabling the [`DeconvDiagnostic::ModelMisspecified`] gate).
    /// When `None`, $\hat{\sigma}^2 = \mathrm{RSS}/\nu$ is estimated from the fit residuals so
    /// `reduced_chi2` is identically `1.0`, the [`DeconvDiagnostic::ModelMisspecified`] gate
    /// cannot fire, and an interior, well-conditioned misspecified fit still reports
    /// [`DeconvDiagnostic::Converged`].
    pub noise_sigma: Option<f64>,
    /// Optional explicit upper threshold on reduced $\chi^2_\nu$ when `noise_sigma` is `Some(_)`.
    /// When `None` (default), uses the exact upper-tail critical value
    /// $\chi^2_{0.999}(\nu) / \nu$ (`p = 1e-3`).
    pub max_reduced_chi2: Option<f64>,
    /// Minimum allowed fitted FWHM as a multiple of the axis grid spacing.
    pub min_fwhm_grid_frac: f64,
    /// Maximum allowed fitted FWHM as a multiple of the axis ROI span.
    pub max_fwhm_span_frac: f64,
    /// Optional maximum center shift from seed in multiples of the seeded FWHM.
    pub max_center_shift_fwhm: Option<f64>,
}

impl Default for DeconvConfig {
    fn default() -> Self {
        Self {
            lineshape: LineshapeKind::PseudoVoigt,
            linewidth_sharing: LinewidthSharing::SharedAcrossPlanes,
            fit_mixing: true,
            fit_baseline: false,
            max_iterations: 200,
            param_tol: 1.0e-7,
            cost_tol: 1.0e-10,
            grad_tol: 1.0e-8,
            initial_lambda: 1.0e-3,
            svd_rcond: 1.0e-12,
            ill_conditioned_kappa: DEFAULT_ILL_CONDITIONED_KAPPA,
            degenerate_separation_frac: DEFAULT_DEGENERATE_SEPARATION_FRAC,
            noise_sigma: None,
            max_reduced_chi2: None,
            min_fwhm_grid_frac: 0.25,
            max_fwhm_span_frac: 2.5,
            max_center_shift_fwhm: None,
        }
    }
}

// ============================================================================
// 1D Lineshape Evaluation & Thompson–Cox–Hastings Pseudo-Voigt
// ============================================================================

/// Value and exact analytical partial derivatives of a unit-height 1D lineshape.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct LineshapeEval {
    /// Unit-peak-height lineshape value $f(x; \mu, \gamma, \eta) \in [0, 1]$.
    pub value: f64,
    /// Exact partial derivative $\partial f / \partial \mu$.
    pub d_mu: f64,
    /// Exact partial derivative $\partial f / \partial \gamma$ (FWHM).
    pub d_fwhm: f64,
    /// Exact partial derivative $\partial f / \partial \eta$ (`0.0` for pure Lorentzian/Gaussian).
    pub d_eta: f64,
}

/// Evaluate a unit-height 1D Lorentzian, Gaussian, or pseudo-Voigt lineshape and its exact
/// closed-form analytical derivatives with respect to center `mu`, FWHM `fwhm`, and mixing `eta`.
#[inline]
pub fn eval_lineshape_1d(
    kind: LineshapeKind,
    x: f64,
    mu: f64,
    fwhm: f64,
    eta: f64,
) -> LineshapeEval {
    let inv_gamma = 1.0 / fwhm;
    let u = 2.0 * (x - mu) * inv_gamma;
    let u2 = u * u;

    match kind {
        LineshapeKind::Lorentzian => {
            let l = 1.0 / (1.0 + u2);
            let l2 = l * l;
            LineshapeEval {
                value: l,
                d_mu: 4.0 * u * l2 * inv_gamma,
                d_fwhm: 2.0 * u2 * l2 * inv_gamma,
                d_eta: 0.0,
            }
        }
        LineshapeKind::Gaussian => {
            let g = (-LN_2 * u2).exp();
            LineshapeEval {
                value: g,
                d_mu: 4.0 * LN_2 * u * g * inv_gamma,
                d_fwhm: 2.0 * LN_2 * u2 * g * inv_gamma,
                d_eta: 0.0,
            }
        }
        LineshapeKind::PseudoVoigt => {
            let eta_c = eta.clamp(0.0, 1.0);
            let l = 1.0 / (1.0 + u2);
            let l2 = l * l;
            let dl_dmu = 4.0 * u * l2 * inv_gamma;
            let dl_dgamma = 2.0 * u2 * l2 * inv_gamma;

            let g = (-LN_2 * u2).exp();
            let dg_dmu = 4.0 * LN_2 * u * g * inv_gamma;
            let dg_dgamma = 2.0 * LN_2 * u2 * g * inv_gamma;

            let om = 1.0 - eta_c;
            LineshapeEval {
                value: eta_c * l + om * g,
                d_mu: eta_c * dl_dmu + om * dg_dmu,
                d_fwhm: eta_c * dl_dgamma + om * dg_dgamma,
                d_eta: l - g,
            }
        }
    }
}

/// Analytical 1D integral $\int_{-\infty}^{\infty} f(x; \mu, \gamma, \eta)\,dx$ of a unit-height
/// lineshape.
#[inline]
pub fn lineshape_area_1d(kind: LineshapeKind, fwhm: f64, eta: f64) -> f64 {
    let lor_coeff = 0.5 * PI;
    let gau_coeff = 0.5 * (PI / LN_2).sqrt();
    match kind {
        LineshapeKind::Lorentzian => fwhm * lor_coeff,
        LineshapeKind::Gaussian => fwhm * gau_coeff,
        LineshapeKind::PseudoVoigt => {
            let eta_c = eta.clamp(0.0, 1.0);
            fwhm * (eta_c * lor_coeff + (1.0 - eta_c) * gau_coeff)
        }
    }
}

/// Thompson–Cox–Hastings (*J. Appl. Crystallogr.* 1987, 20, 79–83) mapping from Gaussian FWHM
/// $\gamma_G \ge 0$ and Lorentzian FWHM $\gamma_L \ge 0$ to Voigt total FWHM $\gamma_V$ and
/// pseudo-Voigt mixing parameter $\eta \in [0, 1]$.
pub fn thompson_cox_hastings_voigt(fwhm_gaussian: f64, fwhm_lorentzian: f64) -> (f64, f64) {
    let fg = fwhm_gaussian.max(0.0);
    let fl = fwhm_lorentzian.max(0.0);
    if fg == 0.0 && fl == 0.0 {
        return (0.0, 0.0);
    }
    let fg2 = fg * fg;
    let fg3 = fg2 * fg;
    let fg4 = fg2 * fg2;
    let fg5 = fg4 * fg;
    let fl2 = fl * fl;
    let fl3 = fl2 * fl;
    let fl4 = fl2 * fl2;
    let fl5 = fl4 * fl;

    let poly = fg5
        + 2.69269 * fg4 * fl
        + 2.42843 * fg3 * fl2
        + 4.47163 * fg2 * fl3
        + 0.07842 * fg * fl4
        + fl5;
    let fv = poly.powf(0.2);
    if fv <= 0.0 {
        return (0.0, 0.0);
    }
    let r = fl / fv;
    let eta = (1.36603 * r - 0.47719 * r * r + 0.11116 * r * r * r).clamp(0.0, 1.0);
    (fv, eta)
}

// ============================================================================
// ROI, Seed, Model & Result Structs (2D, 3D, Pseudo-3D Plane Series)
// ============================================================================

/// 2D ROI patch stored row-major as `values[i1 * n2 + i2]` over `(F1, F2)`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Roi2D {
    /// Indirect axis (`F1`, e.g. $^{15}\text{N}$ or $^{13}\text{C}$) coordinates (length `n1`).
    pub coords_f1: Vec<f64>,
    /// Direct axis (`F2`, e.g. $^1\text{H}$) coordinates (length `n2`).
    pub coords_f2: Vec<f64>,
    /// Voxel intensities in row-major order `values[i1 * coords_f2.len() + i2]`.
    pub values: Vec<f64>,
}

impl Roi2D {
    /// Construct a 2D ROI on integer grid coordinates `0..n1` and `0..n2`.
    pub fn from_grid(n1: usize, n2: usize, values: Vec<f64>) -> Result<Self, DeconvError> {
        let voxels = checked_voxel_count(&[n1, n2])?;
        if values.len() != voxels {
            return Err(DeconvError::DimensionMismatch(format!(
                "expected {voxels} values for {n1}x{n2} grid, got {}",
                values.len()
            )));
        }
        Ok(Self {
            coords_f1: (0..n1).map(|i| i as f64).collect(),
            coords_f2: (0..n2).map(|i| i as f64).collect(),
            values,
        })
    }

    /// Extract a 2D ROI patch from a [`Spectrum2D`] (`F1 = axis_y`, `F2 = axis_x`) covering the
    /// inclusive ppm intervals `f1_ppm_range` and `f2_ppm_range`.
    pub fn from_spectrum_window(
        spec: &Spectrum2D,
        f1_ppm_range: (f64, f64),
        f2_ppm_range: (f64, f64),
    ) -> Result<Self, DeconvError> {
        let (y0, y1) = axis_index_bounds(&spec.axis_y, f1_ppm_range)?;
        let (x0, x1) = axis_index_bounds(&spec.axis_x, f2_ppm_range)?;
        let n1 = y1 - y0 + 1;
        let n2 = x1 - x0 + 1;
        let voxels = checked_voxel_count(&[n1, n2])?;
        let mut coords_f1 = Vec::with_capacity(n1);
        for iy in y0..=y1 {
            coords_f1.push(spec.axis_y.index_to_ppm(iy as f64));
        }
        let mut coords_f2 = Vec::with_capacity(n2);
        for ix in x0..=x1 {
            coords_f2.push(spec.axis_x.index_to_ppm(ix as f64));
        }
        let mut values = Vec::with_capacity(voxels);
        for iy in y0..=y1 {
            for ix in x0..=x1 {
                let idx = iy * spec.nx() + ix;
                let v = *spec.data.get(idx).ok_or_else(|| {
                    DeconvError::DimensionMismatch("spectrum data truncated".to_string())
                })?;
                values.push(v as f64);
            }
        }
        Ok(Self {
            coords_f1,
            coords_f2,
            values,
        })
    }

    #[inline]
    pub fn n1(&self) -> usize {
        self.coords_f1.len()
    }

    #[inline]
    pub fn n2(&self) -> usize {
        self.coords_f2.len()
    }
}

/// 3D ROI patch stored row-major as `values[(i1 * n2 + i2) * n3 + i3]` over `(F1, F2, F3)`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Roi3D {
    /// First indirect axis (`F1`, outermost, e.g. `axis_z`).
    pub coords_f1: Vec<f64>,
    /// Second indirect axis (`F2`, middle, e.g. `axis_y`).
    pub coords_f2: Vec<f64>,
    /// Direct axis (`F3`, fastest, e.g. `axis_x`).
    pub coords_f3: Vec<f64>,
    /// Voxel intensities in row-major order `values[(i1 * n2 + i2) * n3 + i3]`.
    pub values: Vec<f64>,
}

impl Roi3D {
    /// Construct a 3D ROI on integer grid coordinates `0..n1`, `0..n2`, `0..n3`.
    pub fn from_grid(
        n1: usize,
        n2: usize,
        n3: usize,
        values: Vec<f64>,
    ) -> Result<Self, DeconvError> {
        let voxels = checked_voxel_count(&[n1, n2, n3])?;
        if values.len() != voxels {
            return Err(DeconvError::DimensionMismatch(format!(
                "expected {voxels} values for {n1}x{n2}x{n3} grid, got {}",
                values.len()
            )));
        }
        Ok(Self {
            coords_f1: (0..n1).map(|i| i as f64).collect(),
            coords_f2: (0..n2).map(|i| i as f64).collect(),
            coords_f3: (0..n3).map(|i| i as f64).collect(),
            values,
        })
    }

    /// Extract a 3D ROI patch from a [`Spectrum3D`] (`F1 = axis_z`, `F2 = axis_y`, `F3 = axis_x`).
    pub fn from_spectrum_window(
        spec: &Spectrum3D,
        f1_ppm_range: (f64, f64),
        f2_ppm_range: (f64, f64),
        f3_ppm_range: (f64, f64),
    ) -> Result<Self, DeconvError> {
        let (z0, z1) = axis_index_bounds(&spec.axis_z, f1_ppm_range)?;
        let (y0, y1) = axis_index_bounds(&spec.axis_y, f2_ppm_range)?;
        let (x0, x1) = axis_index_bounds(&spec.axis_x, f3_ppm_range)?;
        let n1 = z1 - z0 + 1;
        let n2 = y1 - y0 + 1;
        let n3 = x1 - x0 + 1;
        let voxels = checked_voxel_count(&[n1, n2, n3])?;
        let coords_f1: Vec<f64> = (z0..=z1)
            .map(|iz| spec.axis_z.index_to_ppm(iz as f64))
            .collect();
        let coords_f2: Vec<f64> = (y0..=y1)
            .map(|iy| spec.axis_y.index_to_ppm(iy as f64))
            .collect();
        let coords_f3: Vec<f64> = (x0..=x1)
            .map(|ix| spec.axis_x.index_to_ppm(ix as f64))
            .collect();
        let mut values = Vec::with_capacity(voxels);
        let nx = spec.nx();
        let ny = spec.ny();
        for iz in z0..=z1 {
            for iy in y0..=y1 {
                for ix in x0..=x1 {
                    let idx = (iz * ny + iy) * nx + ix;
                    let v = *spec.data.get(idx).ok_or_else(|| {
                        DeconvError::DimensionMismatch("3D spectrum data truncated".to_string())
                    })?;
                    values.push(v as f64);
                }
            }
        }
        Ok(Self {
            coords_f1,
            coords_f2,
            coords_f3,
            values,
        })
    }
}

/// Coupled pseudo-3D plane series (e.g. $R_1$, $R_2$, CPMG, CEST, or titration planes) over a
/// shared 2D `(F1, F2)` ROI grid.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlaneSeries2D {
    /// Indirect axis (`F1`) coordinates (length `n1`).
    pub coords_f1: Vec<f64>,
    /// Direct axis (`F2`) coordinates (length `n2`).
    pub coords_f2: Vec<f64>,
    /// Per-plane voxel intensities; each inner `Vec<f64>` has length `n1 * n2` in row-major order.
    pub plane_values: Vec<Vec<f64>>,
    /// Optional plane coordinate (e.g. relaxation delay in seconds or titration point); if empty,
    /// defaults to `0.0, 1.0, ...` in downstream metadata.
    pub plane_coords: Vec<f64>,
}

/// Initial nonlinear shape seed for a 2D peak (linear amplitude is solved analytically by VarPro).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PeakSeed2D {
    pub center_f1: f64,
    pub center_f2: f64,
    pub fwhm_f1: f64,
    pub fwhm_f2: f64,
    pub eta_f1: f64,
    pub eta_f2: f64,
}

/// Full 2D peak model (amplitude + shape) for forward synthesis.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PeakModel2D {
    pub amplitude: f64,
    pub center_f1: f64,
    pub center_f2: f64,
    pub fwhm_f1: f64,
    pub fwhm_f2: f64,
    pub eta_f1: f64,
    pub eta_f2: f64,
}

/// Fitted 2D peak parameters and Cramér–Rao standard errors (`None` when a parameter is fixed,
/// pinned on an active box bound, or when the overall fit fails CRLB regularity).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct FittedPeak2D {
    pub amplitude: f64,
    pub amplitude_se: Option<f64>,
    pub volume: f64,
    pub volume_se: Option<f64>,
    pub center_f1: f64,
    pub center_f1_se: Option<f64>,
    pub center_f2: f64,
    pub center_f2_se: Option<f64>,
    pub fwhm_f1: f64,
    pub fwhm_f1_se: Option<f64>,
    pub fwhm_f2: f64,
    pub fwhm_f2_se: Option<f64>,
    pub eta_f1: f64,
    pub eta_f1_se: Option<f64>,
    pub eta_f2: f64,
    pub eta_f2_se: Option<f64>,
}

/// Initial nonlinear shape seed for a 3D peak.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PeakSeed3D {
    pub center_f1: f64,
    pub center_f2: f64,
    pub center_f3: f64,
    pub fwhm_f1: f64,
    pub fwhm_f2: f64,
    pub fwhm_f3: f64,
    pub eta_f1: f64,
    pub eta_f2: f64,
    pub eta_f3: f64,
}

/// Full 3D peak model (amplitude + shape) for forward synthesis.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PeakModel3D {
    pub amplitude: f64,
    pub center_f1: f64,
    pub center_f2: f64,
    pub center_f3: f64,
    pub fwhm_f1: f64,
    pub fwhm_f2: f64,
    pub fwhm_f3: f64,
    pub eta_f1: f64,
    pub eta_f2: f64,
    pub eta_f3: f64,
}

/// Fitted 3D peak parameters and Cramér–Rao standard errors (`None` when a parameter is fixed,
/// pinned on an active box bound, or when the overall fit fails CRLB regularity).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct FittedPeak3D {
    pub amplitude: f64,
    pub amplitude_se: Option<f64>,
    pub volume: f64,
    pub volume_se: Option<f64>,
    pub center_f1: f64,
    pub center_f1_se: Option<f64>,
    pub center_f2: f64,
    pub center_f2_se: Option<f64>,
    pub center_f3: f64,
    pub center_f3_se: Option<f64>,
    pub fwhm_f1: f64,
    pub fwhm_f1_se: Option<f64>,
    pub fwhm_f2: f64,
    pub fwhm_f2_se: Option<f64>,
    pub fwhm_f3: f64,
    pub fwhm_f3_se: Option<f64>,
    pub eta_f1: f64,
    pub eta_f1_se: Option<f64>,
    pub eta_f2: f64,
    pub eta_f2_se: Option<f64>,
    pub eta_f3: f64,
    pub eta_f3_se: Option<f64>,
}

/// Result of 2D VarPro deconvolution with exact full Cramér–Rao covariance and diagnostics.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DeconvResult2D {
    pub peaks: Vec<FittedPeak2D>,
    /// Fitted additive constant offset for the plane (`0.0` when `fit_baseline == false`).
    pub baseline: f64,
    pub baseline_se: Option<f64>,
    /// Peak amplitude covariance submatrix ($K \times K$) extracted from the full joint
    /// $(J_{\text{full}}^T J_{\text{full}})^{-1} \hat{\sigma}^2$ matrix, or `None` when
    /// `crlb_valid == false` so invalid Fisher inverses are never exposed.
    pub amplitude_covariance: Option<Vec<Vec<f64>>>,
    /// Peak amplitude overlap correlation matrix ($K \times K$, entries in $[-1, 1]$), or `None`
    /// when `crlb_valid == false`.
    pub amplitude_correlation: Option<Vec<Vec<f64>>>,
    /// Full joint covariance matrix over `[linear_params..., nonlinear_params...]`, or `None` when
    /// `crlb_valid == false`.
    pub full_covariance: Option<Vec<Vec<f64>>>,
    pub rss: f64,
    pub dof: usize,
    pub residual_sigma: f64,
    /// Reduced chi-squared $\chi^2_\nu = \mathrm{RSS}/(\nu \sigma^2)$ when `noise_sigma` is
    /// `Some(sigma)`, or identically `1.0` when `noise_sigma` is `None` (unknown $\sigma$, in
    /// which case model misspecification cannot be detected from $\chi^2_\nu$ and an interior fit
    /// still reports [`DeconvDiagnostic::Converged`]).
    pub reduced_chi2: f64,
    pub basis_condition_number: f64,
    pub jacobian_condition_number: f64,
    pub min_pair_separation_fwhm: f64,
    pub degenerate_pairs: Vec<(usize, usize)>,
    /// Nonlinear parameters sitting on an active box bound at the solution.
    pub active_bounds: Vec<ActiveBound>,
    /// `true` iff the unconstrained full Cramér–Rao covariance satisfies regularity conditions
    /// (converged, interior optimum with no active bounds, well-conditioned, and not misspecified).
    /// When `false`, `amplitude_covariance`, `amplitude_correlation`, and `full_covariance` are
    /// suppressed to `None`.
    pub crlb_valid: bool,
    pub iterations: usize,
    pub converged: bool,
    pub diagnostic: DeconvDiagnostic,
    pub all_diagnostics: Vec<DeconvDiagnostic>,
    pub fitted_values: Vec<f64>,
    pub residuals: Vec<f64>,
}

/// Result of 3D VarPro deconvolution with exact full Cramér–Rao covariance and diagnostics.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DeconvResult3D {
    pub peaks: Vec<FittedPeak3D>,
    /// Fitted additive constant offset for the 3D ROI (`0.0` when `fit_baseline == false`).
    pub baseline: f64,
    pub baseline_se: Option<f64>,
    /// Peak amplitude covariance submatrix ($K \times K$), or `None` when `crlb_valid == false`.
    pub amplitude_covariance: Option<Vec<Vec<f64>>>,
    /// Peak amplitude overlap correlation matrix ($K \times K$, entries in $[-1, 1]$), or `None`
    /// when `crlb_valid == false`.
    pub amplitude_correlation: Option<Vec<Vec<f64>>>,
    /// Full joint covariance matrix over `[linear_params..., nonlinear_params...]`, or `None` when
    /// `crlb_valid == false`.
    pub full_covariance: Option<Vec<Vec<f64>>>,
    pub rss: f64,
    pub dof: usize,
    pub residual_sigma: f64,
    /// Reduced chi-squared $\chi^2_\nu = \mathrm{RSS}/(\nu \sigma^2)$ when `noise_sigma` is
    /// `Some(sigma)`, or identically `1.0` when `noise_sigma` is `None` (unknown $\sigma$, in
    /// which case model misspecification cannot be detected from $\chi^2_\nu$ and an interior fit
    /// still reports [`DeconvDiagnostic::Converged`]).
    pub reduced_chi2: f64,
    pub basis_condition_number: f64,
    pub jacobian_condition_number: f64,
    pub min_pair_separation_fwhm: f64,
    pub degenerate_pairs: Vec<(usize, usize)>,
    /// Nonlinear parameters sitting on an active box bound at the solution.
    pub active_bounds: Vec<ActiveBound>,
    /// `true` iff the unconstrained full Cramér–Rao covariance satisfies regularity conditions.
    /// When `false`, `amplitude_covariance`, `amplitude_correlation`, and `full_covariance` are
    /// suppressed to `None`.
    pub crlb_valid: bool,
    pub iterations: usize,
    pub converged: bool,
    pub diagnostic: DeconvDiagnostic,
    pub all_diagnostics: Vec<DeconvDiagnostic>,
    pub fitted_values: Vec<f64>,
    pub residuals: Vec<f64>,
}

/// Per-peak fitted trajectories across a coupled pseudo-3D 2D plane series.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlanePeakSeries {
    pub center_f1: f64,
    pub center_f1_se: Option<f64>,
    pub center_f2: f64,
    pub center_f2_se: Option<f64>,
    /// Per-plane FWHM on `F1` (length `P`; identical across planes under `SharedAcrossPlanes` /
    /// `SharedPerCluster`).
    pub fwhm_f1: Vec<f64>,
    pub fwhm_f1_se: Vec<Option<f64>>,
    /// Per-plane FWHM on `F2` (length `P`).
    pub fwhm_f2: Vec<f64>,
    pub fwhm_f2_se: Vec<Option<f64>>,
    pub eta_f1: f64,
    pub eta_f1_se: Option<f64>,
    pub eta_f2: f64,
    pub eta_f2_se: Option<f64>,
    /// Linearly projected amplitudes across planes (length `P`).
    pub amplitudes: Vec<f64>,
    /// Exact coupled Cramér–Rao standard errors on amplitudes across planes (length `P`;
    /// `None` when fit fails CRLB regularity).
    pub amplitude_se: Vec<Option<f64>>,
    /// Integrated analytical peak volumes across planes (length `P`).
    pub volumes: Vec<f64>,
}

/// Result of coupled pseudo-3D plane-series 2D deconvolution.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlaneSeriesDeconvResult {
    pub peaks: Vec<PlanePeakSeries>,
    /// Fitted additive constant offset per plane (length `P`; all `0.0` when
    /// `fit_baseline == false`).
    pub baselines: Vec<f64>,
    pub baseline_se: Vec<Option<f64>>,
    /// Per-plane peak amplitude covariance matrices (`P` matrices of size $K \times K$), or `None`
    /// when `crlb_valid == false` so invalid Fisher inverses are never exposed.
    pub plane_amplitude_covariances: Option<Vec<Vec<Vec<f64>>>>,
    /// Per-plane peak amplitude correlation matrices (`P` matrices of size $K \times K$), or `None`
    /// when `crlb_valid == false`.
    pub plane_amplitude_correlations: Option<Vec<Vec<Vec<f64>>>>,
    pub rss: f64,
    pub plane_rss: Vec<f64>,
    pub dof: usize,
    pub residual_sigma: f64,
    /// Reduced chi-squared $\chi^2_\nu = \mathrm{RSS}/(\nu \sigma^2)$ when `noise_sigma` is
    /// `Some(sigma)`, or identically `1.0` when `noise_sigma` is `None` (unknown $\sigma$, in
    /// which case model misspecification cannot be detected from $\chi^2_\nu$ and an interior fit
    /// still reports [`DeconvDiagnostic::Converged`]).
    pub reduced_chi2: f64,
    pub basis_condition_number: f64,
    pub jacobian_condition_number: f64,
    pub min_pair_separation_fwhm: f64,
    pub degenerate_pairs: Vec<(usize, usize)>,
    /// Nonlinear parameters sitting on an active box bound at the solution.
    pub active_bounds: Vec<ActiveBound>,
    /// `true` iff the unconstrained full Cramér–Rao covariance satisfies regularity conditions.
    /// When `false`, `plane_amplitude_covariances` and `plane_amplitude_correlations` are
    /// suppressed to `None`.
    pub crlb_valid: bool,
    pub iterations: usize,
    pub converged: bool,
    pub diagnostic: DeconvDiagnostic,
    pub all_diagnostics: Vec<DeconvDiagnostic>,
}

// ============================================================================
// Forward Synthesis Helpers
// ============================================================================

/// Synthesize a noiseless 2D ROI patch from exact separable lineshapes.
pub fn synthesize_roi_2d(
    coords_f1: &[f64],
    coords_f2: &[f64],
    kind: LineshapeKind,
    peaks: &[PeakModel2D],
    baseline: f64,
) -> Result<Roi2D, DeconvError> {
    let voxels = checked_voxel_count(&[coords_f1.len(), coords_f2.len()])?;
    validate_axis("F1", coords_f1)?;
    validate_axis("F2", coords_f2)?;
    if !baseline.is_finite() {
        return Err(DeconvError::NonFiniteInput("baseline".to_string()));
    }
    let n2 = coords_f2.len();
    let mut values = vec![baseline; voxels];
    for (pk_idx, pk) in peaks.iter().enumerate() {
        if !pk.amplitude.is_finite()
            || !pk.center_f1.is_finite()
            || !pk.center_f2.is_finite()
            || !pk.fwhm_f1.is_finite()
            || !pk.fwhm_f2.is_finite()
            || !pk.eta_f1.is_finite()
            || !pk.eta_f2.is_finite()
        {
            return Err(DeconvError::NonFiniteInput(format!("peak {pk_idx}")));
        }
        if pk.fwhm_f1 <= 0.0 || pk.fwhm_f2 <= 0.0 {
            return Err(DeconvError::InvalidParameter(format!(
                "peak {pk_idx} has non-positive FWHM"
            )));
        }
        let prof1: Vec<f64> = coords_f1
            .iter()
            .map(|&x| eval_lineshape_1d(kind, x, pk.center_f1, pk.fwhm_f1, pk.eta_f1).value)
            .collect();
        let prof2: Vec<f64> = coords_f2
            .iter()
            .map(|&x| eval_lineshape_1d(kind, x, pk.center_f2, pk.fwhm_f2, pk.eta_f2).value)
            .collect();
        for (i1, &p1) in prof1.iter().enumerate() {
            let a1 = pk.amplitude * p1;
            let row = i1 * n2;
            for (i2, &p2) in prof2.iter().enumerate() {
                values[row + i2] += a1 * p2;
            }
        }
    }
    Ok(Roi2D {
        coords_f1: coords_f1.to_vec(),
        coords_f2: coords_f2.to_vec(),
        values,
    })
}

/// Synthesize a noiseless 3D ROI patch from exact separable lineshapes.
pub fn synthesize_roi_3d(
    coords_f1: &[f64],
    coords_f2: &[f64],
    coords_f3: &[f64],
    kind: LineshapeKind,
    peaks: &[PeakModel3D],
    baseline: f64,
) -> Result<Roi3D, DeconvError> {
    let voxels = checked_voxel_count(&[coords_f1.len(), coords_f2.len(), coords_f3.len()])?;
    validate_axis("F1", coords_f1)?;
    validate_axis("F2", coords_f2)?;
    validate_axis("F3", coords_f3)?;
    if !baseline.is_finite() {
        return Err(DeconvError::NonFiniteInput("baseline".to_string()));
    }
    let n2 = coords_f2.len();
    let n3 = coords_f3.len();
    let mut values = vec![baseline; voxels];
    for (pk_idx, pk) in peaks.iter().enumerate() {
        if !pk.amplitude.is_finite()
            || !pk.center_f1.is_finite()
            || !pk.center_f2.is_finite()
            || !pk.center_f3.is_finite()
            || !pk.fwhm_f1.is_finite()
            || !pk.fwhm_f2.is_finite()
            || !pk.fwhm_f3.is_finite()
            || !pk.eta_f1.is_finite()
            || !pk.eta_f2.is_finite()
            || !pk.eta_f3.is_finite()
        {
            return Err(DeconvError::NonFiniteInput(format!("peak {pk_idx}")));
        }
        if pk.fwhm_f1 <= 0.0 || pk.fwhm_f2 <= 0.0 || pk.fwhm_f3 <= 0.0 {
            return Err(DeconvError::InvalidParameter(format!(
                "peak {pk_idx} has non-positive FWHM"
            )));
        }
        let prof1: Vec<f64> = coords_f1
            .iter()
            .map(|&x| eval_lineshape_1d(kind, x, pk.center_f1, pk.fwhm_f1, pk.eta_f1).value)
            .collect();
        let prof2: Vec<f64> = coords_f2
            .iter()
            .map(|&x| eval_lineshape_1d(kind, x, pk.center_f2, pk.fwhm_f2, pk.eta_f2).value)
            .collect();
        let prof3: Vec<f64> = coords_f3
            .iter()
            .map(|&x| eval_lineshape_1d(kind, x, pk.center_f3, pk.fwhm_f3, pk.eta_f3).value)
            .collect();
        for (i1, &p1) in prof1.iter().enumerate() {
            let a1 = pk.amplitude * p1;
            for (i2, &p2) in prof2.iter().enumerate() {
                let a12 = a1 * p2;
                let base = (i1 * n2 + i2) * n3;
                for (i3, &p3) in prof3.iter().enumerate() {
                    values[base + i3] += a12 * p3;
                }
            }
        }
    }
    Ok(Roi3D {
        coords_f1: coords_f1.to_vec(),
        coords_f2: coords_f2.to_vec(),
        coords_f3: coords_f3.to_vec(),
        values,
    })
}

// ============================================================================
// Input Validation Helpers
// ============================================================================

fn checked_voxel_count(dims: &[usize]) -> Result<usize, DeconvError> {
    if dims.is_empty() {
        return Err(DeconvError::EmptyRoi);
    }
    let mut total: usize = 1;
    for &d in dims {
        if d == 0 {
            return Err(DeconvError::EmptyRoi);
        }
        total = total.checked_mul(d).ok_or(DeconvError::OversizedRoi {
            voxels: usize::MAX,
            max_voxels: MAX_DECONV_VOXELS,
        })?;
    }
    if total > MAX_DECONV_VOXELS {
        return Err(DeconvError::OversizedRoi {
            voxels: total,
            max_voxels: MAX_DECONV_VOXELS,
        });
    }
    Ok(total)
}

fn axis_index_bounds(axis: &AxisPpm, ppm_range: (f64, f64)) -> Result<(usize, usize), DeconvError> {
    if axis.size == 0 || !axis.step_ppm.is_finite() || axis.step_ppm == 0.0 {
        return Err(DeconvError::DegenerateAxis(axis.label.clone()));
    }
    if !ppm_range.0.is_finite() || !ppm_range.1.is_finite() {
        return Err(DeconvError::NonFiniteInput(format!(
            "ppm range on {}",
            axis.label
        )));
    }
    let idx_a = axis.ppm_to_index(ppm_range.0);
    let idx_b = axis.ppm_to_index(ppm_range.1);
    let lo_f = idx_a.min(idx_b).floor();
    let hi_f = idx_a.max(idx_b).ceil();
    let max_idx = (axis.size - 1) as f64;
    if hi_f < 0.0 || lo_f > max_idx {
        return Err(DeconvError::EmptyRoi);
    }
    let lo = lo_f.clamp(0.0, max_idx) as usize;
    let hi = hi_f.clamp(0.0, max_idx) as usize;
    if hi < lo {
        return Err(DeconvError::EmptyRoi);
    }
    Ok((lo, hi))
}

#[derive(Clone, Copy, Debug)]
struct AxisBounds {
    min_coord: f64,
    max_coord: f64,
    span: f64,
    mean_step: f64,
}

fn validate_axis(name: &str, coords: &[f64]) -> Result<AxisBounds, DeconvError> {
    if coords.len() < 2 {
        return Err(DeconvError::DegenerateAxis(format!(
            "{name} requires at least 2 grid points, got {}",
            coords.len()
        )));
    }
    let mut min_c = f64::INFINITY;
    let mut max_c = f64::NEG_INFINITY;
    for &c in coords {
        if !c.is_finite() {
            return Err(DeconvError::NonFiniteInput(format!("{name} coordinate")));
        }
        if c < min_c {
            min_c = c;
        }
        if c > max_c {
            max_c = c;
        }
    }
    let span = max_c - min_c;
    if span <= 0.0 {
        return Err(DeconvError::DegenerateAxis(format!(
            "{name} has zero coordinate span"
        )));
    }
    let mut step_sum = 0.0;
    let first_diff = coords[1] - coords[0];
    if first_diff == 0.0 {
        return Err(DeconvError::DegenerateAxis(format!(
            "{name} has duplicate adjacent coordinates"
        )));
    }
    let sign = first_diff.signum();
    for w in coords.windows(2) {
        let diff = w[1] - w[0];
        if diff == 0.0 || diff.signum() != sign {
            return Err(DeconvError::DegenerateAxis(format!(
                "{name} coordinates must be strictly monotonic"
            )));
        }
        step_sum += diff.abs();
    }
    let mean_step = step_sum / ((coords.len() - 1) as f64);
    Ok(AxisBounds {
        min_coord: min_c,
        max_coord: max_c,
        span,
        mean_step,
    })
}

fn validate_config(cfg: &DeconvConfig) -> Result<(), DeconvError> {
    if cfg.max_iterations == 0 {
        return Err(DeconvError::InvalidParameter(
            "max_iterations must be >= 1".to_string(),
        ));
    }
    let finite_pos = [
        ("param_tol", cfg.param_tol),
        ("cost_tol", cfg.cost_tol),
        ("grad_tol", cfg.grad_tol),
        ("initial_lambda", cfg.initial_lambda),
        ("svd_rcond", cfg.svd_rcond),
        ("ill_conditioned_kappa", cfg.ill_conditioned_kappa),
        ("degenerate_separation_frac", cfg.degenerate_separation_frac),
        ("min_fwhm_grid_frac", cfg.min_fwhm_grid_frac),
        ("max_fwhm_span_frac", cfg.max_fwhm_span_frac),
    ];
    for (name, val) in finite_pos {
        if !val.is_finite() || val <= 0.0 {
            return Err(DeconvError::InvalidParameter(format!(
                "{name} must be finite and > 0"
            )));
        }
    }
    if let Some(sigma) = cfg.noise_sigma {
        if !sigma.is_finite() || sigma <= 0.0 {
            return Err(DeconvError::InvalidParameter(
                "noise_sigma must be finite and > 0".to_string(),
            ));
        }
    }
    if let Some(max_chi2) = cfg.max_reduced_chi2 {
        if !max_chi2.is_finite() || max_chi2 <= 0.0 {
            return Err(DeconvError::InvalidParameter(
                "max_reduced_chi2 must be finite and > 0".to_string(),
            ));
        }
    }
    if let Some(shift) = cfg.max_center_shift_fwhm {
        if !shift.is_finite() || shift <= 0.0 {
            return Err(DeconvError::InvalidParameter(
                "max_center_shift_fwhm must be finite and > 0".to_string(),
            ));
        }
    }
    Ok(())
}

/// Compute the exact nonlinear parameter count `n_nonlinear` before allocating any solver buffers.
fn expected_n_nonlinear(
    n_planes: usize,
    n_peaks: usize,
    n_dims: usize,
    cfg: &DeconvConfig,
) -> Option<usize> {
    let n_centers = n_peaks.checked_mul(n_dims)?;
    let n_fwhms = match cfg.linewidth_sharing {
        LinewidthSharing::SharedPerCluster => n_dims.checked_sub(1)?.checked_add(n_peaks)?,
        LinewidthSharing::SharedAcrossPlanes => n_peaks.checked_mul(n_dims)?,
        LinewidthSharing::Independent => n_planes.checked_mul(n_peaks)?.checked_mul(n_dims)?,
    };
    let n_etas = if cfg.lineshape == LineshapeKind::PseudoVoigt && cfg.fit_mixing {
        if cfg.linewidth_sharing == LinewidthSharing::SharedPerCluster {
            n_dims.checked_sub(1)?.checked_add(n_peaks)?
        } else {
            n_peaks.checked_mul(n_dims)?
        }
    } else {
        0
    };
    n_centers.checked_add(n_fwhms)?.checked_add(n_etas)
}

/// Fail-closed memory budget check on the VarPro Jacobian (`n_nl * total_obs`) and LM Hessian
/// (`n_nl * n_nl`) performed with checked arithmetic strictly before allocating solver matrices.
fn check_memory_budget(total_obs: usize, n_nl: usize) -> Result<(), DeconvError> {
    let jac_elems =
        n_nl.checked_mul(total_obs)
            .ok_or_else(|| DeconvError::MemoryBudgetExceeded {
                matrix: "varpro_jacobian".to_string(),
                elements: usize::MAX,
                max_elements: MAX_DECONV_JACOBIAN_ELEMS,
            })?;
    if jac_elems > MAX_DECONV_JACOBIAN_ELEMS {
        return Err(DeconvError::MemoryBudgetExceeded {
            matrix: "varpro_jacobian".to_string(),
            elements: jac_elems,
            max_elements: MAX_DECONV_JACOBIAN_ELEMS,
        });
    }
    let hess_elems = n_nl
        .checked_mul(n_nl)
        .ok_or_else(|| DeconvError::MemoryBudgetExceeded {
            matrix: "lm_hessian".to_string(),
            elements: usize::MAX,
            max_elements: MAX_DECONV_HESSIAN_ELEMS,
        })?;
    if hess_elems > MAX_DECONV_HESSIAN_ELEMS {
        return Err(DeconvError::MemoryBudgetExceeded {
            matrix: "lm_hessian".to_string(),
            elements: hess_elems,
            max_elements: MAX_DECONV_HESSIAN_ELEMS,
        });
    }
    Ok(())
}

/// Lanczos approximation to $\ln \Gamma(z)$ ($g = 7, n = 9$, relative accuracy $< 10^{-14}$ for
/// $z \ge 0.5$).
fn ln_gamma(z: f64) -> f64 {
    const COEFFS: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    let x = z - 1.0;
    let mut acc = COEFFS[0];
    for (i, &c) in COEFFS.iter().enumerate().skip(1) {
        acc += c / (x + i as f64);
    }
    let t = x + 7.5;
    0.5 * (2.0 * PI).ln() + (x + 0.5) * t.ln() - t + acc.ln()
}

/// Regularized upper incomplete gamma function $Q(a, x) = \Gamma(a, x) / \Gamma(a)$ for $a > 0$,
/// $x \ge 0$ (Abramowitz & Stegun §6.5; power series for $x < a + 1$, modified Lentz continued
/// fraction for $x \ge a + 1$).
fn gamma_q(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    let gln = ln_gamma(a);
    let log_pref = -x + a * x.ln() - gln;
    if x < a + 1.0 {
        let mut ap = a;
        let mut sum = 1.0 / a;
        let mut del = sum;
        for _ in 0..1000 {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if del.abs() <= sum.abs() * 1.0e-15 {
                break;
            }
        }
        let p = sum * log_pref.exp();
        (1.0 - p).clamp(0.0, 1.0)
    } else {
        const TINY: f64 = 1.0e-300;
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / TINY;
        let mut d = if b.abs() < TINY { 1.0 / TINY } else { 1.0 / b };
        let mut h = d;
        for i in 1..=1000 {
            let an = -(i as f64) * ((i as f64) - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < TINY {
                d = TINY;
            }
            c = b + an / c;
            if c.abs() < TINY {
                c = TINY;
            }
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() <= 1.0e-15 {
                break;
            }
        }
        (log_pref.exp() * h).clamp(0.0, 1.0)
    }
}

/// Exact upper-tail critical reduced chi-squared value $\chi^2_{0.999}(\nu) / \nu$
/// ($P(\chi^2_\nu > \nu \cdot c) = 10^{-3}$) via Wilson–Hilferty (1931) initialization and
/// bracketed Newton–Raphson refinement on $\ln Q(\nu/2, x/2) = \ln(10^{-3})$.
pub fn chi2_reduced_critical_0999(dof: usize) -> f64 {
    let nu = dof.max(1);
    if nu == 2 {
        // Exact closed form: Q(1, z) = exp(-z) => z = -ln(alpha), chi2 = 2*z, reduced = z.
        return -DEFAULT_CHI2_TAIL_PROB.ln();
    }
    let df = nu as f64;
    // Standard normal 99.9% quantile: Phi^{-1}(0.999).
    const Z_0999: f64 = 3.090_232_306_167_813;
    let two_over_9df = 2.0 / (9.0 * df);
    let wh_term = (1.0 - two_over_9df + Z_0999 * two_over_9df.sqrt()).max(0.1);
    let wh_reduced = wh_term * wh_term * wh_term;
    if nu > 10_000 {
        // For dof > 10_000, Wilson–Hilferty is accurate to < 1e-6 relative error.
        return wh_reduced;
    }

    let a = 0.5 * df;
    let gln = ln_gamma(a);
    let target_ln = DEFAULT_CHI2_TAIL_PROB.ln();

    // Solve f(z) = ln Q(a, z) - ln(alpha) = 0 for z = chi2 / 2, starting from Wilson–Hilferty.
    let mut z = (a * wh_reduced).max(1.0e-6);
    let mut lo = (0.5 * z).max(1.0e-12);
    let mut hi = (2.0 * z).max(z + 20.0);
    while gamma_q(a, lo) < DEFAULT_CHI2_TAIL_PROB {
        lo *= 0.5;
        if lo <= 1.0e-15 {
            break;
        }
    }
    while gamma_q(a, hi) > DEFAULT_CHI2_TAIL_PROB {
        hi *= 1.5;
        if hi > 1.0e12 {
            break;
        }
    }

    for _ in 0..40 {
        let q = gamma_q(a, z);
        if q <= 0.0 {
            hi = z;
            z = 0.5 * (lo + hi);
            continue;
        }
        let f = q.ln() - target_ln;
        if f.abs() < 1.0e-12 {
            break;
        }
        if f > 0.0 {
            lo = z;
        } else {
            hi = z;
        }
        // d/dz [ln Q(a, z)] = -z^{a-1} e^{-z} / (Gamma(a) * Q(a, z))
        let ln_dq = (a - 1.0) * z.ln() - z - gln;
        let df_dz = -(ln_dq - q.ln()).exp();
        let cand = if df_dz.is_finite() && df_dz.abs() > 1.0e-30 {
            z - f / df_dz
        } else {
            0.5 * (lo + hi)
        };
        if cand > lo && cand < hi {
            let step = (cand - z).abs();
            z = cand;
            if step <= 1.0e-13 * z.max(1.0) {
                break;
            }
        } else {
            z = 0.5 * (lo + hi);
        }
    }
    (2.0 * z) / df
}

// ============================================================================
// Self-Contained Pure-Rust Numerical Linear Algebra (QR + One-Sided Jacobi SVD)
// ============================================================================

/// Thin Householder QR factorization of a column-major matrix $A \in \mathbb{R}^{m \times n}$
/// ($m \ge n$). Returns `(q_cols, r_cols)` where `q_cols` has `n` orthonormal columns of length
/// `m`, and `r_cols` is upper-triangular $n \times n$ stored column-major (`r_cols[j][i] = R_{ij}`).
#[allow(clippy::needless_range_loop)]
fn householder_qr(a_cols: &[Vec<f64>], m: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let n = a_cols.len();
    if n == 0 || m == 0 {
        return (Vec::new(), Vec::new());
    }
    let k_max = m.min(n);
    let mut w: Vec<Vec<f64>> = a_cols.to_vec();
    let mut vs: Vec<Vec<f64>> = Vec::with_capacity(k_max);
    let mut betas: Vec<f64> = vec![0.0; k_max];

    for k in 0..k_max {
        let len = m - k;
        let mut norm_sq = 0.0;
        for i in k..m {
            let val = w[k][i];
            norm_sq += val * val;
        }
        let sigma = norm_sq.sqrt();
        if sigma == 0.0 {
            vs.push(Vec::new());
            betas[k] = 0.0;
            continue;
        }
        let x0 = w[k][k];
        let alpha = if x0 >= 0.0 { -sigma } else { sigma };
        let v0 = x0 - alpha;
        let mut v = vec![0.0; len];
        v[0] = 1.0;
        let mut v_norm_sq = 1.0;
        for i in 1..len {
            let vi = w[k][k + i] / v0;
            v[i] = vi;
            v_norm_sq += vi * vi;
        }
        let beta = 2.0 / v_norm_sq;
        betas[k] = beta;

        w[k][k] = alpha;
        for i in (k + 1)..m {
            w[k][i] = 0.0;
        }

        for j in (k + 1)..n {
            let mut dot = 0.0;
            for i in 0..len {
                dot += v[i] * w[j][k + i];
            }
            let tau = beta * dot;
            for i in 0..len {
                w[j][k + i] -= tau * v[i];
            }
        }
        vs.push(v);
    }

    let mut r_cols = vec![vec![0.0; n]; n];
    for j in 0..n {
        let i_end = (j + 1).min(k_max);
        for i in 0..i_end {
            r_cols[j][i] = w[j][i];
        }
    }

    // Reconstruct thin Q (m x n) by applying reflections in reverse order.
    let mut q_cols = vec![vec![0.0; m]; n];
    for j in 0..k_max {
        q_cols[j][j] = 1.0;
    }
    for k in (0..k_max).rev() {
        let beta = betas[k];
        if beta == 0.0 {
            continue;
        }
        let v = &vs[k];
        let len = m - k;
        for j in k..k_max {
            let mut dot = 0.0;
            for i in 0..len {
                dot += v[i] * q_cols[j][k + i];
            }
            let tau = beta * dot;
            for i in 0..len {
                q_cols[j][k + i] -= tau * v[i];
            }
        }
    }

    (q_cols, r_cols)
}

struct ThinSvd {
    /// Orthonormal left singular columns `u_cols[k]` of length `m` (`k = 0..n`).
    u_cols: Vec<Vec<f64>>,
    /// Singular values sorted descending (`s[0] >= s[1] >= ... >= s[n-1] >= 0`).
    s: Vec<f64>,
    /// Orthogonal right singular columns `v_cols[k]` of length `n` (`k = 0..n`).
    v_cols: Vec<Vec<f64>>,
}

/// One-sided Jacobi SVD (Demmel & Veselić 1992) on a square column-major matrix $R \in \mathbb{R}^{n \times n}$.
#[allow(clippy::needless_range_loop)]
fn jacobi_svd_square(r_cols: &[Vec<f64>]) -> ThinSvd {
    let n = r_cols.len();
    if n == 0 {
        return ThinSvd {
            u_cols: Vec::new(),
            s: Vec::new(),
            v_cols: Vec::new(),
        };
    }
    let mut a: Vec<Vec<f64>> = r_cols.to_vec();
    let mut v: Vec<Vec<f64>> = vec![vec![0.0; n]; n];
    for j in 0..n {
        v[j][j] = 1.0;
    }

    const MAX_SWEEPS: usize = 64;
    const TOL: f64 = 1.0e-15;

    for _sweep in 0..MAX_SWEEPS {
        let mut converged = true;
        for p in 0..n {
            for q in (p + 1)..n {
                let mut alpha = 0.0;
                let mut beta = 0.0;
                let mut gamma = 0.0;
                for i in 0..n {
                    let ap = a[p][i];
                    let aq = a[q][i];
                    alpha += ap * ap;
                    beta += aq * aq;
                    gamma += ap * aq;
                }
                let denom = (alpha * beta).sqrt();
                if denom == 0.0 || gamma.abs() <= TOL * denom {
                    continue;
                }
                converged = false;
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = if zeta >= 0.0 {
                    1.0 / (zeta + (1.0 + zeta * zeta).sqrt())
                } else {
                    -1.0 / (-zeta + (1.0 + zeta * zeta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;

                for i in 0..n {
                    let ap = a[p][i];
                    let aq = a[q][i];
                    a[p][i] = c * ap - s * aq;
                    a[q][i] = s * ap + c * aq;

                    let vp = v[p][i];
                    let vq = v[q][i];
                    v[p][i] = c * vp - s * vq;
                    v[q][i] = s * vp + c * vq;
                }
            }
        }
        if converged {
            break;
        }
    }

    let mut norms: Vec<(usize, f64)> = (0..n)
        .map(|j| {
            let nrm = a[j].iter().map(|&x| x * x).sum::<f64>().sqrt();
            (j, nrm)
        })
        .collect();
    norms.sort_by(|x, y| y.1.total_cmp(&x.1));

    let mut u_cols = Vec::with_capacity(n);
    let mut s_vals = Vec::with_capacity(n);
    let mut v_cols = Vec::with_capacity(n);

    for &(idx, nrm) in &norms {
        s_vals.push(nrm);
        v_cols.push(v[idx].clone());
        if nrm > 0.0 {
            let inv = 1.0 / nrm;
            u_cols.push(a[idx].iter().map(|&x| x * inv).collect());
        } else {
            u_cols.push(vec![0.0; n]);
        }
    }

    // Complete any zero-norm columns of U to an orthonormal basis via modified Gram–Schmidt.
    for j in 0..n {
        if s_vals[j] > 0.0 {
            continue;
        }
        let mut best_vec = vec![0.0; n];
        let mut best_norm = 0.0;
        for basis_idx in 0..n {
            let mut cand = vec![0.0; n];
            cand[basis_idx] = 1.0;
            for prev in 0..j {
                let dot: f64 = cand
                    .iter()
                    .zip(u_cols[prev].iter())
                    .map(|(&c, &u)| c * u)
                    .sum();
                for i in 0..n {
                    cand[i] -= dot * u_cols[prev][i];
                }
            }
            let cn = cand.iter().map(|&x| x * x).sum::<f64>().sqrt();
            if cn > best_norm {
                best_norm = cn;
                best_vec = cand;
            }
        }
        if best_norm > 0.0 {
            let inv = 1.0 / best_norm;
            for x in &mut best_vec {
                *x *= inv;
            }
        }
        u_cols[j] = best_vec;
    }

    ThinSvd {
        u_cols,
        s: s_vals,
        v_cols,
    }
}

/// Thin SVD $A = U \Sigma V^T$ of a tall-skinny column-major matrix $A \in \mathbb{R}^{m \times n}$
/// ($m \ge n$), computed via Householder QR followed by one-sided Jacobi SVD of $R$ so that the
/// condition number is never squared.
#[allow(clippy::needless_range_loop)]
fn thin_svd(a_cols: &[Vec<f64>], m: usize) -> ThinSvd {
    let n = a_cols.len();
    if n == 0 || m == 0 {
        return ThinSvd {
            u_cols: Vec::new(),
            s: Vec::new(),
            v_cols: Vec::new(),
        };
    }
    let (q_cols, r_cols) = householder_qr(a_cols, m);
    let r_svd = jacobi_svd_square(&r_cols);

    // U = Q * U_R
    let mut u_cols = vec![vec![0.0; m]; n];
    for j in 0..n {
        let ur_col = &r_svd.u_cols[j];
        for k in 0..n {
            let coeff = ur_col[k];
            if coeff == 0.0 {
                continue;
            }
            let qk = &q_cols[k];
            for i in 0..m {
                u_cols[j][i] += coeff * qk[i];
            }
        }
    }

    ThinSvd {
        u_cols,
        s: r_svd.s,
        v_cols: r_svd.v_cols,
    }
}

#[allow(clippy::needless_range_loop)]
impl ThinSvd {
    fn condition_number(&self) -> f64 {
        if self.s.is_empty() {
            return 1.0;
        }
        let smax = self.s[0];
        let smin = self.s[self.s.len() - 1];
        if smax == 0.0 {
            return f64::INFINITY;
        }
        if smin <= 0.0 {
            return f64::INFINITY;
        }
        smax / smin
    }

    /// Truncated pseudoinverse singular values $\sigma_i^+$ with threshold `rcond * s_max`.
    fn inv_singular_values(&self, rcond: f64) -> Vec<f64> {
        if self.s.is_empty() {
            return Vec::new();
        }
        let cutoff = self.s[0] * rcond;
        self.s
            .iter()
            .map(|&sv| {
                if sv > cutoff && sv > 0.0 {
                    1.0 / sv
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Solve $\min_a \|A a - y\|_2$ via $a = V \Sigma^+ U^T y$.
    fn solve(&self, y: &[f64], s_inv: &[f64]) -> Vec<f64> {
        let n = self.s.len();
        let mut uty = vec![0.0; n];
        for j in 0..n {
            let mut dot = 0.0;
            for (&u, &yi) in self.u_cols[j].iter().zip(y.iter()) {
                dot += u * yi;
            }
            uty[j] = dot * s_inv[j];
        }
        let mut a = vec![0.0; n];
        for j in 0..n {
            let coeff = uty[j];
            if coeff == 0.0 {
                continue;
            }
            for i in 0..n {
                a[i] += coeff * self.v_cols[j][i];
            }
        }
        a
    }

    /// Compute $U^T v \in \mathbb{R}^n$.
    fn u_transpose_mul(&self, v: &[f64]) -> Vec<f64> {
        let n = self.s.len();
        let mut out = vec![0.0; n];
        for j in 0..n {
            let mut dot = 0.0;
            for (&u, &vi) in self.u_cols[j].iter().zip(v.iter()) {
                dot += u * vi;
            }
            out[j] = dot;
        }
        out
    }

    /// Compute orthogonal projection $P_A^\perp v = v - U_r (U_r^T v)$ over retained singular
    /// directions (`s_inv[j] > 0.0`) so completed nullspace columns never corrupt $\operatorname{Range}(A)^\perp$.
    fn project_perp(&self, v: &[f64], s_inv: &[f64]) -> Vec<f64> {
        let utv = self.u_transpose_mul(v);
        let mut out = v.to_vec();
        for (j, &coeff) in utv.iter().enumerate() {
            if s_inv[j] == 0.0 || coeff == 0.0 {
                continue;
            }
            let uj = &self.u_cols[j];
            for i in 0..out.len() {
                out[i] -= coeff * uj[i];
            }
        }
        out
    }

    /// Compute $(A^+)^T w = U \Sigma^+ V^T w \in \mathbb{R}^m$.
    fn pinv_transpose_mul(&self, w: &[f64], s_inv: &[f64], m: usize) -> Vec<f64> {
        let n = self.s.len();
        let mut out = vec![0.0; m];
        for j in 0..n {
            if s_inv[j] == 0.0 {
                continue;
            }
            let mut vtw = 0.0;
            for (&vj, &wi) in self.v_cols[j].iter().zip(w.iter()) {
                vtw += vj * wi;
            }
            let coeff = s_inv[j] * vtw;
            if coeff == 0.0 {
                continue;
            }
            let uj = &self.u_cols[j];
            for i in 0..m {
                out[i] += coeff * uj[i];
            }
        }
        out
    }

    /// Compute $(A^T A)^+ = V (\Sigma^+)^2 V^T$ as a dense $n \times n$ matrix.
    fn gram_pinv(&self, s_inv: &[f64]) -> Vec<Vec<f64>> {
        let n = self.s.len();
        let mut inv = vec![vec![0.0; n]; n];
        for k in 0..n {
            let w = s_inv[k] * s_inv[k];
            if w == 0.0 {
                continue;
            }
            let vk = &self.v_cols[k];
            for i in 0..n {
                let wi = w * vk[i];
                for j in 0..n {
                    inv[i][j] += wi * vk[j];
                }
            }
        }
        inv
    }
}

/// Solve symmetric positive-definite $(H + \lambda \operatorname{diag}(d^2)) x = g$ via Cholesky
/// decomposition with automatic Jacobi-SVD fallback if numerically ill-conditioned.
#[allow(clippy::needless_range_loop)]
fn solve_damped_normal_eqs(h: &[Vec<f64>], g: &[f64], d_sq: &[f64], lambda: f64) -> Vec<f64> {
    let n = g.len();
    if n == 0 {
        return Vec::new();
    }
    let mut a = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            a[i][j] = h[i][j];
        }
        a[i][i] += lambda * d_sq[i] + 1.0e-15;
    }

    // Attempt Cholesky L * L^T = A
    let mut l = vec![vec![0.0; n]; n];
    let mut chol_ok = true;
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }
            if i == j {
                if sum <= 0.0 || !sum.is_finite() {
                    chol_ok = false;
                    break;
                }
                l[i][j] = sum.sqrt();
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
        if !chol_ok {
            break;
        }
    }

    if chol_ok {
        let mut y = vec![0.0; n];
        for i in 0..n {
            let mut sum = g[i];
            for k in 0..i {
                sum -= l[i][k] * y[k];
            }
            y[i] = sum / l[i][i];
        }
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let mut sum = y[i];
            for k in (i + 1)..n {
                sum -= l[k][i] * x[k];
            }
            x[i] = sum / l[i][i];
        }
        if x.iter().all(|v| v.is_finite()) {
            return x;
        }
    }

    // Fallback: Jacobi SVD on the square matrix A (stored column-major)
    let mut a_cols = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            a_cols[j][i] = a[i][j];
        }
    }
    let svd = jacobi_svd_square(&a_cols);
    let s_inv = svd.inv_singular_values(1.0e-12);
    svd.solve(g, &s_inv)
}

// ============================================================================
// Unified N-D (2D / 3D) Multi-Plane VarPro Engine
// ============================================================================

#[derive(Clone, Debug)]
struct ParamLayout {
    n_planes: usize,
    n_peaks: usize,
    n_nonlinear: usize,
    /// `center_idx[peak][dim]` -> nonlinear parameter index.
    center_idx: Vec<Vec<usize>>,
    /// `fwhm_idx[plane][peak][dim]` -> nonlinear parameter index.
    fwhm_idx: Vec<Vec<Vec<usize>>>,
    /// `eta_idx[peak][dim]` -> optional nonlinear parameter index (`None` if fixed).
    eta_idx: Vec<Vec<Option<usize>>>,
    /// Fixed mixing values used when `eta_idx[peak][dim]` is `None`.
    fixed_eta: Vec<Vec<f64>>,
    /// Box bounds for each nonlinear parameter.
    lower: Vec<f64>,
    upper: Vec<f64>,
    /// Characteristic scale for relative step stopping criterion.
    scales: Vec<f64>,
    /// Human-readable parameter name for each nonlinear parameter.
    names: Vec<String>,
    /// Whether all planes share the exact same basis matrix $\Phi(\boldsymbol{\theta})$.
    shared_basis_across_planes: bool,
}

impl ParamLayout {
    fn build(
        n_planes: usize,
        n_peaks: usize,
        axes: &[AxisBounds],
        seed_centers: &[Vec<f64>],
        seed_fwhms: &[Vec<f64>],
        seed_etas: &[Vec<f64>],
        cfg: &DeconvConfig,
    ) -> Result<(Self, Vec<f64>), DeconvError> {
        let n_dims = axes.len();
        let axis_tags = ["f1", "f2", "f3"];
        let mut theta = Vec::new();
        let mut lower = Vec::new();
        let mut upper = Vec::new();
        let mut scales = Vec::new();
        let mut names = Vec::new();

        // 1. Peak centers: always 1 per (peak, dim), shared across planes.
        let mut center_idx = vec![vec![0usize; n_dims]; n_peaks];
        for k in 0..n_peaks {
            for d in 0..n_dims {
                let c0 = seed_centers[k][d];
                let f0 = seed_fwhms[k][d];
                let ax = axes[d];
                let (lo, hi) = if let Some(shift_mult) = cfg.max_center_shift_fwhm {
                    let delta = shift_mult * f0;
                    (
                        (c0 - delta).max(ax.min_coord),
                        (c0 + delta).min(ax.max_coord),
                    )
                } else {
                    (ax.min_coord, ax.max_coord)
                };
                let (lo, hi) = if lo <= hi {
                    (lo, hi)
                } else {
                    (ax.min_coord, ax.max_coord)
                };
                let idx = theta.len();
                center_idx[k][d] = idx;
                theta.push(c0.clamp(lo, hi));
                lower.push(lo);
                upper.push(hi);
                scales.push(f0.max(ax.mean_step));
                names.push(format!("peak[{k}].center_{}", axis_tags[d]));
            }
        }

        // 2. Linewidths (FWHM) according to LinewidthSharing policy.
        let mut fwhm_idx = vec![vec![vec![0usize; n_dims]; n_peaks]; n_planes];
        let shared_basis_across_planes = match cfg.linewidth_sharing {
            LinewidthSharing::SharedPerCluster => {
                // Indirect axes (d < n_dims - 1): 1 shared FWHM per axis across all peaks & planes.
                for d in 0..(n_dims - 1) {
                    let ax = axes[d];
                    let f_lo = (cfg.min_fwhm_grid_frac * ax.mean_step).max(1.0e-12);
                    let f_hi = (cfg.max_fwhm_span_frac * ax.span).max(f_lo * 2.0);
                    let mean_seed: f64 =
                        seed_fwhms.iter().map(|v| v[d]).sum::<f64>() / (n_peaks.max(1) as f64);
                    let idx = theta.len();
                    theta.push(mean_seed.clamp(f_lo, f_hi));
                    lower.push(f_lo);
                    upper.push(f_hi);
                    scales.push(mean_seed.max(ax.mean_step));
                    names.push(format!("cluster.fwhm_{}", axis_tags[d]));
                    for plane_entry in &mut fwhm_idx {
                        for peak_entry in plane_entry.iter_mut() {
                            peak_entry[d] = idx;
                        }
                    }
                }
                // Direct axis (d = n_dims - 1): 1 FWHM per peak, shared across planes.
                let d = n_dims - 1;
                let ax = axes[d];
                let f_lo = (cfg.min_fwhm_grid_frac * ax.mean_step).max(1.0e-12);
                let f_hi = (cfg.max_fwhm_span_frac * ax.span).max(f_lo * 2.0);
                for k in 0..n_peaks {
                    let idx = theta.len();
                    let f0 = seed_fwhms[k][d];
                    theta.push(f0.clamp(f_lo, f_hi));
                    lower.push(f_lo);
                    upper.push(f_hi);
                    scales.push(f0.max(ax.mean_step));
                    names.push(format!("peak[{k}].fwhm_{}", axis_tags[d]));
                    for plane_entry in &mut fwhm_idx {
                        plane_entry[k][d] = idx;
                    }
                }
                true
            }
            LinewidthSharing::SharedAcrossPlanes => {
                for k in 0..n_peaks {
                    for d in 0..n_dims {
                        let ax = axes[d];
                        let f_lo = (cfg.min_fwhm_grid_frac * ax.mean_step).max(1.0e-12);
                        let f_hi = (cfg.max_fwhm_span_frac * ax.span).max(f_lo * 2.0);
                        let f0 = seed_fwhms[k][d];
                        let idx = theta.len();
                        theta.push(f0.clamp(f_lo, f_hi));
                        lower.push(f_lo);
                        upper.push(f_hi);
                        scales.push(f0.max(ax.mean_step));
                        names.push(format!("peak[{k}].fwhm_{}", axis_tags[d]));
                        for plane_entry in &mut fwhm_idx {
                            plane_entry[k][d] = idx;
                        }
                    }
                }
                true
            }
            LinewidthSharing::Independent => {
                for (p, plane_entry) in fwhm_idx.iter_mut().enumerate() {
                    for k in 0..n_peaks {
                        for d in 0..n_dims {
                            let ax = axes[d];
                            let f_lo = (cfg.min_fwhm_grid_frac * ax.mean_step).max(1.0e-12);
                            let f_hi = (cfg.max_fwhm_span_frac * ax.span).max(f_lo * 2.0);
                            let f0 = seed_fwhms[k][d];
                            let idx = theta.len();
                            theta.push(f0.clamp(f_lo, f_hi));
                            lower.push(f_lo);
                            upper.push(f_hi);
                            scales.push(f0.max(ax.mean_step));
                            if n_planes == 1 {
                                names.push(format!("peak[{k}].fwhm_{}", axis_tags[d]));
                            } else {
                                names.push(format!("plane[{p}].peak[{k}].fwhm_{}", axis_tags[d]));
                            }
                            plane_entry[k][d] = idx;
                        }
                    }
                }
                n_planes == 1
            }
        };

        // 3. Pseudo-Voigt mixing parameters (eta in [0, 1]).
        let mut eta_idx = vec![vec![None; n_dims]; n_peaks];
        let mut fixed_eta = vec![vec![0.0; n_dims]; n_peaks];
        for k in 0..n_peaks {
            for d in 0..n_dims {
                fixed_eta[k][d] = match cfg.lineshape {
                    LineshapeKind::Lorentzian => 1.0,
                    LineshapeKind::Gaussian => 0.0,
                    LineshapeKind::PseudoVoigt => seed_etas[k][d].clamp(0.0, 1.0),
                };
            }
        }

        if cfg.lineshape == LineshapeKind::PseudoVoigt && cfg.fit_mixing {
            if cfg.linewidth_sharing == LinewidthSharing::SharedPerCluster {
                // Share indirect mixing across cluster, per-peak on direct axis.
                for d in 0..(n_dims - 1) {
                    let mean_eta: f64 =
                        seed_etas.iter().map(|v| v[d]).sum::<f64>() / (n_peaks.max(1) as f64);
                    let idx = theta.len();
                    theta.push(mean_eta.clamp(0.0, 1.0));
                    lower.push(0.0);
                    upper.push(1.0);
                    scales.push(1.0);
                    names.push(format!("cluster.eta_{}", axis_tags[d]));
                    for entry in &mut eta_idx {
                        entry[d] = Some(idx);
                    }
                }
                let d = n_dims - 1;
                for k in 0..n_peaks {
                    let idx = theta.len();
                    theta.push(seed_etas[k][d].clamp(0.0, 1.0));
                    lower.push(0.0);
                    upper.push(1.0);
                    scales.push(1.0);
                    names.push(format!("peak[{k}].eta_{}", axis_tags[d]));
                    eta_idx[k][d] = Some(idx);
                }
            } else {
                for k in 0..n_peaks {
                    for d in 0..n_dims {
                        let idx = theta.len();
                        theta.push(seed_etas[k][d].clamp(0.0, 1.0));
                        lower.push(0.0);
                        upper.push(1.0);
                        scales.push(1.0);
                        names.push(format!("peak[{k}].eta_{}", axis_tags[d]));
                        eta_idx[k][d] = Some(idx);
                    }
                }
            }
        }

        let n_nonlinear = theta.len();
        Ok((
            Self {
                n_planes,
                n_peaks,
                n_nonlinear,
                center_idx,
                fwhm_idx,
                eta_idx,
                fixed_eta,
                lower,
                upper,
                scales,
                names,
                shared_basis_across_planes,
            },
            theta,
        ))
    }

    #[inline]
    fn bound_tol(&self, i: usize) -> f64 {
        let span = (self.upper[i] - self.lower[i]).abs();
        let ref_scale = self.scales[i]
            .max(span)
            .max(self.lower[i].abs())
            .max(self.upper[i].abs())
            .max(1.0e-15);
        1.0e-12 * ref_scale
    }

    #[inline]
    fn center(&self, theta: &[f64], k: usize, d: usize) -> f64 {
        theta[self.center_idx[k][d]]
    }

    #[inline]
    fn fwhm(&self, theta: &[f64], p: usize, k: usize, d: usize) -> f64 {
        theta[self.fwhm_idx[p][k][d]]
    }

    #[inline]
    fn eta(&self, theta: &[f64], k: usize, d: usize) -> f64 {
        match self.eta_idx[k][d] {
            Some(idx) => theta[idx],
            None => self.fixed_eta[k][d],
        }
    }
}

/// Sparse per-peak basis and derivative evaluation on a single plane.
struct PlaneBasis {
    /// Columns of $\Phi \in \mathbb{R}^{M \times L}$ (`L = K` or `K + 1` with baseline).
    phi_cols: Vec<Vec<f64>>,
    /// Derivative entries `(param_j, peak_k, d_phi_k_over_d_theta_j)` where only non-zero pairs
    /// are stored.
    deriv_entries: Vec<(usize, usize, Vec<f64>)>,
}

/// Evaluate separable 2D or 3D basis columns $\Phi(\boldsymbol{\theta})$ and exact analytical
/// partial derivatives $\partial \phi_k / \partial \theta_j$ for plane `p`.
#[allow(clippy::needless_range_loop)]
fn eval_plane_basis(
    coords: &[&[f64]],
    layout: &ParamLayout,
    theta: &[f64],
    p: usize,
    kind: LineshapeKind,
    fit_baseline: bool,
    need_derivs: bool,
) -> PlaneBasis {
    let n_dims = coords.len();
    let m: usize = coords.iter().map(|c| c.len()).product();
    let n_lin = layout.n_peaks + usize::from(fit_baseline);
    let mut phi_cols = Vec::with_capacity(n_lin);
    let mut deriv_entries = Vec::new();

    for k in 0..layout.n_peaks {
        let mut evals_1d: Vec<Vec<LineshapeEval>> = Vec::with_capacity(n_dims);
        for (d, axis_coords) in coords.iter().enumerate() {
            let mu = layout.center(theta, k, d);
            let fwhm = layout.fwhm(theta, p, k, d);
            let eta = layout.eta(theta, k, d);
            let prof: Vec<LineshapeEval> = axis_coords
                .iter()
                .map(|&x| eval_lineshape_1d(kind, x, mu, fwhm, eta))
                .collect();
            evals_1d.push(prof);
        }

        if n_dims == 2 {
            let n1 = coords[0].len();
            let n2 = coords[1].len();
            let e1 = &evals_1d[0];
            let e2 = &evals_1d[1];

            let mut col = vec![0.0; m];
            for i1 in 0..n1 {
                let v1 = e1[i1].value;
                let row = i1 * n2;
                for i2 in 0..n2 {
                    col[row + i2] = v1 * e2[i2].value;
                }
            }
            phi_cols.push(col);

            if need_derivs {
                // d / d(center_f1)
                let mut dc1 = vec![0.0; m];
                // d / d(fwhm_f1)
                let mut df1 = vec![0.0; m];
                // d / d(center_f2)
                let mut dc2 = vec![0.0; m];
                // d / d(fwhm_f2)
                let mut df2 = vec![0.0; m];
                let want_e1 = layout.eta_idx[k][0].is_some();
                let want_e2 = layout.eta_idx[k][1].is_some();
                let mut de1 = if want_e1 { vec![0.0; m] } else { Vec::new() };
                let mut de2 = if want_e2 { vec![0.0; m] } else { Vec::new() };

                for i1 in 0..n1 {
                    let a1 = e1[i1];
                    let row = i1 * n2;
                    for i2 in 0..n2 {
                        let a2 = e2[i2];
                        let idx = row + i2;
                        dc1[idx] = a1.d_mu * a2.value;
                        df1[idx] = a1.d_fwhm * a2.value;
                        dc2[idx] = a1.value * a2.d_mu;
                        df2[idx] = a1.value * a2.d_fwhm;
                        if want_e1 {
                            de1[idx] = a1.d_eta * a2.value;
                        }
                        if want_e2 {
                            de2[idx] = a1.value * a2.d_eta;
                        }
                    }
                }
                deriv_entries.push((layout.center_idx[k][0], k, dc1));
                deriv_entries.push((layout.fwhm_idx[p][k][0], k, df1));
                deriv_entries.push((layout.center_idx[k][1], k, dc2));
                deriv_entries.push((layout.fwhm_idx[p][k][1], k, df2));
                if let Some(ei1) = layout.eta_idx[k][0] {
                    deriv_entries.push((ei1, k, de1));
                }
                if let Some(ei2) = layout.eta_idx[k][1] {
                    deriv_entries.push((ei2, k, de2));
                }
            }
        } else {
            // 3D separable tensor (F1, F2, F3)
            let n1 = coords[0].len();
            let n2 = coords[1].len();
            let n3 = coords[2].len();
            let e1 = &evals_1d[0];
            let e2 = &evals_1d[1];
            let e3 = &evals_1d[2];

            let mut col = vec![0.0; m];
            for i1 in 0..n1 {
                let v1 = e1[i1].value;
                for i2 in 0..n2 {
                    let v12 = v1 * e2[i2].value;
                    let base = (i1 * n2 + i2) * n3;
                    for i3 in 0..n3 {
                        col[base + i3] = v12 * e3[i3].value;
                    }
                }
            }
            phi_cols.push(col);

            if need_derivs {
                let mut dc1 = vec![0.0; m];
                let mut df1 = vec![0.0; m];
                let mut dc2 = vec![0.0; m];
                let mut df2 = vec![0.0; m];
                let mut dc3 = vec![0.0; m];
                let mut df3 = vec![0.0; m];
                let want_e1 = layout.eta_idx[k][0].is_some();
                let want_e2 = layout.eta_idx[k][1].is_some();
                let want_e3 = layout.eta_idx[k][2].is_some();
                let mut de1 = if want_e1 { vec![0.0; m] } else { Vec::new() };
                let mut de2 = if want_e2 { vec![0.0; m] } else { Vec::new() };
                let mut de3 = if want_e3 { vec![0.0; m] } else { Vec::new() };

                for i1 in 0..n1 {
                    let a1 = e1[i1];
                    for i2 in 0..n2 {
                        let a2 = e2[i2];
                        let v12 = a1.value * a2.value;
                        let dm1_v2 = a1.d_mu * a2.value;
                        let df1_v2 = a1.d_fwhm * a2.value;
                        let v1_dm2 = a1.value * a2.d_mu;
                        let v1_df2 = a1.value * a2.d_fwhm;
                        let de1_v2 = a1.d_eta * a2.value;
                        let v1_de2 = a1.value * a2.d_eta;
                        let base = (i1 * n2 + i2) * n3;
                        for i3 in 0..n3 {
                            let a3 = e3[i3];
                            let idx = base + i3;
                            dc1[idx] = dm1_v2 * a3.value;
                            df1[idx] = df1_v2 * a3.value;
                            dc2[idx] = v1_dm2 * a3.value;
                            df2[idx] = v1_df2 * a3.value;
                            dc3[idx] = v12 * a3.d_mu;
                            df3[idx] = v12 * a3.d_fwhm;
                            if want_e1 {
                                de1[idx] = de1_v2 * a3.value;
                            }
                            if want_e2 {
                                de2[idx] = v1_de2 * a3.value;
                            }
                            if want_e3 {
                                de3[idx] = v12 * a3.d_eta;
                            }
                        }
                    }
                }
                deriv_entries.push((layout.center_idx[k][0], k, dc1));
                deriv_entries.push((layout.fwhm_idx[p][k][0], k, df1));
                deriv_entries.push((layout.center_idx[k][1], k, dc2));
                deriv_entries.push((layout.fwhm_idx[p][k][1], k, df2));
                deriv_entries.push((layout.center_idx[k][2], k, dc3));
                deriv_entries.push((layout.fwhm_idx[p][k][2], k, df3));
                if let Some(ei1) = layout.eta_idx[k][0] {
                    deriv_entries.push((ei1, k, de1));
                }
                if let Some(ei2) = layout.eta_idx[k][1] {
                    deriv_entries.push((ei2, k, de2));
                }
                if let Some(ei3) = layout.eta_idx[k][2] {
                    deriv_entries.push((ei3, k, de3));
                }
            }
        }
    }

    if fit_baseline {
        phi_cols.push(vec![1.0; m]);
    }

    PlaneBasis {
        phi_cols,
        deriv_entries,
    }
}

struct VarProEval {
    /// Per-plane linear solutions $\mathbf{a}^{(p)} \in \mathbb{R}^L$.
    plane_amplitudes: Vec<Vec<f64>>,
    /// Per-plane fitted signal $\hat{\mathbf{y}}^{(p)} = \Phi^{(p)} \mathbf{a}^{(p)}$.
    plane_fitted: Vec<Vec<f64>>,
    /// Per-plane residuals $\mathbf{r}^{(p)} = \mathbf{y}^{(p)} - \hat{\mathbf{y}}^{(p)}$.
    plane_residuals: Vec<Vec<f64>>,
    /// Per-plane RSS and total RSS across all planes.
    plane_rss: Vec<f64>,
    total_rss: f64,
    /// Worst basis condition number $\max_p \kappa(\Phi^{(p)})$.
    max_basis_kappa: f64,
    /// Exact Golub–Pereyra VarPro Jacobian columns for $\hat{\mathbf{y}}(\boldsymbol{\theta})$:
    /// `j_vp_cols[param_j]` has length `P * M` (only populated when `need_jacobian` is true).
    j_vp_cols: Vec<Vec<f64>>,
}

/// Evaluate VarPro projected linear amplitudes, residuals, and exact Golub–Pereyra (1973)
/// VarPro Jacobian $\partial \hat{\mathbf{y}} / \partial \boldsymbol{\theta}$ across all planes.
fn eval_varpro_system(
    coords: &[&[f64]],
    planes: &[&[f64]],
    layout: &ParamLayout,
    theta: &[f64],
    cfg: &DeconvConfig,
    need_jacobian: bool,
) -> Result<VarProEval, DeconvError> {
    let m: usize = coords.iter().map(|c| c.len()).product();
    let n_planes = planes.len();
    let n_lin = layout.n_peaks + usize::from(cfg.fit_baseline);
    let n_nl = layout.n_nonlinear;
    let total_obs = n_planes.checked_mul(m).ok_or(DeconvError::OversizedRoi {
        voxels: usize::MAX,
        max_voxels: MAX_DECONV_VOXELS,
    })?;
    if need_jacobian {
        check_memory_budget(total_obs, n_nl)?;
    }

    let mut plane_amplitudes = Vec::with_capacity(n_planes);
    let mut plane_fitted = Vec::with_capacity(n_planes);
    let mut plane_residuals = Vec::with_capacity(n_planes);
    let mut plane_rss = Vec::with_capacity(n_planes);
    let mut total_rss = 0.0;
    let mut max_basis_kappa = 0.0_f64;

    let mut j_vp_cols = if need_jacobian {
        vec![vec![0.0; total_obs]; n_nl]
    } else {
        Vec::new()
    };

    // Reuse basis & thin SVD across all planes when Phi is shared.
    let shared_basis_and_svd = if layout.shared_basis_across_planes {
        let basis = eval_plane_basis(
            coords,
            layout,
            theta,
            0,
            cfg.lineshape,
            cfg.fit_baseline,
            need_jacobian,
        );
        let svd = thin_svd(&basis.phi_cols, m);
        let s_inv = svd.inv_singular_values(cfg.svd_rcond);
        Some((basis, svd, s_inv))
    } else {
        None
    };

    for (p, &y_p) in planes.iter().enumerate() {
        let owned;
        let (basis, svd, s_inv) = if let Some(ref shared) = shared_basis_and_svd {
            (&shared.0, &shared.1, &shared.2)
        } else {
            let b = eval_plane_basis(
                coords,
                layout,
                theta,
                p,
                cfg.lineshape,
                cfg.fit_baseline,
                need_jacobian,
            );
            let s = thin_svd(&b.phi_cols, m);
            let si = s.inv_singular_values(cfg.svd_rcond);
            owned = (b, s, si);
            (&owned.0, &owned.1, &owned.2)
        };

        let kappa = svd.condition_number();
        if kappa > max_basis_kappa {
            max_basis_kappa = kappa;
        }

        let a_p = svd.solve(y_p, s_inv);
        let mut yhat_p = vec![0.0; m];
        for (l, &al) in a_p.iter().enumerate() {
            if al == 0.0 {
                continue;
            }
            let col = &basis.phi_cols[l];
            for i in 0..m {
                yhat_p[i] += al * col[i];
            }
        }

        let mut r_p = vec![0.0; m];
        let mut rss_p = 0.0;
        for i in 0..m {
            let ri = y_p[i] - yhat_p[i];
            r_p[i] = ri;
            rss_p += ri * ri;
        }

        if need_jacobian {
            // Accumulate D_j * a_p (length m) and D_j^T * r_p (length n_lin) for each touched param_j.
            let mut d_a: Vec<Option<Vec<f64>>> = vec![None; n_nl];
            let mut d_t_r: Vec<Option<Vec<f64>>> = vec![None; n_nl];

            for (param_j, peak_k, dcol) in &basis.deriv_entries {
                let ak = a_p[*peak_k];
                let da_vec = d_a[*param_j].get_or_insert_with(|| vec![0.0; m]);
                if ak != 0.0 {
                    for i in 0..m {
                        da_vec[i] += ak * dcol[i];
                    }
                }
                let mut dot = 0.0;
                for i in 0..m {
                    dot += dcol[i] * r_p[i];
                }
                let dtr_vec = d_t_r[*param_j].get_or_insert_with(|| vec![0.0; n_lin]);
                dtr_vec[*peak_k] += dot;
            }

            let offset = p * m;
            for j in 0..n_nl {
                let has_da = d_a[j].is_some();
                let has_dtr = d_t_r[j].is_some();
                if !has_da && !has_dtr {
                    continue;
                }
                // Exact Golub–Pereyra (1973) VarPro Jacobian column for yhat = Phi * Phi^+ * y:
                //   d(yhat)/d(theta_j) = P_Phi^\perp (D_j a) + (Phi^+)^T (D_j^T r)
                let term1 = if let Some(ref v) = d_a[j] {
                    svd.project_perp(v, s_inv)
                } else {
                    vec![0.0; m]
                };
                let term2 = if let Some(ref w) = d_t_r[j] {
                    svd.pinv_transpose_mul(w, s_inv, m)
                } else {
                    vec![0.0; m]
                };
                let dst = &mut j_vp_cols[j][offset..(offset + m)];
                for i in 0..m {
                    dst[i] = term1[i] + term2[i];
                }
            }
        }

        total_rss += rss_p;
        plane_rss.push(rss_p);
        plane_amplitudes.push(a_p);
        plane_fitted.push(yhat_p);
        plane_residuals.push(r_p);
    }

    Ok(VarProEval {
        plane_amplitudes,
        plane_fitted,
        plane_residuals,
        plane_rss,
        total_rss,
        max_basis_kappa,
        j_vp_cols,
    })
}

/// Exact full Cramér–Rao covariance computed via block Schur complement with unsquared QR + SVD
/// (combining all linear amplitudes and free interior nonlinear shape parameters; parameters
/// sitting on an active box bound are excluded from the free Fisher information and left zeroed
/// so their CRLB SEs are suppressed as `None`).
struct FullCovarianceResult {
    /// Per-plane full linear covariance matrices (`P` matrices of size $L \times L$).
    plane_linear_cov: Vec<Vec<Vec<f64>>>,
    /// Nonlinear parameter covariance matrix ($N \times N$; zero on active-bound rows/cols).
    nonlinear_cov: Vec<Vec<f64>>,
    /// Full $(L + N) \times (L + N)$ covariance matrix when `P == 1` (empty when `P > 1`).
    single_plane_full_cov: Vec<Vec<f64>>,
    /// Condition number of the full Schur Jacobian $F$ (and basis $\Phi$).
    jacobian_condition_number: f64,
    /// Whether any singular value was truncated (rank deficiency).
    rank_deficient: bool,
}

#[allow(clippy::needless_range_loop)]
fn compute_full_crlb_covariance(
    coords: &[&[f64]],
    layout: &ParamLayout,
    theta: &[f64],
    plane_amplitudes: &[Vec<f64>],
    active_bound_mask: &[bool],
    sigma_sq: f64,
    cfg: &DeconvConfig,
) -> FullCovarianceResult {
    let m: usize = coords.iter().map(|c| c.len()).product();
    let n_planes = layout.n_planes;
    let n_lin = layout.n_peaks + usize::from(cfg.fit_baseline);
    let n_nl = layout.n_nonlinear;

    let mut free_nl: Vec<usize> = Vec::with_capacity(n_nl);
    let mut nl_to_free: Vec<Option<usize>> = vec![None; n_nl];
    for j in 0..n_nl {
        if !active_bound_mask.get(j).copied().unwrap_or(false) {
            nl_to_free[j] = Some(free_nl.len());
            free_nl.push(j);
        }
    }
    let n_free = free_nl.len();

    let shared_basis_and_svd = if layout.shared_basis_across_planes {
        let basis = eval_plane_basis(
            coords,
            layout,
            theta,
            0,
            cfg.lineshape,
            cfg.fit_baseline,
            n_free > 0,
        );
        let svd = thin_svd(&basis.phi_cols, m);
        let s_inv = svd.inv_singular_values(cfg.svd_rcond);
        Some((basis, svd, s_inv))
    } else {
        None
    };

    // Per-plane cached A_p^{-1} = V_p Sigma_p^{-2} V_p^T and G_p = V_p Sigma_p^{-1} U_p^T J_theta^(p)
    let mut a_inv_list: Vec<Vec<Vec<f64>>> = Vec::with_capacity(n_planes);
    let mut g_list: Vec<Vec<Vec<f64>>> = Vec::with_capacity(n_planes);
    // Compressed triangular rows from QR(F_p) stacked across planes so we never square kappa(F).
    let mut r_stack_cols: Vec<Vec<f64>> = vec![Vec::new(); n_free];
    let mut r_stack_rows = 0usize;
    let mut rank_deficient = false;

    for (p, a_p) in plane_amplitudes.iter().enumerate() {
        let owned;
        let (basis, svd, s_inv) = if let Some(ref shared) = shared_basis_and_svd {
            (&shared.0, &shared.1, &shared.2)
        } else {
            let b = eval_plane_basis(
                coords,
                layout,
                theta,
                p,
                cfg.lineshape,
                cfg.fit_baseline,
                n_free > 0,
            );
            let s = thin_svd(&b.phi_cols, m);
            let si = s.inv_singular_values(cfg.svd_rcond);
            owned = (b, s, si);
            (&owned.0, &owned.1, &owned.2)
        };

        if s_inv.contains(&0.0) {
            rank_deficient = true;
        }

        // Build J_theta^(p) (m x n_free) holding a_p and active-bound params fixed.
        let mut j_theta_cols = vec![vec![0.0; m]; n_free];
        for (param_j, peak_k, dcol) in &basis.deriv_entries {
            let Some(free_j) = nl_to_free[*param_j] else {
                continue;
            };
            let ak = a_p[*peak_k];
            if ak == 0.0 {
                continue;
            }
            let dst = &mut j_theta_cols[free_j];
            for i in 0..m {
                dst[i] += ak * dcol[i];
            }
        }

        // Project each free column j of J_theta^(p) over retained singular directions (s_inv[l] > 0.0):
        //   E_p[l, j] = U_p[:, l]^T J_theta^(p)[:, j]  if s_inv[l] > 0 else 0
        //   F_p[:, j] = J_theta^(p)[:, j] - sum_{l: s_inv[l] > 0} E_p[l, j] U_p[:, l]
        let mut e_p = vec![vec![0.0; n_free]; n_lin];
        let mut f_p_cols = Vec::with_capacity(n_free);
        for j in 0..n_free {
            let utj = svd.u_transpose_mul(&j_theta_cols[j]);
            let mut fj = j_theta_cols[j].clone();
            for (l, &coeff) in utj.iter().enumerate() {
                if s_inv[l] == 0.0 {
                    continue;
                }
                e_p[l][j] = coeff;
                if coeff == 0.0 {
                    continue;
                }
                let ul = &svd.u_cols[l];
                for i in 0..m {
                    fj[i] -= coeff * ul[i];
                }
            }
            f_p_cols.push(fj);
        }

        // Compress F_p (m x n_free) via QR into R_p (n_free x n_free) if m >= n_free, else push F_p rows directly.
        if m >= n_free && n_free > 0 {
            let (_q_f, r_f) = householder_qr(&f_p_cols, m);
            for j in 0..n_free {
                r_stack_cols[j].extend_from_slice(&r_f[j]);
            }
            r_stack_rows += n_free;
        } else if n_free > 0 {
            for j in 0..n_free {
                r_stack_cols[j].extend_from_slice(&f_p_cols[j]);
            }
            r_stack_rows += m;
        }

        // A_p^{-1} = V_p Sigma_p^{-2} V_p^T
        let a_inv = svd.gram_pinv(s_inv);
        a_inv_list.push(a_inv);

        // G_p = V_p Sigma_p^{-1} E_p (n_lin x n_free)
        let mut g_p = vec![vec![0.0; n_free]; n_lin];
        for k in 0..n_lin {
            let si = s_inv[k];
            if si == 0.0 {
                continue;
            }
            let vk = &svd.v_cols[k];
            for r in 0..n_lin {
                let coeff = vk[r] * si;
                if coeff == 0.0 {
                    continue;
                }
                for j in 0..n_free {
                    g_p[r][j] += coeff * e_p[k][j];
                }
            }
        }
        g_list.push(g_p);
    }

    // Thin SVD of stacked F (represented identically by R_stack) gives S^{-1} = V_F Sigma_F^{-2} V_F^T.
    let (s_inv_free, jacobian_condition_number) = if n_free == 0 {
        (Vec::new(), 1.0)
    } else if r_stack_rows >= n_free {
        let f_svd = thin_svd(&r_stack_cols, r_stack_rows);
        let f_s_inv = f_svd.inv_singular_values(cfg.svd_rcond);
        if f_s_inv.contains(&0.0) {
            rank_deficient = true;
        }
        (f_svd.gram_pinv(&f_s_inv), f_svd.condition_number())
    } else {
        rank_deficient = true;
        (vec![vec![0.0; n_free]; n_free], f64::INFINITY)
    };

    // Embed free nonlinear covariance into full n_nl x n_nl matrix (active-bound rows/cols stay 0).
    let mut nonlinear_cov = vec![vec![0.0; n_nl]; n_nl];
    for (fi, &orig_i) in free_nl.iter().enumerate() {
        for (fj, &orig_j) in free_nl.iter().enumerate() {
            nonlinear_cov[orig_i][orig_j] = sigma_sq * s_inv_free[fi][fj];
        }
    }

    // Per-plane linear amplitude covariance:
    //   Sigma_{a^(p), a^(p)} = sigma^2 * (A_p^{-1} + G_p S^{-1} G_p^T)
    let mut plane_linear_cov = Vec::with_capacity(n_planes);
    for p in 0..n_planes {
        let a_inv = &a_inv_list[p];
        let g_p = &g_list[p];
        // Compute tmp = G_p * S^{-1} (n_lin x n_free)
        let mut tmp = vec![vec![0.0; n_free]; n_lin];
        for r in 0..n_lin {
            for c in 0..n_free {
                let mut sum = 0.0;
                for k in 0..n_free {
                    sum += g_p[r][k] * s_inv_free[k][c];
                }
                tmp[r][c] = sum;
            }
        }
        let mut cov_p = vec![vec![0.0; n_lin]; n_lin];
        for r in 0..n_lin {
            for c in 0..n_lin {
                let mut extra = 0.0;
                for k in 0..n_free {
                    extra += tmp[r][k] * g_p[c][k];
                }
                cov_p[r][c] = sigma_sq * (a_inv[r][c] + extra);
            }
        }
        plane_linear_cov.push(cov_p);
    }

    // For single-plane ROIs, assemble the complete (L + N) x (L + N) joint covariance matrix.
    let single_plane_full_cov = if n_planes == 1 {
        let total = n_lin + n_nl;
        let mut full = vec![vec![0.0; total]; total];
        for i in 0..n_lin {
            for j in 0..n_lin {
                full[i][j] = plane_linear_cov[0][i][j];
            }
        }
        // Off-diagonal block: -sigma^2 * G_0 * S^{-1}
        let g_0 = &g_list[0];
        for i in 0..n_lin {
            for (fj, &orig_j) in free_nl.iter().enumerate() {
                let mut val = 0.0;
                for k in 0..n_free {
                    val += g_0[i][k] * s_inv_free[k][fj];
                }
                let entry = -sigma_sq * val;
                full[i][n_lin + orig_j] = entry;
                full[n_lin + orig_j][i] = entry;
            }
        }
        for i in 0..n_nl {
            for j in 0..n_nl {
                full[n_lin + i][n_lin + j] = nonlinear_cov[i][j];
            }
        }
        full
    } else {
        Vec::new()
    };

    FullCovarianceResult {
        plane_linear_cov,
        nonlinear_cov,
        single_plane_full_cov,
        jacobian_condition_number,
        rank_deficient,
    }
}

struct NdSolveOutput {
    layout: ParamLayout,
    theta: Vec<f64>,
    eval: VarProEval,
    cov: FullCovarianceResult,
    dof: usize,
    residual_sigma: f64,
    reduced_chi2: f64,
    min_pair_separation_fwhm: f64,
    degenerate_pairs: Vec<(usize, usize)>,
    active_bound_mask: Vec<bool>,
    active_bounds: Vec<ActiveBound>,
    model_misspecified: bool,
    crlb_valid: bool,
    iterations: usize,
    converged: bool,
    diagnostic: DeconvDiagnostic,
    all_diagnostics: Vec<DeconvDiagnostic>,
}

fn correlation_from_covariance(cov: &[Vec<f64>], k: usize) -> Vec<Vec<f64>> {
    let mut corr = vec![vec![0.0; k]; k];
    for i in 0..k {
        let vi = cov[i][i].max(0.0).sqrt();
        for j in 0..k {
            if i == j {
                corr[i][j] = 1.0;
            } else {
                let vj = cov[j][j].max(0.0).sqrt();
                let denom = vi * vj;
                if denom > 0.0 && denom.is_finite() {
                    corr[i][j] = (cov[i][j] / denom).clamp(-1.0, 1.0);
                }
            }
        }
    }
    corr
}

fn extract_peak_cov_submatrix(linear_cov: &[Vec<f64>], n_peaks: usize) -> Vec<Vec<f64>> {
    let mut out = vec![vec![0.0; n_peaks]; n_peaks];
    for i in 0..n_peaks {
        for j in 0..n_peaks {
            out[i][j] = linear_cov[i][j];
        }
    }
    out
}

#[allow(clippy::needless_range_loop)]
fn solve_varpro_nd(
    coords: &[&[f64]],
    planes: &[&[f64]],
    seed_centers: &[Vec<f64>],
    seed_fwhms: &[Vec<f64>],
    seed_etas: &[Vec<f64>],
    cfg: &DeconvConfig,
) -> Result<NdSolveOutput, DeconvError> {
    validate_config(cfg)?;
    let n_dims = coords.len();
    let n_planes = planes.len();
    let n_peaks = seed_centers.len();
    if n_planes == 0 {
        return Err(DeconvError::EmptyPlanes);
    }
    if n_planes > MAX_DECONV_PLANES {
        return Err(DeconvError::TooManyPlanes {
            planes: n_planes,
            max_planes: MAX_DECONV_PLANES,
        });
    }
    if n_peaks == 0 {
        return Err(DeconvError::EmptyPeaks);
    }
    if n_peaks > MAX_DECONV_PEAKS_PER_CLUSTER {
        return Err(DeconvError::TooManyPeaks {
            peaks: n_peaks,
            max_peaks: MAX_DECONV_PEAKS_PER_CLUSTER,
        });
    }

    let dim_lens: Vec<usize> = coords.iter().map(|c| c.len()).collect();
    let m = checked_voxel_count(&dim_lens)?;
    let axis_names = ["F1", "F2", "F3"];
    let mut axes = Vec::with_capacity(n_dims);
    for (d, axis_coords) in coords.iter().enumerate() {
        axes.push(validate_axis(axis_names[d], axis_coords)?);
    }

    // Check per-plane lengths in O(n_planes) before memory budget & voxel scan.
    for (p, &plane) in planes.iter().enumerate() {
        if plane.len() != m {
            return Err(DeconvError::DimensionMismatch(format!(
                "plane {p} has {} values, expected {m}",
                plane.len()
            )));
        }
    }

    // Check joint Jacobian / Hessian memory budget BEFORE scanning total_obs voxels or allocating
    // any solver arrays (closes 22.1:P1-1 with O(1) fail-closed rejection).
    let total_obs = n_planes.checked_mul(m).ok_or(DeconvError::OversizedRoi {
        voxels: usize::MAX,
        max_voxels: MAX_DECONV_VOXELS,
    })?;
    let expected_n_nl = expected_n_nonlinear(n_planes, n_peaks, n_dims, cfg).ok_or_else(|| {
        DeconvError::MemoryBudgetExceeded {
            matrix: "varpro_jacobian".to_string(),
            elements: usize::MAX,
            max_elements: MAX_DECONV_JACOBIAN_ELEMS,
        }
    })?;
    check_memory_budget(total_obs, expected_n_nl)?;

    // Validate plane values: finiteness and non-constant / non-zero energy.
    let mut global_min = f64::INFINITY;
    let mut global_max = f64::NEG_INFINITY;
    for (p, &plane) in planes.iter().enumerate() {
        for &v in plane {
            if !v.is_finite() {
                return Err(DeconvError::NonFiniteInput(format!("plane {p} voxel")));
            }
            if v < global_min {
                global_min = v;
            }
            if v > global_max {
                global_max = v;
            }
        }
    }
    let dynamic_range = global_max - global_min;
    let scale_mag = global_max.abs().max(global_min.abs()).max(1.0);
    if dynamic_range <= 1.0e-15 * scale_mag {
        return Err(DeconvError::ConstantOrZeroPatch);
    }

    // Validate seeds.
    for k in 0..n_peaks {
        for d in 0..n_dims {
            if !seed_centers[k][d].is_finite()
                || !seed_fwhms[k][d].is_finite()
                || !seed_etas[k][d].is_finite()
            {
                return Err(DeconvError::NonFiniteInput(format!(
                    "peak seed {k} axis {}",
                    axis_names[d]
                )));
            }
            if seed_fwhms[k][d] <= 0.0 {
                return Err(DeconvError::InvalidParameter(format!(
                    "peak seed {k} axis {} has non-positive FWHM",
                    axis_names[d]
                )));
            }
            if cfg.lineshape == LineshapeKind::PseudoVoigt
                && !(0.0..=1.0).contains(&seed_etas[k][d])
            {
                return Err(DeconvError::InvalidParameter(format!(
                    "peak seed {k} axis {} has eta outside [0, 1]",
                    axis_names[d]
                )));
            }
        }
    }

    let (layout, mut theta) = ParamLayout::build(
        n_planes,
        n_peaks,
        &axes,
        seed_centers,
        seed_fwhms,
        seed_etas,
        cfg,
    )?;

    check_memory_budget(total_obs, layout.n_nonlinear)?;

    let n_lin = n_peaks + usize::from(cfg.fit_baseline);
    let total_params = n_planes
        .checked_mul(n_lin)
        .and_then(|lin| lin.checked_add(layout.n_nonlinear))
        .ok_or(DeconvError::InsufficientDegreesOfFreedom {
            observations: total_obs,
            parameters: usize::MAX,
        })?;
    if total_obs <= total_params {
        return Err(DeconvError::InsufficientDegreesOfFreedom {
            observations: total_obs,
            parameters: total_params,
        });
    }
    let dof = total_obs - total_params;

    // Projected Levenberg–Marquardt with Nielsen trust-region damping on exact Golub–Pereyra VarPro.
    let n_nl = layout.n_nonlinear;
    let mut current = eval_varpro_system(coords, planes, &layout, &theta, cfg, true)?;
    let mut lambda = cfg.initial_lambda;
    let mut nu_up = 2.0_f64;
    let mut converged = false;
    let mut iterations = 0usize;

    for iter in 0..cfg.max_iterations {
        iterations = iter + 1;

        // Form H = J^T J (n_nl x n_nl) and g = J^T r (n_nl)
        let mut h = vec![vec![0.0; n_nl]; n_nl];
        let mut g = vec![0.0; n_nl];
        for i in 0..n_nl {
            let ji = &current.j_vp_cols[i];
            let mut gi = 0.0;
            for (p, r_p) in current.plane_residuals.iter().enumerate() {
                let off = p * m;
                for idx in 0..m {
                    gi += ji[off + idx] * r_p[idx];
                }
            }
            g[i] = gi;
            for j in i..n_nl {
                let jj = &current.j_vp_cols[j];
                let mut dot = 0.0;
                for idx in 0..ji.len() {
                    dot += ji[idx] * jj[idx];
                }
                h[i][j] = dot;
                h[j][i] = dot;
            }
        }

        // Check projected gradient optimality (respecting active box bounds).
        let r_norm = current.total_rss.sqrt();
        let mut max_scaled_grad = 0.0_f64;
        let mut d_sq = vec![0.0; n_nl];
        for i in 0..n_nl {
            let diag = h[i][i].max(1.0e-16);
            d_sq[i] = diag;
            let bound_tol = layout.bound_tol(i);
            let at_lo = theta[i] <= layout.lower[i] + bound_tol && g[i] < 0.0;
            let at_hi = theta[i] >= layout.upper[i] - bound_tol && g[i] > 0.0;
            if !at_lo && !at_hi {
                let sg = g[i].abs() / (diag.sqrt() * r_norm).max(1.0e-30);
                if sg > max_scaled_grad {
                    max_scaled_grad = sg;
                }
            }
        }
        if r_norm <= 1.0e-15 * scale_mag || max_scaled_grad <= cfg.grad_tol {
            converged = true;
            break;
        }

        // Inner damping loop to find a strictly improving projected trial step.
        let mut accepted = false;
        for _inner in 0..24 {
            let step = solve_damped_normal_eqs(&h, &g, &d_sq, lambda);
            let mut trial_theta = vec![0.0; n_nl];
            let mut actual_step = vec![0.0; n_nl];
            let mut max_rel_step = 0.0_f64;
            for i in 0..n_nl {
                let t = (theta[i] + step[i]).clamp(layout.lower[i], layout.upper[i]);
                trial_theta[i] = t;
                let ds = t - theta[i];
                actual_step[i] = ds;
                let rel = ds.abs() / theta[i].abs().max(layout.scales[i]).max(1.0e-12);
                if rel > max_rel_step {
                    max_rel_step = rel;
                }
            }

            if max_rel_step <= cfg.param_tol {
                // Step is negligible even under current damping: converged.
                let trial_eval =
                    eval_varpro_system(coords, planes, &layout, &trial_theta, cfg, false)?;
                if trial_eval.total_rss <= current.total_rss {
                    theta = trial_theta;
                    current = trial_eval;
                }
                converged = true;
                accepted = true;
                break;
            }

            let trial_eval = eval_varpro_system(coords, planes, &layout, &trial_theta, cfg, false)?;
            let rss_diff = current.total_rss - trial_eval.total_rss;

            // Predicted reduction from quadratic model: 2 g^T dx - dx^T H dx
            let mut pred_red = 0.0;
            for i in 0..n_nl {
                let mut h_dx_i = 0.0;
                for j in 0..n_nl {
                    h_dx_i += h[i][j] * actual_step[j];
                }
                pred_red += actual_step[i] * (2.0 * g[i] - h_dx_i);
            }
            if pred_red <= 0.0 {
                // Fallback positive lower bound for gain ratio denominator
                let mut damp_term = 0.0;
                for i in 0..n_nl {
                    damp_term += actual_step[i] * (lambda * d_sq[i] * actual_step[i] + g[i]);
                }
                pred_red = damp_term.max(1.0e-30);
            }

            if rss_diff > 0.0 {
                let rel_cost = rss_diff / current.total_rss.max(1.0e-30);
                let rho = rss_diff / pred_red;
                theta = trial_theta;
                lambda *= (1.0_f64 / 3.0).max(1.0 - (2.0 * rho - 1.0).powi(3));
                lambda = lambda.clamp(1.0e-12, 1.0e12);
                nu_up = 2.0;
                accepted = true;

                if rel_cost <= cfg.cost_tol || max_rel_step <= cfg.param_tol {
                    current = trial_eval;
                    converged = true;
                } else {
                    current = eval_varpro_system(coords, planes, &layout, &theta, cfg, true)?;
                }
                break;
            } else {
                lambda *= nu_up;
                nu_up *= 2.0;
                if lambda > 1.0e12 {
                    break;
                }
            }
        }

        if converged {
            break;
        }
        if !accepted {
            // No improving step could be found even at extreme damping; check if already at stationary point.
            if max_scaled_grad <= cfg.grad_tol * 100.0 {
                converged = true;
            }
            break;
        }
    }

    // Detect any nonlinear parameters sitting on an active box bound.
    let mut active_bound_mask = vec![false; n_nl];
    let mut active_bounds = Vec::new();
    for i in 0..n_nl {
        let tol = layout.bound_tol(i);
        if theta[i] <= layout.lower[i] + tol {
            active_bound_mask[i] = true;
            active_bounds.push(ActiveBound {
                param_index: i,
                param_name: layout.names[i].clone(),
                side: BoundSide::Lower,
                value: theta[i],
                limit: layout.lower[i],
            });
        } else if theta[i] >= layout.upper[i] - tol {
            active_bound_mask[i] = true;
            active_bounds.push(ActiveBound {
                param_index: i,
                param_name: layout.names[i].clone(),
                side: BoundSide::Upper,
                value: theta[i],
                limit: layout.upper[i],
            });
        }
    }

    // Final residual & variance statistics.
    let residual_var = current.total_rss / (dof as f64);
    let residual_sigma = residual_var.max(0.0).sqrt();
    let (cov_sigma_sq, reduced_chi2) = if let Some(known_sigma) = cfg.noise_sigma {
        let s2 = known_sigma * known_sigma;
        (s2, residual_var / s2)
    } else {
        (residual_var, 1.0)
    };

    // Known-sigma goodness-of-fit gate (upper-tail chi^2_0.999(dof)/dof or configured override).
    let model_misspecified = if cfg.noise_sigma.is_some() {
        let chi2_limit = cfg
            .max_reduced_chi2
            .unwrap_or_else(|| chi2_reduced_critical_0999(dof));
        !reduced_chi2.is_finite() || reduced_chi2 > chi2_limit
    } else {
        false
    };

    let cov = compute_full_crlb_covariance(
        coords,
        &layout,
        &theta,
        &current.plane_amplitudes,
        &active_bound_mask,
        cov_sigma_sq,
        cfg,
    );

    // Pairwise normalized center separation check for degenerate peak pairs.
    let mut min_pair_sep = f64::INFINITY;
    let mut degenerate_pairs = Vec::new();
    for i in 0..n_peaks {
        for j in (i + 1)..n_peaks {
            let mut sq_sum = 0.0;
            for d in 0..n_dims {
                let ci = layout.center(&theta, i, d);
                let cj = layout.center(&theta, j, d);
                let mut f_mean_i = 0.0;
                let mut f_mean_j = 0.0;
                for p in 0..n_planes {
                    f_mean_i += layout.fwhm(&theta, p, i, d);
                    f_mean_j += layout.fwhm(&theta, p, j, d);
                }
                let mean_fwhm = 0.5 * (f_mean_i + f_mean_j) / (n_planes as f64);
                let ratio = (ci - cj) / mean_fwhm.max(1.0e-12);
                sq_sum += ratio * ratio;
            }
            let sep = sq_sum.sqrt();
            if sep < min_pair_sep {
                min_pair_sep = sep;
            }
            if sep < cfg.degenerate_separation_frac {
                degenerate_pairs.push((i, j));
            }
        }
    }

    let mut all_diagnostics = Vec::new();
    if !degenerate_pairs.is_empty() {
        all_diagnostics.push(DeconvDiagnostic::DegeneratePeakPair);
    }
    let ill_cond = !current.max_basis_kappa.is_finite()
        || current.max_basis_kappa > cfg.ill_conditioned_kappa
        || !cov.jacobian_condition_number.is_finite()
        || cov.jacobian_condition_number > cfg.ill_conditioned_kappa * 10.0
        || cov.rank_deficient;
    if ill_cond {
        all_diagnostics.push(DeconvDiagnostic::IllConditionedCluster);
    }
    if !converged {
        all_diagnostics.push(DeconvDiagnostic::NonConverged);
    }
    if !active_bounds.is_empty() {
        all_diagnostics.push(DeconvDiagnostic::ActiveBoundHit);
    }
    if model_misspecified {
        all_diagnostics.push(DeconvDiagnostic::ModelMisspecified { reduced_chi2 });
    }
    if all_diagnostics.is_empty() {
        all_diagnostics.push(DeconvDiagnostic::Converged);
    }
    let diagnostic = all_diagnostics[0];
    let crlb_valid =
        all_diagnostics.len() == 1 && all_diagnostics[0] == DeconvDiagnostic::Converged;

    Ok(NdSolveOutput {
        layout,
        theta,
        eval: current,
        cov,
        dof,
        residual_sigma,
        reduced_chi2,
        min_pair_separation_fwhm: min_pair_sep,
        degenerate_pairs,
        active_bound_mask,
        active_bounds,
        model_misspecified,
        crlb_valid,
        iterations,
        converged,
        diagnostic,
        all_diagnostics,
    })
}

struct PeakVolumeSpec<'a> {
    kind: LineshapeKind,
    amplitude: f64,
    peak_idx: usize,
    n_lin: usize,
    fwhms: &'a [f64],
    etas: &'a [f64],
    fwhm_nl_indices: &'a [usize],
    eta_nl_indices: &'a [Option<usize>],
    active_bound_mask: &'a [bool],
    suppress_se: bool,
    full_cov: &'a [Vec<f64>],
}

/// Delta-method volume standard error for a single-plane peak from the full joint covariance.
/// Returns `None` for the SE if `suppress_se` is set or if any FWHM / fitted mixing parameter
/// contributing to the peak's integral sits on an active box bound.
fn peak_volume_and_se(spec: &PeakVolumeSpec<'_>) -> (f64, Option<f64>) {
    let mut total_area = 1.0;
    for (&fwhm, &eta) in spec.fwhms.iter().zip(spec.etas.iter()) {
        total_area *= lineshape_area_1d(spec.kind, fwhm, eta);
    }
    let volume = spec.amplitude * total_area;
    let total_params = spec.full_cov.len();
    if spec.suppress_se || total_params == 0 {
        return (volume, None);
    }
    for &fi in spec.fwhm_nl_indices {
        if spec.active_bound_mask.get(fi).copied().unwrap_or(false) {
            return (volume, None);
        }
    }
    if spec.kind == LineshapeKind::PseudoVoigt {
        for &ei_opt in spec.eta_nl_indices {
            if let Some(ei) = ei_opt {
                if spec.active_bound_mask.get(ei).copied().unwrap_or(false) {
                    return (volume, None);
                }
            }
        }
    }
    let mut grad = vec![0.0; total_params];
    grad[spec.peak_idx] = total_area;

    let lor_coeff = 0.5 * PI;
    let gau_coeff = 0.5 * (PI / LN_2).sqrt();
    for d in 0..spec.fwhms.len() {
        let f_idx = spec.n_lin + spec.fwhm_nl_indices[d];
        if f_idx < total_params && spec.fwhms[d] > 0.0 {
            grad[f_idx] += volume / spec.fwhms[d];
        }
        if spec.kind == LineshapeKind::PseudoVoigt {
            if let Some(ei) = spec.eta_nl_indices[d] {
                let e_idx = spec.n_lin + ei;
                if e_idx < total_params {
                    let denom = spec.etas[d] * lor_coeff + (1.0 - spec.etas[d]) * gau_coeff;
                    if denom > 0.0 {
                        grad[e_idx] += volume * (lor_coeff - gau_coeff) / denom;
                    }
                }
            }
        }
    }

    let mut var = 0.0;
    for i in 0..total_params {
        if grad[i] == 0.0 {
            continue;
        }
        for j in 0..total_params {
            if grad[j] == 0.0 {
                continue;
            }
            var += grad[i] * spec.full_cov[i][j] * grad[j];
        }
    }
    let se = if var.is_finite() {
        Some(var.max(0.0).sqrt())
    } else {
        None
    };
    (volume, se)
}

#[inline]
fn checked_se_from_var(var: f64) -> Option<f64> {
    if var.is_finite() {
        Some(var.max(0.0).sqrt())
    } else {
        None
    }
}

// ============================================================================
// Public Entry Points
// ============================================================================

fn validate_seeds_and_values(
    values: &[f64],
    centers: &[Vec<f64>],
    fwhms: &[Vec<f64>],
    etas: &[Vec<f64>],
    cfg: &DeconvConfig,
) -> Result<(), DeconvError> {
    if centers.is_empty() {
        return Err(DeconvError::EmptyPeaks);
    }
    if centers.len() > MAX_DECONV_PEAKS_PER_CLUSTER {
        return Err(DeconvError::TooManyPeaks {
            peaks: centers.len(),
            max_peaks: MAX_DECONV_PEAKS_PER_CLUSTER,
        });
    }
    for &v in values {
        if !v.is_finite() {
            return Err(DeconvError::NonFiniteInput("ROI voxel".to_string()));
        }
    }
    let axis_names = ["F1", "F2", "F3"];
    for k in 0..centers.len() {
        for d in 0..centers[k].len() {
            if !centers[k][d].is_finite() || !fwhms[k][d].is_finite() || !etas[k][d].is_finite() {
                return Err(DeconvError::NonFiniteInput(format!(
                    "peak seed {k} axis {}",
                    axis_names[d]
                )));
            }
            if fwhms[k][d] <= 0.0 {
                return Err(DeconvError::InvalidParameter(format!(
                    "peak seed {k} axis {} has non-positive FWHM",
                    axis_names[d]
                )));
            }
            if cfg.lineshape == LineshapeKind::PseudoVoigt && !(0.0..=1.0).contains(&etas[k][d]) {
                return Err(DeconvError::InvalidParameter(format!(
                    "peak seed {k} axis {} has eta outside [0, 1]",
                    axis_names[d]
                )));
            }
        }
    }
    Ok(())
}

/// Evaluate the exact VarPro projected model $\hat{\mathbf{y}}(\boldsymbol{\theta})$, residual
/// $\mathbf{r} = \mathbf{y} - \hat{\mathbf{y}}$, exact Golub–Pereyra VarPro Jacobian columns
/// $\partial \hat{\mathbf{y}} / \partial \theta_j$, and basis condition number $\kappa(\Phi)$
/// for a 2D ROI at the supplied peak seeds.
#[allow(clippy::type_complexity)]
pub fn eval_varpro_residual_and_jacobian_2d(
    roi: &Roi2D,
    seeds: &[PeakSeed2D],
    config: &DeconvConfig,
) -> Result<(Vec<f64>, Vec<f64>, Vec<Vec<f64>>, f64), DeconvError> {
    validate_config(config)?;
    let m = checked_voxel_count(&[roi.coords_f1.len(), roi.coords_f2.len()])?;
    if roi.values.len() != m {
        return Err(DeconvError::DimensionMismatch(format!(
            "expected {m} values, got {}",
            roi.values.len()
        )));
    }
    if seeds.is_empty() {
        return Err(DeconvError::EmptyPeaks);
    }
    if seeds.len() > MAX_DECONV_PEAKS_PER_CLUSTER {
        return Err(DeconvError::TooManyPeaks {
            peaks: seeds.len(),
            max_peaks: MAX_DECONV_PEAKS_PER_CLUSTER,
        });
    }
    let ax1 = validate_axis("F1", &roi.coords_f1)?;
    let ax2 = validate_axis("F2", &roi.coords_f2)?;
    let expected_n_nl = expected_n_nonlinear(1, seeds.len(), 2, config).ok_or_else(|| {
        DeconvError::MemoryBudgetExceeded {
            matrix: "varpro_jacobian".to_string(),
            elements: usize::MAX,
            max_elements: MAX_DECONV_JACOBIAN_ELEMS,
        }
    })?;
    check_memory_budget(m, expected_n_nl)?;
    let centers: Vec<Vec<f64>> = seeds
        .iter()
        .map(|s| vec![s.center_f1, s.center_f2])
        .collect();
    let fwhms: Vec<Vec<f64>> = seeds.iter().map(|s| vec![s.fwhm_f1, s.fwhm_f2]).collect();
    let etas: Vec<Vec<f64>> = seeds.iter().map(|s| vec![s.eta_f1, s.eta_f2]).collect();
    validate_seeds_and_values(&roi.values, &centers, &fwhms, &etas, config)?;
    let n_lin = seeds.len() + usize::from(config.fit_baseline);
    if m < n_lin {
        return Err(DeconvError::InsufficientDegreesOfFreedom {
            observations: m,
            parameters: n_lin,
        });
    }
    let (layout, theta) =
        ParamLayout::build(1, seeds.len(), &[ax1, ax2], &centers, &fwhms, &etas, config)?;
    let coords: [&[f64]; 2] = [&roi.coords_f1, &roi.coords_f2];
    let planes: [&[f64]; 1] = [&roi.values];
    let eval = eval_varpro_system(&coords, &planes, &layout, &theta, config, true)?;
    Ok((
        eval.plane_fitted[0].clone(),
        eval.plane_residuals[0].clone(),
        eval.j_vp_cols,
        eval.max_basis_kappa,
    ))
}

/// Evaluate the exact VarPro projected model, residual, exact Golub–Pereyra VarPro Jacobian
/// columns, and basis condition number for a 3D ROI at the supplied peak seeds.
#[allow(clippy::type_complexity)]
pub fn eval_varpro_residual_and_jacobian_3d(
    roi: &Roi3D,
    seeds: &[PeakSeed3D],
    config: &DeconvConfig,
) -> Result<(Vec<f64>, Vec<f64>, Vec<Vec<f64>>, f64), DeconvError> {
    validate_config(config)?;
    let m = checked_voxel_count(&[
        roi.coords_f1.len(),
        roi.coords_f2.len(),
        roi.coords_f3.len(),
    ])?;
    if roi.values.len() != m {
        return Err(DeconvError::DimensionMismatch(format!(
            "expected {m} values, got {}",
            roi.values.len()
        )));
    }
    if seeds.is_empty() {
        return Err(DeconvError::EmptyPeaks);
    }
    if seeds.len() > MAX_DECONV_PEAKS_PER_CLUSTER {
        return Err(DeconvError::TooManyPeaks {
            peaks: seeds.len(),
            max_peaks: MAX_DECONV_PEAKS_PER_CLUSTER,
        });
    }
    let ax1 = validate_axis("F1", &roi.coords_f1)?;
    let ax2 = validate_axis("F2", &roi.coords_f2)?;
    let ax3 = validate_axis("F3", &roi.coords_f3)?;
    let expected_n_nl = expected_n_nonlinear(1, seeds.len(), 3, config).ok_or_else(|| {
        DeconvError::MemoryBudgetExceeded {
            matrix: "varpro_jacobian".to_string(),
            elements: usize::MAX,
            max_elements: MAX_DECONV_JACOBIAN_ELEMS,
        }
    })?;
    check_memory_budget(m, expected_n_nl)?;
    let centers: Vec<Vec<f64>> = seeds
        .iter()
        .map(|s| vec![s.center_f1, s.center_f2, s.center_f3])
        .collect();
    let fwhms: Vec<Vec<f64>> = seeds
        .iter()
        .map(|s| vec![s.fwhm_f1, s.fwhm_f2, s.fwhm_f3])
        .collect();
    let etas: Vec<Vec<f64>> = seeds
        .iter()
        .map(|s| vec![s.eta_f1, s.eta_f2, s.eta_f3])
        .collect();
    validate_seeds_and_values(&roi.values, &centers, &fwhms, &etas, config)?;
    let n_lin = seeds.len() + usize::from(config.fit_baseline);
    if m < n_lin {
        return Err(DeconvError::InsufficientDegreesOfFreedom {
            observations: m,
            parameters: n_lin,
        });
    }
    let (layout, theta) = ParamLayout::build(
        1,
        seeds.len(),
        &[ax1, ax2, ax3],
        &centers,
        &fwhms,
        &etas,
        config,
    )?;
    let coords: [&[f64]; 3] = [&roi.coords_f1, &roi.coords_f2, &roi.coords_f3];
    let planes: [&[f64]; 1] = [&roi.values];
    let eval = eval_varpro_system(&coords, &planes, &layout, &theta, config, true)?;
    Ok((
        eval.plane_fitted[0].clone(),
        eval.plane_residuals[0].clone(),
        eval.j_vp_cols,
        eval.max_basis_kappa,
    ))
}

/// Deconvolve a crowded 2D `(F1, F2)` ROI patch via separable Variable Projection (`VarPro`)
/// nonlinear least squares with exact full Cramér–Rao amplitude covariance.
pub fn deconvolve_roi_2d(
    roi: &Roi2D,
    seeds: &[PeakSeed2D],
    config: &DeconvConfig,
) -> Result<DeconvResult2D, DeconvError> {
    let seed_centers: Vec<Vec<f64>> = seeds
        .iter()
        .map(|s| vec![s.center_f1, s.center_f2])
        .collect();
    let seed_fwhms: Vec<Vec<f64>> = seeds.iter().map(|s| vec![s.fwhm_f1, s.fwhm_f2]).collect();
    let seed_etas: Vec<Vec<f64>> = seeds.iter().map(|s| vec![s.eta_f1, s.eta_f2]).collect();
    let coords: [&[f64]; 2] = [&roi.coords_f1, &roi.coords_f2];
    let planes: [&[f64]; 1] = [&roi.values];

    let out = solve_varpro_nd(
        &coords,
        &planes,
        &seed_centers,
        &seed_fwhms,
        &seed_etas,
        config,
    )?;
    let n_peaks = seeds.len();
    let n_lin = n_peaks + usize::from(config.fit_baseline);
    let amps = &out.eval.plane_amplitudes[0];
    let lin_cov = &out.cov.plane_linear_cov[0];
    let nl_cov = &out.cov.nonlinear_cov;
    let suppress_all_se = out.model_misspecified || out.cov.rank_deficient || !out.converged;
    let nl_se = |idx: usize| -> Option<f64> {
        if suppress_all_se || out.active_bound_mask[idx] {
            None
        } else {
            checked_se_from_var(nl_cov[idx][idx])
        }
    };
    let lin_se = |idx: usize| -> Option<f64> {
        if suppress_all_se {
            None
        } else {
            checked_se_from_var(lin_cov[idx][idx])
        }
    };

    let mut peaks = Vec::with_capacity(n_peaks);
    for (k, &amp) in amps.iter().enumerate().take(n_peaks) {
        let c1_i = out.layout.center_idx[k][0];
        let c2_i = out.layout.center_idx[k][1];
        let f1_i = out.layout.fwhm_idx[0][k][0];
        let f2_i = out.layout.fwhm_idx[0][k][1];
        let e1_opt = out.layout.eta_idx[k][0];
        let e2_opt = out.layout.eta_idx[k][1];

        let f1 = out.theta[f1_i];
        let f2 = out.theta[f2_i];
        let e1 = out.layout.eta(&out.theta, k, 0);
        let e2 = out.layout.eta(&out.theta, k, 1);
        let (volume, volume_se) = peak_volume_and_se(&PeakVolumeSpec {
            kind: config.lineshape,
            amplitude: amp,
            peak_idx: k,
            n_lin,
            fwhms: &[f1, f2],
            etas: &[e1, e2],
            fwhm_nl_indices: &[f1_i, f2_i],
            eta_nl_indices: &[e1_opt, e2_opt],
            active_bound_mask: &out.active_bound_mask,
            suppress_se: suppress_all_se,
            full_cov: &out.cov.single_plane_full_cov,
        });

        peaks.push(FittedPeak2D {
            amplitude: amp,
            amplitude_se: lin_se(k),
            volume,
            volume_se,
            center_f1: out.theta[c1_i],
            center_f1_se: nl_se(c1_i),
            center_f2: out.theta[c2_i],
            center_f2_se: nl_se(c2_i),
            fwhm_f1: f1,
            fwhm_f1_se: nl_se(f1_i),
            fwhm_f2: f2,
            fwhm_f2_se: nl_se(f2_i),
            eta_f1: e1,
            eta_f1_se: e1_opt.and_then(nl_se),
            eta_f2: e2,
            eta_f2_se: e2_opt.and_then(nl_se),
        });
    }

    let (baseline, baseline_se) = if config.fit_baseline {
        (amps[n_peaks], lin_se(n_peaks))
    } else {
        (0.0, None)
    };
    let (amplitude_covariance, amplitude_correlation, full_covariance) = if out.crlb_valid {
        let amp_cov = extract_peak_cov_submatrix(lin_cov, n_peaks);
        let amp_corr = correlation_from_covariance(&amp_cov, n_peaks);
        (
            Some(amp_cov),
            Some(amp_corr),
            Some(out.cov.single_plane_full_cov),
        )
    } else {
        (None, None, None)
    };

    Ok(DeconvResult2D {
        peaks,
        baseline,
        baseline_se,
        amplitude_covariance,
        amplitude_correlation,
        full_covariance,
        rss: out.eval.total_rss,
        dof: out.dof,
        residual_sigma: out.residual_sigma,
        reduced_chi2: out.reduced_chi2,
        basis_condition_number: out.eval.max_basis_kappa,
        jacobian_condition_number: out.cov.jacobian_condition_number,
        min_pair_separation_fwhm: out.min_pair_separation_fwhm,
        degenerate_pairs: out.degenerate_pairs,
        active_bounds: out.active_bounds,
        crlb_valid: out.crlb_valid,
        iterations: out.iterations,
        converged: out.converged,
        diagnostic: out.diagnostic,
        all_diagnostics: out.all_diagnostics,
        fitted_values: out.eval.plane_fitted[0].clone(),
        residuals: out.eval.plane_residuals[0].clone(),
    })
}

/// Deconvolve a crowded 3D `(F1, F2, F3)` ROI patch via separable Variable Projection (`VarPro`)
/// nonlinear least squares with exact full Cramér–Rao amplitude covariance.
pub fn deconvolve_roi_3d(
    roi: &Roi3D,
    seeds: &[PeakSeed3D],
    config: &DeconvConfig,
) -> Result<DeconvResult3D, DeconvError> {
    let seed_centers: Vec<Vec<f64>> = seeds
        .iter()
        .map(|s| vec![s.center_f1, s.center_f2, s.center_f3])
        .collect();
    let seed_fwhms: Vec<Vec<f64>> = seeds
        .iter()
        .map(|s| vec![s.fwhm_f1, s.fwhm_f2, s.fwhm_f3])
        .collect();
    let seed_etas: Vec<Vec<f64>> = seeds
        .iter()
        .map(|s| vec![s.eta_f1, s.eta_f2, s.eta_f3])
        .collect();
    let coords: [&[f64]; 3] = [&roi.coords_f1, &roi.coords_f2, &roi.coords_f3];
    let planes: [&[f64]; 1] = [&roi.values];

    let out = solve_varpro_nd(
        &coords,
        &planes,
        &seed_centers,
        &seed_fwhms,
        &seed_etas,
        config,
    )?;
    let n_peaks = seeds.len();
    let n_lin = n_peaks + usize::from(config.fit_baseline);
    let amps = &out.eval.plane_amplitudes[0];
    let lin_cov = &out.cov.plane_linear_cov[0];
    let nl_cov = &out.cov.nonlinear_cov;
    let suppress_all_se = out.model_misspecified || out.cov.rank_deficient || !out.converged;
    let nl_se = |idx: usize| -> Option<f64> {
        if suppress_all_se || out.active_bound_mask[idx] {
            None
        } else {
            checked_se_from_var(nl_cov[idx][idx])
        }
    };
    let lin_se = |idx: usize| -> Option<f64> {
        if suppress_all_se {
            None
        } else {
            checked_se_from_var(lin_cov[idx][idx])
        }
    };

    let mut peaks = Vec::with_capacity(n_peaks);
    for (k, &amp) in amps.iter().enumerate().take(n_peaks) {
        let c1_i = out.layout.center_idx[k][0];
        let c2_i = out.layout.center_idx[k][1];
        let c3_i = out.layout.center_idx[k][2];
        let f1_i = out.layout.fwhm_idx[0][k][0];
        let f2_i = out.layout.fwhm_idx[0][k][1];
        let f3_i = out.layout.fwhm_idx[0][k][2];
        let e1_opt = out.layout.eta_idx[k][0];
        let e2_opt = out.layout.eta_idx[k][1];
        let e3_opt = out.layout.eta_idx[k][2];

        let f1 = out.theta[f1_i];
        let f2 = out.theta[f2_i];
        let f3 = out.theta[f3_i];
        let e1 = out.layout.eta(&out.theta, k, 0);
        let e2 = out.layout.eta(&out.theta, k, 1);
        let e3 = out.layout.eta(&out.theta, k, 2);
        let (volume, volume_se) = peak_volume_and_se(&PeakVolumeSpec {
            kind: config.lineshape,
            amplitude: amp,
            peak_idx: k,
            n_lin,
            fwhms: &[f1, f2, f3],
            etas: &[e1, e2, e3],
            fwhm_nl_indices: &[f1_i, f2_i, f3_i],
            eta_nl_indices: &[e1_opt, e2_opt, e3_opt],
            active_bound_mask: &out.active_bound_mask,
            suppress_se: suppress_all_se,
            full_cov: &out.cov.single_plane_full_cov,
        });

        peaks.push(FittedPeak3D {
            amplitude: amp,
            amplitude_se: lin_se(k),
            volume,
            volume_se,
            center_f1: out.theta[c1_i],
            center_f1_se: nl_se(c1_i),
            center_f2: out.theta[c2_i],
            center_f2_se: nl_se(c2_i),
            center_f3: out.theta[c3_i],
            center_f3_se: nl_se(c3_i),
            fwhm_f1: f1,
            fwhm_f1_se: nl_se(f1_i),
            fwhm_f2: f2,
            fwhm_f2_se: nl_se(f2_i),
            fwhm_f3: f3,
            fwhm_f3_se: nl_se(f3_i),
            eta_f1: e1,
            eta_f1_se: e1_opt.and_then(nl_se),
            eta_f2: e2,
            eta_f2_se: e2_opt.and_then(nl_se),
            eta_f3: e3,
            eta_f3_se: e3_opt.and_then(nl_se),
        });
    }

    let (baseline, baseline_se) = if config.fit_baseline {
        (amps[n_peaks], lin_se(n_peaks))
    } else {
        (0.0, None)
    };
    let (amplitude_covariance, amplitude_correlation, full_covariance) = if out.crlb_valid {
        let amp_cov = extract_peak_cov_submatrix(lin_cov, n_peaks);
        let amp_corr = correlation_from_covariance(&amp_cov, n_peaks);
        (
            Some(amp_cov),
            Some(amp_corr),
            Some(out.cov.single_plane_full_cov),
        )
    } else {
        (None, None, None)
    };

    Ok(DeconvResult3D {
        peaks,
        baseline,
        baseline_se,
        amplitude_covariance,
        amplitude_correlation,
        full_covariance,
        rss: out.eval.total_rss,
        dof: out.dof,
        residual_sigma: out.residual_sigma,
        reduced_chi2: out.reduced_chi2,
        basis_condition_number: out.eval.max_basis_kappa,
        jacobian_condition_number: out.cov.jacobian_condition_number,
        min_pair_separation_fwhm: out.min_pair_separation_fwhm,
        degenerate_pairs: out.degenerate_pairs,
        active_bounds: out.active_bounds,
        crlb_valid: out.crlb_valid,
        iterations: out.iterations,
        converged: out.converged,
        diagnostic: out.diagnostic,
        all_diagnostics: out.all_diagnostics,
        fitted_values: out.eval.plane_fitted[0].clone(),
        residuals: out.eval.plane_residuals[0].clone(),
    })
}

/// Jointly deconvolve a coupled pseudo-3D 2D plane series (`P >= 1` planes, e.g. $R_1$, $R_2$,
/// CPMG, CEST, or titration planes) where peak positions and shared lineshape parameters are
/// refined simultaneously across all planes while per-plane peak amplitudes $\mathbf{a}^{(p)}$
/// are solved linearly in closed form at every VarPro Levenberg–Marquardt iteration.
pub fn deconvolve_plane_series_2d(
    series: &PlaneSeries2D,
    seeds: &[PeakSeed2D],
    config: &DeconvConfig,
) -> Result<PlaneSeriesDeconvResult, DeconvError> {
    if series.plane_values.is_empty() {
        return Err(DeconvError::EmptyPlanes);
    }
    if !series.plane_coords.is_empty() && series.plane_coords.len() != series.plane_values.len() {
        return Err(DeconvError::DimensionMismatch(format!(
            "plane_coords len {} != plane_values len {}",
            series.plane_coords.len(),
            series.plane_values.len()
        )));
    }
    for (p, &c) in series.plane_coords.iter().enumerate() {
        if !c.is_finite() {
            return Err(DeconvError::NonFiniteInput(format!("plane_coords[{p}]")));
        }
    }

    let seed_centers: Vec<Vec<f64>> = seeds
        .iter()
        .map(|s| vec![s.center_f1, s.center_f2])
        .collect();
    let seed_fwhms: Vec<Vec<f64>> = seeds.iter().map(|s| vec![s.fwhm_f1, s.fwhm_f2]).collect();
    let seed_etas: Vec<Vec<f64>> = seeds.iter().map(|s| vec![s.eta_f1, s.eta_f2]).collect();
    let coords: [&[f64]; 2] = [&series.coords_f1, &series.coords_f2];
    let planes: Vec<&[f64]> = series.plane_values.iter().map(Vec::as_slice).collect();

    let out = solve_varpro_nd(
        &coords,
        &planes,
        &seed_centers,
        &seed_fwhms,
        &seed_etas,
        config,
    )?;
    let n_planes = series.plane_values.len();
    let n_peaks = seeds.len();
    let nl_cov = &out.cov.nonlinear_cov;
    let suppress_all_se = out.model_misspecified || out.cov.rank_deficient || !out.converged;
    let nl_se = |idx: usize| -> Option<f64> {
        if suppress_all_se || out.active_bound_mask[idx] {
            None
        } else {
            checked_se_from_var(nl_cov[idx][idx])
        }
    };
    let plane_lin_se = |p: usize, idx: usize| -> Option<f64> {
        if suppress_all_se {
            None
        } else {
            checked_se_from_var(out.cov.plane_linear_cov[p][idx][idx])
        }
    };

    let mut peaks = Vec::with_capacity(n_peaks);
    for k in 0..n_peaks {
        let c1_i = out.layout.center_idx[k][0];
        let c2_i = out.layout.center_idx[k][1];
        let e1_opt = out.layout.eta_idx[k][0];
        let e2_opt = out.layout.eta_idx[k][1];
        let e1 = out.layout.eta(&out.theta, k, 0);
        let e2 = out.layout.eta(&out.theta, k, 1);

        let mut fwhm_f1 = Vec::with_capacity(n_planes);
        let mut fwhm_f1_se = Vec::with_capacity(n_planes);
        let mut fwhm_f2 = Vec::with_capacity(n_planes);
        let mut fwhm_f2_se = Vec::with_capacity(n_planes);
        let mut amplitudes = Vec::with_capacity(n_planes);
        let mut amplitude_se = Vec::with_capacity(n_planes);
        let mut volumes = Vec::with_capacity(n_planes);

        for p in 0..n_planes {
            let f1_i = out.layout.fwhm_idx[p][k][0];
            let f2_i = out.layout.fwhm_idx[p][k][1];
            let f1 = out.theta[f1_i];
            let f2 = out.theta[f2_i];
            let amp = out.eval.plane_amplitudes[p][k];
            let amp_se = plane_lin_se(p, k);
            let vol = amp
                * lineshape_area_1d(config.lineshape, f1, e1)
                * lineshape_area_1d(config.lineshape, f2, e2);

            fwhm_f1.push(f1);
            fwhm_f1_se.push(nl_se(f1_i));
            fwhm_f2.push(f2);
            fwhm_f2_se.push(nl_se(f2_i));
            amplitudes.push(amp);
            amplitude_se.push(amp_se);
            volumes.push(vol);
        }

        peaks.push(PlanePeakSeries {
            center_f1: out.theta[c1_i],
            center_f1_se: nl_se(c1_i),
            center_f2: out.theta[c2_i],
            center_f2_se: nl_se(c2_i),
            fwhm_f1,
            fwhm_f1_se,
            fwhm_f2,
            fwhm_f2_se,
            eta_f1: e1,
            eta_f1_se: e1_opt.and_then(nl_se),
            eta_f2: e2,
            eta_f2_se: e2_opt.and_then(nl_se),
            amplitudes,
            amplitude_se,
            volumes,
        });
    }

    let mut baselines = Vec::with_capacity(n_planes);
    let mut baseline_se = Vec::with_capacity(n_planes);

    for p in 0..n_planes {
        if config.fit_baseline {
            baselines.push(out.eval.plane_amplitudes[p][n_peaks]);
            baseline_se.push(plane_lin_se(p, n_peaks));
        } else {
            baselines.push(0.0);
            baseline_se.push(None);
        }
    }

    let (plane_amplitude_covariances, plane_amplitude_correlations) = if out.crlb_valid {
        let mut covs = Vec::with_capacity(n_planes);
        let mut corrs = Vec::with_capacity(n_planes);
        for p in 0..n_planes {
            let lin_cov = &out.cov.plane_linear_cov[p];
            let amp_cov = extract_peak_cov_submatrix(lin_cov, n_peaks);
            let amp_corr = correlation_from_covariance(&amp_cov, n_peaks);
            covs.push(amp_cov);
            corrs.push(amp_corr);
        }
        (Some(covs), Some(corrs))
    } else {
        (None, None)
    };

    Ok(PlaneSeriesDeconvResult {
        peaks,
        baselines,
        baseline_se,
        plane_amplitude_covariances,
        plane_amplitude_correlations,
        rss: out.eval.total_rss,
        plane_rss: out.eval.plane_rss,
        dof: out.dof,
        residual_sigma: out.residual_sigma,
        reduced_chi2: out.reduced_chi2,
        basis_condition_number: out.eval.max_basis_kappa,
        jacobian_condition_number: out.cov.jacobian_condition_number,
        min_pair_separation_fwhm: out.min_pair_separation_fwhm,
        degenerate_pairs: out.degenerate_pairs,
        active_bounds: out.active_bounds,
        crlb_valid: out.crlb_valid,
        iterations: out.iterations,
        converged: out.converged,
        diagnostic: out.diagnostic,
        all_diagnostics: out.all_diagnostics,
    })
}
