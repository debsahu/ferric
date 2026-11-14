//! Two-state (`7x7`) and three-state (`10x10`) homogeneous Bloch–McConnell Chemical Exchange
//! Saturation Transfer (`CEST`) and Dark-state Exchange Saturation Transfer (`DEST`) matrix-exponential
//! forward engine, Gauss–Hermite `B1`-inhomogeneity quadrature, Schur-complement global
//! Levenberg–Marquardt fitter, `1D`/`2D` `chi^2` confidence surfaces, and TSV/JSON I/O (`T13.4`).
//!
//! Provenance: Clean-room original implementation from peer-reviewed NMR spectroscopy and
//! numerical linear algebra literature:
//! - McConnell, H. M. (*J. Chem. Phys.* 1958, 28, 430–431 — coupled Bloch equations with chemical
//!   exchange).
//! - Helgstrand, M., Härd, T. & Allard, P. (*J. Biomol. NMR* 2000, 18, 47–61 — homogeneous
//!   Cartesian operator basis incorporating thermal longitudinal equilibrium `[E, Ix, Iy, Iz]`).
//! - Vallurupalli, P., Bouvignies, G. & Kay, L. E. (*J. Am. Chem. Soc.* 2012, 134, 8148–8161 —
//!   studying invisible excited protein states in slow exchange via `15N` `CEST` with `B1`
//!   inhomogeneity integration).
//! - Fawzi, N. L., Ying, J., Ghirlando, R., Torchia, D. A. & Clore, G. M. (*Nature* 2011, 480,
//!   268–272 — atomic-resolution protofibril surface dynamics via high-`R2B` `DEST`).
//! - Higham, N. J. (*SIAM J. Matrix Anal. Appl.* 2005, 26, 1179–1193 — scaling-and-squaring
//!   `[6/6]` Padé matrix exponential).

#![allow(clippy::needless_range_loop)]

use std::collections::BTreeMap;
use std::f64::consts::TAU;

use serde::{Deserialize, Serialize};

/// Higham (2005) `[6/6]` Padé numerator/denominator normalized coefficients `b_k = B_k / B_0`
/// where `B = [665280, 332640, 75600, 10080, 840, 42, 1]`.
pub const PADE66_COEFFS: [f64; 7] = [
    1.0,
    332_640.0 / 665_280.0,
    75_600.0 / 665_280.0,
    10_080.0 / 665_280.0,
    840.0 / 665_280.0,
    42.0 / 665_280.0,
    1.0 / 665_280.0,
];

/// Scaling threshold `theta_6` for `[6/6]` Padé scaling-and-squaring (`||A / 2^s||_1 <= 0.5`).
pub const PADE66_THETA_F64: f64 = 0.5;

/// Reference intensity normalization convention for `CEST` and `DEST` saturation profiles.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum CestNormalization {
    /// Phase-cycled `+-Iz` difference detection normalized by a `T_EX = 0` reference plane
    /// (Vallurupalli, Bouvignies & Kay *JACS* 2012): initial vector has `E = 0`, baseline far
    /// off-resonance equals `exp(-R1A * T_EX)`, directly constraining `R1A`.
    #[default]
    ZeroTimePhaseCycled,
    /// Un-cycled Boltzmann thermal recovery during `T_EX` normalized by equilibrium `I0 = p_A`
    /// (Fawzi et al. *Nature* 2011 `DEST`): initial vector has `E = 1`, baseline far off-resonance
    /// equals `1.0`.
    ThermalEquilibrium,
    /// Phase-cycled `+-Iz` difference detection normalized by a `B1 = 0` plane acquired with the
    /// same `T_EX` delay (`I(w1, T_EX) / I(0, T_EX)`): baseline far off-resonance approaches `1.0`.
    SameDelayZeroB1,
}

impl CestNormalization {
    pub const fn as_code(self) -> u32 {
        match self {
            Self::ZeroTimePhaseCycled => 0,
            Self::ThermalEquilibrium => 1,
            Self::SameDelayZeroB1 => 2,
        }
    }

    pub fn from_str_label(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "zero_time" | "zerotime" | "phase_cycled" | "cest" => Ok(Self::ZeroTimePhaseCycled),
            "thermal" | "boltzmann" | "dest" => Ok(Self::ThermalEquilibrium),
            "same_delay" | "zero_b1" => Ok(Self::SameDelayZeroB1),
            other => Err(format!("unknown CEST normalization '{other}'")),
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ZeroTimePhaseCycled => "zero_time",
            Self::ThermalEquilibrium => "thermal",
            Self::SameDelayZeroB1 => "same_delay",
        }
    }
}

/// Two-state (`A <-> B`) Bloch–McConnell parameters for a single residue and `B1` field.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Cest2StateParams {
    /// Fractional population of minor state `B` (`p_B in (0, 1)`, `p_A = 1 - p_B`).
    pub p_b: f64,
    /// Total chemical exchange rate `k_ex = k_AB + k_BA` (`s^-1`).
    pub k_ex_s_inv: f64,
    /// Major-state chemical shift `varpi_A` (`ppm`).
    pub major_ppm: f64,
    /// Excited-state chemical shift difference `Delta_varpi_AB = varpi_B - varpi_A` (`ppm`).
    pub dw_ab_ppm: f64,
    /// Longitudinal relaxation rate `R1A` (`s^-1`).
    pub r1a_s_inv: f64,
    /// Transverse relaxation rate `R2A` (`s^-1`).
    pub r2a_s_inv: f64,
    /// Longitudinal relaxation rate `R1B` (`s^-1`; typically equal to `R1A` in `CEST`).
    pub r1b_s_inv: f64,
    /// Transverse relaxation rate `R2B` (`s^-1`; `10..200 s^-1` in `CEST`, `1e3..5e4 s^-1` in `DEST`).
    pub r2b_s_inv: f64,
    /// Heteronuclear Larmor frequency `|nu_0|` in `MHz` (e.g. `60.8106 MHz` for `15N` at `600 MHz` `1H`).
    pub larmor_mhz: f64,
    /// Saturation period `T_EX` (`s`).
    pub t_ex_s: f64,
    /// Continuous-wave RF nutation frequency `nu_1 = omega_1 / (2*pi)` (`Hz`).
    pub b1_hz: f64,
    /// Fractional `B1` inhomogeneity standard deviation (`sigma_B1 / B1`, e.g. `0.0` or `0.10`).
    pub b1_inhomogeneity_frac: f64,
    /// Number of Gauss–Hermite quadrature nodes (`1, 3, 5, 7, 9`).
    pub n_b1_quad: usize,
    /// Reference intensity normalization mode.
    pub normalization: CestNormalization,
}

/// Three-state (`A <-> B <-> C` linear or triangular) Bloch–McConnell parameters (`10x10` basis).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Cest3StateParams {
    /// Minor population `p_B` (`p_A = 1 - p_B - p_C > 0`).
    pub p_b: f64,
    /// Minor population `p_C` (`p_A = 1 - p_B - p_C > 0`).
    pub p_c: f64,
    /// Pairwise exchange rate `k_ex_AB = k_AB + k_BA` (`s^-1`).
    pub k_ex_ab_s_inv: f64,
    /// Pairwise exchange rate `k_ex_BC = k_BC + k_CB` (`s^-1`).
    pub k_ex_bc_s_inv: f64,
    /// Pairwise exchange rate `k_ex_AC = k_AC + k_CA` (`s^-1`; `0.0` for linear `A <-> B <-> C`,
    /// `> 0.0` for triangular exchange).
    pub k_ex_ac_s_inv: f64,
    /// Major-state chemical shift `varpi_A` (`ppm`).
    pub major_ppm: f64,
    /// Shift difference `Delta_varpi_AB = varpi_B - varpi_A` (`ppm`).
    pub dw_ab_ppm: f64,
    /// Shift difference `Delta_varpi_AC = varpi_C - varpi_A` (`ppm`).
    pub dw_ac_ppm: f64,
    pub r1a_s_inv: f64,
    pub r2a_s_inv: f64,
    pub r1b_s_inv: f64,
    pub r2b_s_inv: f64,
    pub r1c_s_inv: f64,
    pub r2c_s_inv: f64,
    pub larmor_mhz: f64,
    pub t_ex_s: f64,
    pub b1_hz: f64,
    pub b1_inhomogeneity_frac: f64,
    pub n_b1_quad: usize,
    pub normalization: CestNormalization,
}

/// Experimental `CEST` or `DEST` saturation profile for one residue at one `B1` nutation field.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CestProfileDataset {
    pub seq_id: i32,
    pub res_name: String,
    pub larmor_mhz: f64,
    pub t_ex_s: f64,
    pub b1_hz: f64,
    pub b1_inhomogeneity_frac: f64,
    pub normalization: CestNormalization,
    /// Optional known major-state resonance position (`ppm`), or auto-detected from deepest dip.
    pub major_ppm_hint: Option<f64>,
    /// RF carrier positions `varpi_rf` (`ppm`).
    pub offsets_ppm: Vec<f64>,
    /// Normalized peak intensity ratios `I(varpi_rf) / I0`.
    pub intensities: Vec<f64>,
    /// `1-sigma` experimental uncertainties on `I / I0`.
    pub uncertainties: Vec<f64>,
}

/// Configuration for multi-`B1`, multi-residue global `CEST` / `DEST` fitting.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CestFitConfig {
    pub n_b1_quad: usize,
    pub max_lm_iters: usize,
    pub grad_tol: f64,
    pub step_tol: f64,
    /// Whether to optimize `major_ppm` per residue (`true` by default).
    pub refine_major_ppm: bool,
    /// Whether to compute `1D` and `2D` `(p_B, k_ex)` profile `chi^2` surfaces.
    pub compute_chi2_surface: bool,
    /// Grid resolution `(n_pb, n_kex)` for optional `2D` `chi^2` surface.
    pub surface_grid_dims: (usize, usize),
}

impl Default for CestFitConfig {
    fn default() -> Self {
        Self {
            n_b1_quad: 5,
            max_lm_iters: 45,
            grad_tol: 1e-8,
            step_tol: 1e-8,
            refine_major_ppm: true,
            compute_chi2_surface: true,
            surface_grid_dims: (15, 15),
        }
    }
}

/// Fitted per-residue `CEST` / `DEST` parameters and uncertainties (`1-sigma` covariance SEs).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CestResidueFit {
    pub seq_id: i32,
    pub res_name: String,
    pub major_ppm: f64,
    pub major_ppm_err: f64,
    pub dw_ab_ppm: f64,
    pub dw_ab_ppm_err: f64,
    pub r1a_s_inv: f64,
    pub r1a_err: f64,
    pub r2a_s_inv: f64,
    pub r2a_err: f64,
    pub r2b_s_inv: f64,
    pub r2b_err: f64,
    pub chi2: f64,
    pub n_points: usize,
}

/// `1D` and `2D` `(p_B, k_ex)` `chi^2` confidence profiles around the global minimum.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CestChi2Surface {
    pub pb_grid: Vec<f64>,
    pub kex_grid: Vec<f64>,
    /// `1D` slice `chi^2(p_B, k_ex_best)`.
    pub chi2_1d_pb: Vec<f64>,
    /// `1D` slice `chi^2(p_B_best, k_ex)`.
    pub chi2_1d_kex: Vec<f64>,
    /// Row-major `[i_pb * kex_grid.len() + i_kex]` `2D` `chi^2(p_B, k_ex)` surface.
    pub chi2_2d: Vec<f64>,
}

/// Complete multi-residue, multi-`B1` global `CEST` / `DEST` fit result.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CestGlobalFitResult {
    pub p_b: f64,
    pub p_b_err: f64,
    pub k_ex_s_inv: f64,
    pub k_ex_err: f64,
    pub k_ab_s_inv: f64,
    pub k_ba_s_inv: f64,
    pub total_chi2: f64,
    pub reduced_chi2: f64,
    pub dof: usize,
    pub iterations: usize,
    pub converged: bool,
    pub residues: Vec<CestResidueFit>,
    pub chi2_surface: Option<CestChi2Surface>,
}

// ============================================================================
// 1. Fixed-Size Square Matrix Algebra & Higham (2005) [6/6] Padé Exponential
// ============================================================================

#[inline]
pub fn mat_zero<const N: usize>() -> [[f64; N]; N] {
    [[0.0; N]; N]
}

#[inline]
pub fn mat_identity<const N: usize>() -> [[f64; N]; N] {
    let mut m = [[0.0; N]; N];
    for (i, row) in m.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    m
}

#[inline]
pub fn mat_mul<const N: usize>(a: &[[f64; N]; N], b: &[[f64; N]; N]) -> [[f64; N]; N] {
    let mut c = [[0.0; N]; N];
    for i in 0..N {
        for k in 0..N {
            let aik = a[i][k];
            if aik == 0.0 {
                continue;
            }
            for j in 0..N {
                c[i][j] += aik * b[k][j];
            }
        }
    }
    c
}

#[inline]
pub fn mat_vec_mul<const N: usize>(a: &[[f64; N]; N], v: &[f64; N]) -> [f64; N] {
    let mut out = [0.0; N];
    for i in 0..N {
        let mut s = 0.0;
        for j in 0..N {
            s += a[i][j] * v[j];
        }
        out[i] = s;
    }
    out
}

#[inline]
pub fn mat_norm1<const N: usize>(a: &[[f64; N]; N]) -> f64 {
    let mut max_col = 0.0f64;
    for j in 0..N {
        let mut col_sum = 0.0f64;
        for row in a.iter().take(N) {
            col_sum += row[j].abs();
        }
        if col_sum > max_col {
            max_col = col_sum;
        }
    }
    max_col
}

/// Solve `A * X = B` for `N x N` matrices using Gaussian elimination with partial pivoting.
pub fn mat_solve<const N: usize>(
    a: &[[f64; N]; N],
    b: &[[f64; N]; N],
) -> Result<[[f64; N]; N], String> {
    let mut lu = *a;
    let mut rhs = *b;

    for col in 0..N {
        let mut piv = col;
        let mut max_abs = lu[col][col].abs();
        for (r, row) in lu.iter().enumerate().take(N).skip(col + 1) {
            let v = row[col].abs();
            if v > max_abs {
                max_abs = v;
                piv = r;
            }
        }
        if !max_abs.is_finite() || max_abs < 1e-30 {
            return Err(format!(
                "singular or non-finite matrix in mat_solve at column {col} (pivot={max_abs})"
            ));
        }
        if piv != col {
            lu.swap(col, piv);
            rhs.swap(col, piv);
        }

        let inv_diag = 1.0 / lu[col][col];
        for r in (col + 1)..N {
            let factor = lu[r][col] * inv_diag;
            if factor == 0.0 {
                continue;
            }
            for c in (col + 1)..N {
                lu[r][c] -= factor * lu[col][c];
            }
            for c in 0..N {
                rhs[r][c] -= factor * rhs[col][c];
            }
        }
    }

    let mut x = [[0.0; N]; N];
    for i in (0..N).rev() {
        let inv_diag = 1.0 / lu[i][i];
        for col in 0..N {
            let mut s = rhs[i][col];
            for k in (i + 1)..N {
                s -= lu[i][k] * x[k][col];
            }
            x[i][col] = s * inv_diag;
        }
    }
    Ok(x)
}

/// Compute the matrix exponential `exp(A)` via Higham (*SIAM J. Matrix Anal. Appl.* 2005)
/// scaling-and-squaring `[6/6]` Padé approximant, propagating `F = exp(A / 2^s) - I` via
/// `(V - U) F_0 = 2 U` and `F <- 2 F + F^2` to eliminate subtraction cancellation near `I`.
pub fn expm_pade66<const N: usize>(a: &[[f64; N]; N]) -> Result<[[f64; N]; N], String> {
    let norm1 = mat_norm1(a);
    if !norm1.is_finite() {
        return Err("non-finite matrix entry in expm_pade66".to_string());
    }
    if norm1 == 0.0 {
        return Ok(mat_identity());
    }

    let s: u32 = if norm1 > PADE66_THETA_F64 {
        (norm1 / PADE66_THETA_F64).log2().ceil().max(0.0) as u32
    } else {
        0
    };
    let inv_scale = 0.5_f64.powi(s as i32);
    let mut a_s = [[0.0; N]; N];
    for i in 0..N {
        for j in 0..N {
            a_s[i][j] = a[i][j] * inv_scale;
        }
    }

    let a2 = mat_mul(&a_s, &a_s);
    let a4 = mat_mul(&a2, &a2);
    let a6 = mat_mul(&a2, &a4);

    let b = PADE66_COEFFS;
    let mut inner_odd = [[0.0; N]; N];
    let mut v = [[0.0; N]; N];
    for i in 0..N {
        for j in 0..N {
            inner_odd[i][j] = b[5] * a4[i][j] + b[3] * a2[i][j];
            v[i][j] = b[6] * a6[i][j] + b[4] * a4[i][j] + b[2] * a2[i][j];
        }
        inner_odd[i][i] += b[1];
        v[i][i] += b[0];
    }
    let u = mat_mul(&a_s, &inner_odd);

    let mut v_minus_u = [[0.0; N]; N];
    let mut two_u = [[0.0; N]; N];
    for i in 0..N {
        for j in 0..N {
            v_minus_u[i][j] = v[i][j] - u[i][j];
            two_u[i][j] = 2.0 * u[i][j];
        }
    }

    // F = R_6(A_s) - I = 2 * (V - U)^(-1) * U
    let mut f = mat_solve(&v_minus_u, &two_u)?;

    // Square `I + F` `s` times without losing precision near `I`: `(I + F)^2 - I = 2 F + F^2`
    for _ in 0..s {
        let f2 = mat_mul(&f, &f);
        for i in 0..N {
            for j in 0..N {
                f[i][j] = 2.0 * f[i][j] + f2[i][j];
            }
        }
    }

    for (i, row) in f.iter_mut().enumerate().take(N) {
        row[i] += 1.0;
    }
    Ok(f)
}

/// Independent high-order (`24`-term Taylor / `8`th+ order ODE reference with ultra-fine scaling
/// `||A / 2^s||_1 <= 1/32`) matrix exponential reference for verifying `expm_pade66` to `< 1e-13`.
pub fn expm_high_order_reference<const N: usize>(a: &[[f64; N]; N]) -> [[f64; N]; N] {
    let norm1 = mat_norm1(a);
    if norm1 == 0.0 {
        return mat_identity();
    }
    let theta = 1.0 / 32.0;
    let s: u32 = if norm1 > theta {
        (norm1 / theta).log2().ceil().max(0.0) as u32
    } else {
        0
    };
    let scale = 0.5_f64.powi(s as i32);
    let mut a_s = [[0.0; N]; N];
    for i in 0..N {
        for j in 0..N {
            a_s[i][j] = a[i][j] * scale;
        }
    }

    // Accumulate F = sum_{k=1}^{24} A_s^k / k!
    let mut f = a_s;
    let mut term = a_s;
    for k in 2..=24 {
        term = mat_mul(&term, &a_s);
        let inv_k = 1.0 / (k as f64);
        for i in 0..N {
            for j in 0..N {
                term[i][j] *= inv_k;
                f[i][j] += term[i][j];
            }
        }
    }

    for _ in 0..s {
        let f2 = mat_mul(&f, &f);
        for i in 0..N {
            for j in 0..N {
                f[i][j] = 2.0 * f[i][j] + f2[i][j];
            }
        }
    }
    for (i, row) in f.iter_mut().enumerate().take(N) {
        row[i] += 1.0;
    }
    f
}

/// Independent complex Francis-QR Schur + eigenvector spectral decomposition `V diag(exp(lambda)) V^-1`
/// reference for diagonalizable `N x N` Bloch–McConnell matrices.
pub fn expm_eigen_reference<const N: usize>(a: &[[f64; N]; N]) -> Result<[[f64; N]; N], String> {
    if N == 0 {
        return Ok([[0.0; N]; N]);
    }
    expm_complex_qr_full(a)
}

fn expm_complex_qr_full<const N: usize>(a: &[[f64; N]; N]) -> Result<[[f64; N]; N], String> {
    let c_add = |x: (f64, f64), y: (f64, f64)| (x.0 + y.0, x.1 + y.1);
    let c_sub = |x: (f64, f64), y: (f64, f64)| (x.0 - y.0, x.1 - y.1);
    let c_mul = |x: (f64, f64), y: (f64, f64)| (x.0 * y.0 - x.1 * y.1, x.0 * y.1 + x.1 * y.0);
    let c_conj = |x: (f64, f64)| (x.0, -x.1);
    let c_abs = |x: (f64, f64)| x.0.hypot(x.1);
    let c_div = |x: (f64, f64), y: (f64, f64)| {
        let d = y.0 * y.0 + y.1 * y.1;
        ((x.0 * y.0 + x.1 * y.1) / d, (x.1 * y.0 - x.0 * y.1) / d)
    };
    let c_sqrt = |z: (f64, f64)| {
        let r = c_abs(z);
        if r == 0.0 {
            (0.0, 0.0)
        } else {
            let re = ((r + z.0) * 0.5).max(0.0).sqrt();
            let im = ((r - z.0) * 0.5).max(0.0).sqrt() * if z.1 >= 0.0 { 1.0 } else { -1.0 };
            (re, im)
        }
    };

    // 1. Real Householder Hessenberg reduction
    let mut h = *a;
    for k in 0..N.saturating_sub(2) {
        let mut norm_sq = 0.0;
        for row in h.iter().take(N).skip(k + 1) {
            norm_sq += row[k] * row[k];
        }
        let norm = norm_sq.sqrt();
        if norm < 1e-15 {
            continue;
        }
        let sign = if h[k + 1][k] >= 0.0 { 1.0 } else { -1.0 };
        let mut v = vec![0.0; N - (k + 1)];
        v[0] = h[k + 1][k] + sign * norm;
        for idx in 1..v.len() {
            v[idx] = h[k + 1 + idx][k];
        }
        let v_norm_sq: f64 = v.iter().map(|x| x * x).sum();
        if v_norm_sq < 1e-30 {
            continue;
        }
        let beta = 2.0 / v_norm_sq;
        for col in k..N {
            let mut dot = 0.0;
            for idx in 0..v.len() {
                dot += v[idx] * h[k + 1 + idx][col];
            }
            for idx in 0..v.len() {
                h[k + 1 + idx][col] -= beta * v[idx] * dot;
            }
        }
        for row in h.iter_mut().take(N) {
            let mut dot = 0.0;
            for idx in 0..v.len() {
                dot += row[k + 1 + idx] * v[idx];
            }
            for idx in 0..v.len() {
                row[k + 1 + idx] -= beta * dot * v[idx];
            }
        }
    }

    // 2. Shifted complex Hessenberg QR iteration -> upper triangular Schur form T (`A = Q T Q*` up to H similarity)
    let mut hc = vec![vec![(0.0f64, 0.0f64); N]; N];
    for i in 0..N {
        for j in 0..N {
            hc[i][j] = (h[i][j], 0.0);
        }
    }
    let mut eigvals = vec![(0.0f64, 0.0f64); N];
    let mut m = N;
    let mut iter_count = 0usize;
    while m > 1 {
        iter_count += 1;
        if iter_count > 4000 {
            return Err("complex QR eigenvalue iteration exceeded 4000 steps".to_string());
        }
        let sub_mag = c_abs(hc[m - 1][m - 2]);
        let diag_scale = (c_abs(hc[m - 2][m - 2]) + c_abs(hc[m - 1][m - 1])).max(1.0);
        if sub_mag <= 1e-14 * diag_scale {
            eigvals[m - 1] = hc[m - 1][m - 1];
            m -= 1;
            continue;
        }

        let a11 = hc[m - 2][m - 2];
        let a12 = hc[m - 2][m - 1];
        let a21 = hc[m - 1][m - 2];
        let a22 = hc[m - 1][m - 1];
        let half_diff = ((a11.0 - a22.0) * 0.5, (a11.1 - a22.1) * 0.5);
        let root = c_sqrt(c_add(c_mul(half_diff, half_diff), c_mul(a12, a21)));
        let l1 = c_add(a22, c_sub(half_diff, root));
        let l2 = c_add(a22, c_add(half_diff, root));
        let mut shift = if c_abs(c_sub(l1, a22)) < c_abs(c_sub(l2, a22)) {
            l1
        } else {
            l2
        };
        if iter_count.is_multiple_of(25) {
            shift = (shift.0 + 0.731 * sub_mag, shift.1 - 0.417 * sub_mag);
        }

        for (i, row) in hc.iter_mut().enumerate().take(m) {
            row[i] = c_sub(row[i], shift);
        }
        let mut alphas = vec![(1.0f64, 0.0f64); m - 1];
        let mut betas = vec![(0.0f64, 0.0f64); m - 1];
        for i in 0..(m - 1) {
            let x = hc[i][i];
            let y = hc[i + 1][i];
            let r = x.0.hypot(x.1).hypot(y.0.hypot(y.1));
            if r < 1e-30 {
                continue;
            }
            let al = (x.0 / r, x.1 / r);
            let be = (y.0 / r, y.1 / r);
            alphas[i] = al;
            betas[i] = be;
            let al_c = c_conj(al);
            let be_c = c_conj(be);
            for col in i..m {
                let top = hc[i][col];
                let bot = hc[i + 1][col];
                hc[i][col] = c_add(c_mul(al_c, top), c_mul(be_c, bot));
                hc[i + 1][col] = c_sub(c_mul(al, bot), c_mul(be, top));
            }
        }
        // Right multiply by G_i^* = [[al, -be^*], [be, al^*]]
        for i in 0..(m - 1) {
            let al = alphas[i];
            let be = betas[i];
            let al_c = c_conj(al);
            let be_c = c_conj(be);
            for row in hc.iter_mut().take(m) {
                let left = row[i];
                let right = row[i + 1];
                row[i] = c_add(c_mul(left, al), c_mul(right, be));
                row[i + 1] = c_add(c_mul(left, (-be_c.0, -be_c.1)), c_mul(right, al_c));
            }
        }
        for (i, row) in hc.iter_mut().enumerate().take(m) {
            row[i] = c_add(row[i], shift);
        }
    }
    if m == 1 {
        eigvals[0] = hc[0][0];
    }

    // 3. Inverse iteration on `(A - (lambda_k + eps) I) v_k = rhs` for each eigenvalue `lambda_k`
    let norm_a = mat_norm1(a).max(1.0);
    let mut v_mat = vec![vec![(0.0f64, 0.0f64); N]; N];
    for (k, &lam) in eigvals.iter().enumerate() {
        let eps = 1e-13 * norm_a;
        let mut vec_k: Vec<(f64, f64)> = (0..N)
            .map(|i| {
                let phase = ((i + 1) * (k + 3)) as f64;
                (phase.cos(), phase.sin())
            })
            .collect();
        for _step in 0..3 {
            let mut m_sys = vec![vec![(0.0f64, 0.0f64); N]; N];
            for i in 0..N {
                for j in 0..N {
                    m_sys[i][j] = (a[i][j], 0.0);
                }
                m_sys[i][i] = c_sub(m_sys[i][i], (lam.0 + eps, lam.1 - 0.37 * eps));
            }
            vec_k = complex_linear_solve_vec(&m_sys, &vec_k, c_sub, c_mul, c_div, c_abs)?;
            let nrm: f64 = vec_k
                .iter()
                .map(|z| z.0 * z.0 + z.1 * z.1)
                .sum::<f64>()
                .sqrt();
            if nrm > 0.0 {
                for z in &mut vec_k {
                    z.0 /= nrm;
                    z.1 /= nrm;
                }
            }
        }
        for i in 0..N {
            v_mat[i][k] = vec_k[i];
        }
    }

    // 4. Solve `V^T X^T = (V * diag(exp(lambda)))^T` -> `X = V * diag(exp(lambda)) * V^{-1}`
    let mut vd = vec![vec![(0.0f64, 0.0f64); N]; N];
    for k in 0..N {
        let er = eigvals[k].0.exp();
        let elam = (er * eigvals[k].1.cos(), er * eigvals[k].1.sin());
        for i in 0..N {
            vd[i][k] = c_mul(v_mat[i][k], elam);
        }
    }

    let mut vt = vec![vec![(0.0f64, 0.0f64); N]; N];
    for i in 0..N {
        for j in 0..N {
            vt[i][j] = v_mat[j][i];
        }
    }

    let mut out = [[0.0; N]; N];
    for row_idx in 0..N {
        let rhs: Vec<(f64, f64)> = (0..N).map(|k| vd[row_idx][k]).collect();
        let sol = complex_linear_solve_vec(&vt, &rhs, c_sub, c_mul, c_div, c_abs)?;
        for col_idx in 0..N {
            out[row_idx][col_idx] = sol[col_idx].0;
        }
    }
    Ok(out)
}

fn complex_linear_solve_vec<FSub, FMul, FDiv, FAbs>(
    a: &[Vec<(f64, f64)>],
    b: &[(f64, f64)],
    c_sub: FSub,
    c_mul: FMul,
    c_div: FDiv,
    c_abs: FAbs,
) -> Result<Vec<(f64, f64)>, String>
where
    FSub: Fn((f64, f64), (f64, f64)) -> (f64, f64),
    FMul: Fn((f64, f64), (f64, f64)) -> (f64, f64),
    FDiv: Fn((f64, f64), (f64, f64)) -> (f64, f64),
    FAbs: Fn((f64, f64)) -> f64,
{
    let n = b.len();
    let mut lu = a.to_vec();
    let mut rhs = b.to_vec();

    for col in 0..n {
        let mut piv = col;
        let mut max_v = c_abs(lu[col][col]);
        for (r, row) in lu.iter().enumerate().take(n).skip(col + 1) {
            let v = c_abs(row[col]);
            if v > max_v {
                max_v = v;
                piv = r;
            }
        }
        if max_v < 1e-28 {
            return Err("ill-conditioned complex eigenvector system".to_string());
        }
        if piv != col {
            lu.swap(col, piv);
            rhs.swap(col, piv);
        }
        for r in (col + 1)..n {
            let factor = c_div(lu[r][col], lu[col][col]);
            for c in (col + 1)..n {
                lu[r][c] = c_sub(lu[r][c], c_mul(factor, lu[col][c]));
            }
            rhs[r] = c_sub(rhs[r], c_mul(factor, rhs[col]));
        }
    }

    let mut x = vec![(0.0, 0.0); n];
    for i in (0..n).rev() {
        let mut s = rhs[i];
        for k in (i + 1)..n {
            s = c_sub(s, c_mul(lu[i][k], x[k]));
        }
        x[i] = c_div(s, lu[i][i]);
    }
    Ok(x)
}

// ============================================================================
// 2. Gauss–Hermite B1 Inhomogeneity Quadrature & Bloch–McConnell Liouvillians
// ============================================================================

/// Return normalized Gauss–Hermite quadrature `(x_q, w_q / sqrt(pi))` pairs for `n_points`
/// (`n >= 1`), satisfying `sum w_norm_q = 1` so `E[f(X)]` for `X ~ N(0, 1)` is approximated by
/// `sum w_norm_q * f(sqrt(2) * x_q)`. Uses exact closed-form tables for `1, 3, 5, 7, 9` and the
/// Golub–Welsch (*Math. Comp.* 1969) symmetric tridiagonal Hermite Jacobi eigensolver for general `n`.
pub fn gauss_hermite_normalized(n_points: usize) -> Vec<(f64, f64)> {
    match n_points {
        0 | 1 => vec![(0.0, 1.0)],
        3 => vec![
            (-1.224_744_871_391_589, 1.0 / 6.0),
            (0.0, 2.0 / 3.0),
            (1.224_744_871_391_589, 1.0 / 6.0),
        ],
        5 => vec![
            (-2.020_182_870_456_085_6, 0.011_257_411_327_720_69),
            (-0.958_572_464_613_818_5, 0.222_075_922_005_612_64),
            (0.0, 0.533_333_333_333_333_3),
            (0.958_572_464_613_818_5, 0.222_075_922_005_612_64),
            (2.020_182_870_456_085_6, 0.011_257_411_327_720_69),
        ],
        7 => vec![
            (-2.651_961_356_835_233_5, 0.000_548_268_855_972_427_6),
            (-1.673_551_628_767_471_4, 0.030_757_123_967_589_49),
            (-0.816_287_882_858_964_7, 0.240_123_178_605_012_93),
            (0.0, 0.457_142_857_142_857_13),
            (0.816_287_882_858_964_7, 0.240_123_178_605_012_93),
            (1.673_551_628_767_471_4, 0.030_757_123_967_589_49),
            (2.651_961_356_835_233_5, 0.000_548_268_855_972_427_6),
        ],
        9 => vec![
            (-3.190_993_201_781_528, 0.000_022_345_844_013_966_6),
            (-2.266_580_584_531_843, 0.002_789_141_321_179_884_7),
            (-1.468_553_289_216_668, 0.049_916_406_768_527_22),
            (-0.723_551_018_752_837_6, 0.244_097_502_894_927_14),
            (0.0, 0.406_349_206_349_206_35),
            (0.723_551_018_752_837_6, 0.244_097_502_894_927_14),
            (1.468_553_289_216_668, 0.049_916_406_768_527_22),
            (2.266_580_584_531_843, 0.002_789_141_321_179_884_7),
            (3.190_993_201_781_528, 0.000_022_345_844_013_966_6),
        ],
        n => {
            let n = n.min(64);
            let mut a = vec![vec![0.0f64; n]; n];
            let mut v = vec![vec![0.0f64; n]; n];
            for (i, row) in v.iter_mut().enumerate() {
                row[i] = 1.0;
            }
            for k in 0..(n - 1) {
                let b = ((k + 1) as f64 * 0.5).sqrt();
                a[k][k + 1] = b;
                a[k + 1][k] = b;
            }
            for _sweep in 0..80 {
                let mut max_off = 0.0f64;
                for p in 0..n {
                    for q in (p + 1)..n {
                        max_off = max_off.max(a[p][q].abs());
                    }
                }
                if max_off < 1e-15 {
                    break;
                }
                for p in 0..n {
                    for q in (p + 1)..n {
                        let apq = a[p][q];
                        if apq.abs() < 1e-16 {
                            continue;
                        }
                        let tau = (a[q][q] - a[p][p]) / (2.0 * apq);
                        let t = if tau >= 0.0 {
                            1.0 / (tau + (1.0 + tau * tau).sqrt())
                        } else {
                            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                        };
                        let c = 1.0 / (1.0 + t * t).sqrt();
                        let s = t * c;
                        let app = a[p][p];
                        let aqq = a[q][q];
                        a[p][p] = app - t * apq;
                        a[q][q] = aqq + t * apq;
                        a[p][q] = 0.0;
                        a[q][p] = 0.0;
                        for r in 0..n {
                            if r != p && r != q {
                                let arp = a[r][p];
                                let arq = a[r][q];
                                a[r][p] = c * arp - s * arq;
                                a[p][r] = a[r][p];
                                a[r][q] = s * arp + c * arq;
                                a[q][r] = a[r][q];
                            }
                            let vrp = v[r][p];
                            let vrq = v[r][q];
                            v[r][p] = c * vrp - s * vrq;
                            v[r][q] = s * vrp + c * vrq;
                        }
                    }
                }
            }
            let mut nw: Vec<(f64, f64)> = (0..n).map(|i| (a[i][i], v[0][i] * v[0][i])).collect();
            nw.sort_by(|x, y| x.0.total_cmp(&y.0));
            nw
        }
    }
}

/// Construct the `7 x 7` homogeneous two-state Bloch–McConnell Liouvillian `L` in the basis
/// `[E, IxA, IyA, IzA, IxB, IyB, IzB]^T` (Helgstrand et al. 2000; Vallurupalli et al. 2012).
///
/// Angular offsets `omega_a_rad`, `omega_b_rad` and nutation components `w1x_rad`, `w1y_rad` are
/// in `rad/s`; rates `r1a`, `r2a`, `r1b`, `r2b`, `k_ex` are in `s^-1`.
#[allow(clippy::too_many_arguments)]
pub fn build_liouvillian_2state(
    p_b: f64,
    k_ex: f64,
    omega_a_rad: f64,
    omega_b_rad: f64,
    w1x_rad: f64,
    w1y_rad: f64,
    r1a: f64,
    r2a: f64,
    r1b: f64,
    r2b: f64,
) -> [[f64; 7]; 7] {
    let pb = p_b.clamp(0.0, 1.0);
    let pa = 1.0 - pb;
    let kab = pb * k_ex;
    let kba = pa * k_ex;

    let mut l = [[0.0; 7]; 7];
    // State A (`1..=3`: IxA, IyA, IzA)
    l[1][1] = -r2a - kab;
    l[1][2] = -omega_a_rad;
    l[1][3] = w1y_rad;
    l[1][4] = kba;

    l[2][1] = omega_a_rad;
    l[2][2] = -r2a - kab;
    l[2][3] = -w1x_rad;
    l[2][5] = kba;

    l[3][0] = r1a * pa;
    l[3][1] = -w1y_rad;
    l[3][2] = w1x_rad;
    l[3][3] = -r1a - kab;
    l[3][6] = kba;

    // State B (`4..=6`: IxB, IyB, IzB)
    l[4][1] = kab;
    l[4][4] = -r2b - kba;
    l[4][5] = -omega_b_rad;
    l[4][6] = w1y_rad;

    l[5][2] = kab;
    l[5][4] = omega_b_rad;
    l[5][5] = -r2b - kba;
    l[5][6] = -w1x_rad;

    l[6][0] = r1b * pb;
    l[6][3] = kab;
    l[6][4] = -w1y_rad;
    l[6][5] = w1x_rad;
    l[6][6] = -r1b - kba;

    l
}

/// Construct the `10 x 10` homogeneous three-state (`A <-> B <-> C` linear or triangular)
/// Bloch–McConnell Liouvillian `L` in basis `[E, IxA, IyA, IzA, IxB, IyB, IzB, IxC, IyC, IzC]^T`
/// enforcing detailed balance on every pair (`k_ij * p_i = k_ji * p_j`).
#[allow(clippy::too_many_arguments)]
pub fn build_liouvillian_3state(
    p_b: f64,
    p_c: f64,
    k_ex_ab: f64,
    k_ex_bc: f64,
    k_ex_ac: f64,
    omega_a_rad: f64,
    omega_b_rad: f64,
    omega_c_rad: f64,
    w1x_rad: f64,
    w1y_rad: f64,
    r1a: f64,
    r2a: f64,
    r1b: f64,
    r2b: f64,
    r1c: f64,
    r2c: f64,
) -> [[f64; 10]; 10] {
    let pb = p_b.max(0.0);
    let pc = p_c.max(0.0);
    let pa = (1.0 - pb - pc).max(0.0);

    let (kab, kba) = if pa + pb > 0.0 {
        (k_ex_ab * pb / (pa + pb), k_ex_ab * pa / (pa + pb))
    } else {
        (0.0, 0.0)
    };
    let (kbc, kcb) = if pb + pc > 0.0 {
        (k_ex_bc * pc / (pb + pc), k_ex_bc * pb / (pb + pc))
    } else {
        (0.0, 0.0)
    };
    let (kac, kca) = if pa + pc > 0.0 {
        (k_ex_ac * pc / (pa + pc), k_ex_ac * pa / (pa + pc))
    } else {
        (0.0, 0.0)
    };

    let mut l = [[0.0; 10]; 10];
    let ka_out = kab + kac;
    let kb_out = kba + kbc;
    let kc_out = kca + kcb;

    // State A (`1..=3`)
    l[1][1] = -r2a - ka_out;
    l[1][2] = -omega_a_rad;
    l[1][3] = w1y_rad;
    l[1][4] = kba;
    l[1][7] = kca;

    l[2][1] = omega_a_rad;
    l[2][2] = -r2a - ka_out;
    l[2][3] = -w1x_rad;
    l[2][5] = kba;
    l[2][8] = kca;

    l[3][0] = r1a * pa;
    l[3][1] = -w1y_rad;
    l[3][2] = w1x_rad;
    l[3][3] = -r1a - ka_out;
    l[3][6] = kba;
    l[3][9] = kca;

    // State B (`4..=6`)
    l[4][1] = kab;
    l[4][4] = -r2b - kb_out;
    l[4][5] = -omega_b_rad;
    l[4][6] = w1y_rad;
    l[4][7] = kcb;

    l[5][2] = kab;
    l[5][4] = omega_b_rad;
    l[5][5] = -r2b - kb_out;
    l[5][6] = -w1x_rad;
    l[5][8] = kcb;

    l[6][0] = r1b * pb;
    l[6][3] = kab;
    l[6][4] = -w1y_rad;
    l[6][5] = w1x_rad;
    l[6][6] = -r1b - kb_out;
    l[6][9] = kcb;

    // State C (`7..=9`)
    l[7][1] = kac;
    l[7][4] = kbc;
    l[7][7] = -r2c - kc_out;
    l[7][8] = -omega_c_rad;
    l[7][9] = w1y_rad;

    l[8][2] = kac;
    l[8][5] = kbc;
    l[8][7] = omega_c_rad;
    l[8][8] = -r2c - kc_out;
    l[8][9] = -w1x_rad;

    l[9][0] = r1c * pc;
    l[9][3] = kac;
    l[9][6] = kbc;
    l[9][7] = -w1y_rad;
    l[9][8] = w1x_rad;
    l[9][9] = -r1c - kc_out;

    l
}

/// Evaluate a two-state (`CEST` or `DEST`) normalized intensity profile `I(varpi_rf) / I0` across
/// `offsets_ppm` using Higham `[6/6]` Padé matrix exponentiation and Gauss–Hermite `B1` quadrature.
pub fn simulate_cest_2state_profile(
    params: &Cest2StateParams,
    offsets_ppm: &[f64],
) -> Result<Vec<f64>, String> {
    let pb = params.p_b.clamp(0.0, 0.999_999);
    let pa = 1.0 - pb;
    let e0 = match params.normalization {
        CestNormalization::ZeroTimePhaseCycled | CestNormalization::SameDelayZeroB1 => 0.0,
        CestNormalization::ThermalEquilibrium => 1.0,
    };
    let v0 = [e0, 0.0, 0.0, pa, 0.0, 0.0, pb];

    let i0 = match params.normalization {
        CestNormalization::ZeroTimePhaseCycled | CestNormalization::ThermalEquilibrium => pa,
        CestNormalization::SameDelayZeroB1 => {
            let mut l0 = build_liouvillian_2state(
                pb,
                params.k_ex_s_inv,
                0.0,
                0.0,
                0.0,
                0.0,
                params.r1a_s_inv,
                params.r2a_s_inv,
                params.r1b_s_inv,
                params.r2b_s_inv,
            );
            for row in &mut l0 {
                for val in row {
                    *val *= params.t_ex_s;
                }
            }
            let p0 = expm_pade66(&l0)?;
            mat_vec_mul(&p0, &v0)[3].max(1e-12)
        }
    };

    let quad = if params.b1_inhomogeneity_frac > 0.0 && params.n_b1_quad > 1 {
        gauss_hermite_normalized(params.n_b1_quad)
    } else {
        vec![(0.0, 1.0)]
    };

    let scale_rad_per_ppm = TAU * params.larmor_mhz; // 1 ppm * MHz = Hz -> * 2*pi = rad/s
    let w_b_ppm = params.major_ppm + params.dw_ab_ppm;
    let w1_nom_rad = TAU * params.b1_hz;
    let sqrt2_sig = std::f64::consts::SQRT_2 * params.b1_inhomogeneity_frac.max(0.0);

    let mut out = Vec::with_capacity(offsets_ppm.len());
    for &rf_ppm in offsets_ppm {
        let om_a = scale_rad_per_ppm * (params.major_ppm - rf_ppm);
        let om_b = scale_rad_per_ppm * (w_b_ppm - rf_ppm);
        let mut iz_a_acc = 0.0;
        for &(xq, wq) in &quad {
            let w1_q = (w1_nom_rad * (1.0 + sqrt2_sig * xq)).max(0.0);
            let mut l_t = build_liouvillian_2state(
                pb,
                params.k_ex_s_inv,
                om_a,
                om_b,
                w1_q,
                0.0,
                params.r1a_s_inv,
                params.r2a_s_inv,
                params.r1b_s_inv,
                params.r2b_s_inv,
            );
            for row in &mut l_t {
                for val in row {
                    *val *= params.t_ex_s;
                }
            }
            let prop = expm_pade66(&l_t)?;
            let vt = mat_vec_mul(&prop, &v0);
            iz_a_acc += wq * vt[3];
        }
        out.push(iz_a_acc / i0);
    }
    Ok(out)
}

/// Evaluate a three-state (`10x10` linear or triangular `A <-> B <-> C`) normalized saturation
/// profile `I(varpi_rf) / I0` across `offsets_ppm`.
pub fn simulate_cest_3state_profile(
    params: &Cest3StateParams,
    offsets_ppm: &[f64],
) -> Result<Vec<f64>, String> {
    let pb = params.p_b.max(0.0);
    let pc = params.p_c.max(0.0);
    if pb + pc >= 0.999_999 {
        return Err("3-state CEST requires p_B + p_C < 1".to_string());
    }
    let pa = 1.0 - pb - pc;
    let e0 = match params.normalization {
        CestNormalization::ZeroTimePhaseCycled | CestNormalization::SameDelayZeroB1 => 0.0,
        CestNormalization::ThermalEquilibrium => 1.0,
    };
    let v0 = [e0, 0.0, 0.0, pa, 0.0, 0.0, pb, 0.0, 0.0, pc];

    let i0 = match params.normalization {
        CestNormalization::ZeroTimePhaseCycled | CestNormalization::ThermalEquilibrium => pa,
        CestNormalization::SameDelayZeroB1 => {
            let mut l0 = build_liouvillian_3state(
                pb,
                pc,
                params.k_ex_ab_s_inv,
                params.k_ex_bc_s_inv,
                params.k_ex_ac_s_inv,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                params.r1a_s_inv,
                params.r2a_s_inv,
                params.r1b_s_inv,
                params.r2b_s_inv,
                params.r1c_s_inv,
                params.r2c_s_inv,
            );
            for row in &mut l0 {
                for val in row {
                    *val *= params.t_ex_s;
                }
            }
            let p0 = expm_pade66(&l0)?;
            mat_vec_mul(&p0, &v0)[3].max(1e-12)
        }
    };

    let quad = if params.b1_inhomogeneity_frac > 0.0 && params.n_b1_quad > 1 {
        gauss_hermite_normalized(params.n_b1_quad)
    } else {
        vec![(0.0, 1.0)]
    };

    let scale_rad_per_ppm = TAU * params.larmor_mhz;
    let w_b_ppm = params.major_ppm + params.dw_ab_ppm;
    let w_c_ppm = params.major_ppm + params.dw_ac_ppm;
    let w1_nom_rad = TAU * params.b1_hz;
    let sqrt2_sig = std::f64::consts::SQRT_2 * params.b1_inhomogeneity_frac.max(0.0);

    let mut out = Vec::with_capacity(offsets_ppm.len());
    for &rf_ppm in offsets_ppm {
        let om_a = scale_rad_per_ppm * (params.major_ppm - rf_ppm);
        let om_b = scale_rad_per_ppm * (w_b_ppm - rf_ppm);
        let om_c = scale_rad_per_ppm * (w_c_ppm - rf_ppm);
        let mut iz_a_acc = 0.0;
        for &(xq, wq) in &quad {
            let w1_q = (w1_nom_rad * (1.0 + sqrt2_sig * xq)).max(0.0);
            let mut l_t = build_liouvillian_3state(
                pb,
                pc,
                params.k_ex_ab_s_inv,
                params.k_ex_bc_s_inv,
                params.k_ex_ac_s_inv,
                om_a,
                om_b,
                om_c,
                w1_q,
                0.0,
                params.r1a_s_inv,
                params.r2a_s_inv,
                params.r1b_s_inv,
                params.r2b_s_inv,
                params.r1c_s_inv,
                params.r2c_s_inv,
            );
            for row in &mut l_t {
                for val in row {
                    *val *= params.t_ex_s;
                }
            }
            let prop = expm_pade66(&l_t)?;
            let vt = mat_vec_mul(&prop, &v0);
            iz_a_acc += wq * vt[3];
        }
        out.push(iz_a_acc / i0);
    }
    Ok(out)
}

// ============================================================================
// 3. Initial Parameter Estimation & Schur-Complement Global LM Optimization
// ============================================================================

/// Initial per-residue local parameter seed `(major_ppm, dw_ab_ppm, r1a, r2a, r2b)`.
#[derive(Clone, Copy, Debug)]
pub struct CestResidueSeed {
    pub major_ppm: f64,
    pub dw_ab_ppm: f64,
    pub r1a_s_inv: f64,
    pub r2a_s_inv: f64,
    pub r2b_s_inv: f64,
}

/// Automatically estimate initial per-residue `(major_ppm, dw_ab_ppm, r1a, r2a, r2b)` from one or
/// more multi-`B1` `CEST` / `DEST` profiles of the same residue.
pub fn estimate_residue_initial_seed(res_profiles: &[&CestProfileDataset]) -> CestResidueSeed {
    let p0 = res_profiles[0];
    // 1. Major-state resonance position: hint or minimum intensity in lowest-B1 profile
    let lowest_b1 = res_profiles
        .iter()
        .copied()
        .min_by(|a, b| a.b1_hz.total_cmp(&b.b1_hz))
        .unwrap_or(p0);

    let major_ppm = p0.major_ppm_hint.unwrap_or_else(|| {
        let mut best_i = 0usize;
        let mut min_val = f64::INFINITY;
        for (i, &y) in lowest_b1.intensities.iter().enumerate() {
            if y < min_val {
                min_val = y;
                best_i = i;
            }
        }
        // Parabolic 3-point interpolation around minimum
        let x0 = lowest_b1.offsets_ppm[best_i];
        if best_i > 0 && best_i + 1 < lowest_b1.offsets_ppm.len() {
            let xm = lowest_b1.offsets_ppm[best_i - 1];
            let xp = lowest_b1.offsets_ppm[best_i + 1];
            let ym = lowest_b1.intensities[best_i - 1];
            let yc = lowest_b1.intensities[best_i];
            let yp = lowest_b1.intensities[best_i + 1];
            let step = 0.5 * (xp - xm);
            let denom = ym - 2.0 * yc + yp;
            if denom > 1e-9 && step.abs() > 0.0 {
                let delta = 0.5 * (ym - yp) / denom;
                if delta.abs() <= 1.0 {
                    return x0 + delta * step;
                }
            }
        }
        x0
    });

    // 2. Baseline intensity far from major dip -> R1A estimate
    let mut far_vals: Vec<f64> = lowest_b1
        .offsets_ppm
        .iter()
        .zip(lowest_b1.intensities.iter())
        .filter(|(&x, _)| (x - major_ppm).abs() >= 4.0)
        .map(|(_, &y)| y)
        .collect();
    if far_vals.is_empty() {
        far_vals = lowest_b1.intensities.clone();
    }
    far_vals.sort_by(|a, b| a.total_cmp(b));
    let base =
        far_vals[(far_vals.len() * 4 / 5).min(far_vals.len().saturating_sub(1))].clamp(0.05, 1.0);

    let r1a_s_inv = match lowest_b1.normalization {
        CestNormalization::ZeroTimePhaseCycled => {
            ((-base.clamp(0.05, 0.995).ln()) / lowest_b1.t_ex_s.max(1e-3)).clamp(0.2, 6.0)
        }
        CestNormalization::ThermalEquilibrium | CestNormalization::SameDelayZeroB1 => 1.4,
    };

    // 3. Estimate R2A from single-state major-dip width
    let mut best_r2a = 15.0;
    let mut best_single_err = f64::INFINITY;
    for &cand_r2a in &[4.0, 8.0, 12.0, 16.0, 22.0, 32.0, 50.0] {
        let single_params = Cest2StateParams {
            p_b: 0.0,
            k_ex_s_inv: 0.0,
            major_ppm,
            dw_ab_ppm: 0.0,
            r1a_s_inv,
            r2a_s_inv: cand_r2a,
            r1b_s_inv: r1a_s_inv,
            r2b_s_inv: cand_r2a,
            larmor_mhz: lowest_b1.larmor_mhz,
            t_ex_s: lowest_b1.t_ex_s,
            b1_hz: lowest_b1.b1_hz,
            b1_inhomogeneity_frac: 0.0,
            n_b1_quad: 1,
            normalization: lowest_b1.normalization,
        };
        if let Ok(sim) = simulate_cest_2state_profile(&single_params, &lowest_b1.offsets_ppm) {
            // Fit core of major dip (|x - major_ppm| <= 1.2 ppm)
            let err: f64 = lowest_b1
                .offsets_ppm
                .iter()
                .zip(lowest_b1.intensities.iter())
                .zip(sim.iter())
                .filter(|((&x, _), _)| (x - major_ppm).abs() <= 1.2)
                .map(|((_, &yo), &ys)| (yo - ys).powi(2))
                .sum();
            if err < best_single_err {
                best_single_err = err;
                best_r2a = cand_r2a;
            }
        }
    }

    // 4. Locate minor dip `dw_ab_ppm` as maximum positive residual `I_single(x) - I_obs(x)` summed
    // across all B1 profiles outside immediate major-dip core (`|x - major| >= 0.45 ppm`).
    let mut best_dw = 3.0;
    let mut max_dip_score = -1.0f64;
    for prof in res_profiles {
        let single_params = Cest2StateParams {
            p_b: 0.0,
            k_ex_s_inv: 0.0,
            major_ppm,
            dw_ab_ppm: 0.0,
            r1a_s_inv,
            r2a_s_inv: best_r2a,
            r1b_s_inv: r1a_s_inv,
            r2b_s_inv: best_r2a,
            larmor_mhz: prof.larmor_mhz,
            t_ex_s: prof.t_ex_s,
            b1_hz: prof.b1_hz,
            b1_inhomogeneity_frac: 0.0,
            n_b1_quad: 1,
            normalization: prof.normalization,
        };
        if let Ok(sim) = simulate_cest_2state_profile(&single_params, &prof.offsets_ppm) {
            for (idx, &x) in prof.offsets_ppm.iter().enumerate() {
                let dw = x - major_ppm;
                if dw.abs() < 0.45 {
                    continue;
                }
                let sig = prof
                    .uncertainties
                    .get(idx)
                    .copied()
                    .unwrap_or(0.005)
                    .max(1e-4);
                let mut score = (sim[idx] - prof.intensities[idx]) / sig;
                if idx > 0 {
                    score += 0.5 * (sim[idx - 1] - prof.intensities[idx - 1]) / sig;
                }
                if idx + 1 < prof.offsets_ppm.len() {
                    score += 0.5 * (sim[idx + 1] - prof.intensities[idx + 1]) / sig;
                }
                if score > max_dip_score {
                    max_dip_score = score;
                    best_dw = dw;
                }
            }
        }
    }

    CestResidueSeed {
        major_ppm,
        dw_ab_ppm: best_dw,
        r1a_s_inv,
        r2a_s_inv: best_r2a,
        r2b_s_inv: (best_r2a * 1.25).clamp(5.0, 200.0),
    }
}

/// Packed description of a single `(curve, profile)` forward evaluation request so CPU and WebGPU
/// backends share the exact same global Schur-complement Levenberg–Marquardt optimizer!
#[derive(Clone, Copy, Debug)]
pub struct CestCurveEvalSpec {
    pub profile_idx: usize,
    pub p_b: f64,
    pub k_ex_s_inv: f64,
    pub major_ppm: f64,
    pub dw_ab_ppm: f64,
    pub r1a_s_inv: f64,
    pub r2a_s_inv: f64,
    pub r1b_s_inv: f64,
    pub r2b_s_inv: f64,
}

/// Evaluate a batch of `CestCurveEvalSpec` items on CPU (`f64` Padé `[6/6]`).
pub fn eval_cest_curve_batch_cpu(
    profiles: &[CestProfileDataset],
    specs: &[CestCurveEvalSpec],
    n_b1_quad: usize,
) -> Result<Vec<Vec<f64>>, String> {
    let mut out = Vec::with_capacity(specs.len());
    for sp in specs {
        let prof = &profiles[sp.profile_idx];
        let params = Cest2StateParams {
            p_b: sp.p_b,
            k_ex_s_inv: sp.k_ex_s_inv,
            major_ppm: sp.major_ppm,
            dw_ab_ppm: sp.dw_ab_ppm,
            r1a_s_inv: sp.r1a_s_inv,
            r2a_s_inv: sp.r2a_s_inv,
            r1b_s_inv: sp.r1b_s_inv,
            r2b_s_inv: sp.r2b_s_inv,
            larmor_mhz: prof.larmor_mhz,
            t_ex_s: prof.t_ex_s,
            b1_hz: prof.b1_hz,
            b1_inhomogeneity_frac: prof.b1_inhomogeneity_frac,
            n_b1_quad,
            normalization: prof.normalization,
        };
        out.push(simulate_cest_2state_profile(&params, &prof.offsets_ppm)?);
    }
    Ok(out)
}

/// Fit global `(p_B, k_ex)` and per-residue `(major_ppm, dw_ab_ppm, R1A, R2A, R2B)` on CPU (`f64`).
pub fn fit_cest_global_cpu(
    profiles: &[CestProfileDataset],
    config: &CestFitConfig,
) -> Result<CestGlobalFitResult, String> {
    fit_cest_global_with_evaluator(profiles, config, |specs, n_quad| {
        eval_cest_curve_batch_cpu(profiles, specs, n_quad)
    })
}

/// Generic multi-`B1`, multi-residue global `CEST` / `DEST` Schur-complement Levenberg–Marquardt
/// solver accepting a batched curve evaluator (CPU `f64` or WebGPU `bloch_cest.wgsl` + final `f64`
/// covariance polish).
pub fn fit_cest_global_with_evaluator<FEval>(
    profiles: &[CestProfileDataset],
    config: &CestFitConfig,
    mut eval_batch: FEval,
) -> Result<CestGlobalFitResult, String>
where
    FEval: FnMut(&[CestCurveEvalSpec], usize) -> Result<Vec<Vec<f64>>, String>,
{
    if profiles.is_empty() {
        return Err("fit_cest_global requires at least one profile".to_string());
    }
    for (idx, p) in profiles.iter().enumerate() {
        if p.offsets_ppm.is_empty()
            || p.offsets_ppm.len() != p.intensities.len()
            || p.offsets_ppm.len() != p.uncertainties.len()
        {
            return Err(format!(
                "profile {idx} (seq_id={}) has empty or mismatched offsets/intensities/uncertainties",
                p.seq_id
            ));
        }
        if !p.t_ex_s.is_finite()
            || p.t_ex_s <= 0.0
            || !p.larmor_mhz.is_finite()
            || p.larmor_mhz <= 0.0
        {
            return Err(format!("profile {idx} has invalid t_ex_s or larmor_mhz"));
        }
    }

    // Group profiles by seq_id in deterministic sorted order
    let mut by_res: BTreeMap<i32, Vec<usize>> = BTreeMap::new();
    for (idx, p) in profiles.iter().enumerate() {
        by_res.entry(p.seq_id).or_default().push(idx);
    }
    let res_groups: Vec<(i32, Vec<usize>)> = by_res.into_iter().collect();
    let n_res = res_groups.len();

    let mut locals: Vec<CestLocalState> = res_groups
        .iter()
        .map(|(_, pidxs)| {
            let refs: Vec<&CestProfileDataset> = pidxs.iter().map(|&pi| &profiles[pi]).collect();
            let seed = estimate_residue_initial_seed(&refs);
            CestLocalState {
                major_ppm: seed.major_ppm,
                dw_ab_ppm: seed.dw_ab_ppm,
                ln_r1a: seed.r1a_s_inv.ln(),
                ln_r2a: seed.r2a_s_inv.ln(),
                ln_r2b: seed.r2b_s_inv.ln(),
            }
        })
        .collect();

    // Coarse batched `(p_B, k_ex)` grid search in 1 single batch call to start near global basin!
    let pb_candidates = [0.005, 0.010, 0.015, 0.025, 0.045, 0.080];
    let kex_candidates = [40.0, 90.0, 150.0, 260.0, 450.0, 800.0];
    let mut init_specs =
        Vec::with_capacity(pb_candidates.len() * kex_candidates.len() * profiles.len());
    for &pb_c in &pb_candidates {
        for &kex_c in &kex_candidates {
            for (ri, (_, pidxs)) in res_groups.iter().enumerate() {
                let loc = locals[ri];
                let r1a = loc.ln_r1a.exp();
                let r2a = loc.ln_r2a.exp();
                let r2b = loc.ln_r2b.exp();
                for &pi in pidxs {
                    init_specs.push(CestCurveEvalSpec {
                        profile_idx: pi,
                        p_b: pb_c,
                        k_ex_s_inv: kex_c,
                        major_ppm: loc.major_ppm,
                        dw_ab_ppm: loc.dw_ab_ppm,
                        r1a_s_inv: r1a,
                        r2a_s_inv: r2a,
                        r1b_s_inv: r1a,
                        r2b_s_inv: r2b,
                    });
                }
            }
        }
    }
    let init_curves = eval_batch(&init_specs, 1)?;
    let n_prof_total: usize = res_groups.iter().map(|(_, v)| v.len()).sum();
    let mut best_pb = 0.015;
    let mut best_kex = 150.0;
    let mut best_init_chi2 = f64::INFINITY;
    let mut cursor = 0usize;
    for &pb_c in &pb_candidates {
        for &kex_c in &kex_candidates {
            let mut chi2 = 0.0;
            for k in 0..n_prof_total {
                let sp = &init_specs[cursor + k];
                let prof = &profiles[sp.profile_idx];
                let sim = &init_curves[cursor + k];
                for j in 0..prof.intensities.len() {
                    let r = (prof.intensities[j] - sim[j]) / prof.uncertainties[j].max(1e-5);
                    chi2 += r * r;
                }
            }
            cursor += n_prof_total;
            if chi2 < best_init_chi2 {
                best_init_chi2 = chi2;
                best_pb = pb_c;
                best_kex = kex_c;
            }
        }
    }

    let mut ln_pb = logit_pb(best_pb);
    let mut ln_kex = best_kex.ln();
    let n_local_cols = if config.refine_major_ppm {
        5usize
    } else {
        4usize
    };

    // Evaluate base curves
    type BatchEvalFn<'a> =
        dyn FnMut(&[CestCurveEvalSpec], usize) -> Result<Vec<Vec<f64>>, String> + 'a;
    let eval_base = |ln_pb_v: f64,
                     ln_kex_v: f64,
                     locs: &[CestLocalState],
                     eval_fn: &mut BatchEvalFn<'_>|
     -> Result<(Vec<Vec<f64>>, f64), String> {
        let pb = inv_logit_pb(ln_pb_v);
        let kex = ln_kex_v.exp().clamp(1.0, 25_000.0);
        let mut specs = Vec::with_capacity(n_prof_total);
        for (ri, (_, pidxs)) in res_groups.iter().enumerate() {
            let loc = locs[ri];
            let r1a = loc.ln_r1a.exp().clamp(0.05, 50.0);
            let r2a = loc.ln_r2a.exp().clamp(0.2, 500.0);
            let r2b = loc.ln_r2b.exp().clamp(0.2, 100_000.0);
            for &pi in pidxs {
                specs.push(CestCurveEvalSpec {
                    profile_idx: pi,
                    p_b: pb,
                    k_ex_s_inv: kex,
                    major_ppm: loc.major_ppm,
                    dw_ab_ppm: loc.dw_ab_ppm,
                    r1a_s_inv: r1a,
                    r2a_s_inv: r2a,
                    r1b_s_inv: r1a,
                    r2b_s_inv: r2b,
                });
            }
        }
        let curves = eval_fn(&specs, config.n_b1_quad)?;
        let mut chi2 = 0.0;
        for (sp, sim) in specs.iter().zip(curves.iter()) {
            let prof = &profiles[sp.profile_idx];
            for j in 0..prof.intensities.len() {
                let r = (prof.intensities[j] - sim[j]) / prof.uncertainties[j].max(1e-5);
                chi2 += r * r;
            }
        }
        Ok((curves, chi2))
    };

    let (mut base_curves, mut current_chi2) = eval_base(ln_pb, ln_kex, &locals, &mut eval_batch)?;
    let mut lambda = 1e-2f64;
    let mut iterations = 0usize;
    let mut converged = false;

    // Finite-difference step sizes (`1e-3` is sweet-spot for both `f32` GPU and `f64` CPU Jacobians)
    let h_g = [2e-3f64, 2e-3f64];
    let h_l = [1e-3f64, 2e-3f64, 2e-3f64, 2e-3f64, 1e-3f64]; // [dw_ab, ln_r1a, ln_r2a, ln_r2b, major_ppm]

    for iter in 0..config.max_lm_iters {
        iterations = iter + 1;
        let n_cols = 2 + n_local_cols;
        let mut jac_specs = Vec::with_capacity(n_prof_total * n_cols);
        let pb0 = inv_logit_pb(ln_pb);
        let kex0 = ln_kex.exp().clamp(1.0, 25_000.0);
        let pb_plus = inv_logit_pb(ln_pb + h_g[0]);
        let kex_plus = (ln_kex + h_g[1]).exp().clamp(1.0, 25_000.0);

        for (ri, (_, pidxs)) in res_groups.iter().enumerate() {
            let loc = locals[ri];
            let r1a0 = loc.ln_r1a.exp().clamp(0.05, 50.0);
            let r2a0 = loc.ln_r2a.exp().clamp(0.2, 500.0);
            let r2b0 = loc.ln_r2b.exp().clamp(0.2, 100_000.0);
            let r1a_p = (loc.ln_r1a + h_l[1]).exp().clamp(0.05, 50.0);
            let r2a_p = (loc.ln_r2a + h_l[2]).exp().clamp(0.2, 500.0);
            let r2b_p = (loc.ln_r2b + h_l[3]).exp().clamp(0.2, 100_000.0);

            for &pi in pidxs {
                let base_sp = CestCurveEvalSpec {
                    profile_idx: pi,
                    p_b: pb0,
                    k_ex_s_inv: kex0,
                    major_ppm: loc.major_ppm,
                    dw_ab_ppm: loc.dw_ab_ppm,
                    r1a_s_inv: r1a0,
                    r2a_s_inv: r2a0,
                    r1b_s_inv: r1a0,
                    r2b_s_inv: r2b0,
                };
                // Col 0: +h ln_pb
                jac_specs.push(CestCurveEvalSpec {
                    p_b: pb_plus,
                    ..base_sp
                });
                // Col 1: +h ln_kex
                jac_specs.push(CestCurveEvalSpec {
                    k_ex_s_inv: kex_plus,
                    ..base_sp
                });
                // Col 2: +h dw_ab
                jac_specs.push(CestCurveEvalSpec {
                    dw_ab_ppm: loc.dw_ab_ppm + h_l[0],
                    ..base_sp
                });
                // Col 3: +h ln_r1a
                jac_specs.push(CestCurveEvalSpec {
                    r1a_s_inv: r1a_p,
                    r1b_s_inv: r1a_p,
                    ..base_sp
                });
                // Col 4: +h ln_r2a
                jac_specs.push(CestCurveEvalSpec {
                    r2a_s_inv: r2a_p,
                    ..base_sp
                });
                // Col 5: +h ln_r2b
                jac_specs.push(CestCurveEvalSpec {
                    r2b_s_inv: r2b_p,
                    ..base_sp
                });
                if config.refine_major_ppm {
                    // Col 6: +h major_ppm
                    jac_specs.push(CestCurveEvalSpec {
                        major_ppm: loc.major_ppm + h_l[4],
                        ..base_sp
                    });
                }
            }
        }

        let jac_curves = eval_batch(&jac_specs, config.n_b1_quad)?;

        // Assemble sparse arrow-head normal equations `J^T J` and `J^T r` (`r = (y - f)/sigma`)
        let mut h_gg = [[0.0f64; 2]; 2];
        let mut g_g = [0.0f64; 2];
        let mut h_ll = vec![vec![vec![0.0f64; n_local_cols]; n_local_cols]; n_res];
        let mut h_gl = vec![vec![[0.0f64; 2]; n_local_cols]; n_res]; // h_gl[ri][lc][gc]
        let mut g_l = vec![vec![0.0f64; n_local_cols]; n_res];

        let mut prof_flat_idx = 0usize;
        for (ri, (_, pidxs)) in res_groups.iter().enumerate() {
            for &pi in pidxs {
                let prof = &profiles[pi];
                let base_sim = &base_curves[prof_flat_idx];
                let jc_off = prof_flat_idx * n_cols;

                for j in 0..prof.intensities.len() {
                    let inv_sig = 1.0 / prof.uncertainties[j].max(1e-5);
                    let res_j = (prof.intensities[j] - base_sim[j]) * inv_sig;

                    let jg0 = (jac_curves[jc_off][j] - base_sim[j]) * inv_sig / h_g[0];
                    let jg1 = (jac_curves[jc_off + 1][j] - base_sim[j]) * inv_sig / h_g[1];
                    let jg = [jg0, jg1];

                    let mut jl = [0.0f64; 5];
                    for lc in 0..n_local_cols {
                        jl[lc] = (jac_curves[jc_off + 2 + lc][j] - base_sim[j]) * inv_sig / h_l[lc];
                    }

                    for a in 0..2 {
                        g_g[a] += jg[a] * res_j;
                        for b in 0..2 {
                            h_gg[a][b] += jg[a] * jg[b];
                        }
                    }
                    for a in 0..n_local_cols {
                        g_l[ri][a] += jl[a] * res_j;
                        for gc in 0..2 {
                            h_gl[ri][a][gc] += jl[a] * jg[gc];
                        }
                        for b in 0..n_local_cols {
                            h_ll[ri][a][b] += jl[a] * jl[b];
                        }
                    }
                }
                prof_flat_idx += 1;
            }
        }

        let max_grad = g_g
            .iter()
            .copied()
            .chain(g_l.iter().flat_map(|v| v.iter().copied()))
            .fold(0.0f64, |m, v| m.max(v.abs()));
        if max_grad < config.grad_tol {
            converged = true;
            break;
        }

        // Try LM damping step via Schur complement
        let mut accepted = false;
        for _sub in 0..8 {
            if let Ok((d_global, d_locals)) =
                solve_arrowhead_lm(&h_gg, &g_g, &h_ll, &h_gl, &g_l, lambda, n_local_cols)
            {
                let step_norm = d_global[0].hypot(d_global[1]).max(
                    d_locals
                        .iter()
                        .flat_map(|v| v.iter())
                        .fold(0.0f64, |m, v| m.max(v.abs())),
                );

                let trial_ln_pb = (ln_pb + d_global[0]).clamp(-11.0, -0.05);
                let trial_ln_kex = (ln_kex + d_global[1]).clamp(0.0, 10.1);
                let mut trial_locals = locals.clone();
                for (ri, tl) in trial_locals.iter_mut().enumerate() {
                    tl.dw_ab_ppm = (tl.dw_ab_ppm + d_locals[ri][0]).clamp(-40.0, 40.0);
                    tl.ln_r1a = (tl.ln_r1a + d_locals[ri][1]).clamp(-3.0, 3.9);
                    tl.ln_r2a = (tl.ln_r2a + d_locals[ri][2]).clamp(-1.6, 6.2);
                    tl.ln_r2b = (tl.ln_r2b + d_locals[ri][3]).clamp(-1.6, 11.5);
                    if config.refine_major_ppm {
                        tl.major_ppm += d_locals[ri][4].clamp(-1.5, 1.5);
                    }
                }

                let (trial_curves, trial_chi2) =
                    eval_base(trial_ln_pb, trial_ln_kex, &trial_locals, &mut eval_batch)?;
                if trial_chi2 < current_chi2 {
                    let rel_impr = (current_chi2 - trial_chi2) / current_chi2.max(1.0);
                    ln_pb = trial_ln_pb;
                    ln_kex = trial_ln_kex;
                    locals = trial_locals;
                    base_curves = trial_curves;
                    current_chi2 = trial_chi2;
                    lambda = (lambda * 0.35).max(1e-9);
                    accepted = true;
                    if step_norm < config.step_tol || rel_impr < 1e-10 {
                        converged = true;
                    }
                    break;
                }
            }
            lambda = (lambda * 4.0).min(1e8);
        }

        if converged || !accepted {
            break;
        }
    }

    // Final exact `f64` CPU polish (2 steps) & covariance calculation so GPU `f32` evaluations
    // never limit asymptotic parameter precision or `1-sigma` standard errors!
    for _polish in 0..3 {
        let (cpu_base, cpu_chi2) = eval_base(ln_pb, ln_kex, &locals, &mut |sp, nq| {
            eval_cest_curve_batch_cpu(profiles, sp, nq)
        })?;
        base_curves = cpu_base;
        current_chi2 = cpu_chi2;

        let h_fine = 1e-5f64;
        let n_cols = 2 + n_local_cols;
        let mut jac_specs = Vec::with_capacity(n_prof_total * n_cols);
        let pb0 = inv_logit_pb(ln_pb);
        let kex0 = ln_kex.exp();
        let pb_p = inv_logit_pb(ln_pb + h_fine);
        let kex_p = (ln_kex + h_fine).exp();
        for (ri, (_, pidxs)) in res_groups.iter().enumerate() {
            let loc = locals[ri];
            let r1a0 = loc.ln_r1a.exp();
            let r2a0 = loc.ln_r2a.exp();
            let r2b0 = loc.ln_r2b.exp();
            let r1a_p = (loc.ln_r1a + h_fine).exp();
            let r2a_p = (loc.ln_r2a + h_fine).exp();
            let r2b_p = (loc.ln_r2b + h_fine).exp();
            for &pi in pidxs {
                let bsp = CestCurveEvalSpec {
                    profile_idx: pi,
                    p_b: pb0,
                    k_ex_s_inv: kex0,
                    major_ppm: loc.major_ppm,
                    dw_ab_ppm: loc.dw_ab_ppm,
                    r1a_s_inv: r1a0,
                    r2a_s_inv: r2a0,
                    r1b_s_inv: r1a0,
                    r2b_s_inv: r2b0,
                };
                jac_specs.push(CestCurveEvalSpec { p_b: pb_p, ..bsp });
                jac_specs.push(CestCurveEvalSpec {
                    k_ex_s_inv: kex_p,
                    ..bsp
                });
                jac_specs.push(CestCurveEvalSpec {
                    dw_ab_ppm: loc.dw_ab_ppm + h_fine,
                    ..bsp
                });
                jac_specs.push(CestCurveEvalSpec {
                    r1a_s_inv: r1a_p,
                    r1b_s_inv: r1a_p,
                    ..bsp
                });
                jac_specs.push(CestCurveEvalSpec {
                    r2a_s_inv: r2a_p,
                    ..bsp
                });
                jac_specs.push(CestCurveEvalSpec {
                    r2b_s_inv: r2b_p,
                    ..bsp
                });
                if config.refine_major_ppm {
                    jac_specs.push(CestCurveEvalSpec {
                        major_ppm: loc.major_ppm + h_fine,
                        ..bsp
                    });
                }
            }
        }
        let jac_curves = eval_cest_curve_batch_cpu(profiles, &jac_specs, config.n_b1_quad)?;

        let mut h_gg = [[0.0f64; 2]; 2];
        let mut g_g = [0.0f64; 2];
        let mut h_ll = vec![vec![vec![0.0f64; n_local_cols]; n_local_cols]; n_res];
        let mut h_gl = vec![vec![[0.0f64; 2]; n_local_cols]; n_res];
        let mut g_l = vec![vec![0.0f64; n_local_cols]; n_res];
        let mut pf = 0usize;
        for (ri, (_, pidxs)) in res_groups.iter().enumerate() {
            for &pi in pidxs {
                let prof = &profiles[pi];
                let bsim = &base_curves[pf];
                let off = pf * n_cols;
                for j in 0..prof.intensities.len() {
                    let inv_sig = 1.0 / prof.uncertainties[j].max(1e-5);
                    let rj = (prof.intensities[j] - bsim[j]) * inv_sig;
                    let jg = [
                        (jac_curves[off][j] - bsim[j]) * inv_sig / h_fine,
                        (jac_curves[off + 1][j] - bsim[j]) * inv_sig / h_fine,
                    ];
                    let mut jl = [0.0f64; 5];
                    for lc in 0..n_local_cols {
                        jl[lc] = (jac_curves[off + 2 + lc][j] - bsim[j]) * inv_sig / h_fine;
                    }
                    for a in 0..2 {
                        g_g[a] += jg[a] * rj;
                        for b in 0..2 {
                            h_gg[a][b] += jg[a] * jg[b];
                        }
                    }
                    for a in 0..n_local_cols {
                        g_l[ri][a] += jl[a] * rj;
                        for gc in 0..2 {
                            h_gl[ri][a][gc] += jl[a] * jg[gc];
                        }
                        for b in 0..n_local_cols {
                            h_ll[ri][a][b] += jl[a] * jl[b];
                        }
                    }
                }
                pf += 1;
            }
        }
        if let Ok((dg, dl)) =
            solve_arrowhead_lm(&h_gg, &g_g, &h_ll, &h_gl, &g_l, 1e-6, n_local_cols)
        {
            let t_pb = (ln_pb + dg[0]).clamp(-11.0, -0.05);
            let t_kex = (ln_kex + dg[1]).clamp(0.0, 10.1);
            let mut t_locs = locals.clone();
            for (ri, tl) in t_locs.iter_mut().enumerate() {
                tl.dw_ab_ppm += dl[ri][0];
                tl.ln_r1a += dl[ri][1];
                tl.ln_r2a += dl[ri][2];
                tl.ln_r2b += dl[ri][3];
                if config.refine_major_ppm {
                    tl.major_ppm += dl[ri][4];
                }
            }
            if let Ok((tc, tchi2)) = eval_base(t_pb, t_kex, &t_locs, &mut |sp, nq| {
                eval_cest_curve_batch_cpu(profiles, sp, nq)
            }) {
                if tchi2 < current_chi2 {
                    ln_pb = t_pb;
                    ln_kex = t_kex;
                    locals = t_locs;
                    base_curves = tc;
                    current_chi2 = tchi2;
                }
            }
        }
    }

    // Extract asymptotic covariance SEs via Schur complement at optimum
    let (cov_global, cov_locals) = compute_cest_covariance(
        profiles,
        &res_groups,
        ln_pb,
        ln_kex,
        &locals,
        config,
        &base_curves,
    )?;

    let p_b = inv_logit_pb(ln_pb);
    let dp_dlogit = d_inv_logit_pb(ln_pb);
    let p_b_err = (cov_global[0][0].max(0.0)).sqrt() * dp_dlogit;
    let k_ex_s_inv = ln_kex.exp();
    let k_ex_err = (cov_global[1][1].max(0.0)).sqrt() * k_ex_s_inv;

    let mut total_pts = 0usize;
    let mut residue_fits = Vec::with_capacity(n_res);
    let mut pf_cursor = 0usize;
    for (ri, (seq_id, pidxs)) in res_groups.iter().enumerate() {
        let loc = locals[ri];
        let r1a = loc.ln_r1a.exp();
        let r2a = loc.ln_r2a.exp();
        let r2b = loc.ln_r2b.exp();
        let mut r_chi2 = 0.0;
        let mut r_pts = 0usize;
        for &pi in pidxs {
            let prof = &profiles[pi];
            let sim = &base_curves[pf_cursor];
            pf_cursor += 1;
            r_pts += prof.intensities.len();
            for j in 0..prof.intensities.len() {
                let r = (prof.intensities[j] - sim[j]) / prof.uncertainties[j].max(1e-5);
                r_chi2 += r * r;
            }
        }
        total_pts += r_pts;
        let cl = &cov_locals[ri];
        residue_fits.push(CestResidueFit {
            seq_id: *seq_id,
            res_name: profiles[pidxs[0]].res_name.clone(),
            major_ppm: loc.major_ppm,
            major_ppm_err: if config.refine_major_ppm {
                cl[4].max(0.0).sqrt()
            } else {
                0.0
            },
            dw_ab_ppm: loc.dw_ab_ppm,
            dw_ab_ppm_err: cl[0].max(0.0).sqrt(),
            r1a_s_inv: r1a,
            r1a_err: cl[1].max(0.0).sqrt() * r1a,
            r2a_s_inv: r2a,
            r2a_err: cl[2].max(0.0).sqrt() * r2a,
            r2b_s_inv: r2b,
            r2b_err: cl[3].max(0.0).sqrt() * r2b,
            chi2: r_chi2,
            n_points: r_pts,
        });
    }

    let n_params = 2 + n_res * n_local_cols;
    let dof = total_pts.saturating_sub(n_params);
    let reduced_chi2 = current_chi2 / (dof.max(1) as f64);

    let chi2_surface = if config.compute_chi2_surface {
        let (npb, nkex) = config.surface_grid_dims;
        if npb >= 3 && nkex >= 3 {
            let pb_lo = (p_b * 0.60).max(0.0005);
            let pb_hi = (p_b * 1.40).min(0.45);
            let kex_lo = (k_ex_s_inv * 0.60).max(5.0);
            let kex_hi = (k_ex_s_inv * 1.40).min(10_000.0);
            let pb_grid: Vec<f64> = (0..npb)
                .map(|i| pb_lo + (pb_hi - pb_lo) * (i as f64) / ((npb - 1) as f64))
                .collect();
            let kex_grid: Vec<f64> = (0..nkex)
                .map(|i| kex_lo + (kex_hi - kex_lo) * (i as f64) / ((nkex - 1) as f64))
                .collect();

            let mut surf_specs = Vec::with_capacity((npb * nkex + npb + nkex) * n_prof_total);
            let push_pt = |pb_v: f64, kex_v: f64, dst: &mut Vec<CestCurveEvalSpec>| {
                for (ri, (_, pidxs)) in res_groups.iter().enumerate() {
                    let loc = locals[ri];
                    let r1a = loc.ln_r1a.exp();
                    let r2a = loc.ln_r2a.exp();
                    let r2b = loc.ln_r2b.exp();
                    for &pi in pidxs {
                        dst.push(CestCurveEvalSpec {
                            profile_idx: pi,
                            p_b: pb_v,
                            k_ex_s_inv: kex_v,
                            major_ppm: loc.major_ppm,
                            dw_ab_ppm: loc.dw_ab_ppm,
                            r1a_s_inv: r1a,
                            r2a_s_inv: r2a,
                            r1b_s_inv: r1a,
                            r2b_s_inv: r2b,
                        });
                    }
                }
            };
            for &pb_v in &pb_grid {
                push_pt(pb_v, k_ex_s_inv, &mut surf_specs);
            }
            for &kex_v in &kex_grid {
                push_pt(p_b, kex_v, &mut surf_specs);
            }
            for &pb_v in &pb_grid {
                for &kex_v in &kex_grid {
                    push_pt(pb_v, kex_v, &mut surf_specs);
                }
            }
            let surf_curves = eval_batch(&surf_specs, config.n_b1_quad)?;
            let score_chunk = |chunk_idx: usize| -> f64 {
                let start = chunk_idx * n_prof_total;
                let mut c2 = 0.0;
                for k in 0..n_prof_total {
                    let sp = &surf_specs[start + k];
                    let prof = &profiles[sp.profile_idx];
                    let sim = &surf_curves[start + k];
                    for j in 0..prof.intensities.len() {
                        let r = (prof.intensities[j] - sim[j]) / prof.uncertainties[j].max(1e-5);
                        c2 += r * r;
                    }
                }
                c2
            };
            let chi2_1d_pb: Vec<f64> = (0..npb).map(&score_chunk).collect();
            let chi2_1d_kex: Vec<f64> = (0..nkex).map(|i| score_chunk(npb + i)).collect();
            let chi2_2d: Vec<f64> = (0..(npb * nkex))
                .map(|i| score_chunk(npb + nkex + i))
                .collect();

            Some(CestChi2Surface {
                pb_grid,
                kex_grid,
                chi2_1d_pb,
                chi2_1d_kex,
                chi2_2d,
            })
        } else {
            None
        }
    } else {
        None
    };

    Ok(CestGlobalFitResult {
        p_b,
        p_b_err,
        k_ex_s_inv,
        k_ex_err,
        k_ab_s_inv: p_b * k_ex_s_inv,
        k_ba_s_inv: (1.0 - p_b) * k_ex_s_inv,
        total_chi2: current_chi2,
        reduced_chi2,
        dof,
        iterations,
        converged,
        residues: residue_fits,
        chi2_surface,
    })
}

#[inline]
fn logit_pb(pb: f64) -> f64 {
    let p = pb.clamp(1e-5, 0.495) / 0.50;
    (p / (1.0 - p)).ln()
}

#[inline]
fn inv_logit_pb(z: f64) -> f64 {
    let s = 1.0 / (1.0 + (-z).exp());
    0.50 * s
}

#[inline]
fn d_inv_logit_pb(z: f64) -> f64 {
    let s = 1.0 / (1.0 + (-z).exp());
    0.50 * s * (1.0 - s)
}

fn invert_small_sym(a: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
    let n = a.len();
    let mut lu = a.to_vec();
    let mut rhs = vec![vec![0.0; n]; n];
    for (i, r) in rhs.iter_mut().enumerate() {
        r[i] = 1.0;
    }

    for col in 0..n {
        let mut piv = col;
        let mut max_v = lu[col][col].abs();
        for (r, row) in lu.iter().enumerate().take(n).skip(col + 1) {
            if row[col].abs() > max_v {
                max_v = row[col].abs();
                piv = r;
            }
        }
        if !max_v.is_finite() || max_v < 1e-24 {
            return Err("singular block in Schur-complement solve".to_string());
        }
        if piv != col {
            lu.swap(col, piv);
            rhs.swap(col, piv);
        }
        let inv_d = 1.0 / lu[col][col];
        for r in (col + 1)..n {
            let f = lu[r][col] * inv_d;
            for c in (col + 1)..n {
                lu[r][c] -= f * lu[col][c];
            }
            for c in 0..n {
                rhs[r][c] -= f * rhs[col][c];
            }
        }
    }

    let mut inv = vec![vec![0.0; n]; n];
    for i in (0..n).rev() {
        let inv_d = 1.0 / lu[i][i];
        for col in 0..n {
            let mut s = rhs[i][col];
            for k in (i + 1)..n {
                s -= lu[i][k] * inv[k][col];
            }
            inv[i][col] = s * inv_d;
        }
    }
    Ok(inv)
}

fn solve_arrowhead_lm(
    h_gg: &[[f64; 2]; 2],
    g_g: &[f64; 2],
    h_ll: &[Vec<Vec<f64>>],
    h_gl: &[Vec<[f64; 2]>],
    g_l: &[Vec<f64>],
    lambda: f64,
    n_l: usize,
) -> Result<([f64; 2], Vec<Vec<f64>>), String> {
    let n_res = h_ll.len();
    let mut s_gg = *h_gg;
    for a in 0..2 {
        s_gg[a][a] += lambda * (h_gg[a][a].abs().max(1.0));
    }
    let mut rhs_g = *g_g;
    let mut inv_ll = Vec::with_capacity(n_res);

    for ri in 0..n_res {
        let mut blk = h_ll[ri].clone();
        for c in 0..n_l {
            blk[c][c] += lambda * (blk[c][c].abs().max(1.0));
        }
        let inv_b = invert_small_sym(&blk)?;
        // Subtract H_gi * inv_b * H_ig from s_gg and H_gi * inv_b * g_l from rhs_g
        // Let w[c][gc] = sum_k inv_b[c][k] * h_gl[ri][k][gc]
        let mut w = vec![[0.0f64; 2]; n_l];
        let mut u = vec![0.0f64; n_l];
        for c in 0..n_l {
            for k in 0..n_l {
                w[c][0] += inv_b[c][k] * h_gl[ri][k][0];
                w[c][1] += inv_b[c][k] * h_gl[ri][k][1];
                u[c] += inv_b[c][k] * g_l[ri][k];
            }
        }
        for ga in 0..2 {
            for c in 0..n_l {
                rhs_g[ga] -= h_gl[ri][c][ga] * u[c];
                for gb in 0..2 {
                    s_gg[ga][gb] -= h_gl[ri][c][ga] * w[c][gb];
                }
            }
        }
        inv_ll.push((inv_b, w, u));
    }

    let det = s_gg[0][0] * s_gg[1][1] - s_gg[0][1] * s_gg[1][0];
    if !det.is_finite() || det.abs() < 1e-24 {
        return Err("singular global 2x2 Schur complement".to_string());
    }
    let d_global = [
        (s_gg[1][1] * rhs_g[0] - s_gg[0][1] * rhs_g[1]) / det,
        (-s_gg[1][0] * rhs_g[0] + s_gg[0][0] * rhs_g[1]) / det,
    ];

    let mut d_locals = Vec::with_capacity(n_res);
    for (_inv_b, w, u) in &inv_ll {
        let mut dl = vec![0.0f64; n_l];
        for c in 0..n_l {
            dl[c] = u[c] - (w[c][0] * d_global[0] + w[c][1] * d_global[1]);
        }
        d_locals.push(dl);
    }

    Ok((d_global, d_locals))
}

#[allow(clippy::type_complexity)]
fn compute_cest_covariance<LocalT: Copy + CestLocalFields>(
    profiles: &[CestProfileDataset],
    res_groups: &[(i32, Vec<usize>)],
    ln_pb: f64,
    ln_kex: f64,
    locals: &[LocalT],
    config: &CestFitConfig,
    base_curves: &[Vec<f64>],
) -> Result<([[f64; 2]; 2], Vec<[f64; 5]>), String> {
    let n_res = res_groups.len();
    let n_l = if config.refine_major_ppm {
        5usize
    } else {
        4usize
    };
    let n_cols = 2 + n_l;
    let n_prof_total: usize = res_groups.iter().map(|(_, v)| v.len()).sum();
    let h = 1e-5f64;

    let pb0 = inv_logit_pb(ln_pb);
    let kex0 = ln_kex.exp();
    let pb_p = inv_logit_pb(ln_pb + h);
    let kex_p = (ln_kex + h).exp();

    let mut jac_specs = Vec::with_capacity(n_prof_total * n_cols);
    for (ri, (_, pidxs)) in res_groups.iter().enumerate() {
        let (maj, dw, lr1a, lr2a, lr2b) = locals[ri].fields();
        let r1a0 = lr1a.exp();
        let r2a0 = lr2a.exp();
        let r2b0 = lr2b.exp();
        let r1a_p = (lr1a + h).exp();
        let r2a_p = (lr2a + h).exp();
        let r2b_p = (lr2b + h).exp();
        for &pi in pidxs {
            let bsp = CestCurveEvalSpec {
                profile_idx: pi,
                p_b: pb0,
                k_ex_s_inv: kex0,
                major_ppm: maj,
                dw_ab_ppm: dw,
                r1a_s_inv: r1a0,
                r2a_s_inv: r2a0,
                r1b_s_inv: r1a0,
                r2b_s_inv: r2b0,
            };
            jac_specs.push(CestCurveEvalSpec { p_b: pb_p, ..bsp });
            jac_specs.push(CestCurveEvalSpec {
                k_ex_s_inv: kex_p,
                ..bsp
            });
            jac_specs.push(CestCurveEvalSpec {
                dw_ab_ppm: dw + h,
                ..bsp
            });
            jac_specs.push(CestCurveEvalSpec {
                r1a_s_inv: r1a_p,
                r1b_s_inv: r1a_p,
                ..bsp
            });
            jac_specs.push(CestCurveEvalSpec {
                r2a_s_inv: r2a_p,
                ..bsp
            });
            jac_specs.push(CestCurveEvalSpec {
                r2b_s_inv: r2b_p,
                ..bsp
            });
            if config.refine_major_ppm {
                jac_specs.push(CestCurveEvalSpec {
                    major_ppm: maj + h,
                    ..bsp
                });
            }
        }
    }

    let jac_curves = eval_cest_curve_batch_cpu(profiles, &jac_specs, config.n_b1_quad)?;
    let mut h_gg = [[0.0f64; 2]; 2];
    let mut h_ll = vec![vec![vec![0.0f64; n_l]; n_l]; n_res];
    let mut h_gl = vec![vec![[0.0f64; 2]; n_l]; n_res];

    let mut pf = 0usize;
    for (ri, (_, pidxs)) in res_groups.iter().enumerate() {
        for &pi in pidxs {
            let prof = &profiles[pi];
            let bsim = &base_curves[pf];
            let off = pf * n_cols;
            for j in 0..prof.intensities.len() {
                let inv_sig = 1.0 / prof.uncertainties[j].max(1e-5);
                let jg = [
                    (jac_curves[off][j] - bsim[j]) * inv_sig / h,
                    (jac_curves[off + 1][j] - bsim[j]) * inv_sig / h,
                ];
                let mut jl = [0.0f64; 5];
                for lc in 0..n_l {
                    jl[lc] = (jac_curves[off + 2 + lc][j] - bsim[j]) * inv_sig / h;
                }
                for a in 0..2 {
                    for b in 0..2 {
                        h_gg[a][b] += jg[a] * jg[b];
                    }
                }
                for a in 0..n_l {
                    for gc in 0..2 {
                        h_gl[ri][a][gc] += jl[a] * jg[gc];
                    }
                    for b in 0..n_l {
                        h_ll[ri][a][b] += jl[a] * jl[b];
                    }
                }
            }
            pf += 1;
        }
    }

    let mut s_gg = h_gg;
    s_gg[0][0] += 1e-10;
    s_gg[1][1] += 1e-10;
    let mut inv_blocks = Vec::with_capacity(n_res);
    for ri in 0..n_res {
        let mut blk = h_ll[ri].clone();
        for c in 0..n_l {
            blk[c][c] += 1e-10;
        }
        let inv_b = invert_small_sym(&blk).unwrap_or_else(|_| {
            let mut diag = vec![vec![0.0; n_l]; n_l];
            for c in 0..n_l {
                diag[c][c] = 1.0 / blk[c][c].max(1e-6);
            }
            diag
        });
        let mut w = vec![[0.0f64; 2]; n_l];
        for c in 0..n_l {
            for k in 0..n_l {
                w[c][0] += inv_b[c][k] * h_gl[ri][k][0];
                w[c][1] += inv_b[c][k] * h_gl[ri][k][1];
            }
        }
        for ga in 0..2 {
            for c in 0..n_l {
                for gb in 0..2 {
                    s_gg[ga][gb] -= h_gl[ri][c][ga] * w[c][gb];
                }
            }
        }
        inv_blocks.push((inv_b, w));
    }

    let det = (s_gg[0][0] * s_gg[1][1] - s_gg[0][1] * s_gg[1][0]).max(1e-18);
    let cov_g = [
        [s_gg[1][1] / det, -s_gg[0][1] / det],
        [-s_gg[1][0] / det, s_gg[0][0] / det],
    ];

    let mut cov_l = Vec::with_capacity(n_res);
    for (inv_b, w) in &inv_blocks {
        let mut diag_var = [0.0f64; 5];
        for c in 0..n_l {
            // Var(theta_l[c]) = inv_b[c][c] + w[c]^T * cov_g * w[c]
            let extra = w[c][0] * (cov_g[0][0] * w[c][0] + cov_g[0][1] * w[c][1])
                + w[c][1] * (cov_g[1][0] * w[c][0] + cov_g[1][1] * w[c][1]);
            diag_var[c] = (inv_b[c][c] + extra).max(0.0);
        }
        cov_l.push(diag_var);
    }

    Ok((cov_g, cov_l))
}

#[derive(Clone, Copy)]
struct CestLocalState {
    major_ppm: f64,
    dw_ab_ppm: f64,
    ln_r1a: f64,
    ln_r2a: f64,
    ln_r2b: f64,
}

trait CestLocalFields {
    fn fields(self) -> (f64, f64, f64, f64, f64);
}

impl CestLocalFields for CestLocalState {
    fn fields(self) -> (f64, f64, f64, f64, f64) {
        (
            self.major_ppm,
            self.dw_ab_ppm,
            self.ln_r1a,
            self.ln_r2a,
            self.ln_r2b,
        )
    }
}

// ============================================================================
// 4. TSV & JSON Serialization / Deserialization
// ============================================================================

/// Serialize experimental `CEST` / `DEST` profiles to a clean tab-separated table.
pub fn write_cest_profiles_tsv(profiles: &[CestProfileDataset]) -> String {
    let mut out = String::from(
        "seq_id\tres_name\tlarmor_mhz\tt_ex_s\tb1_hz\tb1_inhom_frac\tnormalization\tmajor_ppm_hint\toffset_ppm\tintensity\tuncertainty\n",
    );
    for p in profiles {
        let hint_str = p
            .major_ppm_hint
            .map(|v| format!("{v:.5}"))
            .unwrap_or_else(|| "NA".to_string());
        for i in 0..p.offsets_ppm.len() {
            out.push_str(&format!(
                "{}\t{}\t{:.5}\t{:.5}\t{:.4}\t{:.4}\t{}\t{}\t{:.5}\t{:.6}\t{:.6}\n",
                p.seq_id,
                p.res_name,
                p.larmor_mhz,
                p.t_ex_s,
                p.b1_hz,
                p.b1_inhomogeneity_frac,
                p.normalization.as_str(),
                hint_str,
                p.offsets_ppm[i],
                p.intensities[i],
                p.uncertainties[i],
            ));
        }
    }
    out
}

/// Parse experimental `CEST` / `DEST` profiles from TSV text.
pub fn read_cest_profiles_tsv(tsv: &str) -> Result<Vec<CestProfileDataset>, String> {
    type GroupKey = (i32, String, u64, u64, u64, u32);
    let mut groups: BTreeMap<GroupKey, CestProfileDataset> = BTreeMap::new();

    for (line_no, raw) in tsv.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols[0] == "seq_id" {
            continue;
        }
        if cols.len() < 11 {
            return Err(format!(
                "line {}: expected 11 TSV columns in CEST profile table, got {}",
                line_no + 1,
                cols.len()
            ));
        }
        let seq_id: i32 = cols[0]
            .parse()
            .map_err(|e| format!("line {}: invalid seq_id: {e}", line_no + 1))?;
        let res_name = cols[1].trim().to_string();
        let larmor_mhz: f64 = cols[2]
            .parse()
            .map_err(|e| format!("line {}: invalid larmor_mhz: {e}", line_no + 1))?;
        let t_ex_s: f64 = cols[3]
            .parse()
            .map_err(|e| format!("line {}: invalid t_ex_s: {e}", line_no + 1))?;
        let b1_hz: f64 = cols[4]
            .parse()
            .map_err(|e| format!("line {}: invalid b1_hz: {e}", line_no + 1))?;
        let b1_inhom: f64 = cols[5]
            .parse()
            .map_err(|e| format!("line {}: invalid b1_inhom_frac: {e}", line_no + 1))?;
        let norm = CestNormalization::from_str_label(cols[6])?;
        let major_hint = if cols[7].eq_ignore_ascii_case("na") || cols[7].trim().is_empty() {
            None
        } else {
            Some(
                cols[7]
                    .parse::<f64>()
                    .map_err(|e| format!("line {}: invalid major_ppm_hint: {e}", line_no + 1))?,
            )
        };
        let offset_ppm: f64 = cols[8]
            .parse()
            .map_err(|e| format!("line {}: invalid offset_ppm: {e}", line_no + 1))?;
        let intensity: f64 = cols[9]
            .parse()
            .map_err(|e| format!("line {}: invalid intensity: {e}", line_no + 1))?;
        let uncertainty: f64 = cols[10]
            .parse()
            .map_err(|e| format!("line {}: invalid uncertainty: {e}", line_no + 1))?;

        let key = (
            seq_id,
            res_name.clone(),
            (larmor_mhz * 1e5).round() as u64,
            (t_ex_s * 1e6).round() as u64,
            (b1_hz * 1e4).round() as u64,
            norm.as_code(),
        );
        let entry = groups.entry(key).or_insert_with(|| CestProfileDataset {
            seq_id,
            res_name,
            larmor_mhz,
            t_ex_s,
            b1_hz,
            b1_inhomogeneity_frac: b1_inhom,
            normalization: norm,
            major_ppm_hint: major_hint,
            offsets_ppm: Vec::new(),
            intensities: Vec::new(),
            uncertainties: Vec::new(),
        });
        entry.offsets_ppm.push(offset_ppm);
        entry.intensities.push(intensity);
        entry.uncertainties.push(uncertainty);
    }

    Ok(groups.into_values().collect())
}

/// Format a [`CestGlobalFitResult`] summary as TSV.
pub fn format_cest_fit_tsv(res: &CestGlobalFitResult) -> String {
    let mut out = format!(
        "# p_b={:.6}\tp_b_err={:.6}\tk_ex_s_inv={:.4}\tk_ex_err={:.4}\ttotal_chi2={:.4}\treduced_chi2={:.4}\tdof={}\n",
        res.p_b, res.p_b_err, res.k_ex_s_inv, res.k_ex_err, res.total_chi2, res.reduced_chi2, res.dof
    );
    out.push_str(
        "seq_id\tres_name\tmajor_ppm\tmajor_ppm_err\tdw_ab_ppm\tdw_ab_ppm_err\tr1a_s_inv\tr1a_err\tr2a_s_inv\tr2a_err\tr2b_s_inv\tr2b_err\tchi2\tn_points\n",
    );
    for r in &res.residues {
        out.push_str(&format!(
            "{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{}\n",
            r.seq_id,
            r.res_name,
            r.major_ppm,
            r.major_ppm_err,
            r.dw_ab_ppm,
            r.dw_ab_ppm_err,
            r.r1a_s_inv,
            r.r1a_err,
            r.r2a_s_inv,
            r.r2a_err,
            r.r2b_s_inv,
            r.r2b_err,
            r.chi2,
            r.n_points
        ));
    }
    out
}

/// Format a [`CestGlobalFitResult`] as pretty JSON.
pub fn format_cest_fit_json(res: &CestGlobalFitResult) -> Result<String, String> {
    serde_json::to_string_pretty(res).map_err(|e| format!("failed to serialize CEST JSON: {e}"))
}
