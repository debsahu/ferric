//! Batched CPU exponential relaxation curve fitter ($R_1, R_2, R_{1\rho}$), steady-state
//! $\{^1\text{H}\}\text{–}^{15}\text{N}$ heteronuclear NOE ratio & uncertainty propagation,
//! outlier-trimmed $R_2 / R_1$ global $\tau_m$ & axial $D_\parallel / D_\perp$ estimator, and
//! reduced spectral density mapping $J(0), J(\omega_N), J(0.87\omega_H)$ (`T13.2`,
//! `docs/specs/16-dynamics-and-csp.md` §1.4–§1.7).
//!
//! Provenance: Clean-room implementation strictly from peer-reviewed NMR and numerical optimization
//! literature (zero external GPL/academic tool code consulted; zero external product names in
//! identifiers):
//! - Kay, L. E., Torchia, D. A. & Bax, A. (*Biochemistry* **1989**, 28, 8972–8979) — backbone
//!   $^{15}\text{N}$ dynamics and outlier-trimmed $R_2 / R_1$ correlation-time estimation.
//! - Farrow, N. A., Muhandiram, R., Singer, A. U., Pascal, S. M., Kay, C. M., Gish, G.,
//!   Shoelson, S. E., Pawson, T., Forman-Kay, J. D. & Kay, L. E. (*Biochemistry* **1994**, 33,
//!   5984–6003) — $T_1, T_2$, and steady-state $\{^1\text{H}\}\text{–}^{15}\text{N}$ NOE analysis.
//! - Farrow, N. A., Zhang, O., Szabo, A., Torchia, D. A. & Kay, L. E. (*J. Biomol. NMR* **1995**,
//!   6, 153–162) & Lefèvre, J.-F., Dayie, K. T., Peng, J. W. & Wagner, G. (*Biochemistry* **1996**,
//!   35, 2674–2686) — reduced spectral density mapping $J(0), J(\omega_N), J(0.87\omega_H)$.
//! - Tjandra, N., Feller, S. E., Pastor, R. W. & Bax, A. (*J. Am. Chem. Soc.* **1995**, 117,
//!   12562–12566) & Woessner, D. E. (*J. Chem. Phys.* **1962**, 37, 647–654) — rotational diffusion
//!   anisotropy from $R_2 / R_1$ ratios.
//! - Lipari, G. & Szabo, A. (*J. Am. Chem. Soc.* **1982**, 104, 4546–4559 & 4559–4570).
//! - Clore, G. M. et al. (*J. Am. Chem. Soc.* **1990**, 112, 4989–4991).
//! - Mandel, A. M., Akke, M. & Palmer, A. G. III (*J. Mol. Biol.* **1995**, 246, 144–163).
//! - d'Auvergne, E. J. & Gooley, P. R. (*J. Biomol. NMR* **2003**, 25, 25–39).
//! - Levenberg, K. (*Quart. Appl. Math.* **1944**, 2, 164–168); Marquardt, D. W. (*J. Soc. Indust.
//!   Appl. Math.* **1963**, 11, 431–441); Golub, G. H. & Pereyra, V. (*SIAM J. Numer. Anal.*
//!   **1973**, 10, 413–432 — separable Variable Projection).

use std::f64::consts::PI;

use serde::{Deserialize, Serialize};

use crate::relax_io::RelaxSeriesDataset;
use crate::rng::Rng;

// ============================================================================
// FundamentalCODATA 2018 Spin-Relaxation Constants
// ============================================================================

/// Vacuum permeability factor $\mu_0 / (4\pi)$ ($\text{T}^2\text{ m}^3\text{ J}^{-1}$).
pub const MU0_OVER_4PI: f64 = 1.0e-7;
/// Reduced Planck constant $\hbar = h / (2\pi)$ ($\text{J s}$).
pub const HBAR_J_S: f64 = 1.054_571_817e-34;
/// Proton gyromagnetic ratio $\gamma_H > 0$ ($\text{rad s}^{-1}\text{ T}^{-1}$).
pub const GAMMA_1H_RAD_S_T: f64 = 2.675_221_874_4e8;
/// Nitrogen-15 signed gyromagnetic ratio $\gamma_N < 0$ ($\text{rad s}^{-1}\text{ T}^{-1}$).
pub const GAMMA_15N_RAD_S_T: f64 = -2.712_618_04e7;

/// Configurable physical constants for $^{15}\text{N}$–$^1\text{H}$ dipolar + CSA relaxation.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RelaxPhysicsConfig {
    /// Effective $\text{N–H}$ internuclear bond length in Ångströms (default `1.02 Å`).
    pub r_nh_angstrom: f64,
    /// Axially symmetric $^{15}\text{N}$ chemical shift anisotropy $\Delta\sigma = \sigma_\parallel - \sigma_\perp$ in ppm (default `-172.0 ppm`).
    pub csa_n_ppm: f64,
    /// Proton gyromagnetic ratio $\gamma_H$ ($\text{rad s}^{-1}\text{ T}^{-1}$).
    pub gamma_h: f64,
    /// Nitrogen-15 signed gyromagnetic ratio $\gamma_N$ ($\text{rad s}^{-1}\text{ T}^{-1}$, negative).
    pub gamma_n: f64,
}

impl Default for RelaxPhysicsConfig {
    fn default() -> Self {
        Self {
            r_nh_angstrom: 1.02,
            csa_n_ppm: -172.0,
            gamma_h: GAMMA_1H_RAD_S_T,
            gamma_n: GAMMA_15N_RAD_S_T,
        }
    }
}

impl RelaxPhysicsConfig {
    /// Dipolar coupling constant $d = -\frac{\mu_0 \hbar \gamma_H \gamma_N}{4\pi r_{\text{NH}}^3}$ ($\text{rad s}^{-1}$).
    ///
    /// Note: $d^2$ enters Abragam's equations with $J(\omega) = \frac{2}{5}\frac{\tau_c}{1 + (\omega\tau_c)^2}$
    /// as $R_1^{\text{DD}} = \frac{d^2}{4}[J(\omega_H - \omega_N) + 3 J(\omega_N) + 6 J(\omega_H + \omega_N)]$.
    #[inline]
    pub fn dipolar_d_rad_s(&self) -> f64 {
        let r_m = self.r_nh_angstrom * 1.0e-10;
        -(MU0_OVER_4PI * HBAR_J_S * self.gamma_h * self.gamma_n) / (r_m * r_m * r_m)
    }

    /// Signed Larmor angular frequencies $(\omega_H > 0, \omega_N < 0)$ in $\text{rad s}^{-1}$ at proton spectrometer frequency `field_mhz`.
    #[inline]
    pub fn signed_omegas_rad_s(&self, field_mhz: f64) -> (f64, f64) {
        let omega_h = 2.0 * PI * field_mhz * 1.0e6;
        let omega_n = (self.gamma_n / self.gamma_h) * omega_h;
        (omega_h, omega_n)
    }

    /// $^{15}\text{N}$ CSA coupling constant $c = \frac{\omega_N \Delta\sigma}{\sqrt{3}}$ ($\text{rad s}^{-1}$) at `field_mhz`.
    #[inline]
    pub fn csa_c_rad_s(&self, field_mhz: f64) -> f64 {
        let (_, omega_n) = self.signed_omegas_rad_s(field_mhz);
        (omega_n * (self.csa_n_ppm * 1.0e-6)) / 3.0_f64.sqrt()
    }
}

// ============================================================================
// Exponential Decay Curve Fitting (2-Param & AIC/BIC-Gated 3-Param + MC)
// ============================================================================

/// Statistical gate for selecting 3-parameter ($I_0 e^{-R t} + C$) over 2-parameter ($I_0 e^{-R t}$) decay.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OffsetModelGate {
    /// Strictly fit 2-parameter $I(t) = I_0 e^{-R t}$ ($C = 0$).
    Disabled,
    /// Select 3-parameter offset model when $\mathrm{AIC}_3 < \mathrm{AIC}_2$ ($\Delta\chi^2 > 2$).
    Aic,
    /// Small-sample corrected $\mathrm{AIC}_c$ gate.
    Aicc,
    /// Bayesian Information Criterion $\mathrm{BIC} = \chi^2 + k \ln n$ gate.
    Bic,
    /// Always fit 3-parameter $I_0 e^{-R t} + C$ when $n \ge 3$.
    AlwaysThreeParam,
}

/// Configuration for batched exponential relaxation curve fitting and parametric Monte Carlo bootstrap.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExpFitConfig {
    pub offset_gate: OffsetModelGate,
    /// Number of parametric Monte Carlo bootstrap replicates per residue (`0` = analytical Fisher SE only, default `1000`).
    pub mc_replicates: usize,
    /// Deterministic PRNG seed for parametric Monte Carlo bootstrap.
    pub seed: u64,
    /// Maximum physical relaxation rate bound $R_{\max}$ ($\text{s}^{-1}$, default `200.0`).
    pub max_rate_s_inv: f64,
}

impl Default for ExpFitConfig {
    fn default() -> Self {
        Self {
            offset_gate: OffsetModelGate::Disabled,
            mc_replicates: 1000,
            seed: 20261302,
            max_rate_s_inv: 200.0,
        }
    }
}

/// Input decay curve for a single spin system / residue.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DecayCurveInput {
    pub seq_id: i32,
    pub res_name: String,
    pub times_sec: Vec<f64>,
    pub intensities: Vec<f64>,
    pub sigmas: Vec<f64>,
}

/// Fitted exponential relaxation rate, initial amplitude, analytical Cramér–Rao / Fisher SE, and Monte Carlo CIs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DecayCurveFit {
    pub seq_id: i32,
    pub res_name: String,
    /// Whether the 3-parameter offset model $I_0 e^{-R t} + C$ was selected (`false` = 2-param $I_0 e^{-R t}$).
    pub used_three_param: bool,
    /// Fitted initial amplitude $\hat{I}_0$.
    pub i0: f64,
    /// Fitted relaxation rate $\hat{R}$ ($\text{s}^{-1}$).
    pub rate: f64,
    /// Fitted baseline asymptote $\hat{C}$ (`0.0` when `used_three_param == false`).
    pub offset_c: f64,
    /// Analytical Fisher / Cramér–Rao $1\sigma$ standard error on $\hat{I}_0$.
    pub sigma_i0_fisher: f64,
    /// Analytical Fisher / Cramér–Rao $1\sigma$ standard error on $\hat{R}$.
    pub sigma_rate_fisher: f64,
    /// Parametric Monte Carlo $1\sigma$ standard deviation on $\hat{I}_0$ (`sigma_i0_fisher` if `mc_replicates == 0`).
    pub sigma_i0_mc: f64,
    /// Parametric Monte Carlo $1\sigma$ standard deviation on $\hat{R}$ (`sigma_rate_fisher` if `mc_replicates == 0`).
    pub sigma_rate_mc: f64,
    /// Asymmetric `68.27%` (`[p16, p84]`) Monte Carlo interval for $R$.
    pub rate_ci68: [f64; 2],
    /// Asymmetric `95.0%` (`[p2.5, p97.5]`) Monte Carlo interval for $R$.
    pub rate_ci95: [f64; 2],
    /// Weighted residual sum of squares $\chi^2 = \sum_k ((I_k - \hat{I}_k)/\sigma_k)^2$.
    pub chi2: f64,
    /// Residual degrees of freedom $\nu = n - k$.
    pub dof: usize,
    pub aic: f64,
    pub bic: f64,
}

/// Exact dimensionless 2-parameter weighted VarPro + Levenberg–Marquardt solver shared by CPU and
/// mirrored identically in `crates/nmr-gpu/src/shaders/relax_mc.wgsl` for `< 1e-5` parity.
pub fn fit_two_param_normalized(
    u: &[f64],
    y: &[f64],
    w: &[f64],
    r_tilde_max: f64,
) -> (f64, f64, f64) {
    let n = u.len();
    let varpro_eval = |r_t: f64| -> (f64, f64) {
        let mut s1 = 0.0_f64;
        let mut s2 = 0.0_f64;
        for i in 0..n {
            let e = (-r_t * u[i]).exp();
            let we = w[i] * e;
            s1 += we * y[i];
            s2 += we * e;
        }
        let i0_t = if s2 > 1.0e-30 { s1 / s2 } else { 0.0 };
        let mut chi2 = 0.0_f64;
        for i in 0..n {
            let e = (-r_t * u[i]).exp();
            let d = y[i] - i0_t * e;
            chi2 += w[i] * d * d;
        }
        (i0_t, chi2)
    };

    // 1. Coarse 32-point VarPro grid scan across `[0, min(r_tilde_max, 25.0)]`
    let grid_hi = r_tilde_max.clamp(0.5, 25.0);
    let mut best_r = 0.0_f64;
    let (mut best_i0, mut best_chi2) = varpro_eval(0.0);
    for g in 1..=32 {
        let frac = (g as f64) / 32.0;
        // Quadratic spacing concentrates grid resolution in `[0, 6]` while spanning up to `grid_hi`
        let r_cand = grid_hi * frac * frac;
        let (i0_c, c2_c) = varpro_eval(r_cand);
        if c2_c < best_chi2 {
            best_chi2 = c2_c;
            best_i0 = i0_c;
            best_r = r_cand;
        }
    }

    // Also test weighted log-linear regression on positive points
    {
        let mut sw = 0.0_f64;
        let mut su = 0.0_f64;
        let mut sl = 0.0_f64;
        let mut suu = 0.0_f64;
        let mut sul = 0.0_f64;
        let mut cnt = 0usize;
        for i in 0..n {
            if y[i] > 0.05 {
                let wi = w[i] * y[i] * y[i];
                let ly = y[i].ln();
                sw += wi;
                su += wi * u[i];
                sl += wi * ly;
                suu += wi * u[i] * u[i];
                sul += wi * u[i] * ly;
                cnt += 1;
            }
        }
        let denom = sw * suu - su * su;
        if cnt >= 2 && denom > 1.0e-24 {
            let r_lin = (-(sw * sul - su * sl) / denom).clamp(0.0, r_tilde_max);
            let (i0_l, c2_l) = varpro_eval(r_lin);
            if c2_l < best_chi2 {
                best_chi2 = c2_l;
                best_i0 = i0_l;
                best_r = r_lin;
            }
        }
    }

    // 2. Damped 2x2 Levenberg–Marquardt with exact linear-amplitude VarPro polish
    let mut i0_cur = best_i0;
    let mut r_cur = best_r;
    let mut chi2_cur = best_chi2;
    let mut lambda = 1.0e-4_f64;

    for _iter in 0..24 {
        let mut a00 = 0.0_f64;
        let mut a01 = 0.0_f64;
        let mut a11 = 0.0_f64;
        let mut g0 = 0.0_f64;
        let mut g1 = 0.0_f64;

        for i in 0..n {
            let e = (-r_cur * u[i]).exp();
            let res = y[i] - i0_cur * e;
            let j0 = e;
            let j1 = -i0_cur * u[i] * e;
            let wi = w[i];
            a00 += wi * j0 * j0;
            a01 += wi * j0 * j1;
            a11 += wi * j1 * j1;
            g0 += wi * j0 * res;
            g1 += wi * j1 * res;
        }

        let mut accepted = false;
        for _sub in 0..6 {
            let d00 = a00 + lambda * a00.max(1.0e-6);
            let d11 = a11 + lambda * a11.max(1.0e-6);
            let det = d00 * d11 - a01 * a01;
            if det <= 1.0e-30 {
                lambda *= 10.0;
                continue;
            }
            let step_i0 = (d11 * g0 - a01 * g1) / det;
            let step_r = (d00 * g1 - a01 * g0) / det;

            let r_cand = (r_cur + step_r).clamp(0.0, r_tilde_max);
            let i0_cand_lm = i0_cur + step_i0;
            let mut c2_lm = 0.0_f64;
            for i in 0..n {
                let d = y[i] - i0_cand_lm * (-r_cand * u[i]).exp();
                c2_lm += w[i] * d * d;
            }
            let (i0_vp, c2_vp) = varpro_eval(r_cand);
            let (i0_next, c2_next) = if c2_vp <= c2_lm {
                (i0_vp, c2_vp)
            } else {
                (i0_cand_lm, c2_lm)
            };

            if c2_next <= chi2_cur + 1.0e-15 {
                let dr = (r_cand - r_cur).abs();
                let di = (i0_next - i0_cur).abs();
                i0_cur = i0_next;
                r_cur = r_cand;
                chi2_cur = c2_next;
                lambda = (lambda * 0.25).max(1.0e-9);
                accepted = true;
                if dr < 1.0e-10 && di < 1.0e-10 {
                    return (i0_cur, r_cur, chi2_cur);
                }
                break;
            }
            lambda = (lambda * 8.0).min(1.0e7);
        }
        if !accepted {
            break;
        }
    }

    (i0_cur, r_cur, chi2_cur)
}

fn solve_3x3(a: [[f64; 3]; 3], b: [f64; 3]) -> Option<[f64; 3]> {
    let mut m = [
        [a[0][0], a[0][1], a[0][2], b[0]],
        [a[1][0], a[1][1], a[1][2], b[1]],
        [a[2][0], a[2][1], a[2][2], b[2]],
    ];
    for col in 0..3 {
        let mut piv = col;
        let mut max_v = m[col][col].abs();
        for (r, row) in m.iter().enumerate().skip(col + 1) {
            if row[col].abs() > max_v {
                max_v = row[col].abs();
                piv = r;
            }
        }
        if max_v <= 1.0e-24 {
            return None;
        }
        if piv != col {
            m.swap(col, piv);
        }
        let inv_p = 1.0 / m[col][col];
        let piv_row = m[col];
        for row_r in m.iter_mut().skip(col + 1) {
            let f = row_r[col] * inv_p;
            for (dst, &src) in row_r[col..4].iter_mut().zip(piv_row[col..4].iter()) {
                *dst -= f * src;
            }
        }
    }
    let mut x = [0.0_f64; 3];
    for i in (0..3).rev() {
        let mut s = m[i][3];
        for j in (i + 1)..3 {
            s -= m[i][j] * x[j];
        }
        x[i] = s / m[i][i];
    }
    Some(x)
}

/// Fit 3-parameter normalized decay $y(u) = \tilde{I}_0 e^{-\tilde{R} u} + \tilde{C}$.
fn fit_three_param_normalized(
    u: &[f64],
    y: &[f64],
    w: &[f64],
    r_tilde_max: f64,
) -> (f64, f64, f64, f64) {
    let n = u.len();
    let varpro_3 = |r_t: f64| -> (f64, f64, f64) {
        let mut s_ee = 0.0_f64;
        let mut s_e1 = 0.0_f64;
        let mut s_11 = 0.0_f64;
        let mut s_ey = 0.0_f64;
        let mut s_1y = 0.0_f64;
        for i in 0..n {
            let e = (-r_t * u[i]).exp();
            let wi = w[i];
            s_ee += wi * e * e;
            s_e1 += wi * e;
            s_11 += wi;
            s_ey += wi * e * y[i];
            s_1y += wi * y[i];
        }
        let det = s_ee * s_11 - s_e1 * s_e1;
        let (i0_t, c_t) = if det > 1.0e-24 {
            (
                (s_11 * s_ey - s_e1 * s_1y) / det,
                (s_ee * s_1y - s_e1 * s_ey) / det,
            )
        } else {
            (if s_ee > 1.0e-24 { s_ey / s_ee } else { 0.0 }, 0.0)
        };
        let mut chi2 = 0.0_f64;
        for i in 0..n {
            let d = y[i] - (i0_t * (-r_t * u[i]).exp() + c_t);
            chi2 += w[i] * d * d;
        }
        (i0_t, c_t, chi2)
    };

    let grid_hi = r_tilde_max.clamp(0.5, 25.0);
    let mut best_r = 0.5_f64;
    let (mut best_i0, mut best_c, mut best_chi2) = varpro_3(best_r);
    for g in 1..=40 {
        let frac = (g as f64) / 40.0;
        let r_cand = 0.02 + (grid_hi - 0.02) * frac * frac;
        let (i0_c, c_c, c2_c) = varpro_3(r_cand);
        if c2_c < best_chi2 {
            best_chi2 = c2_c;
            best_i0 = i0_c;
            best_c = c_c;
            best_r = r_cand;
        }
    }

    let mut i0_cur = best_i0;
    let mut r_cur = best_r;
    let mut c_cur = best_c;
    let mut chi2_cur = best_chi2;
    let mut lambda = 1.0e-4_f64;

    for _iter in 0..30 {
        let mut a = [[0.0_f64; 3]; 3];
        let mut g = [0.0_f64; 3];
        for i in 0..n {
            let e = (-r_cur * u[i]).exp();
            let res = y[i] - (i0_cur * e + c_cur);
            let j = [e, -i0_cur * u[i] * e, 1.0];
            let wi = w[i];
            for r in 0..3 {
                g[r] += wi * j[r] * res;
                for c in 0..3 {
                    a[r][c] += wi * j[r] * j[c];
                }
            }
        }

        let mut accepted = false;
        for _sub in 0..6 {
            let mut d = a;
            for k in 0..3 {
                d[k][k] += lambda * a[k][k].max(1.0e-6);
            }
            let Some(step) = solve_3x3(d, g) else {
                lambda *= 10.0;
                continue;
            };
            let r_cand = (r_cur + step[1]).clamp(0.0, r_tilde_max);
            let i0_lm = i0_cur + step[0];
            let c_lm = c_cur + step[2];
            let mut c2_lm = 0.0_f64;
            for i in 0..n {
                let diff = y[i] - (i0_lm * (-r_cand * u[i]).exp() + c_lm);
                c2_lm += w[i] * diff * diff;
            }
            let (i0_vp, c_vp, c2_vp) = varpro_3(r_cand);
            let (i0_n, c_n, c2_n) = if c2_vp <= c2_lm {
                (i0_vp, c_vp, c2_vp)
            } else {
                (i0_lm, c_lm, c2_lm)
            };
            if c2_n <= chi2_cur + 1.0e-15 {
                let dr = (r_cand - r_cur).abs();
                i0_cur = i0_n;
                r_cur = r_cand;
                c_cur = c_n;
                chi2_cur = c2_n;
                lambda = (lambda * 0.3).max(1.0e-9);
                accepted = true;
                if dr < 1.0e-10 {
                    return (i0_cur, r_cur, c_cur, chi2_cur);
                }
                break;
            }
            lambda = (lambda * 8.0).min(1.0e7);
        }
        if !accepted {
            break;
        }
    }

    (i0_cur, r_cur, c_cur, chi2_cur)
}

/// Compute exact analytical Fisher Information / Cramér–Rao $1\sigma$ standard errors
/// $(\sigma_{\text{Fisher}}(I_0), \sigma_{\text{Fisher}}(R))$ for $I(t) = I_0 e^{-R t}$.
pub fn fisher_crlb_two_param(times_sec: &[f64], sigmas: &[f64], i0: f64, rate: f64) -> (f64, f64) {
    let mut f00 = 0.0_f64;
    let mut f01 = 0.0_f64;
    let mut f11 = 0.0_f64;
    for (&t, &sig) in times_sec.iter().zip(sigmas.iter()) {
        let w = 1.0 / (sig * sig).max(1.0e-30);
        let e2 = (-2.0 * rate * t).exp();
        f00 += w * e2;
        f01 += -i0 * w * t * e2;
        f11 += i0 * i0 * w * t * t * e2;
    }
    let det = f00 * f11 - f01 * f01;
    if det <= 1.0e-30 {
        return (0.0, 0.0);
    }
    ((f11 / det).max(0.0).sqrt(), (f00 / det).max(0.0).sqrt())
}

fn fisher_crlb_three_param(times_sec: &[f64], sigmas: &[f64], i0: f64, rate: f64) -> (f64, f64) {
    let mut f = [[0.0_f64; 3]; 3];
    for (&t, &sig) in times_sec.iter().zip(sigmas.iter()) {
        let w = 1.0 / (sig * sig).max(1.0e-30);
        let e = (-rate * t).exp();
        let j = [e, -i0 * t * e, 1.0];
        for r in 0..3 {
            for c in 0..3 {
                f[r][c] += w * j[r] * j[c];
            }
        }
    }
    let e0 = solve_3x3(f, [1.0, 0.0, 0.0]).unwrap_or([0.0; 3]);
    let e1 = solve_3x3(f, [0.0, 1.0, 0.0]).unwrap_or([0.0; 3]);
    (e0[0].max(0.0).sqrt(), e1[1].max(0.0).sqrt())
}

fn info_criterion(chi2: f64, k: usize, n: usize, gate: OffsetModelGate) -> f64 {
    let kf = k as f64;
    let nf = (n.max(1)) as f64;
    match gate {
        OffsetModelGate::Aicc => {
            let denom = ((n as isize) - (k as isize) - 1).max(1) as f64;
            chi2 + 2.0 * kf + (2.0 * kf * (kf + 1.0)) / denom
        }
        OffsetModelGate::Bic => chi2 + kf * nf.ln(),
        _ => chi2 + 2.0 * kf,
    }
}

pub(crate) fn quantile_sorted(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let idx = (p.clamp(0.0, 1.0)) * ((sorted.len() - 1) as f64);
    let lo = idx.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    let frac = idx - (lo as f64);
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// Fit a single decay curve on CPU with analytical Fisher bounds and optional parametric Monte Carlo bootstrap.
pub fn fit_decay_curve_cpu(
    curve: &DecayCurveInput,
    cfg: &ExpFitConfig,
) -> Result<DecayCurveFit, String> {
    let n = curve.times_sec.len();
    if n < 2 || curve.intensities.len() != n || curve.sigmas.len() != n {
        return Err(format!(
            "residue {} decay curve requires >= 2 matching (t, I, sigma) points",
            curve.seq_id
        ));
    }

    let t_scale = curve
        .times_sec
        .iter()
        .copied()
        .fold(0.0_f64, f64::max)
        .max(1.0e-6);
    let s_scale = curve
        .intensities
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f64, f64::max)
        .max(1.0e-12);
    let r_tilde_max = (cfg.max_rate_s_inv * t_scale).max(1.0);

    let u: Vec<f64> = curve.times_sec.iter().map(|&t| t / t_scale).collect();
    let y: Vec<f64> = curve.intensities.iter().map(|&v| v / s_scale).collect();
    let s_norm: Vec<f64> = curve
        .sigmas
        .iter()
        .map(|&s| (s / s_scale).max(1.0e-9))
        .collect();
    let w: Vec<f64> = s_norm.iter().map(|&sn| 1.0 / (sn * sn)).collect();

    let (i0_2t, r_2t, chi2_2) = fit_two_param_normalized(&u, &y, &w, r_tilde_max);

    let (use_3p, i0_t, r_t, c_t, chi2) = match cfg.offset_gate {
        OffsetModelGate::Disabled => (false, i0_2t, r_2t, 0.0, chi2_2),
        OffsetModelGate::AlwaysThreeParam if n >= 3 => {
            let (i3, r3, c3, chi3) = fit_three_param_normalized(&u, &y, &w, r_tilde_max);
            (true, i3, r3, c3, chi3)
        }
        gate if n >= 4 => {
            let (i3, r3, c3, chi3) = fit_three_param_normalized(&u, &y, &w, r_tilde_max);
            let ic2 = info_criterion(chi2_2, 2, n, gate);
            let ic3 = info_criterion(chi3, 3, n, gate);
            if ic3 < ic2 {
                (true, i3, r3, c3, chi3)
            } else {
                (false, i0_2t, r_2t, 0.0, chi2_2)
            }
        }
        _ => (false, i0_2t, r_2t, 0.0, chi2_2),
    };

    let i0 = i0_t * s_scale;
    let rate = r_t / t_scale;
    let offset_c = c_t * s_scale;
    let k_params = if use_3p { 3 } else { 2 };
    let dof = n.saturating_sub(k_params);
    let aic = chi2 + 2.0 * (k_params as f64);
    let bic = chi2 + (k_params as f64) * (n as f64).ln();

    let (sigma_i0_fisher, sigma_rate_fisher) = if use_3p {
        fisher_crlb_three_param(&curve.times_sec, &curve.sigmas, i0, rate)
    } else {
        fisher_crlb_two_param(&curve.times_sec, &curve.sigmas, i0, rate)
    };

    if cfg.mc_replicates == 0 {
        return Ok(DecayCurveFit {
            seq_id: curve.seq_id,
            res_name: curve.res_name.clone(),
            used_three_param: use_3p,
            i0,
            rate,
            offset_c,
            sigma_i0_fisher,
            sigma_rate_fisher,
            sigma_i0_mc: sigma_i0_fisher,
            sigma_rate_mc: sigma_rate_fisher,
            rate_ci68: [rate - sigma_rate_fisher, rate + sigma_rate_fisher],
            rate_ci95: [
                rate - 1.95996 * sigma_rate_fisher,
                rate + 1.95996 * sigma_rate_fisher,
            ],
            chi2,
            dof,
            aic,
            bic,
        });
    }

    let mut rng = Rng::new(cfg.seed ^ ((curve.seq_id as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)));
    let y_hat: Vec<f64> = u.iter().map(|&uk| i0_t * (-r_t * uk).exp() + c_t).collect();
    let mut y_syn = vec![0.0_f64; n];
    let mut mc_i0 = Vec::with_capacity(cfg.mc_replicates);
    let mut mc_r = Vec::with_capacity(cfg.mc_replicates);

    for _rep in 0..cfg.mc_replicates {
        for i in 0..n {
            y_syn[i] = y_hat[i] + rng.normal(0.0, s_norm[i]);
        }
        let (mi0, mr) = if use_3p {
            let (a, b, _, _) = fit_three_param_normalized(&u, &y_syn, &w, r_tilde_max);
            (a * s_scale, b / t_scale)
        } else {
            let (a, b, _) = fit_two_param_normalized(&u, &y_syn, &w, r_tilde_max);
            (a * s_scale, b / t_scale)
        };
        mc_i0.push(mi0);
        mc_r.push(mr);
    }

    let m_f = cfg.mc_replicates as f64;
    let mean_i0: f64 = mc_i0.iter().sum::<f64>() / m_f;
    let mean_r: f64 = mc_r.iter().sum::<f64>() / m_f;
    let denom = (cfg.mc_replicates.saturating_sub(1)).max(1) as f64;
    let var_i0: f64 = mc_i0
        .iter()
        .map(|&v| (v - mean_i0) * (v - mean_i0))
        .sum::<f64>()
        / denom;
    let var_r: f64 = mc_r
        .iter()
        .map(|&v| (v - mean_r) * (v - mean_r))
        .sum::<f64>()
        / denom;

    mc_r.sort_by(|a, b| a.total_cmp(b));
    let rate_ci68 = [
        quantile_sorted(&mc_r, 0.158_655),
        quantile_sorted(&mc_r, 0.841_345),
    ];
    let rate_ci95 = [quantile_sorted(&mc_r, 0.025), quantile_sorted(&mc_r, 0.975)];

    Ok(DecayCurveFit {
        seq_id: curve.seq_id,
        res_name: curve.res_name.clone(),
        used_three_param: use_3p,
        i0,
        rate,
        offset_c,
        sigma_i0_fisher,
        sigma_rate_fisher,
        sigma_i0_mc: var_i0.sqrt(),
        sigma_rate_mc: var_r.sqrt(),
        rate_ci68,
        rate_ci95,
        chi2,
        dof,
        aic,
        bic,
    })
}

/// Multi-threaded CPU batch exponential fitter across $N$ residues (`std::thread::scope` on native,
/// sequential on `wasm32`).
pub fn fit_decay_batch_cpu(
    curves: &[DecayCurveInput],
    cfg: &ExpFitConfig,
) -> Result<Vec<DecayCurveFit>, String> {
    if curves.is_empty() {
        return Ok(Vec::new());
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let n_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .max(1);
        let chunk_sz = curves.len().div_ceil(n_threads);
        let mut results: Vec<Option<Result<DecayCurveFit, String>>> = vec![None; curves.len()];
        std::thread::scope(|s| {
            for (chunk_in, chunk_out) in curves.chunks(chunk_sz).zip(results.chunks_mut(chunk_sz)) {
                s.spawn(move || {
                    for (cin, cout) in chunk_in.iter().zip(chunk_out.iter_mut()) {
                        *cout = Some(fit_decay_curve_cpu(cin, cfg));
                    }
                });
            }
        });
        results
            .into_iter()
            .map(|opt| opt.unwrap())
            .collect::<Result<Vec<_>, _>>()
    }
    #[cfg(target_arch = "wasm32")]
    {
        curves.iter().map(|c| fit_decay_curve_cpu(c, cfg)).collect()
    }
}

/// Convert an integrated `RelaxSeriesDataset` (`T13.1`) directly into `Vec<DecayCurveInput>`.
pub fn decay_inputs_from_series(ds: &RelaxSeriesDataset) -> Vec<DecayCurveInput> {
    let times = ds.schedule.primary_values();
    ds.traces
        .iter()
        .map(|tr| DecayCurveInput {
            seq_id: tr.footprint.seq_id,
            res_name: tr.footprint.res_name.clone(),
            times_sec: times.clone(),
            intensities: tr.intensities.clone(),
            sigmas: tr.sigmas.clone(),
        })
        .collect()
}

// ============================================================================
// Steady-State {1H}-15N Heteronuclear NOE Ratio & Error Propagation
// ============================================================================

/// Steady-state $\{^1\text{H}\}\text{–}^{15}\text{N}$ hetNOE input for one residue.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HetNoeInput {
    pub seq_id: i32,
    pub res_name: String,
    pub i_sat: f64,
    pub sigma_sat: f64,
    pub i_unsat: f64,
    pub sigma_unsat: f64,
}

/// Steady-state $\{^1\text{H}\}\text{–}^{15}\text{N}$ hetNOE result with analytical and asymmetric MC uncertainties.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HetNoeResult {
    pub seq_id: i32,
    pub res_name: String,
    /// Steady-state ratio $\mathrm{NOE} = I_{\text{sat}} / I_{\text{unsat}}$.
    pub noe: f64,
    /// Exact first-order Gaussian propagated uncertainty $\sigma_{\mathrm{NOE}} = \sqrt{\sigma_{\text{sat}}^2 + \mathrm{NOE}^2 \sigma_{\text{unsat}}^2} / |I_{\text{unsat}}|$.
    pub sigma_noe: f64,
    /// Parametric Monte Carlo standard deviation (`sigma_noe` when `mc_replicates == 0`).
    pub sigma_noe_mc: f64,
    /// Asymmetric `68.27%` (`[p16, p84]`) Monte Carlo interval.
    pub noe_ci68: [f64; 2],
    /// Asymmetric `95.0%` (`[p2.5, p97.5]`) Monte Carlo interval.
    pub noe_ci95: [f64; 2],
}

/// Evaluate steady-state $\{^1\text{H}\}\text{–}^{15}\text{N}$ hetNOE ratio $I_{\text{sat}} / I_{\text{unsat}}$
/// with exact first-order analytical error propagation and optional parametric Monte Carlo CIs.
pub fn compute_hetnoe(
    inp: &HetNoeInput,
    mc_replicates: usize,
    seed: u64,
) -> Result<HetNoeResult, String> {
    if !inp.i_unsat.is_finite() || inp.i_unsat.abs() <= 1.0e-20 {
        return Err(format!(
            "hetNOE unsaturated reference intensity on residue {} must be non-zero",
            inp.seq_id
        ));
    }
    if !inp.sigma_sat.is_finite()
        || inp.sigma_sat < 0.0
        || !inp.sigma_unsat.is_finite()
        || inp.sigma_unsat < 0.0
    {
        return Err(format!(
            "hetNOE uncertainties on residue {} must be finite and >= 0",
            inp.seq_id
        ));
    }

    let noe = inp.i_sat / inp.i_unsat;
    let sigma_noe = (inp.sigma_sat.hypot(noe * inp.sigma_unsat)) / inp.i_unsat.abs();

    if mc_replicates == 0 {
        return Ok(HetNoeResult {
            seq_id: inp.seq_id,
            res_name: inp.res_name.clone(),
            noe,
            sigma_noe,
            sigma_noe_mc: sigma_noe,
            noe_ci68: [noe - sigma_noe, noe + sigma_noe],
            noe_ci95: [noe - 1.95996 * sigma_noe, noe + 1.95996 * sigma_noe],
        });
    }

    let mut rng = Rng::new(seed ^ ((inp.seq_id as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9)));
    let mut samples = Vec::with_capacity(mc_replicates);
    for _ in 0..mc_replicates {
        let s_mc = inp.i_sat + rng.normal(0.0, inp.sigma_sat);
        let mut u_mc = inp.i_unsat + rng.normal(0.0, inp.sigma_unsat);
        if u_mc.abs() < 1.0e-6 * inp.i_unsat.abs() {
            u_mc = inp.i_unsat.signum() * 1.0e-6 * inp.i_unsat.abs();
        }
        samples.push(s_mc / u_mc);
    }
    let mean: f64 = samples.iter().sum::<f64>() / (mc_replicates as f64);
    let denom = (mc_replicates.saturating_sub(1)).max(1) as f64;
    let var: f64 = samples
        .iter()
        .map(|&v| (v - mean) * (v - mean))
        .sum::<f64>()
        / denom;
    samples.sort_by(|a, b| a.total_cmp(b));

    Ok(HetNoeResult {
        seq_id: inp.seq_id,
        res_name: inp.res_name.clone(),
        noe,
        sigma_noe,
        sigma_noe_mc: var.sqrt(),
        noe_ci68: [
            quantile_sorted(&samples, 0.158_655),
            quantile_sorted(&samples, 0.841_345),
        ],
        noe_ci95: [
            quantile_sorted(&samples, 0.025),
            quantile_sorted(&samples, 0.975),
        ],
    })
}

/// Batch compute steady-state $\{^1\text{H}\}\text{–}^{15}\text{N}$ hetNOEs across a protein.
pub fn compute_hetnoe_batch(
    inputs: &[HetNoeInput],
    mc_replicates: usize,
    seed: u64,
) -> Result<Vec<HetNoeResult>, String> {
    inputs
        .iter()
        .map(|inp| compute_hetnoe(inp, mc_replicates, seed))
        .collect()
}

// ============================================================================
// Outlier-Trimmed R2/R1 Global tau_m & Axial Diffusion Estimator (Kay et al. 1989)
// ============================================================================

/// Per-residue experimental $(R_1, R_2, \mathrm{hetNOE})$ triplet at a single static magnetic field.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResidueRelaxRates {
    pub seq_id: i32,
    pub res_name: String,
    pub field_mhz: f64,
    pub r1: f64,
    pub sigma_r1: f64,
    pub r2: f64,
    pub sigma_r2: f64,
    pub hetnoe: Option<f64>,
    pub sigma_hetnoe: Option<f64>,
}

/// Configuration for Kay, Torchia & Bax (1989) outlier-trimmed $R_2 / R_1$ $\tau_m$ estimation.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrimmedTauMConfig {
    /// Exclude flexible residues with $\mathrm{hetNOE} < \text{min\_hetnoe}$ (default `0.65`).
    pub min_hetnoe: f64,
    /// Iterative outlier rejection cutoff in standard-deviation units around mean $R_2 / R_1$ (default `1.5`).
    pub trim_sd: f64,
    /// Number of iterative trimming passes (default `3`).
    pub max_trim_passes: usize,
    pub physics: RelaxPhysicsConfig,
}

impl Default for TrimmedTauMConfig {
    fn default() -> Self {
        Self {
            min_hetnoe: 0.65,
            trim_sd: 1.5,
            max_trim_passes: 3,
            physics: RelaxPhysicsConfig::default(),
        }
    }
}

/// Result of Kay, Torchia & Bax (1989) outlier-trimmed $R_2 / R_1$ isotropic $\tau_m$ estimation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrimmedTauMEstimate {
    /// Estimated global isotropic rotational correlation time $\tau_m$ in seconds.
    pub tau_m_sec: f64,
    /// Propagated $1\sigma$ uncertainty on $\tau_m$ in seconds.
    pub sigma_tau_m_sec: f64,
    /// Inverse-variance weighted mean $R_2 / R_1$ over retained core residues.
    pub mean_r2_over_r1: f64,
    /// Sequence IDs of core rigid residues retained after hetNOE and iterative SD trimming.
    pub retained_seq_ids: Vec<i32>,
    /// Sequence IDs of excluded outlier residues (flexible loops/termini or $R_{\text{ex}}$ sites).
    pub excluded_seq_ids: Vec<i32>,
}

/// Initial axial rotational diffusion estimate $(D_\parallel, D_\perp, \theta, \phi)$ from $R_2 / R_1$
/// orientation dependence (Kay, Torchia & Bax 1989; Tjandra et al. 1995; Woessner 1962).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AxialDiffusionEstimate {
    /// Effective isotropic correlation time $\tau_m = (6 D_{\text{iso}})^{-1}$ in seconds.
    pub tau_m_sec: f64,
    /// Parallel rotational diffusion rate $D_\parallel$ ($\text{s}^{-1}$).
    pub d_par_s_inv: f64,
    /// Perpendicular rotational diffusion rate $D_\perp$ ($\text{s}^{-1}$).
    pub d_perp_s_inv: f64,
    /// Anisotropy ratio $D_\parallel / D_\perp$ (`> 1` prolate, `< 1` oblate).
    pub anisotropy_ratio: f64,
    /// Polar angle $\theta \in [0, \pi]$ (radians) of the symmetry axis in the molecular PDB frame.
    pub theta_rad: f64,
    /// Azimuthal angle $\phi \in (-\pi, \pi]$ (radians) of the symmetry axis in the molecular PDB frame.
    pub phi_rad: f64,
    pub retained_seq_ids: Vec<i32>,
    pub chi2: f64,
}

/// Exact rigid-tumbling ($S^2 = 1, \tau_e = 0, R_{\text{ex}} = 0$) theoretical $R_1$ and $R_2$ ($\text{s}^{-1}$)
/// for a multi-exponential rotational correlation function $\sum_a c_a e^{-t / \tau_a}$.
pub fn rigid_r1_r2_from_rot_components(
    components: &[(f64, f64)], // (weight c_a, tau_a_sec)
    field_mhz: f64,
    phys: &RelaxPhysicsConfig,
) -> (f64, f64) {
    let d = phys.dipolar_d_rad_s();
    let d2 = d * d;
    let c = phys.csa_c_rad_s(field_mhz);
    let c2 = c * c;
    let (wh, wn) = phys.signed_omegas_rad_s(field_mhz);

    let eval_j = |omega: f64| -> f64 {
        let mut s = 0.0_f64;
        for &(ca, tau) in components {
            let wt = omega * tau;
            s += ca * tau / (1.0 + wt * wt);
        }
        0.4 * s
    };

    let j0 = eval_j(0.0);
    let jn = eval_j(wn);
    let jh = eval_j(wh);
    let j_diff = eval_j(wh - wn);
    let j_sum = eval_j(wh + wn);

    let r1 = 0.25 * d2 * (j_diff + 3.0 * jn + 6.0 * j_sum) + c2 * jn;
    let r2 = 0.125 * d2 * (4.0 * j0 + j_diff + 3.0 * jn + 6.0 * jh + 6.0 * j_sum)
        + (c2 / 6.0) * (4.0 * j0 + 3.0 * jn);
    (r1, r2)
}

/// Invert a single observed $R_2 / R_1$ ratio into an exact rigid-limit $\tau_c$ (seconds) via monotonic bisection.
pub fn invert_r2_over_r1_to_tau_c(
    r2_over_r1: f64,
    field_mhz: f64,
    phys: &RelaxPhysicsConfig,
) -> f64 {
    let mut lo = 0.05e-9_f64;
    let mut hi = 250.0e-9_f64;
    let ratio_at = |tau: f64| -> f64 {
        let (r1, r2) = rigid_r1_r2_from_rot_components(&[(1.0, tau)], field_mhz, phys);
        r2 / r1.max(1.0e-30)
    };
    if r2_over_r1 <= ratio_at(lo) {
        return lo;
    }
    if r2_over_r1 >= ratio_at(hi) {
        return hi;
    }
    for _ in 0..64 {
        let mid = 0.5 * (lo + hi);
        if ratio_at(mid) < r2_over_r1 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Estimate initial global $\tau_m$ from outlier-trimmed $R_2 / R_1$ ratios (Kay, Torchia & Bax 1989).
pub fn estimate_trimmed_tau_m(
    rates: &[ResidueRelaxRates],
    cfg: &TrimmedTauMConfig,
) -> Result<TrimmedTauMEstimate, String> {
    if rates.is_empty() {
        return Err("estimate_trimmed_tau_m requires >= 1 residue".to_string());
    }
    let field_mhz = rates[0].field_mhz;

    // Step 1: Pre-filter by positive rates and steady-state hetNOE >= min_hetnoe
    let mut candidates: Vec<(usize, f64, f64)> = Vec::new(); // (idx, q = R2/R1, sigma_q)
    for (idx, r) in rates.iter().enumerate() {
        if r.r1 <= 0.0 || r.r2 <= 0.0 || !r.r1.is_finite() || !r.r2.is_finite() {
            continue;
        }
        if let Some(noe) = r.hetnoe {
            if noe < cfg.min_hetnoe {
                continue;
            }
        }
        let q = r.r2 / r.r1;
        let rel1 = (r.sigma_r1 / r.r1).max(1.0e-4);
        let rel2 = (r.sigma_r2 / r.r2).max(1.0e-4);
        let sq = q * rel1.hypot(rel2);
        candidates.push((idx, q, sq));
    }
    if candidates.is_empty() {
        // Fallback: relax hetNOE filter if all residues had hetNOE < min_hetnoe
        for (idx, r) in rates.iter().enumerate() {
            if r.r1 > 0.0 && r.r2 > 0.0 && r.r1.is_finite() && r.r2.is_finite() {
                let q = r.r2 / r.r1;
                let sq = q * (r.sigma_r1 / r.r1).hypot(r.sigma_r2 / r.r2).max(1.0e-4);
                candidates.push((idx, q, sq));
            }
        }
    }
    if candidates.is_empty() {
        return Err("no valid positive R1/R2 rate pairs for tau_m estimation".to_string());
    }

    // Step 2: Iterative mean +- trim_sd * SD outlier rejection
    let mut active = vec![true; candidates.len()];
    for _pass in 0..cfg.max_trim_passes {
        let act_vals: Vec<f64> = candidates
            .iter()
            .zip(active.iter())
            .filter_map(|(&(_, q, _), &on)| on.then_some(q))
            .collect();
        if act_vals.len() <= 4 {
            break;
        }
        let mean = act_vals.iter().sum::<f64>() / (act_vals.len() as f64);
        let sd = (act_vals
            .iter()
            .map(|&q| (q - mean) * (q - mean))
            .sum::<f64>()
            / ((act_vals.len() - 1) as f64))
            .sqrt();
        if sd <= 1.0e-12 {
            break;
        }
        let next_active: Vec<bool> = candidates
            .iter()
            .zip(active.iter())
            .map(|(&(_, q, _), &on)| on && (q - mean).abs() <= cfg.trim_sd * sd)
            .collect();
        if next_active.iter().filter(|&&b| b).count() >= 3 {
            active = next_active;
        } else {
            break;
        }
    }

    let mut w_sum = 0.0_f64;
    let mut wq_sum = 0.0_f64;
    let mut retained_mask = vec![false; rates.len()];
    for (&(orig_idx, q, sq), &on) in candidates.iter().zip(active.iter()) {
        if on {
            retained_mask[orig_idx] = true;
            let w = 1.0 / (sq * sq).max(1.0e-12);
            w_sum += w;
            wq_sum += w * q;
        }
    }
    let mean_q = wq_sum / w_sum.max(1.0e-30);
    let sigma_mean_q = (1.0 / w_sum.max(1.0e-30)).sqrt();

    let tau_m = invert_r2_over_r1_to_tau_c(mean_q, field_mhz, &cfg.physics);
    let tau_hi = invert_r2_over_r1_to_tau_c(mean_q + sigma_mean_q, field_mhz, &cfg.physics);
    let tau_lo =
        invert_r2_over_r1_to_tau_c((mean_q - sigma_mean_q).max(1.001), field_mhz, &cfg.physics);
    let sigma_tau_m = 0.5 * (tau_hi - tau_lo).abs();

    let mut retained_seq_ids = Vec::new();
    let mut excluded_seq_ids = Vec::new();
    for (idx, r) in rates.iter().enumerate() {
        if retained_mask[idx] {
            retained_seq_ids.push(r.seq_id);
        } else {
            excluded_seq_ids.push(r.seq_id);
        }
    }

    Ok(TrimmedTauMEstimate {
        tau_m_sec: tau_m,
        sigma_tau_m_sec: sigma_tau_m,
        mean_r2_over_r1: mean_q,
        retained_seq_ids,
        excluded_seq_ids,
    })
}

/// Woessner (1962) 3-component axial rotational correlation weights and correlation times
/// `[(c_1, tau_1), (c_2, tau_2), (c_3, tau_3)]` for unit bond vector `u` and symmetry axis `axis`.
#[inline]
pub fn woessner_axial_components(
    u: [f64; 3],
    axis: [f64; 3],
    d_perp: f64,
    d_par: f64,
) -> [(f64, f64); 3] {
    let mu = (u[0] * axis[0] + u[1] * axis[1] + u[2] * axis[2]).clamp(-1.0, 1.0);
    let mu2 = mu * mu;
    let one_m_mu2 = 1.0 - mu2;
    let c1 = 0.25 * (3.0 * mu2 - 1.0) * (3.0 * mu2 - 1.0);
    let c2 = 3.0 * mu2 * one_m_mu2;
    let c3 = 0.75 * one_m_mu2 * one_m_mu2;
    let tau1 = 1.0 / (6.0 * d_perp).max(1.0e-3);
    let tau2 = 1.0 / (5.0 * d_perp + d_par).max(1.0e-3);
    let tau3 = 1.0 / (2.0 * d_perp + 4.0 * d_par).max(1.0e-3);
    [(c1, tau1), (c2, tau2), (c3, tau3)]
}

/// Estimate initial axial rotational diffusion tensor $(D_\parallel, D_\perp, \theta, \phi)$ from
/// outlier-trimmed $R_2 / R_1$ ratios and $\text{N–H}$ unit bond vectors (Kay, Torchia & Bax 1989;
/// Tjandra et al. 1995; Woessner 1962).
pub fn estimate_initial_axial_diffusion_from_r2_r1(
    rates: &[ResidueRelaxRates],
    nh_vectors: &[(i32, [f64; 3])],
    cfg: &TrimmedTauMConfig,
) -> Result<AxialDiffusionEstimate, String> {
    let iso = estimate_trimmed_tau_m(rates, cfg)?;
    let vec_map: std::collections::BTreeMap<i32, [f64; 3]> = nh_vectors.iter().copied().collect();
    let retained_set: std::collections::BTreeSet<i32> =
        iso.retained_seq_ids.iter().copied().collect();

    let mut obs: Vec<([f64; 3], f64, f64, f64, i32)> = Vec::new(); // (u, q, sq, field_mhz, seq_id)
    for r in rates {
        if !retained_set.contains(&r.seq_id) {
            continue;
        }
        if let Some(&u_raw) = vec_map.get(&r.seq_id) {
            let nrm = (u_raw[0] * u_raw[0] + u_raw[1] * u_raw[1] + u_raw[2] * u_raw[2]).sqrt();
            if nrm > 1.0e-6 {
                let u = [u_raw[0] / nrm, u_raw[1] / nrm, u_raw[2] / nrm];
                let q = r.r2 / r.r1;
                let sq = (q * (r.sigma_r1 / r.r1).hypot(r.sigma_r2 / r.r2)).max(1.0e-3);
                obs.push((u, q, sq, r.field_mhz, r.seq_id));
            }
        }
    }
    if obs.len() < 5 {
        return Err(format!(
            "axial R2/R1 diffusion estimation requires >= 5 core residues with N-H vectors, got {}",
            obs.len()
        ));
    }

    let d_iso_0 = 1.0 / (6.0 * iso.tau_m_sec);
    let eval_chi2 = |p: &[f64; 4]| -> f64 {
        let d_perp = p[0];
        let d_par = p[1];
        if d_perp <= 1.0e4 || d_par <= 1.0e4 {
            return 1.0e30;
        }
        let (st, ct) = p[2].sin_cos();
        let (sp, cp) = p[3].sin_cos();
        let axis = [st * cp, st * sp, ct];
        let mut c2 = 0.0_f64;
        for &(u, q_obs, sq, mhz, _) in &obs {
            let comps = woessner_axial_components(u, axis, d_perp, d_par);
            let (r1, r2) = rigid_r1_r2_from_rot_components(&comps, mhz, &cfg.physics);
            let q_calc = r2 / r1.max(1.0e-30);
            let z = (q_obs - q_calc) / sq;
            c2 += z * z;
        }
        c2
    };

    // Multi-start spherical axis scan (hemisphere) x prolate/oblate initial ratios
    let mut best_p = [d_iso_0, d_iso_0, 0.5, 0.5];
    let mut best_c2 = eval_chi2(&best_p);
    for &ratio in &[0.65_f64, 0.85, 1.15, 1.45, 1.80] {
        let dp = 3.0 * d_iso_0 / (2.0 + ratio);
        let dl = ratio * dp;
        for ti in 0..6 {
            let theta = ((ti as f64) + 0.5) * (0.5 * PI) / 6.0;
            for pi_idx in 0..12 {
                let phi = -PI + ((pi_idx as f64) + 0.5) * (2.0 * PI) / 12.0;
                let cand = [dp, dl, theta, phi];
                let c2 = eval_chi2(&cand);
                if c2 < best_c2 {
                    best_c2 = c2;
                    best_p = cand;
                }
            }
        }
    }

    // Damped 4x4 Levenberg–Marquardt polish
    let mut p = best_p;
    let mut chi2_cur = best_c2;
    let mut lambda = 1.0e-3_f64;
    let n_obs = obs.len();

    let residual_vec = |q_par: &[f64; 4], out: &mut [f64]| {
        let (st, ct) = q_par[2].sin_cos();
        let (sp, cp) = q_par[3].sin_cos();
        let axis = [st * cp, st * sp, ct];
        for (i, &(u, q_obs, sq, mhz, _)) in obs.iter().enumerate() {
            let comps = woessner_axial_components(u, axis, q_par[0], q_par[1]);
            let (r1, r2) = rigid_r1_r2_from_rot_components(&comps, mhz, &cfg.physics);
            out[i] = ((r2 / r1.max(1.0e-30)) - q_obs) / sq;
        }
    };

    let mut r_cur = vec![0.0_f64; n_obs];
    let mut r_tmp = vec![0.0_f64; n_obs];
    residual_vec(&p, &mut r_cur);

    for _iter in 0..35 {
        let eps = [1.0e-5 * p[0], 1.0e-5 * p[1], 1.0e-5, 1.0e-5];
        let mut j_mat = vec![[0.0_f64; 4]; n_obs];
        for k in 0..4 {
            let mut pk = p;
            pk[k] += eps[k];
            residual_vec(&pk, &mut r_tmp);
            let inv_h = 1.0 / eps[k];
            for i in 0..n_obs {
                j_mat[i][k] = (r_tmp[i] - r_cur[i]) * inv_h;
            }
        }
        let mut jtj = [0.0_f64; 16];
        let mut jtr = [0.0_f64; 4];
        for i in 0..n_obs {
            for a in 0..4 {
                jtr[a] -= j_mat[i][a] * r_cur[i];
                for b in 0..4 {
                    jtj[a * 4 + b] += j_mat[i][a] * j_mat[i][b];
                }
            }
        }

        let mut accepted = false;
        for _sub in 0..6 {
            let mut lhs = jtj;
            for d in 0..4 {
                lhs[d * 4 + d] += lambda * jtj[d * 4 + d].max(1.0e-8);
            }
            let step = solve_4x4(&lhs, &jtr);
            let mut cand = [
                (p[0] + step[0]).clamp(0.1 * d_iso_0, 10.0 * d_iso_0),
                (p[1] + step[1]).clamp(0.1 * d_iso_0, 10.0 * d_iso_0),
                p[2] + step[2],
                p[3] + step[3],
            ];
            // Canonicalize polar angles into theta in [0, pi], phi in (-pi, pi]
            let (st, ct) = cand[2].sin_cos();
            let (sp, cp) = cand[3].sin_cos();
            let mut ax = [st * cp, st * sp, ct];
            if ax[2] < 0.0 {
                ax = [-ax[0], -ax[1], -ax[2]]; // antipodal symmetry of axial diffusion tensor!
            }
            cand[2] = ax[2].clamp(-1.0, 1.0).acos();
            cand[3] = ax[1].atan2(ax[0]);

            let c2_c = eval_chi2(&cand);
            if c2_c < chi2_cur {
                let rel = (chi2_cur - c2_c) / chi2_cur.max(1.0e-12);
                p = cand;
                chi2_cur = c2_c;
                residual_vec(&p, &mut r_cur);
                lambda = (lambda * 0.3).max(1.0e-8);
                accepted = true;
                if rel < 1.0e-9 {
                    break;
                }
                break;
            }
            lambda = (lambda * 6.0).min(1.0e6);
        }
        if !accepted {
            break;
        }
    }

    let d_iso = (p[1] + 2.0 * p[0]) / 3.0;
    Ok(AxialDiffusionEstimate {
        tau_m_sec: 1.0 / (6.0 * d_iso),
        d_par_s_inv: p[1],
        d_perp_s_inv: p[0],
        anisotropy_ratio: p[1] / p[0],
        theta_rad: p[2],
        phi_rad: p[3],
        retained_seq_ids: obs.iter().map(|o| o.4).collect(),
        chi2: chi2_cur,
    })
}

fn solve_4x4(a: &[f64; 16], b: &[f64; 4]) -> [f64; 4] {
    let mut m = [[0.0_f64; 5]; 4];
    for r in 0..4 {
        for c in 0..4 {
            m[r][c] = a[r * 4 + c];
        }
        m[r][4] = b[r];
    }
    for col in 0..4 {
        let mut piv = col;
        let mut mv = m[col][col].abs();
        for (r, row) in m.iter().enumerate().skip(col + 1) {
            if row[col].abs() > mv {
                mv = row[col].abs();
                piv = r;
            }
        }
        if mv <= 1.0e-24 {
            return [0.0; 4];
        }
        m.swap(col, piv);
        let inv = 1.0 / m[col][col];
        let piv_row = m[col];
        for row_r in m.iter_mut().skip(col + 1) {
            let f = row_r[col] * inv;
            for (dst, &src) in row_r[col..5].iter_mut().zip(piv_row[col..5].iter()) {
                *dst -= f * src;
            }
        }
    }
    let mut x = [0.0_f64; 4];
    for i in (0..4).rev() {
        let mut s = m[i][4];
        for j in (i + 1)..4 {
            s -= m[i][j] * x[j];
        }
        x[i] = s / m[i][i];
    }
    x
}

// ============================================================================
// Reduced Spectral Density Mapping J(0), J(omega_N), J(0.87 * omega_H)
// (Farrow et al., J. Biomol. NMR 1995, 6, 153-162; Lefevre et al., 1996)
// ============================================================================

/// Reduced spectral density values $(\text{s rad}^{-1})$ and propagated $1\sigma$ uncertainties for one residue.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReducedSpectralDensity {
    pub seq_id: i32,
    pub res_name: String,
    pub field_mhz: f64,
    pub j_0: f64,
    pub sigma_j_0: f64,
    pub j_omega_n: f64,
    pub sigma_j_omega_n: f64,
    pub j_087_omega_h: f64,
    pub sigma_j_087_omega_h: f64,
}

/// Compute Farrow et al. (*J. Biomol. NMR* 1995) reduced spectral density values
/// $J(0), J(\omega_N), J(0.87\omega_H)$ (under exact $\omega^{-2}$ high-frequency proton scaling
/// with signed $\gamma_N < 0$) and analytical $1\sigma$ uncertainties.
pub fn map_reduced_spectral_density(
    obs: &ResidueRelaxRates,
    phys: &RelaxPhysicsConfig,
) -> Result<ReducedSpectralDensity, String> {
    let noe = obs.hetnoe.ok_or_else(|| {
        format!(
            "reduced spectral density mapping on residue {} requires hetNOE",
            obs.seq_id
        )
    })?;
    let sigma_noe = obs.sigma_hetnoe.unwrap_or(0.02);

    let d = phys.dipolar_d_rad_s();
    let d2 = d * d;
    let c = phys.csa_c_rad_s(obs.field_mhz);
    let c2 = c * c;

    // Signed ratio r_gn = gamma_N / gamma_H < 0 (~ -0.101398)
    let r_gn = phys.gamma_n / phys.gamma_h;
    let f_sum = 1.0 + r_gn; // |omega_H + omega_N| / omega_H ~ 0.89860
    let f_dif = 1.0 - r_gn; // |omega_H - omega_N| / omega_H ~ 1.10140

    // Under J(omega) ~ omega^-2 near omega_H:
    // 6 J(omega_H + omega_N) - J(omega_H - omega_N) = [6 f_sum^-2 - f_dif^-2] J(omega_H) = 5 J(f_eff * omega_H)
    // where f_eff = sqrt(5 / (6 f_sum^-2 - f_dif^-2)) = 0.87004!
    let k_sigma = 6.0 / (f_sum * f_sum) - 1.0 / (f_dif * f_dif); // ~ 6.6063
    let k_r1_hf = 6.0 / (f_sum * f_sum) + 1.0 / (f_dif * f_dif); // ~ 8.2547
    let k_r2_hf = 6.0 + 6.0 / (f_sum * f_sum) + 1.0 / (f_dif * f_dif); // ~ 14.2547

    // sigma_NH = (NOE - 1) * R1 * (gamma_N / gamma_H) = (d^2 / 4) * 5 * J(0.87 omega_H)
    let sigma_nh = (noe - 1.0) * obs.r1 * r_gn;
    let j_087 = (4.0 * sigma_nh) / (5.0 * d2);

    // High-frequency proton contributions expressed strictly in terms of J(0.87 omega_H):
    // (d^2/4)[J(wH-wN) + 6 J(wH+wN)] = (d^2/4) * (5 * k_r1_hf / k_sigma) * J(0.87 wH)
    let alpha_r1 = 0.25 * d2 * (5.0 * k_r1_hf / k_sigma);
    let denom_jn = 0.75 * d2 + c2;
    let j_wn = (obs.r1 - alpha_r1 * j_087) / denom_jn;

    // (d^2/8)[J(wH-wN) + 6 J(wH) + 6 J(wH+wN)] = (d^2/8) * (5 * k_r2_hf / k_sigma) * J(0.87 wH)
    let alpha_r2 = 0.125 * d2 * (5.0 * k_r2_hf / k_sigma);
    let beta_jn = 0.375 * d2 + 0.5 * c2;
    let denom_j0 = 0.5 * d2 + (2.0 / 3.0) * c2;
    let j_0 = (obs.r2 - beta_jn * j_wn - alpha_r2 * j_087) / denom_j0;

    // Exact linear Jacobian error propagation from independent (R1, R2, NOE):
    let dj87_dr1 = (4.0 * (noe - 1.0) * r_gn) / (5.0 * d2);
    let dj87_dnoe = (4.0 * obs.r1 * r_gn) / (5.0 * d2);
    let var_j87 = (dj87_dr1 * obs.sigma_r1).powi(2) + (dj87_dnoe * sigma_noe).powi(2);

    let djn_dr1 = (1.0 - alpha_r1 * dj87_dr1) / denom_jn;
    let djn_dnoe = (-alpha_r1 * dj87_dnoe) / denom_jn;
    let var_jn = (djn_dr1 * obs.sigma_r1).powi(2) + (djn_dnoe * sigma_noe).powi(2);

    let dj0_dr2 = 1.0 / denom_j0;
    let dj0_dr1 = (-beta_jn * djn_dr1 - alpha_r2 * dj87_dr1) / denom_j0;
    let dj0_dnoe = (-beta_jn * djn_dnoe - alpha_r2 * dj87_dnoe) / denom_j0;
    let var_j0 = (dj0_dr2 * obs.sigma_r2).powi(2)
        + (dj0_dr1 * obs.sigma_r1).powi(2)
        + (dj0_dnoe * sigma_noe).powi(2);

    Ok(ReducedSpectralDensity {
        seq_id: obs.seq_id,
        res_name: obs.res_name.clone(),
        field_mhz: obs.field_mhz,
        j_0,
        sigma_j_0: var_j0.sqrt(),
        j_omega_n: j_wn,
        sigma_j_omega_n: var_jn.sqrt(),
        j_087_omega_h: j_087,
        sigma_j_087_omega_h: var_j87.sqrt(),
    })
}

/// Batch reduced spectral density mapping across a residue list.
pub fn map_reduced_spectral_density_batch(
    rates: &[ResidueRelaxRates],
    phys: &RelaxPhysicsConfig,
) -> Result<Vec<ReducedSpectralDensity>, String> {
    rates
        .iter()
        .map(|r| map_reduced_spectral_density(r, phys))
        .collect()
}
