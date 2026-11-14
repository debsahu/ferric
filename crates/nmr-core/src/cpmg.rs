//! Carr–Purcell–Meiboom–Gill (`CPMG`) and off-resonance rotating-frame (`R1rho`) chemical exchange
//! relaxation dispersion forward models, replicate intensity error propagation, statistical
//! exchange gating (`AIC` + nested `F`-test), multi-field global fitting, and TSV/JSON I/O (`T13.5`).
//!
//! Provenance: Clean-room original implementation from peer-reviewed biomolecular NMR literature:
//! - Luz, Z. & Meiboom, S. (*J. Chem. Phys.* 1963, 39, 366–370 — fast-exchange `CPMG` dispersion
//!   `LM63`).
//! - Carver, J. P. & Richards, R. E. (*J. Magn. Reson.* 1972, 6, 89–105 — general two-site
//!   eigenvalue solution `CR72`).
//! - Baldwin, A. J. (*J. Magn. Reson.* 2014, 244, 114–124 — exact closed-form two-site
//!   `R2,eff(nu_CPMG)` across all exchange timescales including `Delta_R2 = R2B - R2A != 0` and
//!   eigenvector projection weights).
//! - Trott, O. & Palmer, A. G. III (*J. Magn. Reson.* 2002, 154, 157–160 — asymmetric
//!   off-resonance `R1rho(omega_eff, theta)` relaxation outside the fast-exchange limit).

#![allow(clippy::needless_range_loop)]

use std::collections::BTreeMap;
use std::f64::consts::TAU;

use serde::{Deserialize, Serialize};

/// Dispersion model family for `CPMG` curve evaluation and fitting.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum CpmgModel {
    /// Constant horizontal baseline `R2,eff(nu) = R20` (`0` exchange contribution).
    Flat,
    /// Luz–Meiboom (1963) fast-exchange approximation (`Phi_ex = p_A * p_B * dw^2`, `k_ex`).
    Lm63,
    /// Carver–Richards (1972) dominant-eigenvalue 2-site dispersion (`CR72`).
    Cr72,
    /// Baldwin (2014) exact closed-form 2-site dispersion including `Delta_R2 != 0` and finite-`T`
    /// eigenvector projection weights.
    #[default]
    Baldwin2014,
    /// Explicit numerical `2x2` complex spin-echo matrix propagation `[U(tau) U*(2*tau) U(tau)]^n`.
    Numerical2x2,
}

impl CpmgModel {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Flat => "flat",
            Self::Lm63 => "lm63",
            Self::Cr72 => "cr72",
            Self::Baldwin2014 => "baldwin2014",
            Self::Numerical2x2 => "numerical",
        }
    }

    pub fn from_str_label(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "flat" | "null" => Ok(Self::Flat),
            "lm63" | "luz_meiboom" => Ok(Self::Lm63),
            "cr72" | "carver_richards" => Ok(Self::Cr72),
            "baldwin2014" | "baldwin" | "b14" => Ok(Self::Baldwin2014),
            "numerical" | "numerical2x2" | "matrix" => Ok(Self::Numerical2x2),
            other => Err(format!("unknown CPMG dispersion model '{other}'")),
        }
    }
}

/// Raw constant-time `CPMG` peak intensity plane measurement (`nu_cpmg_hz = 0` denotes `I_0`
/// reference planes acquired with `T_relax = 0`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RawCpmgIntensityPlane {
    /// Refocusing frequency `nu_CPMG = n_cyc / T_relax` (`Hz`), or `0.0` for reference `I_0`.
    pub nu_cpmg_hz: f64,
    /// Measured peak height or integrated volume.
    pub intensity: f64,
    /// Optional per-plane spectral baseline RMS noise `sigma_plane`.
    pub baseline_noise: Option<f64>,
}

/// Processed dispersion curve `R2,eff(nu_CPMG) +- sigma` for one residue at one static `B0` field.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CpmgSeriesDataset {
    pub seq_id: i32,
    pub res_name: String,
    /// Heteronuclear Larmor frequency `|nu_0|` (`MHz`, e.g. `60.8106` for `15N` at `600 MHz` `1H`,
    /// `81.0808` at `800 MHz` `1H`).
    pub larmor_mhz: f64,
    /// Constant-time CPMG relaxation period `T_relax` (`s`, e.g. `0.040`).
    pub t_relax_s: f64,
    /// Unique sorted refocusing frequencies `nu_CPMG` (`Hz`).
    pub nu_cpmg_hz: Vec<f64>,
    /// Effective transverse relaxation rates `R2,eff` (`s^-1`).
    pub r2_eff_s_inv: Vec<f64>,
    /// `1-sigma` uncertainties on `R2,eff` (`s^-1`).
    pub r2_eff_err: Vec<f64>,
}

/// Statistical exchange gate configuration testing dispersion against the `Flat` null model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExchangeGateConfig {
    /// Minimum required dispersion step `max(R2,eff) - min(R2,eff)` (`s^-1`; default `2.0`).
    pub min_delta_r2_eff_s_inv: f64,
    /// Minimum AIC improvement `AIC_flat - AIC_ex` (`default 2.0`).
    pub min_delta_aic: f64,
    /// Maximum nested `F`-test `p`-value (`default 0.05`).
    pub max_f_pvalue: f64,
}

impl Default for ExchangeGateConfig {
    fn default() -> Self {
        Self {
            min_delta_r2_eff_s_inv: 2.0,
            min_delta_aic: 2.0,
            max_f_pvalue: 0.05,
        }
    }
}

/// Result of testing a residue's dispersion curves against the `Flat` no-exchange null model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExchangeGateVerdict {
    pub seq_id: i32,
    pub res_name: String,
    pub significant_exchange: bool,
    pub observed_rex_s_inv: f64,
    pub flat_chi2: f64,
    pub ex_chi2: f64,
    pub delta_aic: f64,
    pub f_stat: f64,
    pub f_pvalue: f64,
}

/// Configuration for multi-field global `CPMG` dispersion fitting.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CpmgFitConfig {
    pub model: CpmgModel,
    pub gate: ExchangeGateConfig,
    /// If `true`, only include residues passing `gate` in the global `(p_B, k_ex)` fit.
    pub filter_by_exchange_gate: bool,
    /// `(n_pb, n_kex)` grid dimensions for global `(p_B, k_ex)` search.
    pub grid_dims: (usize, usize),
    pub max_lm_iters: usize,
}

impl Default for CpmgFitConfig {
    fn default() -> Self {
        Self {
            model: CpmgModel::Baldwin2014,
            gate: ExchangeGateConfig::default(),
            filter_by_exchange_gate: true,
            grid_dims: (36, 44),
            max_lm_iters: 40,
        }
    }
}

/// Per-residue multi-field `CPMG` fit output.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CpmgResidueFit {
    pub seq_id: i32,
    pub res_name: String,
    /// Magnitude of chemical shift difference `|Delta_varpi|` (`ppm`).
    pub dw_ppm: f64,
    pub dw_ppm_err: f64,
    /// Intrinsic high-frequency baseline `R20` (`s^-1`) per static field `(larmor_mhz, r20, r20_err)`.
    pub field_r20: Vec<(f64, f64, f64)>,
    /// Millet et al. (*JACS* 2000) exchange timescale scaling exponent `alpha in [0, 2]` when `>= 2` fields.
    pub alpha_rex_scaling: Option<f64>,
    pub gate: ExchangeGateVerdict,
    pub chi2: f64,
    pub n_points: usize,
}

/// Complete multi-field global `CPMG` fit summary.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CpmgGlobalFitResult {
    pub model: CpmgModel,
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
    pub residues: Vec<CpmgResidueFit>,
}

/// Off-resonance rotating-frame `R1rho` measurement & Trott–Palmer (2002) geometry point.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct R1RhoOffResPoint {
    /// Spin-lock carrier frequency `varpi_rf` (`ppm`).
    pub carrier_ppm: f64,
    /// Spin-lock nutation frequency `nu_1 = omega_1 / (2*pi)` (`Hz`).
    pub spin_lock_hz: f64,
    /// Effective field magnitude `omega_eff` (`rad/s`) in population-averaged frame.
    pub omega_eff_rad_s: f64,
    /// Tilt angle `theta = atan2(omega_1, Omega_bar)` (`rad`, `0` along `+z`, `pi/2` on-resonance).
    pub theta_rad: f64,
    /// Measured or simulated `R1rho` (`s^-1`).
    pub r1rho_s_inv: f64,
    pub r1rho_err: f64,
}

/// Fitted Trott–Palmer (2002) asymmetric off-resonance `R1rho` result for a single residue
/// (recovering the **signed** excited-state chemical shift difference `dw_ab_ppm`!).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct R1RhoFitResult {
    pub p_b: f64,
    pub k_ex_s_inv: f64,
    /// Signed chemical shift difference `Delta_varpi_AB = varpi_B - varpi_A` (`ppm`).
    pub dw_ab_ppm: f64,
    pub r1_s_inv: f64,
    pub r20_s_inv: f64,
    pub chi2: f64,
}

// ============================================================================
// 1. Intensity-to-R2,eff Conversion with Replicate Plane Error Propagation
// ============================================================================

/// Convert raw constant-time `CPMG` intensity planes (`I_0` at `nu_cpmg_hz == 0` and refocused
/// planes at `nu_cpmg_hz > 0`) into a [`CpmgSeriesDataset`] with propagated uncertainties from
/// replicate planes and optional spectral baseline noise (`sigma_floor_s_inv` minimum).
pub fn compute_r2eff_from_intensities(
    seq_id: i32,
    res_name: impl Into<String>,
    larmor_mhz: f64,
    t_relax_s: f64,
    planes: &[RawCpmgIntensityPlane],
    sigma_floor_s_inv: f64,
) -> Result<CpmgSeriesDataset, String> {
    if !t_relax_s.is_finite() || t_relax_s <= 0.0 {
        return Err(format!("invalid t_relax_s={t_relax_s}: must be > 0"));
    }
    if !larmor_mhz.is_finite() || larmor_mhz <= 0.0 {
        return Err(format!("invalid larmor_mhz={larmor_mhz}: must be > 0"));
    }

    // Group planes by quantized nu_cpmg_hz (0.001 Hz resolution)
    let mut i0_vals = Vec::new();
    let mut i0_noises = Vec::new();
    let mut groups: BTreeMap<i64, (f64, Vec<f64>, Vec<f64>)> = BTreeMap::new();

    for p in planes {
        if !p.nu_cpmg_hz.is_finite() || p.nu_cpmg_hz < 0.0 || !p.intensity.is_finite() {
            return Err("invalid non-finite or negative CPMG intensity plane".to_string());
        }
        if p.intensity <= 0.0 {
            return Err(format!(
                "non-positive CPMG plane intensity {} at nu={} Hz",
                p.intensity, p.nu_cpmg_hz
            ));
        }
        if p.nu_cpmg_hz == 0.0 {
            i0_vals.push(p.intensity);
            if let Some(n) = p.baseline_noise {
                if n.is_finite() && n > 0.0 {
                    i0_noises.push(n);
                }
            }
        } else {
            let key = (p.nu_cpmg_hz * 1000.0).round() as i64;
            let entry = groups
                .entry(key)
                .or_insert_with(|| (p.nu_cpmg_hz, Vec::new(), Vec::new()));
            entry.1.push(p.intensity);
            if let Some(n) = p.baseline_noise {
                if n.is_finite() && n > 0.0 {
                    entry.2.push(n);
                }
            }
        }
    }

    if i0_vals.is_empty() {
        return Err("compute_r2eff_from_intensities requires at least one reference I0 plane (nu_cpmg_hz = 0)".to_string());
    }
    if groups.is_empty() {
        return Err("compute_r2eff_from_intensities requires at least one refocused CPMG plane (nu_cpmg_hz > 0)".to_string());
    }

    // Estimate pooled single-plane intensity SD across all replicate groups (including replicate I0)
    let mut ss_rep = 0.0f64;
    let mut dof_rep = 0usize;
    if i0_vals.len() >= 2 {
        let mean0 = i0_vals.iter().sum::<f64>() / (i0_vals.len() as f64);
        ss_rep += i0_vals.iter().map(|&v| (v - mean0).powi(2)).sum::<f64>();
        dof_rep += i0_vals.len() - 1;
    }
    for (_, vals, _) in groups.values() {
        if vals.len() >= 2 {
            let m = vals.iter().sum::<f64>() / (vals.len() as f64);
            ss_rep += vals.iter().map(|&v| (v - m).powi(2)).sum::<f64>();
            dof_rep += vals.len() - 1;
        }
    }
    let sigma_rep = if dof_rep > 0 {
        (ss_rep / (dof_rep as f64)).sqrt()
    } else {
        0.0
    };

    let mean_i0 = i0_vals.iter().sum::<f64>() / (i0_vals.len() as f64);
    let noise_i0 = if !i0_noises.is_empty() {
        i0_noises.iter().sum::<f64>() / (i0_noises.len() as f64)
    } else {
        0.0
    };
    let single_sig_i0 = sigma_rep.max(noise_i0).max(1e-4 * mean_i0);
    let rel_var_i0 = (single_sig_i0 * single_sig_i0 / (i0_vals.len() as f64)) / (mean_i0 * mean_i0);

    let mut nu_out = Vec::with_capacity(groups.len());
    let mut r2_out = Vec::with_capacity(groups.len());
    let mut err_out = Vec::with_capacity(groups.len());

    for (_, (nu, vals, noises)) in groups {
        let m_k = vals.len() as f64;
        let mean_ik = vals.iter().sum::<f64>() / m_k;
        let r2_eff = -(mean_ik / mean_i0).ln() / t_relax_s;

        let plane_noise = if !noises.is_empty() {
            noises.iter().sum::<f64>() / (noises.len() as f64)
        } else {
            noise_i0
        };
        let single_sig_ik = sigma_rep.max(plane_noise).max(1e-4 * mean_i0);
        let rel_var_ik = (single_sig_ik * single_sig_ik / m_k) / (mean_ik * mean_ik);
        let sig_r2 = ((rel_var_ik + rel_var_i0).sqrt() / t_relax_s).max(sigma_floor_s_inv.max(0.0));

        nu_out.push(nu);
        r2_out.push(r2_eff);
        err_out.push(sig_r2);
    }

    Ok(CpmgSeriesDataset {
        seq_id,
        res_name: res_name.into(),
        larmor_mhz,
        t_relax_s,
        nu_cpmg_hz: nu_out,
        r2_eff_s_inv: r2_out,
        r2_eff_err: err_out,
    })
}

// ============================================================================
// 2. Analytical & Numerical CPMG + Off-Resonance R1rho Forward Equations
// ============================================================================

/// Flat null model `R2,eff(nu_CPMG) = R20`.
#[inline]
pub fn r2eff_flat(r20_s_inv: f64) -> f64 {
    r20_s_inv
}

/// Luz & Meiboom (*J. Chem. Phys.* 1963) fast-exchange CPMG formula (`Phi_ex = p_A p_B dw^2` in `rad^2 s^-2`).
#[inline]
pub fn r2eff_lm63(r20_s_inv: f64, phi_ex_rad2: f64, k_ex_s_inv: f64, nu_cpmg_hz: f64) -> f64 {
    if k_ex_s_inv <= 1e-9 || phi_ex_rad2 <= 0.0 || nu_cpmg_hz <= 0.0 {
        return r20_s_inv;
    }
    let x = k_ex_s_inv / (4.0 * nu_cpmg_hz);
    let factor = if x.abs() < 1e-4 {
        let x2 = x * x;
        x2 / 3.0 - 2.0 * x2 * x2 / 15.0
    } else {
        1.0 - x.tanh() / x
    };
    r20_s_inv + (phi_ex_rad2 / k_ex_s_inv) * factor
}

/// Carver & Richards (*J. Magn. Reson.* 1972) dominant-eigenvalue 2-site `R2,eff(nu_CPMG)`.
#[inline]
pub fn r2eff_cr72(
    r2a_s_inv: f64,
    r2b_s_inv: f64,
    p_b: f64,
    k_ex_s_inv: f64,
    dw_rad_s: f64,
    nu_cpmg_hz: f64,
) -> f64 {
    let pb = p_b.clamp(0.0, 0.999_999);
    if pb == 0.0 || k_ex_s_inv <= 0.0 || nu_cpmg_hz <= 0.0 {
        return r2a_s_inv;
    }
    let pa = 1.0 - pb;
    let kab = pb * k_ex_s_inv;
    let kba = pa * k_ex_s_inv;
    let dr_ab = r2a_s_inv - r2b_s_inv;
    let alpha_m = dr_ab + kab - kba;
    let dw2 = dw_rad_s * dw_rad_s;
    let psi = alpha_m * alpha_m - dw2 + 4.0 * kab * kba;
    let zeta = 2.0 * dw_rad_s * alpha_m;
    let hyp = psi.hypot(zeta);
    if hyp < 1e-30 {
        return 0.5 * (r2a_s_inv + r2b_s_inv + k_ex_s_inv);
    }

    let inv_scale = 1.0 / (2.0 * std::f64::consts::SQRT_2 * nu_cpmg_hz);
    let eta_p = inv_scale * (hyp + psi).max(0.0).sqrt();
    let eta_m = inv_scale * (hyp - psi).max(0.0).sqrt();

    let ratio = (psi + 2.0 * dw2) / hyp;
    let d_p = 0.5 * (1.0 + ratio);
    let d_m = 0.5 * (-1.0 + ratio);

    let arg = (d_p * eta_p.cosh() - d_m * eta_m.cos()).max(1.0);
    0.5 * (r2a_s_inv + r2b_s_inv + k_ex_s_inv) - nu_cpmg_hz * arg.acosh()
}

/// Baldwin (*J. Magn. Reson.* 2014, 244, 114–124) exact closed-form two-site `R2,eff(nu_CPMG)`
/// across all exchange timescales, arbitrary minor population `p_B`, and arbitrary differential
/// transverse relaxation `Delta_R2 = R2B - R2A != 0`, incorporating exact finite-`T_relax`
/// eigenvector projection weights (`F_0 / F_2` / `y` correction).
pub fn r2eff_baldwin2014(
    r2a_s_inv: f64,
    r2b_s_inv: f64,
    p_b: f64,
    k_ex_s_inv: f64,
    dw_rad_s: f64,
    t_relax_s: f64,
    nu_cpmg_hz: f64,
) -> f64 {
    let pb = p_b.clamp(0.0, 0.999_999);
    if pb == 0.0 || k_ex_s_inv <= 0.0 || nu_cpmg_hz <= 0.0 || t_relax_s <= 0.0 {
        return r2a_s_inv;
    }
    let pa = 1.0 - pb;
    let kab = pb * k_ex_s_inv;
    let kba = pa * k_ex_s_inv;
    let r2_bar = 0.5 * (r2a_s_inv + r2b_s_inv);
    let dr_ab = r2a_s_inv - r2b_s_inv;
    let alpha_m = dr_ab + kab - kba;
    let dw2 = dw_rad_s * dw_rad_s;

    let psi = alpha_m * alpha_m - dw2 + 4.0 * kab * kba;
    let zeta = 2.0 * dw_rad_s * alpha_m;
    let hyp = psi.hypot(zeta);

    // Complex eigenvalue `z = 0.5 * sqrt(psi + i * zeta)` of traceless generator `M`
    let z_re = 0.5 * (0.5 * (hyp + psi).max(0.0)).sqrt();
    let z_im = 0.5 * (0.5 * (hyp - psi).max(0.0)).sqrt() * if zeta >= 0.0 { 1.0 } else { -1.0 };

    let tau = 1.0 / (4.0 * nu_cpmg_hz);
    let (_c1, s1) = complex_cosh_sinc((z_re, z_im), tau);
    let (c2, s2) = complex_cosh_sinc((z_re, z_im), 2.0 * tau);

    let t_m = 0.25 * (psi + 2.0 * dw2);
    let u_trace = (c2.0 * c2.0 + c2.1 * c2.1) + (s2.0 * s2.0 + s2.1 * s2.1) * t_m;

    let c2_conj = (c2.0, -c2.1);
    let s2_conj = (s2.0, -s2.1);
    let c_mul = |a: (f64, f64), b: (f64, f64)| (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0);
    let s1_sq = c_mul(s1, s1);
    let s1_sq_s2c = c_mul(s1_sq, s2_conj);
    let c2c_s2 = c_mul(c2_conj, s2);
    let beta1 = (
        c2c_s2.0 + 2.0 * t_m * s1_sq_s2c.0,
        c2c_s2.1 + 2.0 * t_m * s1_sq_s2c.1,
    );
    let beta2 = s2_conj;

    // `(M * v0)_A / p_A = 0.5 * (k_ex - dr_ab - i * dw)`
    let vm = (0.5 * (k_ex_s_inv - dr_ab), -0.5 * dw_rad_s);
    let vm_conj = (vm.0, -vm.1);
    let w_a_re = c_mul(beta1, vm).0 + c_mul(beta2, vm_conj).0;

    let n_cyc = t_relax_s * nu_cpmg_hz;
    let base_rate = r2_bar + 0.5 * k_ex_s_inv;

    if u_trace > 1.0 + 1e-11 {
        let gamma = u_trace.acosh();
        let sinh_g = (u_trace * u_trace - 1.0).sqrt(); // exact sinh(acosh(u))
        let exp_m2ng = (-2.0 * n_cyc * gamma).exp();
        let proj = 0.5 * (1.0 + exp_m2ng) + w_a_re * 0.5 * (1.0 - exp_m2ng) / sinh_g;
        base_rate - nu_cpmg_hz * gamma - proj.max(1e-300).ln() / t_relax_s
    } else if u_trace >= 1.0 - 1e-11 {
        let proj = (1.0 + n_cyc * w_a_re).max(1e-300);
        base_rate - proj.ln() / t_relax_s
    } else {
        let theta = u_trace.clamp(-1.0, 1.0).acos();
        let sin_t = theta.sin().max(1e-15);
        let proj = ((n_cyc * theta).cos() + w_a_re * (n_cyc * theta).sin() / sin_t).max(1e-300);
        base_rate - proj.ln() / t_relax_s
    }
}

#[inline]
fn complex_cosh_sinc(z: (f64, f64), t: f64) -> ((f64, f64), (f64, f64)) {
    let u = (z.0 * t, z.1 * t);
    let u_abs2 = u.0 * u.0 + u.1 * u.1;
    if u_abs2 < 1e-8 {
        let u2 = (u.0 * u.0 - u.1 * u.1, 2.0 * u.0 * u.1);
        let u4 = (u2.0 * u2.0 - u2.1 * u2.1, 2.0 * u2.0 * u2.1);
        let c = (1.0 + 0.5 * u2.0 + u4.0 / 24.0, 0.5 * u2.1 + u4.1 / 24.0);
        let s = (
            t * (1.0 + u2.0 / 6.0 + u4.0 / 120.0),
            t * (u2.1 / 6.0 + u4.1 / 120.0),
        );
        (c, s)
    } else {
        let c = (u.0.cosh() * u.1.cos(), u.0.sinh() * u.1.sin());
        let sh = (u.0.sinh() * u.1.cos(), u.0.cosh() * u.1.sin());
        let z_abs2 = z.0 * z.0 + z.1 * z.1;
        let s = (
            (sh.0 * z.0 + sh.1 * z.1) / z_abs2,
            (sh.1 * z.0 - sh.0 * z.1) / z_abs2,
        );
        (c, s)
    }
}

/// Exact numerical `2x2` complex spin-echo matrix propagation `[U(tau) * U*(2*tau) * U(tau)]^n_cyc`
/// for `CPMG` dispersion (`tau = 1 / (4 * nu_CPMG)`), serving as the gold-standard propagator.
pub fn r2eff_numerical_2x2(
    r2a_s_inv: f64,
    r2b_s_inv: f64,
    p_b: f64,
    k_ex_s_inv: f64,
    dw_rad_s: f64,
    t_relax_s: f64,
    nu_cpmg_hz: f64,
) -> f64 {
    let pb = p_b.clamp(0.0, 0.999_999);
    if pb == 0.0 || k_ex_s_inv <= 0.0 || nu_cpmg_hz <= 0.0 || t_relax_s <= 0.0 {
        return r2a_s_inv;
    }
    let pa = 1.0 - pb;
    let kab = pb * k_ex_s_inv;
    let kba = pa * k_ex_s_inv;

    let n_cyc = (t_relax_s * nu_cpmg_hz).round().max(1.0) as u64;
    let t_eff = (n_cyc as f64) / nu_cpmg_hz;
    let tau = 1.0 / (4.0 * nu_cpmg_hz);

    // 2x2 complex generator G = [[-R2A - kAB, kBA], [kAB, -R2B - kBA + i*dw]]
    let g = [
        [(-r2a_s_inv - kab, 0.0), (kba, 0.0)],
        [(kab, 0.0), (-r2b_s_inv - kba, dw_rad_s)],
    ];
    let u_tau = expm_complex_2x2(g, tau);
    let u_2tau = cmat2_mul(u_tau, u_tau);
    let u_2tau_conj = cmat2_conj(u_2tau);
    let e_block = cmat2_mul(cmat2_mul(u_tau, u_2tau_conj), u_tau);

    let e_total = cmat2_pow(e_block, n_cyc);
    // Initial equilibrium M0 = [p_A, p_B]^T; detected state A normalized by p_A
    let m_a_re = (e_total[0][0].0 * pa + e_total[0][1].0 * pb) / pa;
    -(m_a_re.max(1e-300).ln()) / t_eff
}

type CMat2 = [[(f64, f64); 2]; 2];

#[inline]
fn cmat2_mul(a: CMat2, b: CMat2) -> CMat2 {
    let cm = |x: (f64, f64), y: (f64, f64)| (x.0 * y.0 - x.1 * y.1, x.0 * y.1 + x.1 * y.0);
    let ca = |x: (f64, f64), y: (f64, f64)| (x.0 + y.0, x.1 + y.1);
    [
        [
            ca(cm(a[0][0], b[0][0]), cm(a[0][1], b[1][0])),
            ca(cm(a[0][0], b[0][1]), cm(a[0][1], b[1][1])),
        ],
        [
            ca(cm(a[1][0], b[0][0]), cm(a[1][1], b[1][0])),
            ca(cm(a[1][0], b[0][1]), cm(a[1][1], b[1][1])),
        ],
    ]
}

#[inline]
fn cmat2_conj(a: CMat2) -> CMat2 {
    [
        [(a[0][0].0, -a[0][0].1), (a[0][1].0, -a[0][1].1)],
        [(a[1][0].0, -a[1][0].1), (a[1][1].0, -a[1][1].1)],
    ]
}

#[inline]
fn cmat2_pow(mut base: CMat2, mut exp: u64) -> CMat2 {
    let mut acc: CMat2 = [[(1.0, 0.0), (0.0, 0.0)], [(0.0, 0.0), (1.0, 0.0)]];
    while exp > 0 {
        if exp & 1 == 1 {
            acc = cmat2_mul(acc, base);
        }
        exp >>= 1;
        if exp > 0 {
            base = cmat2_mul(base, base);
        }
    }
    acc
}

/// Exact `2x2` complex matrix exponential `exp(G * t)` via Putzer / trace-determinant identity.
fn expm_complex_2x2(g: CMat2, t: f64) -> CMat2 {
    let tr = (0.5 * (g[0][0].0 + g[1][1].0), 0.5 * (g[0][0].1 + g[1][1].1));
    // Traceless part M = G - tr * I
    let m00 = (g[0][0].0 - tr.0, g[0][0].1 - tr.1);
    let m01 = g[0][1];
    let m10 = g[1][0];
    let m11 = (g[1][1].0 - tr.0, g[1][1].1 - tr.1);
    // z^2 = m00^2 + m01*m10
    let z2 = (
        (m00.0 * m00.0 - m00.1 * m00.1) + (m01.0 * m10.0 - m01.1 * m10.1),
        2.0 * m00.0 * m00.1 + m01.0 * m10.1 + m01.1 * m10.0,
    );
    let r = z2.0.hypot(z2.1);
    let z = if r == 0.0 {
        (0.0, 0.0)
    } else {
        (
            (0.5 * (r + z2.0).max(0.0)).sqrt(),
            (0.5 * (r - z2.0).max(0.0)).sqrt() * if z2.1 >= 0.0 { 1.0 } else { -1.0 },
        )
    };
    let (c, s) = complex_cosh_sinc(z, t);
    let etr_mag = (tr.0 * t).exp();
    let etr = (etr_mag * (tr.1 * t).cos(), etr_mag * (tr.1 * t).sin());
    let cm = |x: (f64, f64), y: (f64, f64)| (x.0 * y.0 - x.1 * y.1, x.0 * y.1 + x.1 * y.0);
    let sm00 = cm(s, m00);
    let sm01 = cm(s, m01);
    let sm10 = cm(s, m10);
    let sm11 = cm(s, m11);
    [
        [cm(etr, (c.0 + sm00.0, c.1 + sm00.1)), cm(etr, sm01)],
        [cm(etr, sm10), cm(etr, (c.0 + sm11.0, c.1 + sm11.1))],
    ]
}

/// Evaluate `R2,eff(nu_CPMG)` for any [`CpmgModel`].
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn evaluate_cpmg_model(
    model: CpmgModel,
    r2a_s_inv: f64,
    r2b_s_inv: f64,
    p_b: f64,
    k_ex_s_inv: f64,
    dw_rad_s: f64,
    t_relax_s: f64,
    nu_cpmg_hz: f64,
) -> f64 {
    match model {
        CpmgModel::Flat => r2eff_flat(r2a_s_inv),
        CpmgModel::Lm63 => {
            let pb = p_b.clamp(0.0, 0.999_999);
            let phi_ex = (1.0 - pb) * pb * dw_rad_s * dw_rad_s;
            r2eff_lm63(r2a_s_inv, phi_ex, k_ex_s_inv, nu_cpmg_hz)
        }
        CpmgModel::Cr72 => r2eff_cr72(r2a_s_inv, r2b_s_inv, p_b, k_ex_s_inv, dw_rad_s, nu_cpmg_hz),
        CpmgModel::Baldwin2014 => r2eff_baldwin2014(
            r2a_s_inv, r2b_s_inv, p_b, k_ex_s_inv, dw_rad_s, t_relax_s, nu_cpmg_hz,
        ),
        CpmgModel::Numerical2x2 => r2eff_numerical_2x2(
            r2a_s_inv, r2b_s_inv, p_b, k_ex_s_inv, dw_rad_s, t_relax_s, nu_cpmg_hz,
        ),
    }
}

/// Trott & Palmer (*J. Magn. Reson.* 2002, 154, 157–160, Eq. 12) asymmetric off-resonance
/// rotating-frame relaxation rate `R1rho(omega_eff, theta)`, returning `(omega_eff_rad_s, theta_rad, r1rho_s_inv)`.
///
/// Here `omega_a_offset_rad_s = omega_A - omega_rf` and `dw_ab_rad_s = omega_B - omega_A` (signed!).
#[inline]
pub fn r1rho_trott_palmer_2002(
    r1_s_inv: f64,
    r20_s_inv: f64,
    p_b: f64,
    k_ex_s_inv: f64,
    omega_a_offset_rad_s: f64,
    dw_ab_rad_s: f64,
    w1_rad_s: f64,
) -> (f64, f64, f64) {
    let pb = p_b.clamp(0.0, 0.999_999);
    let pa = 1.0 - pb;
    let om_a = omega_a_offset_rad_s;
    let om_b = omega_a_offset_rad_s + dw_ab_rad_s;
    let om_bar = pa * om_a + pb * om_b;

    let w1_sq = w1_rad_s * w1_rad_s;
    let w_ae_sq = om_a * om_a + w1_sq;
    let w_be_sq = om_b * om_b + w1_sq;
    let w_e_sq = (om_bar * om_bar + w1_sq).max(1e-24);
    let w_eff = w_e_sq.sqrt();
    let theta = w1_rad_s.atan2(om_bar);

    let sin2 = w1_sq / w_e_sq;
    let cos2 = (om_bar * om_bar) / w_e_sq;

    let denom = (w_ae_sq * w_be_sq) / w_e_sq + k_ex_s_inv * k_ex_s_inv;
    let rex = if denom > 1e-24 {
        (pa * pb * dw_ab_rad_s * dw_ab_rad_s * k_ex_s_inv) / denom
    } else {
        0.0
    };
    let r1rho = r1_s_inv * cos2 + r20_s_inv * sin2 + sin2 * rex;
    (w_eff, theta, r1rho)
}

/// Fit Trott–Palmer (2002) asymmetric off-resonance `R1rho` data for a single residue (`major_ppm`,
/// `larmor_mhz`), recovering both exchange kinetics `(p_B, k_ex)` and the **signed** excited-state
/// chemical shift difference `dw_ab_ppm`!
pub fn fit_r1rho_trott_palmer_cpu(
    major_ppm: f64,
    larmor_mhz: f64,
    points: &[R1RhoOffResPoint],
) -> Result<R1RhoFitResult, String> {
    if points.len() < 6 {
        return Err(
            "fit_r1rho_trott_palmer_cpu requires at least 6 off-resonance R1rho points".to_string(),
        );
    }
    let scale = TAU * larmor_mhz;

    // Coarse signed search over `(dw_ab_ppm, p_B, k_ex)` with exact linear WLS for `(R1, R20)`!
    let eval_wls = |pb: f64, kex: f64, dw_ppm: f64| -> (f64, f64, f64) {
        let dw_rad = scale * dw_ppm;
        let mut s_cc = 0.0;
        let mut s_cs = 0.0;
        let mut s_ss = 0.0;
        let mut s_cy = 0.0;
        let mut s_sy = 0.0;
        let mut cached = Vec::with_capacity(points.len());
        for pt in points {
            let om_a = scale * (major_ppm - pt.carrier_ppm);
            let w1 = TAU * pt.spin_lock_hz;
            let (_, _, r_ex_only) = r1rho_trott_palmer_2002(0.0, 0.0, pb, kex, om_a, dw_rad, w1);
            let om_bar = om_a + pb * dw_rad;
            let we2 = (om_bar * om_bar + w1 * w1).max(1e-24);
            let cos2 = om_bar * om_bar / we2;
            let sin2 = w1 * w1 / we2;
            let y_rem = pt.r1rho_s_inv - r_ex_only;
            let w = 1.0 / (pt.r1rho_err.max(1e-3)).powi(2);
            s_cc += w * cos2 * cos2;
            s_cs += w * cos2 * sin2;
            s_ss += w * sin2 * sin2;
            s_cy += w * cos2 * y_rem;
            s_sy += w * sin2 * y_rem;
            cached.push((cos2, sin2, r_ex_only, w, pt.r1rho_s_inv));
        }
        let det = (s_cc * s_ss - s_cs * s_cs).max(1e-18);
        let r1 = ((s_ss * s_cy - s_cs * s_sy) / det).clamp(0.05, 20.0);
        let r20 = ((-s_cs * s_cy + s_cc * s_sy) / det).clamp(0.2, 200.0);
        let mut chi2 = 0.0;
        for &(c2, s2, rex, w, y) in &cached {
            let pred = r1 * c2 + r20 * s2 + rex;
            chi2 += w * (y - pred).powi(2);
        }
        (chi2, r1, r20)
    };

    let mut best = (f64::INFINITY, 0.02, 500.0, 3.0, 1.5, 12.0);
    for &pb in &[0.005, 0.015, 0.03, 0.06, 0.12] {
        for &kex in &[100.0, 300.0, 800.0, 2000.0, 4500.0] {
            for step in -24..=24 {
                if step == 0 {
                    continue;
                }
                let dw = (step as f64) * 0.4;
                let (c2, r1, r20) = eval_wls(pb, kex, dw);
                if c2 < best.0 {
                    best = (c2, pb, kex, dw, r1, r20);
                }
            }
        }
    }

    // 3x3 Levenberg–Marquardt polish on `(ln_pb, ln_kex, dw_ppm)` with variable-projection WLS `(R1, R20)`
    // so curved `(p_B, dw_AB)` valleys are traversed quadratically without coordinate-search stalling.
    let eval_residuals = |x_vec: [f64; 3]| -> (f64, f64, f64, Vec<f64>) {
        let pb = x_vec[0].exp().clamp(1e-4, 0.45);
        let kex = x_vec[1].exp().clamp(5.0, 30_000.0);
        let dw = x_vec[2].clamp(-30.0, 30.0);
        let (c2, r1, r20) = eval_wls(pb, kex, dw);
        let dw_rad = scale * dw;
        let mut res = Vec::with_capacity(points.len());
        for pt in points {
            let om_a = scale * (major_ppm - pt.carrier_ppm);
            let w1 = TAU * pt.spin_lock_hz;
            let (_, _, pred) = r1rho_trott_palmer_2002(r1, r20, pb, kex, om_a, dw_rad, w1);
            let inv_s = 1.0 / pt.r1rho_err.max(1e-3);
            res.push((pred - pt.r1rho_s_inv) * inv_s);
        }
        (c2, r1, r20, res)
    };

    let mut x = [best.1.ln(), best.2.ln(), best.3];
    let (mut best_c2, mut best_r1, mut best_r20, mut r_cur) = eval_residuals(x);
    let mut lambda = 1e-2_f64;

    for _iter in 0..50 {
        let m = r_cur.len();
        let mut jac = vec![[0.0f64; 3]; m];
        let eps = [1e-4, 1e-4, 1e-4];
        for d in 0..3 {
            let mut xp = x;
            xp[d] += eps[d];
            let (_, _, _, rp) = eval_residuals(xp);
            for i in 0..m {
                jac[i][d] = (rp[i] - r_cur[i]) / eps[d];
            }
        }
        let mut jtj = [[0.0f64; 3]; 3];
        let mut jtr = [0.0f64; 3];
        for i in 0..m {
            for a in 0..3 {
                jtr[a] -= jac[i][a] * r_cur[i];
                for b in 0..3 {
                    jtj[a][b] += jac[i][a] * jac[i][b];
                }
            }
        }
        let mut accepted = false;
        for _sub in 0..8 {
            let mut a_mat = jtj;
            for (d, row) in a_mat.iter_mut().enumerate() {
                row[d] += lambda * row[d].max(1e-3);
            }
            // Solve 3x3 linear system `a_mat * step = jtr` via Gaussian elimination
            let mut aug = [
                [a_mat[0][0], a_mat[0][1], a_mat[0][2], jtr[0]],
                [a_mat[1][0], a_mat[1][1], a_mat[1][2], jtr[1]],
                [a_mat[2][0], a_mat[2][1], a_mat[2][2], jtr[2]],
            ];
            let mut ok = true;
            for col in 0..3 {
                let mut piv = col;
                for r in (col + 1)..3 {
                    if aug[r][col].abs() > aug[piv][col].abs() {
                        piv = r;
                    }
                }
                if aug[piv][col].abs() < 1e-18 {
                    ok = false;
                    break;
                }
                aug.swap(col, piv);
                let inv = 1.0 / aug[col][col];
                for r in (col + 1)..3 {
                    let f = aug[r][col] * inv;
                    for c in col..4 {
                        aug[r][c] -= f * aug[col][c];
                    }
                }
            }
            if !ok {
                lambda *= 10.0;
                continue;
            }
            let mut step = [0.0f64; 3];
            for i in (0..3).rev() {
                let mut s = aug[i][3];
                for k in (i + 1)..3 {
                    s -= aug[i][k] * step[k];
                }
                step[i] = s / aug[i][i];
            }
            let cand = [
                (x[0] + step[0].clamp(-0.8, 0.8)).clamp((1e-4_f64).ln(), (0.45_f64).ln()),
                (x[1] + step[1].clamp(-0.8, 0.8)).clamp((5.0_f64).ln(), (30_000.0_f64).ln()),
                (x[2] + step[2].clamp(-3.0, 3.0)).clamp(-30.0, 30.0),
            ];
            let (c2_new, r1_new, r20_new, r_new) = eval_residuals(cand);
            if c2_new < best_c2 {
                let rel = (best_c2 - c2_new) / best_c2.max(1.0);
                x = cand;
                best_c2 = c2_new;
                best_r1 = r1_new;
                best_r20 = r20_new;
                r_cur = r_new;
                lambda = (lambda * 0.3).max(1e-7);
                accepted = true;
                if rel < 1e-9 {
                    return Ok(R1RhoFitResult {
                        p_b: x[0].exp(),
                        k_ex_s_inv: x[1].exp(),
                        dw_ab_ppm: x[2],
                        r1_s_inv: best_r1,
                        r20_s_inv: best_r20,
                        chi2: best_c2,
                    });
                }
                break;
            }
            lambda *= 5.0;
        }
        if !accepted && lambda > 1e8 {
            break;
        }
    }

    Ok(R1RhoFitResult {
        p_b: x[0].exp(),
        k_ex_s_inv: x[1].exp(),
        dw_ab_ppm: x[2],
        r1_s_inv: best_r1,
        r20_s_inv: best_r20,
        chi2: best_c2,
    })
}

// ============================================================================
// 3. Statistical Exchange Gate (`Flat` Null vs Dispersion: AIC + Nested F-test)
// ============================================================================

#[inline]
fn hurvich_tsai_aicc(chi2: f64, n: usize, k: usize) -> f64 {
    if n <= k + 1 {
        f64::INFINITY
    } else {
        let kf = k as f64;
        let nf = n as f64;
        chi2 + 2.0 * kf + (2.0 * kf * (kf + 1.0)) / (nf - kf - 1.0)
    }
}

/// Evaluate the statistical exchange gate on all static-field series of a single residue.
pub fn evaluate_exchange_gate(
    residue_series: &[&CpmgSeriesDataset],
    gate_cfg: &ExchangeGateConfig,
) -> ExchangeGateVerdict {
    let Some(&first) = residue_series.first() else {
        return ExchangeGateVerdict {
            seq_id: 0,
            res_name: String::new(),
            significant_exchange: false,
            observed_rex_s_inv: 0.0,
            flat_chi2: 0.0,
            ex_chi2: 0.0,
            delta_aic: f64::NEG_INFINITY,
            f_stat: 0.0,
            f_pvalue: 1.0,
        };
    };

    for s in residue_series {
        if s.r2_eff_s_inv.is_empty()
            || s.r2_eff_s_inv.len() != s.r2_eff_err.len()
            || s.nu_cpmg_hz.len() != s.r2_eff_s_inv.len()
        {
            return ExchangeGateVerdict {
                seq_id: first.seq_id,
                res_name: first.res_name.clone(),
                significant_exchange: false,
                observed_rex_s_inv: 0.0,
                flat_chi2: 0.0,
                ex_chi2: 0.0,
                delta_aic: f64::NEG_INFINITY,
                f_stat: 0.0,
                f_pvalue: 1.0,
            };
        }
    }

    let n_fields = residue_series.len();
    let mut total_pts = 0usize;
    let mut flat_chi2 = 0.0f64;
    let mut max_obs_rex = 0.0f64;

    for s in residue_series {
        total_pts += s.r2_eff_s_inv.len();
        let mut w_sum = 0.0;
        let mut wy_sum = 0.0;
        let mut r_min = f64::INFINITY;
        let mut r_max = f64::NEG_INFINITY;
        for (&r, &e) in s.r2_eff_s_inv.iter().zip(s.r2_eff_err.iter()) {
            let w = 1.0 / e.max(1e-3).powi(2);
            w_sum += w;
            wy_sum += w * r;
            r_min = r_min.min(r);
            r_max = r_max.max(r);
        }
        if r_max > r_min {
            max_obs_rex = max_obs_rex.max(r_max - r_min);
        }
        let mean_r = wy_sum / w_sum.max(1e-12);
        for (&r, &e) in s.r2_eff_s_inv.iter().zip(s.r2_eff_err.iter()) {
            let z = (r - mean_r) / e.max(1e-3);
            flat_chi2 += z * z;
        }
    }

    // Fit fast-exchange LM63 (`k = n_fields + 2` params: per-field R20 + k_ex + phi_ex_ppm2)
    // where `phi_ex_rad2(field) = phi_ex_ppm2 * (2*pi*larmor_mhz)^2`!
    // Notice for fixed `k_ex`, both `R20_f` AND `phi_ex_ppm2` are 100% LINEAR parameters!!
    let mut best_ex_chi2 = flat_chi2;
    for ik in 0..80 {
        let k_ex = 20.0 * (8000.0_f64 / 20.0).powf((ik as f64) / 79.0);
        if let Some(c2) = fit_lm63_linear_profile_chi2(residue_series, k_ex) {
            if c2 < best_ex_chi2 {
                best_ex_chi2 = c2;
            }
        }
    }

    let k_flat = n_fields;
    let k_ex_params = n_fields + 2;
    let aic_flat = hurvich_tsai_aicc(flat_chi2, total_pts, k_flat);
    let aic_ex = hurvich_tsai_aicc(best_ex_chi2, total_pts, k_ex_params);
    let delta_aic = if aic_ex.is_finite() && aic_flat.is_finite() {
        aic_flat - aic_ex
    } else if !aic_ex.is_finite() && aic_flat.is_finite() {
        f64::NEG_INFINITY
    } else {
        0.0
    };

    let df1 = (k_ex_params - k_flat) as f64;
    let df2 = total_pts.saturating_sub(k_ex_params) as f64;
    let (f_stat, f_pvalue) = if total_pts > k_ex_params && flat_chi2 > best_ex_chi2 {
        let num = (flat_chi2 - best_ex_chi2) / df1;
        let den = (best_ex_chi2 / df2).max(1e-15);
        let f = num / den;
        let p = f_distribution_sf(f, df1, df2);
        (f, p)
    } else {
        (0.0, 1.0)
    };

    let significant = total_pts > k_ex_params + 1
        && max_obs_rex >= gate_cfg.min_delta_r2_eff_s_inv
        && delta_aic.is_finite()
        && delta_aic >= gate_cfg.min_delta_aic
        && f_pvalue <= gate_cfg.max_f_pvalue;

    ExchangeGateVerdict {
        seq_id: first.seq_id,
        res_name: first.res_name.clone(),
        significant_exchange: significant,
        observed_rex_s_inv: max_obs_rex,
        flat_chi2,
        ex_chi2: best_ex_chi2,
        delta_aic,
        f_stat,
        f_pvalue,
    }
}

/// Exact profiled weighted least squares for `LM63` at fixed `k_ex` (`phi_ex_ppm2 >= 0` + per-field `R20_f`).
fn fit_lm63_linear_profile_chi2(series: &[&CpmgSeriesDataset], k_ex: f64) -> Option<f64> {
    // Eliminate each field's intercept `R20_f` by centering `y` and `g_f(nu)` with weights `w`!
    let mut s_gg = 0.0f64;
    let mut s_gy = 0.0f64;
    let mut per_field_data = Vec::with_capacity(series.len());

    for s in series {
        if s.nu_cpmg_hz.is_empty()
            || s.nu_cpmg_hz.len() != s.r2_eff_s_inv.len()
            || s.nu_cpmg_hz.len() != s.r2_eff_err.len()
        {
            return None;
        }
        let scale2 = (TAU * s.larmor_mhz).powi(2);
        let mut w_sum = 0.0;
        let mut wg_sum = 0.0;
        let mut wy_sum = 0.0;
        let mut gs = Vec::with_capacity(s.nu_cpmg_hz.len());
        let mut ws = Vec::with_capacity(s.nu_cpmg_hz.len());
        for i in 0..s.nu_cpmg_hz.len() {
            let x = k_ex / (4.0 * s.nu_cpmg_hz[i]);
            let g = (scale2 / k_ex) * (1.0 - x.tanh() / x.max(1e-12));
            let w = 1.0 / s.r2_eff_err[i].max(1e-3).powi(2);
            w_sum += w;
            wg_sum += w * g;
            wy_sum += w * s.r2_eff_s_inv[i];
            gs.push(g);
            ws.push(w);
        }
        let g_bar = wg_sum / w_sum.max(1e-15);
        let y_bar = wy_sum / w_sum.max(1e-15);
        for i in 0..gs.len() {
            let dg = gs[i] - g_bar;
            let dy = s.r2_eff_s_inv[i] - y_bar;
            s_gg += ws[i] * dg * dg;
            s_gy += ws[i] * dg * dy;
        }
        per_field_data.push((gs, ws, g_bar, y_bar));
    }

    let phi_ppm2 = if s_gg > 1e-18 {
        (s_gy / s_gg).max(0.0)
    } else {
        0.0
    };

    let mut chi2 = 0.0;
    for (s, (gs, ws, g_bar, y_bar)) in series.iter().zip(per_field_data.iter()) {
        let r20 = y_bar - phi_ppm2 * g_bar;
        for i in 0..gs.len() {
            let pred = r20 + phi_ppm2 * gs[i];
            chi2 += ws[i] * (s.r2_eff_s_inv[i] - pred).powi(2);
        }
    }
    Some(chi2)
}

/// Upper-tail survival function `P(F_{d1, d2} >= f)` via regularized incomplete beta `I_x(d2/2, d1/2)`.
pub fn f_distribution_sf(f: f64, d1: f64, d2: f64) -> f64 {
    if !f.is_finite() || f <= 0.0 {
        return 1.0;
    }
    let x = d2 / (d2 + d1 * f);
    regularized_inc_beta(x.clamp(0.0, 1.0), 0.5 * d2, 0.5 * d1)
}

fn regularized_inc_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let ln_beta = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b);
    let front = (a * x.ln() + b * (1.0 - x).ln() - ln_beta).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        front * beta_cf(x, a, b) / a
    } else {
        1.0 - front * beta_cf(1.0 - x, b, a) / b
    }
}

fn beta_cf(x: f64, a: f64, b: f64) -> f64 {
    let fpmin = 1e-30;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < fpmin {
        d = fpmin;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=200 {
        let mf = m as f64;
        let m2 = 2.0 * mf;
        let aa1 = mf * (b - mf) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa1 * d;
        if d.abs() < fpmin {
            d = fpmin;
        }
        c = 1.0 + aa1 / c;
        if c.abs() < fpmin {
            c = fpmin;
        }
        d = 1.0 / d;
        h *= d * c;

        let aa2 = -(a + mf) * (qab + mf) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa2 * d;
        if d.abs() < fpmin {
            d = fpmin;
        }
        c = 1.0 + aa2 / c;
        if c.abs() < fpmin {
            c = fpmin;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-14 {
            break;
        }
    }
    h
}

/// Lanczos `g = 7` `ln(Gamma(z))` accurate to `< 1e-15` for `z > 0`.
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
    let zm = z - 1.0;
    let mut x = COEFFS[0];
    for (i, &c) in COEFFS.iter().enumerate().skip(1) {
        x += c / (zm + (i as f64));
    }
    let t = zm + 7.5;
    0.5 * TAU.ln() + (zm + 0.5) * t.ln() - t + x.ln()
}

// ============================================================================
// 4. Multi-Field Global CPMG Grid + Schur-Complement Levenberg–Marquardt Fitter
// ============================================================================

/// Result of profiling `|dw_ppm|` and per-field `R20` for one `(grid_idx, residue_idx)` pair.
#[derive(Clone, Copy, Debug)]
pub struct CpmgGridCellResult {
    pub best_chi2: f64,
    pub best_dw_ppm: f64,
}

/// Profile optimal `(dw_ppm, R20_fields)` on CPU for one residue at fixed `(p_B, k_ex)`.
pub fn profile_residue_dw_cpu(
    res_series: &[&CpmgSeriesDataset],
    model: CpmgModel,
    p_b: f64,
    k_ex_s_inv: f64,
) -> (f64, f64, Vec<f64>) {
    let eval_dw = |dw_ppm: f64| -> (f64, Vec<f64>) {
        let mut total_c2 = 0.0;
        let mut r20s = Vec::with_capacity(res_series.len());
        for s in res_series {
            let dw_rad = TAU * s.larmor_mhz * dw_ppm;
            // Notice R20 enters strictly additively (`R2,eff(R20, ...) = R20 + R_ex(0, ...)` when dR2 = 0)!
            let mut w_sum = 0.0;
            let mut w_diff = 0.0;
            let mut rex_vals = Vec::with_capacity(s.nu_cpmg_hz.len());
            for i in 0..s.nu_cpmg_hz.len() {
                let rex_i = evaluate_cpmg_model(
                    model,
                    0.0,
                    0.0,
                    p_b,
                    k_ex_s_inv,
                    dw_rad,
                    s.t_relax_s,
                    s.nu_cpmg_hz[i],
                );
                let w = 1.0 / s.r2_eff_err[i].max(1e-3).powi(2);
                w_sum += w;
                w_diff += w * (s.r2_eff_s_inv[i] - rex_i);
                rex_vals.push(rex_i);
            }
            let r20 = (w_diff / w_sum.max(1e-15)).max(0.0);
            for i in 0..s.nu_cpmg_hz.len() {
                let pred = r20 + rex_vals[i];
                let z = (s.r2_eff_s_inv[i] - pred) / s.r2_eff_err[i].max(1e-3);
                total_c2 += z * z;
            }
            r20s.push(r20);
        }
        (total_c2, r20s)
    };

    // Coarse sweep over `0.15 .. 12.0 ppm` + golden-section refinement
    let mut best_dw = 1.0;
    let mut best_c2 = f64::INFINITY;
    for k in 1..=36 {
        let dw = 0.25 * (k as f64);
        let (c2, _) = eval_dw(dw);
        if c2 < best_c2 {
            best_c2 = c2;
            best_dw = dw;
        }
    }

    let mut lo = (best_dw - 0.35).max(0.02);
    let mut hi = (best_dw + 0.35).min(20.0);
    let phi = 0.5 * (5.0_f64.sqrt() - 1.0);
    let mut x1 = hi - phi * (hi - lo);
    let mut x2 = lo + phi * (hi - lo);
    let (mut f1, _) = eval_dw(x1);
    let (mut f2, _) = eval_dw(x2);
    for _ in 0..18 {
        if f1 < f2 {
            hi = x2;
            x2 = x1;
            f2 = f1;
            x1 = hi - phi * (hi - lo);
            (f1, _) = eval_dw(x1);
        } else {
            lo = x1;
            x1 = x2;
            f1 = f2;
            x2 = lo + phi * (hi - lo);
            (f2, _) = eval_dw(x2);
        }
    }
    let opt_dw = 0.5 * (lo + hi);
    let (opt_c2, opt_r20s) = eval_dw(opt_dw);
    (opt_c2, opt_dw, opt_r20s)
}

/// Multi-field global `(p_B, k_ex)` + per-residue `(|dw_ppm|, R20_fields)` `CPMG` solver on CPU.
pub fn fit_cpmg_global_cpu(
    series: &[CpmgSeriesDataset],
    config: &CpmgFitConfig,
) -> Result<CpmgGlobalFitResult, String> {
    fit_cpmg_global_with_grid(series, config, |res_groups, pb_grid, kex_grid, model| {
        let n_grid = pb_grid.len() * kex_grid.len();
        let n_res = res_groups.len();
        let mut out = Vec::with_capacity(n_grid * n_res);
        for &pb in pb_grid {
            for &kex in kex_grid {
                for r_series in res_groups {
                    let (c2, dw, _) = profile_residue_dw_cpu(r_series, model, pb, kex);
                    out.push(CpmgGridCellResult {
                        best_chi2: c2,
                        best_dw_ppm: dw,
                    });
                }
            }
        }
        Ok(out)
    })
}

/// Generic multi-field global `CPMG` solver accepting a pluggable `(p_B, k_ex)` grid evaluator
/// (CPU or WebGPU `cpmg_fit.wgsl`) followed by exact `f64` Schur-complement Levenberg–Marquardt
/// polish and analytical covariance propagation.
pub fn fit_cpmg_global_with_grid<FGrid>(
    series: &[CpmgSeriesDataset],
    config: &CpmgFitConfig,
    mut eval_grid: FGrid,
) -> Result<CpmgGlobalFitResult, String>
where
    FGrid: FnMut(
        &[Vec<&CpmgSeriesDataset>],
        &[f64],
        &[f64],
        CpmgModel,
    ) -> Result<Vec<CpmgGridCellResult>, String>,
{
    if series.is_empty() {
        return Err("fit_cpmg_global requires at least one CPMG series".to_string());
    }
    for (i, s) in series.iter().enumerate() {
        if s.nu_cpmg_hz.is_empty()
            || s.nu_cpmg_hz.len() != s.r2_eff_s_inv.len()
            || s.nu_cpmg_hz.len() != s.r2_eff_err.len()
        {
            return Err(format!(
                "CPMG series {i} has empty or mismatched array lengths"
            ));
        }
    }

    // Group series by residue `seq_id` (each residue may have 1, 2, or 3 static fields)
    let mut by_res: BTreeMap<i32, Vec<&CpmgSeriesDataset>> = BTreeMap::new();
    for s in series {
        by_res.entry(s.seq_id).or_default().push(s);
    }
    for list in by_res.values_mut() {
        list.sort_by(|a, b| a.larmor_mhz.total_cmp(&b.larmor_mhz));
    }

    let mut all_groups = Vec::new();
    let mut gates = Vec::new();
    for (_, list) in by_res {
        let gate = evaluate_exchange_gate(&list, &config.gate);
        if !config.filter_by_exchange_gate || gate.significant_exchange {
            all_groups.push(list);
            gates.push(gate);
        }
    }
    if all_groups.is_empty() {
        return Err("no residues passed the statistical chemical exchange gate".to_string());
    }

    let n_res = all_groups.len();

    if config.model == CpmgModel::Flat {
        let mut residues = Vec::with_capacity(n_res);
        let mut total_chi2 = 0.0;
        let mut total_pts = 0usize;
        let mut n_params = 0usize;
        for (ri, r_list) in all_groups.iter().enumerate() {
            let mut field_r20 = Vec::with_capacity(r_list.len());
            let mut r_chi2 = 0.0;
            let mut r_pts = 0usize;
            for s in r_list {
                let mut w_sum = 0.0;
                let mut wy_sum = 0.0;
                for (&y, &e) in s.r2_eff_s_inv.iter().zip(s.r2_eff_err.iter()) {
                    let w = 1.0 / e.max(1e-3).powi(2);
                    w_sum += w;
                    wy_sum += w * y;
                }
                let r20 = wy_sum / w_sum.max(1e-12);
                let r20_err = (1.0 / w_sum.max(1e-12)).sqrt();
                for (&y, &e) in s.r2_eff_s_inv.iter().zip(s.r2_eff_err.iter()) {
                    r_chi2 += ((y - r20) / e.max(1e-3)).powi(2);
                }
                r_pts += s.nu_cpmg_hz.len();
                n_params += 1;
                field_r20.push((s.larmor_mhz, r20, r20_err));
            }
            total_chi2 += r_chi2;
            total_pts += r_pts;
            residues.push(CpmgResidueFit {
                seq_id: r_list[0].seq_id,
                res_name: r_list[0].res_name.clone(),
                dw_ppm: 0.0,
                dw_ppm_err: 0.0,
                field_r20,
                alpha_rex_scaling: None,
                gate: gates[ri].clone(),
                chi2: r_chi2,
                n_points: r_pts,
            });
        }
        let dof = total_pts.saturating_sub(n_params);
        return Ok(CpmgGlobalFitResult {
            model: CpmgModel::Flat,
            p_b: 0.0,
            p_b_err: 0.0,
            k_ex_s_inv: 0.0,
            k_ex_err: 0.0,
            k_ab_s_inv: 0.0,
            k_ba_s_inv: 0.0,
            total_chi2,
            reduced_chi2: total_chi2 / (dof.max(1) as f64),
            dof,
            iterations: 1,
            converged: true,
            residues,
        });
    }

    // 1. Global `(p_B, k_ex)` log-grid evaluation (dispatched on GPU or CPU)
    let (npb, nkex) = (config.grid_dims.0.max(8), config.grid_dims.1.max(8));
    let pb_grid: Vec<f64> = (0..npb)
        .map(|i| 0.0025 * (0.30_f64 / 0.0025).powf((i as f64) / ((npb - 1) as f64)))
        .collect();
    let kex_grid: Vec<f64> = (0..nkex)
        .map(|i| 25.0 * (7500.0_f64 / 25.0).powf((i as f64) / ((nkex - 1) as f64)))
        .collect();

    let grid_model = match config.model {
        CpmgModel::Numerical2x2 => CpmgModel::Baldwin2014,
        other => other,
    };
    let grid_cells = eval_grid(&all_groups, &pb_grid, &kex_grid, grid_model)?;
    if grid_cells.len() != npb * nkex * n_res {
        return Err("invalid grid cell output length in fit_cpmg_global".to_string());
    }

    let mut best_g_idx = 0usize;
    let mut best_grid_chi2 = f64::INFINITY;
    for g_idx in 0..(npb * nkex) {
        let mut sum_c2 = 0.0;
        for ri in 0..n_res {
            sum_c2 += grid_cells[g_idx * n_res + ri].best_chi2;
        }
        if sum_c2 < best_grid_chi2 {
            best_grid_chi2 = sum_c2;
            best_g_idx = g_idx;
        }
    }

    let init_pb = pb_grid[best_g_idx / nkex];
    let init_kex = kex_grid[best_g_idx % nkex];

    // Seed per-residue `(dw_ppm, field_r20s)` from exact CPU profile at `(init_pb, init_kex)`
    let mut ln_pb = cpmg_logit_pb(init_pb);
    let mut ln_kex = init_kex.ln();
    let mut res_dw: Vec<f64> = Vec::with_capacity(n_res);
    let mut res_r20: Vec<Vec<f64>> = Vec::with_capacity(n_res);
    for r_list in &all_groups {
        let (_, dw, r20s) = profile_residue_dw_cpu(r_list, config.model, init_pb, init_kex);
        res_dw.push(dw);
        res_r20.push(r20s);
    }

    // 2. Schur-complement Levenberg–Marquardt global polish on `(logit_pb, ln_kex)` + per-residue
    // `(dw_ppm, R20_0, R20_1, ...)`
    let eval_all =
        |z_pb: f64, z_kex: f64, dws: &[f64], r20s: &[Vec<f64>]| -> (Vec<Vec<Vec<f64>>>, f64) {
            let pb = cpmg_inv_logit_pb(z_pb);
            let kex = z_kex.exp().clamp(1.0, 50_000.0);
            let mut all_preds = Vec::with_capacity(n_res);
            let mut c2 = 0.0;
            for (ri, r_list) in all_groups.iter().enumerate() {
                let mut f_preds = Vec::with_capacity(r_list.len());
                for (fi, s) in r_list.iter().enumerate() {
                    let dw_rad = TAU * s.larmor_mhz * dws[ri].abs();
                    let r20 = r20s[ri][fi];
                    let mut preds = Vec::with_capacity(s.nu_cpmg_hz.len());
                    for (pi, &nu) in s.nu_cpmg_hz.iter().enumerate() {
                        let p = evaluate_cpmg_model(
                            config.model,
                            r20,
                            r20,
                            pb,
                            kex,
                            dw_rad,
                            s.t_relax_s,
                            nu,
                        );
                        let z = (s.r2_eff_s_inv[pi] - p) / s.r2_eff_err[pi].max(1e-3);
                        c2 += z * z;
                        preds.push(p);
                    }
                    f_preds.push(preds);
                }
                all_preds.push(f_preds);
            }
            (all_preds, c2)
        };

    let (mut base_preds, mut current_chi2) = eval_all(ln_pb, ln_kex, &res_dw, &res_r20);
    let mut lambda = 1e-3f64;
    let mut iterations = 0usize;
    let mut converged = false;
    let h = 1e-5f64;

    for iter in 0..config.max_lm_iters {
        iterations = iter + 1;
        let pb0 = cpmg_inv_logit_pb(ln_pb);
        let kex0 = ln_kex.exp();
        let pb_p = cpmg_inv_logit_pb(ln_pb + h);
        let kex_p = (ln_kex + h).exp();

        let mut h_gg = [[0.0f64; 2]; 2];
        let mut g_g = [0.0f64; 2];
        let mut h_ll = Vec::with_capacity(n_res);
        let mut h_gl = Vec::with_capacity(n_res);
        let mut g_l = Vec::with_capacity(n_res);

        for (ri, r_list) in all_groups.iter().enumerate() {
            let n_l = 1 + r_list.len(); // [dw_ppm, R20_0, R20_1, ...]
            let mut h_rr = vec![vec![0.0f64; n_l]; n_l];
            let mut h_rg = vec![[0.0f64; 2]; n_l];
            let mut g_r = vec![0.0f64; n_l];

            for (fi, s) in r_list.iter().enumerate() {
                let dw0_rad = TAU * s.larmor_mhz * res_dw[ri].abs();
                let dwp_rad = TAU * s.larmor_mhz * (res_dw[ri].abs() + h);
                let r20 = res_r20[ri][fi];

                for (pi, &nu) in s.nu_cpmg_hz.iter().enumerate() {
                    let b_val = base_preds[ri][fi][pi];
                    let inv_sig = 1.0 / s.r2_eff_err[pi].max(1e-3);
                    let res_p = (s.r2_eff_s_inv[pi] - b_val) * inv_sig;

                    let f_pb = evaluate_cpmg_model(
                        config.model,
                        r20,
                        r20,
                        pb_p,
                        kex0,
                        dw0_rad,
                        s.t_relax_s,
                        nu,
                    );
                    let f_kex = evaluate_cpmg_model(
                        config.model,
                        r20,
                        r20,
                        pb0,
                        kex_p,
                        dw0_rad,
                        s.t_relax_s,
                        nu,
                    );
                    let f_dw = evaluate_cpmg_model(
                        config.model,
                        r20,
                        r20,
                        pb0,
                        kex0,
                        dwp_rad,
                        s.t_relax_s,
                        nu,
                    );

                    let jg = [(f_pb - b_val) * inv_sig / h, (f_kex - b_val) * inv_sig / h];
                    let mut jl = vec![0.0f64; n_l];
                    jl[0] = (f_dw - b_val) * inv_sig / h;
                    jl[1 + fi] = inv_sig; // exact d(R2,eff)/d(R20) == 1 when dR2 = 0!

                    for a in 0..2 {
                        g_g[a] += jg[a] * res_p;
                        for b in 0..2 {
                            h_gg[a][b] += jg[a] * jg[b];
                        }
                    }
                    for a in 0..n_l {
                        g_r[a] += jl[a] * res_p;
                        for gc in 0..2 {
                            h_rg[a][gc] += jl[a] * jg[gc];
                        }
                        for b in 0..n_l {
                            h_rr[a][b] += jl[a] * jl[b];
                        }
                    }
                }
            }
            h_ll.push(h_rr);
            h_gl.push(h_rg);
            g_l.push(g_r);
        }

        let max_grad = g_g
            .iter()
            .copied()
            .chain(g_l.iter().flat_map(|v| v.iter().copied()))
            .fold(0.0f64, |m, v| m.max(v.abs()));
        if max_grad < 1e-8 {
            converged = true;
            break;
        }

        let mut accepted = false;
        for _sub in 0..8 {
            if let Ok((dg, dl)) = solve_cpmg_arrowhead(&h_gg, &g_g, &h_ll, &h_gl, &g_l, lambda) {
                let t_pb = (ln_pb + dg[0]).clamp(-11.0, -0.02);
                let t_kex = (ln_kex + dg[1]).clamp(0.5, 10.8);
                let mut t_dw = res_dw.clone();
                let mut t_r20 = res_r20.clone();
                let mut step_max = dg[0].hypot(dg[1]);
                for ri in 0..n_res {
                    t_dw[ri] = (t_dw[ri] + dl[ri][0]).abs().clamp(0.005, 30.0);
                    step_max = step_max.max(dl[ri][0].abs());
                    for fi in 0..t_r20[ri].len() {
                        t_r20[ri][fi] = (t_r20[ri][fi] + dl[ri][1 + fi]).clamp(0.0, 250.0);
                        step_max = step_max.max(dl[ri][1 + fi].abs());
                    }
                }
                let (t_preds, t_chi2) = eval_all(t_pb, t_kex, &t_dw, &t_r20);
                if t_chi2 < current_chi2 {
                    let rel = (current_chi2 - t_chi2) / current_chi2.max(1.0);
                    ln_pb = t_pb;
                    ln_kex = t_kex;
                    res_dw = t_dw;
                    res_r20 = t_r20;
                    base_preds = t_preds;
                    current_chi2 = t_chi2;
                    lambda = (lambda * 0.3).max(1e-10);
                    accepted = true;
                    if step_max < 1e-8 || rel < 1e-11 {
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

    // 3. Analytical covariance via Schur complement at optimum
    let (cov_g, cov_locals) = compute_cpmg_covariance(
        &all_groups,
        config.model,
        ln_pb,
        ln_kex,
        &res_dw,
        &res_r20,
        &base_preds,
    );

    let p_b = cpmg_inv_logit_pb(ln_pb);
    let p_b_err = cov_g[0][0].max(0.0).sqrt() * cpmg_d_inv_logit_pb(ln_pb);
    let k_ex_s_inv = ln_kex.exp();
    let k_ex_err = cov_g[1][1].max(0.0).sqrt() * k_ex_s_inv;

    let mut total_pts = 0usize;
    let mut n_params = 2usize;
    let mut residues = Vec::with_capacity(n_res);
    for (ri, r_list) in all_groups.iter().enumerate() {
        n_params += 1 + r_list.len();
        let mut r_chi2 = 0.0;
        let mut r_pts = 0usize;
        let mut field_r20 = Vec::with_capacity(r_list.len());
        for (fi, s) in r_list.iter().enumerate() {
            r_pts += s.nu_cpmg_hz.len();
            for pi in 0..s.nu_cpmg_hz.len() {
                let z = (s.r2_eff_s_inv[pi] - base_preds[ri][fi][pi]) / s.r2_eff_err[pi].max(1e-3);
                r_chi2 += z * z;
            }
            let r20_se = cov_locals[ri][1 + fi].max(0.0).sqrt();
            field_r20.push((s.larmor_mhz, res_r20[ri][fi], r20_se));
        }
        total_pts += r_pts;

        // Millet et al. (2000) alpha exchange timescale scaling parameter when >= 2 fields
        let alpha_rex_scaling = if r_list.len() >= 2 {
            let s_lo = r_list[0];
            let s_hi = r_list[r_list.len() - 1];
            let rex_lo = (s_lo.r2_eff_s_inv[0] - res_r20[ri][0]).max(1e-4);
            let rex_hi = (s_hi.r2_eff_s_inv[0] - res_r20[ri][r_list.len() - 1]).max(1e-4);
            let b_sum = s_hi.larmor_mhz + s_lo.larmor_mhz;
            let b_diff = (s_hi.larmor_mhz - s_lo.larmor_mhz).max(1e-6);
            let r_sum = rex_hi + rex_lo;
            let r_diff = rex_hi - rex_lo;
            Some(((b_sum / b_diff) * (r_diff / r_sum)).clamp(0.0, 2.2))
        } else {
            None
        };

        residues.push(CpmgResidueFit {
            seq_id: r_list[0].seq_id,
            res_name: r_list[0].res_name.clone(),
            dw_ppm: res_dw[ri],
            dw_ppm_err: cov_locals[ri][0].max(0.0).sqrt(),
            field_r20,
            alpha_rex_scaling,
            gate: gates[ri].clone(),
            chi2: r_chi2,
            n_points: r_pts,
        });
    }

    let dof = total_pts.saturating_sub(n_params);
    Ok(CpmgGlobalFitResult {
        model: config.model,
        p_b,
        p_b_err,
        k_ex_s_inv,
        k_ex_err,
        k_ab_s_inv: p_b * k_ex_s_inv,
        k_ba_s_inv: (1.0 - p_b) * k_ex_s_inv,
        total_chi2: current_chi2,
        reduced_chi2: current_chi2 / (dof.max(1) as f64),
        dof,
        iterations,
        converged,
        residues,
    })
}

#[inline]
fn cpmg_logit_pb(pb: f64) -> f64 {
    let p = pb.clamp(1e-5, 0.495) / 0.50;
    (p / (1.0 - p)).ln()
}

#[inline]
fn cpmg_inv_logit_pb(z: f64) -> f64 {
    0.50 / (1.0 + (-z).exp())
}

#[inline]
fn cpmg_d_inv_logit_pb(z: f64) -> f64 {
    let s = 1.0 / (1.0 + (-z).exp());
    0.50 * s * (1.0 - s)
}

fn solve_cpmg_arrowhead(
    h_gg: &[[f64; 2]; 2],
    g_g: &[f64; 2],
    h_ll: &[Vec<Vec<f64>>],
    h_gl: &[Vec<[f64; 2]>],
    g_l: &[Vec<f64>],
    lambda: f64,
) -> Result<([f64; 2], Vec<Vec<f64>>), String> {
    let n_res = h_ll.len();
    let mut s_gg = *h_gg;
    for a in 0..2 {
        s_gg[a][a] += lambda * h_gg[a][a].abs().max(1.0);
    }
    let mut rhs_g = *g_g;
    let mut saved = Vec::with_capacity(n_res);

    for ri in 0..n_res {
        let n_l = h_ll[ri].len();
        let mut blk = h_ll[ri].clone();
        for c in 0..n_l {
            blk[c][c] += lambda * blk[c][c].abs().max(1.0);
        }
        let inv_b = invert_sym_cpmg(&blk)?;
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
        saved.push((w, u));
    }

    let det = s_gg[0][0] * s_gg[1][1] - s_gg[0][1] * s_gg[1][0];
    if !det.is_finite() || det.abs() < 1e-24 {
        return Err("singular global 2x2 Schur complement in CPMG LM".to_string());
    }
    let dg = [
        (s_gg[1][1] * rhs_g[0] - s_gg[0][1] * rhs_g[1]) / det,
        (-s_gg[1][0] * rhs_g[0] + s_gg[0][0] * rhs_g[1]) / det,
    ];

    let mut dl = Vec::with_capacity(n_res);
    for (w, u) in &saved {
        let mut v = vec![0.0f64; u.len()];
        for c in 0..u.len() {
            v[c] = u[c] - (w[c][0] * dg[0] + w[c][1] * dg[1]);
        }
        dl.push(v);
    }
    Ok((dg, dl))
}

fn compute_cpmg_covariance(
    all_groups: &[Vec<&CpmgSeriesDataset>],
    model: CpmgModel,
    ln_pb: f64,
    ln_kex: f64,
    res_dw: &[f64],
    res_r20: &[Vec<f64>],
    base_preds: &[Vec<Vec<f64>>],
) -> ([[f64; 2]; 2], Vec<Vec<f64>>) {
    let n_res = all_groups.len();
    let h = 1e-5f64;
    let pb0 = cpmg_inv_logit_pb(ln_pb);
    let kex0 = ln_kex.exp();
    let pb_p = cpmg_inv_logit_pb(ln_pb + h);
    let kex_p = (ln_kex + h).exp();

    let mut h_gg = [[0.0f64; 2]; 2];
    let mut s_gg_sub = [[0.0f64; 2]; 2];
    let mut blocks = Vec::with_capacity(n_res);

    for (ri, r_list) in all_groups.iter().enumerate() {
        let n_l = 1 + r_list.len();
        let mut h_rr = vec![vec![0.0f64; n_l]; n_l];
        let mut h_rg = vec![[0.0f64; 2]; n_l];

        for (fi, s) in r_list.iter().enumerate() {
            let dw0 = TAU * s.larmor_mhz * res_dw[ri].abs();
            let dwp = TAU * s.larmor_mhz * (res_dw[ri].abs() + h);
            let r20 = res_r20[ri][fi];
            for (pi, &nu) in s.nu_cpmg_hz.iter().enumerate() {
                let b_val = base_preds[ri][fi][pi];
                let inv_sig = 1.0 / s.r2_eff_err[pi].max(1e-3);
                let f_pb = evaluate_cpmg_model(model, r20, r20, pb_p, kex0, dw0, s.t_relax_s, nu);
                let f_kex = evaluate_cpmg_model(model, r20, r20, pb0, kex_p, dw0, s.t_relax_s, nu);
                let f_dw = evaluate_cpmg_model(model, r20, r20, pb0, kex0, dwp, s.t_relax_s, nu);
                let jg = [(f_pb - b_val) * inv_sig / h, (f_kex - b_val) * inv_sig / h];
                let mut jl = vec![0.0f64; n_l];
                jl[0] = (f_dw - b_val) * inv_sig / h;
                jl[1 + fi] = inv_sig;

                for a in 0..2 {
                    for b in 0..2 {
                        h_gg[a][b] += jg[a] * jg[b];
                    }
                }
                for a in 0..n_l {
                    for gc in 0..2 {
                        h_rg[a][gc] += jl[a] * jg[gc];
                    }
                    for b in 0..n_l {
                        h_rr[a][b] += jl[a] * jl[b];
                    }
                }
            }
        }
        for c in 0..n_l {
            h_rr[c][c] += 1e-10;
        }
        let inv_b = invert_sym_cpmg(&h_rr).unwrap_or_else(|_| {
            let mut d = vec![vec![0.0; n_l]; n_l];
            for c in 0..n_l {
                d[c][c] = 1.0 / h_rr[c][c].max(1e-6);
            }
            d
        });
        let mut w = vec![[0.0f64; 2]; n_l];
        for c in 0..n_l {
            for k in 0..n_l {
                w[c][0] += inv_b[c][k] * h_rg[k][0];
                w[c][1] += inv_b[c][k] * h_rg[k][1];
            }
        }
        for ga in 0..2 {
            for c in 0..n_l {
                for gb in 0..2 {
                    s_gg_sub[ga][gb] += h_rg[c][ga] * w[c][gb];
                }
            }
        }
        blocks.push((inv_b, w));
    }

    let s00 = h_gg[0][0] - s_gg_sub[0][0] + 1e-10;
    let s01 = h_gg[0][1] - s_gg_sub[0][1];
    let s10 = h_gg[1][0] - s_gg_sub[1][0];
    let s11 = h_gg[1][1] - s_gg_sub[1][1] + 1e-10;
    let det = (s00 * s11 - s01 * s10).max(1e-18);
    let cov_g = [[s11 / det, -s01 / det], [-s10 / det, s00 / det]];

    let mut cov_l = Vec::with_capacity(n_res);
    for (inv_b, w) in &blocks {
        let mut diag = vec![0.0f64; w.len()];
        for c in 0..w.len() {
            let extra = w[c][0] * (cov_g[0][0] * w[c][0] + cov_g[0][1] * w[c][1])
                + w[c][1] * (cov_g[1][0] * w[c][0] + cov_g[1][1] * w[c][1]);
            diag[c] = (inv_b[c][c] + extra).max(0.0);
        }
        cov_l.push(diag);
    }
    (cov_g, cov_l)
}

fn invert_sym_cpmg(a: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
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
            return Err("singular block in CPMG Schur solve".to_string());
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

// ============================================================================
// 5. TSV & JSON Serialization / Deserialization
// ============================================================================

/// Serialize [`CpmgSeriesDataset`] entries to TSV.
pub fn write_cpmg_series_tsv(series: &[CpmgSeriesDataset]) -> String {
    let mut out = String::from(
        "seq_id\tres_name\tlarmor_mhz\tt_relax_s\tnu_cpmg_hz\tr2_eff_s_inv\tr2_eff_err\n",
    );
    for s in series {
        for i in 0..s.nu_cpmg_hz.len() {
            out.push_str(&format!(
                "{}\t{}\t{:.5}\t{:.5}\t{:.4}\t{:.5}\t{:.5}\n",
                s.seq_id,
                s.res_name,
                s.larmor_mhz,
                s.t_relax_s,
                s.nu_cpmg_hz[i],
                s.r2_eff_s_inv[i],
                s.r2_eff_err[i],
            ));
        }
    }
    out
}

/// Parse [`CpmgSeriesDataset`] entries from TSV text.
pub fn read_cpmg_series_tsv(tsv: &str) -> Result<Vec<CpmgSeriesDataset>, String> {
    type Key = (i32, String, u64, u64);
    let mut groups: BTreeMap<Key, CpmgSeriesDataset> = BTreeMap::new();

    for (line_no, raw) in tsv.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols[0] == "seq_id" {
            continue;
        }
        if cols.len() < 7 {
            return Err(format!(
                "line {}: expected 7 TSV columns in CPMG series table, got {}",
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
        let t_relax_s: f64 = cols[3]
            .parse()
            .map_err(|e| format!("line {}: invalid t_relax_s: {e}", line_no + 1))?;
        let nu: f64 = cols[4]
            .parse()
            .map_err(|e| format!("line {}: invalid nu_cpmg_hz: {e}", line_no + 1))?;
        let r2: f64 = cols[5]
            .parse()
            .map_err(|e| format!("line {}: invalid r2_eff_s_inv: {e}", line_no + 1))?;
        let err: f64 = cols[6]
            .parse()
            .map_err(|e| format!("line {}: invalid r2_eff_err: {e}", line_no + 1))?;

        let key = (
            seq_id,
            res_name.clone(),
            (larmor_mhz * 1e5).round() as u64,
            (t_relax_s * 1e6).round() as u64,
        );
        let entry = groups.entry(key).or_insert_with(|| CpmgSeriesDataset {
            seq_id,
            res_name,
            larmor_mhz,
            t_relax_s,
            nu_cpmg_hz: Vec::new(),
            r2_eff_s_inv: Vec::new(),
            r2_eff_err: Vec::new(),
        });
        entry.nu_cpmg_hz.push(nu);
        entry.r2_eff_s_inv.push(r2);
        entry.r2_eff_err.push(err);
    }

    Ok(groups.into_values().collect())
}

/// Format a [`CpmgGlobalFitResult`] summary as TSV.
pub fn format_cpmg_fit_tsv(res: &CpmgGlobalFitResult) -> String {
    let mut out = format!(
        "# model={}\tp_b={:.6}\tp_b_err={:.6}\tk_ex_s_inv={:.4}\tk_ex_err={:.4}\ttotal_chi2={:.4}\treduced_chi2={:.4}\tdof={}\n",
        res.model.as_str(),
        res.p_b,
        res.p_b_err,
        res.k_ex_s_inv,
        res.k_ex_err,
        res.total_chi2,
        res.reduced_chi2,
        res.dof
    );
    out.push_str(
        "seq_id\tres_name\tdw_ppm\tdw_ppm_err\talpha_rex\tsignificant_exchange\tdelta_aic\tf_pvalue\tchi2\tn_points\n",
    );
    for r in &res.residues {
        let alpha_s = r
            .alpha_rex_scaling
            .map(|v| format!("{v:.3}"))
            .unwrap_or_else(|| "NA".to_string());
        out.push_str(&format!(
            "{}\t{}\t{:.4}\t{:.4}\t{}\t{}\t{:.3}\t{:.4e}\t{:.4}\t{}\n",
            r.seq_id,
            r.res_name,
            r.dw_ppm,
            r.dw_ppm_err,
            alpha_s,
            r.gate.significant_exchange,
            r.gate.delta_aic,
            r.gate.f_pvalue,
            r.chi2,
            r.n_points
        ));
    }
    out
}

/// Format a [`CpmgGlobalFitResult`] as pretty JSON.
pub fn format_cpmg_fit_json(res: &CpmgGlobalFitResult) -> Result<String, String> {
    serde_json::to_string_pretty(res).map_err(|e| format!("failed to serialize CPMG JSON: {e}"))
}

/// Format an [`R1RhoFitResult`] as pretty JSON.
pub fn format_r1rho_fit_json(res: &R1RhoFitResult) -> Result<String, String> {
    serde_json::to_string_pretty(res).map_err(|e| format!("failed to serialize R1rho JSON: {e}"))
}
