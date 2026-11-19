//! Multi-frequency `DANTE-CEST` & cosine-modulated `cos-CEST` dual-sideband Bloch–McConnell
//! propagators (`T22.6`) and amortized neural `CEST` initial-parameter & posterior-uncertainty
//! inference (`T22.7`).
//!
//! Provenance: Clean-room original implementation strictly from peer-reviewed biomolecular NMR
//! spectroscopy and numerical analysis literature:
//! - McConnell, H. M. (*J. Chem. Phys.* 1958, 28, 430–431 — reaction-modified Bloch equations).
//! - Helgstrand, M., Härd, T. & Allard, P. (*J. Biomol. NMR* 2000, 18, 49–63 — homogeneous
//!   `7x7` Cartesian operator basis `[E, M_Ax, M_Ay, M_Az, M_Bx, M_By, M_Bz]^T`).
//! - Vallurupalli, P., Bouvignies, G. & Kay, L. E. (*J. Am. Chem. Soc.* 2012, 134, 8148–8161 —
//!   2-state `7x7` homogeneous Bloch–McConnell `CEST` and `B1`-inhomogeneity quadrature).
//! - Carneiro, M. G., Reddy, J. G., Griesinger, C. & Lee, D. (*J. Biomol. NMR* 2015, 63,
//!   237–244 — Fourier-transform multi-frequency acceleration of `CEST` acquisition).
//! - Yuwen, T., Kay, L. E. & Bouvignies, G. (*ChemPhysChem* 2018, 19, 1707–1710 — multi-site
//!   `DANTE-CEST` pulse-train Bloch–McConnell propagator
//!   `U = (exp(L_delay * tau_d) * exp(L_pulse * tau_p))^N_pulses`).
//! - Yuwen, T., Bouvignies, G. & Kay, L. E. (*J. Magn. Reson.* 2018, 292, 1–7 — band-selective
//!   `DANTE-CEST` and cosine-modulated `cos-CEST` dual-sideband saturation propagator
//!   `omega_1(t) = 2 * omega_{1,eff} * cos(2 * pi * nu_mod * t)`).
//! - Higham, N. J. (*SIAM J. Matrix Anal. Appl.* 2005, 26, 1179–1193 — scaling-and-squaring
//!   `[6/6]` Padé matrix exponential).
//! - Nix, D. A. & Weigend, A. S. (*Proc. IEEE ICNN* 1994, 55–60 — heteroscedastic Gaussian
//!   negative log-likelihood uncertainty estimation) and Kingma, D. P. & Ba, J. (*ICLR* 2015 —
//!   Adam stochastic optimization).

#![allow(clippy::needless_range_loop)]

use std::f64::consts::{PI, SQRT_2, TAU};

use serde::{Deserialize, Serialize};

use crate::cest::{
    build_liouvillian_2state, gauss_hermite_normalized, mat_identity, mat_mul, mat_norm1,
    mat_solve, mat_vec_mul, CestNormalization, PADE66_COEFFS, PADE66_THETA_F64,
};
use crate::rng::Rng;

/// Type alias for a `7x7` homogeneous two-state Bloch–McConnell matrix.
pub type Mat7 = [[f64; 7]; 7];

// ============================================================================
// 1. Exact 7x7 Scaling-and-Squaring Padé(6,6) & O(log N) Binary Exponentiation
// ============================================================================

/// Compute `exp(A) - I` for a `7x7` matrix via Higham (2005) scaling-and-squaring `[6/6]` Padé
/// approximant without adding/subtracting the identity matrix, eliminating cancellation for
/// short-step propagators (`||A|| << 1`).
pub fn expm_m1_7x7(a: &Mat7) -> Result<Mat7, String> {
    let norm1 = mat_norm1(a);
    if !norm1.is_finite() {
        return Err("non-finite matrix entry in expm_m1_7x7".to_string());
    }
    if norm1 == 0.0 {
        return Ok([[0.0; 7]; 7]);
    }

    let s: u32 = if norm1 > PADE66_THETA_F64 {
        (norm1 / PADE66_THETA_F64).log2().ceil().max(0.0) as u32
    } else {
        0
    };
    if s > 60 {
        return Err(format!(
            "expm_m1_7x7 scaling exponent {s} exceeds safe range (norm1={norm1:.3e})"
        ));
    }
    let inv_scale = 0.5_f64.powi(s as i32);
    let mut a_s = [[0.0; 7]; 7];
    for i in 0..7 {
        for j in 0..7 {
            a_s[i][j] = a[i][j] * inv_scale;
        }
    }

    let a2 = mat_mul(&a_s, &a_s);
    let a4 = mat_mul(&a2, &a2);
    let a6 = mat_mul(&a2, &a4);

    let b = PADE66_COEFFS;
    let mut inner_odd = [[0.0; 7]; 7];
    let mut v = [[0.0; 7]; 7];
    for i in 0..7 {
        for j in 0..7 {
            inner_odd[i][j] = b[5] * a4[i][j] + b[3] * a2[i][j];
            v[i][j] = b[6] * a6[i][j] + b[4] * a4[i][j] + b[2] * a2[i][j];
        }
        inner_odd[i][i] += b[1];
        v[i][i] += b[0];
    }
    let u = mat_mul(&a_s, &inner_odd);

    let mut v_minus_u = [[0.0; 7]; 7];
    let mut two_u = [[0.0; 7]; 7];
    for i in 0..7 {
        for j in 0..7 {
            v_minus_u[i][j] = v[i][j] - u[i][j];
            two_u[i][j] = 2.0 * u[i][j];
        }
    }

    let mut f = mat_solve(&v_minus_u, &two_u)?;
    for _ in 0..s {
        let f2 = mat_mul(&f, &f);
        for i in 0..7 {
            for j in 0..7 {
                f[i][j] = 2.0 * f[i][j] + f2[i][j];
            }
        }
    }
    Ok(f)
}

/// Compute `exp(A)` for a `7x7` homogeneous Bloch–McConnell matrix via scaling-and-squaring
/// Higham `[6/6]` Padé approximant.
pub fn expm_7x7(a: &Mat7) -> Result<Mat7, String> {
    let mut f = expm_m1_7x7(a)?;
    for i in 0..7 {
        f[i][i] += 1.0;
    }
    Ok(f)
}

/// Compose two identity-deviation propagators: given `F_a = P_a - I` and `F_b = P_b - I`,
/// returns `P_a * P_b - I = F_a + F_b + F_a * F_b` without cancellation near `I`.
#[inline]
pub fn compose_m1_7x7(fa: &Mat7, fb: &Mat7) -> Mat7 {
    let ab = mat_mul(fa, fb);
    let mut out = [[0.0; 7]; 7];
    for i in 0..7 {
        for j in 0..7 {
            out[i][j] = fa[i][j] + fb[i][j] + ab[i][j];
        }
    }
    out
}

/// Raise an identity-deviation propagator `I + F` to the integer power `n` in `O(log n)`
/// matrix multiplications, returning `(I + F)^n - I`.
pub fn mat_pow_m1_7x7(f: &Mat7, mut n: u32) -> Mat7 {
    let mut res = [[0.0; 7]; 7];
    if n == 0 {
        return res;
    }
    let mut base = *f;
    while n > 0 {
        if (n & 1) == 1 {
            res = compose_m1_7x7(&res, &base);
        }
        n >>= 1;
        if n > 0 {
            let b2 = mat_mul(&base, &base);
            for i in 0..7 {
                for j in 0..7 {
                    base[i][j] = 2.0 * base[i][j] + b2[i][j];
                }
            }
        }
    }
    res
}

/// Raise a `7x7` propagator `P` to integer power `n` via `O(log n)` binary exponentiation.
pub fn mat_pow_7x7(p: &Mat7, mut n: u32) -> Mat7 {
    let mut res = mat_identity::<7>();
    if n == 0 {
        return res;
    }
    let mut base = *p;
    while n > 0 {
        if (n & 1) == 1 {
            res = mat_mul(&res, &base);
        }
        n >>= 1;
        if n > 0 {
            base = mat_mul(&base, &base);
        }
    }
    res
}

#[inline]
fn scale_mat7(a: &Mat7, factor: f64) -> Mat7 {
    let mut out = [[0.0; 7]; 7];
    for i in 0..7 {
        for j in 0..7 {
            out[i][j] = a[i][j] * factor;
        }
    }
    out
}

// ============================================================================
// 2. Multi-Frequency CEST Parameter Types & Forward Propagators (T22.6)
// ============================================================================

/// Maximum allowed saturation duration `T_EX` (`s`) for physical `CEST` schemes.
pub const MAX_CEST_DURATION_S: f64 = 10.0;
/// Maximum allowed RF nutation frequency (`Hz`) for physical `CEST` schemes.
pub const MAX_CEST_B1_HZ: f64 = 100_000.0;
/// Maximum allowed number of `DANTE-CEST` pulse-train cycles.
pub const MAX_DANTE_PULSES: u32 = 10_000;
/// Maximum allowed midpoint Trotter steps per cosine modulation period (`cos-CEST`).
pub const MAX_COS_CEST_STEPS_PER_CYCLE: u32 = 256;
/// Maximum allowed cosine modulation frequency `nu_mod` (`Hz`) for `cos-CEST`.
pub const MAX_COS_CEST_MOD_FREQ_HZ: f64 = 20_000.0;
/// Maximum allowed cosine modulation cycles (`t_ex_s * mod_freq_hz`) to prevent `u32` saturation
/// and `f32` precision loss in remainder integration.
pub const MAX_COS_CEST_CYCLES: u32 = 200_000;
/// Maximum allowed Gauss–Hermite `B1` quadrature nodes per offset.
pub const MAX_CEST_B1_QUAD_NODES: usize = 31;
/// Maximum allowed total propagator steps per simulation call or GPU batch (work budget).
pub const MAX_TOTAL_PROPAGATOR_STEPS: u64 = 5_000_000;

/// Irradiation scheme for two-state multi-frequency `CEST` saturation profiles.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum MultiCestScheme {
    /// Standard continuous-wave `CEST` saturation (Vallurupalli, Bouvignies & Kay 2012).
    Cw {
        /// Saturation duration `T_EX` (`s`).
        t_ex_s: f64,
        /// Continuous-wave RF nutation frequency `nu_1` (`Hz`).
        b1_hz: f64,
    },
    /// `DANTE-CEST` pulse-train saturation (Yuwen, Kay & Bouvignies, *ChemPhysChem* 2018, 19,
    /// 1707–1710; Yuwen, Bouvignies & Kay, *J. Magn. Reson.* 2018, 292, 1–7):
    /// `U = (exp(L_delay * tau_delay_s) * exp(L_pulse * tau_pulse_s))^n_pulses`.
    Dante {
        /// RF pulse duration `tau_p` per cycle (`s`).
        tau_pulse_s: f64,
        /// Inter-pulse free-precession delay `tau_d` per cycle (`s`).
        tau_delay_s: f64,
        /// Number of repeated `DANTE` cycles `N_pulses`.
        n_pulses: u32,
        /// Peak RF nutation frequency `nu_{1,pulse}` during each pulse (`Hz`).
        b1_pulse_hz: f64,
    },
    /// Cosine-modulated dual-sideband `cos-CEST` saturation (Yuwen, Bouvignies & Kay,
    /// *J. Magn. Reson.* 2018, 292, 1–7; Yuwen, Kay & Bouvignies, *ChemPhysChem* 2018, 19,
    /// 1707–1710):
    /// `omega_1(t) = 2 * omega_{1,eff} * cos(2 * pi * nu_mod * t)`.
    CosModulated {
        /// Total saturation duration `T_EX` (`s`).
        t_ex_s: f64,
        /// Effective single-sideband RF nutation frequency `nu_{1,eff}` (`Hz`);
        /// peak modulation amplitude is `2 * b1_eff_hz`.
        b1_eff_hz: f64,
        /// Cosine modulation frequency `nu_mod` (`Hz`), creating dual sidebands at `+-nu_mod`.
        mod_freq_hz: f64,
        /// Number of midpoint Trotter time steps per modulation period `T_mod = 1 / nu_mod`.
        /// Even values `>= 16` (default `32`) exploit exact cosine half-period symmetry.
        n_steps_per_cycle: u32,
    },
}

impl MultiCestScheme {
    /// Total saturation duration `T_EX` (`s`).
    pub fn total_duration_s(&self) -> f64 {
        match *self {
            Self::Cw { t_ex_s, .. } | Self::CosModulated { t_ex_s, .. } => t_ex_s,
            Self::Dante {
                tau_pulse_s,
                tau_delay_s,
                n_pulses,
                ..
            } => (tau_pulse_s + tau_delay_s) * (n_pulses as f64),
        }
    }

    /// Effective on-resonance single-sideband RF nutation frequency (`Hz`).
    pub fn effective_b1_hz(&self) -> f64 {
        match *self {
            Self::Cw { b1_hz, .. } => b1_hz,
            Self::Dante {
                tau_pulse_s,
                tau_delay_s,
                b1_pulse_hz,
                ..
            } => {
                let cycle = tau_pulse_s + tau_delay_s;
                if cycle > 0.0 {
                    b1_pulse_hz * (tau_pulse_s / cycle)
                } else {
                    0.0
                }
            }
            Self::CosModulated { b1_eff_hz, .. } => b1_eff_hz,
        }
    }

    /// Compute bounded propagator steps (7x7 matrix exponentials / binary-squaring multiplies)
    /// required per `(offset, B1 quadrature node)` evaluation.
    pub fn propagator_steps_per_eval(&self) -> Result<u64, String> {
        match *self {
            Self::Cw { .. } => Ok(1),
            Self::Dante { n_pulses, .. } => {
                if n_pulses == 0 || n_pulses > MAX_DANTE_PULSES {
                    return Err(format!(
                        "DANTE-CEST requires 1 <= n_pulses <= {MAX_DANTE_PULSES}, got {n_pulses}"
                    ));
                }
                let pow_steps = 32u32.saturating_sub(n_pulses.leading_zeros());
                Ok(2u64 + u64::from(pow_steps))
            }
            Self::CosModulated {
                t_ex_s,
                mod_freq_hz,
                n_steps_per_cycle,
                ..
            } => {
                if !t_ex_s.is_finite() || t_ex_s <= 0.0 || t_ex_s > MAX_CEST_DURATION_S {
                    return Err(format!(
                        "cos-CEST requires finite t_ex_s in (0, {MAX_CEST_DURATION_S}], got {t_ex_s}"
                    ));
                }
                if !mod_freq_hz.is_finite()
                    || mod_freq_hz <= 0.0
                    || mod_freq_hz > MAX_COS_CEST_MOD_FREQ_HZ
                {
                    return Err(format!(
                        "cos-CEST requires finite mod_freq_hz in (0, {MAX_COS_CEST_MOD_FREQ_HZ}], got {mod_freq_hz}"
                    ));
                }
                if !(4..=MAX_COS_CEST_STEPS_PER_CYCLE).contains(&n_steps_per_cycle) {
                    return Err(format!(
                        "cos-CEST requires n_steps_per_cycle in 4..={MAX_COS_CEST_STEPS_PER_CYCLE}, got {n_steps_per_cycle}"
                    ));
                }
                let cycles_f = t_ex_s * mod_freq_hz;
                if !cycles_f.is_finite()
                    || !(0.0..=(MAX_COS_CEST_CYCLES as f64)).contains(&cycles_f)
                {
                    return Err(format!(
                        "cos-CEST cycle count t_ex_s * mod_freq_hz ({cycles_f:.3e}) exceeds MAX_COS_CEST_CYCLES ({MAX_COS_CEST_CYCLES})"
                    ));
                }
                let n_full = (cycles_f + 1e-6).floor().max(0.0) as u32;
                let cycle_steps = if n_steps_per_cycle.is_multiple_of(2) {
                    n_steps_per_cycle / 2
                } else {
                    n_steps_per_cycle
                };
                let pow_steps = if n_full > 0 {
                    32u32.saturating_sub(n_full.leading_zeros())
                } else {
                    0
                };
                let t_mod = 1.0 / mod_freq_hz;
                let t_rem = (t_ex_s - (n_full as f64) * t_mod).max(0.0);
                let rem_steps = if t_rem > 1e-6 * t_mod {
                    ((n_steps_per_cycle as f64) * (t_rem / t_mod))
                        .ceil()
                        .clamp(1.0, n_steps_per_cycle as f64) as u32
                } else {
                    0
                };
                u64::from(cycle_steps)
                    .checked_add(u64::from(pow_steps))
                    .and_then(|v| v.checked_add(u64::from(rem_steps)))
                    .ok_or_else(|| "cos-CEST propagator step count overflowed u64".to_string())
            }
        }
    }

    /// Validate physical constraints and upper work bounds on the irradiation scheme.
    pub fn validate(&self) -> Result<(), String> {
        match *self {
            Self::Cw { t_ex_s, b1_hz } => {
                if !t_ex_s.is_finite() || t_ex_s <= 0.0 || t_ex_s > MAX_CEST_DURATION_S {
                    return Err(format!(
                        "CW CEST requires finite t_ex_s in (0, {MAX_CEST_DURATION_S}], got {t_ex_s}"
                    ));
                }
                if !b1_hz.is_finite() || !(0.0..=MAX_CEST_B1_HZ).contains(&b1_hz) {
                    return Err(format!(
                        "CW CEST requires finite b1_hz in [0, {MAX_CEST_B1_HZ}], got {b1_hz}"
                    ));
                }
            }
            Self::Dante {
                tau_pulse_s,
                tau_delay_s,
                n_pulses,
                b1_pulse_hz,
            } => {
                if !tau_pulse_s.is_finite()
                    || tau_pulse_s <= 0.0
                    || tau_pulse_s > MAX_CEST_DURATION_S
                {
                    return Err(format!(
                        "DANTE-CEST requires finite tau_pulse_s in (0, {MAX_CEST_DURATION_S}], got {tau_pulse_s}"
                    ));
                }
                if !tau_delay_s.is_finite() || !(0.0..=MAX_CEST_DURATION_S).contains(&tau_delay_s) {
                    return Err(format!(
                        "DANTE-CEST requires finite tau_delay_s in [0, {MAX_CEST_DURATION_S}], got {tau_delay_s}"
                    ));
                }
                if n_pulses == 0 || n_pulses > MAX_DANTE_PULSES {
                    return Err(format!(
                        "DANTE-CEST requires n_pulses in 1..={MAX_DANTE_PULSES}, got {n_pulses}"
                    ));
                }
                let total_dur = (tau_pulse_s + tau_delay_s) * (n_pulses as f64);
                if !total_dur.is_finite() || total_dur <= 0.0 || total_dur > MAX_CEST_DURATION_S {
                    return Err(format!(
                        "DANTE-CEST total duration ({total_dur:.3e} s) must be finite in (0, {MAX_CEST_DURATION_S}]"
                    ));
                }
                if !b1_pulse_hz.is_finite() || !(0.0..=MAX_CEST_B1_HZ).contains(&b1_pulse_hz) {
                    return Err(format!(
                        "DANTE-CEST requires finite b1_pulse_hz in [0, {MAX_CEST_B1_HZ}], got {b1_pulse_hz}"
                    ));
                }
            }
            Self::CosModulated {
                t_ex_s,
                b1_eff_hz,
                mod_freq_hz,
                n_steps_per_cycle,
            } => {
                if !t_ex_s.is_finite() || t_ex_s <= 0.0 || t_ex_s > MAX_CEST_DURATION_S {
                    return Err(format!(
                        "cos-CEST requires finite t_ex_s in (0, {MAX_CEST_DURATION_S}], got {t_ex_s}"
                    ));
                }
                if !b1_eff_hz.is_finite() || !(0.0..=MAX_CEST_B1_HZ).contains(&b1_eff_hz) {
                    return Err(format!(
                        "cos-CEST requires finite b1_eff_hz in [0, {MAX_CEST_B1_HZ}], got {b1_eff_hz}"
                    ));
                }
                if !mod_freq_hz.is_finite()
                    || mod_freq_hz <= 0.0
                    || mod_freq_hz > MAX_COS_CEST_MOD_FREQ_HZ
                {
                    return Err(format!(
                        "cos-CEST requires finite mod_freq_hz in (0, {MAX_COS_CEST_MOD_FREQ_HZ}], got {mod_freq_hz}"
                    ));
                }
                if !(4..=MAX_COS_CEST_STEPS_PER_CYCLE).contains(&n_steps_per_cycle) {
                    return Err(format!(
                        "cos-CEST requires n_steps_per_cycle in 4..={MAX_COS_CEST_STEPS_PER_CYCLE}, got {n_steps_per_cycle}"
                    ));
                }
                let cycles_f = t_ex_s * mod_freq_hz;
                if !cycles_f.is_finite()
                    || !(0.0..=(MAX_COS_CEST_CYCLES as f64)).contains(&cycles_f)
                {
                    return Err(format!(
                        "cos-CEST total cycles t_ex_s * mod_freq_hz ({cycles_f:.3e}) exceeds MAX_COS_CEST_CYCLES ({MAX_COS_CEST_CYCLES})"
                    ));
                }
            }
        }
        let steps = self.propagator_steps_per_eval()?;
        if steps > MAX_TOTAL_PROPAGATOR_STEPS {
            return Err(format!(
                "multi-CEST single-eval propagator steps ({steps}) exceeds MAX_TOTAL_PROPAGATOR_STEPS ({MAX_TOTAL_PROPAGATOR_STEPS})"
            ));
        }
        Ok(())
    }
}

/// Validate the total propagator-step work budget (`n_offsets * eff_quad * steps_per_eval`)
/// using checked arithmetic to prevent CPU/GPU DoS or TDR hangs.
pub fn validate_multi_cest_work_budget(
    scheme: &MultiCestScheme,
    n_offsets: usize,
    n_quad: usize,
) -> Result<u64, String> {
    scheme.validate()?;
    if n_quad > MAX_CEST_B1_QUAD_NODES {
        return Err(format!(
            "n_b1_quad ({n_quad}) exceeds MAX_CEST_B1_QUAD_NODES ({MAX_CEST_B1_QUAD_NODES})"
        ));
    }
    let steps_per_eval = scheme.propagator_steps_per_eval()?;
    let eff_quad = n_quad.max(1) as u64;
    let total = (n_offsets as u64)
        .checked_mul(eff_quad)
        .and_then(|v| v.checked_mul(steps_per_eval))
        .ok_or_else(|| "multi-CEST total propagator work overflowed u64".to_string())?;
    if total > MAX_TOTAL_PROPAGATOR_STEPS {
        return Err(format!(
            "multi-CEST total propagator steps ({total}) exceeds MAX_TOTAL_PROPAGATOR_STEPS ({MAX_TOTAL_PROPAGATOR_STEPS})"
        ));
    }
    Ok(total)
}

/// Complete physical parameters for a 2-state `DANTE-CEST` simulation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DanteCestParams {
    pub p_b: f64,
    pub k_ex_s_inv: f64,
    pub major_ppm: f64,
    pub dw_ab_ppm: f64,
    pub r1a_s_inv: f64,
    pub r2a_s_inv: f64,
    pub r1b_s_inv: f64,
    pub r2b_s_inv: f64,
    pub larmor_mhz: f64,
    pub tau_pulse_s: f64,
    pub tau_delay_s: f64,
    pub n_pulses: u32,
    pub b1_pulse_hz: f64,
    pub b1_inhomogeneity_frac: f64,
    pub n_b1_quad: usize,
    pub normalization: CestNormalization,
}

impl DanteCestParams {
    /// Comb sideband spacing `sw_DANTE = 1 / (tau_p + tau_d)` in `Hz`.
    pub fn comb_spacing_hz(&self) -> f64 {
        let cycle = self.tau_pulse_s + self.tau_delay_s;
        if cycle > 0.0 {
            1.0 / cycle
        } else {
            0.0
        }
    }

    /// Comb sideband spacing `sw_DANTE / nu_0` in `ppm`.
    pub fn comb_spacing_ppm(&self) -> f64 {
        if self.larmor_mhz > 0.0 {
            self.comb_spacing_hz() / self.larmor_mhz
        } else {
            0.0
        }
    }

    /// Time-averaged RF nutation field `b1_eff = b1_pulse_hz * tau_p / (tau_p + tau_d)` (`Hz`).
    pub fn effective_b1_hz(&self) -> f64 {
        let cycle = self.tau_pulse_s + self.tau_delay_s;
        if cycle > 0.0 {
            self.b1_pulse_hz * (self.tau_pulse_s / cycle)
        } else {
            0.0
        }
    }

    /// Total `DANTE` train duration `N_pulses * (tau_p + tau_d)` (`s`).
    pub fn total_duration_s(&self) -> f64 {
        (self.tau_pulse_s + self.tau_delay_s) * (self.n_pulses as f64)
    }

    /// Convert to unified [`MultiCestSimParams`].
    pub fn to_multi_params(&self) -> MultiCestSimParams {
        MultiCestSimParams {
            p_b: self.p_b,
            k_ex_s_inv: self.k_ex_s_inv,
            major_ppm: self.major_ppm,
            dw_ab_ppm: self.dw_ab_ppm,
            r1a_s_inv: self.r1a_s_inv,
            r2a_s_inv: self.r2a_s_inv,
            r1b_s_inv: self.r1b_s_inv,
            r2b_s_inv: self.r2b_s_inv,
            larmor_mhz: self.larmor_mhz,
            b1_inhomogeneity_frac: self.b1_inhomogeneity_frac,
            n_b1_quad: self.n_b1_quad,
            normalization: self.normalization,
            scheme: MultiCestScheme::Dante {
                tau_pulse_s: self.tau_pulse_s,
                tau_delay_s: self.tau_delay_s,
                n_pulses: self.n_pulses,
                b1_pulse_hz: self.b1_pulse_hz,
            },
        }
    }
}

/// Complete physical parameters for a 2-state cosine-modulated `cos-CEST` simulation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CosCestParams {
    pub p_b: f64,
    pub k_ex_s_inv: f64,
    pub major_ppm: f64,
    pub dw_ab_ppm: f64,
    pub r1a_s_inv: f64,
    pub r2a_s_inv: f64,
    pub r1b_s_inv: f64,
    pub r2b_s_inv: f64,
    pub larmor_mhz: f64,
    pub t_ex_s: f64,
    pub b1_eff_hz: f64,
    pub mod_freq_hz: f64,
    pub n_steps_per_cycle: u32,
    pub b1_inhomogeneity_frac: f64,
    pub n_b1_quad: usize,
    pub normalization: CestNormalization,
}

impl CosCestParams {
    /// Sideband offset `nu_mod / nu_0` (`ppm`) on either side of each resonance.
    pub fn sideband_offset_ppm(&self) -> f64 {
        if self.larmor_mhz > 0.0 {
            self.mod_freq_hz / self.larmor_mhz
        } else {
            0.0
        }
    }

    /// Convert to unified [`MultiCestSimParams`].
    pub fn to_multi_params(&self) -> MultiCestSimParams {
        MultiCestSimParams {
            p_b: self.p_b,
            k_ex_s_inv: self.k_ex_s_inv,
            major_ppm: self.major_ppm,
            dw_ab_ppm: self.dw_ab_ppm,
            r1a_s_inv: self.r1a_s_inv,
            r2a_s_inv: self.r2a_s_inv,
            r1b_s_inv: self.r1b_s_inv,
            r2b_s_inv: self.r2b_s_inv,
            larmor_mhz: self.larmor_mhz,
            b1_inhomogeneity_frac: self.b1_inhomogeneity_frac,
            n_b1_quad: self.n_b1_quad,
            normalization: self.normalization,
            scheme: MultiCestScheme::CosModulated {
                t_ex_s: self.t_ex_s,
                b1_eff_hz: self.b1_eff_hz,
                mod_freq_hz: self.mod_freq_hz,
                n_steps_per_cycle: self.n_steps_per_cycle,
            },
        }
    }
}

/// Unified 2-state Bloch–McConnell parameters supporting `CW`, `DANTE-CEST`, and `cos-CEST`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MultiCestSimParams {
    pub p_b: f64,
    pub k_ex_s_inv: f64,
    pub major_ppm: f64,
    pub dw_ab_ppm: f64,
    pub r1a_s_inv: f64,
    pub r2a_s_inv: f64,
    pub r1b_s_inv: f64,
    pub r2b_s_inv: f64,
    pub larmor_mhz: f64,
    pub b1_inhomogeneity_frac: f64,
    pub n_b1_quad: usize,
    pub normalization: CestNormalization,
    pub scheme: MultiCestScheme,
}

impl MultiCestSimParams {
    /// Validate all physical parameters and reject hostile/non-finite inputs.
    pub fn validate(&self) -> Result<(), String> {
        if !self.p_b.is_finite() || self.p_b < 0.0 || self.p_b >= 1.0 {
            return Err(format!("p_b must be finite in [0, 1), got {}", self.p_b));
        }
        if !self.k_ex_s_inv.is_finite() || self.k_ex_s_inv < 0.0 {
            return Err(format!(
                "k_ex_s_inv must be finite >= 0, got {}",
                self.k_ex_s_inv
            ));
        }
        if !self.major_ppm.is_finite() || !self.dw_ab_ppm.is_finite() {
            return Err("major_ppm and dw_ab_ppm must be finite".to_string());
        }
        for (name, val) in [
            ("r1a_s_inv", self.r1a_s_inv),
            ("r2a_s_inv", self.r2a_s_inv),
            ("r1b_s_inv", self.r1b_s_inv),
            ("r2b_s_inv", self.r2b_s_inv),
        ] {
            if !val.is_finite() || val < 0.0 {
                return Err(format!("{name} must be finite >= 0, got {val}"));
            }
        }
        if !self.larmor_mhz.is_finite() || self.larmor_mhz <= 0.0 {
            return Err(format!(
                "larmor_mhz must be finite > 0, got {}",
                self.larmor_mhz
            ));
        }
        if !self.b1_inhomogeneity_frac.is_finite() || self.b1_inhomogeneity_frac < 0.0 {
            return Err(format!(
                "b1_inhomogeneity_frac must be finite >= 0, got {}",
                self.b1_inhomogeneity_frac
            ));
        }
        if self.n_b1_quad > MAX_CEST_B1_QUAD_NODES {
            return Err(format!(
                "n_b1_quad ({}) exceeds MAX_CEST_B1_QUAD_NODES ({MAX_CEST_B1_QUAD_NODES})",
                self.n_b1_quad
            ));
        }
        self.scheme.validate()
    }
}

/// Compute `(e0, inv_i0)` normalization constants for a unified multi-frequency `CEST` scheme.
pub fn compute_multi_cest_e0_inv_i0(p: &MultiCestSimParams) -> Result<(f64, f64), String> {
    p.validate()?;
    let pb = p.p_b;
    let pa = 1.0 - pb;
    let e0 = match p.normalization {
        CestNormalization::ZeroTimePhaseCycled | CestNormalization::SameDelayZeroB1 => 0.0,
        CestNormalization::ThermalEquilibrium => 1.0,
    };
    let i0 = match p.normalization {
        CestNormalization::ZeroTimePhaseCycled | CestNormalization::ThermalEquilibrium => pa,
        CestNormalization::SameDelayZeroB1 => {
            let t_ex = p.scheme.total_duration_s();
            let l0 = build_liouvillian_2state(
                pb,
                p.k_ex_s_inv,
                0.0,
                0.0,
                0.0,
                0.0,
                p.r1a_s_inv,
                p.r2a_s_inv,
                p.r1b_s_inv,
                p.r2b_s_inv,
            );
            let lt = scale_mat7(&l0, t_ex);
            let prop = expm_7x7(&lt)?;
            let v0 = [e0, 0.0, 0.0, pa, 0.0, 0.0, pb];
            mat_vec_mul(&prop, &v0)[3].max(1e-12)
        }
    };
    Ok((e0, 1.0 / i0.max(1e-12)))
}

/// Build the single-cycle `DANTE-CEST` propagator `P_cycle = exp(L_delay * tau_d) * exp(L_pulse * tau_p)`
/// (Yuwen, Kay & Bouvignies, *ChemPhysChem* 2018, 19, 1707–1710) as identity deviation `P_cycle - I`.
#[allow(clippy::too_many_arguments)]
pub fn dante_cycle_propagator_m1(
    p_b: f64,
    k_ex: f64,
    omega_a_rad: f64,
    omega_b_rad: f64,
    w1_pulse_rad: f64,
    r1a: f64,
    r2a: f64,
    r1b: f64,
    r2b: f64,
    tau_pulse_s: f64,
    tau_delay_s: f64,
    f_delay_cached: Option<&Mat7>,
) -> Result<Mat7, String> {
    let f_delay = match f_delay_cached {
        Some(cached) => *cached,
        None => {
            if tau_delay_s > 0.0 {
                let l_delay = build_liouvillian_2state(
                    p_b,
                    k_ex,
                    omega_a_rad,
                    omega_b_rad,
                    0.0,
                    0.0,
                    r1a,
                    r2a,
                    r1b,
                    r2b,
                );
                expm_m1_7x7(&scale_mat7(&l_delay, tau_delay_s))?
            } else {
                [[0.0; 7]; 7]
            }
        }
    };
    let l_pulse = build_liouvillian_2state(
        p_b,
        k_ex,
        omega_a_rad,
        omega_b_rad,
        w1_pulse_rad,
        0.0,
        r1a,
        r2a,
        r1b,
        r2b,
    );
    let f_pulse = expm_m1_7x7(&scale_mat7(&l_pulse, tau_pulse_s))?;
    Ok(compose_m1_7x7(&f_delay, &f_pulse))
}

/// Build the single-period cosine-modulated `cos-CEST` Trotter propagator over
/// `T_mod = 1 / mod_freq_hz` with `omega_1(t) = 2 * w1_eff_rad * cos(2 * pi * mod_freq_hz * t)`
/// (Yuwen, Bouvignies & Kay, *J. Magn. Reson.* 2018, 292, 1–7), returning `P_cycle - I`.
#[allow(clippy::too_many_arguments)]
pub fn cos_cest_cycle_propagator_m1(
    p_b: f64,
    k_ex: f64,
    omega_a_rad: f64,
    omega_b_rad: f64,
    w1_eff_rad: f64,
    r1a: f64,
    r2a: f64,
    r1b: f64,
    r2b: f64,
    mod_freq_hz: f64,
    n_steps_per_cycle: u32,
) -> Result<Mat7, String> {
    if !mod_freq_hz.is_finite() || mod_freq_hz <= 0.0 || mod_freq_hz > MAX_COS_CEST_MOD_FREQ_HZ {
        return Err(format!(
            "cos_cest_cycle_propagator_m1 requires finite mod_freq_hz in (0, {MAX_COS_CEST_MOD_FREQ_HZ}], got {mod_freq_hz}"
        ));
    }
    if !(4..=MAX_COS_CEST_STEPS_PER_CYCLE).contains(&n_steps_per_cycle) {
        return Err(format!(
            "cos_cest_cycle_propagator_m1 requires n_steps_per_cycle in 4..={MAX_COS_CEST_STEPS_PER_CYCLE}, got {n_steps_per_cycle}"
        ));
    }
    let m = n_steps_per_cycle;
    let t_mod = 1.0 / mod_freq_hz;
    let dt = t_mod / (m as f64);

    if m.is_multiple_of(2) {
        // Exact half-period cosine symmetry: cos(2*pi*(M - 1 - k + 0.5)/M) == cos(2*pi*(k + 0.5)/M)
        // P_cycle = U_{M-1} ... U_0 = (U_0 ... U_{M/2 - 1}) * (U_{M/2 - 1} ... U_0)
        let half = m / 2;
        let mut f_first = [[0.0; 7]; 7];
        let mut f_second = [[0.0; 7]; 7];
        for k in 0..half {
            let phase = TAU * ((k as f64) + 0.5) / (m as f64);
            let w1_k = 2.0 * w1_eff_rad * phase.cos();
            let l_k = build_liouvillian_2state(
                p_b,
                k_ex,
                omega_a_rad,
                omega_b_rad,
                w1_k,
                0.0,
                r1a,
                r2a,
                r1b,
                r2b,
            );
            let f_k = expm_m1_7x7(&scale_mat7(&l_k, dt))?;
            f_first = compose_m1_7x7(&f_k, &f_first);
            f_second = compose_m1_7x7(&f_second, &f_k);
        }
        Ok(compose_m1_7x7(&f_second, &f_first))
    } else {
        let mut f_cycle = [[0.0; 7]; 7];
        for k in 0..m {
            let phase = TAU * ((k as f64) + 0.5) / (m as f64);
            let w1_k = 2.0 * w1_eff_rad * phase.cos();
            let l_k = build_liouvillian_2state(
                p_b,
                k_ex,
                omega_a_rad,
                omega_b_rad,
                w1_k,
                0.0,
                r1a,
                r2a,
                r1b,
                r2b,
            );
            let f_k = expm_m1_7x7(&scale_mat7(&l_k, dt))?;
            f_cycle = compose_m1_7x7(&f_k, &f_cycle);
        }
        Ok(f_cycle)
    }
}

/// Evaluate a unified multi-frequency `CEST` saturation profile (`CW`, `DANTE-CEST`, or `cos-CEST`)
/// across RF carrier `offsets_ppm` (`ppm`).
pub fn simulate_multi_cest_profile(
    p: &MultiCestSimParams,
    offsets_ppm: &[f64],
) -> Result<Vec<f64>, String> {
    p.validate()?;
    for (idx, &off) in offsets_ppm.iter().enumerate() {
        if !off.is_finite() {
            return Err(format!("non-finite offset at index {idx}: {off}"));
        }
    }
    if offsets_ppm.is_empty() {
        return Ok(Vec::new());
    }

    let (e0, inv_i0) = compute_multi_cest_e0_inv_i0(p)?;
    let pb = p.p_b;
    let pa = 1.0 - pb;
    let scale_rad = TAU * p.larmor_mhz;

    let use_quad = p.b1_inhomogeneity_frac > 0.0 && p.n_b1_quad > 1;
    let eff_quad = if use_quad { p.n_b1_quad } else { 1 };
    validate_multi_cest_work_budget(&p.scheme, offsets_ppm.len(), eff_quad)?;
    let quad_nodes = if use_quad {
        gauss_hermite_normalized(p.n_b1_quad)
    } else {
        vec![(0.0, 1.0)]
    };
    let sqrt2_sig = SQRT_2 * p.b1_inhomogeneity_frac;

    let mut out = Vec::with_capacity(offsets_ppm.len());
    for &rf_ppm in offsets_ppm {
        let om_a = scale_rad * (p.major_ppm - rf_ppm);
        let om_b = scale_rad * ((p.major_ppm + p.dw_ab_ppm) - rf_ppm);

        let iz_a_avg = match p.scheme {
            MultiCestScheme::Cw { t_ex_s, b1_hz } => {
                let w1_nom = TAU * b1_hz;
                let mut acc = 0.0;
                for &(xq, wq) in &quad_nodes {
                    let w1 = (w1_nom * (1.0 + sqrt2_sig * xq)).max(0.0);
                    let l = build_liouvillian_2state(
                        pb,
                        p.k_ex_s_inv,
                        om_a,
                        om_b,
                        w1,
                        0.0,
                        p.r1a_s_inv,
                        p.r2a_s_inv,
                        p.r1b_s_inv,
                        p.r2b_s_inv,
                    );
                    let f = expm_m1_7x7(&scale_mat7(&l, t_ex_s))?;
                    let iz_a = f[3][0] * e0 + (1.0 + f[3][3]) * pa + f[3][6] * pb;
                    acc += wq * iz_a;
                }
                acc
            }
            MultiCestScheme::Dante {
                tau_pulse_s,
                tau_delay_s,
                n_pulses,
                b1_pulse_hz,
            } => {
                // Free-precession delay propagator is independent of B1: compute once per offset.
                let f_delay = if tau_delay_s > 0.0 {
                    let l_delay = build_liouvillian_2state(
                        pb,
                        p.k_ex_s_inv,
                        om_a,
                        om_b,
                        0.0,
                        0.0,
                        p.r1a_s_inv,
                        p.r2a_s_inv,
                        p.r1b_s_inv,
                        p.r2b_s_inv,
                    );
                    expm_m1_7x7(&scale_mat7(&l_delay, tau_delay_s))?
                } else {
                    [[0.0; 7]; 7]
                };
                let w1_nom = TAU * b1_pulse_hz;
                let mut acc = 0.0;
                for &(xq, wq) in &quad_nodes {
                    let w1 = (w1_nom * (1.0 + sqrt2_sig * xq)).max(0.0);
                    let f_cycle = dante_cycle_propagator_m1(
                        pb,
                        p.k_ex_s_inv,
                        om_a,
                        om_b,
                        w1,
                        p.r1a_s_inv,
                        p.r2a_s_inv,
                        p.r1b_s_inv,
                        p.r2b_s_inv,
                        tau_pulse_s,
                        tau_delay_s,
                        Some(&f_delay),
                    )?;
                    let f_tot = mat_pow_m1_7x7(&f_cycle, n_pulses);
                    let iz_a = f_tot[3][0] * e0 + (1.0 + f_tot[3][3]) * pa + f_tot[3][6] * pb;
                    acc += wq * iz_a;
                }
                acc
            }
            MultiCestScheme::CosModulated {
                t_ex_s,
                b1_eff_hz,
                mod_freq_hz,
                n_steps_per_cycle,
            } => {
                let t_mod = 1.0 / mod_freq_hz;
                let cycles_f = t_ex_s / t_mod;
                let n_full = (cycles_f + 1e-6).floor().max(0.0) as u32;
                let t_rem = (t_ex_s - (n_full as f64) * t_mod).max(0.0);
                let has_rem = t_rem > 1e-6 * t_mod;
                let w1_nom = TAU * b1_eff_hz;

                let mut acc = 0.0;
                for &(xq, wq) in &quad_nodes {
                    let w1_eff = (w1_nom * (1.0 + sqrt2_sig * xq)).max(0.0);
                    let mut f_tot = if n_full > 0 {
                        let f_cycle = cos_cest_cycle_propagator_m1(
                            pb,
                            p.k_ex_s_inv,
                            om_a,
                            om_b,
                            w1_eff,
                            p.r1a_s_inv,
                            p.r2a_s_inv,
                            p.r1b_s_inv,
                            p.r2b_s_inv,
                            mod_freq_hz,
                            n_steps_per_cycle,
                        )?;
                        mat_pow_m1_7x7(&f_cycle, n_full)
                    } else {
                        [[0.0; 7]; 7]
                    };

                    if has_rem {
                        let m_rem = ((n_steps_per_cycle as f64) * (t_rem / t_mod))
                            .ceil()
                            .clamp(1.0, n_steps_per_cycle as f64)
                            as u32;
                        let dt_rem = t_rem / (m_rem as f64);
                        for k in 0..m_rem {
                            let t_k = ((k as f64) + 0.5) * dt_rem;
                            let w1_k = 2.0 * w1_eff * (TAU * mod_freq_hz * t_k).cos();
                            let l_k = build_liouvillian_2state(
                                pb,
                                p.k_ex_s_inv,
                                om_a,
                                om_b,
                                w1_k,
                                0.0,
                                p.r1a_s_inv,
                                p.r2a_s_inv,
                                p.r1b_s_inv,
                                p.r2b_s_inv,
                            );
                            let f_k = expm_m1_7x7(&scale_mat7(&l_k, dt_rem))?;
                            f_tot = compose_m1_7x7(&f_k, &f_tot);
                        }
                    }

                    let iz_a = f_tot[3][0] * e0 + (1.0 + f_tot[3][3]) * pa + f_tot[3][6] * pb;
                    acc += wq * iz_a;
                }
                acc
            }
        };

        out.push(iz_a_avg * inv_i0);
    }

    Ok(out)
}

/// Evaluate a 2-state `DANTE-CEST` saturation profile across `offsets_ppm` (`ppm`).
pub fn simulate_dante_cest_profile(
    p: &DanteCestParams,
    offsets_ppm: &[f64],
) -> Result<Vec<f64>, String> {
    simulate_multi_cest_profile(&p.to_multi_params(), offsets_ppm)
}

/// Evaluate a 2-state cosine-modulated `cos-CEST` saturation profile across `offsets_ppm` (`ppm`).
pub fn simulate_cos_cest_profile(
    p: &CosCestParams,
    offsets_ppm: &[f64],
) -> Result<Vec<f64>, String> {
    simulate_multi_cest_profile(&p.to_multi_params(), offsets_ppm)
}

// ============================================================================
// 3. Exact Multi-B1 Levenberg–Marquardt Fitter with Cramér–Rao Covariance
// ============================================================================

/// Experimental or synthetic multi-frequency `CEST` profile dataset at a single `B1` / scheme.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MultiCestProfile {
    pub larmor_mhz: f64,
    pub major_ppm: f64,
    pub b1_inhomogeneity_frac: f64,
    pub n_b1_quad: usize,
    pub normalization: CestNormalization,
    pub scheme: MultiCestScheme,
    pub offsets_ppm: Vec<f64>,
    pub intensities: Vec<f64>,
    pub uncertainties: Vec<f64>,
}

impl MultiCestProfile {
    /// Validate profile arrays and physical configuration.
    pub fn validate(&self) -> Result<(), String> {
        if !self.larmor_mhz.is_finite() || self.larmor_mhz <= 0.0 {
            return Err(format!(
                "profile larmor_mhz must be finite > 0, got {}",
                self.larmor_mhz
            ));
        }
        if !self.major_ppm.is_finite() {
            return Err(format!(
                "profile major_ppm must be finite, got {}",
                self.major_ppm
            ));
        }
        if !self.b1_inhomogeneity_frac.is_finite() || self.b1_inhomogeneity_frac < 0.0 {
            return Err(format!(
                "profile b1_inhomogeneity_frac must be finite >= 0, got {}",
                self.b1_inhomogeneity_frac
            ));
        }
        if self.n_b1_quad > MAX_CEST_B1_QUAD_NODES {
            return Err(format!(
                "profile n_b1_quad ({}) exceeds MAX_CEST_B1_QUAD_NODES ({MAX_CEST_B1_QUAD_NODES})",
                self.n_b1_quad
            ));
        }
        self.scheme.validate()?;
        let n = self.offsets_ppm.len();
        if n == 0 {
            return Err("profile offsets_ppm must not be empty".to_string());
        }
        let eff_quad = if self.b1_inhomogeneity_frac > 0.0 && self.n_b1_quad > 1 {
            self.n_b1_quad
        } else {
            1
        };
        validate_multi_cest_work_budget(&self.scheme, n, eff_quad)?;
        if self.intensities.len() != n || self.uncertainties.len() != n {
            return Err(format!(
                "profile array length mismatch: offsets={n}, intensities={}, uncertainties={}",
                self.intensities.len(),
                self.uncertainties.len()
            ));
        }
        for i in 0..n {
            if !self.offsets_ppm[i].is_finite()
                || !self.intensities[i].is_finite()
                || !self.uncertainties[i].is_finite()
                || self.uncertainties[i] <= 0.0
            {
                return Err(format!(
                    "invalid data point at index {i}: offset={}, intensity={}, sigma={}",
                    self.offsets_ppm[i], self.intensities[i], self.uncertainties[i]
                ));
            }
        }
        Ok(())
    }
}

/// Six-parameter global 2-state exchange model `(p_B, k_ex, dw_ppm, R1, R2a, R2b)`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct MultiCestFitParams {
    pub p_b: f64,
    pub k_ex_s_inv: f64,
    pub dw_ab_ppm: f64,
    pub r1_s_inv: f64,
    pub r2a_s_inv: f64,
    pub r2b_s_inv: f64,
}

impl MultiCestFitParams {
    /// Convert to physical 6-vector `[p_b, k_ex, dw_ppm, r1, r2a, r2b]`.
    pub fn to_array(self) -> [f64; 6] {
        [
            self.p_b,
            self.k_ex_s_inv,
            self.dw_ab_ppm,
            self.r1_s_inv,
            self.r2a_s_inv,
            self.r2b_s_inv,
        ]
    }

    /// Construct from physical 6-vector `[p_b, k_ex, dw_ppm, r1, r2a, r2b]`.
    pub fn from_array(v: [f64; 6]) -> Self {
        Self {
            p_b: v[0],
            k_ex_s_inv: v[1],
            dw_ab_ppm: v[2],
            r1_s_inv: v[3],
            r2a_s_inv: v[4],
            r2b_s_inv: v[5],
        }
    }

    /// Validate all 6 physical fit parameters.
    pub fn validate(&self) -> Result<(), String> {
        if !self.p_b.is_finite() || self.p_b <= 0.0 || self.p_b >= 1.0 {
            return Err(format!("p_b must be in (0, 1), got {}", self.p_b));
        }
        if !self.k_ex_s_inv.is_finite() || self.k_ex_s_inv <= 0.0 {
            return Err(format!("k_ex must be > 0, got {}", self.k_ex_s_inv));
        }
        if !self.dw_ab_ppm.is_finite() {
            return Err(format!("dw_ab_ppm must be finite, got {}", self.dw_ab_ppm));
        }
        if !self.r1_s_inv.is_finite() || self.r1_s_inv <= 0.0 {
            return Err(format!("r1 must be > 0, got {}", self.r1_s_inv));
        }
        if !self.r2a_s_inv.is_finite() || self.r2a_s_inv <= 0.0 {
            return Err(format!("r2a must be > 0, got {}", self.r2a_s_inv));
        }
        if !self.r2b_s_inv.is_finite() || self.r2b_s_inv <= 0.0 {
            return Err(format!("r2b must be > 0, got {}", self.r2b_s_inv));
        }
        Ok(())
    }

    fn to_opt_vec(self) -> [f64; 6] {
        [
            self.p_b.clamp(1e-4, 0.49).ln(),
            self.k_ex_s_inv.clamp(1.0, 5_000.0).ln(),
            self.dw_ab_ppm.clamp(-30.0, 30.0),
            self.r1_s_inv.clamp(0.05, 25.0).ln(),
            self.r2a_s_inv.clamp(0.5, 500.0).ln(),
            self.r2b_s_inv.clamp(0.5, 50_000.0).ln(),
        ]
    }

    fn from_opt_vec(z: [f64; 6]) -> Self {
        Self {
            p_b: z[0].clamp(1e-4_f64.ln(), 0.49_f64.ln()).exp(),
            k_ex_s_inv: z[1].clamp(1.0_f64.ln(), 5_000.0_f64.ln()).exp(),
            dw_ab_ppm: z[2].clamp(-30.0, 30.0),
            r1_s_inv: z[3].clamp(0.05_f64.ln(), 25.0_f64.ln()).exp(),
            r2a_s_inv: z[4].clamp(0.5_f64.ln(), 500.0_f64.ln()).exp(),
            r2b_s_inv: z[5].clamp(0.5_f64.ln(), 50_000.0_f64.ln()).exp(),
        }
    }
}

/// Configuration for multi-`B1` `DANTE-CEST` / `cos-CEST` Levenberg–Marquardt fitting.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MultiCestLmConfig {
    pub max_iters: usize,
    pub grad_tol: f64,
    pub step_tol: f64,
    pub initial_lambda: f64,
    /// Minimum and maximum `dw_ab_ppm` explored by grid seeding.
    pub grid_dw_range_ppm: (f64, f64),
    /// Step size (`ppm`) for `dw_ab_ppm` grid search.
    pub grid_dw_step_ppm: f64,
}

impl Default for MultiCestLmConfig {
    fn default() -> Self {
        Self {
            max_iters: 40,
            grad_tol: 1e-7,
            step_tol: 1e-7,
            initial_lambda: 1e-2,
            grid_dw_range_ppm: (-8.0, 8.0),
            grid_dw_step_ppm: 0.6,
        }
    }
}

/// Result of an exact multi-`B1` Bloch–McConnell Levenberg–Marquardt fit.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MultiCestFitResult {
    pub params: MultiCestFitParams,
    /// `1-sigma` Cramér–Rao standard errors `[p_b, k_ex, dw_ppm, r1, r2a, r2b]`, or `None` when
    /// the Fisher information matrix is rank-deficient or cannot be validly inverted.
    pub std_errors: Option<[f64; 6]>,
    /// Exact `6x6` Cramér–Rao covariance matrix `(J^T J)^(-1)` in physical coordinates, or `None`
    /// when Fisher inversion fails or yields non-finite / non-positive diagonal variances.
    pub cramer_rao_cov: Option<[[f64; 6]; 6]>,
    /// True iff `cramer_rao_cov` and `std_errors` came from a valid finite, positive-variance
    /// Fisher matrix inversion (never silently zero-filled on failure).
    pub covariance_valid: bool,
    /// True if mild Tikhonov damping was needed to invert a near-singular Fisher matrix.
    pub covariance_regularized: bool,
    pub chi2: f64,
    pub reduced_chi2: f64,
    pub n_points: usize,
    pub dof: usize,
    pub iterations: usize,
    pub n_evals: usize,
    pub converged: bool,
}

fn evaluate_residuals_all(
    profiles: &[MultiCestProfile],
    params: MultiCestFitParams,
) -> Result<(Vec<f64>, f64), String> {
    let total_pts: usize = profiles.iter().map(|p| p.offsets_ppm.len()).sum();
    let mut res = Vec::with_capacity(total_pts);
    let mut chi2 = 0.0;
    for prof in profiles {
        let sim_p = MultiCestSimParams {
            p_b: params.p_b,
            k_ex_s_inv: params.k_ex_s_inv,
            major_ppm: prof.major_ppm,
            dw_ab_ppm: params.dw_ab_ppm,
            r1a_s_inv: params.r1_s_inv,
            r2a_s_inv: params.r2a_s_inv,
            r1b_s_inv: params.r1_s_inv,
            r2b_s_inv: params.r2b_s_inv,
            larmor_mhz: prof.larmor_mhz,
            b1_inhomogeneity_frac: prof.b1_inhomogeneity_frac,
            n_b1_quad: prof.n_b1_quad,
            normalization: prof.normalization,
            scheme: prof.scheme.clone(),
        };
        let pred = simulate_multi_cest_profile(&sim_p, &prof.offsets_ppm)?;
        for i in 0..pred.len() {
            let r = (pred[i] - prof.intensities[i]) / prof.uncertainties[i];
            chi2 += r * r;
            res.push(r);
        }
    }
    Ok((res, chi2))
}

fn solve_6x6(a: &[[f64; 6]; 6], b: &[f64; 6]) -> Result<[f64; 6], String> {
    let mut lu = *a;
    let mut rhs = *b;
    for col in 0..6 {
        let mut piv = col;
        let mut max_v = lu[col][col].abs();
        for r in (col + 1)..6 {
            let v = lu[r][col].abs();
            if v > max_v {
                max_v = v;
                piv = r;
            }
        }
        if !max_v.is_finite() || max_v < 1e-24 {
            return Err(format!("singular 6x6 normal system at col {col}"));
        }
        if piv != col {
            lu.swap(col, piv);
            rhs.swap(col, piv);
        }
        let inv_diag = 1.0 / lu[col][col];
        for r in (col + 1)..6 {
            let fac = lu[r][col] * inv_diag;
            if fac == 0.0 {
                continue;
            }
            for c in (col + 1)..6 {
                lu[r][c] -= fac * lu[col][c];
            }
            rhs[r] -= fac * rhs[col];
        }
    }
    let mut x = [0.0; 6];
    for i in (0..6).rev() {
        let mut s = rhs[i];
        for k in (i + 1)..6 {
            s -= lu[i][k] * x[k];
        }
        x[i] = s / lu[i][i];
    }
    Ok(x)
}

fn invert_6x6(a: &[[f64; 6]; 6]) -> Result<[[f64; 6]; 6], String> {
    let eye = mat_identity::<6>();
    mat_solve(a, &eye)
}

struct CramerRaoCovResult {
    cov: Option<[[f64; 6]; 6]>,
    std_errors: Option<[f64; 6]>,
    valid: bool,
    regularized: bool,
    evals: usize,
}

fn is_valid_cramer_rao_cov(cov: &[[f64; 6]; 6]) -> bool {
    cov.iter().all(|row| row.iter().all(|v| v.is_finite()))
        && (0..6).all(|i| cov[i][i].is_finite() && cov[i][i] > 0.0)
}

fn std_errors_from_cov(cov: &[[f64; 6]; 6]) -> [f64; 6] {
    let mut se = [0.0; 6];
    for i in 0..6 {
        se[i] = cov[i][i].sqrt();
    }
    se
}

fn compute_cramer_rao_covariance(
    profiles: &[MultiCestProfile],
    params: MultiCestFitParams,
) -> Result<CramerRaoCovResult, String> {
    let base_arr = params.to_array();
    let steps = [
        (base_arr[0] * 1e-4)
            .max(1e-6)
            .min(0.25 * (1.0 - base_arr[0])),
        (base_arr[1] * 1e-4).max(1e-3),
        1e-4,
        (base_arr[3] * 1e-4).max(1e-4),
        (base_arr[4] * 1e-4).max(1e-4),
        (base_arr[5] * 1e-4).max(1e-4),
    ];

    let n_pts: usize = profiles.iter().map(|p| p.offsets_ppm.len()).sum();
    let mut jac_cols = vec![vec![0.0; n_pts]; 6];
    let mut evals = 0usize;

    for j in 0..6 {
        let mut p_plus = base_arr;
        let mut p_minus = base_arr;
        let h = steps[j];
        p_plus[j] += h;
        p_minus[j] -= h;
        if j != 2 && p_minus[j] <= 1e-8 {
            let (r_plus, _) =
                evaluate_residuals_all(profiles, MultiCestFitParams::from_array(p_plus))?;
            let (r_base, _) = evaluate_residuals_all(profiles, params)?;
            evals += 2;
            for i in 0..n_pts {
                jac_cols[j][i] = (r_plus[i] - r_base[i]) / h;
            }
        } else {
            let (r_plus, _) =
                evaluate_residuals_all(profiles, MultiCestFitParams::from_array(p_plus))?;
            let (r_minus, _) =
                evaluate_residuals_all(profiles, MultiCestFitParams::from_array(p_minus))?;
            evals += 2;
            let inv_2h = 0.5 / h;
            for i in 0..n_pts {
                jac_cols[j][i] = (r_plus[i] - r_minus[i]) * inv_2h;
            }
        }
    }

    let mut fisher = [[0.0; 6]; 6];
    for a in 0..6 {
        for b in a..6 {
            let mut s = 0.0;
            for i in 0..n_pts {
                s += jac_cols[a][i] * jac_cols[b][i];
            }
            fisher[a][b] = s;
            fisher[b][a] = s;
        }
    }

    let all_diag_informative = (0..6).all(|i| fisher[i][i].is_finite() && fisher[i][i] > 0.0);
    let (cov, std_errors, valid, regularized) = if !all_diag_informative {
        (None, None, false, false)
    } else {
        match invert_6x6(&fisher) {
            Ok(c) if is_valid_cramer_rao_cov(&c) => {
                (Some(c), Some(std_errors_from_cov(&c)), true, false)
            }
            _ => {
                let mut reg = fisher;
                for i in 0..6 {
                    let d = reg[i][i].abs().max(1.0);
                    reg[i][i] += 1e-8 * d;
                }
                match invert_6x6(&reg) {
                    Ok(c) if is_valid_cramer_rao_cov(&c) => {
                        (Some(c), Some(std_errors_from_cov(&c)), true, true)
                    }
                    _ => (None, None, false, true),
                }
            }
        }
    };

    Ok(CramerRaoCovResult {
        cov,
        std_errors,
        valid,
        regularized,
        evals,
    })
}

/// Fit `(p_B, k_ex, dw_ppm, R1, R2a, R2b)` globally across multi-`B1` `DANTE-CEST`, `cos-CEST`,
/// or `CW-CEST` profiles via damped Levenberg–Marquardt optimization from an initial seed.
pub fn fit_multi_cest_lm(
    profiles: &[MultiCestProfile],
    initial: MultiCestFitParams,
    cfg: &MultiCestLmConfig,
) -> Result<MultiCestFitResult, String> {
    if profiles.is_empty() {
        return Err("fit_multi_cest_lm requires at least one profile".to_string());
    }
    for prof in profiles {
        prof.validate()?;
    }
    initial.validate()?;

    let n_points: usize = profiles.iter().map(|p| p.offsets_ppm.len()).sum();
    if n_points < 6 {
        return Err(format!(
            "fit_multi_cest_lm requires at least 6 data points, got {n_points}"
        ));
    }

    let mut z = initial.to_opt_vec();
    let mut cur_params = MultiCestFitParams::from_opt_vec(z);
    let (mut cur_res, mut cur_chi2) = evaluate_residuals_all(profiles, cur_params)?;
    let mut n_evals = 1usize;

    let mut lambda = cfg.initial_lambda.clamp(1e-6, 1e3);
    let mut converged = false;
    let mut iters = 0usize;
    let fd_eps = [1e-4, 1e-4, 5e-4, 1e-4, 1e-4, 1e-4];

    for iter in 0..cfg.max_iters {
        iters = iter + 1;

        // Forward finite-difference Jacobian in optimization coordinates
        let mut jac_cols = vec![vec![0.0; n_points]; 6];
        for j in 0..6 {
            let mut z_p = z;
            z_p[j] += fd_eps[j];
            let p_p = MultiCestFitParams::from_opt_vec(z_p);
            let (res_p, _) = evaluate_residuals_all(profiles, p_p)?;
            n_evals += 1;
            let inv_h = 1.0 / fd_eps[j];
            for i in 0..n_points {
                jac_cols[j][i] = (res_p[i] - cur_res[i]) * inv_h;
            }
        }

        let mut jtj = [[0.0; 6]; 6];
        let mut grad = [0.0; 6];
        for a in 0..6 {
            let mut g = 0.0;
            for i in 0..n_points {
                g += jac_cols[a][i] * cur_res[i];
            }
            grad[a] = g;
            for b in a..6 {
                let mut s = 0.0;
                for i in 0..n_points {
                    s += jac_cols[a][i] * jac_cols[b][i];
                }
                jtj[a][b] = s;
                jtj[b][a] = s;
            }
        }

        let max_grad = grad.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        if max_grad < cfg.grad_tol {
            converged = true;
            break;
        }

        let mut accepted = false;
        let mut step_norm = 0.0;
        for _attempt in 0..10 {
            let mut lhs = jtj;
            let mut rhs = [0.0; 6];
            for i in 0..6 {
                let d = lhs[i][i].abs().max(1e-6);
                lhs[i][i] += lambda * d;
                rhs[i] = -grad[i];
            }
            let Ok(mut dz) = solve_6x6(&lhs, &rhs) else {
                lambda = (lambda * 10.0).min(1e9);
                continue;
            };
            // Trust-region cap per coordinate so log-rates and dw never leap wildly
            let caps = [0.8_f64, 0.8, 1.5, 0.8, 0.8, 1.0];
            let mut scale = 1.0_f64;
            for j in 0..6 {
                if dz[j].abs() > caps[j] {
                    scale = scale.min(caps[j] / dz[j].abs());
                }
            }
            for j in 0..6 {
                dz[j] *= scale;
            }

            let mut z_try = z;
            for j in 0..6 {
                z_try[j] += dz[j];
            }
            let p_try = MultiCestFitParams::from_opt_vec(z_try);
            let z_clamped = p_try.to_opt_vec();
            let (res_try, chi2_try) = evaluate_residuals_all(profiles, p_try)?;
            n_evals += 1;

            if chi2_try.is_finite() && chi2_try < cur_chi2 {
                step_norm = (0..6)
                    .map(|j| (z_clamped[j] - z[j]).abs())
                    .fold(0.0_f64, f64::max);
                let rel_improv = (cur_chi2 - chi2_try) / cur_chi2.max(1e-12);
                z = z_clamped;
                cur_params = p_try;
                cur_res = res_try;
                cur_chi2 = chi2_try;
                lambda = (lambda * 0.35).max(1e-7);
                accepted = true;
                if step_norm < cfg.step_tol || rel_improv < 1e-10 {
                    converged = true;
                }
                break;
            } else {
                lambda = (lambda * 4.5).min(1e9);
            }
        }

        if converged || (!accepted && lambda >= 1e8) {
            if !accepted && max_grad < 1e-3 {
                converged = true;
            }
            break;
        }
        if step_norm > 0.0 && step_norm < cfg.step_tol {
            converged = true;
            break;
        }
    }

    let cr = compute_cramer_rao_covariance(profiles, cur_params)?;
    n_evals += cr.evals;
    let dof = n_points.saturating_sub(6);
    let reduced_chi2 = cur_chi2 / (dof.max(1) as f64);

    Ok(MultiCestFitResult {
        params: cur_params,
        std_errors: cr.std_errors,
        cramer_rao_cov: cr.cov,
        covariance_valid: cr.valid,
        covariance_regularized: cr.regularized,
        chi2: cur_chi2,
        reduced_chi2,
        n_points,
        dof,
        iterations: iters,
        n_evals,
        converged,
    })
}

/// Estimate rough baseline `(R1, R2a)` directly from the outer wing and major-dip depth of
/// multi-`B1` `CEST` profiles.
pub fn estimate_baseline_relaxation(profiles: &[MultiCestProfile]) -> (f64, f64) {
    let mut r1_est_acc = 0.0;
    let mut r1_cnt = 0usize;
    for prof in profiles {
        let mut sorted = prof.intensities.clone();
        sorted.sort_by(f64::total_cmp);
        let q90_idx = ((sorted.len() as f64) * 0.90)
            .floor()
            .clamp(0.0, sorted.len().saturating_sub(1) as f64) as usize;
        let base_i = sorted[q90_idx].clamp(0.05, 0.99);
        if prof.normalization == CestNormalization::ZeroTimePhaseCycled {
            let t_ex = prof.scheme.total_duration_s().max(1e-3);
            let r1 = (-base_i.ln() / t_ex).clamp(0.3, 6.0);
            r1_est_acc += r1;
            r1_cnt += 1;
        }
    }
    let r1 = if r1_cnt > 0 {
        r1_est_acc / (r1_cnt as f64)
    } else {
        1.3
    };
    (r1, 14.0)
}

fn profile_baseline_q90(intensities: &[f64]) -> f64 {
    if intensities.is_empty() {
        return 1.0;
    }
    let mut sorted = intensities.to_vec();
    sorted.sort_by(f64::total_cmp);
    let q90_idx = ((sorted.len() as f64) * 0.90)
        .floor()
        .clamp(0.0, sorted.len().saturating_sub(1) as f64) as usize;
    sorted[q90_idx]
}

fn interp_sorted_pairs(pairs: &[(f64, f64)], query: f64) -> Option<f64> {
    if pairs.is_empty() || query < pairs[0].0 || query > pairs[pairs.len() - 1].0 {
        return None;
    }
    let pos = pairs.partition_point(|&(x, _)| x < query);
    if pos == 0 {
        return Some(pairs[0].1);
    }
    if pos >= pairs.len() {
        return Some(pairs[pairs.len() - 1].1);
    }
    let (x0, y0) = pairs[pos - 1];
    let (x1, y1) = pairs[pos];
    let dx = x1 - x0;
    if dx.abs() < 1e-12 {
        Some(0.5 * (y0 + y1))
    } else {
        Some(y0 + ((query - x0) / dx) * (y1 - y0))
    }
}

/// Non-neural scheme-aware deepest-dip heuristic seed for `CW`, `DANTE-CEST`, and `cos-CEST`
/// profiles: masks major-state dip(s), locates the deepest non-major minor-state dip (folding
/// comb periodicity for `DANTE-CEST` and pairing `+-nu_mod` sidebands for `cos-CEST`), and
/// combines the estimated `dw_ab_ppm` with baseline `(R1, R2a)` and mid-range `(p_B, k_ex)`.
pub fn deepest_dip_heuristic_seed(
    profiles: &[MultiCestProfile],
) -> Result<MultiCestFitParams, String> {
    if profiles.is_empty() {
        return Err("deepest_dip_heuristic_seed requires at least one profile".to_string());
    }
    for prof in profiles {
        prof.validate()?;
    }

    let (r1_init, r2a_init) = estimate_baseline_relaxation(profiles);
    let major_excl_ppm = 1.2_f64;

    // Precompute sorted (delta_ppm, normalized_depth) curves for every profile.
    let mut depth_curves: Vec<Vec<(f64, f64)>> = Vec::with_capacity(profiles.len());
    for prof in profiles {
        let base_i = profile_baseline_q90(&prof.intensities).max(1e-6);
        let mut pairs: Vec<(f64, f64)> = prof
            .offsets_ppm
            .iter()
            .zip(prof.intensities.iter())
            .map(|(&off, &y)| (off - prof.major_ppm, ((base_i - y) / base_i).max(0.0)))
            .collect();
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
        depth_curves.push(pairs);
    }

    let dw_est = match profiles[0].scheme {
        MultiCestScheme::Cw { .. } => {
            let mut best_local_dw = None;
            let mut best_local_score = -1.0_f64;
            let mut best_any_dw = 3.0_f64;
            let mut best_any_score = -1.0_f64;
            for curve in &depth_curves {
                let n = curve.len();
                for i in 0..n {
                    let delta = curve[i].0;
                    if delta.abs() < major_excl_ppm {
                        continue;
                    }
                    let mut sum = 0.0;
                    let mut cnt = 0usize;
                    for c2 in &depth_curves {
                        if let Some(d) = interp_sorted_pairs(c2, delta) {
                            sum += d;
                            cnt += 1;
                        }
                    }
                    let score = if cnt > 0 { sum / (cnt as f64) } else { 0.0 };
                    if score > best_any_score {
                        best_any_score = score;
                        best_any_dw = delta;
                    }
                    let is_local_peak = i > 0
                        && i + 1 < n
                        && curve[i].1 > curve[i - 1].1
                        && curve[i].1 >= curve[i + 1].1;
                    if is_local_peak && score > best_local_score {
                        best_local_score = score;
                        best_local_dw = Some(delta);
                    }
                }
            }
            best_local_dw.unwrap_or(best_any_dw)
        }
        MultiCestScheme::Dante {
            tau_pulse_s,
            tau_delay_s,
            ..
        } => {
            let sw_ppm = 1.0 / ((tau_pulse_s + tau_delay_s) * profiles[0].larmor_mhz);
            let mut best_local_dw = None;
            let mut best_local_score = -1.0_f64;
            let mut best_any_dw = 3.0_f64;
            let mut best_any_score = -1.0_f64;
            for curve in &depth_curves {
                let n = curve.len();
                for i in 0..n {
                    let delta = curve[i].0;
                    let folded = delta - (delta / sw_ppm).round() * sw_ppm;
                    if folded.abs() < major_excl_ppm {
                        continue;
                    }
                    let mut sum = 0.0;
                    let mut cnt = 0usize;
                    for c2 in &depth_curves {
                        if let Some(d) = interp_sorted_pairs(c2, folded) {
                            sum += d;
                            cnt += 1;
                        }
                    }
                    let score = if cnt > 0 { sum / (cnt as f64) } else { 0.0 };
                    if score > best_any_score {
                        best_any_score = score;
                        best_any_dw = folded;
                    }
                    let is_local_peak = i > 0
                        && i + 1 < n
                        && curve[i].1 > curve[i - 1].1
                        && curve[i].1 >= curve[i + 1].1;
                    if is_local_peak && score > best_local_score {
                        best_local_score = score;
                        best_local_dw = Some(folded);
                    }
                }
            }
            best_local_dw.unwrap_or(best_any_dw)
        }
        MultiCestScheme::CosModulated { mod_freq_hz, .. } => {
            let sb_ppm = mod_freq_hz / profiles[0].larmor_mhz;
            let mut local_candidates: Vec<f64> = Vec::new();
            let mut any_candidates: Vec<f64> = Vec::new();
            for curve in &depth_curves {
                let n = curve.len();
                for i in 0..n {
                    let delta = curve[i].0;
                    if (delta - sb_ppm).abs() < major_excl_ppm
                        || (delta + sb_ppm).abs() < major_excl_ppm
                    {
                        continue;
                    }
                    let is_local_peak = i > 0
                        && i + 1 < n
                        && curve[i].1 > curve[i - 1].1
                        && curve[i].1 >= curve[i + 1].1;
                    for cand in [delta + sb_ppm, delta - sb_ppm] {
                        if cand.abs() >= 1.0 && cand.abs() <= 15.0 {
                            any_candidates.push(cand);
                            if is_local_peak {
                                local_candidates.push(cand);
                            }
                        }
                    }
                }
            }
            let cands = if local_candidates.is_empty() {
                &any_candidates
            } else {
                &local_candidates
            };
            let mut best_dw = 3.0_f64;
            let mut best_score = -1.0_f64;
            for &dw_c in cands {
                let s_lo = dw_c - sb_ppm;
                let s_hi = dw_c + sb_ppm;
                let lo_ok = (s_lo - sb_ppm).abs() >= major_excl_ppm
                    && (s_lo + sb_ppm).abs() >= major_excl_ppm;
                let hi_ok = (s_hi - sb_ppm).abs() >= major_excl_ppm
                    && (s_hi + sb_ppm).abs() >= major_excl_ppm;
                let mut total = 0.0;
                for c2 in &depth_curves {
                    if lo_ok {
                        if let Some(d) = interp_sorted_pairs(c2, s_lo) {
                            total += d;
                        }
                    }
                    if hi_ok {
                        if let Some(d) = interp_sorted_pairs(c2, s_hi) {
                            total += d;
                        }
                    }
                }
                if total > best_score {
                    best_score = total;
                    best_dw = dw_c;
                }
            }
            best_dw
        }
    };

    Ok(MultiCestFitParams {
        p_b: 0.02,
        k_ex_s_inv: 150.0,
        dw_ab_ppm: dw_est.clamp(-25.0, 25.0),
        r1_s_inv: r1_init,
        r2a_s_inv: r2a_init,
        r2b_s_inv: 20.0,
    })
}

/// Global grid-seeded multi-`B1` Levenberg–Marquardt fit for `DANTE-CEST`, `cos-CEST`, or
/// `CW-CEST` profiles. Scans `dw_ab_ppm` and coarse `(p_B, k_ex)` before launching LM from the
/// top seeds.
pub fn fit_multi_cest_grid_lm(
    profiles: &[MultiCestProfile],
    cfg: &MultiCestLmConfig,
) -> Result<MultiCestFitResult, String> {
    if profiles.is_empty() {
        return Err("fit_multi_cest_grid_lm requires at least one profile".to_string());
    }
    for prof in profiles {
        prof.validate()?;
    }
    let (r1_init, r2a_init) = estimate_baseline_relaxation(profiles);

    let (dw_lo, dw_hi) = if cfg.grid_dw_range_ppm.0 <= cfg.grid_dw_range_ppm.1 {
        cfg.grid_dw_range_ppm
    } else {
        (cfg.grid_dw_range_ppm.1, cfg.grid_dw_range_ppm.0)
    };
    let step = cfg.grid_dw_step_ppm.max(0.15);

    let mut dw_grid = Vec::new();
    let mut x = dw_lo;
    while x <= dw_hi + 1e-9 {
        if x.abs() >= 0.35 {
            dw_grid.push(x);
        }
        x += step;
    }
    if dw_grid.is_empty() {
        dw_grid.push(2.0);
        dw_grid.push(-2.0);
    }

    let pb_grid = [0.012, 0.035];
    let kex_grid = [80.0, 260.0];

    let mut scored: Vec<(f64, MultiCestFitParams)> = Vec::new();
    let mut grid_evals = 0usize;

    for &dw in &dw_grid {
        for &pb in &pb_grid {
            for &kex in &kex_grid {
                let cand = MultiCestFitParams {
                    p_b: pb,
                    k_ex_s_inv: kex,
                    dw_ab_ppm: dw,
                    r1_s_inv: r1_init,
                    r2a_s_inv: r2a_init,
                    r2b_s_inv: 20.0,
                };
                let (_, chi2) = evaluate_residuals_all(profiles, cand)?;
                grid_evals += 1;
                if chi2.is_finite() {
                    scored.push((chi2, cand));
                }
            }
        }
    }

    scored.sort_by(|a, b| a.0.total_cmp(&b.0));

    // Launch LM from the top 2 well-separated dw seeds
    let mut seeds = Vec::new();
    for (_, cand) in &scored {
        if seeds.is_empty()
            || seeds
                .iter()
                .all(|s: &MultiCestFitParams| (s.dw_ab_ppm - cand.dw_ab_ppm).abs() > 0.75)
        {
            seeds.push(*cand);
            if seeds.len() >= 2 {
                break;
            }
        }
    }
    if seeds.is_empty() {
        seeds.push(MultiCestFitParams {
            p_b: 0.02,
            k_ex_s_inv: 150.0,
            dw_ab_ppm: 3.0,
            r1_s_inv: r1_init,
            r2a_s_inv: r2a_init,
            r2b_s_inv: 20.0,
        });
    }

    let mut best: Option<MultiCestFitResult> = None;
    let mut total_evals = grid_evals;
    let mut total_iters = 0usize;

    for seed in seeds {
        let fit = fit_multi_cest_lm(profiles, seed, cfg)?;
        total_evals += fit.n_evals;
        total_iters += fit.iterations;
        let replace = match &best {
            None => true,
            Some(prev) => fit.chi2 < prev.chi2,
        };
        if replace {
            best = Some(fit);
        }
    }

    let mut out = best.ok_or_else(|| "grid-seeded LM failed to produce a fit".to_string())?;
    out.n_evals = total_evals;
    out.iterations = total_iters;
    Ok(out)
}

// ============================================================================
// 4. T22.7: Amortized Neural CEST Posterior Estimator (1D-CNN + MLP)
// ============================================================================

/// Architectural and normalization configuration for [`AmortizedCestNet`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AmortizedCestNetConfig {
    /// Number of multi-`B1` input channels (e.g. `2` for dual-`B1` `CEST`).
    pub n_channels: usize,
    /// Number of uniform resampled relative offset points `delta_ppm = offset - major_ppm`.
    pub n_grid: usize,
    /// Relative offset resampling window `(min_delta_ppm, max_delta_ppm)`.
    pub offset_window_ppm: (f64, f64),
    /// First 1D convolution output channels.
    pub conv1_channels: usize,
    /// Second 1D convolution output channels.
    pub conv2_channels: usize,
    /// 1D convolution kernel size (must be odd `>= 3`).
    pub kernel_size: usize,
    /// Hidden MLP layer width.
    pub hidden_dim: usize,
    /// Target standardization means for `[log10(p_b), log10(k_ex), dw_ppm]`.
    pub target_mean: [f64; 3],
    /// Target standardization standard deviations for `[log10(p_b), log10(k_ex), dw_ppm]`.
    pub target_std: [f64; 3],
    /// Minimum standardized posterior standard deviation floor.
    pub sigma_floor: f64,
}

impl Default for AmortizedCestNetConfig {
    fn default() -> Self {
        Self {
            n_channels: 2,
            n_grid: 48,
            offset_window_ppm: (-9.0, 9.0),
            conv1_channels: 8,
            conv2_channels: 12,
            kernel_size: 5,
            hidden_dim: 48,
            target_mean: [-1.65, 2.15, 0.0],
            target_std: [0.35, 0.40, 4.5],
            sigma_floor: 0.02,
        }
    }
}

impl AmortizedCestNetConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.n_channels == 0 || self.n_grid < 8 {
            return Err("AmortizedCestNet requires n_channels >= 1 and n_grid >= 8".to_string());
        }
        if !self.offset_window_ppm.0.is_finite()
            || !self.offset_window_ppm.1.is_finite()
            || self.offset_window_ppm.1 <= self.offset_window_ppm.0
        {
            return Err("invalid offset_window_ppm".to_string());
        }
        if self.conv1_channels == 0
            || self.conv2_channels == 0
            || self.hidden_dim == 0
            || self.kernel_size < 3
            || self.kernel_size.is_multiple_of(2)
        {
            return Err(
                "invalid conv/MLP layer dimensions (kernel_size must be odd >= 3)".to_string(),
            );
        }
        for j in 0..3 {
            if !self.target_mean[j].is_finite()
                || !self.target_std[j].is_finite()
                || self.target_std[j] <= 0.0
            {
                return Err("target_mean/target_std must be finite with std > 0".to_string());
            }
        }
        if !self.sigma_floor.is_finite() || self.sigma_floor <= 0.0 {
            return Err("sigma_floor must be finite > 0".to_string());
        }
        Ok(())
    }

    #[inline]
    fn l1(&self) -> usize {
        self.n_grid.div_ceil(2)
    }

    #[inline]
    fn l2(&self) -> usize {
        self.l1().div_ceil(2)
    }

    #[inline]
    fn flat_dim(&self) -> usize {
        self.conv2_channels * self.l2()
    }
}

/// Predicted amortized Gaussian posterior for `(log10(p_B), log10(k_ex), dw_ab_ppm)`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct AmortizedCestPosterior {
    pub log10_pb_mean: f64,
    pub log10_pb_std: f64,
    pub log10_kex_mean: f64,
    pub log10_kex_std: f64,
    pub dw_ppm_mean: f64,
    pub dw_ppm_std: f64,
}

impl AmortizedCestPosterior {
    /// Convert posterior mean into physical `(p_B, k_ex, dw_ppm)` values clamped to valid bounds.
    pub fn physical_mean(&self) -> (f64, f64, f64) {
        let pb = 10.0_f64.powf(self.log10_pb_mean.clamp(-3.5, -0.35));
        let kex = 10.0_f64.powf(self.log10_kex_mean.clamp(0.5, 3.6));
        let dw = self.dw_ppm_mean.clamp(-25.0, 25.0);
        (pb, kex, dw)
    }
}

#[inline]
fn sigmoid_safe(x: f64) -> f64 {
    if x >= 0.0 {
        let e = (-x).exp();
        1.0 / (1.0 + e)
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

#[inline]
fn silu(x: f64) -> f64 {
    x * sigmoid_safe(x)
}

#[inline]
fn silu_deriv(x: f64) -> f64 {
    let s = sigmoid_safe(x);
    s * (1.0 + x * (1.0 - s))
}

#[inline]
fn softplus_safe(x: f64) -> f64 {
    if x > 25.0 {
        x + (-x).exp()
    } else if x < -25.0 {
        x.exp()
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// Resample a single `(offsets_ppm, intensities)` profile onto a uniform relative offset grid
/// `delta_ppm in [win_lo, win_hi]` via piecewise-linear interpolation, encoded as saturation
/// depth `1.0 - intensity` (so dips become positive localized features).
pub fn resample_single_profile(
    offsets_ppm: &[f64],
    intensities: &[f64],
    major_ppm: f64,
    n_grid: usize,
    window_ppm: (f64, f64),
) -> Result<Vec<f64>, String> {
    if offsets_ppm.is_empty() || offsets_ppm.len() != intensities.len() {
        return Err("resample_single_profile requires non-empty equal-length slices".to_string());
    }
    if n_grid < 2 {
        return Err("resample_single_profile requires n_grid >= 2".to_string());
    }
    let mut pairs: Vec<(f64, f64)> = offsets_ppm
        .iter()
        .zip(intensities.iter())
        .map(|(&o, &y)| (o - major_ppm, y))
        .collect();
    for &(dx, y) in &pairs {
        if !dx.is_finite() || !y.is_finite() {
            return Err("non-finite offset or intensity in resample_single_profile".to_string());
        }
    }
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));

    let (w_lo, w_hi) = window_ppm;
    let step = (w_hi - w_lo) / ((n_grid - 1) as f64);
    let mut out = Vec::with_capacity(n_grid);
    let mut cursor = 0usize;

    for g in 0..n_grid {
        let q = w_lo + (g as f64) * step;
        let val = if q <= pairs[0].0 {
            pairs[0].1
        } else if q >= pairs[pairs.len() - 1].0 {
            pairs[pairs.len() - 1].1
        } else {
            while cursor + 1 < pairs.len() && pairs[cursor + 1].0 < q {
                cursor += 1;
            }
            let (x0, y0) = pairs[cursor];
            let (x1, y1) = pairs[cursor + 1];
            let dx = x1 - x0;
            if dx.abs() < 1e-12 {
                0.5 * (y0 + y1)
            } else {
                let t = (q - x0) / dx;
                y0 + t * (y1 - y0)
            }
        };
        out.push(1.0 - val);
    }
    Ok(out)
}

/// Resample multi-`B1` `CEST` profiles into a flattened `[n_channels * n_grid]` input tensor
/// sorted by ascending effective `B1` field.
pub fn resample_profiles_for_net(
    profiles: &[MultiCestProfile],
    cfg: &AmortizedCestNetConfig,
) -> Result<Vec<f64>, String> {
    cfg.validate()?;
    if profiles.is_empty() {
        return Err("resample_profiles_for_net requires at least one profile".to_string());
    }
    let mut order: Vec<usize> = (0..profiles.len()).collect();
    order.sort_by(|&a, &b| {
        profiles[a]
            .scheme
            .effective_b1_hz()
            .total_cmp(&profiles[b].scheme.effective_b1_hz())
    });

    let mut tensor = Vec::with_capacity(cfg.n_channels * cfg.n_grid);
    for ch in 0..cfg.n_channels {
        let prof_idx = order[ch.min(order.len() - 1)];
        let prof = &profiles[prof_idx];
        let resampled = resample_single_profile(
            &prof.offsets_ppm,
            &prof.intensities,
            prof.major_ppm,
            cfg.n_grid,
            cfg.offset_window_ppm,
        )?;
        tensor.extend_from_slice(&resampled);
    }
    Ok(tensor)
}

/// Pure-Rust compact `1D-CNN + MLP` amortized heteroscedastic Gaussian posterior estimator for
/// `(log10(p_B), log10(k_ex), dw_ab_ppm)`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AmortizedCestNet {
    pub cfg: AmortizedCestNetConfig,
    /// Conv1 weights `[conv1_channels * n_channels * kernel_size]`.
    pub conv1_w: Vec<f64>,
    pub conv1_b: Vec<f64>,
    /// Conv2 weights `[conv2_channels * conv1_channels * kernel_size]`.
    pub conv2_w: Vec<f64>,
    pub conv2_b: Vec<f64>,
    /// FC1 weights `[hidden_dim * flat_dim]`.
    pub fc1_w: Vec<f64>,
    pub fc1_b: Vec<f64>,
    /// FC2 weights `[6 * hidden_dim]` -> `[mu_0..2, raw_s_0..2]`.
    pub fc2_w: Vec<f64>,
    pub fc2_b: Vec<f64>,
}

/// Parameter gradient bundle matching [`AmortizedCestNet`].
#[derive(Clone, Debug)]
pub struct AmortizedCestGrads {
    pub conv1_w: Vec<f64>,
    pub conv1_b: Vec<f64>,
    pub conv2_w: Vec<f64>,
    pub conv2_b: Vec<f64>,
    pub fc1_w: Vec<f64>,
    pub fc1_b: Vec<f64>,
    pub fc2_w: Vec<f64>,
    pub fc2_b: Vec<f64>,
}

impl AmortizedCestGrads {
    fn zeros_like(net: &AmortizedCestNet) -> Self {
        Self {
            conv1_w: vec![0.0; net.conv1_w.len()],
            conv1_b: vec![0.0; net.conv1_b.len()],
            conv2_w: vec![0.0; net.conv2_w.len()],
            conv2_b: vec![0.0; net.conv2_b.len()],
            fc1_w: vec![0.0; net.fc1_w.len()],
            fc1_b: vec![0.0; net.fc1_b.len()],
            fc2_w: vec![0.0; net.fc2_w.len()],
            fc2_b: vec![0.0; net.fc2_b.len()],
        }
    }

    /// Flatten all gradients in canonical parameter order for finite-difference verification.
    pub fn as_flat_vec(&self) -> Vec<f64> {
        let mut v = Vec::with_capacity(
            self.conv1_w.len()
                + self.conv1_b.len()
                + self.conv2_w.len()
                + self.conv2_b.len()
                + self.fc1_w.len()
                + self.fc1_b.len()
                + self.fc2_w.len()
                + self.fc2_b.len(),
        );
        v.extend_from_slice(&self.conv1_w);
        v.extend_from_slice(&self.conv1_b);
        v.extend_from_slice(&self.conv2_w);
        v.extend_from_slice(&self.conv2_b);
        v.extend_from_slice(&self.fc1_w);
        v.extend_from_slice(&self.fc1_b);
        v.extend_from_slice(&self.fc2_w);
        v.extend_from_slice(&self.fc2_b);
        v
    }
}

struct ForwardCache {
    c1_pre: Vec<f64>,
    c1_act: Vec<f64>,
    c2_pre: Vec<f64>,
    c2_act: Vec<f64>,
    h_pre: Vec<f64>,
    h_act: Vec<f64>,
    out_raw: [f64; 6],
}

impl AmortizedCestNet {
    /// Initialize a deterministic He/Xavier-scaled [`AmortizedCestNet`] using `seed`.
    pub fn new(cfg: AmortizedCestNetConfig, seed: u64) -> Result<Self, String> {
        cfg.validate()?;
        let mut rng = Rng::new(seed);
        let mut sample_w = |n: usize, fan_in: usize| -> Vec<f64> {
            let sd = (2.0 / (fan_in.max(1) as f64)).sqrt() * 0.7;
            (0..n).map(|_| rng.normal(0.0, sd)).collect()
        };

        let c1_w = sample_w(
            cfg.conv1_channels * cfg.n_channels * cfg.kernel_size,
            cfg.n_channels * cfg.kernel_size,
        );
        let c1_b = vec![0.0; cfg.conv1_channels];
        let c2_w = sample_w(
            cfg.conv2_channels * cfg.conv1_channels * cfg.kernel_size,
            cfg.conv1_channels * cfg.kernel_size,
        );
        let c2_b = vec![0.0; cfg.conv2_channels];
        let flat = cfg.flat_dim();
        let fc1_w = sample_w(cfg.hidden_dim * flat, flat);
        let fc1_b = vec![0.0; cfg.hidden_dim];
        let fc2_w = sample_w(6 * cfg.hidden_dim, cfg.hidden_dim);
        // Initialize scale logits near 0 -> softplus(0) = ln(2) ~ 0.69
        let fc2_b = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        Ok(Self {
            cfg,
            conv1_w: c1_w,
            conv1_b: c1_b,
            conv2_w: c2_w,
            conv2_b: c2_b,
            fc1_w,
            fc1_b,
            fc2_w,
            fc2_b,
        })
    }

    /// Total number of trainable scalar parameters.
    pub fn num_params(&self) -> usize {
        self.conv1_w.len()
            + self.conv1_b.len()
            + self.conv2_w.len()
            + self.conv2_b.len()
            + self.fc1_w.len()
            + self.fc1_b.len()
            + self.fc2_w.len()
            + self.fc2_b.len()
    }

    /// Get a mutable reference to parameter `idx` in canonical flat order (for FD gradient checks).
    pub fn param_mut(&mut self, mut idx: usize) -> Option<&mut f64> {
        let slices: [&mut [f64]; 8] = [
            &mut self.conv1_w,
            &mut self.conv1_b,
            &mut self.conv2_w,
            &mut self.conv2_b,
            &mut self.fc1_w,
            &mut self.fc1_b,
            &mut self.fc2_w,
            &mut self.fc2_b,
        ];
        for s in slices {
            if idx < s.len() {
                return Some(&mut s[idx]);
            }
            idx -= s.len();
        }
        None
    }

    fn forward_with_cache(&self, x: &[f64]) -> Result<ForwardCache, String> {
        let expected = self.cfg.n_channels * self.cfg.n_grid;
        if x.len() != expected {
            return Err(format!(
                "AmortizedCestNet expected input length {expected}, got {}",
                x.len()
            ));
        }
        let k_sz = self.cfg.kernel_size;
        let pad = (k_sz / 2) as isize;

        // Layer 1: Conv1D (stride 2) + SiLU
        let l0 = self.cfg.n_grid;
        let l1 = self.cfg.l1();
        let c0 = self.cfg.n_channels;
        let c1 = self.cfg.conv1_channels;
        let mut c1_pre = vec![0.0; c1 * l1];
        let mut c1_act = vec![0.0; c1 * l1];

        for oc in 0..c1 {
            let b = self.conv1_b[oc];
            for pos in 0..l1 {
                let mut acc = b;
                let base_in = (pos * 2) as isize - pad;
                for ic in 0..c0 {
                    let w_base = (oc * c0 + ic) * k_sz;
                    let x_base = ic * l0;
                    for kk in 0..k_sz {
                        let ip = base_in + (kk as isize);
                        if ip >= 0 && (ip as usize) < l0 {
                            acc += self.conv1_w[w_base + kk] * x[x_base + (ip as usize)];
                        }
                    }
                }
                let out_idx = oc * l1 + pos;
                c1_pre[out_idx] = acc;
                c1_act[out_idx] = silu(acc);
            }
        }

        // Layer 2: Conv1D (stride 2) + SiLU
        let l2 = self.cfg.l2();
        let c2 = self.cfg.conv2_channels;
        let mut c2_pre = vec![0.0; c2 * l2];
        let mut c2_act = vec![0.0; c2 * l2];

        for oc in 0..c2 {
            let b = self.conv2_b[oc];
            for pos in 0..l2 {
                let mut acc = b;
                let base_in = (pos * 2) as isize - pad;
                for ic in 0..c1 {
                    let w_base = (oc * c1 + ic) * k_sz;
                    let in_base = ic * l1;
                    for kk in 0..k_sz {
                        let ip = base_in + (kk as isize);
                        if ip >= 0 && (ip as usize) < l1 {
                            acc += self.conv2_w[w_base + kk] * c1_act[in_base + (ip as usize)];
                        }
                    }
                }
                let out_idx = oc * l2 + pos;
                c2_pre[out_idx] = acc;
                c2_act[out_idx] = silu(acc);
            }
        }

        // Layer 3: FC1 + SiLU
        let flat = c2 * l2;
        let h_dim = self.cfg.hidden_dim;
        let mut h_pre = vec![0.0; h_dim];
        let mut h_act = vec![0.0; h_dim];
        for h in 0..h_dim {
            let mut acc = self.fc1_b[h];
            let w_row = h * flat;
            for f in 0..flat {
                acc += self.fc1_w[w_row + f] * c2_act[f];
            }
            h_pre[h] = acc;
            h_act[h] = silu(acc);
        }

        // Layer 4: FC2 -> [mu_0..2, s_0..2]
        let mut out_raw = [0.0; 6];
        for o in 0..6 {
            let mut acc = self.fc2_b[o];
            let w_row = o * h_dim;
            for h in 0..h_dim {
                acc += self.fc2_w[w_row + h] * h_act[h];
            }
            out_raw[o] = acc;
        }

        Ok(ForwardCache {
            c1_pre,
            c1_act,
            c2_pre,
            c2_act,
            h_pre,
            h_act,
            out_raw,
        })
    }

    /// Predict the amortized Gaussian posterior `(mu, sigma)` for `(log10(p_B), log10(k_ex), dw_ppm)`.
    pub fn predict_posterior(
        &self,
        input_tensor: &[f64],
    ) -> Result<AmortizedCestPosterior, String> {
        let cache = self.forward_with_cache(input_tensor)?;
        let mut mu_phys = [0.0; 3];
        let mut sd_phys = [0.0; 3];
        for j in 0..3 {
            let mu_std = cache.out_raw[j];
            let sd_std = self.cfg.sigma_floor + softplus_safe(cache.out_raw[3 + j]);
            mu_phys[j] = self.cfg.target_mean[j] + self.cfg.target_std[j] * mu_std;
            sd_phys[j] = self.cfg.target_std[j] * sd_std;
        }
        Ok(AmortizedCestPosterior {
            log10_pb_mean: mu_phys[0],
            log10_pb_std: sd_phys[0],
            log10_kex_mean: mu_phys[1],
            log10_kex_std: sd_phys[1],
            dw_ppm_mean: mu_phys[2],
            dw_ppm_std: sd_phys[2],
        })
    }

    /// Evaluate heteroscedastic Gaussian negative log-likelihood loss on a single sample
    /// `(input_tensor, target_phys = [log10_pb, log10_kex, dw_ppm])`.
    pub fn nll_loss_single(
        &self,
        input_tensor: &[f64],
        target_phys: [f64; 3],
    ) -> Result<f64, String> {
        let cache = self.forward_with_cache(input_tensor)?;
        let mut loss = 0.0;
        let half_ln_2pi = 0.5 * (TAU).ln();
        for j in 0..3 {
            let y_std = (target_phys[j] - self.cfg.target_mean[j]) / self.cfg.target_std[j];
            let mu = cache.out_raw[j];
            let sigma = self.cfg.sigma_floor + softplus_safe(cache.out_raw[3 + j]);
            let diff = y_std - mu;
            loss += 0.5 * (diff * diff) / (sigma * sigma) + sigma.ln() + half_ln_2pi;
        }
        Ok(loss / 3.0)
    }

    /// Compute heteroscedastic Gaussian NLL loss and exact analytical gradients via backpropagation
    /// on a single sample, accumulating gradients into `grads` with weight `batch_scale`.
    pub fn backprop_single_accumulate(
        &self,
        input_tensor: &[f64],
        target_phys: [f64; 3],
        batch_scale: f64,
        grads: &mut AmortizedCestGrads,
    ) -> Result<f64, String> {
        let cache = self.forward_with_cache(input_tensor)?;
        let half_ln_2pi = 0.5 * TAU.ln();

        let mut d_out = [0.0; 6];
        let mut loss = 0.0;
        for j in 0..3 {
            let y_std = (target_phys[j] - self.cfg.target_mean[j]) / self.cfg.target_std[j];
            let mu = cache.out_raw[j];
            let s_raw = cache.out_raw[3 + j];
            let sigma = self.cfg.sigma_floor + softplus_safe(s_raw);
            let diff = mu - y_std;
            let inv_sig = 1.0 / sigma;
            let inv_sig2 = inv_sig * inv_sig;

            loss += 0.5 * (diff * diff) * inv_sig2 + sigma.ln() + half_ln_2pi;

            let w = batch_scale / 3.0;
            d_out[j] = w * diff * inv_sig2;
            let d_sigma = w * (inv_sig - (diff * diff) * inv_sig2 * inv_sig);
            d_out[3 + j] = d_sigma * sigmoid_safe(s_raw);
        }

        // Backprop FC2
        let h_dim = self.cfg.hidden_dim;
        let mut d_h_act = vec![0.0; h_dim];
        for o in 0..6 {
            let go = d_out[o];
            grads.fc2_b[o] += go;
            let w_row = o * h_dim;
            for h in 0..h_dim {
                grads.fc2_w[w_row + h] += go * cache.h_act[h];
                d_h_act[h] += go * self.fc2_w[w_row + h];
            }
        }

        // Backprop FC1 + SiLU
        let l2 = self.cfg.l2();
        let c2 = self.cfg.conv2_channels;
        let flat = c2 * l2;
        let mut d_c2_act = vec![0.0; flat];
        for h in 0..h_dim {
            let gh = d_h_act[h] * silu_deriv(cache.h_pre[h]);
            grads.fc1_b[h] += gh;
            let w_row = h * flat;
            for f in 0..flat {
                grads.fc1_w[w_row + f] += gh * cache.c2_act[f];
                d_c2_act[f] += gh * self.fc1_w[w_row + f];
            }
        }

        // Backprop Conv2 + SiLU
        let k_sz = self.cfg.kernel_size;
        let pad = (k_sz / 2) as isize;
        let l1 = self.cfg.l1();
        let c1 = self.cfg.conv1_channels;
        let mut d_c1_act = vec![0.0; c1 * l1];

        for oc in 0..c2 {
            for pos in 0..l2 {
                let out_idx = oc * l2 + pos;
                let g = d_c2_act[out_idx] * silu_deriv(cache.c2_pre[out_idx]);
                grads.conv2_b[oc] += g;
                let base_in = (pos * 2) as isize - pad;
                for ic in 0..c1 {
                    let w_base = (oc * c1 + ic) * k_sz;
                    let in_base = ic * l1;
                    for kk in 0..k_sz {
                        let ip = base_in + (kk as isize);
                        if ip >= 0 && (ip as usize) < l1 {
                            let i_idx = in_base + (ip as usize);
                            grads.conv2_w[w_base + kk] += g * cache.c1_act[i_idx];
                            d_c1_act[i_idx] += g * self.conv2_w[w_base + kk];
                        }
                    }
                }
            }
        }

        // Backprop Conv1 + SiLU
        let l0 = self.cfg.n_grid;
        let c0 = self.cfg.n_channels;
        for oc in 0..c1 {
            for pos in 0..l1 {
                let out_idx = oc * l1 + pos;
                let g = d_c1_act[out_idx] * silu_deriv(cache.c1_pre[out_idx]);
                grads.conv1_b[oc] += g;
                let base_in = (pos * 2) as isize - pad;
                for ic in 0..c0 {
                    let w_base = (oc * c0 + ic) * k_sz;
                    let x_base = ic * l0;
                    for kk in 0..k_sz {
                        let ip = base_in + (kk as isize);
                        if ip >= 0 && (ip as usize) < l0 {
                            grads.conv1_w[w_base + kk] += g * input_tensor[x_base + (ip as usize)];
                        }
                    }
                }
            }
        }

        Ok(loss / 3.0)
    }

    /// Compute NLL loss and exact analytical gradients over a batch of `(input, target_phys)` pairs.
    pub fn batch_loss_and_grads(
        &self,
        batch: &[CestTrainingSample],
        weight_decay: f64,
    ) -> Result<(f64, AmortizedCestGrads), String> {
        if batch.is_empty() {
            return Err("batch_loss_and_grads requires a non-empty batch".to_string());
        }
        let mut grads = AmortizedCestGrads::zeros_like(self);
        let scale = 1.0 / (batch.len() as f64);
        let mut total_loss = 0.0;
        for sample in batch {
            total_loss += self.backprop_single_accumulate(
                &sample.input_tensor,
                sample.target_phys,
                scale,
                &mut grads,
            )?;
        }
        let mut loss = total_loss * scale;
        if weight_decay > 0.0 {
            let mut l2_sum = 0.0;
            for (w, g) in [
                (&self.conv1_w, &mut grads.conv1_w),
                (&self.conv2_w, &mut grads.conv2_w),
                (&self.fc1_w, &mut grads.fc1_w),
                (&self.fc2_w, &mut grads.fc2_w),
            ] {
                for (wi, gi) in w.iter().zip(g.iter_mut()) {
                    l2_sum += wi * wi;
                    *gi += weight_decay * wi;
                }
            }
            loss += 0.5 * weight_decay * l2_sum;
        }
        Ok((loss, grads))
    }
}

/// One labeled synthetic Bloch–McConnell `CEST` training/validation example.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CestTrainingSample {
    /// Resampled multi-`B1` saturation-depth tensor `[n_channels * n_grid]`.
    pub input_tensor: Vec<f64>,
    /// Ground-truth `[log10(p_b), log10(k_ex), dw_ab_ppm]`.
    pub target_phys: [f64; 3],
}

/// Specification for generating synthetic Bloch–McConnell `CEST` training ensembles.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CestEnsembleGenConfig {
    pub larmor_mhz: f64,
    pub major_ppm: f64,
    pub schemes: Vec<MultiCestScheme>,
    pub normalization: CestNormalization,
    pub b1_inhomogeneity_frac: f64,
    pub n_b1_quad: usize,
    pub log10_pb_range: (f64, f64),
    pub log10_kex_range: (f64, f64),
    pub dw_abs_range_ppm: (f64, f64),
    pub r1_range: (f64, f64),
    pub r2a_range: (f64, f64),
    pub r2b_range: (f64, f64),
    pub noise_sigma: f64,
}

/// Generate an ensemble of `n_samples` synthetic multi-`B1` Bloch–McConnell `CEST` training
/// examples for [`AmortizedCestNet`].
pub fn generate_cest_training_ensemble(
    net_cfg: &AmortizedCestNetConfig,
    gen_cfg: &CestEnsembleGenConfig,
    n_samples: usize,
    seed: u64,
) -> Result<Vec<CestTrainingSample>, String> {
    net_cfg.validate()?;
    if gen_cfg.schemes.is_empty() {
        return Err("CestEnsembleGenConfig requires at least one scheme".to_string());
    }
    for s in &gen_cfg.schemes {
        s.validate()?;
    }

    let (w_lo, w_hi) = net_cfg.offset_window_ppm;
    let step = (w_hi - w_lo) / ((net_cfg.n_grid - 1) as f64);
    let offsets_ppm: Vec<f64> = (0..net_cfg.n_grid)
        .map(|i| gen_cfg.major_ppm + w_lo + (i as f64) * step)
        .collect();

    let mut rng = Rng::new(seed);
    let mut samples = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let log10_pb = rng.uniform(gen_cfg.log10_pb_range.0, gen_cfg.log10_pb_range.1);
        let log10_kex = rng.uniform(gen_cfg.log10_kex_range.0, gen_cfg.log10_kex_range.1);
        let dw_mag = rng.uniform(gen_cfg.dw_abs_range_ppm.0, gen_cfg.dw_abs_range_ppm.1);
        let dw = if rng.f64() < 0.5 { dw_mag } else { -dw_mag };
        let r1 = rng.uniform(gen_cfg.r1_range.0, gen_cfg.r1_range.1);
        let r2a = rng.uniform(gen_cfg.r2a_range.0, gen_cfg.r2a_range.1);
        let r2b = rng.uniform(gen_cfg.r2b_range.0, gen_cfg.r2b_range.1);

        let p_b = 10.0_f64.powf(log10_pb);
        let k_ex = 10.0_f64.powf(log10_kex);

        let mut profiles = Vec::with_capacity(gen_cfg.schemes.len());
        for scheme in &gen_cfg.schemes {
            let sim_p = MultiCestSimParams {
                p_b,
                k_ex_s_inv: k_ex,
                major_ppm: gen_cfg.major_ppm,
                dw_ab_ppm: dw,
                r1a_s_inv: r1,
                r2a_s_inv: r2a,
                r1b_s_inv: r1,
                r2b_s_inv: r2b,
                larmor_mhz: gen_cfg.larmor_mhz,
                b1_inhomogeneity_frac: gen_cfg.b1_inhomogeneity_frac,
                n_b1_quad: gen_cfg.n_b1_quad,
                normalization: gen_cfg.normalization,
                scheme: scheme.clone(),
            };
            let mut intensities = simulate_multi_cest_profile(&sim_p, &offsets_ppm)?;
            if gen_cfg.noise_sigma > 0.0 {
                for y in &mut intensities {
                    *y += rng.normal(0.0, gen_cfg.noise_sigma);
                }
            }
            profiles.push(MultiCestProfile {
                larmor_mhz: gen_cfg.larmor_mhz,
                major_ppm: gen_cfg.major_ppm,
                b1_inhomogeneity_frac: gen_cfg.b1_inhomogeneity_frac,
                n_b1_quad: gen_cfg.n_b1_quad,
                normalization: gen_cfg.normalization,
                scheme: scheme.clone(),
                offsets_ppm: offsets_ppm.clone(),
                intensities,
                uncertainties: vec![gen_cfg.noise_sigma.max(0.005); offsets_ppm.len()],
            });
        }
        let input_tensor = resample_profiles_for_net(&profiles, net_cfg)?;
        samples.push(CestTrainingSample {
            input_tensor,
            target_phys: [log10_pb, log10_kex, dw],
        });
    }

    Ok(samples)
}

/// Hyperparameters for Adam online backpropagation training of [`AmortizedCestNet`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AmortizedTrainConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub grad_clip_norm: f64,
    pub seed: u64,
}

impl Default for AmortizedTrainConfig {
    fn default() -> Self {
        Self {
            epochs: 60,
            batch_size: 16,
            learning_rate: 3e-3,
            weight_decay: 1e-4,
            grad_clip_norm: 5.0,
            seed: 42,
        }
    }
}

/// Summary metrics from online backpropagation training of [`AmortizedCestNet`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AmortizedTrainReport {
    pub initial_nll: f64,
    pub final_nll: f64,
    pub epochs_completed: usize,
    pub updates_completed: usize,
}

/// Train an [`AmortizedCestNet`] with real analytical backpropagation and Adam optimization
/// under heteroscedastic Gaussian negative log-likelihood loss.
pub fn train_amortized_cest_net(
    net: &mut AmortizedCestNet,
    dataset: &[CestTrainingSample],
    cfg: &AmortizedTrainConfig,
) -> Result<AmortizedTrainReport, String> {
    if dataset.is_empty() {
        return Err("train_amortized_cest_net requires non-empty dataset".to_string());
    }
    if cfg.epochs == 0
        || cfg.batch_size == 0
        || !cfg.learning_rate.is_finite()
        || cfg.learning_rate <= 0.0
    {
        return Err("invalid AmortizedTrainConfig hyperparameters".to_string());
    }

    let (initial_nll, _) = net.batch_loss_and_grads(dataset, 0.0)?;
    let n_params = net.num_params();
    let mut m_vec = vec![0.0; n_params];
    let mut v_vec = vec![0.0; n_params];
    let beta1 = 0.9_f64;
    let beta2 = 0.999_f64;
    let eps = 1e-8_f64;

    let mut rng = Rng::new(cfg.seed);
    let mut indices: Vec<usize> = (0..dataset.len()).collect();
    let mut step = 0usize;

    for epoch in 0..cfg.epochs {
        rng.shuffle(&mut indices);
        // Smooth cosine learning-rate decay from lr to 0.1 * lr
        let progress = (epoch as f64) / (cfg.epochs.max(1) as f64);
        let lr = cfg.learning_rate * (0.1 + 0.45 * (1.0 + (PI * progress).cos()));

        for chunk in indices.chunks(cfg.batch_size) {
            let batch: Vec<CestTrainingSample> =
                chunk.iter().map(|&i| dataset[i].clone()).collect();
            let (_, grads) = net.batch_loss_and_grads(&batch, cfg.weight_decay)?;
            let mut g_flat = grads.as_flat_vec();

            if cfg.grad_clip_norm > 0.0 {
                let norm: f64 = g_flat.iter().map(|v| v * v).sum::<f64>().sqrt();
                if norm > cfg.grad_clip_norm && norm.is_finite() {
                    let s = cfg.grad_clip_norm / norm;
                    for gi in &mut g_flat {
                        *gi *= s;
                    }
                }
            }

            step += 1;
            let bc1 = 1.0 - beta1.powi(step as i32);
            let bc2 = 1.0 - beta2.powi(step as i32);

            let mut p_idx = 0usize;
            for slice in [
                &mut net.conv1_w[..],
                &mut net.conv1_b[..],
                &mut net.conv2_w[..],
                &mut net.conv2_b[..],
                &mut net.fc1_w[..],
                &mut net.fc1_b[..],
                &mut net.fc2_w[..],
                &mut net.fc2_b[..],
            ] {
                for w in slice.iter_mut() {
                    let g = g_flat[p_idx];
                    m_vec[p_idx] = beta1 * m_vec[p_idx] + (1.0 - beta1) * g;
                    v_vec[p_idx] = beta2 * v_vec[p_idx] + (1.0 - beta2) * g * g;
                    let m_hat = m_vec[p_idx] / bc1;
                    let v_hat = v_vec[p_idx] / bc2;
                    *w -= lr * m_hat / (v_hat.sqrt() + eps);
                    p_idx += 1;
                }
            }
        }
    }

    let (final_nll, _) = net.batch_loss_and_grads(dataset, 0.0)?;
    Ok(AmortizedTrainReport {
        initial_nll,
        final_nll,
        epochs_completed: cfg.epochs,
        updates_completed: step,
    })
}

// ============================================================================
// 5. Hybrid Amortized-Seeded LM Pipeline with Abstention Fallback
// ============================================================================

/// Outcome diagnostic of the neural-amortized seeding stage in [`fit_cest_amortized_seeded_lm`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AmortizedSeedStatus {
    /// Neural amortized posterior mean (plus `+-1sigma` multi-start) converged to a physically
    /// acceptable reduced `chi^2` without requiring full grid search.
    AcceptedNeuralSeed,
    /// Neural seed produced poor reduced `chi^2` or excessive posterior uncertainty; cleanly
    /// abstained and fell back to exact physical grid-search + Levenberg–Marquardt.
    AbstainToGridFallback,
}

/// Configuration for the hybrid amortized-seeded Bloch–McConnell Levenberg–Marquardt fitter.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AmortizedHybridFitConfig {
    pub lm_cfg: MultiCestLmConfig,
    /// Maximum reduced `chi^2` allowed to accept the neural-seeded LM fit without fallback.
    pub fallback_reduced_chi2_threshold: f64,
    /// Maximum number of `+-1sigma` multi-start seeds (in addition to posterior mean) to refine
    /// with LM if the posterior mean alone has not already reached `reduced_chi2 <= early_accept_chi2`.
    pub max_sigma_multistarts: usize,
    /// If the posterior-mean LM fit achieves `reduced_chi2 <= early_accept_chi2` and converges,
    /// skip extra `+-1sigma` LM runs for maximum speed.
    pub early_accept_reduced_chi2: f64,
    /// Optional uncertainty abstention cap on `dw_ppm_std` (`ppm`); above this, also trigger fallback.
    pub max_dw_std_abstain_ppm: f64,
}

impl Default for AmortizedHybridFitConfig {
    fn default() -> Self {
        Self {
            lm_cfg: MultiCestLmConfig::default(),
            fallback_reduced_chi2_threshold: 2.5,
            max_sigma_multistarts: 2,
            early_accept_reduced_chi2: 1.5,
            max_dw_std_abstain_ppm: 6.0,
        }
    }
}

/// Complete output of [`fit_cest_amortized_seeded_lm`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AmortizedHybridFitResult {
    pub posterior: AmortizedCestPosterior,
    pub status: AmortizedSeedStatus,
    pub fit: MultiCestFitResult,
    /// Best reduced `chi^2` achieved by the neural-seeded LM attempts prior to any fallback.
    pub neural_seeded_reduced_chi2: f64,
    /// Number of LM seeds evaluated from the neural posterior.
    pub neural_seeds_refined: usize,
}

/// Hybrid amortized-seeded multi-`B1` Bloch–McConnell `CEST` fitter (`T22.7`):
/// 1. Predicts the Gaussian posterior `(mu, sigma)` for `(log10(p_B), log10(k_ex), dw_ppm)` via
///    [`AmortizedCestNet`].
/// 2. Generates the posterior-mean seed plus `+-1sigma` multi-start perturbation seeds.
/// 3. Runs exact multi-`B1` Bloch–McConnell Levenberg–Marquardt optimization, scoring strictly
///    with the exact physical likelihood (`chi^2`).
/// 4. If the neural seed yields poor `chi^2_nu` (`> fallback_reduced_chi2_threshold`) or
///    excessive uncertainty, abstains cleanly to [`fit_multi_cest_grid_lm`] with
///    [`AmortizedSeedStatus::AbstainToGridFallback`].
pub fn fit_cest_amortized_seeded_lm(
    net: &AmortizedCestNet,
    profiles: &[MultiCestProfile],
    cfg: &AmortizedHybridFitConfig,
) -> Result<AmortizedHybridFitResult, String> {
    if profiles.is_empty() {
        return Err("fit_cest_amortized_seeded_lm requires at least one profile".to_string());
    }
    for prof in profiles {
        prof.validate()?;
    }

    let input_tensor = resample_profiles_for_net(profiles, &net.cfg)?;
    let posterior = net.predict_posterior(&input_tensor)?;
    let (r1_init, r2a_init) = estimate_baseline_relaxation(profiles);

    let make_seed = |l10_pb: f64, l10_kex: f64, dw: f64| -> MultiCestFitParams {
        MultiCestFitParams {
            p_b: 10.0_f64.powf(l10_pb.clamp(-3.5, -0.35)),
            k_ex_s_inv: 10.0_f64.powf(l10_kex.clamp(0.5, 3.6)),
            dw_ab_ppm: dw.clamp(-25.0, 25.0),
            r1_s_inv: r1_init,
            r2a_s_inv: r2a_init,
            r2b_s_inv: 20.0,
        }
    };

    let mean_seed = make_seed(
        posterior.log10_pb_mean,
        posterior.log10_kex_mean,
        posterior.dw_ppm_mean,
    );

    // Run LM from posterior mean first
    let mut best_neural = fit_multi_cest_lm(profiles, mean_seed, &cfg.lm_cfg)?;
    let mut total_evals = best_neural.n_evals;
    let mut total_iters = best_neural.iterations;
    let mut neural_seeds_refined = 1usize;

    // If mean seed didn't already hit tight physical chi^2_nu, score +-1sigma multi-start seeds
    if !(best_neural.converged && best_neural.reduced_chi2 <= cfg.early_accept_reduced_chi2)
        && cfg.max_sigma_multistarts > 0
    {
        let s_pb = posterior.log10_pb_std.clamp(0.05, 0.7);
        let s_kex = posterior.log10_kex_std.clamp(0.05, 0.7);
        let s_dw = posterior.dw_ppm_std.clamp(0.15, 3.0);

        let sigma_candidates = [
            make_seed(
                posterior.log10_pb_mean,
                posterior.log10_kex_mean,
                posterior.dw_ppm_mean + s_dw,
            ),
            make_seed(
                posterior.log10_pb_mean,
                posterior.log10_kex_mean,
                posterior.dw_ppm_mean - s_dw,
            ),
            make_seed(
                posterior.log10_pb_mean + s_pb,
                posterior.log10_kex_mean - s_kex,
                posterior.dw_ppm_mean,
            ),
            make_seed(
                posterior.log10_pb_mean - s_pb,
                posterior.log10_kex_mean + s_kex,
                posterior.dw_ppm_mean,
            ),
            make_seed(
                posterior.log10_pb_mean + s_pb,
                posterior.log10_kex_mean + s_kex,
                posterior.dw_ppm_mean,
            ),
            make_seed(
                posterior.log10_pb_mean - s_pb,
                posterior.log10_kex_mean - s_kex,
                posterior.dw_ppm_mean,
            ),
        ];

        let mut ranked: Vec<(f64, MultiCestFitParams)> = Vec::with_capacity(sigma_candidates.len());
        for cand in sigma_candidates {
            let (_, chi2) = evaluate_residuals_all(profiles, cand)?;
            total_evals += 1;
            if chi2.is_finite() {
                ranked.push((chi2, cand));
            }
        }
        ranked.sort_by(|a, b| a.0.total_cmp(&b.0));

        for (_, cand) in ranked.into_iter().take(cfg.max_sigma_multistarts) {
            let fit = fit_multi_cest_lm(profiles, cand, &cfg.lm_cfg)?;
            total_evals += fit.n_evals;
            total_iters += fit.iterations;
            neural_seeds_refined += 1;
            if fit.chi2 < best_neural.chi2 {
                best_neural = fit;
            }
            if best_neural.converged && best_neural.reduced_chi2 <= cfg.early_accept_reduced_chi2 {
                break;
            }
        }
    }

    let neural_seeded_reduced_chi2 = best_neural.reduced_chi2;
    let should_abstain = !best_neural.converged
        || !neural_seeded_reduced_chi2.is_finite()
        || neural_seeded_reduced_chi2 > cfg.fallback_reduced_chi2_threshold
        || posterior.dw_ppm_std > cfg.max_dw_std_abstain_ppm;

    if should_abstain {
        let mut grid_fit = fit_multi_cest_grid_lm(profiles, &cfg.lm_cfg)?;
        total_evals += grid_fit.n_evals;
        total_iters += grid_fit.iterations;
        let mut winner = if grid_fit.chi2 <= best_neural.chi2 {
            grid_fit.n_evals = total_evals;
            grid_fit.iterations = total_iters;
            grid_fit
        } else {
            best_neural.n_evals = total_evals;
            best_neural.iterations = total_iters;
            best_neural
        };
        winner.n_evals = total_evals;
        winner.iterations = total_iters;
        Ok(AmortizedHybridFitResult {
            posterior,
            status: AmortizedSeedStatus::AbstainToGridFallback,
            fit: winner,
            neural_seeded_reduced_chi2,
            neural_seeds_refined,
        })
    } else {
        best_neural.n_evals = total_evals;
        best_neural.iterations = total_iters;
        Ok(AmortizedHybridFitResult {
            posterior,
            status: AmortizedSeedStatus::AcceptedNeuralSeed,
            fit: best_neural,
            neural_seeded_reduced_chi2,
            neural_seeds_refined,
        })
    }
}
