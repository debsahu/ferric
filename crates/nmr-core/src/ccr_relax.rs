//! Transverse & longitudinal $^{15}\text{N}\text{–}^1\text{H}$ dipolar/CSA cross-correlated
//! relaxation ($\eta_{xy}$, $\eta_z$, and `TRACT` $\alpha/\beta$) for $R_{\text{ex}}$-free
//! rotational correlation time $\tau_m$, model-free $R_{\text{ex}}$ separation, and site-specific
//! $^{15}\text{N}$ chemical shift anisotropy $\Delta\sigma_N$ (`T22.5`).
//!
//! Provenance: Clean-room implementation strictly from peer-reviewed NMR spin-relaxation
//! literature (zero external GPL, AGPL, or proprietary code consulted):
//! - Goldman, M. (*J. Magn. Reson.* **1984**, 60, 437–452) — exact dipolar–CSA cross-correlated
//!   relaxation interference equations.
//! - Tjandra, N., Szabo, A. & Bax, A. (*J. Am. Chem. Soc.* **1996**, 118, 6986–6991) —
//!   $^{15}\text{N}$ CSA/dipolar cross-correlation $\eta_{xy}, \eta_z$, model-independent
//!   $\eta_{xy}/\eta_z$ correlation-time ratio, and site-specific $^{15}\text{N}$ CSA
//!   determination.
//! - Kroenke, C. D., Loria, J. P., Cavanagh, J., Rance, M. & Palmer, A. G. III (*J. Am. Chem.
//!   Soc.* **1998**, 120, 7905–7915) — separation of $\mu\text{s–ms}$ chemical exchange
//!   $R_{\text{ex}}$ from transverse auto-relaxation via $\kappa = R_2 / \eta_{xy}$.
//! - Lee, D., Hilty, C., Wider, G. & Wüthrich, K. (*J. Magn. Reson.* **2006**, 178, 72–76) —
//!   `TRACT` (TROSY for rotational correlation times) $\alpha/\beta$ transverse decay difference
//!   $R_\beta - R_\alpha = 2\eta_{xy}$ and exact cubic inversion for $R_{\text{ex}}$-free $\tau_m$.

use std::f64::consts::PI;

use serde::{Deserialize, Serialize};

use crate::relax_fit::{
    fit_decay_curve_cpu, DecayCurveInput, ExpFitConfig, OffsetModelGate, RelaxPhysicsConfig,
    GAMMA_15N_RAD_S_T, GAMMA_1H_RAD_S_T, HBAR_J_S, MU0_OVER_4PI,
};

// ============================================================================
// Physical Constants & Configuration
// ============================================================================

/// Default angle (degrees) between the axially symmetric $^{15}\text{N}$ CSA unique principal
/// axis ($\sigma_\parallel$) and the amide $\text{N–H}$ dipolar vector (Tjandra et al. 1996;
/// Lee et al. 2006).
pub const DEFAULT_CSA_DIPOLAR_THETA_DEG: f64 = 17.0;

/// Lower bound of the canonical physical $^{15}\text{N}$ backbone CSA range (ppm).
pub const CSA_N_PHYSICAL_MIN_PPM: f64 = -250.0;

/// Upper bound of the canonical physical $^{15}\text{N}$ backbone CSA range (ppm).
pub const CSA_N_PHYSICAL_MAX_PPM: f64 = -100.0;

/// Extreme-narrowing lower limit of $\eta_{xy} / \eta_z = (4 J(0) + 3 J(\omega_N)) / (6 J(\omega_N))$
/// as $\omega_N \tau_m \to 0$.
pub const ETA_RATIO_EXTREME_NARROWING_LIMIT: f64 = 7.0 / 6.0;

/// Configurable physical constants for $^{15}\text{N}\text{–}^1\text{H}$ dipolar/CSA
/// cross-correlated relaxation (CODATA-2018 defaults).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CcrPhysicsConfig {
    /// Effective amide $\text{N–H}$ internuclear bond length in Ångströms (default `1.02 Å`).
    pub r_nh_angstrom: f64,
    /// Axially symmetric $^{15}\text{N}$ chemical shift anisotropy $\Delta\sigma_N = \sigma_\parallel - \sigma_\perp$
    /// in ppm (default `-172.0 ppm`, negative for backbone amides).
    pub csa_n_ppm: f64,
    /// Angle $\theta$ (degrees) between the $^{15}\text{N}$ CSA principal axis and the $\text{N–H}$
    /// dipolar bond vector (default `17.0°`).
    pub theta_deg: f64,
    /// Proton gyromagnetic ratio $\gamma_H > 0$ ($\text{rad s}^{-1}\text{ T}^{-1}$).
    pub gamma_h: f64,
    /// Nitrogen-15 signed gyromagnetic ratio $\gamma_N < 0$ ($\text{rad s}^{-1}\text{ T}^{-1}$).
    pub gamma_n: f64,
}

impl Default for CcrPhysicsConfig {
    fn default() -> Self {
        Self {
            r_nh_angstrom: 1.02,
            csa_n_ppm: -172.0,
            theta_deg: DEFAULT_CSA_DIPOLAR_THETA_DEG,
            gamma_h: GAMMA_1H_RAD_S_T,
            gamma_n: GAMMA_15N_RAD_S_T,
        }
    }
}

impl CcrPhysicsConfig {
    /// Build a `CcrPhysicsConfig` from an existing `RelaxPhysicsConfig` and CSA–dipolar angle.
    pub fn from_relax_physics(phys: RelaxPhysicsConfig, theta_deg: f64) -> Self {
        Self {
            r_nh_angstrom: phys.r_nh_angstrom,
            csa_n_ppm: phys.csa_n_ppm,
            theta_deg,
            gamma_h: phys.gamma_h,
            gamma_n: phys.gamma_n,
        }
    }

    /// Convert into `RelaxPhysicsConfig` sharing identical CODATA-2018 constants.
    pub fn to_relax_physics(&self) -> RelaxPhysicsConfig {
        RelaxPhysicsConfig {
            r_nh_angstrom: self.r_nh_angstrom,
            csa_n_ppm: self.csa_n_ppm,
            gamma_h: self.gamma_h,
            gamma_n: self.gamma_n,
        }
    }

    /// Validate that all physical parameters are finite and physically well-posed.
    pub fn validate(&self) -> Result<(), String> {
        if !self.r_nh_angstrom.is_finite() || self.r_nh_angstrom <= 0.0 {
            return Err("r_nh_angstrom must be finite and > 0".to_string());
        }
        if !self.csa_n_ppm.is_finite() {
            return Err("csa_n_ppm must be finite".to_string());
        }
        if !self.theta_deg.is_finite() {
            return Err("theta_deg must be finite".to_string());
        }
        if !self.gamma_h.is_finite() || self.gamma_h <= 0.0 {
            return Err("gamma_h must be finite and > 0".to_string());
        }
        if !self.gamma_n.is_finite() || self.gamma_n >= 0.0 {
            return Err("gamma_n must be finite and < 0 for 15N".to_string());
        }
        Ok(())
    }

    /// Second-order Legendre polynomial $P_2(\cos\theta) = \frac{1}{2}(3\cos^2\theta - 1)$.
    #[inline]
    pub fn p2_cos_theta(&self) -> f64 {
        let c = self.theta_deg.to_radians().cos();
        0.5 * (3.0 * c * c - 1.0)
    }

    /// Static magnetic field strength $B_0 = \omega_H / \gamma_H$ in Tesla at proton Larmor
    /// frequency `field_mhz` (MHz).
    #[inline]
    pub fn b0_tesla(&self, field_mhz: f64) -> f64 {
        (2.0 * PI * field_mhz * 1.0e6) / self.gamma_h
    }

    /// Signed Larmor angular frequencies $(\omega_H > 0, \omega_N < 0)$ in $\text{rad s}^{-1}$.
    #[inline]
    pub fn signed_omegas_rad_s(&self, field_mhz: f64) -> (f64, f64) {
        let omega_h = 2.0 * PI * field_mhz * 1.0e6;
        let omega_n = (self.gamma_n / self.gamma_h) * omega_h;
        (omega_h, omega_n)
    }

    /// Signed dipolar coupling constant $d_{\text{signed}} = \frac{\mu_0 \hbar \gamma_H \gamma_N}{4\pi r_{\text{NH}}^3} < 0$
    /// ($\text{rad s}^{-1}$).
    #[inline]
    pub fn dipolar_d_signed_rad_s(&self) -> f64 {
        let r_m = self.r_nh_angstrom * 1.0e-10;
        (MU0_OVER_4PI * HBAR_J_S * self.gamma_h * self.gamma_n) / (r_m * r_m * r_m)
    }

    /// Unsigned dipolar coupling constant $d = -d_{\text{signed}} > 0$ ($\text{rad s}^{-1}$),
    /// matching [`RelaxPhysicsConfig::dipolar_d_rad_s`].
    #[inline]
    pub fn dipolar_d_rad_s(&self) -> f64 {
        -self.dipolar_d_signed_rad_s()
    }

    /// Lee et al. (2006) dipolar constant $p = \frac{\mu_0 \gamma_H \gamma_N \hbar}{4\pi \cdot 2\sqrt{2}\, r_{\text{NH}}^3} < 0$
    /// ($\text{rad s}^{-1}$), satisfying $p = d_{\text{signed}} / (2\sqrt{2})$.
    #[inline]
    pub fn dipolar_p_rad_s(&self) -> f64 {
        self.dipolar_d_signed_rad_s() / (2.0 * 2.0_f64.sqrt())
    }

    /// $^{15}\text{N}$ CSA coupling constant $c = \frac{\omega_N \Delta\sigma_N}{\sqrt{3}}$ ($\text{rad s}^{-1}$),
    /// matching [`RelaxPhysicsConfig::csa_c_rad_s`] (positive for $\gamma_N < 0, \Delta\sigma_N < 0$).
    #[inline]
    pub fn csa_c_rad_s(&self, field_mhz: f64) -> f64 {
        let (_, omega_n) = self.signed_omegas_rad_s(field_mhz);
        (omega_n * (self.csa_n_ppm * 1.0e-6)) / 3.0_f64.sqrt()
    }

    /// Lee et al. (2006) $^{15}\text{N}$ CSA constant $\delta_N = \frac{\gamma_N B_0 \Delta\sigma_N}{3\sqrt{2}}$
    /// ($\text{rad s}^{-1}$), satisfying $\delta_N = c / \sqrt{6}$ (positive for $\gamma_N < 0, \Delta\sigma_N < 0$).
    #[inline]
    pub fn csa_delta_n_rad_s(&self, field_mhz: f64) -> f64 {
        let b0 = self.b0_tesla(field_mhz);
        (self.gamma_n * b0 * (self.csa_n_ppm * 1.0e-6)) / (3.0 * 2.0_f64.sqrt())
    }

    /// Exact dipolar–CSA cross-correlation prefactor $K_{\text{CCR}} P_2(\cos\theta)$
    /// ($\text{rad}^2\text{ s}^{-2}$) multiplying $[4 J(0) + 3 J(\omega_N)]$ for $\eta_{xy}$ and
    /// $[6 J(\omega_N)]$ for $\eta_z$ under the $J(\omega) = \frac{2}{5}\frac{\tau_c}{1+(\omega\tau_c)^2}$
    /// spectral density normalization:
    ///
    /// $$K_{\text{CCR}} P_2(\cos\theta) = -2 p \delta_N P_2(\cos\theta) = -\frac{\sqrt{3}}{6} c \, d_{\text{signed}} P_2(\cos\theta) = \frac{1}{6} d \, (\omega_N \Delta\sigma_N) P_2(\cos\theta)$$
    ///
    /// Positive for backbone $^{15}\text{N}$ ($\gamma_N < 0$, $\Delta\sigma_N < 0$, $\theta \approx 17^\circ$),
    /// guaranteeing $\eta_{xy} > 0$, $\eta_z > 0$, $R_\beta - R_\alpha = 2\eta_{xy} > 0$, and exact
    /// Cauchy–Schwarz positivity of $R_\alpha = R_2 - \eta_{xy}$ at $\theta = 0^\circ$.
    #[inline]
    pub fn ccr_prefactor(&self, field_mhz: f64) -> f64 {
        -2.0 * self.dipolar_p_rad_s() * self.csa_delta_n_rad_s(field_mhz) * self.p2_cos_theta()
    }

    /// Theoretical secular baseline ratio $\kappa_{\text{sec}} = (R_2^{\text{DD,sec}} + R_2^{\text{CSA}}) / \eta_{xy}$
    /// (Kroenke et al. 1998):
    ///
    /// $$\kappa_{\text{sec}} = \frac{\frac{1}{8} d^2 + \frac{1}{6} c^2}{K_{\text{CCR}} P_2(\cos\theta)} = \frac{3 d^2 + 4 c^2}{4\sqrt{3}\, c \, d \, P_2(\cos\theta)}$$
    #[inline]
    pub fn theoretical_kappa0(&self, field_mhz: f64) -> Result<f64, String> {
        validate_field_mhz(field_mhz)?;
        self.validate()?;
        let pref = self.ccr_prefactor(field_mhz);
        if pref.abs() <= 1.0e-30 {
            return Err(
                "CCR prefactor vanishes (csa_n_ppm == 0 or P2(cos theta) == 0)".to_string(),
            );
        }
        let d = self.dipolar_d_rad_s();
        let c = self.csa_c_rad_s(field_mhz);
        Ok((0.125 * d * d + (c * c) / 6.0) / pref)
    }
}

// ============================================================================
// Spectral Density & Exact Forward CCR / Auto-Relaxation Physics
// ============================================================================

/// Isotropic rigid-rotor spectral density $J(\omega) = \frac{2}{5} \frac{\tau_m}{1 + (\omega\tau_m)^2}$
/// ($\text{s rad}^{-1}$), matching `relax_fit` and `modelfree`.
#[inline]
pub fn isotropic_spectral_density(omega_rad_s: f64, tau_m_sec: f64) -> f64 {
    let wt = omega_rad_s * tau_m_sec;
    0.4 * tau_m_sec / (1.0 + wt * wt)
}

/// Lipari–Szabo model-free isotropic spectral density ($\text{s rad}^{-1}$):
///
/// $$J(\omega) = \frac{2}{5}\left[\frac{S^2 \tau_m}{1 + (\omega\tau_m)^2} + \frac{(1 - S^2)\tau}{1 + (\omega\tau)^2}\right], \quad \tau^{-1} = \tau_m^{-1} + \tau_e^{-1}$$
#[inline]
pub fn model_free_spectral_density(
    omega_rad_s: f64,
    tau_m_sec: f64,
    s2: f64,
    tau_e_sec: f64,
) -> f64 {
    let wt_m = omega_rad_s * tau_m_sec;
    let slow = s2 * tau_m_sec / (1.0 + wt_m * wt_m);
    let fast = if tau_e_sec > 0.0 && s2 < 1.0 {
        let tau = (tau_m_sec * tau_e_sec) / (tau_m_sec + tau_e_sec);
        let wt_i = omega_rad_s * tau;
        (1.0 - s2) * tau / (1.0 + wt_i * wt_i)
    } else {
        0.0
    };
    0.4 * (slow + fast)
}

/// Multi-exponential rotational diffusion + Lipari–Szabo spectral density for arbitrary
/// Woessner components `[(c_a, tau_a_sec)]` ($\sum_a c_a = 1$).
pub fn multi_component_spectral_density(
    omega_rad_s: f64,
    components: &[(f64, f64)],
    s2: f64,
    tau_e_sec: f64,
) -> f64 {
    let mut sum = 0.0_f64;
    for &(ca, tau_a) in components {
        sum += ca * model_free_spectral_density(omega_rad_s, tau_a, s2, tau_e_sec);
    }
    sum
}

/// Transverse and longitudinal dipolar–CSA cross-correlated relaxation rates ($\text{s}^{-1}$).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CcrRates {
    /// Transverse dipolar–CSA cross-correlation rate $\eta_{xy}$ ($\text{s}^{-1}$).
    pub eta_xy: f64,
    /// Longitudinal dipolar–CSA cross-correlation rate $\eta_z$ ($\text{s}^{-1}$).
    pub eta_z: f64,
    /// Model-independent ratio $\eta_{xy} / \eta_z = (4 J(0) + 3 J(\omega_N)) / (6 J(\omega_N))$.
    pub ratio_xy_over_z: f64,
    /// `TRACT` anti-TROSY minus TROSY transverse rate difference $R_\beta - R_\alpha = 2\eta_{xy}$ ($\text{s}^{-1}$).
    pub r_beta_minus_r_alpha: f64,
}

/// Complete self-consistent set of isotropic $^{15}\text{N}$ auto-relaxation and CCR observables.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct FullIsotropicRelaxObservables {
    pub field_mhz: f64,
    pub r1: f64,
    pub r2: f64,
    pub hetnoe: f64,
    pub eta_xy: f64,
    pub eta_z: f64,
    /// TROSY slow-relaxing transverse rate $R_\alpha = R_2 - \eta_{xy}$ ($\text{s}^{-1}$).
    pub r_alpha: f64,
    /// Anti-TROSY fast-relaxing transverse rate $R_\beta = R_2 + \eta_{xy}$ ($\text{s}^{-1}$).
    pub r_beta: f64,
}

/// Evaluate exact Goldman (1984) / Tjandra et al. (1996) / Lee et al. (2006) cross-correlated
/// relaxation rates $(\eta_{xy}, \eta_z)$ from spectral densities $(J(0), J(\omega_N))$.
pub fn compute_ccr_rates_from_j(
    j_0: f64,
    j_omega_n: f64,
    field_mhz: f64,
    phys: &CcrPhysicsConfig,
) -> Result<CcrRates, String> {
    validate_field_mhz(field_mhz)?;
    phys.validate()?;
    if !j_0.is_finite() || j_0 < 0.0 || !j_omega_n.is_finite() || j_omega_n <= 0.0 {
        return Err("spectral densities j_0 >= 0 and j_omega_n > 0 must be finite".to_string());
    }
    let pref = phys.ccr_prefactor(field_mhz);
    let eta_xy = pref * (4.0 * j_0 + 3.0 * j_omega_n);
    let eta_z = pref * (6.0 * j_omega_n);
    let ratio_xy_over_z = (4.0 * j_0 + 3.0 * j_omega_n) / (6.0 * j_omega_n);
    Ok(CcrRates {
        eta_xy,
        eta_z,
        ratio_xy_over_z,
        r_beta_minus_r_alpha: 2.0 * eta_xy,
    })
}

/// Compute isotropic Lipari–Szabo $(\eta_{xy}, \eta_z)$ at magnetic field `field_mhz`.
pub fn compute_ccr_rates_isotropic(
    tau_m_sec: f64,
    s2: f64,
    tau_e_sec: f64,
    field_mhz: f64,
    phys: &CcrPhysicsConfig,
) -> Result<CcrRates, String> {
    validate_field_mhz(field_mhz)?;
    phys.validate()?;
    validate_dynamics_params(tau_m_sec, s2, tau_e_sec)?;
    let (_, omega_n) = phys.signed_omegas_rad_s(field_mhz);
    let j_0 = model_free_spectral_density(0.0, tau_m_sec, s2, tau_e_sec);
    let j_n = model_free_spectral_density(omega_n, tau_m_sec, s2, tau_e_sec);
    compute_ccr_rates_from_j(j_0, j_n, field_mhz, phys)
}

/// Compute exact self-consistent $(R_1, R_2, \text{hetNOE}, \eta_{xy}, \eta_z, R_\alpha, R_\beta)$
/// for an isotropic Lipari–Szabo spin with optional chemical exchange $R_{\text{ex}} \ge 0$.
pub fn compute_full_relax_and_ccr_isotropic(
    tau_m_sec: f64,
    s2: f64,
    tau_e_sec: f64,
    r_ex: f64,
    field_mhz: f64,
    phys: &CcrPhysicsConfig,
) -> Result<FullIsotropicRelaxObservables, String> {
    validate_field_mhz(field_mhz)?;
    phys.validate()?;
    validate_dynamics_params(tau_m_sec, s2, tau_e_sec)?;
    if !r_ex.is_finite() || r_ex < 0.0 {
        return Err("r_ex must be finite and >= 0".to_string());
    }

    let (wh, wn) = phys.signed_omegas_rad_s(field_mhz);
    let j0 = model_free_spectral_density(0.0, tau_m_sec, s2, tau_e_sec);
    let jn = model_free_spectral_density(wn, tau_m_sec, s2, tau_e_sec);
    let jh = model_free_spectral_density(wh, tau_m_sec, s2, tau_e_sec);
    let j_diff = model_free_spectral_density(wh - wn, tau_m_sec, s2, tau_e_sec);
    let j_sum = model_free_spectral_density(wh + wn, tau_m_sec, s2, tau_e_sec);

    let d = phys.dipolar_d_rad_s();
    let d2 = d * d;
    let c = phys.csa_c_rad_s(field_mhz);
    let c2 = c * c;

    let r1 = 0.25 * d2 * (j_diff + 3.0 * jn + 6.0 * j_sum) + c2 * jn;
    let r2 = 0.125 * d2 * (4.0 * j0 + j_diff + 3.0 * jn + 6.0 * jh + 6.0 * j_sum)
        + (c2 / 6.0) * (4.0 * j0 + 3.0 * jn)
        + r_ex;
    let sigma_nh = 0.25 * d2 * (6.0 * j_sum - j_diff);
    let hetnoe = 1.0 + (phys.gamma_h / phys.gamma_n) * (sigma_nh / r1);

    let ccr = compute_ccr_rates_from_j(j0, jn, field_mhz, phys)?;
    let r_alpha = r2 - ccr.eta_xy;
    let r_beta = r2 + ccr.eta_xy;

    Ok(FullIsotropicRelaxObservables {
        field_mhz,
        r1,
        r2,
        hetnoe,
        eta_xy: ccr.eta_xy,
        eta_z: ccr.eta_z,
        r_alpha,
        r_beta,
    })
}

// ============================================================================
// TRACT alpha/beta Decay Fitter & R_ex-Free tau_m Inversion (Lee et al. 2006)
// ============================================================================

/// Result of inverting $\eta_{xy}$ (or $R_\beta - R_\alpha = 2\eta_{xy}$) into isotropic $\tau_m$.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TractTauMEstimate {
    /// Inverted rotational correlation time $\tau_m$ in seconds.
    pub tau_m_sec: f64,
    /// Analytically propagated $1\sigma$ uncertainty on $\tau_m$ in seconds.
    pub sigma_tau_m_sec: f64,
    /// Order parameter $S^2$ assumed in the inversion (`1.0` for canonical rigid-body `TRACT` lower bound).
    pub assumed_s2: f64,
    /// Internal correlation time $\tau_e$ (seconds) assumed in the inversion (`0.0` for rigid limit).
    pub assumed_tau_e_sec: f64,
    /// Dimensionless product $|\omega_N| \tau_m$.
    pub omega_n_tau_m: f64,
}

/// Result of model-independent $\eta_{xy} / \eta_z$ inversion to $\tau_m$ (Tjandra et al. 1996).
///
/// Because $\eta_{xy} / \eta_z = (4 J(0) + 3 J(\omega_N)) / (6 J(\omega_N)) = \frac{7}{6} + \frac{2}{3}\omega_N^2 \tau_m^2$
/// for isotropic tumbling ($\tau_e \ll \tau_m$), this estimate is simultaneously invariant to $S^2$,
/// $r_{\text{NH}}$, $\Delta\sigma_N$, and $P_2(\cos\theta)$.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct EtaRatioTauMEstimate {
    /// Model-independent rotational correlation time $\tau_m$ in seconds.
    pub tau_m_sec: f64,
    /// Propagated $1\sigma$ uncertainty on $\tau_m$ in seconds.
    pub sigma_tau_m_sec: f64,
    /// Observed ratio $\eta_{xy} / \eta_z$.
    pub ratio_xy_over_z: f64,
    /// Propagated $1\sigma$ uncertainty on $\eta_{xy} / \eta_z$.
    pub sigma_ratio_xy_over_z: f64,
    /// Implied Lipari–Szabo order parameter $S^2 = \eta_{xy,\text{obs}} / \eta_{xy,\text{rigid}}(\tau_m)$
    /// under the supplied `CcrPhysicsConfig`.
    pub implied_s2: f64,
    /// Propagated $1\sigma$ uncertainty on `implied_s2`.
    pub sigma_implied_s2: f64,
}

/// Solve the strictly monotonic Lee et al. (2006) dimensionless cubic equation
/// $4 x + \frac{3 x}{1 + x^2} = c_{\text{dim}}$ (equivalently $4 x^3 - c_{\text{dim}} x^2 + 7 x - c_{\text{dim}} = 0$)
/// for its unique real positive root $x = |\omega_N| \tau_m > 0$ using Cardano's closed-form
/// solution followed by Newton–Raphson polishing.
fn solve_lee_cubic_dimensionless(c_dim: f64) -> f64 {
    // Depressed cubic: t^3 + p_c t + q_c = 0 with x = t + c_dim / 12.
    // Because f'(x) = 4 + 3(1 - x^2)/(1 + x^2)^2 >= 29/8 > 0 for all real x,
    // the discriminant delta = (q_c/2)^2 + (p_c/3)^3 is strictly positive for all c_dim.
    let p_c = (84.0 - c_dim * c_dim) / 48.0;
    let q_c = -c_dim * (c_dim * c_dim + 90.0) / 864.0;
    let half_q = 0.5 * q_c;
    let third_p = p_c / 3.0;
    let disc = (half_q * half_q + third_p * third_p * third_p).max(0.0);
    let sqrt_disc = disc.sqrt();
    let t = (-half_q + sqrt_disc).cbrt() + (-half_q - sqrt_disc).cbrt();
    let mut x = (t + c_dim / 12.0).max(0.0);

    // Polish with 6 Newton–Raphson steps to eliminate any large-c or small-c cancellation.
    for _ in 0..6 {
        let x2 = x * x;
        let denom = 1.0 + x2;
        let f = 4.0 * x + (3.0 * x) / denom - c_dim;
        let df = 4.0 + 3.0 * (1.0 - x2) / (denom * denom);
        let step = f / df;
        x = (x - step).max(0.0);
        if step.abs() <= 1.0e-15 * x.max(1.0) {
            break;
        }
    }
    x
}

/// Invert a measured transverse dipolar–CSA cross-correlation rate $\eta_{xy}$ ($\text{s}^{-1}$)
/// into an isotropic rotational correlation time $\tau_m$ (seconds) via the exact Lee et al. (2006)
/// cubic/spectral-density relation.
///
/// - Pass `s2 = 1.0, tau_e_sec = 0.0` for the canonical rigid-body `TRACT` lower bound $\tau_m$.
/// - Pass `0 < s2 < 1` (and optional `tau_e_sec >= 0`) for the order-parameter-corrected $\tau_m$.
pub fn invert_tract_eta_xy_to_tau_m(
    eta_xy: f64,
    sigma_eta_xy: f64,
    field_mhz: f64,
    s2: f64,
    tau_e_sec: f64,
    phys: &CcrPhysicsConfig,
) -> Result<TractTauMEstimate, String> {
    validate_field_mhz(field_mhz)?;
    phys.validate()?;
    if !eta_xy.is_finite() || eta_xy <= 0.0 {
        return Err("eta_xy must be finite and > 0 for TRACT tau_m inversion".to_string());
    }
    if !sigma_eta_xy.is_finite() || sigma_eta_xy < 0.0 {
        return Err("sigma_eta_xy must be finite and >= 0".to_string());
    }
    if !s2.is_finite() || s2 <= 0.0 || s2 > 1.0 {
        return Err("order parameter s2 must be finite and in (0, 1]".to_string());
    }
    if !tau_e_sec.is_finite() || tau_e_sec < 0.0 {
        return Err("tau_e_sec must be finite and >= 0".to_string());
    }

    let pref = phys.ccr_prefactor(field_mhz);
    if !pref.is_finite() || pref <= 0.0 {
        return Err(
            "CCR prefactor must be strictly positive (check csa_n_ppm < 0 and P2(cos theta) > 0)"
                .to_string(),
        );
    }

    let (_, omega_n) = phys.signed_omegas_rad_s(field_mhz);
    let abs_wn = omega_n.abs();

    if tau_e_sec == 0.0 || s2 >= 1.0 - 1.0e-15 {
        // Exact closed-form Cardano + Newton inversion of Lee et al. (2006):
        // eta_xy = pref * (2/5) * s2 * (1/|wN|) * [4 x + 3 x / (1 + x^2)], x = |wN| tau_m
        let c_dim = (2.5 * eta_xy * abs_wn) / (pref * s2);
        let x = solve_lee_cubic_dimensionless(c_dim);
        let tau_m_sec = x / abs_wn;
        let x2 = x * x;
        let denom = 1.0 + x2;
        let df_dx = 4.0 + 3.0 * (1.0 - x2) / (denom * denom);
        let d_eta_d_tau = 0.4 * pref * s2 * df_dx;
        let sigma_tau_m_sec = sigma_eta_xy / d_eta_d_tau.abs().max(1.0e-30);
        return Ok(TractTauMEstimate {
            tau_m_sec,
            sigma_tau_m_sec,
            assumed_s2: s2,
            assumed_tau_e_sec: 0.0,
            omega_n_tau_m: x,
        });
    }

    // General Lipari–Szabo case (0 < s2 < 1, tau_e_sec > 0):
    // Both slow and fast terms of eta_xy(tau_m) are strictly increasing in tau_m > 0.
    let eval_eta = |tau_m: f64| -> f64 {
        let j0 = model_free_spectral_density(0.0, tau_m, s2, tau_e_sec);
        let jn = model_free_spectral_density(omega_n, tau_m, s2, tau_e_sec);
        pref * (4.0 * j0 + 3.0 * jn)
    };

    // Bracket root cleanly across [lo, hi]
    let mut lo = 1.0e-14_f64;
    let mut hi = (2.5 * eta_xy) / (4.0 * pref * s2).max(1.0e-30);
    if hi <= lo {
        hi = 1.0e-9;
    }
    for _ in 0..80 {
        if eval_eta(hi) >= eta_xy {
            break;
        }
        hi *= 2.0;
    }
    if eval_eta(hi) < eta_xy {
        return Err("failed to bracket TRACT tau_m for given eta_xy".to_string());
    }
    for _ in 0..80 {
        if eval_eta(lo) <= eta_xy {
            break;
        }
        lo *= 0.5;
    }

    // Safeguarded bisection + secant/Newton refinement
    for _ in 0..90 {
        let mid = 0.5 * (lo + hi);
        let f_mid = eval_eta(mid) - eta_xy;
        if f_mid.abs() <= 1.0e-14 * eta_xy.max(1.0) || (hi - lo) <= 1.0e-15 * mid.max(1.0e-15) {
            lo = mid;
            hi = mid;
            break;
        }
        if f_mid < 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let tau_m_sec = 0.5 * (lo + hi);
    let h = (tau_m_sec * 1.0e-6).max(1.0e-15);
    let d_eta_d_tau = (eval_eta(tau_m_sec + h) - eval_eta((tau_m_sec - h).max(1.0e-15)))
        / ((tau_m_sec + h) - (tau_m_sec - h).max(1.0e-15));
    let sigma_tau_m_sec = sigma_eta_xy / d_eta_d_tau.abs().max(1.0e-30);

    Ok(TractTauMEstimate {
        tau_m_sec,
        sigma_tau_m_sec,
        assumed_s2: s2,
        assumed_tau_e_sec: tau_e_sec,
        omega_n_tau_m: abs_wn * tau_m_sec,
    })
}

/// Invert a paired $(\eta_{xy}, \eta_z)$ measurement into $\tau_m$ via Tjandra, Szabo & Bax (1996)
/// Eq. (5):
///
/// $$\frac{\eta_{xy}}{\eta_z} = \frac{7}{6} + \frac{2}{3} \omega_N^2 \tau_m^2 \implies \tau_m = \frac{1}{|\omega_N|}\sqrt{\frac{3}{2}\frac{\eta_{xy}}{\eta_z} - \frac{7}{4}}$$
///
/// This ratio simultaneously cancels $S^2$, $r_{\text{NH}}$, $\Delta\sigma_N$, and $P_2(\cos\theta)$.
pub fn invert_eta_ratio_to_tau_m(
    eta_xy: f64,
    sigma_eta_xy: f64,
    eta_z: f64,
    sigma_eta_z: f64,
    field_mhz: f64,
    phys: &CcrPhysicsConfig,
) -> Result<EtaRatioTauMEstimate, String> {
    validate_field_mhz(field_mhz)?;
    phys.validate()?;
    if !eta_xy.is_finite() || eta_xy <= 0.0 || !eta_z.is_finite() || eta_z <= 0.0 {
        return Err("eta_xy and eta_z must be finite and > 0".to_string());
    }
    if !sigma_eta_xy.is_finite()
        || sigma_eta_xy < 0.0
        || !sigma_eta_z.is_finite()
        || sigma_eta_z < 0.0
    {
        return Err("sigma_eta_xy and sigma_eta_z must be finite and >= 0".to_string());
    }

    let ratio = eta_xy / eta_z;
    if ratio <= ETA_RATIO_EXTREME_NARROWING_LIMIT {
        return Err(format!(
            "eta_xy / eta_z = {:.6} must exceed the extreme-narrowing limit 7/6 ({:.6})",
            ratio, ETA_RATIO_EXTREME_NARROWING_LIMIT
        ));
    }

    let (_, omega_n) = phys.signed_omegas_rad_s(field_mhz);
    let abs_wn = omega_n.abs();
    let rad = 1.5 * ratio - 1.75;
    let tau_m_sec = rad.sqrt() / abs_wn;

    let sigma_ratio = ratio * (sigma_eta_xy / eta_xy).hypot(sigma_eta_z / eta_z);
    let d_tau_d_ratio = 0.75 / (abs_wn * rad.sqrt());
    let sigma_tau_m_sec = d_tau_d_ratio * sigma_ratio;

    // Implied order parameter S^2 under canonical physics (S^2 = eta_xy / eta_xy_rigid(tau_m))
    let rigid = compute_ccr_rates_isotropic(tau_m_sec, 1.0, 0.0, field_mhz, phys)?;
    let (implied_s2, sigma_implied_s2) = if rigid.eta_xy.abs() > 1.0e-30 {
        let s2_est = eta_xy / rigid.eta_xy;
        let x = abs_wn * tau_m_sec;
        let x2 = x * x;
        let denom = 1.0 + x2;
        let df_dx = 4.0 + 3.0 * (1.0 - x2) / (denom * denom);
        let d_rigid_d_tau = 0.4 * phys.ccr_prefactor(field_mhz) * df_dx;
        let d_s2_d_eta_xy = 1.0 / rigid.eta_xy
            - (eta_xy * d_rigid_d_tau * d_tau_d_ratio) / (rigid.eta_xy * rigid.eta_xy * eta_z);
        let d_s2_d_eta_z = (eta_xy * d_rigid_d_tau * d_tau_d_ratio * ratio)
            / (rigid.eta_xy * rigid.eta_xy * eta_z);
        let sig_s2 = (d_s2_d_eta_xy * sigma_eta_xy).hypot(d_s2_d_eta_z * sigma_eta_z);
        (s2_est, sig_s2)
    } else {
        (0.0, 0.0)
    };

    Ok(EtaRatioTauMEstimate {
        tau_m_sec,
        sigma_tau_m_sec,
        ratio_xy_over_z: ratio,
        sigma_ratio_xy_over_z: sigma_ratio,
        implied_s2,
        sigma_implied_s2,
    })
}

/// Paired `TRACT` $\alpha$-state (TROSY) and $\beta$-state (anti-TROSY) intensity decay series
/// (Lee et al. 2006) for 1D envelope or 2D residue-resolved experiments.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TractDecayInput {
    pub label: String,
    pub seq_id: Option<i32>,
    pub field_mhz: f64,
    pub alpha_times_sec: Vec<f64>,
    pub alpha_intensities: Vec<f64>,
    pub alpha_sigmas: Vec<f64>,
    pub beta_times_sec: Vec<f64>,
    pub beta_intensities: Vec<f64>,
    pub beta_sigmas: Vec<f64>,
    /// Optional paired longitudinal CCR rate $\eta_z$ ($\text{s}^{-1}$) for model-independent
    /// $\eta_{xy}/\eta_z$ correlation-time inversion.
    pub eta_z: Option<f64>,
    /// Optional $1\sigma$ uncertainty on $\eta_z$ ($\text{s}^{-1}$).
    pub sigma_eta_z: Option<f64>,
}

/// Configuration for `TRACT` $\alpha/\beta$ decay fitting and $\tau_m$ inversion.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TractConfig {
    pub physics: CcrPhysicsConfig,
    /// Optional backbone order parameter $S^2 \in (0, 1]$ for order-parameter-corrected $\tau_m$
    /// (in addition to the canonical rigid-body $S^2 = 1$ `TRACT` lower bound).
    pub order_parameter_s2: Option<f64>,
    /// Internal correlation time $\tau_e$ (seconds) used when `order_parameter_s2` is `Some`.
    pub tau_e_sec: f64,
    /// Upper bound on fitted relaxation rates $R_\alpha, R_\beta$ ($\text{s}^{-1}$, default `500.0`).
    pub max_rate_s_inv: f64,
    /// Number of parametric Monte Carlo bootstrap replicates for exponential decay fitting
    /// (`0` = exact analytical Cramér–Rao / Fisher SE only, default `0`).
    pub mc_replicates: usize,
    /// Deterministic seed when `mc_replicates > 0`.
    pub seed: u64,
}

impl Default for TractConfig {
    fn default() -> Self {
        Self {
            physics: CcrPhysicsConfig::default(),
            order_parameter_s2: None,
            tau_e_sec: 0.0,
            max_rate_s_inv: 500.0,
            mc_replicates: 0,
            seed: 20262205,
        }
    }
}

/// Fitted `TRACT` $\alpha/\beta$ rates, $R_{\text{ex}}$-free $\eta_{xy}$, and inverted $\tau_m$.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TractFitResult {
    pub label: String,
    pub seq_id: Option<i32>,
    pub field_mhz: f64,
    /// Fitted initial amplitude $I_{0,\alpha}$ of the TROSY ($\alpha$) component.
    pub i0_alpha: f64,
    /// Fitted TROSY transverse decay rate $R_\alpha$ ($\text{s}^{-1}$).
    pub r_alpha: f64,
    /// $1\sigma$ uncertainty on $R_\alpha$ ($\text{s}^{-1}$).
    pub sigma_r_alpha: f64,
    pub chi2_alpha: f64,
    /// Fitted initial amplitude $I_{0,\beta}$ of the anti-TROSY ($\beta$) component.
    pub i0_beta: f64,
    /// Fitted anti-TROSY transverse decay rate $R_\beta$ ($\text{s}^{-1}$).
    pub r_beta: f64,
    /// $1\sigma$ uncertainty on $R_\beta$ ($\text{s}^{-1}$).
    pub sigma_r_beta: f64,
    pub chi2_beta: f64,
    /// Transverse dipolar–CSA CCR rate $\eta_{xy} = (R_\beta - R_\alpha) / 2$ ($\text{s}^{-1}$),
    /// strictly invariant to chemical exchange $R_{\text{ex}}$.
    pub eta_xy: f64,
    /// Propagated $1\sigma$ uncertainty $\sigma_{\eta_{xy}} = \frac{1}{2}\sqrt{\sigma_{R_\alpha}^2 + \sigma_{R_\beta}^2}$.
    pub sigma_eta_xy: f64,
    /// Canonical Lee et al. (2006) rigid-body ($S^2 = 1$) `TRACT` $\tau_m$ lower bound.
    pub tau_m_rigid: TractTauMEstimate,
    /// Order-parameter-corrected $\tau_m$ when `TractConfig::order_parameter_s2` is `Some`.
    pub tau_m_s2_corrected: Option<TractTauMEstimate>,
    /// Model-independent $\eta_{xy}/\eta_z$ $\tau_m$ when paired `eta_z` was supplied.
    pub tau_m_from_eta_ratio: Option<EtaRatioTauMEstimate>,
}

/// Fit paired `TRACT` $\alpha$ (TROSY) and $\beta$ (anti-TROSY) decay curves via weighted
/// Variable-Projection + Levenberg–Marquardt nonlinear least squares and invert $\eta_{xy}$
/// into an $R_{\text{ex}}$-free rotational correlation time $\tau_m$ (Lee et al. 2006).
pub fn fit_tract_decay_pair(
    input: &TractDecayInput,
    cfg: &TractConfig,
) -> Result<TractFitResult, String> {
    validate_field_mhz(input.field_mhz)?;
    cfg.physics.validate()?;
    if !cfg.max_rate_s_inv.is_finite() || cfg.max_rate_s_inv <= 0.0 {
        return Err("max_rate_s_inv must be finite and > 0".to_string());
    }
    if !cfg.tau_e_sec.is_finite() || cfg.tau_e_sec < 0.0 {
        return Err("tau_e_sec must be finite and >= 0".to_string());
    }
    if let Some(s2) = cfg.order_parameter_s2 {
        if !s2.is_finite() || s2 <= 0.0 || s2 > 1.0 {
            return Err("order_parameter_s2 must be finite and in (0, 1]".to_string());
        }
    }

    validate_decay_series(
        "alpha",
        &input.alpha_times_sec,
        &input.alpha_intensities,
        &input.alpha_sigmas,
    )?;
    validate_decay_series(
        "beta",
        &input.beta_times_sec,
        &input.beta_intensities,
        &input.beta_sigmas,
    )?;

    let exp_cfg = ExpFitConfig {
        offset_gate: OffsetModelGate::Disabled,
        mc_replicates: cfg.mc_replicates,
        seed: cfg.seed,
        max_rate_s_inv: cfg.max_rate_s_inv,
    };
    let seq_id = input.seq_id.unwrap_or(0);

    let alpha_curve = DecayCurveInput {
        seq_id,
        res_name: format!("{}:alpha", input.label),
        times_sec: input.alpha_times_sec.clone(),
        intensities: input.alpha_intensities.clone(),
        sigmas: input.alpha_sigmas.clone(),
    };
    let beta_curve = DecayCurveInput {
        seq_id: seq_id.wrapping_add(1_000_000),
        res_name: format!("{}:beta", input.label),
        times_sec: input.beta_times_sec.clone(),
        intensities: input.beta_intensities.clone(),
        sigmas: input.beta_sigmas.clone(),
    };

    let fit_a = fit_decay_curve_cpu(&alpha_curve, &exp_cfg)?;
    let fit_b = fit_decay_curve_cpu(&beta_curve, &exp_cfg)?;

    let sigma_ra = if cfg.mc_replicates > 0 {
        fit_a.sigma_rate_mc
    } else {
        fit_a.sigma_rate_fisher
    };
    let sigma_rb = if cfg.mc_replicates > 0 {
        fit_b.sigma_rate_mc
    } else {
        fit_b.sigma_rate_fisher
    };

    if !fit_a.rate.is_finite() || !fit_b.rate.is_finite() {
        return Err("TRACT decay fit produced non-finite relaxation rate".to_string());
    }
    if fit_b.rate <= fit_a.rate {
        return Err(format!(
            "TRACT requires R_beta > R_alpha (got R_beta = {:.4} s^-1 <= R_alpha = {:.4} s^-1)",
            fit_b.rate, fit_a.rate
        ));
    }

    let eta_xy = 0.5 * (fit_b.rate - fit_a.rate);
    let sigma_eta_xy = 0.5 * sigma_ra.hypot(sigma_rb);

    let tau_m_rigid = invert_tract_eta_xy_to_tau_m(
        eta_xy,
        sigma_eta_xy,
        input.field_mhz,
        1.0,
        0.0,
        &cfg.physics,
    )?;

    let tau_m_s2_corrected = match cfg.order_parameter_s2 {
        Some(s2) => Some(invert_tract_eta_xy_to_tau_m(
            eta_xy,
            sigma_eta_xy,
            input.field_mhz,
            s2,
            cfg.tau_e_sec,
            &cfg.physics,
        )?),
        None => None,
    };

    let tau_m_from_eta_ratio = match input.eta_z {
        Some(eta_z) => {
            let sigma_ez = input.sigma_eta_z.unwrap_or(0.0);
            Some(invert_eta_ratio_to_tau_m(
                eta_xy,
                sigma_eta_xy,
                eta_z,
                sigma_ez,
                input.field_mhz,
                &cfg.physics,
            )?)
        }
        None => None,
    };

    Ok(TractFitResult {
        label: input.label.clone(),
        seq_id: input.seq_id,
        field_mhz: input.field_mhz,
        i0_alpha: fit_a.i0,
        r_alpha: fit_a.rate,
        sigma_r_alpha: sigma_ra,
        chi2_alpha: fit_a.chi2,
        i0_beta: fit_b.i0,
        r_beta: fit_b.rate,
        sigma_r_beta: sigma_rb,
        chi2_beta: fit_b.chi2,
        eta_xy,
        sigma_eta_xy,
        tau_m_rigid,
        tau_m_s2_corrected,
        tau_m_from_eta_ratio,
    })
}

/// Batch fit a set of residue-resolved 2D `TRACT` decay pairs.
pub fn fit_tract_batch(
    inputs: &[TractDecayInput],
    cfg: &TractConfig,
) -> Result<Vec<TractFitResult>, String> {
    if inputs.is_empty() {
        return Err("fit_tract_batch requires >= 1 TRACT decay pair".to_string());
    }
    inputs
        .iter()
        .map(|inp| fit_tract_decay_pair(inp, cfg))
        .collect()
}

// ============================================================================
// Kroenke–Palmer (1998) Model-Free R_ex Separation from Paired (R2, eta_xy)
// ============================================================================

/// Per-residue paired $(R_2, \eta_{xy})$ input for Kroenke–Palmer (1998) $R_{\text{ex}}$ separation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KroenkeResidueInput {
    pub seq_id: i32,
    pub res_name: String,
    pub r2: f64,
    pub sigma_r2: f64,
    pub eta_xy: f64,
    pub sigma_eta_xy: f64,
    /// Optional $R_1$ ($\text{s}^{-1}$) for subtracting high-frequency proton dipolar $R_2^{\text{HF}}$.
    pub r1: Option<f64>,
    pub sigma_r1: Option<f64>,
    /// Optional steady-state $\{^1\text{H}\}\text{–}^{15}\text{N}$ hetNOE for $R_2^{\text{HF}}$ correction.
    pub hetnoe: Option<f64>,
    pub sigma_hetnoe: Option<f64>,
}

/// Configuration for Kroenke, Loria, Cavanagh, Rance & Palmer (1998) $R_{\text{ex}}$ separation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KroenkeConfig {
    pub field_mhz: f64,
    pub physics: CcrPhysicsConfig,
    /// Statistical significance threshold $z = R_{\text{ex},i} / \sigma_{R_{\text{ex}},i}$ (default `3.0`).
    pub significance_z: f64,
    /// Upper-tail trimming threshold used to exclude exchanging residues from the rigid baseline
    /// $\kappa_0 = \text{median}(R_{2,\text{rigid}} / \eta_{xy})$ (default `2.5`).
    pub trim_z: f64,
    /// Maximum iterative rigid-core trimming passes (default `5`).
    pub max_trim_passes: usize,
    /// Subtract high-frequency proton dipolar $R_2^{\text{HF}}$ when `r1` and `hetnoe` are present
    /// on a residue (default `true`).
    pub subtract_high_freq_dd: bool,
    /// Optional fixed baseline $\kappa_0$ override (bypassing empirical median estimation).
    pub kappa0_override: Option<f64>,
}

impl Default for KroenkeConfig {
    fn default() -> Self {
        Self {
            field_mhz: 600.0,
            physics: CcrPhysicsConfig::default(),
            significance_z: 3.0,
            trim_z: 2.5,
            max_trim_passes: 5,
            subtract_high_freq_dd: true,
            kappa0_override: None,
        }
    }
}

/// Per-residue chemical exchange separation result from Kroenke–Palmer (1998) analysis.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KroenkeResidueResult {
    pub seq_id: i32,
    pub res_name: String,
    /// $R_2$ used in the separation ($R_2 - R_2^{\text{HF}}$ if high-frequency corrected, else $R_2$).
    pub r2_used: f64,
    pub eta_xy: f64,
    /// Per-residue ratio $\kappa_i = R_{2,\text{used},i} / \eta_{xy,i}$.
    pub kappa_i: f64,
    pub sigma_kappa_i: f64,
    /// Separated chemical exchange contribution $R_{\text{ex},i} = R_{2,\text{used},i} - \kappa_0 \eta_{xy,i}$ ($\text{s}^{-1}$).
    pub r_ex: f64,
    /// Propagated $1\sigma$ uncertainty $\sigma_{R_{\text{ex}},i}$ ($\text{s}^{-1}$).
    pub sigma_r_ex: f64,
    /// Statistical significance $z_i = R_{\text{ex},i} / \sigma_{R_{\text{ex}},i}$.
    pub z_score: f64,
    /// `true` when $R_{\text{ex},i} > 0$ and $z_i > \text{significance\_z}$ (default $3\sigma$).
    pub has_significant_rex: bool,
    /// Whether this residue was retained in the rigid core used to estimate $\kappa_0$.
    pub used_in_rigid_baseline: bool,
}

/// Protein-wide Kroenke–Palmer (1998) $R_{\text{ex}}$ separation report.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KroenkeRexReport {
    /// Estimated rigid-core baseline ratio $\kappa_0 = \text{median}(R_{2,\text{rigid}} / \eta_{xy})$.
    pub kappa0: f64,
    /// Propagated $1\sigma$ uncertainty on $\kappa_0$.
    pub sigma_kappa0: f64,
    /// Theoretical secular $\kappa_{\text{sec}}$ from `CcrPhysicsConfig` at `field_mhz`.
    pub theoretical_kappa0: f64,
    pub rigid_seq_ids: Vec<i32>,
    pub exchanging_seq_ids: Vec<i32>,
    pub residues: Vec<KroenkeResidueResult>,
}

/// Separate chemical exchange $R_{\text{ex},i} = R_{2,i} - \kappa_0 \eta_{xy,i}$ from paired
/// $(R_2, \eta_{xy})$ observations across residues (Kroenke, Loria, Cavanagh, Rance & Palmer 1998).
pub fn separate_rex_kroenke_palmer(
    inputs: &[KroenkeResidueInput],
    cfg: &KroenkeConfig,
) -> Result<KroenkeRexReport, String> {
    if inputs.is_empty() {
        return Err("separate_rex_kroenke_palmer requires >= 1 residue".to_string());
    }
    validate_field_mhz(cfg.field_mhz)?;
    cfg.physics.validate()?;
    if !cfg.significance_z.is_finite() || cfg.significance_z <= 0.0 {
        return Err("significance_z must be finite and > 0".to_string());
    }
    if !cfg.trim_z.is_finite() || cfg.trim_z <= 0.0 {
        return Err("trim_z must be finite and > 0".to_string());
    }
    if let Some(k0) = cfg.kappa0_override {
        if !k0.is_finite() || k0 <= 0.0 {
            return Err("kappa0_override must be finite and > 0".to_string());
        }
    }

    let theoretical_kappa0 = cfg.physics.theoretical_kappa0(cfg.field_mhz)?;
    let (_, alpha_r2_hf, r_gn) = high_freq_dipolar_factors(&cfg.physics);

    // Validate and prepare effective (r2_used, sigma_r2_used, kappa_i, sigma_kappa_i)
    struct Prep {
        r2_used: f64,
        sigma_r2_used: f64,
        eta_xy: f64,
        sigma_eta_xy: f64,
        kappa_i: f64,
        sigma_kappa_i: f64,
    }
    let mut preps = Vec::with_capacity(inputs.len());
    for inp in inputs {
        if !inp.r2.is_finite() || inp.r2 <= 0.0 {
            return Err(format!("residue {} r2 must be finite and > 0", inp.seq_id));
        }
        if !inp.sigma_r2.is_finite() || inp.sigma_r2 <= 0.0 {
            return Err(format!(
                "residue {} sigma_r2 must be finite and > 0",
                inp.seq_id
            ));
        }
        if !inp.eta_xy.is_finite() || inp.eta_xy <= 0.0 {
            return Err(format!(
                "residue {} eta_xy must be finite and > 0",
                inp.seq_id
            ));
        }
        if !inp.sigma_eta_xy.is_finite() || inp.sigma_eta_xy <= 0.0 {
            return Err(format!(
                "residue {} sigma_eta_xy must be finite and > 0",
                inp.seq_id
            ));
        }

        let (r2_used, sigma_r2_used) = if cfg.subtract_high_freq_dd {
            if let (Some(r1), Some(noe)) = (inp.r1, inp.hetnoe) {
                if !r1.is_finite() || r1 <= 0.0 || !noe.is_finite() {
                    return Err(format!(
                        "residue {} optional r1 (> 0) and hetnoe must be finite",
                        inp.seq_id
                    ));
                }
                let sr1 = inp.sigma_r1.unwrap_or(0.0);
                let snoe = inp.sigma_hetnoe.unwrap_or(0.0);
                if !sr1.is_finite() || sr1 < 0.0 || !snoe.is_finite() || snoe < 0.0 {
                    return Err(format!(
                        "residue {} optional sigma_r1 and sigma_hetnoe must be finite and >= 0",
                        inp.seq_id
                    ));
                }
                let sigma_nh = (noe - 1.0) * r1 * r_gn;
                let r2_hf = alpha_r2_hf * sigma_nh;
                let r2_corr = inp.r2 - r2_hf;
                if r2_corr <= 0.0 {
                    return Err(format!(
                        "residue {} high-frequency-corrected R2 <= 0 ({:.4})",
                        inp.seq_id, r2_corr
                    ));
                }
                let d_r2_dr1 = -alpha_r2_hf * (noe - 1.0) * r_gn;
                let d_r2_dnoe = -alpha_r2_hf * r1 * r_gn;
                let var = inp.sigma_r2 * inp.sigma_r2
                    + (d_r2_dr1 * sr1).powi(2)
                    + (d_r2_dnoe * snoe).powi(2);
                (r2_corr, var.sqrt())
            } else {
                (inp.r2, inp.sigma_r2)
            }
        } else {
            (inp.r2, inp.sigma_r2)
        };

        let kappa_i = r2_used / inp.eta_xy;
        let sigma_kappa_i =
            kappa_i * (sigma_r2_used / r2_used).hypot(inp.sigma_eta_xy / inp.eta_xy);
        preps.push(Prep {
            r2_used,
            sigma_r2_used,
            eta_xy: inp.eta_xy,
            sigma_eta_xy: inp.sigma_eta_xy,
            kappa_i,
            sigma_kappa_i,
        });
    }

    let n = preps.len();
    let mut active = vec![true; n];

    let (kappa0, sigma_kappa0) = if let Some(k0) = cfg.kappa0_override {
        (k0, 0.0)
    } else {
        // Iterative upper-tail outlier trimming to isolate rigid-core baseline kappa_0
        let min_rigid = (n.div_ceil(2)).min(n).max(1.min(n));
        for _pass in 0..cfg.max_trim_passes {
            let act_kappas: Vec<f64> = preps
                .iter()
                .zip(active.iter())
                .filter_map(|(p, &on)| on.then_some(p.kappa_i))
                .collect();
            let k_med = median_slice(&act_kappas);
            let abs_devs: Vec<f64> = act_kappas.iter().map(|&k| (k - k_med).abs()).collect();
            let sd_rob = 1.482_602_218_5 * median_slice(&abs_devs);

            let mut next_active = active.clone();
            let mut changed = false;
            for (i, p) in preps.iter().enumerate() {
                if !active[i] {
                    continue;
                }
                let r_ex_tmp = p.r2_used - k_med * p.eta_xy;
                let sig_meas = p.sigma_r2_used.hypot(k_med * p.sigma_eta_xy);
                let upper_sig = r_ex_tmp > cfg.trim_z * sig_meas;
                let upper_mad = sd_rob > 1.0e-12
                    && (p.kappa_i - k_med) > 3.5 * sd_rob
                    && r_ex_tmp > 2.0 * sig_meas;
                if upper_sig || upper_mad {
                    next_active[i] = false;
                    changed = true;
                }
            }

            let count_on = next_active.iter().filter(|&&b| b).count();
            if count_on >= min_rigid {
                active = next_active;
                if !changed {
                    break;
                }
            } else {
                // Keep the `min_rigid` smallest-kappa residues as rigid core
                let mut order: Vec<usize> = (0..n).collect();
                order.sort_by(|&a, &b| preps[a].kappa_i.total_cmp(&preps[b].kappa_i));
                active.fill(false);
                for &idx in order.iter().take(min_rigid) {
                    active[idx] = true;
                }
                break;
            }
        }

        let rigid_kappas: Vec<f64> = preps
            .iter()
            .zip(active.iter())
            .filter_map(|(p, &on)| on.then_some(p.kappa_i))
            .collect();
        let k0 = median_slice(&rigid_kappas);
        let m = rigid_kappas.len();

        // Propagated measurement SE of the rigid-core median (asymptotic sqrt(pi/2) factor)
        let inv_var_sum: f64 = preps
            .iter()
            .zip(active.iter())
            .filter_map(|(p, &on)| on.then_some(1.0 / (p.sigma_kappa_i * p.sigma_kappa_i)))
            .sum();
        let se_meas = if inv_var_sum > 1.0e-30 {
            1.253_314_137_3 / inv_var_sum.sqrt()
        } else {
            0.0
        };
        let se_emp = if m >= 2 {
            let mean = rigid_kappas.iter().sum::<f64>() / (m as f64);
            let var = rigid_kappas
                .iter()
                .map(|&k| (k - mean) * (k - mean))
                .sum::<f64>()
                / ((m - 1) as f64);
            1.253_314_137_3 * (var.sqrt() / (m as f64).sqrt())
        } else {
            se_meas
        };
        (k0, se_emp.max(se_meas))
    };

    let mut residues = Vec::with_capacity(n);
    let mut rigid_seq_ids = Vec::new();
    let mut exchanging_seq_ids = Vec::new();

    for (i, (inp, p)) in inputs.iter().zip(preps.iter()).enumerate() {
        let r_ex = p.r2_used - kappa0 * p.eta_xy;
        let sigma_r_ex = (p.sigma_r2_used * p.sigma_r2_used
            + (kappa0 * p.sigma_eta_xy).powi(2)
            + (p.eta_xy * sigma_kappa0).powi(2))
        .sqrt();
        let z_score = if sigma_r_ex > 1.0e-30 {
            r_ex / sigma_r_ex
        } else {
            0.0
        };
        let has_significant_rex = r_ex > 0.0 && z_score > cfg.significance_z;

        if has_significant_rex {
            exchanging_seq_ids.push(inp.seq_id);
        }
        if active[i] {
            rigid_seq_ids.push(inp.seq_id);
        }

        residues.push(KroenkeResidueResult {
            seq_id: inp.seq_id,
            res_name: inp.res_name.clone(),
            r2_used: p.r2_used,
            eta_xy: p.eta_xy,
            kappa_i: p.kappa_i,
            sigma_kappa_i: p.sigma_kappa_i,
            r_ex,
            sigma_r_ex,
            z_score,
            has_significant_rex,
            used_in_rigid_baseline: active[i],
        });
    }

    Ok(KroenkeRexReport {
        kappa0,
        sigma_kappa0,
        theoretical_kappa0,
        rigid_seq_ids,
        exchanging_seq_ids,
        residues,
    })
}

// ============================================================================
// Tjandra, Szabo & Bax (1996) Site-Specific 15N CSA Solver
// ============================================================================

/// Single-field relaxation and cross-correlated relaxation observations for one residue.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SiteCsaFieldObs {
    pub field_mhz: f64,
    pub r1: Option<f64>,
    pub sigma_r1: Option<f64>,
    pub r2: Option<f64>,
    pub sigma_r2: Option<f64>,
    pub hetnoe: Option<f64>,
    pub sigma_hetnoe: Option<f64>,
    pub eta_xy: Option<f64>,
    pub sigma_eta_xy: Option<f64>,
    pub eta_z: Option<f64>,
    pub sigma_eta_z: Option<f64>,
}

/// Multi-field (or single-field) relaxation/CCR dataset for one residue's $^{15}\text{N}$ CSA fit.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SiteCsaInput {
    pub seq_id: i32,
    pub res_name: String,
    pub fields: Vec<SiteCsaFieldObs>,
}

/// Configuration for Tjandra, Szabo & Bax (1996) site-specific $^{15}\text{N}$ CSA determination.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SiteCsaConfig {
    /// Physical constants (`r_nh_angstrom`, `theta_deg`, `gamma_h`, `gamma_n`); `csa_n_ppm` serves
    /// only as the canonical prior center (`-172.0 ppm`) for breaking single-field quadratic branch
    /// degeneracy.
    pub physics: CcrPhysicsConfig,
    /// Include transverse $R_2' / \eta_{xy}$ ratios when available (automatically excluded at any
    /// field where $q_{xy}$ exceeds $q_z$ by $> \text{rex\_exclusion\_z} \cdot \sigma$).
    pub use_r2_eta_xy: bool,
    /// Significance threshold (in $\sigma$) for automatically excluding $R_2' / \eta_{xy}$ when
    /// chemical exchange $R_{\text{ex}}$ is detected relative to $R_1' / \eta_z$ (default `3.0`).
    pub rex_exclusion_z: f64,
    /// Lower bound of physical $^{15}\text{N}$ CSA range (default `-250.0 ppm`).
    pub physical_min_ppm: f64,
    /// Upper bound of physical $^{15}\text{N}$ CSA range (default `-100.0 ppm`).
    pub physical_max_ppm: f64,
}

impl Default for SiteCsaConfig {
    fn default() -> Self {
        Self {
            physics: CcrPhysicsConfig::default(),
            use_r2_eta_xy: true,
            rex_exclusion_z: 3.0,
            physical_min_ppm: CSA_N_PHYSICAL_MIN_PPM,
            physical_max_ppm: CSA_N_PHYSICAL_MAX_PPM,
        }
    }
}

/// Physical-range diagnostic classification for fitted $^{15}\text{N}$ CSA $\Delta\sigma_N$.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CsaRangeStatus {
    /// Fitted $\Delta\sigma_N \in [-250, -100]\text{ ppm}$ (canonical backbone amide range).
    InPhysicalRange,
    /// Fitted $\Delta\sigma_N < -250\text{ ppm}$ (anomalously large CSA magnitude).
    BelowPhysicalRange,
    /// Fitted $\Delta\sigma_N > -100\text{ ppm}$ (anomalously small CSA magnitude or inverted sign).
    AbovePhysicalRange,
}

/// Result of Tjandra, Szabo & Bax (1996) site-specific $^{15}\text{N}$ CSA determination.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SiteCsaResult {
    pub seq_id: i32,
    pub res_name: String,
    /// Fitted site-specific $^{15}\text{N}$ CSA $\Delta\sigma_N = \sigma_\parallel - \sigma_\perp$ (ppm).
    pub csa_n_ppm: f64,
    /// Propagated $1\sigma$ Cramér–Rao / Fisher uncertainty on $\Delta\sigma_N$ (ppm).
    pub sigma_csa_n_ppm: f64,
    pub range_status: CsaRangeStatus,
    pub in_physical_range: bool,
    /// `true` if $R_2'/\eta_{xy}$ significantly exceeded $R_1'/\eta_z$ at one or more fields and
    /// was excluded to prevent chemical exchange $R_{\text{ex}}$ from biasing $\Delta\sigma_N$.
    pub rex_detected_on_r2: bool,
    pub n_observables_used: usize,
    pub chi2: f64,
    pub dof: usize,
    pub warnings: Vec<String>,
}

/// Solve site-specific $^{15}\text{N}$ CSA $\Delta\sigma_{N,i}$ (ppm) from single- or multi-field
/// paired $(\eta_{xy}, \eta_z, R_1, R_2, \text{hetNOE})$ observables via the exact model-free
/// Tjandra, Szabo & Bax (1996) ratio relation:
///
/// $$\frac{R_1 - R_1^{\text{HF}}}{\eta_z} = \frac{R_2 - R_2^{\text{HF}}}{\eta_{xy}} = \frac{\frac{3}{4} + u^2}{\sqrt{3}\, P_2(\cos\theta)\, u}, \qquad u = \frac{\omega_N \Delta\sigma_N}{\sqrt{3}\, d}$$
///
/// Spectral densities $J(0)$ and $J(\omega_N)$ cancel identically in both ratios, making the
/// recovered $\Delta\sigma_{N,i}$ independent of $\tau_m$, $S^2$, $\tau_e$, and rotational
/// diffusion anisotropy.
pub fn fit_site_specific_csa(
    input: &SiteCsaInput,
    cfg: &SiteCsaConfig,
) -> Result<SiteCsaResult, String> {
    cfg.physics.validate()?;
    if !cfg.physical_min_ppm.is_finite()
        || !cfg.physical_max_ppm.is_finite()
        || cfg.physical_min_ppm >= cfg.physical_max_ppm
    {
        return Err("physical CSA range bounds must be finite with min < max".to_string());
    }
    if !cfg.rex_exclusion_z.is_finite() || cfg.rex_exclusion_z <= 0.0 {
        return Err("rex_exclusion_z must be finite and > 0".to_string());
    }
    if input.fields.is_empty() {
        return Err(format!(
            "residue {} site-specific CSA fit requires >= 1 magnetic field entry",
            input.seq_id
        ));
    }

    let p2 = cfg.physics.p2_cos_theta();
    if p2 <= 1.0e-12 {
        return Err(
            "P2(cos theta) must be strictly positive for site-specific CSA fit".to_string(),
        );
    }
    let d = cfg.physics.dipolar_d_rad_s();
    let (alpha_r1_hf, alpha_r2_hf, r_gn) = high_freq_dipolar_factors(&cfg.physics);

    struct RatioObs {
        field_mhz: f64,
        q_obs: f64,
        sigma_q: f64,
    }
    let mut obs: Vec<RatioObs> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();
    let mut rex_detected_on_r2 = false;

    for f in &input.fields {
        validate_field_mhz(f.field_mhz)?;
        validate_opt_pos("r1", f.r1)?;
        validate_opt_pos("sigma_r1", f.sigma_r1)?;
        validate_opt_pos("r2", f.r2)?;
        validate_opt_pos("sigma_r2", f.sigma_r2)?;
        validate_opt_pos("eta_xy", f.eta_xy)?;
        validate_opt_pos("sigma_eta_xy", f.sigma_eta_xy)?;
        validate_opt_pos("eta_z", f.eta_z)?;
        validate_opt_pos("sigma_eta_z", f.sigma_eta_z)?;
        if let Some(noe) = f.hetnoe {
            if !noe.is_finite() || !(-5.0..=2.0).contains(&noe) {
                return Err(format!(
                    "residue {} hetNOE must be finite and within [-5, 2]",
                    input.seq_id
                ));
            }
        }
        validate_opt_pos("sigma_hetnoe", f.sigma_hetnoe)?;

        // Longitudinal ratio q_z = (R1 - R1^HF) / eta_z (strictly R_ex-free)
        let q_z_entry = match (f.r1, f.hetnoe, f.eta_z) {
            (Some(r1), Some(noe), Some(eta_z)) => {
                let sr1 = f.sigma_r1.ok_or_else(|| {
                    format!(
                        "residue {} missing sigma_r1 at {:.1} MHz",
                        input.seq_id, f.field_mhz
                    )
                })?;
                let snoe = f.sigma_hetnoe.ok_or_else(|| {
                    format!(
                        "residue {} missing sigma_hetnoe at {:.1} MHz",
                        input.seq_id, f.field_mhz
                    )
                })?;
                let sez = f.sigma_eta_z.ok_or_else(|| {
                    format!(
                        "residue {} missing sigma_eta_z at {:.1} MHz",
                        input.seq_id, f.field_mhz
                    )
                })?;
                let sigma_nh = (noe - 1.0) * r1 * r_gn;
                let r1_prime = r1 - alpha_r1_hf * sigma_nh;
                if r1_prime <= 0.0 {
                    return Err(format!(
                        "residue {} high-frequency-corrected R1' <= 0 at {:.1} MHz",
                        input.seq_id, f.field_mhz
                    ));
                }
                let d_r1p_dr1 = 1.0 - alpha_r1_hf * (noe - 1.0) * r_gn;
                let d_r1p_dnoe = -alpha_r1_hf * r1 * r_gn;
                let sig_r1p = (d_r1p_dr1 * sr1).hypot(d_r1p_dnoe * snoe);
                let q_z = r1_prime / eta_z;
                let sig_qz = (sig_r1p.hypot(q_z * sez)) / eta_z;
                Some((q_z, sig_qz))
            }
            _ => None,
        };

        // Transverse ratio q_xy = (R2 - R2^HF) / eta_xy (requires R_ex == 0)
        let q_xy_entry = if cfg.use_r2_eta_xy {
            match (f.r2, f.eta_xy) {
                (Some(r2), Some(eta_xy)) => {
                    let sr2 = f.sigma_r2.ok_or_else(|| {
                        format!(
                            "residue {} missing sigma_r2 at {:.1} MHz",
                            input.seq_id, f.field_mhz
                        )
                    })?;
                    let sexy = f.sigma_eta_xy.ok_or_else(|| {
                        format!(
                            "residue {} missing sigma_eta_xy at {:.1} MHz",
                            input.seq_id, f.field_mhz
                        )
                    })?;
                    let (r2_prime, sig_r2p) = if let (Some(r1), Some(noe)) = (f.r1, f.hetnoe) {
                        let sr1 = f.sigma_r1.unwrap_or(0.0);
                        let snoe = f.sigma_hetnoe.unwrap_or(0.0);
                        let sigma_nh = (noe - 1.0) * r1 * r_gn;
                        let r2p = r2 - alpha_r2_hf * sigma_nh;
                        if r2p <= 0.0 {
                            return Err(format!(
                                "residue {} high-frequency-corrected R2' <= 0 at {:.1} MHz",
                                input.seq_id, f.field_mhz
                            ));
                        }
                        let d_r2p_dr1 = -alpha_r2_hf * (noe - 1.0) * r_gn;
                        let d_r2p_dnoe = -alpha_r2_hf * r1 * r_gn;
                        let var =
                            sr2 * sr2 + (d_r2p_dr1 * sr1).powi(2) + (d_r2p_dnoe * snoe).powi(2);
                        (r2p, var.sqrt())
                    } else {
                        warnings.push(format!(
                            "field {:.1} MHz: R2 high-frequency dipolar term not subtracted (missing R1/hetNOE)",
                            f.field_mhz
                        ));
                        (r2, sr2)
                    };
                    let q_xy = r2_prime / eta_xy;
                    let sig_qxy = (sig_r2p.hypot(q_xy * sexy)) / eta_xy;
                    Some((q_xy, sig_qxy))
                }
                _ => None,
            }
        } else {
            None
        };

        if let Some((q_z, sig_qz)) = q_z_entry {
            obs.push(RatioObs {
                field_mhz: f.field_mhz,
                q_obs: q_z,
                sigma_q: sig_qz,
            });
        }
        if let Some((q_xy, sig_qxy)) = q_xy_entry {
            let contaminated = if let Some((q_z, sig_qz)) = q_z_entry {
                let z_diff = (q_xy - q_z) / sig_qxy.hypot(sig_qz).max(1.0e-30);
                z_diff > cfg.rex_exclusion_z
            } else {
                false
            };
            if contaminated {
                rex_detected_on_r2 = true;
                warnings.push(format!(
                    "field {:.1} MHz: R2'/eta_xy significantly exceeds R1'/eta_z; excluded R2 due to chemical exchange R_ex",
                    f.field_mhz
                ));
            } else {
                obs.push(RatioObs {
                    field_mhz: f.field_mhz,
                    q_obs: q_xy,
                    sigma_q: sig_qxy,
                });
            }
        }
    }

    if obs.is_empty() {
        return Err(format!(
            "residue {} has no valid (R1, hetNOE, eta_z) or (R2, eta_xy) ratio observables",
            input.seq_id
        ));
    }

    let sqrt3 = 3.0_f64.sqrt();
    let eval_q_and_dq = |csa_ppm: f64, field_mhz: f64| -> (f64, f64) {
        let (_, omega_n) = cfg.physics.signed_omegas_rad_s(field_mhz);
        let du_dcsa = (omega_n * 1.0e-6) / (sqrt3 * d); // negative because omega_n < 0
        let u = du_dcsa * csa_ppm; // positive for csa_ppm < 0
        let q = (0.75 + u * u) / (sqrt3 * p2 * u);
        let dq_du = (1.0 - 0.75 / (u * u)) / (sqrt3 * p2);
        (q, dq_du * du_dcsa)
    };

    let eval_chi2 = |csa_ppm: f64| -> f64 {
        let mut s = 0.0_f64;
        for o in &obs {
            let (q, _) = eval_q_and_dq(csa_ppm, o.field_mhz);
            let r = (o.q_obs - q) / o.sigma_q;
            s += r * r;
        }
        s
    };

    // Global 1D scan across [-600, -20] ppm (0.5 ppm step) to capture all candidate local minima
    // including both quadratic branches if single-field data is supplied.
    let grid_min = -600.0_f64;
    let grid_max = -20.0_f64;
    let n_steps = 1160usize;
    let step = (grid_max - grid_min) / (n_steps as f64);
    let mut grid_x = Vec::with_capacity(n_steps + 1);
    let mut grid_y = Vec::with_capacity(n_steps + 1);
    for i in 0..=n_steps {
        let x = grid_min + (i as f64) * step;
        grid_x.push(x);
        grid_y.push(eval_chi2(x));
    }

    // Collect local minima indices
    let mut minima: Vec<(f64, f64)> = Vec::new();
    for i in 0..=n_steps {
        let is_min =
            (i == 0 || grid_y[i] <= grid_y[i - 1]) && (i == n_steps || grid_y[i] <= grid_y[i + 1]);
        if is_min {
            let lo = if i == 0 { grid_x[0] } else { grid_x[i - 1] };
            let hi = if i == n_steps {
                grid_x[n_steps]
            } else {
                grid_x[i + 1]
            };
            let (x_ref, y_ref) = golden_section_min(lo, hi, &eval_chi2, 80);
            minima.push((x_ref, y_ref));
        }
    }

    // Select global minimum; if two branches are mathematically degenerate (delta_chi2 < 1e-6),
    // prefer the root closest to canonical backbone prior (cfg.physics.csa_n_ppm, default -172 ppm).
    let best_chi2 = minima.iter().map(|&(_, y)| y).fold(f64::INFINITY, f64::min);
    let prior_center = cfg.physics.csa_n_ppm;
    let mut chosen = (prior_center, eval_chi2(prior_center));
    let mut best_dist = f64::INFINITY;
    let mut n_degenerate = 0usize;
    for &(x, y) in &minima {
        if (y - best_chi2) <= 1.0e-6 * best_chi2.max(1.0) {
            n_degenerate += 1;
            let dist = (x - prior_center).abs();
            if dist < best_dist {
                best_dist = dist;
                chosen = (x, y);
            }
        }
    }
    if n_degenerate > 1 {
        warnings.push(
            "single-field quadratic branch degeneracy resolved toward canonical backbone CSA prior"
                .to_string(),
        );
    }

    let csa_n_ppm = chosen.0;
    let chi2 = chosen.1;

    // Exact Cramér–Rao / Fisher standard error on Delta sigma_N
    let mut fisher_info = 0.0_f64;
    for o in &obs {
        let (_, dq_dcsa) = eval_q_and_dq(csa_n_ppm, o.field_mhz);
        let term = dq_dcsa / o.sigma_q;
        fisher_info += term * term;
    }

    let sigma_csa_n_ppm = if fisher_info > 1.0e-14 {
        1.0 / fisher_info.sqrt()
    } else {
        // Fallback curvature near the u = sqrt(3)/2 turning point (~1 GHz)
        warnings.push(
            "CSA fit lies near the single-field turning point (u ~ sqrt(3)/2); uncertainty inflated"
                .to_string(),
        );
        let h = 0.25_f64;
        let d2 = (eval_chi2(csa_n_ppm + h) - 2.0 * chi2 + eval_chi2(csa_n_ppm - h)) / (h * h);
        if d2 > 1.0e-12 {
            (2.0 / d2).sqrt()
        } else {
            f64::INFINITY
        }
    };

    let range_status = if csa_n_ppm < cfg.physical_min_ppm {
        warnings.push(format!(
            "fitted 15N CSA ({:.2} ppm) is below physical range [{:.1}, {:.1}] ppm",
            csa_n_ppm, cfg.physical_min_ppm, cfg.physical_max_ppm
        ));
        CsaRangeStatus::BelowPhysicalRange
    } else if csa_n_ppm > cfg.physical_max_ppm {
        warnings.push(format!(
            "fitted 15N CSA ({:.2} ppm) is above physical range [{:.1}, {:.1}] ppm",
            csa_n_ppm, cfg.physical_min_ppm, cfg.physical_max_ppm
        ));
        CsaRangeStatus::AbovePhysicalRange
    } else {
        CsaRangeStatus::InPhysicalRange
    };

    Ok(SiteCsaResult {
        seq_id: input.seq_id,
        res_name: input.res_name.clone(),
        csa_n_ppm,
        sigma_csa_n_ppm,
        range_status,
        in_physical_range: range_status == CsaRangeStatus::InPhysicalRange,
        rex_detected_on_r2,
        n_observables_used: obs.len(),
        chi2,
        dof: obs.len().saturating_sub(1),
        warnings,
    })
}

/// Batch fit site-specific $^{15}\text{N}$ CSA across a residue slice.
pub fn fit_site_specific_csa_batch(
    inputs: &[SiteCsaInput],
    cfg: &SiteCsaConfig,
) -> Result<Vec<SiteCsaResult>, String> {
    if inputs.is_empty() {
        return Err("fit_site_specific_csa_batch requires >= 1 residue".to_string());
    }
    inputs
        .iter()
        .map(|inp| fit_site_specific_csa(inp, cfg))
        .collect()
}

// ============================================================================
// Internal Helpers
// ============================================================================

/// High-frequency dipolar scaling constants `(alpha_r1_hf, alpha_r2_hf, gamma_n / gamma_h)`
/// matching `relax_fit::map_reduced_spectral_density` under $\omega^{-2}$ scaling near $\omega_H$.
fn high_freq_dipolar_factors(phys: &CcrPhysicsConfig) -> (f64, f64, f64) {
    let r_gn = phys.gamma_n / phys.gamma_h;
    let f_sum = 1.0 + r_gn;
    let f_dif = 1.0 - r_gn;
    let k_sigma = 6.0 / (f_sum * f_sum) - 1.0 / (f_dif * f_dif);
    let k_r1_hf = 6.0 / (f_sum * f_sum) + 1.0 / (f_dif * f_dif);
    let k_r2_hf = 6.0 + 6.0 / (f_sum * f_sum) + 1.0 / (f_dif * f_dif);
    (k_r1_hf / k_sigma, k_r2_hf / (2.0 * k_sigma), r_gn)
}

fn validate_field_mhz(field_mhz: f64) -> Result<(), String> {
    if !field_mhz.is_finite() || field_mhz <= 0.0 {
        return Err("field_mhz must be finite and > 0".to_string());
    }
    Ok(())
}

fn validate_dynamics_params(tau_m_sec: f64, s2: f64, tau_e_sec: f64) -> Result<(), String> {
    if !tau_m_sec.is_finite() || tau_m_sec <= 0.0 {
        return Err("tau_m_sec must be finite and > 0".to_string());
    }
    if !s2.is_finite() || s2 <= 0.0 || s2 > 1.0 {
        return Err("s2 must be finite and in (0, 1]".to_string());
    }
    if !tau_e_sec.is_finite() || tau_e_sec < 0.0 {
        return Err("tau_e_sec must be finite and >= 0".to_string());
    }
    Ok(())
}

fn validate_opt_pos(name: &str, v: Option<f64>) -> Result<(), String> {
    if let Some(val) = v {
        if !val.is_finite() || val <= 0.0 {
            return Err(format!("{name} must be finite and > 0"));
        }
    }
    Ok(())
}

fn validate_decay_series(
    tag: &str,
    times: &[f64],
    intensities: &[f64],
    sigmas: &[f64],
) -> Result<(), String> {
    let n = times.len();
    if n < 2 || intensities.len() != n || sigmas.len() != n {
        return Err(format!(
            "TRACT {tag} series requires >= 2 matching (time, intensity, sigma) points"
        ));
    }
    let mut t_min = f64::INFINITY;
    let mut t_max = f64::NEG_INFINITY;
    let mut any_pos_i = false;
    for i in 0..n {
        let t = times[i];
        let y = intensities[i];
        let s = sigmas[i];
        if !t.is_finite() || t < 0.0 {
            return Err(format!(
                "TRACT {tag} delay at index {i} must be finite and >= 0"
            ));
        }
        if !y.is_finite() {
            return Err(format!("TRACT {tag} intensity at index {i} must be finite"));
        }
        if !s.is_finite() || s <= 0.0 {
            return Err(format!(
                "TRACT {tag} sigma at index {i} must be finite and > 0"
            ));
        }
        if t < t_min {
            t_min = t;
        }
        if t > t_max {
            t_max = t;
        }
        if y > 0.0 {
            any_pos_i = true;
        }
    }
    if t_max <= t_min {
        return Err(format!(
            "TRACT {tag} delays must span a strictly positive time range"
        ));
    }
    if !any_pos_i {
        return Err(format!(
            "TRACT {tag} intensities must contain at least one positive value"
        ));
    }
    Ok(())
}

fn median_slice(vals: &[f64]) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    let mut sorted = vals.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
    }
}

fn golden_section_min<F: Fn(f64) -> f64>(
    mut a: f64,
    mut b: f64,
    f: &F,
    iters: usize,
) -> (f64, f64) {
    let inv_phi = 0.5 * (5.0_f64.sqrt() - 1.0);
    let mut c = b - inv_phi * (b - a);
    let mut d = a + inv_phi * (b - a);
    let mut fc = f(c);
    let mut fd = f(d);
    for _ in 0..iters {
        if (b - a) <= 1.0e-12 * b.abs().max(a.abs()).max(1.0) {
            break;
        }
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - inv_phi * (b - a);
            fc = f(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + inv_phi * (b - a);
            fd = f(d);
        }
    }
    let mid = 0.5 * (a + b);
    (mid, f(mid))
}
