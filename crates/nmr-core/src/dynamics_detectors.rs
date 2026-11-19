//! Model-Free Linear Optimal-Window Timescale Dynamics Detectors ($\rho_n(\tau)$),
//! Fixed-Timescale `LeMaster` Spectral Density Decomposition, and Multi-Field Stacked `IMPACT`
//! Simplex Correlation-Time Decomposition (`T22.3`, `T22.4`).
//!
//! Provenance: Clean-room implementation strictly from peer-reviewed NMR spin-relaxation,
//! spectral-density decomposition, and convex optimization literature (zero external GPL, AGPL,
//! or proprietary/academic-licence software consulted; zero external product names in identifiers):
//! - Smith, A. A., Ernst, M. & Meier, B. H. (*J. Chem. Phys.* **2018**, 148, 044110) and
//!   Smith, A. A., Zumpfe, K. & Meier, B. H. (*J. Chem. Phys.* **2019**, 151, 034102) —
//!   linear optimal-window model-free timescale dynamics detectors $\rho_n(\tau)$ from SVD of
//!   the relaxation sensitivity matrix $R_i(\tau)$, localized non-negative unit-peak windows,
//!   analytical error propagation, and back-calculated relaxation residuals.
//! - LeMaster, D. M. (*J. Biomol. NMR* **1995**, 6, 366–374) — fixed-timescale analytical
//!   decomposition of single-field reduced spectral densities $(J(0), J(\omega_N), J(0.87\omega_H))$
//!   at $(\tau_m, |\omega_N|^{-1}, (0.87\omega_H)^{-1})$.
//! - Khan, S. N., Charlier, C., Augustyniak, R., Salvi, N., Déjean, V., Bodenhausen, G.,
//!   Lequin, O., Pelupessy, P. & Ferrage, F. (*Biophys. J.* **2015**, 109, 988–999) — `IMPACT`
//!   (Interpretation of Motions by a Projection onto an Array of Correlation Times), constrained
//!   non-negative sum-to-unity simplex weighted least-squares spectral density decomposition across
//!   multiple magnetic fields with parametric Monte Carlo uncertainty estimation and $\mathrm{AIC}_c$
//!   model-size selection.
//! - Kay, L. E., Torchia, D. A. & Bax, A. (*Biochemistry* **1989**, 28, 8972–8979) and
//!   Farrow, N. A. et al. (*J. Biomol. NMR* **1995**, 6, 153–162) — $^{15}\text{N}$ dipolar + CSA
//!   relaxation equations and reduced spectral density mapping.
//! - Duchi, J., Shalev-Shwartz, S., Singer, Y. & Chandra, T. (*ICML* **2008**, 272–279) — exact
//!   Euclidean projection onto the probability simplex.
//! - Lacoste-Julien, S. & Jaggi, M. (*NeurIPS* **2015**, 496–504) — away-step Frank–Wolfe linear
//!   convergence on polytopes.

#![allow(clippy::needless_range_loop)]

use serde::{Deserialize, Serialize};

use crate::relax_fit::{
    map_reduced_spectral_density, ReducedSpectralDensity, RelaxPhysicsConfig, ResidueRelaxRates,
};
use crate::rng::Rng;

// ============================================================================
// Physical Validation & Multi-Field 15N Relaxation Sensitivity Kernels
// ============================================================================

/// Linear $^{15}\text{N}$ backbone relaxation observable kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelaxObservableKind {
    /// Longitudinal relaxation rate $R_1$ ($\text{s}^{-1}$).
    R1,
    /// Transverse relaxation rate $R_2$ ($\text{s}^{-1}$).
    R2,
    /// Dipolar cross-relaxation rate $\Gamma_{\text{NOE}} = \sigma_{\text{NH}} = R_1(\mathrm{hetNOE} - 1)\gamma_N/\gamma_H$ ($\text{s}^{-1}$).
    GammaNoe,
}

impl RelaxObservableKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::R1 => "R1",
            Self::R2 => "R2",
            Self::GammaNoe => "GammaNOE",
        }
    }
}

/// Specification of one $^{15}\text{N}$ relaxation experiment and its $1\sigma$ measurement uncertainty.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RelaxExperiment {
    pub kind: RelaxObservableKind,
    /// Static magnetic field expressed as $^1\text{H}$ Larmor frequency in MHz (e.g. `400.0..=1200.0`).
    pub field_mhz: f64,
    /// $1\sigma$ measurement uncertainty in $\text{s}^{-1}$ (strictly positive).
    pub sigma: f64,
    /// Optional descriptive label (auto-populated by [`RelaxExperiment::new`]).
    pub label: String,
}

impl RelaxExperiment {
    pub fn new(kind: RelaxObservableKind, field_mhz: f64, sigma: f64) -> Self {
        Self {
            kind,
            field_mhz,
            sigma,
            label: format!("{}_{:.0}MHz", kind.as_str(), field_mhz),
        }
    }
}

/// Treatment of overall isotropic molecular tumbling $\tau_m$ in the relaxation sensitivity matrix.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, Default)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum TumblingMode {
    /// Pure total correlation-time spectrum $C(t) = \int \theta(\tau) e^{-t/\tau}\,d\ln\tau$
    /// (Smith et al. 2018; no tumbling separation).
    #[default]
    PureCorrelationSpectrum,
    /// Solution-NMR internal-motion detector basis (Smith et al. 2019) with known isotropic overall
    /// tumbling $\tau_m$: sensitivity kernel $\Delta R_i(\tau; \tau_m) = R_i^{(0)}(\tau_m) - R_i^{(0)}(\tau_{\text{eff}}(\tau))$
    /// where $\tau_{\text{eff}}^{-1} = \tau_m^{-1} + \tau^{-1}$, and detector input $y_i = R_i^{(0)}(\tau_m) - R_i^{\text{obs}}$.
    IsotropicInternal { tau_m_sec: f64 },
    /// Direct effective-timescale kernel $R_i(\tau) = R_i^{(0)}(\tau_{\text{eff}}(\tau))$ where
    /// $\tau_{\text{eff}}^{-1} = \tau_m^{-1} + \tau^{-1}$ and detector input $y_i = R_i^{\text{obs}}$.
    EffectiveTimescale { tau_m_sec: f64 },
}

/// Validate physical relaxation constants fail-closed.
pub fn validate_physics(phys: &RelaxPhysicsConfig) -> Result<(), String> {
    if !phys.r_nh_angstrom.is_finite() || phys.r_nh_angstrom <= 0.0 {
        return Err(format!(
            "invalid N-H bond length r_nh_angstrom = {} (must be finite and > 0)",
            phys.r_nh_angstrom
        ));
    }
    if !phys.csa_n_ppm.is_finite() {
        return Err(format!(
            "invalid CSA csa_n_ppm = {} (must be finite)",
            phys.csa_n_ppm
        ));
    }
    if !phys.gamma_h.is_finite() || phys.gamma_h <= 0.0 {
        return Err(format!(
            "invalid proton gyromagnetic ratio gamma_h = {} (must be finite and > 0)",
            phys.gamma_h
        ));
    }
    if !phys.gamma_n.is_finite() || phys.gamma_n == 0.0 {
        return Err(format!(
            "invalid 15N gyromagnetic ratio gamma_n = {} (must be finite and non-zero)",
            phys.gamma_n
        ));
    }
    Ok(())
}

/// Validate tumbling mode fail-closed.
fn validate_tumbling(tumbling: TumblingMode) -> Result<(), String> {
    match tumbling {
        TumblingMode::PureCorrelationSpectrum => Ok(()),
        TumblingMode::IsotropicInternal { tau_m_sec }
        | TumblingMode::EffectiveTimescale { tau_m_sec } => {
            if !tau_m_sec.is_finite() || tau_m_sec <= 0.0 {
                return Err(format!(
                    "invalid overall tumbling tau_m_sec = {} (must be finite and > 0)",
                    tau_m_sec
                ));
            }
            Ok(())
        }
    }
}

/// Generate a strictly increasing log-spaced correlation-time grid $\tau_j \in [\tau_{\min}, \tau_{\max}]$ (seconds).
pub fn log_spaced_grid(
    tau_min_sec: f64,
    tau_max_sec: f64,
    count: usize,
) -> Result<Vec<f64>, String> {
    if count == 0 {
        return Err("correlation-time grid count must be >= 1".to_string());
    }
    if !tau_min_sec.is_finite() || tau_min_sec <= 0.0 {
        return Err(format!(
            "invalid tau_min_sec = {} (must be finite and > 0)",
            tau_min_sec
        ));
    }
    if !tau_max_sec.is_finite() || tau_max_sec <= 0.0 {
        return Err(format!(
            "invalid tau_max_sec = {} (must be finite and > 0)",
            tau_max_sec
        ));
    }
    if count == 1 {
        return Ok(vec![tau_min_sec]);
    }
    if tau_max_sec <= tau_min_sec {
        return Err(format!(
            "tau_max_sec ({}) must be strictly greater than tau_min_sec ({})",
            tau_max_sec, tau_min_sec
        ));
    }
    let z_min = tau_min_sec.log10();
    let z_max = tau_max_sec.log10();
    let denom = (count - 1) as f64;
    let mut grid = Vec::with_capacity(count);
    for j in 0..count {
        let frac = (j as f64) / denom;
        grid.push(10.0_f64.powf(z_min + frac * (z_max - z_min)));
    }
    Ok(grid)
}

/// Single-exponential Lorentzian spectral density $J(\omega; \tau) = \frac{2}{5}\frac{\tau}{1 + (\omega\tau)^2}$ ($\text{s rad}^{-1}$).
#[inline]
pub fn lorentzian_j(omega_rad_s: f64, tau_sec: f64) -> f64 {
    let wt = omega_rad_s * tau_sec;
    0.4 * tau_sec / (1.0 + wt * wt)
}

/// Effective correlation time $\tau_{\text{eff}} = (\tau_m^{-1} + \tau_{\text{int}}^{-1})^{-1}$.
#[inline]
pub fn effective_tau_sec(tau_int_sec: f64, tau_m_sec: f64) -> f64 {
    if tau_int_sec <= 0.0 {
        return 0.0;
    }
    if !tau_int_sec.is_finite() {
        return tau_m_sec;
    }
    (tau_m_sec * tau_int_sec) / (tau_m_sec + tau_int_sec)
}

/// Exact single-exponential $^{15}\text{N}$ dipolar + CSA relaxation rate $R_i^{(0)}(B_0, \tau)$ ($\text{s}^{-1}$).
///
/// Uses the identical CODATA-2018 dipolar ($d^2$) and CSA ($c^2(\omega_N)$) physical kernels as
/// `crate::modelfree::predict_r1_r2_noe`.
#[inline]
pub fn single_tau_rate(
    kind: RelaxObservableKind,
    field_mhz: f64,
    tau_sec: f64,
    phys: &RelaxPhysicsConfig,
) -> f64 {
    let d = phys.dipolar_d_rad_s();
    let d2 = d * d;
    let (wh, wn) = phys.signed_omegas_rad_s(field_mhz);

    match kind {
        RelaxObservableKind::R1 => {
            let c = phys.csa_c_rad_s(field_mhz);
            let c2 = c * c;
            let jn = lorentzian_j(wn, tau_sec);
            let j_diff = lorentzian_j(wh - wn, tau_sec);
            let j_sum = lorentzian_j(wh + wn, tau_sec);
            0.25 * d2 * (j_diff + 3.0 * jn + 6.0 * j_sum) + c2 * jn
        }
        RelaxObservableKind::R2 => {
            let c = phys.csa_c_rad_s(field_mhz);
            let c2 = c * c;
            let j0 = lorentzian_j(0.0, tau_sec);
            let jn = lorentzian_j(wn, tau_sec);
            let jh = lorentzian_j(wh, tau_sec);
            let j_diff = lorentzian_j(wh - wn, tau_sec);
            let j_sum = lorentzian_j(wh + wn, tau_sec);
            0.125 * d2 * (4.0 * j0 + j_diff + 3.0 * jn + 6.0 * jh + 6.0 * j_sum)
                + (c2 / 6.0) * (4.0 * j0 + 3.0 * jn)
        }
        RelaxObservableKind::GammaNoe => {
            let j_diff = lorentzian_j(wh - wn, tau_sec);
            let j_sum = lorentzian_j(wh + wn, tau_sec);
            0.25 * d2 * (6.0 * j_sum - j_diff)
        }
    }
}

/// Convert steady-state heteronuclear NOE and $R_1$ to linear dipolar cross-relaxation rate
/// $\Gamma_{\text{NOE}} = R_1 (\mathrm{hetNOE} - 1) \gamma_N / \gamma_H$ ($\text{s}^{-1}$).
#[inline]
pub fn gamma_noe_from_r1_hetnoe(r1: f64, hetnoe: f64, phys: &RelaxPhysicsConfig) -> f64 {
    r1 * (hetnoe - 1.0) * (phys.gamma_n / phys.gamma_h)
}

/// Convert $(R_1, \sigma_{R_1}, \mathrm{hetNOE}, \sigma_{\mathrm{hetNOE}})$ to
/// $(\Gamma_{\text{NOE}}, \sigma_{\Gamma_{\text{NOE}}}, \mathrm{Cov}(R_1, \Gamma_{\text{NOE}}))$
/// via exact first-order Gaussian error propagation.
#[inline]
pub fn gamma_noe_with_sigma(
    r1: f64,
    sigma_r1: f64,
    hetnoe: f64,
    sigma_hetnoe: f64,
    phys: &RelaxPhysicsConfig,
) -> (f64, f64, f64) {
    let r_gn = phys.gamma_n / phys.gamma_h;
    let gamma = r1 * (hetnoe - 1.0) * r_gn;
    let dg_dr1 = (hetnoe - 1.0) * r_gn;
    let dg_dnoe = r1 * r_gn;
    let var_g = (dg_dr1 * sigma_r1).powi(2) + (dg_dnoe * sigma_hetnoe).powi(2);
    let cov_r1_g = dg_dr1 * sigma_r1 * sigma_r1;
    (gamma, var_g.sqrt(), cov_r1_g)
}

/// Evaluate the relaxation sensitivity kernel $R_i(\tau)$ under the chosen [`TumblingMode`].
#[inline]
pub fn sensitivity_value(
    kind: RelaxObservableKind,
    field_mhz: f64,
    tau_sec: f64,
    tumbling: TumblingMode,
    phys: &RelaxPhysicsConfig,
) -> f64 {
    match tumbling {
        TumblingMode::PureCorrelationSpectrum => single_tau_rate(kind, field_mhz, tau_sec, phys),
        TumblingMode::IsotropicInternal { tau_m_sec } => {
            let r_rigid = single_tau_rate(kind, field_mhz, tau_m_sec, phys);
            let t_eff = effective_tau_sec(tau_sec, tau_m_sec);
            let r_eff = single_tau_rate(kind, field_mhz, t_eff, phys);
            r_rigid - r_eff
        }
        TumblingMode::EffectiveTimescale { tau_m_sec } => {
            let t_eff = effective_tau_sec(tau_sec, tau_m_sec);
            single_tau_rate(kind, field_mhz, t_eff, phys)
        }
    }
}

/// Evaluated relaxation sensitivity matrix on a correlation-time grid.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SensitivityMatrix {
    /// Raw sensitivity matrix $R_{i,j}$ ($M \times N_z$).
    pub raw: Vec<Vec<f64>>,
    /// Uncertainty-normalized sensitivity matrix $S_{i,j} = R_{i,j} / \sigma_i$ ($M \times N_z$).
    pub normalized: Vec<Vec<f64>>,
    /// Rigid-limit baselines $R_i^{(0)}(\tau_m)$ for [`TumblingMode::IsotropicInternal`] (`0.0` otherwise).
    pub rigid_baselines: Vec<f64>,
}

/// Build the raw and uncertainty-normalized $^{15}\text{N}$ relaxation sensitivity matrices.
pub fn build_sensitivity_matrix(
    experiments: &[RelaxExperiment],
    tau_grid_sec: &[f64],
    tumbling: TumblingMode,
    phys: &RelaxPhysicsConfig,
) -> Result<SensitivityMatrix, String> {
    validate_physics(phys)?;
    validate_tumbling(tumbling)?;
    if experiments.is_empty() {
        return Err("experiment list must not be empty".to_string());
    }
    if tau_grid_sec.is_empty() {
        return Err("correlation-time grid must not be empty".to_string());
    }
    for (idx, exp) in experiments.iter().enumerate() {
        if !exp.field_mhz.is_finite() || exp.field_mhz <= 0.0 {
            return Err(format!(
                "experiment {} ({}) has invalid field_mhz = {} (must be finite and > 0)",
                idx, exp.label, exp.field_mhz
            ));
        }
        if !exp.sigma.is_finite() || exp.sigma <= 0.0 {
            return Err(format!(
                "experiment {} ({}) has invalid sigma = {} (must be finite and > 0)",
                idx, exp.label, exp.sigma
            ));
        }
    }
    for (j, &tau) in tau_grid_sec.iter().enumerate() {
        if !tau.is_finite() || tau <= 0.0 {
            return Err(format!(
                "tau_grid_sec[{}] = {} must be finite and > 0",
                j, tau
            ));
        }
    }

    let m = experiments.len();
    let nz = tau_grid_sec.len();
    let mut raw = vec![vec![0.0_f64; nz]; m];
    let mut normalized = vec![vec![0.0_f64; nz]; m];
    let mut rigid_baselines = vec![0.0_f64; m];

    for (i, exp) in experiments.iter().enumerate() {
        if let TumblingMode::IsotropicInternal { tau_m_sec } = tumbling {
            rigid_baselines[i] = single_tau_rate(exp.kind, exp.field_mhz, tau_m_sec, phys);
        }
        let inv_sig = 1.0 / exp.sigma;
        for (j, &tau) in tau_grid_sec.iter().enumerate() {
            let val = sensitivity_value(exp.kind, exp.field_mhz, tau, tumbling, phys);
            raw[i][j] = val;
            normalized[i][j] = val * inv_sig;
        }
    }

    Ok(SensitivityMatrix {
        raw,
        normalized,
        rigid_baselines,
    })
}

/// Converted multi-field relaxation experiments, observed linear rates, and experimental covariance matrix.
pub type ExperimentObservationBundle = (Vec<RelaxExperiment>, Vec<f64>, Vec<Vec<f64>>);

/// Convert multi-field [`ResidueRelaxRates`] into `(experiments, observed_rates, cov_exp)` including
/// exact analytical off-diagonal $\mathrm{Cov}(R_1, \Gamma_{\text{NOE}})$ when `hetNOE` is present.
pub fn experiments_and_observations_from_rates(
    rates: &[ResidueRelaxRates],
    phys: &RelaxPhysicsConfig,
) -> Result<ExperimentObservationBundle, String> {
    validate_physics(phys)?;
    if rates.is_empty() {
        return Err("multi-field relaxation rates slice must not be empty".to_string());
    }

    let mut experiments = Vec::new();
    let mut obs = Vec::new();
    let mut cov_pairs: Vec<(usize, usize, f64)> = Vec::new();

    for (idx, r) in rates.iter().enumerate() {
        if !r.field_mhz.is_finite() || r.field_mhz <= 0.0 {
            return Err(format!(
                "field entry {} has invalid field_mhz = {} (must be finite and > 0)",
                idx, r.field_mhz
            ));
        }
        if !r.r1.is_finite() || r.r1 <= 0.0 {
            return Err(format!(
                "field entry {} ({} MHz) has invalid r1 = {}",
                idx, r.field_mhz, r.r1
            ));
        }
        if !r.r2.is_finite() || r.r2 <= 0.0 {
            return Err(format!(
                "field entry {} ({} MHz) has invalid r2 = {}",
                idx, r.field_mhz, r.r2
            ));
        }
        if !r.sigma_r1.is_finite() || r.sigma_r1 <= 0.0 {
            return Err(format!(
                "field entry {} ({} MHz) has invalid sigma_r1 = {}",
                idx, r.field_mhz, r.sigma_r1
            ));
        }
        if !r.sigma_r2.is_finite() || r.sigma_r2 <= 0.0 {
            return Err(format!(
                "field entry {} ({} MHz) has invalid sigma_r2 = {}",
                idx, r.field_mhz, r.sigma_r2
            ));
        }

        let r1_idx = experiments.len();
        experiments.push(RelaxExperiment::new(
            RelaxObservableKind::R1,
            r.field_mhz,
            r.sigma_r1,
        ));
        obs.push(r.r1);

        experiments.push(RelaxExperiment::new(
            RelaxObservableKind::R2,
            r.field_mhz,
            r.sigma_r2,
        ));
        obs.push(r.r2);

        if let Some(noe) = r.hetnoe {
            if !noe.is_finite() {
                return Err(format!(
                    "field entry {} ({} MHz) has non-finite hetnoe = {}",
                    idx, r.field_mhz, noe
                ));
            }
            let sigma_noe = r.sigma_hetnoe.ok_or_else(|| {
                format!(
                    "field entry {} ({} MHz) has hetnoe without sigma_hetnoe",
                    idx, r.field_mhz
                )
            })?;
            if !sigma_noe.is_finite() || sigma_noe <= 0.0 {
                return Err(format!(
                    "field entry {} ({} MHz) has invalid sigma_hetnoe = {}",
                    idx, r.field_mhz, sigma_noe
                ));
            }
            let (gamma, sigma_gamma, cov_r1_g) =
                gamma_noe_with_sigma(r.r1, r.sigma_r1, noe, sigma_noe, phys);
            if !sigma_gamma.is_finite() || sigma_gamma <= 0.0 {
                return Err(format!(
                    "field entry {} ({} MHz) yielded non-positive sigma_gamma_noe = {}",
                    idx, r.field_mhz, sigma_gamma
                ));
            }
            let g_idx = experiments.len();
            experiments.push(RelaxExperiment::new(
                RelaxObservableKind::GammaNoe,
                r.field_mhz,
                sigma_gamma,
            ));
            obs.push(gamma);
            cov_pairs.push((r1_idx, g_idx, cov_r1_g));
        }
    }

    let m = experiments.len();
    let mut cov = vec![vec![0.0_f64; m]; m];
    for (i, exp) in experiments.iter().enumerate() {
        cov[i][i] = exp.sigma * exp.sigma;
    }
    for (i, j, c_ij) in cov_pairs {
        cov[i][j] = c_ij;
        cov[j][i] = c_ij;
    }

    Ok((experiments, obs, cov))
}

// ============================================================================
// T22.3 — Linear Optimal-Window Model-Free Timescale Dynamics Detectors
// (Smith, Ernst & Meier 2018; Smith, Zumpfe & Meier 2019)
// ============================================================================

/// Configuration for linear optimal-window [`build_dynamics_detectors`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DetectorConfig {
    pub physics: RelaxPhysicsConfig,
    pub tumbling: TumblingMode,
    /// Lower bound of correlation-time grid $\tau_{\min}$ in seconds (default `1e-13`).
    pub tau_min_sec: f64,
    /// Upper bound of correlation-time grid $\tau_{\max}$ in seconds (default `1e-7`).
    pub tau_max_sec: f64,
    /// Number of log-spaced grid points $N_z$ (default `200`, minimum `16`).
    pub grid_points: usize,
    /// Explicit number of detectors $K$ (`None` = automatic SVD rank selection by `rel_sv_threshold`).
    pub num_detectors: Option<usize>,
    /// Automatic rank selection threshold $\sigma_k / \sigma_1 \ge \text{rel\_sv\_threshold}$ (default `5e-3`).
    pub rel_sv_threshold: f64,
    /// Maximum number of detectors when `num_detectors` is `None` (default `8`).
    pub max_detectors: usize,
    /// Quadratic localization penalty $\beta$ on $(z_j - z_{j_n})^2$ in $\log_{10}\tau$ decades (default `1.0`).
    pub localization_beta: f64,
    /// Cross-peak overlap suppression penalty $\gamma_{\text{sep}}$ (default `2.0`).
    pub overlap_penalty: f64,
    /// Dimensionless Tikhonov variance regularization $\lambda$ on $(\sigma_1/\sigma_k)^2 T_{n,k}^2$ (default `1e-4`).
    pub tikhonov_lambda: f64,
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
            physics: RelaxPhysicsConfig::default(),
            tumbling: TumblingMode::PureCorrelationSpectrum,
            tau_min_sec: 1.0e-13,
            tau_max_sec: 1.0e-7,
            grid_points: 200,
            num_detectors: None,
            rel_sv_threshold: 5.0e-3,
            max_detectors: 8,
            localization_beta: 1.0,
            overlap_penalty: 2.0,
            tikhonov_lambda: 1.0e-4,
        }
    }
}

/// One optimized linear dynamics detector window $\rho_n(\tau) = \sum_i r_{n,i} R_i(\tau)$.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DetectorWindow {
    /// 0-based detector index sorted in ascending order of `peak_tau_sec`.
    pub index: usize,
    /// Detector sensitivity profile $\rho_n(\tau_j)$ evaluated on `tau_grid_sec`.
    pub sensitivities: Vec<f64>,
    /// Linear experiment combination weights $r_{n,i}$ ($\rho_n(\tau) = \sum_i r_{n,i} R_i(\tau)$).
    pub experiment_weights: Vec<f64>,
    /// Transformation coefficients $T_{n,k}$ in the $K$-dimensional right-singular-vector basis.
    pub svd_coeffs: Vec<f64>,
    /// Correlation time $\tau_{\text{peak}}$ (seconds) where $\rho_n(\tau)$ attains its maximum `1.0`.
    pub peak_tau_sec: f64,
    /// $\log_{10}(\tau_{\text{peak}} / 1\text{ s})$.
    pub peak_log10_tau: f64,
    /// Sensitivity-weighted centroid $\bar{z}_n = \frac{\sum_j z_j \max(0, \rho_n(z_j))}{\sum_j \max(0, \rho_n(z_j))}$.
    pub center_log10_tau: f64,
    /// Full width at half maximum ($\rho_n(z) \ge 0.5$) in $\log_{10}\tau$ decades.
    pub fwhm_decades: f64,
    /// Minimum value $\min_j \rho_n(\tau_j)$ across the grid (non-negativity audit).
    pub min_value: f64,
    /// Maximum value $\max_j \rho_n(\tau_j)$ across the grid (unit-peak audit, `1.0`).
    pub max_value: f64,
    /// Trapezoidal integral $\int \rho_n(\tau)\,d\ln\tau$ across the grid.
    pub area_ln_tau: f64,
}

/// Complete linear optimal-window dynamics detector bank constructed from an experiment set.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DetectorDesign {
    pub experiments: Vec<RelaxExperiment>,
    pub tau_grid_sec: Vec<f64>,
    pub log10_tau_grid: Vec<f64>,
    pub tumbling: TumblingMode,
    pub physics: RelaxPhysicsConfig,
    /// All singular values $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r$ of the normalized sensitivity matrix $S$.
    pub singular_values: Vec<f64>,
    /// Number of retained detectors $K$.
    pub num_detectors: usize,
    /// Rigid-limit baselines $R_i^{(0)}(\tau_m)$ (nonzero in [`TumblingMode::IsotropicInternal`]).
    pub rigid_baselines: Vec<f64>,
    /// $K$ optimized detector windows ordered by ascending `peak_tau_sec`.
    pub windows: Vec<DetectorWindow>,
    /// $M \times K$ back-calculation matrix mapping $\hat{\boldsymbol{\rho}}^{(\theta)}$ to $\hat{\mathbf{y}}$.
    pub backcalc_matrix: Vec<Vec<f64>>,
    /// Condition number $\kappa(T)$ of the $K \times K$ SVD-to-detector basis transform.
    pub transform_condition_number: f64,
    /// Maximum pairwise cosine overlap $\max_{m \ne n} \frac{\langle \rho_n, \rho_m \rangle}{\|\rho_n\|_2 \|\rho_m\|_2}$.
    pub max_pairwise_overlap: f64,
}

/// Linear detector estimation result for one residue's multi-field relaxation measurements.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DetectorFitResult {
    /// Linear detector response estimates $\hat{\rho}_n^{(\theta)} = \sum_i r_{n,i} y_i$.
    pub responses: Vec<f64>,
    /// Exact analytical $1\sigma$ standard errors $\sqrt{\mathbf{r}_n^T \Sigma_{\text{exp}} \mathbf{r}_n}$.
    pub sigmas: Vec<f64>,
    /// Full $K \times K$ analytical detector covariance matrix $\mathbf{R}_{\text{det}} \Sigma_{\text{exp}} \mathbf{R}_{\text{det}}^T$.
    pub covariance: Vec<Vec<f64>>,
    /// Back-calculated relaxation rates $\hat{R}_i^{\text{calc}}$ ($\text{s}^{-1}$).
    pub backcalc_rates: Vec<f64>,
    /// Raw residuals $R_i^{\text{obs}} - \hat{R}_i^{\text{calc}}$ ($\text{s}^{-1}$).
    pub residuals: Vec<f64>,
    /// Uncertainty-normalized residuals $(R_i^{\text{obs}} - \hat{R}_i^{\text{calc}}) / \sigma_i$.
    pub weighted_residuals: Vec<f64>,
    /// Goodness-of-fit $\chi^2 = \sum_i ((R_i^{\text{obs}} - \hat{R}_i^{\text{calc}}) / \sigma_i)^2$
    /// (or $(\mathbf{R}^{\text{obs}} - \hat{\mathbf{R}})^T \Sigma_{\text{exp}}^{-1} (\mathbf{R}^{\text{obs}} - \hat{\mathbf{R}})$ when full covariance is supplied).
    pub chi2: f64,
    /// Residual degrees of freedom $\nu = M - K$.
    pub dof: usize,
    /// Reduced $\chi^2_\nu = \chi^2 / \nu$ (`0.0` when $\nu = 0$).
    pub reduced_chi2: f64,
}

impl DetectorDesign {
    /// Evaluate detector `detector_idx` continuously at any correlation time $\tau > 0$ using
    /// exact linear combination $\rho_n(\tau) = \sum_{i=1}^M r_{n,i} R_i(\tau)$.
    pub fn evaluate_window(&self, detector_idx: usize, tau_sec: f64) -> Result<f64, String> {
        let win = self.windows.get(detector_idx).ok_or_else(|| {
            format!(
                "detector_idx {} out of bounds (num_detectors = {})",
                detector_idx, self.num_detectors
            )
        })?;
        if !tau_sec.is_finite() || tau_sec < 0.0 {
            return Err(format!("tau_sec = {} must be finite and >= 0", tau_sec));
        }
        let mut acc = 0.0_f64;
        for (exp, &w) in self.experiments.iter().zip(win.experiment_weights.iter()) {
            acc += w * sensitivity_value(
                exp.kind,
                exp.field_mhz,
                tau_sec,
                self.tumbling,
                &self.physics,
            );
        }
        Ok(acc)
    }

    /// Evaluate exact detector integrals $\rho_n^{(\theta)} = \sum_c w_c \rho_n(\tau_c)$ for a
    /// discrete correlation-time distribution `components = [(weight, tau_sec)]`.
    pub fn evaluate_distribution(&self, components: &[(f64, f64)]) -> Result<Vec<f64>, String> {
        for (idx, &(w, tau)) in components.iter().enumerate() {
            if !w.is_finite() {
                return Err(format!("component {} has non-finite weight {}", idx, w));
            }
            if !tau.is_finite() || tau < 0.0 {
                return Err(format!(
                    "component {} has invalid tau_sec = {} (must be finite and >= 0)",
                    idx, tau
                ));
            }
        }
        let mut out = vec![0.0_f64; self.num_detectors];
        for n in 0..self.num_detectors {
            let mut acc = 0.0_f64;
            for &(w, tau) in components {
                acc += w * self.evaluate_window(n, tau)?;
            }
            out[n] = acc;
        }
        Ok(out)
    }

    /// Compute linear detector estimates $\hat{\rho}_n^{(\theta)}$, exact analytical propagated
    /// variance $\mathbf{r}_n^T \Sigma_{\text{exp}} \mathbf{r}_n$, back-calculated relaxation rates,
    /// residuals, and reduced $\chi^2_\nu$.
    ///
    /// Pass `cov_exp: None` for independent diagonal uncertainties $\Sigma_{\text{exp}} = \mathrm{diag}(\sigma_i^2)$,
    /// or `Some(&cov)` for a full symmetric positive-definite experimental covariance matrix.
    pub fn estimate(
        &self,
        observed_rates: &[f64],
        cov_exp: Option<&[Vec<f64>]>,
    ) -> Result<DetectorFitResult, String> {
        let m = self.experiments.len();
        let k = self.num_detectors;
        if observed_rates.len() != m {
            return Err(format!(
                "observed_rates length ({}) does not match design experiments ({})",
                observed_rates.len(),
                m
            ));
        }
        for (i, &val) in observed_rates.iter().enumerate() {
            if !val.is_finite() {
                return Err(format!("observed_rates[{}] = {} must be finite", i, val));
            }
        }

        // Validate / build experimental covariance matrix Sigma_exp
        let cov_owned: Vec<Vec<f64>>;
        let cov_ref: &[Vec<f64>] = if let Some(c) = cov_exp {
            validate_covariance_matrix(c, m)?;
            c
        } else {
            let mut d = vec![vec![0.0_f64; m]; m];
            for (i, exp) in self.experiments.iter().enumerate() {
                d[i][i] = exp.sigma * exp.sigma;
            }
            cov_owned = d;
            &cov_owned
        };

        // Map raw rates to linear detector inputs y_i
        let mut y = vec![0.0_f64; m];
        for i in 0..m {
            y[i] = match self.tumbling {
                TumblingMode::IsotropicInternal { .. } => {
                    self.rigid_baselines[i] - observed_rates[i]
                }
                TumblingMode::PureCorrelationSpectrum | TumblingMode::EffectiveTimescale { .. } => {
                    observed_rates[i]
                }
            };
        }

        // Exact linear estimator rho_hat = R_det * y
        let mut responses = vec![0.0_f64; k];
        for (n, win) in self.windows.iter().enumerate() {
            let mut acc = 0.0_f64;
            for i in 0..m {
                acc += win.experiment_weights[i] * y[i];
            }
            responses[n] = acc;
        }

        // Exact analytical detector covariance Cov(rho_hat) = R_det * Sigma_exp * R_det^T
        let mut covariance = vec![vec![0.0_f64; k]; k];
        let mut sigmas = vec![0.0_f64; k];
        for a in 0..k {
            let ra = &self.windows[a].experiment_weights;
            let mut sigma_ra = vec![0.0_f64; m];
            for i in 0..m {
                let mut s = 0.0_f64;
                for j in 0..m {
                    s += cov_ref[i][j] * ra[j];
                }
                sigma_ra[i] = s;
            }
            for b in a..k {
                let rb = &self.windows[b].experiment_weights;
                let mut v = 0.0_f64;
                for i in 0..m {
                    v += rb[i] * sigma_ra[i];
                }
                covariance[a][b] = v;
                covariance[b][a] = v;
            }
            sigmas[a] = covariance[a][a].max(0.0).sqrt();
        }

        // Back-calculated y_hat = backcalc_matrix * responses
        let mut backcalc_rates = vec![0.0_f64; m];
        let mut residuals = vec![0.0_f64; m];
        let mut weighted_residuals = vec![0.0_f64; m];
        for i in 0..m {
            let mut y_hat = 0.0_f64;
            for n in 0..k {
                y_hat += self.backcalc_matrix[i][n] * responses[n];
            }
            let r_calc = match self.tumbling {
                TumblingMode::IsotropicInternal { .. } => self.rigid_baselines[i] - y_hat,
                TumblingMode::PureCorrelationSpectrum | TumblingMode::EffectiveTimescale { .. } => {
                    y_hat
                }
            };
            backcalc_rates[i] = r_calc;
            residuals[i] = observed_rates[i] - r_calc;
            weighted_residuals[i] = residuals[i] / self.experiments[i].sigma;
        }

        // Compute chi2 (exact Mahalanobis form if full covariance is provided and invertible, else diagonal)
        let chi2 = if cov_exp.is_some() {
            if let Some(chol) = cholesky_decompose(cov_ref) {
                let z = cholesky_solve_lower(&chol, &residuals);
                z.iter().map(|&v| v * v).sum()
            } else {
                weighted_residuals.iter().map(|&z| z * z).sum()
            }
        } else {
            weighted_residuals.iter().map(|&z| z * z).sum()
        };

        let dof = m.saturating_sub(k);
        let reduced_chi2 = if dof > 0 { chi2 / (dof as f64) } else { 0.0 };

        Ok(DetectorFitResult {
            responses,
            sigmas,
            covariance,
            backcalc_rates,
            residuals,
            weighted_residuals,
            chi2,
            dof,
            reduced_chi2,
        })
    }
}

/// Construct a bank of $K$ maximally localized, non-negative, unit-peak linear dynamics detectors
/// $\rho_n(\tau)$ from an arbitrary set of multi-field $^{15}\text{N}$ relaxation experiments
/// (Smith, Ernst & Meier 2018; Smith, Zumpfe & Meier 2019).
pub fn build_dynamics_detectors(
    experiments: &[RelaxExperiment],
    cfg: &DetectorConfig,
) -> Result<DetectorDesign, String> {
    validate_physics(&cfg.physics)?;
    validate_tumbling(cfg.tumbling)?;
    if experiments.is_empty() {
        return Err("cannot build dynamics detectors from an empty experiment list".to_string());
    }
    if cfg.grid_points < 16 {
        return Err(format!("grid_points = {} must be >= 16", cfg.grid_points));
    }
    if !cfg.rel_sv_threshold.is_finite()
        || cfg.rel_sv_threshold <= 0.0
        || cfg.rel_sv_threshold >= 1.0
    {
        return Err(format!(
            "rel_sv_threshold = {} must be finite and in (0, 1)",
            cfg.rel_sv_threshold
        ));
    }
    if !cfg.localization_beta.is_finite() || cfg.localization_beta < 0.0 {
        return Err(format!(
            "localization_beta = {} must be finite and >= 0",
            cfg.localization_beta
        ));
    }
    if !cfg.overlap_penalty.is_finite() || cfg.overlap_penalty < 0.0 {
        return Err(format!(
            "overlap_penalty = {} must be finite and >= 0",
            cfg.overlap_penalty
        ));
    }
    if !cfg.tikhonov_lambda.is_finite() || cfg.tikhonov_lambda < 0.0 {
        return Err(format!(
            "tikhonov_lambda = {} must be finite and >= 0",
            cfg.tikhonov_lambda
        ));
    }

    let tau_grid_sec = log_spaced_grid(cfg.tau_min_sec, cfg.tau_max_sec, cfg.grid_points)?;
    let log10_tau_grid: Vec<f64> = tau_grid_sec.iter().map(|&t| t.log10()).collect();
    let sens = build_sensitivity_matrix(experiments, &tau_grid_sec, cfg.tumbling, &cfg.physics)?;

    let m = experiments.len();
    let nz = tau_grid_sec.len();
    let svd = thin_svd(&sens.normalized)?;
    let max_rank = svd.s.len();
    if max_rank == 0 || svd.s[0] <= 1.0e-15 {
        return Err("sensitivity matrix is numerically zero".to_string());
    }

    let s0 = svd.s[0];
    let num_detectors = if let Some(req_k) = cfg.num_detectors {
        if req_k == 0 || req_k > max_rank {
            return Err(format!(
                "requested num_detectors = {} is out of valid range 1..={}",
                req_k, max_rank
            ));
        }
        if svd.s[req_k - 1] <= 1.0e-12 * s0 {
            return Err(format!(
                "requested num_detectors = {} exceeds numerical rank of sensitivity matrix (sigma_{} / sigma_1 = {:.3e})",
                req_k,
                req_k,
                svd.s[req_k - 1] / s0
            ));
        }
        req_k
    } else {
        let cap = cfg.max_detectors.max(1).min(max_rank);
        let mut auto_k = 0usize;
        for &sv in svd.s.iter().take(cap) {
            if sv >= cfg.rel_sv_threshold * s0 {
                auto_k += 1;
            }
        }
        auto_k.max(1)
    };

    // Extract retained K singular vectors:
    // U_K is M x K, S_K is length K, V_K^T is K x N_z (column j is v_j in R^K).
    let k = num_detectors;
    let mut v_cols: Vec<Vec<f64>> = vec![vec![0.0_f64; k]; nz];
    for j in 0..nz {
        for r in 0..k {
            v_cols[j][r] = svd.v[j][r];
        }
    }

    // Select K maximally separated, well-conditioned target peak centers across the sensitive grid
    let mut centers = select_max_volume_centers(&v_cols, k);
    centers.sort_unstable();

    // Optimize each detector window via exact dual active-set non-negative localization QP
    let mut t_rows = optimize_detector_rows(
        &v_cols,
        &log10_tau_grid,
        &svd.s[..k],
        &centers,
        cfg.localization_beta,
        cfg.overlap_penalty,
        cfg.tikhonov_lambda,
    );

    // One refinement pass if peaks shifted significantly while remaining strictly distinct
    let actual_peaks: Vec<usize> = t_rows.iter().map(|t| argmax_profile(t, &v_cols)).collect();
    let mut unique_peaks = actual_peaks.clone();
    unique_peaks.sort_unstable();
    unique_peaks.dedup();
    if unique_peaks.len() == k && unique_peaks != centers {
        let refined = optimize_detector_rows(
            &v_cols,
            &log10_tau_grid,
            &svd.s[..k],
            &unique_peaks,
            cfg.localization_beta,
            cfg.overlap_penalty,
            cfg.tikhonov_lambda,
        );
        // Accept refinement if T remains well-conditioned and non-negativity is preserved
        if let Some(cond_ref) = matrix_condition_number(&refined) {
            let cond_orig = matrix_condition_number(&t_rows).unwrap_or(f64::INFINITY);
            let min_ref = min_profile_across_rows(&refined, &v_cols);
            let min_orig = min_profile_across_rows(&t_rows, &v_cols);
            if cond_ref.is_finite() && cond_ref <= cond_orig * 2.5 && min_ref >= min_orig - 1.0e-6 {
                t_rows = refined;
            }
        }
    }

    // Build DetectorWindow entries and sort by ascending peak_tau_sec (tie-break by centroid)
    let mut raw_windows: Vec<(f64, f64, Vec<f64>)> = Vec::with_capacity(k);
    for t in t_rows {
        let prof = eval_profile(&t, &v_cols);
        let peak_idx = argmax_slice(&prof);
        let peak_tau = tau_grid_sec[peak_idx];
        let centroid = profile_centroid(&prof, &log10_tau_grid);
        raw_windows.push((peak_tau, centroid, t));
    }
    raw_windows.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.total_cmp(&b.1)));

    let ordered_t: Vec<Vec<f64>> = raw_windows.into_iter().map(|(_, _, t)| t).collect();

    // Compute exact experiment weights r_n = (t_n^T * Sigma_K^-1 * U_K^T) * D_sigma^-1
    let mut windows = Vec::with_capacity(k);
    for (n, t_row) in ordered_t.iter().enumerate() {
        let mut r_weights = vec![0.0_f64; m];
        for i in 0..m {
            let mut acc = 0.0_f64;
            for c in 0..k {
                acc += (t_row[c] / svd.s[c]) * svd.u[i][c];
            }
            r_weights[i] = acc / experiments[i].sigma;
        }

        // Evaluate sensitivities directly from exact experiment weights (identical to t_row^T V_K^T)
        let mut sensitivities = vec![0.0_f64; nz];
        for j in 0..nz {
            let mut acc = 0.0_f64;
            for i in 0..m {
                acc += r_weights[i] * sens.raw[i][j];
            }
            sensitivities[j] = acc;
        }

        let peak_idx = argmax_slice(&sensitivities);
        let min_value = sensitivities.iter().copied().fold(f64::INFINITY, f64::min);
        let max_value = sensitivities[peak_idx];
        let center_log10_tau = profile_centroid(&sensitivities, &log10_tau_grid);
        let fwhm_decades = profile_fwhm_decades(&sensitivities, &log10_tau_grid, peak_idx);
        let area_ln_tau = trapezoidal_area_ln_tau(&sensitivities, &tau_grid_sec);

        windows.push(DetectorWindow {
            index: n,
            sensitivities,
            experiment_weights: r_weights,
            svd_coeffs: t_row.clone(),
            peak_tau_sec: tau_grid_sec[peak_idx],
            peak_log10_tau: log10_tau_grid[peak_idx],
            center_log10_tau,
            fwhm_decades,
            min_value,
            max_value,
            area_ln_tau,
        });
    }

    // Back-calculation matrix M_back = D_sigma * U_K * Sigma_K * T^-1 (size M x K)
    let t_inv = invert_square_matrix(&ordered_t).ok_or_else(|| {
        "optimized detector transformation matrix T is numerically singular".to_string()
    })?;
    let transform_condition_number = matrix_condition_number(&ordered_t).unwrap_or(f64::INFINITY);

    let mut backcalc_matrix = vec![vec![0.0_f64; k]; m];
    for i in 0..m {
        let sig_i = experiments[i].sigma;
        for col in 0..k {
            let mut acc = 0.0_f64;
            for c in 0..k {
                acc += svd.u[i][c] * svd.s[c] * t_inv[c][col];
            }
            backcalc_matrix[i][col] = sig_i * acc;
        }
    }

    // Maximum pairwise cosine overlap across detector windows
    let mut max_pairwise_overlap = 0.0_f64;
    for a in 0..k {
        let na = l2_norm(&windows[a].sensitivities);
        for b in (a + 1)..k {
            let nb = l2_norm(&windows[b].sensitivities);
            if na > 1.0e-15 && nb > 1.0e-15 {
                let dot: f64 = windows[a]
                    .sensitivities
                    .iter()
                    .zip(windows[b].sensitivities.iter())
                    .map(|(&x, &y)| x * y)
                    .sum();
                let ov = (dot / (na * nb)).abs();
                if ov > max_pairwise_overlap {
                    max_pairwise_overlap = ov;
                }
            }
        }
    }

    Ok(DetectorDesign {
        experiments: experiments.to_vec(),
        tau_grid_sec,
        log10_tau_grid,
        tumbling: cfg.tumbling,
        physics: cfg.physics,
        singular_values: svd.s,
        num_detectors: k,
        rigid_baselines: sens.rigid_baselines,
        windows,
        backcalc_matrix,
        transform_condition_number,
        max_pairwise_overlap,
    })
}

/// Select $K$ grid indices in the sensitive region maximizing the volume $|\det([\hat{\mathbf{v}}_{j_1}, \dots, \hat{\mathbf{v}}_{j_K}])|$
/// via greedy pivoted orthogonalization followed by pairwise swap coordinate ascent.
fn select_max_volume_centers(v_cols: &[Vec<f64>], k: usize) -> Vec<usize> {
    let nz = v_cols.len();
    let norms: Vec<f64> = v_cols.iter().map(|v| l2_norm(v)).collect();
    let max_norm = norms.iter().copied().fold(0.0_f64, f64::max);
    let thresh = 0.10 * max_norm;

    let mut eligible: Vec<usize> = (0..nz).filter(|&j| norms[j] >= thresh).collect();
    if eligible.len() < k {
        // Fallback: sort all grid points by descending norm and take top k * 2
        let mut all: Vec<usize> = (0..nz).collect();
        all.sort_by(|&a, &b| norms[b].total_cmp(&norms[a]));
        eligible = all.into_iter().take((k * 2).min(nz)).collect();
        eligible.sort_unstable();
    }

    // Normalized direction vectors hat_v_j = v_j / ||v_j||
    let hat_v: Vec<Vec<f64>> = v_cols
        .iter()
        .zip(norms.iter())
        .map(|(v, &nrm)| {
            if nrm > 1.0e-30 {
                v.iter().map(|&x| x / nrm).collect()
            } else {
                vec![0.0_f64; k]
            }
        })
        .collect();

    if k == 1 {
        let best = eligible
            .iter()
            .copied()
            .max_by(|&a, &b| norms[a].total_cmp(&norms[b]))
            .unwrap_or(nz / 2);
        return vec![best];
    }

    // Step 1: Greedy pivoted Gram-Schmidt to pick K linearly independent directions
    let mut selected: Vec<usize> = Vec::with_capacity(k);
    let mut basis: Vec<Vec<f64>> = Vec::with_capacity(k);

    // First center: strongest norm point (typically one of the primary sensitivity peaks)
    let first = eligible
        .iter()
        .copied()
        .max_by(|&a, &b| norms[a].total_cmp(&norms[b]))
        .unwrap_or(eligible[0]);
    selected.push(first);
    basis.push(hat_v[first].clone());

    for _ in 1..k {
        let mut best_j = eligible[0];
        let mut best_rem = -1.0_f64;
        for &j in &eligible {
            if selected.contains(&j) {
                continue;
            }
            let mut r = hat_v[j].clone();
            for q in &basis {
                let d = dot_slice(&r, q);
                for c in 0..k {
                    r[c] -= d * q[c];
                }
            }
            let rem_norm = l2_norm(&r);
            // Tiny tie-breaker favoring stronger raw sensitivity
            let score = rem_norm * (1.0 + 1.0e-3 * (norms[j] / max_norm.max(1.0e-30)));
            if score > best_rem {
                best_rem = score;
                best_j = j;
            }
        }
        let mut r = hat_v[best_j].clone();
        for q in &basis {
            let d = dot_slice(&r, q);
            for c in 0..k {
                r[c] -= d * q[c];
            }
        }
        let nrm = l2_norm(&r);
        if nrm > 1.0e-15 {
            for x in &mut r {
                *x /= nrm;
            }
            basis.push(r);
        }
        selected.push(best_j);
    }

    // Step 2: Coordinate-ascent volume maximization (|det([hat_v_{j_1}, ..., hat_v_{j_K}])|)
    let abs_det = |idxs: &[usize]| -> f64 {
        let mut mat = vec![vec![0.0_f64; k]; k];
        for col in 0..k {
            let v = &hat_v[idxs[col]];
            for row in 0..k {
                mat[row][col] = v[row];
            }
        }
        matrix_abs_determinant(&mat)
    };

    let mut cur_vol = abs_det(&selected);
    for _pass in 0..6 {
        let mut improved = false;
        for slot in 0..k {
            let orig = selected[slot];
            let mut best_cand = orig;
            let mut best_v = cur_vol;
            for &j in &eligible {
                if j == orig || selected.contains(&j) {
                    continue;
                }
                selected[slot] = j;
                let vol = abs_det(&selected);
                if vol > best_v * (1.0 + 1.0e-9) {
                    best_v = vol;
                    best_cand = j;
                }
            }
            selected[slot] = best_cand;
            if best_cand != orig {
                cur_vol = best_v;
                improved = true;
            }
        }
        if !improved {
            break;
        }
    }

    selected
}

/// Optimize the $K$ detector transformation rows $\mathbf{t}_n \in \mathbb{R}^K$ for centers `centers`.
fn optimize_detector_rows(
    v_cols: &[Vec<f64>],
    log10_tau_grid: &[f64],
    singular_values: &[f64],
    centers: &[usize],
    beta: f64,
    overlap_penalty: f64,
    tikhonov_lambda: f64,
) -> Vec<Vec<f64>> {
    let k = singular_values.len();
    let nz = v_cols.len();
    let s0 = singular_values[0].max(1.0e-30);

    // Precompute normalized direction vectors at each center for cross-peak suppression
    let center_dirs: Vec<Vec<f64>> = centers
        .iter()
        .map(|&idx| {
            let v = &v_cols[idx];
            let nrm = l2_norm(v).max(1.0e-30);
            v.iter().map(|&x| x / nrm).collect()
        })
        .collect();

    let mut rows = Vec::with_capacity(k);
    for (n, &j_star) in centers.iter().enumerate() {
        let z_star = log10_tau_grid[j_star];

        // Build SPD Hessian H_n = sum_j (1 + beta * (z_j - z_*)^2) v_j v_j^T
        //                       + gamma_sep * sum_{m != n} hat_v_{j_m} hat_v_{j_m}^T
        //                       + lambda * diag((s_0 / s_c)^2)
        let mut h_mat = vec![vec![0.0_f64; k]; k];
        for j in 0..nz {
            let dz = log10_tau_grid[j] - z_star;
            let w = 1.0 + beta * dz * dz;
            let vj = &v_cols[j];
            for r in 0..k {
                let wr = w * vj[r];
                for c in r..k {
                    h_mat[r][c] += wr * vj[c];
                }
            }
        }
        if overlap_penalty > 0.0 {
            for (m_idx, dir) in center_dirs.iter().enumerate() {
                if m_idx == n {
                    continue;
                }
                for r in 0..k {
                    let wr = overlap_penalty * dir[r];
                    for c in r..k {
                        h_mat[r][c] += wr * dir[c];
                    }
                }
            }
        }
        for c in 0..k {
            let ratio = s0 / singular_values[c].max(1.0e-30);
            h_mat[c][c] += tikhonov_lambda * ratio * ratio + 1.0e-12;
        }
        for r in 0..k {
            for c in (r + 1)..k {
                h_mat[c][r] = h_mat[r][c];
            }
        }

        // Solve exact dual active-set non-negative localization QP:
        // min 0.5 t^T H_n t  s.t.  v_{j*}^T t = 1,  v_j^T t >= 0 for all j
        let mut t = solve_nonneg_localized_qp(&h_mat, v_cols, j_star)
            .unwrap_or_else(|| solve_soft_nonneg_localized(&h_mat, v_cols, j_star));

        // Normalize strictly to unit peak max_j (t^T v_j) = 1.0
        let prof = eval_profile(&t, v_cols);
        let max_v = prof.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if max_v.is_finite() && max_v > 1.0e-30 {
            for x in &mut t {
                *x /= max_v;
            }
        }
        rows.push(t);
    }

    rows
}

/// Exact dual active-set solver for the strictly convex localization QP:
/// $\min_{\mathbf{t} \in \mathbb{R}^K} \frac{1}{2}\mathbf{t}^T H \mathbf{t}$ subject to
/// $\mathbf{v}_{j^*}^T \mathbf{t} = 1$ and $\mathbf{v}_j^T \mathbf{t} \ge 0\;\forall j$.
///
/// Returns `None` only if the strict non-negativity cone is infeasible.
fn solve_nonneg_localized_qp(
    h_mat: &[Vec<f64>],
    v_cols: &[Vec<f64>],
    j_star: usize,
) -> Option<Vec<f64>> {
    let k = h_mat.len();
    let nz = v_cols.len();
    let chol = cholesky_decompose(h_mat)?;

    // Transform u = L^T t, a_j = L^-1 v_j so objective is 0.5 ||u||^2.
    // Normalize every inequality constraint a_j^T u >= 0 by ||a_j|| so all dual constraints
    // have unit norm (eliminating 1e10 Gram-matrix scaling mismatch at tiny tau).
    let mut a_cols: Vec<Vec<f64>> = Vec::with_capacity(nz);
    for vj in v_cols {
        a_cols.push(cholesky_solve_lower(&chol, vj));
    }

    let a_star = &a_cols[j_star];
    let a_star_norm = l2_norm(a_star);
    if a_star_norm <= 1.0e-15 {
        return None;
    }
    let e_hat: Vec<f64> = a_star.iter().map(|&x| x / a_star_norm).collect();
    let inv_norm = 1.0 / a_star_norm;

    // Decompose hat_a_j^T u = c_j + b_j^T u_perp >= 0 where u = inv_norm * e_hat + u_perp
    let mut c_vec = vec![0.0_f64; nz];
    let mut b_cols: Vec<Vec<f64>> = Vec::with_capacity(nz);
    for j in 0..nz {
        let an = l2_norm(&a_cols[j]);
        if an <= 1.0e-30 {
            c_vec[j] = 0.0;
            b_cols.push(vec![0.0_f64; k]);
            continue;
        }
        let inv_an = 1.0 / an;
        let hat_a: Vec<f64> = a_cols[j].iter().map(|&x| x * inv_an).collect();
        let proj = dot_slice(&hat_a, &e_hat);
        c_vec[j] = proj * inv_norm;
        let mut bj = vec![0.0_f64; k];
        for r in 0..k {
            bj[r] = hat_a[r] - proj * e_hat[r];
        }
        b_cols.push(bj);
    }

    // Dual: min_{mu >= 0} 0.5 || sum_j mu_j b_j ||^2 + c^T mu
    // Gradient w.r.t. mu_j is g_j = b_j^T u_perp + c_j = (a_j^T u) / ||a_j||.
    let mut mu = vec![0.0_f64; nz];
    let mut active: Vec<usize> = Vec::new();
    let mut u_perp = vec![0.0_f64; k];
    let b_norm_sq: Vec<f64> = b_cols.iter().map(|bj| dot_slice(bj, bj)).collect();

    // Stage 1: Exact Lawson-Hanson active-set
    let max_outer = (6 * k).max(32);
    for _outer in 0..max_outer {
        let mut min_g = f64::INFINITY;
        let mut min_j = 0usize;
        for j in 0..nz {
            let g = dot_slice(&b_cols[j], &u_perp) + c_vec[j];
            if g < min_g {
                min_g = g;
                min_j = j;
            }
        }
        if min_g >= -1.0e-11 {
            let mut u = vec![0.0_f64; k];
            for r in 0..k {
                u[r] = inv_norm * e_hat[r] + u_perp[r];
            }
            return Some(cholesky_solve_upper_transpose(&chol, &u));
        }

        if active.contains(&min_j) || active.len() >= k {
            break;
        }
        active.push(min_j);

        let mut inner_ok = false;
        for _inner in 0..(4 * k + 16) {
            let s_len = active.len();
            if s_len == 0 {
                inner_ok = true;
                break;
            }
            let mut gram = vec![vec![0.0_f64; s_len]; s_len];
            let mut rhs = vec![0.0_f64; s_len];
            for r in 0..s_len {
                let jr = active[r];
                rhs[r] = -c_vec[jr];
                for c in r..s_len {
                    let jc = active[c];
                    let v = dot_slice(&b_cols[jr], &b_cols[jc]);
                    gram[r][c] = v;
                    gram[c][r] = v;
                }
            }
            let Some(s_sol) = solve_linear_system(&gram, &rhs) else {
                break;
            };
            if s_sol.iter().any(|&v| !v.is_finite() || v.abs() > 1.0e14) {
                break;
            }

            if s_sol.iter().all(|&v| v > 1.0e-15) {
                for (slot, &j_act) in active.iter().enumerate() {
                    mu[j_act] = s_sol[slot];
                }
                inner_ok = true;
                break;
            }

            let mut alpha = 1.0_f64;
            for (slot, &j_act) in active.iter().enumerate() {
                if s_sol[slot] <= 1.0e-15 {
                    let denom = mu[j_act] - s_sol[slot];
                    if denom > 1.0e-30 {
                        let cand = mu[j_act] / denom;
                        if cand < alpha {
                            alpha = cand;
                        }
                    }
                }
            }
            alpha = alpha.clamp(0.0, 1.0);
            for (slot, &j_act) in active.iter().enumerate() {
                mu[j_act] += alpha * (s_sol[slot] - mu[j_act]);
                if mu[j_act] <= 1.0e-15 {
                    mu[j_act] = 0.0;
                }
            }
            active.retain(|&j_act| mu[j_act] > 1.0e-15);
        }

        // Always synchronize u_perp = sum_j mu_j b_j even if inner loop exited early
        u_perp.fill(0.0);
        for j in 0..nz {
            if mu[j] > 0.0 {
                for r in 0..k {
                    u_perp[r] += mu[j] * b_cols[j][r];
                }
            }
        }
        if !inner_ok {
            break;
        }
    }

    // Stage 2: Hildreth dual coordinate descent (unconditionally handles collinear/adjacent active grid points)
    let u_bound = 1.0e8 * inv_norm.max(1.0);
    for _sweep in 0..6000 {
        let mut min_g = f64::INFINITY;
        let mut min_j = 0usize;
        for j in 0..nz {
            let g = dot_slice(&b_cols[j], &u_perp) + c_vec[j];
            if g < min_g {
                min_g = g;
                min_j = j;
            }
        }
        if min_g >= -1.0e-11 {
            let mut u = vec![0.0_f64; k];
            for r in 0..k {
                u[r] = inv_norm * e_hat[r] + u_perp[r];
            }
            return Some(cholesky_solve_upper_transpose(&chol, &u));
        }

        if b_norm_sq[min_j] > 1.0e-30 {
            let new_mu = (mu[min_j] - min_g / b_norm_sq[min_j]).max(0.0);
            let delta = new_mu - mu[min_j];
            if delta != 0.0 {
                mu[min_j] = new_mu;
                for r in 0..k {
                    u_perp[r] += delta * b_cols[min_j][r];
                }
            }
        }

        let mut max_delta = 0.0_f64;
        for j in 0..nz {
            if b_norm_sq[j] <= 1.0e-30 {
                continue;
            }
            let g = dot_slice(&b_cols[j], &u_perp) + c_vec[j];
            if mu[j] > 0.0 || g < -1.0e-12 {
                let new_mu = (mu[j] - g / b_norm_sq[j]).max(0.0);
                let delta = new_mu - mu[j];
                if delta.abs() > 1.0e-18 {
                    mu[j] = new_mu;
                    for r in 0..k {
                        u_perp[r] += delta * b_cols[j][r];
                    }
                    if delta.abs() > max_delta {
                        max_delta = delta.abs();
                    }
                }
            }
        }

        let u_nrm = l2_norm(&u_perp);
        if !u_nrm.is_finite() || u_nrm > u_bound {
            return None;
        }
        if max_delta <= 1.0e-15 {
            break;
        }
    }

    let min_g = (0..nz)
        .map(|j| dot_slice(&b_cols[j], &u_perp) + c_vec[j])
        .fold(f64::INFINITY, f64::min);
    if min_g >= -1.0e-8 {
        let mut u = vec![0.0_f64; k];
        for r in 0..k {
            u[r] = inv_norm * e_hat[r] + u_perp[r];
        }
        Some(cholesky_solve_upper_transpose(&chol, &u))
    } else {
        None
    }
}

/// Fallback soft-nonnegativity iterative reweighted localization solver if exact non-negativity
/// cone is infeasible (e.g., degenerate single-rate sign-changing `IsotropicInternal` design).
fn solve_soft_nonneg_localized(
    h_base: &[Vec<f64>],
    v_cols: &[Vec<f64>],
    j_star: usize,
) -> Vec<f64> {
    let k = h_base.len();
    let nz = v_cols.len();
    let v_star = &v_cols[j_star];
    let mut neg_weight = vec![0.0_f64; nz];
    let mut t = vec![0.0_f64; k];

    for _iter in 0..12 {
        let mut h = h_base.to_vec();
        for j in 0..nz {
            let w = neg_weight[j];
            if w > 0.0 {
                let vj = &v_cols[j];
                for r in 0..k {
                    for c in 0..k {
                        h[r][c] += w * vj[r] * vj[c];
                    }
                }
            }
        }
        if let Some(h_inv_v) = solve_linear_system(&h, v_star) {
            let denom = dot_slice(v_star, &h_inv_v);
            if denom.abs() > 1.0e-30 {
                for r in 0..k {
                    t[r] = h_inv_v[r] / denom;
                }
            }
        }
        let mut any_neg = false;
        for j in 0..nz {
            let val = dot_slice(&t, &v_cols[j]);
            if val < -1.0e-6 {
                neg_weight[j] = (neg_weight[j] * 2.0 + 50.0).min(1.0e6);
                any_neg = true;
            }
        }
        if !any_neg {
            break;
        }
    }
    t
}

fn eval_profile(t: &[f64], v_cols: &[Vec<f64>]) -> Vec<f64> {
    v_cols.iter().map(|vj| dot_slice(t, vj)).collect()
}

fn argmax_profile(t: &[f64], v_cols: &[Vec<f64>]) -> usize {
    let prof = eval_profile(t, v_cols);
    argmax_slice(&prof)
}

fn argmax_slice(vals: &[f64]) -> usize {
    let mut best_i = 0usize;
    let mut best_v = f64::NEG_INFINITY;
    for (i, &v) in vals.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    best_i
}

fn min_profile_across_rows(t_rows: &[Vec<f64>], v_cols: &[Vec<f64>]) -> f64 {
    let mut m = f64::INFINITY;
    for t in t_rows {
        for vj in v_cols {
            let v = dot_slice(t, vj);
            if v < m {
                m = v;
            }
        }
    }
    m
}

fn profile_centroid(prof: &[f64], z_grid: &[f64]) -> f64 {
    let mut num = 0.0_f64;
    let mut den = 0.0_f64;
    for (&p, &z) in prof.iter().zip(z_grid.iter()) {
        let w = p.max(0.0);
        num += w * z;
        den += w;
    }
    if den > 1.0e-30 {
        num / den
    } else {
        z_grid.first().copied().unwrap_or(0.0)
    }
}

fn profile_fwhm_decades(prof: &[f64], z_grid: &[f64], peak_idx: usize) -> f64 {
    let half = 0.5 * prof[peak_idx];
    let mut left_z = z_grid[0];
    for j in (0..peak_idx).rev() {
        if prof[j] <= half {
            let denom = prof[j + 1] - prof[j];
            let frac = if denom.abs() > 1.0e-30 {
                ((half - prof[j]) / denom).clamp(0.0, 1.0)
            } else {
                0.0
            };
            left_z = z_grid[j] + frac * (z_grid[j + 1] - z_grid[j]);
            break;
        }
    }
    let mut right_z = z_grid[z_grid.len() - 1];
    for j in (peak_idx + 1)..prof.len() {
        if prof[j] <= half {
            let denom = prof[j - 1] - prof[j];
            let frac = if denom.abs() > 1.0e-30 {
                ((half - prof[j]) / denom).clamp(0.0, 1.0)
            } else {
                0.0
            };
            right_z = z_grid[j] - frac * (z_grid[j] - z_grid[j - 1]);
            break;
        }
    }
    (right_z - left_z).max(0.0)
}

fn trapezoidal_area_ln_tau(prof: &[f64], tau_grid: &[f64]) -> f64 {
    if prof.len() < 2 {
        return 0.0;
    }
    let mut acc = 0.0_f64;
    for j in 0..(prof.len() - 1) {
        let d_ln = (tau_grid[j + 1] / tau_grid[j]).ln();
        acc += 0.5 * (prof[j] + prof[j + 1]) * d_ln;
    }
    acc
}

// ============================================================================
// T22.4a — LeMaster (1995) Fixed-Timescale Analytical Decomposition
// (LeMaster, J. Biomol. NMR 1995, 6, 366-374)
// ============================================================================

/// Result of LeMaster (1995) single-field analytical 3-timescale spectral density decomposition
/// $(J(0), J(\omega_N), J(0.87\omega_H)) \to (A_{\text{slow}}, A_{\text{mid}}, A_{\text{fast}})$
/// at fixed correlation times $(\tau_m, \tau_N = |\omega_N|^{-1}, \tau_H = (0.87\omega_H)^{-1})$.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LeMasterResult {
    pub field_mhz: f64,
    /// Fixed slow correlation time $\tau_m$ (seconds).
    pub tau_m_sec: f64,
    /// Fixed mid correlation time $\tau_N = |\omega_N|^{-1}$ (seconds).
    pub tau_n_sec: f64,
    /// Fixed fast correlation time $\tau_H = (0.87\omega_H)^{-1}$ (seconds).
    pub tau_h_sec: f64,
    /// Amplitude $A_{\text{slow}}$ associated with $\tau_m$.
    pub a_slow: f64,
    pub sigma_a_slow: f64,
    /// Amplitude $A_{\text{mid}}$ associated with $\tau_N = |\omega_N|^{-1}$.
    pub a_mid: f64,
    pub sigma_a_mid: f64,
    /// Amplitude $A_{\text{fast}}$ associated with $\tau_H = (0.87\omega_H)^{-1}$.
    pub a_fast: f64,
    pub sigma_a_fast: f64,
    /// Implicit ultrafast ($\tau \ll \tau_H$) remainder $A_{\text{ultrafast}} = 1 - (A_{\text{slow}} + A_{\text{mid}} + A_{\text{fast}})$.
    pub a_ultrafast: f64,
    pub sigma_a_ultrafast: f64,
    /// Exact analytical $3 \times 3$ covariance matrix of $(A_{\text{slow}}, A_{\text{mid}}, A_{\text{fast}})$.
    pub covariance: [[f64; 3]; 3],
    /// Whether all four weights $(A_{\text{slow}}, A_{\text{mid}}, A_{\text{fast}}, A_{\text{ultrafast}})$ are non-negative within $10^{-9}$.
    pub is_physical: bool,
}

/// Forward evaluation of the 3-timescale LeMaster (1995) spectral density model at
/// $\omega \in \{0, |\omega_N|, 0.87\omega_H\}$.
pub fn lemaster_predict_spectral_density(
    a_slow: f64,
    a_mid: f64,
    a_fast: f64,
    field_mhz: f64,
    tau_m_sec: f64,
    phys: &RelaxPhysicsConfig,
) -> Result<(f64, f64, f64), String> {
    validate_physics(phys)?;
    if !field_mhz.is_finite() || field_mhz <= 0.0 {
        return Err(format!(
            "invalid field_mhz = {} (must be finite and > 0)",
            field_mhz
        ));
    }
    if !tau_m_sec.is_finite() || tau_m_sec <= 0.0 {
        return Err(format!(
            "invalid tau_m_sec = {} (must be finite and > 0)",
            tau_m_sec
        ));
    }
    if !a_slow.is_finite() || !a_mid.is_finite() || !a_fast.is_finite() {
        return Err("LeMaster amplitudes must be finite".to_string());
    }
    let (m_mat, _, _) = lemaster_design_matrix(field_mhz, tau_m_sec, phys)?;
    let a = [a_slow, a_mid, a_fast];
    let j0 = m_mat[0][0] * a[0] + m_mat[0][1] * a[1] + m_mat[0][2] * a[2];
    let jn = m_mat[1][0] * a[0] + m_mat[1][1] * a[1] + m_mat[1][2] * a[2];
    let jh = m_mat[2][0] * a[0] + m_mat[2][1] * a[1] + m_mat[2][2] * a[2];
    Ok((j0, jn, jh))
}

/// Exact analytical LeMaster (1995) 3-timescale decomposition from single-field reduced spectral
/// densities $(J(0), J(\omega_N), J(0.87\omega_H))$ with independent diagonal uncertainties.
pub fn fit_lemaster_from_spectral_density(
    rsdm: &ReducedSpectralDensity,
    tau_m_sec: f64,
    phys: &RelaxPhysicsConfig,
) -> Result<LeMasterResult, String> {
    validate_physics(phys)?;
    if !rsdm.field_mhz.is_finite() || rsdm.field_mhz <= 0.0 {
        return Err(format!(
            "invalid field_mhz = {} (must be finite and > 0)",
            rsdm.field_mhz
        ));
    }
    if !tau_m_sec.is_finite() || tau_m_sec <= 0.0 {
        return Err(format!(
            "invalid tau_m_sec = {} (must be finite and > 0)",
            tau_m_sec
        ));
    }
    for (name, val) in [
        ("j_0", rsdm.j_0),
        ("j_omega_n", rsdm.j_omega_n),
        ("j_087_omega_h", rsdm.j_087_omega_h),
    ] {
        if !val.is_finite() {
            return Err(format!(
                "non-finite reduced spectral density {} = {}",
                name, val
            ));
        }
    }
    for (name, sig) in [
        ("sigma_j_0", rsdm.sigma_j_0),
        ("sigma_j_omega_n", rsdm.sigma_j_omega_n),
        ("sigma_j_087_omega_h", rsdm.sigma_j_087_omega_h),
    ] {
        if !sig.is_finite() || sig <= 0.0 {
            return Err(format!(
                "invalid spectral density uncertainty {} = {} (must be finite and > 0)",
                name, sig
            ));
        }
    }

    let j_vec = [rsdm.j_0, rsdm.j_omega_n, rsdm.j_087_omega_h];
    let cov_j = [
        [rsdm.sigma_j_0 * rsdm.sigma_j_0, 0.0, 0.0],
        [0.0, rsdm.sigma_j_omega_n * rsdm.sigma_j_omega_n, 0.0],
        [
            0.0,
            0.0,
            rsdm.sigma_j_087_omega_h * rsdm.sigma_j_087_omega_h,
        ],
    ];
    fit_lemaster_with_cov(j_vec, cov_j, rsdm.field_mhz, tau_m_sec, phys)
}

/// Exact analytical LeMaster (1995) 3-timescale decomposition directly from single-field
/// $(R_1, R_2, \mathrm{hetNOE})$ measurements, propagating the full correlated $3 \times 3$
/// reduced-spectral-density Jacobian covariance $\Sigma_J = J_{\text{RSDM}} \Sigma_R J_{\text{RSDM}}^T$.
pub fn fit_lemaster_from_rates(
    rates: &ResidueRelaxRates,
    tau_m_sec: f64,
    phys: &RelaxPhysicsConfig,
) -> Result<LeMasterResult, String> {
    validate_physics(phys)?;
    if !rates.field_mhz.is_finite() || rates.field_mhz <= 0.0 {
        return Err(format!(
            "invalid field_mhz = {} (must be finite and > 0)",
            rates.field_mhz
        ));
    }
    if !tau_m_sec.is_finite() || tau_m_sec <= 0.0 {
        return Err(format!(
            "invalid tau_m_sec = {} (must be finite and > 0)",
            tau_m_sec
        ));
    }
    if !rates.r1.is_finite() || rates.r1 <= 0.0 || !rates.r2.is_finite() || rates.r2 <= 0.0 {
        return Err(format!(
            "invalid relaxation rates r1 = {}, r2 = {} (must be finite and > 0)",
            rates.r1, rates.r2
        ));
    }
    if !rates.sigma_r1.is_finite()
        || rates.sigma_r1 <= 0.0
        || !rates.sigma_r2.is_finite()
        || rates.sigma_r2 <= 0.0
    {
        return Err(format!(
            "invalid rate uncertainties sigma_r1 = {}, sigma_r2 = {} (must be finite and > 0)",
            rates.sigma_r1, rates.sigma_r2
        ));
    }
    let noe = rates
        .hetnoe
        .ok_or_else(|| "LeMaster fit from rates requires hetNOE".to_string())?;
    let sigma_noe = rates
        .sigma_hetnoe
        .ok_or_else(|| "LeMaster fit from rates requires sigma_hetnoe".to_string())?;
    if !noe.is_finite() || !sigma_noe.is_finite() || sigma_noe <= 0.0 {
        return Err(format!(
            "invalid hetNOE = {} / sigma_hetnoe = {}",
            noe, sigma_noe
        ));
    }

    let rsdm = map_reduced_spectral_density(rates, phys)?;

    // Build exact 3x3 Jacobian d(J0, JwN, J0.87wH) / d(R1, R2, NOE)
    let d = phys.dipolar_d_rad_s();
    let d2 = d * d;
    let c = phys.csa_c_rad_s(rates.field_mhz);
    let c2 = c * c;
    let r_gn = phys.gamma_n / phys.gamma_h;
    let f_sum = 1.0 + r_gn;
    let f_dif = 1.0 - r_gn;
    let k_sigma = 6.0 / (f_sum * f_sum) - 1.0 / (f_dif * f_dif);
    let k_r1_hf = 6.0 / (f_sum * f_sum) + 1.0 / (f_dif * f_dif);
    let k_r2_hf = 6.0 + 6.0 / (f_sum * f_sum) + 1.0 / (f_dif * f_dif);

    let alpha_r1 = 0.25 * d2 * (5.0 * k_r1_hf / k_sigma);
    let denom_jn = 0.75 * d2 + c2;
    let alpha_r2 = 0.125 * d2 * (5.0 * k_r2_hf / k_sigma);
    let beta_jn = 0.375 * d2 + 0.5 * c2;
    let denom_j0 = 0.5 * d2 + (2.0 / 3.0) * c2;

    let dj87_dr1 = (4.0 * (noe - 1.0) * r_gn) / (5.0 * d2);
    let dj87_dnoe = (4.0 * rates.r1 * r_gn) / (5.0 * d2);

    let djn_dr1 = (1.0 - alpha_r1 * dj87_dr1) / denom_jn;
    let djn_dnoe = (-alpha_r1 * dj87_dnoe) / denom_jn;

    let dj0_dr2 = 1.0 / denom_j0;
    let dj0_dr1 = (-beta_jn * djn_dr1 - alpha_r2 * dj87_dr1) / denom_j0;
    let dj0_dnoe = (-beta_jn * djn_dnoe - alpha_r2 * dj87_dnoe) / denom_j0;

    // Rows: [J0, JwN, J0.87wH], Cols: [R1, R2, NOE]
    let jac = [
        [dj0_dr1, dj0_dr2, dj0_dnoe],
        [djn_dr1, 0.0, djn_dnoe],
        [dj87_dr1, 0.0, dj87_dnoe],
    ];
    let vars = [
        rates.sigma_r1 * rates.sigma_r1,
        rates.sigma_r2 * rates.sigma_r2,
        sigma_noe * sigma_noe,
    ];
    let mut cov_j = [[0.0_f64; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            let mut s = 0.0_f64;
            for p in 0..3 {
                s += jac[a][p] * vars[p] * jac[b][p];
            }
            cov_j[a][b] = s;
        }
    }

    fit_lemaster_with_cov(
        [rsdm.j_0, rsdm.j_omega_n, rsdm.j_087_omega_h],
        cov_j,
        rates.field_mhz,
        tau_m_sec,
        phys,
    )
}

fn lemaster_design_matrix(
    field_mhz: f64,
    tau_m_sec: f64,
    phys: &RelaxPhysicsConfig,
) -> Result<([[f64; 3]; 3], f64, f64), String> {
    let (wh, wn_signed) = phys.signed_omegas_rad_s(field_mhz);
    let wn = wn_signed.abs();
    let w_hf = 0.87 * wh;
    if wn <= 0.0 || w_hf <= 0.0 {
        return Err("non-positive Larmor frequency in LeMaster decomposition".to_string());
    }
    let tau_n = 1.0 / wn;
    let tau_h = 1.0 / w_hf;

    if ((tau_m_sec - tau_n).abs() / tau_n) < 1.0e-3 || ((tau_m_sec - tau_h).abs() / tau_h) < 1.0e-3
    {
        return Err(format!(
            "degenerate LeMaster timescales: tau_m ({:.3e} s) coincides with tau_N ({:.3e} s) or tau_H ({:.3e} s)",
            tau_m_sec, tau_n, tau_h
        ));
    }

    let taus = [tau_m_sec, tau_n, tau_h];
    let omegas = [0.0, wn, w_hf];
    let mut m_mat = [[0.0_f64; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            m_mat[r][c] = lorentzian_j(omegas[r], taus[c]);
        }
    }
    Ok((m_mat, tau_n, tau_h))
}

fn fit_lemaster_with_cov(
    j_vec: [f64; 3],
    cov_j: [[f64; 3]; 3],
    field_mhz: f64,
    tau_m_sec: f64,
    phys: &RelaxPhysicsConfig,
) -> Result<LeMasterResult, String> {
    let (m_mat, tau_n_sec, tau_h_sec) = lemaster_design_matrix(field_mhz, tau_m_sec, phys)?;
    let m_inv = invert_3x3(m_mat).ok_or_else(|| {
        "LeMaster 3x3 spectral density matrix is numerically singular".to_string()
    })?;

    let mut a = [0.0_f64; 3];
    for r in 0..3 {
        a[r] = m_inv[r][0] * j_vec[0] + m_inv[r][1] * j_vec[1] + m_inv[r][2] * j_vec[2];
    }

    // Exact covariance Sigma_A = M^-1 * Sigma_J * (M^-1)^T
    let mut tmp = [[0.0_f64; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            tmp[r][c] =
                m_inv[r][0] * cov_j[0][c] + m_inv[r][1] * cov_j[1][c] + m_inv[r][2] * cov_j[2][c];
        }
    }
    let mut cov_a = [[0.0_f64; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            cov_a[r][c] =
                tmp[r][0] * m_inv[c][0] + tmp[r][1] * m_inv[c][1] + tmp[r][2] * m_inv[c][2];
        }
    }

    let a_ultrafast = 1.0 - (a[0] + a[1] + a[2]);
    let mut var_ultra = 0.0_f64;
    for r in 0..3 {
        for c in 0..3 {
            var_ultra += cov_a[r][c];
        }
    }

    let is_physical =
        a[0] >= -1.0e-9 && a[1] >= -1.0e-9 && a[2] >= -1.0e-9 && a_ultrafast >= -1.0e-9;

    Ok(LeMasterResult {
        field_mhz,
        tau_m_sec,
        tau_n_sec,
        tau_h_sec,
        a_slow: a[0],
        sigma_a_slow: cov_a[0][0].max(0.0).sqrt(),
        a_mid: a[1],
        sigma_a_mid: cov_a[1][1].max(0.0).sqrt(),
        a_fast: a[2],
        sigma_a_fast: cov_a[2][2].max(0.0).sqrt(),
        a_ultrafast,
        sigma_a_ultrafast: var_ultra.max(0.0).sqrt(),
        covariance: cov_a,
        is_physical,
    })
}

// ============================================================================
// T22.4b — Multi-Field Stacked IMPACT (Khan, Charlier, Ferrage et al. 2015)
// ============================================================================

/// Single spectral density observation $J_{\text{obs}}(\omega_k) \pm \sigma_k$ ($\text{s rad}^{-1}$).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SpectralDensityObs {
    /// Angular frequency $\omega_k \ge 0$ ($\text{rad s}^{-1}$).
    pub omega_rad_s: f64,
    /// Observed spectral density $J_{\text{obs}}(\omega_k)$ ($\text{s rad}^{-1}$).
    pub value: f64,
    /// $1\sigma$ uncertainty $\sigma_k > 0$ ($\text{s rad}^{-1}$).
    pub sigma: f64,
    /// Associated static field in MHz (metadata).
    pub field_mhz: f64,
    /// Descriptive label (e.g. `"J(0)_600MHz"`).
    pub label: String,
}

impl SpectralDensityObs {
    pub fn new(
        omega_rad_s: f64,
        value: f64,
        sigma: f64,
        field_mhz: f64,
        label: impl Into<String>,
    ) -> Self {
        Self {
            omega_rad_s,
            value,
            sigma,
            field_mhz,
            label: label.into(),
        }
    }
}

/// Configuration for multi-field stacked `IMPACT` simplex decomposition and $\mathrm{AIC}_c$ selection.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ImpactConfig {
    /// Minimum correlation time $\tau_1$ (seconds) for automatic grid generation in AICc selection (default `1e-11`).
    pub tau_min_sec: f64,
    /// Maximum correlation time $\tau_M$ (seconds) for automatic grid generation in AICc selection (default `2e-8`).
    pub tau_max_sec: f64,
    /// Number of parametric Monte Carlo bootstrap replicates for 95% CIs (`0` = skip MC, default `200`).
    pub mc_replicates: usize,
    /// Deterministic PRNG seed for parametric Monte Carlo bootstrap.
    pub seed: u64,
    /// KKT optimality tolerance for the convex simplex QP solver (default `1e-10`).
    pub kkt_tol: f64,
    /// Maximum optimizer iterations per solve (default `2000`).
    pub max_iter: usize,
}

impl Default for ImpactConfig {
    fn default() -> Self {
        Self {
            tau_min_sec: 1.0e-11,
            tau_max_sec: 2.0e-8,
            mc_replicates: 200,
            seed: 20262204,
            kkt_tol: 1.0e-10,
            max_iter: 2000,
        }
    }
}

/// Result of a single constrained convex simplex weighted least-squares solve.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SimplexWlsSolution {
    /// Optimal probability-simplex amplitudes $A_m \ge 0$, $\sum_m A_m = 1$.
    pub amplitudes: Vec<f64>,
    /// Weighted sum of squared residuals $\chi^2 = \sum_k ((y_{\text{calc},k} - y_k)/\sigma_k)^2$.
    pub chi2: f64,
    /// Exact KKT violation $\max_{m : A_m > 10^{-14}} g_m - \min_l g_l$ on the simplex.
    pub kkt_violation: f64,
    /// Exact Frank–Wolfe duality gap $\mathbf{A}^T \mathbf{g} - \min_l g_l \ge 0$.
    pub fw_gap: f64,
    /// Number of optimization iterations performed.
    pub iterations: usize,
}

/// Converged `IMPACT` correlation-time amplitude distribution and diagnostics.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ImpactFitResult {
    /// Fixed log-spaced correlation times $\{\tau_m\}_{m=1}^M$ (seconds).
    pub tau_grid_sec: Vec<f64>,
    /// Fitted simplex amplitudes $A_m \ge 0$, $\sum_{m=1}^M A_m = 1$.
    pub amplitudes: Vec<f64>,
    /// Parametric Monte Carlo bootstrap $1\sigma$ standard deviations per component (`0.0` if `mc_replicates == 0`).
    pub sigma_amplitudes_mc: Vec<f64>,
    /// Parametric Monte Carlo `68.27%` (`[p16, p84]`) confidence intervals per component.
    pub amplitudes_ci68: Vec<[f64; 2]>,
    /// Parametric Monte Carlo `95.0%` (`[p2.5, p97.5]`) confidence intervals per component.
    pub amplitudes_ci95: Vec<[f64; 2]>,
    /// Back-calculated observations (spectral densities or relaxation rates).
    pub predicted: Vec<f64>,
    /// Raw residuals `obs - predicted`.
    pub residuals: Vec<f64>,
    /// Weighted sum of squared residuals $\chi^2$.
    pub chi2: f64,
    /// Residual degrees of freedom $\nu = N_{\text{obs}} - (M - 1)$.
    pub dof: usize,
    /// Reduced $\chi^2_\nu = \chi^2 / \nu$ (`0.0` when $\nu = 0$).
    pub reduced_chi2: f64,
    /// Akaike Information Criterion $\mathrm{AIC} = \chi^2 + 2 k$ with $k = M - 1$.
    pub aic: f64,
    /// Small-sample corrected $\mathrm{AIC}_c = \chi^2 + 2 k + \frac{2 k (k + 1)}{N_{\text{obs}} - k - 1}$ ($+\infty$ if $N_{\text{obs}} \le k + 1$).
    pub aicc: f64,
    /// Bayesian Information Criterion $\mathrm{BIC} = \chi^2 + k \ln N_{\text{obs}}$.
    pub bic: f64,
    /// Exact KKT optimality violation $\max_{m : A_m > 10^{-14}} g_m - \min_l g_l$.
    pub kkt_violation: f64,
    /// Exact Frank–Wolfe duality gap $\mathbf{A}^T \mathbf{g} - \min_l g_l$.
    pub fw_gap: f64,
    /// Sum of amplitudes $\sum_{m=1}^M A_m$ (enforced to `1.0` within `< 1e-14`).
    pub sum_amplitudes: f64,
}

/// Result of $\mathrm{AIC}_c$ model-size selection over candidate correlation-time grid sizes (e.g. $M \in \{3, 4, 5, 6\}$).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ImpactModelSelectionResult {
    /// Selected optimal number of correlation times $M^*$ minimizing $\mathrm{AIC}_c$.
    pub selected_num_timescales: usize,
    /// Best-fit `IMPACT` decomposition (with parametric Monte Carlo CIs computed for $M^*$).
    pub best_fit: ImpactFitResult,
    /// Evaluated candidate fits for all feasible sizes in `candidate_sizes`.
    pub candidates: Vec<ImpactFitResult>,
}

/// Forward evaluation of the `IMPACT` spectral density $J(\omega; \mathbf{A}) = \frac{2}{5}\sum_{m=1}^M A_m \frac{\tau_m}{1+(\omega\tau_m)^2}$.
#[inline]
pub fn impact_predict_spectral_density(
    omega_rad_s: f64,
    tau_grid_sec: &[f64],
    amplitudes: &[f64],
) -> f64 {
    let mut acc = 0.0_f64;
    for (&tau, &a) in tau_grid_sec.iter().zip(amplitudes.iter()) {
        acc += a * lorentzian_j(omega_rad_s, tau);
    }
    acc
}

/// Convert multi-field $^{15}\text{N}$ $(R_1, R_2, \mathrm{hetNOE})$ measurements into stacked
/// spectral density observations $\{J(0), J(\omega_N), J(0.87\omega_H)\}$ across all fields via
/// Farrow et al. (1995) reduced spectral density mapping (Khan et al. 2015).
pub fn stack_multi_field_spectral_densities(
    rates: &[ResidueRelaxRates],
    phys: &RelaxPhysicsConfig,
) -> Result<Vec<SpectralDensityObs>, String> {
    validate_physics(phys)?;
    if rates.is_empty() {
        return Err("cannot stack spectral densities from empty multi-field rates".to_string());
    }
    let mut stacked = Vec::with_capacity(rates.len() * 3);
    for (idx, r) in rates.iter().enumerate() {
        if !r.field_mhz.is_finite() || r.field_mhz <= 0.0 {
            return Err(format!(
                "field entry {} has invalid field_mhz = {} (must be finite and > 0)",
                idx, r.field_mhz
            ));
        }
        if !r.r1.is_finite() || r.r1 <= 0.0 || !r.r2.is_finite() || r.r2 <= 0.0 {
            return Err(format!(
                "field entry {} ({} MHz) has invalid R1/R2",
                idx, r.field_mhz
            ));
        }
        if !r.sigma_r1.is_finite()
            || r.sigma_r1 <= 0.0
            || !r.sigma_r2.is_finite()
            || r.sigma_r2 <= 0.0
        {
            return Err(format!(
                "field entry {} ({} MHz) has invalid sigma_r1/sigma_r2",
                idx, r.field_mhz
            ));
        }
        let noe = r.hetnoe.ok_or_else(|| {
            format!(
                "field entry {} ({} MHz) is missing hetNOE for reduced spectral density mapping",
                idx, r.field_mhz
            )
        })?;
        let sigma_noe = r.sigma_hetnoe.ok_or_else(|| {
            format!(
                "field entry {} ({} MHz) is missing sigma_hetnoe",
                idx, r.field_mhz
            )
        })?;
        if !noe.is_finite() || !sigma_noe.is_finite() || sigma_noe <= 0.0 {
            return Err(format!(
                "field entry {} ({} MHz) has invalid hetNOE/sigma_hetnoe",
                idx, r.field_mhz
            ));
        }

        let rsdm = map_reduced_spectral_density(r, phys)?;
        let (wh, wn_signed) = phys.signed_omegas_rad_s(r.field_mhz);
        let wn = wn_signed.abs();
        let w_hf = 0.87 * wh;

        stacked.push(SpectralDensityObs::new(
            0.0,
            rsdm.j_0,
            rsdm.sigma_j_0,
            r.field_mhz,
            format!("J(0)_{:.0}MHz", r.field_mhz),
        ));
        stacked.push(SpectralDensityObs::new(
            wn,
            rsdm.j_omega_n,
            rsdm.sigma_j_omega_n,
            r.field_mhz,
            format!("J(wN)_{:.0}MHz", r.field_mhz),
        ));
        stacked.push(SpectralDensityObs::new(
            w_hf,
            rsdm.j_087_omega_h,
            rsdm.sigma_j_087_omega_h,
            r.field_mhz,
            format!("J(0.87wH)_{:.0}MHz", r.field_mhz),
        ));
    }
    Ok(stacked)
}

/// Exact constrained convex weighted least-squares solver on the probability simplex $\Delta_M$:
/// $$\min_{\mathbf{A} \ge 0,\; \sum_{m=1}^M A_m = 1} \sum_{k=1}^N \left(\frac{(G\mathbf{A})_k - y_k}{\sigma_k}\right)^2$$
/// using Exact Active-Set Simplex QP + Away-Step Frank–Wolfe (Lacoste-Julien & Jaggi 2015) with
/// closed-form exact quadratic line search and final KKT polishing.
pub fn solve_simplex_wls(
    design_rows: &[Vec<f64>],
    targets: &[f64],
    sigmas: &[f64],
    kkt_tol: f64,
    max_iter: usize,
) -> Result<SimplexWlsSolution, String> {
    let n = design_rows.len();
    if n == 0 {
        return Err("simplex WLS requires at least one observation".to_string());
    }
    if targets.len() != n || sigmas.len() != n {
        return Err("design_rows, targets, and sigmas lengths must match".to_string());
    }
    let m = design_rows[0].len();
    if m == 0 {
        return Err("design matrix must have at least one column".to_string());
    }
    if !kkt_tol.is_finite() || kkt_tol <= 0.0 {
        return Err(format!(
            "invalid kkt_tol = {} (must be finite and > 0)",
            kkt_tol
        ));
    }
    for k in 0..n {
        if design_rows[k].len() != m {
            return Err(format!(
                "design_rows[{}] length ({}) != {}",
                k,
                design_rows[k].len(),
                m
            ));
        }
        if !targets[k].is_finite() {
            return Err(format!("non-finite target[{}] = {}", k, targets[k]));
        }
        if !sigmas[k].is_finite() || sigmas[k] <= 0.0 {
            return Err(format!(
                "invalid sigma[{}] = {} (must be finite and > 0)",
                k, sigmas[k]
            ));
        }
        for (j, &v) in design_rows[k].iter().enumerate() {
            if !v.is_finite() {
                return Err(format!("non-finite design_rows[{}][{}] = {}", k, j, v));
            }
        }
    }

    // Build weighted matrix G_tilde = W^(1/2) G and target y_tilde = W^(1/2) y
    let mut g_tilde = vec![vec![0.0_f64; m]; n];
    let mut y_tilde = vec![0.0_f64; n];
    for k in 0..n {
        let inv_s = 1.0 / sigmas[k];
        y_tilde[k] = targets[k] * inv_s;
        for j in 0..m {
            g_tilde[k][j] = design_rows[k][j] * inv_s;
        }
    }

    if m == 1 {
        let a = vec![1.0_f64];
        let chi2 = eval_wls_obj(&g_tilde, &y_tilde, &a);
        return Ok(SimplexWlsSolution {
            amplitudes: a,
            chi2,
            kkt_violation: 0.0,
            fw_gap: 0.0,
            iterations: 0,
        });
    }

    // Form Q = G_tilde^T G_tilde (M x M) and b = G_tilde^T y_tilde (M)
    let mut q_mat = vec![vec![0.0_f64; m]; m];
    let mut b_vec = vec![0.0_f64; m];
    for k in 0..n {
        let row = &g_tilde[k];
        let yk = y_tilde[k];
        for r in 0..m {
            b_vec[r] += row[r] * yk;
            for c in r..m {
                q_mat[r][c] += row[r] * row[c];
            }
        }
    }
    for r in 0..m {
        for c in (r + 1)..m {
            q_mat[c][r] = q_mat[r][c];
        }
    }

    // Initialize at the best simplex vertex e_{best_v}
    let mut best_v = 0usize;
    let mut best_f = f64::INFINITY;
    for v in 0..m {
        let mut e = vec![0.0_f64; m];
        e[v] = 1.0;
        let f = eval_wls_obj(&g_tilde, &y_tilde, &e);
        if f < best_f {
            best_f = f;
            best_v = v;
        }
    }
    let mut a = vec![0.0_f64; m];
    a[best_v] = 1.0;
    let mut support: Vec<usize> = vec![best_v];
    let mut total_iters = 0usize;

    let grad_of = |x: &[f64]| -> Vec<f64> {
        // Compute gradient directly via 2 * G_tilde^T (G_tilde x - y_tilde) for maximum accuracy
        let mut r_vec = vec![0.0_f64; n];
        for k in 0..n {
            let mut pred = 0.0_f64;
            for j in 0..m {
                pred += g_tilde[k][j] * x[j];
            }
            r_vec[k] = pred - y_tilde[k];
        }
        let mut g = vec![0.0_f64; m];
        for k in 0..n {
            let rk2 = 2.0 * r_vec[k];
            for j in 0..m {
                g[j] += g_tilde[k][j] * rk2;
            }
        }
        g
    };

    // Phase 1: Exact Active-Set Simplex QP
    let active_limit = (12 * m).max(64).min(max_iter.max(1));
    let mut prev_obj = eval_wls_obj(&g_tilde, &y_tilde, &a);
    let mut stall_count = 0usize;

    for _iter in 0..active_limit {
        total_iters += 1;
        let s_len = support.len();
        if s_len == 0 {
            break;
        }

        // Solve equality-constrained KKT system on current face `support`:
        // [ Q_SS   1 ] [ x_S ] = [ b_S ]
        // [ 1^T    0 ] [ mu  ]   [  1  ]
        let dim = s_len + 1;
        let mut kkt_mat = vec![vec![0.0_f64; dim]; dim];
        let mut kkt_rhs = vec![0.0_f64; dim];
        for r in 0..s_len {
            let ir = support[r];
            kkt_rhs[r] = b_vec[ir];
            for c in 0..s_len {
                kkt_mat[r][c] = q_mat[ir][support[c]];
            }
            kkt_mat[r][s_len] = 1.0;
            kkt_mat[s_len][r] = 1.0;
        }
        kkt_rhs[s_len] = 1.0;

        let Some(sol) = solve_linear_system(&kkt_mat, &kkt_rhs) else {
            // Rank-deficient face: hand off cleanly to Away-Step Frank-Wolfe
            break;
        };
        if sol.iter().any(|&v| !v.is_finite()) {
            break;
        }

        let mut x_cand = vec![0.0_f64; m];
        let mut all_nonneg = true;
        for (slot, &idx) in support.iter().enumerate() {
            x_cand[idx] = sol[slot];
            if sol[slot] < -1.0e-14 {
                all_nonneg = false;
            }
        }

        if all_nonneg {
            for v in &mut x_cand {
                if *v < 0.0 {
                    *v = 0.0;
                }
            }
            normalize_simplex(&mut x_cand);
            let f_new = eval_wls_obj(&g_tilde, &y_tilde, &x_cand);
            if f_new <= prev_obj + 1.0e-12 * prev_obj.max(1.0) {
                a = x_cand;
                support.retain(|&idx| a[idx] > 1.0e-15);
                if support.is_empty() {
                    let top = argmax_slice(&a);
                    support.push(top);
                }
                if (prev_obj - f_new).abs() <= 1.0e-15 * prev_obj.max(1.0) {
                    stall_count += 1;
                } else {
                    stall_count = 0;
                }
                prev_obj = f_new;
            } else {
                break;
            }

            // Check exact KKT optimality on the full simplex
            let g = grad_of(&a);
            let (kkt_viol, fw_gap, min_idx) = simplex_kkt_metrics(&a, &g);
            let g_scale = g.iter().map(|v| v.abs()).fold(1.0_f64, f64::max);
            if kkt_viol <= kkt_tol * g_scale && fw_gap <= kkt_tol * g_scale {
                break;
            }
            if stall_count >= 4 {
                break;
            }
            if !support.contains(&min_idx) {
                support.push(min_idx);
            } else {
                break;
            }
        } else {
            // Compute maximum feasible step alpha in [0, 1] keeping a + alpha*(x_cand - a) >= 0
            let mut alpha = 1.0_f64;
            for &idx in &support {
                if x_cand[idx] < 0.0 {
                    let denom = a[idx] - x_cand[idx];
                    if denom > 1.0e-30 {
                        let cand = a[idx] / denom;
                        if cand < alpha {
                            alpha = cand;
                        }
                    }
                }
            }
            alpha = alpha.clamp(0.0, 1.0);
            let mut a_next = vec![0.0_f64; m];
            for j in 0..m {
                a_next[j] = (a[j] + alpha * (x_cand[j] - a[j])).max(0.0);
            }
            normalize_simplex(&mut a_next);
            let f_next = eval_wls_obj(&g_tilde, &y_tilde, &a_next);
            a = a_next;
            prev_obj = f_next;

            // Drop blocking coordinate(s) with zero amplitude
            let before = support.len();
            support.retain(|&idx| a[idx] > 1.0e-15);
            if support.len() == before {
                // Force drop the most-negative candidate coordinate to prevent stalling
                let mut worst_slot = 0usize;
                let mut worst_val = f64::INFINITY;
                for (slot, &idx) in support.iter().enumerate() {
                    if x_cand[idx] < worst_val {
                        worst_val = x_cand[idx];
                        worst_slot = slot;
                    }
                }
                let dropped = support.remove(worst_slot);
                a[dropped] = 0.0;
                normalize_simplex(&mut a);
            }
            if support.is_empty() {
                let top = argmax_slice(&a);
                support.push(top);
            }
        }
    }

    // Phase 2: Away-Step Frank-Wolfe with exact closed-form quadratic line search
    // Guarantees linear convergence to arbitrary KKT precision even for rank-deficient Q!
    let mut g = grad_of(&a);
    let (mut kkt_viol, mut fw_gap, _) = simplex_kkt_metrics(&a, &g);
    let mut g_scale = g.iter().map(|v| v.abs()).fold(1.0_f64, f64::max);

    if kkt_viol > kkt_tol * g_scale || fw_gap > kkt_tol * g_scale {
        let rem_iters = max_iter.saturating_sub(total_iters).max(200);
        for _iter in 0..rem_iters {
            total_iters += 1;
            g = grad_of(&a);
            let (kv, fg, s_idx) = simplex_kkt_metrics(&a, &g);
            kkt_viol = kv;
            fw_gap = fg;
            g_scale = g.iter().map(|v| v.abs()).fold(1.0_f64, f64::max);
            if kkt_viol <= kkt_tol * g_scale && fw_gap <= kkt_tol * g_scale {
                break;
            }

            // Away vertex: largest gradient among active coordinates (A_m > 0)
            let mut v_idx = s_idx;
            let mut max_g_act = f64::NEG_INFINITY;
            for j in 0..m {
                if a[j] > 1.0e-15 && g[j] > max_g_act {
                    max_g_act = g[j];
                    v_idx = j;
                }
            }
            let a_dot_g = dot_slice(&a, &g);
            let gap_fw = a_dot_g - g[s_idx];
            let gap_away = max_g_act - a_dot_g;

            let (dir, alpha_max) = if gap_fw >= gap_away || a[v_idx] >= 1.0 - 1.0e-15 {
                let mut d = vec![0.0_f64; m];
                for j in 0..m {
                    d[j] = -a[j];
                }
                d[s_idx] += 1.0;
                (d, 1.0_f64)
            } else {
                let mut d = a.clone();
                d[v_idx] -= 1.0;
                (d, a[v_idx] / (1.0 - a[v_idx]))
            };

            let slope = dot_slice(&g, &dir);
            if slope >= 0.0 {
                break;
            }

            // Exact quadratic denominator 2 * ||G_tilde * d||^2
            let mut gd_sq = 0.0_f64;
            for k in 0..n {
                let mut acc = 0.0_f64;
                for j in 0..m {
                    acc += g_tilde[k][j] * dir[j];
                }
                gd_sq += acc * acc;
            }
            let step = if gd_sq > 1.0e-30 {
                (-slope / (2.0 * gd_sq)).clamp(0.0, alpha_max)
            } else {
                alpha_max
            };

            if step <= 0.0 {
                break;
            }

            for j in 0..m {
                a[j] = (a[j] + step * dir[j]).max(0.0);
                if a[j] < 1.0e-15 {
                    a[j] = 0.0;
                }
            }
            normalize_simplex(&mut a);
        }

        // Phase 3: Final exact active-set polish on the converged support
        let final_sup: Vec<usize> = (0..m).filter(|&j| a[j] > 1.0e-14).collect();
        if final_sup.len() >= 2 {
            let s_len = final_sup.len();
            let dim = s_len + 1;
            let mut kkt_mat = vec![vec![0.0_f64; dim]; dim];
            let mut kkt_rhs = vec![0.0_f64; dim];
            for r in 0..s_len {
                let ir = final_sup[r];
                kkt_rhs[r] = b_vec[ir];
                for c in 0..s_len {
                    kkt_mat[r][c] = q_mat[ir][final_sup[c]];
                }
                kkt_mat[r][s_len] = 1.0;
                kkt_mat[s_len][r] = 1.0;
            }
            kkt_rhs[s_len] = 1.0;
            if let Some(sol) = solve_linear_system(&kkt_mat, &kkt_rhs) {
                if sol[..s_len].iter().all(|&v| v.is_finite() && v >= -1.0e-13) {
                    let mut x_pol = vec![0.0_f64; m];
                    for (slot, &idx) in final_sup.iter().enumerate() {
                        x_pol[idx] = sol[slot].max(0.0);
                    }
                    normalize_simplex(&mut x_pol);
                    if eval_wls_obj(&g_tilde, &y_tilde, &x_pol)
                        <= eval_wls_obj(&g_tilde, &y_tilde, &a) + 1.0e-12
                    {
                        a = x_pol;
                    }
                }
            }
        }
    }

    // Final exact projection onto the simplex (guarantees A_m >= 0 and |sum A_m - 1| < 1e-15)
    a = project_onto_simplex(&a);
    let chi2 = eval_wls_obj(&g_tilde, &y_tilde, &a);
    g = grad_of(&a);
    let (final_kkt_viol, final_fw_gap, _) = simplex_kkt_metrics(&a, &g);

    Ok(SimplexWlsSolution {
        amplitudes: a,
        chi2,
        kkt_violation: final_kkt_viol,
        fw_gap: final_fw_gap,
        iterations: total_iters,
    })
}

/// Fit multi-field stacked `IMPACT` correlation-time amplitudes on an explicit fixed grid
/// `tau_grid_sec` from spectral density observations `obs`, with parametric Monte Carlo 95% CIs.
pub fn fit_impact_spectral_density(
    obs: &[SpectralDensityObs],
    tau_grid_sec: &[f64],
    cfg: &ImpactConfig,
) -> Result<ImpactFitResult, String> {
    validate_impact_config(cfg)?;
    validate_tau_grid(tau_grid_sec)?;
    if obs.is_empty() {
        return Err("IMPACT spectral density observations must not be empty".to_string());
    }
    let m = tau_grid_sec.len();
    let k_free = m.saturating_sub(1);
    if obs.len() < k_free.max(1) {
        return Err(format!(
            "under-determined IMPACT problem: {} observations for {} free simplex parameters (M = {})",
            obs.len(),
            k_free,
            m
        ));
    }

    let mut design_rows = Vec::with_capacity(obs.len());
    let mut targets = Vec::with_capacity(obs.len());
    let mut sigmas = Vec::with_capacity(obs.len());

    for (idx, o) in obs.iter().enumerate() {
        if !o.omega_rad_s.is_finite() || o.omega_rad_s < 0.0 {
            return Err(format!(
                "observation {} ({}) has invalid omega_rad_s = {} (must be finite and >= 0)",
                idx, o.label, o.omega_rad_s
            ));
        }
        if !o.value.is_finite() {
            return Err(format!(
                "observation {} ({}) has non-finite value = {}",
                idx, o.label, o.value
            ));
        }
        if !o.sigma.is_finite() || o.sigma <= 0.0 {
            return Err(format!(
                "observation {} ({}) has invalid sigma = {} (must be finite and > 0)",
                idx, o.label, o.sigma
            ));
        }
        let row: Vec<f64> = tau_grid_sec
            .iter()
            .map(|&tau| lorentzian_j(o.omega_rad_s, tau))
            .collect();
        design_rows.push(row);
        targets.push(o.value);
        sigmas.push(o.sigma);
    }

    run_impact_fit_core(&design_rows, &targets, &sigmas, tau_grid_sec, cfg)
}

/// Fit multi-field stacked `IMPACT` directly from multi-field $(R_1, R_2, \mathrm{hetNOE})$ rates
/// by stacking reduced spectral densities $(J(0), J(\omega_N), J(0.87\omega_H))$ across all fields
/// (Khan, Charlier, Ferrage et al. 2015).
pub fn fit_impact_from_multi_field_rates(
    rates: &[ResidueRelaxRates],
    tau_grid_sec: &[f64],
    phys: &RelaxPhysicsConfig,
    cfg: &ImpactConfig,
) -> Result<ImpactFitResult, String> {
    let obs = stack_multi_field_spectral_densities(rates, phys)?;
    fit_impact_spectral_density(&obs, tau_grid_sec, cfg)
}

/// Fit multi-field `IMPACT` directly on exact linear relaxation observables
/// $(R_1(B_0), R_2(B_0), \Gamma_{\text{NOE}}(B_0))$ without the $0.87\omega_H$ high-frequency
/// approximation.
pub fn fit_impact_from_relaxation_experiments(
    experiments: &[RelaxExperiment],
    observed_rates: &[f64],
    tau_grid_sec: &[f64],
    phys: &RelaxPhysicsConfig,
    cfg: &ImpactConfig,
) -> Result<ImpactFitResult, String> {
    validate_impact_config(cfg)?;
    validate_tau_grid(tau_grid_sec)?;
    if experiments.is_empty() {
        return Err("experiment list must not be empty".to_string());
    }
    if observed_rates.len() != experiments.len() {
        return Err(format!(
            "observed_rates length ({}) does not match experiments length ({})",
            observed_rates.len(),
            experiments.len()
        ));
    }
    let m = tau_grid_sec.len();
    let k_free = m.saturating_sub(1);
    if experiments.len() < k_free.max(1) {
        return Err(format!(
            "under-determined IMPACT relaxation fit: {} experiments for {} free simplex parameters (M = {})",
            experiments.len(),
            k_free,
            m
        ));
    }

    let sens = build_sensitivity_matrix(
        experiments,
        tau_grid_sec,
        TumblingMode::PureCorrelationSpectrum,
        phys,
    )?;
    let sigmas: Vec<f64> = experiments.iter().map(|e| e.sigma).collect();
    run_impact_fit_core(&sens.raw, observed_rates, &sigmas, tau_grid_sec, cfg)
}

/// Perform $\mathrm{AIC}_c$ model-size selection for multi-field stacked `IMPACT` across candidate
/// array sizes `candidate_sizes` (typically `&[3, 4, 5, 6]`), failing closed if all candidates are
/// under-determined ($N_{\text{obs}} \le M$).
pub fn select_impact_model_aicc(
    obs: &[SpectralDensityObs],
    candidate_sizes: &[usize],
    cfg: &ImpactConfig,
) -> Result<ImpactModelSelectionResult, String> {
    validate_impact_config(cfg)?;
    if obs.is_empty() {
        return Err("cannot run IMPACT AICc selection on empty observations".to_string());
    }
    if candidate_sizes.is_empty() {
        return Err("candidate_sizes must not be empty".to_string());
    }

    let mut no_mc_cfg = cfg.clone();
    no_mc_cfg.mc_replicates = 0;

    let mut candidates = Vec::new();
    let mut best_idx: Option<usize> = None;
    let mut best_aicc = f64::INFINITY;

    for &m in candidate_sizes {
        if m < 2 {
            return Err(format!(
                "invalid IMPACT candidate size M = {} (must be >= 2)",
                m
            ));
        }
        let k_free = m - 1;
        // Skip sizes that cannot be fit or have undefined/infinite AICc (N_obs <= k + 1 = M)
        if obs.len() < k_free {
            continue;
        }
        let grid = log_spaced_grid(cfg.tau_min_sec, cfg.tau_max_sec, m)?;
        let fit = fit_impact_spectral_density(obs, &grid, &no_mc_cfg)?;
        if fit.aicc.is_finite() && fit.aicc < best_aicc {
            best_aicc = fit.aicc;
            best_idx = Some(candidates.len());
        }
        candidates.push(fit);
    }

    let chosen_idx = best_idx.ok_or_else(|| {
        format!(
            "under-determined IMPACT AICc model selection: {} observations cannot yield finite AICc for any candidate in {:?} (requires N_obs > M)",
            obs.len(),
            candidate_sizes
        )
    })?;

    let selected_m = candidates[chosen_idx].tau_grid_sec.len();
    let best_grid = log_spaced_grid(cfg.tau_min_sec, cfg.tau_max_sec, selected_m)?;
    // Run full parametric Monte Carlo bootstrap on the AICc-selected model
    let best_fit = fit_impact_spectral_density(obs, &best_grid, cfg)?;
    candidates[chosen_idx] = best_fit.clone();

    Ok(ImpactModelSelectionResult {
        selected_num_timescales: selected_m,
        best_fit,
        candidates,
    })
}

/// Perform $\mathrm{AIC}_c$ model-size selection directly from multi-field $^{15}\text{N}$ relaxation rates.
pub fn select_impact_model_aicc_from_rates(
    rates: &[ResidueRelaxRates],
    candidate_sizes: &[usize],
    phys: &RelaxPhysicsConfig,
    cfg: &ImpactConfig,
) -> Result<ImpactModelSelectionResult, String> {
    let obs = stack_multi_field_spectral_densities(rates, phys)?;
    select_impact_model_aicc(&obs, candidate_sizes, cfg)
}

fn validate_impact_config(cfg: &ImpactConfig) -> Result<(), String> {
    if !cfg.tau_min_sec.is_finite() || cfg.tau_min_sec <= 0.0 {
        return Err(format!(
            "invalid ImpactConfig tau_min_sec = {} (must be finite and > 0)",
            cfg.tau_min_sec
        ));
    }
    if !cfg.tau_max_sec.is_finite() || cfg.tau_max_sec <= cfg.tau_min_sec {
        return Err(format!(
            "invalid ImpactConfig tau_max_sec = {} (must be > tau_min_sec = {})",
            cfg.tau_max_sec, cfg.tau_min_sec
        ));
    }
    if !cfg.kkt_tol.is_finite() || cfg.kkt_tol <= 0.0 {
        return Err(format!(
            "invalid ImpactConfig kkt_tol = {} (must be finite and > 0)",
            cfg.kkt_tol
        ));
    }
    if cfg.max_iter == 0 {
        return Err("ImpactConfig max_iter must be >= 1".to_string());
    }
    Ok(())
}

fn validate_tau_grid(tau_grid_sec: &[f64]) -> Result<(), String> {
    if tau_grid_sec.len() < 2 {
        return Err(format!(
            "IMPACT tau_grid_sec must contain at least 2 correlation times, got {}",
            tau_grid_sec.len()
        ));
    }
    for (i, &t) in tau_grid_sec.iter().enumerate() {
        if !t.is_finite() || t <= 0.0 {
            return Err(format!(
                "tau_grid_sec[{}] = {} must be finite and > 0",
                i, t
            ));
        }
        if i > 0 && t <= tau_grid_sec[i - 1] {
            return Err(format!(
                "tau_grid_sec must be strictly increasing (index {} has {} <= {})",
                i,
                t,
                tau_grid_sec[i - 1]
            ));
        }
    }
    Ok(())
}

fn run_impact_fit_core(
    design_rows: &[Vec<f64>],
    targets: &[f64],
    sigmas: &[f64],
    tau_grid_sec: &[f64],
    cfg: &ImpactConfig,
) -> Result<ImpactFitResult, String> {
    let n = targets.len();
    let m = tau_grid_sec.len();
    let sol = solve_simplex_wls(design_rows, targets, sigmas, cfg.kkt_tol, cfg.max_iter)?;

    let mut predicted = vec![0.0_f64; n];
    let mut residuals = vec![0.0_f64; n];
    for k in 0..n {
        let p = dot_slice(&design_rows[k], &sol.amplitudes);
        predicted[k] = p;
        residuals[k] = targets[k] - p;
    }

    let k_free = m - 1;
    let dof = n.saturating_sub(k_free);
    let reduced_chi2 = if dof > 0 {
        sol.chi2 / (dof as f64)
    } else {
        0.0
    };
    let k_f = k_free as f64;
    let n_f = n as f64;
    let aic = sol.chi2 + 2.0 * k_f;
    let aicc = if n > k_free + 1 {
        aic + (2.0 * k_f * (k_f + 1.0)) / ((n - k_free - 1) as f64)
    } else {
        f64::INFINITY
    };
    let bic = sol.chi2 + k_f * n_f.ln();

    // Parametric Monte Carlo bootstrap for 68% and 95% confidence intervals
    let mut sigma_amplitudes_mc = vec![0.0_f64; m];
    let mut amplitudes_ci68: Vec<[f64; 2]> = sol.amplitudes.iter().map(|&a| [a, a]).collect();
    let mut amplitudes_ci95: Vec<[f64; 2]> = sol.amplitudes.iter().map(|&a| [a, a]).collect();

    if cfg.mc_replicates > 0 {
        let mut rng = Rng::new(cfg.seed ^ ((m as u64).wrapping_mul(0x9e3779b97f4a7c15)));
        let mut samples = vec![Vec::with_capacity(cfg.mc_replicates); m];
        let mut y_boot = vec![0.0_f64; n];
        let mc_tol = cfg.kkt_tol.max(1.0e-8);
        let mc_max_iter = cfg.max_iter.min(500);

        for _rep in 0..cfg.mc_replicates {
            for k in 0..n {
                y_boot[k] = rng.normal(predicted[k], sigmas[k]);
            }
            if let Ok(boot_sol) =
                solve_simplex_wls(design_rows, &y_boot, sigmas, mc_tol, mc_max_iter)
            {
                for j in 0..m {
                    samples[j].push(boot_sol.amplitudes[j]);
                }
            }
        }

        for j in 0..m {
            let s = &mut samples[j];
            if s.len() >= 2 {
                let mean = s.iter().sum::<f64>() / (s.len() as f64);
                let var = s.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>()
                    / ((s.len() - 1) as f64);
                sigma_amplitudes_mc[j] = var.max(0.0).sqrt();
                s.sort_by(|a, b| a.total_cmp(b));
                amplitudes_ci68[j] = [sorted_quantile(s, 0.15865), sorted_quantile(s, 0.84135)];
                amplitudes_ci95[j] = [sorted_quantile(s, 0.025), sorted_quantile(s, 0.975)];
            }
        }
    }

    let sum_amplitudes: f64 = sol.amplitudes.iter().sum();

    Ok(ImpactFitResult {
        tau_grid_sec: tau_grid_sec.to_vec(),
        amplitudes: sol.amplitudes,
        sigma_amplitudes_mc,
        amplitudes_ci68,
        amplitudes_ci95,
        predicted,
        residuals,
        chi2: sol.chi2,
        dof,
        reduced_chi2,
        aic,
        aicc,
        bic,
        kkt_violation: sol.kkt_violation,
        fw_gap: sol.fw_gap,
        sum_amplitudes,
    })
}

fn eval_wls_obj(g_tilde: &[Vec<f64>], y_tilde: &[f64], a: &[f64]) -> f64 {
    let mut chi2 = 0.0_f64;
    for (row, &yk) in g_tilde.iter().zip(y_tilde.iter()) {
        let d = dot_slice(row, a) - yk;
        chi2 += d * d;
    }
    chi2
}

fn simplex_kkt_metrics(a: &[f64], g: &[f64]) -> (f64, f64, usize) {
    let mut min_g = f64::INFINITY;
    let mut min_idx = 0usize;
    let mut max_g_active = f64::NEG_INFINITY;
    for (j, (&aj, &gj)) in a.iter().zip(g.iter()).enumerate() {
        if gj < min_g {
            min_g = gj;
            min_idx = j;
        }
        if aj > 1.0e-14 && gj > max_g_active {
            max_g_active = gj;
        }
    }
    let kkt_violation = if max_g_active.is_finite() {
        (max_g_active - min_g).max(0.0)
    } else {
        0.0
    };
    let fw_gap = (dot_slice(a, g) - min_g).max(0.0);
    (kkt_violation, fw_gap, min_idx)
}

/// Exact Euclidean projection onto the probability simplex $\Delta_M$ (Duchi et al., ICML 2008).
fn project_onto_simplex(v: &[f64]) -> Vec<f64> {
    let m = v.len();
    if m == 0 {
        return Vec::new();
    }
    let mut u = v.to_vec();
    u.sort_by(|a, b| b.total_cmp(a));
    let mut cssv = 0.0_f64;
    let mut rho = 0usize;
    for (i, &ui) in u.iter().enumerate() {
        cssv += ui;
        let t = (cssv - 1.0) / ((i + 1) as f64);
        if ui - t > 0.0 {
            rho = i;
        }
    }
    let sum_rho: f64 = u[..=rho].iter().sum();
    let theta = (sum_rho - 1.0) / ((rho + 1) as f64);
    let mut out: Vec<f64> = v.iter().map(|&x| (x - theta).max(0.0)).collect();
    normalize_simplex(&mut out);
    out
}

fn normalize_simplex(a: &mut [f64]) {
    let sum: f64 = a.iter().sum();
    if sum > 1.0e-30 && sum.is_finite() {
        for x in a.iter_mut() {
            *x /= sum;
        }
    } else if !a.is_empty() {
        let inv = 1.0 / (a.len() as f64);
        for x in a.iter_mut() {
            *x = inv;
        }
    }
}

fn sorted_quantile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let pos = p.clamp(0.0, 1.0) * ((sorted.len() - 1) as f64);
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    let frac = pos - (lo as f64);
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

// ============================================================================
// Pure-Rust Full-Precision Dense Linear Algebra (Thin SVD, Cholesky, LU)
// ============================================================================

struct ThinSvd {
    /// Left singular vectors $U \in \mathbb{R}^{m \times r}$ (columns orthonormal).
    u: Vec<Vec<f64>>,
    /// Singular values $\sigma_1 \ge \dots \ge \sigma_r \ge 0$.
    s: Vec<f64>,
    /// Right singular vectors $V \in \mathbb{R}^{n \times r}$ (columns orthonormal).
    v: Vec<Vec<f64>>,
}

/// Full-precision thin SVD $A = U \Sigma V^T$ via Householder QR + one-sided Jacobi SVD.
/// Avoids squaring the condition number (never forms $A^T A$ or $A A^T$).
fn thin_svd(a: &[Vec<f64>]) -> Result<ThinSvd, String> {
    let m = a.len();
    if m == 0 {
        return Err("cannot compute SVD of empty matrix".to_string());
    }
    let n = a[0].len();
    if n == 0 {
        return Err("cannot compute SVD of zero-column matrix".to_string());
    }

    if m <= n {
        // Compute QR on A^T (n x m, tall) -> A^T = Q * R_tri, then Jacobi SVD on R_tri^T (m x m)
        let mut at = vec![vec![0.0_f64; m]; n];
        for i in 0..m {
            for j in 0..n {
                at[j][i] = a[i][j];
            }
        }
        let (q, r_tri) = householder_qr_tall(&at);
        let mut r_t = vec![vec![0.0_f64; m]; m];
        for i in 0..m {
            for j in 0..m {
                r_t[i][j] = r_tri[j][i];
            }
        }
        let (u_sq, s, v_sq) = jacobi_svd_square(&r_t);
        // V = Q * V_sq (n x m)
        let mut v = vec![vec![0.0_f64; m]; n];
        for j in 0..n {
            for c in 0..m {
                let mut acc = 0.0_f64;
                for l in 0..m {
                    acc += q[j][l] * v_sq[l][c];
                }
                v[j][c] = acc;
            }
        }
        Ok(ThinSvd { u: u_sq, s, v })
    } else {
        // Tall matrix A (m x n): QR on A = Q * R_tri, then Jacobi SVD on R_tri (n x n)
        let (q, r_tri) = householder_qr_tall(a);
        let (u_sq, s, v_sq) = jacobi_svd_square(&r_tri);
        let mut u = vec![vec![0.0_f64; n]; m];
        for i in 0..m {
            for c in 0..n {
                let mut acc = 0.0_f64;
                for l in 0..n {
                    acc += q[i][l] * u_sq[l][c];
                }
                u[i][c] = acc;
            }
        }
        Ok(ThinSvd { u, s, v: v_sq })
    }
}

/// Householder thin QR factorization of $A \in \mathbb{R}^{m \times n}$ ($m \ge n$):
/// returns $(Q \in \mathbb{R}^{m \times n}, R \in \mathbb{R}^{n \times n})$ with $Q^T Q = I_n$ and $A = Q R$.
fn householder_qr_tall(a: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let m = a.len();
    let n = a[0].len();
    let mut r = a.to_vec();
    let mut vs: Vec<(Vec<f64>, f64)> = Vec::with_capacity(n);

    for k in 0..n {
        let mut norm_sq = 0.0_f64;
        for row in r.iter().skip(k) {
            norm_sq += row[k] * row[k];
        }
        let norm = norm_sq.sqrt();
        if norm <= 1.0e-300 {
            vs.push((vec![0.0_f64; m - k], 0.0));
            continue;
        }
        let sign = if r[k][k] >= 0.0 { 1.0 } else { -1.0 };
        let alpha = -sign * norm;
        let mut v = vec![0.0_f64; m - k];
        v[0] = r[k][k] - alpha;
        for i in (k + 1)..m {
            v[i - k] = r[i][k];
        }
        let v_norm_sq: f64 = v.iter().map(|&x| x * x).sum();
        let beta = if v_norm_sq > 1.0e-300 {
            2.0 / v_norm_sq
        } else {
            0.0
        };

        // Apply H_k = I - beta v v^T to columns k..n of R
        for col in k..n {
            let mut dot = 0.0_f64;
            for i in k..m {
                dot += v[i - k] * r[i][col];
            }
            let scale = beta * dot;
            for i in k..m {
                r[i][col] -= scale * v[i - k];
            }
        }
        r[k][k] = alpha;
        for row in r.iter_mut().skip(k + 1) {
            row[k] = 0.0;
        }
        vs.push((v, beta));
    }

    // Accumulate thin Q (m x n) backwards from I_{m x n}
    let mut q = vec![vec![0.0_f64; n]; m];
    for k in 0..n {
        q[k][k] = 1.0;
    }
    for k in (0..n).rev() {
        let (ref v, beta) = vs[k];
        if beta == 0.0 {
            continue;
        }
        for col in k..n {
            let mut dot = 0.0_f64;
            for i in k..m {
                dot += v[i - k] * q[i][col];
            }
            let scale = beta * dot;
            for i in k..m {
                q[i][col] -= scale * v[i - k];
            }
        }
    }

    let mut r_top = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in i..n {
            r_top[i][j] = r[i][j];
        }
    }

    (q, r_top)
}

/// One-sided Hestenes–Jacobi SVD of a square matrix $B \in \mathbb{R}^{n \times n}$:
/// returns $(U, \Sigma, V)$ with $B = U \Sigma V^T$, $U^T U = V^T V = I_n$, sorted descending.
fn jacobi_svd_square(b: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>) {
    let n = b.len();
    let mut w = b.to_vec(); // columns converge to U * Sigma
    let mut v = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        v[i][i] = 1.0;
    }

    let max_sweeps = 64usize;
    for _sweep in 0..max_sweeps {
        let mut max_rel = 0.0_f64;
        for p in 0..n {
            for q in (p + 1)..n {
                let mut alpha = 0.0_f64;
                let mut beta = 0.0_f64;
                let mut gamma = 0.0_f64;
                for row in &w {
                    let wp = row[p];
                    let wq = row[q];
                    alpha += wp * wp;
                    beta += wq * wq;
                    gamma += wp * wq;
                }
                let denom = (alpha * beta).sqrt();
                if denom <= 1.0e-300 {
                    continue;
                }
                let rel = gamma.abs() / denom;
                if rel > max_rel {
                    max_rel = rel;
                }
                if rel <= 1.0e-15 {
                    continue;
                }
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = if zeta >= 0.0 {
                    1.0 / (zeta + (1.0 + zeta * zeta).sqrt())
                } else {
                    -1.0 / (-zeta + (1.0 + zeta * zeta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;

                for row in &mut w {
                    let wp = row[p];
                    let wq = row[q];
                    row[p] = c * wp - s * wq;
                    row[q] = s * wp + c * wq;
                }
                for row in &mut v {
                    let vp = row[p];
                    let vq = row[q];
                    row[p] = c * vp - s * vq;
                    row[q] = s * vp + c * vq;
                }
            }
        }
        if max_rel <= 1.0e-14 {
            break;
        }
    }

    let mut col_norms = vec![0.0_f64; n];
    for c in 0..n {
        let mut sq = 0.0_f64;
        for row in &w {
            sq += row[c] * row[c];
        }
        col_norms[c] = sq.sqrt();
    }

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| col_norms[b].total_cmp(&col_norms[a]));

    let mut s_sorted = vec![0.0_f64; n];
    let mut u_sorted = vec![vec![0.0_f64; n]; n];
    let mut v_sorted = vec![vec![0.0_f64; n]; n];
    let mut valid_cols = 0usize;

    for (new_c, &old_c) in order.iter().enumerate() {
        let sv = col_norms[old_c];
        s_sorted[new_c] = sv;
        for r in 0..n {
            v_sorted[r][new_c] = v[r][old_c];
        }
        if sv > 1.0e-15 * col_norms[order[0]].max(1.0e-30) {
            for r in 0..n {
                u_sorted[r][new_c] = w[r][old_c] / sv;
            }
            valid_cols += 1;
        }
    }

    // Complete orthonormal basis for any numerically zero singular values
    for c in valid_cols..n {
        for cand in 0..n {
            let mut vec_u = vec![0.0_f64; n];
            vec_u[cand] = 1.0;
            for prev in 0..c {
                let mut d = 0.0_f64;
                for r in 0..n {
                    d += vec_u[r] * u_sorted[r][prev];
                }
                for r in 0..n {
                    vec_u[r] -= d * u_sorted[r][prev];
                }
            }
            let nrm = l2_norm(&vec_u);
            if nrm > 1.0e-8 {
                for r in 0..n {
                    u_sorted[r][c] = vec_u[r] / nrm;
                }
                break;
            }
        }
    }

    (u_sorted, s_sorted, v_sorted)
}

fn validate_covariance_matrix(cov: &[Vec<f64>], n: usize) -> Result<(), String> {
    if cov.len() != n {
        return Err(format!(
            "covariance matrix row count ({}) != expected dimension ({})",
            cov.len(),
            n
        ));
    }
    let mut max_diag = 0.0_f64;
    for (i, row) in cov.iter().enumerate() {
        if row.len() != n {
            return Err(format!(
                "covariance matrix row {} has length {} != {}",
                i,
                row.len(),
                n
            ));
        }
        if !row[i].is_finite() || row[i] < 0.0 {
            return Err(format!(
                "covariance diagonal [{},{}] = {} must be finite and >= 0",
                i, i, row[i]
            ));
        }
        if row[i] > max_diag {
            max_diag = row[i];
        }
        for (j, &v) in row.iter().enumerate() {
            if !v.is_finite() {
                return Err(format!("non-finite covariance entry [{},{}] = {}", i, j, v));
            }
        }
    }
    let tol = 1.0e-9 * max_diag.max(1.0);
    for i in 0..n {
        for j in (i + 1)..n {
            if (cov[i][j] - cov[j][i]).abs() > tol {
                return Err(format!(
                    "covariance matrix is not symmetric at [{},{}]: {} vs {}",
                    i, j, cov[i][j], cov[j][i]
                ));
            }
        }
    }
    // Verify positive semidefiniteness via Cholesky with tiny jitter
    let mut shifted = cov.to_vec();
    let eps = 1.0e-12 * max_diag.max(1.0);
    for i in 0..n {
        shifted[i][i] += eps;
    }
    if cholesky_decompose(&shifted).is_none() {
        return Err("covariance matrix is not positive semidefinite".to_string());
    }
    Ok(())
}

/// Cholesky decomposition $A = L L^T$ for symmetric positive-definite $A$.
fn cholesky_decompose(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    let mut l = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }
            if i == j {
                if !sum.is_finite() || sum <= 1.0e-28 {
                    return None;
                }
                l[i][j] = sum.sqrt();
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }
    Some(l)
}

/// Forward substitution $L \mathbf{x} = \mathbf{b}$.
fn cholesky_solve_lower(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = l.len();
    let mut x = vec![0.0_f64; n];
    for i in 0..n {
        let mut s = b[i];
        for j in 0..i {
            s -= l[i][j] * x[j];
        }
        x[i] = s / l[i][i];
    }
    x
}

/// Backward substitution $L^T \mathbf{x} = \mathbf{b}$.
fn cholesky_solve_upper_transpose(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = l.len();
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= l[j][i] * x[j];
        }
        x[i] = s / l[i][i];
    }
    x
}

/// Solve square linear system $A \mathbf{x} = \mathbf{b}$ via Gaussian elimination with partial pivoting.
fn solve_linear_system(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = a.len();
    if n == 0 || b.len() != n {
        return None;
    }
    let mut aug = vec![vec![0.0_f64; n + 1]; n];
    let mut max_entry = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            let v = a[i][j];
            if !v.is_finite() {
                return None;
            }
            if v.abs() > max_entry {
                max_entry = v.abs();
            }
            aug[i][j] = v;
        }
        aug[i][n] = b[i];
    }
    let piv_tol = 1.0e-14 * max_entry.max(1.0);

    for col in 0..n {
        let mut piv = col;
        let mut max_v = aug[col][col].abs();
        for (r, row) in aug.iter().enumerate().skip(col + 1) {
            if row[col].abs() > max_v {
                max_v = row[col].abs();
                piv = r;
            }
        }
        if max_v <= piv_tol {
            return None;
        }
        aug.swap(col, piv);
        let inv = 1.0 / aug[col][col];
        let piv_row = aug[col].clone();
        for row_r in aug.iter_mut().skip(col + 1) {
            let factor = row_r[col] * inv;
            if factor != 0.0 {
                for c in col..=n {
                    row_r[c] -= factor * piv_row[c];
                }
            }
        }
    }

    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut s = aug[i][n];
        for j in (i + 1)..n {
            s -= aug[i][j] * x[j];
        }
        let val = s / aug[i][i];
        if !val.is_finite() {
            return None;
        }
        x[i] = val;
    }
    Some(x)
}

/// Invert a general square matrix $A \in \mathbb{R}^{n \times n}$ via Gauss–Jordan with partial pivoting.
fn invert_square_matrix(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    if n == 0 {
        return None;
    }
    let mut aug = vec![vec![0.0_f64; 2 * n]; n];
    let mut max_entry = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            let v = a[i][j];
            if !v.is_finite() {
                return None;
            }
            if v.abs() > max_entry {
                max_entry = v.abs();
            }
            aug[i][j] = v;
        }
        aug[i][n + i] = 1.0;
    }
    let piv_tol = 1.0e-14 * max_entry.max(1.0);

    for col in 0..n {
        let mut piv = col;
        let mut max_v = aug[col][col].abs();
        for (r, row) in aug.iter().enumerate().skip(col + 1) {
            if row[col].abs() > max_v {
                max_v = row[col].abs();
                piv = r;
            }
        }
        if max_v <= piv_tol {
            return None;
        }
        aug.swap(col, piv);
        let inv = 1.0 / aug[col][col];
        for c in col..(2 * n) {
            aug[col][c] *= inv;
        }
        let piv_row = aug[col].clone();
        for (r, row_r) in aug.iter_mut().enumerate() {
            if r == col {
                continue;
            }
            let factor = row_r[col];
            if factor != 0.0 {
                for c in col..(2 * n) {
                    row_r[c] -= factor * piv_row[c];
                }
            }
        }
    }

    let mut inv_mat = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        inv_mat[i].copy_from_slice(&aug[i][n..(2 * n)]);
    }
    Some(inv_mat)
}

fn invert_3x3(m: [[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let a = m[0][0];
    let b = m[0][1];
    let c = m[0][2];
    let d = m[1][0];
    let e = m[1][1];
    let f = m[1][2];
    let g = m[2][0];
    let h = m[2][1];
    let i = m[2][2];

    let c00 = e * i - f * h;
    let c01 = -(d * i - f * g);
    let c02 = d * h - e * g;
    let det = a * c00 + b * c01 + c * c02;

    let scale = m
        .iter()
        .flat_map(|row| row.iter())
        .map(|v| v.abs())
        .fold(0.0_f64, f64::max);
    if !det.is_finite() || det.abs() <= 1.0e-14 * scale.powi(3).max(1.0e-300) {
        return None;
    }

    let inv_det = 1.0 / det;
    Some([
        [
            c00 * inv_det,
            -(b * i - c * h) * inv_det,
            (b * f - c * e) * inv_det,
        ],
        [
            c01 * inv_det,
            (a * i - c * g) * inv_det,
            -(a * f - c * d) * inv_det,
        ],
        [
            c02 * inv_det,
            -(a * h - b * g) * inv_det,
            (a * e - b * d) * inv_det,
        ],
    ])
}

fn matrix_abs_determinant(a: &[Vec<f64>]) -> f64 {
    let n = a.len();
    if n == 0 {
        return 0.0;
    }
    let mut m = a.to_vec();
    let mut det = 1.0_f64;
    for col in 0..n {
        let mut piv = col;
        let mut max_v = m[col][col].abs();
        for (r, row) in m.iter().enumerate().skip(col + 1) {
            if row[col].abs() > max_v {
                max_v = row[col].abs();
                piv = r;
            }
        }
        if max_v <= 1.0e-30 {
            return 0.0;
        }
        m.swap(col, piv);
        det *= m[col][col];
        let inv = 1.0 / m[col][col];
        let piv_row = m[col].clone();
        for row_r in m.iter_mut().skip(col + 1) {
            let factor = row_r[col] * inv;
            for c in col..n {
                row_r[c] -= factor * piv_row[c];
            }
        }
    }
    det.abs()
}

fn matrix_condition_number(a: &[Vec<f64>]) -> Option<f64> {
    if a.is_empty() {
        return None;
    }
    let (_, s, _) = jacobi_svd_square(a);
    let s_max = *s.first()?;
    let s_min = *s.last()?;
    if s_min <= 1.0e-30 {
        None
    } else {
        Some(s_max / s_min)
    }
}

#[inline]
fn dot_slice(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

#[inline]
fn l2_norm(a: &[f64]) -> f64 {
    dot_slice(a, a).sqrt()
}
