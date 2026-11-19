//! Unassigned 2D diamagnetic-to-paramagnetic pseudocontact shift (`PCS`) bipartite peak matching /
//! assignment transfer (Kuhn–Munkres + Sinkhorn–Knopp entropic optimal transport + iterative EM
//! Huber-IRLS $\Delta\boldsymbol{\chi}$-tensor refinement) and Curie-spin / dipolar paramagnetic
//! cross-correlated relaxation ($\Gamma_{\text{CCR}}$) forward prediction & joint
//! `PCS` + $\Gamma_{\text{CCR}}$ susceptibility-tensor fitting.
//!
//! Provenance: Clean-room original implementation derived strictly from peer-reviewed literature:
//! 1. Bertini, I., Luchinat, C. & Parigi, G. (2002). "Magnetic susceptibility in paramagnetic NMR."
//!    *Prog. Nucl. Magn. Reson. Spectrosc.* 40(3), 249–273 (pseudocontact shifts
//!    $\Delta\delta^{\text{PCS}}$, Curie-spin relaxation, and Curie–dipolar cross-correlated
//!    relaxation $\Gamma_{\text{CCR}}$).
//! 2. Pintacuda, G., Hohenthanner, K., Otting, G. & Müller, N. (2003). "Angular dependence of
//!    dipole–dipole–Curie-spin coupled relaxation." *J. Biomol. NMR* 27, 115–132 (exact
//!    $P_2(\cos\theta)$ angular dependence and full susceptibility-tensor $\boldsymbol{\chi}$
//!    Curie–dipolar cross-correlated relaxation).
//! 3. Orton, H. W., Kuprov, I., Loh, C.-T. & Otting, G. (2016). *J. Biomol. NMR* 64, 243–254 &
//!    *Magn. Reson.* (2020), 1, 1–12 (paramagnetic $\Delta\boldsymbol{\chi}$-tensor fitting,
//!    Curie-spin cross-correlated relaxation, and diagonal `PCS` assignment transfer).
//! 4. Schmitz, C., Stanton-Cook, M. J., Su, X.-C., Otting, G. & Huber, T. (2008). *J. Biomol. NMR*
//!    41(3), 179–189 (5-parameter traceless symmetric $\Delta\boldsymbol{\chi}$-tensor SVD and UTS
//!    principal-axis ordering).
//! 5. Cornilescu, G., Marquardt, J. L., Ottiger, M. & Bax, A. (1998). *J. Am. Chem. Soc.* 120(27),
//!    6836–6837 (anisotropic quality factor $Q$ and cross-validated $Q_{\text{free}}$).
//! 6. Kuhn, H. W. (1955). *Naval Res. Logist. Quart.* 2, 83–97; Munkres, J. (1957). *J. Soc.
//!    Indust. Appl. Math.* 5, 32–38 (exact polynomial-time minimum-cost bipartite matching).
//! 7. Cuturi, M. (2013). "Sinkhorn Distances: Lightspeed Computation of Optimal Transport." *Adv.
//!    Neural Inf. Process. Syst.* 26, 2292–2300; Gold, S. & Rangarajan, A. (1996). *IEEE Trans.
//!    Pattern Anal. Mach. Intell.* 18(4), 377–388 (entropic optimal transport / Sinkhorn–Knopp
//!    marginal posteriors with explicit unmatched slack states).
//! 8. Huber, P. J. (1964). *Ann. Math. Stat.* 35(1), 73–101 (robust standardized Huber IRLS).

use std::collections::{BTreeMap, BTreeSet};
use std::f64::consts::PI;

use serde::{Deserialize, Serialize};

use crate::csp::hungarian_min_cost_assignment;
use crate::noesy::{v_dist, Vec3};
use crate::paramagnetic::{
    fit_pcs_tensor_with_options, pcs_design_row, predict_pcs_from_tensor, rotation_matrix_from_zyz,
    zyz_euler_from_rotation_matrix, DeltaChiTensor, PcsFitOptions, PcsObservation,
    MAX_PCS_DESIGN_COND,
};

// ============================================================================
// 1. Exact CODATA-2018 Physical Constants & Magnetic Susceptibility Conversions
// ============================================================================

/// CODATA-2018 vacuum permeability factor $\mu_0 / (4\pi) = 10^{-7}\text{ T}^2\text{ m}^3\text{ J}^{-1}$.
pub const CODATA_MU0_OVER_4PI: f64 = 1.0e-7;
/// CODATA-2018 vacuum magnetic permeability $\mu_0 = 4\pi \times 10^{-7}\text{ N A}^{-2}$.
pub const CODATA_MU0: f64 = 4.0 * PI * CODATA_MU0_OVER_4PI;
/// CODATA-2018 reduced Planck constant $\hbar = h / (2\pi)$ ($\text{J s}$).
pub const CODATA_HBAR_J_S: f64 = 1.054_571_817e-34;
/// CODATA-2018 exact Boltzmann constant $k_B$ ($\text{J K}^{-1}$).
pub const CODATA_KB_J_K: f64 = 1.380_649e-23;
/// CODATA-2018 Bohr magneton $\mu_B$ ($\text{J T}^{-1}$).
pub const CODATA_MU_B_J_T: f64 = 9.274_010_078_3e-24;
/// CODATA-2018 $^1\text{H}$ gyromagnetic ratio $\gamma_H > 0$ ($\text{rad s}^{-1}\text{ T}^{-1}$).
pub const CODATA_GAMMA_1H_RAD_S_T: f64 = 2.675_221_874_4e8;
/// CODATA-2018 $^{15}\text{N}$ signed gyromagnetic ratio $\gamma_N < 0$ ($\text{rad s}^{-1}\text{ T}^{-1}$).
pub const CODATA_GAMMA_15N_RAD_S_T: f64 = -2.712_618_04e7;
/// CODATA-2018 $^{13}\text{C}$ signed gyromagnetic ratio $\gamma_C > 0$ ($\text{rad s}^{-1}\text{ T}^{-1}$).
pub const CODATA_GAMMA_13C_RAD_S_T: f64 = 6.728_284e7;

/// Canonical effective amide $\text{N–H}$ bond length ($1.02\text{ Å}$) for relaxation calculations.
pub const DEFAULT_R_NH_ANGSTROM: f64 = 1.02;
/// Canonical effective aliphatic $\text{C–H}$ bond length ($1.09\text{ Å}$) for relaxation calculations.
pub const DEFAULT_R_CH_ANGSTROM: f64 = 1.09;

/// Minimum physically permitted metal-to-nucleus distance ($\text{Å}$) for dipolar point-dipole models.
const MIN_METAL_NUCLEUS_DIST_A: f64 = 1.2;
/// Minimum permitted internuclear bond length ($\text{Å}$) between coupled spins $I$ and $S$.
const MIN_BOND_VECTOR_NORM_A: f64 = 1.0e-6;
/// Dimensionless equilibration scale for the optional constant PCS referencing-offset column.
const PCS_OFFSET_COL_SCALE: f64 = 0.1;

/// Convert an effective paramagnetic moment $\mu_{\text{eff}}$ (in Bohr magnetons $\mu_B$) at
/// absolute temperature `temperature_k` ($\text{K}$) to isotropic per-ion magnetic susceptibility
/// $\chi_{\text{iso}}$ in units of $10^{-32}\text{ m}^3$ via the exact SI Curie law
/// (Bertini, Luchinat & Parigi 2002):
/// $$\chi_{\text{iso}} = \frac{\mu_0 (\mu_{\text{eff}}\mu_B)^2}{3 k_B T}.$$
pub fn chi_iso_from_mu_eff_bohr(mu_eff_bohr: f64, temperature_k: f64) -> Result<f64, String> {
    if !mu_eff_bohr.is_finite() || mu_eff_bohr < 0.0 {
        return Err(format!(
            "invalid mu_eff_bohr ({mu_eff_bohr}): must be finite and >= 0"
        ));
    }
    if !temperature_k.is_finite() || temperature_k <= 0.0 {
        return Err(format!(
            "invalid temperature_k ({temperature_k}): must be finite and > 0"
        ));
    }
    let mu_si = mu_eff_bohr * CODATA_MU_B_J_T;
    let chi_si_m3 = (CODATA_MU0 * mu_si * mu_si) / (3.0 * CODATA_KB_J_K * temperature_k);
    Ok(chi_si_m3 * 1.0e32)
}

/// Convert an isotropic per-ion magnetic susceptibility $\chi_{\text{iso}} \ge 0$ (in units of
/// $10^{-32}\text{ m}^3$) at absolute temperature `temperature_k` ($\text{K}$) to effective
/// paramagnetic moment $\mu_{\text{eff}}$ in Bohr magnetons $\mu_B$.
pub fn mu_eff_bohr_from_chi_iso(chi_iso_1e32_m3: f64, temperature_k: f64) -> Result<f64, String> {
    if !chi_iso_1e32_m3.is_finite() || chi_iso_1e32_m3 < 0.0 {
        return Err(format!(
            "invalid chi_iso_1e32_m3 ({chi_iso_1e32_m3}): must be finite and >= 0"
        ));
    }
    if !temperature_k.is_finite() || temperature_k <= 0.0 {
        return Err(format!(
            "invalid temperature_k ({temperature_k}): must be finite and > 0"
        ));
    }
    let chi_si_m3 = chi_iso_1e32_m3 * 1.0e-32;
    let mu_si = ((3.0 * CODATA_KB_J_K * temperature_k * chi_si_m3) / CODATA_MU0).sqrt();
    Ok(mu_si / CODATA_MU_B_J_T)
}

/// Compute the Cornilescu quality factor $Q = \sqrt{\sum_i (y_i - \hat{y}_i)^2 / \sum_i y_i^2}$
/// (Cornilescu et al., *J. Am. Chem. Soc.* 1998, 120, 6836–6837).
pub fn cornilescu_q_factor(obs: &[f64], calc: &[f64]) -> Result<f64, String> {
    if obs.is_empty() || obs.len() != calc.len() {
        return Err(format!(
            "cornilescu_q_factor requires non-empty equal-length slices (got {} and {})",
            obs.len(),
            calc.len()
        ));
    }
    let mut sum_sq_err = 0.0_f64;
    let mut sum_sq_obs = 0.0_f64;
    for (&y, &yh) in obs.iter().zip(calc.iter()) {
        if !y.is_finite() || !yh.is_finite() {
            return Err("non-finite value in cornilescu_q_factor".to_string());
        }
        let d = y - yh;
        sum_sq_err += d * d;
        sum_sq_obs += y * y;
    }
    if sum_sq_obs <= 1.0e-24 {
        return Err("degenerate zero-norm observation vector in cornilescu_q_factor".to_string());
    }
    Ok((sum_sq_err / sum_sq_obs).sqrt())
}

/// Construct a canonical [`DeltaChiTensor`] directly from spherical UTS principal parameters
/// $(\Delta\chi_{\text{ax}}, \Delta\chi_{\text{rh}})$ in $10^{-32}\text{ m}^3$ and ZYZ Euler angles
/// $(\alpha, \beta, \gamma)$ in radians (`R = R_z(gamma) * R_y(beta) * R_z(alpha)`), useful for
/// forward PCS prediction and rough-tensor initialization of [`match_pcs_assignment_transfer`] or
/// [`iterative_pcs_assignment_transfer`].
pub fn delta_chi_tensor_from_spherical(
    metal_pos_angstrom: Vec3,
    dchi_ax: f64,
    dchi_rh: f64,
    euler_zyz_rad: [f64; 3],
) -> Result<DeltaChiTensor, String> {
    if !metal_pos_angstrom.iter().all(|v| v.is_finite())
        || !dchi_ax.is_finite()
        || !dchi_rh.is_finite()
        || !euler_zyz_rad.iter().all(|v| v.is_finite())
    {
        return Err("non-finite parameter in delta_chi_tensor_from_spherical".to_string());
    }
    let rot = rotation_matrix_from_zyz(euler_zyz_rad[0], euler_zyz_rad[1], euler_zyz_rad[2]);
    let lam = [
        -dchi_ax / 3.0 + 0.5 * dchi_rh,
        -dchi_ax / 3.0 - 0.5 * dchi_rh,
        (2.0 / 3.0) * dchi_ax,
    ];
    // DeltaChi_mol = R^T * diag(lam) * R
    let mut mat = [[0.0_f64; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            let mut s = 0.0_f64;
            for k in 0..3 {
                s += rot[k][r] * lam[k] * rot[k][c];
            }
            mat[r][c] = s;
        }
    }
    let components_5 = [mat[0][0], mat[1][1], mat[0][1], mat[0][2], mat[1][2]];
    Ok(build_delta_chi_from_components(
        metal_pos_angstrom,
        components_5,
        0.0,
        1.0,
        &[],
    ))
}

// ============================================================================
// 2. Unassigned 2D Diamagnetic -> Paramagnetic PCS Bipartite Matching & EM
// ============================================================================

/// Assigned diamagnetic 2D heteronuclear cross-peak (e.g. $^1\text{H}^N\text{–}^{15}\text{N}$ HSQC
/// or $^1\text{H}\text{–}^{13}\text{C}$ HSQC), optionally paired with 3D atomic coordinates.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DiamagneticPeak2D {
    /// Residue sequence number (`seq_id`).
    pub seq_id: i32,
    /// Human-readable residue/spin label (e.g. `"ILE3"`).
    pub label: String,
    /// Diamagnetic proton chemical shift $\delta_H^{\text{dia}}$ (ppm).
    pub h_ppm: f64,
    /// Diamagnetic heteronucleus chemical shift $\delta_X^{\text{dia}}$ ($^{15}\text{N}$ or $^{13}\text{C}$, ppm).
    pub x_ppm: f64,
    /// Optional 3D Cartesian coordinate of the proton ($^1\text{H}$) in Å.
    pub coord_h_angstrom: Option<Vec3>,
    /// Optional 3D Cartesian coordinate of the attached heteronucleus ($^{15}\text{N}$/$^{13}\text{C}$) in Å.
    pub coord_x_angstrom: Option<Vec3>,
}

/// Unassigned observed paramagnetic 2D heteronuclear cross-peak.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParamagneticPeak2D {
    /// Caller-provided peak identifier.
    pub peak_id: usize,
    /// Observed paramagnetic proton chemical shift $\delta_H^{\text{para}}$ (ppm).
    pub h_ppm: f64,
    /// Observed paramagnetic heteronucleus chemical shift $\delta_X^{\text{para}}$ (ppm).
    pub x_ppm: f64,
    /// Optional peak intensity or signal-to-noise ratio.
    pub intensity: Option<f64>,
}

/// Configuration for single-pass 2D diamagnetic-to-paramagnetic `PCS` assignment transfer
/// (Kuhn–Munkres minimum-cost matching + Sinkhorn–Knopp entropic optimal transport posteriors).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PcsTransferConfig {
    /// Proton chemical-shift measurement standard deviation $\sigma_H$ (ppm, default `0.02`).
    pub sigma_h_ppm: f64,
    /// Heteronucleus chemical-shift measurement standard deviation $\sigma_X$ (ppm, default `0.10`).
    pub sigma_x_ppm: f64,
    /// Correlated common-mode PCS prediction uncertainty $\sigma_{\text{model}} \ge 0$ (ppm) along
    /// the $\Delta\delta_H \approx \Delta\delta_X$ diagonal when a $\Delta\boldsymbol{\chi}$ tensor
    /// is provided (default `0.04`).
    pub model_sigma_ppm: f64,
    /// Prior PCS scale $\sigma_{\text{PCS,prior}} > 0$ (ppm) along the $\Delta\delta_H \approx \Delta\delta_X$
    /// diagonal when running without a $\Delta\boldsymbol{\chi}$ tensor (`tensor == None`, default `1.5`).
    pub diagonal_pcs_prior_sigma_ppm: f64,
    /// Characteristic Solomon–Bloembergen `PRE` bleaching radius $r_{\text{bleach}}$ in Å
    /// (`0.0` disables PRE bleaching priors; default `9.0` for strong paramagnetic lanthanides).
    pub r_bleach_angstrom: f64,
    /// Small baseline non-PRE missing-peak probability $p_{\text{miss,base}} \in (0, 0.5)$ (default `0.02`).
    pub base_missing_prob: f64,
    /// Standardized bivariate Mahalanobis rejection gate $\tau_{\text{gate}} > 0$ (default `5.5`).
    pub mahalanobis_gate_sigma: f64,
    /// Bleaching probability threshold above which an unmatched residue is labelled
    /// [`PcsMatchStatus::UnmatchedBleached`] (`0.5` corresponds exactly to $r \le r_{\text{bleach}}$).
    pub bleach_classification_prob: f64,
    /// Entropic regularization temperature $\varepsilon > 0$ for Sinkhorn–Knopp (`1.0` = exact
    /// Boltzmann/Bayesian negative-log-likelihood posterior).
    pub sinkhorn_temperature: f64,
    /// Maximum Sinkhorn–Knopp log-domain iterations (default `300`).
    pub sinkhorn_max_iters: usize,
    /// Sinkhorn–Knopp marginal convergence tolerance (default `1e-9`).
    pub sinkhorn_tol: f64,
    /// Cost for leaving an observed paramagnetic peak unassigned in Sinkhorn–Knopp (default `0.0`).
    pub extra_peak_cost: f64,
    /// Optional hard upper bound on $|\Delta\delta^{\text{PCS}}|$ (ppm) for rejecting unphysical shifts.
    pub max_pcs_abs_ppm: Option<f64>,
}

impl Default for PcsTransferConfig {
    fn default() -> Self {
        Self {
            sigma_h_ppm: 0.02,
            sigma_x_ppm: 0.10,
            model_sigma_ppm: 0.04,
            diagonal_pcs_prior_sigma_ppm: 1.5,
            r_bleach_angstrom: 9.0,
            base_missing_prob: 0.02,
            mahalanobis_gate_sigma: 5.5,
            bleach_classification_prob: 0.5,
            sinkhorn_temperature: 1.0,
            sinkhorn_max_iters: 300,
            sinkhorn_tol: 1.0e-9,
            extra_peak_cost: 0.0,
            max_pcs_abs_ppm: None,
        }
    }
}

impl PcsTransferConfig {
    /// Validate all configuration parameters (fail-closed on non-finite or out-of-range inputs).
    pub fn validate(&self) -> Result<(), String> {
        if !self.sigma_h_ppm.is_finite() || self.sigma_h_ppm <= 0.0 {
            return Err(format!(
                "invalid sigma_h_ppm ({}): must be finite and > 0",
                self.sigma_h_ppm
            ));
        }
        if !self.sigma_x_ppm.is_finite() || self.sigma_x_ppm <= 0.0 {
            return Err(format!(
                "invalid sigma_x_ppm ({}): must be finite and > 0",
                self.sigma_x_ppm
            ));
        }
        if !self.model_sigma_ppm.is_finite() || self.model_sigma_ppm < 0.0 {
            return Err(format!(
                "invalid model_sigma_ppm ({}): must be finite and >= 0",
                self.model_sigma_ppm
            ));
        }
        if !self.diagonal_pcs_prior_sigma_ppm.is_finite()
            || self.diagonal_pcs_prior_sigma_ppm <= 0.0
        {
            return Err(format!(
                "invalid diagonal_pcs_prior_sigma_ppm ({}): must be finite and > 0",
                self.diagonal_pcs_prior_sigma_ppm
            ));
        }
        if !self.r_bleach_angstrom.is_finite() || self.r_bleach_angstrom < 0.0 {
            return Err(format!(
                "invalid r_bleach_angstrom ({}): must be finite and >= 0",
                self.r_bleach_angstrom
            ));
        }
        if !self.base_missing_prob.is_finite()
            || self.base_missing_prob <= 0.0
            || self.base_missing_prob >= 0.5
        {
            return Err(format!(
                "invalid base_missing_prob ({}): must be finite and in (0, 0.5)",
                self.base_missing_prob
            ));
        }
        if !self.mahalanobis_gate_sigma.is_finite() || self.mahalanobis_gate_sigma <= 0.0 {
            return Err(format!(
                "invalid mahalanobis_gate_sigma ({}): must be finite and > 0",
                self.mahalanobis_gate_sigma
            ));
        }
        if !self.bleach_classification_prob.is_finite()
            || self.bleach_classification_prob <= 0.0
            || self.bleach_classification_prob > 1.0
        {
            return Err(format!(
                "invalid bleach_classification_prob ({}): must be finite and in (0, 1]",
                self.bleach_classification_prob
            ));
        }
        if !self.sinkhorn_temperature.is_finite() || self.sinkhorn_temperature <= 0.0 {
            return Err(format!(
                "invalid sinkhorn_temperature ({}): must be finite and > 0",
                self.sinkhorn_temperature
            ));
        }
        if self.sinkhorn_max_iters == 0 {
            return Err("invalid sinkhorn_max_iters (0): must be >= 1".to_string());
        }
        if !self.sinkhorn_tol.is_finite() || self.sinkhorn_tol <= 0.0 {
            return Err(format!(
                "invalid sinkhorn_tol ({}): must be finite and > 0",
                self.sinkhorn_tol
            ));
        }
        if !self.extra_peak_cost.is_finite() {
            return Err(format!(
                "invalid extra_peak_cost ({}): must be finite",
                self.extra_peak_cost
            ));
        }
        if let Some(m) = self.max_pcs_abs_ppm {
            if !m.is_finite() || m <= 0.0 {
                return Err(format!(
                    "invalid max_pcs_abs_ppm ({m}): must be finite and > 0"
                ));
            }
        }
        Ok(())
    }
}

/// Assignment transfer status for a diamagnetic residue in the paramagnetic 2D spectrum.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PcsMatchStatus {
    /// Matched to paramagnetic peak at index `para_index` (`peak_id = para_peak_id`).
    Matched {
        para_index: usize,
        para_peak_id: usize,
    },
    /// Unmatched and inside the paramagnetic `PRE` bleaching sphere ($p_{\text{bleach}} \ge \text{threshold}$).
    UnmatchedBleached,
    /// Unmatched outside the `PRE` bleaching sphere (no paramagnetic peak within Mahalanobis gate).
    UnmatchedNoPeak,
}

/// Per-residue 2D `PCS` assignment-transfer result, geometric residual, and Sinkhorn posterior.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PcsResidueTransfer {
    pub dia_index: usize,
    pub seq_id: i32,
    pub label: String,
    pub status: PcsMatchStatus,
    /// Solomon–Bloembergen $r_M^{-6}$ `PRE` bleaching probability $\in [0, 1]$.
    pub p_bleach: f64,
    /// Distance from the amide proton (or heteronucleus fallback) to the metal center ($\text{Å}$).
    pub metal_distance_angstrom: Option<f64>,
    /// Predicted $\Delta\delta_H^{\text{PCS}}(\mathbf{r}_H)$ (ppm), when structure + $\Delta\boldsymbol{\chi}$ provided.
    pub predicted_pcs_h_ppm: Option<f64>,
    /// Predicted $\Delta\delta_X^{\text{PCS}}(\mathbf{r}_X)$ (ppm), when structure + $\Delta\boldsymbol{\chi}$ provided.
    pub predicted_pcs_x_ppm: Option<f64>,
    /// Exact geometric residual $\delta_{\text{geom}} = \Delta\delta^{\text{PCS}}(\mathbf{r}_H) - \Delta\delta^{\text{PCS}}(\mathbf{r}_X)$ (ppm).
    pub geometric_residual_ppm: Option<f64>,
    /// Observed $\Delta\delta_H = \delta_H^{\text{para}} - \delta_H^{\text{dia}}$ (ppm) when matched.
    pub observed_pcs_h_ppm: Option<f64>,
    /// Observed $\Delta\delta_X = \delta_X^{\text{para}} - \delta_X^{\text{dia}}$ (ppm) when matched.
    pub observed_pcs_x_ppm: Option<f64>,
    /// Off-diagonal residual $(\Delta\delta_H^{\text{obs}} - \Delta\delta_X^{\text{obs}}) - \delta_{\text{geom}}$ (ppm).
    pub diagonal_residual_ppm: Option<f64>,
    /// Bivariate Mahalanobis distance $d_{\text{Mahal}}$ for the matched pair.
    pub mahalanobis_distance: Option<f64>,
    /// Sinkhorn–Knopp marginal posterior probability of the selected Kuhn–Munkres state.
    pub posterior_confidence: f64,
    /// Sinkhorn–Knopp marginal probability that this residue is unmatched / bleached.
    pub unmatched_posterior: f64,
    /// Index of the best alternative paramagnetic peak (excluding the selected state).
    pub runner_up_para_index: Option<usize>,
    /// Second-highest Sinkhorn–Knopp marginal posterior probability across all alternative states.
    pub runner_up_posterior: f64,
    /// Calibrated margin $P_{\text{selected}} - P_{\text{runner-up}} \in [-1, 1]$.
    pub posterior_margin: f64,
}

/// Sinkhorn–Knopp entropic optimal-transport marginal posterior matrix for partial bipartite matching
/// (Cuturi 2013; Gold & Rangarajan 1996).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SinkhornPosteriorMatrix {
    pub n_dia: usize,
    pub n_para: usize,
    /// $N \times M$ marginal assignment probability matrix $P_{ij} = P(\text{residue } i \leftrightarrow \text{peak } j)$.
    pub match_probs: Vec<Vec<f64>>,
    /// Length-$N$ unmatched/bleached slack probabilities $s_i = 1 - \sum_j P_{ij}$.
    pub unmatched_residue_probs: Vec<f64>,
    /// Length-$M$ unassigned/artifact peak slack probabilities $t_j = 1 - \sum_i P_{ij}$.
    pub unassigned_peak_probs: Vec<f64>,
    pub iterations: usize,
    pub max_marginal_error: f64,
    pub converged: bool,
}

/// Complete output of single-pass [`match_pcs_assignment_transfer`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PcsTransferResult {
    pub transfers: Vec<PcsResidueTransfer>,
    pub sinkhorn: SinkhornPosteriorMatrix,
    pub total_matching_cost: f64,
    pub n_matched: usize,
    pub n_unmatched_bleached: usize,
    pub n_unmatched_no_peak: usize,
    /// Indices into `para_peaks` that were not matched to any diamagnetic residue.
    pub unassigned_para_indices: Vec<usize>,
}

/// Log-sum-exp helper with exact max-shift numerical stability.
fn log_sum_exp(vals: &[f64]) -> f64 {
    let mut m = f64::NEG_INFINITY;
    for &v in vals {
        if v > m {
            m = v;
        }
    }
    if !m.is_finite() {
        return f64::NEG_INFINITY;
    }
    let mut acc = 0.0_f64;
    for &v in vals {
        acc += (v - m).exp();
    }
    m + acc.ln()
}

/// Solve the strictly convex entropic partial-assignment optimal transport problem via log-domain
/// Sinkhorn–Knopp alternating dual scaling (Cuturi 2013; Gold & Rangarajan 1996):
/// $$\min_{P, s, t \ge 0} \sum_{i,j} P_{ij} C_{ij} + \sum_i s_i c_i^{\text{row}} + \sum_j t_j c_j^{\text{col}} + \varepsilon \left(H(P) + H(s) + H(t)\right)$$
/// subject to exact marginal constraints $\sum_j P_{ij} + s_i = 1$ ($\forall i$) and
/// $\sum_i P_{ij} + t_j = 1$ ($\forall j$).
pub fn sinkhorn_partial_assignment(
    cost_matrix: &[Vec<f64>],
    unmatched_row_cost: &[f64],
    unassigned_col_cost: &[f64],
    temperature: f64,
    max_iters: usize,
    tol: f64,
) -> Result<SinkhornPosteriorMatrix, String> {
    let n = cost_matrix.len();
    if unmatched_row_cost.len() != n {
        return Err(format!(
            "sinkhorn_partial_assignment: unmatched_row_cost len ({}) != n_rows ({n})",
            unmatched_row_cost.len()
        ));
    }
    let m = unassigned_col_cost.len();
    for (i, row) in cost_matrix.iter().enumerate() {
        if row.len() != m {
            return Err(format!(
                "sinkhorn_partial_assignment: row {i} len ({}) != n_cols ({m})",
                row.len()
            ));
        }
        if row.iter().any(|v| !v.is_finite()) {
            return Err(format!(
                "sinkhorn_partial_assignment: non-finite cost in row {i}"
            ));
        }
    }
    if unmatched_row_cost.iter().any(|v| !v.is_finite()) {
        return Err("sinkhorn_partial_assignment: non-finite unmatched_row_cost".to_string());
    }
    if unassigned_col_cost.iter().any(|v| !v.is_finite()) {
        return Err("sinkhorn_partial_assignment: non-finite unassigned_col_cost".to_string());
    }
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err(format!(
            "sinkhorn_partial_assignment: temperature ({temperature}) must be finite and > 0"
        ));
    }
    if max_iters == 0 {
        return Err("sinkhorn_partial_assignment: max_iters must be >= 1".to_string());
    }
    if !tol.is_finite() || tol <= 0.0 {
        return Err(format!(
            "sinkhorn_partial_assignment: tol ({tol}) must be finite and > 0"
        ));
    }

    if n == 0 {
        return Ok(SinkhornPosteriorMatrix {
            n_dia: 0,
            n_para: m,
            match_probs: Vec::new(),
            unmatched_residue_probs: Vec::new(),
            unassigned_peak_probs: vec![1.0; m],
            iterations: 0,
            max_marginal_error: 0.0,
            converged: true,
        });
    }
    if m == 0 {
        return Ok(SinkhornPosteriorMatrix {
            n_dia: n,
            n_para: 0,
            match_probs: vec![Vec::new(); n],
            unmatched_residue_probs: vec![1.0; n],
            unassigned_peak_probs: Vec::new(),
            iterations: 0,
            max_marginal_error: 0.0,
            converged: true,
        });
    }

    let inv_t = 1.0 / temperature;
    let log_k: Vec<Vec<f64>> = cost_matrix
        .iter()
        .map(|row| row.iter().map(|&c| -c * inv_t).collect())
        .collect();
    let raw_u: Vec<f64> = unmatched_row_cost.iter().map(|&c| -c * inv_t).collect();
    let raw_v: Vec<f64> = unassigned_col_cost.iter().map(|&c| -c * inv_t).collect();

    // Exact gauge-invariant centering: shifting (u_i, v_j) -> (u_i + delta, v_j - delta)
    // preserves (P_ij, s_i, t_j) identically while centering dual potentials (f, g) near 0
    // so alternating Sinkhorn-Knopp scaling converges quadratically in a few iterations.
    let mean_u = raw_u.iter().sum::<f64>() / (n as f64);
    let mean_v = raw_v.iter().sum::<f64>() / (m as f64);
    let gauge_delta = 0.5 * (mean_v - mean_u);
    let u_slack: Vec<f64> = raw_u.iter().map(|&u| u + gauge_delta).collect();
    let v_slack: Vec<f64> = raw_v.iter().map(|&v| v - gauge_delta).collect();

    // Symmetric dual initialization so tightly coupled clusters start in balanced slack gauge
    let mut f = vec![0.0_f64; n];
    let mut g = vec![0.0_f64; m];
    let mut row_buf = vec![0.0_f64; m + 1];
    let mut col_buf = vec![0.0_f64; n + 1];
    for i in 0..n {
        row_buf[..m].copy_from_slice(&log_k[i][..m]);
        row_buf[m] = u_slack[i];
        f[i] = -0.5 * log_sum_exp(&row_buf);
    }
    for j in 0..m {
        for i in 0..n {
            col_buf[i] = log_k[i][j];
        }
        col_buf[n] = v_slack[j];
        g[j] = -0.5 * log_sum_exp(&col_buf);
    }

    // Exact smooth strictly convex dual potential Phi(f, g) for monotone Anderson safeguard
    let dual_obj = |fv: &[f64], gv: &[f64]| -> f64 {
        let mut val = 0.0_f64;
        for i in 0..n {
            let fi = fv[i];
            val -= fi;
            val += (u_slack[i] + fi).exp();
            for j in 0..m {
                val += (log_k[i][j] + fi + gv[j]).exp();
            }
        }
        for j in 0..m {
            let gj = gv[j];
            val -= gj;
            val += (v_slack[j] + gj).exp();
        }
        val
    };

    let mut iterations = 0usize;
    let mut max_err = f64::INFINITY;
    let mut converged = false;

    let mut prev_t_f: Vec<f64> = Vec::new();
    let mut prev_t_g: Vec<f64> = Vec::new();
    let mut prev_r_f: Vec<f64> = Vec::new();
    let mut prev_r_g: Vec<f64> = Vec::new();

    for iter in 1..=max_iters {
        iterations = iter;
        let f_in = f.clone();
        let g_in = g.clone();

        for i in 0..n {
            for j in 0..m {
                row_buf[j] = log_k[i][j] + g[j];
            }
            row_buf[m] = u_slack[i];
            f[i] = -log_sum_exp(&row_buf);
        }
        for j in 0..m {
            for i in 0..n {
                col_buf[i] = log_k[i][j] + f[i];
            }
            col_buf[n] = v_slack[j];
            g[j] = -log_sum_exp(&col_buf);
        }

        // Check both row and column marginal residuals
        max_err = 0.0_f64;
        for i in 0..n {
            for j in 0..m {
                row_buf[j] = log_k[i][j] + f[i] + g[j];
            }
            row_buf[m] = u_slack[i] + f[i];
            let err = (log_sum_exp(&row_buf).exp() - 1.0).abs();
            if err > max_err {
                max_err = err;
            }
        }
        if max_err <= tol {
            converged = true;
            break;
        }

        // Safeguarded 1-step Anderson extrapolation on the Sinkhorn fixed-point map
        let r_f: Vec<f64> = f.iter().zip(&f_in).map(|(&a, &b)| a - b).collect();
        let r_g: Vec<f64> = g.iter().zip(&g_in).map(|(&a, &b)| a - b).collect();
        if !prev_r_f.is_empty() {
            let mut dot_num = 0.0_f64;
            let mut dot_den = 0.0_f64;
            for (&rk, &rp) in r_f.iter().zip(&prev_r_f).chain(r_g.iter().zip(&prev_r_g)) {
                let dr = rk - rp;
                dot_num += rk * dr;
                dot_den += dr * dr;
            }
            if dot_den > 1.0e-30 {
                let alpha = (dot_num / dot_den).clamp(-1.0e4, 1.0e4);
                let cand_f: Vec<f64> = f
                    .iter()
                    .zip(&prev_t_f)
                    .map(|(&tk, &tp)| tk - alpha * (tk - tp))
                    .collect();
                let cand_g: Vec<f64> = g
                    .iter()
                    .zip(&prev_t_g)
                    .map(|(&tk, &tp)| tk - alpha * (tk - tp))
                    .collect();
                if cand_f.iter().all(|v| v.is_finite()) && cand_g.iter().all(|v| v.is_finite()) {
                    let phi_base = dual_obj(&f, &g);
                    let phi_cand = dual_obj(&cand_f, &cand_g);
                    if phi_cand.is_finite() && phi_cand < phi_base {
                        f = cand_f;
                        g = cand_g;
                        prev_t_f.clear();
                        prev_t_g.clear();
                        prev_r_f.clear();
                        prev_r_g.clear();
                        continue;
                    }
                }
            }
        }
        prev_t_f = f.clone();
        prev_t_g = g.clone();
        prev_r_f = r_f;
        prev_r_g = r_g;
    }

    let mut match_probs = vec![vec![0.0_f64; m]; n];
    let mut unmatched_residue_probs = vec![0.0_f64; n];
    let mut unassigned_peak_probs = vec![0.0_f64; m];

    for i in 0..n {
        let mut row_tot = 0.0_f64;
        for j in 0..m {
            let p = (log_k[i][j] + f[i] + g[j]).exp();
            match_probs[i][j] = p;
            row_tot += p;
        }
        let s = (u_slack[i] + f[i]).exp();
        unmatched_residue_probs[i] = s;
        row_tot += s;
        if row_tot > 0.0 && (row_tot - 1.0).abs() > 1.0e-12 {
            let inv = 1.0 / row_tot;
            for p in &mut match_probs[i] {
                *p *= inv;
            }
            unmatched_residue_probs[i] *= inv;
        }
    }
    for j in 0..m {
        let col_sum: f64 = (0..n).map(|i| match_probs[i][j]).sum();
        unassigned_peak_probs[j] = (1.0 - col_sum).clamp(0.0, 1.0);
    }

    Ok(SinkhornPosteriorMatrix {
        n_dia: n,
        n_para: m,
        match_probs,
        unmatched_residue_probs,
        unassigned_peak_probs,
        iterations,
        max_marginal_error: max_err,
        converged,
    })
}

/// Evaluate the correlated bivariate Gaussian squared Mahalanobis distance for residual `(eh, ex)`
/// under independent measurement variances `(sigma_h^2, sigma_x^2)` plus common-mode correlated
/// diagonal PCS uncertainty `sigma_model^2`:
/// $$\Sigma = \begin{pmatrix}\sigma_H^2 + \sigma_m^2 & \sigma_m^2 \\ \sigma_m^2 & \sigma_X^2 + \sigma_m^2\end{pmatrix}.$$
#[inline]
fn bivariate_pcs_mahalanobis_sq(
    eh: f64,
    ex: f64,
    sigma_h: f64,
    sigma_x: f64,
    sigma_model: f64,
) -> f64 {
    let vh = sigma_h * sigma_h;
    let vx = sigma_x * sigma_x;
    let vm = sigma_model * sigma_model;
    let det = vh * vx + vm * (vh + vx);
    let diff = eh - ex;
    (vx * eh * eh + vh * ex * ex + vm * diff * diff) / det
}

fn validate_peak_slices(
    dia_peaks: &[DiamagneticPeak2D],
    para_peaks: &[ParamagneticPeak2D],
) -> Result<(), String> {
    if dia_peaks.is_empty() {
        return Err(
            "match_pcs_assignment_transfer requires at least 1 diamagnetic peak".to_string(),
        );
    }
    for (i, d) in dia_peaks.iter().enumerate() {
        if !d.h_ppm.is_finite() || !d.x_ppm.is_finite() {
            return Err(format!(
                "non-finite diamagnetic shift ({}, {}) at index {i} (seq_id {})",
                d.h_ppm, d.x_ppm, d.seq_id
            ));
        }
        if let Some(c) = d.coord_h_angstrom {
            if !c.iter().all(|v| v.is_finite()) {
                return Err(format!(
                    "non-finite coord_h_angstrom at index {i} (seq_id {})",
                    d.seq_id
                ));
            }
        }
        if let Some(c) = d.coord_x_angstrom {
            if !c.iter().all(|v| v.is_finite()) {
                return Err(format!(
                    "non-finite coord_x_angstrom at index {i} (seq_id {})",
                    d.seq_id
                ));
            }
        }
    }
    for (j, p) in para_peaks.iter().enumerate() {
        if !p.h_ppm.is_finite() || !p.x_ppm.is_finite() {
            return Err(format!(
                "non-finite paramagnetic shift ({}, {}) at index {j} (peak_id {})",
                p.h_ppm, p.x_ppm, p.peak_id
            ));
        }
        if let Some(inten) = p.intensity {
            if !inten.is_finite() {
                return Err(format!(
                    "non-finite paramagnetic intensity ({inten}) at index {j} (peak_id {})",
                    p.peak_id
                ));
            }
        }
    }
    Ok(())
}

/// Perform unassigned 2D diamagnetic-to-paramagnetic `PCS` bipartite peak matching and assignment
/// transfer via exact **Kuhn–Munkres** minimum-cost assignment (with explicit `PRE`-bleaching
/// dummy states) paired with **Sinkhorn–Knopp** entropic optimal-transport marginal posteriors.
///
/// Supports both:
/// - **Structure + $\Delta\boldsymbol{\chi}$-guided matching** (`tensor = Some(t)`): computes exact
///   predicted $(\delta_H^{\text{para}}, \delta_X^{\text{para}})$ coordinates and exact geometric
///   residual $\delta_{\text{geom}} = \Delta\delta^{\text{PCS}}(\mathbf{r}_H) - \Delta\delta^{\text{PCS}}(\mathbf{r}_X)$.
/// - **Diagonal-line matching** (`tensor = None`): matches along the near-diagonal
///   $\Delta\delta_H \approx \Delta\delta_X$ manifold using `cfg.diagonal_pcs_prior_sigma_ppm`.
pub fn match_pcs_assignment_transfer(
    dia_peaks: &[DiamagneticPeak2D],
    para_peaks: &[ParamagneticPeak2D],
    tensor: Option<&DeltaChiTensor>,
    metal_pos_override: Option<Vec3>,
    cfg: &PcsTransferConfig,
) -> Result<PcsTransferResult, String> {
    cfg.validate()?;
    validate_peak_slices(dia_peaks, para_peaks)?;
    if let Some(m) = metal_pos_override {
        if !m.iter().all(|v| v.is_finite()) {
            return Err(format!(
                "non-finite metal_pos_override ({m:?}) in match_pcs_assignment_transfer"
            ));
        }
    }
    if let Some(t) = tensor {
        if !t.metal_pos_angstrom.iter().all(|v| v.is_finite())
            || !t.components_5.iter().all(|v| v.is_finite())
            || !t.pcs_offset_ppm.is_finite()
        {
            return Err("non-finite DeltaChiTensor in match_pcs_assignment_transfer".to_string());
        }
    }

    let n = dia_peaks.len();
    let m = para_peaks.len();
    let metal_pos = metal_pos_override.or_else(|| tensor.map(|t| t.metal_pos_angstrom));

    let mut p_bleach = vec![0.0_f64; n];
    let mut metal_dists: Vec<Option<f64>> = vec![None; n];
    let mut pred_pcs_h: Vec<Option<f64>> = vec![None; n];
    let mut pred_pcs_x: Vec<Option<f64>> = vec![None; n];
    let mut geom_res: Vec<Option<f64>> = vec![None; n];

    let mut cost_matrix = vec![vec![0.0_f64; m]; n];
    let mut unmatched_row_cost = vec![0.0_f64; n];

    let gate_half_sq = 0.5 * cfg.mahalanobis_gate_sigma * cfg.mahalanobis_gate_sigma;
    let eps_prob = 1.0e-6_f64;

    for i in 0..n {
        let dia = &dia_peaks[i];
        let ref_coord = dia.coord_h_angstrom.or(dia.coord_x_angstrom);
        if let (Some(mp), Some(rc)) = (metal_pos, ref_coord) {
            let r = v_dist(rc, mp);
            metal_dists[i] = Some(r);
            if cfg.r_bleach_angstrom > 0.0 {
                let ratio = (cfg.r_bleach_angstrom / r.max(0.5)).powi(6);
                p_bleach[i] = (ratio / (1.0 + ratio)).clamp(0.0, 1.0);
            }
        }

        let (hat_h, hat_x, sigma_m) = if let Some(t) = tensor {
            if let Some(ch) = dia.coord_h_angstrom {
                let ph = predict_pcs_from_tensor(ch, t);
                let px = match dia.coord_x_angstrom {
                    Some(cx) => predict_pcs_from_tensor(cx, t),
                    None => ph,
                };
                pred_pcs_h[i] = Some(ph);
                pred_pcs_x[i] = Some(px);
                geom_res[i] = Some(ph - px);
                (dia.h_ppm + ph, dia.x_ppm + px, cfg.model_sigma_ppm)
            } else if let Some(cx) = dia.coord_x_angstrom {
                let px = predict_pcs_from_tensor(cx, t);
                pred_pcs_h[i] = Some(px);
                pred_pcs_x[i] = Some(px);
                geom_res[i] = Some(0.0);
                (dia.h_ppm + px, dia.x_ppm + px, cfg.model_sigma_ppm)
            } else {
                (dia.h_ppm, dia.x_ppm, cfg.diagonal_pcs_prior_sigma_ppm)
            }
        } else {
            (dia.h_ppm, dia.x_ppm, cfg.diagonal_pcs_prior_sigma_ppm)
        };

        let p_miss = (p_bleach[i] + (1.0 - p_bleach[i]) * cfg.base_missing_prob)
            .clamp(eps_prob, 1.0 - eps_prob);
        let p_obs = (1.0 - p_miss).clamp(eps_prob, 1.0 - eps_prob);

        let neg_ln_obs = -p_obs.ln();
        let neg_ln_miss = -p_miss.ln();
        unmatched_row_cost[i] = gate_half_sq + neg_ln_miss;

        for j in 0..m {
            let para = &para_peaks[j];
            let obs_pcs_h = para.h_ppm - dia.h_ppm;
            let obs_pcs_x = para.x_ppm - dia.x_ppm;
            let over_bound = cfg
                .max_pcs_abs_ppm
                .is_some_and(|lim| obs_pcs_h.abs() > lim || obs_pcs_x.abs() > lim);
            if over_bound {
                cost_matrix[i][j] = unmatched_row_cost[i] + 50.0;
            } else {
                let eh = para.h_ppm - hat_h;
                let ex = para.x_ppm - hat_x;
                let d2 =
                    bivariate_pcs_mahalanobis_sq(eh, ex, cfg.sigma_h_ppm, cfg.sigma_x_ppm, sigma_m);
                cost_matrix[i][j] = 0.5 * d2 + neg_ln_obs;
            }
        }
    }

    // Build augmented N x (M + N) matrix for exact Kuhn-Munkres partial matching:
    // columns 0..M are observed paramagnetic peaks; columns M..M+N are unmatched/bleached dummies.
    let mut aug_cost = vec![vec![0.0_f64; m + n]; n];
    for i in 0..n {
        aug_cost[i][..m].copy_from_slice(&cost_matrix[i][..m]);
        aug_cost[i][m..m + n].fill(unmatched_row_cost[i]);
    }

    let (row_to_col, total_matching_cost) = hungarian_min_cost_assignment(&aug_cost)?;

    // Compute calibrated Sinkhorn-Knopp entropic optimal-transport posteriors
    let unassigned_col_cost = vec![cfg.extra_peak_cost; m];
    let sinkhorn = sinkhorn_partial_assignment(
        &cost_matrix,
        &unmatched_row_cost,
        &unassigned_col_cost,
        cfg.sinkhorn_temperature,
        cfg.sinkhorn_max_iters,
        cfg.sinkhorn_tol,
    )?;

    let mut matched_para_flags = vec![false; m];
    let mut transfers = Vec::with_capacity(n);
    let mut n_matched = 0usize;
    let mut n_unmatched_bleached = 0usize;
    let mut n_unmatched_no_peak = 0usize;

    for i in 0..n {
        let dia = &dia_peaks[i];
        let assigned_col = row_to_col[i];
        let s_prob = sinkhorn.unmatched_residue_probs[i];

        let (status, obs_h, obs_x, diag_res, mahal_dist, selected_prob, selected_col_opt) =
            if assigned_col < m {
                matched_para_flags[assigned_col] = true;
                n_matched += 1;
                let para = &para_peaks[assigned_col];
                let d_h = para.h_ppm - dia.h_ppm;
                let d_x = para.x_ppm - dia.x_ppm;
                let g_res = geom_res[i].unwrap_or(0.0);
                let d_res = (d_h - d_x) - g_res;
                let (hat_h, hat_x, sigma_m) = match (pred_pcs_h[i], pred_pcs_x[i]) {
                    (Some(ph), Some(px)) => (dia.h_ppm + ph, dia.x_ppm + px, cfg.model_sigma_ppm),
                    _ => (dia.h_ppm, dia.x_ppm, cfg.diagonal_pcs_prior_sigma_ppm),
                };
                let d2 = bivariate_pcs_mahalanobis_sq(
                    para.h_ppm - hat_h,
                    para.x_ppm - hat_x,
                    cfg.sigma_h_ppm,
                    cfg.sigma_x_ppm,
                    sigma_m,
                );
                (
                    PcsMatchStatus::Matched {
                        para_index: assigned_col,
                        para_peak_id: para.peak_id,
                    },
                    Some(d_h),
                    Some(d_x),
                    Some(d_res),
                    Some(d2.sqrt()),
                    sinkhorn.match_probs[i][assigned_col],
                    Some(assigned_col),
                )
            } else if p_bleach[i] >= cfg.bleach_classification_prob {
                n_unmatched_bleached += 1;
                (
                    PcsMatchStatus::UnmatchedBleached,
                    None,
                    None,
                    None,
                    None,
                    s_prob,
                    None,
                )
            } else {
                n_unmatched_no_peak += 1;
                (
                    PcsMatchStatus::UnmatchedNoPeak,
                    None,
                    None,
                    None,
                    None,
                    s_prob,
                    None,
                )
            };

        // Determine runner-up probability across all states other than the selected state
        let mut runner_up_para_index: Option<usize> = None;
        let mut runner_up_prob = 0.0_f64;
        for (j, &pj) in sinkhorn.match_probs[i].iter().enumerate() {
            if selected_col_opt == Some(j) {
                continue;
            }
            if pj > runner_up_prob {
                runner_up_prob = pj;
                runner_up_para_index = Some(j);
            }
        }
        if selected_col_opt.is_some() && s_prob > runner_up_prob {
            runner_up_prob = s_prob;
            runner_up_para_index = None;
        }

        transfers.push(PcsResidueTransfer {
            dia_index: i,
            seq_id: dia.seq_id,
            label: dia.label.clone(),
            status,
            p_bleach: p_bleach[i],
            metal_distance_angstrom: metal_dists[i],
            predicted_pcs_h_ppm: pred_pcs_h[i],
            predicted_pcs_x_ppm: pred_pcs_x[i],
            geometric_residual_ppm: geom_res[i],
            observed_pcs_h_ppm: obs_h,
            observed_pcs_x_ppm: obs_x,
            diagonal_residual_ppm: diag_res,
            mahalanobis_distance: mahal_dist,
            posterior_confidence: selected_prob,
            unmatched_posterior: s_prob,
            runner_up_para_index,
            runner_up_posterior: runner_up_prob,
            posterior_margin: selected_prob - runner_up_prob,
        });
    }

    let unassigned_para_indices: Vec<usize> = matched_para_flags
        .iter()
        .enumerate()
        .filter_map(|(j, &used)| (!used).then_some(j))
        .collect();

    Ok(PcsTransferResult {
        transfers,
        sinkhorn,
        total_matching_cost,
        n_matched,
        n_unmatched_bleached,
        n_unmatched_no_peak,
        unassigned_para_indices,
    })
}

/// Configuration for iterative EM `PCS` assignment transfer and Huber-IRLS $\Delta\boldsymbol{\chi}$
/// tensor refinement ([`iterative_pcs_assignment_transfer`]).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct IterativePcsTransferConfig {
    /// Base single-pass matching configuration (used directly on the final converged pass).
    pub transfer_cfg: PcsTransferConfig,
    /// Minimum Sinkhorn–Knopp posterior probability required for a matched residue to enter
    /// Huber-IRLS $\Delta\boldsymbol{\chi}$ SVD refitting (default `0.85`).
    pub anchor_confidence_threshold: f64,
    /// Standardized Huber IRLS outlier threshold $k_{\text{Huber}} > 0$ (default `1.5`).
    pub huber_k: f64,
    /// Maximum Huber IRLS iterations per M-step (default `25`).
    pub max_irls_iters: usize,
    /// Whether to jointly fit a constant inter-sample PCS referencing offset $\delta_{\text{off}}$.
    pub fit_offset: bool,
    /// Maximum outer EM iterations (default `20`).
    pub max_em_iters: usize,
    /// Relative L2 convergence tolerance on the 5-component $\Delta\boldsymbol{\chi}$ vector (default `1e-4`).
    pub convergence_tol: f64,
    /// Initial correlated diagonal model uncertainty $\sigma_{\text{model}}^{(0)}$ (ppm, default `0.35`)
    /// annealed smoothly down to `transfer_cfg.model_sigma_ppm` across EM iterations.
    pub initial_model_sigma_ppm: f64,
    /// Minimum number of anchor residues required to refit $\Delta\boldsymbol{\chi}$ (must be `>= 5`).
    pub min_anchors_for_fit: usize,
    /// Keep user-supplied `seed_anchors` included in the M-step anchor set throughout refinement.
    pub retain_seed_anchors: bool,
}

impl Default for IterativePcsTransferConfig {
    fn default() -> Self {
        Self {
            transfer_cfg: PcsTransferConfig::default(),
            anchor_confidence_threshold: 0.85,
            huber_k: 1.5,
            max_irls_iters: 25,
            fit_offset: false,
            max_em_iters: 20,
            convergence_tol: 1.0e-4,
            initial_model_sigma_ppm: 0.35,
            min_anchors_for_fit: 5,
            retain_seed_anchors: true,
        }
    }
}

impl IterativePcsTransferConfig {
    pub fn validate(&self) -> Result<(), String> {
        self.transfer_cfg.validate()?;
        if !self.anchor_confidence_threshold.is_finite()
            || self.anchor_confidence_threshold <= 0.0
            || self.anchor_confidence_threshold > 1.0
        {
            return Err(format!(
                "invalid anchor_confidence_threshold ({}): must be finite and in (0, 1]",
                self.anchor_confidence_threshold
            ));
        }
        if !self.huber_k.is_finite() || self.huber_k <= 0.0 {
            return Err(format!(
                "invalid huber_k ({}): must be finite and > 0",
                self.huber_k
            ));
        }
        if self.max_irls_iters == 0 {
            return Err("invalid max_irls_iters (0): must be >= 1".to_string());
        }
        if self.max_em_iters == 0 {
            return Err("invalid max_em_iters (0): must be >= 1".to_string());
        }
        if !self.convergence_tol.is_finite() || self.convergence_tol <= 0.0 {
            return Err(format!(
                "invalid convergence_tol ({}): must be finite and > 0",
                self.convergence_tol
            ));
        }
        if !self.initial_model_sigma_ppm.is_finite() || self.initial_model_sigma_ppm < 0.0 {
            return Err(format!(
                "invalid initial_model_sigma_ppm ({}): must be finite and >= 0",
                self.initial_model_sigma_ppm
            ));
        }
        if self.min_anchors_for_fit < 5 {
            return Err(format!(
                "invalid min_anchors_for_fit ({}): must be >= 5",
                self.min_anchors_for_fit
            ));
        }
        Ok(())
    }
}

/// Diagnostic trace entry recorded at each outer EM iteration.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct EmIterationTrace {
    pub iteration: usize,
    pub model_sigma_ppm: f64,
    pub n_anchors: usize,
    pub n_pcs_obs: usize,
    pub rmsd_ppm: f64,
    pub q_factor: f64,
    pub rel_tensor_change: f64,
}

/// Output of [`iterative_pcs_assignment_transfer`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IterativePcsTransferResult {
    pub transfer: PcsTransferResult,
    pub fitted_tensor: DeltaChiTensor,
    pub n_anchors_used: usize,
    pub iterations: usize,
    pub converged: bool,
    pub trace: Vec<EmIterationTrace>,
}

fn build_pcs_obs_from_pairs(
    dia_peaks: &[DiamagneticPeak2D],
    para_peaks: &[ParamagneticPeak2D],
    pairs: &[(usize, usize)],
    sigma_h: f64,
    sigma_x: f64,
) -> Vec<PcsObservation> {
    let mut obs = Vec::with_capacity(pairs.len() * 2);
    for &(di, pi) in pairs {
        let dia = &dia_peaks[di];
        let para = &para_peaks[pi];
        if let Some(ch) = dia.coord_h_angstrom {
            obs.push(
                PcsObservation::new(dia.seq_id, ch, para.h_ppm - dia.h_ppm)
                    .with_uncertainty(sigma_h),
            );
        }
        if let Some(cx) = dia.coord_x_angstrom {
            obs.push(
                PcsObservation::new(dia.seq_id, cx, para.x_ppm - dia.x_ppm)
                    .with_uncertainty(sigma_x),
            );
        }
    }
    obs
}

/// Iteratively alternate Kuhn–Munkres + Sinkhorn–Knopp 2D `PCS` bipartite matching with
/// high-confidence anchor selection ($p \ge \text{anchor\_confidence\_threshold}$, default `0.85`)
/// and Huber-IRLS 5-parameter traceless symmetric $\Delta\boldsymbol{\chi}$-tensor SVD refitting
/// until convergence.
///
/// Initialization accepts EITHER a rough `initial_tensor` ($\Delta\boldsymbol{\chi}_0$) OR at least
/// `5` `(seq_id, para_peak_index)` seed anchors (or both).
pub fn iterative_pcs_assignment_transfer(
    dia_peaks: &[DiamagneticPeak2D],
    para_peaks: &[ParamagneticPeak2D],
    metal_pos_angstrom: Vec3,
    initial_tensor: Option<&DeltaChiTensor>,
    seed_anchors: &[(i32, usize)],
    cfg: &IterativePcsTransferConfig,
) -> Result<IterativePcsTransferResult, String> {
    cfg.validate()?;
    validate_peak_slices(dia_peaks, para_peaks)?;
    if !metal_pos_angstrom.iter().all(|v| v.is_finite()) {
        return Err(format!(
            "non-finite metal_pos_angstrom ({metal_pos_angstrom:?}) in iterative_pcs_assignment_transfer"
        ));
    }
    if initial_tensor.is_none() && seed_anchors.len() < cfg.min_anchors_for_fit {
        return Err(format!(
            "iterative_pcs_assignment_transfer requires either initial_tensor or >= {} seed anchors (got {})",
            cfg.min_anchors_for_fit,
            seed_anchors.len()
        ));
    }

    // Resolve and validate seed anchors (seq_id -> dia_index, para_index)
    let mut seq_to_dia: BTreeMap<i32, usize> = BTreeMap::new();
    for (i, d) in dia_peaks.iter().enumerate() {
        seq_to_dia.entry(d.seq_id).or_insert(i);
    }
    let mut resolved_seeds: Vec<(usize, usize)> = Vec::with_capacity(seed_anchors.len());
    let mut seen_seed_dia = BTreeSet::new();
    let mut seen_seed_para = BTreeSet::new();
    for &(sid, pi) in seed_anchors {
        let Some(&di) = seq_to_dia.get(&sid) else {
            return Err(format!(
                "seed anchor seq_id {sid} not found in diamagnetic peak list"
            ));
        };
        if pi >= para_peaks.len() {
            return Err(format!(
                "seed anchor para_index {pi} out of range (n_para = {})",
                para_peaks.len()
            ));
        }
        if dia_peaks[di].coord_h_angstrom.is_none() && dia_peaks[di].coord_x_angstrom.is_none() {
            return Err(format!(
                "seed anchor seq_id {sid} has no 3D atomic coordinates"
            ));
        }
        if !seen_seed_dia.insert(di) || !seen_seed_para.insert(pi) {
            return Err(format!(
                "duplicate diamagnetic residue or paramagnetic peak in seed anchors (seq_id {sid}, para_index {pi})"
            ));
        }
        resolved_seeds.push((di, pi));
    }

    let pcs_fit_opts = PcsFitOptions {
        fit_offset: cfg.fit_offset,
        huber_k: Some(cfg.huber_k),
        max_irls_iters: cfg.max_irls_iters,
    };

    // Initialize DeltaChiTensor
    let mut current_tensor = if let Some(t0) = initial_tensor {
        if !t0.components_5.iter().all(|v| v.is_finite())
            || !t0.metal_pos_angstrom.iter().all(|v| v.is_finite())
            || !t0.pcs_offset_ppm.is_finite()
        {
            return Err(
                "non-finite initial_tensor in iterative_pcs_assignment_transfer".to_string(),
            );
        }
        let mut t = t0.clone();
        t.metal_pos_angstrom = metal_pos_angstrom;
        t
    } else {
        let seed_obs = build_pcs_obs_from_pairs(
            dia_peaks,
            para_peaks,
            &resolved_seeds,
            cfg.transfer_cfg.sigma_h_ppm,
            cfg.transfer_cfg.sigma_x_ppm,
        );
        fit_pcs_tensor_with_options(&seed_obs, metal_pos_angstrom, &pcs_fit_opts)?
    };

    let mut prev_matches: Option<Vec<Option<usize>>> = None;
    let mut trace = Vec::with_capacity(cfg.max_em_iters);
    let mut converged = false;
    let mut last_n_anchors = resolved_seeds.len();

    for iter in 0..cfg.max_em_iters {
        let annealed = cfg.initial_model_sigma_ppm * 0.45_f64.powi(iter as i32);
        let sigma_t = annealed.max(cfg.transfer_cfg.model_sigma_ppm);
        let mut step_cfg = cfg.transfer_cfg;
        step_cfg.model_sigma_ppm = sigma_t;

        let step_res = match_pcs_assignment_transfer(
            dia_peaks,
            para_peaks,
            Some(&current_tensor),
            Some(metal_pos_angstrom),
            &step_cfg,
        )?;

        // Collect high-confidence unbleached anchors
        let mut anchor_map: BTreeMap<usize, usize> = BTreeMap::new();
        if cfg.retain_seed_anchors {
            for &(di, pi) in &resolved_seeds {
                anchor_map.insert(di, pi);
            }
        }
        for tr in &step_res.transfers {
            if let PcsMatchStatus::Matched { para_index, .. } = tr.status {
                let has_coord = dia_peaks[tr.dia_index].coord_h_angstrom.is_some()
                    || dia_peaks[tr.dia_index].coord_x_angstrom.is_some();
                if has_coord
                    && tr.p_bleach < cfg.transfer_cfg.bleach_classification_prob
                    && tr.posterior_confidence >= cfg.anchor_confidence_threshold
                {
                    anchor_map.entry(tr.dia_index).or_insert(para_index);
                }
            }
        }

        // Safe bootstrap on iter 0 only when rough initial tensor lowers posteriors below 0.85
        if iter == 0 && anchor_map.len() < cfg.min_anchors_for_fit {
            let mut candidates: Vec<(usize, usize, f64)> = step_res
                .transfers
                .iter()
                .filter_map(|tr| match tr.status {
                    PcsMatchStatus::Matched { para_index, .. }
                        if tr.p_bleach < cfg.transfer_cfg.bleach_classification_prob
                            && tr.posterior_confidence >= 0.50
                            && (dia_peaks[tr.dia_index].coord_h_angstrom.is_some()
                                || dia_peaks[tr.dia_index].coord_x_angstrom.is_some()) =>
                    {
                        Some((tr.dia_index, para_index, tr.posterior_confidence))
                    }
                    _ => None,
                })
                .collect();
            candidates.sort_by(|a, b| b.2.total_cmp(&a.2));
            for (di, pi, _) in candidates {
                anchor_map.entry(di).or_insert(pi);
                if anchor_map.len() >= cfg.min_anchors_for_fit + 3 {
                    break;
                }
            }
        }

        let anchor_pairs: Vec<(usize, usize)> = anchor_map.into_iter().collect();
        if anchor_pairs.len() < cfg.min_anchors_for_fit {
            return Err(format!(
                "EM iteration {iter} yielded only {} high-confidence anchors (< {} required)",
                anchor_pairs.len(),
                cfg.min_anchors_for_fit
            ));
        }

        let pcs_obs = build_pcs_obs_from_pairs(
            dia_peaks,
            para_peaks,
            &anchor_pairs,
            cfg.transfer_cfg.sigma_h_ppm,
            cfg.transfer_cfg.sigma_x_ppm,
        );
        let next_tensor = fit_pcs_tensor_with_options(&pcs_obs, metal_pos_angstrom, &pcs_fit_opts)?;

        let diff_norm = next_tensor
            .components_5
            .iter()
            .zip(&current_tensor.components_5)
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        let next_norm = next_tensor
            .components_5
            .iter()
            .map(|&a| a * a)
            .sum::<f64>()
            .sqrt()
            .max(1.0e-12);
        let rel_change = diff_norm / next_norm;

        let cur_matches: Vec<Option<usize>> = step_res
            .transfers
            .iter()
            .map(|t| match t.status {
                PcsMatchStatus::Matched { para_index, .. } => Some(para_index),
                _ => None,
            })
            .collect();
        let assignments_unchanged = prev_matches.as_ref() == Some(&cur_matches);

        trace.push(EmIterationTrace {
            iteration: iter,
            model_sigma_ppm: sigma_t,
            n_anchors: anchor_pairs.len(),
            n_pcs_obs: pcs_obs.len(),
            rmsd_ppm: next_tensor.rmsd_ppm,
            q_factor: next_tensor.q_factor,
            rel_tensor_change: rel_change,
        });

        current_tensor = next_tensor;
        prev_matches = Some(cur_matches);
        last_n_anchors = anchor_pairs.len();

        let sigma_tightened = sigma_t <= cfg.transfer_cfg.model_sigma_ppm + 1.0e-12;
        if sigma_tightened && assignments_unchanged && rel_change < cfg.convergence_tol {
            converged = true;
            break;
        }
    }

    // Final consistent E-step at the converged tensor & target model_sigma_ppm
    let final_transfer = match_pcs_assignment_transfer(
        dia_peaks,
        para_peaks,
        Some(&current_tensor),
        Some(metal_pos_angstrom),
        &cfg.transfer_cfg,
    )?;

    Ok(IterativePcsTransferResult {
        transfer: final_transfer,
        fitted_tensor: current_tensor,
        n_anchors_used: last_n_anchors,
        iterations: trace.len(),
        converged,
        trace,
    })
}

// ============================================================================
// 3. Curie-Spin / Dipolar Paramagnetic Cross-Correlated Relaxation (Gamma_CCR)
// ============================================================================

/// Heteronuclear spin-1/2 pair `(S_observed, I_coupled)` for Curie–dipolar cross-correlated
/// relaxation $\Gamma_{\text{CCR}}$ (Bertini et al. 2002; Pintacuda et al. 2003; Orton et al. 2016).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CcrSpinPair {
    /// Observed $^{15}\text{N}$ transverse coherence coupled to amide $^1\text{H}^N$ ($r_{\text{NH}} = 1.02\text{ Å}$).
    N15H1,
    /// Observed amide $^1\text{H}^N$ transverse coherence coupled to $^{15}\text{N}$ ($r_{\text{NH}} = 1.02\text{ Å}$).
    H1N15,
    /// Observed $^{13}\text{C}$ transverse coherence coupled to $^1\text{H}$ ($r_{\text{CH}} = 1.09\text{ Å}$).
    C13H1,
    /// Observed $^1\text{H}$ transverse coherence coupled to $^{13}\text{C}$ ($r_{\text{CH}} = 1.09\text{ Å}$).
    H1C13,
}

impl CcrSpinPair {
    /// Signed gyromagnetic ratio $\gamma_S$ ($\text{rad s}^{-1}\text{ T}^{-1}$) of the observed spin $S$.
    #[inline]
    pub fn gamma_s_rad_s_t(self) -> f64 {
        match self {
            Self::N15H1 => CODATA_GAMMA_15N_RAD_S_T,
            Self::H1N15 | Self::H1C13 => CODATA_GAMMA_1H_RAD_S_T,
            Self::C13H1 => CODATA_GAMMA_13C_RAD_S_T,
        }
    }

    /// Signed gyromagnetic ratio $\gamma_I$ ($\text{rad s}^{-1}\text{ T}^{-1}$) of the coupled spin $I$.
    #[inline]
    pub fn gamma_i_rad_s_t(self) -> f64 {
        match self {
            Self::N15H1 | Self::C13H1 => CODATA_GAMMA_1H_RAD_S_T,
            Self::H1N15 => CODATA_GAMMA_15N_RAD_S_T,
            Self::H1C13 => CODATA_GAMMA_13C_RAD_S_T,
        }
    }

    /// Canonical effective internuclear bond length $r_{IS}$ in Å.
    #[inline]
    pub fn default_bond_length_angstrom(self) -> f64 {
        match self {
            Self::N15H1 | Self::H1N15 => DEFAULT_R_NH_ANGSTROM,
            Self::C13H1 | Self::H1C13 => DEFAULT_R_CH_ANGSTROM,
        }
    }
}

/// Mandatory experimental & physical parameters for Curie–dipolar cross-correlated relaxation
/// ($\Gamma_{\text{CCR}}$). Intentionally implements **no `Default`**: `tau_m_ns`, `proton_mhz`,
/// and `temperature_k` MUST be explicitly supplied and validated (zero silent hardcoded defaults).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CcrPhysicsParams {
    /// Global isotropic rotational correlation time $\tau_m$ in nanoseconds ($> 0$).
    pub tau_m_ns: f64,
    /// Spectrometer $^1\text{H}$ Larmor frequency $\nu_H$ in MHz ($> 0$).
    pub proton_mhz: f64,
    /// Sample temperature $T$ in Kelvin ($> 0$).
    pub temperature_k: f64,
    /// Observed/coupled heteronuclear spin pair.
    pub spin_pair: CcrSpinPair,
    /// Effective internuclear bond length $r_{IS}$ in Å ($> 0$).
    pub bond_length_angstrom: f64,
}

impl CcrPhysicsParams {
    /// Construct and strictly validate mandatory Curie–DD CCR physical parameters.
    pub fn new(
        tau_m_ns: f64,
        proton_mhz: f64,
        temperature_k: f64,
        spin_pair: CcrSpinPair,
    ) -> Result<Self, String> {
        let params = Self {
            tau_m_ns,
            proton_mhz,
            temperature_k,
            spin_pair,
            bond_length_angstrom: spin_pair.default_bond_length_angstrom(),
        };
        params.validate()?;
        Ok(params)
    }

    /// Override the effective internuclear bond length $r_{IS}$ ($\text{Å}$) with validation.
    pub fn with_bond_length_angstrom(mut self, bond_length_angstrom: f64) -> Result<Self, String> {
        self.bond_length_angstrom = bond_length_angstrom;
        self.validate()?;
        Ok(self)
    }

    /// Validate all mandatory fields (fail-closed on NaN, Inf, zero, or negative values).
    pub fn validate(&self) -> Result<(), String> {
        if !self.tau_m_ns.is_finite() || self.tau_m_ns <= 0.0 {
            return Err(format!(
                "invalid tau_m_ns ({}): mandatory rotational correlation time must be finite and > 0",
                self.tau_m_ns
            ));
        }
        if !self.proton_mhz.is_finite() || self.proton_mhz <= 0.0 {
            return Err(format!(
                "invalid proton_mhz ({}): mandatory spectrometer frequency must be finite and > 0",
                self.proton_mhz
            ));
        }
        if !self.temperature_k.is_finite() || self.temperature_k <= 0.0 {
            return Err(format!(
                "invalid temperature_k ({}): mandatory temperature must be finite and > 0",
                self.temperature_k
            ));
        }
        if !self.bond_length_angstrom.is_finite() || self.bond_length_angstrom <= 0.0 {
            return Err(format!(
                "invalid bond_length_angstrom ({}): must be finite and > 0",
                self.bond_length_angstrom
            ));
        }
        Ok(())
    }

    /// Static magnetic field strength $B_0 = 2\pi \nu_H / \gamma_H$ in Tesla.
    #[inline]
    pub fn b0_tesla(&self) -> f64 {
        (2.0 * PI * self.proton_mhz * 1.0e6) / CODATA_GAMMA_1H_RAD_S_T
    }

    /// Signed Larmor angular frequency $\omega_S = \gamma_S B_0$ ($\text{rad s}^{-1}$) of the observed spin $S$.
    #[inline]
    pub fn omega_s_rad_s(&self) -> f64 {
        self.spin_pair.gamma_s_rad_s_t() * self.b0_tesla()
    }

    /// Isotropic Lorentzian spectral density $J(\omega) = \tau_m / (1 + (\omega \tau_m)^2)$ in seconds.
    #[inline]
    pub fn spectral_density_s(&self, omega_rad_s: f64) -> f64 {
        let tau_s = self.tau_m_ns * 1.0e-9;
        let wt = omega_rad_s * tau_s;
        tau_s / (1.0 + wt * wt)
    }

    /// Transverse Curie–DD spectral density combination $4 J(0) + 3 J(\omega_S)$ in seconds.
    #[inline]
    pub fn spectral_density_combination_s(&self) -> f64 {
        4.0 * self.spectral_density_s(0.0) + 3.0 * self.spectral_density_s(self.omega_s_rad_s())
    }
}

/// Susceptibility tensor model used in Curie–dipolar cross-correlated relaxation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum CcrTensorModel {
    /// Classical Bertini et al. (2002) isotropic Curie susceptibility model:
    /// $\Gamma_{\text{CCR}} \propto \chi_{\text{iso}}\, P_2(\cos\theta_{M,IS})$.
    #[default]
    IsotropicChiOnly,
    /// Full susceptibility tensor $\boldsymbol{\chi} = \chi_{\text{iso}}\mathbf{I} + \Delta\boldsymbol{\chi}$
    /// (Pintacuda et al. 2003; Orton et al. 2016):
    /// $\Gamma_{\text{CCR}} \propto \chi_{\text{iso}}\, P_2(\cos\theta_{M,IS}) + \frac{1}{6}\operatorname{tr}(\mathbf{D}(\hat{\mathbf{u}}_M)\Delta\boldsymbol{\chi}\mathbf{D}(\hat{\mathbf{u}}_{IS}))$.
    FullChiTensor,
}

/// Experimental Curie–dipolar cross-correlated relaxation observation $\Gamma_{\text{CCR}}$ ($\text{s}^{-1}$).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CcrObservation {
    pub seq_id: i32,
    /// 3D Cartesian coordinate of the observed spin $S$ (e.g. $^{15}\text{N}$ or $^{13}\text{C}$) in Å.
    pub coord_s_angstrom: Vec3,
    /// 3D Cartesian coordinate of the coupled spin $I$ (e.g. $^1\text{H}^N$) in Å.
    pub coord_i_angstrom: Vec3,
    /// Observed Curie–dipolar cross-correlated relaxation rate $\Gamma_{\text{CCR}}$ ($\text{s}^{-1}$).
    pub ccr_rate_s_inv: f64,
    /// Measurement standard deviation $\sigma > 0$ ($\text{s}^{-1}$).
    pub uncertainty_s_inv: f64,
    /// Lipari–Szabo generalized order parameter $S^2 \in (0, 1]$ (default `1.0` for rigid body).
    pub order_parameter_s2: f64,
}

impl CcrObservation {
    pub fn new(
        seq_id: i32,
        coord_s_angstrom: Vec3,
        coord_i_angstrom: Vec3,
        ccr_rate_s_inv: f64,
        uncertainty_s_inv: f64,
    ) -> Self {
        Self {
            seq_id,
            coord_s_angstrom,
            coord_i_angstrom,
            ccr_rate_s_inv,
            uncertainty_s_inv,
            order_parameter_s2: 1.0,
        }
    }

    pub fn with_order_parameter(mut self, s2: f64) -> Self {
        self.order_parameter_s2 = s2;
        self
    }
}

/// Compute the symmetric traceless dipolar orientation matrix $\mathbf{D}(\hat{\mathbf{u}}) = 3\hat{\mathbf{u}}\hat{\mathbf{u}}^T - \mathbf{I}$.
#[inline]
fn dipolar_tensor_3x3(u: Vec3) -> [[f64; 3]; 3] {
    [
        [
            3.0 * u[0] * u[0] - 1.0,
            3.0 * u[0] * u[1],
            3.0 * u[0] * u[2],
        ],
        [
            3.0 * u[1] * u[0],
            3.0 * u[1] * u[1] - 1.0,
            3.0 * u[1] * u[2],
        ],
        [
            3.0 * u[2] * u[0],
            3.0 * u[2] * u[1],
            3.0 * u[2] * u[2] - 1.0,
        ],
    ]
}

/// Compute exact geometric CCR sensitivities:
/// returns `(phys_prefactor_per_1e32, p2_cos_theta, aniso_row_5)` where:
/// - `Gamma_CCR_iso = phys_prefactor_per_1e32 * p2_cos_theta * chi_iso_1e32`
/// - `Gamma_CCR_aniso = phys_prefactor_per_1e32 * dot(aniso_row_5, delta_chi_5)`
fn ccr_geometry_and_design_row(
    coord_s: Vec3,
    coord_i: Vec3,
    metal_pos: Vec3,
    params: &CcrPhysicsParams,
    order_parameter_s2: f64,
) -> Result<(f64, f64, [f64; 5]), String> {
    if !coord_s.iter().all(|v| v.is_finite())
        || !coord_i.iter().all(|v| v.is_finite())
        || !metal_pos.iter().all(|v| v.is_finite())
    {
        return Err("non-finite 3D coordinate in Curie-DD CCR evaluation".to_string());
    }
    if !order_parameter_s2.is_finite() || order_parameter_s2 <= 0.0 || order_parameter_s2 > 1.0 {
        return Err(format!(
            "invalid order_parameter_s2 ({order_parameter_s2}): must be finite and in (0, 1]"
        ));
    }

    let vm = [
        coord_s[0] - metal_pos[0],
        coord_s[1] - metal_pos[1],
        coord_s[2] - metal_pos[2],
    ];
    let r_m = (vm[0] * vm[0] + vm[1] * vm[1] + vm[2] * vm[2]).sqrt();
    if r_m < MIN_METAL_NUCLEUS_DIST_A {
        return Err(format!(
            "unphysical metal-to-spin distance {r_m:.3} Å (< {MIN_METAL_NUCLEUS_DIST_A} Å) in CCR evaluation"
        ));
    }
    let u_m = [vm[0] / r_m, vm[1] / r_m, vm[2] / r_m];

    let vis = [
        coord_i[0] - coord_s[0],
        coord_i[1] - coord_s[1],
        coord_i[2] - coord_s[2],
    ];
    let r_is_geom = (vis[0] * vis[0] + vis[1] * vis[1] + vis[2] * vis[2]).sqrt();
    if r_is_geom < MIN_BOND_VECTOR_NORM_A {
        return Err(format!(
            "degenerate zero-length internuclear vector ({r_is_geom:.3e} Å) in CCR evaluation"
        ));
    }
    let u_is = [vis[0] / r_is_geom, vis[1] / r_is_geom, vis[2] / r_is_geom];

    let cos_theta = (u_m[0] * u_is[0] + u_m[1] * u_is[1] + u_m[2] * u_is[2]).clamp(-1.0, 1.0);
    let p2 = 0.5 * (3.0 * cos_theta * cos_theta - 1.0);

    // Anisotropic full-chi geometric coupling: (1/6) * Tr(D_M * DeltaChi * D_IS)
    // = (1/6) * Tr(M_sym * DeltaChi) where M_sym = (1/2) * (D_IS * D_M + D_M * D_IS)
    let dm = dipolar_tensor_3x3(u_m);
    let dis = dipolar_tensor_3x3(u_is);
    let mut prod = [[0.0_f64; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            let mut s = 0.0_f64;
            for k in 0..3 {
                s += dis[r][k] * dm[k][c];
            }
            prod[r][c] = s;
        }
    }
    let mut m_sym = [[0.0_f64; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            m_sym[r][c] = 0.5 * (prod[r][c] + prod[c][r]);
        }
    }
    let inv6 = 1.0 / 6.0;
    let aniso_row_5 = [
        inv6 * (m_sym[0][0] - m_sym[2][2]),
        inv6 * (m_sym[1][1] - m_sym[2][2]),
        inv6 * (2.0 * m_sym[0][1]),
        inv6 * (2.0 * m_sym[0][2]),
        inv6 * (2.0 * m_sym[1][2]),
    ];

    let b0 = params.b0_tesla();
    let r_m_m = r_m * 1.0e-10;
    let r_is_m = params.bond_length_angstrom * 1.0e-10;
    let j_comb = params.spectral_density_combination_s();
    let gamma_i = params.spin_pair.gamma_i_rad_s_t();
    let gamma_s = params.spin_pair.gamma_s_rad_s_t();

    let phys_prefactor_per_1e32 =
        0.2 * CODATA_MU0_OVER_4PI * (CODATA_HBAR_J_S * gamma_i * gamma_s * gamma_s * b0 * 1.0e-32)
            / (4.0 * PI * r_m_m.powi(3) * r_is_m.powi(3))
            * j_comb
            * order_parameter_s2;

    Ok((phys_prefactor_per_1e32, p2, aniso_row_5))
}

/// Forward prediction of the Curie-spin / dipolar cross-correlated relaxation rate
/// $\Gamma_{\text{CCR}}$ ($\text{s}^{-1}$) for a heteronuclear spin pair $(S, I)$
/// (Bertini et al. 2002; Pintacuda et al. 2003; Orton et al. 2016):
/// $$\Gamma_{\text{CCR}} = \frac{1}{5}\left(\frac{\mu_0}{4\pi}\right)\frac{\hbar \gamma_I \gamma_S^2 B_0}{4\pi\, r_M^3\, r_{IS}^3}\left[\chi_{\text{iso}} P_2(\cos\theta_{M,IS}) + \frac{1}{6}\operatorname{tr}(\mathbf{D}_M \Delta\boldsymbol{\chi} \mathbf{D}_{IS})\right]\left(4 J(0) + 3 J(\omega_S)\right) S^2.$$
///
/// Pass `delta_chi = None` for the classical isotropic-$\chi_{\text{iso}}$ Curie model, or
/// `delta_chi = Some(tensor)` for the full susceptibility tensor $\boldsymbol{\chi} = \chi_{\text{iso}}\mathbf{I} + \Delta\boldsymbol{\chi}$.
pub fn predict_curie_dd_ccr_s_inv(
    coord_s_angstrom: Vec3,
    coord_i_angstrom: Vec3,
    metal_pos_angstrom: Vec3,
    chi_iso_1e32_m3: f64,
    delta_chi: Option<&DeltaChiTensor>,
    params: &CcrPhysicsParams,
    order_parameter_s2: f64,
) -> Result<f64, String> {
    params.validate()?;
    if !chi_iso_1e32_m3.is_finite() {
        return Err(format!(
            "non-finite chi_iso_1e32_m3 ({chi_iso_1e32_m3}) in predict_curie_dd_ccr_s_inv"
        ));
    }
    let (pref, p2, aniso_row) = ccr_geometry_and_design_row(
        coord_s_angstrom,
        coord_i_angstrom,
        metal_pos_angstrom,
        params,
        order_parameter_s2,
    )?;
    let mut geom = chi_iso_1e32_m3 * p2;
    if let Some(t) = delta_chi {
        if !t.components_5.iter().all(|v| v.is_finite()) {
            return Err("non-finite DeltaChiTensor in predict_curie_dd_ccr_s_inv".to_string());
        }
        let aniso: f64 = aniso_row
            .iter()
            .zip(&t.components_5)
            .map(|(&a, &c)| a * c)
            .sum();
        geom += aniso;
    }
    Ok(pref * geom)
}

/// Fitted isotropic susceptibility $\chi_{\text{iso}}$ from $\Gamma_{\text{CCR}}$ observations alone.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CcrFitResult {
    pub metal_pos_angstrom: Vec3,
    /// Fitted isotropic magnetic susceptibility $\chi_{\text{iso}}$ in $10^{-32}\text{ m}^3$.
    pub chi_iso_1e32_m3: f64,
    /// Propagated linear Fisher uncertainty $\sigma(\chi_{\text{iso}})$ in $10^{-32}\text{ m}^3$.
    pub chi_iso_uncertainty_1e32_m3: f64,
    /// Effective magnetic moment $\mu_{\text{eff}}$ ($\mu_B$) evaluated at `params.temperature_k` (`NaN` if $\chi_{\text{iso}} < 0$).
    pub mu_eff_bohr: f64,
    pub rmsd_s_inv: f64,
    pub q_factor: f64,
    pub q_free: Option<f64>,
    pub n_obs: usize,
}

fn validate_ccr_observations(ccr_obs: &[CcrObservation]) -> Result<(), String> {
    if ccr_obs.is_empty() {
        return Err("CCR fit requires at least 1 observation".to_string());
    }
    for o in ccr_obs {
        if !o.ccr_rate_s_inv.is_finite() {
            return Err(format!(
                "non-finite ccr_rate_s_inv ({}) at seq_id {}",
                o.ccr_rate_s_inv, o.seq_id
            ));
        }
        if !o.uncertainty_s_inv.is_finite() || o.uncertainty_s_inv <= 0.0 {
            return Err(format!(
                "invalid CCR uncertainty_s_inv ({}) at seq_id {}: must be finite and > 0",
                o.uncertainty_s_inv, o.seq_id
            ));
        }
        if !o.order_parameter_s2.is_finite()
            || o.order_parameter_s2 <= 0.0
            || o.order_parameter_s2 > 1.0
        {
            return Err(format!(
                "invalid CCR order_parameter_s2 ({}) at seq_id {}: must be finite and in (0, 1]",
                o.order_parameter_s2, o.seq_id
            ));
        }
        if !o.coord_s_angstrom.iter().all(|v| v.is_finite())
            || !o.coord_i_angstrom.iter().all(|v| v.is_finite())
        {
            return Err(format!(
                "non-finite nuclear coordinates in CCR observation at seq_id {}",
                o.seq_id
            ));
        }
    }
    Ok(())
}

/// Fit isotropic paramagnetic susceptibility $\chi_{\text{iso}}$ ($10^{-32}\text{ m}^3$) from
/// experimental $\Gamma_{\text{CCR}}$ rates via exact weighted linear least squares, with
/// Cornilescu $Q$ and deterministic $K$-fold holdout $Q_{\text{free}}$.
pub fn fit_curie_ccr_chi_iso(
    ccr_obs: &[CcrObservation],
    metal_pos_angstrom: Vec3,
    fixed_delta_chi: Option<&DeltaChiTensor>,
    params: &CcrPhysicsParams,
    cv_folds: usize,
) -> Result<CcrFitResult, String> {
    params.validate()?;
    if !metal_pos_angstrom.iter().all(|v| v.is_finite()) {
        return Err(format!(
            "non-finite metal_pos_angstrom ({metal_pos_angstrom:?}) in fit_curie_ccr_chi_iso"
        ));
    }
    validate_ccr_observations(ccr_obs)?;
    if let Some(t) = fixed_delta_chi {
        if !t.components_5.iter().all(|v| v.is_finite()) {
            return Err("non-finite fixed_delta_chi in fit_curie_ccr_chi_iso".to_string());
        }
    }

    let n = ccr_obs.len();
    let mut a_iso = Vec::with_capacity(n);
    let mut y_adj = Vec::with_capacity(n);
    let mut y_raw = Vec::with_capacity(n);
    let mut aniso_part = Vec::with_capacity(n);
    let mut weights = Vec::with_capacity(n);

    for o in ccr_obs {
        let (pref, p2, aniso_row) = ccr_geometry_and_design_row(
            o.coord_s_angstrom,
            o.coord_i_angstrom,
            metal_pos_angstrom,
            params,
            o.order_parameter_s2,
        )?;
        let aniso_val = match fixed_delta_chi {
            Some(t) => {
                pref * aniso_row
                    .iter()
                    .zip(&t.components_5)
                    .map(|(&a, &c)| a * c)
                    .sum::<f64>()
            }
            None => 0.0,
        };
        a_iso.push(pref * p2);
        aniso_part.push(aniso_val);
        y_adj.push(o.ccr_rate_s_inv - aniso_val);
        y_raw.push(o.ccr_rate_s_inv);
        weights.push(1.0 / o.uncertainty_s_inv);
    }

    let solve_1d = |mask: &[bool]| -> Result<(f64, f64), String> {
        let mut num = 0.0_f64;
        let mut den = 0.0_f64;
        for i in 0..n {
            if !mask[i] {
                continue;
            }
            let wa = a_iso[i] * weights[i];
            let wy = y_adj[i] * weights[i];
            num += wa * wy;
            den += wa * wa;
        }
        if !den.is_finite() || den <= 1.0e-24 {
            return Err(
                "degenerate CCR geometry: zero Fisher information for chi_iso (e.g. magic-angle vectors)"
                    .to_string(),
            );
        }
        Ok((num / den, (1.0 / den).sqrt()))
    };

    let all_true = vec![true; n];
    let (chi_iso, chi_unc) = solve_1d(&all_true)?;
    let calc: Vec<f64> = (0..n).map(|i| a_iso[i] * chi_iso + aniso_part[i]).collect();
    let sum_sq_err: f64 = y_raw
        .iter()
        .zip(&calc)
        .map(|(&y, &yh)| (y - yh).powi(2))
        .sum();
    let rmsd_s_inv = (sum_sq_err / (n as f64)).sqrt();
    let q_factor = cornilescu_q_factor(&y_raw, &calc)?;

    // Deterministic residue-grouped K-fold Q_free
    let q_free = if cv_folds >= 2 && n >= 3 {
        let unique_res: Vec<i32> = ccr_obs
            .iter()
            .map(|o| o.seq_id)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let k_eff = cv_folds.min(unique_res.len());
        if k_eff >= 2 {
            let mut fold_of_res: BTreeMap<i32, usize> = BTreeMap::new();
            for (rank, &sid) in unique_res.iter().enumerate() {
                fold_of_res.insert(sid, rank % k_eff);
            }
            let mut held_obs = Vec::new();
            let mut held_pred = Vec::new();
            for fold in 0..k_eff {
                let train_mask: Vec<bool> = ccr_obs
                    .iter()
                    .map(|o| fold_of_res.get(&o.seq_id).copied() != Some(fold))
                    .collect();
                if let Ok((chi_tr, _)) = solve_1d(&train_mask) {
                    for i in 0..n {
                        if !train_mask[i] {
                            held_obs.push(y_raw[i]);
                            held_pred.push(a_iso[i] * chi_tr + aniso_part[i]);
                        }
                    }
                }
            }
            if held_obs.len() >= 2 {
                cornilescu_q_factor(&held_obs, &held_pred).ok()
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };

    let mu_eff_bohr = if chi_iso >= 0.0 {
        mu_eff_bohr_from_chi_iso(chi_iso, params.temperature_k).unwrap_or(f64::NAN)
    } else {
        f64::NAN
    };

    Ok(CcrFitResult {
        metal_pos_angstrom,
        chi_iso_1e32_m3: chi_iso,
        chi_iso_uncertainty_1e32_m3: chi_unc,
        mu_eff_bohr,
        rmsd_s_inv,
        q_factor,
        q_free,
        n_obs: n,
    })
}

/// Configuration options for joint `PCS` + $\Gamma_{\text{CCR}}$ fitting ([`fit_joint_pcs_ccr`]).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct JointPcsCcrFitOptions {
    /// Whether $\Gamma_{\text{CCR}}$ uses the isotropic $\chi_{\text{iso}}$ approximation or the full
    /// $\boldsymbol{\chi} = \chi_{\text{iso}}\mathbf{I} + \Delta\boldsymbol{\chi}$ tensor.
    pub tensor_model: CcrTensorModel,
    /// Jointly fit a constant inter-sample PCS referencing offset $\delta_{\text{off}}$ (ppm).
    pub fit_pcs_offset: bool,
    /// Optional standardized-residual Huber IRLS threshold $k_{\text{Huber}} > 0$ (`None` = exact GLS/SVD).
    pub huber_k: Option<f64>,
    /// Maximum Huber IRLS iterations when `huber_k` is `Some(...)` (default `25`).
    pub max_irls_iters: usize,
    /// Deterministic residue-grouped $K$-fold cross-validation folds for holdout $Q_{\text{free}}$
    /// (`0` disables cross-validation; default `5`).
    pub cv_folds: usize,
}

impl Default for JointPcsCcrFitOptions {
    fn default() -> Self {
        Self {
            tensor_model: CcrTensorModel::IsotropicChiOnly,
            fit_pcs_offset: false,
            huber_k: None,
            max_irls_iters: 25,
            cv_folds: 5,
        }
    }
}

/// Result of joint `PCS` + $\Gamma_{\text{CCR}}$ susceptibility-tensor fitting ([`fit_joint_pcs_ccr`]).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct JointPcsCcrFitResult {
    /// Fitted 5-parameter traceless symmetric $\Delta\boldsymbol{\chi}$ tensor (UTS principal-axis convention).
    pub delta_chi: DeltaChiTensor,
    /// Fitted isotropic magnetic susceptibility $\chi_{\text{iso}}$ ($10^{-32}\text{ m}^3$).
    pub chi_iso_1e32_m3: f64,
    /// Propagated linear Fisher standard uncertainty $\sigma(\chi_{\text{iso}})$ ($10^{-32}\text{ m}^3$).
    pub chi_iso_uncertainty_1e32_m3: f64,
    /// Effective magnetic moment $\mu_{\text{eff}}$ ($\mu_B$) at `params.temperature_k` (`NaN` if $\chi_{\text{iso}} < 0$).
    pub mu_eff_bohr: f64,
    /// Full $3 \times 3$ molecular-frame susceptibility tensor $\boldsymbol{\chi} = \chi_{\text{iso}}\mathbf{I} + \Delta\boldsymbol{\chi}$ ($10^{-32}\text{ m}^3$).
    pub full_chi_matrix_mol: [[f64; 3]; 3],
    pub pcs_rmsd_ppm: f64,
    pub pcs_q_factor: f64,
    pub pcs_q_free: Option<f64>,
    pub ccr_rmsd_s_inv: f64,
    pub ccr_q_factor: f64,
    pub ccr_q_free: Option<f64>,
    pub joint_weighted_q_factor: f64,
    pub joint_q_free: Option<f64>,
    pub condition_number: f64,
    pub n_pcs_obs: usize,
    pub n_ccr_obs: usize,
}

// ============================================================================
// 4. Internal Thin Jacobi SVD (Column-Equilibrated) & 3x3 Symmetric Eigen
// ============================================================================

/// Column-equilibrated one-sided Hestenes Jacobi SVD for $N \times K$ ($K \in \{6, 7\}$)
/// weighted linear least squares, returning `(x, cond, col_variances)`.
fn solve_equilibrated_svd<const K: usize>(
    a_rows: &[[f64; K]],
    b: &[f64],
) -> Result<([f64; K], f64, [f64; K]), String> {
    let n = a_rows.len();
    if n < K + 1 || b.len() != n {
        return Err(format!(
            "joint {K}-parameter SVD requires at least {} observations (>= 1 DoF, got {n})",
            K + 1
        ));
    }
    if a_rows.iter().any(|r| r.iter().any(|v| !v.is_finite())) || b.iter().any(|v| !v.is_finite()) {
        return Err("non-finite entry in joint PCS+CCR design matrix or target vector".to_string());
    }

    // Column equilibration scales (keeps heterogeneous PCS/CCR physical units well-scaled)
    let mut col_scales = [0.0_f64; K];
    for j in 0..K {
        let norm_sq: f64 = a_rows.iter().map(|r| r[j] * r[j]).sum();
        let norm = norm_sq.sqrt();
        if !norm.is_finite() || norm <= 1.0e-15 {
            return Err(format!(
                "degenerate joint PCS+CCR geometry: column {j} has zero norm"
            ));
        }
        col_scales[j] = norm;
    }

    let mut cols: Vec<Vec<f64>> = (0..K)
        .map(|j| a_rows.iter().map(|r| r[j] / col_scales[j]).collect())
        .collect();
    let mut v = [[0.0_f64; K]; K];
    for (j, row) in v.iter_mut().enumerate() {
        row[j] = 1.0;
    }

    for _sweep in 0..50 {
        let mut max_off = 0.0_f64;
        for p in 0..K {
            for q in (p + 1)..K {
                let mut app = 0.0_f64;
                let mut aqq = 0.0_f64;
                let mut apq = 0.0_f64;
                for (&wp, &wq) in cols[p].iter().zip(&cols[q]) {
                    app += wp * wp;
                    aqq += wq * wq;
                    apq += wp * wq;
                }
                let denom = (app * aqq).sqrt();
                if denom > 0.0 {
                    max_off = max_off.max(apq.abs() / denom);
                }
                if apq.abs() <= 1.0e-15 * denom {
                    continue;
                }
                let tau = (aqq - app) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                let (left, right) = cols.split_at_mut(q);
                for (wp, wq) in left[p].iter_mut().zip(&mut right[0]) {
                    let up = *wp;
                    let uq = *wq;
                    *wp = c * up - s * uq;
                    *wq = s * up + c * uq;
                }
                for row in &mut v {
                    let vp = row[p];
                    let vq = row[q];
                    row[p] = c * vp - s * vq;
                    row[q] = s * vp + c * vq;
                }
            }
        }
        if max_off < 1.0e-14 {
            break;
        }
    }

    let mut sigmas = [0.0_f64; K];
    for j in 0..K {
        let norm_sq: f64 = cols[j].iter().map(|&val| val * val).sum();
        sigmas[j] = norm_sq.sqrt();
    }
    let s_max = sigmas.iter().copied().fold(0.0_f64, f64::max);
    let s_min = sigmas.iter().copied().fold(f64::INFINITY, f64::min);
    if !s_max.is_finite() || !s_min.is_finite() || s_max <= 1.0e-15 || s_min <= 0.0 {
        return Err(
            "degenerate joint PCS+CCR geometry: singular or rank-deficient design matrix"
                .to_string(),
        );
    }
    let cond = s_max / s_min;
    if !cond.is_finite() || cond > MAX_PCS_DESIGN_COND {
        return Err(format!(
            "ill-conditioned joint PCS+CCR geometry: condition number {cond:.3e} exceeds {:.1e}",
            MAX_PCS_DESIGN_COND
        ));
    }

    let mut x_scaled = [0.0_f64; K];
    for j in 0..K {
        let mut ut_b = 0.0_f64;
        for i in 0..n {
            ut_b += (cols[j][i] / sigmas[j]) * b[i];
        }
        let coeff = ut_b / sigmas[j];
        for k in 0..K {
            x_scaled[k] += coeff * v[k][j];
        }
    }

    let mut x = [0.0_f64; K];
    let mut vars = [0.0_f64; K];
    for k in 0..K {
        x[k] = x_scaled[k] / col_scales[k];
        let mut v_sum = 0.0_f64;
        for j in 0..K {
            let ratio = v[k][j] / sigmas[j];
            v_sum += ratio * ratio;
        }
        vars[k] = v_sum / (col_scales[k] * col_scales[k]);
    }
    if !x.iter().all(|val| val.is_finite()) {
        return Err("joint PCS+CCR SVD produced non-finite parameters".to_string());
    }
    Ok((x, cond, vars))
}

fn jacobi_sym_3x3(mut a: [[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut q = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..32 {
        let pairs = [(0usize, 1usize), (0, 2), (1, 2)];
        let mut off_sq = 0.0_f64;
        for &(p, r) in &pairs {
            off_sq += a[p][r] * a[p][r];
        }
        if off_sq.sqrt() < 1.0e-15 {
            break;
        }
        for &(p, r) in &pairs {
            let apr = a[p][r];
            if apr.abs() < 1.0e-16 {
                continue;
            }
            let tau = (a[r][r] - a[p][p]) / (2.0 * apr);
            let t = if tau >= 0.0 {
                1.0 / (tau + (1.0 + tau * tau).sqrt())
            } else {
                -1.0 / (-tau + (1.0 + tau * tau).sqrt())
            };
            let c = 1.0 / (1.0 + t * t).sqrt();
            let s = t * c;

            let app = a[p][p];
            let arr = a[r][r];
            a[p][p] = app - t * apr;
            a[r][r] = arr + t * apr;
            a[p][r] = 0.0;
            a[r][p] = 0.0;

            let m = 3 - p - r;
            let amp = a[m][p];
            let amr = a[m][r];
            a[m][p] = c * amp - s * amr;
            a[p][m] = a[m][p];
            a[m][r] = s * amp + c * amr;
            a[r][m] = a[m][r];

            for row in &mut q {
                let qp = row[p];
                let qr = row[r];
                row[p] = c * qp - s * qr;
                row[r] = s * qp + c * qr;
            }
        }
    }
    ([a[0][0], a[1][1], a[2][2]], q)
}

fn build_delta_chi_from_components(
    metal_pos: Vec3,
    x: [f64; 5],
    pcs_offset_ppm: f64,
    cond: f64,
    obs: &[PcsObservation],
) -> DeltaChiTensor {
    let (xx, yy, xy, xz, yz) = (x[0], x[1], x[2], x[3], x[4]);
    let zz = -(xx + yy);
    let mat = [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]];

    let (evals, q_cols) = jacobi_sym_3x3(mat);
    let mut order = [0usize, 1, 2];
    order.sort_by(|&a, &b| evals[b].abs().total_cmp(&evals[a].abs()));
    let idx_z = order[0];
    let idx_y = order[1];
    let idx_x = order[2];

    let dchi_zz_p = evals[idx_z];
    let dchi_xx_p = evals[idx_x];
    let dchi_yy_p = evals[idx_y];

    let mut ex = [q_cols[0][idx_x], q_cols[1][idx_x], q_cols[2][idx_x]];
    let mut ey = [q_cols[0][idx_y], q_cols[1][idx_y], q_cols[2][idx_y]];
    let mut ez = [q_cols[0][idx_z], q_cols[1][idx_z], q_cols[2][idx_z]];

    if ez[2] < 0.0 || (ez[2].abs() < 1.0e-12 && ez[1] < 0.0) {
        ez = [-ez[0], -ez[1], -ez[2]];
    }
    if ex[0] < 0.0 {
        ex = [-ex[0], -ex[1], -ex[2]];
    }
    let cross_xy = [
        ex[1] * ey[2] - ex[2] * ey[1],
        ex[2] * ey[0] - ex[0] * ey[2],
        ex[0] * ey[1] - ex[1] * ey[0],
    ];
    let det = cross_xy[0] * ez[0] + cross_xy[1] * ez[1] + cross_xy[2] * ez[2];
    if det < 0.0 {
        ey = [-ey[0], -ey[1], -ey[2]];
    }

    let rot = [ex, ey, ez];
    let euler_zyz_rad = zyz_euler_from_rotation_matrix(&rot);

    let dchi_ax = dchi_zz_p - 0.5 * (dchi_xx_p + dchi_yy_p);
    let dchi_rh = dchi_xx_p - dchi_yy_p;
    let rhombicity = if dchi_ax.abs() > 1.0e-15 {
        (dchi_rh / dchi_ax).clamp(0.0, 2.0 / 3.0)
    } else {
        0.0
    };

    let mut sum_sq_err = 0.0_f64;
    let mut sum_sq_obs = 0.0_f64;
    for o in obs {
        let row = pcs_design_row(o.coord_angstrom, metal_pos);
        let calc: f64 = row.iter().zip(&x).map(|(&r, &c)| r * c).sum::<f64>() + pcs_offset_ppm;
        let diff = o.pcs_ppm - calc;
        sum_sq_err += diff * diff;
        sum_sq_obs += o.pcs_ppm * o.pcs_ppm;
    }
    let rmsd_ppm = if !obs.is_empty() {
        (sum_sq_err / (obs.len() as f64)).sqrt()
    } else {
        0.0
    };
    let q_factor = if sum_sq_obs > 1.0e-15 && sum_sq_err.is_finite() {
        (sum_sq_err / sum_sq_obs).sqrt()
    } else {
        0.0
    };

    DeltaChiTensor {
        metal_pos_angstrom: metal_pos,
        components_5: x,
        matrix_mol: mat,
        eigenvalues_paf: [dchi_xx_p, dchi_yy_p, dchi_zz_p],
        rotation_mol_to_paf: rot,
        dchi_ax,
        dchi_rh,
        rhombicity,
        euler_zyz_rad,
        pcs_offset_ppm,
        rmsd_ppm,
        q_factor,
        condition_number: cond,
    }
}

struct JointLinearSolution {
    dchi_5: [f64; 5],
    chi_iso: f64,
    chi_iso_unc: f64,
    pcs_offset: f64,
    cond: f64,
}

fn solve_joint_pcs_ccr_core(
    pcs_obs: &[PcsObservation],
    ccr_obs: &[CcrObservation],
    metal_pos: Vec3,
    params: &CcrPhysicsParams,
    opts: &JointPcsCcrFitOptions,
) -> Result<JointLinearSolution, String> {
    let min_pcs = if opts.fit_pcs_offset { 7 } else { 6 };
    if pcs_obs.len() < min_pcs {
        return Err(format!(
            "joint PCS+CCR fit requires at least {min_pcs} PCS observations, got {}",
            pcs_obs.len()
        ));
    }
    if ccr_obs.is_empty() {
        return Err("joint PCS+CCR fit requires at least 1 CCR observation".to_string());
    }

    // Precompute un-weighted design rows for PCS and CCR
    let mut pcs_rows = Vec::with_capacity(pcs_obs.len());
    for o in pcs_obs {
        let r = v_dist(o.coord_angstrom, metal_pos);
        if r < MIN_METAL_NUCLEUS_DIST_A {
            return Err(format!(
                "unphysical metal-nucleus distance {r:.3} Å (< {MIN_METAL_NUCLEUS_DIST_A} Å) at PCS seq_id {}",
                o.seq_id
            ));
        }
        pcs_rows.push(pcs_design_row(o.coord_angstrom, metal_pos));
    }

    let mut ccr_rows: Vec<([f64; 5], f64)> = Vec::with_capacity(ccr_obs.len());
    for o in ccr_obs {
        let (pref, p2, aniso_row) = ccr_geometry_and_design_row(
            o.coord_s_angstrom,
            o.coord_i_angstrom,
            metal_pos,
            params,
            o.order_parameter_s2,
        )?;
        let dchi_part = match opts.tensor_model {
            CcrTensorModel::IsotropicChiOnly => [0.0; 5],
            CcrTensorModel::FullChiTensor => [
                pref * aniso_row[0],
                pref * aniso_row[1],
                pref * aniso_row[2],
                pref * aniso_row[3],
                pref * aniso_row[4],
            ],
        };
        ccr_rows.push((dchi_part, pref * p2));
    }

    let n_pcs = pcs_obs.len();
    let n_ccr = ccr_obs.len();
    let mut pcs_w: Vec<f64> = pcs_obs.iter().map(|o| 1.0 / o.uncertainty_ppm).collect();
    let mut ccr_w: Vec<f64> = ccr_obs.iter().map(|o| 1.0 / o.uncertainty_s_inv).collect();

    let solve_step = |pw: &[f64], cw: &[f64]| -> Result<JointLinearSolution, String> {
        if opts.fit_pcs_offset {
            let mut a7 = Vec::with_capacity(n_pcs + n_ccr);
            let mut b = Vec::with_capacity(n_pcs + n_ccr);
            for i in 0..n_pcs {
                let r = pcs_rows[i];
                let w = pw[i];
                a7.push([
                    r[0] * w,
                    r[1] * w,
                    r[2] * w,
                    r[3] * w,
                    r[4] * w,
                    0.0,
                    PCS_OFFSET_COL_SCALE * w,
                ]);
                b.push(pcs_obs[i].pcs_ppm * w);
            }
            for j in 0..n_ccr {
                let (dr, c_iso) = ccr_rows[j];
                let w = cw[j];
                a7.push([
                    dr[0] * w,
                    dr[1] * w,
                    dr[2] * w,
                    dr[3] * w,
                    dr[4] * w,
                    c_iso * w,
                    0.0,
                ]);
                b.push(ccr_obs[j].ccr_rate_s_inv * w);
            }
            let (x7, cond, vars7) = solve_equilibrated_svd::<7>(&a7, &b)?;
            Ok(JointLinearSolution {
                dchi_5: [x7[0], x7[1], x7[2], x7[3], x7[4]],
                chi_iso: x7[5],
                chi_iso_unc: vars7[5].max(0.0).sqrt(),
                pcs_offset: x7[6] * PCS_OFFSET_COL_SCALE,
                cond,
            })
        } else {
            let mut a6 = Vec::with_capacity(n_pcs + n_ccr);
            let mut b = Vec::with_capacity(n_pcs + n_ccr);
            for i in 0..n_pcs {
                let r = pcs_rows[i];
                let w = pw[i];
                a6.push([r[0] * w, r[1] * w, r[2] * w, r[3] * w, r[4] * w, 0.0]);
                b.push(pcs_obs[i].pcs_ppm * w);
            }
            for j in 0..n_ccr {
                let (dr, c_iso) = ccr_rows[j];
                let w = cw[j];
                a6.push([
                    dr[0] * w,
                    dr[1] * w,
                    dr[2] * w,
                    dr[3] * w,
                    dr[4] * w,
                    c_iso * w,
                ]);
                b.push(ccr_obs[j].ccr_rate_s_inv * w);
            }
            let (x6, cond, vars6) = solve_equilibrated_svd::<6>(&a6, &b)?;
            Ok(JointLinearSolution {
                dchi_5: [x6[0], x6[1], x6[2], x6[3], x6[4]],
                chi_iso: x6[5],
                chi_iso_unc: vars6[5].max(0.0).sqrt(),
                pcs_offset: 0.0,
                cond,
            })
        }
    };

    let mut sol = solve_step(&pcs_w, &ccr_w)?;

    if let Some(hk) = opts.huber_k {
        let k = hk.max(1.0e-6);
        for _ in 0..opts.max_irls_iters.max(1) {
            for i in 0..n_pcs {
                let pred: f64 = pcs_rows[i]
                    .iter()
                    .zip(&sol.dchi_5)
                    .map(|(&r, &c)| r * c)
                    .sum::<f64>()
                    + sol.pcs_offset;
                let z = (pcs_obs[i].pcs_ppm - pred).abs() / pcs_obs[i].uncertainty_ppm;
                let hw = if z <= k { 1.0 } else { (k / z).sqrt() };
                pcs_w[i] = hw / pcs_obs[i].uncertainty_ppm;
            }
            for j in 0..n_ccr {
                let (dr, c_iso) = ccr_rows[j];
                let pred: f64 = dr
                    .iter()
                    .zip(&sol.dchi_5)
                    .map(|(&r, &c)| r * c)
                    .sum::<f64>()
                    + c_iso * sol.chi_iso;
                let z = (ccr_obs[j].ccr_rate_s_inv - pred).abs() / ccr_obs[j].uncertainty_s_inv;
                let hw = if z <= k { 1.0 } else { (k / z).sqrt() };
                ccr_w[j] = hw / ccr_obs[j].uncertainty_s_inv;
            }
            let next = solve_step(&pcs_w, &ccr_w)?;
            let step_sq = next
                .dchi_5
                .iter()
                .zip(&sol.dchi_5)
                .map(|(&a, &b)| (a - b).powi(2))
                .sum::<f64>()
                + (next.chi_iso - sol.chi_iso).powi(2)
                + (next.pcs_offset - sol.pcs_offset).powi(2);
            sol = next;
            if step_sq.sqrt() < 1.0e-10 {
                break;
            }
        }
    }

    Ok(sol)
}

/// Jointly fit the 5-parameter traceless magnetic susceptibility anisotropy tensor
/// $\Delta\boldsymbol{\chi}$ and isotropic paramagnetic susceptibility $\chi_{\text{iso}}$
/// (in $10^{-32}\text{ m}^3$) from combined `PCS` and Curie–dipolar $\Gamma_{\text{CCR}}$
/// observations via column-equilibrated weighted thin Jacobi SVD (plus optional standardized Huber
/// IRLS), reporting Cornilescu $Q$-factors and residue-grouped $K$-fold holdout $Q_{\text{free}}$.
pub fn fit_joint_pcs_ccr(
    pcs_obs: &[PcsObservation],
    ccr_obs: &[CcrObservation],
    metal_pos_angstrom: Vec3,
    params: &CcrPhysicsParams,
    opts: &JointPcsCcrFitOptions,
) -> Result<JointPcsCcrFitResult, String> {
    params.validate()?;
    if !metal_pos_angstrom.iter().all(|v| v.is_finite()) {
        return Err(format!(
            "non-finite metal_pos_angstrom ({metal_pos_angstrom:?}) in fit_joint_pcs_ccr"
        ));
    }
    if let Some(hk) = opts.huber_k {
        if !hk.is_finite() || hk <= 0.0 {
            return Err(format!("invalid huber_k ({hk}): must be finite and > 0"));
        }
    }
    for o in pcs_obs {
        if !o.pcs_ppm.is_finite()
            || !o.coord_angstrom.iter().all(|v| v.is_finite())
            || !o.uncertainty_ppm.is_finite()
            || o.uncertainty_ppm <= 0.0
            || !o.coord_uncertainty_angstrom.is_finite()
            || o.coord_uncertainty_angstrom < 0.0
        {
            return Err(format!(
                "invalid PCS observation at seq_id {} in fit_joint_pcs_ccr",
                o.seq_id
            ));
        }
    }
    validate_ccr_observations(ccr_obs)?;

    let sol = solve_joint_pcs_ccr_core(pcs_obs, ccr_obs, metal_pos_angstrom, params, opts)?;
    let delta_chi = build_delta_chi_from_components(
        metal_pos_angstrom,
        sol.dchi_5,
        sol.pcs_offset,
        sol.cond,
        pcs_obs,
    );

    let pcs_y: Vec<f64> = pcs_obs.iter().map(|o| o.pcs_ppm).collect();
    let pcs_pred: Vec<f64> = pcs_obs
        .iter()
        .map(|o| predict_pcs_from_tensor(o.coord_angstrom, &delta_chi))
        .collect();
    let pcs_q_factor = cornilescu_q_factor(&pcs_y, &pcs_pred)?;
    let pcs_rmsd_ppm = (pcs_y
        .iter()
        .zip(&pcs_pred)
        .map(|(&y, &yh)| (y - yh).powi(2))
        .sum::<f64>()
        / (pcs_obs.len() as f64))
        .sqrt();

    let ccr_y: Vec<f64> = ccr_obs.iter().map(|o| o.ccr_rate_s_inv).collect();
    let mut ccr_pred = Vec::with_capacity(ccr_obs.len());
    let dchi_opt = match opts.tensor_model {
        CcrTensorModel::IsotropicChiOnly => None,
        CcrTensorModel::FullChiTensor => Some(&delta_chi),
    };
    for o in ccr_obs {
        ccr_pred.push(predict_curie_dd_ccr_s_inv(
            o.coord_s_angstrom,
            o.coord_i_angstrom,
            metal_pos_angstrom,
            sol.chi_iso,
            dchi_opt,
            params,
            o.order_parameter_s2,
        )?);
    }
    let ccr_q_factor = cornilescu_q_factor(&ccr_y, &ccr_pred)?;
    let ccr_rmsd_s_inv = (ccr_y
        .iter()
        .zip(&ccr_pred)
        .map(|(&y, &yh)| (y - yh).powi(2))
        .sum::<f64>()
        / (ccr_obs.len() as f64))
        .sqrt();

    let mut w_obs = Vec::with_capacity(pcs_obs.len() + ccr_obs.len());
    let mut w_pred = Vec::with_capacity(pcs_obs.len() + ccr_obs.len());
    for (o, &yh) in pcs_obs.iter().zip(&pcs_pred) {
        w_obs.push(o.pcs_ppm / o.uncertainty_ppm);
        w_pred.push(yh / o.uncertainty_ppm);
    }
    for (o, &yh) in ccr_obs.iter().zip(&ccr_pred) {
        w_obs.push(o.ccr_rate_s_inv / o.uncertainty_s_inv);
        w_pred.push(yh / o.uncertainty_s_inv);
    }
    let joint_weighted_q_factor = cornilescu_q_factor(&w_obs, &w_pred)?;

    // Deterministic residue-grouped K-fold holdout Q_free (PCS & CCR of same seq_id held out together)
    let mut pcs_q_free = None;
    let mut ccr_q_free = None;
    let mut joint_q_free = None;
    if opts.cv_folds >= 2 {
        let mut all_res = BTreeSet::new();
        for o in pcs_obs {
            all_res.insert(o.seq_id);
        }
        for o in ccr_obs {
            all_res.insert(o.seq_id);
        }
        let res_vec: Vec<i32> = all_res.into_iter().collect();
        let k_eff = opts.cv_folds.min(res_vec.len());
        if k_eff >= 2 {
            let mut fold_of: BTreeMap<i32, usize> = BTreeMap::new();
            for (rank, &sid) in res_vec.iter().enumerate() {
                fold_of.insert(sid, rank % k_eff);
            }
            let mut hold_pcs_y = Vec::new();
            let mut hold_pcs_p = Vec::new();
            let mut hold_ccr_y = Vec::new();
            let mut hold_ccr_p = Vec::new();
            let mut hold_w_y = Vec::new();
            let mut hold_w_p = Vec::new();

            for fold in 0..k_eff {
                let tr_pcs: Vec<PcsObservation> = pcs_obs
                    .iter()
                    .copied()
                    .filter(|o| fold_of.get(&o.seq_id).copied() != Some(fold))
                    .collect();
                let tr_ccr: Vec<CcrObservation> = ccr_obs
                    .iter()
                    .copied()
                    .filter(|o| fold_of.get(&o.seq_id).copied() != Some(fold))
                    .collect();
                if let Ok(tr_sol) =
                    solve_joint_pcs_ccr_core(&tr_pcs, &tr_ccr, metal_pos_angstrom, params, opts)
                {
                    let tr_dchi = build_delta_chi_from_components(
                        metal_pos_angstrom,
                        tr_sol.dchi_5,
                        tr_sol.pcs_offset,
                        tr_sol.cond,
                        &tr_pcs,
                    );
                    let tr_dchi_opt = match opts.tensor_model {
                        CcrTensorModel::IsotropicChiOnly => None,
                        CcrTensorModel::FullChiTensor => Some(&tr_dchi),
                    };
                    for o in pcs_obs {
                        if fold_of.get(&o.seq_id).copied() == Some(fold) {
                            let p = predict_pcs_from_tensor(o.coord_angstrom, &tr_dchi);
                            hold_pcs_y.push(o.pcs_ppm);
                            hold_pcs_p.push(p);
                            hold_w_y.push(o.pcs_ppm / o.uncertainty_ppm);
                            hold_w_p.push(p / o.uncertainty_ppm);
                        }
                    }
                    for o in ccr_obs {
                        if fold_of.get(&o.seq_id).copied() == Some(fold) {
                            if let Ok(p) = predict_curie_dd_ccr_s_inv(
                                o.coord_s_angstrom,
                                o.coord_i_angstrom,
                                metal_pos_angstrom,
                                tr_sol.chi_iso,
                                tr_dchi_opt,
                                params,
                                o.order_parameter_s2,
                            ) {
                                hold_ccr_y.push(o.ccr_rate_s_inv);
                                hold_ccr_p.push(p);
                                hold_w_y.push(o.ccr_rate_s_inv / o.uncertainty_s_inv);
                                hold_w_p.push(p / o.uncertainty_s_inv);
                            }
                        }
                    }
                }
            }

            if hold_pcs_y.len() >= 2 {
                pcs_q_free = cornilescu_q_factor(&hold_pcs_y, &hold_pcs_p).ok();
            }
            if hold_ccr_y.len() >= 2 {
                ccr_q_free = cornilescu_q_factor(&hold_ccr_y, &hold_ccr_p).ok();
            }
            if hold_w_y.len() >= 2 {
                joint_q_free = cornilescu_q_factor(&hold_w_y, &hold_w_p).ok();
            }
        }
    }

    let mut full_chi_matrix_mol = delta_chi.matrix_mol;
    full_chi_matrix_mol[0][0] += sol.chi_iso;
    full_chi_matrix_mol[1][1] += sol.chi_iso;
    full_chi_matrix_mol[2][2] += sol.chi_iso;

    let mu_eff_bohr = if sol.chi_iso >= 0.0 {
        mu_eff_bohr_from_chi_iso(sol.chi_iso, params.temperature_k).unwrap_or(f64::NAN)
    } else {
        f64::NAN
    };

    Ok(JointPcsCcrFitResult {
        delta_chi,
        chi_iso_1e32_m3: sol.chi_iso,
        chi_iso_uncertainty_1e32_m3: sol.chi_iso_unc,
        mu_eff_bohr,
        full_chi_matrix_mol,
        pcs_rmsd_ppm,
        pcs_q_factor,
        pcs_q_free,
        ccr_rmsd_s_inv,
        ccr_q_factor,
        ccr_q_free,
        joint_weighted_q_factor,
        joint_q_free,
        condition_number: sol.cond,
        n_pcs_obs: pcs_obs.len(),
        n_ccr_obs: ccr_obs.len(),
    })
}
