//! Pseudo-2D Diffusion-Ordered Spectroscopy (`DOSY`) gradient-attenuation parser, discrete
//! multi-exponential Stejskal–Tanner nonlinear least-squares with `AIC`/`BIC` selection and
//! bootstrap confidence intervals, opt-in Tikhonov-regularized Inverse Laplace Transform (`ILT`),
//! 2D `(\delta, \log_{10} D)` grid synthesis, and Stokes–Einstein hydrodynamics (`T14.4`).
//!
//! Provenance: Clean-room original implementation derived strictly from peer-reviewed NMR
//! spectroscopy, numerical optimization, and inverse-problem literature:
//! - Stejskal, E. O. & Tanner, J. E. (*J. Chem. Phys.* 1965, 42, 288–292 — pulsed-gradient
//!   spin-echo diffusion attenuation $I(g) = I_0 \exp(-D \gamma^2 g^2 \delta^2 (\Delta - \delta/3))$).
//! - Wu, D. H., Chen, A. D. & Johnson, C. S. Jr. (*J. Magn. Reson. A* 1995, 115, 260–264 —
//!   bipolar gradient pulse pair longitudinal-eddy-current delay `BPP-LED` sequence and
//!   inter-pulse $\tau / 2$ effective diffusion-time correction).
//! - Pelta, M. D., Morris, G. A., Stchedroff, M. J. & Hammond, S. J. (*Magn. Reson. Chem.*
//!   2002, 40, S147–S152 — shaped pulsed-field-gradient attenuation factors and convection-
//!   compensated / oneshot `DOSY` sequences).
//! - Johnson, C. S. Jr. (*Prog. Nucl. Magn. Reson. Spectrosc.* 1999, 34, 203–256 — `DOSY`
//!   processing and 2D Gaussian $(\delta, \log_{10} D)$ display synthesis).
//! - Provencher, S. W. (*Comput. Phys. Commun.* 1982, 27, 213–227 & 229–242 — constrained
//!   second-derivative Tikhonov regularization for inverse Laplace transforms).
//! - Lawson, C. L. & Hanson, R. J. (*Solving Least Squares Problems*, Prentice-Hall, 1974 —
//!   active-set non-negative least squares `NNLS`).
//! - Golub, G. H., Heath, M. & Wahba, G. (*Technometrics* 1979, 21, 215–223 — generalized
//!   cross-validation `GCV`) and Hansen, P. C. (*SIAM Rev.* 1992, 34, 561–580 — L-curve
//!   corner regularization selection).
//! - Einstein, A. (*Ann. Phys.* 1905, 322, 549–560 — Stokes–Einstein translational diffusion
//!   relation $R_h = k_B T / (6\pi \eta D)$) and Evans, R., Deng, Z., Rogerson, A. K.,
//!   McLachlan, A. S., Richards, J. J., Nilsson, M. & Morris, G. A. (*Angew. Chem. Int. Ed.*
//!   2013, 52, 3199–3202 — Stokes–Einstein diffusivity-to-molecular-weight estimation across
//!   deuterated NMR solvents).
//! - Published PFG-STE self-diffusion calibration standards (`D_ref` at $298.15\text{ K}$):
//!   Longsworth, L. G. (*J. Phys. Chem.* 1960, 64, 1914–1917);
//!   Holz, M. & Weingärtner, H. (*J. Magn. Reson.* 1991, 92, 115–125);
//!   Holz, M., Heil, S. R. & Sacco, A. (*Phys. Chem. Chem. Phys.* 2000, 2, 4740–4742);
//!   Wilkins, D. K. et al. (*Biochemistry* 1999, 38, 16424–16431).

use std::collections::HashMap;
use std::f64::consts::PI;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::rng::Rng;

// ============================================================================
// Fundamental Physical Constants (CODATA 2018)
// ============================================================================

/// Exact Boltzmann constant $k_B$ (`J / K`, CODATA 2018).
pub const BOLTZMANN_J_PER_K: f64 = 1.380_649e-23;

/// Exact Avogadro constant $N_A$ (`mol^-1`, CODATA 2018).
pub const AVOGADRO_PER_MOL: f64 = 6.022_140_76e23;

/// Proton ($^1\text{H}$) gyromagnetic ratio $\gamma_{^1\text{H}}$ (`rad s^-1 T^-1`).
pub const GAMMA_H1_RAD_S_T: f64 = 2.675_221_874_4e8;

/// Fluorine-19 ($^{19}\text{F}$) gyromagnetic ratio $\gamma_{^{19}\text{F}}$ (`rad s^-1 T^-1`).
pub const GAMMA_F19_RAD_S_T: f64 = 2.518_148e8;

/// Phosphorus-31 ($^{31}\text{P}$) gyromagnetic ratio $\gamma_{^{31}\text{P}}$ (`rad s^-1 T^-1`).
pub const GAMMA_P31_RAD_S_T: f64 = 1.084_192e8;

/// Carbon-13 ($^{13}\text{C}$) gyromagnetic ratio $\gamma_{^{13}\text{C}}$ (`rad s^-1 T^-1`).
pub const GAMMA_C13_RAD_S_T: f64 = 6.728_284e7;

/// Lithium-7 ($^7\text{Li}$) gyromagnetic ratio $\gamma_{^7\text{Li}}$ (`rad s^-1 T^-1`).
pub const GAMMA_LI7_RAD_S_T: f64 = 1.039_62e8;

/// Deuterium ($^2\text{H}$) gyromagnetic ratio $\gamma_{^2\text{H}}$ (`rad s^-1 T^-1`).
pub const GAMMA_H2_RAD_S_T: f64 = 4.106_627_91e7;

/// Unit conversion factor from `G / cm` (`Gauss / cm`) to SI `T / m` (`Tesla / m`):
/// $1\text{ G cm}^{-1} = 10^{-4}\text{ T} / 10^{-2}\text{ m} = 0.01\text{ T m}^{-1}$.
pub const GAUSS_PER_CM_TO_TESLA_PER_M: f64 = 0.01;

/// Explicit scientific caveat attached to every Inverse Laplace Transform (`ILT`) result and UI view.
pub const ILT_ARTIFACT_CAVEAT: &str = "Caveat: Numerical Inverse Laplace Transform (ILT) is an \
ill-posed problem. Spectral overlap between unresolved 1D multiplets shifts apparent decay rates \
toward intermediate pseudo-components, and under-regularized lambda values can split polydisperse \
or noisy single-species decays into spurious multi-modal spikes. Prefer BIC-selected discrete \
Stejskal-Tanner fits for well-defined mixtures and cross-check ILT peaks across the lambda sweep.";

// ============================================================================
// Error Type
// ============================================================================

/// Error returned by `DOSY` ingestion, fitting, `ILT`, or hydrodynamic estimation routines.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DosyError {
    pub message: String,
}

impl DosyError {
    pub fn new(msg: impl Into<String>) -> Self {
        Self {
            message: msg.into(),
        }
    }
}

impl fmt::Display for DosyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for DosyError {}

// ============================================================================
// 1. Gradient-List, Pulse-Sequence Attenuation `b`-Factor Builder & Parsers
// ============================================================================

/// Observed NMR nucleus determining the gyromagnetic ratio $\gamma$ (`rad s^-1 T^-1`).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum DosyNucleus {
    H1,
    F19,
    P31,
    C13,
    Li7,
    H2,
    Custom { gamma_rad_s_t: f64 },
}

impl DosyNucleus {
    /// Gyromagnetic ratio $|\gamma|$ in SI `rad s^-1 T^-1`.
    #[inline]
    pub fn gamma_rad_s_t(self) -> f64 {
        match self {
            Self::H1 => GAMMA_H1_RAD_S_T,
            Self::F19 => GAMMA_F19_RAD_S_T,
            Self::P31 => GAMMA_P31_RAD_S_T,
            Self::C13 => GAMMA_C13_RAD_S_T,
            Self::Li7 => GAMMA_LI7_RAD_S_T,
            Self::H2 => GAMMA_H2_RAD_S_T,
            Self::Custom { gamma_rad_s_t } => gamma_rad_s_t.abs(),
        }
    }

    /// Parse Bruker `NUC1` or Varian `tn` nucleus label (e.g. `"1H"`, `"H1"`, `"19F"`, `"31P"`).
    pub fn from_label(label: &str) -> Option<Self> {
        let cleaned: String = label
            .trim()
            .trim_matches(|c| c == '<' || c == '>' || c == '"' || c == '\'')
            .chars()
            .filter(|c| !c.is_whitespace())
            .collect::<String>()
            .to_ascii_uppercase();
        match cleaned.as_str() {
            "1H" | "H1" | "PROTON" => Some(Self::H1),
            "19F" | "F19" | "FLUORINE" => Some(Self::F19),
            "31P" | "P31" | "PHOSPHORUS" => Some(Self::P31),
            "13C" | "C13" | "CARBON" => Some(Self::C13),
            "7LI" | "LI7" => Some(Self::Li7),
            "2H" | "H2" | "D" => Some(Self::H2),
            _ => None,
        }
    }
}

/// Pulsed-field-gradient envelope shape controlling the effective gradient area $g_{\text{eff}}$.
///
/// Per Pelta et al. (*Magn. Reson. Chem.* 2002, 40, S147–S152), a half-sine bell pulse has peak-to-area
/// factor $2 / \pi$, giving $g_{\text{eff}} = (2 / \pi)\,g$, whereas a rectangular pulse has factor $1.0$.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum GradientShape {
    /// Ideal rectangular pulse (`g_eff = g`, shape factor `1.0`).
    Rectangular,
    /// Half-sine bell pulse (`SINE.100`, `g_eff = (2 / pi) * g`, shape factor `0.63661977...`).
    SineBell,
    /// Smoothed-square pulse (e.g. Bruker `SMSQ10.100` with area integral factor `0.90`).
    SmoothedSquare { area_factor: f64 },
    /// Arbitrary user-calibrated effective area factor (`g_eff = factor * g`).
    CustomFactor(f64),
}

impl GradientShape {
    /// Dimensionless ratio $g_{\text{eff}} / g_{\text{peak}}$ (`1.0` for rectangular, `2 / pi` for sine-bell).
    #[inline]
    pub fn effective_factor(self) -> f64 {
        match self {
            Self::Rectangular => 1.0,
            Self::SineBell => 2.0 / PI,
            Self::SmoothedSquare { area_factor } => area_factor,
            Self::CustomFactor(f) => f,
        }
    }

    /// Infer [`GradientShape`] from a Bruker `GPNAM` or Varian `gshape` token.
    pub fn from_shape_name(name: &str) -> Self {
        let up = name
            .trim()
            .trim_matches(|c| c == '<' || c == '>' || c == '"' || c == '\'')
            .to_ascii_uppercase();
        if up.is_empty() || up.starts_with("RECT") || up.starts_with("SQUARE") || up == "R" {
            Self::Rectangular
        } else if up.starts_with("SIN") || up == "S" {
            Self::SineBell
        } else if up.starts_with("SMSQ") {
            Self::SmoothedSquare { area_factor: 0.90 }
        } else {
            Self::Rectangular
        }
    }
}

/// Diffusion pulse-sequence family determining $\Delta_{\text{eff}}$ in the Stejskal–Tanner exponent.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DosySequenceType {
    /// Standard spin-echo or monopolar stimulated-echo (`STE`, Stejskal & Tanner 1965):
    /// $\Delta_{\text{eff}} = \Delta - \delta / 3$.
    Ste,
    /// Bipolar gradient pulse pair stimulated-echo / longitudinal eddy-current delay
    /// (`BPP-LED`, Wu, Chen & Johnson *J. Magn. Reson. A* 1995, 115, 260–264):
    /// $\Delta_{\text{eff}} = \Delta - \delta / 3 - \tau / 2$, where $\delta$ is the total gradient
    /// encoding duration per defocusing/refocusing pair ($\delta = 2 p_{30}$ for bipolar lobes of width $p_{30}$)
    /// and $\tau$ is the inter-lobe bipolar delay.
    BppLed,
}

/// Interpretation of Bruker `P30` pulse length relative to total diffusion gradient width $\delta$.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BrukerP30Mode {
    /// Automatically set $\delta = 2 \times \text{P30}$ for bipolar sequences (`*bp*` / `BppLed`)
    /// and $\delta = \text{P30}$ for monopolar `STE`/`LED` sequences.
    Auto,
    /// Bruker `P30` represents half of a bipolar pair ($\delta = 2 \times \text{P30}$).
    HalfDeltaBipolar,
    /// Bruker `P30` represents the full monopolar gradient pulse duration ($\delta = \text{P30}$).
    FullDeltaMonopolar,
}

/// Fully resolved experimental parameters and computed Stejskal–Tanner attenuation factors $b_m$ (`s / m^2`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DosyExperimentParams {
    /// Nucleus determining $\gamma$ (`rad s^-1 T^-1`).
    pub nucleus: DosyNucleus,
    /// Diffusion pulse sequence family (`STE` vs `BPP-LED`).
    pub sequence_type: DosySequenceType,
    /// Gradient envelope shape (`Rectangular`, `SineBell`, etc.).
    pub gradient_shape: GradientShape,
    /// Diffusion delay $\Delta$ (`s`, "big delta").
    pub big_delta_s: f64,
    /// Total diffusion-encoding gradient width $\delta$ (`s`, "little delta").
    pub little_delta_s: f64,
    /// Bipolar inter-pulse delay $\tau$ (`s`, used in `BPP-LED`, `0.0` for monopolar `STE`).
    pub tau_s: f64,
    /// Peak gradient amplitudes in `G / cm`.
    pub gradients_g_per_cm: Vec<f64>,
    /// Effective gradient amplitudes $g_{\text{eff}}$ in SI `T / m`.
    pub effective_gradients_t_per_m: Vec<f64>,
    /// Computed Stejskal–Tanner / Wu–Johnson attenuation factors $b_m$ in SI `s / m^2`.
    pub b_values_s_m2: Vec<f64>,
    /// Optional source pulse-program identifier (`PULPROG` or `seqfil`).
    pub pulse_program: Option<String>,
    /// Pelta et al. (2002) Oneshot bipolar gradient unbalancing factor $\alpha \in [0, 1)$
    /// (`0.0` for standard balanced `STE` / `BPP-LED`, typically `0.2` for `Oneshot` sequences).
    #[serde(default)]
    pub oneshot_alpha: f64,
}

impl DosyExperimentParams {
    /// Construct and validate [`DosyExperimentParams`] from peak gradient strengths in `G / cm`,
    /// computing $b_m = (\gamma \, g_{\text{eff},m} \, \delta)^2 \, \Delta_{\text{eff}}$ in SI `s / m^2`.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        nucleus: DosyNucleus,
        sequence_type: DosySequenceType,
        gradient_shape: GradientShape,
        big_delta_s: f64,
        little_delta_s: f64,
        tau_s: f64,
        gradients_g_per_cm: Vec<f64>,
        pulse_program: Option<String>,
    ) -> Result<Self, DosyError> {
        Self::build_with_oneshot_alpha(
            nucleus,
            sequence_type,
            gradient_shape,
            big_delta_s,
            little_delta_s,
            tau_s,
            0.0,
            gradients_g_per_cm,
            pulse_program,
        )
    }

    /// Construct and validate [`DosyExperimentParams`] with optional Pelta et al. (2002) Oneshot
    /// bipolar unbalancing factor $\alpha \in [0, 1)$ (`F29`).
    #[allow(clippy::too_many_arguments)]
    pub fn build_with_oneshot_alpha(
        nucleus: DosyNucleus,
        sequence_type: DosySequenceType,
        gradient_shape: GradientShape,
        big_delta_s: f64,
        little_delta_s: f64,
        tau_s: f64,
        oneshot_alpha: f64,
        gradients_g_per_cm: Vec<f64>,
        pulse_program: Option<String>,
    ) -> Result<Self, DosyError> {
        if gradients_g_per_cm.is_empty() {
            return Err(DosyError::new("DOSY gradient list must not be empty"));
        }
        let gamma = nucleus.gamma_rad_s_t();
        if !gamma.is_finite() || gamma <= 0.0 {
            return Err(DosyError::new(format!(
                "invalid gyromagnetic ratio gamma = {gamma}"
            )));
        }
        if !little_delta_s.is_finite() || little_delta_s <= 0.0 {
            return Err(DosyError::new(format!(
                "invalid gradient pulse duration delta = {little_delta_s} s (must be > 0)"
            )));
        }
        if !big_delta_s.is_finite() || big_delta_s <= 0.0 {
            return Err(DosyError::new(format!(
                "invalid diffusion delay Delta = {big_delta_s} s (must be > 0)"
            )));
        }
        if !tau_s.is_finite() || tau_s < 0.0 {
            return Err(DosyError::new(format!(
                "invalid bipolar delay tau = {tau_s} s (must be >= 0)"
            )));
        }
        if !oneshot_alpha.is_finite() || !(0.0..1.0).contains(&oneshot_alpha) {
            return Err(DosyError::new(format!(
                "invalid Oneshot unbalancing factor alpha = {oneshot_alpha} (must be in [0.0, 1.0))"
            )));
        }
        let shape_factor = gradient_shape.effective_factor();
        if !shape_factor.is_finite() || shape_factor <= 0.0 {
            return Err(DosyError::new(format!(
                "invalid gradient shape factor = {shape_factor} (must be > 0)"
            )));
        }

        let delta_eff = if oneshot_alpha > 0.0 && sequence_type == DosySequenceType::BppLed {
            effective_diffusion_time_oneshot_s(
                gradient_shape,
                big_delta_s,
                little_delta_s,
                tau_s,
                oneshot_alpha,
            )?
        } else {
            effective_diffusion_time_s(sequence_type, big_delta_s, little_delta_s, tau_s)?
        };

        let mut effective_gradients_t_per_m = Vec::with_capacity(gradients_g_per_cm.len());
        let mut b_values_s_m2 = Vec::with_capacity(gradients_g_per_cm.len());

        for (idx, &g_gcm) in gradients_g_per_cm.iter().enumerate() {
            if !g_gcm.is_finite() {
                return Err(DosyError::new(format!(
                    "non-finite gradient strength at index {idx}: {g_gcm}"
                )));
            }
            let g_t_per_m = g_gcm * GAUSS_PER_CM_TO_TESLA_PER_M;
            let g_eff = g_t_per_m * shape_factor;
            let q = gamma * g_eff * little_delta_s;
            let b = q * q * delta_eff;
            effective_gradients_t_per_m.push(g_eff);
            b_values_s_m2.push(b);
        }

        Ok(Self {
            nucleus,
            sequence_type,
            gradient_shape,
            big_delta_s,
            little_delta_s,
            tau_s,
            gradients_g_per_cm,
            effective_gradients_t_per_m,
            b_values_s_m2,
            pulse_program,
            oneshot_alpha,
        })
    }

    /// Effective diffusion delay $\Delta_{\text{eff}}$ (`s`).
    pub fn effective_diffusion_time_s(&self) -> Result<f64, DosyError> {
        if self.oneshot_alpha > 0.0 && self.sequence_type == DosySequenceType::BppLed {
            effective_diffusion_time_oneshot_s(
                self.gradient_shape,
                self.big_delta_s,
                self.little_delta_s,
                self.tau_s,
                self.oneshot_alpha,
            )
        } else {
            effective_diffusion_time_s(
                self.sequence_type,
                self.big_delta_s,
                self.little_delta_s,
                self.tau_s,
            )
        }
    }
}

/// Compute Pelta et al. (*Magn. Reson. Chem.* **2002**, 40, S147–S152) effective diffusion delay
/// $\Delta_{\text{eff}}(\alpha)$ (`s`) for unbalanced bipolar `Oneshot` sequences (`0 <= alpha < 1`):
/// - Rectangular gradient lobes: $\Delta - \delta (2 - \alpha^2) / 6 - \tau (1 - \alpha^2) / 2$
/// - Half-sine (`SineBell`) / smoothed lobes: $\Delta - \delta (5 - 3\alpha^2) / 16 - \tau (1 - \alpha^2) / 2$
///   (reduces cleanly to balanced `BPP-LED` $\Delta - \delta / 3 - \tau / 2$ when $\alpha = 0$).
pub fn effective_diffusion_time_oneshot_s(
    gradient_shape: GradientShape,
    big_delta_s: f64,
    little_delta_s: f64,
    tau_s: f64,
    alpha: f64,
) -> Result<f64, DosyError> {
    if !alpha.is_finite() || !(0.0..1.0).contains(&alpha) {
        return Err(DosyError::new(format!(
            "invalid Oneshot alpha = {alpha} (expected 0.0 <= alpha < 1.0)"
        )));
    }
    if alpha == 0.0 {
        return effective_diffusion_time_s(
            DosySequenceType::BppLed,
            big_delta_s,
            little_delta_s,
            tau_s,
        );
    }
    let a2 = alpha * alpha;
    let lobe_coeff = match gradient_shape {
        GradientShape::SineBell => (5.0 - 3.0 * a2) / 16.0,
        GradientShape::Rectangular
        | GradientShape::SmoothedSquare { .. }
        | GradientShape::CustomFactor(_) => (2.0 - a2) / 6.0,
    };
    let delta_eff = big_delta_s - little_delta_s * lobe_coeff - 0.5 * tau_s * (1.0 - a2);
    if !delta_eff.is_finite() || delta_eff <= 0.0 {
        return Err(DosyError::new(format!(
            "non-positive Oneshot effective diffusion delay Delta_eff = {delta_eff:.6e} s \
             (Delta = {big_delta_s:.6e} s, delta = {little_delta_s:.6e} s, tau = {tau_s:.6e} s, alpha = {alpha})"
        )));
    }
    Ok(delta_eff)
}

/// Compute effective diffusion delay $\Delta_{\text{eff}}$ (`s`) for `STE` or `BPP-LED`:
/// - `STE` (Stejskal & Tanner 1965): $\Delta - \delta / 3$
/// - `BPP-LED` (Wu, Chen & Johnson 1995): $\Delta - \delta / 3 - \tau / 2$
pub fn effective_diffusion_time_s(
    sequence_type: DosySequenceType,
    big_delta_s: f64,
    little_delta_s: f64,
    tau_s: f64,
) -> Result<f64, DosyError> {
    let delta_eff = match sequence_type {
        DosySequenceType::Ste => big_delta_s - little_delta_s / 3.0,
        DosySequenceType::BppLed => big_delta_s - little_delta_s / 3.0 - 0.5 * tau_s,
    };
    if !delta_eff.is_finite() || delta_eff <= 0.0 {
        return Err(DosyError::new(format!(
            "non-positive effective diffusion delay Delta_eff = {delta_eff:.6e} s \
             (Delta = {big_delta_s:.6e} s, delta = {little_delta_s:.6e} s, tau = {tau_s:.6e} s)"
        )));
    }
    Ok(delta_eff)
}

/// Convenience helper computing Stejskal–Tanner / Wu–Johnson $b$-factors (`s / m^2`) directly
/// from gradient strengths in `G / cm`.
pub fn compute_b_values_s_m2(
    gradients_g_per_cm: &[f64],
    nucleus: DosyNucleus,
    sequence_type: DosySequenceType,
    gradient_shape: GradientShape,
    big_delta_s: f64,
    little_delta_s: f64,
    tau_s: f64,
) -> Result<Vec<f64>, DosyError> {
    Ok(DosyExperimentParams::build(
        nucleus,
        sequence_type,
        gradient_shape,
        big_delta_s,
        little_delta_s,
        tau_s,
        gradients_g_per_cm.to_vec(),
        None,
    )?
    .b_values_s_m2)
}

/// Optional caller overrides when parsing Bruker `difflist` + `acqu`/`acqus` diffusion metadata.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BrukerDosyOverrides {
    pub nucleus: Option<DosyNucleus>,
    pub sequence_type: Option<DosySequenceType>,
    pub gradient_shape: Option<GradientShape>,
    pub p30_mode: Option<BrukerP30Mode>,
    pub big_delta_s: Option<f64>,
    pub little_delta_s: Option<f64>,
    pub tau_s: Option<f64>,
    /// Index in Bruker `GPNAM` array for the diffusion gradient pulse (defaults to `6` when non-empty,
    /// falling back to the first non-empty diffusion `GPNAM` entry).
    pub gpnam_index: Option<usize>,
    /// Optional Pelta et al. (2002) Oneshot bipolar gradient unbalancing factor $\alpha \in [0, 1)$.
    #[serde(default)]
    pub oneshot_alpha: Option<f64>,
}

/// Parse a Bruker `difflist` file containing gradient strengths in `G / cm`.
pub fn parse_bruker_difflist(difflist_text: &str) -> Result<Vec<f64>, DosyError> {
    let mut vals = Vec::new();
    for (line_no, raw) in difflist_text.lines().enumerate() {
        let trimmed = raw.trim();
        if trimmed.is_empty()
            || trimmed.starts_with('#')
            || trimmed.starts_with("//")
            || trimmed.starts_with(';')
        {
            continue;
        }
        // Strip inline `#` comment if present
        let active = trimmed.split('#').next().unwrap_or(trimmed).trim();
        for tok in active.split_whitespace() {
            let v: f64 = tok.parse().map_err(|_| {
                DosyError::new(format!(
                    "invalid numeric token '{tok}' in Bruker difflist at line {}",
                    line_no + 1
                ))
            })?;
            if !v.is_finite() {
                return Err(DosyError::new(format!(
                    "non-finite gradient value in Bruker difflist at line {}",
                    line_no + 1
                )));
            }
            vals.push(v);
        }
    }
    if vals.is_empty() {
        return Err(DosyError::new(
            "Bruker difflist contains no gradient values",
        ));
    }
    Ok(vals)
}

#[derive(Default)]
struct BrukerJcampParsed {
    scalars: HashMap<String, String>,
    num_arrays: HashMap<String, Vec<f64>>,
    str_arrays: HashMap<String, Vec<String>>,
}

fn parse_bruker_jcamp_subset(text: &str) -> BrukerJcampParsed {
    let mut out = BrukerJcampParsed::default();
    let lines: Vec<&str> = text.lines().collect();
    let mut i = 0usize;
    while i < lines.len() {
        let line = lines[i].trim();
        i += 1;
        if !line.starts_with("##$") {
            continue;
        }
        let Some((lhs, rhs)) = line[3..].split_once('=') else {
            continue;
        };
        let key = lhs.trim().to_ascii_uppercase();
        let rhs_trim = rhs.trim();

        if rhs_trim.starts_with('(') && rhs_trim.contains("..") {
            // JCAMP array header `##$D= (0..63)` or `##$GPNAM= (0..31)`
            let mut body = String::new();
            if let Some(after_paren) = rhs_trim.split_once(')').map(|(_, tail)| tail.trim()) {
                if !after_paren.is_empty() {
                    body.push_str(after_paren);
                    body.push(' ');
                }
            }
            while i < lines.len() {
                let peek = lines[i].trim();
                if peek.starts_with("##") || peek.starts_with("$$") {
                    break;
                }
                body.push_str(peek);
                body.push(' ');
                i += 1;
            }
            if body.contains('<') {
                let mut items = Vec::new();
                let mut rest = body.as_str();
                while let Some(open_idx) = rest.find('<') {
                    let after = &rest[open_idx + 1..];
                    if let Some(close_idx) = after.find('>') {
                        items.push(after[..close_idx].trim().to_string());
                        rest = &after[close_idx + 1..];
                    } else {
                        break;
                    }
                }
                out.str_arrays.insert(key, items);
            } else {
                let nums: Vec<f64> = body
                    .split_whitespace()
                    .filter_map(|t| t.parse::<f64>().ok())
                    .collect();
                out.num_arrays.insert(key, nums);
            }
        } else {
            let cleaned = rhs_trim
                .trim_matches(|c| c == '<' || c == '>' || c == '"' || c == '\'')
                .trim()
                .to_string();
            out.scalars.insert(key, cleaned);
        }
    }
    out
}

/// Parse Bruker `acqu` / `acqus` JCAMP-DX parameters (`D20` = $\Delta$, `P30` = $\delta/2$ or $\delta$,
/// `D16`, `PULPROG`, `NUC1`, `GPNAM`) together with `difflist` (`G / cm`) to construct [`DosyExperimentParams`].
pub fn parse_bruker_dosy(
    acqus_text: &str,
    difflist_text: &str,
    overrides: &BrukerDosyOverrides,
) -> Result<DosyExperimentParams, DosyError> {
    let gradients_g_per_cm = parse_bruker_difflist(difflist_text)?;
    let jc = parse_bruker_jcamp_subset(acqus_text);

    let pulprog = jc.scalars.get("PULPROG").cloned().filter(|s| !s.is_empty());
    let pulprog_low = pulprog.as_deref().unwrap_or("").to_ascii_lowercase();

    let nucleus = if let Some(nuc) = overrides.nucleus {
        nuc
    } else if let Some(nuc_str) = jc.scalars.get("NUC1") {
        DosyNucleus::from_label(nuc_str).unwrap_or(DosyNucleus::H1)
    } else {
        DosyNucleus::H1
    };

    let is_oneshot = pulprog_low.contains("oneshot");

    let sequence_type = if let Some(seq) = overrides.sequence_type {
        seq
    } else if is_oneshot
        || pulprog_low.contains("bpp")
        || (pulprog_low.contains("bpgp") && pulprog_low.contains("led"))
        || pulprog_low.contains("dstebpgp")
    {
        DosySequenceType::BppLed
    } else {
        DosySequenceType::Ste
    };

    // Extract D20 (Delta in seconds)
    let big_delta_s = if let Some(d) = overrides.big_delta_s {
        d
    } else if let Some(d20) = jc.num_arrays.get("D").and_then(|arr| arr.get(20)).copied() {
        d20
    } else if let Some(d20_s) = jc
        .scalars
        .get("D20")
        .or_else(|| jc.scalars.get("D[20]"))
        .and_then(|s| s.parse::<f64>().ok())
    {
        d20_s
    } else {
        return Err(DosyError::new(
            "missing Bruker diffusion delay D20 (Delta) in acqus/acqu",
        ));
    };

    // Extract P30 (in microseconds in Bruker JCAMP `P` array) and convert to total little_delta_s
    let little_delta_s = if let Some(ld) = overrides.little_delta_s {
        ld
    } else {
        let p30_us = jc
            .num_arrays
            .get("P")
            .and_then(|arr| arr.get(30))
            .copied()
            .or_else(|| {
                jc.scalars
                    .get("P30")
                    .or_else(|| jc.scalars.get("P[30]"))
                    .and_then(|s| s.parse::<f64>().ok())
            })
            .ok_or_else(|| {
                DosyError::new("missing Bruker gradient pulse length P30 in acqus/acqu")
            })?;
        let p30_s = p30_us * 1.0e-6;
        let p30_mode = overrides.p30_mode.unwrap_or(BrukerP30Mode::Auto);
        let is_bipolar = match p30_mode {
            BrukerP30Mode::HalfDeltaBipolar => true,
            BrukerP30Mode::FullDeltaMonopolar => false,
            BrukerP30Mode::Auto => {
                if pulprog_low.contains("bp") || is_oneshot {
                    true
                } else if !pulprog_low.is_empty() {
                    false
                } else {
                    sequence_type == DosySequenceType::BppLed
                }
            }
        };
        if is_bipolar {
            2.0 * p30_s
        } else {
            p30_s
        }
    };

    // Extract Pelta et al. (2002) Oneshot bipolar unbalancing factor `alpha` (`0 <= alpha < 1`).
    // Supports Manchester Oneshot (`CNST[14] = alpha` with `CNST[15] = 1 + alpha`, `CNST[16] = 1 - alpha`)
    // and Bruker `CNST[19]` (`0 < CNST[19] < 1`, ignoring Bruker's unused `1.0` default fill),
    // falling back cleanly to `alpha = 0.0` (standard balanced bipolar `2 * P30`).
    let get_cnst = |idx: usize| -> Option<f64> {
        jc.num_arrays
            .get("CNST")
            .and_then(|arr| arr.get(idx))
            .copied()
            .or_else(|| {
                let k1 = format!("CNST{idx}");
                let k2 = format!("CNST[{idx}]");
                jc.scalars
                    .get(&k1)
                    .or_else(|| jc.scalars.get(&k2))
                    .and_then(|s| s.parse::<f64>().ok())
            })
            .filter(|v| v.is_finite())
    };
    let oneshot_alpha = if let Some(a) = overrides.oneshot_alpha {
        a
    } else if is_oneshot {
        let c14 = get_cnst(14);
        let c15 = get_cnst(15);
        let c16 = get_cnst(16);
        let c19 = get_cnst(19);
        if let (Some(a), Some(p), Some(m)) = (c14, c15, c16) {
            if (0.0..1.0).contains(&a)
                && (p - (1.0 + a)).abs() <= 1.0e-3
                && (m - (1.0 - a)).abs() <= 1.0e-3
            {
                a
            } else if let Some(a19) = c19.filter(|&v| v > 0.0 && v < 1.0) {
                a19
            } else if (0.0..1.0).contains(&a) {
                a
            } else {
                0.0
            }
        } else if let Some(a19) = c19.filter(|&v| v > 0.0 && v < 1.0) {
            a19
        } else {
            c14.filter(|&v| v > 0.0 && v < 1.0).unwrap_or(0.0)
        }
    } else {
        0.0
    };

    // Extract bipolar inter-pulse delay tau (for Oneshot, `CNST[17] = 2*P1 + D16` when present, else `D16`).
    let tau_s = if let Some(t) = overrides.tau_s {
        t
    } else if sequence_type == DosySequenceType::BppLed {
        let d16_s = jc
            .num_arrays
            .get("D")
            .and_then(|arr| arr.get(16))
            .copied()
            .or_else(|| {
                jc.scalars
                    .get("D16")
                    .or_else(|| jc.scalars.get("D[16]"))
                    .and_then(|s| s.parse::<f64>().ok())
            })
            .unwrap_or(0.0);
        if is_oneshot {
            let p1_s = jc
                .num_arrays
                .get("P")
                .and_then(|arr| arr.get(1))
                .copied()
                .or_else(|| {
                    jc.scalars
                        .get("P1")
                        .or_else(|| jc.scalars.get("P[1]"))
                        .and_then(|s| s.parse::<f64>().ok())
                })
                .map(|us| us.max(0.0) * 1.0e-6)
                .unwrap_or(0.0);
            if let Some(c17) = get_cnst(17).filter(|&v| v > 0.0 && v < 0.05) {
                if p1_s > 0.0 && (c17 - (2.0 * p1_s + d16_s)).abs() <= 1.0e-4 {
                    c17
                } else {
                    2.0 * p1_s + d16_s
                }
            } else {
                2.0 * p1_s + d16_s
            }
        } else {
            d16_s
        }
    } else {
        0.0
    };

    // Extract gradient shape from GPNAM (for Oneshot, diffusion pulses use `GP1` -> `GPNAM[1]`,
    // falling back to `GPNAM[6]` or first non-empty shape).
    let gradient_shape = if let Some(sh) = overrides.gradient_shape {
        sh
    } else {
        let gpnam_str = if let Some(arr) = jc.str_arrays.get("GPNAM") {
            let pref_idx = overrides
                .gpnam_index
                .unwrap_or(if is_oneshot { 1 } else { 6 });
            arr.get(pref_idx)
                .filter(|s| !s.is_empty())
                .or_else(|| arr.get(6).filter(|s| !s.is_empty()))
                .or_else(|| arr.get(1).filter(|s| !s.is_empty()))
                .or_else(|| arr.iter().find(|s| !s.is_empty()))
                .cloned()
        } else {
            jc.scalars
                .get("GPNAM6")
                .or_else(|| jc.scalars.get("GPNAM[6]"))
                .or_else(|| jc.scalars.get("GPNAM1"))
                .or_else(|| jc.scalars.get("GPNAM[1]"))
                .or_else(|| jc.scalars.get("GPNAM"))
                .cloned()
        };
        gpnam_str
            .as_deref()
            .map(GradientShape::from_shape_name)
            .unwrap_or(GradientShape::Rectangular)
    };

    DosyExperimentParams::build_with_oneshot_alpha(
        nucleus,
        sequence_type,
        gradient_shape,
        big_delta_s,
        little_delta_s,
        tau_s,
        oneshot_alpha,
        gradients_g_per_cm,
        pulprog,
    )
}

fn find_file_case_insensitive(dir: &Path, target: &str) -> Option<PathBuf> {
    let entries = fs::read_dir(dir).ok()?;
    let mut ci_hit = None;
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        if let Some(name) = entry.file_name().to_str() {
            if name == target {
                return Some(path);
            }
            if ci_hit.is_none() && name.eq_ignore_ascii_case(target) {
                ci_hit = Some(path);
            }
        }
    }
    ci_hit
}

/// Read Bruker `difflist` and `acqus` (or `acqu`) from an experiment directory on disk.
pub fn read_bruker_dosy_dir(
    exp_dir: impl AsRef<Path>,
    overrides: &BrukerDosyOverrides,
) -> Result<DosyExperimentParams, DosyError> {
    let dir = exp_dir.as_ref();
    let difflist_path = find_file_case_insensitive(dir, "difflist")
        .ok_or_else(|| DosyError::new(format!("missing Bruker difflist in {}", dir.display())))?;
    let acqus_path = find_file_case_insensitive(dir, "acqus")
        .or_else(|| find_file_case_insensitive(dir, "acqu"))
        .ok_or_else(|| DosyError::new(format!("missing Bruker acqus/acqu in {}", dir.display())))?;
    let difflist_text = fs::read_to_string(&difflist_path)
        .map_err(|e| DosyError::new(format!("failed to read difflist: {e}")))?;
    let acqus_text = fs::read_to_string(&acqus_path)
        .map_err(|e| DosyError::new(format!("failed to read acqus: {e}")))?;
    parse_bruker_dosy(&acqus_text, &difflist_text, overrides)
}

/// Optional caller overrides and DAC calibration when parsing Varian/Agilent `procpar` diffusion metadata.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct VarianDosyOverrides {
    /// Spectrometer DAC-to-`G/cm` calibration factor (`G / (cm * DAC)`). Overrides `gcal` in `procpar`.
    pub dac_to_g_per_cm: Option<f64>,
    pub nucleus: Option<DosyNucleus>,
    pub sequence_type: Option<DosySequenceType>,
    pub gradient_shape: Option<GradientShape>,
    pub big_delta_s: Option<f64>,
    pub little_delta_s: Option<f64>,
    pub tau_s: Option<f64>,
}

enum VarianParamVal {
    Numbers(Vec<f64>),
    Strings(Vec<String>),
}

const MAX_VARIAN_ARRAY_LEN: usize = 65_536;

fn unquote_varian(s: &str) -> String {
    let t = s.trim();
    if t.len() >= 2
        && ((t.starts_with('"') && t.ends_with('"')) || (t.starts_with('\'') && t.ends_with('\'')))
    {
        t[1..t.len() - 1].to_string()
    } else {
        t.to_string()
    }
}

fn parse_varian_procpar_map(text: &str) -> Result<HashMap<String, VarianParamVal>, DosyError> {
    let mut map = HashMap::new();
    let lines: Vec<&str> = text.lines().collect();
    let mut idx = 0usize;

    while idx < lines.len() {
        let hdr = lines[idx].trim();
        idx += 1;
        if hdr.is_empty() || hdr.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = hdr.split_whitespace().collect();
        if parts.len() < 11 {
            continue;
        }
        let name = parts[0].to_string();
        let Ok(basictype) = parts[2].parse::<i32>() else {
            continue;
        };
        if idx >= lines.len() {
            break;
        }

        if basictype == 2 {
            let first_val = lines[idx].trim();
            idx += 1;
            let (cnt_str, rem) = first_val
                .split_once(char::is_whitespace)
                .unwrap_or((first_val, ""));
            let count: usize = cnt_str.parse().unwrap_or(1);
            if count > MAX_VARIAN_ARRAY_LEN {
                return Err(DosyError::new(format!(
                    "Varian procpar parameter '{name}' count ({count}) exceeds limit"
                )));
            }
            let mut strs = Vec::with_capacity(count.min(256));
            if !rem.trim().is_empty() && count > 0 {
                strs.push(unquote_varian(rem.trim()));
            }
            while strs.len() < count && idx < lines.len() {
                strs.push(unquote_varian(lines[idx]));
                idx += 1;
            }
            map.insert(name, VarianParamVal::Strings(strs));
        } else {
            let first_toks: Vec<&str> = lines[idx].split_whitespace().collect();
            idx += 1;
            let mut nums = Vec::new();
            if let Some(cnt_tok) = first_toks.first() {
                let count: usize = cnt_tok.parse().unwrap_or(0);
                if count > MAX_VARIAN_ARRAY_LEN {
                    return Err(DosyError::new(format!(
                        "Varian procpar parameter '{name}' count ({count}) exceeds limit"
                    )));
                }
                nums.reserve(count.min(256));
                for &t in &first_toks[1..] {
                    if nums.len() >= count {
                        break;
                    }
                    if let Ok(v) = t.parse::<f64>() {
                        nums.push(v);
                    }
                }
                while nums.len() < count && idx < lines.len() {
                    let more: Vec<&str> = lines[idx].split_whitespace().collect();
                    idx += 1;
                    if more.is_empty() {
                        break;
                    }
                    for &t in &more {
                        if nums.len() >= count {
                            break;
                        }
                        if let Ok(v) = t.parse::<f64>() {
                            nums.push(v);
                        }
                    }
                }
            }
            map.insert(name, VarianParamVal::Numbers(nums));
        }

        // Skip enumerated values trailer line (`0` or `<n_enum> ...`)
        if idx < lines.len() {
            idx += 1;
        }
    }

    Ok(map)
}

/// Parse Varian/Agilent `procpar` diffusion parameters (`gzlvl1`, `gt1`, `del`, and `gcal` /
/// `DAC-to-G/cm` calibration) into [`DosyExperimentParams`].
pub fn parse_varian_procpar_dosy(
    procpar_text: &str,
    overrides: &VarianDosyOverrides,
) -> Result<DosyExperimentParams, DosyError> {
    let pp = parse_varian_procpar_map(procpar_text)?;

    let get_nums = |k: &str| -> Option<&[f64]> {
        match pp.get(k)? {
            VarianParamVal::Numbers(v) if !v.is_empty() => Some(v.as_slice()),
            _ => None,
        }
    };
    let get_first_str = |k: &str| -> Option<&str> {
        match pp.get(k)? {
            VarianParamVal::Strings(v) => v.first().map(String::as_str).filter(|s| !s.is_empty()),
            _ => None,
        }
    };

    let gzlvl1 = get_nums("gzlvl1")
        .ok_or_else(|| DosyError::new("missing Varian `gzlvl1` gradient DAC array in procpar"))?;

    let dac_cal = overrides
        .dac_to_g_per_cm
        .or_else(|| get_nums("gcal").map(|v| v[0]))
        .or_else(|| get_nums("gcal_").map(|v| v[0]))
        .ok_or_else(|| {
            DosyError::new(
                "missing Varian DAC-to-G/cm calibration (`gcal` in procpar or `dac_to_g_per_cm` override)",
            )
        })?;
    if !dac_cal.is_finite() || dac_cal <= 0.0 {
        return Err(DosyError::new(format!(
            "invalid Varian DAC-to-G/cm calibration factor: {dac_cal}"
        )));
    }

    let gradients_g_per_cm: Vec<f64> = gzlvl1.iter().map(|&dac| dac * dac_cal).collect();

    let seqfil = get_first_str("seqfil")
        .or_else(|| get_first_str("pslabel"))
        .map(str::to_string);
    let seq_low = seqfil.as_deref().unwrap_or("").to_ascii_lowercase();

    let nucleus = if let Some(n) = overrides.nucleus {
        n
    } else if let Some(tn) = get_first_str("tn") {
        DosyNucleus::from_label(tn).unwrap_or(DosyNucleus::H1)
    } else {
        DosyNucleus::H1
    };

    let sequence_type = if let Some(st) = overrides.sequence_type {
        st
    } else if seq_low.contains("bpp") {
        DosySequenceType::BppLed
    } else {
        DosySequenceType::Ste
    };

    let little_delta_s = overrides
        .little_delta_s
        .or_else(|| get_nums("gt1").map(|v| v[0]))
        .ok_or_else(|| DosyError::new("missing Varian `gt1` gradient duration in procpar"))?;

    let big_delta_s = overrides
        .big_delta_s
        .or_else(|| get_nums("del").map(|v| v[0]))
        .ok_or_else(|| DosyError::new("missing Varian `del` diffusion delay in procpar"))?;

    let tau_s = if let Some(t) = overrides.tau_s {
        t
    } else if sequence_type == DosySequenceType::BppLed {
        get_nums("tau").map(|v| v[0]).unwrap_or(0.0)
    } else {
        0.0
    };

    let gradient_shape = if let Some(sh) = overrides.gradient_shape {
        sh
    } else if let Some(gs) = get_first_str("gshape").or_else(|| get_first_str("gradshape")) {
        GradientShape::from_shape_name(gs)
    } else {
        GradientShape::Rectangular
    };

    DosyExperimentParams::build(
        nucleus,
        sequence_type,
        gradient_shape,
        big_delta_s,
        little_delta_s,
        tau_s,
        gradients_g_per_cm,
        seqfil,
    )
}

/// Read Varian/Agilent `procpar` from an `.fid` directory on disk.
pub fn read_varian_dosy_dir(
    exp_dir: impl AsRef<Path>,
    overrides: &VarianDosyOverrides,
) -> Result<DosyExperimentParams, DosyError> {
    let dir = exp_dir.as_ref();
    let procpar_path = find_file_case_insensitive(dir, "procpar")
        .ok_or_else(|| DosyError::new(format!("missing Varian procpar in {}", dir.display())))?;
    let text = fs::read_to_string(&procpar_path)
        .map_err(|e| DosyError::new(format!("failed to read Varian procpar: {e}")))?;
    parse_varian_procpar_dosy(&text, overrides)
}

// ============================================================================
// 2. Stejskal–Tanner Multi-Exponential LM Fitter, AIC/BIC & Bootstrap CIs
// ============================================================================

/// Statistical model-order selection criterion across $K \in \{1, 2, 3\}$ exponential components.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelSelectionCriterion {
    /// Bayesian / Schwarz Information Criterion ($\text{BIC} = M \ln(\hat{\sigma}^2) + p \ln M$, default).
    Bic,
    /// Akaike Information Criterion ($\text{AIC} = M \ln(\hat{\sigma}^2) + 2 p$).
    Aic,
}

/// Resampling strategy for 95% confidence intervals on fitted diffusion coefficients $D_k$.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BootstrapMethod {
    /// Resample leverage-adjusted fit residuals with replacement.
    Residual,
    /// Draw synthetic Gaussian noise $\mathcal{N}(0, \hat{\sigma}^2)$ around the fitted decay curve.
    Parametric,
}

/// Configuration for discrete Stejskal–Tanner nonlinear Levenberg–Marquardt fitting (`K in 1..=3`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DosyFitConfig {
    /// Minimum physically allowed diffusion coefficient $D_{\min}$ (`m^2 / s`).
    pub d_min_m2_s: f64,
    /// Maximum physically allowed diffusion coefficient $D_{\max}$ (`m^2 / s`).
    pub d_max_m2_s: f64,
    /// Maximum component count evaluated during model selection (`1`, `2`, or `3`).
    pub max_components: usize,
    /// Information criterion used to select $K \in 1..=K_{\max}$.
    pub criterion: ModelSelectionCriterion,
    /// Minimum amplitude fraction $I_{0,k} / \sum_j I_{0,j}$ for a multi-exponential component to be
    /// accepted as physically distinct rather than noise overfitting.
    pub min_component_fraction: f64,
    /// Minimum diffusivity ratio $D_{k+1} / D_k$ between adjacent sorted components for $K > 1$
    /// to be considered resolved.
    pub min_d_ratio: f64,
    /// Maximum Levenberg–Marquardt iterations per seed.
    pub max_lm_iters: usize,
    /// Number of bootstrap replicates for 95% confidence intervals (`0` disables bootstrap).
    pub bootstrap_samples: usize,
    /// Bootstrap resampling mode (`Residual` or `Parametric`).
    pub bootstrap_method: BootstrapMethod,
    /// Deterministic RNG seed for reproducible bootstrap confidence intervals.
    pub rng_seed: u64,
    /// Optional known baseline intensity noise $\sigma$ (when `None`, estimated from residuals).
    pub known_noise_sigma: Option<f64>,
}

impl Default for DosyFitConfig {
    fn default() -> Self {
        Self {
            d_min_m2_s: 1.0e-12,
            d_max_m2_s: 1.0e-8,
            max_components: 3,
            criterion: ModelSelectionCriterion::Bic,
            min_component_fraction: 0.03,
            min_d_ratio: 1.35,
            max_lm_iters: 80,
            bootstrap_samples: 200,
            bootstrap_method: BootstrapMethod::Residual,
            rng_seed: 0xD057_2026_0014_0004,
            known_noise_sigma: None,
        }
    }
}

/// One resolved exponential component $I_{0,k} \exp(-b D_k)$ sorted by ascending $D_k$.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DosyComponent {
    /// Pre-exponential amplitude $I_{0,k}$ at $b = 0$.
    pub amplitude: f64,
    /// Fractional population $I_{0,k} / \sum_j I_{0,j} \in [0, 1]$.
    pub fraction: f64,
    /// Translational diffusion coefficient $D_k$ (`m^2 / s`).
    pub d_m2_s: f64,
    /// Base-10 logarithm $\log_{10}(D_k / \text{m}^2\text{s}^{-1})$.
    pub log10_d: f64,
    /// Bootstrap 95% confidence interval `(d_low_2p5, d_high_97p5)` (`m^2 / s`).
    pub ci95_d_m2_s: (f64, f64),
    /// Bootstrap standard error of $\log_{10} D_k$ (used for 2D DOSY Gaussian projection width).
    pub sd_log10_d: f64,
}

/// Fit result for a specific fixed component count $K \in \{1, 2, 3\}$.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DosyExpModelFit {
    /// Number of exponential components $K$.
    pub k_components: usize,
    /// Resolved components sorted by ascending $D_1 \le \dots \le D_K$.
    pub components: Vec<DosyComponent>,
    /// Residual sum of squares $\sum_{m=1}^M (y_m - \hat{y}_m)^2$.
    pub rss: f64,
    /// Root-mean-square residual $\sqrt{\text{RSS} / M}$.
    pub rmse: f64,
    /// Akaike Information Criterion score (lower is better).
    pub aic: f64,
    /// Bayesian Information Criterion score (lower is better).
    pub bic: f64,
    /// Fitted attenuation curve $\hat{y}_m = \sum_{k=1}^K I_{0,k} \exp(-b_m D_k)$.
    pub fitted_curve: Vec<f64>,
    /// Fit residuals $r_m = y_m - \hat{y}_m$.
    pub residuals: Vec<f64>,
    /// Whether all components satisfy minimum amplitude share and diffusivity separation guards.
    pub is_well_resolved: bool,
    /// Number of Levenberg–Marquardt iterations executed on the winning seed.
    pub lm_iterations: usize,
}

/// Multi-model (`K in 1..=K_max`) Stejskal–Tanner fit and `AIC`/`BIC` selection summary.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DosyMultiExpSelection {
    /// Selected optimal component count $K^\star \in \{1, 2, 3\}$.
    pub selected_k: usize,
    /// Criterion used to choose `selected_k`.
    pub criterion: ModelSelectionCriterion,
    /// Winning [`DosyExpModelFit`] (with bootstrap 95% CIs populated).
    pub best_fit: DosyExpModelFit,
    /// All candidate fits evaluated for $K = 1 \dots K_{\max}$ (index `k - 1`).
    pub candidate_fits: Vec<DosyExpModelFit>,
}

/// Number of log-spaced diffusivity knots in the shared CPU/GPU VarPro initializer for $K = 1, 2$.
pub const DOSY_VARPRO_GRID_K12: usize = 24;

/// Solve a small dense linear system $A x = b$ ($n \le 8$) via Gaussian elimination with partial pivoting.
fn solve_small_linear(n: usize, a: &[f64], b: &[f64]) -> Option<Vec<f64>> {
    let mut aug = vec![0.0; n * (n + 1)];
    for r in 0..n {
        for c in 0..n {
            aug[r * (n + 1) + c] = a[r * n + c];
        }
        aug[r * (n + 1) + n] = b[r];
    }

    for col in 0..n {
        let mut piv_row = col;
        let mut max_abs = aug[col * (n + 1) + col].abs();
        for r in (col + 1)..n {
            let v = aug[r * (n + 1) + col].abs();
            if v > max_abs {
                max_abs = v;
                piv_row = r;
            }
        }
        if !max_abs.is_finite() || max_abs < 1.0e-18 {
            return None;
        }
        if piv_row != col {
            for c in col..=n {
                aug.swap(col * (n + 1) + c, piv_row * (n + 1) + c);
            }
        }
        let diag = aug[col * (n + 1) + col];
        for r in (col + 1)..n {
            let factor = aug[r * (n + 1) + col] / diag;
            aug[r * (n + 1) + col] = 0.0;
            for c in (col + 1)..=n {
                aug[r * (n + 1) + c] -= factor * aug[col * (n + 1) + c];
            }
        }
    }

    let mut x = vec![0.0; n];
    for r in (0..n).rev() {
        let mut s = aug[r * (n + 1) + n];
        for c in (r + 1)..n {
            s -= aug[r * (n + 1) + c] * x[c];
        }
        let d = aug[r * (n + 1) + r];
        if !d.is_finite() || d.abs() < 1.0e-18 {
            return None;
        }
        x[r] = s / d;
    }
    Some(x)
}

/// Kaufman (1975) / Golub–Pereyra (1973) Variable-Projection (`VarPro`) damped Gauss–Newton
/// stage in log-rate space $u = (u_1, \dots, u_K)$, eliminating linear amplitude parameters
/// $\hat{a}(u) = G(u)^{-1} E(u)^\top \tilde{y}$ at every step to break multi-exponential valley
/// ill-conditioning before joint $(\tilde{a}, u)$ Levenberg–Marquardt polish.
fn varpro_polish_dimensionless(
    b_tilde: &[f64],
    y_tilde: &[f64],
    k_comp: usize,
    init_a: &[f64],
    init_u: &[f64],
    (u_min, u_max): (f64, f64),
    max_iters: usize,
) -> (Vec<f64>, Vec<f64>) {
    let m = b_tilde.len();
    let mut u: Vec<f64> = init_u.iter().map(|&uk| uk.clamp(u_min, u_max)).collect();
    let mut best_a: Vec<f64> = init_a.iter().map(|&ak| ak.max(1.0e-6)).collect();

    let eval_vp = |u_try: &[f64]| -> Option<(Vec<f64>, f64)> {
        let d: Vec<f64> = u_try.iter().map(|&uk| uk.exp()).collect();
        let mut g = vec![0.0; k_comp * k_comp];
        let mut c = vec![0.0; k_comp];
        let mut ek = vec![0.0; k_comp];
        for i in 0..m {
            let bt = b_tilde[i];
            for k in 0..k_comp {
                ek[k] = (-bt * d[k]).exp();
                c[k] += ek[k] * y_tilde[i];
            }
            for p in 0..k_comp {
                for q in p..k_comp {
                    g[p * k_comp + q] += ek[p] * ek[q];
                }
            }
        }
        for p in 0..k_comp {
            for q in 0..p {
                g[p * k_comp + q] = g[q * k_comp + p];
            }
            g[p * k_comp + p] += 1.0e-14;
        }
        let a_opt = solve_small_linear(k_comp, &g, &c)?;
        if a_opt.iter().any(|&ak| !ak.is_finite() || ak <= 1.0e-5) {
            return None;
        }
        let mut rss = 0.0;
        for i in 0..m {
            let bt = b_tilde[i];
            let mut pred = 0.0;
            for k in 0..k_comp {
                pred += a_opt[k] * (-bt * d[k]).exp();
            }
            let r = pred - y_tilde[i];
            rss += r * r;
        }
        Some((a_opt, rss))
    };

    let mut cur_rss = if let Some((a0, r0)) = eval_vp(&u) {
        best_a = a0;
        r0
    } else {
        return (best_a, u);
    };

    let mut lam = 1.0e-3_f64;
    for _it in 0..max_iters {
        let d: Vec<f64> = u.iter().map(|&uk| uk.exp()).collect();
        let mut e_mat = vec![0.0; m * k_comp];
        let mut phi_dot = vec![0.0; m * k_comp];
        let mut g_gram = vec![0.0; k_comp * k_comp];
        for i in 0..m {
            let bt = b_tilde[i];
            for k in 0..k_comp {
                let bd = bt * d[k];
                let ek = (-bd).exp();
                e_mat[i * k_comp + k] = ek;
                phi_dot[i * k_comp + k] = -best_a[k] * bd * ek;
            }
            for p in 0..k_comp {
                for q in p..k_comp {
                    g_gram[p * k_comp + q] += e_mat[i * k_comp + p] * e_mat[i * k_comp + q];
                }
            }
        }
        for p in 0..k_comp {
            for q in 0..p {
                g_gram[p * k_comp + q] = g_gram[q * k_comp + p];
            }
            g_gram[p * k_comp + p] += 1.0e-14;
        }

        // Kaufman projected columns J_vp[:, k] = phi_dot[:, k] - E * (G^{-1} E^T phi_dot[:, k])
        let mut alpha_proj = vec![0.0; k_comp * k_comp]; // col k stored at k * k_comp ..
        let mut proj_ok = true;
        for k in 0..k_comp {
            let mut et_phi = vec![0.0; k_comp];
            for p in 0..k_comp {
                let mut s = 0.0;
                for i in 0..m {
                    s += e_mat[i * k_comp + p] * phi_dot[i * k_comp + k];
                }
                et_phi[p] = s;
            }
            if let Some(ak_proj) = solve_small_linear(k_comp, &g_gram, &et_phi) {
                alpha_proj[k * k_comp..(k + 1) * k_comp].copy_from_slice(&ak_proj);
            } else {
                proj_ok = false;
                break;
            }
        }
        if !proj_ok {
            break;
        }

        let mut h_vp = vec![0.0; k_comp * k_comp];
        let mut g_vp = vec![0.0; k_comp];
        let mut j_row = vec![0.0; k_comp];
        for i in 0..m {
            let mut pred = 0.0;
            for k in 0..k_comp {
                pred += best_a[k] * e_mat[i * k_comp + k];
                let mut proj_val = 0.0;
                for p in 0..k_comp {
                    proj_val += alpha_proj[k * k_comp + p] * e_mat[i * k_comp + p];
                }
                j_row[k] = phi_dot[i * k_comp + k] - proj_val;
            }
            let r = pred - y_tilde[i];
            for p in 0..k_comp {
                g_vp[p] += j_row[p] * r;
                for q in p..k_comp {
                    h_vp[p * k_comp + q] += j_row[p] * j_row[q];
                }
            }
        }
        for p in 0..k_comp {
            for q in 0..p {
                h_vp[p * k_comp + q] = h_vp[q * k_comp + p];
            }
        }

        if g_vp.iter().all(|&v| v.abs() < 1.0e-11) {
            break;
        }

        let mut accepted = false;
        for _sub in 0..7 {
            let mut hd = h_vp.clone();
            for p in 0..k_comp {
                hd[p * k_comp + p] += lam * h_vp[p * k_comp + p].max(1.0e-4) + 1.0e-9;
            }
            let neg_g: Vec<f64> = g_vp.iter().map(|&v| -v).collect();
            let Some(du) = solve_small_linear(k_comp, &hd, &neg_g) else {
                lam = (lam * 8.0).min(1.0e7);
                continue;
            };
            let mut u_try = u.clone();
            let mut max_du = 0.0_f64;
            for k in 0..k_comp {
                u_try[k] = (u[k] + du[k].clamp(-1.0, 1.0)).clamp(u_min, u_max);
                max_du = max_du.max((u_try[k] - u[k]).abs());
            }
            if let Some((a_try, rss_try)) = eval_vp(&u_try) {
                if rss_try <= cur_rss {
                    let rel = (cur_rss - rss_try) / (1.0e-15 + cur_rss);
                    u = u_try;
                    best_a = a_try;
                    cur_rss = rss_try;
                    lam = (lam * 0.3).max(1.0e-11);
                    accepted = true;
                    if rel < 1.0e-13 && max_du < 1.0e-8 {
                        return (best_a, u);
                    }
                    break;
                }
            }
            lam = (lam * 5.0).min(1.0e7);
        }
        if !accepted && lam >= 1.0e7 {
            break;
        }
    }

    (best_a, u)
}

/// Dimensionless Levenberg–Marquardt polish on $\theta = (\tilde{a}_1, u_1, \dots, \tilde{a}_K, u_K)$
/// where $\tilde{D}_k = \exp(u_k)$, matching the WGSL GPU formulation for $K = 1, 2$.
fn lm_polish_dimensionless(
    b_tilde: &[f64],
    y_tilde: &[f64],
    k_comp: usize,
    init_a: &[f64],
    init_u: &[f64],
    (u_min, u_max): (f64, f64),
    max_iters: usize,
) -> (Vec<f64>, Vec<f64>, f64, usize) {
    let vp_iters = (max_iters / 2).clamp(6, 25);
    let (vp_a, vp_u) = varpro_polish_dimensionless(
        b_tilde,
        y_tilde,
        k_comp,
        init_a,
        init_u,
        (u_min, u_max),
        vp_iters,
    );
    let m = b_tilde.len();
    let n_par = 2 * k_comp;
    let mut a = vp_a;
    let mut u = vp_u;
    for k in 0..k_comp {
        a[k] = a[k].max(1.0e-6);
        u[k] = u[k].clamp(u_min, u_max);
    }

    let eval_rss = |a_v: &[f64], u_v: &[f64]| -> f64 {
        let mut rss = 0.0;
        let d_v: Vec<f64> = u_v.iter().map(|&uk| uk.exp()).collect();
        for i in 0..m {
            let mut pred = 0.0;
            for k in 0..k_comp {
                pred += a_v[k] * (-b_tilde[i] * d_v[k]).exp();
            }
            let r = pred - y_tilde[i];
            rss += r * r;
        }
        rss
    };

    let mut cur_rss = eval_rss(&a, &u);
    let mut lambda = 1.0e-3_f64;
    let mut iters_used = 0usize;

    let mut j_row = vec![0.0; n_par];
    let mut h_mat = vec![0.0; n_par * n_par];
    let mut g_vec = vec![0.0; n_par];
    let mut h_damped = vec![0.0; n_par * n_par];

    for iter in 0..max_iters {
        iters_used = iter + 1;
        h_mat.fill(0.0);
        g_vec.fill(0.0);

        let d_cur: Vec<f64> = u.iter().map(|&uk| uk.exp()).collect();
        for i in 0..m {
            let bt = b_tilde[i];
            let mut pred = 0.0;
            for k in 0..k_comp {
                let bd = bt * d_cur[k];
                let ek = (-bd).exp();
                pred += a[k] * ek;
                j_row[2 * k] = ek;
                j_row[2 * k + 1] = -a[k] * bd * ek;
            }
            let r = pred - y_tilde[i];
            for p in 0..n_par {
                g_vec[p] += j_row[p] * r;
                for q in p..n_par {
                    h_mat[p * n_par + q] += j_row[p] * j_row[q];
                }
            }
        }
        for p in 0..n_par {
            for q in 0..p {
                h_mat[p * n_par + q] = h_mat[q * n_par + p];
            }
        }

        let grad_inf = g_vec.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        if grad_inf < 1.0e-11 {
            break;
        }

        let mut accepted = false;
        for _sub in 0..8 {
            h_damped.copy_from_slice(&h_mat);
            for p in 0..n_par {
                let diag = h_mat[p * n_par + p].max(1.0e-3);
                h_damped[p * n_par + p] += lambda * diag + 1.0e-8;
            }
            let neg_g: Vec<f64> = g_vec.iter().map(|&v| -v).collect();
            let Some(step) = solve_small_linear(n_par, &h_damped, &neg_g) else {
                lambda = (lambda * 8.0).min(1.0e8);
                continue;
            };

            let mut trial_a = a.clone();
            let mut trial_u = u.clone();
            let mut max_step = 0.0_f64;
            for k in 0..k_comp {
                let da = step[2 * k].clamp(-2.0, 2.0);
                let du = step[2 * k + 1].clamp(-1.5, 1.5);
                trial_a[k] = (a[k] + da).max(1.0e-7);
                trial_u[k] = (u[k] + du).clamp(u_min, u_max);
                max_step = max_step
                    .max((trial_a[k] - a[k]).abs())
                    .max((trial_u[k] - u[k]).abs());
            }

            let trial_rss = eval_rss(&trial_a, &trial_u);
            if trial_rss <= cur_rss {
                let rel_impr = (cur_rss - trial_rss) / (1.0e-15 + cur_rss);
                a = trial_a;
                u = trial_u;
                cur_rss = trial_rss;
                lambda = (lambda * 0.3).max(1.0e-11);
                accepted = true;
                if rel_impr < 1.0e-13 && max_step < 1.0e-8 {
                    return (a, u, cur_rss, iters_used);
                }
                break;
            } else {
                lambda = (lambda * 5.0).min(1.0e8);
            }
        }
        if !accepted && lambda >= 1.0e8 {
            break;
        }
    }

    (a, u, cur_rss, iters_used)
}

/// Generate Variable-Projection (`VarPro`) initial seeds in dimensionless $(\tilde{a}, u)$ space.
fn generate_varpro_seeds(
    b_tilde: &[f64],
    y_tilde: &[f64],
    k_comp: usize,
    u_min: f64,
    u_max: f64,
) -> Vec<(Vec<f64>, Vec<f64>)> {
    let m = b_tilde.len();
    let y_norm2: f64 = y_tilde.iter().map(|&v| v * v).sum();

    if k_comp == 1 {
        let mut seeds = Vec::with_capacity(2);
        // Seed A: Weighted log-linear regression (`w_m = y_m^2`)
        let mut sw = 0.0;
        let mut swx = 0.0;
        let mut swy = 0.0;
        let mut swxx = 0.0;
        let mut swxy = 0.0;
        let mut count = 0usize;
        for i in 0..m {
            if y_tilde[i] > 0.05 {
                let w = y_tilde[i] * y_tilde[i];
                let x = b_tilde[i];
                let ly = y_tilde[i].ln();
                sw += w;
                swx += w * x;
                swy += w * ly;
                swxx += w * x * x;
                swxy += w * x * ly;
                count += 1;
            }
        }
        if count >= 2 {
            let denom = sw * swxx - swx * swx;
            if denom.abs() > 1.0e-15 {
                let slope = (sw * swxy - swx * swy) / denom;
                let intercept = (swy - slope * swx) / sw;
                let d_est = (-slope).max(1.0e-6);
                let a_est = intercept.exp().clamp(0.05, 2.5);
                seeds.push((vec![a_est], vec![d_est.ln().clamp(u_min, u_max)]));
            }
        }

        // Seed B: Exact 24-point 1D VarPro grid (shared with WGSL shader)
        let q_len = DOSY_VARPRO_GRID_K12;
        let mut best_rss = f64::INFINITY;
        let mut best_au = (vec![1.0], vec![0.5 * (u_min + u_max)]);
        for q in 0..q_len {
            let uq = u_min + (q as f64) * (u_max - u_min) / ((q_len - 1) as f64);
            let dq = uq.exp();
            let mut g_qq = 0.0;
            let mut c_q = 0.0;
            for i in 0..m {
                let e = (-b_tilde[i] * dq).exp();
                g_qq += e * e;
                c_q += e * y_tilde[i];
            }
            if g_qq > 1.0e-15 {
                let aq = (c_q / g_qq).max(1.0e-5);
                let rss = (y_norm2 - 2.0 * aq * c_q + aq * aq * g_qq).max(0.0);
                if rss < best_rss {
                    best_rss = rss;
                    best_au = (vec![aq], vec![uq]);
                }
            }
        }
        seeds.push(best_au);
        return seeds;
    }

    if k_comp == 2 {
        let q_len = DOSY_VARPRO_GRID_K12;
        let mut u_grid = vec![0.0; q_len];
        let mut e_mat = vec![0.0; q_len * m];
        let mut c_vec = vec![0.0; q_len];
        for q in 0..q_len {
            let uq = u_min + (q as f64) * (u_max - u_min) / ((q_len - 1) as f64);
            u_grid[q] = uq;
            let dq = uq.exp();
            let mut cq = 0.0;
            for i in 0..m {
                let e = (-b_tilde[i] * dq).exp();
                e_mat[q * m + i] = e;
                cq += e * y_tilde[i];
            }
            c_vec[q] = cq;
        }
        let mut g_mat = vec![0.0; q_len * q_len];
        for p in 0..q_len {
            for q in p..q_len {
                let mut s = 0.0;
                for i in 0..m {
                    s += e_mat[p * m + i] * e_mat[q * m + i];
                }
                g_mat[p * q_len + q] = s;
                g_mat[q * q_len + p] = s;
            }
        }

        let min_du = 1.25_f64.ln();
        let mut candidates: Vec<(f64, [f64; 2], [f64; 2])> = Vec::new();
        for p in 0..q_len {
            for q in (p + 1)..q_len {
                if u_grid[q] - u_grid[p] < min_du {
                    continue;
                }
                let g_pp = g_mat[p * q_len + p];
                let g_pq = g_mat[p * q_len + q];
                let g_qq = g_mat[q * q_len + q];
                let det = g_pp * g_qq - g_pq * g_pq;
                if det <= 1.0e-12 * g_pp * g_qq {
                    continue;
                }
                let cp = c_vec[p];
                let cq = c_vec[q];
                let a1 = (g_qq * cp - g_pq * cq) / det;
                let a2 = (g_pp * cq - g_pq * cp) / det;
                if a1 > 1.0e-3 && a2 > 1.0e-3 {
                    let rss = (y_norm2 - 2.0 * (a1 * cp + a2 * cq)
                        + a1 * a1 * g_pp
                        + 2.0 * a1 * a2 * g_pq
                        + a2 * a2 * g_qq)
                        .max(0.0);
                    candidates.push((rss, [a1, a2], [u_grid[p], u_grid[q]]));
                }
            }
        }
        candidates.sort_by(|x, y| x.0.total_cmp(&y.0));
        let mut seeds: Vec<(Vec<f64>, Vec<f64>)> = candidates
            .into_iter()
            .take(3)
            .map(|(_, a, u)| (a.to_vec(), u.to_vec()))
            .collect();
        if seeds.is_empty() {
            let mid = 0.5 * (u_min + u_max);
            seeds.push((
                vec![0.5, 0.5],
                vec![(mid - 0.6).max(u_min), (mid + 0.6).min(u_max)],
            ));
        }
        return seeds;
    }

    // k_comp == 3: 15-point 3D VarPro grid
    let q_len = 15usize;
    let mut u_grid = vec![0.0; q_len];
    let mut e_mat = vec![0.0; q_len * m];
    let mut c_vec = vec![0.0; q_len];
    for q in 0..q_len {
        let uq = u_min + (q as f64) * (u_max - u_min) / ((q_len - 1) as f64);
        u_grid[q] = uq;
        let dq = uq.exp();
        let mut cq = 0.0;
        for i in 0..m {
            let e = (-b_tilde[i] * dq).exp();
            e_mat[q * m + i] = e;
            cq += e * y_tilde[i];
        }
        c_vec[q] = cq;
    }
    let mut g_mat = vec![0.0; q_len * q_len];
    for p in 0..q_len {
        for q in p..q_len {
            let mut s = 0.0;
            for i in 0..m {
                s += e_mat[p * m + i] * e_mat[q * m + i];
            }
            g_mat[p * q_len + q] = s;
            g_mat[q * q_len + p] = s;
        }
    }

    let min_du = 1.25_f64.ln();
    let mut candidates: Vec<(f64, [f64; 3], [f64; 3])> = Vec::new();
    for p in 0..q_len {
        for q in (p + 1)..q_len {
            if u_grid[q] - u_grid[p] < min_du {
                continue;
            }
            for r in (q + 1)..q_len {
                if u_grid[r] - u_grid[q] < min_du {
                    continue;
                }
                let g3 = [
                    g_mat[p * q_len + p],
                    g_mat[p * q_len + q],
                    g_mat[p * q_len + r],
                    g_mat[q * q_len + p],
                    g_mat[q * q_len + q],
                    g_mat[q * q_len + r],
                    g_mat[r * q_len + p],
                    g_mat[r * q_len + q],
                    g_mat[r * q_len + r],
                ];
                let rhs = [c_vec[p], c_vec[q], c_vec[r]];
                let Some(sol) = solve_small_linear(3, &g3, &rhs) else {
                    continue;
                };
                if sol[0] > 1.0e-3 && sol[1] > 1.0e-3 && sol[2] > 1.0e-3 {
                    let mut quad = y_norm2;
                    for ia in 0..3 {
                        quad -= 2.0 * sol[ia] * rhs[ia];
                        for ib in 0..3 {
                            quad += sol[ia] * sol[ib] * g3[ia * 3 + ib];
                        }
                    }
                    candidates.push((
                        quad.max(0.0),
                        [sol[0], sol[1], sol[2]],
                        [u_grid[p], u_grid[q], u_grid[r]],
                    ));
                }
            }
        }
    }
    candidates.sort_by(|x, y| x.0.total_cmp(&y.0));
    let mut seeds: Vec<(Vec<f64>, Vec<f64>)> = candidates
        .into_iter()
        .take(3)
        .map(|(_, a, u)| (a.to_vec(), u.to_vec()))
        .collect();
    if seeds.is_empty() {
        let span = u_max - u_min;
        seeds.push((
            vec![0.34, 0.33, 0.33],
            vec![
                u_min + 0.25 * span,
                u_min + 0.50 * span,
                u_min + 0.75 * span,
            ],
        ));
    }
    seeds
}

fn compute_aic_bic(
    rss: f64,
    m: usize,
    k_comp: usize,
    y_scale: f64,
    known_noise_sigma: Option<f64>,
) -> (f64, f64) {
    let p = (2 * k_comp) as f64;
    let m_f = m as f64;
    // Davies (1977) look-elsewhere penalty (`2 ln(6)` per added exponential component) accounting
    // for global search of unidentified rate parameter D_k under the single-species null.
    let look_elsewhere = 2.0 * (k_comp.saturating_sub(1) as f64) * 6.0_f64.ln();
    if let Some(sig) = known_noise_sigma.filter(|&s| s.is_finite() && s > 0.0) {
        let chi2 = rss / (sig * sig);
        (
            chi2 + 2.0 * p + look_elsewhere,
            chi2 + p * m_f.ln() + look_elsewhere,
        )
    } else {
        // Guard exact noiseless synthetic curves against log(0) floating-point noise differences
        let sigma_num_sq = (1.0e-6 * y_scale).powi(2);
        let s2 = (rss / m_f).max(sigma_num_sq);
        let ll_term = m_f * s2.ln();
        (
            ll_term + 2.0 * p + look_elsewhere,
            ll_term + p * m_f.ln() + look_elsewhere,
        )
    }
}

/// Fit a fixed-$K$ (`K in 1..=3`) Stejskal–Tanner model $I(b) = \sum_{k=1}^K I_{0,k} \exp(-b D_k)$
/// via multi-start Variable-Projection + Levenberg–Marquardt nonlinear least squares.
pub fn fit_stejskal_tanner_k(
    b_s_m2: &[f64],
    intensities: &[f64],
    k_components: usize,
    cfg: &DosyFitConfig,
    with_bootstrap: bool,
) -> Result<DosyExpModelFit, DosyError> {
    if !(1..=3).contains(&k_components) {
        return Err(DosyError::new(format!(
            "unsupported component count K = {k_components} (must be 1, 2, or 3)"
        )));
    }
    let m = b_s_m2.len();
    if m != intensities.len() {
        return Err(DosyError::new(format!(
            "b_s_m2 length ({m}) != intensities length ({})",
            intensities.len()
        )));
    }
    if m < 2 * k_components + 1 {
        return Err(DosyError::new(format!(
            "insufficient gradient points ({m}) to fit K = {k_components} exponential components \
             (require >= {})",
            2 * k_components + 1
        )));
    }
    if !cfg.d_min_m2_s.is_finite()
        || !cfg.d_max_m2_s.is_finite()
        || cfg.d_min_m2_s <= 0.0
        || cfg.d_max_m2_s <= cfg.d_min_m2_s
    {
        return Err(DosyError::new(format!(
            "invalid diffusivity bounds [{}, {}] m^2/s",
            cfg.d_min_m2_s, cfg.d_max_m2_s
        )));
    }

    let mut b_scale = 0.0_f64;
    let mut y_scale = 0.0_f64;
    for i in 0..m {
        if !b_s_m2[i].is_finite() || b_s_m2[i] < 0.0 {
            return Err(DosyError::new(format!(
                "invalid b-value at index {i}: {}",
                b_s_m2[i]
            )));
        }
        if !intensities[i].is_finite() {
            return Err(DosyError::new(format!(
                "non-finite intensity at index {i}: {}",
                intensities[i]
            )));
        }
        b_scale = b_scale.max(b_s_m2[i]);
        y_scale = y_scale.max(intensities[i].abs());
    }
    if b_scale <= 0.0 {
        return Err(DosyError::new(
            "all b-values are zero; cannot fit diffusivity",
        ));
    }
    if y_scale <= 0.0 {
        return Err(DosyError::new("all intensities are zero; cannot fit decay"));
    }

    let b_tilde: Vec<f64> = b_s_m2.iter().map(|&b| b / b_scale).collect();
    let y_tilde: Vec<f64> = intensities.iter().map(|&y| y / y_scale).collect();
    let u_min = (cfg.d_min_m2_s * b_scale).ln();
    let u_max = (cfg.d_max_m2_s * b_scale).ln();

    let seeds = generate_varpro_seeds(&b_tilde, &y_tilde, k_components, u_min, u_max);
    let mut best_a = vec![1.0; k_components];
    let mut best_u = vec![0.5 * (u_min + u_max); k_components];
    let mut best_rss_tilde = f64::INFINITY;
    let mut best_iters = 0usize;

    for (sa, su) in &seeds {
        let (ca, cu, crss, iters) = lm_polish_dimensionless(
            &b_tilde,
            &y_tilde,
            k_components,
            sa,
            su,
            (u_min, u_max),
            cfg.max_lm_iters.max(10),
        );
        if crss < best_rss_tilde {
            best_rss_tilde = crss;
            best_a = ca;
            best_u = cu;
            best_iters = iters;
        }
    }

    // Sort components by ascending diffusivity u_1 <= u_2 <= ... <= u_K
    let mut order: Vec<usize> = (0..k_components).collect();
    order.sort_by(|&i, &j| best_u[i].total_cmp(&best_u[j]));
    let sorted_a: Vec<f64> = order.iter().map(|&i| best_a[i]).collect();
    let sorted_u: Vec<f64> = order.iter().map(|&i| best_u[i]).collect();

    let amplitudes: Vec<f64> = sorted_a.iter().map(|&ak| ak * y_scale).collect();
    let d_vals: Vec<f64> = sorted_u.iter().map(|&uk| uk.exp() / b_scale).collect();
    let amp_sum: f64 = amplitudes.iter().sum::<f64>().max(1.0e-30);

    let mut fitted_curve = Vec::with_capacity(m);
    let mut residuals = Vec::with_capacity(m);
    let mut rss = 0.0_f64;
    for i in 0..m {
        let mut yhat = 0.0;
        for k in 0..k_components {
            yhat += amplitudes[k] * (-b_s_m2[i] * d_vals[k]).exp();
        }
        let r = intensities[i] - yhat;
        rss += r * r;
        fitted_curve.push(yhat);
        residuals.push(r);
    }

    let rmse = (rss / (m as f64)).sqrt();
    let (aic, bic) = compute_aic_bic(rss, m, k_components, y_scale, cfg.known_noise_sigma);

    let mut is_well_resolved = true;
    if k_components > 1 {
        let dof = (m.saturating_sub(2 * k_components)).max(1) as f64;
        let sigma_hat = cfg
            .known_noise_sigma
            .filter(|&s| s.is_finite() && s > 0.0)
            .unwrap_or_else(|| (rss / dof).sqrt());
        for k in 0..k_components {
            if amplitudes[k] / amp_sum < cfg.min_component_fraction {
                is_well_resolved = false;
            }
            if k + 1 < k_components && d_vals[k + 1] / d_vals[k].max(1.0e-30) < cfg.min_d_ratio {
                is_well_resolved = false;
            }
            let comp_l2_norm = b_s_m2
                .iter()
                .map(|&b| {
                    let v = amplitudes[k] * (-b * d_vals[k]).exp();
                    v * v
                })
                .sum::<f64>()
                .sqrt();
            if comp_l2_norm < 4.0 * sigma_hat {
                is_well_resolved = false;
            }
        }
    }

    // Bootstrap 95% confidence intervals on D_k
    let mut ci95 = vec![(0.0, 0.0); k_components];
    let mut sd_log10 = vec![0.02; k_components];
    for k in 0..k_components {
        ci95[k] = (d_vals[k], d_vals[k]);
    }

    if with_bootstrap && cfg.bootstrap_samples >= 8 {
        let mut rng = Rng::new(cfg.rng_seed ^ ((k_components as u64) * 0x9E37_79B9));
        let dof = (m.saturating_sub(2 * k_components)).max(1) as f64;
        let lev_scale = ((m as f64) / dof).sqrt();
        let sigma_hat_tilde = cfg
            .known_noise_sigma
            .map(|s| s / y_scale)
            .unwrap_or_else(|| (best_rss_tilde / dof).sqrt());
        let res_tilde: Vec<f64> = residuals
            .iter()
            .map(|&r| (r / y_scale) * lev_scale)
            .collect();
        let yhat_tilde: Vec<f64> = fitted_curve.iter().map(|&yh| yh / y_scale).collect();

        let mut samples_per_k: Vec<Vec<f64>> =
            vec![Vec::with_capacity(cfg.bootstrap_samples); k_components];
        let mut y_boot = vec![0.0; m];

        for _b in 0..cfg.bootstrap_samples {
            for i in 0..m {
                let noise = match cfg.bootstrap_method {
                    BootstrapMethod::Residual => {
                        let pick = rng.usize(m);
                        res_tilde[pick]
                    }
                    BootstrapMethod::Parametric => rng.normal(0.0, sigma_hat_tilde),
                };
                y_boot[i] = yhat_tilde[i] + noise;
            }
            let (_, bu, _, _) = lm_polish_dimensionless(
                &b_tilde,
                &y_boot,
                k_components,
                &sorted_a,
                &sorted_u,
                (u_min, u_max),
                35,
            );
            let mut bd: Vec<f64> = bu.into_iter().map(|uk| uk.exp() / b_scale).collect();
            bd.sort_by(|x, y| x.total_cmp(y));
            for k in 0..k_components {
                samples_per_k[k].push(bd[k]);
            }
        }

        for k in 0..k_components {
            let s = &mut samples_per_k[k];
            s.sort_by(|x, y| x.total_cmp(y));
            let n_s = s.len();
            let lo_idx = ((0.025 * ((n_s - 1) as f64)).round() as usize).min(n_s - 1);
            let hi_idx = ((0.975 * ((n_s - 1) as f64)).round() as usize).min(n_s - 1);
            ci95[k] = (s[lo_idx], s[hi_idx]);

            let logs: Vec<f64> = s.iter().map(|&d| d.max(1.0e-30).log10()).collect();
            let mean_l = logs.iter().sum::<f64>() / (n_s as f64);
            let var_l = logs
                .iter()
                .map(|&l| (l - mean_l) * (l - mean_l))
                .sum::<f64>()
                / ((n_s - 1).max(1) as f64);
            sd_log10[k] = var_l.sqrt().max(1.0e-4);
        }
    }

    let components = (0..k_components)
        .map(|k| DosyComponent {
            amplitude: amplitudes[k],
            fraction: amplitudes[k] / amp_sum,
            d_m2_s: d_vals[k],
            log10_d: d_vals[k].log10(),
            ci95_d_m2_s: ci95[k],
            sd_log10_d: sd_log10[k],
        })
        .collect();

    Ok(DosyExpModelFit {
        k_components,
        components,
        rss,
        rmse,
        aic,
        bic,
        fitted_curve,
        residuals,
        is_well_resolved,
        lm_iterations: best_iters,
    })
}

/// Evaluate mono-, bi-, and tri-exponential Stejskal–Tanner fits (`K in 1..=cfg.max_components`),
/// select the optimal component order via `BIC` or `AIC` (penalizing unresolved/overfit components),
/// and compute bootstrap 95% confidence intervals on the selected model.
pub fn fit_stejskal_tanner_multi(
    b_s_m2: &[f64],
    intensities: &[f64],
    cfg: &DosyFitConfig,
) -> Result<DosyMultiExpSelection, DosyError> {
    let k_max = cfg.max_components.clamp(1, 3);
    let m = b_s_m2.len();
    if m < 3 {
        return Err(DosyError::new(format!(
            "DOSY multi-exponential fit requires >= 3 gradient points, got {m}"
        )));
    }

    let mut candidate_fits = Vec::with_capacity(k_max);
    for k in 1..=k_max {
        if m < 2 * k + 1 {
            break;
        }
        let fit = fit_stejskal_tanner_k(b_s_m2, intensities, k, cfg, false)?;
        candidate_fits.push(fit);
    }

    let score_of = |fit: &DosyExpModelFit| -> f64 {
        match cfg.criterion {
            ModelSelectionCriterion::Bic => fit.bic,
            ModelSelectionCriterion::Aic => fit.aic,
        }
    };

    let mut best_idx = 0usize;
    let mut best_score = score_of(&candidate_fits[0]);
    for (idx, fit) in candidate_fits.iter().enumerate().skip(1) {
        if !fit.is_well_resolved {
            continue;
        }
        let sc = score_of(fit);
        if sc < best_score {
            best_score = sc;
            best_idx = idx;
        }
    }

    let selected_k = candidate_fits[best_idx].k_components;
    if cfg.bootstrap_samples >= 8 {
        let best_with_ci = fit_stejskal_tanner_k(b_s_m2, intensities, selected_k, cfg, true)?;
        candidate_fits[best_idx] = best_with_ci;
    }
    let best_fit = candidate_fits[best_idx].clone();

    Ok(DosyMultiExpSelection {
        selected_k,
        criterion: cfg.criterion,
        best_fit,
        candidate_fits,
    })
}

// ============================================================================
// 3. Opt-In Regularized Inverse Laplace Transform (`ILT`, Provencher 1982 + NNLS)
// ============================================================================

/// Regularization penalty operator $L$ in $\min_{f \ge 0} \|K f - y\|_2^2 + \lambda \|L f\|_2^2$.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TikhonovRegularizer {
    /// Constrained second-derivative smoothness penalty (`CONTIN`-style, Provencher 1982) with
    /// gentle endpoint damping so smooth polydisperse distributions are not fractured into spikes.
    SecondDerivative,
    /// Zeroth-order identity penalty ($L = I_N$).
    Identity,
}

/// Automatic or manual regularization parameter ($\lambda > 0$) selection strategy for `ILT`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum IltLambdaStrategy {
    /// Generalized Cross-Validation (`GCV`, Golub, Heath & Wahba 1979 / Provencher 1982) over the
    /// active positive subspace.
    Gcv,
    /// Maximum log-log Pareto curvature corner of $(\ln \|K f - y\|_2, \ln \|L f\|_2)$ (Hansen 1992).
    LCurve,
    /// Morozov discrepancy principle choosing the largest $\lambda$ with $\|K f - y\|_2^2 \le M \sigma^2$.
    DiscrepancyPrinciple { noise_sigma: f64 },
    /// User-specified fixed regularization parameter $\lambda > 0$.
    Fixed(f64),
}

/// Configuration for continuous non-negative Tikhonov Inverse Laplace Transform (`ILT`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IltConfig {
    /// Number of log-spaced diffusivity grid points $N$ (`>= 16`, default `64`).
    pub n_grid: usize,
    /// Minimum grid diffusivity $D_{\min}$ (`m^2 / s`).
    pub d_min_m2_s: f64,
    /// Maximum grid diffusivity $D_{\max}$ (`m^2 / s`).
    pub d_max_m2_s: f64,
    /// Penalty operator $L$ (`SecondDerivative` default).
    pub regularizer: TikhonovRegularizer,
    /// Selection strategy for $\lambda$.
    pub lambda_strategy: IltLambdaStrategy,
    /// Minimum $\lambda$ scanned when `lambda_strategy` sweeps a candidate grid.
    pub lambda_min: f64,
    /// Maximum $\lambda$ scanned when `lambda_strategy` sweeps a candidate grid.
    pub lambda_max: f64,
    /// Number of log-spaced $\lambda$ values in the regularization sweep (`>= 5`).
    pub n_lambda_sweep: usize,
}

impl Default for IltConfig {
    fn default() -> Self {
        Self {
            n_grid: 64,
            d_min_m2_s: 1.0e-11,
            d_max_m2_s: 5.0e-9,
            regularizer: TikhonovRegularizer::SecondDerivative,
            lambda_strategy: IltLambdaStrategy::Gcv,
            lambda_min: 1.0e-5,
            lambda_max: 1.0e1,
            n_lambda_sweep: 25,
        }
    }
}

/// Diagnostic metrics for one candidate $\lambda$ along the Tikhonov `ILT` sweep.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IltSweepPoint {
    pub lambda: f64,
    pub residual_norm: f64,
    pub roughness_norm: f64,
    pub effective_dof: f64,
    pub gcv_score: f64,
}

/// Resolved continuous mode (local peak lobe) extracted from the non-negative `ILT` spectrum $f(D)$.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IltModePeak {
    /// Mode peak diffusivity $D$ (`m^2 / s`) computed as log-weighted lobe centroid.
    pub d_m2_s: f64,
    /// Mode centroid in $\log_{10}(D / \text{m}^2\text{s}^{-1})$.
    pub log10_d: f64,
    /// Fractional area of this lobe relative to $\sum_j f_j$.
    pub area_fraction: f64,
    /// Lobe standard deviation along $\log_{10} D$.
    pub sd_log10_d: f64,
}

/// Output of the opt-in continuous Tikhonov-regularized Inverse Laplace Transform (`ILT`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IltResult {
    /// Log-spaced diffusivity grid $D_j$ (`m^2 / s`, length `n_grid`).
    pub d_grid_m2_s: Vec<f64>,
    /// Uniform grid in $\log_{10}(D_j / \text{m}^2\text{s}^{-1})$.
    pub log10_d_grid: Vec<f64>,
    /// Non-negative diffusion spectrum $f_j \ge 0$ in intensity units ($I(b) \approx \sum_j f_j e^{-b D_j}$).
    pub spectrum: Vec<f64>,
    /// Selected regularization weight $\lambda^\star$.
    pub selected_lambda: f64,
    /// Forward-reconstructed decay curve $\hat{y}_m = (K f)_m$.
    pub fitted_curve: Vec<f64>,
    /// Residuals $r_m = y_m - \hat{y}_m$.
    pub residuals: Vec<f64>,
    /// Detected continuous lobes sorted by ascending $D$.
    pub modes: Vec<IltModePeak>,
    /// Full diagnostic $\lambda$-sweep (`GCV` / L-curve inspection).
    pub lambda_sweep: Vec<IltSweepPoint>,
    /// Mandatory caveat describing ill-posedness and polydisperse/overlap artifact sensitivity.
    pub caveat: String,
}

/// Active-set Lawson & Hanson (1974) Non-Negative Least Squares (`NNLS`) solver minimizing
/// $\|A x - b\|_2^2$ subject to $x_j \ge 0$, where `a_row_major` has dimensions `n_rows x n_cols`.
pub fn solve_nnls_lawson_hanson(
    a_row_major: &[f64],
    b_rhs: &[f64],
    n_rows: usize,
    n_cols: usize,
) -> Result<Vec<f64>, DosyError> {
    if a_row_major.len() != n_rows * n_cols || b_rhs.len() != n_rows {
        return Err(DosyError::new("NNLS matrix/vector dimension mismatch"));
    }
    // Precompute Gram matrix G = A^T A and c = A^T b for fast inner passive-set Cholesky/LU solves
    let mut g_mat = vec![0.0; n_cols * n_cols];
    let mut at_b = vec![0.0; n_cols];
    for i in 0..n_cols {
        let mut s_b = 0.0;
        for r in 0..n_rows {
            s_b += a_row_major[r * n_cols + i] * b_rhs[r];
        }
        at_b[i] = s_b;
        for j in i..n_cols {
            let mut s_g = 0.0;
            for r in 0..n_rows {
                s_g += a_row_major[r * n_cols + i] * a_row_major[r * n_cols + j];
            }
            g_mat[i * n_cols + j] = s_g;
            g_mat[j * n_cols + i] = s_g;
        }
    }

    let mut x = vec![0.0; n_cols];
    let mut passive = vec![false; n_cols];
    let tol = 1.0e-11 * at_b.iter().fold(1.0_f64, |acc, &v| acc.max(v.abs()));
    let max_outer = (5 * n_cols).max(50);

    for _outer in 0..max_outer {
        // Dual gradient w = A^T (b - A x) = at_b - G x
        let mut best_j = None;
        let mut best_w = tol;
        for j in 0..n_cols {
            if passive[j] {
                continue;
            }
            let mut gx = 0.0;
            for k in 0..n_cols {
                if x[k] != 0.0 {
                    gx += g_mat[j * n_cols + k] * x[k];
                }
            }
            let wj = at_b[j] - gx;
            if wj > best_w {
                best_w = wj;
                best_j = Some(j);
            }
        }
        let Some(enter_j) = best_j else {
            break;
        };
        passive[enter_j] = true;

        for _inner in 0..(3 * n_cols + 8) {
            let p_indices: Vec<usize> = (0..n_cols).filter(|&j| passive[j]).collect();
            let p_len = p_indices.len();
            if p_len == 0 {
                break;
            }
            let mut sub_g = vec![0.0; p_len * p_len];
            let mut sub_rhs = vec![0.0; p_len];
            for (ri, &pi) in p_indices.iter().enumerate() {
                sub_rhs[ri] = at_b[pi];
                for (ci, &pj) in p_indices.iter().enumerate() {
                    sub_g[ri * p_len + ci] = g_mat[pi * n_cols + pj];
                }
                sub_g[ri * p_len + ri] += 1.0e-14;
            }

            let Some(z_sub) = solve_stable_dense(p_len, &sub_g, &sub_rhs) else {
                passive[enter_j] = false;
                break;
            };

            let mut z = vec![0.0; n_cols];
            let mut min_z = f64::INFINITY;
            for (ri, &pi) in p_indices.iter().enumerate() {
                z[pi] = z_sub[ri];
                min_z = min_z.min(z_sub[ri]);
            }

            if min_z > 0.0 {
                x = z;
                break;
            }

            let mut alpha = 1.0_f64;
            for &pi in &p_indices {
                if z[pi] <= 0.0 {
                    let ratio = x[pi] / (x[pi] - z[pi]).max(1.0e-30);
                    if ratio < alpha {
                        alpha = ratio;
                    }
                }
            }
            for j in 0..n_cols {
                if passive[j] {
                    x[j] += alpha * (z[j] - x[j]);
                    if x[j] <= 1.0e-14 {
                        x[j] = 0.0;
                        passive[j] = false;
                    }
                } else {
                    x[j] = 0.0;
                }
            }
        }
    }

    Ok(x)
}

fn solve_stable_dense(n: usize, a: &[f64], b: &[f64]) -> Option<Vec<f64>> {
    solve_small_linear(n, a, b)
}

fn build_regularization_matrix(n: usize, reg: TikhonovRegularizer) -> Vec<f64> {
    let mut l = vec![0.0; n * n];
    match reg {
        TikhonovRegularizer::Identity => {
            for i in 0..n {
                l[i * n + i] = 1.0;
            }
        }
        TikhonovRegularizer::SecondDerivative => {
            // Gentle endpoint boundary stabilization + interior `[1, -2, 1]` second-difference operator
            l[0] = 0.5;
            for r in 1..(n - 1) {
                l[r * n + (r - 1)] = 1.0;
                l[r * n + r] = -2.0;
                l[r * n + (r + 1)] = 1.0;
            }
            l[(n - 1) * n + (n - 1)] = 0.5;
        }
    }
    l
}

/// Solve the opt-in continuous Tikhonov-regularized Inverse Laplace Transform (`ILT`):
/// $\min_{f \ge 0} \|K f - y\|_2^2 + \lambda \|L f\|_2^2$ across a log-spaced diffusivity grid,
/// selecting $\lambda$ via `GCV`, `L-curve`, `DiscrepancyPrinciple`, or a fixed user weight.
pub fn solve_tikhonov_ilt(
    b_s_m2: &[f64],
    intensities: &[f64],
    cfg: &IltConfig,
) -> Result<IltResult, DosyError> {
    let m = b_s_m2.len();
    if m != intensities.len() || m < 4 {
        return Err(DosyError::new(format!(
            "Tikhonov ILT requires matching b/intensity arrays of length >= 4, got {m}"
        )));
    }
    if !b_s_m2.iter().all(|b| b.is_finite() && *b >= 0.0) {
        return Err(DosyError::new(
            "Tikhonov ILT requires finite non-negative b-values",
        ));
    }
    if !intensities.iter().all(|y| y.is_finite()) {
        return Err(DosyError::new("Tikhonov ILT requires finite intensities"));
    }
    let n = cfg.n_grid;
    if !(16..=512).contains(&n) {
        return Err(DosyError::new(format!(
            "IltConfig.n_grid ({n}) must be in 16..=512"
        )));
    }
    if !cfg.d_min_m2_s.is_finite()
        || !cfg.d_max_m2_s.is_finite()
        || cfg.d_min_m2_s <= 0.0
        || cfg.d_max_m2_s <= cfg.d_min_m2_s
    {
        return Err(DosyError::new("invalid diffusivity bounds in IltConfig"));
    }

    let y_scale = intensities.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    if y_scale <= 0.0 {
        return Err(DosyError::new("all intensities are zero in Tikhonov ILT"));
    }
    let y_norm: Vec<f64> = intensities.iter().map(|&v| v / y_scale).collect();

    let log_min = cfg.d_min_m2_s.log10();
    let log_max = cfg.d_max_m2_s.log10();
    let mut log10_d_grid = Vec::with_capacity(n);
    let mut d_grid_m2_s = Vec::with_capacity(n);
    for j in 0..n {
        let lj = log_min + (j as f64) * (log_max - log_min) / ((n - 1) as f64);
        log10_d_grid.push(lj);
        d_grid_m2_s.push(10.0_f64.powf(lj));
    }

    // Kernel matrix K (M x N)
    let mut k_mat = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            k_mat[i * n + j] = (-b_s_m2[i] * d_grid_m2_s[j]).exp();
        }
    }
    let l_mat = build_regularization_matrix(n, cfg.regularizer);

    let lambdas: Vec<f64> = match cfg.lambda_strategy {
        IltLambdaStrategy::Fixed(lam) => {
            if !lam.is_finite() || lam <= 0.0 {
                return Err(DosyError::new(format!("invalid fixed ILT lambda = {lam}")));
            }
            vec![lam]
        }
        _ => {
            let n_sw = cfg.n_lambda_sweep.max(5);
            let l_lo = cfg.lambda_min.max(1.0e-10).ln();
            let l_hi = cfg.lambda_max.max(cfg.lambda_min * 10.0).ln();
            (0..n_sw)
                .map(|s| (l_lo + (s as f64) * (l_hi - l_lo) / ((n_sw - 1) as f64)).exp())
                .collect()
        }
    };

    let n_aug = m + n;
    let mut aug_a = vec![0.0; n_aug * n];
    let mut aug_b = vec![0.0; n_aug];
    aug_b[..m].copy_from_slice(&y_norm);
    aug_a[..(m * n)].copy_from_slice(&k_mat);

    let mut sweep_points = Vec::with_capacity(lambdas.len());
    let mut solutions = Vec::with_capacity(lambdas.len());

    for &lam in &lambdas {
        let sqrt_lam = lam.sqrt();
        for r in 0..n {
            for c in 0..n {
                aug_a[(m + r) * n + c] = sqrt_lam * l_mat[r * n + c];
            }
        }
        let f_norm = solve_nnls_lawson_hanson(&aug_a, &aug_b, n_aug, n)?;

        // Residual ||K f - y||_2 and roughness ||L f||_2 (in normalized y units)
        let mut rss_norm = 0.0;
        for i in 0..m {
            let mut pred = 0.0;
            for j in 0..n {
                pred += k_mat[i * n + j] * f_norm[j];
            }
            let r = pred - y_norm[i];
            rss_norm += r * r;
        }
        let mut rough_sq = 0.0;
        for r in 0..n {
            let mut lf = 0.0;
            for c in 0..n {
                lf += l_mat[r * n + c] * f_norm[c];
            }
            rough_sq += lf * lf;
        }

        // Effective DOF on active positive support P = {j : f_j > 0}
        let active: Vec<usize> = (0..n).filter(|&j| f_norm[j] > 1.0e-10).collect();
        let eff_dof = if active.is_empty() {
            0.0
        } else {
            let p = active.len();
            let mut kt_k = vec![0.0; p * p];
            let mut h_p = vec![0.0; p * p];
            for (ia, &ja) in active.iter().enumerate() {
                for (ib, &jb) in active.iter().enumerate() {
                    let mut sk = 0.0;
                    for i in 0..m {
                        sk += k_mat[i * n + ja] * k_mat[i * n + jb];
                    }
                    let mut sl = 0.0;
                    for r in 0..n {
                        sl += l_mat[r * n + ja] * l_mat[r * n + jb];
                    }
                    kt_k[ia * p + ib] = sk;
                    h_p[ia * p + ib] = sk + lam * sl;
                }
                h_p[ia * p + ia] += 1.0e-12;
            }
            // Trace( (K_P^T K_P + lam L_P^T L_P)^{-1} K_P^T K_P )
            let mut tr = 0.0;
            for col in 0..p {
                let rhs: Vec<f64> = (0..p).map(|r| kt_k[r * p + col]).collect();
                if let Some(sol) = solve_stable_dense(p, &h_p, &rhs) {
                    tr += sol[col];
                }
            }
            tr.clamp(0.0, (m as f64) * 0.95)
        };

        let m_f = m as f64;
        let denom = (1.0 - eff_dof / m_f).max(0.05);
        let gcv = (rss_norm * y_scale * y_scale / m_f) / (denom * denom);

        sweep_points.push(IltSweepPoint {
            lambda: lam,
            residual_norm: rss_norm.sqrt() * y_scale,
            roughness_norm: rough_sq.sqrt() * y_scale,
            effective_dof: eff_dof,
            gcv_score: gcv,
        });
        solutions.push(f_norm);
    }

    let chosen_idx = match cfg.lambda_strategy {
        IltLambdaStrategy::Fixed(_) => 0,
        IltLambdaStrategy::Gcv => sweep_points
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.gcv_score.total_cmp(&b.1.gcv_score))
            .map(|(idx, _)| idx)
            .unwrap_or(0),
        IltLambdaStrategy::DiscrepancyPrinciple { noise_sigma } => {
            let target = (m as f64).sqrt() * noise_sigma.abs();
            // Largest lambda whose residual_norm <= target (iterating from largest lambda down)
            sweep_points
                .iter()
                .enumerate()
                .rev()
                .find(|(_, pt)| pt.residual_norm <= target)
                .map(|(idx, _)| idx)
                .unwrap_or(0)
        }
        IltLambdaStrategy::LCurve => select_l_curve_corner(&sweep_points),
    };

    let selected_lambda = sweep_points[chosen_idx].lambda;
    let spectrum: Vec<f64> = solutions[chosen_idx].iter().map(|&v| v * y_scale).collect();

    let mut fitted_curve = Vec::with_capacity(m);
    let mut residuals = Vec::with_capacity(m);
    for i in 0..m {
        let mut pred = 0.0;
        for j in 0..n {
            pred += k_mat[i * n + j] * spectrum[j];
        }
        fitted_curve.push(pred);
        residuals.push(intensities[i] - pred);
    }

    let modes = extract_ilt_modes(&log10_d_grid, &spectrum);

    Ok(IltResult {
        d_grid_m2_s,
        log10_d_grid,
        spectrum,
        selected_lambda,
        fitted_curve,
        residuals,
        modes,
        lambda_sweep: sweep_points,
        caveat: ILT_ARTIFACT_CAVEAT.to_string(),
    })
}

fn select_l_curve_corner(pts: &[IltSweepPoint]) -> usize {
    if pts.len() < 3 {
        return 0;
    }
    let xi: Vec<f64> = pts
        .iter()
        .map(|p| p.residual_norm.max(1.0e-30).ln())
        .collect();
    let eta: Vec<f64> = pts
        .iter()
        .map(|p| p.roughness_norm.max(1.0e-30).ln())
        .collect();

    let mut best_idx = pts.len() / 2;
    let mut best_curv = f64::NEG_INFINITY;
    for i in 1..(pts.len() - 1) {
        let dx1 = xi[i] - xi[i - 1];
        let dy1 = eta[i] - eta[i - 1];
        let dx2 = xi[i + 1] - xi[i];
        let dy2 = eta[i + 1] - eta[i];
        let dx3 = xi[i + 1] - xi[i - 1];
        let dy3 = eta[i + 1] - eta[i - 1];
        let d1 = (dx1 * dx1 + dy1 * dy1).sqrt();
        let d2 = (dx2 * dx2 + dy2 * dy2).sqrt();
        let d3 = (dx3 * dx3 + dy3 * dy3).sqrt();
        let denom = d1 * d2 * d3;
        if denom > 1.0e-12 {
            // As lambda increases, xi increases and eta decreases; positive L-curve bend has cross > 0
            let cross = dx1 * dy2 - dy1 * dx2;
            let curv = 2.0 * cross / denom;
            if curv > best_curv {
                best_curv = curv;
                best_idx = i;
            }
        }
    }
    best_idx
}

fn extract_ilt_modes(log10_d: &[f64], spectrum: &[f64]) -> Vec<IltModePeak> {
    let n = spectrum.len();
    let total_area: f64 = spectrum.iter().sum();
    if n == 0 || total_area <= 1.0e-30 {
        return Vec::new();
    }
    let max_val = spectrum.iter().fold(0.0_f64, |a, &b| a.max(b));
    let floor = 0.02 * max_val;

    // Partition contiguous above-floor segments; also split at deep interior valleys (< 35% of both flanking maxima)
    let mut raw_segments: Vec<(usize, usize)> = Vec::new();
    let mut idx = 0usize;
    while idx < n {
        if spectrum[idx] > floor {
            let start = idx;
            while idx < n && spectrum[idx] > floor {
                idx += 1;
            }
            raw_segments.push((start, idx - 1));
        } else {
            idx += 1;
        }
    }

    let mut lobes: Vec<(usize, usize)> = Vec::new();
    for (s, e) in raw_segments {
        let mut sub_start = s;
        for j in (s + 1)..e {
            if spectrum[j] < spectrum[j - 1] && spectrum[j] <= spectrum[j + 1] {
                let left_max = spectrum[sub_start..j]
                    .iter()
                    .fold(0.0_f64, |a, &b| a.max(b));
                let right_max = spectrum[(j + 1)..=e].iter().fold(0.0_f64, |a, &b| a.max(b));
                if spectrum[j] < 0.50 * left_max.min(right_max) {
                    lobes.push((sub_start, j));
                    sub_start = j;
                }
            }
        }
        lobes.push((sub_start, e));
    }

    let step_log = if n > 1 {
        (log10_d[n - 1] - log10_d[0]) / ((n - 1) as f64)
    } else {
        0.03
    };

    let mut modes = Vec::new();
    for (s, e) in lobes {
        let area: f64 = spectrum[s..=e].iter().sum();
        if area / total_area < 0.02 {
            continue;
        }
        let mean_log = spectrum[s..=e]
            .iter()
            .enumerate()
            .map(|(off, &w)| w * log10_d[s + off])
            .sum::<f64>()
            / area;
        let var_log = spectrum[s..=e]
            .iter()
            .enumerate()
            .map(|(off, &w)| {
                let dl = log10_d[s + off] - mean_log;
                w * dl * dl
            })
            .sum::<f64>()
            / area;
        modes.push(IltModePeak {
            d_m2_s: 10.0_f64.powf(mean_log),
            log10_d: mean_log,
            area_fraction: area / total_area,
            sd_log10_d: var_log.sqrt().max(0.5 * step_log),
        });
    }
    modes.sort_by(|a, b| a.d_m2_s.total_cmp(&b.d_m2_s));
    modes
}

// ============================================================================
// 4. Pseudo-2D DOSY Grid Synthesizer (`(ppm, log10(D))` + 1D Top Projection)
// ============================================================================

/// Configuration for synthesizing a 2D `(\delta\text{ ppm}, \log_{10} D)` DOSY spectrum.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DosyGridConfig {
    /// Number of rows along the indirect $\log_{10}(D / \text{m}^2\text{s}^{-1})$ axis.
    pub n_log10_d: usize,
    /// Lower bound of $\log_{10}(D / \text{m}^2\text{s}^{-1})$ (e.g. `-11.0`).
    pub log10_d_min: f64,
    /// Upper bound of $\log_{10}(D / \text{m}^2\text{s}^{-1})$ (e.g. `-8.5`).
    pub log10_d_max: f64,
    /// Minimum Gaussian spot width $\sigma_{\log_{10} D}$ preventing sub-row aliasing.
    pub min_sigma_log10_d: f64,
    /// Maximum Gaussian spot width $\sigma_{\log_{10} D}$.
    pub max_sigma_log10_d: f64,
}

impl Default for DosyGridConfig {
    fn default() -> Self {
        Self {
            n_log10_d: 128,
            log10_d_min: -11.0,
            log10_d_max: -8.5,
            min_sigma_log10_d: 0.025,
            max_sigma_log10_d: 0.30,
        }
    }
}

/// Fitted 1D peak entry carrying its chemical-shift profile and discrete/continuous diffusion fits.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DosyPeakEntry {
    /// Human-readable peak label (e.g. `"Peak 1 (7.32 ppm)"`).
    pub label: String,
    /// Peak center chemical shift $\delta$ (`ppm`).
    pub center_ppm: f64,
    /// Half-width at half-maximum (`ppm`) used when rendering peak-based 2D spots.
    pub fwhm_ppm: f64,
    /// Experimental decay intensities $I(b_m)$ across the gradient list.
    pub intensities: Vec<f64>,
    /// Multi-exponential `AIC`/`BIC` fit summary.
    pub multi_exp: DosyMultiExpSelection,
    /// Optional continuous Tikhonov `ILT` inversion for side-by-side inspection.
    pub ilt: Option<IltResult>,
}

/// Synthesized pseudo-2D `(ppm, log10(D))` DOSY matrix with aligned 1D top projection.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DosyGrid2D {
    /// Direct chemical-shift axis $\delta$ (`ppm`, length `n_ppm`).
    pub ppm_axis: Vec<f64>,
    /// Indirect diffusion axis $\log_{10}(D / \text{m}^2\text{s}^{-1})$ (ascending, length `n_log10_d`).
    pub log10_d_axis: Vec<f64>,
    /// Aligned 1D spectrum top projection (length `n_ppm`).
    pub top_projection_1d: Vec<f64>,
    /// Row-major 2D intensity grid (`n_log10_d` rows $\times$ `n_ppm` columns).
    pub data: Vec<f64>,
}

impl DosyGrid2D {
    /// Read intensity at `(row_log_d, col_ppm)`.
    #[inline]
    pub fn get(&self, row_log_d: usize, col_ppm: usize) -> f64 {
        self.data[row_log_d * self.ppm_axis.len() + col_ppm]
    }
}

/// Synthesize a pseudo-2D `(ppm, log10(D))` DOSY spectrum from per-frequency-column discrete fits
/// (Johnson, *Prog. Nucl. Magn. Reson. Spectrosc.* 1999, 34, 203–256).
pub fn synthesize_dosy_grid_from_columns(
    ppm_axis: &[f64],
    top_spectrum_1d: &[f64],
    column_fits: &[Option<DosyExpModelFit>],
    cfg: &DosyGridConfig,
) -> Result<DosyGrid2D, DosyError> {
    let n_ppm = ppm_axis.len();
    if n_ppm == 0 || top_spectrum_1d.len() != n_ppm || column_fits.len() != n_ppm {
        return Err(DosyError::new(
            "ppm_axis, top_spectrum_1d, and column_fits must have identical non-zero length",
        ));
    }
    if cfg.n_log10_d < 4 || cfg.log10_d_max <= cfg.log10_d_min {
        return Err(DosyError::new("invalid DosyGridConfig log10(D) axis"));
    }

    let n_rows = cfg.n_log10_d;
    let log10_d_axis: Vec<f64> = (0..n_rows)
        .map(|r| {
            cfg.log10_d_min
                + (r as f64) * (cfg.log10_d_max - cfg.log10_d_min) / ((n_rows - 1) as f64)
        })
        .collect();

    let mut data = vec![0.0; n_rows * n_ppm];
    let inv_sqrt_2pi = 1.0 / (2.0 * PI).sqrt();

    for col in 0..n_ppm {
        let top_i = top_spectrum_1d[col].max(0.0);
        let Some(fit) = &column_fits[col] else {
            continue;
        };
        if top_i <= 0.0 {
            continue;
        }
        for comp in &fit.components {
            let mu = comp.log10_d;
            let sig = comp
                .sd_log10_d
                .clamp(cfg.min_sigma_log10_d, cfg.max_sigma_log10_d);
            let weight = top_i * comp.fraction * inv_sqrt_2pi / sig;
            let inv_2sig2 = 0.5 / (sig * sig);
            for (r, &ld) in log10_d_axis.iter().enumerate() {
                let dl = ld - mu;
                data[r * n_ppm + col] += weight * (-dl * dl * inv_2sig2).exp();
            }
        }
    }

    Ok(DosyGrid2D {
        ppm_axis: ppm_axis.to_vec(),
        log10_d_axis,
        top_projection_1d: top_spectrum_1d.to_vec(),
        data,
    })
}

/// Synthesize a pseudo-2D `(ppm, log10(D))` DOSY spectrum from resolved [`DosyPeakEntry`] items,
/// supporting either discrete multi-exponential Gaussian projection (`use_ilt = false`) or
/// continuous Tikhonov `ILT` projection (`use_ilt = true`).
pub fn synthesize_dosy_grid_from_peaks(
    ppm_axis: &[f64],
    top_spectrum_1d: &[f64],
    peaks: &[DosyPeakEntry],
    cfg: &DosyGridConfig,
    use_ilt: bool,
) -> Result<DosyGrid2D, DosyError> {
    let n_ppm = ppm_axis.len();
    if n_ppm == 0 || top_spectrum_1d.len() != n_ppm {
        return Err(DosyError::new(
            "ppm_axis and top_spectrum_1d must have identical non-zero length",
        ));
    }
    if cfg.n_log10_d < 4 || cfg.log10_d_max <= cfg.log10_d_min {
        return Err(DosyError::new("invalid DosyGridConfig log10(D) axis"));
    }

    let n_rows = cfg.n_log10_d;
    let log10_d_axis: Vec<f64> = (0..n_rows)
        .map(|r| {
            cfg.log10_d_min
                + (r as f64) * (cfg.log10_d_max - cfg.log10_d_min) / ((n_rows - 1) as f64)
        })
        .collect();
    let mut data = vec![0.0; n_rows * n_ppm];
    let inv_sqrt_2pi = 1.0 / (2.0 * PI).sqrt();

    for pk in peaks {
        let hwhm = (0.5 * pk.fwhm_ppm).max(1.0e-4);
        let i0_total: f64 = pk
            .multi_exp
            .best_fit
            .components
            .iter()
            .map(|c| c.amplitude)
            .sum::<f64>()
            .max(0.0);
        if i0_total <= 0.0 {
            continue;
        }

        // Precompute horizontal Lorentzian-Gaussian envelope along ppm_axis
        let mut x_env = vec![0.0; n_ppm];
        for (c, &ppm) in ppm_axis.iter().enumerate() {
            let dx = (ppm - pk.center_ppm) / hwhm;
            if dx.abs() <= 12.0 {
                // 50/50 pseudo-Voigt horizontal peak shape
                let lor = 1.0 / (1.0 + dx * dx);
                let gau = (-std::f64::consts::LN_2 * dx * dx).exp();
                x_env[c] = 0.5 * (lor + gau);
            }
        }

        // Precompute vertical log10(D) profile for this peak
        let mut y_prof = vec![0.0; n_rows];
        if use_ilt {
            if let Some(ilt) = &pk.ilt {
                let spec_sum: f64 = ilt.spectrum.iter().sum();
                if spec_sum > 0.0 && ilt.log10_d_grid.len() >= 2 {
                    for (r, &ld) in log10_d_axis.iter().enumerate() {
                        y_prof[r] = i0_total * interp_linear(&ilt.log10_d_grid, &ilt.spectrum, ld)
                            / spec_sum;
                    }
                }
            }
        }
        if y_prof.iter().all(|&v| v == 0.0) {
            for comp in &pk.multi_exp.best_fit.components {
                let mu = comp.log10_d;
                let sig = comp
                    .sd_log10_d
                    .clamp(cfg.min_sigma_log10_d, cfg.max_sigma_log10_d);
                let w = i0_total * comp.fraction * inv_sqrt_2pi / sig;
                let inv_2s2 = 0.5 / (sig * sig);
                for (r, &ld) in log10_d_axis.iter().enumerate() {
                    let dl = ld - mu;
                    y_prof[r] += w * (-dl * dl * inv_2s2).exp();
                }
            }
        }

        for r in 0..n_rows {
            let yr = y_prof[r];
            if yr <= 1.0e-12 * i0_total {
                continue;
            }
            for c in 0..n_ppm {
                if x_env[c] > 0.0 {
                    data[r * n_ppm + c] += yr * x_env[c];
                }
            }
        }
    }

    Ok(DosyGrid2D {
        ppm_axis: ppm_axis.to_vec(),
        log10_d_axis,
        top_projection_1d: top_spectrum_1d.to_vec(),
        data,
    })
}

fn interp_linear(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    if x <= xs[0] || x >= xs[xs.len() - 1] {
        return 0.0;
    }
    let pos = xs.partition_point(|&v| v < x);
    if pos == 0 {
        return ys[0];
    }
    let x0 = xs[pos - 1];
    let x1 = xs[pos];
    let t = (x - x0) / (x1 - x0).max(1.0e-30);
    ys[pos - 1] * (1.0 - t) + ys[pos] * t
}

// ============================================================================
// 5. Stokes–Einstein Hydrodynamics & Empirical Power-Law MW Estimation
// ============================================================================

/// Built-in NMR solvents with calibrated temperature-dependent dynamic viscosities $\eta(T)$ (`Pa s`).
///
/// Reference values at $T_0 = 298.15\text{ K}$ and Arrhenius activation parameters $E_a / R$ from
/// Holz et al. (*Phys. Chem. Chem. Phys.* 2000, 2, 4740–4742), Hardy & Cottington (*J. Res. Natl.
/// Bur. Stand.* 1949, 42, 573–578), and Evans et al. (*Angew. Chem. Int. Ed.* 2013, 52, 3199–3202).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DosySolvent {
    D2O,
    H2O,
    CDCl3,
    DmsoD6,
    CD3OD,
    CD3CN,
}

impl DosySolvent {
    pub const ALL: [Self; 6] = [
        Self::D2O,
        Self::H2O,
        Self::CDCl3,
        Self::DmsoD6,
        Self::CD3OD,
        Self::CD3CN,
    ];

    /// Canonical display/CLI name (`"D2O"`, `"H2O"`, `"CDCl3"`, `"DMSO-d6"`, `"CD3OD"`, `"CD3CN"`).
    pub const fn label(self) -> &'static str {
        match self {
            Self::D2O => "D2O",
            Self::H2O => "H2O",
            Self::CDCl3 => "CDCl3",
            Self::DmsoD6 => "DMSO-d6",
            Self::CD3OD => "CD3OD",
            Self::CD3CN => "CD3CN",
        }
    }

    /// Parse solvent identifier case-insensitively.
    pub fn from_name(name: &str) -> Option<Self> {
        let up = name.trim().to_ascii_uppercase().replace(['-', '_'], "");
        match up.as_str() {
            "D2O" => Some(Self::D2O),
            "H2O" | "WATER" => Some(Self::H2O),
            "CDCL3" | "CHLOROFORMD" => Some(Self::CDCl3),
            "DMSOD6" | "DMSO" => Some(Self::DmsoD6),
            "CD3OD" | "METHANOLD4" | "MEODD4" => Some(Self::CD3OD),
            "CD3CN" | "ACETONITRILED3" | "MECND3" => Some(Self::CD3CN),
            _ => None,
        }
    }

    /// Solvent molecular weight (`g / mol`) for Gierer–Wirtz microfriction calculations.
    pub const fn solvent_mw_da(self) -> f64 {
        match self {
            Self::D2O => 20.028,
            Self::H2O => 18.015,
            Self::CDCl3 => 120.384,
            Self::DmsoD6 => 84.170,
            Self::CD3OD => 36.067,
            Self::CD3CN => 44.071,
        }
    }

    /// Dynamic viscosity $\eta(T)$ in SI `Pa s` (`kg m^-1 s^-1`) at absolute temperature `temp_k` (`K`),
    /// using Andrade/Arrhenius relation $\eta(T) = \eta(298.15\text{ K}) \exp\!\bigl((E_a/R)(1/T - 1/298.15)\bigr)$.
    pub fn viscosity_pa_s(self, temp_k: f64) -> Result<f64, DosyError> {
        if !temp_k.is_finite() || !(200.0..=450.0).contains(&temp_k) {
            return Err(DosyError::new(format!(
                "temperature T = {temp_k} K outside calibrated liquid range [200, 450] K"
            )));
        }
        let (eta_298_pa_s, b_kelvin) = match self {
            Self::D2O => (1.099e-3, 2195.0),
            Self::H2O => (0.8900e-3, 2058.0),
            Self::CDCl3 => (0.542e-3, 820.0),
            Self::DmsoD6 => (2.195e-3, 1690.0),
            Self::CD3OD => (0.585e-3, 1280.0),
            Self::CD3CN => (0.356e-3, 860.0),
        };
        let t_ref = 298.15_f64;
        Ok(eta_298_pa_s * (b_kelvin * (1.0 / temp_k - 1.0 / t_ref)).exp())
    }
}

/// Solute structural compactness class selecting empirical power-law exponent $\beta$ in
/// $D = \alpha \, \mathrm{MW}^{-\beta}$ (Evans et al., *Angew. Chem. Int. Ed.* 2013, 52, 3199–3202).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SoluteCompactnessClass {
    /// Small-to-medium organic molecules (`100 - 2,000 Da`, effective $\beta = 0.55$).
    SmallMolecule,
    /// Folded globular proteins (`3 - 100 kDa`, hydrated compact sphere scaling $\beta = 0.355$).
    GlobularProtein,
    /// Expanded polymers / intrinsically disordered chains ($\beta = 0.58$).
    ExpandedPolymer,
}

impl SoluteCompactnessClass {
    pub const fn label(self) -> &'static str {
        match self {
            Self::SmallMolecule => "Small Molecule",
            Self::GlobularProtein => "Globular Protein",
            Self::ExpandedPolymer => "Expanded Polymer / IDP",
        }
    }
}

/// Empirical power-law diffusivity-to-MW calibration $D = \alpha \cdot \mathrm{MW}^{-\beta}$
/// (`D` in `m^2 / s`, `MW` in `Da` = `g / mol`).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PowerLawCalibration {
    pub alpha: f64,
    pub beta: f64,
}

impl PowerLawCalibration {
    /// Build viscosity- and temperature-compensated empirical power-law parameters $(\alpha, \beta)$
    /// anchored at $T_0 = 298.15\text{ K}$ and scaled by $(T / T_0) \cdot (\eta_{\text{ref}} / \eta(T))$
    /// per Evans et al. (*Angew. Chem. Int. Ed.* 2013, 52, 3199–3202).
    pub fn for_solvent(
        solvent: DosySolvent,
        temp_k: f64,
        class: SoluteCompactnessClass,
    ) -> Result<Self, DosyError> {
        let eta = solvent.viscosity_pa_s(temp_k)?;
        let scale = (temp_k / 298.15) * (1.0e-3 / eta);
        let (alpha_ref, beta) = match class {
            // Anchored so MW = 300 Da gives D = 5.2e-10 m^2/s at eta = 1.0 mPa*s, T = 298.15 K
            SoluteCompactnessClass::SmallMolecule => (1.197_4e-8, 0.55),
            // Anchored so native Lysozyme (14.3 kDa) gives D = 1.03e-10 m^2/s in D2O at 298.15 K
            SoluteCompactnessClass::GlobularProtein => (3.388e-9, 0.355),
            // Expanded flexible chain
            SoluteCompactnessClass::ExpandedPolymer => (1.58e-8, 0.58),
        };
        Ok(Self {
            alpha: alpha_ref * scale,
            beta,
        })
    }

    /// Calibrate $(\alpha, \beta)$ directly from $\ge 2$ internal/external `(mw_da, d_m2_s)` standards
    /// via log-log ordinary least squares ($\ln D = \ln \alpha - \beta \ln \mathrm{MW}$).
    pub fn fit_from_standards(standards: &[(f64, f64)]) -> Result<Self, DosyError> {
        if standards.len() < 2 {
            return Err(DosyError::new(
                "power-law MW calibration requires >= 2 (mw_da, d_m2_s) standards",
            ));
        }
        let n = standards.len() as f64;
        let mut sx = 0.0;
        let mut sy = 0.0;
        let mut sxx = 0.0;
        let mut sxy = 0.0;
        for &(mw, d) in standards {
            if !mw.is_finite() || mw <= 0.0 || !d.is_finite() || d <= 0.0 {
                return Err(DosyError::new(
                    "standards must have positive finite MW and D",
                ));
            }
            let x = mw.ln();
            let y = d.ln();
            sx += x;
            sy += y;
            sxx += x * x;
            sxy += x * y;
        }
        let denom = n * sxx - sx * sx;
        if denom.abs() < 1.0e-15 {
            return Err(DosyError::new(
                "degenerate MW standards (all molecular weights identical)",
            ));
        }
        let slope = (n * sxy - sx * sy) / denom;
        let intercept = (sy - slope * sx) / n;
        let beta = -slope;
        if beta <= 0.0 {
            return Err(DosyError::new(format!(
                "non-physical negative power-law exponent beta = {beta:.4}"
            )));
        }
        Ok(Self {
            alpha: intercept.exp(),
            beta,
        })
    }

    /// Estimate molecular weight (`Da`) from diffusion coefficient $D$ (`m^2 / s`):
    /// $\mathrm{MW} = (\alpha / D)^{1 / \beta}$.
    pub fn estimate_mw_da(self, d_m2_s: f64) -> Result<f64, DosyError> {
        if !d_m2_s.is_finite() || d_m2_s <= 0.0 {
            return Err(DosyError::new(format!(
                "invalid diffusion coefficient D = {d_m2_s} m^2/s"
            )));
        }
        if !self.alpha.is_finite()
            || self.alpha <= 0.0
            || !self.beta.is_finite()
            || self.beta <= 0.0
        {
            return Err(DosyError::new("invalid PowerLawCalibration parameters"));
        }
        Ok((self.alpha / d_m2_s).powf(1.0 / self.beta))
    }

    /// Predict diffusion coefficient $D$ (`m^2 / s`) from molecular weight (`Da`).
    pub fn predict_d_m2_s(self, mw_da: f64) -> Result<f64, DosyError> {
        if !mw_da.is_finite() || mw_da <= 0.0 {
            return Err(DosyError::new(format!(
                "invalid molecular weight MW = {mw_da}"
            )));
        }
        Ok(self.alpha * mw_da.powf(-self.beta))
    }
}

/// Complete Stokes–Einstein hydrodynamic summary for a single diffusion coefficient $D$.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct HydrodynamicEstimate {
    /// Input translational diffusion coefficient $D$ (`m^2 / s`).
    pub d_m2_s: f64,
    /// Sample temperature $T$ (`K`).
    pub temp_k: f64,
    /// Dynamic solvent viscosity $\eta(T)$ (`Pa s`).
    pub viscosity_pa_s: f64,
    /// Classical Stokes–Einstein hydrodynamic radius $R_h = k_B T / (6 \pi \eta D)$ (`m`).
    pub r_h_m: f64,
    /// Hydrodynamic radius $R_h$ in Ångströms (`10^-10 m`).
    pub r_h_angstrom: f64,
    /// Hydrodynamic radius $R_h$ in nanometers (`10^-9 m`).
    pub r_h_nm: f64,
    /// Empirical power-law estimated molecular weight (`Da` = `g / mol`).
    pub estimated_mw_da: f64,
}

/// Compute exact Stokes–Einstein hydrodynamic radius $R_h = k_B T / (6 \pi \eta D)$ (`m`).
pub fn stokes_einstein_radius_m(
    d_m2_s: f64,
    temp_k: f64,
    viscosity_pa_s: f64,
) -> Result<f64, DosyError> {
    if !d_m2_s.is_finite() || d_m2_s <= 0.0 {
        return Err(DosyError::new(format!(
            "invalid diffusion coefficient D = {d_m2_s} m^2/s"
        )));
    }
    if !temp_k.is_finite() || temp_k <= 0.0 {
        return Err(DosyError::new(format!(
            "invalid temperature T = {temp_k} K"
        )));
    }
    if !viscosity_pa_s.is_finite() || viscosity_pa_s <= 0.0 {
        return Err(DosyError::new(format!(
            "invalid solvent viscosity eta = {viscosity_pa_s} Pa*s"
        )));
    }
    Ok((BOLTZMANN_J_PER_K * temp_k) / (6.0 * PI * viscosity_pa_s * d_m2_s))
}

/// Compute Stokes–Einstein hydrodynamic radius ($R_h$ in `m`, `Å`, `nm`) and power-law `MW` (`Da`)
/// for a built-in [`DosySolvent`] at temperature `temp_k` (`K`).
pub fn estimate_hydrodynamics(
    d_m2_s: f64,
    solvent: DosySolvent,
    temp_k: f64,
    class: SoluteCompactnessClass,
) -> Result<HydrodynamicEstimate, DosyError> {
    let viscosity_pa_s = solvent.viscosity_pa_s(temp_k)?;
    let r_h_m = stokes_einstein_radius_m(d_m2_s, temp_k, viscosity_pa_s)?;
    let cal = PowerLawCalibration::for_solvent(solvent, temp_k, class)?;
    let estimated_mw_da = cal.estimate_mw_da(d_m2_s)?;
    Ok(HydrodynamicEstimate {
        d_m2_s,
        temp_k,
        viscosity_pa_s,
        r_h_m,
        r_h_angstrom: r_h_m * 1.0e10,
        r_h_nm: r_h_m * 1.0e9,
        estimated_mw_da,
    })
}

// ============================================================================
// 6. Widened D2O Water-Skirt Exclusion & Published PFG-STE D_ref Corpus (`F29`)
// ============================================================================

/// Default `D2O` residual `HDO` + radiation-damped water-skirt exclusion band `[4.50, 4.98] ppm`
/// for `${}^{1}\text{H}$ `DOSY` reference-plane peak picking (`F29`, widened from `[4.65, 4.95]`
/// so `4.56..4.59 ppm` water-skirt phase/truncation ripples are suppressed).
pub const DEFAULT_DOSY_D2O_WATER_EXCLUSION_PPM: (f64, f64) = (4.50, 4.98);

/// Configuration for 1D `DOSY` reference-plane peak picking (`F29`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DosyPeakPickConfig {
    /// Minimum peak signal-to-noise ratio (`default = 10.0`).
    pub min_snr: f64,
    /// Excluded chemical-shift windows `(ppm_min, ppm_max)` (defaults to `[DEFAULT_DOSY_D2O_WATER_EXCLUSION_PPM]`).
    pub excluded_ppm_regions: Vec<(f64, f64)>,
}

impl Default for DosyPeakPickConfig {
    fn default() -> Self {
        Self {
            min_snr: 10.0,
            excluded_ppm_regions: vec![DEFAULT_DOSY_D2O_WATER_EXCLUSION_PPM],
        }
    }
}

/// Returns `true` if `ppm` falls inside the default `DOSY` `D2O` water-skirt exclusion band `[4.50, 4.98] ppm`.
#[inline]
pub fn is_in_dosy_d2o_water_skirt(ppm: f64) -> bool {
    let (lo, hi) = DEFAULT_DOSY_D2O_WATER_EXCLUSION_PPM;
    ppm >= lo && ppm <= hi
}

/// Pick solute resonances on a lowest-gradient 1D `DOSY` reference plane (`Spectrum1D`),
/// suppressing residual `HDO` and water-skirt ripples inside `cfg.excluded_ppm_regions`
/// (`[4.50, 4.98] ppm` by default, `F29`).
pub fn pick_dosy_peaks_1d(
    ref_plane: &crate::nmr1d::Spectrum1D,
    cfg: &DosyPeakPickConfig,
) -> Vec<crate::nmr1d::Peak1DPicked> {
    let raw = crate::nmr1d::pick_peaks_1d(
        ref_plane,
        &crate::nmr1d::PeakPickConfig1D {
            min_snr: cfg.min_snr.max(1.0),
            detect_shoulders: false,
            ..crate::nmr1d::PeakPickConfig1D::default()
        },
    );
    raw.into_iter()
        .filter(|pk| {
            pk.snr.abs() >= cfg.min_snr
                && !cfg.excluded_ppm_regions.iter().any(|&(a, b)| {
                    let (lo, hi) = (a.min(b), a.max(b));
                    pk.ppm >= lo && pk.ppm <= hi
                })
        })
        .collect()
}

/// Clean-room published literature PFG-STE self-diffusion calibration standard (`D_ref` at $298.15\text{ K}$).
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct DosyPublishedStandard {
    /// Canonical solute name.
    pub solute: &'static str,
    /// Case-insensitive lookup aliases.
    pub aliases: &'static [&'static str],
    /// Deuterated or protonated NMR solvent (`D2O`, `H2O`, `CDCl3`, `DMSO-d6`).
    pub solvent: DosySolvent,
    /// Reference temperature in Kelvin (`298.15 K`).
    pub temp_k: f64,
    /// Solute molecular weight (`Da` = `g / mol`).
    pub mw_da: f64,
    /// Published translational self-diffusion coefficient $D_{\text{ref}}$ (`m^2 / s`).
    pub d_ref_m2_s: f64,
    /// Published `1 sigma` experimental uncertainty $\sigma_D$ (`m^2 / s`).
    pub d_uncertainty_m2_s: f64,
    /// Representative `${}^{1}\text{H}$ chemical shift (`ppm`) for peak matching.
    pub representative_h1_ppm: f64,
    /// Peer-reviewed literature provenance citation.
    pub citation: &'static str,
}

const PUBLISHED_DOSY_D_REF_CORPUS: &[DosyPublishedStandard] = &[
    DosyPublishedStandard {
        solute: "HDO",
        aliases: &["hdo", "residual hdo", "trace hdo"],
        solvent: DosySolvent::D2O,
        temp_k: 298.15,
        mw_da: 19.021,
        d_ref_m2_s: 1.902e-9,
        d_uncertainty_m2_s: 0.004e-9,
        representative_h1_ppm: 4.79,
        citation: "Holz, Heil & Sacco, Phys. Chem. Chem. Phys. 2000, 2, 4740-4742; Longsworth, J. Phys. Chem. 1960, 64, 1914-1917",
    },
    DosyPublishedStandard {
        solute: "H2O",
        aliases: &["h2o", "water", "neat water"],
        solvent: DosySolvent::H2O,
        temp_k: 298.15,
        mw_da: 18.015,
        d_ref_m2_s: 2.299e-9,
        d_uncertainty_m2_s: 0.005e-9,
        representative_h1_ppm: 4.79,
        citation: "Holz, Heil & Sacco, Phys. Chem. Chem. Phys. 2000, 2, 4740-4742; Holz & Weingartner, J. Magn. Reson. 1991, 92, 115-125",
    },
    DosyPublishedStandard {
        solute: "Glycine",
        aliases: &["glycine", "gly"],
        solvent: DosySolvent::D2O,
        temp_k: 298.15,
        mw_da: 75.067,
        d_ref_m2_s: 8.55e-10,
        d_uncertainty_m2_s: 0.15e-10,
        representative_h1_ppm: 3.55,
        citation: "Longsworth, J. Phys. Chem. 1960, 64, 1914-1917 (D2O viscosity-corrected at 298.15 K)",
    },
    DosyPublishedStandard {
        solute: "L-Alanine",
        aliases: &["l-alanine", "alanine", "ala"],
        solvent: DosySolvent::D2O,
        temp_k: 298.15,
        mw_da: 89.093,
        d_ref_m2_s: 7.37e-10,
        d_uncertainty_m2_s: 0.15e-10,
        representative_h1_ppm: 1.48,
        citation: "Longsworth, J. Phys. Chem. 1960, 64, 1914-1917 (D2O viscosity-corrected at 298.15 K)",
    },
    DosyPublishedStandard {
        solute: "DSS",
        aliases: &["dss", "trimethylsilylpropanesulfonate"],
        solvent: DosySolvent::D2O,
        temp_k: 298.15,
        mw_da: 218.32,
        d_ref_m2_s: 5.58e-10,
        d_uncertainty_m2_s: 0.12e-10,
        representative_h1_ppm: 0.00,
        citation: "Wilkins et al., Biochemistry 1999, 38, 16424-16431; Evans et al., Angew. Chem. Int. Ed. 2013, 52, 3199-3202",
    },
    DosyPublishedStandard {
        solute: "D-Glucose",
        aliases: &["d-glucose", "glucose", "glc"],
        solvent: DosySolvent::D2O,
        temp_k: 298.15,
        mw_da: 180.156,
        d_ref_m2_s: 5.45e-10,
        d_uncertainty_m2_s: 0.12e-10,
        representative_h1_ppm: 5.23,
        citation: "Longsworth, J. Phys. Chem. 1960, 64, 1914-1917 (D2O viscosity-corrected at 298.15 K)",
    },
    DosyPublishedStandard {
        solute: "Sucrose",
        aliases: &["sucrose", "saccharose"],
        solvent: DosySolvent::D2O,
        temp_k: 298.15,
        mw_da: 342.297,
        d_ref_m2_s: 4.23e-10,
        d_uncertainty_m2_s: 0.10e-10,
        representative_h1_ppm: 5.41,
        citation: "Longsworth, J. Phys. Chem. 1960, 64, 1914-1917; Holz & Weingartner, J. Magn. Reson. 1991, 92, 115-125",
    },
    DosyPublishedStandard {
        solute: "beta-Cyclodextrin",
        aliases: &["beta-cyclodextrin", "b-cd", "cyclodextrin"],
        solvent: DosySolvent::D2O,
        temp_k: 298.15,
        mw_da: 1134.98,
        d_ref_m2_s: 2.62e-10,
        d_uncertainty_m2_s: 0.08e-10,
        representative_h1_ppm: 5.08,
        citation: "Longsworth, J. Phys. Chem. 1960, 64, 1914-1917; Pelta et al., Magn. Reson. Chem. 2002, 40, S147-S152",
    },
    DosyPublishedStandard {
        solute: "Lysozyme",
        aliases: &["lysozyme", "hewl", "hen egg-white lysozyme"],
        solvent: DosySolvent::D2O,
        temp_k: 298.15,
        mw_da: 14307.0,
        d_ref_m2_s: 1.08e-10,
        d_uncertainty_m2_s: 0.04e-10,
        representative_h1_ppm: 7.20,
        citation: "Wilkins et al., Biochemistry 1999, 38, 16424-16431; Evans et al., Angew. Chem. Int. Ed. 2013, 52, 3199-3202",
    },
    DosyPublishedStandard {
        solute: "TMS",
        aliases: &["tms", "tetramethylsilane", "me4si"],
        solvent: DosySolvent::CDCl3,
        temp_k: 298.15,
        mw_da: 88.224,
        d_ref_m2_s: 2.25e-9,
        d_uncertainty_m2_s: 0.06e-9,
        representative_h1_ppm: 0.00,
        citation: "Evans et al., Angew. Chem. Int. Ed. 2013, 52, 3199-3202",
    },
    DosyPublishedStandard {
        solute: "CHCl3",
        aliases: &["chcl3", "chloroform", "residual chcl3"],
        solvent: DosySolvent::CDCl3,
        temp_k: 298.15,
        mw_da: 119.378,
        d_ref_m2_s: 2.35e-9,
        d_uncertainty_m2_s: 0.06e-9,
        representative_h1_ppm: 7.26,
        citation: "Holz & Weingartner, J. Magn. Reson. 1991, 92, 115-125; Evans et al., Angew. Chem. Int. Ed. 2013, 52, 3199-3202",
    },
    DosyPublishedStandard {
        solute: "Toluene",
        aliases: &["toluene", "methylbenzene"],
        solvent: DosySolvent::CDCl3,
        temp_k: 298.15,
        mw_da: 92.141,
        d_ref_m2_s: 2.01e-9,
        d_uncertainty_m2_s: 0.05e-9,
        representative_h1_ppm: 2.36,
        citation: "Evans et al., Angew. Chem. Int. Ed. 2013, 52, 3199-3202",
    },
    DosyPublishedStandard {
        solute: "Naphthalene",
        aliases: &["naphthalene"],
        solvent: DosySolvent::CDCl3,
        temp_k: 298.15,
        mw_da: 128.174,
        d_ref_m2_s: 1.62e-9,
        d_uncertainty_m2_s: 0.04e-9,
        representative_h1_ppm: 7.84,
        citation: "Evans et al., Angew. Chem. Int. Ed. 2013, 52, 3199-3202",
    },
    DosyPublishedStandard {
        solute: "TMSS",
        aliases: &["tmss", "tetrakis(trimethylsilyl)silane"],
        solvent: DosySolvent::CDCl3,
        temp_k: 298.15,
        mw_da: 320.84,
        d_ref_m2_s: 1.02e-9,
        d_uncertainty_m2_s: 0.03e-9,
        representative_h1_ppm: 0.25,
        citation: "Evans et al., Angew. Chem. Int. Ed. 2013, 52, 3199-3202",
    },
    DosyPublishedStandard {
        solute: "DMSO-d5",
        aliases: &["dmso-d5", "chd2socd3", "residual dmso"],
        solvent: DosySolvent::DmsoD6,
        temp_k: 298.15,
        mw_da: 83.164,
        d_ref_m2_s: 6.80e-10,
        d_uncertainty_m2_s: 0.20e-10,
        representative_h1_ppm: 2.50,
        citation: "Holz & Weingartner, J. Magn. Reson. 1991, 92, 115-125; Evans et al., Angew. Chem. Int. Ed. 2013, 52, 3199-3202",
    },
    DosyPublishedStandard {
        solute: "Caffeine",
        aliases: &["caffeine"],
        solvent: DosySolvent::DmsoD6,
        temp_k: 298.15,
        mw_da: 194.194,
        d_ref_m2_s: 3.48e-10,
        d_uncertainty_m2_s: 0.12e-10,
        representative_h1_ppm: 3.87,
        citation: "Evans et al., Angew. Chem. Int. Ed. 2013, 52, 3199-3202",
    },
    DosyPublishedStandard {
        solute: "Quinine",
        aliases: &["quinine"],
        solvent: DosySolvent::DmsoD6,
        temp_k: 298.15,
        mw_da: 324.424,
        d_ref_m2_s: 2.28e-10,
        d_uncertainty_m2_s: 0.08e-10,
        representative_h1_ppm: 8.73,
        citation: "Pelta et al., Magn. Reson. Chem. 2002, 40, S147-S152; Evans et al., Angew. Chem. Int. Ed. 2013, 52, 3199-3202",
    },
];

/// Return the built-in clean-room published literature PFG-STE `D_ref` calibration corpus (`F29`).
pub fn published_dosy_d_ref_corpus() -> &'static [DosyPublishedStandard] {
    PUBLISHED_DOSY_D_REF_CORPUS
}

/// Look up a published PFG-STE `D_ref` standard by solute name/alias and optional solvent.
pub fn lookup_published_dosy_d_ref(
    solute_name: &str,
    solvent: Option<DosySolvent>,
) -> Option<&'static DosyPublishedStandard> {
    let q = solute_name.trim().to_ascii_lowercase();
    PUBLISHED_DOSY_D_REF_CORPUS.iter().find(|entry| {
        let solv_ok = solvent.is_none_or(|s| entry.solvent == s);
        let name_ok = entry.solute.eq_ignore_ascii_case(&q)
            || entry.aliases.iter().any(|a| a.eq_ignore_ascii_case(&q));
        solv_ok && name_ok
    })
}
