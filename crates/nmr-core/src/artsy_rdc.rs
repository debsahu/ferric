//! Intensity-based `ARTSY` (Amide RDCs by TROSY Spectroscopy) extraction of one-bond
//! `^1J_NH + ^1D_NH` couplings and `^1D_NH` residual dipolar couplings from interleaved
//! 2D reference and attenuated `^1H-^15N TROSY-HSQC` spectra.
//!
//! Provenance: Clean-room implementation strictly from peer-reviewed biomolecular NMR literature:
//! - Fitzkee, N. C. & Bax, A. (2010). "Facile measurement of `^1H-^15N` residual dipolar couplings
//!   in larger perdeuterated proteins." *J. Biomol. NMR* 48(2), 65-70 (quantitative `J`-correlation
//!   `ARTSY` intensity ratio `Q = I_att / I_ref = sin(pi |J+D| T) / sin(pi |J+D| T / 2) = 2 cos(pi |J+D| T / 2)`,
//!   canonical dephasing delay `T = 1 / |^1J_NH| ~ 10.75 ms`, isotropic vs aligned `^1J_NH`
//!   baseline subtraction, scan-count scaling, and differential `T_2`-relaxation mismatch correction).
//! - Ying, J., Wang, J., Grishaev, A., Yu, P., Wang, Y.-X. & Bax, A. (2011). "Measurement of
//!   `^1H-^15N` and `^1H-^13C` residual dipolar couplings in nucleic acids from TROSY intensities."
//!   *J. Biomol. NMR* 51(1-2), 89-103 (exact bivariate Gaussian uncertainty propagation, saturation
//!   boundary diagnostics as `|Q/2| -> 1`, and Monte Carlo resampling near the `arcsin`/`arccos`
//!   singularity).
//!
//! ## Physical Forward and Inverse Relations
//!
//! In an interleaved `ARTSY` experiment with full dephasing period `T` (default `10.75 ms`,
//! matched to `1 / |^1J_NH| ~ 1 / 93.0 Hz`):
//! - Reference `TROSY-HSQC` (active `^1H-^15N` INEPT dephasing `T / 2`):
//!   `I_ref = I_0 * sin(pi * |J + D| * T / 2)` (maximal at `|J + D| = 1 / T`).
//! - Attenuated `TROSY-HSQC` (active `^1H-^15N` INEPT dephasing `T`):
//!   `I_att = I_0 * s * exp(-Delta_R2 * T / 2) * sin(pi * |J + D| * T)`
//!   (zero-crossing at `|J + D| = 1 / T`, positive for `|J + D| < 1 / T`, inverted negative for
//!   `|J + D| > 1 / T`), where `s = N_att / N_ref` is the optional scan-count ratio and
//!   `Delta_R2 = R2_att - R2_ref` is the optional differential transverse relaxation mismatch.
//!
//! Defining the scan- and relaxation-corrected ratio `Q = (I_att / I_ref) / (s * exp(-Delta_R2 * T / 2))`
//! and the normalized ratio `q = Q / 2`:
//! ```text
//! q = Q / 2 = cos(pi * |J + D| * T / 2) = sin(pi * (1 - |J + D| * T) / 2) in (-1, 1)
//! |J + D|   = (2 / (pi * T)) * arccos(q) = (1 / T) - (2 / (pi * T)) * arcsin(q)
//! ```
//!
//! Differentiating with respect to independent Gaussian intensities `(I_ref, I_att)` with
//! standard deviations `(sigma_ref, sigma_att)` yields the exact first-order analytical error:
//! ```text
//! sigma_q        = (1 / (2 * c)) * sqrt((sigma_att / I_ref)^2 + (I_att * sigma_ref / I_ref^2)^2)
//! sigma_{|J + D|} = (2 / (pi * T * sqrt(1 - q^2))) * sigma_q
//! ```
//! where `c = s * exp(-Delta_R2 * T / 2)`. At `q = 0`, `c = 1`, `T = 10.75 ms`, and equal noise
//! `sigma_att = sigma_ref = N`, this reduces to Fitzkee & Bax's `sigma_{|J+D|} = 1 / (pi * T * SNR) ~ 30 / SNR Hz`.

use std::f64::consts::PI;

use serde::{Deserialize, Serialize};

use crate::rdc::{RdcBondType, RdcMediumSet, RdcRestraint, RdcSignConvention};
use crate::rng::Rng;
use crate::spectrum::Spectrum2D;

/// Default full `ARTSY` dephasing delay `T` in seconds (`10.75 ms ≈ 1 / 93.0 Hz`; Fitzkee & Bax 2010).
pub const DEFAULT_ARTSY_DEPHASING_S: f64 = 0.01075;

/// Canonical signed one-bond backbone amide scalar coupling `^1J_NH` in Hz (`-93.0 Hz`).
pub const DEFAULT_J_NH_ISO_HZ: f64 = -93.0;

/// Default normalized ratio threshold `|q| = |Q / 2|` above which [`ArtsyQualityFlag::NearSaturation`]
/// is emitted (`1 / sqrt(1 - 0.90^2) ≈ 2.29x` noise amplification).
pub const DEFAULT_NEAR_SATURATION_THRESHOLD: f64 = 0.90;

/// Default minimum reference-spectrum SNR `|I_ref| / sigma_ref` below which
/// [`ArtsyQualityFlag::LowSnr`] is emitted.
pub const DEFAULT_MIN_REF_SNR: f64 = 10.0;

/// Default deterministic seed for 2D bivariate Gaussian Monte Carlo uncertainty resampling.
pub const DEFAULT_ARTSY_MC_SEED: u64 = 0xA475_0022_0009;

/// Gaussian consistency factor converting Median Absolute Deviation (MAD) to 1-sigma (`1 / Phi^-1(0.75)`).
const MAD_TO_SIGMA: f64 = 1.482_602_218_505_602;

/// Quality and branch-saturation classification for an extracted `ARTSY` coupling.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ArtsyQualityFlag {
    /// Well-conditioned ratio inside the linear `arcsin`/`arccos` regime with adequate reference SNR.
    Good,
    /// In-domain (`|q| < 1`), but close to the saturation boundary (`|q| >= near_saturation_threshold`)
    /// where `1 / sqrt(1 - q^2)` amplifies noise and linear error propagation becomes inaccurate.
    NearSaturation,
    /// Corrected normalized ratio `|q| = |Q / 2| >= 1.0`, outside the physical `(-1, 1)` domain
    /// of `cos(pi * |J + D| * T / 2)`; inversion fails closed (`None`).
    OutOfDomainRatio,
    /// Reference peak SNR `|I_ref| / sigma_ref` is below `min_ref_snr`.
    LowSnr,
}

impl ArtsyQualityFlag {
    /// Severity rank used when combining aligned and isotropic quality flags.
    #[inline]
    fn severity(self) -> u8 {
        match self {
            Self::Good => 0,
            Self::NearSaturation => 1,
            Self::LowSnr => 2,
            Self::OutOfDomainRatio => 3,
        }
    }

    /// Combine two quality flags, keeping the more severe diagnostic.
    #[inline]
    pub fn combine(self, other: Self) -> Self {
        if other.severity() > self.severity() {
            other
        } else {
            self
        }
    }
}

/// Configuration for `ARTSY` coupling inversion, baseline subtraction, and uncertainty estimation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ArtsyConfig {
    /// Full `ARTSY` `^1H` dephasing delay `T` in seconds (default [`DEFAULT_ARTSY_DEPHASING_S`]).
    pub dephasing_delay_s: f64,
    /// Canonical signed isotropic scalar coupling `^1J_NH` in Hz (default [`DEFAULT_J_NH_ISO_HZ`] = `-93.0`),
    /// used whenever no per-residue isotropic `ARTSY` measurement is supplied.
    pub j_iso_default_hz: f64,
    /// Optional 1-sigma uncertainty (Hz) on `j_iso_default_hz` when per-residue isotropic
    /// measurements are absent (default `0.0` Hz).
    pub j_iso_default_sigma_hz: f64,
    /// Output RDC sign convention (default [`RdcSignConvention::SignNormalized`], matching `nmr_core::rdc`).
    pub sign_convention: RdcSignConvention,
    /// Differential transverse relaxation mismatch `Delta_R2 = R2_att - R2_ref` (s^-1) over the
    /// half-delay `T / 2` between attenuated and reference paths (Fitzkee & Bax 2010).
    /// Forward scaling is `exp(-delta_r2_s_inv * T / 2)`; default `0.0`.
    pub delta_r2_s_inv: f64,
    /// Scan-count ratio `N_att / N_ref` between attenuated and reference spectra (default `1.0`).
    pub scan_ratio_att_over_ref: f64,
    /// Normalized ratio magnitude threshold `|q| = |Q / 2|` at or above which
    /// [`ArtsyQualityFlag::NearSaturation`] is emitted (must lie in `(0.0, 1.0)`; default `0.90`).
    pub near_saturation_threshold: f64,
    /// Minimum reference peak SNR `|I_ref| / sigma_ref` below which [`ArtsyQualityFlag::LowSnr`]
    /// is emitted (default `10.0`).
    pub min_ref_snr: f64,
    /// Number of 2D Monte Carlo bivariate Gaussian resampling trials (`0` disables MC; default `0`).
    pub mc_trials: usize,
    /// Deterministic RNG seed for Monte Carlo resampling.
    pub mc_seed: u64,
}

impl Default for ArtsyConfig {
    fn default() -> Self {
        Self {
            dephasing_delay_s: DEFAULT_ARTSY_DEPHASING_S,
            j_iso_default_hz: DEFAULT_J_NH_ISO_HZ,
            j_iso_default_sigma_hz: 0.0,
            sign_convention: RdcSignConvention::SignNormalized,
            delta_r2_s_inv: 0.0,
            scan_ratio_att_over_ref: 1.0,
            near_saturation_threshold: DEFAULT_NEAR_SATURATION_THRESHOLD,
            min_ref_snr: DEFAULT_MIN_REF_SNR,
            mc_trials: 0,
            mc_seed: DEFAULT_ARTSY_MC_SEED,
        }
    }
}

impl ArtsyConfig {
    /// Validate all physical and numerical parameters, rejecting non-finite or out-of-range inputs.
    pub fn validate(&self) -> Result<(), String> {
        if !self.dephasing_delay_s.is_finite() || self.dephasing_delay_s <= 0.0 {
            return Err(format!(
                "invalid ARTSY dephasing_delay_s ({}); must be finite and > 0",
                self.dephasing_delay_s
            ));
        }
        if !self.j_iso_default_hz.is_finite() || self.j_iso_default_hz.abs() <= 1e-6 {
            return Err(format!(
                "invalid ARTSY j_iso_default_hz ({}); must be finite and non-zero",
                self.j_iso_default_hz
            ));
        }
        if !self.j_iso_default_sigma_hz.is_finite() || self.j_iso_default_sigma_hz < 0.0 {
            return Err(format!(
                "invalid ARTSY j_iso_default_sigma_hz ({}); must be finite and >= 0",
                self.j_iso_default_sigma_hz
            ));
        }
        if !self.delta_r2_s_inv.is_finite() {
            return Err(format!(
                "invalid ARTSY delta_r2_s_inv ({}); must be finite",
                self.delta_r2_s_inv
            ));
        }
        if !self.scan_ratio_att_over_ref.is_finite() || self.scan_ratio_att_over_ref <= 0.0 {
            return Err(format!(
                "invalid ARTSY scan_ratio_att_over_ref ({}); must be finite and > 0",
                self.scan_ratio_att_over_ref
            ));
        }
        let scale = self.intensity_scale_factor();
        if !scale.is_finite() || scale <= 0.0 {
            return Err(format!(
                "ARTSY relaxation/scan scale factor underflowed or overflowed ({scale})"
            ));
        }
        if !self.near_saturation_threshold.is_finite()
            || self.near_saturation_threshold <= 0.0
            || self.near_saturation_threshold >= 1.0
        {
            return Err(format!(
                "invalid ARTSY near_saturation_threshold ({}); must be in (0, 1)",
                self.near_saturation_threshold
            ));
        }
        if !self.min_ref_snr.is_finite() || self.min_ref_snr < 0.0 {
            return Err(format!(
                "invalid ARTSY min_ref_snr ({}); must be finite and >= 0",
                self.min_ref_snr
            ));
        }
        Ok(())
    }

    /// Multiplicative factor `c = (N_att / N_ref) * exp(-Delta_R2 * T / 2)` relating raw intensity
    /// ratio `I_att / I_ref` to the ideal relaxation-free ratio `2 cos(pi |J + D| T / 2)`.
    #[inline]
    pub fn intensity_scale_factor(&self) -> f64 {
        self.scan_ratio_att_over_ref * (-0.5 * self.delta_r2_s_inv * self.dephasing_delay_s).exp()
    }
}

/// Paired peak intensities and 1-sigma noise standard deviations from an interleaved `ARTSY` experiment.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ArtsyIntensityPair {
    /// Peak intensity (height or integrated volume) in the reference spectrum (`I_ref`).
    pub i_ref: f64,
    /// Peak intensity (same operator/box) in the attenuated spectrum (`I_att`).
    pub i_att: f64,
    /// 1-sigma noise standard deviation for `i_ref` (`>= 0`).
    pub sigma_ref: f64,
    /// 1-sigma noise standard deviation for `i_att` (`>= 0`).
    pub sigma_att: f64,
}

impl ArtsyIntensityPair {
    /// Validate that intensities and noise levels are finite, `|I_ref| > 0`, and noise `>= 0`.
    pub fn validate(&self) -> Result<(), String> {
        if !self.i_ref.is_finite() {
            return Err(format!("non-finite ARTSY i_ref ({})", self.i_ref));
        }
        if self.i_ref.abs() <= 1e-30 {
            return Err(format!(
                "zero or near-zero ARTSY i_ref ({}); reference intensity must be non-zero",
                self.i_ref
            ));
        }
        if !self.i_att.is_finite() {
            return Err(format!("non-finite ARTSY i_att ({})", self.i_att));
        }
        if !self.sigma_ref.is_finite() || self.sigma_ref < 0.0 {
            return Err(format!(
                "invalid ARTSY sigma_ref ({}); must be finite and >= 0",
                self.sigma_ref
            ));
        }
        if !self.sigma_att.is_finite() || self.sigma_att < 0.0 {
            return Err(format!(
                "invalid ARTSY sigma_att ({}); must be finite and >= 0",
                self.sigma_att
            ));
        }
        Ok(())
    }
}

/// 2D Monte Carlo bivariate Gaussian resampling summary for `|J + D|` uncertainty estimation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ArtsyMcUncertainty {
    /// Total number of bivariate Gaussian draws attempted.
    pub n_trials: usize,
    /// Number of draws producing a valid in-domain ratio (`|q| < 1` and non-zero `I_ref` with
    /// matching sign to the nominal reference intensity).
    pub n_in_domain: usize,
    /// Fraction of draws rejected as out-of-domain (`1.0 - n_in_domain / n_trials`).
    pub out_of_domain_fraction: f64,
    /// Sample mean of in-domain `|J + D|` draws in Hz (`None` if `n_in_domain == 0`).
    pub mean_abs_j_plus_d_hz: Option<f64>,
    /// Sample standard deviation (Bessel-corrected) of in-domain `|J + D|` draws in Hz
    /// (`None` if `n_in_domain < 2`).
    pub sigma_mc_hz: Option<f64>,
    /// Sample median of in-domain `|J + D|` draws in Hz (`None` if `n_in_domain == 0`).
    pub median_abs_j_plus_d_hz: Option<f64>,
    /// Empirical 68.27% central confidence interval `(p15.865, p84.135)` in Hz
    /// (`None` if `n_in_domain < 2`).
    pub ci68_hz: Option<(f64, f64)>,
}

/// Inverted `ARTSY` coupling magnitude `|J + D|` and uncertainty diagnostics from one intensity pair.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ArtsyCouplingEstimate {
    /// Raw intensity ratio `I_att / I_ref`.
    pub raw_ratio: f64,
    /// Scan- and differential-`T2`-corrected ratio `Q = (I_att / I_ref) / c`.
    pub corrected_ratio_q: f64,
    /// Normalized ratio `q = Q / 2 = cos(pi * |J + D| * T / 2) = sin(pi * (1 - |J + D| * T) / 2)`.
    pub normalized_ratio_q_half: f64,
    /// Reference peak SNR `|I_ref| / sigma_ref` (`f64::INFINITY` when `sigma_ref == 0`).
    pub snr_ref: f64,
    /// Extracted total coupling magnitude `|J + D|` in Hz (`None` when [`ArtsyQualityFlag::OutOfDomainRatio`]).
    pub abs_j_plus_d_hz: Option<f64>,
    /// Exact first-order bivariate Gaussian analytical uncertainty on `|J + D|` in Hz
    /// (`None` when [`ArtsyQualityFlag::OutOfDomainRatio`]).
    pub sigma_analytical_hz: Option<f64>,
    /// Optional 2D Monte Carlo uncertainty estimate (populated when `cfg.mc_trials > 0`).
    pub mc_uncertainty: Option<ArtsyMcUncertainty>,
    /// Quality and branch-saturation classification.
    pub quality: ArtsyQualityFlag,
}

/// Extracted `^1D_NH` residual dipolar coupling for a single amide residue.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ArtsyResidueRdc {
    pub seq_id: i32,
    pub res_name: String,
    /// Aligned-state `ARTSY` coupling extraction.
    pub aligned: ArtsyCouplingEstimate,
    /// Optional per-residue isotropic-state `ARTSY` coupling extraction.
    pub isotropic: Option<ArtsyCouplingEstimate>,
    /// Isotropic scalar coupling magnitude `|^1J_NH|` (Hz) subtracted from `|J + D|`.
    pub j_iso_abs_hz: f64,
    /// 1-sigma uncertainty (Hz) on `j_iso_abs_hz`.
    pub j_iso_sigma_hz: f64,
    /// Extracted `^1D_NH` in Hz under [`Self::sign_convention`] (`None` when inversion failed closed).
    pub d_nh_hz: Option<f64>,
    /// Combined 1-sigma uncertainty on `d_nh_hz` in Hz (`None` when `d_nh_hz` is `None`).
    pub sigma_d_nh_hz: Option<f64>,
    /// Sign convention applied to `d_nh_hz`.
    pub sign_convention: RdcSignConvention,
    /// Worst-case combined quality flag across aligned (and optional isotropic) measurement.
    pub quality: ArtsyQualityFlag,
}

impl ArtsyResidueRdc {
    /// Convert this `ARTSY` residue measurement into an [`RdcRestraint`] compatible with
    /// `nmr_core::rdc` (Saupe SVD, Cornilescu `Q`, NEF/TSV export, and structure refinement).
    ///
    /// Returns `None` if inversion failed closed (`d_nh_hz.is_none()`) or if `include_flagged`
    /// is `false` and `self.quality != ArtsyQualityFlag::Good`.
    pub fn to_rdc_restraint(
        &self,
        medium_id: &str,
        min_sigma_hz: f64,
        include_flagged: bool,
    ) -> Option<RdcRestraint> {
        if medium_id.trim().is_empty() {
            return None;
        }
        if !include_flagged && self.quality != ArtsyQualityFlag::Good {
            return None;
        }
        let d_obs_hz = self.d_nh_hz?;
        let raw_sigma = self.sigma_d_nh_hz?;
        if !d_obs_hz.is_finite() || !raw_sigma.is_finite() {
            return None;
        }
        let floor = if min_sigma_hz.is_finite() && min_sigma_hz > 0.0 {
            min_sigma_hz
        } else {
            1e-6
        };
        let sigma_hz = raw_sigma.max(floor);
        let res_name = if self.res_name.trim().is_empty() {
            "UNK".to_string()
        } else {
            self.res_name.trim().to_ascii_uppercase()
        };
        Some(RdcRestraint {
            medium_id: medium_id.trim().to_string(),
            seq_id_1: self.seq_id,
            res_name_1: res_name.clone(),
            atom_name_1: "N".to_string(),
            seq_id_2: self.seq_id,
            res_name_2: res_name,
            atom_name_2: "H".to_string(),
            bond_type: RdcBondType::NH,
            d_obs_hz,
            sigma_hz,
            d_max_hz: RdcBondType::NH.signed_d_max_hz(self.sign_convention),
            is_free: false,
        })
    }
}

/// Integration operator applied identically to reference and attenuated 2D `TROSY-HSQC` planes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ArtsyIntegrationMode {
    /// Sum voxels over a `(2*box_radius_x + 1) x (2*box_radius_y + 1)` window locked on the
    /// reference peak center (lineshape cancels identically in `I_att / I_ref`).
    #[default]
    BoxSum,
    /// Separable 3-point parabolic sub-voxel interpolation locked to the reference peak's
    /// sub-voxel center `(dx, dy) in [-0.5, 0.5]^2` (identical linear weights applied to both planes).
    ParabolicSubvoxel,
}

/// Assigned backbone amide peak target `(1H ppm, 15N ppm)` for 2D `ARTSY` extraction.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ArtsyPeakTarget {
    pub seq_id: i32,
    pub res_name: String,
    /// Amide proton shift (`1H`, direct `axis_x`) in ppm.
    pub h_ppm: f64,
    /// Amide nitrogen shift (`15N`, indirect `axis_y`) in ppm.
    pub n_ppm: f64,
    /// Optional explicit per-residue isotropic baseline `(j_iso_hz, sigma_j_iso_hz)`.
    /// Overrides `cfg.artsy.j_iso_default_hz` when no isotropic spectrum pair is supplied.
    pub j_iso_override: Option<(f64, f64)>,
}

/// Configuration for 2D interleaved `TROSY-HSQC` `ARTSY` peak integration and extraction.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ArtsySpectrumExtractConfig {
    pub artsy: ArtsyConfig,
    /// Linear peak integration operator applied identically to reference and attenuated planes.
    pub integration: ArtsyIntegrationMode,
    /// Box half-width in direct (`1H`) voxels for [`ArtsyIntegrationMode::BoxSum`] (default `1` -> 3 voxels).
    pub box_radius_x: usize,
    /// Box half-width in indirect (`15N`) voxels for [`ArtsyIntegrationMode::BoxSum`] (default `1` -> 3 voxels).
    pub box_radius_y: usize,
    /// Optional local extremum snap radius (in voxels) on the reference plane before locking
    /// the integration center for both planes (default `1`; `0` disables snapping).
    pub ref_snap_radius: usize,
    /// Optional explicit per-voxel noise `(sigma_vox_ref, sigma_vox_att)`.
    /// When `None`, robustly estimated from each plane via two-pass outlier-trimmed MAD.
    pub noise_override: Option<(f64, f64)>,
    /// Minimum uncertainty floor (Hz) applied when emitting [`RdcRestraint`] records (default `0.05` Hz).
    pub min_restraint_sigma_hz: f64,
    /// If `true`, `NearSaturation` and `LowSnr` residues (excluding `OutOfDomainRatio`) are
    /// also included in the emitted `restraints` vector (default `false`).
    pub include_flagged_in_restraints: bool,
}

impl Default for ArtsySpectrumExtractConfig {
    fn default() -> Self {
        Self {
            artsy: ArtsyConfig::default(),
            integration: ArtsyIntegrationMode::BoxSum,
            box_radius_x: 1,
            box_radius_y: 1,
            ref_snap_radius: 1,
            noise_override: None,
            min_restraint_sigma_hz: 0.05,
            include_flagged_in_restraints: false,
        }
    }
}

/// Output of 2D interleaved `TROSY-HSQC` `ARTSY` extraction.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ArtsySpectrumExtractionResult {
    pub medium_id: String,
    /// Per-voxel noise standard deviation used for the reference plane.
    pub sigma_vox_ref: f64,
    /// Per-voxel noise standard deviation used for the attenuated plane.
    pub sigma_vox_att: f64,
    /// Per-residue extracted `ARTSY` coupling and RDC records in input peak order.
    pub residues: Vec<ArtsyResidueRdc>,
    /// Filtered `RdcRestraint` records compatible with `nmr_core::rdc::fit_saupe_tensor_svd`.
    pub restraints: Vec<RdcRestraint>,
}

// ============================================================================
// Exact Forward & Inverse ARTSY Coupling Physics
// ============================================================================

/// Convert a signed `^1D_NH` value (Hz) under `convention` and isotropic `^1J_NH` (Hz)
/// into the total coupling magnitude `|J + D|` (Hz).
///
/// - Under [`RdcSignConvention::SignNormalized`] (`D_max > 0`, Bax `1D3Z` / `nmr_core::rdc` default):
///   `|J + D| = |J_iso| + D_norm`.
/// - Under [`RdcSignConvention::PhysicalGammaSign`] (`D_max < 0`, physical `gamma_15N < 0` sign):
///   `J + D_phys = -|J_iso| + D_phys`, so `|J + D| = |J_iso| - D_phys`.
pub fn artsy_abs_coupling_from_rdc(
    d_nh_hz: f64,
    j_iso_hz: f64,
    convention: RdcSignConvention,
) -> Result<f64, String> {
    if !d_nh_hz.is_finite() {
        return Err(format!("non-finite d_nh_hz ({d_nh_hz})"));
    }
    if !j_iso_hz.is_finite() || j_iso_hz.abs() <= 1e-6 {
        return Err(format!(
            "invalid j_iso_hz ({j_iso_hz}); must be finite and non-zero"
        ));
    }
    let abs_j = j_iso_hz.abs();
    let abs_jd = match convention {
        RdcSignConvention::SignNormalized => abs_j + d_nh_hz,
        RdcSignConvention::PhysicalGammaSign => abs_j - d_nh_hz,
    };
    if !abs_jd.is_finite() || abs_jd <= 0.0 {
        return Err(format!(
            "total coupling magnitude |J + D| = {abs_jd} Hz must be finite and > 0"
        ));
    }
    Ok(abs_jd)
}

/// Convert total coupling magnitude `|J + D|` (Hz) and isotropic baseline `|J_iso|` (Hz)
/// into signed `^1D_NH` (Hz) under `convention`.
#[inline]
pub fn artsy_rdc_from_abs_coupling(
    abs_j_plus_d_hz: f64,
    j_iso_abs_hz: f64,
    convention: RdcSignConvention,
) -> f64 {
    match convention {
        RdcSignConvention::SignNormalized => abs_j_plus_d_hz - j_iso_abs_hz,
        RdcSignConvention::PhysicalGammaSign => j_iso_abs_hz - abs_j_plus_d_hz,
    }
}

/// Exact `ARTSY` forward model predicting raw intensity ratio `I_att / I_ref` for total coupling
/// magnitude `abs_j_plus_d_hz = |^1J_NH + ^1D_NH|` (Hz):
/// ```text
/// I_att / I_ref = 2 * s * exp(-Delta_R2 * T / 2) * cos(pi * |J + D| * T / 2)
/// ```
/// Valid for `0 < |J + D| < 2 / T` (where `sin(pi * |J + D| * T / 2) > 0`).
pub fn artsy_forward_ratio(abs_j_plus_d_hz: f64, cfg: &ArtsyConfig) -> Result<f64, String> {
    cfg.validate()?;
    if !abs_j_plus_d_hz.is_finite() || abs_j_plus_d_hz <= 0.0 {
        return Err(format!(
            "invalid abs_j_plus_d_hz ({abs_j_plus_d_hz}); must be finite and > 0"
        ));
    }
    let phase = 0.5 * PI * abs_j_plus_d_hz * cfg.dephasing_delay_s;
    Ok(2.0 * cfg.intensity_scale_factor() * phase.cos())
}

/// Exact `ARTSY` forward model returning `(I_ref, I_att)` from the physical INEPT transfer
/// efficiencies (Fitzkee & Bax 2010):
/// - `I_ref = i0 * sin(pi * |J + D| * T / 2)`
/// - `I_att = i0 * s * exp(-Delta_R2 * T / 2) * sin(pi * |J + D| * T)`
pub fn artsy_forward_intensities(
    abs_j_plus_d_hz: f64,
    i0: f64,
    cfg: &ArtsyConfig,
) -> Result<(f64, f64), String> {
    cfg.validate()?;
    if !abs_j_plus_d_hz.is_finite() || abs_j_plus_d_hz <= 0.0 {
        return Err(format!(
            "invalid abs_j_plus_d_hz ({abs_j_plus_d_hz}); must be finite and > 0"
        ));
    }
    if !i0.is_finite() || i0.abs() <= 1e-30 {
        return Err(format!(
            "invalid reference amplitude i0 ({i0}); must be finite and non-zero"
        ));
    }
    let half_phase = 0.5 * PI * abs_j_plus_d_hz * cfg.dephasing_delay_s;
    let i_ref = i0 * half_phase.sin();
    let i_att = i0 * cfg.intensity_scale_factor() * (2.0 * half_phase).sin();
    Ok((i_ref, i_att))
}

/// Invert a single `ARTSY` intensity pair `(I_ref, I_att)` into total coupling magnitude `|J + D|`
/// with exact bivariate Gaussian analytical uncertainty propagation and optional Monte Carlo
/// resampling (`cfg.mc_trials > 0`).
pub fn extract_artsy_coupling(
    pair: &ArtsyIntensityPair,
    cfg: &ArtsyConfig,
) -> Result<ArtsyCouplingEstimate, String> {
    cfg.validate()?;
    pair.validate()?;

    let t = cfg.dephasing_delay_s;
    let scale = cfg.intensity_scale_factor();
    let raw_ratio = pair.i_att / pair.i_ref;
    let corrected_ratio_q = raw_ratio / scale;
    let q = 0.5 * corrected_ratio_q;
    let snr_ref = if pair.sigma_ref == 0.0 {
        f64::INFINITY
    } else {
        pair.i_ref.abs() / pair.sigma_ref
    };

    let mc_uncertainty = if cfg.mc_trials > 0 {
        Some(estimate_artsy_uncertainty_mc(
            pair,
            cfg,
            cfg.mc_trials,
            cfg.mc_seed,
        )?)
    } else {
        None
    };

    // Fail closed whenever the normalized ratio lies on or outside (-1, 1)
    if !q.is_finite() || q.abs() >= 1.0 {
        return Ok(ArtsyCouplingEstimate {
            raw_ratio,
            corrected_ratio_q,
            normalized_ratio_q_half: q,
            snr_ref,
            abs_j_plus_d_hz: None,
            sigma_analytical_hz: None,
            mc_uncertainty,
            quality: ArtsyQualityFlag::OutOfDomainRatio,
        });
    }

    // |J + D| = (2 / (pi * T)) * arccos(q) = (1 / T) - (2 / (pi * T)) * arcsin(q)
    // Writing via arcsin(q) preserves full f64 precision around the sweet spot q = 0 (|J + D| = 1 / T).
    let abs_j_plus_d_hz = (1.0 / t) - (2.0 / (PI * t)) * q.asin();

    // Exact bivariate Gaussian first-order error propagation (Ying et al. 2011; Fitzkee & Bax 2010):
    // sigma_q = (1 / (2 * scale)) * sqrt((sigma_att / I_ref)^2 + (I_att * sigma_ref / I_ref^2)^2)
    // sigma_|J+D| = (2 / (pi * T * sqrt(1 - q^2))) * sigma_q
    let inv_i_ref = 1.0 / pair.i_ref;
    let term_att = pair.sigma_att * inv_i_ref;
    let term_ref = pair.i_att * pair.sigma_ref * inv_i_ref * inv_i_ref;
    let sigma_raw_ratio = term_att.hypot(term_ref);
    let sigma_q = 0.5 * sigma_raw_ratio / scale;
    let denom = (1.0 - q * q).sqrt();
    let sigma_analytical_hz = if denom > 0.0 {
        (2.0 / (PI * t * denom)) * sigma_q
    } else {
        f64::INFINITY
    };

    let quality = if snr_ref < cfg.min_ref_snr {
        ArtsyQualityFlag::LowSnr
    } else if q.abs() >= cfg.near_saturation_threshold {
        ArtsyQualityFlag::NearSaturation
    } else {
        ArtsyQualityFlag::Good
    };

    Ok(ArtsyCouplingEstimate {
        raw_ratio,
        corrected_ratio_q,
        normalized_ratio_q_half: q,
        snr_ref,
        abs_j_plus_d_hz: Some(abs_j_plus_d_hz),
        sigma_analytical_hz: Some(sigma_analytical_hz),
        mc_uncertainty,
        quality,
    })
}

/// 2D bivariate Gaussian Monte Carlo uncertainty estimator for `ARTSY` coupling inversion.
///
/// Independently resamples `I_ref* ~ N(I_ref, sigma_ref^2)` and `I_att* ~ N(I_att, sigma_att^2)`
/// over `n_trials` deterministic draws, retaining physically valid in-domain draws
/// (`|q*| < 1` and `I_ref*` having the same sign as `I_ref` with `|I_ref*| > 1e-30`).
/// Particularly valuable near the saturation boundary `|q| -> 1` where first-order analytical
/// error propagation `1 / sqrt(1 - q^2)` diverges.
pub fn estimate_artsy_uncertainty_mc(
    pair: &ArtsyIntensityPair,
    cfg: &ArtsyConfig,
    n_trials: usize,
    seed: u64,
) -> Result<ArtsyMcUncertainty, String> {
    cfg.validate()?;
    pair.validate()?;
    if n_trials < 2 {
        return Err(format!(
            "ARTSY Monte Carlo resampling requires n_trials >= 2, got {n_trials}"
        ));
    }

    let t = cfg.dephasing_delay_s;
    let inv_t = 1.0 / t;
    let two_over_pi_t = 2.0 / (PI * t);
    let two_scale = 2.0 * cfg.intensity_scale_factor();
    let ref_sign = pair.i_ref.signum();

    let mut rng = Rng::new(seed);
    let mut draws = Vec::with_capacity(n_trials);
    let mut mean = 0.0_f64;
    let mut m2 = 0.0_f64;

    for _ in 0..n_trials {
        let ir = if pair.sigma_ref > 0.0 {
            rng.normal(pair.i_ref, pair.sigma_ref)
        } else {
            pair.i_ref
        };
        let ia = if pair.sigma_att > 0.0 {
            rng.normal(pair.i_att, pair.sigma_att)
        } else {
            pair.i_att
        };
        if !ir.is_finite() || !ia.is_finite() || ir.abs() <= 1e-30 || ir.signum() != ref_sign {
            continue;
        }
        let q = ia / (two_scale * ir);
        if !q.is_finite() || q.abs() >= 1.0 {
            continue;
        }
        let val = inv_t - two_over_pi_t * q.asin();
        if !val.is_finite() {
            continue;
        }
        draws.push(val);
        let n_f = draws.len() as f64;
        let delta = val - mean;
        mean += delta / n_f;
        let delta2 = val - mean;
        m2 += delta * delta2;
    }

    let n_in_domain = draws.len();
    let out_of_domain_fraction = 1.0 - (n_in_domain as f64) / (n_trials as f64);

    if n_in_domain == 0 {
        return Ok(ArtsyMcUncertainty {
            n_trials,
            n_in_domain: 0,
            out_of_domain_fraction: 1.0,
            mean_abs_j_plus_d_hz: None,
            sigma_mc_hz: None,
            median_abs_j_plus_d_hz: None,
            ci68_hz: None,
        });
    }

    draws.sort_by(f64::total_cmp);
    let median = percentile_sorted(&draws, 0.5);

    if n_in_domain < 2 {
        return Ok(ArtsyMcUncertainty {
            n_trials,
            n_in_domain,
            out_of_domain_fraction,
            mean_abs_j_plus_d_hz: Some(mean),
            sigma_mc_hz: None,
            median_abs_j_plus_d_hz: Some(median),
            ci68_hz: None,
        });
    }

    let variance = (m2 / ((n_in_domain - 1) as f64)).max(0.0);
    let sigma_mc = variance.sqrt();
    let p_lo = percentile_sorted(&draws, 0.158_655_253_931_457_07);
    let p_hi = percentile_sorted(&draws, 0.841_344_746_068_542_9);

    Ok(ArtsyMcUncertainty {
        n_trials,
        n_in_domain,
        out_of_domain_fraction,
        mean_abs_j_plus_d_hz: Some(mean),
        sigma_mc_hz: Some(sigma_mc),
        median_abs_j_plus_d_hz: Some(median),
        ci68_hz: Some((p_lo, p_hi)),
    })
}

/// Extract `^1D_NH` for a single amide residue from an aligned `ARTSY` intensity pair and an
/// optional per-residue isotropic `ARTSY` intensity pair (or explicit baseline override).
///
/// When `isotropic_pair` is `Some`, its extracted `|J_iso|` and uncertainty are subtracted
/// in quadrature; otherwise `j_iso_override` (if `Some`) or `(cfg.j_iso_default_hz, cfg.j_iso_default_sigma_hz)`
/// is used.
pub fn extract_artsy_rdc(
    seq_id: i32,
    res_name: &str,
    aligned_pair: &ArtsyIntensityPair,
    isotropic_pair: Option<&ArtsyIntensityPair>,
    j_iso_override: Option<(f64, f64)>,
    cfg: &ArtsyConfig,
) -> Result<ArtsyResidueRdc, String> {
    cfg.validate()?;
    let aligned = extract_artsy_coupling(aligned_pair, cfg)?;

    let (isotropic, j_iso_abs_hz, j_iso_sigma_hz, iso_failed, iso_quality) =
        if let Some(iso_p) = isotropic_pair {
            let mut iso_cfg = cfg.clone();
            if iso_cfg.mc_trials > 0 {
                iso_cfg.mc_seed = cfg.mc_seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
            }
            let iso_est = extract_artsy_coupling(iso_p, &iso_cfg)?;
            let iso_q = iso_est.quality;
            match (iso_est.abs_j_plus_d_hz, iso_est.sigma_analytical_hz) {
                (Some(j_val), Some(j_sig)) => (Some(iso_est), j_val, j_sig, false, iso_q),
                _ => (Some(iso_est), cfg.j_iso_default_hz.abs(), 0.0, true, iso_q),
            }
        } else if let Some((j_hz, j_sig)) = j_iso_override {
            if !j_hz.is_finite() || j_hz.abs() <= 1e-6 {
                return Err(format!(
                    "invalid j_iso_override coupling ({j_hz}) on residue {seq_id}:{res_name}"
                ));
            }
            if !j_sig.is_finite() || j_sig < 0.0 {
                return Err(format!(
                    "invalid j_iso_override sigma ({j_sig}) on residue {seq_id}:{res_name}"
                ));
            }
            (None, j_hz.abs(), j_sig, false, ArtsyQualityFlag::Good)
        } else {
            (
                None,
                cfg.j_iso_default_hz.abs(),
                cfg.j_iso_default_sigma_hz,
                false,
                ArtsyQualityFlag::Good,
            )
        };

    let quality = aligned.quality.combine(iso_quality);
    let (d_nh_hz, sigma_d_nh_hz) = if iso_failed {
        (None, None)
    } else if let (Some(abs_jd), Some(sig_jd)) =
        (aligned.abs_j_plus_d_hz, aligned.sigma_analytical_hz)
    {
        let d_val = artsy_rdc_from_abs_coupling(abs_jd, j_iso_abs_hz, cfg.sign_convention);
        let sig_d = sig_jd.hypot(j_iso_sigma_hz);
        (Some(d_val), Some(sig_d))
    } else {
        (None, None)
    };

    Ok(ArtsyResidueRdc {
        seq_id,
        res_name: res_name.trim().to_ascii_uppercase(),
        aligned,
        isotropic,
        j_iso_abs_hz,
        j_iso_sigma_hz,
        d_nh_hz,
        sigma_d_nh_hz,
        sign_convention: cfg.sign_convention,
        quality,
    })
}

/// Collect usable [`ArtsyResidueRdc`] measurements into an [`RdcMediumSet`] ready for
/// `nmr_core::rdc::fit_saupe_tensor_svd` or TSV/NEF export.
pub fn artsy_results_to_rdc_medium_set(
    medium_id: &str,
    residues: &[ArtsyResidueRdc],
    min_sigma_hz: f64,
    include_flagged: bool,
) -> Result<RdcMediumSet, String> {
    if medium_id.trim().is_empty() {
        return Err("medium_id must not be empty".to_string());
    }
    let restraints: Vec<RdcRestraint> = residues
        .iter()
        .filter_map(|r| r.to_rdc_restraint(medium_id, min_sigma_hz, include_flagged))
        .collect();
    if restraints.is_empty() {
        return Err(format!(
            "no usable ARTSY RDC restraints produced for medium '{medium_id}'"
        ));
    }
    Ok(RdcMediumSet {
        medium_id: medium_id.trim().to_string(),
        restraints,
    })
}

// ============================================================================
// 2D Spectrum Pair Extraction (Box / Parabolic Sub-Voxel + MAD Noise)
// ============================================================================

/// Robust two-pass outlier-trimmed Median Absolute Deviation (MAD) per-voxel noise estimator
/// for a 2D NMR spectrum plane.
///
/// Pass 1 estimates global median and MAD; pass 2 discards peak voxels outside `±3.5 * MAD_0`
/// and recomputes `1.4826 * median(|bg - median(bg)|)` on the baseline noise floor.
pub fn estimate_spectrum_2d_noise_mad(spec: &Spectrum2D) -> Result<f64, String> {
    let expected = spec
        .nx()
        .checked_mul(spec.ny())
        .ok_or_else(|| format!("spectrum '{}' dimensions overflow usize", spec.name))?;
    if expected == 0 || spec.data.len() != expected {
        return Err(format!(
            "spectrum '{}' data length ({}) does not match {}x{} grid",
            spec.name,
            spec.data.len(),
            spec.nx(),
            spec.ny()
        ));
    }

    let stride = (spec.data.len() / 65_536).max(1);
    let mut sample: Vec<f64> = Vec::with_capacity(spec.data.len().div_ceil(stride));
    for &v in spec.data.iter().step_by(stride) {
        let vf = v as f64;
        if !vf.is_finite() {
            return Err(format!(
                "spectrum '{}' contains non-finite voxel value ({v})",
                spec.name
            ));
        }
        sample.push(vf);
    }
    if sample.is_empty() {
        return Err(format!("spectrum '{}' is empty", spec.name));
    }

    let med0 = median_f64(&mut sample);
    let mut dev0: Vec<f64> = sample.iter().map(|&v| (v - med0).abs()).collect();
    let mad0 = MAD_TO_SIGMA * median_f64(&mut dev0);
    if mad0 <= 0.0 {
        return Ok(0.0);
    }

    let mut bg: Vec<f64> = sample
        .into_iter()
        .filter(|&v| (v - med0).abs() <= 3.5 * mad0)
        .collect();
    if bg.len() < 16 {
        return Ok(mad0);
    }
    let med1 = median_f64(&mut bg);
    for v in &mut bg {
        *v = (*v - med1).abs();
    }
    Ok(MAD_TO_SIGMA * median_f64(&mut bg))
}

/// Extract `ARTSY` `^1J_NH + ^1D_NH` couplings and `^1D_NH` restraints directly from a paired
/// 2D interleaved `TROSY-HSQC` reference (`ref_spec`) and attenuated (`att_spec`) spectrum.
///
/// Integration coordinates (and optional local reference snap / sub-voxel offset) are locked
/// on `ref_spec` and applied identically to `att_spec` so that lineshape cancels identically
/// in `I_att / I_ref` even when `I_att` passes through zero or inverts sign near `|J + D| = 1 / T`.
pub fn extract_artsy_from_spectra_2d(
    medium_id: &str,
    ref_spec: &Spectrum2D,
    att_spec: &Spectrum2D,
    peaks: &[ArtsyPeakTarget],
    cfg: &ArtsySpectrumExtractConfig,
) -> Result<ArtsySpectrumExtractionResult, String> {
    if medium_id.trim().is_empty() {
        return Err("medium_id must not be empty".to_string());
    }
    cfg.artsy.validate()?;
    if peaks.is_empty() {
        return Err("extract_artsy_from_spectra_2d requires at least one peak target".to_string());
    }
    validate_paired_spectra_2d(ref_spec, att_spec, cfg)?;

    let (sigma_vox_ref, sigma_vox_att) = resolve_pair_noise_2d(ref_spec, att_spec, cfg)?;

    let mut residues = Vec::with_capacity(peaks.len());
    let mut restraints = Vec::new();

    for (idx, peak) in peaks.iter().enumerate() {
        let pair =
            integrate_paired_peak_2d(ref_spec, att_spec, peak, sigma_vox_ref, sigma_vox_att, cfg)?;
        let mut res_cfg = cfg.artsy.clone();
        if res_cfg.mc_trials > 0 {
            res_cfg.mc_seed = cfg.artsy.mc_seed.wrapping_add(idx as u64);
        }
        let res = extract_artsy_rdc(
            peak.seq_id,
            &peak.res_name,
            &pair,
            None,
            peak.j_iso_override,
            &res_cfg,
        )?;
        if let Some(restraint) = res.to_rdc_restraint(
            medium_id,
            cfg.min_restraint_sigma_hz,
            cfg.include_flagged_in_restraints,
        ) {
            restraints.push(restraint);
        }
        residues.push(res);
    }

    Ok(ArtsySpectrumExtractionResult {
        medium_id: medium_id.trim().to_string(),
        sigma_vox_ref,
        sigma_vox_att,
        residues,
        restraints,
    })
}

/// Extract per-residue `^1D_NH` restraints from **both** an aligned `(aligned_ref, aligned_att)`
/// 2D `ARTSY` pair and an isotropic `(iso_ref, iso_att)` 2D `ARTSY` pair, performing exact
/// per-residue `^1J_NH` baseline subtraction with quadrature error propagation (Fitzkee & Bax 2010).
#[allow(clippy::too_many_arguments)]
pub fn extract_artsy_aligned_and_isotropic_spectra_2d(
    medium_id: &str,
    aligned_ref: &Spectrum2D,
    aligned_att: &Spectrum2D,
    iso_ref: &Spectrum2D,
    iso_att: &Spectrum2D,
    peaks: &[ArtsyPeakTarget],
    cfg: &ArtsySpectrumExtractConfig,
) -> Result<ArtsySpectrumExtractionResult, String> {
    if medium_id.trim().is_empty() {
        return Err("medium_id must not be empty".to_string());
    }
    cfg.artsy.validate()?;
    if peaks.is_empty() {
        return Err(
            "extract_artsy_aligned_and_isotropic_spectra_2d requires at least one peak target"
                .to_string(),
        );
    }
    validate_paired_spectra_2d(aligned_ref, aligned_att, cfg)?;
    validate_paired_spectra_2d(iso_ref, iso_att, cfg)?;

    let (sigma_vox_ref, sigma_vox_att) = resolve_pair_noise_2d(aligned_ref, aligned_att, cfg)?;
    let (sigma_vox_iso_ref, sigma_vox_iso_att) = resolve_pair_noise_2d(iso_ref, iso_att, cfg)?;

    let mut residues = Vec::with_capacity(peaks.len());
    let mut restraints = Vec::new();

    for (idx, peak) in peaks.iter().enumerate() {
        let aligned_pair = integrate_paired_peak_2d(
            aligned_ref,
            aligned_att,
            peak,
            sigma_vox_ref,
            sigma_vox_att,
            cfg,
        )?;
        let iso_pair = integrate_paired_peak_2d(
            iso_ref,
            iso_att,
            peak,
            sigma_vox_iso_ref,
            sigma_vox_iso_att,
            cfg,
        )?;
        let mut res_cfg = cfg.artsy.clone();
        if res_cfg.mc_trials > 0 {
            res_cfg.mc_seed = cfg.artsy.mc_seed.wrapping_add(idx as u64);
        }
        let res = extract_artsy_rdc(
            peak.seq_id,
            &peak.res_name,
            &aligned_pair,
            Some(&iso_pair),
            None,
            &res_cfg,
        )?;
        if let Some(restraint) = res.to_rdc_restraint(
            medium_id,
            cfg.min_restraint_sigma_hz,
            cfg.include_flagged_in_restraints,
        ) {
            restraints.push(restraint);
        }
        residues.push(res);
    }

    Ok(ArtsySpectrumExtractionResult {
        medium_id: medium_id.trim().to_string(),
        sigma_vox_ref,
        sigma_vox_att,
        residues,
        restraints,
    })
}

// ============================================================================
// Internal Helpers
// ============================================================================

fn validate_paired_spectra_2d(
    ref_spec: &Spectrum2D,
    att_spec: &Spectrum2D,
    cfg: &ArtsySpectrumExtractConfig,
) -> Result<(), String> {
    let (rx, ry) = required_stencil_radii(cfg);
    let min_nx = 2 * rx + 1;
    let min_ny = 2 * ry + 1;

    if ref_spec.nx() < min_nx || ref_spec.ny() < min_ny {
        return Err(format!(
            "reference spectrum '{}' grid ({}x{}) is too small for stencil ({min_nx}x{min_ny})",
            ref_spec.name,
            ref_spec.nx(),
            ref_spec.ny()
        ));
    }
    if ref_spec.nx() != att_spec.nx() || ref_spec.ny() != att_spec.ny() {
        return Err(format!(
            "reference ('{}': {}x{}) and attenuated ('{}': {}x{}) 2D spectrum dimensions do not match",
            ref_spec.name,
            ref_spec.nx(),
            ref_spec.ny(),
            att_spec.name,
            att_spec.nx(),
            att_spec.ny()
        ));
    }

    let expected = ref_spec
        .nx()
        .checked_mul(ref_spec.ny())
        .ok_or_else(|| "2D spectrum dimensions overflow usize".to_string())?;
    if ref_spec.data.len() != expected {
        return Err(format!(
            "reference spectrum '{}' buffer length ({}) != {}x{}",
            ref_spec.name,
            ref_spec.data.len(),
            ref_spec.nx(),
            ref_spec.ny()
        ));
    }
    if att_spec.data.len() != expected {
        return Err(format!(
            "attenuated spectrum '{}' buffer length ({}) != {}x{}",
            att_spec.name,
            att_spec.data.len(),
            att_spec.nx(),
            att_spec.ny()
        ));
    }

    for (label, ax_r, ax_a) in [
        ("X", &ref_spec.axis_x, &att_spec.axis_x),
        ("Y", &ref_spec.axis_y, &att_spec.axis_y),
    ] {
        if !ax_r.first_ppm.is_finite()
            || !ax_r.step_ppm.is_finite()
            || ax_r.step_ppm.abs() <= 1e-12
            || !ax_a.first_ppm.is_finite()
            || !ax_a.step_ppm.is_finite()
            || ax_a.step_ppm.abs() <= 1e-12
        {
            return Err(format!(
                "invalid non-finite or zero-step {label} axis in paired ARTSY spectra"
            ));
        }
        if (ax_r.first_ppm - ax_a.first_ppm).abs() > 1e-4
            || (ax_r.step_ppm - ax_a.step_ppm).abs() > 1e-6
        {
            return Err(format!(
                "mismatched {label} axis calibration between reference and attenuated spectra"
            ));
        }
    }

    if let Some((sr, sa)) = cfg.noise_override {
        if !sr.is_finite() || sr < 0.0 || !sa.is_finite() || sa < 0.0 {
            return Err(format!(
                "invalid noise_override ({sr}, {sa}); must be finite and >= 0"
            ));
        }
    }

    Ok(())
}

fn resolve_pair_noise_2d(
    ref_spec: &Spectrum2D,
    att_spec: &Spectrum2D,
    cfg: &ArtsySpectrumExtractConfig,
) -> Result<(f64, f64), String> {
    if let Some((sr, sa)) = cfg.noise_override {
        return Ok((sr, sa));
    }
    let sr = estimate_spectrum_2d_noise_mad(ref_spec)?;
    let sa = estimate_spectrum_2d_noise_mad(att_spec)?;
    Ok((sr, sa))
}

#[inline]
fn required_stencil_radii(cfg: &ArtsySpectrumExtractConfig) -> (usize, usize) {
    match cfg.integration {
        ArtsyIntegrationMode::BoxSum => (cfg.box_radius_x, cfg.box_radius_y),
        ArtsyIntegrationMode::ParabolicSubvoxel => (1, 1),
    }
}

fn integrate_paired_peak_2d(
    ref_spec: &Spectrum2D,
    att_spec: &Spectrum2D,
    peak: &ArtsyPeakTarget,
    sigma_vox_ref: f64,
    sigma_vox_att: f64,
    cfg: &ArtsySpectrumExtractConfig,
) -> Result<ArtsyIntensityPair, String> {
    if !peak.h_ppm.is_finite() || !peak.n_ppm.is_finite() {
        return Err(format!(
            "non-finite peak coordinates ({}, {}) on residue {}:{}",
            peak.h_ppm, peak.n_ppm, peak.seq_id, peak.res_name
        ));
    }

    let (rx, ry) = required_stencil_radii(cfg);
    let nx = ref_spec.nx();
    let ny = ref_spec.ny();

    let fx = ref_spec.axis_x.ppm_to_index(peak.h_ppm);
    let fy = ref_spec.axis_y.ppm_to_index(peak.n_ppm);
    let min_x = rx as f64;
    let max_x = (nx - 1 - rx) as f64;
    let min_y = ry as f64;
    let max_y = (ny - 1 - ry) as f64;

    if !fx.is_finite() || !fy.is_finite() || fx < min_x || fx > max_x || fy < min_y || fy > max_y {
        return Err(format!(
            "peak {}:{} ({:.3} ppm, {:.3} ppm -> idx {:.2}, {:.2}) lies outside valid interior [{min_x}..{max_x}] x [{min_y}..{max_y}]",
            peak.seq_id, peak.res_name, peak.h_ppm, peak.n_ppm, fx, fy
        ));
    }

    let ix0 = fx.round() as usize;
    let iy0 = fy.round() as usize;

    // Optional local extremum snap on the reference plane (never independently on attenuated!)
    let (ix, iy) = if cfg.ref_snap_radius > 0 {
        let snap = cfg.ref_snap_radius;
        let x_lo = ix0.saturating_sub(snap).max(rx);
        let x_hi = ix0.saturating_add(snap).min(nx - 1 - rx);
        let y_lo = iy0.saturating_sub(snap).max(ry);
        let y_hi = iy0.saturating_add(snap).min(ny - 1 - ry);
        let mut best_x = ix0;
        let mut best_y = iy0;
        let mut best_abs = -1.0_f64;
        for yy in y_lo..=y_hi {
            for xx in x_lo..=x_hi {
                let v = (ref_spec.get(xx, yy) as f64).abs();
                if v.is_finite() && v > best_abs {
                    best_abs = v;
                    best_x = xx;
                    best_y = yy;
                }
            }
        }
        (best_x, best_y)
    } else {
        (ix0, iy0)
    };

    match cfg.integration {
        ArtsyIntegrationMode::BoxSum => {
            let mut sum_ref = 0.0_f64;
            let mut sum_att = 0.0_f64;
            let mut count = 0usize;
            for yy in (iy - ry)..=(iy + ry) {
                for xx in (ix - rx)..=(ix + rx) {
                    let vr = ref_spec.get(xx, yy) as f64;
                    let va = att_spec.get(xx, yy) as f64;
                    if !vr.is_finite() || !va.is_finite() {
                        return Err(format!(
                            "non-finite voxel in integration box for {}:{} at ({xx}, {yy})",
                            peak.seq_id, peak.res_name
                        ));
                    }
                    sum_ref += vr;
                    sum_att += va;
                    count += 1;
                }
            }
            let sqrt_n = (count as f64).sqrt();
            Ok(ArtsyIntensityPair {
                i_ref: sum_ref,
                i_att: sum_att,
                sigma_ref: sigma_vox_ref * sqrt_n,
                sigma_att: sigma_vox_att * sqrt_n,
            })
        }
        ArtsyIntegrationMode::ParabolicSubvoxel => {
            let rm_x = ref_spec.get(ix - 1, iy) as f64;
            let r0 = ref_spec.get(ix, iy) as f64;
            let rp_x = ref_spec.get(ix + 1, iy) as f64;
            let rm_y = ref_spec.get(ix, iy - 1) as f64;
            let rp_y = ref_spec.get(ix, iy + 1) as f64;

            let dx = parabolic_offset_3pt(rm_x, r0, rp_x);
            let dy = parabolic_offset_3pt(rm_y, r0, rp_y);
            let wx = lagrange_weights_3pt(dx);
            let wy = lagrange_weights_3pt(dy);

            let mut val_ref = 0.0_f64;
            let mut val_att = 0.0_f64;
            let mut w_sq_sum = 0.0_f64;
            for (jy, &wy_j) in wy.iter().enumerate() {
                let yy = iy + jy - 1;
                for (jx, &wx_i) in wx.iter().enumerate() {
                    let xx = ix + jx - 1;
                    let w = wx_i * wy_j;
                    let vr = ref_spec.get(xx, yy) as f64;
                    let va = att_spec.get(xx, yy) as f64;
                    if !vr.is_finite() || !va.is_finite() {
                        return Err(format!(
                            "non-finite voxel in parabolic stencil for {}:{} at ({xx}, {yy})",
                            peak.seq_id, peak.res_name
                        ));
                    }
                    val_ref += w * vr;
                    val_att += w * va;
                    w_sq_sum += w * w;
                }
            }
            let w_norm = w_sq_sum.sqrt();
            Ok(ArtsyIntensityPair {
                i_ref: val_ref,
                i_att: val_att,
                sigma_ref: sigma_vox_ref * w_norm,
                sigma_att: sigma_vox_att * w_norm,
            })
        }
    }
}

/// 3-point parabolic sub-voxel extremum offset in `[-0.5, 0.5]`.
#[inline]
fn parabolic_offset_3pt(ym1: f64, y0: f64, yp1: f64) -> f64 {
    let denom = ym1 - 2.0 * y0 + yp1;
    if !denom.is_finite() || denom.abs() <= 1e-15 {
        return 0.0;
    }
    // Require true local extremum (concave down for positive peak, concave up for negative peak)
    if y0 * denom >= 0.0 {
        return 0.0;
    }
    (0.5 * (ym1 - yp1) / denom).clamp(-0.5, 0.5)
}

/// Exact 3-point quadratic Lagrange interpolation weights at sub-voxel offset `d in [-0.5, 0.5]`.
#[inline]
fn lagrange_weights_3pt(d: f64) -> [f64; 3] {
    [0.5 * d * (d - 1.0), 1.0 - d * d, 0.5 * d * (d + 1.0)]
}

/// Exact median of a non-empty `f64` slice using partial selection (`O(N)`).
fn median_f64(vals: &mut [f64]) -> f64 {
    let n = vals.len();
    if n == 0 {
        return 0.0;
    }
    let mid = n / 2;
    let (_, upper_ref, _) = vals.select_nth_unstable_by(mid, f64::total_cmp);
    let upper = *upper_ref;
    if n % 2 == 1 {
        upper
    } else {
        let lower = vals[..mid]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        0.5 * (lower + upper)
    }
}

/// Linear-interpolated percentile `p in [0, 1]` on a non-empty sorted `f64` slice.
fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return sorted[0];
    }
    let pos = p.clamp(0.0, 1.0) * ((n - 1) as f64);
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = pos - (lo as f64);
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}
