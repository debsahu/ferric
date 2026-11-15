//! Clean-room untargeted 1D NMR metabolomics, interval FFT peak alignment, Adaptive Intelligent
//! Binning, Probabilistic Quotient Normalization (`PQN`), deterministic PCA / NIPALS `PLS-DA` /
//! `OPLS-DA` chemometrics with stratified cross-validation & permutation safeguards, `S-plot` /
//! `VIP` / Benjamini–Hochberg `FDR` Volcano biomarker discovery, and Statistical Total Correlation
//! Spectroscopy (`STOCSY`) coupled to a public-domain CC0 `BMRB` biofluid metabolite library.
//!
//! Provenance: Original clean-room implementation of `docs/specs/20-metabolomics-chemometrics.md`
//! (`T14.5`) derived strictly from peer-reviewed literature and CC0 `BMRB` reference data (zero
//! external source code, proprietary libraries, or copyleft packages consulted):
//! 1. Interval Fast Fourier cross-correlation local peak alignment:
//!    Savorani, F., Tomasi, G. & Engelsen, S. B., *J. Magn. Reson.* **2010**, 202, 190–202.
//! 2. Uniform fixed-width & Adaptive Intelligent Binning (`AIB`, recursive valley-seeking):
//!    Holmes, E., Foxall, P. J. D., Nicholson, J. K. et al., *Anal. Biochem.* **1994**, 220, 284–296;
//!    De Meyer, T., Sinnaeve, D., Van Gasse, B., Tsiporkova, E., Rietzschel, E. R., De Buyzere,
//!    M. L., Gillebert, T. C., Bekaert, S., Martins, J. C. & Van Criekinge, W.,
//!    *Anal. Chem.* **2008**, 80, 3783–3790.
//! 3. Probabilistic Quotient Normalization (`PQN`), Total Spectral Area (`TSA`), internal-standard
//!    normalization, and Pareto / Unit-Variance / Mean-Center scaling:
//!    Dieterle, F., Ross, A., Schlotterbeck, G. & Senn, H., *Anal. Chem.* **2006**, 78, 4281–4290;
//!    van den Berg, R. A., Hoefsloot, H. C. J., Westerhuis, J. A., Smilde, A. K. &
//!    van der Werf, M. J., *BMC Genomics* **2006**, 7, 142.
//! 4. PCA Hotelling $T^2$ & $Q$-residual (`SPE`), NIPALS `PLS-DA`, Orthogonal Projections to
//!    Latent Structures Discriminant Analysis (`OPLS-DA`), stratified cross-validation $Q^2$, and
//!    permutation $p$-value validation:
//!    Hotelling, H., *Ann. Math. Stat.* **1931**, 2, 360–378;
//!    Jackson, J. E. & Mudholkar, G. S., *Technometrics* **1979**, 21, 341–349;
//!    Trygg, J. & Wold, S., *J. Chemom.* **2002**, 16, 119–128;
//!    Bylesjö, M., Rantalainen, M., Cloarec, O., Nicholson, J. K., Holmes, E. & Trygg, J.,
//!    *J. Chemom.* **2006**, 20, 341–351;
//!    Westerhuis, J. A., Hoefsloot, H. C. J., Smit, S., Vis, D. J., Smilde, A. K.,
//!    van Velzen, E. J. J., van Duijnhoven, J. P. M. & van Dorsten, F. A.,
//!    *Metabolomics* **2008**, 4, 81–89.
//! 5. OPLS-DA `S-plot`, `VIP` scores, Welch two-sample $t$-test & Benjamini–Hochberg `FDR` Volcano:
//!    Wiklund, S., Johansson, E., Sjöström, L., Mellerowicz, E. J., Edlund, U., Shockcor, J. P.,
//!    Gottfries, J., Moritz, T. & Trygg, J., *Anal. Chem.* **2008**, 80, 115–122;
//!    Welch, B. L., *Biometrika* **1947**, 34, 28–35;
//!    Benjamini, Y. & Hochberg, Y., *J. R. Stat. Soc. B* **1995**, 57, 289–300.
//! 6. Statistical Total Correlation Spectroscopy (`STOCSY`) & CC0 `BMRB` biofluid library:
//!    Cloarec, O., Dumas, M.-E., Craig, A., Barton, R. H., Trygg, J., Hudson, J., Blancher, C.,
//!    Gauguier, D., Lindon, J. C., Holmes, E. & Nicholson, J. K.,
//!    *Anal. Chem.* **2005**, 77, 1282–1289;
//!    Ulrich, E. L. et al., *Nucleic Acids Res.* **2008**, 36, D402–D408 (`BMRB` CC0 archive).

use std::collections::BTreeMap;
use std::f64::consts::PI;

use serde::{Deserialize, Serialize};

use crate::nmr1d::{correct_baseline_1d, BaselineConfig1D, Experiment1DKind, Spectrum1D};
use crate::ops::fft_arbitrary_in_place;
use crate::rng::Rng;
use crate::spectrum::AxisPpm;

// ============================================================================
// 1. Data Schemas & Pipeline Configuration
// ============================================================================

/// A single 1D NMR sample in a metabolomics cohort study.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CohortSample1D {
    /// Unique sample identifier (e.g., `"Control_01"`, `"Case_12"`).
    pub id: String,
    /// Experimental class or phenotype label (e.g., `"Control"`, `"Case"`).
    pub group: String,
    /// Processed 1D NMR spectrum.
    pub spectrum: Spectrum1D,
    /// Optional numeric covariates (e.g., `"pH"`, `"age"`).
    pub covariates: BTreeMap<String, f64>,
}

impl CohortSample1D {
    /// Construct a cohort sample from an identifier, group label, and `Spectrum1D`.
    pub fn new(id: impl Into<String>, group: impl Into<String>, spectrum: Spectrum1D) -> Self {
        Self {
            id: id.into(),
            group: group.into(),
            spectrum,
            covariates: BTreeMap::new(),
        }
    }
}

/// Chemical-shift exclusion interval `[low_ppm, high_ppm]` (`low_ppm <= high_ppm`).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExclusionZonePpm {
    pub low_ppm: f64,
    pub high_ppm: f64,
}

impl ExclusionZonePpm {
    pub fn new(a_ppm: f64, b_ppm: f64) -> Self {
        Self {
            low_ppm: a_ppm.min(b_ppm),
            high_ppm: a_ppm.max(b_ppm),
        }
    }

    #[inline]
    pub fn contains(&self, ppm: f64) -> bool {
        ppm >= self.low_ppm && ppm <= self.high_ppm
    }
}

/// Canonical solvent, residual water, internal-standard, and urea masking presets.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum BiofluidMaskPreset {
    /// No spectral regions masked.
    None,
    /// Suppress residual $\text{H}_2\text{O}/\text{HDO}$ (`[4.70, 4.95]` ppm) only.
    AqueousMinimal,
    /// Serum / plasma preset: `TSP`/`DSS` collar (`[-0.50, 0.20]` ppm) + water (`[4.68, 5.00]` ppm).
    SerumPlasma,
    /// Urine preset: `TSP`/`DSS` (`[-0.50, 0.20]` ppm), water (`[4.65, 5.05]` ppm), and broad
    /// exchangeable urea $\text{CO(NH}_2)_2$ (`[5.55, 6.05]` ppm).
    Urine,
    /// User-defined list of `[low_ppm, high_ppm]` intervals.
    Custom(Vec<ExclusionZonePpm>),
}

impl BiofluidMaskPreset {
    /// Concrete list of `[low_ppm, high_ppm]` exclusion intervals.
    pub fn zones(&self) -> Vec<ExclusionZonePpm> {
        match self {
            Self::None => Vec::new(),
            Self::AqueousMinimal => vec![ExclusionZonePpm::new(4.70, 4.95)],
            Self::SerumPlasma => vec![
                ExclusionZonePpm::new(-0.50, 0.20),
                ExclusionZonePpm::new(4.68, 5.00),
            ],
            Self::Urine => vec![
                ExclusionZonePpm::new(-0.50, 0.20),
                ExclusionZonePpm::new(4.65, 5.05),
                ExclusionZonePpm::new(5.55, 6.05),
            ],
            Self::Custom(z) => z.clone(),
        }
    }
}

/// Target reference spectrum strategy for interval FFT cross-correlation alignment.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlignmentReference {
    /// Pointwise median across all cohort spectra (robust default).
    Median,
    /// Pointwise arithmetic mean across all cohort spectra.
    Mean,
    /// Explicit cohort sample index (`0..N`).
    SampleIndex(usize),
}

/// Interval segmentation strategy for Savorani et al. (2010) FFT peak alignment.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum AlignmentMode {
    /// Disable peak alignment.
    Disabled,
    /// Partition unmasked spectral blocks into valley-snapped intervals of `interval_width_ppm`
    /// with maximum allowed shift `max_shift_ppm`.
    AutomaticIntervals {
        interval_width_ppm: f64,
        max_shift_ppm: f64,
        reference: AlignmentReference,
    },
    /// Align only explicit `[low_ppm, high_ppm]` windows (e.g. pH-sensitive organic acids / imidazoles).
    CustomIntervals {
        intervals_ppm: Vec<(f64, f64)>,
        max_shift_ppm: f64,
        reference: AlignmentReference,
    },
}

impl Default for AlignmentMode {
    fn default() -> Self {
        Self::AutomaticIntervals {
            interval_width_ppm: 0.24,
            max_shift_ppm: 0.05,
            reference: AlignmentReference::Median,
        }
    }
}

/// Spectral bucketing strategy converting aligned 1D traces into a `N x P` feature matrix.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum BucketingMethod {
    /// Classical fixed-width uniform integration (Holmes et al. 1994).
    Uniform {
        bin_width_ppm: f64,
        /// Optional SNR pruning threshold (`0.0` retains all unmasked bins).
        min_bucket_snr: f64,
    },
    /// Adaptive Intelligent Binning (De Meyer et al. 2008): recursive valley-seeking maximization
    /// of $V_b(a, b)$ over a Hz-calibrated multiplet envelope.
    AdaptiveIntelligent {
        /// Resolution exponent $r \in (0, 1)$ (`0.5` canonical).
        resolution: f64,
        /// Minimum allowed bucket width in ppm (`0.020` default).
        min_width_ppm: f64,
        /// Maximum allowed bucket width in ppm (`0.100` default).
        max_width_ppm: f64,
        /// Gaussian multiplet-coalescing envelope $\sigma$ in Hz (`6.5 Hz` merges $J \le 13\text{ Hz}$
        /// intra-multiplet splittings while preserving inter-metabolite valleys).
        multiplet_coalesce_hz: f64,
        /// Minimum peak SNR required to retain a bucket (`3.0` default; `<= 0.0` keeps baseline bins).
        min_bucket_snr: f64,
    },
}

impl Default for BucketingMethod {
    fn default() -> Self {
        Self::AdaptiveIntelligent {
            resolution: 0.5,
            min_width_ppm: 0.020,
            max_width_ppm: 0.100,
            multiplet_coalesce_hz: 6.5,
            min_bucket_snr: 3.0,
        }
    }
}

/// Sample-wise dilution normalization method (`X_raw -> X_norm`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub enum NormalizationMethod {
    /// No sample normalization.
    None,
    /// Constant-sum / Total Spectral Area (`TSA`) normalization.
    TotalSpectralArea,
    /// Normalize to internal standard (`TSP` / `DSS` at `target_ppm`, typically `0.00 ppm`) area
    /// integrated inside `[target_ppm - half_window_ppm, target_ppm + half_window_ppm]`.
    InternalStandard {
        target_ppm: f64,
        half_window_ppm: f64,
    },
    /// Probabilistic Quotient Normalization (`PQN`, Dieterle et al. 2006).
    #[default]
    ProbabilisticQuotient,
}

/// Variable-wise centering & variance scaling (`X_norm -> X_scaled`, van den Berg et al. 2006).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScalingMethod {
    /// Leave matrix uncentered and unscaled.
    None,
    /// Subtract column mean: $\tilde{X}_{i,j} = X_{i,j} - \bar{X}_j$.
    MeanCenter,
    /// Pareto scaling: $\tilde{X}_{i,j} = (X_{i,j} - \bar{X}_j) / \sqrt{s_j}$ (canonical NMR default).
    #[default]
    Pareto,
    /// Unit-variance (autoscaling): $\tilde{X}_{i,j} = (X_{i,j} - \bar{X}_j) / s_j$.
    UnitVariance,
}

/// End-to-end configuration for an untargeted 1D NMR metabolomics study.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MetabolomicsPipelineConfig {
    /// Lower chemical-shift bound of the active study grid (ppm, default `0.50`).
    pub min_ppm: f64,
    /// Upper chemical-shift bound of the active study grid (ppm, default `9.50`).
    pub max_ppm: f64,
    /// Uniform resampling step size in ppm (default `0.001`).
    pub grid_step_ppm: f64,
    /// Automatically lock each spectrum's `0.00 ppm` `TSP`/`DSS` peak apex before grid resampling.
    pub calibrate_internal_standard: bool,
    /// Optional per-spectrum baseline correction prior to alignment & bucketing.
    pub baseline_config: Option<BaselineConfig1D>,
    /// Solvent, residual water, and urea exclusion mask.
    pub mask_preset: BiofluidMaskPreset,
    /// Local interval FFT cross-correlation peak alignment settings.
    pub alignment: AlignmentMode,
    /// Spectral bucketing strategy (`AdaptiveIntelligent` or `Uniform`).
    pub bucketing: BucketingMethod,
    /// Dilution normalization strategy (`PQN`, `TSA`, `InternalStandard`, `None`).
    pub normalization: NormalizationMethod,
    /// Variable scaling strategy (`Pareto`, `UnitVariance`, `MeanCenter`, `None`).
    pub scaling: ScalingMethod,
    /// Number of principal components to extract in PCA (`default = 5`).
    pub pca_components: usize,
    /// Number of PLS latent variables for NIPALS `PLS-DA` (`default = 2`).
    pub pls_components: usize,
    /// Number of orthogonal components $A_o$ in `OPLS-DA` (`1` predictive + `A_o` orthogonal; `default = 1`).
    pub opls_orthogonal_components: usize,
    /// Number of stratified cross-validation folds (`default = 7`).
    pub cv_folds: usize,
    /// Number of random label permutations for `Westerhuis et al. (2008)` overfitting validation (`default = 100`).
    pub n_permutations: usize,
    /// Deterministic PRNG seed for permutation validation.
    pub rng_seed: u64,
    /// Explicit control / baseline class label (`None` selects lexicographically first group).
    pub control_group: Option<String>,
    /// Explicit case / perturbation class label (`None` selects second group).
    pub case_group: Option<String>,
}

impl Default for MetabolomicsPipelineConfig {
    fn default() -> Self {
        Self {
            min_ppm: 0.50,
            max_ppm: 9.50,
            grid_step_ppm: 0.001,
            calibrate_internal_standard: false,
            baseline_config: None,
            mask_preset: BiofluidMaskPreset::Urine,
            alignment: AlignmentMode::default(),
            bucketing: BucketingMethod::default(),
            normalization: NormalizationMethod::ProbabilisticQuotient,
            scaling: ScalingMethod::Pareto,
            pca_components: 5,
            pls_components: 2,
            opls_orthogonal_components: 1,
            cv_folds: 7,
            n_permutations: 100,
            rng_seed: 20261115,
            control_group: None,
            case_group: None,
        }
    }
}

/// Spectral bucket definition on the common descending chemical-shift grid.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SpectralBucket {
    /// `0`-based column index in the bucket matrix.
    pub index: usize,
    /// Lower chemical-shift boundary (ppm).
    pub low_ppm: f64,
    /// Upper chemical-shift boundary (ppm).
    pub high_ppm: f64,
    /// Intensity-weighted centroid chemical shift (ppm) on the cohort reference spectrum.
    pub center_ppm: f64,
    /// Bucket width `high_ppm - low_ppm` (ppm).
    pub width_ppm: f64,
    /// Inclusive start grid index on the common resampled axis.
    pub start_grid_idx: usize,
    /// Inclusive end grid index on the common resampled axis (`start_grid_idx < end_grid_idx`).
    pub end_grid_idx: usize,
    /// Maximum signal-to-noise ratio across the cohort inside this bucket.
    pub max_snr: f64,
}

/// Parametric 2D confidence ellipse (e.g. 95% Hotelling $T^2$) for score-space visualization.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConfidenceEllipse2D {
    /// Group label (`"Global 95% T2"` for overall Hotelling limit, or a specific class name).
    pub label: String,
    /// Ellipse center `(cx, cy)` in score coordinates.
    pub center: [f64; 2],
    /// Major and minor semi-axis lengths `(a, b)`.
    pub radii: [f64; 2],
    /// Counter-clockwise rotation angle of the major axis in radians.
    pub angle_rad: f64,
}

impl ConfidenceEllipse2D {
    /// Sample `n_points` closed polygon vertices around the ellipse perimeter.
    pub fn sample_points(&self, n_points: usize) -> Vec<[f64; 2]> {
        let m = n_points.max(12);
        let (sin_a, cos_a) = self.angle_rad.sin_cos();
        let mut pts = Vec::with_capacity(m + 1);
        for k in 0..=m {
            let th = 2.0 * PI * (k as f64) / (m as f64);
            let (st, ct) = th.sin_cos();
            let lx = self.radii[0] * ct;
            let ly = self.radii[1] * st;
            pts.push([
                self.center[0] + lx * cos_a - ly * sin_a,
                self.center[1] + lx * sin_a + ly * cos_a,
            ]);
        }
        pts
    }
}

/// Deterministic Principal Component Analysis (`PCA`) model and outlier diagnostics.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PcaResult {
    /// Number of extracted principal components $A$.
    pub n_components: usize,
    /// Score matrix $T \in \mathbb{R}^{N \times A}$ (`scores[sample_idx][comp_idx]`).
    pub scores: Vec<Vec<f64>>,
    /// Orthonormal loading matrix $P \in \mathbb{R}^{P \times A}$ (`loadings[comp_idx][bucket_idx]`).
    pub loadings: Vec<Vec<f64>>,
    /// Singular values $\sigma_1 \ge \dots \ge \sigma_A$.
    pub singular_values: Vec<f64>,
    /// Fraction of scaled variance explained per component ($R^2 X_a \in [0, 1]$).
    pub explained_variance_ratio: Vec<f64>,
    /// Cumulative explained variance ratio ($\sum_{k \le a} R^2 X_k$).
    pub cumulative_variance_ratio: Vec<f64>,
    /// Per-sample $A$-component Hotelling $T_i^2$ leverage statistic.
    pub hotelling_t2: Vec<f64>,
    /// 95% critical threshold $T_{\text{crit}}^2(0.05; A, N)$ from exact $F$-distribution.
    pub hotelling_t2_limit_95: f64,
    /// 99% critical threshold $T_{\text{crit}}^2(0.01; A, N)$.
    pub hotelling_t2_limit_99: f64,
    /// Per-sample $Q$-residual (`SPE`) squared reconstruction error $\|\tilde{x}_i - \hat{x}_i\|_2^2$.
    pub q_residuals: Vec<f64>,
    /// Jackson–Mudholkar 95% $Q$-residual critical limit.
    pub q_residual_limit_95: f64,
    /// Global 95% Hotelling $T^2$ ellipse on `(PC1, PC2)`.
    pub global_ellipse_95: ConfidenceEllipse2D,
    /// Per-group 95% Hotelling $T^2$ covariance ellipses on `(PC1, PC2)`.
    pub group_ellipses_95: Vec<ConfidenceEllipse2D>,
}

/// Multi-component NIPALS Partial Least Squares Discriminant Analysis (`PLS-DA`) result.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlsDaResult {
    pub control_group: String,
    pub case_group: String,
    /// Number of latent variables $A$.
    pub n_components: usize,
    /// Sample indices from the full cohort included in this binary contrast.
    pub active_sample_indices: Vec<usize>,
    /// Binary target vector $y_i \in \{0.0, 1.0\}$ across `active_sample_indices`.
    pub y_binary: Vec<f64>,
    /// Latent score matrix $T \in \mathbb{R}^{N_{\text{act}} \times A}$.
    pub scores: Vec<Vec<f64>>,
    /// $X$-loading matrix $P \in \mathbb{R}^{A \times P}$.
    pub x_loadings: Vec<Vec<f64>>,
    /// Normalized weight matrix $W \in \mathbb{R}^{A \times P}$.
    pub x_weights: Vec<Vec<f64>>,
    /// $Y$-loadings $c \in \mathbb{R}^A$.
    pub y_loadings: Vec<f64>,
    /// Fitted in-sample response $\hat{y} \in \mathbb{R}^{N_{\text{act}}}$.
    pub y_fitted: Vec<f64>,
    /// Out-of-fold cross-validated response $\hat{y}^{\text{CV}} \in \mathbb{R}^{N_{\text{act}}}$.
    pub y_cv_predicted: Vec<f64>,
    /// Cumulative explained $X$ variance $R^2 X$.
    pub r2_x: f64,
    /// Cumulative explained $Y$ variance $R^2 Y$.
    pub r2_y: f64,
    /// Leak-free stratified $k$-fold cross-validated $Q^2 = 1 - \text{PRESS}/\text{TSS}$.
    pub q2: f64,
    /// Variable Importance in Projection (`VIP`) across all $P$ buckets.
    pub vip_scores: Vec<f64>,
}

/// Single permutation trial summary for `Westerhuis et al. (2008)` model validation.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PermutationTrial {
    /// Absolute Pearson correlation $|r(y_{\text{true}}, y_{\text{perm}})| \in [0, 1]$.
    pub abs_label_corr: f64,
    /// In-sample $R^2Y$ on permuted labels.
    pub r2_y: f64,
    /// Leak-free stratified $k$-fold CV $Q^2$ on permuted labels.
    pub q2: f64,
}

/// `N`-permutation test summary guarding supervised chemometric models against overfitting.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PermutationTestSummary {
    pub n_permutations: usize,
    pub observed_r2_y: f64,
    pub observed_q2: f64,
    /// Exact Monte Carlo $p$-value $(1 + \#\{Q^2_{\text{perm}} \ge Q^2_{\text{obs}}\}) / (1 + N_{\text{perm}})$.
    pub p_value_q2: f64,
    /// Exact Monte Carlo $p$-value $(1 + \#\{R^2Y_{\text{perm}} \ge R^2Y_{\text{obs}}\}) / (1 + N_{\text{perm}})$.
    pub p_value_r2_y: f64,
    /// Mean permuted $Q^2$ under the null distribution.
    pub mean_null_q2: f64,
    /// Linear regression intercept of $Q^2_{\text{perm}}$ at $|r(y, y_{\text{perm}})| = 0$.
    pub q2_intercept: f64,
    /// `true` iff `observed_q2 >= 0.40 && p_value_q2 <= 0.05`.
    pub is_validated: bool,
    /// Recorded permutation trials.
    pub trials: Vec<PermutationTrial>,
}

/// Orthogonal Projections to Latent Structures Discriminant Analysis (`OPLS-DA`,
/// Trygg & Wold 2002; Bylesjö et al. 2006; Wiklund et al. 2008).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OplsDaResult {
    pub control_group: String,
    pub case_group: String,
    /// Number of orthogonal components $A_o$ extracted.
    pub n_orthogonal: usize,
    /// Sample indices from the full cohort included in this binary contrast.
    pub active_sample_indices: Vec<usize>,
    /// Binary class vector $y_i \in \{0.0, 1.0\}$.
    pub y_binary: Vec<f64>,
    /// Predictive score vector $t_p \in \mathbb{R}^{N_{\text{act}}}$.
    pub predictive_scores: Vec<f64>,
    /// Predictive loading vector $p_p \in \mathbb{R}^P$.
    pub predictive_loadings: Vec<f64>,
    /// Unit-norm predictive weight vector $w_p \in \mathbb{R}^P$.
    pub predictive_weights: Vec<f64>,
    /// Predictive scalar regression weight $c_p$.
    pub predictive_y_weight: f64,
    /// Orthogonal score vectors $T_o \in \mathbb{R}^{A_o \times N_{\text{act}}}$ (`orthogonal_scores[k][i]`).
    pub orthogonal_scores: Vec<Vec<f64>>,
    /// Orthogonal loading vectors $P_o \in \mathbb{R}^{A_o \times P}$.
    pub orthogonal_loadings: Vec<Vec<f64>>,
    /// Orthogonal weight vectors $W_o \in \mathbb{R}^{A_o \times P}$.
    pub orthogonal_weights: Vec<Vec<f64>>,
    /// Fitted in-sample response $\hat{y} \in \mathbb{R}^{N_{\text{act}}}$.
    pub y_fitted: Vec<f64>,
    /// Out-of-fold cross-validated response $\hat{y}^{\text{CV}} \in \mathbb{R}^{N_{\text{act}}}$.
    pub y_cv_predicted: Vec<f64>,
    /// Fraction of $X$ variance captured by the predictive component ($R^2 X_p$).
    pub r2_x_predictive: f64,
    /// Fraction of $X$ variance captured by orthogonal components ($R^2 X_o$).
    pub r2_x_orthogonal: f64,
    /// Total explained $X$ variance ($R^2 X_p + R^2 X_o$).
    pub r2_x_total: f64,
    /// Explained class variance ($R^2 Y$).
    pub r2_y: f64,
    /// Leak-free stratified $k$-fold cross-validated $Q^2$.
    pub q2: f64,
    /// Predictive `VIP` scores ($\sqrt{P}\,|w_{p,j}|$).
    pub vip_predictive: Vec<f64>,
    /// Wiklund et al. (2008) `S-plot` covariance vector $\operatorname{Cov}(t_p, \tilde{X}_j)$.
    pub s_plot_cov: Vec<f64>,
    /// Wiklund et al. (2008) `S-plot` correlation vector $\operatorname{Corr}(t_p, \tilde{X}_j) \in [-1, 1]$.
    pub s_plot_corr: Vec<f64>,
    /// Global 95% Hotelling $T^2$ ellipse in `(t_p, t_o1)` score space.
    pub global_ellipse_95: ConfidenceEllipse2D,
    /// Per-class 95% Hotelling $T^2$ ellipses in `(t_p, t_o1)` score space.
    pub group_ellipses_95: Vec<ConfidenceEllipse2D>,
    /// Permutation overfitting validation report.
    pub permutation: PermutationTestSummary,
}

/// Ranked candidate biomarker combining multivariate (`OPLS-DA` `S-plot` + `VIP`) and univariate
/// (`Welch t` + `BH-FDR` + $\log_2\text{FC}$) evidence with `BMRB` metabolite annotation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BiomarkerCandidate {
    /// `1`-based biomarker rank (`1` = strongest combined evidence).
    pub rank: usize,
    /// Bucket index in `buckets` (`0..P`).
    pub bucket_index: usize,
    pub center_ppm: f64,
    pub low_ppm: f64,
    pub high_ppm: f64,
    pub control_mean: f64,
    pub case_mean: f64,
    /// Ratio `case_mean / control_mean` (stabilized).
    pub fold_change: f64,
    /// $\log_2(\text{case\_mean} / \text{control\_mean})$.
    pub log2_fold_change: f64,
    /// Welch unequal-variance two-sample $t$-statistic (`case - control`).
    pub welch_t: f64,
    /// Two-sided Welch–Satterthwaite $p$-value.
    pub p_value: f64,
    /// Benjamini–Hochberg FDR-adjusted $q$-value.
    pub fdr_q: f64,
    /// Predictive `VIP` score (`OPLS-DA`, or `PLS-DA` fallback).
    pub vip: f64,
    /// `OPLS-DA` `S-plot` $\operatorname{Cov}(t_p, \tilde{X}_j)$.
    pub s_plot_cov: f64,
    /// `OPLS-DA` `S-plot` $\operatorname{Corr}(t_p, \tilde{X}_j) \in [-1, 1]$.
    pub s_plot_corr: f64,
    /// Composite ranking score combining `VIP`, `|p(corr)|`, and `-log10(FDR q)`.
    pub composite_score: f64,
    /// Top matched `BMRB` metabolite name (if within chemical-shift & `STOCSY` criteria).
    pub putative_metabolite: Option<String>,
    /// Top matched `BMRB` accession ID (e.g., `"bmse000036"`).
    pub putative_bmrb_id: Option<String>,
}

/// Statistical Total Correlation Spectroscopy (`STOCSY`, Cloarec et al. 2005) result from a driver peak.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StocsyResult {
    /// Selected driver bucket index (`0..P`).
    pub driver_bucket_index: usize,
    /// Driver chemical shift in ppm.
    pub driver_ppm: f64,
    /// Bucket-level covariance $\operatorname{Cov}(d, X_j)$ across `0..P`.
    pub bucket_covariance: Vec<f64>,
    /// Bucket-level Pearson correlation $r_j \in [-1, 1]$ across `0..P`.
    pub bucket_correlation: Vec<f64>,
    /// Bucket-level squared Pearson correlation $r_j^2 \in [0, 1]$ across `0..P`.
    pub bucket_r2: Vec<f64>,
    /// Optional full-resolution spectral covariance trace $\operatorname{Cov}(d, S(\delta_m))$ across `0..M`.
    pub spectral_covariance: Vec<f64>,
    /// Optional full-resolution spectral $r^2(\delta_m) \in [0, 1]$ heat-color vector across `0..M`.
    pub spectral_r2: Vec<f64>,
    /// Ranked `BMRB` reference metabolite matches co-varying with this driver peak.
    pub metabolite_matches: Vec<MetaboliteMatchScore>,
}

/// Multiplet pattern code for reference `BMRB` biofluid metabolites.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MultipletCode {
    Singlet,
    Doublet,
    Triplet,
    Quartet,
    DoubletOfDoublets,
    Multiplet,
}

impl MultipletCode {
    pub fn short_label(&self) -> &'static str {
        match self {
            Self::Singlet => "s",
            Self::Doublet => "d",
            Self::Triplet => "t",
            Self::Quartet => "q",
            Self::DoubletOfDoublets => "dd",
            Self::Multiplet => "m",
        }
    }
}

/// Single ${}^{1}\text{H}$ multiplet resonance of a reference `BMRB` metabolite (`D2O`, pH `7.0–7.4`).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ReferenceMultiplet1H {
    /// Functional group label (e.g., `"CH3"`, `"CH"`, `"gamma1-CH3"`).
    pub assignment: &'static str,
    /// Canonical chemical shift in ppm relative to `DSS`/`TSP` (`0.00 ppm`).
    pub center_ppm: f64,
    /// Equivalent proton stoichiometry ($n_H \ge 1$).
    pub protons: u8,
    /// Spin-coupling pattern.
    pub multiplicity: MultipletCode,
    /// Primary scalar coupling constants $J$ in Hz (`empty` for singlets).
    pub couplings_hz: &'static [f64],
    /// Chemical-shift matching tolerance in ppm (wider for pH-sensitive ionizable groups).
    pub shift_tol_ppm: f64,
}

/// Public-domain CC0 `BMRB` human biofluid metabolite spin-system entry.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ReferenceMetaboliteEntry {
    /// Common biochemical name (e.g., `"L-Lactate"`).
    pub name: &'static str,
    /// Public `BMRB` metabolomics accession (`bmse*`).
    pub bmrb_id: &'static str,
    /// Molecular formula.
    pub formula: &'static str,
    /// Observable non-exchangeable ${}^{1}\text{H}$ multiplets in $\text{D}_2\text{O}$ at pH `7.0–7.4`.
    pub multiplets: &'static [ReferenceMultiplet1H],
}

/// Scored `STOCSY` match against a `BMRB` reference metabolite.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MetaboliteMatchScore {
    pub name: String,
    pub bmrb_id: String,
    pub formula: String,
    /// Overall match confidence score in `[0, 1]`.
    pub score: f64,
    /// Number of observable (unmasked) multiplets in the reference spin system.
    pub observable_multiplets: usize,
    /// Number of observable multiplets recovered with positive `STOCSY` $r^2 \ge 0.50$.
    pub matched_multiplets: usize,
    /// Proton-weighted mean positive `STOCSY` $r^2$ across observable multiplets.
    pub mean_stocsy_r2: f64,
    /// Cosine similarity between observed positive `STOCSY` covariances and theoretical proton counts.
    pub stoichiometry_cosine: f64,
    /// Per-multiplet `(ref_ppm, matched_bucket_ppm, stocsy_r2)` summary.
    pub matched_peaks: Vec<(f64, f64, f64)>,
}

/// Complete output of the untargeted 1D NMR metabolomics & biomarker studio pipeline.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MetabolomicsStudyResult {
    pub config: MetabolomicsPipelineConfig,
    pub sample_ids: Vec<String>,
    pub sample_groups: Vec<String>,
    /// Shared descending chemical-shift grid $\delta_0 > \delta_1 > \dots > \delta_{M-1}$.
    pub common_ppm_grid: Vec<f64>,
    /// Boolean mask (`true` = excluded solvent/water/urea point) across `0..M`.
    pub masked_grid_points: Vec<bool>,
    /// Aligned, masked `N x M` spectral matrix (`aligned_spectra[sample_idx][grid_idx]`).
    pub aligned_spectra: Vec<Vec<f64>>,
    /// Extracted spectral buckets (`0..P`), ordered from high ppm to low ppm.
    pub buckets: Vec<SpectralBucket>,
    /// Unnormalized bucket integral matrix $X^{\text{raw}} \in \mathbb{R}^{N \times P}$.
    pub raw_bucket_matrix: Vec<Vec<f64>>,
    /// Dilution-normalized bucket matrix $X^{\text{norm}} \in \mathbb{R}^{N \times P}$.
    pub normalized_bucket_matrix: Vec<Vec<f64>>,
    /// Centered & scaled matrix $\tilde{X} \in \mathbb{R}^{N \times P}$ fed to PCA / PLS-DA / OPLS-DA.
    pub scaled_bucket_matrix: Vec<Vec<f64>>,
    /// Estimated sample dilution / scale factors $d_i$ (`X_norm[i] = X_raw[i] / d_i`).
    pub dilution_factors: Vec<f64>,
    /// Principal Component Analysis model across all $N$ samples.
    pub pca: PcaResult,
    /// Optional binary NIPALS `PLS-DA` model (`Some` whenever $\ge 2$ groups with $\ge 2$ samples exist).
    pub pls_da: Option<PlsDaResult>,
    /// Optional binary `OPLS-DA` model with stratified CV $Q^2$ and permutation test.
    pub opls_da: Option<OplsDaResult>,
    /// Ranked candidate biomarkers across all $P$ buckets.
    pub biomarkers: Vec<BiomarkerCandidate>,
    /// Precomputed `STOCSY` from the #1 ranked biomarker bucket (`None` if `buckets.is_empty()`).
    pub top_biomarker_stocsy: Option<StocsyResult>,
}

// ============================================================================
// 2. Built-In CC0 / BMRB Human Biofluid Metabolite Reference Library (28 Entries)
// ============================================================================

const MULT_LACTATE: &[ReferenceMultiplet1H] = &[
    ReferenceMultiplet1H {
        assignment: "CH3",
        center_ppm: 1.330,
        protons: 3,
        multiplicity: MultipletCode::Doublet,
        couplings_hz: &[6.9],
        shift_tol_ppm: 0.025,
    },
    ReferenceMultiplet1H {
        assignment: "CH",
        center_ppm: 4.110,
        protons: 1,
        multiplicity: MultipletCode::Quartet,
        couplings_hz: &[6.9],
        shift_tol_ppm: 0.030,
    },
];

const MULT_ALANINE: &[ReferenceMultiplet1H] = &[
    ReferenceMultiplet1H {
        assignment: "CH3",
        center_ppm: 1.478,
        protons: 3,
        multiplicity: MultipletCode::Doublet,
        couplings_hz: &[7.2],
        shift_tol_ppm: 0.025,
    },
    ReferenceMultiplet1H {
        assignment: "alpha-CH",
        center_ppm: 3.778,
        protons: 1,
        multiplicity: MultipletCode::Quartet,
        couplings_hz: &[7.2],
        shift_tol_ppm: 0.025,
    },
];

const MULT_VALINE: &[ReferenceMultiplet1H] = &[
    ReferenceMultiplet1H {
        assignment: "gamma1-CH3",
        center_ppm: 0.985,
        protons: 3,
        multiplicity: MultipletCode::Doublet,
        couplings_hz: &[7.0],
        shift_tol_ppm: 0.025,
    },
    ReferenceMultiplet1H {
        assignment: "gamma2-CH3",
        center_ppm: 1.037,
        protons: 3,
        multiplicity: MultipletCode::Doublet,
        couplings_hz: &[7.0],
        shift_tol_ppm: 0.025,
    },
    ReferenceMultiplet1H {
        assignment: "beta-CH",
        center_ppm: 2.268,
        protons: 1,
        multiplicity: MultipletCode::Multiplet,
        couplings_hz: &[7.0, 4.4],
        shift_tol_ppm: 0.030,
    },
    ReferenceMultiplet1H {
        assignment: "alpha-CH",
        center_ppm: 3.605,
        protons: 1,
        multiplicity: MultipletCode::Doublet,
        couplings_hz: &[4.4],
        shift_tol_ppm: 0.025,
    },
];

const MULT_LEUCINE: &[ReferenceMultiplet1H] = &[
    ReferenceMultiplet1H {
        assignment: "delta1-CH3",
        center_ppm: 0.952,
        protons: 3,
        multiplicity: MultipletCode::Doublet,
        couplings_hz: &[6.4],
        shift_tol_ppm: 0.025,
    },
    ReferenceMultiplet1H {
        assignment: "delta2-CH3",
        center_ppm: 0.964,
        protons: 3,
        multiplicity: MultipletCode::Doublet,
        couplings_hz: &[6.4],
        shift_tol_ppm: 0.025,
    },
    ReferenceMultiplet1H {
        assignment: "beta-CH2/gamma-CH",
        center_ppm: 1.706,
        protons: 3,
        multiplicity: MultipletCode::Multiplet,
        couplings_hz: &[6.4],
        shift_tol_ppm: 0.030,
    },
    ReferenceMultiplet1H {
        assignment: "alpha-CH",
        center_ppm: 3.728,
        protons: 1,
        multiplicity: MultipletCode::Triplet,
        couplings_hz: &[6.2],
        shift_tol_ppm: 0.025,
    },
];

const MULT_ISOLEUCINE: &[ReferenceMultiplet1H] = &[
    ReferenceMultiplet1H {
        assignment: "delta-CH3",
        center_ppm: 0.933,
        protons: 3,
        multiplicity: MultipletCode::Triplet,
        couplings_hz: &[7.4],
        shift_tol_ppm: 0.025,
    },
    ReferenceMultiplet1H {
        assignment: "gamma2-CH3",
        center_ppm: 1.005,
        protons: 3,
        multiplicity: MultipletCode::Doublet,
        couplings_hz: &[7.0],
        shift_tol_ppm: 0.025,
    },
    ReferenceMultiplet1H {
        assignment: "gamma1a-CH2",
        center_ppm: 1.255,
        protons: 1,
        multiplicity: MultipletCode::Multiplet,
        couplings_hz: &[7.4, 7.0],
        shift_tol_ppm: 0.030,
    },
    ReferenceMultiplet1H {
        assignment: "gamma1b-CH2",
        center_ppm: 1.465,
        protons: 1,
        multiplicity: MultipletCode::Multiplet,
        couplings_hz: &[7.4, 7.0],
        shift_tol_ppm: 0.030,
    },
    ReferenceMultiplet1H {
        assignment: "beta-CH",
        center_ppm: 1.975,
        protons: 1,
        multiplicity: MultipletCode::Multiplet,
        couplings_hz: &[7.0, 4.0],
        shift_tol_ppm: 0.030,
    },
    ReferenceMultiplet1H {
        assignment: "alpha-CH",
        center_ppm: 3.667,
        protons: 1,
        multiplicity: MultipletCode::Doublet,
        couplings_hz: &[4.0],
        shift_tol_ppm: 0.025,
    },
];

const MULT_ALPHA_D_GLUCOSE: &[ReferenceMultiplet1H] = &[
    ReferenceMultiplet1H {
        assignment: "H1-alpha",
        center_ppm: 5.233,
        protons: 1,
        multiplicity: MultipletCode::Doublet,
        couplings_hz: &[3.7],
        shift_tol_ppm: 0.025,
    },
    ReferenceMultiplet1H {
        assignment: "H2",
        center_ppm: 3.533,
        protons: 1,
        multiplicity: MultipletCode::DoubletOfDoublets,
        couplings_hz: &[9.8, 3.7],
        shift_tol_ppm: 0.025,
    },
    ReferenceMultiplet1H {
        assignment: "H3",
        center_ppm: 3.710,
        protons: 1,
        multiplicity: MultipletCode::Triplet,
        couplings_hz: &[9.5],
        shift_tol_ppm: 0.025,
    },
    ReferenceMultiplet1H {
        assignment: "H4",
        center_ppm: 3.408,
        protons: 1,
        multiplicity: MultipletCode::Triplet,
        couplings_hz: &[9.4],
        shift_tol_ppm: 0.025,
    },
    ReferenceMultiplet1H {
        assignment: "H5/H6",
        center_ppm: 3.833,
        protons: 3,
        multiplicity: MultipletCode::Multiplet,
        couplings_hz: &[9.5, 4.5],
        shift_tol_ppm: 0.030,
    },
];

const MULT_BETA_D_GLUCOSE: &[ReferenceMultiplet1H] = &[
    ReferenceMultiplet1H {
        assignment: "H1-beta",
        center_ppm: 4.642,
        protons: 1,
        multiplicity: MultipletCode::Doublet,
        couplings_hz: &[7.9],
        shift_tol_ppm: 0.025,
    },
    ReferenceMultiplet1H {
        assignment: "H2",
        center_ppm: 3.242,
        protons: 1,
        multiplicity: MultipletCode::DoubletOfDoublets,
        couplings_hz: &[9.2, 7.9],
        shift_tol_ppm: 0.025,
    },
    ReferenceMultiplet1H {
        assignment: "H3/H5",
        center_ppm: 3.475,
        protons: 2,
        multiplicity: MultipletCode::Triplet,
        couplings_hz: &[9.2],
        shift_tol_ppm: 0.028,
    },
    ReferenceMultiplet1H {
        assignment: "H4",
        center_ppm: 3.400,
        protons: 1,
        multiplicity: MultipletCode::Triplet,
        couplings_hz: &[9.3],
        shift_tol_ppm: 0.025,
    },
    ReferenceMultiplet1H {
        assignment: "H6b",
        center_ppm: 3.720,
        protons: 1,
        multiplicity: MultipletCode::DoubletOfDoublets,
        couplings_hz: &[12.2, 5.8],
        shift_tol_ppm: 0.025,
    },
    ReferenceMultiplet1H {
        assignment: "H6a",
        center_ppm: 3.895,
        protons: 1,
        multiplicity: MultipletCode::DoubletOfDoublets,
        couplings_hz: &[12.2, 2.1],
        shift_tol_ppm: 0.025,
    },
];

const MULT_CITRATE: &[ReferenceMultiplet1H] = &[
    ReferenceMultiplet1H {
        assignment: "CH2a",
        center_ppm: 2.535,
        protons: 2,
        multiplicity: MultipletCode::Doublet,
        couplings_hz: &[15.3],
        shift_tol_ppm: 0.065,
    },
    ReferenceMultiplet1H {
        assignment: "CH2b",
        center_ppm: 2.675,
        protons: 2,
        multiplicity: MultipletCode::Doublet,
        couplings_hz: &[15.3],
        shift_tol_ppm: 0.065,
    },
];

const MULT_SUCCINATE: &[ReferenceMultiplet1H] = &[ReferenceMultiplet1H {
    assignment: "2xCH2",
    center_ppm: 2.405,
    protons: 4,
    multiplicity: MultipletCode::Singlet,
    couplings_hz: &[],
    shift_tol_ppm: 0.030,
}];

const MULT_PYRUVATE: &[ReferenceMultiplet1H] = &[ReferenceMultiplet1H {
    assignment: "CH3",
    center_ppm: 2.368,
    protons: 3,
    multiplicity: MultipletCode::Singlet,
    couplings_hz: &[],
    shift_tol_ppm: 0.025,
}];

const MULT_ACETATE: &[ReferenceMultiplet1H] = &[ReferenceMultiplet1H {
    assignment: "CH3",
    center_ppm: 1.918,
    protons: 3,
    multiplicity: MultipletCode::Singlet,
    couplings_hz: &[],
    shift_tol_ppm: 0.025,
}];

const MULT_ACETOACETATE: &[ReferenceMultiplet1H] = &[
    ReferenceMultiplet1H {
        assignment: "CH3",
        center_ppm: 2.278,
        protons: 3,
        multiplicity: MultipletCode::Singlet,
        couplings_hz: &[],
        shift_tol_ppm: 0.025,
    },
    ReferenceMultiplet1H {
        assignment: "CH2",
        center_ppm: 3.442,
        protons: 2,
        multiplicity: MultipletCode::Singlet,
        couplings_hz: &[],
        shift_tol_ppm: 0.025,
    },
];

const MULT_3_HYDROXYBUTYRATE: &[ReferenceMultiplet1H] = &[
    ReferenceMultiplet1H {
        assignment: "gamma-CH3",
        center_ppm: 1.196,
        protons: 3,
        multiplicity: MultipletCode::Doublet,
        couplings_hz: &[6.3],
        shift_tol_ppm: 0.025,
    },
    ReferenceMultiplet1H {
        assignment: "alpha-CH2a",
        center_ppm: 2.305,
        protons: 1,
        multiplicity: MultipletCode::DoubletOfDoublets,
        couplings_hz: &[14.5, 6.2],
        shift_tol_ppm: 0.028,
    },
    ReferenceMultiplet1H {
        assignment: "alpha-CH2b",
        center_ppm: 2.405,
        protons: 1,
        multiplicity: MultipletCode::DoubletOfDoublets,
        couplings_hz: &[14.5, 7.3],
        shift_tol_ppm: 0.028,
    },
    ReferenceMultiplet1H {
        assignment: "beta-CH",
        center_ppm: 4.150,
        protons: 1,
        multiplicity: MultipletCode::Multiplet,
        couplings_hz: &[6.5],
        shift_tol_ppm: 0.030,
    },
];

const MULT_CREATINE: &[ReferenceMultiplet1H] = &[
    ReferenceMultiplet1H {
        assignment: "N-CH3",
        center_ppm: 3.032,
        protons: 3,
        multiplicity: MultipletCode::Singlet,
        couplings_hz: &[],
        shift_tol_ppm: 0.022,
    },
    ReferenceMultiplet1H {
        assignment: "N-CH2",
        center_ppm: 3.928,
        protons: 2,
        multiplicity: MultipletCode::Singlet,
        couplings_hz: &[],
        shift_tol_ppm: 0.025,
    },
];

const MULT_CREATININE: &[ReferenceMultiplet1H] = &[
    ReferenceMultiplet1H {
        assignment: "N-CH3",
        center_ppm: 3.043,
        protons: 3,
        multiplicity: MultipletCode::Singlet,
        couplings_hz: &[],
        shift_tol_ppm: 0.025,
    },
    ReferenceMultiplet1H {
        assignment: "CH2",
        center_ppm: 4.052,
        protons: 2,
        multiplicity: MultipletCode::Singlet,
        couplings_hz: &[],
        shift_tol_ppm: 0.035,
    },
];

const MULT_TMAO: &[ReferenceMultiplet1H] = &[ReferenceMultiplet1H {
    assignment: "3xCH3",
    center_ppm: 3.266,
    protons: 9,
    multiplicity: MultipletCode::Singlet,
    couplings_hz: &[],
    shift_tol_ppm: 0.025,
}];

const MULT_TAURINE: &[ReferenceMultiplet1H] = &[
    ReferenceMultiplet1H {
        assignment: "CH2-NH3",
        center_ppm: 3.260,
        protons: 2,
        multiplicity: MultipletCode::Triplet,
        couplings_hz: &[6.6],
        shift_tol_ppm: 0.025,
    },
    ReferenceMultiplet1H {
        assignment: "CH2-SO3",
        center_ppm: 3.425,
        protons: 2,
        multiplicity: MultipletCode::Triplet,
        couplings_hz: &[6.6],
        shift_tol_ppm: 0.025,
    },
];

const MULT_GLYCINE: &[ReferenceMultiplet1H] = &[ReferenceMultiplet1H {
    assignment: "CH2",
    center_ppm: 3.555,
    protons: 2,
    multiplicity: MultipletCode::Singlet,
    couplings_hz: &[],
    shift_tol_ppm: 0.025,
}];

const MULT_GLUTAMINE: &[ReferenceMultiplet1H] = &[
    ReferenceMultiplet1H {
        assignment: "beta-CH2",
        center_ppm: 2.135,
        protons: 2,
        multiplicity: MultipletCode::Multiplet,
        couplings_hz: &[7.2],
        shift_tol_ppm: 0.028,
    },
    ReferenceMultiplet1H {
        assignment: "gamma-CH2",
        center_ppm: 2.452,
        protons: 2,
        multiplicity: MultipletCode::Multiplet,
        couplings_hz: &[7.5],
        shift_tol_ppm: 0.028,
    },
    ReferenceMultiplet1H {
        assignment: "alpha-CH",
        center_ppm: 3.770,
        protons: 1,
        multiplicity: MultipletCode::Triplet,
        couplings_hz: &[6.2],
        shift_tol_ppm: 0.025,
    },
];

const MULT_GLUTAMATE: &[ReferenceMultiplet1H] = &[
    ReferenceMultiplet1H {
        assignment: "beta-CH2",
        center_ppm: 2.062,
        protons: 2,
        multiplicity: MultipletCode::Multiplet,
        couplings_hz: &[7.2],
        shift_tol_ppm: 0.028,
    },
    ReferenceMultiplet1H {
        assignment: "gamma-CH2",
        center_ppm: 2.348,
        protons: 2,
        multiplicity: MultipletCode::Multiplet,
        couplings_hz: &[7.4],
        shift_tol_ppm: 0.028,
    },
    ReferenceMultiplet1H {
        assignment: "alpha-CH",
        center_ppm: 3.755,
        protons: 1,
        multiplicity: MultipletCode::DoubletOfDoublets,
        couplings_hz: &[7.2, 4.7],
        shift_tol_ppm: 0.025,
    },
];

const MULT_HISTIDINE: &[ReferenceMultiplet1H] = &[
    ReferenceMultiplet1H {
        assignment: "beta-CH2a",
        center_ppm: 3.155,
        protons: 1,
        multiplicity: MultipletCode::DoubletOfDoublets,
        couplings_hz: &[15.5, 7.8],
        shift_tol_ppm: 0.035,
    },
    ReferenceMultiplet1H {
        assignment: "beta-CH2b",
        center_ppm: 3.245,
        protons: 1,
        multiplicity: MultipletCode::DoubletOfDoublets,
        couplings_hz: &[15.5, 4.8],
        shift_tol_ppm: 0.035,
    },
    ReferenceMultiplet1H {
        assignment: "alpha-CH",
        center_ppm: 3.985,
        protons: 1,
        multiplicity: MultipletCode::DoubletOfDoublets,
        couplings_hz: &[7.8, 4.8],
        shift_tol_ppm: 0.030,
    },
    ReferenceMultiplet1H {
        assignment: "ring-C4H",
        center_ppm: 7.085,
        protons: 1,
        multiplicity: MultipletCode::Singlet,
        couplings_hz: &[],
        shift_tol_ppm: 0.085,
    },
    ReferenceMultiplet1H {
        assignment: "ring-C2H",
        center_ppm: 7.865,
        protons: 1,
        multiplicity: MultipletCode::Singlet,
        couplings_hz: &[],
        shift_tol_ppm: 0.095,
    },
];

const MULT_PHENYLALANINE: &[ReferenceMultiplet1H] = &[
    ReferenceMultiplet1H {
        assignment: "beta-CH2a",
        center_ppm: 3.125,
        protons: 1,
        multiplicity: MultipletCode::DoubletOfDoublets,
        couplings_hz: &[14.5, 8.0],
        shift_tol_ppm: 0.028,
    },
    ReferenceMultiplet1H {
        assignment: "beta-CH2b",
        center_ppm: 3.280,
        protons: 1,
        multiplicity: MultipletCode::DoubletOfDoublets,
        couplings_hz: &[14.5, 5.2],
        shift_tol_ppm: 0.028,
    },
    ReferenceMultiplet1H {
        assignment: "alpha-CH",
        center_ppm: 3.990,
        protons: 1,
        multiplicity: MultipletCode::DoubletOfDoublets,
        couplings_hz: &[8.0, 5.2],
        shift_tol_ppm: 0.028,
    },
    ReferenceMultiplet1H {
        assignment: "ortho-CH",
        center_ppm: 7.328,
        protons: 2,
        multiplicity: MultipletCode::Doublet,
        couplings_hz: &[7.3],
        shift_tol_ppm: 0.025,
    },
    ReferenceMultiplet1H {
        assignment: "para-CH",
        center_ppm: 7.375,
        protons: 1,
        multiplicity: MultipletCode::Triplet,
        couplings_hz: &[7.3],
        shift_tol_ppm: 0.025,
    },
    ReferenceMultiplet1H {
        assignment: "meta-CH",
        center_ppm: 7.425,
        protons: 2,
        multiplicity: MultipletCode::Triplet,
        couplings_hz: &[7.4],
        shift_tol_ppm: 0.025,
    },
];

const MULT_TYROSINE: &[ReferenceMultiplet1H] = &[
    ReferenceMultiplet1H {
        assignment: "beta-CH2a",
        center_ppm: 3.055,
        protons: 1,
        multiplicity: MultipletCode::DoubletOfDoublets,
        couplings_hz: &[14.5, 8.0],
        shift_tol_ppm: 0.028,
    },
    ReferenceMultiplet1H {
        assignment: "beta-CH2b",
        center_ppm: 3.195,
        protons: 1,
        multiplicity: MultipletCode::DoubletOfDoublets,
        couplings_hz: &[14.5, 5.2],
        shift_tol_ppm: 0.028,
    },
    ReferenceMultiplet1H {
        assignment: "alpha-CH",
        center_ppm: 3.935,
        protons: 1,
        multiplicity: MultipletCode::DoubletOfDoublets,
        couplings_hz: &[8.0, 5.2],
        shift_tol_ppm: 0.028,
    },
    ReferenceMultiplet1H {
        assignment: "ring-H3/H5",
        center_ppm: 6.895,
        protons: 2,
        multiplicity: MultipletCode::Doublet,
        couplings_hz: &[8.4],
        shift_tol_ppm: 0.025,
    },
    ReferenceMultiplet1H {
        assignment: "ring-H2/H6",
        center_ppm: 7.190,
        protons: 2,
        multiplicity: MultipletCode::Doublet,
        couplings_hz: &[8.4],
        shift_tol_ppm: 0.025,
    },
];

const MULT_FORMATE: &[ReferenceMultiplet1H] = &[ReferenceMultiplet1H {
    assignment: "CH",
    center_ppm: 8.448,
    protons: 1,
    multiplicity: MultipletCode::Singlet,
    couplings_hz: &[],
    shift_tol_ppm: 0.025,
}];

const MULT_HIPPURATE: &[ReferenceMultiplet1H] = &[
    ReferenceMultiplet1H {
        assignment: "CH2",
        center_ppm: 3.968,
        protons: 2,
        multiplicity: MultipletCode::Doublet,
        couplings_hz: &[5.8],
        shift_tol_ppm: 0.028,
    },
    ReferenceMultiplet1H {
        assignment: "meta-CH",
        center_ppm: 7.548,
        protons: 2,
        multiplicity: MultipletCode::Triplet,
        couplings_hz: &[7.6],
        shift_tol_ppm: 0.025,
    },
    ReferenceMultiplet1H {
        assignment: "para-CH",
        center_ppm: 7.635,
        protons: 1,
        multiplicity: MultipletCode::Triplet,
        couplings_hz: &[7.4],
        shift_tol_ppm: 0.025,
    },
    ReferenceMultiplet1H {
        assignment: "ortho-CH",
        center_ppm: 7.832,
        protons: 2,
        multiplicity: MultipletCode::Doublet,
        couplings_hz: &[7.6],
        shift_tol_ppm: 0.025,
    },
];

const MULT_MYO_INOSITOL: &[ReferenceMultiplet1H] = &[
    ReferenceMultiplet1H {
        assignment: "H5",
        center_ppm: 3.278,
        protons: 1,
        multiplicity: MultipletCode::Triplet,
        couplings_hz: &[9.3],
        shift_tol_ppm: 0.025,
    },
    ReferenceMultiplet1H {
        assignment: "H1/H3",
        center_ppm: 3.532,
        protons: 2,
        multiplicity: MultipletCode::DoubletOfDoublets,
        couplings_hz: &[9.9, 2.9],
        shift_tol_ppm: 0.025,
    },
    ReferenceMultiplet1H {
        assignment: "H4/H6",
        center_ppm: 3.620,
        protons: 2,
        multiplicity: MultipletCode::Triplet,
        couplings_hz: &[9.5],
        shift_tol_ppm: 0.025,
    },
    ReferenceMultiplet1H {
        assignment: "H2",
        center_ppm: 4.058,
        protons: 1,
        multiplicity: MultipletCode::Triplet,
        couplings_hz: &[2.9],
        shift_tol_ppm: 0.025,
    },
];

const MULT_ACETONE: &[ReferenceMultiplet1H] = &[ReferenceMultiplet1H {
    assignment: "2xCH3",
    center_ppm: 2.225,
    protons: 6,
    multiplicity: MultipletCode::Singlet,
    couplings_hz: &[],
    shift_tol_ppm: 0.025,
}];

const MULT_DIMETHYLAMINE: &[ReferenceMultiplet1H] = &[ReferenceMultiplet1H {
    assignment: "2xCH3",
    center_ppm: 2.720,
    protons: 6,
    multiplicity: MultipletCode::Singlet,
    couplings_hz: &[],
    shift_tol_ppm: 0.025,
}];

const BMRB_BIOFLUID_LIBRARY: &[ReferenceMetaboliteEntry] = &[
    ReferenceMetaboliteEntry {
        name: "L-Lactate",
        bmrb_id: "bmse000036",
        formula: "C3H5O3-",
        multiplets: MULT_LACTATE,
    },
    ReferenceMetaboliteEntry {
        name: "L-Alanine",
        bmrb_id: "bmse000028",
        formula: "C3H7NO2",
        multiplets: MULT_ALANINE,
    },
    ReferenceMetaboliteEntry {
        name: "L-Valine",
        bmrb_id: "bmse000049",
        formula: "C5H11NO2",
        multiplets: MULT_VALINE,
    },
    ReferenceMetaboliteEntry {
        name: "L-Leucine",
        bmrb_id: "bmse000038",
        formula: "C6H13NO2",
        multiplets: MULT_LEUCINE,
    },
    ReferenceMetaboliteEntry {
        name: "L-Isoleucine",
        bmrb_id: "bmse000035",
        formula: "C6H13NO2",
        multiplets: MULT_ISOLEUCINE,
    },
    ReferenceMetaboliteEntry {
        name: "alpha-D-Glucose",
        bmrb_id: "bmse000016a",
        formula: "C6H12O6",
        multiplets: MULT_ALPHA_D_GLUCOSE,
    },
    ReferenceMetaboliteEntry {
        name: "beta-D-Glucose",
        bmrb_id: "bmse000016b",
        formula: "C6H12O6",
        multiplets: MULT_BETA_D_GLUCOSE,
    },
    ReferenceMetaboliteEntry {
        name: "Citrate",
        bmrb_id: "bmse000074",
        formula: "C6H5O7^3-",
        multiplets: MULT_CITRATE,
    },
    ReferenceMetaboliteEntry {
        name: "Succinate",
        bmrb_id: "bmse000181",
        formula: "C4H4O4^2-",
        multiplets: MULT_SUCCINATE,
    },
    ReferenceMetaboliteEntry {
        name: "Pyruvate",
        bmrb_id: "bmse000174",
        formula: "C3H3O3-",
        multiplets: MULT_PYRUVATE,
    },
    ReferenceMetaboliteEntry {
        name: "Acetate",
        bmrb_id: "bmse000191",
        formula: "C2H3O2-",
        multiplets: MULT_ACETATE,
    },
    ReferenceMetaboliteEntry {
        name: "Acetoacetate",
        bmrb_id: "bmse000370",
        formula: "C4H5O3-",
        multiplets: MULT_ACETOACETATE,
    },
    ReferenceMetaboliteEntry {
        name: "3-Hydroxybutyrate",
        bmrb_id: "bmse000143",
        formula: "C4H7O3-",
        multiplets: MULT_3_HYDROXYBUTYRATE,
    },
    ReferenceMetaboliteEntry {
        name: "Creatine",
        bmrb_id: "bmse000059",
        formula: "C4H9N3O2",
        multiplets: MULT_CREATINE,
    },
    ReferenceMetaboliteEntry {
        name: "Creatinine",
        bmrb_id: "bmse000060",
        formula: "C4H7N3O",
        multiplets: MULT_CREATININE,
    },
    ReferenceMetaboliteEntry {
        name: "TMAO",
        bmrb_id: "bmse000279",
        formula: "C3H9NO",
        multiplets: MULT_TMAO,
    },
    ReferenceMetaboliteEntry {
        name: "Taurine",
        bmrb_id: "bmse000140",
        formula: "C2H7NO3S",
        multiplets: MULT_TAURINE,
    },
    ReferenceMetaboliteEntry {
        name: "Glycine",
        bmrb_id: "bmse000087",
        formula: "C2H5NO2",
        multiplets: MULT_GLYCINE,
    },
    ReferenceMetaboliteEntry {
        name: "L-Glutamine",
        bmrb_id: "bmse000033",
        formula: "C5H10N2O3",
        multiplets: MULT_GLUTAMINE,
    },
    ReferenceMetaboliteEntry {
        name: "L-Glutamate",
        bmrb_id: "bmse000032",
        formula: "C5H8NO4-",
        multiplets: MULT_GLUTAMATE,
    },
    ReferenceMetaboliteEntry {
        name: "L-Histidine",
        bmrb_id: "bmse000034",
        formula: "C6H9N3O2",
        multiplets: MULT_HISTIDINE,
    },
    ReferenceMetaboliteEntry {
        name: "L-Phenylalanine",
        bmrb_id: "bmse000043",
        formula: "C9H11NO2",
        multiplets: MULT_PHENYLALANINE,
    },
    ReferenceMetaboliteEntry {
        name: "L-Tyrosine",
        bmrb_id: "bmse000048",
        formula: "C9H11NO3",
        multiplets: MULT_TYROSINE,
    },
    ReferenceMetaboliteEntry {
        name: "Formate",
        bmrb_id: "bmse000206",
        formula: "CHO2-",
        multiplets: MULT_FORMATE,
    },
    ReferenceMetaboliteEntry {
        name: "Hippurate",
        bmrb_id: "bmse000222",
        formula: "C9H8NO3-",
        multiplets: MULT_HIPPURATE,
    },
    ReferenceMetaboliteEntry {
        name: "Myo-inositol",
        bmrb_id: "bmse000112",
        formula: "C6H12O6",
        multiplets: MULT_MYO_INOSITOL,
    },
    ReferenceMetaboliteEntry {
        name: "Acetone",
        bmrb_id: "bmse000201",
        formula: "C3H6O",
        multiplets: MULT_ACETONE,
    },
    ReferenceMetaboliteEntry {
        name: "Dimethylamine",
        bmrb_id: "bmse000277",
        formula: "C2H7N",
        multiplets: MULT_DIMETHYLAMINE,
    },
];

/// Return the built-in CC0 `BMRB` human biofluid ${}^{1}\text{H}$ metabolite reference library (`28` entries).
pub fn bmrb_biofluid_metabolite_library() -> &'static [ReferenceMetaboliteEntry] {
    BMRB_BIOFLUID_LIBRARY
}

/// Look up a reference metabolite entry by case-insensitive name or `BMRB` accession (`bmse*`).
pub fn lookup_bmrb_metabolite(query: &str) -> Option<&'static ReferenceMetaboliteEntry> {
    let q = query.trim().to_ascii_lowercase();
    BMRB_BIOFLUID_LIBRARY
        .iter()
        .find(|m| m.name.to_ascii_lowercase() == q || m.bmrb_id.to_ascii_lowercase() == q)
}

// ============================================================================
// 3. Synthetic Biofluid 1D Spectrum & Cohort Generator
// ============================================================================

/// Expand a `ReferenceMultiplet1H` into `(delta_ppm, relative_area)` Lorentzian sub-lines at `obs_mhz`.
fn expand_multiplet_lines(
    mult: &ReferenceMultiplet1H,
    obs_mhz: f64,
    shift_offset_ppm: f64,
) -> Vec<(f64, f64)> {
    let c0 = mult.center_ppm + shift_offset_ppm;
    let total_h = mult.protons as f64;
    let mhz = obs_mhz.max(100.0);
    let j0 = mult.couplings_hz.first().copied().unwrap_or(0.0) / mhz;
    let j1 = mult.couplings_hz.get(1).copied().unwrap_or(j0) / mhz;

    match mult.multiplicity {
        MultipletCode::Singlet => vec![(c0, total_h)],
        MultipletCode::Doublet => vec![
            (c0 - 0.5 * j0, 0.5 * total_h),
            (c0 + 0.5 * j0, 0.5 * total_h),
        ],
        MultipletCode::Triplet => vec![
            (c0 - j0, 0.25 * total_h),
            (c0, 0.50 * total_h),
            (c0 + j0, 0.25 * total_h),
        ],
        MultipletCode::Quartet => vec![
            (c0 - 1.5 * j0, 0.125 * total_h),
            (c0 - 0.5 * j0, 0.375 * total_h),
            (c0 + 0.5 * j0, 0.375 * total_h),
            (c0 + 1.5 * j0, 0.125 * total_h),
        ],
        MultipletCode::DoubletOfDoublets => vec![
            (c0 - 0.5 * (j0 + j1), 0.25 * total_h),
            (c0 - 0.5 * (j0 - j1), 0.25 * total_h),
            (c0 + 0.5 * (j0 - j1), 0.25 * total_h),
            (c0 + 0.5 * (j0 + j1), 0.25 * total_h),
        ],
        MultipletCode::Multiplet => {
            let d = if j0 > 0.0 { j0 } else { 6.5 / mhz };
            vec![
                (c0 - 1.5 * d, 0.10 * total_h),
                (c0 - 0.75 * d, 0.22 * total_h),
                (c0, 0.36 * total_h),
                (c0 + 0.75 * d, 0.22 * total_h),
                (c0 + 1.5 * d, 0.10 * total_h),
            ]
        }
    }
}

/// Parameters for synthesizing a realistic 1D ${}^{1}\text{H}$ biofluid spectrum.
#[derive(Clone, Debug)]
pub struct BiofluidSynthOptions {
    pub high_ppm: f64,
    pub low_ppm: f64,
    pub n_points: usize,
    pub obs_mhz: f64,
    pub linewidth_hz: f64,
    /// Overall sample dilution multiplier (`1.0` = undiluted, `0.2` = 5x diluted).
    pub dilution_factor: f64,
    /// Optional `TSP`/`DSS` internal standard concentration (`9H` singlet at `0.000 + tsp_offset_ppm`).
    pub tsp_concentration: f64,
    pub tsp_offset_ppm: f64,
    /// Optional residual water (`4.80 ppm`) & urea (`5.78 ppm`) artefact amplitudes.
    pub residual_water_amp: f64,
    pub urea_amp: f64,
    /// Optional per-metabolite pH shift overrides (`metabolite_name -> delta_ppm`).
    pub metabolite_shift_offsets: BTreeMap<String, f64>,
    /// Thermal Gaussian baseline noise $\sigma$.
    pub noise_sigma: f64,
}

impl Default for BiofluidSynthOptions {
    fn default() -> Self {
        Self {
            high_ppm: 9.60,
            low_ppm: -0.40,
            n_points: 10_001,
            obs_mhz: 600.0,
            linewidth_hz: 1.4,
            dilution_factor: 1.0,
            tsp_concentration: 0.0,
            tsp_offset_ppm: 0.0,
            residual_water_amp: 0.0,
            urea_amp: 0.0,
            metabolite_shift_offsets: BTreeMap::new(),
            noise_sigma: 1e-3,
        }
    }
}

/// Synthesize an area-calibrated Lorentzian 1D ${}^{1}\text{H}$ biofluid spectrum from `(metabolite_name, concentration)` pairs.
///
/// A unit-area normalized Lorentzian on the ppm axis is
/// $L(\delta; \delta_0, \gamma) = \frac{1}{\pi} \frac{\gamma}{(\delta - \delta_0)^2 + \gamma^2}$
/// where $\gamma = \text{FWHM}_{\text{ppm}} / 2 = \text{linewidth\_hz} / (2 \nu_{\text{obs}})$, so
/// integrating a multiplet of concentration $C$ and proton count $n_H$ yields exact area
/// $\text{dilution\_factor} \cdot C \cdot n_H$.
pub fn synthesize_biofluid_spectrum_1d(
    name: impl Into<String>,
    concentrations: &[(&str, f64)],
    opts: &BiofluidSynthOptions,
    rng: &mut Rng,
) -> Spectrum1D {
    let axis = AxisPpm::from_bounds(
        "1H",
        opts.n_points.max(64),
        opts.high_ppm,
        opts.low_ppm,
        opts.obs_mhz,
    );
    let gamma_ppm = (opts.linewidth_hz.max(0.2) / (2.0 * opts.obs_mhz.max(100.0))).max(1e-5);
    let mut lines: Vec<(f64, f64, f64)> = Vec::new(); // (center_ppm, area, hwhm_ppm)

    for &(met_name, conc) in concentrations {
        if conc <= 0.0 {
            continue;
        }
        if let Some(entry) = lookup_bmrb_metabolite(met_name) {
            let shift_off = opts
                .metabolite_shift_offsets
                .get(entry.name)
                .or_else(|| opts.metabolite_shift_offsets.get(met_name))
                .copied()
                .unwrap_or(0.0);
            for mult in entry.multiplets {
                for (l_ppm, h_area) in expand_multiplet_lines(mult, opts.obs_mhz, shift_off) {
                    lines.push((
                        l_ppm + opts.tsp_offset_ppm,
                        opts.dilution_factor * conc * h_area,
                        gamma_ppm,
                    ));
                }
            }
        }
    }

    if opts.tsp_concentration > 0.0 {
        lines.push((opts.tsp_offset_ppm, 9.0 * opts.tsp_concentration, gamma_ppm));
    }

    let inv_pi = 1.0 / PI;
    let mut real = vec![0.0; axis.size];
    // Evaluate each Lorentzian inside a generous ±80 HWHM window (captures >99.2% of Lorentzian area
    // while running in <0.5 ms per 10k-point spectrum).
    let step_abs = axis.step_ppm.abs().max(1e-9);
    for &(c_ppm, area, g_ppm) in &lines {
        let win_ppm = (80.0 * g_ppm).max(0.12);
        let idx_lo_f = axis
            .ppm_to_index(c_ppm + win_ppm)
            .min(axis.ppm_to_index(c_ppm - win_ppm));
        let idx_hi_f = axis
            .ppm_to_index(c_ppm + win_ppm)
            .max(axis.ppm_to_index(c_ppm - win_ppm));
        let i0 = (idx_lo_f.floor().max(0.0)) as usize;
        let i1 = (idx_hi_f.ceil().max(0.0) as usize).min(axis.size.saturating_sub(1));
        if i0 > i1 || i0 >= axis.size {
            continue;
        }
        let g2 = g_ppm * g_ppm;
        let scale = area * inv_pi * g_ppm;
        for (idx, slot) in real[i0..=i1].iter_mut().enumerate() {
            let ppm = axis.index_to_ppm((i0 + idx) as f64);
            let d = ppm - c_ppm;
            *slot += scale / (d * d + g2);
        }
    }

    // Presaturated residual HDO (4.80 ppm) and proton-exchanging urinary urea (5.78 ppm) exhibit
    // localized Gaussian chemical-exchange / presaturation envelopes confined to their solvent zones.
    let mut add_gaussian_hump = |center_ppm: f64, area: f64, sigma_ppm: f64| {
        if area <= 0.0 || sigma_ppm <= 0.0 {
            return;
        }
        let win = 5.5 * sigma_ppm;
        let idx_lo_f = axis
            .ppm_to_index(center_ppm + win)
            .min(axis.ppm_to_index(center_ppm - win));
        let idx_hi_f = axis
            .ppm_to_index(center_ppm + win)
            .max(axis.ppm_to_index(center_ppm - win));
        let i0 = (idx_lo_f.floor().max(0.0)) as usize;
        let i1 = (idx_hi_f.ceil().max(0.0) as usize).min(axis.size.saturating_sub(1));
        let norm_amp = area / ((2.0 * PI).sqrt() * sigma_ppm);
        let inv_2s2 = 0.5 / (sigma_ppm * sigma_ppm);
        for (idx, slot) in real[i0..=i1].iter_mut().enumerate() {
            let ppm = axis.index_to_ppm((i0 + idx) as f64);
            let d = ppm - center_ppm;
            *slot += norm_amp * (-d * d * inv_2s2).exp();
        }
    };
    add_gaussian_hump(4.80 + opts.tsp_offset_ppm, opts.residual_water_amp, 0.026);
    add_gaussian_hump(5.78 + opts.tsp_offset_ppm, opts.urea_amp, 0.048);

    if opts.noise_sigma > 0.0 {
        let sigma = opts.noise_sigma / step_abs.sqrt().max(1e-3);
        for v in &mut real {
            *v += rng.normal(0.0, sigma * 0.03);
        }
    }

    Spectrum1D::new(name, Experiment1DKind::Proton1D, axis, real, None)
}

/// Generate a complete 24-spectrum Case vs Control synthetic human biofluid cohort (`12 Control`,
/// `12 Case`) with realistic sample dilutions (`0.45x..1.65x`), pH-induced Citrate/Histidine jitter,
/// residual water/urea humps, and ground-truth biomarker perturbations (`L-Lactate +3.0x`,
/// `L-Valine +2.0x`, `TMAO -2.5x`).
pub fn generate_demo_biofluid_cohort(seed: u64) -> Vec<CohortSample1D> {
    let mut rng = Rng::new(seed);
    let mut samples = Vec::with_capacity(24);
    let base_metabolites: &[(&str, f64)] = &[
        ("L-Lactate", 1.20),
        ("L-Alanine", 0.85),
        ("L-Valine", 0.65),
        ("Citrate", 1.40),
        ("Succinate", 0.45),
        ("Acetate", 0.55),
        ("Creatine", 0.90),
        ("Creatinine", 1.80),
        ("TMAO", 1.50),
        ("Glycine", 1.10),
        ("Formate", 0.50),
        ("Hippurate", 0.95),
        ("alpha-D-Glucose", 0.55),
        ("beta-D-Glucose", 0.95),
    ];

    for i in 0..24 {
        let is_case = i >= 12;
        let group = if is_case { "Case" } else { "Control" };
        let sample_id = if is_case {
            format!("Case_{:02}", i - 11)
        } else {
            format!("Control_{:02}", i + 1)
        };

        let dilution = rng.uniform(0.45, 1.65);
        let mut concs = Vec::with_capacity(base_metabolites.len());
        for &(m_name, base_c) in base_metabolites {
            // Biological inter-individual CV ~ 12%
            let bio_jit = (1.0 + rng.normal(0.0, 0.12)).clamp(0.55, 1.55);
            let fold = match (is_case, m_name) {
                (true, "L-Lactate") => 3.0,
                (true, "L-Valine") => 2.0,
                (true, "TMAO") => 0.40, // -2.5x down-regulated
                _ => 1.0,
            };
            concs.push((m_name, base_c * fold * bio_jit));
        }

        let mut shift_offsets = BTreeMap::new();
        shift_offsets.insert("Citrate".to_string(), rng.uniform(-0.022, 0.022));

        let opts = BiofluidSynthOptions {
            high_ppm: 9.50,
            low_ppm: -0.30,
            n_points: 9_801,
            obs_mhz: 600.0,
            linewidth_hz: 1.5,
            dilution_factor: dilution,
            tsp_concentration: 0.5,
            tsp_offset_ppm: 0.0,
            residual_water_amp: rng.uniform(15.0, 35.0),
            urea_amp: rng.uniform(6.0, 14.0),
            metabolite_shift_offsets: shift_offsets,
            noise_sigma: 0.008,
        };
        let spec = synthesize_biofluid_spectrum_1d(&sample_id, &concs, &opts, &mut rng);
        let mut s = CohortSample1D::new(sample_id, group, spec);
        s.covariates.insert("true_dilution".to_string(), dilution);
        samples.push(s);
    }

    samples
}

// ============================================================================
// 4. Preprocessing, Referencing, Grid Resampling & Region Masking
// ============================================================================

/// Locate the parabolic sub-bin peak maximum and trapezoidal positive integral around `target_ppm`.
pub fn integrate_internal_standard_1d(
    spec: &Spectrum1D,
    target_ppm: f64,
    half_window_ppm: f64,
) -> (f64, f64) {
    if spec.len() < 3 {
        return (target_ppm, 1.0);
    }
    let lo = target_ppm - half_window_ppm.abs();
    let hi = target_ppm + half_window_ppm.abs();
    let mut best_idx = None;
    let mut best_val = f64::NEG_INFINITY;
    for i in 0..spec.len() {
        let p = spec.ppm_at(i);
        if p >= lo && p <= hi && spec.real[i] > best_val {
            best_val = spec.real[i];
            best_idx = Some(i);
        }
    }
    let apex_ppm = if let Some(bi) = best_idx {
        if bi > 0 && bi + 1 < spec.len() {
            let y0 = spec.real[bi - 1];
            let y1 = spec.real[bi];
            let y2 = spec.real[bi + 1];
            let denom = y0 - 2.0 * y1 + y2;
            let delta = if denom.abs() > 1e-15 {
                (0.5 * (y0 - y2) / denom).clamp(-0.5, 0.5)
            } else {
                0.0
            };
            spec.axis.index_to_ppm(bi as f64 + delta)
        } else {
            spec.ppm_at(bi)
        }
    } else {
        target_ppm
    };

    let int_lo = apex_ppm - half_window_ppm.abs();
    let int_hi = apex_ppm + half_window_ppm.abs();
    let step = spec.bin_width_ppm();
    let mut area = 0.0;
    for i in 0..spec.len().saturating_sub(1) {
        let p0 = spec.ppm_at(i);
        let p1 = spec.ppm_at(i + 1);
        let mid = 0.5 * (p0 + p1);
        if mid >= int_lo && mid <= int_hi {
            area += 0.5 * (spec.real[i].max(0.0) + spec.real[i + 1].max(0.0)) * step;
        }
    }
    (apex_ppm, area.max(1e-12))
}

/// Linearly resample a `Spectrum1D` onto a target descending `common_ppm_grid`.
pub fn resample_spectrum_to_grid(spec: &Spectrum1D, common_ppm_grid: &[f64]) -> Vec<f64> {
    let m = common_ppm_grid.len();
    if m == 0 || spec.is_empty() {
        return vec![0.0; m];
    }
    if spec.len() == m {
        let same_first = (spec.ppm_at(0) - common_ppm_grid[0]).abs() <= 1e-9;
        let same_last = (spec.ppm_at(m - 1) - common_ppm_grid[m - 1]).abs() <= 1e-9;
        if same_first && same_last {
            return spec.real.clone();
        }
    }

    let n = spec.len();
    let mut out = vec![0.0; m];
    for (k, &ppm) in common_ppm_grid.iter().enumerate() {
        let idx_f = spec.axis.ppm_to_index(ppm);
        if idx_f <= 0.0 {
            out[k] = if idx_f >= -1.0 { spec.real[0] } else { 0.0 };
        } else if idx_f >= (n - 1) as f64 {
            out[k] = if idx_f <= n as f64 {
                spec.real[n - 1]
            } else {
                0.0
            };
        } else {
            let i0 = idx_f.floor() as usize;
            let frac = idx_f - (i0 as f64);
            out[k] = spec.real[i0] * (1.0 - frac) + spec.real[i0 + 1] * frac;
        }
    }
    out
}

/// Tuple returned by [`preprocess_cohort_1d`]:
/// `(common_ppm_grid, masked_points, resampled_matrix, internal_standard_areas, obs_mhz)`.
pub type PreprocessedCohort1D = (Vec<f64>, Vec<bool>, Vec<Vec<f64>>, Vec<f64>, f64);

/// Build the shared descending chemical-shift grid and boolean exclusion mask, and preprocess all cohort traces.
pub fn preprocess_cohort_1d(
    samples: &[CohortSample1D],
    config: &MetabolomicsPipelineConfig,
) -> PreprocessedCohort1D {
    let lo_ppm = config.min_ppm.min(config.max_ppm);
    let hi_ppm = config.min_ppm.max(config.max_ppm);
    let step = config.grid_step_ppm.abs().max(1e-5);
    let n_pts = (((hi_ppm - lo_ppm) / step).round() as usize + 1).max(16);
    let actual_step = (hi_ppm - lo_ppm) / ((n_pts - 1) as f64);
    let common_grid: Vec<f64> = (0..n_pts)
        .map(|k| hi_ppm - (k as f64) * actual_step)
        .collect();

    let zones = config.mask_preset.zones();
    let masked: Vec<bool> = common_grid
        .iter()
        .map(|&p| zones.iter().any(|z| z.contains(p)))
        .collect();

    let obs_mhz = samples
        .first()
        .map(|s| s.spectrum.axis.obs_mhz.max(100.0))
        .unwrap_or(600.0);

    let (is_target, is_half_win) = match config.normalization {
        NormalizationMethod::InternalStandard {
            target_ppm,
            half_window_ppm,
        } => (target_ppm, half_window_ppm.max(0.03)),
        _ => (0.0, 0.10),
    };

    let mut matrix = Vec::with_capacity(samples.len());
    let mut is_areas = Vec::with_capacity(samples.len());

    for s in samples {
        let mut work = s.spectrum.clone();
        if let Some(ref bl_cfg) = config.baseline_config {
            let _ = correct_baseline_1d(&mut work, bl_cfg);
        }
        let (apex_ppm, is_area) = integrate_internal_standard_1d(&work, is_target, is_half_win);
        if config.calibrate_internal_standard {
            work.apply_reference_offset(is_target - apex_ppm);
        }
        is_areas.push(is_area);

        let mut row = resample_spectrum_to_grid(&work, &common_grid);
        for (m_idx, &is_m) in masked.iter().enumerate() {
            if is_m {
                row[m_idx] = 0.0;
            }
        }
        matrix.push(row);
    }

    (common_grid, masked, matrix, is_areas, obs_mhz)
}

// ============================================================================
// 5. Interval FFT Cross-Correlation Peak Alignment (Savorani et al. 2010)
// ============================================================================

fn median_in_place(vals: &mut [f64]) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    vals.sort_by(|a, b| a.total_cmp(b));
    let mid = vals.len() / 2;
    if vals.len() % 2 == 1 {
        vals[mid]
    } else {
        0.5 * (vals[mid - 1] + vals[mid])
    }
}

fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let pos = (q.clamp(0.0, 1.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[pos.min(sorted.len() - 1)]
}

fn estimate_matrix_noise_floor(spectra: &[Vec<f64>], masked: &[bool]) -> f64 {
    if spectra.is_empty() || spectra[0].len() < 32 {
        return 1e-6;
    }
    let m = spectra[0].len();
    let win = 32usize;
    let mut win_stds = Vec::new();
    for start in (0..m.saturating_sub(win)).step_by(win) {
        if masked[start..start + win].iter().any(|&b| b) {
            continue;
        }
        for row in spectra.iter().take(8) {
            let seg = &row[start..start + win];
            let mean = seg.iter().sum::<f64>() / (win as f64);
            let var = seg
                .iter()
                .map(|&v| {
                    let d = v - mean;
                    d * d
                })
                .sum::<f64>()
                / ((win - 1) as f64);
            win_stds.push(var.sqrt());
        }
    }
    if win_stds.is_empty() {
        return 1e-6;
    }
    win_stds.sort_by(|a, b| a.total_cmp(b));
    // Median of quietest 25% of windows
    let q_len = (win_stds.len() / 4).max(1);
    median_in_place(&mut win_stds[..q_len]).max(1e-12)
}

fn build_alignment_reference(spectra: &[Vec<f64>], reference: AlignmentReference) -> Vec<f64> {
    let n = spectra.len();
    if n == 0 {
        return Vec::new();
    }
    let m = spectra[0].len();
    match reference {
        AlignmentReference::SampleIndex(idx) => spectra[idx.min(n - 1)].clone(),
        AlignmentReference::Mean => {
            let mut out = vec![0.0; m];
            for row in spectra {
                for (j, &v) in row.iter().enumerate() {
                    out[j] += v;
                }
            }
            let inv = 1.0 / (n as f64);
            for v in &mut out {
                *v *= inv;
            }
            out
        }
        AlignmentReference::Median => {
            let mut out = vec![0.0; m];
            let mut col = vec![0.0; n];
            for (j, slot) in out.iter_mut().enumerate() {
                for i in 0..n {
                    col[i] = spectra[i][j];
                }
                *slot = median_in_place(&mut col);
            }
            out
        }
    }
}

fn unmasked_contiguous_spans(masked: &[bool]) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut i = 0usize;
    while i < masked.len() {
        if masked[i] {
            i += 1;
            continue;
        }
        let start = i;
        while i + 1 < masked.len() && !masked[i + 1] {
            i += 1;
        }
        if i > start + 2 {
            spans.push((start, i));
        }
        i += 1;
    }
    spans
}

fn align_single_interval_fft(
    ref_seg: &[f64],
    samp_seg: &[f64],
    max_lag_pts: usize,
    noise_floor: f64,
) -> (Vec<f64>, f64) {
    let len = ref_seg.len();
    if len < 8 || max_lag_pts == 0 {
        return (samp_seg.to_vec(), 0.0);
    }
    let k_max = max_lag_pts.min(len / 2);
    if k_max == 0 {
        return (samp_seg.to_vec(), 0.0);
    }

    let mut r_sorted = ref_seg.to_vec();
    r_sorted.sort_by(|a, b| a.total_cmp(b));
    let r_base = quantile_sorted(&r_sorted, 0.10);

    let mut s_sorted = samp_seg.to_vec();
    s_sorted.sort_by(|a, b| a.total_cmp(b));
    let s_base = quantile_sorted(&s_sorted, 0.10);

    let mut r_clean = Vec::with_capacity(len);
    let mut s_clean = Vec::with_capacity(len);
    let mut r_norm2 = 0.0;
    let mut s_norm2 = 0.0;
    for i in 0..len {
        let rv = (ref_seg[i] - r_base).max(0.0);
        let sv = (samp_seg[i] - s_base).max(0.0);
        r_clean.push(rv);
        s_clean.push(sv);
        r_norm2 += rv * rv;
        s_norm2 += sv * sv;
    }

    let min_energy = (3.5 * noise_floor) * (3.5 * noise_floor) * (len as f64).min(25.0);
    if r_norm2 <= min_energy || s_norm2 <= min_energy {
        return (samp_seg.to_vec(), 0.0);
    }

    let fft_len = (2 * len + 2 * k_max + 2).next_power_of_two();
    let mut f_ref = vec![[0.0, 0.0]; fft_len];
    let mut f_samp = vec![[0.0, 0.0]; fft_len];
    for i in 0..len {
        f_ref[i] = [r_clean[i], 0.0];
        f_samp[i] = [s_clean[i], 0.0];
    }
    fft_arbitrary_in_place(&mut f_ref, false);
    fft_arbitrary_in_place(&mut f_samp, false);

    // Cross-spectrum C[m] = conj(R[m]) * S[m] so peak at +tau aligns samp[n + tau] -> ref[n].
    let mut cross = vec![[0.0, 0.0]; fft_len];
    for m in 0..fft_len {
        let (rr, ri) = (f_ref[m][0], f_ref[m][1]);
        let (sr, si) = (f_samp[m][0], f_samp[m][1]);
        cross[m] = [rr * sr + ri * si, rr * si - ri * sr];
    }
    fft_arbitrary_in_place(&mut cross, true);

    let corr_at = |lag: isize| -> f64 {
        let idx = lag.rem_euclid(fft_len as isize) as usize;
        cross[idx][0]
    };

    let c_zero = corr_at(0);
    let mut best_lag: isize = 0;
    let mut best_val = c_zero;
    for lag in -(k_max as isize)..=(k_max as isize) {
        let c = corr_at(lag);
        if c > best_val {
            best_val = c;
            best_lag = lag;
        }
    }

    if best_lag == 0 || best_val <= c_zero * 1.0001 {
        return (samp_seg.to_vec(), 0.0);
    }

    let sub_delta = if best_lag.unsigned_abs() < k_max {
        let cm = corr_at(best_lag - 1);
        let c0 = best_val;
        let cp = corr_at(best_lag + 1);
        let denom = cm - 2.0 * c0 + cp;
        if denom < -1e-15 {
            (0.5 * (cm - cp) / denom).clamp(-0.5, 0.5)
        } else {
            0.0
        }
    } else {
        0.0
    };
    let shift_pts = (best_lag as f64) + sub_delta;

    // Quiet baseline edge fill values (5-point edge medians)
    let edge_n = len.min(5);
    let mut left_edge = samp_seg[..edge_n].to_vec();
    let mut right_edge = samp_seg[len - edge_n..].to_vec();
    let fill_left = median_in_place(&mut left_edge);
    let fill_right = median_in_place(&mut right_edge);

    let mut aligned = vec![0.0; len];
    for (n, out_v) in aligned.iter_mut().enumerate() {
        let src = (n as f64) + shift_pts;
        if src < 0.0 {
            *out_v = fill_left;
        } else if src >= (len - 1) as f64 {
            *out_v = fill_right;
        } else {
            let i0 = src.floor() as usize;
            let frac = src - (i0 as f64);
            *out_v = samp_seg[i0] * (1.0 - frac) + samp_seg[i0 + 1] * frac;
        }
    }

    (aligned, shift_pts)
}

/// Apply Savorani et al. (*J. Magn. Reson.* 2010, 202, 190–202) interval FFT cross-correlation
/// local peak alignment across a preprocessed cohort spectral matrix.
pub fn align_cohort_interval_fft(
    spectra: &mut [Vec<f64>],
    common_ppm_grid: &[f64],
    masked: &[bool],
    mode: &AlignmentMode,
) {
    let n = spectra.len();
    let m = common_ppm_grid.len();
    if n < 2 || m < 16 {
        return;
    }
    let step_ppm = (common_ppm_grid[0] - common_ppm_grid[m - 1]).abs() / ((m - 1) as f64);
    if step_ppm <= 0.0 {
        return;
    }

    let (intervals, max_shift_ppm, ref_kind) = match mode {
        AlignmentMode::Disabled => return,
        AlignmentMode::AutomaticIntervals {
            interval_width_ppm,
            max_shift_ppm,
            reference,
        } => {
            // Build cohort maximum envelope so interval boundaries never slice through any sample's shifted multiplet!
            let mut max_env = vec![0.0_f64; m];
            for row in spectra.iter() {
                for (j, &v) in row.iter().enumerate() {
                    if v > max_env[j] {
                        max_env[j] = v;
                    }
                }
            }
            let target_pts = ((interval_width_ppm.max(0.06) / step_ppm).round() as usize).max(24);
            let search_pts = ((0.045 / step_ppm).round() as usize).min(target_pts / 3);
            let mut ivs = Vec::new();
            for (sp_start, sp_end) in unmasked_contiguous_spans(masked) {
                let mut cur = sp_start;
                while cur < sp_end {
                    let nominal = cur + target_pts;
                    if nominal + target_pts / 2 >= sp_end {
                        ivs.push((cur, sp_end));
                        break;
                    }
                    let lo = nominal.saturating_sub(search_pts).max(cur + 8);
                    let hi = (nominal + search_pts).min(sp_end.saturating_sub(8));
                    let mut cut = nominal;
                    if lo <= hi {
                        let mut min_v = f64::INFINITY;
                        for (k, &ev) in max_env[lo..=hi].iter().enumerate() {
                            if ev < min_v {
                                min_v = ev;
                                cut = lo + k;
                            }
                        }
                    }
                    ivs.push((cur, cut));
                    cur = cut + 1;
                }
            }
            (ivs, *max_shift_ppm, *reference)
        }
        AlignmentMode::CustomIntervals {
            intervals_ppm,
            max_shift_ppm,
            reference,
        } => {
            let mut ivs = Vec::new();
            for &(p_a, p_b) in intervals_ppm {
                let hi = p_a.max(p_b);
                let lo = p_a.min(p_b);
                let i_start = common_ppm_grid.partition_point(|&p| p > hi);
                let i_end = common_ppm_grid
                    .partition_point(|&p| p >= lo)
                    .saturating_sub(1);
                if i_start + 8 <= i_end && i_end < m && !masked[i_start..=i_end].iter().any(|&b| b)
                {
                    ivs.push((i_start, i_end));
                }
            }
            (ivs, *max_shift_ppm, *reference)
        }
    };

    let max_lag_pts = ((max_shift_ppm.abs() / step_ppm).round() as usize).max(1);
    let noise_floor = estimate_matrix_noise_floor(spectra, masked);
    let ref_trace = build_alignment_reference(spectra, ref_kind);

    for row in spectra.iter_mut() {
        for &(a, b) in &intervals {
            let (aligned_seg, _) =
                align_single_interval_fft(&ref_trace[a..=b], &row[a..=b], max_lag_pts, noise_floor);
            row[a..=b].copy_from_slice(&aligned_seg);
        }
    }
}

// ============================================================================
// 6. Uniform & Adaptive Intelligent Binning (De Meyer et al. 2008)
// ============================================================================

fn smooth_gaussian_segment(seg: &[f64], sigma_pts: f64) -> Vec<f64> {
    let n = seg.len();
    if n == 0 || sigma_pts <= 0.25 {
        return seg.to_vec();
    }
    let radius = ((3.5 * sigma_pts).ceil() as usize).clamp(1, 256);
    let mut kernel = Vec::with_capacity(2 * radius + 1);
    let inv_2s2 = 1.0 / (2.0 * sigma_pts * sigma_pts);
    let mut ksum = 0.0;
    for d in -(radius as isize)..=(radius as isize) {
        let w = (-(d as f64) * (d as f64) * inv_2s2).exp();
        kernel.push(w);
        ksum += w;
    }
    for w in &mut kernel {
        *w /= ksum;
    }

    let mut out = vec![0.0; n];
    for (i, slot) in out.iter_mut().enumerate() {
        let mut acc = 0.0;
        for (ki, &w) in kernel.iter().enumerate() {
            let j_isize = (i as isize) + (ki as isize) - (radius as isize);
            let j = j_isize.clamp(0, (n - 1) as isize) as usize;
            acc += seg[j] * w;
        }
        *slot = acc;
    }
    out
}

#[inline]
fn de_meyer_vb(env: &[f64], a: usize, b: usize, step_ppm: f64, resolution: f64) -> f64 {
    if b <= a {
        return 0.0;
    }
    let mut max_v = f64::NEG_INFINITY;
    for &v in &env[a..=b] {
        if v > max_v {
            max_v = v;
        }
    }
    let edge_hi = env[a].max(env[b]);
    let prom = (max_v - edge_hi).max(0.0);
    let width = ((b - a) as f64) * step_ppm;
    width.powf(resolution) * prom
}

#[derive(Clone, Copy)]
struct DeMeyerParams {
    min_pts: usize,
    max_pts: usize,
    step_ppm: f64,
    resolution: f64,
}

fn subdivide_adaptive_de_meyer(
    env: &[f64],
    a: usize,
    b: usize,
    params: DeMeyerParams,
    out_bins: &mut Vec<(usize, usize)>,
) {
    if b <= a + 1 {
        return;
    }
    if b - a < 2 * params.min_pts {
        out_bins.push((a, b));
        return;
    }

    let parent_vb = de_meyer_vb(env, a, b, params.step_ppm, params.resolution);
    let mut best_split = None;
    let mut best_sum_vb = f64::NEG_INFINITY;

    for v in (a + params.min_pts)..=(b - params.min_pts) {
        if env[v] <= env[v - 1]
            && env[v] <= env[v + 1]
            && (env[v] < env[v - 1] || env[v] < env[v + 1])
        {
            let score = de_meyer_vb(env, a, v, params.step_ppm, params.resolution)
                + de_meyer_vb(env, v, b, params.step_ppm, params.resolution);
            if score > best_sum_vb {
                best_sum_vb = score;
                best_split = Some(v);
            }
        }
    }

    let must_split = (b - a) > params.max_pts;
    if let Some(v) = best_split {
        if best_sum_vb > parent_vb * 1.000_001 || must_split {
            subdivide_adaptive_de_meyer(env, a, v, params, out_bins);
            subdivide_adaptive_de_meyer(env, v, b, params, out_bins);
            return;
        }
    }

    if must_split {
        // Monotone or broad plateau wider than max_width_ppm: split at lowest interior point or midpoint.
        let mut min_idx = (a + b) / 2;
        let mut min_val = f64::INFINITY;
        for (off, &val) in env[(a + params.min_pts)..=(b - params.min_pts)]
            .iter()
            .enumerate()
        {
            if val < min_val {
                min_val = val;
                min_idx = a + params.min_pts + off;
            }
        }
        subdivide_adaptive_de_meyer(env, a, min_idx, params, out_bins);
        subdivide_adaptive_de_meyer(env, min_idx, b, params, out_bins);
        return;
    }

    out_bins.push((a, b));
}

/// Partition aligned cohort spectra into buckets (`Uniform` or `AdaptiveIntelligent`) and integrate
/// the `N x P` raw bucket area matrix via composite trapezoidal integration.
pub fn bucket_cohort_spectra(
    spectra: &[Vec<f64>],
    common_ppm_grid: &[f64],
    masked: &[bool],
    obs_mhz: f64,
    method: &BucketingMethod,
) -> (Vec<SpectralBucket>, Vec<Vec<f64>>) {
    let n = spectra.len();
    let m = common_ppm_grid.len();
    if n == 0 || m < 4 {
        return (Vec::new(), vec![Vec::new(); n]);
    }
    let step_ppm = (common_ppm_grid[0] - common_ppm_grid[m - 1]).abs() / ((m - 1) as f64);
    let noise_floor = estimate_matrix_noise_floor(spectra, masked);

    // Unbiased cohort mean profile R_mean[k] = max((1/N) * sum_i S_i[k], 0) supplemented by
    // extreme-value-safe single-sample spikes (S_max[k] - 5.5 * sigma_noise)+ so rare outlier peaks
    // are captured without rectifying multi-sample Gaussian baseline noise into positive bias.
    let mut r_mean = vec![0.0_f64; m];
    let mut r_max = vec![f64::NEG_INFINITY; m];
    for row in spectra {
        for k in 0..m {
            let v = row[k];
            r_mean[k] += v;
            if v > r_max[k] {
                r_max[k] = v;
            }
        }
    }
    let inv_n = 1.0 / (n as f64);
    let mut r_raw = vec![0.0_f64; m];
    for k in 0..m {
        let mu = (r_mean[k] * inv_n).max(0.0);
        r_mean[k] = mu;
        let spike_excess = (r_max[k] - 5.5 * noise_floor).max(0.0);
        r_raw[k] = mu.max(0.5 * spike_excess);
    }

    let mut raw_intervals: Vec<(usize, usize)> = Vec::new();
    let min_snr_gate = match *method {
        BucketingMethod::Uniform {
            bin_width_ppm,
            min_bucket_snr,
        } => {
            let bin_pts = ((bin_width_ppm.max(0.002) / step_ppm).round() as usize).max(2);
            for (sp_a, sp_b) in unmasked_contiguous_spans(masked) {
                let mut cur = sp_a;
                while cur < sp_b {
                    let next = (cur + bin_pts).min(sp_b);
                    if next > cur {
                        raw_intervals.push((cur, next));
                    }
                    if next == sp_b {
                        break;
                    }
                    cur = next;
                }
            }
            min_bucket_snr
        }
        BucketingMethod::AdaptiveIntelligent {
            resolution,
            min_width_ppm,
            max_width_ppm,
            multiplet_coalesce_hz,
            min_bucket_snr,
        } => {
            let min_pts = ((min_width_ppm.max(0.004) / step_ppm).round() as usize).max(3);
            let max_pts = ((max_width_ppm.max(min_width_ppm * 1.5) / step_ppm).round() as usize)
                .max(2 * min_pts);
            let sigma_ppm = multiplet_coalesce_hz.max(0.0) / obs_mhz.max(100.0);
            let sigma_pts = sigma_ppm / step_ppm;
            let res = resolution.clamp(0.1, 0.95);
            let dm_params = DeMeyerParams {
                min_pts,
                max_pts,
                step_ppm,
                resolution: res,
            };

            for (sp_a, sp_b) in unmasked_contiguous_spans(masked) {
                let env_seg = smooth_gaussian_segment(&r_raw[sp_a..=sp_b], sigma_pts);
                let seg_len = env_seg.len();
                if min_bucket_snr > 0.0 {
                    // Extract active signal islands above baseline foot threshold, dilated by ~0.006 ppm
                    // so full Lorentzian multiplet wings are enclosed inside each bucket!
                    let foot_thr = 2.0 * noise_floor;
                    let pad_pts = ((0.006 / step_ppm).round() as usize).max(2);
                    let mut active = vec![false; seg_len];
                    for (idx, &v) in env_seg.iter().enumerate() {
                        if v > foot_thr || r_raw[sp_a + idx] > 3.0 * noise_floor {
                            let lo = idx.saturating_sub(pad_pts);
                            let hi = (idx + pad_pts).min(seg_len - 1);
                            for slot in &mut active[lo..=hi] {
                                *slot = true;
                            }
                        }
                    }

                    let mut idx = 0usize;
                    while idx < seg_len {
                        if !active[idx] {
                            idx += 1;
                            continue;
                        }
                        let isl_start = idx;
                        while idx + 1 < seg_len && active[idx + 1] {
                            idx += 1;
                        }
                        let isl_end = idx;
                        if isl_end > isl_start + 1 {
                            // Ensure minimum bin width around narrow singlets if room exists
                            let cur_w = isl_end - isl_start;
                            let (ia, ib) = if cur_w < min_pts {
                                let extra = (min_pts - cur_w).div_ceil(2);
                                (
                                    isl_start.saturating_sub(extra),
                                    (isl_end + extra).min(seg_len - 1),
                                )
                            } else {
                                (isl_start, isl_end)
                            };
                            let mut local_bins = Vec::new();
                            subdivide_adaptive_de_meyer(
                                &env_seg,
                                ia,
                                ib,
                                dm_params,
                                &mut local_bins,
                            );
                            for (la, lb) in local_bins {
                                raw_intervals.push((sp_a + la, sp_a + lb));
                            }
                        }
                        idx += 1;
                    }
                } else {
                    let mut local_bins = Vec::new();
                    subdivide_adaptive_de_meyer(
                        &env_seg,
                        0,
                        seg_len - 1,
                        dm_params,
                        &mut local_bins,
                    );
                    for (la, lb) in local_bins {
                        raw_intervals.push((sp_a + la, sp_a + lb));
                    }
                }
            }
            min_bucket_snr
        }
    };

    let mut buckets = Vec::new();
    let mut kept_bounds = Vec::new();
    for (a, b) in raw_intervals {
        if b <= a || b >= m {
            continue;
        }
        let max_peak = r_raw[a..=b].iter().copied().fold(0.0_f64, f64::max);
        let snr = max_peak / noise_floor.max(1e-12);
        if min_snr_gate > 0.0 && snr < min_snr_gate {
            continue;
        }
        let high_ppm = common_ppm_grid[a].max(common_ppm_grid[b]);
        let low_ppm = common_ppm_grid[a].min(common_ppm_grid[b]);
        let mut w_sum = 0.0;
        let mut wp_sum = 0.0;
        for k in a..=b {
            let w = r_raw[k].max(0.0);
            w_sum += w;
            wp_sum += w * common_ppm_grid[k];
        }
        let center_ppm = if w_sum > 1e-15 {
            wp_sum / w_sum
        } else {
            0.5 * (high_ppm + low_ppm)
        };
        let b_idx = buckets.len();
        buckets.push(SpectralBucket {
            index: b_idx,
            low_ppm,
            high_ppm,
            center_ppm,
            width_ppm: (high_ppm - low_ppm).max(step_ppm),
            start_grid_idx: a,
            end_grid_idx: b,
            max_snr: snr,
        });
        kept_bounds.push((a, b));
    }

    let p = buckets.len();
    let mut x_raw = vec![vec![0.0; p]; n];
    for (i, row) in spectra.iter().enumerate() {
        for (j, &(a, b)) in kept_bounds.iter().enumerate() {
            let mut area = 0.0;
            for k in a..b {
                area += 0.5 * (row[k] + row[k + 1]) * step_ppm;
            }
            x_raw[i][j] = area;
        }
    }

    (buckets, x_raw)
}

// ============================================================================
// 7. Dilution Normalization (PQN, TSA, Internal Standard) & Scaling
// ============================================================================

/// Normalize a `N x P` raw bucket matrix according to `method`, returning `(X_norm, dilution_factors)`
/// where `X_norm[i][j] = X_raw[i][j] / dilution_factors[i]`.
pub fn normalize_bucket_matrix(
    raw_matrix: &[Vec<f64>],
    internal_standard_areas: &[f64],
    method: NormalizationMethod,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let n = raw_matrix.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    let p = raw_matrix[0].len();
    if p == 0 {
        return (raw_matrix.to_vec(), vec![1.0; n]);
    }

    let factors: Vec<f64> = match method {
        NormalizationMethod::None => vec![1.0; n],
        NormalizationMethod::TotalSpectralArea => {
            let areas: Vec<f64> = raw_matrix
                .iter()
                .map(|r| r.iter().map(|&v| v.max(0.0)).sum::<f64>().max(1e-15))
                .collect();
            let mut sorted_a = areas.clone();
            let ref_area = median_in_place(&mut sorted_a).max(1e-15);
            areas.into_iter().map(|a| a / ref_area).collect()
        }
        NormalizationMethod::InternalStandard { .. } => {
            let mut valid_is: Vec<f64> = internal_standard_areas
                .iter()
                .copied()
                .map(|a| a.max(1e-15))
                .collect();
            let ref_is = median_in_place(&mut valid_is).max(1e-15);
            (0..n)
                .map(|i| {
                    internal_standard_areas
                        .get(i)
                        .copied()
                        .unwrap_or(ref_is)
                        .max(1e-15)
                        / ref_is
                })
                .collect()
        }
        NormalizationMethod::ProbabilisticQuotient => {
            // Two-pass Dieterle et al. (2006) PQN:
            // Pass 1: scale each sample roughly by its upper-quartile positive intensity (robust to
            // empty baseline bins while retaining >=25% breakdown resistance to massive spikes) so
            // 2x/5x sample dilutions cannot skew the cohort median reference spectrum `x_ref`.
            let mut sample_scales = Vec::with_capacity(n);
            for row in raw_matrix {
                let mut pos: Vec<f64> = row.iter().copied().filter(|&v| v > 1e-12).collect();
                let s_val = if pos.is_empty() {
                    1.0
                } else {
                    pos.sort_by(|a, b| a.total_cmp(b));
                    let q75_idx = ((3 * (pos.len() - 1)) / 4).min(pos.len() - 1);
                    pos[q75_idx].max(1e-15)
                };
                sample_scales.push(s_val);
            }
            let mut med_of_scales = sample_scales.clone();
            let anchor_scale = median_in_place(&mut med_of_scales).max(1e-15);

            let mut x_ref = vec![0.0; p];
            let mut col = vec![0.0; n];
            for (j, ref_slot) in x_ref.iter_mut().enumerate() {
                for i in 0..n {
                    col[i] = raw_matrix[i][j] * (anchor_scale / sample_scales[i]);
                }
                *ref_slot = median_in_place(&mut col);
            }

            // Identify active signal-bearing reference buckets (exclude low-intensity wing/noise bins per Dieterle 2006)
            let mut pos_ref: Vec<f64> = x_ref.iter().copied().filter(|&v| v > 1e-12).collect();
            let sig_thr = if pos_ref.is_empty() {
                0.0
            } else {
                pos_ref.sort_by(|a, b| a.total_cmp(b));
                let q75_ref = pos_ref[(3 * (pos_ref.len() - 1)) / 4];
                0.05 * q75_ref
            };
            let sig_cols: Vec<usize> = (0..p).filter(|&j| x_ref[j] > sig_thr).collect();

            if sig_cols.is_empty() {
                vec![1.0; n]
            } else {
                let mut pass1_factors = Vec::with_capacity(n);
                for row in raw_matrix {
                    let mut quots: Vec<f64> = sig_cols
                        .iter()
                        .filter_map(|&j| {
                            let q = row[j] / x_ref[j];
                            (q > 0.0 && q.is_finite()).then_some(q)
                        })
                        .collect();
                    let d = if quots.is_empty() {
                        1.0
                    } else {
                        median_in_place(&mut quots).max(1e-15)
                    };
                    pass1_factors.push(d);
                }

                // Pass 2: refine x_ref on Pass-1 dilution-equalized spectra and re-evaluate exact median quotients
                for (j, ref_slot) in x_ref.iter_mut().enumerate() {
                    for i in 0..n {
                        col[i] = raw_matrix[i][j] / pass1_factors[i];
                    }
                    *ref_slot = median_in_place(&mut col);
                }
                let mut final_factors = Vec::with_capacity(n);
                for row in raw_matrix {
                    let mut quots: Vec<f64> = sig_cols
                        .iter()
                        .filter_map(|&j| {
                            if x_ref[j] <= 1e-15 {
                                return None;
                            }
                            let q = row[j] / x_ref[j];
                            (q > 0.0 && q.is_finite()).then_some(q)
                        })
                        .collect();
                    let d = if quots.is_empty() {
                        1.0
                    } else {
                        median_in_place(&mut quots).max(1e-15)
                    };
                    final_factors.push(d);
                }
                final_factors
            }
        }
    };

    let mut norm = vec![vec![0.0; p]; n];
    for i in 0..n {
        let inv_f = 1.0 / factors[i].max(1e-15);
        for j in 0..p {
            norm[i][j] = raw_matrix[i][j] * inv_f;
        }
    }
    (norm, factors)
}

/// Fitted column centering & scaling parameters (strictly estimated on training rows to prevent CV leakage).
#[derive(Clone, Debug)]
pub struct FittedScaler {
    pub method: ScalingMethod,
    pub means: Vec<f64>,
    pub scales: Vec<f64>,
}

impl FittedScaler {
    /// Fit column means and scaling denominators on `matrix[train_rows]`.
    pub fn fit_rows(matrix: &[Vec<f64>], train_rows: &[usize], method: ScalingMethod) -> Self {
        let p = matrix.first().map(|r| r.len()).unwrap_or(0);
        let n = train_rows.len();
        if n == 0 || p == 0 {
            return Self {
                method,
                means: vec![0.0; p],
                scales: vec![1.0; p],
            };
        }
        let mut means = vec![0.0; p];
        if method != ScalingMethod::None {
            let inv_n = 1.0 / (n as f64);
            for &r in train_rows {
                for (j, &v) in matrix[r].iter().enumerate() {
                    means[j] += v;
                }
            }
            for m in &mut means {
                *m *= inv_n;
            }
        }

        let mut scales = vec![1.0; p];
        if matches!(method, ScalingMethod::Pareto | ScalingMethod::UnitVariance) && n > 1 {
            let inv_dof = 1.0 / ((n - 1) as f64);
            for (j, sc) in scales.iter_mut().enumerate() {
                let mu = means[j];
                let mut var = 0.0;
                for &r in train_rows {
                    let d = matrix[r][j] - mu;
                    var += d * d;
                }
                let sd = (var * inv_dof).sqrt();
                *sc = match method {
                    ScalingMethod::Pareto => sd.sqrt(),
                    ScalingMethod::UnitVariance => sd,
                    _ => 1.0,
                };
            }
        }

        Self {
            method,
            means,
            scales,
        }
    }

    /// Transform a single row vector using the fitted parameters.
    pub fn transform_row(&self, row: &[f64]) -> Vec<f64> {
        row.iter()
            .enumerate()
            .map(|(j, &v)| match self.method {
                ScalingMethod::None => v,
                ScalingMethod::MeanCenter => v - self.means[j],
                ScalingMethod::Pareto | ScalingMethod::UnitVariance => {
                    let sc = self.scales[j];
                    if sc > 1e-14 {
                        (v - self.means[j]) / sc
                    } else {
                        0.0
                    }
                }
            })
            .collect()
    }

    /// Transform all rows of `matrix`.
    pub fn transform_matrix(&self, matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
        matrix.iter().map(|r| self.transform_row(r)).collect()
    }
}

/// Fit and apply `method` across all rows of `matrix`.
pub fn scale_bucket_matrix(matrix: &[Vec<f64>], method: ScalingMethod) -> Vec<Vec<f64>> {
    let all_rows: Vec<usize> = (0..matrix.len()).collect();
    FittedScaler::fit_rows(matrix, &all_rows, method).transform_matrix(matrix)
}

// ============================================================================
// 8. Exact Statistical Distributions (Incomplete Beta, Student-t, F, BH-FDR)
// ============================================================================

fn ln_gamma_lanczos(z: f64) -> f64 {
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
    if z < 0.5 {
        return PI.ln() - (PI * z).sin().ln() - ln_gamma_lanczos(1.0 - z);
    }
    let zm1 = z - 1.0;
    let mut x = COEFFS[0];
    for (i, &c) in COEFFS.iter().enumerate().skip(1) {
        x += c / (zm1 + (i as f64));
    }
    let t = zm1 + 7.5;
    0.5 * (2.0 * PI).ln() + (zm1 + 0.5) * t.ln() - t + x.ln()
}

fn betacf_lentz(a: f64, b: f64, x: f64) -> f64 {
    let fpmin = 1e-300;
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
        let aa = mf * (b - mf) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < fpmin {
            d = fpmin;
        }
        c = 1.0 + aa / c;
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

/// Regularized incomplete beta function $I_x(a, b) \in [0, 1]$ for $x \in [0, 1]$, $a, b > 0$.
pub fn regularized_incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let ln_bt = ln_gamma_lanczos(a + b) - ln_gamma_lanczos(a) - ln_gamma_lanczos(b)
        + a * x.ln()
        + b * (1.0 - x).ln();
    let bt = ln_bt.exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        (bt * betacf_lentz(a, b, x) / a).clamp(0.0, 1.0)
    } else {
        (1.0 - bt * betacf_lentz(b, a, 1.0 - x) / b).clamp(0.0, 1.0)
    }
}

/// Exact two-sided Student-$t$ $p$-value $P(|T_\nu| \ge |t|)$ for degrees of freedom $\nu > 0$.
pub fn student_t_two_sided_p_value(t_stat: f64, dof: f64) -> f64 {
    if !t_stat.is_finite() || dof <= 0.0 {
        return 1.0;
    }
    let t2 = t_stat * t_stat;
    if t2 == 0.0 {
        return 1.0;
    }
    let x = dof / (dof + t2);
    regularized_incomplete_beta(x, 0.5 * dof, 0.5)
}

/// Exact $(1 - \alpha)$ quantile $F_{1-\alpha}(d_1, d_2)$ of the $F(d_1, d_2)$ distribution.
pub fn f_distribution_quantile(prob: f64, d1: f64, d2: f64) -> f64 {
    if d1 <= 0.0 || d2 <= 0.0 {
        return 1.0;
    }
    let p = prob.clamp(1e-12, 1.0 - 1e-12);
    if (d1 - 2.0).abs() < 1e-12 {
        let alpha = 1.0 - p;
        return 0.5 * d2 * (alpha.powf(-2.0 / d2) - 1.0);
    }
    let cdf = |f: f64| -> f64 {
        if f <= 0.0 {
            0.0
        } else {
            let x = (d1 * f) / (d1 * f + d2);
            regularized_incomplete_beta(x, 0.5 * d1, 0.5 * d2)
        }
    };
    let mut lo = 0.0;
    let mut hi = 1.0;
    while cdf(hi) < p && hi < 1e12 {
        hi *= 2.0;
    }
    for _ in 0..64 {
        let mid = 0.5 * (lo + hi);
        if cdf(mid) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Benjamini–Hochberg step-up False Discovery Rate (`BH-FDR`, 1995) adjustment.
pub fn benjamini_hochberg_fdr(p_values: &[f64]) -> Vec<f64> {
    let m = p_values.len();
    if m == 0 {
        return Vec::new();
    }
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| p_values[a].total_cmp(&p_values[b]));

    let mut q_sorted = vec![1.0; m];
    let mf = m as f64;
    let mut running_min = 1.0_f64;
    for rank_m1 in (0..m).rev() {
        let idx = order[rank_m1];
        let rank = (rank_m1 + 1) as f64;
        let raw_q = (p_values[idx].clamp(0.0, 1.0) * mf / rank).min(1.0);
        running_min = running_min.min(raw_q);
        q_sorted[rank_m1] = running_min;
    }

    let mut q_out = vec![1.0; m];
    for (rank_m1, &orig_idx) in order.iter().enumerate() {
        q_out[orig_idx] = q_sorted[rank_m1];
    }
    q_out
}

// ============================================================================
// 9. Deterministic PCA, Hotelling T^2 Ellipses & Q-Residuals (SPE)
// ============================================================================

/// Deterministic cyclic Jacobi symmetric eigensolver returning `(eigenvalues_desc, eigenvectors_cols)`.
/// Input `sym` is a `d x d` row-major symmetric matrix (`sym[i * d + j]`).
fn jacobi_symmetric_eigen_desc(sym: &[f64], d: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    if d == 0 {
        return (Vec::new(), Vec::new());
    }
    let mut a = sym.to_vec();
    let mut v = vec![0.0; d * d];
    for i in 0..d {
        v[i * d + i] = 1.0;
    }

    for _sweep in 0..80 {
        let mut max_off = 0.0_f64;
        for p in 0..d {
            for q in (p + 1)..d {
                max_off = max_off.max(a[p * d + q].abs());
            }
        }
        if max_off <= 1e-14 {
            break;
        }
        for p in 0..d {
            for q in (p + 1)..d {
                let apq = a[p * d + q];
                if apq.abs() <= 1e-15 {
                    continue;
                }
                let app = a[p * d + p];
                let aqq = a[q * d + q];
                let tau = (aqq - app) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                a[p * d + p] = app - t * apq;
                a[q * d + q] = aqq + t * apq;
                a[p * d + q] = 0.0;
                a[q * d + p] = 0.0;

                for r in 0..d {
                    if r != p && r != q {
                        let arp = a[r * d + p];
                        let arq = a[r * d + q];
                        a[r * d + p] = c * arp - s * arq;
                        a[p * d + r] = a[r * d + p];
                        a[r * d + q] = s * arp + c * arq;
                        a[q * d + r] = a[r * d + q];
                    }
                    let vrp = v[r * d + p];
                    let vrq = v[r * d + q];
                    v[r * d + p] = c * vrp - s * vrq;
                    v[r * d + q] = s * vrp + c * vrq;
                }
            }
        }
    }

    let mut order: Vec<usize> = (0..d).collect();
    order.sort_by(|&i, &j| a[j * d + j].total_cmp(&a[i * d + i]));
    let evals: Vec<f64> = order.iter().map(|&col| a[col * d + col].max(0.0)).collect();
    let evecs: Vec<Vec<f64>> = order
        .iter()
        .map(|&col| (0..d).map(|row| v[row * d + col]).collect())
        .collect();
    (evals, evecs)
}

fn compute_group_ellipse_95(label: &str, pts: &[[f64; 2]]) -> Option<ConfidenceEllipse2D> {
    let n = pts.len();
    if n < 3 {
        return None;
    }
    let nf = n as f64;
    let cx = pts.iter().map(|p| p[0]).sum::<f64>() / nf;
    let cy = pts.iter().map(|p| p[1]).sum::<f64>() / nf;
    let inv_dof = 1.0 / ((n - 1) as f64);
    let (mut sxx, mut sxy, mut syy) = (0.0, 0.0, 0.0);
    for p in pts {
        let dx = p[0] - cx;
        let dy = p[1] - cy;
        sxx += dx * dx;
        sxy += dx * dy;
        syy += dy * dy;
    }
    sxx *= inv_dof;
    sxy *= inv_dof;
    syy *= inv_dof;

    let tr = sxx + syy;
    let disc = ((sxx - syy) * (sxx - syy) + 4.0 * sxy * sxy).sqrt();
    let l1 = (0.5 * (tr + disc)).max(1e-15);
    let l2 = (0.5 * (tr - disc)).max(1e-15);
    let angle = if sxy.abs() > 1e-15 || sxx < syy {
        (l1 - sxx).atan2(sxy)
    } else {
        0.0
    };

    let scale2 = if n > 3 {
        let f95 = f_distribution_quantile(0.95, 2.0, (n - 2) as f64);
        (2.0 * (n - 1) as f64 / ((n - 2) as f64)) * f95
    } else {
        5.991_464_547_107_979 // chi^2_{0.95, 2} fallback when n == 3
    };

    Some(ConfidenceEllipse2D {
        label: label.to_string(),
        center: [cx, cy],
        radii: [(l1 * scale2).sqrt(), (l2 * scale2).sqrt()],
        angle_rad: angle,
    })
}

/// Fit deterministic PCA, 95%/99% Hotelling $T^2$ critical limits, 2D confidence ellipses, and
/// Jackson–Mudholkar $Q$-residual (`SPE`) outlier thresholds on scaled matrix `scaled_x`.
pub fn fit_pca(
    scaled_x: &[Vec<f64>],
    sample_groups: &[String],
    requested_components: usize,
) -> PcaResult {
    let n = scaled_x.len();
    let p = scaled_x.first().map(|r| r.len()).unwrap_or(0);
    if n == 0 || p == 0 {
        return PcaResult {
            n_components: 0,
            scores: vec![Vec::new(); n],
            loadings: Vec::new(),
            singular_values: Vec::new(),
            explained_variance_ratio: Vec::new(),
            cumulative_variance_ratio: Vec::new(),
            hotelling_t2: vec![0.0; n],
            hotelling_t2_limit_95: 0.0,
            hotelling_t2_limit_99: 0.0,
            q_residuals: vec![0.0; n],
            q_residual_limit_95: 0.0,
            global_ellipse_95: ConfidenceEllipse2D {
                label: "Global 95% T2".to_string(),
                center: [0.0, 0.0],
                radii: [0.0, 0.0],
                angle_rad: 0.0,
            },
            group_ellipses_95: Vec::new(),
        };
    }

    let max_rank = n.saturating_sub(1).max(1).min(p);
    let a_comp = requested_components.clamp(1, max_rank);

    let total_ss: f64 = scaled_x
        .iter()
        .flat_map(|r| r.iter())
        .map(|&v| v * v)
        .sum::<f64>()
        .max(1e-30);

    // SVD via smaller Gram matrix for speed & determinism (`N x N` when N <= P, `P x P` otherwise)
    let (all_sigmas, mut loadings, mut scores) = if n <= p {
        let mut gram = vec![0.0; n * n];
        for i in 0..n {
            for k in i..n {
                let dot: f64 = scaled_x[i]
                    .iter()
                    .zip(scaled_x[k].iter())
                    .map(|(&x, &y)| x * y)
                    .sum();
                gram[i * n + k] = dot;
                gram[k * n + i] = dot;
            }
        }
        let (evals, u_cols) = jacobi_symmetric_eigen_desc(&gram, n);
        let sigmas: Vec<f64> = evals.iter().map(|&ev| ev.max(0.0).sqrt()).collect();
        let mut p_vecs = Vec::with_capacity(a_comp);
        let mut t_mat = vec![vec![0.0; a_comp]; n];
        for c in 0..a_comp {
            let sig = sigmas[c];
            let mut p_c = vec![0.0; p];
            if sig > 1e-12 {
                let inv_s = 1.0 / sig;
                for i in 0..n {
                    let ui = u_cols[c][i] * inv_s;
                    for j in 0..p {
                        p_c[j] += scaled_x[i][j] * ui;
                    }
                }
                // Re-normalize loading vector and apply canonical deterministic sign
                let p_norm = p_c.iter().map(|&v| v * v).sum::<f64>().sqrt().max(1e-15);
                for v in &mut p_c {
                    *v /= p_norm;
                }
            } else if c < p {
                p_c[c] = 1.0;
            }
            let max_j = p_c
                .iter()
                .enumerate()
                .max_by(|(_, x), (_, y)| x.abs().total_cmp(&y.abs()))
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            if p_c[max_j] < 0.0 {
                for v in &mut p_c {
                    *v = -*v;
                }
            }
            for i in 0..n {
                t_mat[i][c] = scaled_x[i]
                    .iter()
                    .zip(p_c.iter())
                    .map(|(&x, &pv)| x * pv)
                    .sum();
            }
            p_vecs.push(p_c);
        }
        (sigmas, p_vecs, t_mat)
    } else {
        let mut cov = vec![0.0; p * p];
        for r in scaled_x {
            for j1 in 0..p {
                let x1 = r[j1];
                for j2 in j1..p {
                    cov[j1 * p + j2] += x1 * r[j2];
                }
            }
        }
        for j1 in 0..p {
            for j2 in (j1 + 1)..p {
                cov[j2 * p + j1] = cov[j1 * p + j2];
            }
        }
        let (evals, v_cols) = jacobi_symmetric_eigen_desc(&cov, p);
        let sigmas: Vec<f64> = evals.iter().map(|&ev| ev.max(0.0).sqrt()).collect();
        let mut p_vecs = Vec::with_capacity(a_comp);
        let mut t_mat = vec![vec![0.0; a_comp]; n];
        for c in 0..a_comp {
            let mut p_c = v_cols[c].clone();
            let max_j = p_c
                .iter()
                .enumerate()
                .max_by(|(_, x), (_, y)| x.abs().total_cmp(&y.abs()))
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            if p_c[max_j] < 0.0 {
                for v in &mut p_c {
                    *v = -*v;
                }
            }
            for i in 0..n {
                t_mat[i][c] = scaled_x[i]
                    .iter()
                    .zip(p_c.iter())
                    .map(|(&x, &pv)| x * pv)
                    .sum();
            }
            p_vecs.push(p_c);
        }
        (sigmas, p_vecs, t_mat)
    };

    let singular_values: Vec<f64> = (0..a_comp)
        .map(|c| scores.iter().map(|r| r[c] * r[c]).sum::<f64>().sqrt())
        .collect();

    let mut explained = Vec::with_capacity(a_comp);
    let mut cumulative = Vec::with_capacity(a_comp);
    let mut cum_acc = 0.0;
    for &sv in &singular_values {
        let r2 = ((sv * sv) / total_ss).clamp(0.0, 1.0);
        cum_acc = (cum_acc + r2).min(1.0);
        explained.push(r2);
        cumulative.push(cum_acc);
    }

    let dof = (n.saturating_sub(1)).max(1) as f64;
    let comp_vars: Vec<f64> = singular_values
        .iter()
        .map(|&sv| (sv * sv / dof).max(1e-30))
        .collect();

    let mut hotelling_t2 = vec![0.0; n];
    for i in 0..n {
        let mut t2 = 0.0;
        for c in 0..a_comp {
            if singular_values[c] > 1e-11 {
                t2 += (scores[i][c] * scores[i][c]) / comp_vars[c];
            }
        }
        hotelling_t2[i] = t2;
    }

    let eff_a = a_comp.min(n.saturating_sub(2).max(1));
    let (t2_95, t2_99) = if n > eff_a + 1 {
        let af = eff_a as f64;
        let nf = n as f64;
        let factor = af * (nf - 1.0) / (nf - af);
        (
            factor * f_distribution_quantile(0.95, af, nf - af),
            factor * f_distribution_quantile(0.99, af, nf - af),
        )
    } else {
        (5.991_464_5, 9.210_340_4)
    };

    // Q-residuals (SPE) & Jackson-Mudholkar (1979) 95% threshold
    let mut q_residuals = vec![0.0; n];
    for i in 0..n {
        let mut q = 0.0;
        for j in 0..p {
            let mut x_hat = 0.0;
            for c in 0..a_comp {
                x_hat += scores[i][c] * loadings[c][j];
            }
            let e = scaled_x[i][j] - x_hat;
            q += e * e;
        }
        q_residuals[i] = q;
    }

    let rem_lambdas: Vec<f64> = all_sigmas
        .iter()
        .skip(a_comp)
        .map(|&s| (s * s) / dof)
        .filter(|&l| l > 1e-18)
        .collect();
    let q_limit_95 = if !rem_lambdas.is_empty() {
        let theta1: f64 = rem_lambdas.iter().sum();
        let theta2: f64 = rem_lambdas.iter().map(|&l| l * l).sum();
        let theta3: f64 = rem_lambdas.iter().map(|&l| l * l * l).sum();
        let h0 = (1.0 - (2.0 * theta1 * theta3) / (3.0 * theta2 * theta2).max(1e-30)).max(0.05);
        let z95 = 1.644_853_626_951_472_2; // one-sided upper 95% normal quantile
        let term = (z95 * (2.0 * theta2 * h0 * h0).sqrt()) / theta1
            + 1.0
            + (theta2 * h0 * (h0 - 1.0)) / (theta1 * theta1);
        theta1 * term.max(0.0).powf(1.0 / h0)
    } else {
        0.0
    };

    // 2D score ellipses on (PC1, PC2)
    let ell_factor = if n > 3 {
        let nf = n as f64;
        (2.0 * (nf - 1.0) / (nf - 2.0)) * f_distribution_quantile(0.95, 2.0, nf - 2.0)
    } else {
        5.991_464_5
    };
    let rx = (comp_vars[0] * ell_factor).sqrt();
    let ry = if a_comp >= 2 {
        (comp_vars[1] * ell_factor).sqrt()
    } else {
        0.0
    };
    let global_ellipse_95 = ConfidenceEllipse2D {
        label: "Global 95% T2".to_string(),
        center: [0.0, 0.0],
        radii: [rx, ry],
        angle_rad: 0.0,
    };

    let mut group_pts: BTreeMap<String, Vec<[f64; 2]>> = BTreeMap::new();
    for (i, grp) in sample_groups.iter().enumerate().take(n) {
        let px = scores[i][0];
        let py = if a_comp >= 2 { scores[i][1] } else { 0.0 };
        group_pts.entry(grp.clone()).or_default().push([px, py]);
    }
    let group_ellipses_95 = group_pts
        .iter()
        .filter_map(|(g, pts)| compute_group_ellipse_95(g, pts))
        .collect();

    loadings.truncate(a_comp);
    for row in &mut scores {
        row.truncate(a_comp);
    }

    PcaResult {
        n_components: a_comp,
        scores,
        loadings,
        singular_values,
        explained_variance_ratio: explained,
        cumulative_variance_ratio: cumulative,
        hotelling_t2,
        hotelling_t2_limit_95: t2_95,
        hotelling_t2_limit_99: t2_99,
        q_residuals,
        q_residual_limit_95: q_limit_95,
        global_ellipse_95,
        group_ellipses_95,
    }
}

// ============================================================================
// 10. NIPALS PLS-DA & OPLS-DA with Leak-Free Stratified CV & Permutation Test
// ============================================================================

fn select_binary_contrast_groups(
    sample_groups: &[String],
    control_override: Option<&str>,
    case_override: Option<&str>,
) -> Option<(String, String, Vec<usize>, Vec<f64>)> {
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for g in sample_groups {
        *counts.entry(g.clone()).or_insert(0) += 1;
    }
    let valid_groups: Vec<String> = counts
        .into_iter()
        .filter_map(|(g, c)| (c >= 2).then_some(g))
        .collect();
    if valid_groups.len() < 2 {
        return None;
    }

    let ctrl = if let Some(c) = control_override {
        valid_groups
            .iter()
            .find(|g| g.as_str() == c)
            .cloned()
            .unwrap_or_else(|| valid_groups[0].clone())
    } else if valid_groups
        .iter()
        .any(|g| g.eq_ignore_ascii_case("control"))
    {
        valid_groups
            .iter()
            .find(|g| g.eq_ignore_ascii_case("control"))
            .unwrap()
            .clone()
    } else {
        valid_groups[0].clone()
    };

    let case = if let Some(c) = case_override {
        valid_groups
            .iter()
            .find(|g| g.as_str() == c && g.as_str() != ctrl)
            .cloned()
            .unwrap_or_else(|| {
                valid_groups
                    .iter()
                    .find(|g| g.as_str() != ctrl)
                    .unwrap()
                    .clone()
            })
    } else {
        valid_groups
            .iter()
            .find(|g| g.as_str() != ctrl)
            .unwrap()
            .clone()
    };

    let mut active_indices = Vec::new();
    let mut y_bin = Vec::new();
    for (idx, g) in sample_groups.iter().enumerate() {
        if g == &ctrl {
            active_indices.push(idx);
            y_bin.push(0.0);
        } else if g == &case {
            active_indices.push(idx);
            y_bin.push(1.0);
        }
    }
    let n0 = y_bin.iter().filter(|&&v| v < 0.5).count();
    let n1 = y_bin.len() - n0;
    if n0 < 2 || n1 < 2 {
        return None;
    }
    Some((ctrl, case, active_indices, y_bin))
}

fn build_stratified_folds(y_binary: &[f64], requested_folds: usize) -> Vec<Vec<usize>> {
    let mut c0 = Vec::new();
    let mut c1 = Vec::new();
    for (i, &y) in y_binary.iter().enumerate() {
        if y < 0.5 {
            c0.push(i);
        } else {
            c1.push(i);
        }
    }
    let k = requested_folds.max(2).min(c0.len()).min(c1.len());
    let mut folds = vec![Vec::new(); k];
    for (pos, idx) in c0.into_iter().enumerate() {
        folds[pos % k].push(idx);
    }
    for (pos, idx) in c1.into_iter().enumerate() {
        folds[pos % k].push(idx);
    }
    for f in &mut folds {
        f.sort_unstable();
    }
    folds
}

#[derive(Clone, Debug)]
struct PlsCoreFit {
    scores: Vec<Vec<f64>>,
    x_loadings: Vec<Vec<f64>>,
    x_weights: Vec<Vec<f64>>,
    y_loadings: Vec<f64>,
    y_mean: f64,
    y_fitted: Vec<f64>,
    r2_x: f64,
    r2_y: f64,
}

impl PlsCoreFit {
    fn predict_scaled_row(&self, x_scaled: &[f64]) -> f64 {
        let mut x_res = x_scaled.to_vec();
        let mut y_hat = self.y_mean;
        for a in 0..self.y_loadings.len() {
            let t_a: f64 = x_res
                .iter()
                .zip(self.x_weights[a].iter())
                .map(|(&x, &w)| x * w)
                .sum();
            y_hat += t_a * self.y_loadings[a];
            for (j, slot) in x_res.iter_mut().enumerate() {
                *slot -= t_a * self.x_loadings[a][j];
            }
        }
        y_hat
    }
}

fn fit_pls_core(scaled_x: &[Vec<f64>], y: &[f64], n_components: usize) -> PlsCoreFit {
    let n = scaled_x.len();
    let p = scaled_x.first().map(|r| r.len()).unwrap_or(0);
    let y_mean = y.iter().sum::<f64>() / (n.max(1) as f64);
    let mut y_res: Vec<f64> = y.iter().map(|&v| v - y_mean).collect();
    let y_tss: f64 = y_res.iter().map(|&v| v * v).sum::<f64>().max(1e-30);
    let x_tss: f64 = scaled_x
        .iter()
        .flat_map(|r| r.iter())
        .map(|&v| v * v)
        .sum::<f64>()
        .max(1e-30);

    let a_max = n_components.clamp(1, n.saturating_sub(1).max(1).min(p.max(1)));
    let mut x_work = scaled_x.to_vec();
    let mut scores = vec![Vec::new(); n];
    let mut x_loadings = Vec::new();
    let mut x_weights = Vec::new();
    let mut y_loadings = Vec::new();

    for _a in 0..a_max {
        let mut w = vec![0.0; p];
        for i in 0..n {
            let yi = y_res[i];
            for j in 0..p {
                w[j] += x_work[i][j] * yi;
            }
        }
        let w_norm = w.iter().map(|&v| v * v).sum::<f64>().sqrt();
        if w_norm <= 1e-13 {
            break;
        }
        for v in &mut w {
            *v /= w_norm;
        }

        let mut t = vec![0.0; n];
        let mut tt = 0.0;
        for i in 0..n {
            let ti: f64 = x_work[i].iter().zip(w.iter()).map(|(&x, &wj)| x * wj).sum();
            t[i] = ti;
            tt += ti * ti;
        }
        if tt <= 1e-15 {
            break;
        }

        let mut p_vec = vec![0.0; p];
        for i in 0..n {
            let ti = t[i];
            for j in 0..p {
                p_vec[j] += x_work[i][j] * ti;
            }
        }
        for v in &mut p_vec {
            *v /= tt;
        }

        let c_a: f64 = y_res
            .iter()
            .zip(t.iter())
            .map(|(&yi, &ti)| yi * ti)
            .sum::<f64>()
            / tt;

        for i in 0..n {
            let ti = t[i];
            for j in 0..p {
                x_work[i][j] -= ti * p_vec[j];
            }
            y_res[i] -= ti * c_a;
            scores[i].push(ti);
        }
        x_weights.push(w);
        x_loadings.push(p_vec);
        y_loadings.push(c_a);
    }

    let x_rss: f64 = x_work.iter().flat_map(|r| r.iter()).map(|&v| v * v).sum();
    let y_rss: f64 = y_res.iter().map(|&v| v * v).sum();
    let y_fitted: Vec<f64> = (0..n).map(|i| y[i] - y_res[i]).collect();

    PlsCoreFit {
        scores,
        x_loadings,
        x_weights,
        y_loadings,
        y_mean,
        y_fitted,
        r2_x: (1.0 - x_rss / x_tss).clamp(0.0, 1.0),
        r2_y: (1.0 - y_rss / y_tss).clamp(0.0, 1.0),
    }
}

/// Fit multi-component NIPALS `PLS-DA` with leak-free stratified $k$-fold cross-validated $Q^2$ and `VIP`.
pub fn fit_pls_da(
    norm_matrix: &[Vec<f64>],
    sample_groups: &[String],
    scaling: ScalingMethod,
    n_components: usize,
    cv_folds: usize,
    control_override: Option<&str>,
    case_override: Option<&str>,
) -> Option<PlsDaResult> {
    let (ctrl, case, active_idx, y_bin) =
        select_binary_contrast_groups(sample_groups, control_override, case_override)?;
    let n_act = active_idx.len();
    let p = norm_matrix.first()?.len();
    if p == 0 {
        return None;
    }

    let sub_x: Vec<Vec<f64>> = active_idx.iter().map(|&i| norm_matrix[i].clone()).collect();
    let all_rows: Vec<usize> = (0..n_act).collect();
    let full_scaler = FittedScaler::fit_rows(&sub_x, &all_rows, scaling);
    let scaled_full = full_scaler.transform_matrix(&sub_x);
    let core = fit_pls_core(&scaled_full, &y_bin, n_components);
    let a_used = core.y_loadings.len();

    // Leak-free stratified k-fold CV
    let folds = build_stratified_folds(&y_bin, cv_folds);
    let mut y_cv = vec![core.y_mean; n_act];
    for val_indices in &folds {
        let mut is_val = vec![false; n_act];
        for &vi in val_indices {
            is_val[vi] = true;
        }
        let train_indices: Vec<usize> = (0..n_act).filter(|&i| !is_val[i]).collect();
        let fold_scaler = FittedScaler::fit_rows(&sub_x, &train_indices, scaling);
        let x_tr: Vec<Vec<f64>> = train_indices
            .iter()
            .map(|&i| fold_scaler.transform_row(&sub_x[i]))
            .collect();
        let y_tr: Vec<f64> = train_indices.iter().map(|&i| y_bin[i]).collect();
        let fold_model = fit_pls_core(&x_tr, &y_tr, a_used.max(1));
        for &vi in val_indices {
            let x_val = fold_scaler.transform_row(&sub_x[vi]);
            y_cv[vi] = fold_model.predict_scaled_row(&x_val);
        }
    }

    let y_mean = core.y_mean;
    let tss: f64 = y_bin
        .iter()
        .map(|&y| (y - y_mean) * (y - y_mean))
        .sum::<f64>()
        .max(1e-30);
    let press: f64 = y_bin
        .iter()
        .zip(y_cv.iter())
        .map(|(&y, &yh)| (y - yh) * (y - yh))
        .sum();
    let q2 = 1.0 - press / tss;

    // VIP_j = sqrt(P * sum_a SSY_a * w_{a,j}^2 / sum_a SSY_a)
    let mut vip = vec![0.0; p];
    let mut ssy_list = Vec::with_capacity(a_used);
    let mut ssy_tot = 0.0;
    for a in 0..a_used {
        let tt: f64 = core.scores.iter().map(|r| r[a] * r[a]).sum();
        let ssy = core.y_loadings[a] * core.y_loadings[a] * tt;
        ssy_list.push(ssy);
        ssy_tot += ssy;
    }
    if ssy_tot > 1e-30 {
        for (j, vip_j) in vip.iter_mut().enumerate() {
            let mut weighted_w2 = 0.0;
            for (&ssy_a, w_row) in ssy_list.iter().zip(core.x_weights.iter()).take(a_used) {
                let waj = w_row[j];
                weighted_w2 += ssy_a * waj * waj;
            }
            *vip_j = ((p as f64) * weighted_w2 / ssy_tot).sqrt();
        }
    }

    Some(PlsDaResult {
        control_group: ctrl,
        case_group: case,
        n_components: a_used,
        active_sample_indices: active_idx,
        y_binary: y_bin,
        scores: core.scores,
        x_loadings: core.x_loadings,
        x_weights: core.x_weights,
        y_loadings: core.y_loadings,
        y_fitted: core.y_fitted,
        y_cv_predicted: y_cv,
        r2_x: core.r2_x,
        r2_y: core.r2_y,
        q2,
        vip_scores: vip,
    })
}

#[derive(Clone, Debug)]
struct OplsCoreFit {
    w_p: Vec<f64>,
    t_p: Vec<f64>,
    p_p: Vec<f64>,
    c_p: f64,
    y_mean: f64,
    t_ortho: Vec<Vec<f64>>,
    p_ortho: Vec<Vec<f64>>,
    w_ortho: Vec<Vec<f64>>,
    y_fitted: Vec<f64>,
    r2_x_pred: f64,
    r2_x_ortho: f64,
    r2_y: f64,
}

impl OplsCoreFit {
    fn predict_scaled_row(&self, x_scaled: &[f64]) -> f64 {
        let mut x_filt = x_scaled.to_vec();
        for k in 0..self.w_ortho.len() {
            let t_ok: f64 = x_filt
                .iter()
                .zip(self.w_ortho[k].iter())
                .map(|(&x, &w)| x * w)
                .sum();
            for (j, slot) in x_filt.iter_mut().enumerate() {
                *slot -= t_ok * self.p_ortho[k][j];
            }
        }
        let t_p_new: f64 = x_filt
            .iter()
            .zip(self.w_p.iter())
            .map(|(&x, &w)| x * w)
            .sum();
        self.y_mean + self.c_p * t_p_new
    }
}

/// Single-$y$ Trygg & Wold (*J. Chemom.* 2002, 16, 119–128) NIPALS `OPLS` solver.
fn fit_opls_core(scaled_x: &[Vec<f64>], y: &[f64], n_orthogonal: usize) -> OplsCoreFit {
    let n = scaled_x.len();
    let p = scaled_x.first().map(|r| r.len()).unwrap_or(0);
    let y_mean = y.iter().sum::<f64>() / (n.max(1) as f64);
    let y_tilde: Vec<f64> = y.iter().map(|&v| v - y_mean).collect();
    let y_tss: f64 = y_tilde.iter().map(|&v| v * v).sum::<f64>().max(1e-30);
    let x_tss: f64 = scaled_x
        .iter()
        .flat_map(|r| r.iter())
        .map(|&v| v * v)
        .sum::<f64>()
        .max(1e-30);

    // 1. Predictive weight vector w_p = X^T y_tilde / ||X^T y_tilde||_2
    let mut w_p = vec![0.0; p];
    for i in 0..n {
        let yi = y_tilde[i];
        for j in 0..p {
            w_p[j] += scaled_x[i][j] * yi;
        }
    }
    let wp_norm = w_p.iter().map(|&v| v * v).sum::<f64>().sqrt();
    if wp_norm > 1e-15 {
        for v in &mut w_p {
            *v /= wp_norm;
        }
    } else if p > 0 {
        w_p[0] = 1.0;
    }

    // 2. Iteratively extract up to A_o orthogonal components (Trygg & Wold 2002, Algorithm 1)
    let max_ortho = n_orthogonal
        .min(n.saturating_sub(2))
        .min(p.saturating_sub(1));
    let mut x_work = scaled_x.to_vec();
    let mut t_ortho = Vec::with_capacity(max_ortho);
    let mut p_ortho = Vec::with_capacity(max_ortho);
    let mut w_ortho = Vec::with_capacity(max_ortho);

    for _k in 0..max_ortho {
        let mut t_cur = vec![0.0; n];
        let mut tt = 0.0;
        for i in 0..n {
            let ti: f64 = x_work[i].iter().zip(w_p.iter()).map(|(&x, &w)| x * w).sum();
            t_cur[i] = ti;
            tt += ti * ti;
        }
        if tt <= 1e-15 {
            break;
        }

        let mut p_cur = vec![0.0; p];
        for i in 0..n {
            let ti = t_cur[i];
            for j in 0..p {
                p_cur[j] += x_work[i][j] * ti;
            }
        }
        for v in &mut p_cur {
            *v /= tt;
        }

        // w_o = p - (w_p^T p) * w_p  (orthogonal to w_p!)
        let wp_dot_p: f64 = w_p.iter().zip(p_cur.iter()).map(|(&w, &pv)| w * pv).sum();
        let mut w_ok: Vec<f64> = p_cur
            .iter()
            .zip(w_p.iter())
            .map(|(&pv, &wv)| pv - wp_dot_p * wv)
            .collect();
        let wok_norm = w_ok.iter().map(|&v| v * v).sum::<f64>().sqrt();
        if wok_norm <= 1e-12 {
            break;
        }
        for v in &mut w_ok {
            *v /= wok_norm;
        }

        let mut t_ok = vec![0.0; n];
        let mut t_ok_ss = 0.0;
        for i in 0..n {
            let tok_i: f64 = x_work[i]
                .iter()
                .zip(w_ok.iter())
                .map(|(&x, &w)| x * w)
                .sum();
            t_ok[i] = tok_i;
            t_ok_ss += tok_i * tok_i;
        }
        if t_ok_ss <= 1e-15 {
            break;
        }

        let mut p_ok = vec![0.0; p];
        for i in 0..n {
            let tok_i = t_ok[i];
            for j in 0..p {
                p_ok[j] += x_work[i][j] * tok_i;
            }
        }
        for v in &mut p_ok {
            *v /= t_ok_ss;
        }

        // Deflate X by orthogonal component t_ok * p_ok^T
        for i in 0..n {
            let tok_i = t_ok[i];
            for j in 0..p {
                x_work[i][j] -= tok_i * p_ok[j];
            }
        }

        t_ortho.push(t_ok);
        p_ortho.push(p_ok);
        w_ortho.push(w_ok);
    }

    // 3. Final predictive component on orthogonal-filtered matrix X_filt = x_work
    let mut t_p = vec![0.0; n];
    let mut tp_ss = 0.0;
    for i in 0..n {
        let ti: f64 = x_work[i].iter().zip(w_p.iter()).map(|(&x, &w)| x * w).sum();
        t_p[i] = ti;
        tp_ss += ti * ti;
    }

    let mut p_p = vec![0.0; p];
    let mut c_p = 0.0;
    if tp_ss > 1e-15 {
        for i in 0..n {
            let ti = t_p[i];
            for j in 0..p {
                p_p[j] += x_work[i][j] * ti;
            }
        }
        for v in &mut p_p {
            *v /= tp_ss;
        }
        c_p = y_tilde
            .iter()
            .zip(t_p.iter())
            .map(|(&yi, &ti)| yi * ti)
            .sum::<f64>()
            / tp_ss;
    }

    // Variance explained decomposition
    let mut ss_pred = 0.0;
    let mut ss_ortho = 0.0;
    for i in 0..n {
        let tpi = t_p[i];
        for j in 0..p {
            let xp = tpi * p_p[j];
            ss_pred += xp * xp;
            let xo = scaled_x[i][j] - x_work[i][j];
            ss_ortho += xo * xo;
        }
    }

    let mut y_fitted = vec![0.0; n];
    let mut y_rss = 0.0;
    for i in 0..n {
        let yh = y_mean + c_p * t_p[i];
        y_fitted[i] = yh;
        let err = y[i] - yh;
        y_rss += err * err;
    }

    OplsCoreFit {
        w_p,
        t_p,
        p_p,
        c_p,
        y_mean,
        t_ortho,
        p_ortho,
        w_ortho,
        y_fitted,
        r2_x_pred: (ss_pred / x_tss).clamp(0.0, 1.0),
        r2_x_ortho: (ss_ortho / x_tss).clamp(0.0, 1.0),
        r2_y: (1.0 - y_rss / y_tss).clamp(0.0, 1.0),
    }
}

fn eval_opls_stratified_cv(
    sub_x: &[Vec<f64>],
    y_bin: &[f64],
    scaling: ScalingMethod,
    n_orthogonal: usize,
    cv_folds: usize,
) -> (f64, Vec<f64>) {
    let n = sub_x.len();
    let y_mean = y_bin.iter().sum::<f64>() / (n.max(1) as f64);
    let tss: f64 = y_bin
        .iter()
        .map(|&y| (y - y_mean) * (y - y_mean))
        .sum::<f64>()
        .max(1e-30);
    let folds = build_stratified_folds(y_bin, cv_folds);
    let mut y_cv = vec![y_mean; n];

    for val_indices in &folds {
        let mut is_val = vec![false; n];
        for &vi in val_indices {
            is_val[vi] = true;
        }
        let train_indices: Vec<usize> = (0..n).filter(|&i| !is_val[i]).collect();
        let fold_scaler = FittedScaler::fit_rows(sub_x, &train_indices, scaling);
        let x_tr: Vec<Vec<f64>> = train_indices
            .iter()
            .map(|&i| fold_scaler.transform_row(&sub_x[i]))
            .collect();
        let y_tr: Vec<f64> = train_indices.iter().map(|&i| y_bin[i]).collect();
        let fold_fit = fit_opls_core(&x_tr, &y_tr, n_orthogonal);
        for &vi in val_indices {
            let x_val = fold_scaler.transform_row(&sub_x[vi]);
            y_cv[vi] = fold_fit.predict_scaled_row(&x_val);
        }
    }

    let press: f64 = y_bin
        .iter()
        .zip(y_cv.iter())
        .map(|(&y, &yh)| (y - yh) * (y - yh))
        .sum();
    (1.0 - press / tss, y_cv)
}

fn pearson_corr_slices(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n < 2 {
        return 0.0;
    }
    let nf = n as f64;
    let ma = a[..n].iter().sum::<f64>() / nf;
    let mb = b[..n].iter().sum::<f64>() / nf;
    let (mut cov, mut va, mut vb) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let da = a[i] - ma;
        let db = b[i] - mb;
        cov += da * db;
        va += da * da;
        vb += db * db;
    }
    let denom = (va * vb).sqrt();
    if denom > 1e-15 {
        (cov / denom).clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

/// Fit `OPLS-DA` (`1` predictive + `n_orthogonal` orthogonal components) with Wiklund `S-plot`,
/// predictive `VIP`, 95% Hotelling $T^2$ ellipses, leak-free stratified $k$-fold CV $Q^2$, and
/// deterministic `Westerhuis et al. (2008)` $N$-permutation overfitting validation.
#[allow(clippy::too_many_arguments)]
pub fn fit_opls_da(
    norm_matrix: &[Vec<f64>],
    sample_groups: &[String],
    scaling: ScalingMethod,
    n_orthogonal: usize,
    cv_folds: usize,
    n_permutations: usize,
    rng_seed: u64,
    control_override: Option<&str>,
    case_override: Option<&str>,
) -> Option<OplsDaResult> {
    let (ctrl, case, active_idx, y_bin) =
        select_binary_contrast_groups(sample_groups, control_override, case_override)?;
    let n_act = active_idx.len();
    let p = norm_matrix.first()?.len();
    if p == 0 {
        return None;
    }

    let sub_x: Vec<Vec<f64>> = active_idx.iter().map(|&i| norm_matrix[i].clone()).collect();
    let all_rows: Vec<usize> = (0..n_act).collect();
    let full_scaler = FittedScaler::fit_rows(&sub_x, &all_rows, scaling);
    let scaled_full = full_scaler.transform_matrix(&sub_x);

    let core = fit_opls_core(&scaled_full, &y_bin, n_orthogonal);
    let a_ortho_used = core.w_ortho.len();
    let (obs_q2, y_cv) = eval_opls_stratified_cv(&sub_x, &y_bin, scaling, a_ortho_used, cv_folds);

    // Wiklund et al. (2008) S-plot: Cov(t_p, X_tilde_j) vs Corr(t_p, X_tilde_j)
    let dof = (n_act.saturating_sub(1)).max(1) as f64;
    let tp_mean = core.t_p.iter().sum::<f64>() / (n_act as f64);
    let tp_var = core
        .t_p
        .iter()
        .map(|&t| (t - tp_mean) * (t - tp_mean))
        .sum::<f64>()
        / dof;
    let tp_sd = tp_var.sqrt().max(1e-15);

    let mut s_plot_cov = vec![0.0; p];
    let mut s_plot_corr = vec![0.0; p];
    for j in 0..p {
        let xj_mean = scaled_full.iter().map(|r| r[j]).sum::<f64>() / (n_act as f64);
        let mut cov_num = 0.0;
        let mut xj_ss = 0.0;
        for (&tp_i, row) in core.t_p.iter().zip(scaled_full.iter()).take(n_act) {
            let dt = tp_i - tp_mean;
            let dx = row[j] - xj_mean;
            cov_num += dt * dx;
            xj_ss += dx * dx;
        }
        let cov = cov_num / dof;
        let xj_sd = (xj_ss / dof).sqrt();
        s_plot_cov[j] = cov;
        s_plot_corr[j] = if xj_sd > 1e-15 && tp_var > 1e-24 {
            (cov / (tp_sd * xj_sd)).clamp(-1.0, 1.0)
        } else {
            0.0
        };
    }

    let sqrt_p = (p as f64).sqrt();
    let vip_pred: Vec<f64> = core.w_p.iter().map(|&w| sqrt_p * w.abs()).collect();

    // 95% Hotelling T^2 ellipses in (t_p, t_o1) plane
    let to1: Vec<f64> = if let Some(first_to) = core.t_ortho.first() {
        first_to.clone()
    } else {
        vec![0.0; n_act]
    };
    let to1_mean = to1.iter().sum::<f64>() / (n_act as f64);
    let to1_var = to1
        .iter()
        .map(|&v| (v - to1_mean) * (v - to1_mean))
        .sum::<f64>()
        / dof;
    let ell_factor = if n_act > 3 {
        let nf = n_act as f64;
        (2.0 * (nf - 1.0) / (nf - 2.0)) * f_distribution_quantile(0.95, 2.0, nf - 2.0)
    } else {
        5.991_464_5
    };
    let global_ellipse_95 = ConfidenceEllipse2D {
        label: "Global 95% T2".to_string(),
        center: [tp_mean, to1_mean],
        radii: [(tp_var * ell_factor).sqrt(), (to1_var * ell_factor).sqrt()],
        angle_rad: 0.0,
    };

    let mut ctrl_pts = Vec::new();
    let mut case_pts = Vec::new();
    for i in 0..n_act {
        if y_bin[i] < 0.5 {
            ctrl_pts.push([core.t_p[i], to1[i]]);
        } else {
            case_pts.push([core.t_p[i], to1[i]]);
        }
    }
    let mut group_ellipses_95 = Vec::new();
    if let Some(e0) = compute_group_ellipse_95(&ctrl, &ctrl_pts) {
        group_ellipses_95.push(e0);
    }
    if let Some(e1) = compute_group_ellipse_95(&case, &case_pts) {
        group_ellipses_95.push(e1);
    }

    // Deterministic Westerhuis et al. (2008) N-permutation validation
    let mut rng = Rng::new(rng_seed);
    let mut trials = Vec::with_capacity(n_permutations);
    let mut ge_q2 = 0usize;
    let mut ge_r2y = 0usize;
    let mut y_perm = y_bin.clone();

    for _ in 0..n_permutations {
        rng.shuffle(&mut y_perm);
        let abs_corr = pearson_corr_slices(&y_bin, &y_perm).abs();
        let perm_core = fit_opls_core(&scaled_full, &y_perm, a_ortho_used);
        let (perm_q2, _) =
            eval_opls_stratified_cv(&sub_x, &y_perm, scaling, a_ortho_used, cv_folds);
        if perm_q2 >= obs_q2 - 1e-12 {
            ge_q2 += 1;
        }
        if perm_core.r2_y >= core.r2_y - 1e-12 {
            ge_r2y += 1;
        }
        trials.push(PermutationTrial {
            abs_label_corr: abs_corr,
            r2_y: perm_core.r2_y,
            q2: perm_q2,
        });
    }

    let denom_p = (1 + n_permutations) as f64;
    let p_value_q2 = ((1 + ge_q2) as f64) / denom_p;
    let p_value_r2_y = ((1 + ge_r2y) as f64) / denom_p;
    let mean_null_q2 = if trials.is_empty() {
        0.0
    } else {
        trials.iter().map(|t| t.q2).sum::<f64>() / (trials.len() as f64)
    };

    // OLS regression intercept of Q^2 against |r(y, y_perm)| anchored with observed point (1.0, obs_q2)
    let q2_intercept = if !trials.is_empty() {
        let m_tot = (trials.len() + 1) as f64;
        let sum_x = trials.iter().map(|t| t.abs_label_corr).sum::<f64>() + 1.0;
        let sum_y = trials.iter().map(|t| t.q2).sum::<f64>() + obs_q2;
        let mx = sum_x / m_tot;
        let my = sum_y / m_tot;
        let mut sxx = (1.0 - mx) * (1.0 - mx);
        let mut sxy = (1.0 - mx) * (obs_q2 - my);
        for tr in &trials {
            let dx = tr.abs_label_corr - mx;
            sxx += dx * dx;
            sxy += dx * (tr.q2 - my);
        }
        let slope = if sxx > 1e-12 { sxy / sxx } else { 0.0 };
        my - slope * mx
    } else {
        0.0
    };

    let is_validated = obs_q2 >= 0.40 && (n_permutations == 0 || p_value_q2 <= 0.05);

    Some(OplsDaResult {
        control_group: ctrl,
        case_group: case,
        n_orthogonal: a_ortho_used,
        active_sample_indices: active_idx,
        y_binary: y_bin,
        predictive_scores: core.t_p,
        predictive_loadings: core.p_p,
        predictive_weights: core.w_p,
        predictive_y_weight: core.c_p,
        orthogonal_scores: core.t_ortho,
        orthogonal_loadings: core.p_ortho,
        orthogonal_weights: core.w_ortho,
        y_fitted: core.y_fitted,
        y_cv_predicted: y_cv,
        r2_x_predictive: core.r2_x_pred,
        r2_x_orthogonal: core.r2_x_ortho,
        r2_x_total: (core.r2_x_pred + core.r2_x_ortho).min(1.0),
        r2_y: core.r2_y,
        q2: obs_q2,
        vip_predictive: vip_pred,
        s_plot_cov,
        s_plot_corr,
        global_ellipse_95,
        group_ellipses_95,
        permutation: PermutationTestSummary {
            n_permutations,
            observed_r2_y: core.r2_y,
            observed_q2: obs_q2,
            p_value_q2,
            p_value_r2_y,
            mean_null_q2,
            q2_intercept,
            is_validated,
            trials,
        },
    })
}

// ============================================================================
// 11. STOCSY Driver Analysis & BMRB Reference Library Matching
// ============================================================================

/// Match a `STOCSY` driver profile (`driver_ppm`, bucket covariances & correlations) against the
/// built-in CC0 `BMRB` human biofluid metabolite library (`bmrb_biofluid_metabolite_library()`).
pub fn match_stocsy_to_bmrb_library(
    driver_bucket_idx: usize,
    buckets: &[SpectralBucket],
    bucket_cov: &[f64],
    bucket_corr: &[f64],
    exclusion_zones: &[ExclusionZonePpm],
) -> Vec<MetaboliteMatchScore> {
    if driver_bucket_idx >= buckets.len() {
        return Vec::new();
    }
    let drv_bucket = &buckets[driver_bucket_idx];
    let drv_ppm = drv_bucket.center_ppm;
    let half_drv_w = 0.5 * drv_bucket.width_ppm;

    let mut matches = Vec::new();
    for entry in BMRB_BIOFLUID_LIBRARY {
        // Observable multiplets (excluding resonances masked by water/urea exclusion zones)
        let obs_mults: Vec<&ReferenceMultiplet1H> = entry
            .multiplets
            .iter()
            .filter(|m| !exclusion_zones.iter().any(|z| z.contains(m.center_ppm)))
            .collect();
        if obs_mults.is_empty() {
            continue;
        }

        // Driver bucket must match at least one resonance of this metabolite within tolerance
        let driver_hits_entry = obs_mults.iter().any(|m| {
            let tol = m.shift_tol_ppm + half_drv_w;
            (drv_ppm - m.center_ppm).abs() <= tol
                || (m.center_ppm >= drv_bucket.low_ppm - m.shift_tol_ppm
                    && m.center_ppm <= drv_bucket.high_ppm + m.shift_tol_ppm)
        });
        if !driver_hits_entry {
            continue;
        }

        let mut matched_count = 0usize;
        let mut h_weighted_r2 = 0.0;
        let mut h_total = 0.0;
        let mut shift_ll = 0.0;
        let mut obs_covs = Vec::with_capacity(obs_mults.len());
        let mut theo_hs = Vec::with_capacity(obs_mults.len());
        let mut peak_pairs = Vec::with_capacity(obs_mults.len());

        for mult in &obs_mults {
            let h = mult.protons as f64;
            h_total += h;
            theo_hs.push(h);

            // Find best candidate bucket within tolerance (preferring high positive STOCSY r^2 + close shift)
            let mut best_j = None;
            let mut best_merit = f64::NEG_INFINITY;
            for (j, b) in buckets.iter().enumerate() {
                let tol = mult.shift_tol_ppm + 0.5 * b.width_ppm;
                let dist = (b.center_ppm - mult.center_ppm).abs();
                if dist <= tol {
                    let r_pos = bucket_corr[j].max(0.0);
                    let r2_pos = r_pos * r_pos;
                    let prox = (-0.5 * (dist / mult.shift_tol_ppm.max(0.01)).powi(2)).exp();
                    let merit = 0.70 * r2_pos + 0.30 * prox;
                    if merit > best_merit {
                        best_merit = merit;
                        best_j = Some(j);
                    }
                }
            }

            if let Some(j) = best_j {
                let r_pos = bucket_corr[j].max(0.0);
                let r2_pos = r_pos * r_pos;
                if r2_pos >= 0.50 {
                    matched_count += 1;
                }
                h_weighted_r2 += h * r2_pos;
                let dist = (buckets[j].center_ppm - mult.center_ppm).abs();
                shift_ll += (-0.5 * (dist / mult.shift_tol_ppm.max(0.015)).powi(2)).exp();
                obs_covs.push(bucket_cov[j].max(0.0));
                peak_pairs.push((mult.center_ppm, buckets[j].center_ppm, r2_pos));
            } else {
                obs_covs.push(0.0);
                peak_pairs.push((mult.center_ppm, mult.center_ppm, 0.0));
            }
        }

        let n_obs = obs_mults.len();
        let coverage = (matched_count as f64) / (n_obs as f64);
        let mean_r2 = if h_total > 0.0 {
            h_weighted_r2 / h_total
        } else {
            0.0
        };
        let shift_prox = shift_ll / (n_obs as f64);

        let stoich_cos = if n_obs >= 2 {
            let dot: f64 = obs_covs
                .iter()
                .zip(theo_hs.iter())
                .map(|(&c, &h)| c * h)
                .sum();
            let nc = obs_covs.iter().map(|&c| c * c).sum::<f64>().sqrt();
            let nh = theo_hs.iter().map(|&h| h * h).sum::<f64>().sqrt();
            if nc > 1e-18 && nh > 1e-18 {
                (dot / (nc * nh)).clamp(0.0, 1.0)
            } else {
                0.0
            }
        } else {
            1.0
        };

        // Multi-spin specificity bonus when >= 2 intramolecular multiplets co-vary strongly
        let multi_spin_bonus = if n_obs >= 2 && matched_count == n_obs && mean_r2 >= 0.75 {
            0.06
        } else {
            0.0
        };

        let raw_score = 0.42 * mean_r2
            + 0.28 * coverage
            + 0.16 * shift_prox
            + 0.14 * stoich_cos
            + multi_spin_bonus;
        matches.push(MetaboliteMatchScore {
            name: entry.name.to_string(),
            bmrb_id: entry.bmrb_id.to_string(),
            formula: entry.formula.to_string(),
            score: raw_score.clamp(0.0, 1.0),
            observable_multiplets: n_obs,
            matched_multiplets: matched_count,
            mean_stocsy_r2: mean_r2,
            stoichiometry_cosine: stoich_cos,
            matched_peaks: peak_pairs,
        });
    }

    matches.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.name.cmp(&b.name))
    });
    matches
}

/// Compute Cloarec et al. (*Anal. Chem.* 2005, 77, 1282–1289) `STOCSY` from `driver_bucket_idx`,
/// including bucket-level covariance/$r$/$r^2$, optional full-resolution spectral trace covariance/$r^2$,
/// and `BMRB` metabolite spin-system matching.
pub fn compute_stocsy(
    driver_bucket_idx: usize,
    buckets: &[SpectralBucket],
    normalized_bucket_matrix: &[Vec<f64>],
    aligned_spectra: Option<(&[Vec<f64>], &[f64])>,
    exclusion_zones: &[ExclusionZonePpm],
) -> Option<StocsyResult> {
    let n = normalized_bucket_matrix.len();
    let p = buckets.len();
    if n < 3 || driver_bucket_idx >= p {
        return None;
    }

    let driver: Vec<f64> = normalized_bucket_matrix
        .iter()
        .map(|r| r[driver_bucket_idx])
        .collect();
    let nf = n as f64;
    let dof = (n - 1) as f64;
    let d_mean = driver.iter().sum::<f64>() / nf;
    let d_ss: f64 = driver.iter().map(|&v| (v - d_mean) * (v - d_mean)).sum();
    let d_sd = (d_ss / dof).sqrt();

    let mut bucket_cov = vec![0.0; p];
    let mut bucket_corr = vec![0.0; p];
    let mut bucket_r2 = vec![0.0; p];

    for j in 0..p {
        let xj_mean = normalized_bucket_matrix.iter().map(|r| r[j]).sum::<f64>() / nf;
        let mut cov_num = 0.0;
        let mut xj_ss = 0.0;
        for i in 0..n {
            let dd = driver[i] - d_mean;
            let dx = normalized_bucket_matrix[i][j] - xj_mean;
            cov_num += dd * dx;
            xj_ss += dx * dx;
        }
        let cov = cov_num / dof;
        let xj_sd = (xj_ss / dof).sqrt();
        let r = if d_sd > 1e-15 && xj_sd > 1e-15 {
            (cov / (d_sd * xj_sd)).clamp(-1.0, 1.0)
        } else if j == driver_bucket_idx {
            1.0
        } else {
            0.0
        };
        bucket_cov[j] = cov;
        bucket_corr[j] = r;
        bucket_r2[j] = r * r;
    }

    // Optional full-resolution spectral STOCSY trace (normalized by dilution_factors)
    let (spectral_covariance, spectral_r2) = if let Some((specs, dil_factors)) = aligned_spectra {
        let m = specs.first().map(|r| r.len()).unwrap_or(0);
        if specs.len() == n && m > 0 {
            let mut s_cov = vec![0.0; m];
            let mut s_r2 = vec![0.0; m];
            for k in 0..m {
                let mut sm_mean = 0.0;
                for (i, sp_row) in specs.iter().enumerate().take(n) {
                    let df = dil_factors.get(i).copied().unwrap_or(1.0).max(1e-15);
                    sm_mean += sp_row[k] / df;
                }
                sm_mean /= nf;

                let mut cov_num = 0.0;
                let mut sm_ss = 0.0;
                for (i, sp_row) in specs.iter().enumerate().take(n) {
                    let df = dil_factors.get(i).copied().unwrap_or(1.0).max(1e-15);
                    let ds = (sp_row[k] / df) - sm_mean;
                    let dd = driver[i] - d_mean;
                    cov_num += dd * ds;
                    sm_ss += ds * ds;
                }
                let cov = cov_num / dof;
                let sm_sd = (sm_ss / dof).sqrt();
                let r = if d_sd > 1e-15 && sm_sd > 1e-15 {
                    (cov / (d_sd * sm_sd)).clamp(-1.0, 1.0)
                } else {
                    0.0
                };
                s_cov[k] = cov;
                s_r2[k] = r * r;
            }
            (s_cov, s_r2)
        } else {
            (Vec::new(), Vec::new())
        }
    } else {
        (Vec::new(), Vec::new())
    };

    let metabolite_matches = match_stocsy_to_bmrb_library(
        driver_bucket_idx,
        buckets,
        &bucket_cov,
        &bucket_corr,
        exclusion_zones,
    );

    Some(StocsyResult {
        driver_bucket_index: driver_bucket_idx,
        driver_ppm: buckets[driver_bucket_idx].center_ppm,
        bucket_covariance: bucket_cov,
        bucket_correlation: bucket_corr,
        bucket_r2,
        spectral_covariance,
        spectral_r2,
        metabolite_matches,
    })
}

/// Convenience wrapper locating the bucket closest to `driver_ppm` and running `compute_stocsy`.
pub fn compute_stocsy_at_ppm(
    driver_ppm: f64,
    study: &MetabolomicsStudyResult,
) -> Option<StocsyResult> {
    let best_idx = study
        .buckets
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            (a.center_ppm - driver_ppm)
                .abs()
                .total_cmp(&(b.center_ppm - driver_ppm).abs())
        })
        .map(|(i, _)| i)?;
    compute_stocsy(
        best_idx,
        &study.buckets,
        &study.normalized_bucket_matrix,
        Some((&study.aligned_spectra, &study.dilution_factors)),
        &study.config.mask_preset.zones(),
    )
}

// ============================================================================
// 12. Volcano + S-Plot + VIP Biomarker Ranking & Full Pipeline Runner
// ============================================================================

/// Compute univariate Welch two-sample $t$-tests, `BH-FDR` $q$-values, $\log_2$ fold changes, and
/// combine with `OPLS-DA` `S-plot` + `VIP` + `STOCSY`-assisted `BMRB` annotation.
pub fn rank_biomarkers(
    buckets: &[SpectralBucket],
    normalized_bucket_matrix: &[Vec<f64>],
    opls_da: Option<&OplsDaResult>,
    pls_da: Option<&PlsDaResult>,
    exclusion_zones: &[ExclusionZonePpm],
) -> Vec<BiomarkerCandidate> {
    let p = buckets.len();
    if p == 0 || normalized_bucket_matrix.is_empty() {
        return Vec::new();
    }

    let (active_idx, y_bin): (Vec<usize>, Vec<f64>) = if let Some(op) = opls_da {
        (op.active_sample_indices.clone(), op.y_binary.clone())
    } else if let Some(pl) = pls_da {
        (pl.active_sample_indices.clone(), pl.y_binary.clone())
    } else {
        let idx: Vec<usize> = (0..normalized_bucket_matrix.len()).collect();
        let y = vec![0.0; idx.len()];
        (idx, y)
    };

    let ctrl_rows: Vec<usize> = active_idx
        .iter()
        .zip(y_bin.iter())
        .filter_map(|(&ri, &y)| (y < 0.5).then_some(ri))
        .collect();
    let case_rows: Vec<usize> = active_idx
        .iter()
        .zip(y_bin.iter())
        .filter_map(|(&ri, &y)| (y >= 0.5).then_some(ri))
        .collect();

    // Small positive concentration floor for finite log2 fold-change
    let mut pos_vals: Vec<f64> = normalized_bucket_matrix
        .iter()
        .flat_map(|r| r.iter().copied())
        .filter(|&v| v > 1e-12)
        .collect();
    let fc_eps = if pos_vals.is_empty() {
        1e-9
    } else {
        1e-6 * median_in_place(&mut pos_vals)
    };

    let mut raw_p_vals = vec![1.0; p];
    let mut stats_cache = Vec::with_capacity(p);

    for j in 0..p {
        if ctrl_rows.len() >= 2 && case_rows.len() >= 2 {
            let n0 = ctrl_rows.len() as f64;
            let n1 = case_rows.len() as f64;
            let m0 = ctrl_rows
                .iter()
                .map(|&i| normalized_bucket_matrix[i][j])
                .sum::<f64>()
                / n0;
            let m1 = case_rows
                .iter()
                .map(|&i| normalized_bucket_matrix[i][j])
                .sum::<f64>()
                / n1;
            let v0 = ctrl_rows
                .iter()
                .map(|&i| {
                    let d = normalized_bucket_matrix[i][j] - m0;
                    d * d
                })
                .sum::<f64>()
                / (n0 - 1.0);
            let v1 = case_rows
                .iter()
                .map(|&i| {
                    let d = normalized_bucket_matrix[i][j] - m1;
                    d * d
                })
                .sum::<f64>()
                / (n1 - 1.0);
            let se0 = v0 / n0;
            let se1 = v1 / n1;
            let se_sum = se0 + se1;
            let (t_stat, p_val) = if se_sum > 1e-30 {
                let t = (m1 - m0) / se_sum.sqrt();
                let dof_denom =
                    (se0 * se0) / (n0 - 1.0).max(1.0) + (se1 * se1) / (n1 - 1.0).max(1.0);
                let dof = if dof_denom > 1e-30 {
                    (se_sum * se_sum) / dof_denom
                } else {
                    n0 + n1 - 2.0
                };
                (t, student_t_two_sided_p_value(t, dof))
            } else {
                (0.0, 1.0)
            };
            let fc = m1.max(fc_eps) / m0.max(fc_eps);
            let log2_fc = fc.log2();
            raw_p_vals[j] = p_val;
            stats_cache.push((m0, m1, fc, log2_fc, t_stat));
        } else {
            let m_all = normalized_bucket_matrix.iter().map(|r| r[j]).sum::<f64>()
                / (normalized_bucket_matrix.len() as f64);
            stats_cache.push((m_all, m_all, 1.0, 0.0, 0.0));
        }
    }

    let fdr_q = benjamini_hochberg_fdr(&raw_p_vals);

    let mut candidates = Vec::with_capacity(p);
    for j in 0..p {
        let b = &buckets[j];
        let (m0, m1, fc, log2_fc, t_stat) = stats_cache[j];
        let vip = opls_da
            .map(|o| o.vip_predictive[j])
            .or_else(|| pls_da.map(|pl| pl.vip_scores[j]))
            .unwrap_or(0.0);
        let s_cov = opls_da.map(|o| o.s_plot_cov[j]).unwrap_or(0.0);
        let s_corr = opls_da.map(|o| o.s_plot_corr[j]).unwrap_or(0.0);
        let qj = fdr_q[j].max(1e-300);
        let neg_log_q = (-qj.log10()).clamp(0.0, 8.0);
        let composite = if opls_da.is_some() {
            vip * s_corr.abs() * (1.0 + neg_log_q / 6.0)
        } else {
            neg_log_q * log2_fc.abs()
        };

        candidates.push(BiomarkerCandidate {
            rank: 0,
            bucket_index: j,
            center_ppm: b.center_ppm,
            low_ppm: b.low_ppm,
            high_ppm: b.high_ppm,
            control_mean: m0,
            case_mean: m1,
            fold_change: fc,
            log2_fold_change: log2_fc,
            welch_t: t_stat,
            p_value: raw_p_vals[j],
            fdr_q: fdr_q[j],
            vip,
            s_plot_cov: s_cov,
            s_plot_corr: s_corr,
            composite_score: composite,
            putative_metabolite: None,
            putative_bmrb_id: None,
        });
    }

    candidates.sort_by(|a, b| {
        b.composite_score
            .total_cmp(&a.composite_score)
            .then_with(|| a.p_value.total_cmp(&b.p_value))
            .then_with(|| a.bucket_index.cmp(&b.bucket_index))
    });

    // Annotate top 40 candidate biomarkers (and any VIP >= 1.0 bucket) via STOCSY + BMRB library!
    for (pos, cand) in candidates.iter_mut().enumerate() {
        cand.rank = pos + 1;
        if pos < 40 || cand.vip >= 1.0 {
            if let Some(st) = compute_stocsy(
                cand.bucket_index,
                buckets,
                normalized_bucket_matrix,
                None,
                exclusion_zones,
            ) {
                if let Some(top_m) = st.metabolite_matches.first() {
                    if top_m.score >= 0.45 {
                        cand.putative_metabolite = Some(top_m.name.clone());
                        cand.putative_bmrb_id = Some(top_m.bmrb_id.clone());
                    }
                }
            }
        }
    }

    candidates
}

/// Run the complete untargeted 1D NMR metabolomics, chemometrics & `STOCSY` biomarker pipeline.
pub fn run_metabolomics_pipeline(
    samples: &[CohortSample1D],
    config: &MetabolomicsPipelineConfig,
) -> MetabolomicsStudyResult {
    let sample_ids: Vec<String> = samples.iter().map(|s| s.id.clone()).collect();
    let sample_groups: Vec<String> = samples.iter().map(|s| s.group.clone()).collect();

    let (common_ppm_grid, masked_grid_points, mut aligned_spectra, is_areas, obs_mhz) =
        preprocess_cohort_1d(samples, config);

    align_cohort_interval_fft(
        &mut aligned_spectra,
        &common_ppm_grid,
        &masked_grid_points,
        &config.alignment,
    );

    let (buckets, raw_bucket_matrix) = bucket_cohort_spectra(
        &aligned_spectra,
        &common_ppm_grid,
        &masked_grid_points,
        obs_mhz,
        &config.bucketing,
    );

    let (normalized_bucket_matrix, dilution_factors) =
        normalize_bucket_matrix(&raw_bucket_matrix, &is_areas, config.normalization);

    let scaled_bucket_matrix = scale_bucket_matrix(&normalized_bucket_matrix, config.scaling);

    let pca = fit_pca(&scaled_bucket_matrix, &sample_groups, config.pca_components);

    let pls_da = fit_pls_da(
        &normalized_bucket_matrix,
        &sample_groups,
        config.scaling,
        config.pls_components,
        config.cv_folds,
        config.control_group.as_deref(),
        config.case_group.as_deref(),
    );

    let opls_da = fit_opls_da(
        &normalized_bucket_matrix,
        &sample_groups,
        config.scaling,
        config.opls_orthogonal_components,
        config.cv_folds,
        config.n_permutations,
        config.rng_seed,
        config.control_group.as_deref(),
        config.case_group.as_deref(),
    );

    let zones = config.mask_preset.zones();
    let biomarkers = rank_biomarkers(
        &buckets,
        &normalized_bucket_matrix,
        opls_da.as_ref(),
        pls_da.as_ref(),
        &zones,
    );

    let top_biomarker_stocsy = biomarkers.first().and_then(|top| {
        compute_stocsy(
            top.bucket_index,
            &buckets,
            &normalized_bucket_matrix,
            Some((&aligned_spectra, &dilution_factors)),
            &zones,
        )
    });

    MetabolomicsStudyResult {
        config: config.clone(),
        sample_ids,
        sample_groups,
        common_ppm_grid,
        masked_grid_points,
        aligned_spectra,
        buckets,
        raw_bucket_matrix,
        normalized_bucket_matrix,
        scaled_bucket_matrix,
        dilution_factors,
        pca,
        pls_da,
        opls_da,
        biomarkers,
        top_biomarker_stocsy,
    }
}

// ============================================================================
// 13. CSV & JSON Study Exporters
// ============================================================================

fn csv_escape(s: &str) -> String {
    if s.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

/// Export the normalized `N x P` bucket table as RFC-4180 CSV (`sample_id,group,dilution_factor,B0_ppm,...`).
pub fn export_bucket_table_csv(study: &MetabolomicsStudyResult) -> String {
    let mut out = String::from("sample_id,group,dilution_factor");
    for b in &study.buckets {
        out.push_str(&format!(",B{:04}_{:.4}ppm", b.index, b.center_ppm));
    }
    out.push('\n');

    for (i, row) in study.normalized_bucket_matrix.iter().enumerate() {
        out.push_str(&format!(
            "{},{},{:.6}",
            csv_escape(&study.sample_ids[i]),
            csv_escape(&study.sample_groups[i]),
            study.dilution_factors.get(i).copied().unwrap_or(1.0)
        ));
        for &v in row {
            out.push_str(&format!(",{:.6e}", v));
        }
        out.push('\n');
    }
    out
}

/// Export the ranked biomarker discovery table (`VIP`, `S-plot`, `Welch t`, `BH-FDR`, `BMRB` match) as CSV.
pub fn export_biomarker_ranking_csv(study: &MetabolomicsStudyResult) -> String {
    let mut out = String::from(
        "rank,bucket_index,center_ppm,low_ppm,high_ppm,control_mean,case_mean,fold_change,log2_fc,welch_t,p_value,fdr_q,vip,s_plot_cov,s_plot_corr,composite_score,putative_metabolite,bmrb_id\n",
    );
    for c in &study.biomarkers {
        out.push_str(&format!(
            "{},{},{:.4},{:.4},{:.4},{:.6e},{:.6e},{:.4},{:.4},{:.4},{:.4e},{:.4e},{:.4},{:.6e},{:.4},{:.4},{},{}\n",
            c.rank,
            c.bucket_index,
            c.center_ppm,
            c.low_ppm,
            c.high_ppm,
            c.control_mean,
            c.case_mean,
            c.fold_change,
            c.log2_fold_change,
            c.welch_t,
            c.p_value,
            c.fdr_q,
            c.vip,
            c.s_plot_cov,
            c.s_plot_corr,
            c.composite_score,
            csv_escape(c.putative_metabolite.as_deref().unwrap_or("")),
            csv_escape(c.putative_bmrb_id.as_deref().unwrap_or("")),
        ));
    }
    out
}

/// Compact JSON summary report (PCA variance, `OPLS-DA` $R^2X/R^2Y/Q^2$, permutation $p$-value,
/// top biomarkers, and top `STOCSY` metabolite hits) suitable for CLI, REST, or GUI export.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MetabolomicsSummaryReportJson {
    pub n_samples: usize,
    pub n_buckets: usize,
    pub pca_explained_variance_ratio: Vec<f64>,
    pub pca_hotelling_t2_limit_95: f64,
    pub pca_q_residual_limit_95: f64,
    pub opls_r2_x_predictive: Option<f64>,
    pub opls_r2_x_orthogonal: Option<f64>,
    pub opls_r2_y: Option<f64>,
    pub opls_q2: Option<f64>,
    pub opls_permutation_p_q2: Option<f64>,
    pub opls_is_validated: Option<bool>,
    pub top_biomarkers: Vec<BiomarkerCandidate>,
}

/// Serialize a structured JSON summary of a `MetabolomicsStudyResult`.
pub fn export_study_summary_json(
    study: &MetabolomicsStudyResult,
    top_k_biomarkers: usize,
) -> Result<String, serde_json::Error> {
    let summary = MetabolomicsSummaryReportJson {
        n_samples: study.sample_ids.len(),
        n_buckets: study.buckets.len(),
        pca_explained_variance_ratio: study.pca.explained_variance_ratio.clone(),
        pca_hotelling_t2_limit_95: study.pca.hotelling_t2_limit_95,
        pca_q_residual_limit_95: study.pca.q_residual_limit_95,
        opls_r2_x_predictive: study.opls_da.as_ref().map(|o| o.r2_x_predictive),
        opls_r2_x_orthogonal: study.opls_da.as_ref().map(|o| o.r2_x_orthogonal),
        opls_r2_y: study.opls_da.as_ref().map(|o| o.r2_y),
        opls_q2: study.opls_da.as_ref().map(|o| o.q2),
        opls_permutation_p_q2: study.opls_da.as_ref().map(|o| o.permutation.p_value_q2),
        opls_is_validated: study.opls_da.as_ref().map(|o| o.permutation.is_validated),
        top_biomarkers: study
            .biomarkers
            .iter()
            .take(top_k_biomarkers.max(1))
            .cloned()
            .collect(),
    };
    serde_json::to_string_pretty(&summary)
}
