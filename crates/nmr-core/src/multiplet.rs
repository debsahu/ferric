//! Analytical 1D NMR integration, recursive first-order & second-order leaning-corrected
//! multiplet `J`-coupling tree analysis, publication journal report formatting/parsing,
//! internal-standard quantitative NMR (`qNMR`) purity & molar concentration metrology,
//! and bidirectional `Octet` 2D chemical-structure atom-assignment tables.
//!
//! # Clean-Room Provenance & Literature Citations
//! Original clean-room implementation derived strictly from peer-reviewed literature:
//! - Hoye, T. R.; Hanson, P. R.; Vyvyan, J. R. "A Practical Guide to First-Order Multiplet
//!   Analysis in 1H NMR Spectroscopy", *J. Org. Chem.* **1994**, 59 (15), 4096–4103
//!   (recursive sub-doublet subtraction trees, coalesced multiplet topologies, and `dt` vs `td`
//!   descending-`J` nomenclature).
//! - Hoye, T. R.; Zhao, H. "A Method for Easily Determining Coupling Constant Values: An
//!   Addendum to 'A Practical Guide to First-Order Multiplet Analysis in 1H NMR Spectroscopy'",
//!   *J. Org. Chem.* **2002**, 67 (12), 4014–4016 (recursive intensity peeling & closed-form
//!   second-order AB spin-system analysis).
//! - Pople, J. A.; Schneider, W. G.; Bernstein, H. J. *High-Resolution Nuclear Magnetic
//!   Resonance*, McGraw-Hill, New York, **1959**, Ch. 6 (secular determinant & `1 +- sin(2 theta)`
//!   roofing intensity ratios for coupled spin-1/2 systems as a function of `Delta nu / J`).
//! - Pauli, G. F.; Chen, S.-N.; Simmler, C.; Lankin, D. C.; Gödecke, T.; Jaki, B. U.;
//!   Friesen, J. B.; McAlpine, J. B.; Napolitano, J. G. "Importance of Purity Evaluation and
//!   the Potential of Quantitative 1H NMR as a Purity Assay", *J. Med. Chem.* **2014**, 57 (22),
//!   9220–9231 (internal-standard `qNMR` mass-fraction purity, molarity, longitudinal `T1`
//!   relaxation / `5 * T1` recycle-delay criterion, and primary calibrants).
//! - Malz, F.; Jancke, H. "Validation of quantitative NMR", *J. Pharm. Biomed. Anal.* **2005**,
//!   38 (5), 813–823 (GUM metrological uncertainty propagation & replicate RSD analysis).

use crate::octet_analyt::jcamp_spec1d::Spectrum1d;
use crate::octet_analyt::nmr_predict::{
    perceive_topological_env, predict_nmr, NmrNucleus as OctetNmrNucleus,
};
use crate::octet_analyt::AnalytMol;
use crate::octet_mol::{parse_smiles, read_mol, ChemError, Molecule, SdfRecord};
use crate::octet_render::styles::RgbaColor;
use crate::octet_render::RenderScene;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

// ============================================================================
// Error Type
// ============================================================================

/// Error returned by 1D integration, multiplet tree analysis, journal parsing, or `qNMR` routines.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MultipletError {
    /// Spectrum arrays are empty, mismatched in length, or contain non-finite values.
    EmptyOrInvalidSpectrum(String),
    /// Invalid configuration or metrological input parameter.
    InvalidParameter(String),
    /// Malformed journal NMR report string or assignment tag.
    ParseError(String),
}

impl fmt::Display for MultipletError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyOrInvalidSpectrum(msg) => write!(f, "invalid spectrum: {msg}"),
            Self::InvalidParameter(msg) => write!(f, "invalid parameter: {msg}"),
            Self::ParseError(msg) => write!(f, "parse error: {msg}"),
        }
    }
}

impl std::error::Error for MultipletError {}

// ============================================================================
// Section 1: 1D Region Integration Engine (Slope/Bias Baseline + Noise RMS)
// ============================================================================

/// Per-region linear baseline model subtracted prior to numerical integration.
///
/// Over an integral window `[\delta_low, \delta_high]` (`\delta_low <= \delta_high`), the local
/// linear baseline is parameterized at `\delta_low` as:
/// `B(\delta) = bias + slope * (\delta - \delta_low)`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum BaselineMode {
    /// Raw trapezoidal integration with zero baseline subtraction (`bias = 0, slope = 0`).
    None,
    /// User-specified explicit `bias` (intensity offset at `\delta_low`) and `slope`
    /// (intensity change per ppm across `[\delta_low, \delta_high]`).
    Explicit { bias: f64, slope: f64 },
    /// Ordinary least-squares linear baseline fitted automatically to the boundary wing samples
    /// at both edges of the integral region.
    AutoLinearEdges {
        /// Fraction of region width on each wing used to estimate `bias` and `slope` (clamped to `0.02..=0.35`).
        edge_fraction: f64,
        /// Minimum sample count taken from each wing when available.
        min_edge_points: usize,
    },
}

impl Default for BaselineMode {
    fn default() -> Self {
        Self::AutoLinearEdges {
            edge_fraction: 0.12,
            min_edge_points: 3,
        }
    }
}

/// Reference normalization rule mapping raw spectral integrals to relative spin counts (e.g. `1H`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum NormalizationMode {
    /// Keep scale factor `1.0` (`norm_integral == abs_integral`).
    #[default]
    None,
    /// Normalize so that `regions[region_index].norm_integral == target_protons`
    /// (for example `1.00`, `2.00`, or `3.00` H).
    ReferenceRegion {
        region_index: usize,
        target_protons: f64,
    },
    /// Normalize so that the sum of positive region integrals equals `total_protons`.
    TotalProtons { total_protons: f64 },
    /// Multiply every absolute integral by an explicit calibration `factor`.
    ExplicitScale { factor: f64 },
}

/// Optional deconvolved line-shape component (e.g. from global Voigt / Lorentz-Gauss fitting)
/// that can be aggregated alongside direct numerical quadrature inside each integral region.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VoigtPeakComponent {
    pub center_ppm: f64,
    pub fwhm_hz: f64,
    pub area: f64,
    pub area_sigma: f64,
}

/// Specification of a single 1D NMR integration window.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IntegralRegionSpec {
    /// First boundary in ppm (may be downfield high-ppm or upfield low-ppm; ordered internally).
    pub ppm_start: f64,
    /// Second boundary in ppm.
    pub ppm_end: f64,
    /// Local baseline correction mode (`None`, `Explicit { bias, slope }`, or `AutoLinearEdges`).
    pub baseline: BaselineMode,
    /// Optional human-readable label (e.g. `"Ar-H"`, `"OCH3"`, `"Internal Std"`).
    pub label: Option<String>,
}

impl IntegralRegionSpec {
    /// Create an integral region with default automatic edge-linear slope/bias correction.
    pub fn new(ppm_start: f64, ppm_end: f64) -> Self {
        Self {
            ppm_start,
            ppm_end,
            baseline: BaselineMode::default(),
            label: None,
        }
    }

    /// Override the per-region baseline correction mode.
    pub fn with_baseline(mut self, baseline: BaselineMode) -> Self {
        self.baseline = baseline;
        self
    }

    /// Set explicit `bias` (at `min(ppm_start, ppm_end)`) and `slope` (intensity per ppm).
    pub fn with_explicit_bias_slope(mut self, bias: f64, slope: f64) -> Self {
        self.baseline = BaselineMode::Explicit { bias, slope };
        self
    }

    /// Attach a display label to the region.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}

/// Evaluated 1D integral region with baseline parameters, cumulative integral trace,
/// normalized proton count, and propagated spectral noise uncertainties.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IntegratedRegion {
    /// Zero-based index in `IntegrationReport::regions`.
    pub index: usize,
    /// Upfield (minimum) boundary in ppm.
    pub ppm_low: f64,
    /// Downfield (maximum) boundary in ppm.
    pub ppm_high: f64,
    /// Optional region label.
    pub label: Option<String>,
    /// Uncorrected trapezoidal integral `\int_{\delta_low}^{\delta_high} y(\delta) d\delta`.
    pub raw_integral: f64,
    /// Baseline-corrected integral after subtracting `bias + slope * (\delta - \delta_low)`.
    pub abs_integral: f64,
    /// Exact propagated `1 sigma` standard uncertainty on `abs_integral` from `noise_rms`
    /// (including variance inflation from edge OLS baseline estimation when `AutoLinearEdges` is active).
    pub abs_sigma: f64,
    /// Normalized integral value (e.g. relative `1H` count when `ReferenceRegion` or `TotalProtons` is active).
    pub norm_integral: f64,
    /// Propagated `1 sigma` uncertainty on `norm_integral` (propagating both target and reference-region uncertainties).
    pub norm_sigma: f64,
    /// Direct scaled uncertainty `|scale_factor| * abs_sigma` without reference-ratio covariance term.
    pub intrinsic_norm_sigma: f64,
    /// Nearest non-negative integer spin count (`round(norm_integral)`).
    pub nearest_integer_h: usize,
    /// Subtracted baseline bias at `ppm_low`.
    pub applied_bias: f64,
    /// Subtracted baseline slope across `[ppm_low, ppm_high]` (intensity per ppm).
    pub applied_slope: f64,
    /// Signal-to-noise ratio of the integrated region (`|abs_integral| / abs_sigma`).
    pub snr_integral: f64,
    /// Optional sum of deconvolved `VoigtPeakComponent` areas whose centers fall inside `[ppm_low, ppm_high]`.
    pub deconvolved_area: Option<f64>,
    /// Normalized deconvolved integral when `VoigtPeakComponent`s are supplied.
    pub deconvolved_norm_integral: Option<f64>,
    /// Propagated uncertainty on `deconvolved_area`.
    pub deconvolved_sigma: Option<f64>,
    /// Downfield-to-upfield chemical shift coordinates (`ppm_high -> ppm_low`) of the cumulative integral curve.
    pub cumulative_ppm: Vec<f64>,
    /// Running cumulative integral values aligned with `cumulative_ppm` (scaled by `scale_factor`).
    pub cumulative_integral: Vec<f64>,
    /// Unscaled (`scale_factor = 1`) running cumulative integral values for lossless re-referencing.
    raw_cumulative_corr: Vec<f64>,
}

/// Complete multi-region 1D integration report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IntegrationReport {
    /// Spectral intensity RMS noise (`1 sigma` per sample point) used for uncertainty propagation.
    pub noise_rms: f64,
    /// Multiplicative factor mapping `abs_integral` -> `norm_integral`.
    pub scale_factor: f64,
    /// Active normalization mode.
    pub normalization: NormalizationMode,
    /// Integrated regions in caller-supplied order.
    pub regions: Vec<IntegratedRegion>,
}

impl IntegrationReport {
    /// Re-normalize all regions in place (e.g. when the user interactively changes the reference
    /// multiplet from `1.00 H` to `2.00 H` or `3.00 H`) without re-integrating the raw trace.
    pub fn renormalize(&mut self, mode: NormalizationMode) -> Result<(), MultipletError> {
        apply_normalization(&mut self.regions, mode).map(|scale| {
            self.scale_factor = scale;
            self.normalization = mode;
        })
    }
}

/// Estimate per-point spectral baseline RMS noise (`sigma_y`) from either an explicit signal-free
/// chemical shift window `Some((ppm_a, ppm_b))` or automatic 16-segment linear-detrended MAD/RMS
/// baseline selection.
pub fn estimate_spectral_noise_rms(
    ppm: &[f64],
    intensity: &[f64],
    noise_window: Option<(f64, f64)>,
) -> Result<f64, MultipletError> {
    validate_trace(ppm, intensity)?;
    let sorted = sorted_ascending_trace(ppm, intensity);

    if let Some((a, b)) = noise_window {
        let lo = a.min(b);
        let hi = a.max(b);
        let slice: Vec<(f64, f64)> = sorted
            .into_iter()
            .filter(|&(x, _)| x >= lo && x <= hi)
            .collect();
        if slice.len() < 3 {
            return Err(MultipletError::InvalidParameter(format!(
                "noise window [{lo}, {hi}] ppm contains fewer than 3 sample points"
            )));
        }
        return Ok(detrended_rms(&slice));
    }

    let n = sorted.len();
    if n < 8 {
        return Ok(detrended_rms(&sorted));
    }

    let n_seg = 16usize.min(n / 4).max(2);
    let seg_len = n / n_seg;
    let mut seg_rms: Vec<f64> = Vec::with_capacity(n_seg);
    for s in 0..n_seg {
        let start = s * seg_len;
        let end = if s + 1 == n_seg { n } else { (s + 1) * seg_len };
        if end - start >= 3 {
            seg_rms.push(detrended_rms(&sorted[start..end]));
        }
    }
    seg_rms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if seg_rms.is_empty() {
        return Ok(0.0);
    }
    // Average the quietest quartile of segments to avoid peaks while remaining unbiased on Gaussian noise.
    let take_k = (seg_rms.len() / 4).max(1);
    let mean_quiet = seg_rms[..take_k].iter().sum::<f64>() / (take_k as f64);
    Ok(mean_quiet)
}

/// Integrate a list of 1D spectral regions with per-region slope/bias linear baseline correction,
/// exact quadrature + baseline-fit noise propagation, optional Voigt component summation, and
/// reference normalization.
pub fn integrate_regions(
    ppm: &[f64],
    intensity: &[f64],
    specs: &[IntegralRegionSpec],
    normalization: NormalizationMode,
    noise_rms: Option<f64>,
    voigt_components: Option<&[VoigtPeakComponent]>,
) -> Result<IntegrationReport, MultipletError> {
    validate_trace(ppm, intensity)?;
    let sigma_y = match noise_rms {
        Some(s) => {
            if !s.is_finite() || s < 0.0 {
                return Err(MultipletError::InvalidParameter(
                    "noise_rms must be finite and non-negative".to_string(),
                ));
            }
            s
        }
        None => estimate_spectral_noise_rms(ppm, intensity, None)?,
    };

    let sorted = sorted_ascending_trace(ppm, intensity);
    let mut regions = Vec::with_capacity(specs.len());

    for (idx, spec) in specs.iter().enumerate() {
        if !spec.ppm_start.is_finite() || !spec.ppm_end.is_finite() {
            return Err(MultipletError::InvalidParameter(
                "region boundaries must be finite ppm values".to_string(),
            ));
        }
        let lo = spec.ppm_start.min(spec.ppm_end);
        let hi = spec.ppm_start.max(spec.ppm_end);
        if (hi - lo) <= 1e-12 {
            return Err(MultipletError::InvalidParameter(format!(
                "integral region {idx} has zero width ({lo}..{hi} ppm)"
            )));
        }

        let reg = integrate_single_region(&sorted, idx, lo, hi, spec, sigma_y, voigt_components)?;
        regions.push(reg);
    }

    let scale_factor = apply_normalization(&mut regions, normalization)?;

    Ok(IntegrationReport {
        noise_rms: sigma_y,
        scale_factor,
        normalization,
        regions,
    })
}

/// Convenience wrapper integrating regions directly on an `octet_analyt` [`Spectrum1d`].
pub fn integrate_spectrum1d_regions(
    spec: &Spectrum1d,
    specs: &[IntegralRegionSpec],
    normalization: NormalizationMode,
    noise_rms: Option<f64>,
) -> Result<IntegrationReport, MultipletError> {
    let xs = spec.x_axis();
    integrate_regions(&xs, &spec.y, specs, normalization, noise_rms, None)
}

fn validate_trace(ppm: &[f64], intensity: &[f64]) -> Result<(), MultipletError> {
    if ppm.len() < 2 || ppm.len() != intensity.len() {
        return Err(MultipletError::EmptyOrInvalidSpectrum(format!(
            "expected matching ppm and intensity slices of length >= 2 (got {} and {})",
            ppm.len(),
            intensity.len()
        )));
    }
    if ppm.iter().any(|x| !x.is_finite()) || intensity.iter().any(|y| !y.is_finite()) {
        return Err(MultipletError::EmptyOrInvalidSpectrum(
            "spectrum contains NaN or infinite values".to_string(),
        ));
    }
    Ok(())
}

fn sorted_ascending_trace(ppm: &[f64], intensity: &[f64]) -> Vec<(f64, f64)> {
    let mut pairs: Vec<(f64, f64)> = ppm.iter().copied().zip(intensity.iter().copied()).collect();
    // Fast reverse check: standard NMR traces are already strictly descending.
    let strictly_desc = pairs.windows(2).all(|w| w[0].0 > w[1].0);
    if strictly_desc {
        pairs.reverse();
        return pairs;
    }
    let strictly_asc = pairs.windows(2).all(|w| w[0].0 < w[1].0);
    if !strictly_asc {
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    }
    pairs
}

fn detrended_rms(points: &[(f64, f64)]) -> f64 {
    let n = points.len();
    if n <= 2 {
        return 0.0;
    }
    let nf = n as f64;
    let mean_x = points.iter().map(|p| p.0).sum::<f64>() / nf;
    let mean_y = points.iter().map(|p| p.1).sum::<f64>() / nf;
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    for &(x, y) in points {
        let dx = x - mean_x;
        sxx += dx * dx;
        sxy += dx * (y - mean_y);
    }
    let slope = if sxx > 1e-30 { sxy / sxx } else { 0.0 };
    let rss: f64 = points
        .iter()
        .map(|&(x, y)| {
            let r = (y - mean_y) - slope * (x - mean_x);
            r * r
        })
        .sum();
    (rss / ((n - 2) as f64)).sqrt()
}

/// Interpolate `(y, left_idx, right_idx, alpha)` at `target_x` on an ascending `trace`,
/// where `y = (1 - alpha) * trace[left_idx].1 + alpha * trace[right_idx].1`.
fn interpolate_on_ascending(trace: &[(f64, f64)], target_x: f64) -> (f64, usize, usize, f64) {
    let n = trace.len();
    if target_x <= trace[0].0 {
        return (trace[0].1, 0, 0, 0.0);
    }
    if target_x >= trace[n - 1].0 {
        return (trace[n - 1].1, n - 1, n - 1, 0.0);
    }
    let pos = trace.partition_point(|&(x, _)| x < target_x);
    if pos == 0 {
        return (trace[0].1, 0, 0, 0.0);
    }
    let l = pos - 1;
    let r = pos;
    let span = trace[r].0 - trace[l].0;
    if span.abs() <= 1e-15 {
        (trace[l].1, l, l, 0.0)
    } else {
        let alpha = ((target_x - trace[l].0) / span).clamp(0.0, 1.0);
        let y = (1.0 - alpha) * trace[l].1 + alpha * trace[r].1;
        (y, l, r, alpha)
    }
}

fn integrate_single_region(
    trace: &[(f64, f64)],
    region_index: usize,
    ppm_low: f64,
    ppm_high: f64,
    spec: &IntegralRegionSpec,
    sigma_y: f64,
    voigt_components: Option<&[VoigtPeakComponent]>,
) -> Result<IntegratedRegion, MultipletError> {
    let trace_lo = trace[0].0;
    let trace_hi = trace[trace.len() - 1].0;
    let lo = ppm_low.max(trace_lo);
    let hi = ppm_high.min(trace_hi);
    if hi <= lo {
        return Err(MultipletError::InvalidParameter(format!(
            "region [{ppm_low}, {ppm_high}] ppm lies outside spectrum [{trace_lo}, {trace_hi}] ppm"
        )));
    }

    // Build local sub-grid `[lo, interior..., hi]` and track exact linear dependence on original
    // sample indices so `Var(I_corr) = sigma_y^2 * sum_k c_k^2` is exact!
    // Each local node `m` is `(x_m, y_m, idx_a, wt_a, idx_b, wt_b)`.
    let mut nodes: Vec<(f64, f64, usize, f64, usize, f64)> = Vec::new();
    let (y_lo, l0, r0, a0) = interpolate_on_ascending(trace, lo);
    nodes.push((lo, y_lo, l0, 1.0 - a0, r0, a0));

    let first_in = trace.partition_point(|&(x, _)| x <= lo + 1e-13);
    let last_in = trace.partition_point(|&(x, _)| x < hi - 1e-13);
    for (orig_idx, &(x, y)) in trace[first_in..last_in].iter().enumerate() {
        let k = first_in + orig_idx;
        nodes.push((x, y, k, 1.0, k, 0.0));
    }

    let (y_hi, l1, r1, a1) = interpolate_on_ascending(trace, hi);
    if (hi - nodes.last().map(|n| n.0).unwrap_or(lo)).abs() > 1e-13 {
        nodes.push((hi, y_hi, l1, 1.0 - a1, r1, a1));
    }

    let m = nodes.len();
    if m < 2 {
        return Err(MultipletError::InvalidParameter(format!(
            "region [{lo}, {hi}] ppm has insufficient sample coverage"
        )));
    }

    // Trapezoidal weights `w_m` on local nodes `0..m`
    let mut w_local = vec![0.0_f64; m];
    for i in 0..(m - 1) {
        let dx = nodes[i + 1].0 - nodes[i].0;
        w_local[i] += 0.5 * dx;
        w_local[i + 1] += 0.5 * dx;
    }

    let raw_integral: f64 = nodes
        .iter()
        .zip(w_local.iter())
        .map(|(nd, &w)| w * nd.1)
        .sum();
    let width = hi - lo;
    let mid = 0.5 * (lo + hi);

    // Baseline parameters `(bias at lo, slope)` and baseline integral linear weights `b_local`
    let mut b_local = vec![0.0_f64; m];
    let (applied_bias, applied_slope) = match spec.baseline {
        BaselineMode::None => (0.0, 0.0),
        BaselineMode::Explicit { bias, slope } => {
            if !bias.is_finite() || !slope.is_finite() {
                return Err(MultipletError::InvalidParameter(
                    "explicit baseline bias and slope must be finite".to_string(),
                ));
            }
            (bias, slope)
        }
        BaselineMode::AutoLinearEdges {
            edge_fraction,
            min_edge_points,
        } => {
            let frac = edge_fraction.clamp(0.02, 0.35);
            let min_pts = min_edge_points.max(1);
            let left_cut = lo + frac * width;
            let right_cut = hi - frac * width;

            let mut edge_indices: Vec<usize> = Vec::new();
            for (idx_n, nd) in nodes.iter().enumerate() {
                if nd.0 <= left_cut || nd.0 >= right_cut {
                    edge_indices.push(idx_n);
                }
            }
            // Ensure at least `min_pts` from each side when `m >= 2 * min_pts`
            if edge_indices.len() < 2 * min_pts {
                let k_each = min_pts.min(m / 2).max(1);
                let mut set = BTreeSet::new();
                for idx_n in 0..k_each {
                    set.insert(idx_n);
                }
                for idx_n in (m - k_each)..m {
                    set.insert(idx_n);
                }
                edge_indices = set.into_iter().collect();
            }

            let ne = edge_indices.len() as f64;
            let x_bar = edge_indices.iter().map(|&i| nodes[i].0).sum::<f64>() / ne;
            let y_bar = edge_indices.iter().map(|&i| nodes[i].1).sum::<f64>() / ne;
            let mut sxx = 0.0;
            let mut sxy = 0.0;
            for &i in &edge_indices {
                let ux = nodes[i].0 - x_bar;
                sxx += ux * ux;
                sxy += ux * (nodes[i].1 - y_bar);
            }
            let slope = if sxx > 1e-30 { sxy / sxx } else { 0.0 };
            let bias_at_lo = y_bar + slope * (lo - x_bar);

            // Exact linear functional weights for `I_base = width * (y_bar + slope * (mid - x_bar))`
            for &i in &edge_indices {
                let ux = nodes[i].0 - x_bar;
                let slope_coeff = if sxx > 1e-30 {
                    (mid - x_bar) * ux / sxx
                } else {
                    0.0
                };
                b_local[i] = width * ((1.0 / ne) + slope_coeff);
            }
            (bias_at_lo, slope)
        }
    };

    let baseline_integral = width * (applied_bias + 0.5 * applied_slope * width);
    let abs_integral = raw_integral - baseline_integral;

    // Accumulate exact sample-level linear combination coefficients `c_sample[orig_idx]`
    let mut sample_coeffs: BTreeMap<usize, f64> = BTreeMap::new();
    for (i, nd) in nodes.iter().enumerate() {
        let net_local = w_local[i] - b_local[i];
        if nd.3 != 0.0 {
            *sample_coeffs.entry(nd.2).or_insert(0.0) += net_local * nd.3;
        }
        if nd.5 != 0.0 {
            *sample_coeffs.entry(nd.4).or_insert(0.0) += net_local * nd.5;
        }
    }
    let sum_sq: f64 = sample_coeffs.values().map(|c| c * c).sum();
    let abs_sigma = sigma_y * sum_sq.sqrt();

    // Downfield-to-upfield (`hi -> lo`) cumulative baseline-corrected integral curve
    let mut cumulative_ppm = Vec::with_capacity(m);
    let mut raw_cumulative_corr = Vec::with_capacity(m);
    cumulative_ppm.push(nodes[m - 1].0);
    raw_cumulative_corr.push(0.0);
    let mut running = 0.0;
    for i in (0..(m - 1)).rev() {
        let x_hi = nodes[i + 1].0;
        let x_lo = nodes[i].0;
        let dx = x_hi - x_lo;
        let yc_hi = nodes[i + 1].1 - (applied_bias + applied_slope * (x_hi - lo));
        let yc_lo = nodes[i].1 - (applied_bias + applied_slope * (x_lo - lo));
        running += 0.5 * (yc_hi + yc_lo) * dx;
        cumulative_ppm.push(x_lo);
        raw_cumulative_corr.push(running);
    }

    let (deconvolved_area, deconvolved_sigma) = match voigt_components {
        Some(comps) => {
            let mut area_sum = 0.0;
            let mut var_sum = 0.0;
            for c in comps {
                if c.center_ppm >= lo && c.center_ppm <= hi {
                    area_sum += c.area;
                    var_sum += c.area_sigma * c.area_sigma;
                }
            }
            (Some(area_sum), Some(var_sum.sqrt()))
        }
        None => (None, None),
    };

    let snr_integral = if abs_sigma > 0.0 {
        abs_integral.abs() / abs_sigma
    } else {
        f64::INFINITY
    };

    Ok(IntegratedRegion {
        index: region_index,
        ppm_low: lo,
        ppm_high: hi,
        label: spec.label.clone(),
        raw_integral,
        abs_integral,
        abs_sigma,
        norm_integral: abs_integral,
        norm_sigma: abs_sigma,
        intrinsic_norm_sigma: abs_sigma,
        nearest_integer_h: abs_integral.round().max(0.0) as usize,
        applied_bias,
        applied_slope,
        snr_integral,
        deconvolved_area,
        deconvolved_norm_integral: deconvolved_area,
        deconvolved_sigma,
        cumulative_ppm,
        cumulative_integral: raw_cumulative_corr.clone(),
        raw_cumulative_corr,
    })
}

fn apply_normalization(
    regions: &mut [IntegratedRegion],
    mode: NormalizationMode,
) -> Result<f64, MultipletError> {
    if regions.is_empty() {
        return Ok(1.0);
    }

    let (scale, ref_idx_opt, ref_rel_var) = match mode {
        NormalizationMode::None => (1.0, None, 0.0),
        NormalizationMode::ExplicitScale { factor } => {
            if !factor.is_finite() || factor <= 0.0 {
                return Err(MultipletError::InvalidParameter(
                    "explicit normalization scale factor must be finite and positive".to_string(),
                ));
            }
            (factor, None, 0.0)
        }
        NormalizationMode::ReferenceRegion {
            region_index,
            target_protons,
        } => {
            if region_index >= regions.len() {
                return Err(MultipletError::InvalidParameter(format!(
                    "reference region index {region_index} out of bounds (len = {})",
                    regions.len()
                )));
            }
            if !target_protons.is_finite() || target_protons <= 0.0 {
                return Err(MultipletError::InvalidParameter(
                    "target_protons must be finite and positive".to_string(),
                ));
            }
            let ref_area = regions[region_index].abs_integral;
            if ref_area.abs() <= 1e-20 {
                return Err(MultipletError::InvalidParameter(format!(
                    "reference region {region_index} has near-zero integral ({ref_area})"
                )));
            }
            let rel = regions[region_index].abs_sigma / ref_area.abs();
            (target_protons / ref_area, Some(region_index), rel * rel)
        }
        NormalizationMode::TotalProtons { total_protons } => {
            if !total_protons.is_finite() || total_protons <= 0.0 {
                return Err(MultipletError::InvalidParameter(
                    "total_protons must be finite and positive".to_string(),
                ));
            }
            let pos_sum: f64 = regions.iter().map(|r| r.abs_integral.max(0.0)).sum::<f64>();
            if pos_sum <= 1e-20 {
                return Err(MultipletError::InvalidParameter(
                    "sum of positive region integrals is zero; cannot normalize by total protons"
                        .to_string(),
                ));
            }
            let pos_var: f64 = regions
                .iter()
                .filter(|r| r.abs_integral > 0.0)
                .map(|r| r.abs_sigma * r.abs_sigma)
                .sum();
            (total_protons / pos_sum, None, pos_var / (pos_sum * pos_sum))
        }
    };

    for reg in regions.iter_mut() {
        reg.norm_integral = reg.abs_integral * scale;
        reg.intrinsic_norm_sigma = scale.abs() * reg.abs_sigma;
        reg.norm_sigma = if Some(reg.index) == ref_idx_opt {
            // Exact by definition of setting the reference multiplet to `target_protons`;
            // intrinsic spectral noise on the reference is preserved in `intrinsic_norm_sigma`.
            0.0
        } else if reg.abs_integral.abs() > 1e-30 && ref_rel_var > 0.0 {
            let own_rel = reg.abs_sigma / reg.abs_integral.abs();
            reg.norm_integral.abs() * (own_rel * own_rel + ref_rel_var).sqrt()
        } else {
            reg.intrinsic_norm_sigma
        };
        reg.nearest_integer_h = reg.norm_integral.round().max(0.0) as usize;
        reg.deconvolved_norm_integral = reg.deconvolved_area.map(|a| a * scale);
        reg.cumulative_integral = reg.raw_cumulative_corr.iter().map(|&v| v * scale).collect();
    }

    Ok(scale)
}

// ============================================================================
// Section 2: First-Order & Leaning-Corrected Recursive Multiplet Tree Analyzer
// ============================================================================

/// Canonical 1D NMR multiplet classification following IUPAC / ACS / *J. Org. Chem.*
/// descending-`J` conventions (Hoye, Hanson & Vyvyan, *J. Org. Chem.* 1994, 59, 4096–4103).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MultipletClass {
    /// Sharp singlet (`s`).
    Singlet,
    /// Exchange- or relaxation-broadened singlet (`br s`).
    BroadSinglet,
    /// Doublet (`d`, `1:1`).
    Doublet,
    /// Triplet (`t`, `1:2:1`).
    Triplet,
    /// Quartet (`q`, `1:3:3:1`).
    Quartet,
    /// Quintet (`quint`, `1:4:6:4:1`).
    Quintet,
    /// Sextet (`sext`, `1:5:10:10:5:1`).
    Sextet,
    /// Septet (`sept`, `1:6:15:20:15:6:1`).
    Septet,
    /// Doublet of doublets (`dd`, `J_1 > J_2`).
    DoubletOfDoublets,
    /// Doublet of doublet of doublets (`ddd`, `J_1 > J_2 > J_3`).
    DoubletOfDoubletOfDoublets,
    /// Doublet of triplets (`dt`, `J_d > J_t`).
    DoubletOfTriplets,
    /// Triplet of doublets (`td`, `J_t > J_d`).
    TripletOfDoublets,
    /// Doublet of quartets (`dq`, `J_d > J_q`).
    DoubletOfQuartets,
    /// Quartet of doublets (`qd`, `J_q > J_d`).
    QuartetOfDoublets,
    /// Triplet of triplets (`tt`, `J_{t1} > J_{t2}`).
    TripletOfTriplets,
    /// Unresolved / higher-order / overlapping multiplet fallback (`m`).
    Multiplet,
}

impl MultipletClass {
    /// Standard abbreviation (`s`, `br s`, `d`, `t`, `q`, `quint`, `sext`, `sept`, `dd`, `ddd`,
    /// `dt`, `td`, `dq`, `qd`, `tt`, `m`).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Singlet => "s",
            Self::BroadSinglet => "br s",
            Self::Doublet => "d",
            Self::Triplet => "t",
            Self::Quartet => "q",
            Self::Quintet => "quint",
            Self::Sextet => "sext",
            Self::Septet => "sept",
            Self::DoubletOfDoublets => "dd",
            Self::DoubletOfDoubletOfDoublets => "ddd",
            Self::DoubletOfTriplets => "dt",
            Self::TripletOfDoublets => "td",
            Self::DoubletOfQuartets => "dq",
            Self::QuartetOfDoublets => "qd",
            Self::TripletOfTriplets => "tt",
            Self::Multiplet => "m",
        }
    }

    /// Parse a multiplicity token from journal characterization text (case-insensitive, supporting
    /// synonyms such as `bs`/`brs`, `p`/`pent`, `hept`, `hex`, `br m`).
    pub fn from_label(raw: &str) -> Option<Self> {
        let collapsed: String = raw
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .to_ascii_lowercase();
        match collapsed.as_str() {
            "s" | "singlet" => Some(Self::Singlet),
            "br s" | "brs" | "bs" | "br. s" | "broad s" | "broad singlet" => {
                Some(Self::BroadSinglet)
            }
            "d" | "doublet" => Some(Self::Doublet),
            "t" | "triplet" | "app t" => Some(Self::Triplet),
            "q" | "quartet" | "quart" => Some(Self::Quartet),
            "quint" | "quintet" | "p" | "pent" | "pentet" => Some(Self::Quintet),
            "sext" | "sextet" | "hex" | "hextet" => Some(Self::Sextet),
            "sept" | "septet" | "hept" | "heptet" => Some(Self::Septet),
            "dd" => Some(Self::DoubletOfDoublets),
            "ddd" => Some(Self::DoubletOfDoubletOfDoublets),
            "dt" => Some(Self::DoubletOfTriplets),
            "td" => Some(Self::TripletOfDoublets),
            "dq" => Some(Self::DoubletOfQuartets),
            "qd" => Some(Self::QuartetOfDoublets),
            "tt" => Some(Self::TripletOfTriplets),
            "m" | "br m" | "multiplet" => Some(Self::Multiplet),
            _ => None,
        }
    }

    /// Equivalent-spin counts `[n_1, n_2, ...]` ordered from largest `J` to smallest `J`.
    pub fn coupling_multiplicities(self) -> &'static [usize] {
        match self {
            Self::Singlet | Self::BroadSinglet | Self::Multiplet => &[],
            Self::Doublet => &[1],
            Self::Triplet => &[2],
            Self::Quartet => &[3],
            Self::Quintet => &[4],
            Self::Sextet => &[5],
            Self::Septet => &[6],
            Self::DoubletOfDoublets => &[1, 1],
            Self::DoubletOfDoubletOfDoublets => &[1, 1, 1],
            Self::DoubletOfTriplets => &[1, 2],
            Self::TripletOfDoublets => &[2, 1],
            Self::DoubletOfQuartets => &[1, 3],
            Self::QuartetOfDoublets => &[3, 1],
            Self::TripletOfTriplets => &[2, 2],
        }
    }

    /// Construct the canonical `MultipletClass` from descending-`J` equivalent partner counts.
    pub fn from_coupling_multiplicities(mults: &[usize]) -> Self {
        match mults {
            [] => Self::Singlet,
            [1] => Self::Doublet,
            [2] => Self::Triplet,
            [3] => Self::Quartet,
            [4] => Self::Quintet,
            [5] => Self::Sextet,
            [6] => Self::Septet,
            [1, 1] => Self::DoubletOfDoublets,
            [1, 1, 1] => Self::DoubletOfDoubletOfDoublets,
            [1, 2] => Self::DoubletOfTriplets,
            [2, 1] => Self::TripletOfDoublets,
            [1, 3] => Self::DoubletOfQuartets,
            [3, 1] => Self::QuartetOfDoublets,
            [2, 2] => Self::TripletOfTriplets,
            _ => Self::Multiplet,
        }
    }
}

impl fmt::Display for MultipletClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Direction toward which a second-order roofed multiplet leans in chemical shift.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum LeaningDirection {
    /// Symmetric first-order intensity ratios (`|\sin 2\theta| < 0.04`).
    #[default]
    None,
    /// Inner taller lines lie downfield (toward higher `ppm` / higher frequency).
    Downfield,
    /// Inner taller lines lie upfield (toward lower `ppm` / lower frequency).
    Upfield,
}

/// Observed or deconvolved sub-peak inside a 1D multiplet.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MultipletSubPeak {
    /// Sub-peak chemical shift in ppm.
    pub ppm: f64,
    /// Peak height or deconvolved area (positive).
    pub intensity: f64,
    /// Optional full-width at half-maximum in Hz.
    pub fwhm_hz: Option<f64>,
}

impl MultipletSubPeak {
    pub fn new(ppm: f64, intensity: f64) -> Self {
        Self {
            ppm,
            intensity,
            fwhm_hz: None,
        }
    }

    pub fn with_fwhm_hz(mut self, fwhm_hz: f64) -> Self {
        self.fwhm_hz = Some(fwhm_hz);
        self
    }
}

/// Configuration tolerances for recursive Hoye multiplet tree analysis and roofing correction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultipletConfig {
    /// Single-peak FWHM threshold (Hz) distinguishing `br s` from `s` (default: `5.0 Hz`).
    pub broad_singlet_fwhm_hz: f64,
    /// Maximum allowed RMS frequency discrepancy (Hz) between predicted tree lines and observed peaks (default: `0.35 Hz`).
    pub freq_tol_hz: f64,
    /// Sub-line coalescence distance (Hz) within which near-degenerate transitions merge into one peak (default: `0.35 Hz`).
    pub coalesce_hz: f64,
    /// Minimum separation (Hz) required to treat two `J` couplings as distinct rather than equivalent (default: `0.35 Hz`).
    pub equal_j_tol_hz: f64,
    /// Minimum physically meaningful scalar coupling `J` in Hz (default: `0.40 Hz`).
    pub min_j_hz: f64,
    /// Maximum scalar coupling `J` in Hz considered in tree decomposition (default: `60.0 Hz`).
    pub max_j_hz: f64,
    /// Maximum normalized intensity RMSD tolerated after accounting for second-order leaning (default: `0.12`).
    pub intensity_rmsd_tol: f64,
    /// Lower bound on `|\Delta\nu / J|` allowed by the roofing intensity model (default: `1.2`,
    /// corresponding to `|\sin 2\theta| <= 0.640` and inner:outer doublet ratio up to `4.56:1`,
    /// comfortably covering strong roofing such as `\Delta\nu / J = 2.5..3.0`).
    pub min_delta_nu_over_j: f64,
    /// Minimum relative height (fraction of strongest line in region) for sub-peak picking (default: `0.004`).
    pub subpeak_rel_threshold: f64,
    /// Minimum signal-to-noise ratio for sub-peak picking on noisy continuous traces (default: `3.5`).
    pub subpeak_snr_threshold: f64,
    /// Remove Lorentzian tail overlap cross-talk when picking sub-peaks from continuous traces (default: `true`).
    pub deconvolve_lorentzian_crosstalk: bool,
}

impl Default for MultipletConfig {
    fn default() -> Self {
        Self {
            broad_singlet_fwhm_hz: 5.0,
            freq_tol_hz: 0.35,
            coalesce_hz: 0.35,
            equal_j_tol_hz: 0.35,
            min_j_hz: 0.40,
            max_j_hz: 60.0,
            intensity_rmsd_tol: 0.12,
            min_delta_nu_over_j: 1.2,
            subpeak_rel_threshold: 0.004,
            subpeak_snr_threshold: 3.5,
            deconvolve_lorentzian_crosstalk: true,
        }
    }
}

/// Extracted coupling constant `J` (in Hz) and per-coupling second-order leaning estimate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CouplingConstant {
    /// Coupling constant `J` in Hz (`|\Delta \delta| * MHz`).
    pub j_hz: f64,
    /// Number of magnetically equivalent coupled spin-1/2 partners (`1` for `d`, `2` for `t`, `3` for `q`, etc.).
    pub equivalent_partners: usize,
    /// Fitted AB-like mixing parameter `\eta = \sin(2\theta) \in (-1, 1)`, positive when leaning downfield.
    pub leaning_sin2theta: f64,
    /// Estimated `|\Delta\nu / J| = \cot(2\theta) = \sqrt{1 - \eta^2} / |\eta|` when roofing is appreciable (`|\eta| >= 0.04`).
    pub estimated_delta_nu_over_j: Option<f64>,
    /// Direction of the coupled partner inferred from intensity leaning.
    pub leaning_direction: LeaningDirection,
}

/// One level of the recursive multiplet splitting tree (`level 0` = unsplit center, `level G` = final multiplet).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultipletTreeLevel {
    /// Tree depth (`0` for root chemical shift, `1..=G` after applying each coupling group).
    pub depth: usize,
    /// Coupling `J` (Hz) applied at this level (`None` at root `depth == 0`).
    pub applied_j_hz: Option<f64>,
    /// Number of equivalent `J` splittings applied at this level (`0` at root).
    pub equivalent_partners: usize,
    /// Sub-line `(frequency_hz, normalized_intensity)` nodes at this tree depth (sorted high-Hz to low-Hz).
    pub nodes_hz: Vec<(f64, f64)>,
}

/// Multiplet-wide second-order roofing ("leaning") diagnostics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct LeaningSummary {
    /// `true` when appreciable intensity asymmetry (`|\sin 2\theta| >= 0.04` or `|slope| * span >= 0.08`) is present.
    pub is_roofed: bool,
    /// Dominant direction toward which taller inner lines lean.
    pub dominant_direction: LeaningDirection,
    /// Effective inner-to-outer doublet intensity ratio `(1 + |\eta|) / (1 - |\eta|)` (`>= 1.0`).
    pub inner_to_outer_ratio: f64,
    /// Inferred `|\Delta\nu / J|` of the most strongly roofed coupling (`None` if unroofed).
    pub estimated_delta_nu_over_j: Option<f64>,
    /// Fitted global linear intensity tilt `\beta` (`I(\nu) = I_0(\nu) * (1 + \beta (\nu - \nu_0))`, in `Hz^-1`).
    pub linear_roofing_slope_per_hz: f64,
}

/// Complete output of recursive multiplet `J`-coupling tree analysis.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultipletAnalysis {
    /// Classified multiplet pattern (`s`, `br s`, `d`, `t`, `q`, `quint`, `sext`, `sept`, `dd`, `ddd`, `dt`, `td`, `dq`, `qd`, `tt`, `m`).
    pub class: MultipletClass,
    /// Canonical string label (`class.as_str()`).
    pub label: String,
    /// Geometric splitting-tree center chemical shift in ppm (immune to roofing intensity skew; equals `centroid_ppm` for `m`).
    pub center_ppm: f64,
    /// Raw intensity-weighted first-moment centroid chemical shift in ppm.
    pub centroid_ppm: f64,
    /// Second-order AB Zeeman chemical shift in ppm corrected for roofing shift displacement
    /// `\Delta\nu_AB_corr = \mp (J / 2) * (\sqrt{R} - 1) / (\sqrt{R} + 1)`.
    pub ab_zeeman_ppm: f64,
    /// `(min_ppm, max_ppm)` span of the observed multiplet peaks.
    pub range_ppm: (f64, f64),
    /// Spectrometer Larmor observe frequency in MHz used for `ppm <-> Hz` conversion.
    pub spectrometer_mhz: f64,
    /// Extracted scalar coupling constants sorted from largest `J` to smallest `J`.
    pub couplings: Vec<CouplingConstant>,
    /// Observed/picked sub-peaks sorted from downfield (high ppm) to upfield (low ppm).
    pub peaks: Vec<MultipletSubPeak>,
    /// Top-down splitting tree levels for interactive J-tree diagram visualization.
    pub tree_levels: Vec<MultipletTreeLevel>,
    /// Elementary spin-1/2 splittings (Hz, descending) peeled directly by Hoye (1994/2002) recursive subtraction.
    pub hoye_peeled_j_hz: Vec<f64>,
    /// Root-mean-square frequency residual (Hz) between coalesced tree lines and observed peaks.
    pub freq_rmsd_hz: f64,
    /// Root-mean-square normalized intensity residual after leaning correction.
    pub intensity_rmsd: f64,
    /// Second-order roofing diagnostics.
    pub leaning: LeaningSummary,
    /// Optional normalized integer proton count attached from region integration.
    pub proton_count: Option<usize>,
}

impl MultipletAnalysis {
    /// Return the extracted coupling constants `J` in Hz in standard descending order.
    pub fn j_values_hz(&self) -> Vec<f64> {
        self.couplings.iter().map(|c| c.j_hz).collect()
    }
}

/// Exact closed-form solution of a 4-line second-order AB spin system
/// (Hoye & Zhao, *J. Org. Chem.* 2002, 67, 4014–4016; Pople, Schneider & Bernstein 1959).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AbQuartetSolution {
    /// Exact scalar coupling `J_AB = |\nu_1 - \nu_2| = |\nu_3 - \nu_4|` in Hz.
    pub j_ab_hz: f64,
    /// True Zeeman frequency `\nu_A` (downfield spin, Hz; or ppm when `spectrometer_mhz` supplied).
    pub nu_a_hz: f64,
    /// True Zeeman frequency `\nu_B` (upfield spin, Hz).
    pub nu_b_hz: f64,
    /// Chemical shift `\delta_A` in ppm (when `spectrometer_mhz` is supplied).
    pub delta_a_ppm: Option<f64>,
    /// Chemical shift `\delta_B` in ppm (when `spectrometer_mhz` is supplied).
    pub delta_b_ppm: Option<f64>,
    /// True Zeeman frequency difference `\Delta\nu_{AB} = \nu_A - \nu_B = \sqrt{(\nu_1 - \nu_4)(\nu_2 - \nu_3)}` in Hz.
    pub delta_nu_hz: f64,
    /// Second-order ratio `\Delta\nu_{AB} / J_{AB}`.
    pub delta_nu_over_j: f64,
    /// Theoretical inner-to-outer line intensity ratio `(1 + \sin 2\theta) / (1 - \sin 2\theta) = (\nu_1 - \nu_3) / (\nu_2 - \nu_4)`?
    /// Exact QM ratio: `(2C + J) / (2C - J)`? Wait: `sin(2 theta) = J / (2C)`, so `(1 + sin 2theta)/(1 - sin 2theta) = (2C + J)/(2C - J) = (\nu_1 - \nu_4) / (\nu_2 - \nu_3)`.
    pub inner_to_outer_ratio: f64,
}

/// Solve an exact 4-line second-order AB quartet from its 4 transition positions
/// (in ppm when `spectrometer_mhz = Some(mhz)`, or directly in Hz when `spectrometer_mhz = None`),
/// implementing Hoye & Zhao (*J. Org. Chem.* 2002, 67, 4014–4016).
pub fn solve_ab_quartet(
    mut lines: [f64; 4],
    spectrometer_mhz: Option<f64>,
) -> Result<AbQuartetSolution, MultipletError> {
    if lines.iter().any(|v| !v.is_finite()) {
        return Err(MultipletError::InvalidParameter(
            "AB quartet line positions must be finite".to_string(),
        ));
    }
    let scale = match spectrometer_mhz {
        Some(mhz) => {
            if !mhz.is_finite() || mhz <= 0.0 {
                return Err(MultipletError::InvalidParameter(
                    "spectrometer_mhz must be finite and positive".to_string(),
                ));
            }
            mhz
        }
        None => 1.0,
    };
    for v in &mut lines {
        *v *= scale;
    }
    // Sort descending: nu_1 > nu_2 > nu_3 > nu_4
    lines.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let [nu1, nu2, nu3, nu4] = lines;

    let j_left = nu1 - nu2;
    let j_right = nu3 - nu4;
    if j_left <= 0.0 || j_right <= 0.0 {
        return Err(MultipletError::InvalidParameter(
            "AB quartet requires 4 distinct transitions".to_string(),
        ));
    }
    if (j_left - j_right).abs() > 0.50_f64.max(0.10 * j_left) {
        return Err(MultipletError::InvalidParameter(format!(
            "outer doublet splittings do not match for an AB spin system ({j_left:.3} vs {j_right:.3} Hz)"
        )));
    }
    let j_ab_hz = 0.5 * (j_left + j_right);
    let d_outer = nu1 - nu4; // 2C + J
    let d_inner = nu2 - nu3; // 2C - J
    if d_inner <= 0.0 {
        return Err(MultipletError::InvalidParameter(
            "inner AB lines must be separated (nu_2 > nu_3)".to_string(),
        ));
    }
    // Since 2C = (d_outer + d_inner)/2 = nu1 - nu3 = nu2 - nu4, (2C)^2 - J^2 = d_outer * d_inner!
    let delta_nu_hz = (d_outer * d_inner).sqrt();
    let center_hz = 0.25 * (nu1 + nu2 + nu3 + nu4);
    let nu_a_hz = center_hz + 0.5 * delta_nu_hz;
    let nu_b_hz = center_hz - 0.5 * delta_nu_hz;
    let inner_to_outer_ratio = d_outer / d_inner;

    Ok(AbQuartetSolution {
        j_ab_hz,
        nu_a_hz,
        nu_b_hz,
        delta_a_ppm: spectrometer_mhz.map(|mhz| nu_a_hz / mhz),
        delta_b_ppm: spectrometer_mhz.map(|mhz| nu_b_hz / mhz),
        delta_nu_hz,
        delta_nu_over_j: delta_nu_hz / j_ab_hz,
        inner_to_outer_ratio,
    })
}

/// Analyze a 1D spectral region `[ppm_a, ppm_b]`, picking sub-peaks with quadratic sub-sample
/// interpolation, optional Lorentzian tail cross-talk removal, and recursive Hoye `J`-tree
/// classification.
pub fn analyze_multiplet_region(
    ppm: &[f64],
    intensity: &[f64],
    ppm_a: f64,
    ppm_b: f64,
    spectrometer_mhz: f64,
    config: &MultipletConfig,
) -> Result<MultipletAnalysis, MultipletError> {
    validate_trace(ppm, intensity)?;
    if !spectrometer_mhz.is_finite() || spectrometer_mhz <= 0.0 {
        return Err(MultipletError::InvalidParameter(
            "spectrometer_mhz must be finite and positive".to_string(),
        ));
    }
    let lo = ppm_a.min(ppm_b);
    let hi = ppm_a.max(ppm_b);
    let sorted = sorted_ascending_trace(ppm, intensity);
    let slice: Vec<(f64, f64)> = sorted
        .iter()
        .copied()
        .filter(|&(x, _)| x >= lo && x <= hi)
        .collect();
    if slice.len() < 3 {
        return Err(MultipletError::InvalidParameter(format!(
            "multiplet region [{lo}, {hi}] ppm contains fewer than 3 sample points"
        )));
    }

    let global_noise = estimate_spectral_noise_rms(ppm, intensity, None).unwrap_or(0.0);
    let subpeaks = pick_region_subpeaks(&slice, spectrometer_mhz, global_noise, config);
    if subpeaks.is_empty() {
        return Err(MultipletError::EmptyOrInvalidSpectrum(format!(
            "no positive peaks detected inside [{lo}, {hi}] ppm"
        )));
    }
    analyze_multiplet_peaks(&subpeaks, spectrometer_mhz, config)
}

/// Classify a multiplet and extract scalar couplings `J` (in Hz) plus second-order roofing
/// metrics from picked sub-peaks `(ppm, intensity)`.
///
/// Combines recursive Hoye sub-doublet peeling (Hoye, Hanson & Vyvyan, *J. Org. Chem.* 1994,
/// 59, 4096–4103; Hoye & Zhao, *J. Org. Chem.* 2002, 67, 4014–4016) with coalesced generative
/// tree verification and bounded `\Delta\nu / J` leaning optimization.
pub fn analyze_multiplet_peaks(
    peaks: &[MultipletSubPeak],
    spectrometer_mhz: f64,
    config: &MultipletConfig,
) -> Result<MultipletAnalysis, MultipletError> {
    if peaks.is_empty() {
        return Err(MultipletError::EmptyOrInvalidSpectrum(
            "multiplet peak list is empty".to_string(),
        ));
    }
    if !spectrometer_mhz.is_finite() || spectrometer_mhz <= 0.0 {
        return Err(MultipletError::InvalidParameter(
            "spectrometer_mhz must be finite and positive".to_string(),
        ));
    }

    // Filter positive finite peaks, sort descending in ppm (downfield -> upfield) and merge
    // duplicate picks closer than `0.5 * coalesce_hz`.
    let mut work: Vec<MultipletSubPeak> = peaks
        .iter()
        .copied()
        .filter(|p| p.ppm.is_finite() && p.intensity.is_finite() && p.intensity > 0.0)
        .collect();
    if work.is_empty() {
        return Err(MultipletError::EmptyOrInvalidSpectrum(
            "multiplet contains no positive-intensity peaks".to_string(),
        ));
    }
    work.sort_by(|a, b| {
        b.ppm
            .partial_cmp(&a.ppm)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let pre_merge_ppm = (0.45 * config.coalesce_hz) / spectrometer_mhz;
    let mut merged: Vec<MultipletSubPeak> = Vec::with_capacity(work.len());
    for pk in work {
        if let Some(last) = merged.last_mut() {
            if (last.ppm - pk.ppm).abs() <= pre_merge_ppm {
                let tot = last.intensity + pk.intensity;
                last.ppm = (last.ppm * last.intensity + pk.ppm * pk.intensity) / tot;
                last.intensity = tot;
                last.fwhm_hz = match (last.fwhm_hz, pk.fwhm_hz) {
                    (Some(w1), Some(w2)) => Some(w1.max(w2)),
                    (Some(w1), None) => Some(w1),
                    (None, Some(w2)) => Some(w2),
                    (None, None) => None,
                };
                continue;
            }
        }
        merged.push(pk);
    }

    let total_int: f64 = merged.iter().map(|p| p.intensity).sum();
    let obs_hz: Vec<f64> = merged.iter().map(|p| p.ppm * spectrometer_mhz).collect();
    let obs_norm: Vec<f64> = merged.iter().map(|p| p.intensity / total_int).collect();
    let centroid_hz: f64 = obs_hz
        .iter()
        .zip(obs_norm.iter())
        .map(|(&nu, &a)| nu * a)
        .sum();
    let centroid_ppm = centroid_hz / spectrometer_mhz;
    let min_ppm = merged.last().map(|p| p.ppm).unwrap_or(centroid_ppm);
    let max_ppm = merged.first().map(|p| p.ppm).unwrap_or(centroid_ppm);

    // Case 1: Single peak -> `s` or `br s`
    if merged.len() == 1 {
        let fwhm = merged[0].fwhm_hz.unwrap_or(1.0);
        let class = if fwhm >= config.broad_singlet_fwhm_hz {
            MultipletClass::BroadSinglet
        } else {
            MultipletClass::Singlet
        };
        let tree_levels = vec![MultipletTreeLevel {
            depth: 0,
            applied_j_hz: None,
            equivalent_partners: 0,
            nodes_hz: vec![(obs_hz[0], 1.0)],
        }];
        return Ok(MultipletAnalysis {
            class,
            label: class.as_str().to_string(),
            center_ppm: centroid_ppm,
            centroid_ppm,
            ab_zeeman_ppm: centroid_ppm,
            range_ppm: (min_ppm, max_ppm),
            spectrometer_mhz,
            couplings: Vec::new(),
            peaks: merged,
            tree_levels,
            hoye_peeled_j_hz: Vec::new(),
            freq_rmsd_hz: 0.0,
            intensity_rmsd: 0.0,
            leaning: LeaningSummary {
                is_roofed: false,
                dominant_direction: LeaningDirection::None,
                inner_to_outer_ratio: 1.0,
                estimated_delta_nu_over_j: None,
                linear_roofing_slope_per_hz: 0.0,
            },
            proton_count: None,
        });
    }

    // Maximum allowed `|\sin 2\theta|` from `config.min_delta_nu_over_j`
    let r_min = config.min_delta_nu_over_j.max(0.25);
    let eta_max = (1.0 / (r_min * r_min + 1.0).sqrt()).clamp(0.05, 0.92);

    // Step 1: Recursive Hoye sub-doublet subtraction to discover candidate elementary J multisets
    let mut hoye_multisets: Vec<Vec<f64>> = Vec::new();
    hoye_recursive_peel(
        &obs_hz,
        &obs_norm,
        config,
        eta_max,
        &mut Vec::new(),
        &mut hoye_multisets,
    );

    let best_hoye_peeled = hoye_multisets.first().cloned().unwrap_or_default();

    // Step 2: Evaluate all canonical multiplet family hypotheses seeded from both closed-form
    // outer-line Hoye invariants and peeled multisets.
    let candidate_classes = [
        MultipletClass::Doublet,
        MultipletClass::Triplet,
        MultipletClass::Quartet,
        MultipletClass::Quintet,
        MultipletClass::Sextet,
        MultipletClass::Septet,
        MultipletClass::DoubletOfDoublets,
        MultipletClass::DoubletOfTriplets,
        MultipletClass::TripletOfDoublets,
        MultipletClass::DoubletOfDoubletOfDoublets,
        MultipletClass::DoubletOfQuartets,
        MultipletClass::QuartetOfDoublets,
        MultipletClass::TripletOfTriplets,
    ];

    let mut coalesce_radii = vec![config.coalesce_hz];
    let mut fwhms: Vec<f64> = merged.iter().filter_map(|p| p.fwhm_hz).collect();
    if !fwhms.is_empty() {
        fwhms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let med_fwhm = fwhms[fwhms.len() / 2];
        for mult in [0.72, 0.92] {
            let r_hz = (mult * med_fwhm).min(1.60);
            if r_hz > config.coalesce_hz + 0.05
                && !coalesce_radii.iter().any(|&c| (c - r_hz).abs() < 0.05)
            {
                coalesce_radii.push(r_hz);
            }
        }
    }

    let mut best_fit: Option<EvaluatedTreeFit> = None;

    for &cls in &candidate_classes {
        let seeds = generate_initial_j_seeds(cls, &obs_hz, &hoye_multisets, config);
        for seed_js in seeds {
            for &coal_hz in &coalesce_radii {
                if let Some(fit) = fit_multiplet_hypothesis(
                    cls, &seed_js, &obs_hz, &obs_norm, eta_max, coal_hz, config,
                ) {
                    let replace = match &best_fit {
                        None => true,
                        Some(prev) => fit.selection_loss < prev.selection_loss,
                    };
                    if replace {
                        best_fit = Some(fit);
                    }
                }
            }
        }
    }

    if let Some(fit) = best_fit {
        let center_ppm = fit.center_hz / spectrometer_mhz;
        let ab_zeeman_ppm = compute_ab_zeeman_ppm(fit.center_hz, &fit.couplings, spectrometer_mhz);
        let tree_levels = build_tree_levels(fit.center_hz, &fit.couplings, fit.linear_beta);
        let hoye_peeled_j_hz = if !best_hoye_peeled.is_empty() {
            best_hoye_peeled
        } else {
            expand_elementary_js(&fit.couplings)
        };

        return Ok(MultipletAnalysis {
            class: fit.class,
            label: fit.class.as_str().to_string(),
            center_ppm,
            centroid_ppm,
            ab_zeeman_ppm,
            range_ppm: (min_ppm, max_ppm),
            spectrometer_mhz,
            couplings: fit.couplings,
            peaks: merged,
            tree_levels,
            hoye_peeled_j_hz,
            freq_rmsd_hz: fit.freq_rmsd_hz,
            intensity_rmsd: fit.intensity_rmsd,
            leaning: fit.leaning,
            proton_count: None,
        });
    }

    // Fallback: unstructured / higher-order multiplet `m`
    let linear_beta = estimate_global_linear_slope(&obs_hz, &obs_norm, centroid_hz);
    let span_hz = (obs_hz[0] - obs_hz[obs_hz.len() - 1]).abs();
    let is_roofed = (linear_beta * 0.5 * span_hz).abs() >= 0.08;
    let dominant_direction = if !is_roofed {
        LeaningDirection::None
    } else if linear_beta > 0.0 {
        LeaningDirection::Downfield
    } else {
        LeaningDirection::Upfield
    };
    let tree_levels = vec![MultipletTreeLevel {
        depth: 0,
        applied_j_hz: None,
        equivalent_partners: 0,
        nodes_hz: obs_hz
            .iter()
            .copied()
            .zip(obs_norm.iter().copied())
            .collect(),
    }];

    Ok(MultipletAnalysis {
        class: MultipletClass::Multiplet,
        label: MultipletClass::Multiplet.as_str().to_string(),
        center_ppm: centroid_ppm,
        centroid_ppm,
        ab_zeeman_ppm: centroid_ppm,
        range_ppm: (min_ppm, max_ppm),
        spectrometer_mhz,
        couplings: Vec::new(),
        peaks: merged,
        tree_levels,
        hoye_peeled_j_hz: best_hoye_peeled,
        freq_rmsd_hz: 0.0,
        intensity_rmsd: 0.0,
        leaning: LeaningSummary {
            is_roofed,
            dominant_direction,
            inner_to_outer_ratio: 1.0,
            estimated_delta_nu_over_j: None,
            linear_roofing_slope_per_hz: linear_beta,
        },
        proton_count: None,
    })
}

/// Synthesize sub-peaks `(ppm, intensity, fwhm_hz)` for a first-order or second-order leaning
/// multiplet centered at `center_ppm`.
///
/// Each tuple in `couplings` is `(equivalent_partners, j_hz, signed_delta_nu_over_j)`:
/// - `signed_delta_nu_over_j = None`: ideal first-order Pascal/binomial intensities (`1:1`, `1:2:1`, `1:3:3:1`, ...).
/// - `signed_delta_nu_over_j = Some(+3.0)`: coupled partner lies downfield at `+3 * J` Hz, so
///   lines at higher ppm are enhanced by `1 + \sin(2\theta)` (`\sin 2\theta = 1 / \sqrt{3^2 + 1}`).
/// - `signed_delta_nu_over_j = Some(-3.0)`: coupled partner lies upfield at `-3 * J` Hz, so
///   lines at lower ppm are enhanced by `1 + |\sin(2\theta)|`.
pub fn synthesize_multiplet_peaks(
    center_ppm: f64,
    spectrometer_mhz: f64,
    couplings: &[(usize, f64, Option<f64>)],
    fwhm_hz: f64,
) -> Vec<MultipletSubPeak> {
    let center_hz = center_ppm * spectrometer_mhz;
    let mut nodes: Vec<(f64, f64)> = vec![(center_hz, 1.0)];

    for &(n_eq, j_hz, dnu_over_j) in couplings {
        let eta = match dnu_over_j {
            Some(r) if r.abs() > 1e-6 => r.signum() / (r * r + 1.0).sqrt(),
            _ => 0.0,
        };
        for _ in 0..n_eq {
            let mut next = Vec::with_capacity(nodes.len() * 2);
            for &(nu, w) in &nodes {
                // Downfield (+J/2) gets `0.5 * (1 + eta)`, upfield (-J/2) gets `0.5 * (1 - eta)`
                next.push((nu + 0.5 * j_hz, 0.5 * w * (1.0 + eta)));
                next.push((nu - 0.5 * j_hz, 0.5 * w * (1.0 - eta)));
            }
            nodes = next;
        }
    }

    nodes.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let merged = merge_close_lines(&nodes, 0.02);
    merged
        .into_iter()
        .map(|(nu, w)| MultipletSubPeak {
            ppm: nu / spectrometer_mhz,
            intensity: w,
            fwhm_hz: Some(fwhm_hz),
        })
        .collect()
}

/// Render a list of [`MultipletSubPeak`]s onto a 1D `ppm_grid` using normalized pseudo-Voigt
/// line shapes (`gaussian_fraction == 0.0` for pure Lorentzian, `1.0` for pure Gaussian; unit total
/// area in ppm when sub-peak intensities sum to `1.0`).
pub fn synthesize_multiplet_spectrum(
    ppm_grid: &[f64],
    peaks: &[MultipletSubPeak],
    spectrometer_mhz: f64,
    default_fwhm_hz: f64,
    gaussian_fraction: f64,
) -> Vec<f64> {
    let gf = gaussian_fraction.clamp(0.0, 1.0);
    let ln2 = std::f64::consts::LN_2;
    let pi = std::f64::consts::PI;
    let mut y = vec![0.0_f64; ppm_grid.len()];

    for pk in peaks {
        let fwhm_ppm = pk.fwhm_hz.unwrap_or(default_fwhm_hz).max(1e-4) / spectrometer_mhz;
        let hwhm = 0.5 * fwhm_ppm;
        let sigma_g = fwhm_ppm / (2.0 * (2.0 * ln2).sqrt());
        let norm_g = 1.0 / (sigma_g * (2.0 * pi).sqrt());

        for (out_y, &x) in y.iter_mut().zip(ppm_grid.iter()) {
            let dx = x - pk.ppm;
            let lor = (hwhm / pi) / (dx * dx + hwhm * hwhm);
            let gau = if gf > 0.0 {
                norm_g * (-0.5 * (dx / sigma_g).powi(2)).exp()
            } else {
                0.0
            };
            *out_y += pk.intensity * ((1.0 - gf) * lor + gf * gau);
        }
    }
    y
}

/// Exact quantum-mechanical spin-1/2 Zeeman + scalar coupling Hamiltonian simulator
/// (`H = \sum_i \nu_i I_{iz} + \sum_{i < j} J_{ij} \mathbf{I}_i \cdot \mathbf{I}_j`),
/// returning the exact second-order transitions predominantly belonging to `observe_spin`
/// (or coupled spin-system sub-lines inside its spectral neighborhood).
///
/// Partitioned by conserved `M_z = \sum_i m_i` blocks and diagonalized via exact Jacobi
/// rotations (`1 <= n_spins <= 8`).
pub fn simulate_exact_spin_half_multiplet(
    shifts_ppm: &[f64],
    j_matrix_hz: &[Vec<f64>],
    observe_spin: usize,
    spectrometer_mhz: f64,
    min_rel_intensity: f64,
) -> Result<Vec<MultipletSubPeak>, MultipletError> {
    let n = shifts_ppm.len();
    if n == 0 || n > 8 {
        return Err(MultipletError::InvalidParameter(format!(
            "exact spin-1/2 Hamiltonian solver supports 1..=8 spins (got {n})"
        )));
    }
    if observe_spin >= n || j_matrix_hz.len() != n || j_matrix_hz.iter().any(|r| r.len() != n) {
        return Err(MultipletError::InvalidParameter(
            "invalid observe_spin or j_matrix_hz dimensions".to_string(),
        ));
    }
    if !spectrometer_mhz.is_finite() || spectrometer_mhz <= 0.0 {
        return Err(MultipletError::InvalidParameter(
            "spectrometer_mhz must be finite and positive".to_string(),
        ));
    }

    let nu_hz: Vec<f64> = shifts_ppm.iter().map(|&d| d * spectrometer_mhz).collect();
    let dim = 1usize << n;

    // Group basis states `0..2^n` by popcount `k` = number of alpha (+1/2) spins (`k = 0..=n`).
    let mut blocks: Vec<Vec<usize>> = vec![Vec::new(); n + 1];
    for state in 0..dim {
        blocks[state.count_ones() as usize].push(state);
    }

    let mut block_evals: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    let mut block_evecs: Vec<Vec<Vec<f64>>> = Vec::with_capacity(n + 1);

    for basis in &blocks {
        let bdim = basis.len();
        let mut h = vec![vec![0.0_f64; bdim]; bdim];
        for (row, &st_r) in basis.iter().enumerate() {
            // Diagonal: sum_i nu_i m_i + sum_{i<j} J_ij m_i m_j
            let mut diag = 0.0;
            for (i, &nu_i) in nu_hz.iter().enumerate() {
                let mi = if ((st_r >> i) & 1) == 1 { 0.5 } else { -0.5 };
                diag += nu_i * mi;
                for (j, row_j) in j_matrix_hz.iter().enumerate().skip(i + 1) {
                    let mj = if ((st_r >> j) & 1) == 1 { 0.5 } else { -0.5 };
                    diag += 0.5 * (j_matrix_hz[i][j] + row_j[i]) * mi * mj;
                }
            }
            h[row][row] = diag;

            // Off-diagonal flip-flop `0.5 * J_ij (I_i+ I_j- + I_i- I_j+)`
            for (i, row_i) in j_matrix_hz.iter().enumerate().take(n) {
                for (j, row_j) in j_matrix_hz.iter().enumerate().take(n).skip(i + 1) {
                    let bi = (st_r >> i) & 1;
                    let bj = (st_r >> j) & 1;
                    if bi != bj {
                        let flipped = st_r ^ (1usize << i) ^ (1usize << j);
                        if let Ok(col) = basis.binary_search(&flipped) {
                            let j_ij = 0.5 * (row_i[j] + row_j[i]);
                            h[row][col] += 0.5 * j_ij;
                        }
                    }
                }
            }
        }
        let (evals, evecs) = jacobi_symmetric_eigen(&h);
        block_evals.push(evals);
        block_evecs.push(evecs);
    }

    // Allowed single-quantum transitions connect block `k` (lower M_z) to block `k + 1` (higher M_z).
    // In a non-selective 90-deg pulse-acquire 1D NMR experiment, transition intensity is proportional to
    // `<psi_high | F_+ | psi_low> * <psi_low | I_{observe,-} | psi_high>` (or `|<psi_high | I_{observe,+} | psi_low>|^2`
    // when isolating spin `observe_spin`, wait: in coupled AB systems, actual experimental line intensity in 1D
    // comes from total `F_x` excitation & `F_+` detection, where transitions localized near `nu_{observe}` carry
    // `|<psi_high | F_+ | psi_low>|^2`! Specifically, in AB: `|<beta_A'| F_+ |alpha_A'>|^2 = 1 +- sin(2 theta)`,
    // whereas `|<beta_A'| I_{A+} |alpha_A'>|^2 = cos^2(theta)` or `sin^2(theta)` without the cross-term!).
    // Using exact coherence amplitude `<high | F_+ | low> * <low | I_{obs,-} | high>`? Wait:
    // Let's check AB! In AB:
    // |1> = |++>, |2> = cos(theta)|+-> + sin(theta)|-+>, |3> = -sin(theta)|+-> + cos(theta)|-+>, |4> = |-->.
    // Transition |3> -> |1> (frequency bar_nu + C + J/2, spin A outer):
    //   <1 | F_+ | 3> = -sin(theta) + cos(theta), so |<1 | F_+ | 3>|^2 = 1 - sin(2 theta)!
    //   And A-character `|<1 | I_{A+} | 3>|^2 + |<3 | I_{A+} | 4>|^2` is `cos^2(theta) > 0.5`!
    // Transition |4> -> |2> (frequency bar_nu + C - J/2, spin A inner):
    //   <2 | F_+ | 4> = cos(theta) + sin(theta), so |<2 | F_+ | 4>|^2 = 1 + sin(2 theta)!
    //   And A-character `|<2 | I_{A+} | 4>|^2 = cos^2(theta) > 0.5`!
    // Whereas spin B transitions (|2> -> |1> and |4> -> |3>) have A-character `sin^2(theta) < 0.5`!
    // Look how exact that physics is: each eigen-transition `(k -> k+1)` has total observable intensity
    // `|<high | F_+ | low>|^2`, and belongs to `observe_spin` iff `observe_spin` has the largest
    // single-spin raising matrix element `|<high | I_{s,+} | low>|^2` (summed over chemically equivalent
    // shifts if `shifts_ppm[s] == shifts_ppm[observe_spin]`)!
    let obs_shift = shifts_ppm[observe_spin];
    let mut raw_lines: Vec<(f64, f64)> = Vec::new();

    for k in 0..n {
        let basis_lo = &blocks[k];
        let basis_hi = &blocks[k + 1];
        for (a_idx, &e_lo) in block_evals[k].iter().enumerate() {
            for (b_idx, &e_hi) in block_evals[k + 1].iter().enumerate() {
                // Compute `<hi, b | I_{s,+} | lo, a>` for each spin `s = 0..n`
                let mut amp_per_spin = vec![0.0_f64; n];
                for (r_lo, &st_lo) in basis_lo.iter().enumerate() {
                    let c_lo = block_evecs[k][r_lo][a_idx];
                    if c_lo.abs() < 1e-14 {
                        continue;
                    }
                    for (s, amp_s) in amp_per_spin.iter_mut().enumerate() {
                        if ((st_lo >> s) & 1) == 0 {
                            let st_hi = st_lo | (1usize << s);
                            if let Ok(r_hi) = basis_hi.binary_search(&st_hi) {
                                *amp_s += block_evecs[k + 1][r_hi][b_idx] * c_lo;
                            }
                        }
                    }
                }
                let f_plus: f64 = amp_per_spin.iter().sum();
                let intensity = f_plus * f_plus;
                if intensity <= 1e-12 {
                    continue;
                }

                // Assign transition to the chemical-shift group carrying the largest `|amp_s|^2`
                let mut obs_weight = 0.0;
                let mut max_other_weight = 0.0;
                for (s, &amp_s) in amp_per_spin.iter().enumerate() {
                    let w = amp_s * amp_s;
                    if (shifts_ppm[s] - obs_shift).abs() <= 1e-9 {
                        obs_weight += w;
                    } else if w > max_other_weight {
                        max_other_weight = w;
                    }
                }
                if obs_weight > 0.0 && obs_weight >= max_other_weight {
                    raw_lines.push((e_hi - e_lo, intensity));
                }
            }
        }
    }

    raw_lines.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let merged = merge_close_lines(&raw_lines, 0.02);
    let total: f64 = merged.iter().map(|l| l.1).sum::<f64>().max(1e-30);
    let max_i: f64 = merged
        .iter()
        .map(|l| l.1)
        .fold(0.0_f64, f64::max)
        .max(1e-30);

    Ok(merged
        .into_iter()
        .filter(|&(_, inten)| inten / max_i >= min_rel_intensity)
        .map(|(nu, inten)| MultipletSubPeak {
            ppm: nu / spectrometer_mhz,
            intensity: inten / total,
            fwhm_hz: None,
        })
        .collect())
}

// --- Internal helpers for Section 2 ---

struct EvaluatedTreeFit {
    class: MultipletClass,
    center_hz: f64,
    couplings: Vec<CouplingConstant>,
    freq_rmsd_hz: f64,
    intensity_rmsd: f64,
    linear_beta: f64,
    leaning: LeaningSummary,
    selection_loss: f64,
}

fn pick_region_subpeaks(
    slice: &[(f64, f64)],
    spectrometer_mhz: f64,
    global_noise_rms: f64,
    config: &MultipletConfig,
) -> Vec<MultipletSubPeak> {
    let m = slice.len();
    let lo = slice[0].0;
    let hi = slice[m - 1].0;
    let width = (hi - lo).max(1e-12);

    // Local linear edge detrending so sloped baselines do not shift peak tops
    let k_edge = ((m as f64 * 0.10) as usize).clamp(1, (m / 3).max(1));
    let mut edge_pts = Vec::with_capacity(2 * k_edge);
    edge_pts.extend_from_slice(&slice[..k_edge]);
    edge_pts.extend_from_slice(&slice[(m - k_edge)..]);
    let ne = edge_pts.len() as f64;
    let xb = edge_pts.iter().map(|p| p.0).sum::<f64>() / ne;
    let yb = edge_pts.iter().map(|p| p.1).sum::<f64>() / ne;
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    for &(x, y) in &edge_pts {
        sxx += (x - xb).powi(2);
        sxy += (x - xb) * (y - yb);
    }
    let slope = if sxx > 1e-30 && width > 0.0 {
        sxy / sxx
    } else {
        0.0
    };
    let edge_noise_rms = if edge_pts.len() >= 4 {
        let rss: f64 = edge_pts
            .iter()
            .map(|&(x, y)| {
                let r = y - (yb + slope * (x - xb));
                r * r
            })
            .sum();
        (rss / ((edge_pts.len() - 2) as f64)).sqrt()
    } else {
        0.0
    };
    let effective_noise_rms = global_noise_rms.max(edge_noise_rms);

    let y_corr: Vec<f64> = slice
        .iter()
        .map(|&(x, y)| y - (yb + slope * (x - xb)))
        .collect();

    let max_y = y_corr.iter().copied().fold(0.0_f64, f64::max);
    if max_y <= 0.0 {
        return Vec::new();
    }

    // When thermal noise is present on an oversampled 1D grid, apply a narrow zero-phase
    // Gaussian matched filter (`sigma = 0.14 Hz`, `FWHM = 0.33 Hz << min_j_hz`) so sample-level
    // noise ripples on broad peak tops do not masquerade as extra splittings; keep exact raw
    // `y_corr` when noiseless.
    let dx_hz = ((width / ((m - 1).max(1) as f64)) * spectrometer_mhz).max(1e-9);
    let y_det: Vec<f64> = if effective_noise_rms > 1e-8 * max_y && dx_hz < 0.25 {
        let sigma_pts = (0.14 / dx_hz).clamp(0.5, 4.5);
        let radius = (3.0 * sigma_pts).ceil() as usize;
        let mut weights = Vec::with_capacity(2 * radius + 1);
        for d in 0..=radius {
            let z = (d as f64) / sigma_pts;
            weights.push((-0.5 * z * z).exp());
        }
        (0..m)
            .map(|i| {
                let mut acc = weights[0] * y_corr[i];
                let mut wsum = weights[0];
                for (d, &w) in weights.iter().enumerate().skip(1) {
                    if i >= d {
                        acc += w * y_corr[i - d];
                        wsum += w;
                    }
                    if i + d < m {
                        acc += w * y_corr[i + d];
                        wsum += w;
                    }
                }
                acc / wsum
            })
            .collect()
    } else {
        y_corr.clone()
    };

    let max_det = y_det.iter().copied().fold(0.0_f64, f64::max).max(1e-30);
    let thresh = (max_det * config.subpeak_rel_threshold)
        .max(effective_noise_rms * config.subpeak_snr_threshold.max(4.8));

    let mut cand_indices: Vec<usize> = Vec::new();
    for i in 1..(m - 1) {
        let yi = y_det[i];
        if yi < thresh {
            continue;
        }
        let is_local_max =
            yi >= y_det[i - 1] && yi > y_det[i + 1] || (yi > y_det[i - 1] && yi >= y_det[i + 1]);
        if is_local_max {
            cand_indices.push(i);
        }
    }

    // Saddle-prominence pruning against thermal noise ripples on flat/broad peak tops
    if effective_noise_rms > 1e-8 * max_y && cand_indices.len() > 1 {
        let min_prom = (4.2 * effective_noise_rms).max(0.004 * max_det);
        loop {
            let mut drop_slot: Option<usize> = None;
            for w in 0..(cand_indices.len() - 1) {
                let ia = cand_indices[w];
                let ib = cand_indices[w + 1];
                let valley = y_det[ia..=ib].iter().copied().fold(f64::INFINITY, f64::min);
                let smaller_h = y_det[ia].min(y_det[ib]);
                if smaller_h - valley < min_prom {
                    drop_slot = Some(if y_det[ia] < y_det[ib] { w } else { w + 1 });
                    break;
                }
            }
            if let Some(slot) = drop_slot {
                cand_indices.remove(slot);
                if cand_indices.len() <= 1 {
                    break;
                }
            } else {
                break;
            }
        }
    }

    let mut raw_picks: Vec<(f64, f64, f64)> = Vec::with_capacity(cand_indices.len()); // (ppm, height, fwhm_hz)
    for (slot, &i) in cand_indices.iter().enumerate() {
        let yi = y_det[i];
        // 3-point quadratic sub-sample interpolation
        let x_m1 = slice[i - 1].0;
        let x_0 = slice[i].0;
        let x_p1 = slice[i + 1].0;
        let dx_avg = 0.5 * (x_p1 - x_m1);
        let denom = y_det[i - 1] - 2.0 * yi + y_det[i + 1];
        let (peak_ppm, peak_h) = if denom < -1e-30 {
            let delta = (0.5 * (y_det[i - 1] - y_det[i + 1]) / denom).clamp(-0.5, 0.5);
            let h = yi - 0.25 * (y_det[i - 1] - y_det[i + 1]) * delta;
            (x_0 + delta * dx_avg, h.max(yi))
        } else {
            (x_0, yi)
        };

        // Estimate half-maximum width in Hz (bounded at true valley minima toward neighbor peaks)
        let left_saddle = if slot > 0 {
            let prev_i = cand_indices[slot - 1];
            (prev_i..=i)
                .min_by(|&a, &b| {
                    y_det[a]
                        .partial_cmp(&y_det[b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(prev_i)
        } else {
            0
        };
        let right_saddle = if slot + 1 < cand_indices.len() {
            let next_i = cand_indices[slot + 1];
            (i..=next_i)
                .min_by(|&a, &b| {
                    y_det[a]
                        .partial_cmp(&y_det[b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(next_i)
        } else {
            m - 1
        };

        let half_h = 0.5 * peak_h;
        let mut left_x = slice[left_saddle].0;
        for j in (left_saddle..i).rev() {
            if y_det[j] <= half_h {
                let frac =
                    ((half_h - y_det[j]) / (y_det[j + 1] - y_det[j]).max(1e-30)).clamp(0.0, 1.0);
                left_x = slice[j].0 + frac * (slice[j + 1].0 - slice[j].0);
                break;
            }
        }
        let mut right_x = slice[right_saddle].0;
        for j in (i + 1)..=right_saddle {
            if y_det[j] <= half_h {
                let frac =
                    ((half_h - y_det[j]) / (y_det[j - 1] - y_det[j]).max(1e-30)).clamp(0.0, 1.0);
                right_x = slice[j].0 - frac * (slice[j].0 - slice[j - 1].0);
                break;
            }
        }
        let fwhm_hz = ((right_x - left_x).abs() * spectrometer_mhz).max(0.15);
        raw_picks.push((peak_ppm, peak_h, fwhm_hz));
    }

    if raw_picks.is_empty() {
        // Fallback to global maximum inside region
        if let Some((best_i, &best_y)) = y_corr
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        {
            if best_y > 0.0 {
                raw_picks.push((slice[best_i].0, best_y, 1.0));
            }
        }
    }

    // Optional multi-Lorentzian cross-talk deconvolution on picked sub-peaks (`2..=32` lines)
    if config.deconvolve_lorentzian_crosstalk && (2..=32).contains(&raw_picks.len()) {
        // Estimate clean outer-edge HWHM from outermost downfield and upfield peaks' outer wings
        let outer_fwhm_hz =
            estimate_outer_wing_fwhm_hz(slice, &y_det, &raw_picks, spectrometer_mhz);
        let hwhm_ppm = (0.5 * outer_fwhm_hz) / spectrometer_mhz;
        if hwhm_ppm > 1e-7 {
            let k = raw_picks.len();
            let mut centers: Vec<f64> = raw_picks.iter().map(|p| p.0).collect();
            let mut amps: Vec<f64> = raw_picks.iter().map(|p| p.1).collect();
            let h2 = hwhm_ppm * hwhm_ppm;

            // Two outer cycles: solve non-negative Lorentzian cross-talk amplitudes + 1D center polish
            for _ in 0..2 {
                let y_at_centers: Vec<f64> = centers
                    .iter()
                    .map(|&cx| interpolate_y_corr(slice, &y_det, cx))
                    .collect();
                for _iter in 0..20 {
                    for i in 0..k {
                        let mut off = 0.0;
                        for j in 0..k {
                            if i != j {
                                let d = centers[i] - centers[j];
                                off += amps[j] * (h2 / (d * d + h2));
                            }
                        }
                        amps[i] = (y_at_centers[i] - off).max(0.0);
                    }
                }

                // Subtract sibling Lorentzian tails around each peak `i` and re-interpolate its apex!
                for i in 0..k {
                    let pos = slice
                        .partition_point(|&(x, _)| x < centers[i])
                        .clamp(1, m - 2);
                    let mut clean3 = [0.0_f64; 3];
                    for (slot, idx_s) in [pos - 1, pos, pos + 1].into_iter().enumerate() {
                        let x_s = slice[idx_s].0;
                        let mut sibling = 0.0;
                        for j in 0..k {
                            if j != i {
                                let d = x_s - centers[j];
                                sibling += amps[j] * (h2 / (d * d + h2));
                            }
                        }
                        clean3[slot] = y_det[idx_s] - sibling;
                    }
                    let denom = clean3[0] - 2.0 * clean3[1] + clean3[2];
                    if denom < -1e-30 && clean3[1] > clean3[0] && clean3[1] > clean3[2] {
                        let dx_avg = 0.5 * (slice[pos + 1].0 - slice[pos - 1].0);
                        let delta = (0.5 * (clean3[0] - clean3[2]) / denom).clamp(-0.5, 0.5);
                        centers[i] = slice[pos].0 + delta * dx_avg;
                    }
                }
            }

            let max_amp = amps.iter().copied().fold(0.0_f64, f64::max).max(1e-30);
            for i in 0..k {
                if amps[i] >= 0.002 * max_amp {
                    raw_picks[i].0 = centers[i];
                    raw_picks[i].1 = amps[i];
                    raw_picks[i].2 = outer_fwhm_hz;
                }
            }
        }
    }

    raw_picks
        .into_iter()
        .filter(|p| p.1 > 0.0)
        .map(|(ppm_v, h_v, fwhm_v)| MultipletSubPeak {
            ppm: ppm_v,
            intensity: h_v,
            fwhm_hz: Some(fwhm_v),
        })
        .collect()
}

fn interpolate_y_corr(slice: &[(f64, f64)], y_corr: &[f64], target_x: f64) -> f64 {
    let n = slice.len();
    if target_x <= slice[0].0 {
        return y_corr[0];
    }
    if target_x >= slice[n - 1].0 {
        return y_corr[n - 1];
    }
    let r = slice.partition_point(|&(x, _)| x < target_x);
    if r == 0 {
        return y_corr[0];
    }
    let l = r - 1;
    let span = slice[r].0 - slice[l].0;
    if span.abs() <= 1e-15 {
        y_corr[l]
    } else {
        let a = ((target_x - slice[l].0) / span).clamp(0.0, 1.0);
        (1.0 - a) * y_corr[l] + a * y_corr[r]
    }
}

fn estimate_outer_wing_fwhm_hz(
    slice: &[(f64, f64)],
    y_corr: &[f64],
    picks: &[(f64, f64, f64)],
    spectrometer_mhz: f64,
) -> f64 {
    // `slice` is ascending in ppm (`picks[0]` is lowest ppm upfield, `picks[last]` is highest ppm downfield)
    let m = slice.len();
    let mut half_widths_hz = Vec::new();

    if let Some(&(low_ppm, low_h, _)) = picks.first() {
        let idx = slice.partition_point(|&(x, _)| x < low_ppm).min(m - 1);
        let half = 0.5 * low_h;
        for j in (0..idx).rev() {
            if y_corr[j] <= half {
                let frac =
                    ((half - y_corr[j]) / (y_corr[j + 1] - y_corr[j]).max(1e-30)).clamp(0.0, 1.0);
                let x_half = slice[j].0 + frac * (slice[j + 1].0 - slice[j].0);
                let hw = (low_ppm - x_half).abs() * spectrometer_mhz;
                if hw > 0.05 {
                    half_widths_hz.push(2.0 * hw);
                }
                break;
            }
        }
    }

    if let Some(&(high_ppm, high_h, _)) = picks.last() {
        let idx = slice.partition_point(|&(x, _)| x < high_ppm).min(m - 1);
        let half = 0.5 * high_h;
        for j in (idx + 1)..m {
            if y_corr[j] <= half {
                let frac =
                    ((half - y_corr[j]) / (y_corr[j - 1] - y_corr[j]).max(1e-30)).clamp(0.0, 1.0);
                let x_half = slice[j].0 - frac * (slice[j].0 - slice[j - 1].0);
                let hw = (x_half - high_ppm).abs() * spectrometer_mhz;
                if hw > 0.05 {
                    half_widths_hz.push(2.0 * hw);
                }
                break;
            }
        }
    }

    if half_widths_hz.is_empty() {
        picks.first().map(|p| p.2).unwrap_or(1.0)
    } else {
        half_widths_hz.iter().sum::<f64>() / (half_widths_hz.len() as f64)
    }
}

/// Recursive Hoye sub-doublet subtraction (Hoye, Hanson & Vyvyan 1994; Hoye & Zhao 2002)
/// generalized for leaning ratio `R = (1 - \eta) / (1 + \eta)`.
fn hoye_recursive_peel(
    rem_hz: &[f64],
    rem_int: &[f64],
    config: &MultipletConfig,
    eta_max: f64,
    current_js: &mut Vec<f64>,
    results: &mut Vec<Vec<f64>>,
) {
    if results.len() >= 16 {
        return;
    }
    if rem_hz.len() == 1 {
        let mut js = current_js.clone();
        js.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        if !results.iter().any(|existing| {
            existing.len() == js.len()
                && existing
                    .iter()
                    .zip(js.iter())
                    .all(|(x, y)| (x - y).abs() < 0.25)
        }) {
            results.push(js);
        }
        return;
    }
    if current_js.len() >= 6 {
        return;
    }

    let span = rem_hz[0] - rem_hz[rem_hz.len() - 1];
    let r_lo = (1.0 - eta_max) / (1.0 + eta_max);
    let r_hi = (1.0 + eta_max) / (1.0 - eta_max);

    // Candidate splittings from the highest-frequency peak `rem_hz[0]` to nearby peaks `1..=min(4, len-1)`
    let max_partner = rem_hz.len().min(5);
    for idx_p in 1..max_partner {
        let j_cand = rem_hz[0] - rem_hz[idx_p];
        if j_cand < config.min_j_hz
            || j_cand > config.max_j_hz
            || j_cand > span + config.freq_tol_hz
        {
            continue;
        }

        // Try leaning ratios: `1.0` (first-order), direct outer pair ratio `rem_int[idx_p]/rem_int[0]`,
        // and root-of-extremes `(rem_int[last]/rem_int[0])^(1/m)` for degenerate triplets/quartets.
        let mut r_cands = vec![1.0_f64];
        if rem_int[0] > 1e-6 {
            let r_direct = (rem_int[idx_p] / rem_int[0]).clamp(r_lo, r_hi);
            r_cands.push(r_direct);
            let outer_ratio = (rem_int[rem_int.len() - 1] / rem_int[0]).max(1e-6);
            for deg in 1..=6 {
                let r_deg = outer_ratio.powf(1.0 / (deg as f64)).clamp(r_lo, r_hi);
                if r_cands.iter().all(|&rc| (rc - r_deg).abs() > 0.03) {
                    r_cands.push(r_deg);
                }
            }
        }

        for r in r_cands {
            if let Some((next_hz, next_int)) =
                peel_single_splitting(rem_hz, rem_int, j_cand, r, config.freq_tol_hz * 1.5)
            {
                current_js.push(j_cand);
                hoye_recursive_peel(&next_hz, &next_int, config, eta_max, current_js, results);
                current_js.pop();
            }
        }
    }
}

fn peel_single_splitting(
    rem_hz: &[f64],
    rem_int: &[f64],
    j_hz: f64,
    ratio_up_over_down: f64,
    tol_hz: f64,
) -> Option<(Vec<f64>, Vec<f64>)> {
    let n = rem_hz.len();
    let mut avail = rem_int.to_vec();
    let total: f64 = avail.iter().sum::<f64>().max(1e-30);
    let prune_eps = 0.025 * total;
    let mut parents: Vec<(f64, f64)> = Vec::new();

    for i in 0..n {
        if avail[i] <= prune_eps {
            continue;
        }
        let w_down = avail[i];
        let target_nu = rem_hz[i] - j_hz;
        // Find closest peak within `tol_hz`
        let mut best_j = None;
        let mut best_dist = tol_hz;
        for (j, &nu_j) in rem_hz.iter().enumerate().take(n).skip(i + 1) {
            let d = (nu_j - target_nu).abs();
            if d <= best_dist {
                best_dist = d;
                best_j = Some(j);
            }
        }
        let partner = best_j?;
        let needed_up = w_down * ratio_up_over_down;
        if avail[partner] + 0.06 * total < needed_up {
            return None;
        }
        avail[i] = 0.0;
        avail[partner] = (avail[partner] - needed_up).max(0.0);
        let parent_nu = 0.5 * (rem_hz[i] + rem_hz[partner]);
        parents.push((parent_nu, w_down + needed_up));
    }

    if avail.iter().any(|&rem| rem > 0.06 * total) || parents.is_empty() {
        return None;
    }

    let merged = merge_close_lines(&parents, tol_hz);
    let sum_p: f64 = merged.iter().map(|p| p.1).sum::<f64>().max(1e-30);
    let out_hz = merged.iter().map(|p| p.0).collect();
    let out_int = merged.iter().map(|p| p.1 / sum_p).collect();
    Some((out_hz, out_int))
}

fn generate_initial_j_seeds(
    cls: MultipletClass,
    obs_hz: &[f64],
    hoye_multisets: &[Vec<f64>],
    config: &MultipletConfig,
) -> Vec<Vec<f64>> {
    let k = obs_hz.len();
    let mults = cls.coupling_multiplicities();
    let g = mults.len();
    if g == 0 || k < 2 {
        return Vec::new();
    }
    let total_splits: usize = mults.iter().sum();
    let max_lines: usize = mults.iter().map(|&m| m + 1).product();
    // Even with accidental inner coalescence, observed lines cannot exceed `max_lines`
    if k > max_lines {
        return Vec::new();
    }

    let span = obs_hz[0] - obs_hz[k - 1];
    if span <= 0.0 {
        return Vec::new();
    }

    let mut seeds: Vec<Vec<f64>> = Vec::new();
    let push_seed = |seeds: &mut Vec<Vec<f64>>, cand: Vec<f64>| {
        if cand.len() != g {
            return;
        }
        if cand
            .iter()
            .any(|&j| !j.is_finite() || j < config.min_j_hz || j > config.max_j_hz)
        {
            return;
        }
        for w in cand.windows(2) {
            if w[0] < w[1] + config.equal_j_tol_hz {
                return;
            }
        }
        if !seeds
            .iter()
            .any(|s| s.iter().zip(cand.iter()).all(|(a, b)| (a - b).abs() < 0.15))
        {
            seeds.push(cand);
        }
    };

    // Closed-form exact outer-line Hoye invariants
    match g {
        1 => {
            push_seed(&mut seeds, vec![span / (mults[0] as f64)]);
        }
        2 => {
            let n1 = mults[0] as f64;
            let n2 = mults[1] as f64;
            let j2_sym = 0.5 * ((obs_hz[0] - obs_hz[1]) + (obs_hz[k - 2] - obs_hz[k - 1]));
            push_seed(&mut seeds, vec![(span - n2 * j2_sym) / n1, j2_sym]);
            for idx in 1..k.min(4) {
                let j2 = obs_hz[0] - obs_hz[idx];
                push_seed(&mut seeds, vec![(span - n2 * j2) / n1, j2]);
            }
        }
        3 => {
            // `ddd`: mults == [1, 1, 1]. In every `ddd` (`J_1 > J_2 > J_3`),
            // `obs_hz[0] - obs_hz[1] == J_3` and `obs_hz[0] - obs_hz[2] == J_2`!
            if k >= 4 {
                let j3 = 0.5 * ((obs_hz[0] - obs_hz[1]) + (obs_hz[k - 2] - obs_hz[k - 1]));
                let j2 = 0.5 * ((obs_hz[0] - obs_hz[2]) + (obs_hz[k - 3] - obs_hz[k - 1]));
                let j1 = span - j2 - j3;
                push_seed(&mut seeds, vec![j1, j2, j3]);
            }
            for i2 in 2..k.min(5) {
                let j3 = obs_hz[0] - obs_hz[1];
                let j2 = obs_hz[0] - obs_hz[i2];
                let j1 = span - j2 - j3;
                push_seed(&mut seeds, vec![j1, j2, j3]);
            }
        }
        _ => {}
    }

    // Seeds condensed from Hoye recursive peeling multisets of matching total elementary splits
    for ms in hoye_multisets {
        if ms.len() == total_splits {
            let mut offset = 0;
            let mut grouped = Vec::with_capacity(g);
            for &m_count in mults {
                let sub = &ms[offset..(offset + m_count)];
                let avg = sub.iter().sum::<f64>() / (m_count as f64);
                grouped.push(avg);
                offset += m_count;
            }
            push_seed(&mut seeds, grouped);
        }
    }

    seeds
}

#[derive(Clone)]
struct SubTransition {
    coeffs: [f64; 3],
    signs: [i8; 3], // sum of +1/-1 flips for each group `g` (in `-n_g..=+n_g`)
    binom_weight: f64,
}

fn build_elementary_transitions(mults: &[usize]) -> Vec<SubTransition> {
    let mut trans = vec![SubTransition {
        coeffs: [0.0; 3],
        signs: [0; 3],
        binom_weight: 1.0,
    }];
    for (g, &n_eq) in mults.iter().enumerate() {
        let mut next = Vec::with_capacity(trans.len() * (n_eq + 1));
        for k in 0..=n_eq {
            // `k` upfield flips (-J/2) and `n_eq - k` downfield flips (+J/2)
            let net_flips = (n_eq as isize) - 2 * (k as isize); // in `-n_eq..=+n_eq`
            let coeff = 0.5 * (net_flips as f64);
            let binom = binomial_coeff(n_eq, k) as f64;
            for t in &trans {
                let mut c = t.coeffs;
                let mut s = t.signs;
                c[g] = coeff;
                s[g] = net_flips as i8;
                next.push(SubTransition {
                    coeffs: c,
                    signs: s,
                    binom_weight: t.binom_weight * binom,
                });
            }
        }
        trans = next;
    }
    trans
}

fn binomial_coeff(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut res = 1usize;
    for i in 0..k {
        res = res * (n - i) / (i + 1);
    }
    res
}

fn evaluate_forward_coalesced(
    center_hz: f64,
    js: &[f64],
    etas: &[f64],
    linear_beta: f64,
    mults: &[usize],
    transitions: &[SubTransition],
    coalesce_hz: f64,
) -> Vec<(f64, f64, [f64; 3])> {
    let g = js.len();
    let mut raw: Vec<(f64, f64, [f64; 3])> = Vec::with_capacity(transitions.len());
    for tr in transitions {
        let mut offset = 0.0;
        let mut w = tr.binom_weight;
        for idx in 0..g {
            offset += tr.coeffs[idx] * js[idx];
            let n_eq = mults[idx] as i32;
            let net = tr.signs[idx] as i32;
            let n_plus = (n_eq + net) / 2;
            let n_minus = (n_eq - net) / 2;
            let eta = etas[idx];
            w *= (0.5 * (1.0 + eta)).powi(n_plus) * (0.5 * (1.0 - eta)).powi(n_minus);
        }
        if linear_beta != 0.0 {
            w *= (1.0 + linear_beta * offset).max(0.02);
        }
        raw.push((center_hz + offset, w, tr.coeffs));
    }

    raw.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut merged: Vec<(f64, f64, [f64; 3])> = Vec::with_capacity(raw.len());
    for (nu, w, c) in raw {
        if let Some(last) = merged.last_mut() {
            if (last.0 - nu).abs() <= coalesce_hz {
                let tot = last.1 + w;
                if tot > 0.0 {
                    last.0 = (last.0 * last.1 + nu * w) / tot;
                    for (slot, &c_val) in c.iter().enumerate() {
                        last.2[slot] = (last.2[slot] * last.1 + c_val * w) / tot;
                    }
                }
                last.1 = tot;
                continue;
            }
        }
        merged.push((nu, w, c));
    }

    let sum_w: f64 = merged.iter().map(|m| m.1).sum::<f64>().max(1e-30);
    for item in &mut merged {
        item.1 /= sum_w;
    }
    merged
}

fn fit_multiplet_hypothesis(
    cls: MultipletClass,
    seed_js: &[f64],
    obs_hz: &[f64],
    obs_norm: &[f64],
    eta_max: f64,
    coalesce_hz: f64,
    config: &MultipletConfig,
) -> Option<EvaluatedTreeFit> {
    let mults = cls.coupling_multiplicities();
    let g = mults.len();
    let k = obs_hz.len();
    let transitions = build_elementary_transitions(mults);

    let mut center_hz = 0.5 * (obs_hz[0] + obs_hz[k - 1]);
    let mut js = seed_js.to_vec();
    let mut etas = vec![0.0_f64; g];
    let mut linear_beta = 0.0_f64;

    // Alternating Gauss-Markov position/J refinement and bounded leaning refinement
    for _outer in 0..3 {
        let pred = evaluate_forward_coalesced(
            center_hz,
            &js,
            &etas,
            linear_beta,
            mults,
            &transitions,
            coalesce_hz,
        );
        if pred.len() != k {
            return None;
        }

        // Solve `(1 + g) x (1 + g)` weighted linear least squares for `(center_hz, js[0..g])`
        let dim = 1 + g;
        let mut ata = vec![vec![0.0_f64; dim]; dim];
        let mut atb = vec![0.0_f64; dim];
        for i in 0..k {
            let wt = obs_norm[i].max(0.02);
            let mut row = [0.0_f64; 4];
            row[0] = 1.0;
            row[1..(g + 1)].copy_from_slice(&pred[i].2[..g]);
            for r in 0..dim {
                atb[r] += wt * row[r] * obs_hz[i];
                for c in 0..dim {
                    ata[r][c] += wt * row[r] * row[c];
                }
            }
        }
        if let Some(sol) = solve_small_linear(&ata, &atb) {
            center_hz = sol[0];
            js[..g].copy_from_slice(&sol[1..(g + 1)]);
        }

        // Validate monotonic coupling ordering `J_1 > J_2 > ... >= min_j_hz`
        if js
            .iter()
            .any(|&j| !j.is_finite() || j < config.min_j_hz || j > config.max_j_hz)
        {
            return None;
        }
        for w in js.windows(2) {
            if w[0] < w[1] + config.equal_j_tol_hz {
                return None;
            }
        }

        // Coordinate optimization of per-coupling leaning `etas[0..g] \in [-eta_max, +eta_max]`
        let mut best_etas = etas.clone();
        let best_beta = 0.0_f64;
        let mut best_ir = intensity_rmsd_for(
            center_hz,
            &js,
            &best_etas,
            best_beta,
            mults,
            &transitions,
            coalesce_hz,
            obs_norm,
        )?;

        for step in [0.20, 0.06, 0.015, 0.003] {
            for _sweep in 0..4 {
                for idx in 0..g {
                    for dir in [-1.0, 1.0] {
                        let cand_eta = (best_etas[idx] + dir * step).clamp(-eta_max, eta_max);
                        let mut trial = best_etas.clone();
                        trial[idx] = cand_eta;
                        if let Some(ir) = intensity_rmsd_for(
                            center_hz,
                            &js,
                            &trial,
                            0.0,
                            mults,
                            &transitions,
                            coalesce_hz,
                            obs_norm,
                        ) {
                            // Tiny L2 tie-breaker (`1e-5 * sum eta^2`) keeps `eta == 0` on symmetric inputs
                            if ir + 1e-6 * cand_eta * cand_eta
                                < best_ir + 1e-6 * best_etas[idx] * best_etas[idx]
                            {
                                best_ir = ir;
                                best_etas = trial;
                            }
                        }
                    }
                }
            }
        }

        // Also check global linear roofing envelope `I_0 * (1 + beta * d_nu)` and adopt if cleaner
        let span = (obs_hz[0] - obs_hz[k - 1]).abs().max(1.0);
        let beta_max = (2.0 * eta_max / span.max(js[0])).min(1.7 / span);
        let zero_etas = vec![0.0_f64; g];
        let mut best_lin_beta = 0.0_f64;
        let mut best_lin_ir = best_ir;
        for step_frac in [0.25, 0.07, 0.015] {
            let step = step_frac * beta_max;
            for _ in 0..6 {
                for dir in [-1.0, 1.0] {
                    let cb = (best_lin_beta + dir * step).clamp(-beta_max, beta_max);
                    if let Some(ir) = intensity_rmsd_for(
                        center_hz,
                        &js,
                        &zero_etas,
                        cb,
                        mults,
                        &transitions,
                        coalesce_hz,
                        obs_norm,
                    ) {
                        if ir + 1e-5 < best_lin_ir {
                            best_lin_ir = ir;
                            best_lin_beta = cb;
                        }
                    }
                }
            }
        }

        if best_lin_ir + 0.005 < best_ir {
            etas = zero_etas;
            linear_beta = best_lin_beta;
        } else {
            etas = best_etas;
            linear_beta = 0.0;
        }
    }

    let final_pred = evaluate_forward_coalesced(
        center_hz,
        &js,
        &etas,
        linear_beta,
        mults,
        &transitions,
        coalesce_hz,
    );
    if final_pred.len() != k {
        return None;
    }

    let mut max_freq_err = 0.0_f64;
    let mut sum_f2 = 0.0_f64;
    let mut sum_i2 = 0.0_f64;
    for i in 0..k {
        let df = (final_pred[i].0 - obs_hz[i]).abs();
        max_freq_err = max_freq_err.max(df);
        sum_f2 += df * df;
        let di = final_pred[i].1 - obs_norm[i];
        sum_i2 += di * di;
    }
    let freq_rmsd_hz = (sum_f2 / (k as f64)).sqrt();
    let intensity_rmsd = (sum_i2 / (k as f64)).sqrt();

    if freq_rmsd_hz > config.freq_tol_hz
        || max_freq_err > 1.75 * config.freq_tol_hz
        || intensity_rmsd > config.intensity_rmsd_tol
    {
        return None;
    }

    // Build `CouplingConstant` entries & `LeaningSummary`
    let mut couplings = Vec::with_capacity(g);
    let mut strongest_abs_eta = 0.0_f64;
    let mut strongest_signed_eta = 0.0_f64;

    for idx in 0..g {
        let eff_eta = if etas[idx].abs() > 1e-6 {
            etas[idx]
        } else if linear_beta.abs() > 1e-6 {
            (0.5 * linear_beta * js[idx]).clamp(-eta_max, eta_max)
        } else {
            0.0
        };
        if eff_eta.abs() > strongest_abs_eta {
            strongest_abs_eta = eff_eta.abs();
            strongest_signed_eta = eff_eta;
        }
        let (dir, dnu_j) = if eff_eta.abs() >= 0.04 {
            let d = if eff_eta > 0.0 {
                LeaningDirection::Downfield
            } else {
                LeaningDirection::Upfield
            };
            let cot2t = (1.0 - eff_eta * eff_eta).max(0.0).sqrt() / eff_eta.abs();
            (d, Some(cot2t))
        } else {
            (LeaningDirection::None, None)
        };
        couplings.push(CouplingConstant {
            j_hz: js[idx],
            equivalent_partners: mults[idx],
            leaning_sin2theta: eff_eta,
            estimated_delta_nu_over_j: dnu_j,
            leaning_direction: dir,
        });
    }

    let is_roofed = strongest_abs_eta >= 0.04;
    let dominant_direction = if !is_roofed {
        LeaningDirection::None
    } else if strongest_signed_eta > 0.0 {
        LeaningDirection::Downfield
    } else {
        LeaningDirection::Upfield
    };
    let inner_to_outer_ratio = if is_roofed {
        (1.0 + strongest_abs_eta) / (1.0 - strongest_abs_eta).max(1e-6)
    } else {
        1.0
    };
    let estimated_delta_nu_over_j = if is_roofed {
        Some(
            (1.0 - strongest_abs_eta * strongest_abs_eta)
                .max(0.0)
                .sqrt()
                / strongest_abs_eta,
        )
    } else {
        None
    };

    // Occam parsimony penalty (`0.008 * g`) prefers e.g. `t` (1 J) over `dd` (2 Js) when J1 ~= J2
    let selection_loss = (freq_rmsd_hz / config.freq_tol_hz.max(0.05)).powi(2)
        + (intensity_rmsd / config.intensity_rmsd_tol.max(0.02)).powi(2)
        + 0.015 * (g as f64);

    Some(EvaluatedTreeFit {
        class: cls,
        center_hz,
        couplings,
        freq_rmsd_hz,
        intensity_rmsd,
        linear_beta,
        leaning: LeaningSummary {
            is_roofed,
            dominant_direction,
            inner_to_outer_ratio,
            estimated_delta_nu_over_j,
            linear_roofing_slope_per_hz: linear_beta,
        },
        selection_loss,
    })
}

#[allow(clippy::too_many_arguments)]
fn intensity_rmsd_for(
    center_hz: f64,
    js: &[f64],
    etas: &[f64],
    linear_beta: f64,
    mults: &[usize],
    transitions: &[SubTransition],
    coalesce_hz: f64,
    obs_norm: &[f64],
) -> Option<f64> {
    let pred = evaluate_forward_coalesced(
        center_hz,
        js,
        etas,
        linear_beta,
        mults,
        transitions,
        coalesce_hz,
    );
    if pred.len() != obs_norm.len() {
        return None;
    }
    let rss: f64 = pred
        .iter()
        .zip(obs_norm.iter())
        .map(|(p, &o)| (p.1 - o).powi(2))
        .sum();
    Some((rss / (obs_norm.len() as f64)).sqrt())
}

fn compute_ab_zeeman_ppm(
    center_hz: f64,
    couplings: &[CouplingConstant],
    spectrometer_mhz: f64,
) -> f64 {
    // Exact AB second-order shift correction along each strongly roofed coupling:
    // When partner is downfield (`eta > 0`), the observed multiplet midpoint is pushed upfield
    // (away from partner) by `(J / 2) * (\sqrt{R} - 1) / (\sqrt{R} + 1)`, so Zeeman shift adds `+corr`.
    // When partner is upfield (`eta < 0`), midpoint is pushed downfield, so Zeeman shift adds `-corr`.
    let mut shift_hz = center_hz;
    for c in couplings {
        let eta = c.leaning_sin2theta;
        if eta.abs() >= 0.04 && eta.abs() < 0.99 {
            let r = (1.0 + eta.abs()) / (1.0 - eta.abs());
            let sr = r.sqrt();
            let mag = 0.5 * c.j_hz * (sr - 1.0) / (sr + 1.0);
            shift_hz += eta.signum() * mag * (c.equivalent_partners as f64).sqrt();
        }
    }
    shift_hz / spectrometer_mhz
}

fn build_tree_levels(
    center_hz: f64,
    couplings: &[CouplingConstant],
    linear_beta: f64,
) -> Vec<MultipletTreeLevel> {
    let mut levels = Vec::with_capacity(couplings.len() + 1);
    let mut current = vec![(center_hz, 1.0_f64)];
    levels.push(MultipletTreeLevel {
        depth: 0,
        applied_j_hz: None,
        equivalent_partners: 0,
        nodes_hz: current.clone(),
    });

    for (idx, c) in couplings.iter().enumerate() {
        let eta = c.leaning_sin2theta;
        for _ in 0..c.equivalent_partners {
            let mut next = Vec::with_capacity(current.len() * 2);
            for &(nu, w) in &current {
                next.push((nu + 0.5 * c.j_hz, 0.5 * w * (1.0 + eta)));
                next.push((nu - 0.5 * c.j_hz, 0.5 * w * (1.0 - eta)));
            }
            current = merge_close_lines(&next, 0.02);
        }
        let mut level_nodes = current.clone();
        if linear_beta != 0.0 && idx + 1 == couplings.len() {
            for nd in &mut level_nodes {
                nd.1 *= (1.0 + linear_beta * (nd.0 - center_hz)).max(0.02);
            }
            let s: f64 = level_nodes.iter().map(|n| n.1).sum::<f64>().max(1e-30);
            for nd in &mut level_nodes {
                nd.1 /= s;
            }
        }
        levels.push(MultipletTreeLevel {
            depth: idx + 1,
            applied_j_hz: Some(c.j_hz),
            equivalent_partners: c.equivalent_partners,
            nodes_hz: level_nodes,
        });
    }
    levels
}

fn expand_elementary_js(couplings: &[CouplingConstant]) -> Vec<f64> {
    let mut out = Vec::new();
    for c in couplings {
        for _ in 0..c.equivalent_partners {
            out.push(c.j_hz);
        }
    }
    out
}

fn estimate_global_linear_slope(obs_hz: &[f64], obs_norm: &[f64], center_hz: f64) -> f64 {
    // Under symmetric first-order intensities, `\sum a_i (\nu_i - \nu_0) == 0`; under linear tilt
    // `a_i = a_i^{(0)} (1 + \beta \delta\nu_i)`, `\sum a_i \delta\nu_i = \beta \sum a_i^{(0)} \delta\nu_i^2`.
    let mid = 0.5 * (obs_hz[0] + obs_hz[obs_hz.len() - 1]);
    let m1: f64 = obs_hz
        .iter()
        .zip(obs_norm.iter())
        .map(|(&nu, &a)| a * (nu - mid))
        .sum();
    let m2: f64 = obs_hz
        .iter()
        .zip(obs_norm.iter())
        .map(|(&nu, &a)| a * (nu - center_hz).powi(2))
        .sum();
    if m2 > 1e-12 {
        m1 / m2
    } else {
        0.0
    }
}

fn merge_close_lines(lines: &[(f64, f64)], tol_hz: f64) -> Vec<(f64, f64)> {
    let mut sorted = lines.to_vec();
    sorted.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut out: Vec<(f64, f64)> = Vec::with_capacity(sorted.len());
    for (nu, w) in sorted {
        if let Some(last) = out.last_mut() {
            if (last.0 - nu).abs() <= tol_hz {
                let tot = last.1 + w;
                if tot > 0.0 {
                    last.0 = (last.0 * last.1 + nu * w) / tot;
                }
                last.1 = tot;
                continue;
            }
        }
        out.push((nu, w));
    }
    out
}

fn solve_small_linear(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = b.len();
    let mut aug: Vec<Vec<f64>> = (0..n)
        .map(|r| {
            let mut row = a[r].clone();
            row.push(b[r]);
            row
        })
        .collect();

    for col in 0..n {
        let mut piv = col;
        let mut max_v = aug[col][col].abs();
        for (r, row_r) in aug.iter().enumerate().skip(col + 1) {
            if row_r[col].abs() > max_v {
                max_v = row_r[col].abs();
                piv = r;
            }
        }
        if max_v <= 1e-14 {
            return None;
        }
        aug.swap(col, piv);
        let diag = aug[col][col];
        for v in aug[col].iter_mut().skip(col) {
            *v /= diag;
        }
        let piv_row = aug[col].clone();
        for (r, row_r) in aug.iter_mut().enumerate() {
            if r != col {
                let f = row_r[col];
                if f != 0.0 {
                    for c in col..=n {
                        row_r[c] -= f * piv_row[c];
                    }
                }
            }
        }
    }
    Some(aug.into_iter().map(|row| row[n]).collect())
}

fn jacobi_symmetric_eigen(matrix: &[Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = matrix.len();
    let mut a = matrix.to_vec();
    let mut v = vec![vec![0.0_f64; n]; n];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    if n <= 1 {
        let evals = a
            .first()
            .and_then(|r| r.first())
            .copied()
            .into_iter()
            .collect();
        return (evals, v);
    }

    for _sweep in 0..80 {
        let mut max_off = 0.0_f64;
        let mut p = 0;
        let mut q = 1;
        for (i, row_i) in a.iter().enumerate() {
            for (j, &val) in row_i.iter().enumerate().skip(i + 1) {
                if val.abs() > max_off {
                    max_off = val.abs();
                    p = i;
                    q = j;
                }
            }
        }
        if max_off < 1e-14 {
            break;
        }
        let app = a[p][p];
        let aqq = a[q][q];
        let apq = a[p][q];
        let tau = (aqq - app) / (2.0 * apq);
        let t = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;

        a[p][p] = app - t * apq;
        a[q][q] = aqq + t * apq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;

        #[allow(clippy::needless_range_loop)]
        for r in 0..n {
            if r != p && r != q {
                let arp = a[r][p];
                let arq = a[r][q];
                a[r][p] = c * arp - s * arq;
                a[p][r] = a[r][p];
                a[r][q] = s * arp + c * arq;
                a[q][r] = a[r][q];
            }
        }
        for row_v in &mut v {
            let vip = row_v[p];
            let viq = row_v[q];
            row_v[p] = c * vip - s * viq;
            row_v[q] = s * vip + c * viq;
        }
    }

    let evals: Vec<f64> = (0..n).map(|i| a[i][i]).collect();
    (evals, v)
}

// ============================================================================
// Section 3: Publication Journal Report Formatter & Parser
// (`ACS`, `JOC`, `RSC`, `Nature` x `Unicode`, `Ascii`, `Latex`)
// ============================================================================

/// Publisher characterization formatting preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum PublicationStyle {
    /// American Chemical Society general style (`¹H NMR (500 MHz, CDCl₃) δ 7.28 (dd, J = 8.4, 1.8 Hz, 2H), ...`).
    #[default]
    Acs,
    /// *Journal of Organic Chemistry* style (`¹H NMR (500 MHz, CDCl₃) δ ...` and explicit `¹³C{¹H} NMR` decoupling).
    Joc,
    /// Royal Society of Chemistry style (`¹H NMR (500 MHz, CDCl₃) δH 7.28 (2H, dd, J = 8.4, 1.8 Hz), ...`).
    Rsc,
    /// *Nature* / *Nature Chemistry* style (`¹H NMR (500 MHz, CDCl₃): δ = 7.28 (dd, J = 8.4, 1.8 Hz, 2H), ...`).
    Nature,
}

/// Output character encoding for isotopes, chemical formulas, `\delta`, and multiplet ranges.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum TextEncoding {
    /// Proper Unicode superscripts/subscripts, `\u{03B4}` (`δ`), and `\u{2013}` (`–`).
    #[default]
    Unicode,
    /// Plain 7-bit ASCII (`1H`, `13C`, `CDCl3`, `delta`, `-`).
    Ascii,
    /// LaTeX math/chem markup (`$^{1}\text{H}$`, `$\text{CDCl}_{3}$`, `$\delta$`, `$J$`, `--`).
    Latex,
}

/// Observed NMR nucleus in a journal characterization report.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum ReportNucleus {
    #[default]
    H1,
    C13,
    F19,
    P31,
    N15,
}

impl ReportNucleus {
    pub fn mass_and_symbol(self) -> (u16, &'static str) {
        match self {
            Self::H1 => (1, "H"),
            Self::C13 => (13, "C"),
            Self::F19 => (19, "F"),
            Self::P31 => (31, "P"),
            Self::N15 => (15, "N"),
        }
    }

    pub fn default_shift_decimals(self) -> usize {
        match self {
            Self::H1 | Self::F19 => 2,
            Self::C13 | Self::P31 | Self::N15 => 1,
        }
    }

    pub fn from_ascii_token(tok: &str) -> Option<Self> {
        match tok.trim().to_ascii_uppercase().as_str() {
            "1H" | "H1" | "H" => Some(Self::H1),
            "13C" | "C13" | "C" => Some(Self::C13),
            "19F" | "F19" | "F" => Some(Self::F19),
            "31P" | "P31" | "P" => Some(Self::P31),
            "15N" | "N15" | "N" => Some(Self::N15),
            _ => None,
        }
    }
}

/// Single resonance entry in a publication NMR report line.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReportedSignal {
    /// Primary chemical shift `\delta` in ppm.
    pub shift_ppm: f64,
    /// Optional chemical shift range `(high_ppm, low_ppm)` for unresolved `m` envelopes.
    pub shift_range_ppm: Option<(f64, f64)>,
    /// Multiplicity (`Some(...)` for `1H` or coupled heteronuclei; `None` for standard decoupled `13C`).
    pub multiplicity: Option<MultipletClass>,
    /// Scalar couplings `J` in Hz (sorted largest to smallest).
    pub couplings_hz: Vec<f64>,
    /// Integer spin count (e.g. `2` -> `2H`).
    pub proton_count: Option<usize>,
    /// Optional assignment annotation (e.g. `"ArH"`, `"OCH3"`, `"C-4"`).
    pub assignment: Option<String>,
}

impl ReportedSignal {
    /// Construct a `ReportedSignal` directly from an analyzed [`MultipletAnalysis`].
    pub fn from_analysis(
        analysis: &MultipletAnalysis,
        proton_count: Option<usize>,
        assignment: Option<String>,
    ) -> Self {
        let shift_range_ppm = if analysis.class == MultipletClass::Multiplet
            && (analysis.range_ppm.1 - analysis.range_ppm.0).abs() >= 0.03
        {
            Some((analysis.range_ppm.1, analysis.range_ppm.0))
        } else {
            None
        };
        let couplings_hz = if matches!(
            analysis.class,
            MultipletClass::Singlet | MultipletClass::BroadSinglet | MultipletClass::Multiplet
        ) {
            Vec::new()
        } else {
            analysis.j_values_hz()
        };
        Self {
            shift_ppm: analysis.center_ppm,
            shift_range_ppm,
            multiplicity: Some(analysis.class),
            couplings_hz,
            proton_count: proton_count.or(analysis.proton_count),
            assignment,
        }
    }

    /// Construct a decoupled carbon/heteronuclear shift entry.
    pub fn new_decoupled(shift_ppm: f64) -> Self {
        Self {
            shift_ppm,
            shift_range_ppm: None,
            multiplicity: None,
            couplings_hz: Vec::new(),
            proton_count: None,
            assignment: None,
        }
    }
}

/// Complete structured NMR journal characterization statement (`1H`, `13C`, `19F`, `31P`, `15N`)
/// supporting lossless formatting and parsing across `ACS`, `JOC`, `RSC`, and `Nature` styles.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JournalNmrReport {
    pub nucleus: ReportNucleus,
    /// When `true` on heteronuclei (e.g. `13C`), indicates `{1H}` broadband decoupling.
    pub proton_decoupled: bool,
    /// Spectrometer observe frequency in MHz (e.g. `500.0` for `1H`, `126.0` for `13C`).
    pub mhz: f64,
    /// Canonical ASCII solvent identifier (e.g. `"CDCl3"`, `"DMSO-d6"`, `"D2O"`, `"CD3OD"`).
    pub solvent: String,
    /// Signals ordered from downfield (high `\delta`) to upfield (low `\delta`).
    pub signals: Vec<ReportedSignal>,
}

impl JournalNmrReport {
    /// Format this NMR report into publication-ready text under `style` and `encoding`.
    pub fn format(&self, style: PublicationStyle, encoding: TextEncoding) -> String {
        format_journal_nmr_report(self, style, encoding)
    }

    /// Parse a publication NMR characterization line (`ACS`, `JOC`, `RSC`, or `Nature` in
    /// Unicode, plain ASCII, or LaTeX) back into a structured [`JournalNmrReport`].
    pub fn parse(text: &str) -> Result<Self, MultipletError> {
        parse_journal_nmr_report(text)
    }
}

/// Format a [`JournalNmrReport`] in `ACS`, `JOC`, `RSC`, or `Nature` style using Unicode, ASCII, or LaTeX.
pub fn format_journal_nmr_report(
    report: &JournalNmrReport,
    style: PublicationStyle,
    encoding: TextEncoding,
) -> String {
    let mut signals = report.signals.clone();
    signals.sort_by(|a, b| {
        b.shift_ppm
            .partial_cmp(&a.shift_ppm)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let nuc_str = format_nucleus_label(report.nucleus, report.proton_decoupled, style, encoding);
    let mhz_str = if (report.mhz - report.mhz.round()).abs() < 0.05 {
        format!("{}", report.mhz.round() as i64)
    } else {
        format!("{:.1}", report.mhz)
    };
    let solv_str = format_solvent_label(&report.solvent, encoding);
    let delta_str = format_delta_leader(report.nucleus, style, encoding);
    let decimals = report.nucleus.default_shift_decimals();

    let formatted_items: Vec<String> = signals
        .iter()
        .map(|sig| format_single_signal(sig, report.nucleus, decimals, style, encoding))
        .collect();

    match style {
        PublicationStyle::Acs | PublicationStyle::Joc => {
            format!(
                "{nuc_str} NMR ({mhz_str} MHz, {solv_str}) {delta_str} {}.",
                formatted_items.join(", ")
            )
        }
        PublicationStyle::Rsc => {
            format!(
                "{nuc_str} NMR ({mhz_str} MHz, {solv_str}) {delta_str} {}.",
                formatted_items.join(", ")
            )
        }
        PublicationStyle::Nature => {
            format!(
                "{nuc_str} NMR ({mhz_str} MHz, {solv_str}): {delta_str} = {}.",
                formatted_items.join(", ")
            )
        }
    }
}

/// Parse a formatted `1H` / `13C` / `19F` / `31P` / `15N` NMR characterization line
/// (`ACS`, `JOC`, `RSC`, or `Nature` across Unicode, ASCII, or LaTeX) into a [`JournalNmrReport`].
pub fn parse_journal_nmr_report(text: &str) -> Result<JournalNmrReport, MultipletError> {
    let norm = normalize_report_markup_to_ascii(text);
    let trimmed = norm.trim();
    if trimmed.is_empty() {
        return Err(MultipletError::ParseError(
            "empty NMR report string".to_string(),
        ));
    }

    // Locate `NMR` header and parenthesized `(500 MHz, CDCl3)` metadata
    let nmr_pos = trimmed
        .find("NMR")
        .ok_or_else(|| MultipletError::ParseError("missing 'NMR' header token".to_string()))?;
    let nuc_raw = trimmed[..nmr_pos].trim();
    let (nucleus, proton_decoupled) = parse_nucleus_header(nuc_raw)?;

    let after_nmr = trimmed[(nmr_pos + 3)..].trim_start();
    let open_p = after_nmr
        .find('(')
        .ok_or_else(|| MultipletError::ParseError("missing '(' around MHz/solvent".to_string()))?;
    let close_p = after_nmr[open_p..]
        .find(')')
        .map(|i| open_p + i)
        .ok_or_else(|| MultipletError::ParseError("missing ')' around MHz/solvent".to_string()))?;

    let meta_inside = &after_nmr[(open_p + 1)..close_p];
    let (mhz, solvent) = parse_mhz_and_solvent(meta_inside)?;

    // Advance past `:`, `delta_H`, `deltaH`, `delta`, `=`
    let mut body = after_nmr[(close_p + 1)..].trim();
    if let Some(rest) = body.strip_prefix(':') {
        body = rest.trim_start();
    }
    for prefix in [
        "delta_H", "delta_C", "delta_F", "delta_P", "delta_N", "deltaH", "deltaC", "deltaF",
        "deltaP", "deltaN", "delta",
    ] {
        if let Some(rest) = body.strip_prefix(prefix) {
            body = rest.trim_start();
            break;
        }
    }
    if let Some(rest) = body.strip_prefix('=') {
        body = rest.trim_start();
    }
    body = body.trim_end_matches('.').trim();
    for suffix in ["p.p.m", "ppm"] {
        if let Some(rest) = body.strip_suffix(suffix) {
            body = rest.trim_end_matches('.').trim();
        }
    }

    if body.is_empty() {
        return Ok(JournalNmrReport {
            nucleus,
            proton_decoupled,
            mhz,
            solvent,
            signals: Vec::new(),
        });
    }

    let raw_items = split_top_level_commas(body);
    let mut signals = Vec::with_capacity(raw_items.len());
    for item in raw_items {
        let item_t = item.trim();
        if !item_t.is_empty() {
            signals.push(parse_single_signal_entry(item_t)?);
        }
    }

    Ok(JournalNmrReport {
        nucleus,
        proton_decoupled,
        mhz,
        solvent,
        signals,
    })
}

fn format_nucleus_label(
    nucleus: ReportNucleus,
    proton_decoupled: bool,
    style: PublicationStyle,
    encoding: TextEncoding,
) -> String {
    let (mass, sym) = nucleus.mass_and_symbol();
    let show_dec = proton_decoupled
        && nucleus != ReportNucleus::H1
        && (style == PublicationStyle::Joc || proton_decoupled);
    match encoding {
        TextEncoding::Unicode => {
            let sup = to_unicode_superscript(&mass.to_string());
            if show_dec {
                format!("{sup}{sym}{{¹H}}")
            } else {
                format!("{sup}{sym}")
            }
        }
        TextEncoding::Ascii => {
            if show_dec {
                format!("{mass}{sym}{{1H}}")
            } else {
                format!("{mass}{sym}")
            }
        }
        TextEncoding::Latex => {
            if show_dec {
                format!("$^{{{mass}}}\\text{{{sym}}}\\{{^{{1}}\\text{{H}}\\}}$")
            } else {
                format!("$^{{{mass}}}\\text{{{sym}}}$")
            }
        }
    }
}

fn format_solvent_label(solvent: &str, encoding: TextEncoding) -> String {
    let canon = canonicalize_solvent_ascii(solvent);
    match encoding {
        TextEncoding::Ascii => canon,
        TextEncoding::Unicode => match canon.as_str() {
            "CDCl3" => "CDCl₃".to_string(),
            "DMSO-d6" => "DMSO-d₆".to_string(),
            "D2O" => "D₂O".to_string(),
            "CD3OD" => "CD₃OD".to_string(),
            "CD3CN" => "CD₃CN".to_string(),
            "C6D6" => "C₆D₆".to_string(),
            "CD2Cl2" => "CD₂Cl₂".to_string(),
            "acetone-d6" => "acetone-d₆".to_string(),
            "THF-d8" => "THF-d₈".to_string(),
            "toluene-d8" => "toluene-d₈".to_string(),
            _ => canon,
        },
        TextEncoding::Latex => match canon.as_str() {
            "CDCl3" => "$\\text{CDCl}_{3}$".to_string(),
            "DMSO-d6" => "$\\text{DMSO-}d_{6}$".to_string(),
            "D2O" => "$\\text{D}_{2}\\text{O}$".to_string(),
            "CD3OD" => "$\\text{CD}_{3}\\text{OD}$".to_string(),
            "CD3CN" => "$\\text{CD}_{3}\\text{CN}$".to_string(),
            "C6D6" => "$\\text{C}_{6}\\text{D}_{6}$".to_string(),
            "CD2Cl2" => "$\\text{CD}_{2}\\text{Cl}_{2}$".to_string(),
            "acetone-d6" => "$\\text{acetone-}d_{6}$".to_string(),
            "THF-d8" => "$\\text{THF-}d_{8}$".to_string(),
            "toluene-d8" => "$\\text{toluene-}d_{8}$".to_string(),
            _ => canon,
        },
    }
}

fn canonicalize_solvent_ascii(raw: &str) -> String {
    let ascii = replace_unicode_sub_super(raw).trim().to_string();
    let up = ascii.to_ascii_uppercase();
    match up.as_str() {
        "CDCL3" | "CHLOROFORM-D" => "CDCl3".to_string(),
        "DMSO-D6" | "(CD3)2SO" | "DMSO" => "DMSO-d6".to_string(),
        "D2O" => "D2O".to_string(),
        "CD3OD" | "MEOD" | "METHANOL-D4" => "CD3OD".to_string(),
        "CD3CN" | "ACETONITRILE-D3" => "CD3CN".to_string(),
        "C6D6" | "BENZENE-D6" => "C6D6".to_string(),
        "CD2CL2" => "CD2Cl2".to_string(),
        "ACETONE-D6" | "(CD3)2CO" => "acetone-d6".to_string(),
        "THF-D8" => "THF-d8".to_string(),
        "TOLUENE-D8" => "toluene-d8".to_string(),
        _ => ascii,
    }
}

fn format_delta_leader(
    nucleus: ReportNucleus,
    style: PublicationStyle,
    encoding: TextEncoding,
) -> String {
    let (_, sym) = nucleus.mass_and_symbol();
    match (style, encoding) {
        (PublicationStyle::Rsc, TextEncoding::Unicode) => format!("\u{03B4}{sym}"),
        (PublicationStyle::Rsc, TextEncoding::Ascii) => format!("delta_{sym}"),
        (PublicationStyle::Rsc, TextEncoding::Latex) => format!("$\\delta_{{\\text{{{sym}}}}}$"),
        (_, TextEncoding::Unicode) => "\u{03B4}".to_string(),
        (_, TextEncoding::Ascii) => "delta".to_string(),
        (_, TextEncoding::Latex) => "$\\delta$".to_string(),
    }
}

fn format_single_signal(
    sig: &ReportedSignal,
    nucleus: ReportNucleus,
    decimals: usize,
    style: PublicationStyle,
    encoding: TextEncoding,
) -> String {
    let shift_txt = if let Some((hi, lo)) = sig.shift_range_ppm {
        let dash = match encoding {
            TextEncoding::Unicode => "\u{2013}",
            TextEncoding::Ascii => "-",
            TextEncoding::Latex => "--",
        };
        format!("{hi:.decimals$}{dash}{lo:.decimals$}")
    } else {
        format!("{:.decimals$}", sig.shift_ppm)
    };

    let j_sym = match encoding {
        TextEncoding::Latex => "$J$",
        _ => "J",
    };

    let (_, nuc_sym) = nucleus.mass_and_symbol();
    let mut parts: Vec<String> = Vec::new();
    let h_str = sig.proton_count.map(|n| format!("{n}{nuc_sym}"));
    let m_str = sig.multiplicity.map(|m| m.as_str().to_string());
    let j_str = if !sig.couplings_hz.is_empty() {
        let js = sig
            .couplings_hz
            .iter()
            .map(|j| format!("{j:.1}"))
            .collect::<Vec<_>>()
            .join(", ");
        Some(format!("{j_sym} = {js} Hz"))
    } else {
        None
    };

    match style {
        PublicationStyle::Rsc => {
            if let Some(h) = h_str {
                parts.push(h);
            }
            if let Some(m) = m_str {
                parts.push(m);
            }
            if let Some(j) = j_str {
                parts.push(j);
            }
        }
        PublicationStyle::Acs | PublicationStyle::Joc | PublicationStyle::Nature => {
            if let Some(m) = m_str {
                parts.push(m);
            }
            if let Some(j) = j_str {
                parts.push(j);
            }
            if let Some(h) = h_str {
                parts.push(h);
            }
        }
    }
    if let Some(asgn) = &sig.assignment {
        if !asgn.trim().is_empty() {
            parts.push(asgn.trim().to_string());
        }
    }

    if parts.is_empty() {
        shift_txt
    } else {
        format!("{shift_txt} ({})", parts.join(", "))
    }
}

fn to_unicode_superscript(digits: &str) -> String {
    digits
        .chars()
        .map(|c| match c {
            '0' => '⁰',
            '1' => '¹',
            '2' => '²',
            '3' => '³',
            '4' => '⁴',
            '5' => '⁵',
            '6' => '⁶',
            '7' => '⁷',
            '8' => '⁸',
            '9' => '⁹',
            _ => c,
        })
        .collect()
}

fn replace_unicode_sub_super(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for c in input.chars() {
        match c {
            '⁰' | '₀' => out.push('0'),
            '¹' | '₁' => out.push('1'),
            '²' | '₂' => out.push('2'),
            '³' | '₃' => out.push('3'),
            '⁴' | '₄' => out.push('4'),
            '⁵' | '₅' => out.push('5'),
            '⁶' | '₆' => out.push('6'),
            '⁷' | '₇' => out.push('7'),
            '⁸' | '₈' => out.push('8'),
            '⁹' | '₉' => out.push('9'),
            '\u{03B4}' => out.push_str("delta"),
            '\u{2013}' | '\u{2014}' => out.push_str(".."),
            '\u{2212}' => out.push('-'),
            _ => out.push(c),
        }
    }
    out
}

fn normalize_report_markup_to_ascii(input: &str) -> String {
    // Strip LaTeX math `$` delimiters, `\text{...}`, `\mathrm{...}`, `^{...}`, `_{...}`, `\{`, `\}`
    let has_latex = input.contains('$') || input.contains('\\') || input.contains("^{");
    let mut s = input
        .replace("\\delta", "delta")
        .replace("\\{", "{")
        .replace("\\}", "}")
        .replace('$', "");
    if has_latex {
        s = s.replace("---", "..-").replace("--", "..");
    }

    for macro_prefix in ["\\text{", "\\mathrm{", "^{", "_{"] {
        while let Some(start) = s.find(macro_prefix) {
            let inner_start = start + macro_prefix.len();
            if let Some(rel_end) = s[inner_start..].find('}') {
                let inner = s[inner_start..(inner_start + rel_end)].to_string();
                s.replace_range(start..(inner_start + rel_end + 1), &inner);
            } else {
                break;
            }
        }
    }

    replace_unicode_sub_super(&s)
}

fn parse_nucleus_header(raw: &str) -> Result<(ReportNucleus, bool), MultipletError> {
    let clean = raw.trim();
    if let Some(brace_pos) = clean.find('{') {
        let main_part = clean[..brace_pos].trim();
        let nuc = ReportNucleus::from_ascii_token(main_part).ok_or_else(|| {
            MultipletError::ParseError(format!("unrecognized NMR nucleus '{main_part}'"))
        })?;
        return Ok((nuc, true));
    }
    let nuc = ReportNucleus::from_ascii_token(clean)
        .ok_or_else(|| MultipletError::ParseError(format!("unrecognized NMR nucleus '{clean}'")))?;
    Ok((nuc, false))
}

fn parse_mhz_and_solvent(inside: &str) -> Result<(f64, String), MultipletError> {
    let sep_pos = inside.find([',', ';']).ok_or_else(|| {
        MultipletError::ParseError("expected ',' between MHz and solvent".to_string())
    })?;
    let mhz_part = inside[..sep_pos].trim();
    let solv_part = inside[(sep_pos + 1)..].trim();
    let mhz_num_str = mhz_part
        .strip_suffix("MHz")
        .or_else(|| mhz_part.strip_suffix("mhz"))
        .unwrap_or(mhz_part)
        .trim();
    let mhz: f64 = mhz_num_str.parse().map_err(|_| {
        MultipletError::ParseError(format!("invalid spectrometer MHz '{mhz_part}'"))
    })?;
    if !mhz.is_finite() || mhz <= 0.0 {
        return Err(MultipletError::ParseError(
            "spectrometer MHz must be positive".to_string(),
        ));
    }
    Ok((mhz, canonicalize_solvent_ascii(solv_part)))
}

fn split_top_level_commas(s: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0usize;
    let mut start = 0usize;
    for (idx, ch) in s.char_indices() {
        match ch {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                parts.push(&s[start..idx]);
                start = idx + 1;
            }
            _ => {}
        }
    }
    parts.push(&s[start..]);
    parts
}

fn parse_single_signal_entry(entry: &str) -> Result<ReportedSignal, MultipletError> {
    let (shift_part, paren_part) = match entry.find('(') {
        Some(open_idx) => {
            let close_idx = entry
                .rfind(')')
                .ok_or_else(|| MultipletError::ParseError(format!("unclosed '(' in '{entry}'")))?;
            (
                entry[..open_idx].trim(),
                Some(entry[(open_idx + 1)..close_idx].trim()),
            )
        }
        None => (entry.trim(), None),
    };

    let (shift_ppm, shift_range_ppm) = parse_shift_or_range(shift_part)?;
    let mut multiplicity: Option<MultipletClass> = None;
    let mut couplings_hz: Vec<f64> = Vec::new();
    let mut proton_count: Option<usize> = None;
    let mut assignment: Option<String> = None;

    if let Some(inside) = paren_part {
        let mut in_j_list = false;
        for raw_tok in inside.split(',') {
            let tok = raw_tok.trim();
            if tok.is_empty() {
                continue;
            }

            // Check `J = ...` or `J ...`
            if let Some(after_j) = tok
                .strip_prefix("J")
                .or_else(|| tok.strip_prefix("j"))
                .map(str::trim_start)
            {
                if after_j.starts_with('=')
                    || after_j
                        .chars()
                        .next()
                        .is_some_and(|c| c.is_ascii_digit() || c == '-')
                {
                    let rest = after_j.strip_prefix('=').unwrap_or(after_j).trim();
                    let ended = extract_j_numbers(rest, &mut couplings_hz)?;
                    in_j_list = !ended;
                    continue;
                }
            }

            if in_j_list {
                if looks_like_j_continuation(tok) {
                    let ended = extract_j_numbers(tok, &mut couplings_hz)?;
                    if ended {
                        in_j_list = false;
                    }
                    continue;
                } else {
                    in_j_list = false;
                }
            }

            if proton_count.is_none() {
                if let Some(n_h) = parse_proton_count_token(tok) {
                    proton_count = Some(n_h);
                    continue;
                }
            }

            if multiplicity.is_none() {
                if let Some(cls) = MultipletClass::from_label(tok) {
                    multiplicity = Some(cls);
                    continue;
                }
            }

            match &mut assignment {
                Some(existing) => {
                    existing.push_str(", ");
                    existing.push_str(tok);
                }
                None => assignment = Some(tok.to_string()),
            }
        }
    }

    Ok(ReportedSignal {
        shift_ppm,
        shift_range_ppm,
        multiplicity,
        couplings_hz,
        proton_count,
        assignment,
    })
}

fn parse_shift_or_range(s: &str) -> Result<(f64, Option<(f64, f64)>), MultipletError> {
    let clean = s.trim();
    if let Some(sep_idx) = clean.find("..") {
        let left: f64 = clean[..sep_idx]
            .trim()
            .parse()
            .map_err(|_| MultipletError::ParseError(format!("invalid range start in '{clean}'")))?;
        let right: f64 = clean[(sep_idx + 2)..]
            .trim()
            .parse()
            .map_err(|_| MultipletError::ParseError(format!("invalid range end in '{clean}'")))?;
        return Ok((
            0.5 * (left + right),
            Some((left.max(right), left.min(right))),
        ));
    }
    // Look for ASCII range separator `-` after index 0 (ignoring a leading negative sign)
    let bytes = clean.as_bytes();
    for idx in 1..bytes.len() {
        if bytes[idx] == b'-' && bytes[idx - 1].is_ascii_digit() {
            let left: f64 = clean[..idx].trim().parse().map_err(|_| {
                MultipletError::ParseError(format!("invalid range start in '{clean}'"))
            })?;
            let right: f64 = clean[(idx + 1)..].trim().parse().map_err(|_| {
                MultipletError::ParseError(format!("invalid range end in '{clean}'"))
            })?;
            return Ok((
                0.5 * (left + right),
                Some((left.max(right), left.min(right))),
            ));
        }
    }
    let val: f64 = clean
        .parse()
        .map_err(|_| MultipletError::ParseError(format!("invalid chemical shift '{clean}'")))?;
    Ok((val, None))
}

fn looks_like_j_continuation(tok: &str) -> bool {
    let t = tok.trim();
    t.chars()
        .next()
        .is_some_and(|c| c.is_ascii_digit() || c == '.')
        && parse_proton_count_token(t).is_none()
}

fn extract_j_numbers(s: &str, out: &mut Vec<f64>) -> Result<bool, MultipletError> {
    let lower = s.to_ascii_lowercase();
    let has_hz = lower.contains("hz");
    let cleaned = s.replace("Hz", " ").replace("hz", " ").replace("and", " ");
    for part in cleaned.split_whitespace() {
        let p = part.trim_matches(|c: char| c == ',' || c == ';');
        if p.is_empty() {
            continue;
        }
        let v: f64 = p
            .parse()
            .map_err(|_| MultipletError::ParseError(format!("invalid J coupling value '{p}'")))?;
        out.push(v);
    }
    Ok(has_hz)
}

fn parse_proton_count_token(tok: &str) -> Option<usize> {
    let t = tok.trim();
    let num_part = t
        .strip_suffix(['H', 'h', 'F', 'f', 'P', 'p', 'C', 'c', 'N', 'n'])?
        .trim();
    if num_part.is_empty() || !num_part.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    num_part.parse::<usize>().ok()
}

// ============================================================================
// Section 4: Internal-Standard Quantitative NMR (`qNMR`) Purity & Concentration
// (Pauli et al., J. Med. Chem. 2014, 57, 9220-9231; Malz & Jancke 2005)
// ============================================================================

/// Certified primary reference material (`CRM` / `NIST SRM` / `USP`) metadata for `1H qNMR`
/// (public IUPAC weights & reference properties summarized in Pauli et al., *J. Med. Chem.* 2014).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct QnmrReferenceStandardInfo {
    pub name: &'static str,
    pub abbreviation: &'static str,
    pub formula: &'static str,
    pub mw_g_mol: f64,
    pub protons_n: usize,
    pub typical_shift_ppm: f64,
    pub typical_solvent: &'static str,
    pub typical_t1_s: f64,
}

/// Built-in lookup table of common NIST-traceable / pharmacopeial `1H qNMR` internal standards.
#[allow(clippy::approx_constant)]
pub const QNMR_CERTIFIED_STANDARDS: &[QnmrReferenceStandardInfo] = &[
    QnmrReferenceStandardInfo {
        name: "Dimethyl sulfone",
        abbreviation: "DMSO2",
        formula: "C2H6O2S",
        mw_g_mol: 94.133,
        protons_n: 6,
        typical_shift_ppm: 3.00,
        typical_solvent: "DMSO-d6",
        typical_t1_s: 2.6,
    },
    QnmrReferenceStandardInfo {
        name: "Maleic acid",
        abbreviation: "MA",
        formula: "C4H4O4",
        mw_g_mol: 116.072,
        protons_n: 2,
        typical_shift_ppm: 6.28,
        typical_solvent: "D2O",
        typical_t1_s: 3.9,
    },
    QnmrReferenceStandardInfo {
        name: "1,3,5-Trimethoxybenzene",
        abbreviation: "TMB",
        formula: "C9H12O3",
        mw_g_mol: 168.190,
        protons_n: 3, // aromatic 3H singlet at 6.09 ppm (or 9H methoxy at 3.77 ppm)
        typical_shift_ppm: 6.09,
        typical_solvent: "CDCl3",
        typical_t1_s: 3.6,
    },
    QnmrReferenceStandardInfo {
        name: "Dimethyl terephthalate",
        abbreviation: "DMT",
        formula: "C10H10O4",
        mw_g_mol: 194.184,
        protons_n: 4,
        typical_shift_ppm: 8.10,
        typical_solvent: "CDCl3",
        typical_t1_s: 3.1,
    },
    QnmrReferenceStandardInfo {
        name: "Benzoic acid (NIST SRM 350b)",
        abbreviation: "BA",
        formula: "C7H6O2",
        mw_g_mol: 122.121,
        protons_n: 2,
        typical_shift_ppm: 8.12,
        typical_solvent: "CDCl3",
        typical_t1_s: 4.2,
    },
    QnmrReferenceStandardInfo {
        name: "3,5-Dinitrobenzoic acid",
        abbreviation: "DNBA",
        formula: "C7H4N2O6",
        mw_g_mol: 212.116,
        protons_n: 1,
        typical_shift_ppm: 9.23,
        typical_solvent: "DMSO-d6",
        typical_t1_s: 4.8,
    },
];

/// Return the built-in catalog of primary `1H qNMR` internal standards.
pub fn qnmr_certified_standards() -> &'static [QnmrReferenceStandardInfo] {
    QNMR_CERTIFIED_STANDARDS
}

/// Internal standard gravimetric and spectral measurement inputs for `qNMR`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QnmrStandardSpec {
    pub name: String,
    /// Molecular weight `MW_std` in `g/mol`.
    pub mw_g_mol: f64,
    /// Standard uncertainty `u(MW_std)` in `g/mol` (e.g. `0.002`).
    pub mw_sigma_g_mol: f64,
    /// Weighed mass `m_std` in `mg`.
    pub mass_mg: f64,
    /// Microbalance standard uncertainty `u(m_std)` in `mg` (e.g. `0.001`–`0.005 mg`).
    pub mass_sigma_mg: f64,
    /// Certified mass-fraction purity `P_std` in `[0.0, 1.0]` (e.g. `0.9995` for `99.95% w/w`).
    pub purity_fraction: f64,
    /// Standard uncertainty `u(P_std)` in mass fraction (e.g. `0.0005`).
    pub purity_sigma_fraction: f64,
    /// Number of nuclei `N_std` contributing to `integral`.
    pub protons_n: usize,
    /// Measured baseline-corrected integral `I_std`.
    pub integral: f64,
    /// Standard uncertainty `u(I_std)` propagated from spectral noise RMS.
    pub integral_sigma: f64,
    /// Optional longitudinal relaxation time `T_1` (seconds) of the standard resonance.
    pub t1_s: Option<f64>,
}

impl QnmrStandardSpec {
    /// Construct a `QnmrStandardSpec` with `purity_fraction` in `(0, 1]`.
    pub fn new(
        name: impl Into<String>,
        mw_g_mol: f64,
        mass_mg: f64,
        purity_fraction: f64,
        protons_n: usize,
        integral: f64,
    ) -> Self {
        Self {
            name: name.into(),
            mw_g_mol,
            mw_sigma_g_mol: 0.0,
            mass_mg,
            mass_sigma_mg: 0.0,
            purity_fraction,
            purity_sigma_fraction: 0.0,
            protons_n,
            integral,
            integral_sigma: 0.0,
            t1_s: None,
        }
    }

    /// Set certified purity and `1 sigma` uncertainty in percent `% w/w` (`0..=100`).
    pub fn with_purity_percent(mut self, purity_pct: f64, sigma_pct: f64) -> Self {
        self.purity_fraction = purity_pct / 100.0;
        self.purity_sigma_fraction = sigma_pct / 100.0;
        self
    }

    /// Attach `1 sigma` metrological uncertainties `(u_mw, u_mass_mg, u_integral)`.
    pub fn with_uncertainties(
        mut self,
        mw_sigma_g_mol: f64,
        mass_sigma_mg: f64,
        integral_sigma: f64,
    ) -> Self {
        self.mw_sigma_g_mol = mw_sigma_g_mol;
        self.mass_sigma_mg = mass_sigma_mg;
        self.integral_sigma = integral_sigma;
        self
    }

    /// Attach longitudinal relaxation time `T_1` (seconds) for the `5 * T_1` recycle-delay check.
    pub fn with_t1_s(mut self, t1_s: f64) -> Self {
        self.t1_s = Some(t1_s);
        self
    }
}

/// Single non-overlapping quantitative reporter multiplet of the analyte `x`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QnmrAnalyteSignal {
    pub label: Option<String>,
    pub shift_ppm: Option<f64>,
    /// Spin count `N_x` represented by this reporter multiplet.
    pub protons_n: usize,
    /// Measured baseline-corrected integral `I_x`.
    pub integral: f64,
    /// Propagated `1 sigma` integral uncertainty `u(I_x)`.
    pub integral_sigma: f64,
    /// Optional longitudinal relaxation time `T_1` (seconds) of this analyte resonance.
    pub t1_s: Option<f64>,
}

impl QnmrAnalyteSignal {
    pub fn new(protons_n: usize, integral: f64, integral_sigma: f64) -> Self {
        Self {
            label: None,
            shift_ppm: None,
            protons_n,
            integral,
            integral_sigma,
            t1_s: None,
        }
    }

    pub fn with_t1_s(mut self, t1_s: f64) -> Self {
        self.t1_s = Some(t1_s);
        self
    }
}

/// Analyte gravimetric, volumetric, and reporter-signal specification for `qNMR`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QnmrAnalyteSpec {
    pub name: String,
    /// Molecular weight `MW_x` in `g/mol`.
    pub mw_g_mol: f64,
    /// Standard uncertainty `u(MW_x)` in `g/mol`.
    pub mw_sigma_g_mol: f64,
    /// Weighed sample mass `m_x` in `mg`.
    pub mass_mg: f64,
    /// Microbalance uncertainty `u(m_x)` in `mg`.
    pub mass_sigma_mg: f64,
    /// One or more quantitative reporter signals of the analyte.
    pub signals: Vec<QnmrAnalyteSignal>,
    /// Optional total NMR solution volume `V` in `mL` (enables molar concentration `mol/L` & `mM` readout).
    pub solution_volume_ml: Option<f64>,
    /// Optional uncertainty `u(V)` in `mL`.
    pub solution_volume_sigma_ml: Option<f64>,
}

/// Pulse-acquire timing parameters used to audit the `5 * T_1` quantitative relaxation criterion.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct QnmrAcquisitionParams {
    /// Inter-scan relaxation delay `D1` in seconds.
    pub recycle_delay_d1_s: f64,
    /// FID acquisition time `AQ` in seconds (`T_rep = D1 + AQ`).
    pub acquisition_time_aq_s: f64,
    /// Excitation pulse flip angle in degrees (`90.0` for quantitative `90 deg` pulse-acquire).
    pub pulse_flip_angle_deg: f64,
}

impl QnmrAcquisitionParams {
    pub fn new(recycle_delay_d1_s: f64, acquisition_time_aq_s: f64) -> Self {
        Self {
            recycle_delay_d1_s,
            acquisition_time_aq_s,
            pulse_flip_angle_deg: 90.0,
        }
    }

    /// Total scan repetition time `T_rep = D1 + AQ` in seconds.
    pub fn total_repetition_time_s(&self) -> f64 {
        self.recycle_delay_d1_s + self.acquisition_time_aq_s
    }
}

/// Per-reporter-signal `qNMR` purity and concentration breakdown.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QnmrSignalPurity {
    pub signal_index: usize,
    pub label: Option<String>,
    pub protons_n: usize,
    pub purity_fraction: f64,
    pub purity_percent: f64,
    pub combined_uncertainty_fraction: f64,
    pub expanded_uncertainty_k2_percent: f64,
    pub molar_concentration_mol_l: Option<f64>,
    pub molar_concentration_mm: Option<f64>,
}

/// Complete metrological output of internal-standard `qNMR` purity & molarity calculation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QnmrPurityReport {
    /// Mass-fraction purity `P_x` in `[0, 1]` (`(I_x / I_std) * (N_std / N_x) * (MW_x / MW_std) * (m_std / m_x) * P_std`).
    pub purity_fraction: f64,
    /// Mass-fraction purity in `% w/w` (`100 * purity_fraction`).
    pub purity_percent: f64,
    /// Combined standard uncertainty `u_c(P_x)` (`k = 1`, mass fraction).
    pub combined_uncertainty_fraction: f64,
    /// Combined standard uncertainty `u_c` in `% w/w`.
    pub combined_uncertainty_percent: f64,
    /// Expanded `95%` coverage uncertainty `U = 2 * u_c(P_x)` (`k = 2`, mass fraction).
    pub expanded_uncertainty_k2_fraction: f64,
    /// Expanded `95%` coverage uncertainty `U = 2 * u_c` in `% w/w`.
    pub expanded_uncertainty_k2_percent: f64,
    /// Analyte molar concentration `C_x` in `mol/L` (`M`) when `solution_volume_ml` is provided.
    pub molar_concentration_mol_l: Option<f64>,
    /// Propagated `1 sigma` uncertainty on `molar_concentration_mol_l`.
    pub molar_concentration_sigma_mol_l: Option<f64>,
    /// Analyte molar concentration in `mmol/L` (`mM`).
    pub molar_concentration_mm: Option<f64>,
    /// Internal standard molar concentration `C_std` in `mol/L` when `solution_volume_ml` is provided.
    pub standard_concentration_mol_l: Option<f64>,
    /// Individual results for each analyte reporter signal.
    pub per_signal_results: Vec<QnmrSignalPurity>,
    /// Number of independent measurements / reporter signals aggregated.
    pub replicate_count: usize,
    /// Relative Standard Deviation (`RSD = 100 * s / mean`, in `%`) when `>= 2` reporter signals or replicates are present.
    pub replicate_rsd_percent: Option<f64>,
    /// `true` when `T_rep = D1 + AQ < 5 * max(T_1)` (Pauli et al. 2014 incomplete longitudinal relaxation guard).
    pub t1_recycle_warning: bool,
    /// Longest `T_1` (seconds) across standard and analyte signals when supplied.
    pub max_t1_s: Option<f64>,
    /// Minimum recommended `5 * T_1` repetition time (seconds) when `T_1` metadata is supplied.
    pub required_5x_t1_s: Option<f64>,
    /// Analytical Ernst-angle steady-state differential saturation bias (`%`) when both `T_1,x` and `T_1,std` are supplied.
    pub estimated_saturation_bias_percent: Option<f64>,
    /// Human-readable metrological warning messages.
    pub warnings: Vec<String>,
}

/// Calculate internal-standard `qNMR` mass-fraction purity `P_x`, molar concentration `C_x`,
/// GUM combined uncertainty, multi-signal RSD, and `5 * T_1` recycle-delay warning
/// (Pauli et al., *J. Med. Chem.* **2014**, 57, 9220–9231).
pub fn calculate_qnmr_purity(
    analyte: &QnmrAnalyteSpec,
    standard: &QnmrStandardSpec,
    acquisition: Option<&QnmrAcquisitionParams>,
) -> Result<QnmrPurityReport, MultipletError> {
    validate_qnmr_inputs(analyte, standard)?;

    let std_conc_mol_l = analyte.solution_volume_ml.map(|vol_ml| {
        // m_std (mg) * 1e-3 (g/mg) * P_std / (MW_std (g/mol) * vol_ml * 1e-3 (L/mL))
        (standard.mass_mg * standard.purity_fraction) / (standard.mw_g_mol * vol_ml)
    });

    // Shared systematic relative variance across signals of the same tube:
    // `(u(I_std)/I_std)^2 + (u(MW_x)/MW_x)^2 + (u(MW_std)/MW_std)^2 + (u(m_std)/m_std)^2 + (u(m_x)/m_x)^2 + (u(P_std)/P_std)^2`
    let shared_rel_var = sq_rel(standard.integral_sigma, standard.integral)
        + sq_rel(analyte.mw_sigma_g_mol, analyte.mw_g_mol)
        + sq_rel(standard.mw_sigma_g_mol, standard.mw_g_mol)
        + sq_rel(standard.mass_sigma_mg, standard.mass_mg)
        + sq_rel(analyte.mass_sigma_mg, analyte.mass_mg)
        + sq_rel(standard.purity_sigma_fraction, standard.purity_fraction);

    let mut per_signal_results = Vec::with_capacity(analyte.signals.len());
    for (idx, sig) in analyte.signals.iter().enumerate() {
        let px = (sig.integral / standard.integral)
            * ((standard.protons_n as f64) / (sig.protons_n as f64))
            * (analyte.mw_g_mol / standard.mw_g_mol)
            * (standard.mass_mg / analyte.mass_mg)
            * standard.purity_fraction;

        let sig_rel_var = sq_rel(sig.integral_sigma, sig.integral) + shared_rel_var;
        let uc_frac = px.abs() * sig_rel_var.sqrt();

        let cx_mol_l = std_conc_mol_l.map(|c_std| {
            (sig.integral / standard.integral)
                * ((standard.protons_n as f64) / (sig.protons_n as f64))
                * c_std
        });

        per_signal_results.push(QnmrSignalPurity {
            signal_index: idx,
            label: sig.label.clone(),
            protons_n: sig.protons_n,
            purity_fraction: px,
            purity_percent: 100.0 * px,
            combined_uncertainty_fraction: uc_frac,
            expanded_uncertainty_k2_percent: 200.0 * uc_frac,
            molar_concentration_mol_l: cx_mol_l,
            molar_concentration_mm: cx_mol_l.map(|c| c * 1_000.0),
        });
    }

    let k_sig = per_signal_results.len();
    let mean_px = per_signal_results
        .iter()
        .map(|r| r.purity_fraction)
        .sum::<f64>()
        / (k_sig as f64);

    // Average of independent `u(I_{x,k}) / I_{x,k}` variances + shared gravimetric/standard variance,
    // combined with between-signal scatter SEM if `k_sig >= 2`
    let mean_ix_rel_var = analyte
        .signals
        .iter()
        .map(|s| sq_rel(s.integral_sigma, s.integral))
        .sum::<f64>()
        / ((k_sig * k_sig) as f64);

    let (replicate_rsd_percent, sem_rel_var) = if k_sig >= 2 && mean_px.abs() > 1e-30 {
        let var = per_signal_results
            .iter()
            .map(|r| (r.purity_fraction - mean_px).powi(2))
            .sum::<f64>()
            / ((k_sig - 1) as f64);
        let sd = var.sqrt();
        let rsd = 100.0 * sd / mean_px.abs();
        let sem_rv = (var / (k_sig as f64)) / (mean_px * mean_px);
        (Some(rsd), sem_rv)
    } else {
        (None, 0.0)
    };

    let total_rel_var = shared_rel_var + mean_ix_rel_var.max(sem_rel_var);
    let uc_fraction = mean_px.abs() * total_rel_var.sqrt();

    let (molar_concentration_mol_l, molar_concentration_sigma_mol_l, molar_concentration_mm) =
        if let Some(vol_ml) = analyte.solution_volume_ml {
            let mean_cx = per_signal_results
                .iter()
                .filter_map(|r| r.molar_concentration_mol_l)
                .sum::<f64>()
                / (k_sig as f64);
            let vol_sigma = analyte.solution_volume_sigma_ml.unwrap_or(0.0);
            let conc_rel_var = sq_rel(standard.integral_sigma, standard.integral)
                + sq_rel(standard.mw_sigma_g_mol, standard.mw_g_mol)
                + sq_rel(standard.mass_sigma_mg, standard.mass_mg)
                + sq_rel(standard.purity_sigma_fraction, standard.purity_fraction)
                + sq_rel(vol_sigma, vol_ml)
                + mean_ix_rel_var.max(sem_rel_var);
            let sigma_cx = mean_cx.abs() * conc_rel_var.sqrt();
            (Some(mean_cx), Some(sigma_cx), Some(mean_cx * 1_000.0))
        } else {
            (None, None, None)
        };

    // Audit longitudinal relaxation `5 * T_1` recycle delay
    let mut warnings = Vec::new();
    let max_analyte_t1 = analyte
        .signals
        .iter()
        .filter_map(|s| s.t1_s)
        .fold(None::<f64>, |acc, t| Some(acc.map_or(t, |a| a.max(t))));
    let max_t1_s = match (standard.t1_s, max_analyte_t1) {
        (Some(ts), Some(ta)) => Some(ts.max(ta)),
        (Some(ts), None) => Some(ts),
        (None, Some(ta)) => Some(ta),
        (None, None) => None,
    };
    let required_5x_t1_s = max_t1_s.map(|t| 5.0 * t);

    let mut t1_recycle_warning = false;
    let mut estimated_saturation_bias_percent = None;

    if let Some(acq) = acquisition {
        let t_rep = acq.total_repetition_time_s();
        if let Some(req_5x) = required_5x_t1_s {
            if t_rep + 1e-9 < req_5x {
                t1_recycle_warning = true;
                warnings.push(format!(
                    "Recycle time D1 + AQ = {t_rep:.2} s is shorter than 5 * T1_max = {req_5x:.2} s; \
                     incomplete longitudinal relaxation biases quantitative integrals (Pauli et al., J. Med. Chem. 2014)."
                ));
            }
        }
        if let (Some(t1_std), Some(t1_x)) = (standard.t1_s, max_analyte_t1) {
            if t1_std > 0.0 && t1_x > 0.0 && t_rep > 0.0 {
                let f_x = ernst_steady_state_factor(t_rep, t1_x, acq.pulse_flip_angle_deg);
                let f_s = ernst_steady_state_factor(t_rep, t1_std, acq.pulse_flip_angle_deg);
                if f_s > 1e-12 {
                    estimated_saturation_bias_percent = Some(100.0 * ((f_x / f_s) - 1.0));
                }
            }
        }
    }

    if mean_px > 1.02 {
        warnings.push(format!(
            "Apparent mass-fraction purity ({:.2}% w/w) exceeds 102%; inspect standard mass, impurity co-resonance, or T1 saturation.",
            100.0 * mean_px
        ));
    }

    Ok(QnmrPurityReport {
        purity_fraction: mean_px,
        purity_percent: 100.0 * mean_px,
        combined_uncertainty_fraction: uc_fraction,
        combined_uncertainty_percent: 100.0 * uc_fraction,
        expanded_uncertainty_k2_fraction: 2.0 * uc_fraction,
        expanded_uncertainty_k2_percent: 200.0 * uc_fraction,
        molar_concentration_mol_l,
        molar_concentration_sigma_mol_l,
        molar_concentration_mm,
        standard_concentration_mol_l: std_conc_mol_l,
        per_signal_results,
        replicate_count: k_sig,
        replicate_rsd_percent,
        t1_recycle_warning,
        max_t1_s,
        required_5x_t1_s,
        estimated_saturation_bias_percent,
        warnings,
    })
}

/// Aggregate `R >= 1` independent replicate `qNMR` sample preparations / acquisitions,
/// reporting mean purity, inter-replicate `RSD (%)`, and combined metrological + replicate uncertainty.
pub fn calculate_qnmr_replicates(
    replicates: &[(QnmrAnalyteSpec, QnmrStandardSpec)],
    acquisition: Option<&QnmrAcquisitionParams>,
) -> Result<QnmrPurityReport, MultipletError> {
    if replicates.is_empty() {
        return Err(MultipletError::InvalidParameter(
            "qNMR replicate list must contain at least 1 measurement".to_string(),
        ));
    }
    if replicates.len() == 1 {
        return calculate_qnmr_purity(&replicates[0].0, &replicates[0].1, acquisition);
    }

    let mut rep_reports = Vec::with_capacity(replicates.len());
    for (an, st) in replicates {
        rep_reports.push(calculate_qnmr_purity(an, st, acquisition)?);
    }

    let r_n = rep_reports.len();
    let r_f = r_n as f64;
    let mean_px = rep_reports.iter().map(|r| r.purity_fraction).sum::<f64>() / r_f;
    let sample_var = rep_reports
        .iter()
        .map(|r| (r.purity_fraction - mean_px).powi(2))
        .sum::<f64>()
        / ((r_n - 1) as f64);
    let sample_sd = sample_var.sqrt();
    let rsd_pct = if mean_px.abs() > 1e-30 {
        Some(100.0 * sample_sd / mean_px.abs())
    } else {
        None
    };
    let sem = sample_sd / r_f.sqrt();

    // Combine mean single-determination metrological budget with empirical replicate SEM in quadrature
    let mean_metrological_uc = rep_reports
        .iter()
        .map(|r| r.combined_uncertainty_fraction)
        .sum::<f64>()
        / r_f;
    let uc_fraction = (mean_metrological_uc.powi(2) + sem.powi(2)).sqrt();

    let mean_cx = if rep_reports
        .iter()
        .all(|r| r.molar_concentration_mol_l.is_some())
    {
        Some(
            rep_reports
                .iter()
                .filter_map(|r| r.molar_concentration_mol_l)
                .sum::<f64>()
                / r_f,
        )
    } else {
        None
    };
    let sigma_cx = if rep_reports
        .iter()
        .all(|r| r.molar_concentration_sigma_mol_l.is_some())
    {
        let m_sig = rep_reports
            .iter()
            .filter_map(|r| r.molar_concentration_sigma_mol_l)
            .sum::<f64>()
            / r_f;
        Some(m_sig)
    } else {
        None
    };

    let t1_recycle_warning = rep_reports.iter().any(|r| r.t1_recycle_warning);
    let max_t1_s = rep_reports
        .iter()
        .filter_map(|r| r.max_t1_s)
        .fold(None::<f64>, |acc, t| Some(acc.map_or(t, |a| a.max(t))));
    let required_5x_t1_s = max_t1_s.map(|t| 5.0 * t);
    let estimated_saturation_bias_percent = rep_reports[0].estimated_saturation_bias_percent;
    let warnings = rep_reports[0].warnings.clone();

    let mut per_signal_results = Vec::new();
    for (rep_i, rep) in rep_reports.iter().enumerate() {
        for sig in &rep.per_signal_results {
            let mut copy = sig.clone();
            copy.signal_index = per_signal_results.len();
            copy.label = Some(match &sig.label {
                Some(lbl) => format!("Rep {} ({lbl})", rep_i + 1),
                None => format!("Rep {}", rep_i + 1),
            });
            per_signal_results.push(copy);
        }
    }

    Ok(QnmrPurityReport {
        purity_fraction: mean_px,
        purity_percent: 100.0 * mean_px,
        combined_uncertainty_fraction: uc_fraction,
        combined_uncertainty_percent: 100.0 * uc_fraction,
        expanded_uncertainty_k2_fraction: 2.0 * uc_fraction,
        expanded_uncertainty_k2_percent: 200.0 * uc_fraction,
        molar_concentration_mol_l: mean_cx,
        molar_concentration_sigma_mol_l: sigma_cx,
        molar_concentration_mm: mean_cx.map(|c| c * 1_000.0),
        standard_concentration_mol_l: rep_reports[0].standard_concentration_mol_l,
        per_signal_results,
        replicate_count: r_n,
        replicate_rsd_percent: rsd_pct,
        t1_recycle_warning,
        max_t1_s,
        required_5x_t1_s,
        estimated_saturation_bias_percent,
        warnings,
    })
}

/// Direct molar concentration determination `C_x = (I_x / I_std) * (N_std / N_x) * C_std`
/// (returns `(conc_mol_l, sigma_mol_l)`).
#[allow(clippy::too_many_arguments)]
pub fn calculate_molar_concentration_from_standard(
    analyte_integral: f64,
    analyte_integral_sigma: f64,
    analyte_protons_n: usize,
    std_integral: f64,
    std_integral_sigma: f64,
    std_protons_n: usize,
    std_conc_mol_l: f64,
    std_conc_sigma_mol_l: f64,
) -> Result<(f64, f64), MultipletError> {
    if analyte_protons_n == 0 || std_protons_n == 0 {
        return Err(MultipletError::InvalidParameter(
            "proton counts N_x and N_std must be >= 1".to_string(),
        ));
    }
    if !analyte_integral.is_finite()
        || analyte_integral <= 0.0
        || !std_integral.is_finite()
        || std_integral <= 0.0
        || !std_conc_mol_l.is_finite()
        || std_conc_mol_l <= 0.0
    {
        return Err(MultipletError::InvalidParameter(
            "integrals and standard concentration must be finite and positive".to_string(),
        ));
    }
    let cx = (analyte_integral / std_integral)
        * ((std_protons_n as f64) / (analyte_protons_n as f64))
        * std_conc_mol_l;
    let rel_var = sq_rel(analyte_integral_sigma, analyte_integral)
        + sq_rel(std_integral_sigma, std_integral)
        + sq_rel(std_conc_sigma_mol_l, std_conc_mol_l);
    Ok((cx, cx * rel_var.sqrt()))
}

fn validate_qnmr_inputs(
    analyte: &QnmrAnalyteSpec,
    standard: &QnmrStandardSpec,
) -> Result<(), MultipletError> {
    if analyte.signals.is_empty() {
        return Err(MultipletError::InvalidParameter(
            "analyte must provide at least one reporter signal".to_string(),
        ));
    }
    if standard.protons_n == 0 || analyte.signals.iter().any(|s| s.protons_n == 0) {
        return Err(MultipletError::InvalidParameter(
            "spin counts N_x and N_std must be >= 1".to_string(),
        ));
    }
    for (name, val) in [
        ("analyte.mw_g_mol", analyte.mw_g_mol),
        ("analyte.mass_mg", analyte.mass_mg),
        ("standard.mw_g_mol", standard.mw_g_mol),
        ("standard.mass_mg", standard.mass_mg),
        ("standard.purity_fraction", standard.purity_fraction),
        ("standard.integral", standard.integral),
    ] {
        if !val.is_finite() || val <= 0.0 {
            return Err(MultipletError::InvalidParameter(format!(
                "{name} must be finite and strictly positive (got {val})"
            )));
        }
    }
    if standard.purity_fraction > 1.005 {
        return Err(MultipletError::InvalidParameter(format!(
            "standard.purity_fraction must be in (0, 1] (got {}; use with_purity_percent for % values)",
            standard.purity_fraction
        )));
    }
    if let Some(vol) = analyte.solution_volume_ml {
        if !vol.is_finite() || vol <= 0.0 {
            return Err(MultipletError::InvalidParameter(
                "solution_volume_ml must be finite and positive".to_string(),
            ));
        }
    }
    for (idx, sig) in analyte.signals.iter().enumerate() {
        if !sig.integral.is_finite() || sig.integral <= 0.0 {
            return Err(MultipletError::InvalidParameter(format!(
                "analyte signal {idx} integral must be finite and positive"
            )));
        }
    }
    Ok(())
}

fn sq_rel(sigma: f64, val: f64) -> f64 {
    if sigma <= 0.0 || val.abs() <= 1e-30 {
        0.0
    } else {
        (sigma / val).powi(2)
    }
}

fn ernst_steady_state_factor(t_rep_s: f64, t1_s: f64, flip_deg: f64) -> f64 {
    let alpha = flip_deg.to_radians();
    let e1 = (-t_rep_s / t1_s).exp();
    ((1.0 - e1) * alpha.sin()) / (1.0 - e1 * alpha.cos()).max(1e-12)
}

// ============================================================================
// Section 5: `Octet`-Compatible 2D Structure Atom-Index <-> Multiplet Assignment Table
// & SDF / SMILES Atom-Number Mapping
// ============================================================================

/// Atom index and numbering cross-reference entry bridging `Octet` 2D molecular graphs,
/// 1-based MDL `MOL`/`SDF` atom blocks, and SMILES `[atom:map]` labels.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AtomNumberEntry {
    /// Zero-based index into `octet_mol::Molecule::atoms` and `octet_render::RenderScene::atoms`.
    pub octet_atom_idx: usize,
    /// One-based atom block number in MDL V2000/V3000 `.mol` and `.sdf` records (`octet_atom_idx + 1`).
    pub sdf_atom_number: usize,
    /// Optional SMILES atom-to-atom map number (`[CH3:4]` -> `Some(4)`).
    pub smiles_atom_map: Option<u16>,
    /// Optional explicit user/IUPAC atom numbering annotation (`Atom::atom_number`).
    pub custom_atom_number: Option<String>,
    /// Canonical display label for tables & GUI tooltips (e.g. `"C1"`, `"O4"`, or custom `"4a"`).
    pub display_label: String,
    /// Canonical element symbol (`"C"`, `"N"`, `"O"`, ...).
    pub element: String,
    /// Atomic number `Z`.
    pub z: u8,
    /// Total attached `1H` count (implicit valence hydrogens + explicit `1H` neighbors).
    pub attached_hydrogens: u8,
    /// 2D canvas coordinates `(x, y)` in Ångströms from `octet_mol::Molecule`.
    pub coords_2d: (f64, f64),
    /// Topological Morgan/HOSE symmetry equivalence class (`usize::MAX` for explicit `H` nodes).
    pub symmetry_class: usize,
}

/// Complete atom-numbering lookup table for an `Octet` 2D chemical structure.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StructureAtomMap {
    pub molecule_name: String,
    pub entries: Vec<AtomNumberEntry>,
}

impl StructureAtomMap {
    /// Build an `Octet` / `SDF` / `SMILES` atom-number mapping from an [`octet_mol::Molecule`].
    pub fn from_molecule(mol: &Molecule) -> Self {
        let analyt = AnalytMol::from(mol);
        let env = perceive_topological_env(&analyt);
        let mut heavy_sym_by_mol: BTreeMap<usize, usize> = BTreeMap::new();
        for (env_i, &mol_i) in env.from_mol.iter().enumerate() {
            heavy_sym_by_mol.insert(mol_i, env.symmetry_class[env_i]);
        }

        let mut entries = Vec::with_capacity(mol.atoms.len());
        for (idx, atom) in mol.atoms.iter().enumerate() {
            let explicit_h_nb = mol
                .neighbors(idx)
                .into_iter()
                .filter(|&(nb, _)| mol.atoms[nb].z == 1 && mol.atoms[nb].isotope != Some(2))
                .count() as u8;
            let attached_h = if atom.z == 1 {
                0
            } else {
                mol.implicit_hydrogens(idx).saturating_add(explicit_h_nb)
            };
            let sdf_num = idx + 1;
            let display_label = match &atom.atom_number {
                Some(lbl) if !lbl.trim().is_empty() => lbl.trim().to_string(),
                _ => format!("{}{sdf_num}", atom.symbol),
            };
            entries.push(AtomNumberEntry {
                octet_atom_idx: idx,
                sdf_atom_number: sdf_num,
                smiles_atom_map: atom.atom_map,
                custom_atom_number: atom.atom_number.clone(),
                display_label,
                element: atom.symbol.clone(),
                z: atom.z,
                attached_hydrogens: attached_h,
                coords_2d: (atom.x, atom.y),
                symmetry_class: heavy_sym_by_mol.get(&idx).copied().unwrap_or(usize::MAX),
            });
        }

        Self {
            molecule_name: mol.name.clone(),
            entries,
        }
    }

    /// Parse an OpenSMILES string (preserving `[atom:map]` indices) and build its [`StructureAtomMap`].
    pub fn from_smiles(smiles: &str) -> Result<(Molecule, Self), ChemError> {
        let mol = parse_smiles(smiles)?;
        let map = Self::from_molecule(&mol);
        Ok((mol, map))
    }

    /// Parse an MDL V2000 or V3000 `.mol` block and build its 1-based SDF <-> 0-based `Octet` [`StructureAtomMap`].
    pub fn from_mol_block(mol_text: &str) -> Result<(Molecule, Self), ChemError> {
        let mol = read_mol(mol_text)?;
        let map = Self::from_molecule(&mol);
        Ok((mol, map))
    }

    /// Build a [`StructureAtomMap`] from a parsed [`SdfRecord`].
    pub fn from_sdf_record(record: &SdfRecord) -> Self {
        Self::from_molecule(&record.molecule)
    }

    /// Look up an atom entry by 0-based `Octet` atom index.
    pub fn by_octet_index(&self, octet_atom_idx: usize) -> Option<&AtomNumberEntry> {
        self.entries.get(octet_atom_idx)
    }

    /// Look up an atom entry by 1-based MDL `MOL`/`SDF` atom number (`1..=N`).
    pub fn by_sdf_number(&self, sdf_atom_number: usize) -> Option<&AtomNumberEntry> {
        if sdf_atom_number == 0 {
            None
        } else {
            self.entries.get(sdf_atom_number - 1)
        }
    }

    /// Look up all `Octet` atom indices carrying a given SMILES `[atom:map]` number.
    pub fn octet_indices_for_smiles_map(&self, smiles_map: u16) -> Vec<usize> {
        self.entries
            .iter()
            .filter(|e| e.smiles_atom_map == Some(smiles_map))
            .map(|e| e.octet_atom_idx)
            .collect()
    }

    /// Return all symmetry-equivalent 0-based `Octet` atom indices sharing `octet_atom_idx`'s topological class.
    pub fn symmetry_peers(&self, octet_atom_idx: usize) -> Vec<usize> {
        let Some(entry) = self.entries.get(octet_atom_idx) else {
            return Vec::new();
        };
        if entry.symmetry_class == usize::MAX {
            return vec![octet_atom_idx];
        }
        self.entries
            .iter()
            .filter(|e| e.symmetry_class == entry.symmetry_class)
            .map(|e| e.octet_atom_idx)
            .collect()
    }
}

/// Bidirectional link between one experimental 1D NMR multiplet and one or more atoms in an
/// `Octet` 2D chemical structure.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultipletAtomLink {
    /// Identifier of the experimental multiplet (`0..N`).
    pub multiplet_id: usize,
    /// Observed nucleus (`H1`, `C13`, ...).
    pub nucleus: ReportNucleus,
    /// Experimental multiplet chemical shift `\delta` in ppm.
    pub shift_ppm: f64,
    /// Experimental multiplet class (`s`, `d`, `t`, `dd`, ...).
    pub multiplicity: MultipletClass,
    /// Extracted coupling constants `J` in Hz.
    pub couplings_hz: Vec<f64>,
    /// Experimental normalized spin count (e.g. `2` for `2H`).
    pub proton_count: Option<usize>,
    /// Assigned 0-based `Octet` atom indices (`Molecule::atoms` / `RenderScene::atoms`).
    pub octet_atom_indices: Vec<usize>,
    /// Synchronized 1-based MDL `MOL`/`SDF` atom numbers (`octet_idx + 1`).
    pub sdf_atom_numbers: Vec<usize>,
    /// Synchronized SMILES `[atom:map]` numbers present on the assigned atoms.
    pub smiles_atom_maps: Vec<u16>,
    /// Optional HOSE/additivity predicted chemical shift (ppm) of the matched equivalence group.
    pub predicted_shift_ppm: Option<f64>,
    /// Assignment confidence in `[0.0, 1.0]` (`1.0` when `manual_lock == true`).
    pub confidence: f64,
    /// True when locked interactively by the spectroscopist.
    pub manual_lock: bool,
    /// Optional user note (e.g. `"ortho Ar-H"`).
    pub annotation: Option<String>,
}

/// Bidirectional 2D chemical-structure <-> 1D multiplet assignment table ready for
/// `Octet` / `nmr-gui` hover & click cross-highlighting and SDF/TSV/JSON interchange.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultipletAssignmentTable {
    pub atom_map: StructureAtomMap,
    pub links: Vec<MultipletAtomLink>,
}

impl MultipletAssignmentTable {
    /// Create an empty assignment table for `mol`.
    pub fn new(mol: &Molecule) -> Self {
        Self {
            atom_map: StructureAtomMap::from_molecule(mol),
            links: Vec::new(),
        }
    }

    /// Register or update a multiplet <-> `Octet` atom-index assignment, automatically populating
    /// synchronized 1-based `SDF` atom numbers and `SMILES` atom-map tags.
    #[allow(clippy::too_many_arguments)]
    pub fn assign_multiplet(
        &mut self,
        multiplet_id: usize,
        nucleus: ReportNucleus,
        analysis: &MultipletAnalysis,
        octet_atom_indices: &[usize],
        predicted_shift_ppm: Option<f64>,
        confidence: f64,
        manual_lock: bool,
        annotation: Option<String>,
    ) -> Result<(), MultipletError> {
        let mut clean_octet: Vec<usize> = Vec::with_capacity(octet_atom_indices.len());
        for &idx in octet_atom_indices {
            if idx >= self.atom_map.entries.len() {
                return Err(MultipletError::InvalidParameter(format!(
                    "Octet atom index {idx} out of bounds (molecule has {} atoms)",
                    self.atom_map.entries.len()
                )));
            }
            if !clean_octet.contains(&idx) {
                clean_octet.push(idx);
            }
        }
        clean_octet.sort_unstable();

        let sdf_atom_numbers: Vec<usize> = clean_octet.iter().map(|&i| i + 1).collect();
        let mut smiles_atom_maps: Vec<u16> = clean_octet
            .iter()
            .filter_map(|&i| self.atom_map.entries[i].smiles_atom_map)
            .collect();
        smiles_atom_maps.sort_unstable();
        smiles_atom_maps.dedup();

        let link = MultipletAtomLink {
            multiplet_id,
            nucleus,
            shift_ppm: analysis.center_ppm,
            multiplicity: analysis.class,
            couplings_hz: analysis.j_values_hz(),
            proton_count: analysis.proton_count,
            octet_atom_indices: clean_octet,
            sdf_atom_numbers,
            smiles_atom_maps,
            predicted_shift_ppm,
            confidence: if manual_lock {
                1.0
            } else {
                confidence.clamp(0.0, 1.0)
            },
            manual_lock,
            annotation,
        };

        if let Some(existing) = self
            .links
            .iter_mut()
            .find(|l| l.multiplet_id == multiplet_id)
        {
            *existing = link;
        } else {
            self.links.push(link);
            self.links.sort_by_key(|l| l.multiplet_id);
        }
        Ok(())
    }

    /// Assign a multiplet using 1-based MDL `MOL`/`SDF` atom numbers (`1..=N`).
    pub fn assign_multiplet_by_sdf_numbers(
        &mut self,
        multiplet_id: usize,
        nucleus: ReportNucleus,
        analysis: &MultipletAnalysis,
        sdf_numbers: &[usize],
        manual_lock: bool,
    ) -> Result<(), MultipletError> {
        let mut octet_indices = Vec::with_capacity(sdf_numbers.len());
        for &n in sdf_numbers {
            if n == 0 || n > self.atom_map.entries.len() {
                return Err(MultipletError::InvalidParameter(format!(
                    "1-based SDF atom number {n} out of range 1..={}",
                    self.atom_map.entries.len()
                )));
            }
            octet_indices.push(n - 1);
        }
        self.assign_multiplet(
            multiplet_id,
            nucleus,
            analysis,
            &octet_indices,
            None,
            1.0,
            manual_lock,
            None,
        )
    }

    /// Remove an assignment by `multiplet_id`.
    pub fn unassign_multiplet(&mut self, multiplet_id: usize) -> bool {
        let before = self.links.len();
        self.links.retain(|l| l.multiplet_id != multiplet_id);
        self.links.len() < before
    }

    /// Forward GUI lookup: return assigned 0-based `Octet` atom indices for `multiplet_id`.
    pub fn atoms_for_multiplet(&self, multiplet_id: usize) -> &[usize] {
        self.links
            .iter()
            .find(|l| l.multiplet_id == multiplet_id)
            .map(|l| l.octet_atom_indices.as_slice())
            .unwrap_or(&[])
    }

    /// Forward SDF lookup: return 1-based SDF atom numbers for `multiplet_id`.
    pub fn sdf_numbers_for_multiplet(&self, multiplet_id: usize) -> &[usize] {
        self.links
            .iter()
            .find(|l| l.multiplet_id == multiplet_id)
            .map(|l| l.sdf_atom_numbers.as_slice())
            .unwrap_or(&[])
    }

    /// Reverse GUI lookup: return all `multiplet_id`s assigned to 0-based `octet_atom_idx`.
    pub fn multiplets_for_octet_atom(&self, octet_atom_idx: usize) -> Vec<usize> {
        self.links
            .iter()
            .filter(|l| l.octet_atom_indices.contains(&octet_atom_idx))
            .map(|l| l.multiplet_id)
            .collect()
    }

    /// Reverse SDF lookup: return all `multiplet_id`s assigned to 1-based `sdf_atom_number`.
    pub fn multiplets_for_sdf_number(&self, sdf_atom_number: usize) -> Vec<usize> {
        self.links
            .iter()
            .filter(|l| l.sdf_atom_numbers.contains(&sdf_atom_number))
            .map(|l| l.multiplet_id)
            .collect()
    }

    /// Reverse SMILES lookup: return all `multiplet_id`s assigned to atoms bearing `[atom:smiles_map]`.
    pub fn multiplets_for_smiles_map(&self, smiles_map: u16) -> Vec<usize> {
        self.links
            .iter()
            .filter(|l| l.smiles_atom_maps.contains(&smiles_map))
            .map(|l| l.multiplet_id)
            .collect()
    }

    /// Return 0-based `Octet` heavy-atom indices bearing `>= 1` attached hydrogens that are not
    /// yet covered by any `H1` multiplet link.
    pub fn unassigned_proton_bearing_atoms(&self) -> Vec<usize> {
        let assigned: BTreeSet<usize> = self
            .links
            .iter()
            .filter(|l| l.nucleus == ReportNucleus::H1)
            .flat_map(|l| l.octet_atom_indices.iter().copied())
            .collect();
        self.atom_map
            .entries
            .iter()
            .filter(|e| e.attached_hydrogens > 0 && !assigned.contains(&e.octet_atom_idx))
            .map(|e| e.octet_atom_idx)
            .collect()
    }

    /// Apply cross-highlight halos onto an `Octet` [`RenderScene`] for `multiplet_id`
    /// (highlighting both assigned `SceneAtom`s and bonds connecting two assigned atoms).
    pub fn highlight_multiplet_on_scene(
        &self,
        scene: &mut RenderScene,
        multiplet_id: usize,
        highlight_color: RgbaColor,
    ) {
        Self::clear_scene_highlights(scene);
        let atoms = self.atoms_for_multiplet(multiplet_id);
        let atom_set: BTreeSet<usize> = atoms.iter().copied().collect();
        for &idx in &atom_set {
            if let Some(sa) = scene.atoms.get_mut(idx) {
                sa.highlight = Some(highlight_color);
            }
        }
        for sb in &mut scene.bonds {
            if atom_set.contains(&sb.a) && atom_set.contains(&sb.b) {
                sb.highlight = Some(highlight_color);
            }
        }
    }

    /// Clear all atom and bond highlights on an `Octet` [`RenderScene`].
    pub fn clear_scene_highlights(scene: &mut RenderScene) {
        for sa in &mut scene.atoms {
            sa.highlight = None;
        }
        for sb in &mut scene.bonds {
            sb.highlight = None;
        }
    }

    /// Export compact SDF property tag string (`multiplet_id|delta|mult|protons|sdf_atoms`, semicolon-delimited)
    /// suitable for storing inside `SdfRecord::properties["NMR_ASSIGNMENT"]`.
    pub fn to_sdf_property_tag(&self) -> String {
        self.links
            .iter()
            .map(|l| {
                let atoms_csv = l
                    .sdf_atom_numbers
                    .iter()
                    .map(|n| n.to_string())
                    .collect::<Vec<_>>()
                    .join(",");
                let h_txt = l.proton_count.map(|n| n.to_string()).unwrap_or_default();
                format!(
                    "{}|{:.3}|{}|{}|{}",
                    l.multiplet_id,
                    l.shift_ppm,
                    l.multiplicity.as_str(),
                    h_txt,
                    atoms_csv
                )
            })
            .collect::<Vec<_>>()
            .join("; ")
    }

    /// Reconstruct a [`MultipletAssignmentTable`] from an [`Molecule`] and an `NMR_ASSIGNMENT`
    /// SDF property tag string written by [`Self::to_sdf_property_tag`].
    pub fn from_sdf_property_tag(
        mol: &Molecule,
        nucleus: ReportNucleus,
        tag_value: &str,
    ) -> Result<Self, MultipletError> {
        let mut table = Self::new(mol);
        let trimmed = tag_value.trim();
        if trimmed.is_empty() {
            return Ok(table);
        }

        for entry in trimmed.split(';').map(str::trim).filter(|s| !s.is_empty()) {
            let parts: Vec<&str> = entry.split('|').map(str::trim).collect();
            if parts.len() != 5 {
                return Err(MultipletError::ParseError(format!(
                    "expected 5 '|'-delimited fields in SDF assignment tag entry, got `{entry}`"
                )));
            }
            let multiplet_id = parts[0].parse::<usize>().map_err(|_| {
                MultipletError::ParseError(format!("invalid multiplet_id `{}`", parts[0]))
            })?;
            let shift_ppm = parts[1].parse::<f64>().map_err(|_| {
                MultipletError::ParseError(format!("invalid shift_ppm `{}`", parts[1]))
            })?;
            let multiplicity = MultipletClass::from_label(parts[2]).ok_or_else(|| {
                MultipletError::ParseError(format!("unknown multiplicity `{}`", parts[2]))
            })?;
            let proton_count = if parts[3].is_empty() {
                None
            } else {
                Some(parts[3].parse::<usize>().map_err(|_| {
                    MultipletError::ParseError(format!("invalid proton_count `{}`", parts[3]))
                })?)
            };
            let mut sdf_nums = Vec::new();
            if !parts[4].is_empty() {
                for tok in parts[4].split(',').map(str::trim).filter(|s| !s.is_empty()) {
                    let n = tok.parse::<usize>().map_err(|_| {
                        MultipletError::ParseError(format!("invalid SDF atom number `{tok}`"))
                    })?;
                    sdf_nums.push(n);
                }
            }

            let mut oct_indices = Vec::with_capacity(sdf_nums.len());
            let mut smiles_maps = Vec::new();
            for sdf_num in &sdf_nums {
                let atom_entry = table.atom_map.by_sdf_number(*sdf_num).ok_or_else(|| {
                    MultipletError::InvalidParameter(format!(
                        "1-based SDF atom number {sdf_num} out of bounds"
                    ))
                })?;
                oct_indices.push(atom_entry.octet_atom_idx);
                if let Some(m) = atom_entry.smiles_atom_map {
                    if !smiles_maps.contains(&m) {
                        smiles_maps.push(m);
                    }
                }
            }

            table.links.push(MultipletAtomLink {
                multiplet_id,
                nucleus,
                shift_ppm,
                multiplicity,
                couplings_hz: Vec::new(),
                proton_count,
                octet_atom_indices: oct_indices,
                sdf_atom_numbers: sdf_nums,
                smiles_atom_maps: smiles_maps,
                predicted_shift_ppm: None,
                confidence: 1.0,
                manual_lock: true,
                annotation: None,
            });
        }
        table.links.sort_by_key(|l| l.multiplet_id);
        Ok(table)
    }

    /// Export a tab-separated summary table of multiplet <-> `Octet` / `SDF` / `SMILES` assignments.
    pub fn to_tsv(&self) -> String {
        let mut rows = Vec::with_capacity(self.links.len() + 1);
        rows.push(
            [
                "multiplet_id",
                "nucleus",
                "shift_ppm",
                "multiplicity",
                "j_hz",
                "protons",
                "octet_atoms_0based",
                "sdf_atoms_1based",
                "smiles_maps",
                "confidence",
            ]
            .join("\t"),
        );
        for l in &self.links {
            let (_, sym) = l.nucleus.mass_and_symbol();
            let (mass, _) = l.nucleus.mass_and_symbol();
            let js = l
                .couplings_hz
                .iter()
                .map(|j| format!("{j:.1}"))
                .collect::<Vec<_>>()
                .join(",");
            let oct = l
                .octet_atom_indices
                .iter()
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(",");
            let sdf = l
                .sdf_atom_numbers
                .iter()
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(",");
            let smi = l
                .smiles_atom_maps
                .iter()
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(",");
            rows.push(
                [
                    l.multiplet_id.to_string(),
                    format!("{mass}{sym}"),
                    format!("{:.3}", l.shift_ppm),
                    l.multiplicity.as_str().to_string(),
                    js,
                    l.proton_count.map(|n| n.to_string()).unwrap_or_default(),
                    oct,
                    sdf,
                    smi,
                    format!("{:.3}", l.confidence),
                ]
                .join("\t"),
            );
        }
        rows.join("\n")
    }
}

/// Automatically match experimental `1H` [`MultipletAnalysis`] entries to symmetry-equivalent
/// proton-bearing groups of an `Octet` [`Molecule`] using `octet_analyt` HOSE/additivity
/// shift, integral-count, and multiplicity agreement.
pub fn auto_assign_1h_multiplets(
    mol: &Molecule,
    multiplets: &[MultipletAnalysis],
    solvent: Option<&str>,
) -> MultipletAssignmentTable {
    let mut table = MultipletAssignmentTable::new(mol);
    if multiplets.is_empty() || mol.atoms.is_empty() {
        return table;
    }

    let analyt = AnalytMol::from(mol);
    let pred = predict_nmr(&analyt, OctetNmrNucleus::H1, solvent, None);
    if pred.signals.is_empty() {
        return table;
    }

    // Score every `(exp_idx, pred_idx)` pair and greedily assign in best-score order (1-to-1)
    let mut pairs: Vec<(f64, usize, usize)> = Vec::new();
    for (m_idx, m) in multiplets.iter().enumerate() {
        for (p_idx, ps) in pred.signals.iter().enumerate() {
            let d_ppm = (m.center_ppm - ps.shift_ppm).abs();
            let h_pen = match m.proton_count {
                Some(nh) if nh == ps.nuclei => 0.0,
                Some(nh) => 0.85 * ((nh as f64) - (ps.nuclei as f64)).abs(),
                None => 0.0,
            };
            let pred_mult =
                MultipletClass::from_label(&ps.multiplicity()).unwrap_or(MultipletClass::Multiplet);
            let mult_pen = if m.class == pred_mult {
                0.0
            } else if m.class == MultipletClass::Multiplet || pred_mult == MultipletClass::Multiplet
            {
                0.20
            } else {
                0.55
            };
            let cost = d_ppm + h_pen + mult_pen;
            pairs.push((cost, m_idx, p_idx));
        }
    }
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut used_m = vec![false; multiplets.len()];
    let mut used_p = vec![false; pred.signals.len()];
    for (cost, m_idx, p_idx) in pairs {
        if used_m[m_idx] || used_p[p_idx] || cost > 3.5 {
            continue;
        }
        used_m[m_idx] = true;
        used_p[p_idx] = true;
        let ps = &pred.signals[p_idx];
        let conf = (-0.65 * cost).exp().clamp(0.05, 0.99);
        let _ = table.assign_multiplet(
            m_idx,
            ReportNucleus::H1,
            &multiplets[m_idx],
            &ps.atoms,
            Some(ps.shift_ppm),
            conf,
            false,
            Some(ps.group_label.clone()),
        );
    }

    table
}
