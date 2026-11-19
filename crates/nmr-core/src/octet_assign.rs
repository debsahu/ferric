//! Clean-room 1D NMR (`¹H` / `¹³C`) experimental-to-structure automated resonance assignment,
//! manual click-to-swap / lock override, and ACS characterization formatter with atom labels.
//!
//! Provenance:
//! Original clean-room implementation for FERRIC (`nmr-core::octet_assign`) using exact
//! minimum-cost bipartite matching (Hungarian algorithm) over a composite chemical-shift,
//! integral, and multiplet-pattern cost matrix:
//! - Kuhn, H. W. (1955). "The Hungarian method for the assignment problem."
//!   *Naval Research Logistics Quarterly*, 2(1–2), 83–97.
//! - Munkres, J. (1957). "Algorithms for the assignment and transportation problems."
//!   *Journal of the Society for Industrial and Applied Mathematics*, 5(1), 32–38.
//! - Jonker, R.; Volgenant, A. (1987). "A shortest augmenting path algorithm for dense and
//!   sparse linear assignment problems." *Computing*, 38, 325–340.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::csp::hungarian_min_cost_assignment;
use crate::octet_render::{SceneSpectrum1D, SpectrumSignal1D};

/// Ratio of `¹³C` to `¹H` nuclear gyromagnetic ratios (`γ_13C / γ_1H ≈ 0.25145`).
pub const C13_TO_H1_LARMOR_RATIO: f64 = 0.251_45;

/// Convert a proton (`¹H`) Larmor frequency in MHz into the corresponding `¹³C` observation
/// frequency in MHz (`0.25145 * proton_mhz`). Returns `0.0` for non-positive or non-finite input.
pub fn c13_frequency_mhz(proton_mhz: f64) -> f64 {
    if proton_mhz.is_finite() && proton_mhz > 0.0 {
        proton_mhz * C13_TO_H1_LARMOR_RATIO
    } else {
        0.0
    }
}

/// Return unique positive finite scalar `J`-couplings sorted descending (largest `J` first),
/// merging duplicate values within `0.05 Hz` (one-decimal ACS precision) so coupled multiplet
/// counts never produce repeated `J = 7.0, 7.0, 7.0 Hz` entries.
pub fn deduplicate_couplings_hz(couplings_hz: &[f64]) -> Vec<f64> {
    let mut js: Vec<f64> = couplings_hz
        .iter()
        .copied()
        .filter(|j| j.is_finite() && *j > 0.0)
        .collect();
    js.sort_by(|a, b| b.total_cmp(a));
    let mut unique: Vec<f64> = Vec::with_capacity(js.len());
    for j in js {
        let is_dup = unique.last().is_some_and(|&prev| {
            (prev - j).abs() < 0.05 || (prev * 10.0).round() == (j * 10.0).round()
        });
        if !is_dup {
            unique.push(j);
        }
    }
    unique
}

/// Return `true` when `label` is merely a bare nuclei count such as `"1H"`, `"2H"`, `"3H"`,
/// `"1C"`, or `"2C"` rather than a real atom/group identifier (e.g. `"H-3"`, `"CH2"`, `"C-1"`).
fn is_bare_nuclei_count_label(label: &str, is_c13: bool) -> bool {
    let trimmed = label.trim();
    if trimmed.len() < 2 {
        return false;
    }
    let expected_suffix = if is_c13 { 'C' } else { 'H' };
    let mut chars = trimmed.chars();
    let Some(last) = chars.next_back() else {
        return false;
    };
    if !last.eq_ignore_ascii_case(&expected_suffix) {
        return false;
    }
    let digits = chars.as_str();
    !digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit())
}

/// Build a canonical atom-group label (`H-3`, `H-2/6`, `C-1`, etc.) from atom indices or group index.
fn format_atom_group_label(atom_indices: &[usize], group_idx: usize, is_c13: bool) -> String {
    let prefix = if is_c13 { "C" } else { "H" };
    if atom_indices.is_empty() {
        format!("{prefix}-{}", group_idx + 1)
    } else if atom_indices.len() == 1 {
        format!("{prefix}-{}", atom_indices[0] + 1)
    } else {
        let mut sorted: Vec<usize> = atom_indices.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        let nums: Vec<String> = sorted.into_iter().map(|i| (i + 1).to_string()).collect();
        format!("{prefix}-{}", nums.join("/"))
    }
}

/// An observed experimental 1D NMR peak (`¹H` or `¹³C`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExperimentalPeak1D {
    /// Stable 0-based index of this peak in the experimental peak list.
    pub peak_idx: usize,
    /// Observed chemical shift `δ` in ppm.
    pub ppm: f64,
    /// Optional relative integral (e.g., proton count for ¹H; `None` when unintegrated).
    pub integral: Option<f64>,
    /// Optional observed multiplicity code (`"s"`, `"d"`, `"t"`, `"q"`, `"dd"`, `"m"`, `"br s"`).
    pub multiplicity: Option<String>,
    /// Optional observed scalar J-couplings in Hz.
    pub couplings_hz: Vec<f64>,
    /// Peak height / intensity (arbitrary positive units).
    pub intensity: f64,
}

impl ExperimentalPeak1D {
    /// Construct an experimental peak from a [`SpectrumSignal1D`] entry.
    /// Non-finite chemical shifts are preserved as non-finite (never coerced to `0.0`) so
    /// downstream matching and ACS formatting reject them cleanly.
    pub fn from_signal(peak_idx: usize, sig: &SpectrumSignal1D) -> Self {
        let mult = sig.multiplicity.trim();
        Self {
            peak_idx,
            ppm: sig.ppm,
            integral: (sig.integral.is_finite() && sig.integral > 0.0).then_some(sig.integral),
            multiplicity: (!mult.is_empty()).then(|| mult.to_string()),
            couplings_hz: deduplicate_couplings_hz(&sig.couplings_hz),
            intensity: if sig.integral.is_finite() && sig.integral > 0.0 {
                sig.integral
            } else {
                1.0
            },
        }
    }

    /// Construct an experimental peak from a [`SpectrumSignal1D`] entry, returning `None`
    /// if `sig.ppm` is `NaN` or infinite.
    pub fn try_from_signal(peak_idx: usize, sig: &SpectrumSignal1D) -> Option<Self> {
        sig.ppm
            .is_finite()
            .then(|| Self::from_signal(peak_idx, sig))
    }
}

/// A predicted chemically-equivalent nucleus group on the 2D chemical structure.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PredictedGroup1D {
    /// Stable 0-based index of this predicted group.
    pub group_idx: usize,
    /// Indices of atoms in `RenderScene::atoms` / molecular graph belonging to this group.
    pub atom_indices: Vec<usize>,
    /// Human-readable atom/group label (e.g., `"ArH-6"`, `"OAc CH3"`, `"C-1 (C=O)"`).
    pub label: String,
    /// Predicted chemical shift `δ` in ppm.
    pub shift_ppm: f64,
    /// Estimated shift uncertainty `σ` in ppm (floored to sensible defaults if `<= 0`).
    pub sd_ppm: f64,
    /// Expected relative integral (number of equivalent nuclei, e.g., `3.0` for `CH3`).
    pub expected_integral: f64,
    /// Predicted multiplicity code (`"s"`, `"d"`, `"t"`, `"q"`, `"dd"`, `"m"`, `"br s"`).
    pub multiplicity: String,
    /// Predicted scalar J-couplings in Hz.
    pub couplings_hz: Vec<f64>,
}

impl PredictedGroup1D {
    /// Construct a predicted group from a [`SpectrumSignal1D`] entry.
    pub fn from_spectrum_signal(group_idx: usize, sig: &SpectrumSignal1D, is_c13: bool) -> Self {
        let default_label = sig
            .assignment_label
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty() && !is_bare_nuclei_count_label(s, is_c13))
            .map(str::to_string)
            .unwrap_or_else(|| format_atom_group_label(&sig.atom_indices, group_idx, is_c13));
        Self {
            group_idx,
            atom_indices: sig.atom_indices.clone(),
            label: default_label,
            shift_ppm: sig.ppm,
            sd_ppm: if is_c13 { 4.0 } else { 0.40 },
            expected_integral: if sig.integral.is_finite() && sig.integral > 0.0 {
                sig.integral
            } else {
                1.0
            },
            multiplicity: sig.multiplicity.trim().to_string(),
            couplings_hz: deduplicate_couplings_hz(&sig.couplings_hz),
        }
    }
}

/// Assignment pairing one predicted structure group with an optional experimental peak.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AtomPeakAssignment {
    /// Index of the predicted group (`PredictedGroup1D::group_idx`).
    pub group_idx: usize,
    /// Atom indices in the 2D structure assigned to this peak.
    pub atom_indices: Vec<usize>,
    /// Human-readable group / atom label.
    pub label: String,
    /// Predicted chemical shift in ppm.
    pub predicted_ppm: f64,
    /// Assigned experimental peak index (`ExperimentalPeak1D::peak_idx`), or `None` if unassigned.
    pub peak_idx: Option<usize>,
    /// Observed experimental shift in ppm (if assigned).
    pub experimental_ppm: Option<f64>,
    /// Signed shift residual `experimental_ppm - predicted_ppm` in ppm (if assigned).
    pub delta_ppm: Option<f64>,
    /// Pairwise assignment cost (lower is better).
    pub cost: f64,
    /// Assignment confidence score in `[0.0, 1.0]`.
    pub confidence: f64,
    /// True when manually locked/confirmed or swapped by the user.
    pub locked: bool,
}

/// Complete result of matching a collected 1D NMR peak list to a predicted structure spectrum.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AutoAssign1DResult {
    /// Target nucleus (`"1H"` or `"13C"`).
    pub nucleus: String,
    /// Measured spectrometer observation frequency in MHz (`e.g. 500.0` for ¹H, `125.7` for ¹³C);
    /// `0.0` when unknown (never fabricated).
    pub spectrometer_mhz: f64,
    /// Measured solvent identifier (`e.g. "CDCl3"`), or empty string when unknown (never fabricated).
    pub solvent: String,
    /// Optional measured experimental solvent identifier (`None` when unknown).
    #[serde(default)]
    pub experimental_solvent: Option<String>,
    /// Input predicted structure groups.
    pub predicted_groups: Vec<PredictedGroup1D>,
    /// Input experimental peaks.
    pub experimental_peaks: Vec<ExperimentalPeak1D>,
    /// One assignment entry per predicted group.
    pub assignments: Vec<AtomPeakAssignment>,
    /// Experimental peak indices left unassigned (e.g., solvent or impurity signals).
    pub unassigned_peaks: Vec<usize>,
    /// Total minimum assignment cost across all predicted groups.
    pub total_cost: f64,
    /// Root-mean-square shift deviation (ppm) over all matched pairs.
    pub rmsd_ppm: f64,
    /// Mean confidence score in `[0.0, 1.0]` over all predicted groups.
    pub mean_confidence: f64,
    /// Number of predicted groups matched to an experimental peak.
    pub matched_count: usize,
}

impl AutoAssign1DResult {
    /// Look up the assignment entry covering `atom_idx` on the 2D chemical structure.
    pub fn assignment_for_atom(&self, atom_idx: usize) -> Option<&AtomPeakAssignment> {
        self.assignments
            .iter()
            .find(|a| a.atom_indices.contains(&atom_idx))
    }

    /// Look up the first assignment entry matched to experimental `peak_idx`.
    pub fn assignment_for_peak(&self, peak_idx: usize) -> Option<&AtomPeakAssignment> {
        self.assignments
            .iter()
            .find(|a| a.peak_idx == Some(peak_idx))
    }

    /// Look up all assignment entries matched to experimental `peak_idx` (supporting
    /// many-to-one composite multiplets such as an overlapping 5H aromatic multiplet).
    pub fn assignments_for_peak(&self, peak_idx: usize) -> Vec<&AtomPeakAssignment> {
        self.assignments
            .iter()
            .filter(|a| a.peak_idx == Some(peak_idx))
            .collect()
    }

    /// Return all currently locked `(group_idx, peak_idx)` pairs.
    pub fn locked_pairs(&self) -> Vec<(usize, usize)> {
        self.assignments
            .iter()
            .filter_map(|a| (a.locked).then_some(a.peak_idx.map(|pk| (a.group_idx, pk)))?)
            .collect()
    }
}

/// Configuration weights for 1D experimental-to-predicted Hungarian assignment.
#[derive(Debug, Clone, Copy, PartialEq)]
struct AssignWeights {
    is_c13: bool,
    sigma_floor_ppm: f64,
    weight_shift: f64,
    weight_integral: f64,
    weight_multiplicity: f64,
    unassigned_cost: f64,
}

impl AssignWeights {
    fn for_nucleus(nucleus: &str) -> Self {
        let is_c13 = is_carbon_nucleus(nucleus);
        if is_c13 {
            Self {
                is_c13: true,
                sigma_floor_ppm: 3.5,
                weight_shift: 1.0,
                weight_integral: 0.05,
                weight_multiplicity: 0.0,
                unassigned_cost: 9.0,
            }
        } else {
            Self {
                is_c13: false,
                sigma_floor_ppm: 0.35,
                weight_shift: 1.0,
                weight_integral: 0.65,
                weight_multiplicity: 0.55,
                unassigned_cost: 9.0,
            }
        }
    }
}

/// Return true if `nucleus` represents `13C` (`"13C"`, `"C13"`, `"13C{1H}"`, etc.).
pub fn is_carbon_nucleus(nucleus: &str) -> bool {
    let up = nucleus.to_ascii_uppercase();
    up.contains("13C") || up.contains("C13") || up == "C"
}

/// Convert a slice of predicted [`SpectrumSignal1D`] items into [`PredictedGroup1D`] records.
/// Non-finite (`NaN`/`±∞`) chemical shifts are excluded cleanly.
pub fn predicted_groups_from_signals(
    signals: &[SpectrumSignal1D],
    nucleus: &str,
) -> Vec<PredictedGroup1D> {
    let is_c13 = is_carbon_nucleus(nucleus);
    signals
        .iter()
        .enumerate()
        .filter(|(_, s)| s.ppm.is_finite())
        .map(|(i, s)| PredictedGroup1D::from_spectrum_signal(i, s, is_c13))
        .collect()
}

/// Peak-preserving min-max envelope decimation for high-resolution 1D NMR traces.
///
/// Unlike stride decimation (`.step_by(n / 1024)`), which routinely skips over sharp `0.8 Hz`
/// singlets in `64k`-point `600 MHz` spectra, partitioning into `max_bins` buckets and retaining
/// both the minimum and maximum intensity point in each bucket guarantees that every sharp peak
/// apex is preserved at its exact chemical shift and full intensity.
pub fn decimate_trace_peak_preserving(
    ppm: &[f64],
    intensities: &[f64],
    max_bins: usize,
) -> Vec<[f64; 2]> {
    let n = ppm.len().min(intensities.len());
    if n == 0 {
        return Vec::new();
    }
    let pts: Vec<[f64; 2]> = ppm
        .iter()
        .zip(intensities.iter())
        .take(n)
        .filter(|(&x, &y)| x.is_finite() && y.is_finite())
        .map(|(&x, &y)| [x, y])
        .collect();
    if max_bins == 0 || pts.len() <= max_bins.saturating_mul(2) {
        return pts;
    }

    let len = pts.len();
    let mut out = Vec::with_capacity(max_bins.saturating_mul(2).min(len));
    for b in 0..max_bins {
        let start = (b * len) / max_bins;
        let end = (((b + 1) * len) / max_bins).min(len);
        if start >= end {
            continue;
        }
        let mut min_i = start;
        let mut max_i = start;
        let mut min_y = pts[start][1];
        let mut max_y = pts[start][1];
        for (i, &[_, y]) in pts.iter().enumerate().take(end).skip(start + 1) {
            if y < min_y {
                min_y = y;
                min_i = i;
            }
            if y > max_y {
                max_y = y;
                max_i = i;
            }
        }
        if min_i < max_i {
            out.push(pts[min_i]);
            out.push(pts[max_i]);
        } else if max_i < min_i {
            out.push(pts[max_i]);
            out.push(pts[min_i]);
        } else {
            out.push(pts[max_i]);
        }
    }
    out
}

/// Extract experimental 1D peaks directly from full-resolution `(ppm, intensities)` slices with
/// zero stride decimation, preserving sharp `0.8 Hz` FWHM singlets even in `64k`-point `600 MHz`
/// spectra.
///
/// Peaks are ordered deterministically downfield (high ppm) to upfield (low ppm). Multiplicity,
/// scalar `J`-couplings, and integrals are left unpopulated (`None` / empty) because raw trace
/// peak picking measures apex shifts and intensities only.
pub fn extract_experimental_peaks_full_res(
    ppm: &[f64],
    intensities: &[f64],
    nucleus: &str,
) -> Vec<ExperimentalPeak1D> {
    let n = ppm.len().min(intensities.len());
    if n < 3 {
        return Vec::new();
    }

    // Fast path when inputs are finite and already monotonic descending/ascending (no O(N log N) sort)
    let all_finite = ppm
        .iter()
        .zip(intensities.iter())
        .take(n)
        .all(|(&x, &y)| x.is_finite() && y.is_finite());
    let is_desc = all_finite && ppm[..n].windows(2).all(|w| w[0] >= w[1]);
    let is_asc = !is_desc && all_finite && ppm[..n].windows(2).all(|w| w[0] <= w[1]);

    let owned_pts: Vec<[f64; 2]>;
    let (pts_len, get_pt): (usize, Box<dyn Fn(usize) -> [f64; 2] + '_>) = if is_desc {
        (n, Box::new(move |i| [ppm[i], intensities[i]]))
    } else if is_asc {
        (
            n,
            Box::new(move |i| [ppm[n - 1 - i], intensities[n - 1 - i]]),
        )
    } else {
        let mut pts: Vec<[f64; 2]> = ppm
            .iter()
            .zip(intensities.iter())
            .take(n)
            .filter(|(&x, &y)| x.is_finite() && y.is_finite())
            .map(|(&x, &y)| [x, y])
            .collect();
        if pts.len() < 3 {
            return Vec::new();
        }
        // Sort downfield (high ppm) to upfield (low ppm) for deterministic peak ordering
        pts.sort_by(|a, b| b[0].total_cmp(&a[0]));
        owned_pts = pts;
        let len = owned_pts.len();
        (len, Box::new(move |i| owned_pts[i]))
    };

    let mut y_max = 0.0_f64;
    for i in 0..pts_len {
        let y = get_pt(i)[1];
        if y > y_max {
            y_max = y;
        }
    }
    if y_max <= 1e-12 {
        return Vec::new();
    }

    let threshold = 0.05 * y_max;
    let is_c13 = is_carbon_nucleus(nucleus);
    let min_sep_ppm = if is_c13 { 0.45 } else { 0.045 };

    let mut raw_peaks: Vec<(f64, f64)> = Vec::new();
    for i in 1..(pts_len - 1) {
        let p_prev = get_pt(i - 1);
        let p_cur = get_pt(i);
        let p_next = get_pt(i + 1);
        let y = p_cur[1];
        if y >= threshold && y >= p_prev[1] && y > p_next[1] {
            // Quadratic parabolic sub-grid apex refinement
            let y0 = p_prev[1];
            let y1 = y;
            let y2 = p_next[1];
            let denom = y0 - 2.0 * y1 + y2;
            let x_apex = if denom.abs() > 1e-12 {
                let offset = (0.5 * (y0 - y2) / denom).clamp(-0.5, 0.5);
                let step = 0.5 * (p_next[0] - p_prev[0]);
                p_cur[0] + offset * step
            } else {
                p_cur[0]
            };
            if !x_apex.is_finite() {
                continue;
            }
            if let Some(last) = raw_peaks.last_mut() {
                if (last.0 - x_apex).abs() < min_sep_ppm {
                    if y > last.1 {
                        *last = (x_apex, y);
                    }
                    continue;
                }
            }
            raw_peaks.push((x_apex, y));
        }
    }

    raw_peaks
        .into_iter()
        .enumerate()
        .map(|(idx, (peak_ppm, intensity))| ExperimentalPeak1D {
            peak_idx: idx,
            ppm: peak_ppm,
            integral: None,
            multiplicity: None,
            couplings_hz: Vec::new(),
            intensity: intensity / y_max,
        })
        .collect()
}

/// Extract experimental 1D peaks from a [`SceneSpectrum1D`].
///
/// Uses finite `spectrum.signals` directly when present; otherwise performs full-resolution
/// deterministic local-maximum peak picking on `spectrum.custom_trace` via
/// [`extract_experimental_peaks_full_res`].
pub fn extract_experimental_peaks_from_spectrum(
    spectrum: &SceneSpectrum1D,
) -> Vec<ExperimentalPeak1D> {
    if !spectrum.signals.is_empty() {
        return spectrum
            .signals
            .iter()
            .filter(|sig| sig.ppm.is_finite())
            .enumerate()
            .map(|(idx, sig)| ExperimentalPeak1D::from_signal(idx, sig))
            .collect();
    }

    if spectrum.custom_trace.len() < 3 {
        return Vec::new();
    }

    let mut ppms = Vec::with_capacity(spectrum.custom_trace.len());
    let mut ys = Vec::with_capacity(spectrum.custom_trace.len());
    for &[x, y] in &spectrum.custom_trace {
        if x.is_finite() && y.is_finite() {
            ppms.push(x);
            ys.push(y);
        }
    }
    extract_experimental_peaks_full_res(&ppms, &ys, &spectrum.nucleus)
}

/// Normalize a multiplicity string to a canonical lower-case token (`"s"`, `"d"`, `"t"`, `"q"`, etc.).
fn canonical_multiplicity(raw: &str) -> String {
    let trimmed = raw.trim().to_ascii_lowercase();
    if trimmed.is_empty() {
        return String::new();
    }
    let compact: String = trimmed.split_whitespace().collect::<Vec<_>>().join(" ");
    match compact.as_str() {
        "singlet" => "s".to_string(),
        "doublet" => "d".to_string(),
        "triplet" => "t".to_string(),
        "quartet" => "q".to_string(),
        "quintet" | "pentet" => "quint".to_string(),
        "sextet" => "sext".to_string(),
        "septet" => "sept".to_string(),
        "multiplet" => "m".to_string(),
        "br. s" | "brs" | "broad singlet" => "br s".to_string(),
        other => other.to_string(),
    }
}

/// Compute multiplicity inconsistency penalty in `[0.0, 2.0]` between predicted and experimental patterns.
fn multiplicity_mismatch_penalty(pred_raw: &str, exp_raw: Option<&str>) -> f64 {
    let Some(exp_s) = exp_raw else {
        return 0.0;
    };
    let p = canonical_multiplicity(pred_raw);
    let e = canonical_multiplicity(exp_s);
    if p.is_empty() || e.is_empty() || p == e {
        return 0.0;
    }
    // Broad singlet vs sharp singlet
    if (p == "br s" && e == "s") || (p == "s" && e == "br s") {
        return 0.20;
    }
    // Unresolved multiplet `m` is mildly compatible with complex splitting patterns
    let is_complex = |s: &str| {
        matches!(
            s,
            "dd" | "dt" | "td" | "ddd" | "t" | "q" | "quint" | "sext" | "sept" | "dq" | "qd"
        )
    };
    if (e == "m" && is_complex(&p)) || (p == "m" && is_complex(&e)) {
        return 0.25;
    }
    if (e == "m" && p == "d") || (p == "m" && e == "d") {
        return 0.55;
    }
    // Apparent triplet (`t`) vs doublet-of-doublets (`dd`) with similar J constants
    if (p == "dd" && e == "t") || (p == "t" && e == "dd") {
        return 0.35;
    }
    if (p == "ddd" && (e == "dd" || e == "dt" || e == "td"))
        || (e == "ddd" && (p == "dd" || p == "dt" || p == "td"))
    {
        return 0.35;
    }
    if (p == "dt" && e == "td") || (p == "td" && e == "dt") {
        return 0.20;
    }
    if (p == "s" || p == "br s") != (e == "s" || e == "br s") {
        return 2.0;
    }
    1.35
}

/// Compute scale factor bringing raw experimental integrals onto the predicted proton scale
/// when experimental integrals are clearly in arbitrary unnormalized units.
fn integral_scale_factor(
    predicted: &[PredictedGroup1D],
    experimental: &[ExperimentalPeak1D],
    is_c13: bool,
) -> f64 {
    if is_c13 {
        return 1.0;
    }
    let sum_pred: f64 = predicted.iter().map(|g| g.expected_integral.max(0.0)).sum();
    let sum_exp: f64 = experimental
        .iter()
        .filter_map(|p| p.integral.filter(|v| v.is_finite() && *v > 0.0))
        .sum();
    if sum_pred <= 1e-6 || sum_exp <= 1e-6 {
        return 1.0;
    }
    let ratio = sum_exp / sum_pred;
    if !(0.35..=2.5).contains(&ratio) {
        sum_pred / sum_exp
    } else {
        1.0
    }
}

/// Compute the pairwise assignment cost between one predicted group and one experimental peak.
/// Returns a prohibitive cost (`1e6`) when either chemical shift is non-finite (`NaN`/`±∞`).
fn pairwise_cost(
    group: &PredictedGroup1D,
    peak: &ExperimentalPeak1D,
    weights: AssignWeights,
    int_scale: f64,
) -> f64 {
    if !group.shift_ppm.is_finite() || !peak.ppm.is_finite() {
        return 1e6;
    }
    let pred_ppm = group.shift_ppm;
    let obs_ppm = peak.ppm;
    let sigma = if group.sd_ppm.is_finite() && group.sd_ppm > 0.0 {
        group.sd_ppm.max(weights.sigma_floor_ppm)
    } else {
        weights.sigma_floor_ppm
    };

    let z_shift = (obs_ppm - pred_ppm).abs() / sigma;
    let c_shift = weights.weight_shift * z_shift * z_shift;

    let c_int = if weights.weight_integral > 0.0 {
        if let Some(obs_int) = peak.integral.filter(|v| v.is_finite() && *v > 0.0) {
            let exp_int = group.expected_integral.max(0.5);
            let norm_obs = obs_int * int_scale;
            let rel_err = (norm_obs - exp_int).abs() / exp_int;
            weights.weight_integral * rel_err * rel_err
        } else {
            0.0
        }
    } else {
        0.0
    };

    let c_mult = if weights.weight_multiplicity > 0.0 {
        weights.weight_multiplicity
            * multiplicity_mismatch_penalty(&group.multiplicity, peak.multiplicity.as_deref())
    } else {
        0.0
    };

    (c_shift + c_int + c_mult).min(1e4)
}

/// Compute confidence in `[0.0, 1.0]` from pairwise fit cost and assignment margin.
fn confidence_from_cost_and_margin(cost_ij: f64, margin: f64) -> f64 {
    let fit_factor = (-0.25 * cost_ij.max(0.0)).exp();
    let margin_factor = 1.0 / (1.0 + (-margin.max(0.0)).exp());
    (fit_factor * margin_factor).clamp(0.0, 1.0)
}

/// Optimally assign collected experimental 1D NMR peaks (`¹H` or `¹³C`) to predicted
/// chemically-equivalent structure groups via exact Kuhn–Munkres (Hungarian) bipartite matching,
/// followed by integral-capacity-aware many-to-one composite multiplet matching (e.g. a 5H
/// toluene aromatic multiplet covering 2H + 2H + 1H aromatic groups).
///
/// The cost matrix combines:
/// 1. Normalized squared chemical shift residual `(Δδ / σ)^2`,
/// 2. Relative integral consistency (for ¹H when integrals are available), and
/// 3. Multiplicity pattern compatibility (for ¹H when multiplicities are available).
///
/// Unmatched groups (outliers beyond the `3σ` cutoff or missing peaks) and unmatched experimental
/// peaks (solvent/impurity resonances) are handled natively via augmented dummy columns.
pub fn auto_assign_1d_to_structure(
    predicted: &[PredictedGroup1D],
    experimental: &[ExperimentalPeak1D],
    nucleus: &str,
    spectrometer_mhz: f64,
    solvent: &str,
) -> AutoAssign1DResult {
    auto_assign_1d_to_structure_with_locks(
        predicted,
        experimental,
        nucleus,
        spectrometer_mhz,
        solvent,
        &[],
    )
}

/// Variant of [`auto_assign_1d_to_structure`] that strictly preserves user-locked
/// `(group_idx, peak_idx)` pairs while globally optimizing all remaining free groups via Hungarian
/// and many-to-one composite multiplet matching.
pub fn auto_assign_1d_to_structure_with_locks(
    predicted: &[PredictedGroup1D],
    experimental: &[ExperimentalPeak1D],
    nucleus: &str,
    spectrometer_mhz: f64,
    solvent: &str,
    locked_pairs: &[(usize, usize)],
) -> AutoAssign1DResult {
    let weights = AssignWeights::for_nucleus(nucleus);
    let canon_nucleus = if weights.is_c13 { "13C" } else { "1H" }.to_string();
    // Preserve exact measured spectrometer frequency; `0.0` denotes unknown (never fabricated).
    let freq_mhz = if spectrometer_mhz.is_finite() && spectrometer_mhz > 0.0 {
        spectrometer_mhz
    } else {
        0.0
    };
    let solv = solvent.trim().to_string();
    let experimental_solvent = (!solv.is_empty()).then(|| solv.clone());

    let n = predicted.len();
    let m = experimental.len();
    if n == 0 {
        return AutoAssign1DResult {
            nucleus: canon_nucleus,
            spectrometer_mhz: freq_mhz,
            solvent: solv,
            experimental_solvent,
            predicted_groups: Vec::new(),
            experimental_peaks: experimental.to_vec(),
            assignments: Vec::new(),
            unassigned_peaks: experimental.iter().map(|p| p.peak_idx).collect(),
            total_cost: 0.0,
            rmsd_ppm: 0.0,
            mean_confidence: 0.0,
            matched_count: 0,
        };
    }

    let int_scale = integral_scale_factor(predicted, experimental, weights.is_c13);

    // Map `group_idx` -> row position and `peak_idx` -> col position
    let group_pos_by_id: BTreeMap<usize, usize> = predicted
        .iter()
        .enumerate()
        .map(|(row, g)| (g.group_idx, row))
        .collect();
    let peak_pos_by_id: BTreeMap<usize, usize> = experimental
        .iter()
        .enumerate()
        .map(|(col, p)| (p.peak_idx, col))
        .collect();

    // Resolve valid 1-to-1 user locks (rejecting any non-finite shifts)
    let mut locked_row_to_col: BTreeMap<usize, usize> = BTreeMap::new();
    let mut locked_cols: BTreeSet<usize> = BTreeSet::new();
    for &(gid, pid) in locked_pairs {
        if let (Some(&row), Some(&col)) = (group_pos_by_id.get(&gid), peak_pos_by_id.get(&pid)) {
            if predicted[row].shift_ppm.is_finite()
                && experimental[col].ppm.is_finite()
                && !locked_row_to_col.contains_key(&row)
                && !locked_cols.contains(&col)
            {
                locked_row_to_col.insert(row, col);
                locked_cols.insert(col);
            }
        }
    }

    // Collect free rows and free columns for the Hungarian subproblem
    let free_rows: Vec<usize> = (0..n)
        .filter(|r| !locked_row_to_col.contains_key(r))
        .collect();
    let free_cols: Vec<usize> = (0..m).filter(|c| !locked_cols.contains(c)).collect();

    let nf = free_rows.len();
    let mf = free_cols.len();

    let mut matched_col_for_row: Vec<Option<usize>> = vec![None; n];
    let mut row_cost: Vec<f64> = vec![weights.unassigned_cost; n];
    let mut row_confidence: Vec<f64> = vec![0.0; n];

    for (&row, &col) in &locked_row_to_col {
        matched_col_for_row[row] = Some(col);
        row_cost[row] = pairwise_cost(&predicted[row], &experimental[col], weights, int_scale);
        row_confidence[row] = 1.0;
    }

    if nf > 0 {
        // Pass 1: Augmented `nf x (mf + nf)` 1-to-1 Hungarian matching
        let total_cols = mf + nf;
        let mut aug_cost = vec![vec![weights.unassigned_cost; total_cols]; nf];
        for (fr_i, &row) in free_rows.iter().enumerate() {
            for (fc_j, &col) in free_cols.iter().enumerate() {
                aug_cost[fr_i][fc_j] =
                    pairwise_cost(&predicted[row], &experimental[col], weights, int_scale);
            }
        }

        let (assignment_cols, best_total) = hungarian_min_cost_assignment(&aug_cost)
            .unwrap_or_else(|_| {
                (
                    (0..nf).map(|k| mf + k).collect(),
                    nf as f64 * weights.unassigned_cost,
                )
            });

        let use_exact_loo = nf <= 32;
        let mut scratch = if use_exact_loo {
            aug_cost.clone()
        } else {
            Vec::new()
        };

        for (fr_i, &assigned_c) in assignment_cols.iter().enumerate() {
            let row = free_rows[fr_i];
            if assigned_c < mf {
                let col = free_cols[assigned_c];
                if predicted[row].shift_ppm.is_finite() && experimental[col].ppm.is_finite() {
                    matched_col_for_row[row] = Some(col);
                    let c_ij = aug_cost[fr_i][assigned_c];
                    row_cost[row] = c_ij;

                    // Compute assignment margin against best runner-up alternative
                    let margin = if use_exact_loo {
                        let saved = scratch[fr_i][assigned_c];
                        scratch[fr_i][assigned_c] = 1e6;
                        let alt_margin = match hungarian_min_cost_assignment(&scratch) {
                            Ok((_, alt_total)) => (alt_total - best_total).max(0.0),
                            Err(_) => 0.0,
                        };
                        scratch[fr_i][assigned_c] = saved;
                        alt_margin
                    } else {
                        let runner_up = aug_cost[fr_i]
                            .iter()
                            .enumerate()
                            .filter(|&(c, _)| c != assigned_c)
                            .map(|(_, &v)| v)
                            .fold(f64::INFINITY, f64::min);
                        if runner_up.is_finite() {
                            (runner_up - c_ij).max(0.0)
                        } else {
                            0.0
                        }
                    };
                    row_confidence[row] = confidence_from_cost_and_margin(c_ij, margin);
                }
            } else {
                matched_col_for_row[row] = None;
                row_cost[row] = weights.unassigned_cost;
                row_confidence[row] = 0.0;
            }
        }
    }

    // Pass 2: Many-to-one composite multiplet matching when an experimental peak has a measured
    // integral covering multiple overlapping predicted groups within shift tolerance
    // (e.g., toluene's 5H aromatic multiplet covering 2H + 2H + 1H aromatic groups).
    if m > 0 {
        let mut used_integral_by_col = vec![0.0_f64; m];
        for row in 0..n {
            if let Some(col) = matched_col_for_row[row] {
                used_integral_by_col[col] += predicted[row].expected_integral.max(0.5);
            }
        }

        let composite_mult_cost = |g: &PredictedGroup1D, pk: &ExperimentalPeak1D| -> f64 {
            if weights.weight_multiplicity <= 0.0 {
                return 0.0;
            }
            let Some(exp_mult) = pk.multiplicity.as_deref() else {
                return 0.0;
            };
            let e = canonical_multiplicity(exp_mult);
            if e.is_empty() || e == "m" || e == "br s" {
                0.0
            } else {
                0.5 * weights.weight_multiplicity
                    * multiplicity_mismatch_penalty(&g.multiplicity, Some(exp_mult))
            }
        };

        let mut composite_touched_cols: BTreeSet<usize> = BTreeSet::new();
        loop {
            let mut best_move: Option<(usize, usize, f64)> = None;

            for row in 0..n {
                if locked_row_to_col.contains_key(&row) {
                    continue;
                }
                let g = &predicted[row];
                if !g.shift_ppm.is_finite() {
                    continue;
                }
                let g_int = g.expected_integral.max(0.5);
                let sigma = if g.sd_ppm.is_finite() && g.sd_ppm > 0.0 {
                    g.sd_ppm.max(weights.sigma_floor_ppm)
                } else {
                    weights.sigma_floor_ppm
                };

                // Eligible to join a composite multiplet if currently unassigned, OR if forced in
                // Pass 1 onto a severely under-integrated peak (e.g. a tiny residual solvent peak).
                let cur_cost = match matched_col_for_row[row] {
                    None => weights.unassigned_cost,
                    Some(old_col) => {
                        let old_norm_obs = experimental[old_col]
                            .integral
                            .filter(|v| v.is_finite() && *v > 0.0)
                            .map(|v| v * int_scale);
                        if old_norm_obs.is_some_and(|obs| obs < 0.5 * g_int) && row_cost[row] > 1.0
                        {
                            row_cost[row]
                        } else {
                            continue;
                        }
                    }
                };

                for col in 0..m {
                    if matched_col_for_row[row] == Some(col) {
                        continue;
                    }
                    let pk = &experimental[col];
                    if !pk.ppm.is_finite() {
                        continue;
                    }
                    let Some(obs_int) = pk.integral.filter(|v| v.is_finite() && *v > 0.0) else {
                        continue;
                    };
                    let norm_obs = obs_int * int_scale;
                    let used = used_integral_by_col[col];
                    let tol = (0.18 * norm_obs).max(0.35);

                    // Require genuine remaining measured integral capacity for this group
                    if used + g_int > norm_obs + tol || norm_obs < used + 0.5 * g_int {
                        continue;
                    }

                    let z_shift = (pk.ppm - g.shift_ppm).abs() / sigma;
                    let c_shift = weights.weight_shift * z_shift * z_shift;
                    if c_shift > weights.unassigned_cost {
                        continue;
                    }

                    let new_used = used + g_int;
                    let c_int = if weights.weight_integral > 0.0 && new_used > norm_obs {
                        let over = (new_used - norm_obs) / norm_obs.max(0.5);
                        weights.weight_integral * over * over
                    } else {
                        0.0
                    };
                    let c_mult = composite_mult_cost(g, pk);
                    let c_join = (c_shift + c_int + c_mult).min(1e4);

                    if c_join + 1e-9 < cur_cost {
                        let is_better = match best_move {
                            None => true,
                            Some((_, _, best_c)) => c_join < best_c - 1e-9,
                        };
                        if is_better {
                            best_move = Some((row, col, c_join));
                        }
                    }
                }
            }

            let Some((best_row, best_col, _)) = best_move else {
                break;
            };
            let g_int = predicted[best_row].expected_integral.max(0.5);
            if let Some(old_col) = matched_col_for_row[best_row] {
                used_integral_by_col[old_col] = (used_integral_by_col[old_col] - g_int).max(0.0);
            }
            matched_col_for_row[best_row] = Some(best_col);
            used_integral_by_col[best_col] += g_int;
            composite_touched_cols.insert(best_col);
        }

        // Recompute composite cost & honest confidence for every peak modified by Pass 2
        for col in composite_touched_cols {
            let pk = &experimental[col];
            let norm_obs = pk
                .integral
                .filter(|v| v.is_finite() && *v > 0.0)
                .map(|v| v * int_scale);
            let sum_int = used_integral_by_col[col].max(0.5);
            let c_int_comp = if weights.weight_integral > 0.0 {
                if let Some(obs) = norm_obs {
                    let rel = (obs - sum_int).abs() / sum_int;
                    weights.weight_integral * rel * rel
                } else {
                    0.0
                }
            } else {
                0.0
            };
            for row in 0..n {
                if matched_col_for_row[row] != Some(col) {
                    continue;
                }
                let g = &predicted[row];
                let sigma = if g.sd_ppm.is_finite() && g.sd_ppm > 0.0 {
                    g.sd_ppm.max(weights.sigma_floor_ppm)
                } else {
                    weights.sigma_floor_ppm
                };
                let z_shift = (pk.ppm - g.shift_ppm).abs() / sigma;
                let c_shift = weights.weight_shift * z_shift * z_shift;
                let c_mult = composite_mult_cost(g, pk);
                let comp_cost = (c_shift + c_int_comp + c_mult).min(1e4);
                row_cost[row] = comp_cost;

                if locked_row_to_col.contains_key(&row) {
                    row_confidence[row] = 1.0;
                } else {
                    let mut runner_up = weights.unassigned_cost;
                    for (alt_col, alt_pk) in experimental.iter().enumerate() {
                        if alt_col != col {
                            let alt_c = pairwise_cost(g, alt_pk, weights, int_scale);
                            if alt_c < runner_up {
                                runner_up = alt_c;
                            }
                        }
                    }
                    let margin = (runner_up - comp_cost).max(0.0);
                    row_confidence[row] = confidence_from_cost_and_margin(comp_cost, margin);
                }
            }
        }
    }

    let mut assignments = Vec::with_capacity(n);
    for row in 0..n {
        let g = &predicted[row];
        let locked = locked_row_to_col.contains_key(&row);
        if let Some(col) = matched_col_for_row[row] {
            let pk = &experimental[col];
            assignments.push(AtomPeakAssignment {
                group_idx: g.group_idx,
                atom_indices: g.atom_indices.clone(),
                label: g.label.clone(),
                predicted_ppm: g.shift_ppm,
                peak_idx: Some(pk.peak_idx),
                experimental_ppm: Some(pk.ppm),
                delta_ppm: Some(pk.ppm - g.shift_ppm),
                cost: row_cost[row],
                confidence: row_confidence[row],
                locked,
            });
        } else {
            assignments.push(AtomPeakAssignment {
                group_idx: g.group_idx,
                atom_indices: g.atom_indices.clone(),
                label: g.label.clone(),
                predicted_ppm: g.shift_ppm,
                peak_idx: None,
                experimental_ppm: None,
                delta_ppm: None,
                cost: weights.unassigned_cost,
                confidence: 0.0,
                locked: false,
            });
        }
    }

    let mut result = AutoAssign1DResult {
        nucleus: canon_nucleus,
        spectrometer_mhz: freq_mhz,
        solvent: solv,
        experimental_solvent,
        predicted_groups: predicted.to_vec(),
        experimental_peaks: experimental.to_vec(),
        assignments,
        unassigned_peaks: Vec::new(),
        total_cost: 0.0,
        rmsd_ppm: 0.0,
        mean_confidence: 0.0,
        matched_count: 0,
    };
    recompute_result_summaries(&mut result);
    result
}

/// Recompute `unassigned_peaks`, `total_cost`, `rmsd_ppm`, `mean_confidence`, and `matched_count`.
fn recompute_result_summaries(result: &mut AutoAssign1DResult) {
    let assigned_peak_ids: BTreeSet<usize> = result
        .assignments
        .iter()
        .filter_map(|a| a.peak_idx)
        .collect();

    result.unassigned_peaks = result
        .experimental_peaks
        .iter()
        .map(|p| p.peak_idx)
        .filter(|pid| !assigned_peak_ids.contains(pid))
        .collect();

    let mut total_cost = 0.0;
    let mut sq_err_sum = 0.0;
    let mut matched = 0usize;
    let mut conf_sum = 0.0;

    for a in &result.assignments {
        total_cost += a.cost;
        conf_sum += a.confidence.clamp(0.0, 1.0);
        if let Some(d) = a.delta_ppm.filter(|v| v.is_finite()) {
            sq_err_sum += d * d;
            matched += 1;
        }
    }

    result.total_cost = total_cost;
    result.matched_count = matched;
    result.rmsd_ppm = if matched > 0 {
        (sq_err_sum / matched as f64).sqrt()
    } else {
        0.0
    };
    result.mean_confidence = if result.assignments.is_empty() {
        0.0
    } else {
        (conf_sum / result.assignments.len() as f64).clamp(0.0, 1.0)
    };
}

/// Manually override, swap, or lock/unlock an experimental peak assignment on `result`.
///
/// Behavior:
/// - If `target_peak_idx` is `Some(pk)` and another group `other` already holds `Some(pk)`,
///   `group_idx` and `other` **swap** their peaks atomically (preserving 1-to-1 bijection).
/// - If `target_peak_idx` is `Some(pk)` and `pk` is currently unassigned, `group_idx` claims `pk`
///   and its previous peak (if any) becomes unassigned.
/// - If `target_peak_idx` is `None`, `group_idx` is unassigned.
/// - `group_idx`'s `locked` flag is updated to `lock` (when locked & assigned, confidence becomes `1.0`).
pub fn swap_or_lock_assignment(
    result: &mut AutoAssign1DResult,
    group_idx: usize,
    target_peak_idx: Option<usize>,
    lock: bool,
) -> Result<(), String> {
    let weights = AssignWeights::for_nucleus(&result.nucleus);
    let int_scale = integral_scale_factor(
        &result.predicted_groups,
        &result.experimental_peaks,
        weights.is_c13,
    );

    let Some(row) = result
        .assignments
        .iter()
        .position(|a| a.group_idx == group_idx)
    else {
        return Err(format!(
            "Predicted group {group_idx} not found in assignment result"
        ));
    };

    if let Some(pk) = target_peak_idx {
        let Some(exp_pk) = result.experimental_peaks.iter().find(|p| p.peak_idx == pk) else {
            return Err(format!(
                "Experimental peak {pk} not found in assignment result"
            ));
        };
        if !exp_pk.ppm.is_finite() {
            return Err(format!(
                "Experimental peak {pk} has non-finite chemical shift"
            ));
        }
        if !result.assignments[row].predicted_ppm.is_finite() {
            return Err(format!(
                "Predicted group {group_idx} has non-finite chemical shift"
            ));
        }
    }

    let prev_peak = result.assignments[row].peak_idx;

    // Check if another group currently holds `target_peak_idx` -> swap!
    let other_row = target_peak_idx.and_then(|pk| {
        result
            .assignments
            .iter()
            .enumerate()
            .find(|&(idx, a)| idx != row && a.peak_idx == Some(pk))
            .map(|(idx, _)| idx)
    });

    // Helper to apply a peak (or None) to a given assignment row
    let apply_peak_to_row =
        |res: &mut AutoAssign1DResult, r: usize, pk_opt: Option<usize>, locked_flag: bool| {
            let g = res
                .predicted_groups
                .iter()
                .find(|g| g.group_idx == res.assignments[r].group_idx)
                .cloned();
            if let (Some(pk_id), Some(group)) = (pk_opt, g) {
                if let Some(peak) = res.experimental_peaks.iter().find(|p| p.peak_idx == pk_id) {
                    if peak.ppm.is_finite() && group.shift_ppm.is_finite() {
                        let c = pairwise_cost(&group, peak, weights, int_scale);
                        let conf = if locked_flag {
                            1.0
                        } else {
                            confidence_from_cost_and_margin(c, 1.0)
                        };
                        let entry = &mut res.assignments[r];
                        entry.peak_idx = Some(pk_id);
                        entry.experimental_ppm = Some(peak.ppm);
                        entry.delta_ppm = Some(peak.ppm - group.shift_ppm);
                        entry.cost = c;
                        entry.confidence = conf;
                        entry.locked = locked_flag;
                        return;
                    }
                }
            }
            let entry = &mut res.assignments[r];
            entry.peak_idx = None;
            entry.experimental_ppm = None;
            entry.delta_ppm = None;
            entry.cost = weights.unassigned_cost;
            entry.confidence = 0.0;
            entry.locked = false;
        };

    if let Some(orow) = other_row {
        // Atomic swap: `other` gets `prev_peak`, `row` gets `target_peak_idx`
        let other_locked = result.assignments[orow].locked && prev_peak.is_some();
        apply_peak_to_row(result, orow, prev_peak, other_locked);
        apply_peak_to_row(result, row, target_peak_idx, lock);
    } else {
        apply_peak_to_row(result, row, target_peak_idx, lock);
    }

    recompute_result_summaries(result);
    Ok(())
}

/// Format an honest ACS journal characterization report string using **only** measured
/// experimental values (frequency, solvent, chemical shifts, multiplicities, unique sorted
/// `J`-couplings, and integrals) annotated with true structure atom/group labels.
///
/// Rules enforced:
/// - Never fabricates spectrometer frequency (`600 MHz` / `500 MHz`) or solvent (`CDCl3`) when
///   unknown; omits unknown header fields cleanly (`¹H NMR (400 MHz)` or `¹H NMR`).
/// - Never copies predicted multiplicity, predicted `J`-couplings, or predicted integrals onto
///   experimental peaks where those quantities were not measured.
/// - Formats `J`-couplings as unique descending values (`J = 7.0 Hz`, never `7.0, 7.0, 7.0 Hz`).
/// - Uses true atom/group labels (`H-3`, `CH2`, `C-1`), never duplicating nuclei counts (`2H, 2H`).
/// - Groups composite many-to-one multiplet assignments into a single peak entry.
/// - Excludes any non-finite (`NaN`/`±∞`) chemical shifts cleanly.
pub fn format_acs_with_atom_labels(result: &AutoAssign1DResult) -> String {
    let is_c13 = is_carbon_nucleus(&result.nucleus);
    let nuc_label = if is_c13 { "¹³C NMR" } else { "¹H NMR" };
    let freq_part = (result.spectrometer_mhz.is_finite() && result.spectrometer_mhz > 0.0)
        .then(|| format!("{} MHz", result.spectrometer_mhz.round().max(1.0) as usize));
    let solvent_part = result
        .experimental_solvent
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .or_else(|| {
            let s = result.solvent.trim();
            (!s.is_empty()).then_some(s)
        });

    let header = match (freq_part, solvent_part) {
        (Some(f), Some(s)) => format!("{nuc_label} ({f}, {s})"),
        (Some(f), None) => format!("{nuc_label} ({f})"),
        (None, Some(s)) => format!("{nuc_label} ({s})"),
        (None, None) => nuc_label.to_string(),
    };

    // Group finite assigned entries by experimental peak so many-to-one composite multiplets emit
    // one merged entry with all constituent atom labels rather than duplicating the peak/integral.
    struct PeakGroup<'a> {
        peak_idx: Option<usize>,
        ppm: f64,
        members: Vec<&'a AtomPeakAssignment>,
    }

    let mut grouped: Vec<PeakGroup<'_>> = Vec::new();
    let mut by_peak_pos: BTreeMap<usize, usize> = BTreeMap::new();
    for a in &result.assignments {
        let Some(ppm) = a.experimental_ppm.filter(|p| p.is_finite()) else {
            continue;
        };
        if let Some(pid) = a.peak_idx {
            if let Some(&pos) = by_peak_pos.get(&pid) {
                grouped[pos].members.push(a);
            } else {
                let pos = grouped.len();
                by_peak_pos.insert(pid, pos);
                grouped.push(PeakGroup {
                    peak_idx: Some(pid),
                    ppm,
                    members: vec![a],
                });
            }
        } else {
            grouped.push(PeakGroup {
                peak_idx: None,
                ppm,
                members: vec![a],
            });
        }
    }

    // Sort downfield (high ppm) to upfield (low ppm)
    grouped.sort_by(|a, b| b.ppm.total_cmp(&a.ppm));

    if grouped.is_empty() {
        return format!("{header} δ (no assigned peaks).");
    }

    let resolve_labels = |members: &[&AtomPeakAssignment]| -> String {
        let mut labels: Vec<String> = Vec::with_capacity(members.len());
        for a in members {
            let trimmed = a.label.trim();
            let clean = if trimmed.is_empty() || is_bare_nuclei_count_label(trimmed, is_c13) {
                format_atom_group_label(&a.atom_indices, a.group_idx, is_c13)
            } else {
                trimmed.to_string()
            };
            if !labels.contains(&clean) {
                labels.push(clean);
            }
        }
        labels.join(", ")
    };

    if is_c13 {
        let entries: Vec<String> = grouped
            .iter()
            .map(|pg| format!("{:.1} ({})", pg.ppm, resolve_labels(&pg.members)))
            .collect();
        return format!("{header} δ {}.", entries.join(", "));
    }

    let entries: Vec<String> = grouped
        .iter()
        .map(|pg| {
            let peak = pg
                .peak_idx
                .and_then(|pid| result.experimental_peaks.iter().find(|p| p.peak_idx == pid));

            // Strictly use ONLY measured experimental multiplicity, J-couplings, and integral!
            let exp_mult = peak
                .and_then(|p| p.multiplicity.as_deref())
                .map(str::trim)
                .filter(|s| !s.is_empty());

            let exp_js = peak
                .map(|p| deduplicate_couplings_hz(&p.couplings_hz))
                .unwrap_or_default();

            let show_j = !exp_js.is_empty() && !matches!(exp_mult, Some("m" | "s" | "br s"));

            let exp_int = peak
                .and_then(|p| p.integral)
                .filter(|v| v.is_finite() && *v > 0.0)
                .map(|v| v.round().max(1.0) as usize);

            let mut parts: Vec<String> = Vec::with_capacity(4);
            if let Some(m) = exp_mult {
                parts.push(m.to_string());
            }
            if show_j {
                let formatted = exp_js
                    .iter()
                    .map(|j| format!("{j:.1}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                if !formatted.is_empty() {
                    parts.push(format!("J = {formatted} Hz"));
                }
            }
            if let Some(n_h) = exp_int {
                parts.push(format!("{n_h}H"));
            }
            parts.push(resolve_labels(&pg.members));

            format!("{:.2} ({})", pg.ppm, parts.join(", "))
        })
        .collect();

    format!("{header} δ {}.", entries.join(", "))
}
