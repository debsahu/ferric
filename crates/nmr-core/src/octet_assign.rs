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
    pub fn from_signal(peak_idx: usize, sig: &SpectrumSignal1D) -> Self {
        let mult = sig.multiplicity.trim();
        Self {
            peak_idx,
            ppm: if sig.ppm.is_finite() { sig.ppm } else { 0.0 },
            integral: (sig.integral.is_finite() && sig.integral > 0.0).then_some(sig.integral),
            multiplicity: (!mult.is_empty()).then(|| mult.to_string()),
            couplings_hz: sig
                .couplings_hz
                .iter()
                .copied()
                .filter(|j| j.is_finite() && *j > 0.0)
                .collect(),
            intensity: if sig.integral.is_finite() && sig.integral > 0.0 {
                sig.integral
            } else {
                1.0
            },
        }
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
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| {
                let prefix = if is_c13 { "C" } else { "H" };
                if let Some(&first_atom) = sig.atom_indices.first() {
                    format!("{prefix}-{}", first_atom + 1)
                } else {
                    format!("{prefix}-{}", group_idx + 1)
                }
            });
        Self {
            group_idx,
            atom_indices: sig.atom_indices.clone(),
            label: default_label,
            shift_ppm: if sig.ppm.is_finite() { sig.ppm } else { 0.0 },
            sd_ppm: if is_c13 { 4.0 } else { 0.40 },
            expected_integral: if sig.integral.is_finite() && sig.integral > 0.0 {
                sig.integral
            } else {
                1.0
            },
            multiplicity: sig.multiplicity.trim().to_string(),
            couplings_hz: sig
                .couplings_hz
                .iter()
                .copied()
                .filter(|j| j.is_finite() && *j > 0.0)
                .collect(),
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
    /// Spectrometer observation frequency in MHz (`e.g. 500.0` for ¹H, `126.0` for ¹³C).
    pub spectrometer_mhz: f64,
    /// Solvent identifier (`e.g. "CDCl3"`).
    pub solvent: String,
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

    /// Look up the assignment entry matched to experimental `peak_idx`.
    pub fn assignment_for_peak(&self, peak_idx: usize) -> Option<&AtomPeakAssignment> {
        self.assignments
            .iter()
            .find(|a| a.peak_idx == Some(peak_idx))
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
pub fn predicted_groups_from_signals(
    signals: &[SpectrumSignal1D],
    nucleus: &str,
) -> Vec<PredictedGroup1D> {
    let is_c13 = is_carbon_nucleus(nucleus);
    signals
        .iter()
        .enumerate()
        .map(|(i, s)| PredictedGroup1D::from_spectrum_signal(i, s, is_c13))
        .collect()
}

/// Extract experimental 1D peaks from a [`SceneSpectrum1D`].
///
/// Uses `spectrum.signals` directly when present; otherwise performs deterministic local-maximum
/// peak picking on `spectrum.custom_trace` above a 5% relative intensity threshold.
pub fn extract_experimental_peaks_from_spectrum(
    spectrum: &SceneSpectrum1D,
) -> Vec<ExperimentalPeak1D> {
    if !spectrum.signals.is_empty() {
        return spectrum
            .signals
            .iter()
            .enumerate()
            .map(|(idx, sig)| ExperimentalPeak1D::from_signal(idx, sig))
            .collect();
    }

    if spectrum.custom_trace.len() < 3 {
        return Vec::new();
    }

    let mut pts: Vec<[f64; 2]> = spectrum
        .custom_trace
        .iter()
        .copied()
        .filter(|p| p[0].is_finite() && p[1].is_finite())
        .collect();
    if pts.len() < 3 {
        return Vec::new();
    }
    // Sort downfield (high ppm) to upfield (low ppm) for deterministic peak ordering
    pts.sort_by(|a, b| b[0].total_cmp(&a[0]));

    let y_max = pts.iter().map(|p| p[1]).fold(0.0_f64, f64::max);
    if y_max <= 1e-12 {
        return Vec::new();
    }
    let threshold = 0.05 * y_max;
    let is_c13 = is_carbon_nucleus(&spectrum.nucleus);
    let min_sep_ppm = if is_c13 { 0.45 } else { 0.045 };

    let mut raw_peaks: Vec<(f64, f64)> = Vec::new();
    for i in 1..(pts.len() - 1) {
        let y = pts[i][1];
        if y >= threshold && y >= pts[i - 1][1] && y > pts[i + 1][1] {
            // Quadratic parabolic sub-grid apex refinement
            let y0 = pts[i - 1][1];
            let y1 = y;
            let y2 = pts[i + 1][1];
            let denom = y0 - 2.0 * y1 + y2;
            let x_apex = if denom.abs() > 1e-12 {
                let offset = (0.5 * (y0 - y2) / denom).clamp(-0.5, 0.5);
                let step = 0.5 * (pts[i + 1][0] - pts[i - 1][0]);
                pts[i][0] + offset * step
            } else {
                pts[i][0]
            };
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
        .map(|(idx, (ppm, intensity))| ExperimentalPeak1D {
            peak_idx: idx,
            ppm,
            integral: None,
            multiplicity: None,
            couplings_hz: Vec::new(),
            intensity: intensity / y_max,
        })
        .collect()
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
fn pairwise_cost(
    group: &PredictedGroup1D,
    peak: &ExperimentalPeak1D,
    weights: AssignWeights,
    int_scale: f64,
) -> f64 {
    let pred_ppm = if group.shift_ppm.is_finite() {
        group.shift_ppm
    } else {
        0.0
    };
    let obs_ppm = if peak.ppm.is_finite() { peak.ppm } else { 0.0 };
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
/// chemically-equivalent structure groups via exact Kuhn–Munkres (Hungarian) bipartite matching.
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
/// `(group_idx, peak_idx)` pairs while globally optimizing all remaining free groups via Hungarian.
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
    let freq_mhz = if spectrometer_mhz.is_finite() && spectrometer_mhz > 0.0 {
        spectrometer_mhz
    } else if weights.is_c13 {
        125.76
    } else {
        500.0
    };
    let solv = if solvent.trim().is_empty() {
        "CDCl3".to_string()
    } else {
        solvent.trim().to_string()
    };

    let n = predicted.len();
    let m = experimental.len();
    if n == 0 {
        return AutoAssign1DResult {
            nucleus: canon_nucleus,
            spectrometer_mhz: freq_mhz,
            solvent: solv,
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

    // Resolve valid 1-to-1 user locks
    let mut locked_row_to_col: BTreeMap<usize, usize> = BTreeMap::new();
    let mut locked_cols: BTreeSet<usize> = BTreeSet::new();
    for &(gid, pid) in locked_pairs {
        if let (Some(&row), Some(&col)) = (group_pos_by_id.get(&gid), peak_pos_by_id.get(&pid)) {
            if !locked_row_to_col.contains_key(&row) && !locked_cols.contains(&col) {
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
    let mut row_confidence: Vec<f64> = vec![0.0; n];

    for (&row, &col) in &locked_row_to_col {
        matched_col_for_row[row] = Some(col);
        row_confidence[row] = 1.0;
    }

    if nf > 0 {
        // Augmented `nf x (mf + nf)` matrix: cols `0..mf` are real peaks, cols `mf..mf+nf` are dummy unassigned slots
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
                matched_col_for_row[row] = Some(col);
                let c_ij = aug_cost[fr_i][assigned_c];

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
            } else {
                matched_col_for_row[row] = None;
                row_confidence[row] = 0.0;
            }
        }
    }

    let mut assignments = Vec::with_capacity(n);
    for row in 0..n {
        let g = &predicted[row];
        let locked = locked_row_to_col.contains_key(&row);
        if let Some(col) = matched_col_for_row[row] {
            let pk = &experimental[col];
            let cost = pairwise_cost(g, pk, weights, int_scale);
            assignments.push(AtomPeakAssignment {
                group_idx: g.group_idx,
                atom_indices: g.atom_indices.clone(),
                label: g.label.clone(),
                predicted_ppm: g.shift_ppm,
                peak_idx: Some(pk.peak_idx),
                experimental_ppm: Some(pk.ppm),
                delta_ppm: Some(pk.ppm - g.shift_ppm),
                cost,
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
        if let Some(d) = a.delta_ppm {
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
        if !result.experimental_peaks.iter().any(|p| p.peak_idx == pk) {
            return Err(format!(
                "Experimental peak {pk} not found in assignment result"
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

/// Format an ACS journal characterization report string using the assigned experimental
/// chemical shifts and structure atom/group labels.
///
/// Examples:
/// - `¹H NMR (500 MHz, CDCl3) δ 11.42 (s, 1H, COOH), 8.11 (d, J = 7.9 Hz, 1H, ArH-6), 2.34 (s, 3H, OAc CH3).`
/// - `¹³C NMR (126 MHz, CDCl3) δ 169.8 (C=O), 165.4 (COOH), 21.1 (OAc CH3).`
pub fn format_acs_with_atom_labels(result: &AutoAssign1DResult) -> String {
    let is_c13 = is_carbon_nucleus(&result.nucleus);
    let mhz_int = result.spectrometer_mhz.round().max(1.0) as usize;
    let solvent = if result.solvent.trim().is_empty() {
        "CDCl3"
    } else {
        result.solvent.trim()
    };

    let mut matched: Vec<&AtomPeakAssignment> = result
        .assignments
        .iter()
        .filter(|a| a.experimental_ppm.is_some())
        .collect();
    // Sort downfield (high ppm) to upfield (low ppm)
    matched.sort_by(|a, b| {
        let pa = a.experimental_ppm.unwrap_or(0.0);
        let pb = b.experimental_ppm.unwrap_or(0.0);
        pb.total_cmp(&pa)
    });

    if matched.is_empty() {
        return if is_c13 {
            format!("¹³C NMR ({mhz_int} MHz, {solvent}) δ (no assigned peaks).")
        } else {
            format!("¹H NMR ({mhz_int} MHz, {solvent}) δ (no assigned peaks).")
        };
    }

    if is_c13 {
        let entries: Vec<String> = matched
            .iter()
            .map(|a| {
                let ppm = a.experimental_ppm.unwrap_or(a.predicted_ppm);
                format!("{ppm:.1} ({})", a.label)
            })
            .collect();
        return format!(
            "¹³C NMR ({mhz_int} MHz, {solvent}) δ {}.",
            entries.join(", ")
        );
    }

    let entries: Vec<String> = matched
        .iter()
        .map(|a| {
            let ppm = a.experimental_ppm.unwrap_or(a.predicted_ppm);
            let peak = a
                .peak_idx
                .and_then(|pid| result.experimental_peaks.iter().find(|p| p.peak_idx == pid));
            let group = result
                .predicted_groups
                .iter()
                .find(|g| g.group_idx == a.group_idx);

            let mult = peak
                .and_then(|p| p.multiplicity.as_deref())
                .filter(|s| !s.trim().is_empty())
                .or_else(|| group.map(|g| g.multiplicity.as_str()))
                .filter(|s| !s.trim().is_empty())
                .unwrap_or("s");

            let couplings: &[f64] = peak
                .filter(|p| !p.couplings_hz.is_empty())
                .map(|p| p.couplings_hz.as_slice())
                .or_else(|| group.map(|g| g.couplings_hz.as_slice()))
                .unwrap_or(&[]);

            let j_part = if !couplings.is_empty() && mult != "m" && mult != "s" && mult != "br s" {
                let mut js: Vec<f64> = couplings
                    .iter()
                    .copied()
                    .filter(|j| j.is_finite() && *j > 0.0)
                    .collect();
                js.sort_by(|x, y| y.total_cmp(x));
                let formatted = js
                    .iter()
                    .map(|j| format!("{j:.1}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                if formatted.is_empty() {
                    String::new()
                } else {
                    format!(", J = {formatted} Hz")
                }
            } else {
                String::new()
            };

            let integral_n = group
                .map(|g| g.expected_integral.round().max(1.0) as usize)
                .or_else(|| peak.and_then(|p| p.integral.map(|v| v.round().max(1.0) as usize)))
                .unwrap_or(1);

            format!("{ppm:.2} ({mult}{j_part}, {integral_n}H, {})", a.label)
        })
        .collect();

    format!(
        "¹H NMR ({mhz_int} MHz, {solvent}) δ {}.",
        entries.join(", ")
    )
}
