//! Group peaks into backbone spin systems and resolve `i` / `i-1` offsets.
//!
//! Provenance: Clean-room original method implementing `docs/specs/03-assignment.md` sections 2 and 4,
//! including `1J_NCa` vs `2J_NCa` intensity-ratio `i` / `i-1` disambiguation (Kay, Ikura, Tschudin &
//! Bax, *J. Magn. Reson.* 1990, 89, 496–514), distinct-experiment support counting, per-experiment
//! intensity-weighted peak capping, automatic intensity-gated coincident `i == i-1` degenerate
//! candidate detection, and Asn/Gln sidechain `NH2` doublet rejection.

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::peaks::Peak;

#[derive(Clone, Debug)]
pub struct Tolerances {
    pub h: f64,
    pub n: f64,
    pub match_ca: f64,
    pub match_cb: f64,
    pub match_c: f64,
}

impl Default for Tolerances {
    fn default() -> Self {
        Self {
            h: 0.03,
            n: 0.30,
            match_ca: 0.25,
            match_cb: 0.35,
            match_c: 0.25,
        }
    }
}

impl Tolerances {
    #[inline]
    pub fn carbon_tol(&self, atom: &str) -> f64 {
        match atom {
            "CA" => self.match_ca,
            "CB" => self.match_cb,
            "C" => self.match_c,
            _ => 0.25,
        }
    }
}

/// Backbone spin system rooted on one HSQC peak.
///
/// `cands` holds competing hypotheses per `(atom, offset)` as `(ppm, support)` sorted by
/// descending distinct-experiment support (and normalized intensity when peak heights exist),
/// where `support` is the count of distinct experiments agreeing within tolerance
/// (`0` denotes a degenerate `i`/`i-1` hypothesis used only for connectivity).
#[derive(Clone, Debug)]
pub struct SpinSystem {
    pub h: f64,
    pub n: f64,
    pub cands: BTreeMap<(String, i32), Vec<(f64, usize)>>,
    /// Absolute root intensity (`HSQC volume` when positive, else `|HSQC height|`, or max attached 3D intensity).
    pub root_height: Option<f64>,
    /// Volume-preferred relative root intensity (`rel_root_intensity`), retained as an alias for API compatibility.
    pub rel_root_height: Option<f64>,
    /// Root intensity normalized by the median root intensity across carbon-bearing spin systems,
    /// computed from integrated `Peak.volume` when positive (falling back to `|Peak.height|` only when volume is absent).
    pub rel_root_intensity: Option<f64>,
}

impl SpinSystem {
    pub fn new(h: f64, n: f64) -> Self {
        Self {
            h,
            n,
            cands: BTreeMap::new(),
            root_height: None,
            rel_root_height: None,
            rel_root_intensity: None,
        }
    }

    /// Best-supported value per `(atom, offset)`.
    pub fn shifts(&self) -> BTreeMap<(String, i32), f64> {
        let mut out = BTreeMap::new();
        for (k, v) in &self.cands {
            if let Some(&(ppm, _)) = v.first() {
                out.insert(k.clone(), ppm);
            }
        }
        out
    }

    /// True if at least one observed carbon candidate (`support > 0`) is attached to this root.
    pub fn has_observed_carbons(&self) -> bool {
        self.cands
            .values()
            .any(|list| list.iter().any(|&(_, sup)| sup > 0))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum ExpSignKey {
    Hnca,
    Hncoca,
    HncacbPos,
    HncacbNeg,
    Cbcaconh,
    Hnco,
    Hncaco,
}

impl ExpSignKey {
    const ALL: [Self; 7] = [
        Self::Hnca,
        Self::Hncoca,
        Self::HncacbPos,
        Self::HncacbNeg,
        Self::Cbcaconh,
        Self::Hnco,
        Self::Hncaco,
    ];

    #[inline]
    fn from_peak(p: &Peak) -> Option<Self> {
        match p.exp.as_str() {
            "HNCA" => Some(Self::Hnca),
            "HNCOCA" => Some(Self::Hncoca),
            "HNCACB" => Some(if p.sign >= 0 {
                Self::HncacbPos
            } else {
                Self::HncacbNeg
            }),
            "CBCACONH" | "HNCOCACB" => Some(Self::Cbcaconh),
            "HNCO" => Some(Self::Hnco),
            "HNCACO" => Some(Self::Hncaco),
            _ => None,
        }
    }

    /// Physically expected maximum peaks per `(root, experiment, sign_class)`, allowing up to 2x
    /// the nominal single-root multiplicity (`1` for `i-1`-only `HNCO`/`HNCOCA`, `2` otherwise)
    /// so overlapped HSQC roots retain both spin systems' peaks while suppressing ripple tails.
    #[inline]
    fn max_peaks(self) -> usize {
        match self {
            Self::Hnco | Self::Hncoca => 2,
            Self::Hnca | Self::Hncaco | Self::Cbcaconh | Self::HncacbPos | Self::HncacbNeg => 4,
        }
    }

    /// Distinct-experiment bitmask (`HncacbPos` and `HncacbNeg` share the `HNCACB` bit).
    #[inline]
    fn exp_bit(self) -> u8 {
        match self {
            Self::Hnca => 1 << 0,
            Self::Hncoca => 1 << 1,
            Self::HncacbPos | Self::HncacbNeg => 1 << 2,
            Self::Cbcaconh => 1 << 3,
            Self::Hnco => 1 << 4,
            Self::Hncaco => 1 << 5,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct ClusterPoint {
    val: f64,
    hn_dist: f64,
    eff_norm_height: f64,
    rank_score: f64,
    exp_bit: u8,
}

fn mean_val(pts: &[ClusterPoint]) -> f64 {
    pts.iter().map(|p| p.val).sum::<f64>() / (pts.len() as f64)
}

fn cluster_val(pts: &[ClusterPoint], has_heights: bool) -> f64 {
    if !has_heights || pts.len() <= 1 {
        return mean_val(pts);
    }
    let mut w_sum = 0.0;
    let mut v_sum = 0.0;
    for p in pts {
        let w = p.eff_norm_height.clamp(0.2, 5.0);
        w_sum += w;
        v_sum += w * p.val;
    }
    if w_sum > 0.0 {
        v_sum / w_sum
    } else {
        mean_val(pts)
    }
}

fn mean_dist(pts: &[ClusterPoint]) -> f64 {
    pts.iter().map(|p| p.hn_dist).sum::<f64>() / (pts.len() as f64)
}

#[inline]
fn distinct_support(pts: &[ClusterPoint]) -> usize {
    pts.iter().fold(0u8, |acc, p| acc | p.exp_bit).count_ones() as usize
}

#[inline]
fn max_norm_height(pts: &[ClusterPoint]) -> f64 {
    pts.iter().fold(0.0f64, |m, p| m.max(p.eff_norm_height))
}

#[inline]
fn max_rank_score(pts: &[ClusterPoint]) -> f64 {
    pts.iter().fold(0.0f64, |m, p| m.max(p.rank_score))
}

/// Strictly transitive total order over same-support candidate clusters:
/// - When an `i-1`-only experiment exists (`has_im1_spectrum == true`), rank by `max_rank_score`
///   (`norm_height * exp(-1.8 * hn_dist^2)`) so genuine peaks (`~1.0x`) crush sinc/ridge artifacts
///   (`<0.15x`) and `1J` (`~1.4x`) beats `2J` (`~0.6x`), while on-center own peaks (`hd ~ 0.2`)
///   beat overlapped outer-ellipse neighbor `1J` peaks (`hd ~ 0.85`).
/// - When no `i-1`-only experiment exists (`HNCA`-only), keep both `1J` and `2J` signal clusters
///   (`eff_norm_height >= 0.30`) in the credible signal tier and order within that tier by `(H, N)`
///   centering (`mean_dist`) so the weaker `2J` `CA(i-1)` peak is never dropped by neighbor bleed.
#[inline]
fn cmp_cluster_height_then_dist(
    a: &[ClusterPoint],
    b: &[ClusterPoint],
    has_im1_spectrum: bool,
) -> std::cmp::Ordering {
    if has_im1_spectrum {
        max_rank_score(b)
            .total_cmp(&max_rank_score(a))
            .then_with(|| mean_dist(a).total_cmp(&mean_dist(b)))
    } else {
        let ta = max_norm_height(a) >= 0.30;
        let tb = max_norm_height(b) >= 0.30;
        tb.cmp(&ta)
            .then_with(|| mean_dist(a).total_cmp(&mean_dist(b)))
            .then_with(|| max_rank_score(b).total_cmp(&max_rank_score(a)))
    }
}

fn cluster_points(points: &[ClusterPoint], tol: f64) -> Vec<Vec<ClusterPoint>> {
    let mut sorted = points.to_vec();
    sorted.sort_by(|a, b| a.val.total_cmp(&b.val));
    let mut out: Vec<Vec<ClusterPoint>> = Vec::new();
    for p in sorted {
        if let Some(last) = out.last_mut() {
            if p.val - mean_val(last) <= tol {
                last.push(p);
                continue;
            }
        }
        out.push(vec![p]);
    }
    out
}

#[derive(Clone, Debug)]
struct Entry {
    exp: &'static str,
    key: ExpSignKey,
    atom: Option<&'static str>,
    offset: Option<i32>,
    val: f64,
    hn_dist: f64,
    height: Option<f64>,
    eff_norm_height: f64,
    both_offsets: bool,
}

fn median_f64(mut vals: Vec<f64>) -> Option<f64> {
    if vals.is_empty() {
        return None;
    }
    vals.sort_by(|a, b| a.total_cmp(b));
    Some(vals[vals.len() / 2])
}

/// Sign convention override for `HNCACB` (and opposite-sign `CBCACONH`/`HN(CO)CACB`) before spin-system building (`RD-1`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum HncacbPolarity {
    /// Automatically detect per-spectrum polarity from Ser/Thr/Ala CB bands and non-Gly 13C medians.
    #[default]
    Auto,
    /// Assume `CA >= 0`, `CB < 0` (no sign flip).
    CbNegative,
    /// Assume `CB >= 0`, `CA < 0` (flip signs so internal solver convention `CA >= 0, CB < 0` holds).
    CbPositive,
}

impl HncacbPolarity {
    pub fn parse(raw: &str) -> Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "cb-negative" | "cb_negative" | "cb-neg" | "negative" => Ok(Self::CbNegative),
            "cb-positive" | "cb_positive" | "cb-pos" | "positive" => Ok(Self::CbPositive),
            other => Err(format!(
                "invalid --hncacb-polarity {other:?} (expected auto|cb-negative|cb-positive)"
            )),
        }
    }

    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::CbNegative => "cb-negative",
            Self::CbPositive => "cb-positive",
        }
    }
}

impl fmt::Display for HncacbPolarity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Detected polarity convention of one `13CA`/`13CB` triple-resonance peak list (`RD-1`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DetectedPolarity {
    CbNegative,
    CbPositive,
    SingleSign,
}

impl DetectedPolarity {
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::CbNegative => "cb-negative",
            Self::CbPositive => "cb-positive",
            Self::SingleSign => "single-sign",
        }
    }
}

impl fmt::Display for DetectedPolarity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Per-spectrum polarity audit emitted by `normalize_hncacb_and_cbcaconh_polarity` (`RD-1`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SpectrumPolarityReport {
    pub experiment: String,
    pub detected: DetectedPolarity,
    pub signs_inverted: bool,
    pub n_pos: usize,
    pub n_neg: usize,
    pub ala_cb_pos: usize,
    pub ala_cb_neg: usize,
    pub thr_cb_pos: usize,
    pub thr_cb_neg: usize,
    pub non_gly_med_pos: Option<f64>,
    pub non_gly_med_neg: Option<f64>,
    pub summary: String,
    /// Optional guard/ambiguity warning emitted when auto-detection falls back conservatively.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub warning: Option<String>,
}

/// Polarity diagnostics across `HNCACB` and `CBCACONH`/`HN(CO)CACB` (`RD-1`).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PolarityReport {
    pub mode_requested: HncacbPolarity,
    pub hncacb: Option<SpectrumPolarityReport>,
    pub cbcaconh: Option<SpectrumPolarityReport>,
}

/// Pre-assembly HSQC root gating mode (`RD-5`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RootSelectionMode {
    /// Keep every HSQC root (1-to-1 indexing; sidechain NH2 doublets cleared in-place).
    #[default]
    All,
    /// Require HNCO support (falling back to any non-HSQC 3D support if HNCO is absent) and merge
    /// Asn/Gln sidechain NH2 doublet pairs (`N` in `108–115 ppm`, `|dH| ~ 0.7 ppm`) before building spin systems.
    Hnco,
}

impl RootSelectionMode {
    pub fn parse(raw: &str) -> Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "all" | "hsqc" => Ok(Self::All),
            "hnco" | "3d" => Ok(Self::Hnco),
            other => Err(format!("invalid --roots {other:?} (expected all|hnco)")),
        }
    }

    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::Hnco => "hnco",
        }
    }
}

impl fmt::Display for RootSelectionMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Optional preprocessing controls for [`build_spin_systems_with_options`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpinSystemBuildOptions {
    pub polarity: HncacbPolarity,
    pub root_mode: RootSelectionMode,
    /// Explicitly enable Asn/Gln `NH2` pair merging (`N` in `108–115 ppm`, `|dH| ~ 0.7 ppm`) even
    /// when `root_mode == All` (automatically active under `RootSelectionMode::Hnco`).
    pub merge_nh2: bool,
}

/// Result of [`build_spin_systems_with_options`] including RD-1 polarity and RD-5 root telemetry.
#[derive(Clone, Debug)]
pub struct SpinSystemBuildReport {
    pub systems: Vec<SpinSystem>,
    pub polarity: PolarityReport,
    pub n_raw_roots: usize,
    pub n_kept_roots: usize,
    pub n_roots_dropped_hnco: usize,
    pub n_nh2_pairs_merged: usize,
    pub kept_hsqc_indices: Vec<usize>,
}

fn non_gly_median(vals: &[f64]) -> Option<f64> {
    let mut non_gly: Vec<f64> = vals
        .iter()
        .copied()
        .filter(|v| !(42.5..=48.0).contains(v))
        .collect();
    if (non_gly.len() < 2 && vals.len() >= 2) || non_gly.is_empty() {
        non_gly = vals.to_vec();
    }
    median_f64(non_gly)
}

fn detect_ca_cb_spectrum_polarity(
    peaks: &[Peak],
    exp_label: &str,
    matches_exp: fn(&str) -> bool,
    override_mode: Option<HncacbPolarity>,
    only_when_opposite_signs: bool,
) -> Option<SpectrumPolarityReport> {
    let mut pos_c = Vec::new();
    let mut neg_c = Vec::new();
    for p in peaks {
        if !matches_exp(p.exp.as_str()) {
            continue;
        }
        let Some(c) = p.c else {
            continue;
        };
        if !c.is_finite() || !(10.0..=76.0).contains(&c) {
            continue;
        }
        if p.sign >= 0 {
            pos_c.push(c);
        } else {
            neg_c.push(c);
        }
    }
    if pos_c.is_empty() && neg_c.is_empty() {
        return None;
    }

    let n_pos = pos_c.len();
    let n_neg = neg_c.len();
    let total = n_pos + n_neg;
    // Strictly pure CB diagnostic windows (`173.pol:P1-6`):
    // Ala CB (14.5..=22.5 ppm) and Thr CB (65.5..=72.5 ppm).
    // Deliberately excludes 60.0..65.0 ppm to prevent Val/Ile/Thr CA bleed into CB votes.
    let ala_cb_pos = pos_c
        .iter()
        .filter(|&&c| (14.5..=22.5).contains(&c))
        .count();
    let ala_cb_neg = neg_c
        .iter()
        .filter(|&&c| (14.5..=22.5).contains(&c))
        .count();
    let thr_cb_pos = pos_c
        .iter()
        .filter(|&&c| (65.5..=72.5).contains(&c))
        .count();
    let thr_cb_neg = neg_c
        .iter()
        .filter(|&&c| (65.5..=72.5).contains(&c))
        .count();
    let cb_band_pos = ala_cb_pos + thr_cb_pos;
    let cb_band_neg = ala_cb_neg + thr_cb_neg;
    let non_gly_med_pos = non_gly_median(&pos_c);
    let non_gly_med_neg = non_gly_median(&neg_c);

    let make_report = |detected: DetectedPolarity,
                       signs_inverted: bool,
                       summary: String,
                       warning: Option<String>| {
        SpectrumPolarityReport {
            experiment: exp_label.to_string(),
            detected,
            signs_inverted,
            n_pos,
            n_neg,
            ala_cb_pos,
            ala_cb_neg,
            thr_cb_pos,
            thr_cb_neg,
            non_gly_med_pos,
            non_gly_med_neg,
            summary,
            warning,
        }
    };

    if let Some(mode) = override_mode {
        match mode {
            HncacbPolarity::CbPositive => {
                return Some(make_report(
                    DetectedPolarity::CbPositive,
                    true,
                    format!("{exp_label}: override cb-positive -> signs inverted to CA>=0/CB<0"),
                    None,
                ));
            }
            HncacbPolarity::CbNegative => {
                return Some(make_report(
                    DetectedPolarity::CbNegative,
                    false,
                    format!("{exp_label}: override cb-negative -> signs kept as-is"),
                    None,
                ));
            }
            HncacbPolarity::Auto => {}
        }
    }

    let min_cnt = n_pos.min(n_neg);
    if only_when_opposite_signs {
        // CBCA(CO)NH / HN(CO)CACB opposite-sign guard (`173.pol:P1-7`):
        // require minority sign >= 25% (`min_cnt * 4 >= total`) and >= 5 minority peaks.
        if min_cnt < 5 || min_cnt * 4 < total {
            return Some(make_report(
                DetectedPolarity::SingleSign,
                false,
                format!("{exp_label}: single-sign ({n_pos} pos, {n_neg} neg); signs unchanged"),
                None,
            ));
        }
    } else {
        // HNCACB guard (`173.pol:P1-6`): require both signs with sufficient minority support
        // (`>= 3` peaks and `>= 25%` minority fraction); on failure keep `cb-negative` and warn.
        if n_pos == 0 || n_neg == 0 {
            let warn = format!(
                "{exp_label}: single-sign ({n_pos} pos, {n_neg} neg); polarity auto-detection skipped (keeping cb-negative convention)"
            );
            return Some(make_report(
                DetectedPolarity::SingleSign,
                false,
                warn.clone(),
                Some(warn),
            ));
        }
        if min_cnt < 3 || min_cnt * 4 < total {
            let minority_pct = 100.0 * (min_cnt as f64) / (total.max(1) as f64);
            let warn = format!(
                "{exp_label}: sparse minority sign ({n_pos} pos, {n_neg} neg; minority {minority_pct:.1}% < 25% or < 3 peaks); keeping cb-negative convention"
            );
            return Some(make_report(
                DetectedPolarity::CbNegative,
                false,
                warn.clone(),
                Some(warn),
            ));
        }
    }

    let mut cb_pos_votes = 2.5 * (ala_cb_pos as f64) + 1.5 * (thr_cb_pos as f64);
    let mut cb_neg_votes = 2.5 * (ala_cb_neg as f64) + 1.5 * (thr_cb_neg as f64);
    let mut decisive_median_cb_pos = false;
    let mut decisive_median_cb_neg = false;
    if let (Some(mp), Some(mn)) = (non_gly_med_pos, non_gly_med_neg) {
        if mn - mp >= 3.0 {
            cb_pos_votes += ((mn - mp) / 2.5).clamp(1.0, 10.0);
        } else if mp - mn >= 3.0 {
            cb_neg_votes += ((mp - mn) / 2.5).clamp(1.0, 10.0);
        }
        // Decisive non-Gly median separation (CA >= 49.0 ppm vs CB <= 44.5 ppm), still subject
        // to the Ala/Thr CB-band count guards below.
        if n_pos >= 3
            && n_neg >= 3
            && (mp - mn).abs() >= 6.0
            && mp.min(mn) <= 44.5
            && mp.max(mn) >= 49.0
        {
            decisive_median_cb_pos = mn > mp;
            decisive_median_cb_neg = mp > mn;
        }
    }

    let min_cb_band = if only_when_opposite_signs { 5 } else { 3 };
    let vote_margin = 2.5;
    let strong_cb_pos = cb_band_pos >= min_cb_band
        && ((cb_pos_votes >= cb_neg_votes + vote_margin && cb_band_pos > cb_band_neg)
            || (decisive_median_cb_pos && cb_band_pos >= cb_band_neg));
    let strong_cb_neg = cb_band_neg >= min_cb_band
        && ((cb_neg_votes >= cb_pos_votes + vote_margin && cb_band_neg > cb_band_pos)
            || (decisive_median_cb_neg && cb_band_neg >= cb_band_pos));

    if only_when_opposite_signs {
        // CBCA(CO)NH must show >= 5 Ala/Thr CB-band peaks of the chosen CB sign + clear margin;
        // otherwise keep `SingleSign` so unmatched peaks are not force-typed by sign (`173.pol:P1-7`).
        let (detected, signs_inverted) = if strong_cb_pos {
            (DetectedPolarity::CbPositive, true)
        } else if strong_cb_neg {
            (DetectedPolarity::CbNegative, false)
        } else {
            return Some(make_report(
                DetectedPolarity::SingleSign,
                false,
                format!(
                    "{exp_label}: single-sign fallback ({n_pos} pos [Ala+ThrCB={cb_band_pos}], {n_neg} neg [Ala+ThrCB={cb_band_neg}] < {min_cb_band} or weak margin); signs unchanged"
                ),
                None,
            ));
        };
        let med_p_s = non_gly_med_pos
            .map(|v| format!("{v:.1}"))
            .unwrap_or_else(|| "NA".to_string());
        let med_n_s = non_gly_med_neg
            .map(|v| format!("{v:.1}"))
            .unwrap_or_else(|| "NA".to_string());
        let summary = format!(
            "{exp_label}: {detected} (pos={n_pos} [AlaCB={ala_cb_pos}, ThrCB={thr_cb_pos}, med={med_p_s}], neg={n_neg} [AlaCB={ala_cb_neg}, ThrCB={thr_cb_neg}, med={med_n_s}], score +CB={cb_pos_votes:.2} vs -CB={cb_neg_votes:.2}{})",
            if signs_inverted { "; signs normalized" } else { "" }
        );
        return Some(make_report(detected, signs_inverted, summary, None));
    }

    let signs_inverted = strong_cb_pos;
    let detected = if signs_inverted {
        DetectedPolarity::CbPositive
    } else {
        DetectedPolarity::CbNegative
    };
    let warning = if !signs_inverted && cb_pos_votes > cb_neg_votes {
        Some(format!(
            "{exp_label}: ambiguous CB-positive polarity vote (+CB={cb_pos_votes:.2} vs -CB={cb_neg_votes:.2}, Ala+ThrCB pos={cb_band_pos} < {min_cb_band} or margin < {vote_margin:.1}); keeping cb-negative convention"
        ))
    } else {
        None
    };
    let med_p_s = non_gly_med_pos
        .map(|v| format!("{v:.1}"))
        .unwrap_or_else(|| "NA".to_string());
    let med_n_s = non_gly_med_neg
        .map(|v| format!("{v:.1}"))
        .unwrap_or_else(|| "NA".to_string());
    let summary = format!(
        "{exp_label}: {detected} (pos={n_pos} [AlaCB={ala_cb_pos}, ThrCB={thr_cb_pos}, med={med_p_s}], neg={n_neg} [AlaCB={ala_cb_neg}, ThrCB={thr_cb_neg}, med={med_n_s}], score +CB={cb_pos_votes:.2} vs -CB={cb_neg_votes:.2}{})",
        if signs_inverted { "; signs normalized" } else { "" }
    );

    Some(make_report(detected, signs_inverted, summary, warning))
}

/// Detect and normalize inverted `CB`-positive `HNCACB` (and opposite-sign `CBCACONH`/`HNCOCACB`)
/// signs into FERRIC's canonical `CA >= 0, CB < 0` convention prior to spin-system building (`RD-1`).
pub fn normalize_hncacb_and_cbcaconh_polarity(
    peaks: &[Peak],
    mode: HncacbPolarity,
) -> (Vec<Peak>, PolarityReport) {
    let hncacb_ovr = (mode != HncacbPolarity::Auto).then_some(mode);
    let hncacb =
        detect_ca_cb_spectrum_polarity(peaks, "HNCACB", |e| e == "HNCACB", hncacb_ovr, false);
    let cbcaconh = detect_ca_cb_spectrum_polarity(
        peaks,
        "CBCACONH",
        |e| e == "CBCACONH" || e == "HNCOCACB",
        None,
        true,
    );

    let flip_hncacb = hncacb.as_ref().is_some_and(|r| r.signs_inverted);
    let flip_cbca = cbcaconh.as_ref().is_some_and(|r| r.signs_inverted);

    let mut out = peaks.to_vec();
    if flip_hncacb || flip_cbca {
        for p in &mut out {
            if (flip_hncacb && p.exp == "HNCACB")
                || (flip_cbca && (p.exp == "CBCACONH" || p.exp == "HNCOCACB"))
            {
                p.sign = -p.sign;
            }
        }
    }

    (
        out,
        PolarityReport {
            mode_requested: mode,
            hncacb,
            cbcaconh,
        },
    )
}

fn root_has_intra_ca_support(peaks: &[Peak], root: &Peak, tol: &Tolerances) -> bool {
    peaks.iter().any(|p| {
        if !p.c.is_some_and(|c| (35.0..=76.0).contains(&c)) {
            return false;
        }
        let is_intra_ca = p.exp == "HNCA" || (p.exp == "HNCACB" && p.sign >= 0);
        if !is_intra_ca {
            return false;
        }
        ((p.h - root.h) / tol.h).hypot((p.n - root.n) / tol.n) <= 1.25
    })
}

/// Merge paired sidechain `Asn`/`Gln` `NH2` doublets (`15N` in `107.5–115.5 ppm`, both `1H` in
/// `6.4–7.9 ppm`, `|dN| <= 0.15 ppm`, `|dH|` around `~0.7 ppm` `[0.35, 1.10] ppm`) when both
/// prongs lack intra-residue `CA(i)` support (`RD-5`). When `HNCO` peaks are present on either
/// prong, also require a shared `HNCO` `C'` shift within `tol.carbon_tol("C").max(0.35)` ppm so
/// backbone `Gly` amides cannot merge. Returns `(merged_peaks, n_pairs_merged, merged_nh2_root_coords)`.
pub fn merge_nh2_root_pairs(
    peaks: &[Peak],
    tol: &Tolerances,
) -> (Vec<Peak>, usize, Vec<(f64, f64)>) {
    let hsqc_indices: Vec<usize> = peaks
        .iter()
        .enumerate()
        .filter_map(|(idx, p)| (p.exp == "HSQC").then_some(idx))
        .collect();
    if hsqc_indices.len() < 2 {
        return (peaks.to_vec(), 0, Vec::new());
    }

    let has_ca_3d = peaks
        .iter()
        .any(|p| (p.exp == "HNCA" || p.exp == "HNCACB") && p.c.is_some());

    let lacks_ca: Vec<bool> = hsqc_indices
        .iter()
        .map(|&idx| !has_ca_3d || !root_has_intra_ca_support(peaks, &peaks[idx], tol))
        .collect();

    let hnco_peaks: Vec<&Peak> = peaks
        .iter()
        .filter(|p| p.exp == "HNCO" && p.c.is_some())
        .collect();
    let root_hnco_c: Vec<Vec<f64>> = hsqc_indices
        .iter()
        .map(|&idx| {
            let r = &peaks[idx];
            hnco_peaks
                .iter()
                .filter_map(|p| {
                    let d = ((p.h - r.h) / tol.h).hypot((p.n - r.n) / tol.n);
                    (d <= 1.25).then_some(p.c?).filter(|c| c.is_finite())
                })
                .collect()
        })
        .collect();
    let hnco_c_tol = tol.carbon_tol("C").max(0.35);

    let mut cand_pairs: Vec<(f64, usize, usize)> = Vec::new();
    for a in 0..hsqc_indices.len() {
        if !lacks_ca[a] {
            continue;
        }
        let pa = &peaks[hsqc_indices[a]];
        if !(107.5..=115.5).contains(&pa.n) || !(6.4..=7.9).contains(&pa.h) {
            continue;
        }
        for b in (a + 1)..hsqc_indices.len() {
            if !lacks_ca[b] {
                continue;
            }
            let pb = &peaks[hsqc_indices[b]];
            if !(107.5..=115.5).contains(&pb.n) || !(6.4..=7.9).contains(&pb.h) {
                continue;
            }
            let dn = (pa.n - pb.n).abs();
            let dh = (pa.h - pb.h).abs();
            if dn <= 0.15 && (0.35..=1.10).contains(&dh) {
                // If HNCO is present on either prong, require a shared sidechain C' shift
                let ha_c = &root_hnco_c[a];
                let hb_c = &root_hnco_c[b];
                if !ha_c.is_empty() || !hb_c.is_empty() {
                    let shared_c = ha_c
                        .iter()
                        .any(|&ca| hb_c.iter().any(|&cb| (ca - cb).abs() <= hnco_c_tol));
                    if !shared_c {
                        continue;
                    }
                }
                let score = ((dh - 0.70).abs() / 0.25).hypot(dn / 0.08);
                cand_pairs.push((score, a, b));
            }
        }
    }
    cand_pairs.sort_by(|x, y| x.0.total_cmp(&y.0));

    let mut used = vec![false; hsqc_indices.len()];
    let mut drop_peak_idx = vec![false; peaks.len()];
    let mut merged_coords: BTreeMap<usize, (f64, f64, Option<f64>, Option<f64>)> = BTreeMap::new();
    let mut pairs_merged = 0usize;

    for (_, a, b) in cand_pairs {
        if used[a] || used[b] {
            continue;
        }
        used[a] = true;
        used[b] = true;
        pairs_merged += 1;
        let ia = hsqc_indices[a];
        let ib = hsqc_indices[b];
        let pa = &peaks[ia];
        let pb = &peaks[ib];
        let ha = pa.height.unwrap_or(1.0);
        let hb = pb.height.unwrap_or(1.0);
        let (keep_idx, drop_idx) = if ha >= hb { (ia, ib) } else { (ib, ia) };
        let h_mid = 0.5 * (pa.h + pb.h);
        let n_mid = 0.5 * (pa.n + pb.n);
        let sum_h = match (pa.height, pb.height) {
            (Some(x), Some(y)) => Some(x + y),
            (Some(x), None) | (None, Some(x)) => Some(x),
            (None, None) => None,
        };
        let sum_v = match (pa.volume, pb.volume) {
            (Some(x), Some(y)) => Some(x + y),
            (Some(x), None) | (None, Some(x)) => Some(x),
            (None, None) => None,
        };
        drop_peak_idx[drop_idx] = true;
        merged_coords.insert(keep_idx, (h_mid, n_mid, sum_h, sum_v));
    }

    let mut out = Vec::with_capacity(peaks.len().saturating_sub(pairs_merged));
    let mut out_nh2_roots = Vec::with_capacity(pairs_merged);
    for (idx, p) in peaks.iter().enumerate() {
        if drop_peak_idx[idx] {
            continue;
        }
        if let Some(&(h_mid, n_mid, sum_h, sum_v)) = merged_coords.get(&idx) {
            let mut mp = p.clone();
            mp.h = h_mid;
            mp.n = n_mid;
            if sum_h.is_some() {
                mp.height = sum_h;
            }
            if sum_v.is_some() {
                mp.volume = sum_v;
            }
            out_nh2_roots.push((mp.h, mp.n));
            out.push(mp);
        } else {
            out.push(p.clone());
        }
    }

    (out, pairs_merged, out_nh2_roots)
}

pub fn build_spin_systems(
    peaks: &[Peak],
    tol: &Tolerances,
    max_cands: usize,
    degenerate: bool,
) -> Result<Vec<SpinSystem>, String> {
    build_spin_systems_with_options(
        peaks,
        tol,
        max_cands,
        degenerate,
        SpinSystemBuildOptions::default(),
    )
    .map(|r| r.systems)
}

/// Output of [`preprocess_peaks_with_options`] ready for spin-system assembly or solver entry points.
#[derive(Clone, Debug)]
pub struct PreprocessedPeaks {
    pub peaks: Vec<Peak>,
    pub polarity: PolarityReport,
    pub n_raw_roots: usize,
    pub n_roots_dropped_hnco: usize,
    pub n_nh2_pairs_merged: usize,
    pub kept_hsqc_indices: Vec<usize>,
    pub merged_nh2_coords: Vec<(f64, f64)>,
}

/// Apply `RD-1` per-spectrum polarity normalization and `RD-5` root gating / `NH2` pair merging
/// cleanly in one shared helper before spin-system assembly or peak-based solver entry points.
pub fn preprocess_peaks_with_options(
    raw_peaks: &[Peak],
    tol: &Tolerances,
    options: SpinSystemBuildOptions,
) -> Result<PreprocessedPeaks, String> {
    let raw_hsqc_count = raw_peaks.iter().filter(|p| p.exp == "HSQC").count();
    if raw_hsqc_count == 0 {
        return Err("an HSQC peak list is required to define spin system roots".to_string());
    }

    let (pol_peaks, polarity) = normalize_hncacb_and_cbcaconh_polarity(raw_peaks, options.polarity);
    let do_merge_nh2 = options.merge_nh2 || options.root_mode == RootSelectionMode::Hnco;

    let mut working_peaks = pol_peaks;
    let mut n_roots_dropped_hnco = 0usize;
    let mut kept_hsqc_indices: Vec<usize> = (0..raw_hsqc_count).collect();

    if options.root_mode == RootSelectionMode::Hnco {
        let hnco_pks: Vec<&Peak> = working_peaks
            .iter()
            .filter(|p| p.exp == "HNCO" && p.c.is_some())
            .collect();
        let support_pool: Vec<&Peak> = if !hnco_pks.is_empty() {
            hnco_pks
        } else {
            working_peaks
                .iter()
                .filter(|p| p.exp != "HSQC" && p.c.is_some() && ExpSignKey::from_peak(p).is_some())
                .collect()
        };

        if !support_pool.is_empty() {
            let hsqc_roots: Vec<&Peak> = working_peaks.iter().filter(|p| p.exp == "HSQC").collect();
            let mut dhs = Vec::new();
            let mut dns = Vec::new();
            for r in &hsqc_roots {
                let mut best: Option<(f64, f64, f64)> = None;
                for p in &support_pool {
                    let dh = p.h - r.h;
                    let dn = p.n - r.n;
                    let d = (dh / 0.055).hypot(dn / 0.55);
                    if d <= 1.0 && best.is_none_or(|b| d < b.0) {
                        best = Some((d, dh, dn));
                    }
                }
                if let Some((_, dh, dn)) = best {
                    dhs.push(dh);
                    dns.push(dn);
                }
            }
            let (sh_h, sh_n) = if dhs.len() >= 8 {
                let mh = median_f64(dhs).unwrap_or(0.0);
                let mn = median_f64(dns).unwrap_or(0.0);
                if mh.abs() <= 0.045 && mn.abs() <= 0.45 {
                    (mh, mn)
                } else {
                    (0.0, 0.0)
                }
            } else {
                (0.0, 0.0)
            };

            let mut filtered = Vec::with_capacity(working_peaks.len());
            let mut new_kept = Vec::new();
            let mut hsqc_ordinal = 0usize;
            for p in &working_peaks {
                if p.exp == "HSQC" {
                    let supported = support_pool.iter().any(|q| {
                        (((q.h - sh_h) - p.h) / tol.h).hypot(((q.n - sh_n) - p.n) / tol.n) <= 1.0
                    });
                    if supported {
                        new_kept.push(hsqc_ordinal);
                        filtered.push(p.clone());
                    } else {
                        n_roots_dropped_hnco += 1;
                    }
                    hsqc_ordinal += 1;
                } else {
                    filtered.push(p.clone());
                }
            }
            if !new_kept.is_empty() {
                working_peaks = filtered;
                kept_hsqc_indices = new_kept;
            } else {
                n_roots_dropped_hnco = 0;
            }
        }
    }

    let (peaks, n_nh2_pairs_merged, merged_nh2_coords) = if do_merge_nh2 {
        merge_nh2_root_pairs(&working_peaks, tol)
    } else {
        (working_peaks, 0, Vec::new())
    };

    Ok(PreprocessedPeaks {
        peaks,
        polarity,
        n_raw_roots: raw_hsqc_count,
        n_roots_dropped_hnco,
        n_nh2_pairs_merged,
        kept_hsqc_indices,
        merged_nh2_coords,
    })
}

pub fn build_spin_systems_with_options(
    raw_peaks: &[Peak],
    tol: &Tolerances,
    max_cands: usize,
    degenerate: bool,
    options: SpinSystemBuildOptions,
) -> Result<SpinSystemBuildReport, String> {
    let prep = preprocess_peaks_with_options(raw_peaks, tol, options)?;
    let raw_hsqc_count = prep.n_raw_roots;
    let polarity = prep.polarity;
    let n_roots_dropped_hnco = prep.n_roots_dropped_hnco;
    let n_nh2_pairs_merged = prep.n_nh2_pairs_merged;
    let kept_hsqc_indices = prep.kept_hsqc_indices;
    let merged_nh2_coords = prep.merged_nh2_coords;
    let cbcaconh_opposite_signs = polarity
        .cbcaconh
        .as_ref()
        .is_some_and(|r| r.detected != DetectedPolarity::SingleSign);
    let peaks = &prep.peaks;
    let roots: Vec<&Peak> = peaks.iter().filter(|p| p.exp == "HSQC").collect();
    if roots.is_empty() {
        return Err("an HSQC peak list is required to define spin system roots".to_string());
    }

    let has_heights = peaks
        .iter()
        .any(|p| p.exp != "HSQC" && p.c.is_some() && p.height.is_some());

    let mut attached: Vec<Vec<&Peak>> = vec![Vec::new(); roots.len()];
    for p in peaks {
        if p.exp == "HSQC" || p.c.is_none() || ExpSignKey::from_peak(p).is_none() {
            continue;
        }
        for (j, r) in roots.iter().enumerate() {
            let d = ((r.h - p.h) / tol.h).hypot((r.n - p.n) / tol.n);
            if d <= 1.0 {
                attached[j].push(p);
            }
        }
    }

    // Only cap when an experiment group actually exceeds M_exp due to ripple/noise proliferation;
    // rank by height * exp(-2.0 * hd^2) so on-center 2J peaks beat off-center neighbor 1J peaks.
    if has_heights {
        for (j, r) in roots.iter().enumerate() {
            let plist = &mut attached[j];
            if plist.len() <= 2 {
                continue;
            }
            let mut capped: Vec<&Peak> = Vec::with_capacity(plist.len());
            for key in ExpSignKey::ALL {
                let mut grp: Vec<(&Peak, f64)> = plist
                    .iter()
                    .copied()
                    .filter(|p| ExpSignKey::from_peak(p) == Some(key))
                    .map(|p| {
                        let hd = ((r.h - p.h) / tol.h).hypot((r.n - p.n) / tol.n);
                        let score = p.height.unwrap_or(0.0) * (-0.5 * hd * hd).exp();
                        (p, score)
                    })
                    .collect();
                if grp.is_empty() {
                    continue;
                }
                grp.sort_by(|a, b| b.1.total_cmp(&a.1));
                grp.truncate(key.max_peaks());
                capped.extend(grp.into_iter().map(|(p, _)| p));
            }
            *plist = capped;
        }
    }

    // Compute single-peak median height per (experiment, sign_class) across the capped attachments
    // (falling back to all peaks of that class if needed) so normalized heights are O(1) across spectra.
    let mut med_by_key: BTreeMap<ExpSignKey, f64> = BTreeMap::new();
    if has_heights {
        for key in ExpSignKey::ALL {
            let mut hs: Vec<f64> = attached
                .iter()
                .flat_map(|plist| plist.iter().copied())
                .filter(|p| ExpSignKey::from_peak(p) == Some(key))
                .filter_map(|p| p.height.filter(|&h| h > 0.0))
                .collect();
            if hs.is_empty() {
                hs = peaks
                    .iter()
                    .filter(|p| ExpSignKey::from_peak(p) == Some(key))
                    .filter_map(|p| p.height.filter(|&h| h > 0.0))
                    .collect();
            }
            if let Some(m) = median_f64(hs) {
                med_by_key.insert(key, m);
            }
        }

        // Prune residual <5% strip-bleed remnants inside intense strips so weak artifacts in
        // i-1-only spectra cannot mis-tag strong intra-residue peaks.
        for plist in &mut attached {
            let mut filtered = Vec::with_capacity(plist.len());
            for key in ExpSignKey::ALL {
                let med = med_by_key.get(&key).copied().unwrap_or(0.0);
                let grp: Vec<&Peak> = plist
                    .iter()
                    .copied()
                    .filter(|p| ExpSignKey::from_peak(p) == Some(key))
                    .collect();
                let h_max = grp.iter().filter_map(|p| p.height).fold(0.0f64, f64::max);
                for p in grp {
                    if med > 0.0 && h_max > 2.5 * med {
                        if let Some(h) = p.height {
                            if h < 0.05 * h_max {
                                continue;
                            }
                        }
                    }
                    filtered.push(p);
                }
            }
            *plist = filtered;
        }
    }

    let global_im1_exp = (
        peaks.iter().any(|p| {
            p.c.is_some() && (p.exp == "HNCOCA" || p.exp == "CBCACONH" || p.exp == "HNCOCACB")
        }),
        peaks
            .iter()
            .any(|p| p.c.is_some() && (p.exp == "CBCACONH" || p.exp == "HNCOCACB")),
        peaks.iter().any(|p| p.c.is_some() && p.exp == "HNCO"),
    );

    let use_degenerate = degenerate || has_heights;
    let mut systems = Vec::with_capacity(roots.len());
    let mut lacks_ca_support = vec![false; roots.len()];

    for (idx, (&r, plist)) in roots.iter().zip(attached.iter()).enumerate() {
        let (ss, has_credible_ca) = resolve_root(
            r,
            plist,
            tol,
            max_cands,
            use_degenerate,
            has_heights,
            &med_by_key,
            global_im1_exp,
            cbcaconh_opposite_signs,
        );
        lacks_ca_support[idx] = !has_credible_ca;
        systems.push(ss);
    }

    // Clear candidates on merged sidechain NH2 placeholders so they cannot compete for backbone slots.
    if !merged_nh2_coords.is_empty() {
        for ss in &mut systems {
            if merged_nh2_coords
                .iter()
                .any(|&(mh, mn)| (ss.h - mh).abs() <= 1e-6 && (ss.n - mn).abs() <= 1e-6)
            {
                ss.cands.clear();
            }
        }
    }

    // Reject Asn/Gln sidechain NH2 paired 1H doublets sharing a common 15N shift (`|dN| <= 0.12 ppm`,
    // `0.30 <= |dH| <= 1.20 ppm`) when both prongs lack credible intra-residue CA(i) support.
    // Clear candidates in-place rather than removing roots so 1-to-1 HSQC root indexing is preserved.
    let mut reject_nh2 = vec![false; systems.len()];
    for i in 0..systems.len() {
        if !lacks_ca_support[i] {
            continue;
        }
        for j in (i + 1)..systems.len() {
            if !lacks_ca_support[j] {
                continue;
            }
            let dn = (systems[i].n - systems[j].n).abs();
            let dh = (systems[i].h - systems[j].h).abs();
            if dn <= 0.12 && (0.30..=1.20).contains(&dh) {
                reject_nh2[i] = true;
                reject_nh2[j] = true;
            }
        }
    }
    for (ss, rej) in systems.iter_mut().zip(reject_nh2) {
        if rej {
            ss.cands.clear();
        }
    }

    populate_root_heights(&mut systems, &roots, &attached);

    let n_kept_roots = systems.len();
    Ok(SpinSystemBuildReport {
        systems,
        polarity,
        n_raw_roots: raw_hsqc_count,
        n_kept_roots,
        n_roots_dropped_hnco,
        n_nh2_pairs_merged,
        kept_hsqc_indices,
    })
}

/// Populate `root_height`, `rel_root_height` (`height / median_height`), and `rel_root_intensity`
/// (`volume / median_volume` when positive integrated `Peak.volume` is present on HSQC roots,
/// falling back to `rel_root_height` when volume is absent) on assembled `SpinSystem`s without
/// modifying candidate attachment, clustering, or ranking.
pub(crate) fn populate_root_heights(
    systems: &mut [SpinSystem],
    roots: &[&Peak],
    attached: &[Vec<&Peak>],
) {
    if systems.len() != roots.len() || systems.len() != attached.len() {
        return;
    }
    let n_hsqc_vol = roots
        .iter()
        .filter(|r| r.volume.is_some_and(|v| v.is_finite() && v > 0.0))
        .count();
    let n_hsqc_ht = roots
        .iter()
        .filter(|r| r.height.is_some_and(|h| h.is_finite() && h.abs() > 0.0))
        .count();
    let use_hsqc_ht = n_hsqc_ht > 0;
    let use_hsqc_vol = n_hsqc_vol > 0 && (n_hsqc_ht == 0 || n_hsqc_vol * 2 >= n_hsqc_ht);
    let use_3d_vol = !use_hsqc_vol
        && !use_hsqc_ht
        && attached
            .iter()
            .flat_map(|pl| pl.iter())
            .any(|p| p.volume.is_some_and(|v| v.is_finite() && v > 0.0));

    let mut raw_intensities: Vec<Option<f64>> = Vec::with_capacity(systems.len());
    for (j, ss) in systems.iter_mut().enumerate() {
        ss.root_height = if use_hsqc_ht {
            roots[j]
                .height
                .map(f64::abs)
                .filter(|&h| h.is_finite() && h > 0.0)
        } else {
            attached[j]
                .iter()
                .filter_map(|p| p.height.map(f64::abs).filter(|&h| h.is_finite() && h > 0.0))
                .fold(None, |acc: Option<f64>, h| {
                    Some(acc.map_or(h, |a| a.max(h)))
                })
        };
        let vol_or_ht = if use_hsqc_vol {
            roots[j]
                .volume
                .filter(|&v| v.is_finite() && v > 0.0)
                .or(ss.root_height)
        } else if use_3d_vol {
            attached[j]
                .iter()
                .filter_map(|p| p.volume.filter(|&v| v.is_finite() && v > 0.0))
                .fold(None, |acc: Option<f64>, v| {
                    Some(acc.map_or(v, |a| a.max(v)))
                })
                .or(ss.root_height)
        } else {
            ss.root_height
        };
        raw_intensities.push(vol_or_ht);
    }

    let mut ref_heights: Vec<f64> = systems
        .iter()
        .filter(|ss| ss.has_observed_carbons())
        .filter_map(|ss| ss.root_height)
        .collect();
    if ref_heights.is_empty() {
        ref_heights = systems.iter().filter_map(|ss| ss.root_height).collect();
    }
    if let Some(med_h) = median_f64(ref_heights).filter(|&m| m > 0.0) {
        for ss in systems.iter_mut() {
            ss.rel_root_height = ss.root_height.map(|h| h / med_h);
        }
    }

    let mut ref_intensities: Vec<f64> = systems
        .iter()
        .zip(raw_intensities.iter())
        .filter(|(ss, _)| ss.has_observed_carbons())
        .filter_map(|(_, &v)| v)
        .collect();
    if ref_intensities.is_empty() {
        ref_intensities = raw_intensities.iter().copied().flatten().collect();
    }
    if let Some(med_v) = median_f64(ref_intensities).filter(|&m| m > 0.0) {
        for (ss, &vi) in systems.iter_mut().zip(raw_intensities.iter()) {
            ss.rel_root_intensity = vi.map(|v| v / med_v);
        }
    } else {
        for ss in systems.iter_mut() {
            ss.rel_root_intensity = ss.rel_root_height;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn resolve_root(
    root: &Peak,
    plist: &[&Peak],
    tol: &Tolerances,
    max_cands: usize,
    use_degenerate: bool,
    has_heights: bool,
    med_by_key: &BTreeMap<ExpSignKey, f64>,
    global_im1_exp: (bool, bool, bool),
    cbcaconh_opposite_signs: bool,
) -> (SpinSystem, bool) {
    let hn_dist = |p: &Peak| ((root.h - p.h) / tol.h).hypot((root.n - p.n) / tol.n);
    let raw_norm = |height: Option<f64>, key: ExpSignKey| -> f64 {
        let Some(h) = height else {
            return 0.0;
        };
        match med_by_key.get(&key) {
            Some(&m) if m > 0.0 => h / m,
            _ => h,
        }
    };
    let eff_norm = |p: &Peak, key: ExpSignKey, hd: f64| -> f64 {
        raw_norm(p.height, key) * (-0.5 * hd * hd).exp()
    };

    let mut hncacb: Vec<Entry> = plist
        .iter()
        .filter(|p| p.exp == "HNCACB")
        .filter_map(|p| {
            let c = p.c?;
            let key = ExpSignKey::from_peak(p)?;
            let hd = hn_dist(p);
            Some(Entry {
                exp: "HNCACB",
                key,
                atom: Some(if p.sign >= 0 { "CA" } else { "CB" }),
                offset: None,
                val: c,
                hn_dist: hd,
                height: p.height,
                eff_norm_height: eff_norm(p, key, hd),
                both_offsets: true,
            })
        })
        .collect();

    let mut entries: Vec<Entry> = Vec::new();
    for p in plist {
        let Some(c) = p.c else { continue };
        let Some(key) = ExpSignKey::from_peak(p) else {
            continue;
        };
        let hd = hn_dist(p);
        let enh = eff_norm(p, key, hd);
        match p.exp.as_str() {
            "HNCA" => entries.push(Entry {
                exp: "HNCA",
                key,
                atom: Some("CA"),
                offset: None,
                val: c,
                hn_dist: hd,
                height: p.height,
                eff_norm_height: enh,
                both_offsets: true,
            }),
            "HNCOCA" => entries.push(Entry {
                exp: "HNCOCA",
                key,
                atom: Some("CA"),
                offset: Some(-1),
                val: c,
                hn_dist: hd,
                height: p.height,
                eff_norm_height: enh,
                both_offsets: false,
            }),
            "HNCO" => entries.push(Entry {
                exp: "HNCO",
                key,
                atom: Some("C"),
                offset: Some(-1),
                val: c,
                hn_dist: hd,
                height: p.height,
                eff_norm_height: enh,
                both_offsets: false,
            }),
            "HNCACO" => entries.push(Entry {
                exp: "HNCACO",
                key,
                atom: Some("C"),
                offset: None,
                val: c,
                hn_dist: hd,
                height: p.height,
                eff_norm_height: enh,
                both_offsets: true,
            }),
            _ => {}
        }
    }

    // CBCA(CO)NH peaks take atom type from the matching HNCACB peak, and that HNCACB peak becomes i-1.
    // When CBCACONH/HN(CO)CACB was acquired with opposite CA/CB signs, sign itself also breaks unmatched ties.
    for p in plist {
        if p.exp != "CBCACONH" && p.exp != "HNCOCACB" {
            continue;
        }
        let Some(c) = p.c else { continue };
        let hd = hn_dist(p);
        let enh = eff_norm(p, ExpSignKey::Cbcaconh, hd);
        let mut best_idx: Option<usize> = None;
        let mut best_d = f64::INFINITY;
        for (idx, e) in hncacb.iter().enumerate() {
            let atom = e.atom.unwrap_or("CA");
            let d = (e.val - c).abs();
            if d <= tol.carbon_tol(atom) && d < best_d {
                best_d = d;
                best_idx = Some(idx);
            }
        }
        if let Some(idx) = best_idx {
            let matched_atom = hncacb[idx].atom;
            hncacb[idx].offset = Some(-1);
            entries.push(Entry {
                exp: "CBCACONH",
                key: ExpSignKey::Cbcaconh,
                atom: matched_atom,
                offset: Some(-1),
                val: c,
                hn_dist: hd,
                height: p.height,
                eff_norm_height: enh,
                both_offsets: false,
            });
        } else {
            let fallback_atom = if cbcaconh_opposite_signs {
                Some(if p.sign >= 0 { "CA" } else { "CB" })
            } else {
                None
            };
            entries.push(Entry {
                exp: "CBCACONH",
                key: ExpSignKey::Cbcaconh,
                atom: fallback_atom,
                offset: Some(-1),
                val: c,
                hn_dist: hd,
                height: p.height,
                eff_norm_height: enh,
                both_offsets: false,
            });
        }
    }
    entries.extend(hncacb);

    // Type untyped CBCA(CO)NH peaks by matching other i-1 references.
    for i in 0..entries.len() {
        if entries[i].atom.is_none() {
            let v = entries[i].val;
            for atom in ["CA", "CB"] {
                let m = tol.carbon_tol(atom);
                let matched = entries.iter().any(|x| {
                    x.atom == Some(atom) && x.offset == Some(-1) && (x.val - v).abs() <= m
                });
                if matched {
                    entries[i].atom = Some(atom);
                    break;
                }
            }
        }
    }

    // Resolve offsets of "i and i-1" peaks against i-1 references.
    for i in 0..entries.len() {
        if let (Some(atom), None) = (entries[i].atom, entries[i].offset) {
            let v = entries[i].val;
            let m = tol.carbon_tol(atom);
            let mut has_refs = false;
            let mut matches_ref = false;
            for (j, x) in entries.iter().enumerate() {
                if j != i && x.atom == Some(atom) && x.offset == Some(-1) {
                    has_refs = true;
                    if (x.val - v).abs() <= m {
                        matches_ref = true;
                    }
                }
            }
            if has_refs {
                entries[i].offset = Some(if matches_ref { -1 } else { 0 });
            }
        }
    }

    // Intensity-ratio i / i-1 disambiguation when NO i-1-only reference exists for `atom`:
    // Within each "both_offsets" experiment (`HNCA`, `HNCACB`, `HNCACO`), if exactly 2 well-centered
    // peaks exist for `atom` (`hn_dist <= 0.72`) and one is >= 1.6x stronger (`1J` vs `2J`),
    // assign the stronger to `i` (`0`) and the weaker to `i-1` (`-1`).
    for atom in ["CA", "CB", "C"] {
        let m = tol.carbon_tol(atom);
        let any_im1 = entries
            .iter()
            .any(|e| e.atom == Some(atom) && e.offset == Some(-1));
        if any_im1 {
            continue;
        }
        let has_im1_spectrum = match atom {
            "CA" => global_im1_exp.0,
            "CB" => global_im1_exp.1,
            "C" => global_im1_exp.2,
            _ => true,
        };
        for &ename in &["HNCA", "HNCACB", "HNCACO"] {
            let mut idxs: Vec<usize> = entries
                .iter()
                .enumerate()
                .filter(|(_, e)| {
                    e.exp == ename
                        && e.atom == Some(atom)
                        && e.offset.is_none()
                        && e.hn_dist <= 0.72
                        && e.height.is_some()
                })
                .map(|(idx, _)| idx)
                .collect();
            let mut used_fallback = false;
            if !has_im1_spectrum && idxs.len() != 2 {
                // Isolated-strip fallback (only when no i-1-only spectrum was recorded for `atom`):
                // if the entire root ellipse has only 2 peaks within hn_dist <= 0.88 and the stronger
                // peak is not much farther off-center than the weaker peak, disambiguate 1J vs 2J.
                let all_for_exp: Vec<usize> = entries
                    .iter()
                    .enumerate()
                    .filter(|(_, e)| e.exp == ename && e.atom == Some(atom) && e.offset.is_none())
                    .map(|(idx, _)| idx)
                    .collect();
                if all_for_exp.len() == 2
                    && all_for_exp
                        .iter()
                        .all(|&idx| entries[idx].hn_dist <= 0.88 && entries[idx].height.is_some())
                {
                    idxs = all_for_exp;
                    used_fallback = true;
                }
            }
            if idxs.len() == 2 {
                let (i0, i1) = (idxs[0], idxs[1]);
                if (entries[i0].val - entries[i1].val).abs() > m {
                    let h0 = entries[i0].height.unwrap_or(0.0);
                    let h1 = entries[i1].height.unwrap_or(0.0);
                    let (strong, weak, hi, lo) = if h0 >= h1 {
                        (i0, i1, h0, h1)
                    } else {
                        (i1, i0, h1, h0)
                    };
                    let min_ratio = if !has_im1_spectrum {
                        if used_fallback {
                            1.65
                        } else {
                            1.55
                        }
                    } else {
                        1.60
                    };
                    let geom_ok =
                        !used_fallback || entries[strong].hn_dist <= entries[weak].hn_dist + 0.20;
                    if lo > 0.0 && hi / lo >= min_ratio && geom_ok {
                        entries[strong].offset = Some(0);
                        entries[weak].offset = Some(-1);
                        if !has_im1_spectrum {
                            for e in entries.iter_mut() {
                                if e.exp == ename
                                    && e.atom == Some(atom)
                                    && e.offset.is_none()
                                    && e.hn_dist > 0.72
                                {
                                    e.atom = None;
                                }
                            }
                        }
                    }
                }
            }
        }
        // Propagate newly resolved i-1 offset to remaining unresolved entries of the same atom
        let newly_resolved_im1 = entries
            .iter()
            .any(|e| e.atom == Some(atom) && e.offset == Some(-1));
        if newly_resolved_im1 {
            for i in 0..entries.len() {
                if entries[i].atom == Some(atom) && entries[i].offset.is_none() {
                    let v = entries[i].val;
                    let matches = entries.iter().enumerate().any(|(j, x)| {
                        j != i
                            && x.atom == Some(atom)
                            && x.offset == Some(-1)
                            && (x.val - v).abs() <= m
                    });
                    entries[i].offset = Some(if matches { -1 } else { 0 });
                }
            }
        }
    }

    let mut ss = SpinSystem::new(root.h, root.n);
    let mut max_ca_i_sup = 0usize;
    let mut max_ca_i_norm = 0.0f64;

    for atom in ["CA", "CB", "C"] {
        let m = tol.carbon_tol(atom);
        let has_im1_spectrum = match atom {
            "CA" => global_im1_exp.0,
            "CB" => global_im1_exp.1,
            "C" => global_im1_exp.2,
            _ => true,
        };
        for off in [0, -1] {
            let pts: Vec<ClusterPoint> = entries
                .iter()
                .filter(|e| e.atom == Some(atom) && (e.offset == Some(off) || e.offset.is_none()))
                .map(|e| ClusterPoint {
                    val: e.val,
                    hn_dist: e.hn_dist,
                    eff_norm_height: e.eff_norm_height,
                    rank_score: e.eff_norm_height * (-1.3 * e.hn_dist * e.hn_dist).exp(),
                    exp_bit: e.key.exp_bit(),
                })
                .collect();
            if !pts.is_empty() {
                let mut clusters = cluster_points(&pts, m);
                if has_heights {
                    if has_im1_spectrum {
                        clusters.sort_by(|a, b| {
                            distinct_support(b)
                                .cmp(&distinct_support(a))
                                .then_with(|| max_norm_height(b).total_cmp(&max_norm_height(a)))
                                .then_with(|| mean_dist(a).total_cmp(&mean_dist(b)))
                        });
                        clusters.truncate(max_cands);
                        // Among surviving multi-experiment corroborated (`support >= 2`) full-strength
                        // (`max_norm_height >= 0.65`) clusters well inside the root core (`mean_dist <= 0.50`),
                        // prefer the core cluster over a peripheral neighbor leak (`mean_dist > 0.50`) while
                        // preserving `max_norm_height` ordering everywhere else (including `CA(i)` vs `CA(i-1)`).
                        let best_sup2_h = clusters
                            .iter()
                            .filter(|c| distinct_support(c) >= 2)
                            .map(|c| max_norm_height(c))
                            .fold(0.0_f64, f64::max);
                        let min_comparable_h = (0.55 * best_sup2_h).max(0.65);
                        let on_root_corroborated = |c: &[ClusterPoint]| -> bool {
                            distinct_support(c) >= 2
                                && max_norm_height(c) >= min_comparable_h
                                && mean_dist(c) <= 0.28
                        };
                        clusters.sort_by(|a, b| {
                            distinct_support(b)
                                .cmp(&distinct_support(a))
                                .then_with(|| on_root_corroborated(b).cmp(&on_root_corroborated(a)))
                                .then_with(|| max_norm_height(b).total_cmp(&max_norm_height(a)))
                                .then_with(|| mean_dist(a).total_cmp(&mean_dist(b)))
                        });
                    } else {
                        clusters.sort_by(|a, b| {
                            distinct_support(b)
                                .cmp(&distinct_support(a))
                                .then_with(|| cmp_cluster_height_then_dist(a, b, false))
                        });
                        clusters.truncate(max_cands);
                    }
                } else {
                    clusters.sort_by(|a, b| {
                        distinct_support(b)
                            .cmp(&distinct_support(a))
                            .then_with(|| mean_dist(a).total_cmp(&mean_dist(b)))
                    });
                    clusters.truncate(max_cands);
                }
                if atom == "CA" && off == 0 {
                    for c in &clusters {
                        max_ca_i_sup = max_ca_i_sup.max(distinct_support(c));
                        max_ca_i_norm = max_ca_i_norm.max(max_norm_height(c));
                    }
                }
                let cands: Vec<(f64, usize)> = clusters
                    .into_iter()
                    .map(|c| (cluster_val(&c, has_heights), distinct_support(&c)))
                    .collect();
                ss.cands.insert((atom.to_string(), off), cands);
            } else if off == 0 && use_degenerate {
                // Intensity-gated degenerate i == i-1 hypothesis:
                // When heights are present, only offer the i-1 value as residue i if an experiment
                // observing both offsets (`HNCA`, `HNCACB`, `HNCACO`) has total superposed amplitude
                // at that carbon shift >= 1.65x the experiment's median single-peak height
                // (discriminating a coincident i + i-1 peak from a missing i peak).
                let im1_both: Vec<&Entry> = entries
                    .iter()
                    .filter(|e| e.atom == Some(atom) && e.offset == Some(-1) && e.both_offsets)
                    .collect();
                let mut vals_deg: Vec<f64> = Vec::new();
                for &e in &im1_both {
                    let keep = if has_heights {
                        match med_by_key.get(&e.key) {
                            Some(&med) if med > 0.0 && e.height.is_some() => {
                                let summed_h: f64 = im1_both
                                    .iter()
                                    .filter(|o| o.key == e.key && (o.val - e.val).abs() <= m)
                                    .filter_map(|o| o.height)
                                    .sum();
                                summed_h >= 1.65 * med
                            }
                            _ => false,
                        }
                    } else {
                        true
                    };
                    if keep {
                        vals_deg.push(e.val);
                    }
                }
                if !vals_deg.is_empty() {
                    let avg = vals_deg.iter().sum::<f64>() / (vals_deg.len() as f64);
                    ss.cands.insert((atom.to_string(), off), vec![(avg, 0)]);
                }
            }
        }
    }
    let has_credible_ca =
        max_ca_i_sup >= 2 || (max_ca_i_sup == 1 && (!has_heights || max_ca_i_norm >= 0.30));
    (ss, has_credible_ca)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nmrstar::read_star;
    use crate::priors::Priors;
    use crate::qc::evaluate_dataset_qc;
    use crate::simulate::{simulate, SimConfig};
    use crate::solver::{assign, AnnealConfig, ScoreConfig};
    use std::path::Path;

    #[test]
    fn test_rd1_inverted_hncacb_ubiquitin_matches_normal_and_reports_cb_positive() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let star = read_star(root.join("data/fixtures/synthetic_ubiquitin.str"), None).unwrap();
        let priors = Priors::bmrb();
        let (normal_peaks, _) = simulate(&star, &SimConfig::default());

        let mut inverted_peaks = normal_peaks.clone();
        for p in &mut inverted_peaks {
            if p.exp == "HNCACB" {
                p.sign = -p.sign;
            }
        }

        let norm_rep = build_spin_systems_with_options(
            &normal_peaks,
            &Tolerances::default(),
            3,
            false,
            SpinSystemBuildOptions::default(),
        )
        .unwrap();
        let inv_rep = build_spin_systems_with_options(
            &inverted_peaks,
            &Tolerances::default(),
            3,
            false,
            SpinSystemBuildOptions::default(),
        )
        .unwrap();

        assert_eq!(
            norm_rep.polarity.hncacb.as_ref().map(|h| h.detected),
            Some(DetectedPolarity::CbNegative)
        );
        assert_eq!(
            inv_rep.polarity.hncacb.as_ref().map(|h| h.detected),
            Some(DetectedPolarity::CbPositive)
        );
        assert!(inv_rep.polarity.hncacb.as_ref().unwrap().signs_inverted);

        let scfg = ScoreConfig::default();
        let acfg = AnnealConfig::default();
        let res_norm = assign(&norm_rep.systems, &star.sequence, &priors, &scfg, &acfg);
        let res_inv = assign(&inv_rep.systems, &star.sequence, &priors, &scfg, &acfg);
        assert_eq!(res_norm.position, res_inv.position);

        let qc_inv = evaluate_dataset_qc(
            Some(&inverted_peaks),
            &inv_rep.systems,
            &star.sequence,
            &priors,
            &scfg,
            &res_inv,
            None,
        );
        assert_eq!(qc_inv.hncacb_polarity.as_deref(), Some("cb-positive"));
    }

    #[test]
    fn test_rd1_opposite_sign_cbcaconh_auto_detection_and_overrides() {
        let mut peaks = vec![
            Peak::hsqc(8.10, 121.0),
            Peak::hsqc(8.20, 122.0),
            Peak::hsqc(8.30, 123.0),
            Peak::hsqc(8.40, 119.0),
            Peak::hsqc(8.50, 118.0),
            Peak::hsqc(8.60, 120.0),
            // Opposite-sign CBCACONH with inverted polarity (CB positive, CA negative)
            // >= 5 Ala/Thr CB peaks with >= 25% minority sign (`173.pol:P1-7`).
            Peak::triple("CBCACONH", 8.10, 121.0, 52.5, -1).with_intensity(100.0, None),
            Peak::triple("CBCACONH", 8.10, 121.0, 19.0, 1).with_intensity(90.0, None),
            Peak::triple("CBCACONH", 8.20, 122.0, 53.0, -1).with_intensity(100.0, None),
            Peak::triple("CBCACONH", 8.20, 122.0, 18.2, 1).with_intensity(90.0, None),
            Peak::triple("CBCACONH", 8.30, 123.0, 61.8, -1).with_intensity(100.0, None),
            Peak::triple("CBCACONH", 8.30, 123.0, 69.5, 1).with_intensity(85.0, None),
            Peak::triple("CBCACONH", 8.40, 119.0, 62.2, -1).with_intensity(100.0, None),
            Peak::triple("CBCACONH", 8.40, 119.0, 68.8, 1).with_intensity(85.0, None),
            Peak::triple("CBCACONH", 8.50, 118.0, 52.8, -1).with_intensity(100.0, None),
            Peak::triple("CBCACONH", 8.50, 118.0, 17.5, 1).with_intensity(85.0, None),
            Peak::triple("CBCACONH", 8.60, 120.0, 56.0, -1).with_intensity(100.0, None),
            Peak::triple("CBCACONH", 8.60, 120.0, 29.5, 1).with_intensity(85.0, None),
        ];

        let (norm, rep) = normalize_hncacb_and_cbcaconh_polarity(&peaks, HncacbPolarity::Auto);
        assert_eq!(
            rep.cbcaconh.as_ref().map(|c| c.detected),
            Some(DetectedPolarity::CbPositive)
        );
        assert!(norm
            .iter()
            .any(|p| p.exp == "CBCACONH" && p.c == Some(19.0) && p.sign < 0));
        assert!(norm
            .iter()
            .any(|p| p.exp == "CBCACONH" && p.c == Some(52.5) && p.sign > 0));

        // Explicit CbNegative override prevents auto-inversion on HNCACB
        for p in &mut peaks {
            if p.exp == "CBCACONH" {
                p.exp = "HNCACB".to_string();
            }
        }
        let (ovr_kept, rep_ovr) =
            normalize_hncacb_and_cbcaconh_polarity(&peaks, HncacbPolarity::CbNegative);
        assert_eq!(
            rep_ovr.hncacb.as_ref().map(|h| h.detected),
            Some(DetectedPolarity::CbNegative)
        );
        assert!(ovr_kept
            .iter()
            .any(|p| p.exp == "HNCACB" && p.c == Some(19.0) && p.sign > 0));
    }

    #[test]
    fn test_p1_6_hncacb_weak_cb_counterexample_stays_cb_negative() {
        // Exact counterexample from 173.pol:P1-6:
        // 111 positive CA peaks (including 21 Val/Ile/Thr CA in 60.0..65.0 ppm)
        // + 2 negative artifacts at 61.0 and 62.5 ppm MUST stay cb-negative and emit a warning.
        let mut peaks = vec![Peak::hsqc(8.20, 120.0)];
        for i in 0..90 {
            let ca = 50.0 + (i % 10) as f64 * 0.9;
            peaks.push(Peak::triple("HNCACB", 8.20, 120.0, ca, 1).with_intensity(100.0, None));
        }
        for i in 0..21 {
            let vit_ca = 60.2 + (i % 9) as f64 * 0.45;
            peaks.push(Peak::triple("HNCACB", 8.20, 120.0, vit_ca, 1).with_intensity(100.0, None));
        }
        peaks.push(Peak::triple("HNCACB", 8.20, 120.0, 61.0, -1).with_intensity(25.0, None));
        peaks.push(Peak::triple("HNCACB", 8.20, 120.0, 62.5, -1).with_intensity(25.0, None));

        let (norm, rep) = normalize_hncacb_and_cbcaconh_polarity(&peaks, HncacbPolarity::Auto);
        let h = rep.hncacb.as_ref().expect("hncacb polarity report");
        assert_eq!(h.detected, DetectedPolarity::CbNegative);
        assert!(!h.signs_inverted);
        assert!(
            h.warning.is_some(),
            "sparse minority HNCACB must emit a fallback warning"
        );
        assert_eq!(
            norm.iter()
                .filter(|p| p.exp == "HNCACB" && p.sign > 0)
                .count(),
            111
        );

        // Even with >= 25% negative peaks in 60..65 ppm (Val/Ile/Thr CA bleed) and 0 Ala/Thr CB peaks,
        // dropping 60..65 ppm and requiring >= 3 Ala/Thr CB-band peaks keeps cb-negative.
        let mut peaks_bleed = vec![Peak::hsqc(8.20, 120.0)];
        for i in 0..25 {
            let ca = 52.0 + (i % 6) as f64 * 0.5;
            peaks_bleed.push(Peak::triple("HNCACB", 8.20, 120.0, ca, 1));
        }
        for i in 0..15 {
            let vit = 60.2 + (i % 8) as f64 * 0.5;
            peaks_bleed.push(Peak::triple("HNCACB", 8.20, 120.0, vit, 1));
        }
        for i in 0..15 {
            let neg_art = 60.5 + (i % 7) as f64 * 0.5;
            peaks_bleed.push(Peak::triple("HNCACB", 8.20, 120.0, neg_art, -1));
        }
        let (_, rep_bleed) =
            normalize_hncacb_and_cbcaconh_polarity(&peaks_bleed, HncacbPolarity::Auto);
        let hb = rep_bleed.hncacb.as_ref().expect("hncacb bleed report");
        assert_eq!(hb.detected, DetectedPolarity::CbNegative);
        assert!(!hb.signs_inverted);
    }

    #[test]
    fn test_p1_7_cbcaconh_noise_counterexample_stays_single_sign() {
        // Exact counterexample from 173.pol:P1-7:
        // 30 positive peaks (same-sign CBCA(CO)NH) + 6 negative noise peaks -> SingleSign, not flipped.
        let mut peaks = Vec::new();
        for i in 0..15 {
            let h = 7.8 + (i as f64) * 0.08;
            let n = 116.0 + (i as f64) * 0.9;
            peaks.push(Peak::hsqc(h, n));
            // Same-sign positive CA and CB peaks
            peaks.push(Peak::triple("CBCACONH", h, n, 54.0 + (i % 5) as f64, 1));
            peaks.push(Peak::triple(
                "CBCACONH",
                h,
                n,
                if i < 6 { 18.5 } else { 31.0 + (i % 4) as f64 },
                1,
            ));
        }
        // 6 negative noise spikes (16.7% minority < 25%)
        for i in 0..6 {
            let h = 7.8 + (i as f64) * 0.08;
            let n = 116.0 + (i as f64) * 0.9;
            peaks.push(Peak::triple("CBCACONH", h, n, 58.0 + (i as f64) * 0.5, -1));
        }

        let (norm, rep) = normalize_hncacb_and_cbcaconh_polarity(&peaks, HncacbPolarity::Auto);
        let c = rep.cbcaconh.as_ref().expect("cbcaconh polarity report");
        assert_eq!(c.detected, DetectedPolarity::SingleSign);
        assert!(!c.signs_inverted);
        assert_eq!(
            norm.iter()
                .filter(|p| p.exp == "CBCACONH" && p.sign > 0)
                .count(),
            30
        );

        // Verify build_spin_systems does NOT force positive Ala CB (18.5 ppm) into CA(i-1)
        let built = build_spin_systems(&peaks, &Tolerances::default(), 3, false).unwrap();
        let leaked_cb_into_ca = built.iter().any(|ss| {
            ss.cands
                .get(&("CA".to_string(), -1))
                .is_some_and(|v| v.iter().any(|&(x, _)| (x - 18.5).abs() < 0.2))
        });
        assert!(
            !leaked_cb_into_ca,
            "same-sign CBCA(CO)NH CB(i-1)=18.5 ppm must NOT be forced to CA"
        );
    }

    #[test]
    fn test_p2_nh2_merging_blocks_gly_amides_and_mismatched_hnco() {
        let tol = Tolerances::default();
        let peaks = vec![
            // Two Gly amides in 107.5..115.5 ppm with |dH| = 0.70 ppm but 1H outside 6.4..7.9 -> MUST NOT merge
            Peak::hsqc(8.05, 110.20).with_intensity(90.0, None),
            Peak::hsqc(8.75, 110.22).with_intensity(85.0, None),
            // Two peaks inside 6.4..7.9 ppm and 107.5..115.5 ppm but DIFFERENT HNCO C' -> MUST NOT merge
            Peak::hsqc(6.95, 113.40).with_intensity(80.0, None),
            Peak::hsqc(7.65, 113.42).with_intensity(78.0, None),
            Peak::triple("HNCO", 6.95, 113.40, 173.1, 1).with_intensity(70.0, None),
            Peak::triple("HNCO", 7.65, 113.42, 176.9, 1).with_intensity(70.0, None),
            // Genuine Asn/Gln NH2 doublet in 6.4..7.9 ppm with shared HNCO C' -> MUST merge
            Peak::hsqc(6.82, 112.10).with_intensity(85.0, None),
            Peak::hsqc(7.52, 112.12).with_intensity(82.0, None),
            Peak::triple("HNCO", 6.82, 112.10, 176.4, 1).with_intensity(75.0, None),
            Peak::triple("HNCO", 7.52, 112.12, 176.45, 1).with_intensity(72.0, None),
        ];

        let (merged, n_pairs, coords) = merge_nh2_root_pairs(&peaks, &tol);
        assert_eq!(n_pairs, 1);
        assert_eq!(coords.len(), 1);
        assert!((coords[0].1 - 112.11).abs() < 0.05);
        assert_eq!(merged.iter().filter(|p| p.exp == "HSQC").count(), 5);
    }

    #[test]
    fn test_rd5_roots_hnco_and_nh2_pair_merging() {
        let tol = Tolerances::default();
        let peaks = vec![
            // Two genuine backbone roots with HNCO + HNCACB CA(i) support
            Peak::hsqc(8.20, 121.5).with_intensity(100.0, None),
            Peak::triple("HNCO", 8.20, 121.5, 174.2, 1).with_intensity(95.0, None),
            Peak::triple("HNCACB", 8.20, 121.5, 54.0, 1).with_intensity(90.0, None),
            Peak::hsqc(8.45, 119.2).with_intensity(100.0, None),
            Peak::triple("HNCO", 8.45, 119.2, 175.8, 1).with_intensity(95.0, None),
            Peak::triple("HNCACB", 8.45, 119.2, 58.1, 1).with_intensity(90.0, None),
            // Two spurious over-picked HSQC roots lacking HNCO / 3D support
            Peak::hsqc(7.92, 124.1).with_intensity(35.0, None),
            Peak::hsqc(8.71, 116.8).with_intensity(30.0, None),
            // Paired Asn/Gln NH2 doublet in 108-115 ppm separated by 0.69 ppm in 1H (with sidechain HNCO!)
            Peak::hsqc(6.86, 112.30).with_intensity(85.0, None),
            Peak::hsqc(7.55, 112.32).with_intensity(80.0, None),
            Peak::triple("HNCO", 6.86, 112.30, 176.9, 1).with_intensity(70.0, None),
            Peak::triple("HNCO", 7.55, 112.32, 176.9, 1).with_intensity(68.0, None),
        ];

        // Default RootSelectionMode::All preserves all 6 raw roots
        let rep_all = build_spin_systems_with_options(
            &peaks,
            &tol,
            3,
            false,
            SpinSystemBuildOptions::default(),
        )
        .unwrap();
        assert_eq!(rep_all.systems.len(), 6);

        // RootSelectionMode::Hnco drops the 2 unsupported noise roots and merges the NH2 pair (2 -> 1)
        let rep_hnco = build_spin_systems_with_options(
            &peaks,
            &tol,
            3,
            false,
            SpinSystemBuildOptions {
                polarity: HncacbPolarity::Auto,
                root_mode: RootSelectionMode::Hnco,
                merge_nh2: true,
            },
        )
        .unwrap();
        assert_eq!(rep_hnco.n_raw_roots, 6);
        assert_eq!(rep_hnco.n_roots_dropped_hnco, 2);
        assert_eq!(rep_hnco.n_nh2_pairs_merged, 1);
        assert_eq!(rep_hnco.systems.len(), 3);
        let merged_nh2 = rep_hnco
            .systems
            .iter()
            .find(|s| (s.n - 112.31).abs() < 0.05)
            .expect("merged NH2 placeholder");
        assert!(!merged_nh2.has_observed_carbons());
    }
}
