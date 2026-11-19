//! Interactive `i - 1 / i / i + 1` Candidate Match-Finder Strip Tiler for 3D triple-resonance
//! backbone resonance assignment.
//!
//! Provenance: Clean-room implementation of sequential triple-resonance strip walk matching from
//! published peer-reviewed methodology:
//! - Kay, Ikura, Tschudin & Bax, *J. Magn. Reson.* 1990, **89**, 496–514 (`HNCA` / `HNCO`
//!   heteronuclear 3D correlation principles).
//! - Grzesiek & Bax, *J. Am. Chem. Soc.* 1992, **114**, 6291–6293 & *J. Magn. Reson.* 1992,
//!   **96**, 432–440 (`CBCA(CO)NH` and `HNCACB` sequential `CA`/`CB`/`C'` strip matching).

use std::collections::BTreeMap;
use std::f64::consts::TAU;

use eframe::egui;
use nmr_core::priors::three_to_one;
use nmr_core::solver::{Matrix2D, ScoreConfig, FORBID};
use nmr_core::spectrum::{AxisPpm, Spectrum3D};
use nmr_core::spinsystems::SpinSystem;

use crate::contour_painter::SyncCrosshair;
use crate::workbench_mode::TYPE_CAPTION_PX;
use crate::workbook::{infer_3d_cube_nuclei, pretty_nucleus};

/// Resolve the vertical indirect strip axis, canonical physical nucleus tag (`"1H"`, `"13C"`, ...),
/// and exact `(min_ppm, max_ppm)` bounds from a [`Spectrum3D`] cube (matching `Spectrum3D::extract_strip`
/// whether `${}^{15}\text{N}$ sits on `F2/axis_y` or `F1/axis_z`).
pub fn strip_vertical_axis_metadata(s3: &Spectrum3D) -> (&AxisPpm, &'static str, (f64, f64)) {
    let (_, nuc_y, nuc_z) = infer_3d_cube_nuclei(s3);
    let y_is_n = nuc_y == "15N"
        || s3.axis_y.label.contains('N')
        || (s3.axis_y.max_ppm() > 95.0
            && s3.axis_y.min_ppm() < 145.0
            && !s3.axis_z.label.contains('N'));
    let (v_axis, v_nuc) = if y_is_n {
        (&s3.axis_z, nuc_z)
    } else {
        (&s3.axis_y, nuc_y)
    };
    let lo = v_axis.min_ppm();
    let hi = v_axis.max_ppm();
    let bounds = if hi > lo {
        (lo, hi)
    } else if v_nuc == "1H" {
        (-1.0, 11.0)
    } else {
        (10.0, 75.0)
    };
    (v_axis, v_nuc, bounds)
}

/// Generate clean human-readable chemical-shift ticks inside `[lo, hi]`.
pub fn compute_nice_strip_ticks(lo: f64, hi: f64, target_count: usize) -> Vec<f64> {
    let a = lo.min(hi);
    let b = lo.max(hi);
    let span = b - a;
    if !span.is_finite() || span <= 1e-6 || target_count == 0 {
        return Vec::new();
    }
    let raw = span / (target_count as f64);
    let pow10 = 10.0_f64.powf(raw.log10().floor());
    let frac = raw / pow10;
    let step = if frac <= 1.5 {
        1.0 * pow10
    } else if frac <= 2.5 {
        2.0 * pow10
    } else if frac <= 6.0 {
        5.0 * pow10
    } else {
        10.0 * pow10
    };
    if step <= 1e-9 {
        return Vec::new();
    }
    let mut out = Vec::new();
    let first = (a / step).ceil() as i64;
    let last = (b / step).floor() as i64;
    for k in first..=last.min(first + 64) {
        let v = (k as f64) * step;
        if v >= a - 1e-9 && v <= b + 1e-9 {
            out.push(v);
        }
    }
    out
}

/// Direction of a candidate sequential strip relative to the active query root `i`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StripMatchDirection {
    /// Predecessor `p` at `i - 1`, ranked by `P(p, i)`.
    Predecessor,
    /// Successor `q` at `i + 1`, ranked by `P(i, q)`.
    Successor,
}

/// Configuration for [`CandidateMatchFinder`].
#[derive(Clone, Debug)]
pub struct MatchFinderConfig {
    /// Maximum number of ranked predecessor (`i - 1`) and successor (`i + 1`) strips to return.
    pub top_k: usize,
    /// Scoring tolerances (`sigma_ca`, `sigma_cb`, `sigma_c`, `range_ppm`) matching `pair_matrix`.
    pub score_cfg: ScoreConfig,
    /// When `true`, filters out candidates whose user-locked sequence index conflicts with the
    /// active root's locked sequence index.
    pub exclude_conflicting_locks: bool,
}

impl Default for MatchFinderConfig {
    fn default() -> Self {
        Self {
            top_k: 5,
            score_cfg: ScoreConfig::default(),
            exclude_conflicting_locks: false,
        }
    }
}

/// Ranked predecessor (`i - 1`) or successor (`i + 1`) candidate spin system around an active root.
#[derive(Clone, Debug)]
pub struct StripMatchCandidate {
    pub spin_system_idx: usize,
    pub direction: StripMatchDirection,
    /// Exact sequential connectivity log-likelihood ratio `P(p, s)` or `P(s, q)`.
    pub pair_score: f64,
    /// Posterior relative probability in `[0.0, 1.0]` normalized over all candidates plus an
    /// unlinked/gap (`P = 0.0`) null hypothesis.
    pub rel_prob: f64,
    /// Signed `CA` residual `ppm(left, 0) - ppm(right, -1)` when both sides carry `CA`.
    pub delta_ca_ppm: Option<f64>,
    /// Signed `CB` residual `ppm(left, 0) - ppm(right, -1)` when both sides carry `CB`.
    pub delta_cb_ppm: Option<f64>,
    /// Signed `C'` residual `ppm(left, 0) - ppm(right, -1)` when both sides carry `C`.
    pub delta_c_ppm: Option<f64>,
    /// Matched `(ppm_left_i, ppm_right_im1)` pair for `CA`.
    pub ca_pair_ppm: Option<(f64, f64)>,
    /// Matched `(ppm_left_i, ppm_right_im1)` pair for `CB`.
    pub cb_pair_ppm: Option<(f64, f64)>,
    /// Matched `(ppm_left_i, ppm_right_im1)` pair for `C'`.
    pub c_pair_ppm: Option<(f64, f64)>,
    /// Number of nuclei (`0..=3`) contributing positive Gaussian log-likelihood support.
    pub matched_nuclei: usize,
    /// Number of nuclei (`1..=3`) present on both `left(0)` and `right(-1)`.
    pub compared_nuclei: usize,
    /// Optional current solver sequence placement (`0`-based residue index).
    pub assigned_residue: Option<usize>,
    /// Optional hard user lock (`0`-based residue index).
    pub locked_residue: Option<usize>,
    /// Whether this directed sequential edge has been explicitly confirmed by the user.
    pub confirmed_link: bool,
}

impl StripMatchCandidate {
    /// Root-mean-square chemical-shift residual (ppm) across all compared nuclei.
    pub fn rms_residual_ppm(&self) -> f64 {
        let mut sum_sq = 0.0;
        let mut n = 0usize;
        for d in [self.delta_ca_ppm, self.delta_cb_ppm, self.delta_c_ppm]
            .into_iter()
            .flatten()
        {
            sum_sq += d * d;
            n += 1;
        }
        if n > 0 {
            (sum_sq / (n as f64)).sqrt()
        } else {
            f64::INFINITY
        }
    }
}

/// Top-$K$ predecessor (`i - 1`) and successor (`i + 1`) candidate sets for an active spin system.
#[derive(Clone, Debug)]
pub struct StripMatchResult {
    pub root_idx: usize,
    pub predecessors: Vec<StripMatchCandidate>,
    pub successors: Vec<StripMatchCandidate>,
}

/// Stateless ranking engine computing ranked `i - 1` and `i + 1` candidate strips from `SpinSystem`
/// carbon hypotheses and `nmr_core::solver::pair_matrix`.
pub struct CandidateMatchFinder;

impl CandidateMatchFinder {
    /// Convenience entry point ranking top-$K$ predecessors and successors with default settings.
    pub fn rank_top_k(
        systems: &[SpinSystem],
        pair_mat: &Matrix2D,
        spin_system_idx: usize,
        top_k: usize,
    ) -> StripMatchResult {
        let cfg = MatchFinderConfig {
            top_k,
            ..MatchFinderConfig::default()
        };
        Self::rank_neighbors(systems, pair_mat, spin_system_idx, None, None, None, &cfg)
    }

    /// Rank top-$K$ predecessor (`i - 1`) candidates `p` by `P(p, s)` and successor (`i + 1`)
    /// candidates `q` by `P(s, q)` around `spin_system_idx`.
    pub fn rank_neighbors(
        systems: &[SpinSystem],
        pair_mat: &Matrix2D,
        spin_system_idx: usize,
        assignments: Option<&[i32]>,
        locks: Option<&BTreeMap<usize, usize>>,
        confirmed_links: Option<&[(usize, usize)]>,
        cfg: &MatchFinderConfig,
    ) -> StripMatchResult {
        let n_sys = systems.len().min(pair_mat.rows).min(pair_mat.cols);
        if spin_system_idx >= n_sys || cfg.top_k == 0 {
            return StripMatchResult {
                root_idx: spin_system_idx,
                predecessors: Vec::new(),
                successors: Vec::new(),
            };
        }

        let predecessors = Self::rank_direction(
            systems,
            pair_mat,
            n_sys,
            spin_system_idx,
            StripMatchDirection::Predecessor,
            assignments,
            locks,
            confirmed_links,
            cfg,
        );
        let successors = Self::rank_direction(
            systems,
            pair_mat,
            n_sys,
            spin_system_idx,
            StripMatchDirection::Successor,
            assignments,
            locks,
            confirmed_links,
            cfg,
        );

        StripMatchResult {
            root_idx: spin_system_idx,
            predecessors,
            successors,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn rank_direction(
        systems: &[SpinSystem],
        pair_mat: &Matrix2D,
        n_sys: usize,
        s: usize,
        direction: StripMatchDirection,
        assignments: Option<&[i32]>,
        locks: Option<&BTreeMap<usize, usize>>,
        confirmed_links: Option<&[(usize, usize)]>,
        cfg: &MatchFinderConfig,
    ) -> Vec<StripMatchCandidate> {
        let root_lock = locks.and_then(|m| m.get(&s).copied());
        let log_range = cfg.score_cfg.range_ppm.ln();

        let mut pool: Vec<StripMatchCandidate> = Vec::with_capacity(n_sys);
        for cand_idx in 0..n_sys {
            if cand_idx == s {
                continue;
            }
            let (left_idx, right_idx) = match direction {
                StripMatchDirection::Predecessor => (cand_idx, s),
                StripMatchDirection::Successor => (s, cand_idx),
            };

            let pair_score = pair_mat.get(left_idx, right_idx);
            if !pair_score.is_finite() || pair_score <= FORBID * 0.5 {
                continue;
            }

            let cand_lock = locks.and_then(|m| m.get(&cand_idx).copied());
            if cfg.exclude_conflicting_locks {
                if let Some(ks) = root_lock {
                    match direction {
                        StripMatchDirection::Predecessor => {
                            if ks == 0 {
                                continue;
                            }
                            if let Some(kc) = cand_lock {
                                if kc + 1 != ks {
                                    continue;
                                }
                            } else if locks.is_some_and(|m| {
                                m.iter().any(|(&os, &ok)| os != cand_idx && ok == ks - 1)
                            }) {
                                continue;
                            }
                        }
                        StripMatchDirection::Successor => {
                            if let Some(kc) = cand_lock {
                                if ks + 1 != kc {
                                    continue;
                                }
                            } else if locks.is_some_and(|m| {
                                m.iter().any(|(&os, &ok)| os != cand_idx && ok == ks + 1)
                            }) {
                                continue;
                            }
                        }
                    }
                } else if let Some(kc) = cand_lock {
                    // Candidate is locked to kc; exclude if the adjacent slot needed by s is locked to someone else
                    match direction {
                        StripMatchDirection::Predecessor => {
                            if locks
                                .is_some_and(|m| m.iter().any(|(&os, &ok)| os != s && ok == kc + 1))
                            {
                                continue;
                            }
                        }
                        StripMatchDirection::Successor => {
                            if kc == 0
                                || locks.is_some_and(|m| {
                                    m.iter().any(|(&os, &ok)| os != s && ok == kc - 1)
                                })
                            {
                                continue;
                            }
                        }
                    }
                }
            }

            let (ca_pair, delta_ca, ca_pos) = Self::best_atom_match(
                &systems[left_idx],
                &systems[right_idx],
                "CA",
                cfg.score_cfg.sigma_ca,
                log_range,
            );
            let (cb_pair, delta_cb, cb_pos) = Self::best_atom_match(
                &systems[left_idx],
                &systems[right_idx],
                "CB",
                cfg.score_cfg.sigma_cb,
                log_range,
            );
            let (c_pair, delta_c, c_pos) = Self::best_atom_match(
                &systems[left_idx],
                &systems[right_idx],
                "C",
                cfg.score_cfg.sigma_c,
                log_range,
            );

            let compared_nuclei = usize::from(delta_ca.is_some())
                + usize::from(delta_cb.is_some())
                + usize::from(delta_c.is_some());
            if compared_nuclei == 0 {
                continue;
            }
            let matched_nuclei = usize::from(ca_pos) + usize::from(cb_pos) + usize::from(c_pos);

            let assigned_residue = assignments
                .and_then(|pos| pos.get(cand_idx).copied())
                .and_then(|k| (k >= 0).then_some(k as usize));
            let confirmed_link =
                confirmed_links.is_some_and(|links| links.contains(&(left_idx, right_idx)));

            pool.push(StripMatchCandidate {
                spin_system_idx: cand_idx,
                direction,
                pair_score,
                rel_prob: 0.0,
                delta_ca_ppm: delta_ca,
                delta_cb_ppm: delta_cb,
                delta_c_ppm: delta_c,
                ca_pair_ppm: ca_pair,
                cb_pair_ppm: cb_pair,
                c_pair_ppm: c_pair,
                matched_nuclei,
                compared_nuclei,
                assigned_residue,
                locked_residue: cand_lock,
                confirmed_link,
            });
        }

        if pool.is_empty() {
            return pool;
        }

        // Numerically stable log-sum-exp normalization including null gap hypothesis P_null = 0.0
        let max_lp = pool.iter().fold(0.0_f64, |m, c| m.max(c.pair_score));
        let z = (0.0 - max_lp).exp()
            + pool
                .iter()
                .map(|c| (c.pair_score - max_lp).exp())
                .sum::<f64>();
        for c in &mut pool {
            c.rel_prob = ((c.pair_score - max_lp).exp() / z.max(1e-30)).clamp(0.0, 1.0);
        }

        pool.sort_by(|a, b| {
            b.confirmed_link
                .cmp(&a.confirmed_link)
                .then_with(|| b.pair_score.total_cmp(&a.pair_score))
                .then_with(|| b.matched_nuclei.cmp(&a.matched_nuclei))
                .then_with(|| a.rms_residual_ppm().total_cmp(&b.rms_residual_ppm()))
                .then_with(|| a.spin_system_idx.cmp(&b.spin_system_idx))
        });
        pool.truncate(cfg.top_k);
        pool
    }

    fn best_atom_match(
        left_ss: &SpinSystem,
        right_ss: &SpinSystem,
        atom: &str,
        sigma: f64,
        log_range: f64,
    ) -> (Option<(f64, f64)>, Option<f64>, bool) {
        let Some(a_list) = left_ss.cands.get(&(atom.to_string(), 0)) else {
            return (None, None, false);
        };
        let Some(b_list) = right_ss.cands.get(&(atom.to_string(), -1)) else {
            return (None, None, false);
        };
        if a_list.is_empty() || b_list.is_empty() {
            return (None, None, false);
        }

        let mut best_pair = (a_list[0].0, b_list[0].0);
        let mut best_abs = (best_pair.0 - best_pair.1).abs();
        for &(va, _) in a_list {
            for &(vb, _) in b_list {
                let d = (va - vb).abs();
                if d < best_abs {
                    best_abs = d;
                    best_pair = (va, vb);
                }
            }
        }
        let delta = best_pair.0 - best_pair.1;
        let const_term = -(sigma * TAU.sqrt()).ln() + log_range;
        let llr = -0.5 * (delta / sigma.max(1e-6)).powi(2) + const_term;
        (Some(best_pair), Some(delta), llr > 0.0)
    }
}

/// Evaluate a synthetic 1D Gaussian strip intensity profile at `c_ppm` from a `SpinSystem`'s
/// resolved carbon candidates when volumetric 3D cubes are absent (`peak-only` mode).
///
/// Positive lobes represent `CA` and `C'` peaks; `CB(i)` emits a negative lobe (matching standard
/// `HNCACB` sign convention) while `CB(i - 1)` emits a positive lobe (`CBCA(CO)NH` convention).
pub fn synthetic_strip_intensity_at_ppm(ss: &SpinSystem, c_ppm: f64) -> f32 {
    let mut acc = 0.0f64;
    for ((atom, off), cands) in &ss.cands {
        let lw = match atom.as_str() {
            "CA" => 0.32,
            "CB" => 0.38,
            "C" => 0.24,
            _ => 0.30,
        };
        for (rank, &(ppm, sup)) in cands.iter().enumerate() {
            if sup == 0 && rank > 0 {
                continue;
            }
            let d = (c_ppm - ppm) / lw;
            if d.abs() > 4.2 {
                continue;
            }
            let amp_scale = if *off == 0 { 1.0 } else { 0.68 }
                * (1.0 + 0.18 * (sup.saturating_sub(1) as f64))
                / ((rank + 1) as f64);
            let sign = if atom == "CB" && *off == 0 { -1.0 } else { 1.0 };
            acc += sign * amp_scale * (-0.5 * d * d).exp();
        }
    }
    acc.clamp(-1.5, 1.5) as f32
}

/// Optional volumetric 3D cubes used by [`StripTilerPanel`] when experimental 3D data are loaded.
#[derive(Clone, Copy, Default)]
pub struct StripSpectra3D<'a> {
    /// Aliphatic `13C` 3D cube (`HNCACB`, `CBCA(CO)NH`, `HNCA`, `HN(CO)CA`).
    pub aliphatic_3d: Option<&'a Spectrum3D>,
    /// Carbonyl `13C'` 3D cube (`HNCO`, `HN(CA)CO`).
    pub carbonyl_3d: Option<&'a Spectrum3D>,
}

/// Read-only data context passed to [`StripTilerPanel::show`].
pub struct StripTilerContext<'a> {
    pub systems: &'a [SpinSystem],
    pub pair_mat: &'a Matrix2D,
    pub u_mat: Option<&'a Matrix2D>,
    pub sequence: &'a [String],
    pub seq_ids: &'a [i32],
    pub active_root: usize,
    pub assignments: Option<&'a [i32]>,
    pub locks: Option<&'a BTreeMap<usize, usize>>,
    pub confirmed_links: Option<&'a [(usize, usize)]>,
    pub spectra_3d: StripSpectra3D<'a>,
}

/// User actions emitted by 1-click controls inside [`StripTilerPanel`].
#[derive(Clone, Debug, PartialEq)]
pub enum StripTilerAction {
    /// Make `spin_system_idx` the active query strip `i` (`1`-click sequential strip walk).
    SelectRoot { spin_system_idx: usize },
    /// Confirm directed sequential connectivity `pred_idx (i - 1) -> succ_idx (i)`.
    ConfirmSequentialLink { pred_idx: usize, succ_idx: usize },
    /// Lock `spin_system_idx` onto `0`-based sequence slot `residue_seq_idx`.
    LockToResidue {
        spin_system_idx: usize,
        residue_seq_idx: usize,
    },
    /// Nudge the indirect-axis shift (`d_ppm`) of a selected strip peak (`strip` = `spin_system_idx`,
    /// `peak` = 0-based index into `systems[strip].cands` in canonical `BTreeMap` order).
    NudgeStripPeak {
        strip: usize,
        peak: usize,
        d_ppm: f64,
    },
}

/// Interactive `egui` `Predecessors (i - 1) | Active Root (i) | Successors (i + 1)` strip tiler.
#[derive(Clone, Debug)]
pub struct StripTilerPanel {
    pub cfg: MatchFinderConfig,
    pub show_ca_guides: bool,
    pub show_cb_guides: bool,
    pub show_c_guides: bool,
    pub c_aliphatic_range: (f64, f64),
    pub c_carbonyl_range: (f64, f64),
    pub strip_half_width_h_ppm: f64,
    /// Shared indirect (`13C`) zoom range across all strips (`None` = full range).
    pub indirect_zoom: Option<(f64, f64)>,
    /// Current strip canvas height in pixels (auto-fills available height clamped `220..=680`, or user-resized).
    pub strip_height_px: f32,
    /// Vertical/intensity gain multiplier for strip profiles (adjusted by plain scroll wheel).
    pub strip_gain: f64,
    /// Currently selected strip peak `(spin_system_idx, peak_idx_in_cands)`.
    pub selected_peak: Option<(usize, usize)>,
    /// Bounding rectangle of the strip plot canvas from the most recent frame.
    pub last_canvas_rect: Option<egui::Rect>,
    /// Bounding rectangles of each rendered strip column from the most recent frame.
    pub last_column_rects: Vec<egui::Rect>,
    /// Compact docked mode (`true` by default): keeps inline controls inside the inspector `Options`
    /// menu and fits the `i-1 / i / i+1` triplet cleanly to the available dock width without clipping.
    pub compact_dock_mode: bool,
    /// Show inline per-column `Walk`/`Link`/`Lock` footer buttons beneath strips (`false` by default;
    /// all curation actions remain available via right-click context menu on any strip column).
    pub show_footer_buttons: bool,
    user_resized_height: bool,
    zoom_drag_start_ppm: Option<f64>,
    context_menu_ss_idx: Option<usize>,
}

impl Default for StripTilerPanel {
    fn default() -> Self {
        Self {
            cfg: MatchFinderConfig::default(),
            show_ca_guides: true,
            show_cb_guides: true,
            show_c_guides: true,
            c_aliphatic_range: (13.0, 74.0),
            c_carbonyl_range: (167.5, 182.5),
            strip_half_width_h_ppm: 0.04,
            indirect_zoom: None,
            strip_height_px: 0.0,
            strip_gain: 1.0,
            selected_peak: None,
            last_canvas_rect: None,
            last_column_rects: Vec::new(),
            compact_dock_mode: true,
            show_footer_buttons: false,
            user_resized_height: false,
            zoom_drag_start_ppm: None,
            context_menu_ss_idx: None,
        }
    }
}

#[derive(Clone)]
enum ColumnRole {
    Predecessor {
        rank_1based: usize,
        cand: StripMatchCandidate,
    },
    ActiveRoot {
        ss_idx: usize,
    },
    Successor {
        rank_1based: usize,
        cand: StripMatchCandidate,
    },
}

impl ColumnRole {
    fn ss_idx(&self) -> usize {
        match self {
            Self::Predecessor { cand, .. } | Self::Successor { cand, .. } => cand.spin_system_idx,
            Self::ActiveRoot { ss_idx } => *ss_idx,
        }
    }
}

impl StripTilerPanel {
    pub fn new() -> Self {
        Self::default()
    }

    /// Infer the best candidate sequence residue index (`0`-based) for a 1-click `LockToResidue`
    /// button on `col` given the active root's assignment/lock or unary matrix `U`.
    fn suggest_lock_residue(&self, ctx: &StripTilerContext<'_>, col: &ColumnRole) -> Option<usize> {
        let root_k = ctx
            .locks
            .and_then(|m| m.get(&ctx.active_root).copied())
            .or_else(|| {
                ctx.assignments
                    .and_then(|a| a.get(ctx.active_root).copied())
                    .and_then(|k| (k >= 0).then_some(k as usize))
            });

        let ss_idx = col.ss_idx();
        if let Some(existing_lock) = ctx.locks.and_then(|m| m.get(&ss_idx).copied()) {
            return Some(existing_lock);
        }

        match col {
            ColumnRole::Predecessor { .. } => {
                if let Some(rk) = root_k {
                    if rk > 1 && ctx.sequence.get(rk - 1).is_some_and(|r| r != "PRO") {
                        return Some(rk - 1);
                    }
                }
            }
            ColumnRole::Successor { .. } => {
                if let Some(rk) = root_k {
                    if rk + 1 < ctx.sequence.len()
                        && ctx.sequence.get(rk + 1).is_some_and(|r| r != "PRO")
                    {
                        return Some(rk + 1);
                    }
                }
            }
            ColumnRole::ActiveRoot { .. } => {
                if let Some(rk) = root_k {
                    return Some(rk);
                }
            }
        }

        if let Some(assigned_k) = ctx
            .assignments
            .and_then(|a| a.get(ss_idx).copied())
            .and_then(|k| (k >= 0).then_some(k as usize))
        {
            return Some(assigned_k);
        }

        if let Some(u_mat) = ctx.u_mat {
            if ss_idx < u_mat.rows {
                return (1..u_mat.cols.min(ctx.sequence.len()))
                    .filter(|&k| ctx.sequence[k] != "PRO" && u_mat.get(ss_idx, k) > FORBID * 0.5)
                    .max_by(|&ka, &kb| u_mat.get(ss_idx, ka).total_cmp(&u_mat.get(ss_idx, kb)));
            }
        }
        None
    }

    /// Render the interactive `Predecessors (i-1) | Active Root (i) | Successors (i+1)` strip
    /// tiler and return all 1-click [`StripTilerAction`]s triggered during this frame.
    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &StripTilerContext<'_>,
        mut crosshair: Option<&mut SyncCrosshair>,
    ) -> Vec<StripTilerAction> {
        let mut actions = Vec::new();
        if ctx.active_root >= ctx.systems.len() {
            ui.label("Select an HSQC root to inspect ranked i − 1 / i / i + 1 candidate strips.");
            return actions;
        }

        let matches = CandidateMatchFinder::rank_neighbors(
            ctx.systems,
            ctx.pair_mat,
            ctx.active_root,
            ctx.assignments,
            ctx.locks,
            ctx.confirmed_links,
            &self.cfg,
        );

        // Top controls bar: in compact dock mode, controls live in the inspector Options menu to
        // preserve the <= 12 primary-control budget and maximize 3D strip plot area (`UI-B2`, `UI-B7`).
        if !self.compact_dock_mode {
            ui.horizontal_wrapped(|ui| {
                let root_ss = &ctx.systems[ctx.active_root];
                ui.label(
                    egui::RichText::new(format!(
                        "Strip Match-Finder  •  Root #{} (¹H {:.3}, ¹⁵N {:.2})",
                        ctx.active_root, root_ss.h, root_ss.n
                    ))
                    .strong(),
                );
                ui.separator();
                ui.label(egui::RichText::new("Top-K:").size(TYPE_CAPTION_PX));
                for k_opt in [3usize, 5, 8] {
                    if ui
                        .selectable_label(self.cfg.top_k == k_opt, format!("{k_opt}"))
                        .clicked()
                    {
                        self.cfg.top_k = k_opt;
                    }
                }
                ui.separator();
                ui.checkbox(
                    &mut self.show_ca_guides,
                    egui::RichText::new("Cα")
                        .color(egui::Color32::from_rgb(75, 215, 255))
                        .size(TYPE_CAPTION_PX),
                );
                ui.checkbox(
                    &mut self.show_cb_guides,
                    egui::RichText::new("Cβ")
                        .color(egui::Color32::from_rgb(255, 125, 80))
                        .size(TYPE_CAPTION_PX),
                );
                ui.checkbox(
                    &mut self.show_c_guides,
                    egui::RichText::new("C'")
                        .color(egui::Color32::from_rgb(90, 235, 135))
                        .size(TYPE_CAPTION_PX),
                );
                ui.checkbox(
                    &mut self.cfg.exclude_conflicting_locks,
                    egui::RichText::new("Filter Locks").size(TYPE_CAPTION_PX),
                );
            });
        }

        // Build ordered column list: [p_m ... p_2, p_1, Active(i), q_1, q_2 ... q_n]
        // In compact dock mode, show the primary adjacent triplet `[p_1, Active(i), q_1]`
        // so all 3 strips fit cleanly inside the docked inspector with zero horizontal clipping.
        let max_side_cols = if self.compact_dock_mode {
            1usize
        } else {
            self.cfg.top_k
        };
        let shown_preds = &matches.predecessors[..matches.predecessors.len().min(max_side_cols)];
        let shown_succs = &matches.successors[..matches.successors.len().min(max_side_cols)];
        let mut columns: Vec<ColumnRole> =
            Vec::with_capacity(shown_preds.len() + 1 + shown_succs.len());
        for (idx, cand) in shown_preds.iter().cloned().enumerate().rev() {
            columns.push(ColumnRole::Predecessor {
                rank_1based: idx + 1,
                cand,
            });
        }
        let center_col_idx = columns.len();
        columns.push(ColumnRole::ActiveRoot {
            ss_idx: ctx.active_root,
        });
        for (idx, cand) in shown_succs.iter().cloned().enumerate() {
            columns.push(ColumnRole::Successor {
                rank_1based: idx + 1,
                cand,
            });
        }

        let avail_w = ui.available_width().max(220.0);
        let left_axis_w = if self.compact_dock_mode { 40.0f32 } else { 50.0f32 };
        let col_w = if self.compact_dock_mode && columns.len() <= 3 {
            ((avail_w - left_axis_w - 6.0) / (columns.len().max(1) as f32)).clamp(64.0, 124.0)
        } else {
            124.0f32
        };
        if !self.user_resized_height || self.strip_height_px <= 0.0 {
            let reserve_h = if self.show_footer_buttons { 104.0 } else { 14.0 };
            self.strip_height_px = (ui.available_height() - reserve_h).clamp(220.0, 680.0);
        }
        let strip_h = self.strip_height_px.clamp(220.0, 680.0);
        let desired_w = if self.compact_dock_mode && columns.len() <= 3 {
            avail_w
        } else {
            (left_axis_w + (columns.len() as f32) * col_w + 8.0).max(avail_w)
        };

        let (ali_nuc, base_ali_range) = if let Some(s3) = ctx.spectra_3d.aliphatic_3d {
            let (_, nuc, bounds) = strip_vertical_axis_metadata(s3);
            (nuc, bounds)
        } else {
            ("13C", self.c_aliphatic_range)
        };
        let base_ali_lo = base_ali_range.0.min(base_ali_range.1);
        let base_ali_hi = base_ali_range.0.max(base_ali_range.1);
        let ali_range = self
            .indirect_zoom
            .map(|(a, b)| (a.min(b), a.max(b)))
            .unwrap_or((base_ali_lo, base_ali_hi));

        let (co_nuc, co_range) = if let Some(s3_co) = ctx.spectra_3d.carbonyl_3d {
            let (_, nuc, bounds) = strip_vertical_axis_metadata(s3_co);
            (nuc, bounds)
        } else {
            ("13C", self.c_carbonyl_range)
        };
        let has_c_cands = ctx.systems.iter().any(|ss| {
            ss.cands.contains_key(&("C".to_string(), 0))
                || ss.cands.contains_key(&("C".to_string(), -1))
        });
        let show_co_band = ctx.spectra_3d.carbonyl_3d.is_some() || ali_nuc != "1H" || has_c_cands;

        let no_hscroll = self.compact_dock_mode && columns.len() <= 3;
        let mut render_strips = |ui: &mut egui::Ui| {
                // 1. Paint synchronized split-band (C' + CA/CB) strip columns & dashed guides
                let (canvas_rect, canvas_resp) = ui.allocate_exact_size(
                    egui::vec2(desired_w, strip_h),
                    egui::Sense::click_and_drag(),
                );
                self.last_canvas_rect = Some(canvas_rect);
                let painter = ui.painter_at(canvas_rect);
                painter.rect_filled(canvas_rect, 4.0, egui::Color32::from_rgb(13, 17, 24));

                let header_h = 40.0f32;
                let co_top = canvas_rect.top() + header_h;
                let (co_bot, ali_top) = if show_co_band {
                    let cb = co_top + (strip_h * 0.19).clamp(54.0, 96.0);
                    (cb, cb + 10.0)
                } else {
                    (co_top, co_top)
                };
                let ali_bot = canvas_rect.bottom() - 8.0;

                let eff_col_w = if no_hscroll {
                    ((canvas_rect.width() - left_axis_w - 6.0) / (columns.len().max(1) as f32))
                        .max(40.0)
                } else {
                    ((canvas_rect.width() - left_axis_w - 6.0) / (columns.len().max(1) as f32))
                        .max(col_w)
                };

                // Interactive indirect-axis zoom (Cmd/Ctrl+Scroll, Shift+drag box, double-click reset),
                // Shift+Scroll / Space+drag / Right/Middle-drag indirect pan, & plain scroll gain (`UI-C10`, `UI-C18`)
                if canvas_resp.double_clicked() {
                    self.indirect_zoom = None;
                    self.zoom_drag_start_ppm = None;
                }

                let (mut shift_held, mut cmd_or_ctrl, space_down) = ui.input(|i| {
                    (
                        i.modifiers.shift,
                        i.modifiers.command || i.modifiers.ctrl,
                        i.key_down(egui::Key::Space),
                    )
                });

                if let Some(hover_pos) = canvas_resp.hover_pos() {
                    if canvas_rect.contains(hover_pos) {
                        let (wheel_sy, smooth_sy, ev_shift, ev_cmd) = ui.input(|i| {
                            let mut wy = 0.0f32;
                            let mut esh = false;
                            let mut ecm = false;
                            for ev in &i.events {
                                if let egui::Event::MouseWheel {
                                    delta, modifiers, ..
                                } = ev
                                {
                                    wy += delta.y;
                                    esh |= modifiers.shift;
                                    ecm |= modifiers.command || modifiers.ctrl;
                                }
                            }
                            (wy, i.smooth_scroll_delta.y, esh, ecm)
                        });
                        shift_held |= ev_shift;
                        cmd_or_ctrl |= ev_cmd;
                        let sy = if smooth_sy.abs() >= (wheel_sy * 16.0).abs() {
                            smooth_sy
                        } else {
                            wheel_sy * 16.0
                        };
                        if sy.abs() > 0.05 {
                            ui.input_mut(|i| {
                                i.smooth_scroll_delta.y = 0.0;
                            });
                            if cmd_or_ctrl {
                                let anchor_ppm = SyncCrosshair::strip_y_to_ppm(
                                    ali_range.0,
                                    ali_range.1,
                                    ali_top,
                                    ali_bot,
                                    hover_pos.y.clamp(ali_top, ali_bot),
                                );
                                let z_factor = (-sy as f64 * 0.0035).exp().clamp(0.4, 2.5);
                                let new_lo = (anchor_ppm - (anchor_ppm - ali_range.0) * z_factor)
                                    .max(base_ali_lo - 10.0);
                                let new_hi = (anchor_ppm + (ali_range.1 - anchor_ppm) * z_factor)
                                    .min(base_ali_hi + 10.0);
                                if new_hi - new_lo >= 0.5 {
                                    self.indirect_zoom = Some((new_lo, new_hi));
                                }
                            } else if shift_held {
                                let cur = self
                                    .indirect_zoom
                                    .map(|(a, b)| (a.min(b), a.max(b)))
                                    .unwrap_or(ali_range);
                                let span = (cur.1 - cur.0).max(0.5);
                                let d_ppm =
                                    -(sy as f64 / (ali_bot - ali_top).max(1.0) as f64) * span;
                                self.indirect_zoom = Some((cur.0 + d_ppm, cur.1 + d_ppm));
                            } else {
                                let g_factor = (sy as f64 * 0.0045).exp().clamp(0.25, 4.0);
                                self.strip_gain = (self.strip_gain * g_factor).clamp(0.1, 50.0);
                            }
                        }
                    }
                }

                // Space+drag or Right/Middle-drag pans the indirect 13C range (`UI-C5`, `UI-C18`)
                if !shift_held
                    && ((space_down && canvas_resp.dragged_by(egui::PointerButton::Primary))
                        || canvas_resp.dragged_by(egui::PointerButton::Secondary)
                        || canvas_resp.dragged_by(egui::PointerButton::Middle))
                {
                    let dy = canvas_resp.drag_delta().y;
                    if dy.abs() > 0.1 {
                        let cur = self
                            .indirect_zoom
                            .map(|(a, b)| (a.min(b), a.max(b)))
                            .unwrap_or(ali_range);
                        let span = (cur.1 - cur.0).max(0.5);
                        let d_ppm = -(dy as f64 / (ali_bot - ali_top).max(1.0) as f64) * span;
                        self.indirect_zoom = Some((cur.0 + d_ppm, cur.1 + d_ppm));
                    }
                }

                let active_ali_range = self
                    .indirect_zoom
                    .map(|(a, b)| (a.min(b), a.max(b)))
                    .unwrap_or(ali_range);

                if canvas_resp.drag_started() && shift_held {
                    if let Some(pos) = canvas_resp.interact_pointer_pos() {
                        if canvas_rect.contains(pos) {
                            self.zoom_drag_start_ppm = Some(SyncCrosshair::strip_y_to_ppm(
                                active_ali_range.0,
                                active_ali_range.1,
                                ali_top,
                                ali_bot,
                                pos.y.clamp(ali_top, ali_bot),
                            ));
                        }
                    }
                }
                if canvas_resp.drag_stopped() {
                    if let Some(p0) = self.zoom_drag_start_ppm.take() {
                        if let Some(pos) = canvas_resp.interact_pointer_pos() {
                            let p1 = SyncCrosshair::strip_y_to_ppm(
                                active_ali_range.0,
                                active_ali_range.1,
                                ali_top,
                                ali_bot,
                                pos.y.clamp(ali_top, ali_bot),
                            );
                            if (p1 - p0).abs() >= 0.25 {
                                self.indirect_zoom = Some((p0.min(p1), p0.max(p1)));
                            }
                        }
                    }
                }

                let ali_range = self
                    .indirect_zoom
                    .map(|(a, b)| (a.min(b), a.max(b)))
                    .unwrap_or(active_ali_range);

                let co_to_y = |ppm: f64| -> f32 {
                    SyncCrosshair::ppm_to_strip_y(co_range.0, co_range.1, co_top, co_bot, ppm)
                };
                let ali_to_y = |ppm: f64| -> f32 {
                    SyncCrosshair::ppm_to_strip_y(ali_range.0, ali_range.1, ali_top, ali_bot, ppm)
                };

                // Axis nucleus badge in top-left gutter
                let gutter_nuc = if show_co_band && co_nuc != ali_nuc {
                    format!("{}/{}", pretty_nucleus(co_nuc), pretty_nucleus(ali_nuc))
                } else {
                    pretty_nucleus(ali_nuc).to_string()
                };
                painter.text(
                    egui::pos2(canvas_rect.left() + 4.0, canvas_rect.top() + 18.0),
                    egui::Align2::LEFT_CENTER,
                    gutter_nuc,
                    egui::FontId::monospace(TYPE_CAPTION_PX),
                    egui::Color32::from_rgb(165, 185, 210),
                );

                // Draw dynamic carbonyl/upper-band and aliphatic/indirect-band axis ticks
                if show_co_band {
                    for c_tick in compute_nice_strip_ticks(co_range.0, co_range.1, 3) {
                        let y = co_to_y(c_tick);
                        let fmt_lbl = if (co_range.1 - co_range.0).abs() < 6.0 {
                            format!("{c_tick:.1}")
                        } else {
                            format!("{c_tick:.0}")
                        };
                        painter.text(
                            egui::pos2(canvas_rect.left() + 4.0, y),
                            egui::Align2::LEFT_CENTER,
                            fmt_lbl,
                            egui::FontId::monospace(TYPE_CAPTION_PX),
                            egui::Color32::from_rgb(135, 190, 150),
                        );
                        painter.line_segment(
                            [
                                egui::pos2(canvas_rect.left() + left_axis_w, y),
                                egui::pos2(canvas_rect.right() - 4.0, y),
                            ],
                            egui::Stroke::new(0.5, egui::Color32::from_rgb(26, 36, 46)),
                        );
                    }
                }
                let ali_span = (ali_range.1 - ali_range.0).abs();
                for c_tick in compute_nice_strip_ticks(ali_range.0, ali_range.1, 6) {
                    let y = ali_to_y(c_tick);
                    let fmt_lbl = if ali_span < 16.0 && (c_tick.fract().abs() > 1e-4) {
                        format!("{c_tick:.1}")
                    } else {
                        format!("{c_tick:.0}")
                    };
                    painter.text(
                        egui::pos2(canvas_rect.left() + 4.0, y),
                        egui::Align2::LEFT_CENTER,
                        fmt_lbl,
                        egui::FontId::monospace(TYPE_CAPTION_PX),
                        egui::Color32::from_rgb(145, 165, 190),
                    );
                    painter.line_segment(
                        [
                            egui::pos2(canvas_rect.left() + left_axis_w, y),
                            egui::pos2(canvas_rect.right() - 4.0, y),
                        ],
                        egui::Stroke::new(0.5, egui::Color32::from_rgb(26, 35, 48)),
                    );
                }

                // Draw horizontal dashed CA (cyan), CB (orange-red), and C' (green) alignment guides
                // Left half (Predecessors -> Active Root): anchored at Active Root's (i - 1) shifts
                // Right half (Active Root -> Successors): anchored at Active Root's (i) shifts
                let x_pred_left = canvas_rect.left() + left_axis_w + 2.0;
                let x_center_mid =
                    canvas_rect.left() + left_axis_w + (center_col_idx as f32 + 0.5) * eff_col_w;
                let x_succ_right =
                    canvas_rect.left() + left_axis_w + (columns.len() as f32) * eff_col_w - 2.0;

                let root_ss = &ctx.systems[ctx.active_root];
                let guide_specs = [
                    (
                        "CA",
                        self.show_ca_guides,
                        egui::Color32::from_rgba_unmultiplied(75, 215, 255, 195),
                        false,
                    ),
                    (
                        "CB",
                        self.show_cb_guides,
                        egui::Color32::from_rgba_unmultiplied(255, 120, 75, 195),
                        false,
                    ),
                    (
                        "C",
                        self.show_c_guides,
                        egui::Color32::from_rgba_unmultiplied(90, 235, 135, 195),
                        true,
                    ),
                ];

                for (atom, enabled, color, is_carbonyl) in guide_specs {
                    if !enabled {
                        continue;
                    }
                    let (r_lo, r_hi) = if is_carbonyl { co_range } else { ali_range };
                    if center_col_idx > 0 {
                        if let Some(&(ppm_im1, _)) = root_ss
                            .cands
                            .get(&(atom.to_string(), -1))
                            .and_then(|c| c.first())
                        {
                            if (r_lo.min(r_hi)..=r_lo.max(r_hi)).contains(&ppm_im1) {
                                let y = if is_carbonyl {
                                    co_to_y(ppm_im1)
                                } else {
                                    ali_to_y(ppm_im1)
                                };
                                draw_dashed_hline(&painter, x_pred_left, x_center_mid, y, color);
                            }
                        }
                    }
                    if center_col_idx + 1 < columns.len() {
                        if let Some(&(ppm_i, sup)) = root_ss
                            .cands
                            .get(&(atom.to_string(), 0))
                            .and_then(|c| c.first())
                        {
                            if sup > 0 && (r_lo.min(r_hi)..=r_lo.max(r_hi)).contains(&ppm_i) {
                                let y = if is_carbonyl {
                                    co_to_y(ppm_i)
                                } else {
                                    ali_to_y(ppm_i)
                                };
                                draw_dashed_hline(&painter, x_center_mid, x_succ_right, y, color);
                            }
                        }
                    }
                }

                // Collect painted marker hit targets `(spin_system_idx, peak_idx, screen_pos)`
                let mut marker_hits: Vec<(usize, usize, egui::Pos2)> = Vec::new();
                let mut col_rects: Vec<(usize, egui::Rect)> = Vec::with_capacity(columns.len());

                // Render each strip column (header, 3D slice or synthetic Gaussian + sticks, click-to-select)
                for (c_idx, col) in columns.iter().enumerate() {
                    let x0 = canvas_rect.left() + left_axis_w + (c_idx as f32) * eff_col_w + 2.5;
                    let x1 = x0 + eff_col_w - 5.0;
                    let cx = 0.5 * (x0 + x1);
                    let ss_idx = col.ss_idx();
                    let ss = &ctx.systems[ss_idx];
                    col_rects.push((
                        ss_idx,
                        egui::Rect::from_min_max(
                            egui::pos2(x0, canvas_rect.top()),
                            egui::pos2(x1, canvas_rect.bottom()),
                        ),
                    ));

                    let is_center = matches!(col, ColumnRole::ActiveRoot { .. });
                    let border_col = match col {
                        ColumnRole::ActiveRoot { .. } => egui::Color32::from_rgb(95, 210, 255),
                        ColumnRole::Predecessor { cand, .. }
                        | ColumnRole::Successor { cand, .. }
                            if cand.confirmed_link =>
                        {
                            egui::Color32::from_rgb(95, 240, 155)
                        }
                        _ => egui::Color32::from_rgb(50, 62, 82),
                    };

                    let co_box =
                        egui::Rect::from_min_max(egui::pos2(x0, co_top), egui::pos2(x1, co_bot));
                    let ali_box =
                        egui::Rect::from_min_max(egui::pos2(x0, ali_top), egui::pos2(x1, ali_bot));
                    if show_co_band {
                        painter.rect_stroke(
                            co_box,
                            2.0,
                            egui::Stroke::new(if is_center { 1.6 } else { 1.0 }, border_col),
                            egui::StrokeKind::Inside,
                        );
                    }
                    painter.rect_stroke(
                        ali_box,
                        2.0,
                        egui::Stroke::new(if is_center { 1.6 } else { 1.0 }, border_col),
                        egui::StrokeKind::Inside,
                    );

                    // Column header & score subtitle (zero emojis; >= 11 px font)
                    let res_badge = ctx
                        .locks
                        .and_then(|m| m.get(&ss_idx).copied())
                        .map(|k| format!("[L]{}{}", three_to_one(&ctx.sequence[k]), ctx.seq_ids[k]))
                        .or_else(|| {
                            ctx.assignments
                                .and_then(|a| a.get(ss_idx).copied())
                                .filter(|&k| k >= 0 && (k as usize) < ctx.sequence.len())
                                .map(|k| {
                                    let ku = k as usize;
                                    format!(
                                        "{}{}",
                                        three_to_one(&ctx.sequence[ku]),
                                        ctx.seq_ids[ku]
                                    )
                                })
                        })
                        .unwrap_or_else(|| format!("SS#{ss_idx}"));

                    let compact_hdr = self.compact_dock_mode || eff_col_w < 130.0;
                    let (top_label, sub_label, sub_col) = match col {
                        ColumnRole::Predecessor { rank_1based, cand } => (
                            if compact_hdr {
                                format!("i-1 {res_badge}")
                            } else {
                                format!("i−1 #{rank_1based} {res_badge}")
                            },
                            if compact_hdr {
                                format!("P{:+.0} {:.0}%", cand.pair_score, cand.rel_prob * 100.0)
                            } else {
                                format!(
                                    "P{:+.1} {:>2.0}% ({}n)",
                                    cand.pair_score,
                                    cand.rel_prob * 100.0,
                                    cand.matched_nuclei
                                )
                            },
                            egui::Color32::from_rgb(185, 215, 250),
                        ),
                        ColumnRole::ActiveRoot { .. } => (
                            if compact_hdr {
                                format!("i {res_badge}")
                            } else {
                                format!("Root(i) {res_badge}")
                            },
                            if compact_hdr {
                                format!("{:.2}/{:.0}", ss.h, ss.n)
                            } else {
                                format!("¹H {:.2} N {:.1}", ss.h, ss.n)
                            },
                            egui::Color32::from_rgb(110, 230, 255),
                        ),
                        ColumnRole::Successor { rank_1based, cand } => (
                            if compact_hdr {
                                format!("i+1 {res_badge}")
                            } else {
                                format!("i+1 #{rank_1based} {res_badge}")
                            },
                            if compact_hdr {
                                format!("P{:+.0} {:.0}%", cand.pair_score, cand.rel_prob * 100.0)
                            } else {
                                format!(
                                    "P{:+.1} {:>2.0}% ({}n)",
                                    cand.pair_score,
                                    cand.rel_prob * 100.0,
                                    cand.matched_nuclei
                                )
                            },
                            egui::Color32::from_rgb(185, 215, 250),
                        ),
                    };

                    painter.text(
                        egui::pos2(cx, canvas_rect.top() + 11.0),
                        egui::Align2::CENTER_CENTER,
                        top_label,
                        egui::FontId::monospace(TYPE_CAPTION_PX),
                        egui::Color32::from_rgb(232, 242, 255),
                    );
                    painter.text(
                        egui::pos2(cx, canvas_rect.top() + 27.0),
                        egui::Align2::CENTER_CENTER,
                        sub_label,
                        egui::FontId::monospace(TYPE_CAPTION_PX),
                        sub_col,
                    );

                    // Paint volumetric 3D slice OR synthetic Gaussian strip profile
                    if show_co_band {
                        let co_painter = painter.with_clip_rect(co_box);
                        self.paint_band_profile(
                            &co_painter,
                            co_box,
                            ss,
                            ctx.spectra_3d.carbonyl_3d,
                            co_range,
                            &co_to_y,
                        );
                    }
                    let ali_painter = painter.with_clip_rect(ali_box);
                    self.paint_band_profile(
                        &ali_painter,
                        ali_box,
                        ss,
                        ctx.spectra_3d.aliphatic_3d,
                        ali_range,
                        &ali_to_y,
                    );

                    // Paint resolved candidate markers: (i) filled circles on left, (i-1) diamonds on right
                    self.paint_spin_system_markers(
                        &painter,
                        cx,
                        ss_idx,
                        ss,
                        co_range,
                        ali_range,
                        &co_to_y,
                        &ali_to_y,
                        &mut marker_hits,
                    );

                    // Pointer hover -> update synchronized crosshair
                    if let Some(hp) = canvas_resp.hover_pos() {
                        if (x0..=x1).contains(&hp.x) {
                            if show_co_band && co_box.contains(hp) {
                                let c_ppm = SyncCrosshair::strip_y_to_ppm(
                                    co_range.0, co_range.1, co_top, co_bot, hp.y,
                                );
                                if let Some(xh) = crosshair.as_deref_mut() {
                                    xh.update_from_strip_hover(Some(ss.h), Some(ss.n), c_ppm);
                                }
                            } else if ali_box.contains(hp) {
                                let c_ppm = SyncCrosshair::strip_y_to_ppm(
                                    ali_range.0,
                                    ali_range.1,
                                    ali_top,
                                    ali_bot,
                                    hp.y,
                                );
                                if let Some(xh) = crosshair.as_deref_mut() {
                                    xh.update_from_strip_hover(Some(ss.h), Some(ss.n), c_ppm);
                                }
                            }
                        }
                    }
                }

                // Shift+drag indirect zoom box visual overlay across all strips
                if let (Some(p0), Some(cur_pos)) =
                    (self.zoom_drag_start_ppm, canvas_resp.interact_pointer_pos())
                {
                    let y0 = ali_to_y(p0).clamp(ali_top, ali_bot);
                    let y1 = cur_pos.y.clamp(ali_top, ali_bot);
                    let z_rect = egui::Rect::from_min_max(
                        egui::pos2(x_pred_left, y0.min(y1)),
                        egui::pos2(x_succ_right, y0.max(y1)),
                    );
                    painter.rect(
                        z_rect,
                        2.0,
                        egui::Color32::from_rgba_unmultiplied(250, 204, 21, 32),
                        egui::Stroke::new(1.3, egui::Color32::from_rgb(250, 204, 21)),
                        egui::StrokeKind::Inside,
                    );
                }

                // Primary click: peak selection takes priority within 10 px, else header click -> SelectRoot
                if canvas_resp.clicked_by(egui::PointerButton::Primary)
                    && !canvas_resp.double_clicked()
                {
                    if let Some(cp) = canvas_resp.interact_pointer_pos() {
                        if let Some(&(m_ss, m_pk, _)) = marker_hits
                            .iter()
                            .filter(|(_, _, mp)| mp.distance(cp) <= 10.0)
                            .min_by(|(_, _, a), (_, _, b)| {
                                a.distance(cp).total_cmp(&b.distance(cp))
                            })
                        {
                            self.selected_peak = Some((m_ss, m_pk));
                            canvas_resp.request_focus();
                        } else {
                            for (c_idx, col) in columns.iter().enumerate() {
                                let x0 = canvas_rect.left()
                                    + left_axis_w
                                    + (c_idx as f32) * eff_col_w
                                    + 2.5;
                                let x1 = x0 + eff_col_w - 5.0;
                                let hdr_rect = egui::Rect::from_min_max(
                                    egui::pos2(x0, canvas_rect.top()),
                                    egui::pos2(x1, co_top),
                                );
                                let is_center = matches!(col, ColumnRole::ActiveRoot { .. });
                                if hdr_rect.contains(cp) && !is_center {
                                    actions.push(StripTilerAction::SelectRoot {
                                        spin_system_idx: col.ss_idx(),
                                    });
                                    break;
                                }
                            }
                        }
                    }
                }

                // Right-click context menu on strips
                if canvas_resp.secondary_clicked() {
                    if let Some(cp) = canvas_resp
                        .interact_pointer_pos()
                        .or_else(|| canvas_resp.hover_pos())
                    {
                        self.context_menu_ss_idx = col_rects
                            .iter()
                            .find(|(_, r)| r.contains(cp))
                            .map(|(ss_idx, _)| *ss_idx);
                    }
                }
                canvas_resp.context_menu(|ui| {
                    ui.set_min_width(190.0);
                    let target_col = self
                        .context_menu_ss_idx
                        .and_then(|s| columns.iter().find(|c| c.ss_idx() == s))
                        .or_else(|| {
                            columns
                                .iter()
                                .find(|c| matches!(c, ColumnRole::ActiveRoot { .. }))
                        });
                    if let Some(col) = target_col {
                        let ss_idx = col.ss_idx();
                        if ui.button("Walk to Strip").clicked() {
                            actions.push(StripTilerAction::SelectRoot {
                                spin_system_idx: ss_idx,
                            });
                            ui.close();
                        }
                        match col {
                            ColumnRole::Predecessor { .. } => {
                                if ui.button("Confirm Sequential Link").clicked() {
                                    actions.push(StripTilerAction::ConfirmSequentialLink {
                                        pred_idx: ss_idx,
                                        succ_idx: ctx.active_root,
                                    });
                                    ui.close();
                                }
                            }
                            ColumnRole::Successor { .. } => {
                                if ui.button("Confirm Sequential Link").clicked() {
                                    actions.push(StripTilerAction::ConfirmSequentialLink {
                                        pred_idx: ctx.active_root,
                                        succ_idx: ss_idx,
                                    });
                                    ui.close();
                                }
                            }
                            ColumnRole::ActiveRoot { .. } => {}
                        }
                        if let Some(target_k) = self.suggest_lock_residue(ctx, col) {
                            let lbl = format!(
                                "Lock Residue ({}{})",
                                three_to_one(&ctx.sequence[target_k]),
                                ctx.seq_ids[target_k]
                            );
                            if ui.button(lbl).clicked() {
                                actions.push(StripTilerAction::LockToResidue {
                                    spin_system_idx: ss_idx,
                                    residue_seq_idx: target_k,
                                });
                                ui.close();
                            }
                        }
                        ui.separator();
                    }
                    if ui.button("Reset Strip Zoom").clicked() {
                        self.indirect_zoom = None;
                        self.strip_gain = 1.0;
                        ui.close();
                    }
                });

                // Keyboard gestures on Strips (`UI-C6`, `UI-C10`, `UI-C13`, `UI-C18`):
                // - `+` / `-` scales strip intensity gain (`SIGNAL_STEP_RATIO`)
                // - `ArrowUp` / `ArrowDown` (unshifted) pans indirect 13C range by 10%
                // - `Shift + ArrowUp` / `Shift + ArrowDown` nudges selected strip peak
                // - `Esc` clears selected strip peak
                if !ui.ctx().egui_wants_keyboard_input()
                    && (canvas_resp.hovered() || canvas_resp.has_focus())
                {
                    let (more_sig, less_sig, pan_up, pan_down, esc_pressed) = ui.input(|i| {
                        let plain = !i.modifiers.command && !i.modifiers.ctrl && !i.modifiers.alt;
                        let more_txt = i
                            .events
                            .iter()
                            .any(|ev| matches!(ev, egui::Event::Text(s) if s == "+" || s == "="));
                        let less_txt = i
                            .events
                            .iter()
                            .any(|ev| matches!(ev, egui::Event::Text(s) if s == "-" || s == "_"));
                        let more_k = plain
                            && (i.key_pressed(egui::Key::Plus) || i.key_pressed(egui::Key::Equals));
                        let less_k = plain && i.key_pressed(egui::Key::Minus);
                        let p_up = plain && !i.modifiers.shift && i.key_pressed(egui::Key::ArrowUp);
                        let p_dn =
                            plain && !i.modifiers.shift && i.key_pressed(egui::Key::ArrowDown);
                        let esc = i.key_pressed(egui::Key::Escape);
                        (more_txt || more_k, less_txt || less_k, p_up, p_dn, esc)
                    });
                    if more_sig {
                        self.strip_gain = crate::interaction_grammar::gain_after(
                            self.strip_gain,
                            crate::interaction_grammar::SignalDirection::More,
                        )
                        .clamp(0.1, 50.0);
                    } else if less_sig {
                        self.strip_gain = crate::interaction_grammar::gain_after(
                            self.strip_gain,
                            crate::interaction_grammar::SignalDirection::Less,
                        )
                        .clamp(0.1, 50.0);
                    }
                    if pan_up || pan_down {
                        let cur = self
                            .indirect_zoom
                            .map(|(a, b)| (a.min(b), a.max(b)))
                            .unwrap_or(ali_range);
                        let step = ((cur.1 - cur.0).abs() * 0.10).max(0.1);
                        let d_ppm = if pan_up { -step } else { step };
                        self.indirect_zoom = Some((cur.0 + d_ppm, cur.1 + d_ppm));
                    }
                    if esc_pressed {
                        self.selected_peak = None;
                    }
                }
                if !ui.ctx().egui_wants_keyboard_input() {
                    if let Some((sel_strip, sel_peak)) = self.selected_peak {
                        let (up_nudge, down_nudge) = ui.input(|i| {
                            let sh = i.modifiers.shift && !i.modifiers.command && !i.modifiers.ctrl;
                            let ev_up = i.events.iter().any(|ev| {
                                matches!(
                                    ev,
                                    egui::Event::Key {
                                        key: egui::Key::ArrowUp,
                                        pressed: true,
                                        modifiers,
                                        ..
                                    } if modifiers.shift && !modifiers.command && !modifiers.ctrl
                                )
                            });
                            let ev_down = i.events.iter().any(|ev| {
                                matches!(
                                    ev,
                                    egui::Event::Key {
                                        key: egui::Key::ArrowDown,
                                        pressed: true,
                                        modifiers,
                                        ..
                                    } if modifiers.shift && !modifiers.command && !modifiers.ctrl
                                )
                            });
                            (
                                (sh && i.key_pressed(egui::Key::ArrowUp)) || ev_up,
                                (sh && i.key_pressed(egui::Key::ArrowDown)) || ev_down,
                            )
                        });
                        if up_nudge || down_nudge {
                            let step =
                                ((ali_range.1 - ali_range.0).abs() * 0.005).clamp(0.02, 0.25);
                            let d_ppm = if up_nudge { -step } else { step };
                            actions.push(StripTilerAction::NudgeStripPeak {
                                strip: sel_strip,
                                peak: sel_peak,
                                d_ppm,
                            });
                        }
                    }
                }

                // Draw synchronized indirect crosshair line across all strips when c_ppm is active
                if let Some(xh) = crosshair.as_deref() {
                    let line_col = if xh.locked {
                        egui::Color32::from_rgba_unmultiplied(255, 215, 90, 190)
                    } else {
                        egui::Color32::from_rgba_unmultiplied(185, 225, 255, 125)
                    };
                    if show_co_band {
                        if let Some(y_co) =
                            xh.project_c_to_strip_y(co_range.0, co_range.1, co_top, co_bot)
                        {
                            painter.line_segment(
                                [
                                    egui::pos2(x_pred_left, y_co),
                                    egui::pos2(x_succ_right, y_co),
                                ],
                                egui::Stroke::new(1.0, line_col),
                            );
                        }
                    }
                    if let Some(y_ali) =
                        xh.project_c_to_strip_y(ali_range.0, ali_range.1, ali_top, ali_bot)
                    {
                        painter.line_segment(
                            [
                                egui::pos2(x_pred_left, y_ali),
                                egui::pos2(x_succ_right, y_ali),
                            ],
                            egui::Stroke::new(1.0, line_col),
                        );
                    }
                }

                // Drag handle to resize strip height
                let (grip_rect, grip_resp) =
                    ui.allocate_exact_size(egui::vec2(desired_w, 7.0), egui::Sense::drag());
                if grip_resp.hovered() || grip_resp.dragged() {
                    ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeVertical);
                }
                if grip_resp.dragged() {
                    self.user_resized_height = true;
                    self.strip_height_px =
                        (self.strip_height_px + grip_resp.drag_delta().y).clamp(220.0, 680.0);
                }
                let grip_col = if grip_resp.hovered() || grip_resp.dragged() {
                    egui::Color32::from_rgb(95, 210, 255)
                } else {
                    egui::Color32::from_rgb(56, 70, 92)
                };
                ui.painter().line_segment(
                    [
                        egui::pos2(grip_rect.center().x - 28.0, grip_rect.center().y),
                        egui::pos2(grip_rect.center().x + 28.0, grip_rect.center().y),
                    ],
                    egui::Stroke::new(2.0, grip_col),
                );

                self.last_column_rects = col_rects.iter().map(|(_, r)| *r).collect();

                // 2. Per-column residual summary & 1-click curation action buttons (optional; right-click always available)
                if self.show_footer_buttons {
                ui.horizontal(|ui| {
                    ui.add_space(left_axis_w);
                    for col in &columns {
                        let ss_idx = col.ss_idx();
                        ui.allocate_ui(egui::vec2(eff_col_w - 4.0, 66.0), |ui| {
                            ui.vertical_centered(|ui| {
                                match col {
                                    ColumnRole::Predecessor { cand, .. }
                                    | ColumnRole::Successor { cand, .. } => {
                                        let resid_txt = format_residuals_compact(cand);
                                        ui.label(
                                            egui::RichText::new(resid_txt)
                                                .monospace()
                                                .size(TYPE_CAPTION_PX),
                                        );
                                        ui.horizontal(|ui| {
                                            if ui.small_button("Walk").clicked() {
                                                actions.push(StripTilerAction::SelectRoot {
                                                    spin_system_idx: ss_idx,
                                                });
                                            }
                                            let (pred_idx, succ_idx) = match col {
                                                ColumnRole::Predecessor { .. } => {
                                                    (ss_idx, ctx.active_root)
                                                }
                                                _ => (ctx.active_root, ss_idx),
                                            };
                                            if ui.small_button("Link").clicked() {
                                                actions.push(
                                                    StripTilerAction::ConfirmSequentialLink {
                                                        pred_idx,
                                                        succ_idx,
                                                    },
                                                );
                                            }
                                        });
                                    }
                                    ColumnRole::ActiveRoot { .. } => {
                                        ui.label(
                                            egui::RichText::new("Active (i)")
                                                .monospace()
                                                .size(TYPE_CAPTION_PX)
                                                .color(egui::Color32::from_rgb(110, 230, 255)),
                                        );
                                    }
                                }
                                if let Some(target_k) = self.suggest_lock_residue(ctx, col) {
                                    let rlabel = format!(
                                        "Lock {}{}",
                                        three_to_one(&ctx.sequence[target_k]),
                                        ctx.seq_ids[target_k]
                                    );
                                    if ui.small_button(rlabel).clicked() {
                                        actions.push(StripTilerAction::LockToResidue {
                                            spin_system_idx: ss_idx,
                                            residue_seq_idx: target_k,
                                        });
                                    }
                                }
                            });
                        });
                    }
                });
                }
        };
        if no_hscroll {
            render_strips(ui);
        } else {
            egui::ScrollArea::horizontal()
                .id_salt("strip_tiler_hscroll")
                .show(ui, render_strips);
        }

        actions
    }

    fn paint_band_profile(
        &self,
        painter: &egui::Painter,
        band_box: egui::Rect,
        ss: &SpinSystem,
        spec3d: Option<&Spectrum3D>,
        c_range: (f64, f64),
        c_to_y: &dyn Fn(f64) -> f32,
    ) {
        let cx = band_box.center().x;
        let lo = c_range.0.min(c_range.1);
        let hi = c_range.0.max(c_range.1);
        let gain = self.strip_gain as f32;

        if let Some(s3) = spec3d {
            let slice = s3.extract_strip(ss.h, ss.n, self.strip_half_width_h_ppm);
            let max_v = slice.max_abs().max(1.0);
            let nx = slice.nx().max(1);
            let col_dx = (band_box.width() * 0.80) / (nx as f32);
            for ic in 0..slice.ny() {
                let c_ppm = slice.axis_y.index_to_ppm(ic as f64);
                if !(lo..=hi).contains(&c_ppm) {
                    continue;
                }
                let y = c_to_y(c_ppm);
                for ix in 0..nx {
                    let val = slice.get(ix, ic);
                    let norm = ((val.abs() / max_v) * gain).clamp(0.0, 1.0);
                    if norm > 0.14 {
                        let col = if val >= 0.0 {
                            egui::Color32::from_rgba_unmultiplied(
                                45,
                                200,
                                255,
                                (norm * 145.0) as u8,
                            )
                        } else {
                            egui::Color32::from_rgba_unmultiplied(
                                255,
                                105,
                                95,
                                (norm * 145.0) as u8,
                            )
                        };
                        let px0 = band_box.left() + band_box.width() * 0.10 + (ix as f32) * col_dx;
                        painter.line_segment(
                            [egui::pos2(px0, y), egui::pos2(px0 + col_dx.max(1.5), y)],
                            egui::Stroke::new(2.0, col),
                        );
                    }
                }
            }
        } else {
            // Peak-only synthetic Gaussian strip profile
            let n_samples = (band_box.height().round() as usize).clamp(24, 180);
            for s_idx in 0..=n_samples {
                let t = (s_idx as f64) / (n_samples as f64);
                let c_ppm = c_range.0 + t * (c_range.1 - c_range.0);
                let amp = synthetic_strip_intensity_at_ppm(ss, c_ppm);
                let norm = (amp.abs() * gain).clamp(0.0, 1.0);
                if norm > 0.08 {
                    let y = c_to_y(c_ppm);
                    let col = if amp >= 0.0 {
                        egui::Color32::from_rgba_unmultiplied(55, 200, 255, (norm * 115.0) as u8)
                    } else {
                        egui::Color32::from_rgba_unmultiplied(255, 115, 85, (norm * 115.0) as u8)
                    };
                    let half_w = norm * (band_box.width() * 0.34);
                    painter.line_segment(
                        [egui::pos2(cx - half_w, y), egui::pos2(cx + half_w, y)],
                        egui::Stroke::new(1.8, col),
                    );
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn paint_spin_system_markers(
        &self,
        painter: &egui::Painter,
        cx: f32,
        ss_idx: usize,
        ss: &SpinSystem,
        co_range: (f64, f64),
        ali_range: (f64, f64),
        co_to_y: &dyn Fn(f64) -> f32,
        ali_to_y: &dyn Fn(f64) -> f32,
        marker_hits: &mut Vec<(usize, usize, egui::Pos2)>,
    ) {
        let co_lo = co_range.0.min(co_range.1);
        let co_hi = co_range.0.max(co_range.1);
        let ali_lo = ali_range.0.min(ali_range.1);
        let ali_hi = ali_range.0.max(ali_range.1);

        for (peak_idx, ((atom, off), cands)) in ss.cands.iter().enumerate() {
            let (base_col, is_co) = match atom.as_str() {
                "CA" => (egui::Color32::from_rgb(75, 225, 255), false),
                "CB" => (egui::Color32::from_rgb(255, 120, 80), false),
                "C" => (egui::Color32::from_rgb(90, 235, 135), true),
                _ => continue,
            };
            let Some(&(ppm, sup)) = cands.first() else {
                continue;
            };
            if *off == 0 && sup == 0 {
                continue;
            }
            let in_range = if is_co {
                (co_lo..=co_hi).contains(&ppm)
            } else {
                (ali_lo..=ali_hi).contains(&ppm)
            };
            if !in_range {
                continue;
            }
            let y = if is_co { co_to_y(ppm) } else { ali_to_y(ppm) };
            let pos = if *off == 0 {
                let p = egui::pos2(cx - 7.0, y);
                painter.circle_filled(p, 3.6, base_col);
                p
            } else {
                let p = egui::pos2(cx + 7.0, y);
                let pts = vec![
                    egui::pos2(p.x, y - 3.8),
                    egui::pos2(p.x + 3.8, y),
                    egui::pos2(p.x, y + 3.8),
                    egui::pos2(p.x - 3.8, y),
                ];
                painter.add(egui::Shape::convex_polygon(
                    pts,
                    base_col,
                    egui::Stroke::NONE,
                ));
                p
            };
            if self.selected_peak == Some((ss_idx, peak_idx)) {
                painter.circle_stroke(
                    pos,
                    6.0,
                    egui::Stroke::new(1.6, egui::Color32::from_rgb(250, 204, 21)),
                );
            }
            marker_hits.push((ss_idx, peak_idx, pos));
        }
    }
}

fn draw_dashed_hline(
    painter: &egui::Painter,
    x_start: f32,
    x_end: f32,
    y: f32,
    color: egui::Color32,
) {
    let span = (x_end - x_start).abs();
    if span < 2.0 {
        return;
    }
    let dash_px = 6.0f32;
    let gap_px = 4.0f32;
    let step = dash_px + gap_px;
    let x0 = x_start.min(x_end);
    let x1 = x_start.max(x_end);
    let mut x = x0;
    while x < x1 {
        let seg_end = (x + dash_px).min(x1);
        painter.line_segment(
            [egui::pos2(x, y), egui::pos2(seg_end, y)],
            egui::Stroke::new(1.2, color),
        );
        x += step;
    }
}

fn format_residuals_compact(cand: &StripMatchCandidate) -> String {
    let mut parts = Vec::with_capacity(3);
    if let Some(d) = cand.delta_ca_ppm {
        parts.push(format!("α{d:+.2}"));
    }
    if let Some(d) = cand.delta_cb_ppm {
        parts.push(format!("β{d:+.2}"));
    }
    if let Some(d) = cand.delta_c_ppm {
        parts.push(format!("C'{d:+.2}"));
    }
    if parts.is_empty() {
        "—".to_string()
    } else {
        parts.join(" ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::{Path, PathBuf};

    use nmr_core::nmrstar::{read_star_str, Reference};
    use nmr_core::picker::{
        corroborated_re_pick_3d, pick_hsqc_2d, CorroboratedRepickConfig, PickerConfig, StripBank3D,
    };
    use nmr_core::priors::one_to_three;
    use nmr_core::simulate::{simulate, SimConfig};
    use nmr_core::solver::{pair_matrix, unary_matrix};
    use nmr_core::spectrum::{read_ucsf_2d, read_ucsf_3d};
    use nmr_core::spinsystems::{build_spin_systems, Tolerances};

    fn make_ss(h: f64, n: f64, entries: &[(&str, i32, f64, usize)]) -> SpinSystem {
        let mut ss = SpinSystem::new(h, n);
        for &(atom, off, ppm, sup) in entries {
            ss.cands
                .entry((atom.to_string(), off))
                .or_default()
                .push((ppm, sup));
        }
        ss
    }

    #[test]
    fn test_candidate_match_finder_residuals_probabilities_and_locks() {
        // Build a 4-spin-system chain: SS0 -> SS1 -> SS2, plus SS3 (close competitor for SS1 -> ?)
        // and SS4 (carbon-less noise root that must NEVER appear in candidate lists).
        let systems = vec![
            make_ss(
                8.30,
                120.0,
                &[
                    ("CA", 0, 56.20, 2),
                    ("CB", 0, 31.40, 2),
                    ("C", 0, 175.10, 2),
                ],
            ),
            make_ss(
                8.10,
                118.4,
                &[
                    ("CA", -1, 56.24, 2),
                    ("CB", -1, 31.35, 2),
                    ("C", -1, 175.08, 2),
                    ("CA", 0, 61.10, 2),
                    ("CB", 0, 68.50, 2),
                    ("C", 0, 173.80, 2),
                ],
            ),
            make_ss(
                8.65,
                123.2,
                &[
                    ("CA", -1, 61.08, 2),
                    ("CB", -1, 68.54, 2),
                    ("C", -1, 173.82, 2),
                    ("CA", 0, 45.10, 2),
                    ("C", 0, 171.90, 2),
                ],
            ),
            make_ss(
                7.85,
                116.0,
                &[
                    ("CA", -1, 61.22, 1),
                    ("CB", -1, 68.75, 1),
                    ("C", -1, 174.05, 1),
                    ("CA", 0, 52.00, 1),
                ],
            ),
            SpinSystem::new(7.40, 112.1), // zero carbons
        ];

        let scfg = ScoreConfig::default();
        let p_mat = pair_matrix(&systems, &scfg);
        let res = CandidateMatchFinder::rank_top_k(&systems, &p_mat, 1, 5);

        // Carbon-less SS4 must be excluded
        assert!(res.predecessors.iter().all(|c| c.spin_system_idx != 4));
        assert!(res.successors.iter().all(|c| c.spin_system_idx != 4));

        // Top predecessor of SS1 is SS0 with 3 matched nuclei and tiny residuals
        assert_eq!(res.predecessors[0].spin_system_idx, 0);
        assert_eq!(res.predecessors[0].matched_nuclei, 3);
        assert!((res.predecessors[0].delta_ca_ppm.unwrap() - (-0.04)).abs() < 1e-9);
        assert!((res.predecessors[0].delta_cb_ppm.unwrap() - 0.05).abs() < 1e-9);
        assert!((res.predecessors[0].delta_c_ppm.unwrap() - 0.02).abs() < 1e-9);
        assert!(res.predecessors[0].rel_prob > 0.99);

        // Top successor of SS1 is SS2, followed by competitor SS3
        assert_eq!(res.successors[0].spin_system_idx, 2);
        assert_eq!(res.successors[1].spin_system_idx, 3);
        assert!(res.successors[0].rel_prob > res.successors[1].rel_prob);
        let prob_sum: f64 = res.successors.iter().map(|c| c.rel_prob).sum();
        assert!(prob_sum > 0.99 && prob_sum <= 1.0 + 1e-9);

        // Lock SS1 to k=5 and SS2 to conflicting k=9: with exclude_conflicting_locks=true, SS2 drops out
        let mut locks = BTreeMap::new();
        locks.insert(1usize, 5usize);
        locks.insert(2usize, 9usize);
        let locked_res = CandidateMatchFinder::rank_neighbors(
            &systems,
            &p_mat,
            1,
            None,
            Some(&locks),
            None,
            &MatchFinderConfig {
                top_k: 5,
                score_cfg: scfg,
                exclude_conflicting_locks: true,
            },
        );
        assert_eq!(locked_res.successors[0].spin_system_idx, 3);

        // Synthetic Gaussian strip profile polarity check on SS1: CA(0)=61.10 positive, CB(0)=68.50 negative
        assert!(synthetic_strip_intensity_at_ppm(&systems[1], 61.10) > 0.7);
        assert!(synthetic_strip_intensity_at_ppm(&systems[1], 68.50) < -0.7);
    }

    #[test]
    fn test_strip_tiler_panel_headless_egui_render() {
        let systems = vec![
            make_ss(8.30, 120.0, &[("CA", 0, 56.2, 2), ("C", 0, 175.1, 2)]),
            make_ss(
                8.10,
                118.4,
                &[
                    ("CA", -1, 56.2, 2),
                    ("C", -1, 175.1, 2),
                    ("CA", 0, 61.1, 2),
                    ("CB", 0, 68.5, 2),
                ],
            ),
            make_ss(
                8.65,
                123.2,
                &[("CA", -1, 61.1, 2), ("CB", -1, 68.5, 2), ("CA", 0, 45.1, 2)],
            ),
        ];
        let seq = vec![
            "MET".to_string(),
            "GLN".to_string(),
            "ILE".to_string(),
            "THR".to_string(),
        ];
        let seq_ids = vec![1, 2, 3, 4];
        let scfg = ScoreConfig::default();
        let p_mat = pair_matrix(&systems, &scfg);
        let u_mat = unary_matrix(
            &systems,
            &seq,
            &nmr_core::priors::Priors::provisional(),
            &scfg,
        );
        let assignments = vec![1, 2, 3];

        let mut panel = StripTilerPanel::new();
        let mut xh = SyncCrosshair::new(8.10, 118.4, Some(61.1));
        let egui_ctx = egui::Context::default();
        let mut out = egui_ctx.run_ui(egui::RawInput::default(), |ui| {
            let tctx = StripTilerContext {
                systems: &systems,
                pair_mat: &p_mat,
                u_mat: Some(&u_mat),
                sequence: &seq,
                seq_ids: &seq_ids,
                active_root: 1,
                assignments: Some(&assignments),
                locks: None,
                confirmed_links: None,
                spectra_3d: StripSpectra3D::default(),
            };
            assert_eq!(
                panel.suggest_lock_residue(&tctx, &ColumnRole::ActiveRoot { ss_idx: 1 }),
                Some(2)
            );
            let matches = CandidateMatchFinder::rank_top_k(&systems, &p_mat, 1, 3);
            assert_eq!(
                panel.suggest_lock_residue(
                    &tctx,
                    &ColumnRole::Predecessor {
                        rank_1based: 1,
                        cand: matches.predecessors[0].clone(),
                    }
                ),
                Some(1)
            );
            assert_eq!(
                panel.suggest_lock_residue(
                    &tctx,
                    &ColumnRole::Successor {
                        rank_1based: 1,
                        cand: matches.successors[0].clone(),
                    }
                ),
                Some(3)
            );
            let actions = panel.show(ui, &tctx, Some(&mut xh));
            assert!(actions.is_empty());
        });
        out.textures_delta.clear();

        // Also verify volumetric 3D slice path (aliphatic + carbonyl cubes) + confirmed_links
        let hncacb = Spectrum3D {
            name: "HNCACB".to_string(),
            exp: "HNCACB".to_string(),
            axis_x: nmr_core::spectrum::AxisPpm::from_bounds("1H", 16, 9.5, 7.5, 600.0),
            axis_y: nmr_core::spectrum::AxisPpm::from_bounds("15N", 12, 128.0, 112.0, 60.8),
            axis_z: nmr_core::spectrum::AxisPpm::from_bounds("13C", 32, 74.0, 14.0, 150.9),
            data: vec![5.0f32; 16 * 12 * 32],
        };
        let hnco = Spectrum3D {
            name: "HNCO".to_string(),
            exp: "HNCO".to_string(),
            axis_x: nmr_core::spectrum::AxisPpm::from_bounds("1H", 16, 9.5, 7.5, 600.0),
            axis_y: nmr_core::spectrum::AxisPpm::from_bounds("15N", 12, 128.0, 112.0, 60.8),
            axis_z: nmr_core::spectrum::AxisPpm::from_bounds("13C", 16, 182.0, 168.0, 150.9),
            data: vec![8.0f32; 16 * 12 * 16],
        };
        let confirmed = vec![(0usize, 1usize)];
        let mut out2 = egui_ctx.run_ui(egui::RawInput::default(), |ui| {
            let tctx = StripTilerContext {
                systems: &systems,
                pair_mat: &p_mat,
                u_mat: Some(&u_mat),
                sequence: &seq,
                seq_ids: &seq_ids,
                active_root: 1,
                assignments: Some(&assignments),
                locks: None,
                confirmed_links: Some(&confirmed),
                spectra_3d: StripSpectra3D {
                    aliphatic_3d: Some(&hncacb),
                    carbonyl_3d: Some(&hnco),
                },
            };
            let actions = panel.show(ui, &tctx, Some(&mut xh));
            assert!(actions.is_empty());
        });
        out2.textures_delta.clear();
    }

    #[test]
    fn test_strip_tiler_zoom_gain_resize_and_nudge_action() {
        let systems = vec![
            make_ss(8.30, 120.0, &[("CA", 0, 56.2, 2), ("C", 0, 175.1, 2)]),
            make_ss(
                8.10,
                118.4,
                &[
                    ("CA", -1, 56.2, 2),
                    ("C", -1, 175.1, 2),
                    ("CA", 0, 61.1, 2),
                    ("CB", 0, 68.5, 2),
                ],
            ),
            make_ss(
                8.65,
                123.2,
                &[("CA", -1, 61.1, 2), ("CB", -1, 68.5, 2), ("CA", 0, 45.1, 2)],
            ),
        ];
        let seq = vec![
            "MET".to_string(),
            "GLN".to_string(),
            "ILE".to_string(),
            "THR".to_string(),
        ];
        let seq_ids = vec![1, 2, 3, 4];
        let scfg = ScoreConfig::default();
        let p_mat = pair_matrix(&systems, &scfg);
        let assignments = vec![1, 2, 3];
        let mut locks = BTreeMap::new();
        locks.insert(1usize, 2usize);

        let mut panel = StripTilerPanel::new();
        panel.selected_peak = Some((1, 2));

        let raw_1280 = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::pos2(0.0, 0.0),
                egui::vec2(1280.0, 800.0),
            )),
            events: vec![egui::Event::Key {
                key: egui::Key::ArrowDown,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::SHIFT,
            }],
            ..Default::default()
        };

        let egui_ctx = egui::Context::default();
        let mut emitted = Vec::new();
        let mut out = egui_ctx.run_ui(raw_1280, |ui| {
            let tctx = StripTilerContext {
                systems: &systems,
                pair_mat: &p_mat,
                u_mat: None,
                sequence: &seq,
                seq_ids: &seq_ids,
                active_root: 1,
                assignments: Some(&assignments),
                locks: Some(&locks),
                confirmed_links: None,
                spectra_3d: StripSpectra3D::default(),
            };
            emitted = panel.show(ui, &tctx, None);
        });
        let audit = crate::workbench_mode::audit_text_layout(
            &out.shapes,
            egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(1280.0, 800.0)),
        );
        out.textures_delta.clear();

        assert!(
            audit.min_font_px >= 11.0,
            "strip_tiler font below 11px floor: {}",
            audit.min_font_px
        );
        assert!(
            audit.clipped.is_empty(),
            "strip_tiler clipped text at 1280x800: {:?}",
            audit.clipped
        );

        // Strip height auto-filled available height within 220..=680
        assert!(
            (220.0..=680.0).contains(&panel.strip_height_px) && panel.strip_height_px > 286.0,
            "expected responsive strip height > 286 at 1280x800, got {}",
            panel.strip_height_px
        );
        // Shift+Down nudged the selected strip peak
        assert_eq!(emitted.len(), 1);
        match emitted[0] {
            StripTilerAction::NudgeStripPeak { strip, peak, d_ppm } => {
                assert_eq!(strip, 1);
                assert_eq!(peak, 2);
                assert!(d_ppm > 0.0);
            }
            ref other => panic!("expected NudgeStripPeak, got {other:?}"),
        }
    }

    fn evaluate_top3_sequential_neighbor_recall(
        systems: &[SpinSystem],
        pair_mat: &Matrix2D,
        truth_sys_for_res: &[Option<usize>],
    ) -> (usize, usize, f64) {
        let mut hits = 0usize;
        let mut total = 0usize;
        for k in 1..truth_sys_for_res.len() {
            let (Some(s_prev), Some(s_curr)) = (truth_sys_for_res[k - 1], truth_sys_for_res[k])
            else {
                continue;
            };
            if s_prev == s_curr {
                continue;
            }
            // 1. Successor (i + 1) query from s_prev must recover s_curr in top-3
            let res_prev = CandidateMatchFinder::rank_top_k(systems, pair_mat, s_prev, 3);
            total += 1;
            if res_prev
                .successors
                .iter()
                .any(|c| c.spin_system_idx == s_curr)
            {
                hits += 1;
            }

            // 2. Predecessor (i - 1) query from s_curr must recover s_prev in top-3
            let res_curr = CandidateMatchFinder::rank_top_k(systems, pair_mat, s_curr, 3);
            total += 1;
            if res_curr
                .predecessors
                .iter()
                .any(|c| c.spin_system_idx == s_prev)
            {
                hits += 1;
            }
        }
        let recall = if total > 0 {
            (hits as f64) / (total as f64)
        } else {
            0.0
        };
        (hits, total, recall)
    }

    #[test]
    fn test_synthetic_ubiquitin_top3_neighbor_recovery() {
        let star_txt = include_str!("../../../data/fixtures/synthetic_ubiquitin.str");
        let u_ref = read_star_str(star_txt, None).expect("parse synthetic_ubiquitin.str");
        let (peaks, truth) = simulate(
            &u_ref,
            &SimConfig {
                seed: 0,
                ..SimConfig::default()
            },
        );
        let systems = build_spin_systems(&peaks, &Tolerances::default(), 3, false).unwrap();
        let p_mat = pair_matrix(&systems, &ScoreConfig::default());

        let mut truth_sys_for_res = vec![None; u_ref.sequence.len()];
        for (s, &k) in truth.iter().enumerate() {
            if k >= 0 && (k as usize) < truth_sys_for_res.len() {
                truth_sys_for_res[k as usize] = Some(s);
            }
        }

        let (hits, total, recall) =
            evaluate_top3_sequential_neighbor_recall(&systems, &p_mat, &truth_sys_for_res);
        assert!(
            total >= 110 && recall >= 0.97,
            "Expected >= 97% top-3 sequential neighbor recall on synthetic Ubiquitin, got {hits}/{total} ({:.2}%)",
            recall * 100.0
        );
    }

    fn find_local_real_data_dir() -> Option<PathBuf> {
        if let Some(env_dir) = std::env::var_os("NMR_REAL_DATA_DIR") {
            let p = PathBuf::from(env_dir);
            return p.exists().then_some(p);
        }
        if let Some(home) = std::env::var_os("HOME") {
            let p = PathBuf::from(home).join("Workspace/nmr_software/data");
            if p.exists() {
                return Some(p);
            }
        }
        None
    }

    fn parse_fasta_three(path: &Path) -> Vec<String> {
        let txt = fs::read_to_string(path).expect("read FASTA");
        let mut seq = Vec::new();
        for line in txt.lines() {
            let t = line.trim();
            if t.is_empty() || t.starts_with('>') {
                continue;
            }
            for ch in t.chars() {
                if ch.is_ascii_alphabetic() {
                    seq.push(one_to_three(ch).to_string());
                }
            }
        }
        seq
    }

    fn parse_ubiquitin_truth(table_path: &Path, seq: Vec<String>) -> Reference {
        let txt = fs::read_to_string(table_path).expect("read Ubiquitin reference table");
        let n_res = seq.len();
        let mut truth = Reference::new(seq, (1..=n_res as i32).collect());
        for line in txt.lines() {
            let t = line.trim();
            if t.is_empty() || t.starts_with("Index") || t.starts_with('!') || t.starts_with('#') {
                continue;
            }
            let norm = t.replace("?1", "? 1").replace(":H* ?", ":H*?");
            let cols: Vec<&str> = norm.split_whitespace().collect();
            let Some(Ok(sn)) = cols.first().map(|s| s.parse::<usize>()) else {
                continue;
            };
            if sn == 0 || sn > n_res {
                continue;
            }
            let k = sn - 1;
            let pv = |s: &str| {
                (!s.contains('?') && s != "-")
                    .then(|| s.split(':').next().unwrap().parse::<f64>().ok())
                    .flatten()
            };
            if let Some(v) = cols.get(2).and_then(|s| pv(s)) {
                truth.shifts.insert((k, "H".into()), v);
            }
            if let Some(v) = cols.get(3).and_then(|s| pv(s)) {
                truth.shifts.insert((k, "N".into()), v);
            }
            if let Some(v) = cols.get(4).and_then(|s| pv(s)) {
                truth.shifts.insert((k, "C".into()), v);
            }
            if let Some(v) = cols.get(5).and_then(|s| pv(s)) {
                truth.shifts.insert((k, "CA".into()), v);
            }
            if let Some(v) = cols
                .iter()
                .skip(6)
                .find(|s| s.ends_with(":Cb"))
                .and_then(|s| pv(s))
            {
                truth.shifts.insert((k, "CB".into()), v);
            }
        }
        truth
    }

    #[test]
    fn test_real_ubiquitin_ucsf_top3_neighbor_recovery_gate() {
        let Some(droot) = find_local_real_data_dir() else {
            eprintln!("Skipping real Ubiquitin .ucsf top-3 strip match-finder gate: NMR_REAL_DATA_DIR absent");
            return;
        };
        let ucsf_dir = droot.join("ubiquitin/sparky_files/ubiquitin_data");
        let truth_path = droot.join("ubiquitin_backbone_assignment_191104.txt");
        let fasta_path = ucsf_dir.join("1ubq.fasta.txt");
        if !ucsf_dir.join("HSQC.ucsf").exists() || !truth_path.exists() || !fasta_path.exists() {
            eprintln!("Skipping real Ubiquitin .ucsf top-3 gate: local .ucsf files absent");
            return;
        }

        let seq = parse_fasta_three(&fasta_path);
        let truth_ref = parse_ubiquitin_truth(&truth_path, seq.clone());

        let pcfg = PickerConfig::default();
        let repick_cfg = CorroboratedRepickConfig::default();
        let hsqc = read_ucsf_2d(ucsf_dir.join("HSQC.ucsf")).expect("read HSQC.ucsf");
        let roots = pick_hsqc_2d(&hsqc, &pcfg);
        let mut peaks = roots.clone();
        let mut bank = StripBank3D::new();
        for (fname, exp) in [
            ("HNCACB.ucsf", "HNCACB"),
            ("CBCACONH.ucsf", "CBCACONH"),
            ("HNCO.ucsf", "HNCO"),
            ("HNCACO.ucsf", "HNCACO"),
        ] {
            let s3 = read_ucsf_3d(ucsf_dir.join(fname), exp).expect("read 3D .ucsf");
            peaks.extend(bank.pick_and_record_3d(&s3, &roots, &pcfg));
        }
        let rescued = corroborated_re_pick_3d(&bank, &roots, &peaks, &pcfg, &repick_cfg);
        peaks.extend(rescued);

        let systems = build_spin_systems(&peaks, &Tolerances::default(), 3, false)
            .expect("build Ubiquitin .ucsf spin systems");
        let p_mat = pair_matrix(&systems, &ScoreConfig::default());

        // Map each reference amide residue 1-to-1 onto its picked spin system
        let mut used_sys = vec![false; systems.len()];
        let mut truth_sys_for_res = vec![None; seq.len()];
        for (k, aa) in seq.iter().enumerate().skip(1) {
            if aa == "PRO" {
                continue;
            }
            let (Some(&th), Some(&tn)) = (
                truth_ref.shifts.get(&(k, "H".to_string())),
                truth_ref.shifts.get(&(k, "N".to_string())),
            ) else {
                continue;
            };
            let mut best: Option<(usize, f64)> = None;
            for (s, ss) in systems.iter().enumerate() {
                if used_sys[s] || !ss.has_observed_carbons() {
                    continue;
                }
                let d = ((ss.h - th) / 0.08).hypot((ss.n - tn) / 0.65);
                if d <= 1.0 && best.is_none_or(|(_, bd)| d < bd) {
                    best = Some((s, d));
                }
            }
            if let Some((s, _)) = best {
                used_sys[s] = true;
                truth_sys_for_res[k] = Some(s);
            }
        }

        let (hits, total, recall) =
            evaluate_top3_sequential_neighbor_recall(&systems, &p_mat, &truth_sys_for_res);
        println!(
            "[Ubiquitin .ucsf Strip Match-Finder] Top-3 sequential neighbor recovery: {hits}/{total} ({:.2}%)",
            recall * 100.0
        );
        assert!(
            total >= 110 && recall >= 0.97,
            "Ubiquitin .ucsf top-3 sequential neighbor recovery {:.2}% ({hits}/{total}) below 97.0% gate",
            recall * 100.0
        );
    }
}
