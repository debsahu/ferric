//! Zero-emoji stacked 3-track `egui::Painter` vector secondary-structure & AlphaFold-vs-NMR
//! concordance ribbon widget (`T21.3`).
//!
//! Provenance: Clean-room original implementation (`docs/specs/21-alphafold-integrative-nmr.md`
//! §1.2, §1.3 & §3.1) grounded strictly in peer-reviewed structural biology and NMR literature:
//! - Kabsch, W. & Sander, C. (*Biopolymers* 1983, 22, 2577–2637 — electrostatic H-bond helices
//!   and beta-strands).
//! - Adzhubei, A. A., Sternberg, M. J. E. & Makarov, A. A. (*J. Mol. Biol.* 2013, 425, 2100–2132
//!   — left-handed polyproline II `PPII` secondary-structure state).
//! - Marsh, J. A., Singh, V. K., Jia, Z. & Forman-Kay, J. D. (*Protein Sci.* 2006, 15, 2795–2804
//!   — `SSP` secondary-structure propensity score) & Camilloni, C. et al. (*Biochemistry* 2012,
//!   51, 2224–2231 — 4-state `delta2D` conformational populations).
//! - Berjanskii, M. V. & Wishart, D. S. (*J. Am. Chem. Soc.* 2005, 127, 14970–14971 — Random Coil
//!   Index `RCI` and chemical-shift order parameter `S^2_RCI`).
//! - Fowler, N. J., Sljoka, A. & Williamson, M. P. (*Nat. Commun.* 2020, 11, 5982; *Structure*
//!   2022, 30, 925–933) & Alderson, T. R. et al. (*Proc. Natl. Acad. Sci. U.S.A.* 2023, 120,
//!   e2304302120 — `pLDDT` vs solution NMR rigidity concordance & conditionally folding MoRFs).

use std::collections::BTreeMap;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

use eframe::egui;
use nmr_core::idp_propensity::{compute_idp_propensity, PropensityConfig, PropensityReport};
use nmr_core::nmrstar::Reference;
use nmr_core::priors::three_to_one;

use crate::contour_painter::SyncCrosshair;
pub use crate::mol3d_overlay::{AfNmrConcordanceClass, ConcordanceTrackClass, SsState4};
use crate::mol3d_viewer::{Mol3dViewerPanel, StructureEnsemble3d};
use crate::sidechain_widget::SequenceRibbonState;

/// Default horizontal width (in logical points) per residue column at `1.0x` zoom.
pub const DEFAULT_SS_COL_WIDTH_PX: f32 = 22.0;
/// Minimum zoomable residue column width (`px`).
pub const MIN_SS_COL_WIDTH_PX: f32 = 8.0;
/// Maximum zoomable residue column width (`px`).
pub const MAX_SS_COL_WIDTH_PX: f32 = 64.0;
/// Width of the pinned left track-title gutter (`px`).
pub const SS_TRACK_HEADER_WIDTH_PX: f32 = 152.0;

#[inline]
fn monotonic_ms_start() -> u128 {
    #[cfg(not(target_arch = "wasm32"))]
    {
        // Only used through `elapsed_ms_since` below; return 0 placeholder on wasm32.
        0
    }
    #[cfg(target_arch = "wasm32")]
    {
        0
    }
}

struct FastTimer {
    #[cfg(not(target_arch = "wasm32"))]
    t0: Instant,
    #[cfg(target_arch = "wasm32")]
    t0_ms: f64,
}

impl FastTimer {
    #[inline]
    fn start() -> Self {
        let _ = monotonic_ms_start();
        #[cfg(not(target_arch = "wasm32"))]
        {
            Self { t0: Instant::now() }
        }
        #[cfg(target_arch = "wasm32")]
        {
            let t0_ms = web_sys::window()
                .and_then(|w| w.performance())
                .map(|p| p.now())
                .unwrap_or(0.0);
            Self { t0_ms }
        }
    }

    #[inline]
    fn elapsed_ms(&self) -> f64 {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.t0.elapsed().as_secs_f64() * 1000.0
        }
        #[cfg(target_arch = "wasm32")]
        {
            let now = web_sys::window()
                .and_then(|w| w.performance())
                .map(|p| p.now())
                .unwrap_or(self.t0_ms);
            (now - self.t0_ms).max(0.0)
        }
    }
}

// ============================================================================
// 1. Theme Palette (`Paper White` & `Dark Oscilloscope`)
// ============================================================================

/// Theme-aware vector palette for the 3-track secondary-structure & concordance widget.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SsRibbonTheme {
    pub canvas_bg: egui::Color32,
    pub header_bg: egui::Color32,
    pub border_stroke: egui::Color32,
    pub track_divider: egui::Color32,
    pub ruler_text: egui::Color32,
    pub header_text: egui::Color32,
    pub hover_col_fill: egui::Color32,
    pub selected_col_stroke: egui::Color32,
    pub selected_col_fill: egui::Color32,
    /// Track 1 & Track 2 alpha-helix warm coral (`#E05A47`).
    pub helix_coral: egui::Color32,
    /// Shaded cylinder back-lobe stroke for Track 1 sinusoidal helices.
    pub helix_back_lobe: egui::Color32,
    /// Track 1 & Track 2 beta-sheet directional teal (`#2A9D8F`).
    pub sheet_teal: egui::Color32,
    /// Track 1 polyproline II diamond violet (`#7B61FF`).
    pub ppii_violet: egui::Color32,
    /// Track 1 coil baseline & Track 2 zero-axis stroke.
    pub coil_baseline: egui::Color32,
    /// Track 2 hollow dash stroke for unassigned NMR residues.
    pub nmr_unassigned_dash: egui::Color32,
    /// Track 3 `ConcordantOrdered` (`#16A34A`).
    pub concordant_ordered: egui::Color32,
    /// Track 3 `ConcordantDisordered` (`#64748B`).
    pub concordant_disordered: egui::Color32,
    /// Track 3 `ConditionalFoldingOrCrystalArtifact` (`#D97706`).
    pub conditional_folding: egui::Color32,
    /// Diagonal hatch stroke overlaid on `ConditionalFoldingOrCrystalArtifact`.
    pub conditional_hatch_stroke: egui::Color32,
    /// Track 3 `DynamicHingeOrAltState` (`#DC2626`).
    pub dynamic_hinge: egui::Color32,
    /// High-contrast inner highlight border for `DynamicHingeOrAltState` double-border blocks.
    pub dynamic_hinge_inner_border: egui::Color32,
    /// Outer border for `DynamicHingeOrAltState` double-border blocks.
    pub dynamic_hinge_outer_border: egui::Color32,
    /// Track 3 `UnassignedNmr` muted translucent outline & faint fill.
    pub unassigned_outline: egui::Color32,
    pub unassigned_fill: egui::Color32,
}

impl SsRibbonTheme {
    /// Construct the `Paper White` (`paper_white == true`) or `Dark Oscilloscope`
    /// (`paper_white == false`) theme while preserving exact spec semantic hex colors.
    pub const fn for_mode(paper_white: bool) -> Self {
        let helix_coral = egui::Color32::from_rgb(0xE0, 0x5A, 0x47);
        let sheet_teal = egui::Color32::from_rgb(0x2A, 0x9D, 0x8F);
        let ppii_violet = egui::Color32::from_rgb(0x7B, 0x61, 0xFF);
        let concordant_ordered = egui::Color32::from_rgb(0x16, 0xA3, 0x4A);
        let concordant_disordered = egui::Color32::from_rgb(0x64, 0x74, 0x8B);
        let conditional_folding = egui::Color32::from_rgb(0xD9, 0x77, 0x06);
        let dynamic_hinge = egui::Color32::from_rgb(0xDC, 0x26, 0x26);

        if paper_white {
            Self {
                canvas_bg: egui::Color32::from_rgb(248, 250, 252),
                header_bg: egui::Color32::from_rgb(238, 242, 247),
                border_stroke: egui::Color32::from_rgb(203, 213, 225),
                track_divider: egui::Color32::from_rgb(220, 228, 238),
                ruler_text: egui::Color32::from_rgb(71, 85, 105),
                header_text: egui::Color32::from_rgb(30, 41, 59),
                hover_col_fill: egui::Color32::from_rgba_premultiplied(14, 116, 144, 18),
                selected_col_stroke: egui::Color32::from_rgb(2, 132, 199),
                selected_col_fill: egui::Color32::from_rgba_premultiplied(2, 132, 199, 24),
                helix_coral,
                helix_back_lobe: egui::Color32::from_rgb(168, 56, 42),
                sheet_teal,
                ppii_violet,
                coil_baseline: egui::Color32::from_rgb(100, 116, 139),
                nmr_unassigned_dash: egui::Color32::from_rgb(148, 163, 184),
                concordant_ordered,
                concordant_disordered,
                conditional_folding,
                conditional_hatch_stroke: egui::Color32::from_rgb(255, 247, 237),
                dynamic_hinge,
                dynamic_hinge_inner_border: egui::Color32::from_rgb(254, 226, 226),
                dynamic_hinge_outer_border: egui::Color32::from_rgb(127, 29, 29),
                unassigned_outline: egui::Color32::from_rgba_premultiplied(100, 116, 139, 120),
                unassigned_fill: egui::Color32::from_rgba_premultiplied(148, 163, 184, 22),
            }
        } else {
            Self {
                canvas_bg: egui::Color32::from_rgb(15, 20, 29),
                header_bg: egui::Color32::from_rgb(22, 29, 41),
                border_stroke: egui::Color32::from_rgb(46, 58, 78),
                track_divider: egui::Color32::from_rgb(34, 44, 61),
                ruler_text: egui::Color32::from_rgb(148, 163, 184),
                header_text: egui::Color32::from_rgb(226, 232, 240),
                hover_col_fill: egui::Color32::from_rgba_premultiplied(56, 189, 248, 22),
                selected_col_stroke: egui::Color32::from_rgb(103, 232, 249),
                selected_col_fill: egui::Color32::from_rgba_premultiplied(56, 189, 248, 26),
                helix_coral,
                helix_back_lobe: egui::Color32::from_rgb(152, 52, 40),
                sheet_teal,
                ppii_violet,
                coil_baseline: egui::Color32::from_rgb(118, 134, 158),
                nmr_unassigned_dash: egui::Color32::from_rgb(92, 108, 132),
                concordant_ordered,
                concordant_disordered,
                conditional_folding,
                conditional_hatch_stroke: egui::Color32::from_rgb(255, 245, 225),
                dynamic_hinge,
                dynamic_hinge_inner_border: egui::Color32::from_rgb(254, 202, 202),
                dynamic_hinge_outer_border: egui::Color32::from_rgb(252, 165, 165),
                unassigned_outline: egui::Color32::from_rgba_premultiplied(100, 116, 139, 115),
                unassigned_fill: egui::Color32::from_rgba_premultiplied(71, 85, 105, 24),
            }
        }
    }
}

// ============================================================================
// 2. Berjanskii–Wishart `RCI-S^2` & Fowler–Williamson Concordance Evaluator
// ============================================================================

/// Summary metrics for the `AF vs NMR` rigidity concordance strip (`docs/specs/21-alphafold-integrative-nmr.md` §1.3).
#[derive(Clone, Debug, Default)]
pub struct ConcordanceTrackSummary {
    pub classes: Vec<ConcordanceTrackClass>,
    pub rci_s2: Vec<Option<f64>>,
    pub effective_s2: Vec<Option<f64>>,
    pub spearman_rho: Option<f64>,
    pub rigidity_rmsd: Option<f64>,
    pub n_evaluated: usize,
    pub class_counts: [usize; 5],
}

/// Compute per-residue Berjanskii & Wishart (*J. Am. Chem. Soc.* 2005, 127, 14970–14971)
/// chemical-shift order parameters `S^2_RCI` in `[0.0, 1.0]` directly from an offset-, pH-, and
/// temperature-corrected [`PropensityReport`].
///
/// Residues with fewer than `2` observed backbone shifts (`n_observed_nuclei < 2`) emit `None` so
/// downstream concordance classification marks them `UnassignedNmr`.
pub fn compute_rci_s2_from_propensity(report: &PropensityReport) -> Vec<Option<f64>> {
    let n = report.residues.len();
    if n == 0 {
        return Vec::new();
    }

    // Per-residue weighted unsigned secondary shift activity before 3-point RCI smoothing.
    let mut raw_act = vec![None::<f64>; n];
    for (i, r) in report.residues.iter().enumerate() {
        if !r.has_data {
            continue;
        }
        let mut weighted_sum = 0.0f64;
        let mut weight_tot = 0.0f64;
        for (atom, &dd) in &r.secondary_shifts {
            if !dd.is_finite() {
                continue;
            }
            let (scale, w) = match atom.as_str() {
                "CA" => (14.8, 1.30),
                "CB" => (13.2, 1.00),
                "C" => (17.2, 1.10),
                "HA" => (88.0, 1.15),
                "H" => (56.0, 0.70),
                "N" => (10.5, 0.65),
                _ => continue,
            };
            weighted_sum += w * (scale * dd.abs());
            weight_tot += w;
        }
        if weight_tot > 0.0 {
            raw_act[i] = Some(weighted_sum / weight_tot);
        }
    }

    let mut out = vec![None; n];
    for (i, out_slot) in out.iter_mut().enumerate() {
        if report.residues[i].n_observed_nuclei < 2 {
            continue;
        }
        let mut acc = 0.0f64;
        let mut wsum = 0.0f64;
        for (off, kw) in [(-1isize, 0.25f64), (0, 0.50), (1, 0.25)] {
            let j = i as isize + off;
            if j >= 0 && (j as usize) < n {
                if let Some(a) = raw_act[j as usize] {
                    acc += kw * a;
                    wsum += kw;
                }
            }
        }
        if wsum <= 0.0 {
            continue;
        }
        let smoothed_act = (acc / wsum).max(2.6);
        let rci = (1.0 / smoothed_act).clamp(0.015, 0.55);
        // Berjanskii & Wishart (JACS 2005, Eq. 4): S^2_RCI = 1 - 0.4 * ln(1 + 17.3 * RCI)
        let s2_rci = (1.0 - 0.40 * (1.0 + 17.3 * rci).ln()).clamp(0.0, 1.0);
        *out_slot = Some(s2_rci);
    }
    out
}

/// Convert a validated computational [`StructureEnsemble3d`] into [`nmr_core::af_concordance::AfStructureModel`]
/// so `nmr_core::af_concordance::evaluate_af_nmr_concordance` is the single concordance classifier (`21.W1:P1-5`).
///
/// Fails closed (`None`) whenever `ens.require_predicted_model(treat_bfactors_as_plddt)` returns `Err`.
pub fn ensemble_to_af_structure_model(
    ens: &StructureEnsemble3d,
    treat_bfactors_as_plddt: bool,
) -> Option<nmr_core::af_concordance::AfStructureModel> {
    if ens
        .require_predicted_model(treat_bfactors_as_plddt)
        .is_err()
    {
        return None;
    }
    let m0 = ens.models.first()?;
    let c0 = m0.chains.first()?;
    if c0.residues.is_empty() {
        return None;
    }

    let mut raw_plddts: Vec<f64> = if let Some(ref pv) = ens.per_residue_plddt {
        if pv.len() != c0.residues.len() {
            return None;
        }
        pv.clone()
    } else if treat_bfactors_as_plddt {
        let mut bs = Vec::with_capacity(c0.residues.len());
        for res in &c0.residues {
            let b_opt = res
                .atoms
                .iter()
                .find(|a| a.atom_name == "CA")
                .map(|a| a.b_factor as f64)
                .or_else(|| {
                    let mut sum = 0.0f64;
                    let mut cnt = 0usize;
                    for a in &res.atoms {
                        if matches!(a.atom_name.as_str(), "N" | "CA" | "C" | "O") {
                            sum += a.b_factor as f64;
                            cnt += 1;
                        }
                    }
                    (cnt > 0).then(|| sum / (cnt as f64))
                })?;
            bs.push(b_opt);
        }
        let max_b = bs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut sorted = bs.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let med_b = sorted[sorted.len() / 2];
        if max_b <= 1.0 && med_b > 0.15 {
            for v in &mut bs {
                *v *= 100.0;
            }
        }
        bs
    } else {
        return None;
    };

    let mut residues = Vec::with_capacity(c0.residues.len());
    let mut kept_plddts = Vec::with_capacity(c0.residues.len());
    let chain_str = c0.chain_id.to_string();
    for (r_pos, (res, pv)) in c0.residues.iter().zip(raw_plddts.drain(..)).enumerate() {
        let mut atoms_map = BTreeMap::new();
        for a in &res.atoms {
            atoms_map.insert(
                a.atom_name.clone(),
                [a.pos[0] as f64, a.pos[1] as f64, a.pos[2] as f64],
            );
        }
        let Some(&n) = atoms_map.get("N") else {
            continue;
        };
        let Some(&ca) = atoms_map.get("CA") else {
            continue;
        };
        let Some(&c) = atoms_map.get("C") else {
            continue;
        };
        let Some(&o) = atoms_map.get("O") else {
            continue;
        };
        let cb = atoms_map.get("CB").copied();
        let explicit_h = atoms_map.get("H").or_else(|| atoms_map.get("HN")).copied();
        residues.push(nmr_core::af_concordance::AfResidueRecord {
            seq_index: residues.len(),
            seq_id: res.seq_id,
            label_seq_id: Some((r_pos + 1) as i32),
            chain_id: chain_str.clone(),
            res_name: res.res_name.clone(),
            n,
            ca,
            c,
            o,
            cb,
            explicit_h,
            atoms: atoms_map,
            plddt: Some(pv),
        });
        kept_plddts.push(pv);
    }
    if residues.is_empty() {
        return None;
    }

    let plddt_source = match ens.provenance {
        crate::mol3d_viewer::CoordinateProvenance::ModelArchiveQaMetric => {
            nmr_core::af_concordance::PlddtSource::MaQaMetricLocal
        }
        crate::mol3d_viewer::CoordinateProvenance::ComputationalBFactor => {
            nmr_core::af_concordance::PlddtSource::BFactorColumn
        }
        _ => nmr_core::af_concordance::PlddtSource::ForcedExperimentalBFactor,
    };

    Some(nmr_core::af_concordance::AfStructureModel {
        entry_id: Some(ens.title.clone()),
        detected_pdb_id: None,
        detected_uniprot_id: None,
        exptl_method: None,
        chain_id: chain_str,
        provenance: nmr_core::af_concordance::CoordinateProvenance::ComputationalModel,
        plddt_source,
        plddt_per_residue: Some(kept_plddts),
        residues,
        diagnostics: Vec::new(),
    })
}

/// Align an [`nmr_core::af_concordance::AfConcordanceReport`] onto `0..seq_len` by `seq_id`
/// (handling cloning tags, N-Met cleavage, or missing loops cleanly without length-equality assumptions).
pub fn map_core_concordance_report_to_summary(
    seq_len: usize,
    nmr_seq_ids: &[i32],
    rep: &nmr_core::af_concordance::AfConcordanceReport,
    propensity: Option<&PropensityReport>,
    exp_s2: Option<&BTreeMap<usize, f64>>,
) -> ConcordanceTrackSummary {
    let rci_s2 = propensity
        .map(compute_rci_s2_from_propensity)
        .unwrap_or_else(|| vec![None; seq_len]);
    let mut summary = ConcordanceTrackSummary {
        classes: vec![ConcordanceTrackClass::UnassignedNmr; seq_len],
        rci_s2,
        effective_s2: vec![None; seq_len],
        spearman_rho: None,
        rigidity_rmsd: None,
        n_evaluated: 0,
        class_counts: [0; 5],
    };

    let mut by_seq_id = BTreeMap::new();
    for rec in &rep.residues {
        by_seq_id.entry(rec.seq_id).or_insert(rec);
    }

    for k in 0..seq_len {
        let fallback_s2 = exp_s2
            .and_then(|m| m.get(&k).copied())
            .filter(|v| v.is_finite())
            .map(|v| v.clamp(0.0, 1.0))
            .or_else(|| summary.rci_s2.get(k).copied().flatten());

        let target_sid = nmr_seq_ids.get(k).copied().unwrap_or((k + 1) as i32);
        let matched = by_seq_id.get(&target_sid).copied().or_else(|| {
            (nmr_seq_ids.is_empty() && rep.residues.len() == seq_len)
                .then(|| rep.residues.get(k))
                .flatten()
        });

        let cls = if let Some(rec) = matched {
            summary.effective_s2[k] = rec.s2_exp.or(fallback_s2);
            if rec.plddt.is_none() && summary.effective_s2[k].is_some() {
                ConcordanceTrackClass::Indeterminate
            } else {
                ConcordanceTrackClass::from(rec.concordance_class)
            }
        } else {
            summary.effective_s2[k] = fallback_s2;
            if fallback_s2.is_some() {
                ConcordanceTrackClass::Indeterminate
            } else {
                ConcordanceTrackClass::UnassignedNmr
            }
        };

        summary.classes[k] = cls;
        let bucket = match cls {
            ConcordanceTrackClass::ConcordantOrdered => 0,
            ConcordanceTrackClass::ConcordantDisordered => 1,
            ConcordanceTrackClass::ConditionalFoldingOrCrystalArtifact => 2,
            ConcordanceTrackClass::DynamicHingeOrAltState => 3,
            ConcordanceTrackClass::Indeterminate | ConcordanceTrackClass::UnassignedNmr => 4,
        };
        summary.class_counts[bucket] += 1;
    }

    match rep.summary {
        nmr_core::af_concordance::ConcordanceSummary::Evaluated {
            n_eval,
            spearman_rho_s,
            rmsd_rigidity,
            ..
        } => {
            summary.n_evaluated = n_eval;
            summary.spearman_rho = Some(spearman_rho_s);
            summary.rigidity_rmsd = Some(rmsd_rigidity);
        }
        nmr_core::af_concordance::ConcordanceSummary::InsufficientCoverage { n_eval, .. } => {
            summary.n_evaluated = n_eval;
        }
    }
    summary
}

/// Fail-closed neutral Track 3 summary when no validated `pLDDT` model is loaded (`21.W1:P0-2`, `P1-5`).
pub fn neutral_concordance_summary_without_predicted_model(
    seq_len: usize,
    has_3d: bool,
    propensity: Option<&PropensityReport>,
    exp_s2: Option<&BTreeMap<usize, f64>>,
) -> ConcordanceTrackSummary {
    let rci_s2 = propensity
        .map(compute_rci_s2_from_propensity)
        .unwrap_or_else(|| vec![None; seq_len]);
    let mut summary = ConcordanceTrackSummary {
        classes: vec![ConcordanceTrackClass::UnassignedNmr; seq_len],
        rci_s2,
        effective_s2: vec![None; seq_len],
        spearman_rho: None,
        rigidity_rmsd: None,
        n_evaluated: 0,
        class_counts: [0; 5],
    };
    for k in 0..seq_len {
        let s2_opt = exp_s2
            .and_then(|m| m.get(&k).copied())
            .filter(|v| v.is_finite())
            .map(|v| v.clamp(0.0, 1.0))
            .or_else(|| summary.rci_s2.get(k).copied().flatten());
        summary.effective_s2[k] = s2_opt;
        summary.classes[k] = if has_3d && s2_opt.is_some() {
            ConcordanceTrackClass::Indeterminate
        } else {
            ConcordanceTrackClass::UnassignedNmr
        };
    }
    summary.class_counts[4] = seq_len;
    summary
}

// ============================================================================
// 3. Stateful Incremental Cache & Vector Ribbon Telemetry
// ============================================================================

/// Counts of geometric primitives emitted across the 3 stacked vector tracks during the most
/// recent frame (exposed for deterministic headless unit tests).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SsRibbonPrimCounts {
    pub track1_helix_wave_points: usize,
    pub track1_sheet_arrows: usize,
    pub track1_ppii_diamonds: usize,
    pub track1_coil_segments: usize,
    pub track2_upward_helix_bars: usize,
    pub track2_downward_strand_bars: usize,
    pub track2_unassigned_dashes: usize,
    pub track3_concordance_blocks: usize,
    pub track3_hatch_strokes: usize,
    pub track3_double_borders: usize,
}

impl SsRibbonPrimCounts {
    #[inline]
    pub const fn track1_total(&self) -> usize {
        self.track1_helix_wave_points
            + self.track1_sheet_arrows
            + self.track1_ppii_diamonds
            + self.track1_coil_segments
    }

    #[inline]
    pub const fn track2_total(&self) -> usize {
        self.track2_upward_helix_bars
            + self.track2_downward_strand_bars
            + self.track2_unassigned_dashes
    }

    #[inline]
    pub const fn track3_total(&self) -> usize {
        self.track3_concordance_blocks
    }
}

fn fingerprint_reference_shifts(ref_data: Option<&Reference>, cfg: &PropensityConfig) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0100_0000_01b3;
    let mut h = FNV_OFFSET;
    let mut mix = |v: u64| {
        h ^= v;
        h = h.wrapping_mul(FNV_PRIME);
    };
    let Some(r) = ref_data else {
        mix(0);
        return h;
    };
    mix(r.sequence.len() as u64);
    for aa in &r.sequence {
        for &b in aa.as_bytes() {
            mix(b as u64);
        }
    }
    mix(cfg.temperature_kelvin.to_bits());
    mix(cfg.sample_ph.unwrap_or(-1.0).to_bits());
    mix(u64::from(cfg.auto_offset_correction));
    mix(r.shifts.len() as u64);
    for (&(idx, ref atom), &ppm) in &r.shifts {
        mix(idx as u64);
        for &b in atom.as_bytes() {
            mix(b as u64);
        }
        mix(ppm.to_bits());
    }
    h
}

/// Persistent state and `< 1.0 ms` incremental cache for the 3-track secondary-structure and
/// concordance ribbon widget.
#[derive(Clone, Debug)]
pub struct SsRibbonState {
    pub selected_residue: Option<usize>,
    pub col_width_px: f32,
    pub scroll_offset_px: f32,
    /// Track 1 (`3D SS`) 4-state secondary structure per `0`-based sequence index.
    pub ss3d_track: Vec<SsState4>,
    /// Optional per-residue AlphaFold `pLDDT` (`[0, 100]`) aligned to `0`-based sequence index.
    pub plddt_track: Vec<Option<f64>>,
    /// Cached Track 2 (`NMR SS`) Marsh `SSP` & Camilloni 4-state `delta2D` report.
    pub cached_propensity: Option<PropensityReport>,
    /// Cached NMR [`Reference`] shifts used to evaluate `nmr_core::af_concordance::evaluate_af_nmr_concordance`.
    pub cached_reference: Option<Reference>,
    /// Optional external override for Track 3 (`AF vs NMR`, e.g. from `nmr-core::af_concordance`).
    pub concordance_override: Option<Vec<ConcordanceTrackClass>>,
    /// `true` when experimental Lipari–Szabo relaxation `S²` values are loaded; `false` when Track 3
    /// uses chemical-shift-derived `"RCI-S² (shift-derived)"`.
    pub has_experimental_s2: bool,
    /// Evaluated Track 3 concordance summary (live or override-backed).
    pub concordance_summary: ConcordanceTrackSummary,
    /// Wall-clock duration (`ms`) of the most recent Track 2 `SSP` incremental update.
    pub last_nmr_update_ms: f64,
    /// Per-residue clickable column rectangles (`0..L`) from the most recent frame.
    pub last_col_rects: Vec<egui::Rect>,
    /// Bounding rectangles of `[Track 1, Track 2, Track 3]` from the most recent frame.
    pub last_track_rects: [Option<egui::Rect>; 3],
    /// Outer widget canvas rectangle from the most recent frame.
    pub last_widget_rect: Option<egui::Rect>,
    /// Geometric primitive counters from the most recent frame.
    pub last_prim_counts: SsRibbonPrimCounts,
    last_shift_fingerprint: u64,
}

impl Default for SsRibbonState {
    fn default() -> Self {
        Self::new()
    }
}

impl SsRibbonState {
    pub fn new() -> Self {
        Self {
            selected_residue: None,
            col_width_px: DEFAULT_SS_COL_WIDTH_PX,
            scroll_offset_px: 0.0,
            ss3d_track: Vec::new(),
            plddt_track: Vec::new(),
            cached_propensity: None,
            cached_reference: None,
            concordance_override: None,
            has_experimental_s2: false,
            concordance_summary: ConcordanceTrackSummary::default(),
            last_nmr_update_ms: 0.0,
            last_col_rects: Vec::new(),
            last_track_rects: [None; 3],
            last_widget_rect: None,
            last_prim_counts: SsRibbonPrimCounts::default(),
            last_shift_fingerprint: 0,
        }
    }

    /// Explicit provenance label for the solution NMR rigidity source (`"Experimental S²"` vs
    /// `"RCI-S² (shift-derived)"`).
    #[inline]
    pub const fn rigidity_source_label(&self) -> &'static str {
        if self.has_experimental_s2 {
            "Experimental S²"
        } else {
            "RCI-S² (shift-derived)"
        }
    }

    /// Multiplicatively zoom residue column width (`factor > 1.0` widens columns, `< 1.0` compacts).
    pub fn zoom_by(&mut self, factor: f32) {
        if factor.is_finite() && factor > 0.0 {
            self.col_width_px =
                (self.col_width_px * factor).clamp(MIN_SS_COL_WIDTH_PX, MAX_SS_COL_WIDTH_PX);
        }
    }

    /// Reset horizontal zoom to [`DEFAULT_SS_COL_WIDTH_PX`].
    pub fn reset_zoom(&mut self) {
        self.col_width_px = DEFAULT_SS_COL_WIDTH_PX;
    }

    /// Directly set Track 1 3D secondary-structure states (`[0..L]`).
    pub fn set_ss3d_slice(&mut self, ss: &[SsState4]) {
        self.ss3d_track.clear();
        self.ss3d_track.extend_from_slice(ss);
    }

    /// Directly set Track 1 3D secondary-structure states from `nmr_core::kabsch_sander_ss::SecondaryStructure4`.
    pub fn set_core_ss3d_slice(&mut self, ss: &[nmr_core::kabsch_sander_ss::SecondaryStructure4]) {
        self.ss3d_track.clear();
        self.ss3d_track
            .extend(ss.iter().copied().map(SsState4::from));
    }

    /// Directly set per-residue AlphaFold `pLDDT` (`[0, 100]`) values.
    pub fn set_plddt_slice(&mut self, plddt: &[Option<f64>]) {
        self.plddt_track.clear();
        self.plddt_track.extend_from_slice(plddt);
    }

    /// Supply an explicit Track 3 concordance vector from `nmr_core::af_concordance::AfNmrConcordanceClass`.
    pub fn set_core_concordance_override(
        &mut self,
        classes: &[nmr_core::af_concordance::AfNmrConcordanceClass],
    ) {
        let mapped: Vec<ConcordanceTrackClass> = classes
            .iter()
            .copied()
            .map(ConcordanceTrackClass::from)
            .collect();
        self.set_concordance_override(&mapped);
    }

    /// Align an [`nmr_core::af_concordance::AfConcordanceReport`] onto `0..seq_len` by `seq_id`
    /// and store both the override vector and full [`ConcordanceTrackSummary`].
    pub fn set_core_concordance_report_aligned(
        &mut self,
        seq_len: usize,
        nmr_seq_ids: &[i32],
        rep: &nmr_core::af_concordance::AfConcordanceReport,
        exp_s2: Option<&BTreeMap<usize, f64>>,
    ) {
        let summary = map_core_concordance_report_to_summary(
            seq_len,
            nmr_seq_ids,
            rep,
            self.cached_propensity.as_ref(),
            exp_s2,
        );
        self.concordance_override = Some(summary.classes.clone());
        self.concordance_summary = summary;
    }

    /// Supply an explicit Track 3 concordance vector (e.g. from `nmr-core::af_concordance`).
    pub fn set_concordance_override(&mut self, classes: &[ConcordanceTrackClass]) {
        let mut counts = [0usize; 5];
        for &c in classes {
            let b = match c {
                ConcordanceTrackClass::ConcordantOrdered => 0,
                ConcordanceTrackClass::ConcordantDisordered => 1,
                ConcordanceTrackClass::ConditionalFoldingOrCrystalArtifact => 2,
                ConcordanceTrackClass::DynamicHingeOrAltState => 3,
                ConcordanceTrackClass::Indeterminate | ConcordanceTrackClass::UnassignedNmr => 4,
            };
            counts[b] += 1;
        }
        self.concordance_override = Some(classes.to_vec());
        self.concordance_summary.classes = classes.to_vec();
        self.concordance_summary.class_counts = counts;
    }

    /// Clear any external Track 3 override, returning to automatic live evaluation.
    pub fn clear_concordance_override(&mut self) {
        self.concordance_override = None;
    }

    /// Incrementally recompute Track 2 (`NMR SS` via `compute_idp_propensity`) in `< 1.0 ms`
    /// whenever `ref_data` or `cfg` changes. Returns `true` if a recomputation occurred.
    pub fn sync_nmr_shifts(
        &mut self,
        ref_data: Option<&Reference>,
        cfg: &PropensityConfig,
    ) -> bool {
        let fp = fingerprint_reference_shifts(ref_data, cfg);
        if fp == self.last_shift_fingerprint {
            return false;
        }
        self.last_shift_fingerprint = fp;
        let timer = FastTimer::start();
        self.cached_reference = ref_data.filter(|r| !r.sequence.is_empty()).cloned();
        self.cached_propensity = self
            .cached_reference
            .as_ref()
            .map(|r| compute_idp_propensity(r, cfg));
        self.last_nmr_update_ms = timer.elapsed_ms();
        true
    }

    /// Synchronize all 3 tracks from an optional loaded [`StructureEnsemble3d`], NMR [`Reference`]
    /// shifts (or pre-computed [`PropensityReport`]), and optional Lipari–Szabo `S^2` map.
    pub fn sync_from_ensemble_and_nmr(
        &mut self,
        seq_len: usize,
        ensemble: Option<&StructureEnsemble3d>,
        ref_data: Option<&Reference>,
        prop_cfg: &PropensityConfig,
        exp_s2: Option<&BTreeMap<usize, f64>>,
    ) {
        let _ = self.sync_nmr_shifts(ref_data, prop_cfg);
        self.sync_3d_and_concordance(seq_len, ensemble, exp_s2);
    }

    /// Synchronize Track 1 (`3D SS`) and Track 3 (`AF vs NMR`) using an already-computed
    /// `cached_propensity` (avoiding duplicate `compute_idp_propensity` calls when the host app
    /// already maintains a [`PropensityReport`]).
    pub fn sync_with_cached_propensity(
        &mut self,
        seq_len: usize,
        ensemble: Option<&StructureEnsemble3d>,
        propensity: Option<&PropensityReport>,
        exp_s2: Option<&BTreeMap<usize, f64>>,
    ) {
        self.cached_propensity = propensity.cloned();
        self.sync_3d_and_concordance(seq_len, ensemble, exp_s2);
    }

    fn sync_3d_and_concordance(
        &mut self,
        seq_len: usize,
        ensemble: Option<&StructureEnsemble3d>,
        exp_s2: Option<&BTreeMap<usize, f64>>,
    ) {
        self.has_experimental_s2 = exp_s2.is_some_and(|m| !m.is_empty());
        if let Some(ens) = ensemble {
            let mut ss_vec = vec![SsState4::Coil; seq_len];
            let mut plddt_vec = vec![None::<f64>; seq_len];
            if let Some(c0) = ens.models.first().and_then(|m| m.chains.first()) {
                for (r_pos, res) in c0.residues.iter().enumerate() {
                    if let Some(k) = res.sequence_idx.filter(|&k| k < seq_len) {
                        ss_vec[k] = res.ss4;
                        if let Some(pv) = ens
                            .per_residue_plddt
                            .as_ref()
                            .and_then(|v| v.get(r_pos).copied())
                            .filter(|v| v.is_finite())
                        {
                            plddt_vec[k] = Some(pv);
                        }
                    }
                }
            }
            self.ss3d_track = ss_vec;
            self.plddt_track = plddt_vec;
        } else {
            self.ss3d_track.clear();
            self.plddt_track.clear();
        }

        if let Some(ref ov) = self.concordance_override {
            self.concordance_summary.classes = ov.clone();
            return;
        }

        let has_3d = !self.ss3d_track.is_empty();
        if let (Some(ens), Some(shifts)) = (ensemble, self.cached_reference.as_ref()) {
            if let Some(af_model) = ensemble_to_af_structure_model(ens, false) {
                let s2_entries: Option<Vec<nmr_core::af_concordance::ExperimentalS2Entry>> = exp_s2
                    .map(|m| {
                        m.iter()
                            .filter(|&(_, &v)| v.is_finite() && (0.0..=1.0).contains(&v))
                            .map(|(&k, &v)| nmr_core::af_concordance::ExperimentalS2Entry {
                                seq_id: shifts.seq_ids.get(k).copied().unwrap_or((k + 1) as i32),
                                res_name: shifts.sequence.get(k).cloned(),
                                s2: v,
                                sigma_s2: None,
                            })
                            .collect()
                    });
                let cfg = nmr_core::af_concordance::AfConcordanceConfig::default();
                if let Ok(rep) = nmr_core::af_concordance::evaluate_af_nmr_concordance(
                    &af_model,
                    shifts,
                    None,
                    s2_entries.as_deref().filter(|entries| !entries.is_empty()),
                    &cfg,
                ) {
                    self.concordance_summary = map_core_concordance_report_to_summary(
                        seq_len,
                        &shifts.seq_ids,
                        &rep,
                        self.cached_propensity.as_ref(),
                        exp_s2,
                    );
                    return;
                }
            }
        }

        self.concordance_summary = neutral_concordance_summary_without_predicted_model(
            seq_len,
            has_3d,
            self.cached_propensity.as_ref(),
            exp_s2,
        );
    }
}

/// Borrowed workspace view feeding [`SsRibbonPanel::show`].
pub struct SsRibbonWorkspaceInput<'a> {
    pub sequence: &'a [String],
    pub seq_ids: &'a [i32],
    pub paper_white: bool,
    pub has_3d_structure: bool,
}

/// Per-frame interaction output emitted by [`SsRibbonPanel::show`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SsRibbonOutput {
    /// `Some(residue_idx)` when the user clicks a residue column across any of the 3 tracks.
    pub clicked_residue: Option<usize>,
    /// Active selected `0`-based residue index after processing clicks.
    pub selected_residue: Option<usize>,
}

/// Apply a residue click from [`SsRibbonPanel`] to synchronize:
/// 1. `SyncCrosshair` (`1H`, `15N`, `13CA` frequencies for 2D `HSQC` & 3D sequential walk strips),
/// 2. `SequenceRibbonState` (`selected_residue` for the 2D covalent sidechain inspector), and
/// 3. `Mol3dViewerPanel::highlight_sequence_residue(Some(k), true)` (3D highlight & camera focus).
pub fn apply_ss_ribbon_click_sync(
    residue_idx: usize,
    ref_data: Option<&Reference>,
    crosshair: &mut SyncCrosshair,
    sidechain_state: &mut SequenceRibbonState,
    mol3d_panel: &mut Mol3dViewerPanel,
) {
    sidechain_state.selected_residue = residue_idx;
    mol3d_panel.highlight_sequence_residue(Some(residue_idx), true);
    if let Some(r) = ref_data {
        let h_opt = r
            .shifts
            .get(&(residue_idx, "H".to_string()))
            .or_else(|| r.shifts.get(&(residue_idx, "HN".to_string())))
            .copied()
            .filter(|v| v.is_finite());
        let n_opt = r
            .shifts
            .get(&(residue_idx, "N".to_string()))
            .copied()
            .filter(|v| v.is_finite());
        let ca_opt = r
            .shifts
            .get(&(residue_idx, "CA".to_string()))
            .copied()
            .filter(|v| v.is_finite());
        if h_opt.is_some() || n_opt.is_some() || ca_opt.is_some() {
            crosshair.snap_to(
                h_opt.unwrap_or(crosshair.h_ppm),
                n_opt.unwrap_or(crosshair.n_ppm),
                ca_opt.or(crosshair.c_ppm),
            );
        }
    }
}

// ============================================================================
// 4. Pure `egui::Painter` Three-Track Vector Renderer
// ============================================================================

fn paint_clipped_diagonal_hatch(
    painter: &egui::Painter,
    rect: egui::Rect,
    spacing_px: f32,
    stroke: egui::Stroke,
) -> usize {
    let inset = rect.shrink(1.0);
    if inset.width() <= 1.5 || inset.height() <= 1.5 {
        return 0;
    }
    let step = spacing_px.max(3.5);
    let h = inset.height();
    let mut count = 0usize;
    let mut d = -h;
    while d <= inset.width() {
        // Line equation: x - x0 = (y_bot - y) + d  =>  at y in [top, bottom]: x(y) = left + d + (bottom - y)
        let mut pts: [egui::Pos2; 2] = [egui::Pos2::ZERO; 2];
        let mut n_pts = 0usize;
        let mut push_unique = |p: egui::Pos2| {
            if n_pts < 2
                && (n_pts == 0 || (pts[0].x - p.x).abs() > 0.25 || (pts[0].y - p.y).abs() > 0.25)
            {
                pts[n_pts] = p;
                n_pts += 1;
            }
        };

        // Intersect with bottom (y = inset.bottom()): x = inset.left() + d
        let xb = inset.left() + d;
        if xb >= inset.left() - 0.01 && xb <= inset.right() + 0.01 {
            push_unique(egui::pos2(
                xb.clamp(inset.left(), inset.right()),
                inset.bottom(),
            ));
        }
        // Intersect with top (y = inset.top()): x = inset.left() + d + h
        let xt = inset.left() + d + h;
        if xt >= inset.left() - 0.01 && xt <= inset.right() + 0.01 {
            push_unique(egui::pos2(
                xt.clamp(inset.left(), inset.right()),
                inset.top(),
            ));
        }
        // Intersect with left (x = inset.left()): y = inset.bottom() + d  (valid when d in [-h, 0])
        let yl = inset.bottom() + d;
        if yl >= inset.top() - 0.01 && yl <= inset.bottom() + 0.01 {
            push_unique(egui::pos2(
                inset.left(),
                yl.clamp(inset.top(), inset.bottom()),
            ));
        }
        // Intersect with right (x = inset.right()): y = inset.bottom() + d - width
        let yr = inset.bottom() + d - inset.width();
        if yr >= inset.top() - 0.01 && yr <= inset.bottom() + 0.01 {
            push_unique(egui::pos2(
                inset.right(),
                yr.clamp(inset.top(), inset.bottom()),
            ));
        }

        if n_pts == 2 {
            painter.line_segment([pts[0], pts[1]], stroke);
            count += 1;
        }
        d += step;
    }
    count
}

/// Stateless renderer for the 3-track vector secondary-structure & concordance widget.
pub struct SsRibbonPanel;

impl SsRibbonPanel {
    /// Render all 3 stacked vector tracks (`3D SS`, `NMR SS`, `AF vs NMR`) synchronized with
    /// sequence indices `0..L`.
    pub fn show(
        ui: &mut egui::Ui,
        state: &mut SsRibbonState,
        input: &SsRibbonWorkspaceInput<'_>,
    ) -> SsRibbonOutput {
        state.last_col_rects.clear();
        state.last_track_rects = [None; 3];
        state.last_widget_rect = None;
        state.last_prim_counts = SsRibbonPrimCounts::default();

        let seq_len = input.sequence.len();
        if seq_len == 0 {
            state.selected_residue = None;
            return SsRibbonOutput::default();
        }
        if let Some(sel) = state.selected_residue {
            if sel >= seq_len {
                state.selected_residue = Some(seq_len - 1);
            }
        }

        let theme = SsRibbonTheme::for_mode(input.paper_white);
        let ruler_h = 15.0_f32;
        let t1_h = 16.0_f32;
        let t2_h = 17.0_f32;
        let t3_h = 15.0_f32;
        let total_h = ruler_h + t1_h + t2_h + t3_h + 4.0;
        let col_w = state
            .col_width_px
            .clamp(MIN_SS_COL_WIDTH_PX, MAX_SS_COL_WIDTH_PX);

        let mut clicked_residue: Option<usize> = None;

        let outer_frame = egui::Frame::new()
            .fill(theme.canvas_bg)
            .stroke(egui::Stroke::new(1.0, theme.border_stroke))
            .corner_radius(4.0)
            .inner_margin(egui::Margin::ZERO);

        let frame_resp = outer_frame.show(ui, |ui| {
            ui.spacing_mut().item_spacing = egui::Vec2::ZERO;
            ui.horizontal(|ui| {
                // 1. Pinned Left Header Gutter ("Seq", "3D SS", "NMR SS", "AF vs NMR") + zoom controls
                let (hdr_rect, hdr_resp) = ui.allocate_exact_size(
                    egui::vec2(SS_TRACK_HEADER_WIDTH_PX, total_h),
                    egui::Sense::click(),
                );
                let hp = ui.painter_at(hdr_rect);
                hp.rect_filled(hdr_rect, 3.0, theme.header_bg);
                hp.line_segment(
                    [
                        egui::pos2(hdr_rect.right(), hdr_rect.top()),
                        egui::pos2(hdr_rect.right(), hdr_rect.bottom()),
                    ],
                    egui::Stroke::new(1.0, theme.border_stroke),
                );

                let y0 = hdr_rect.top() + 2.0;
                let r0_mid = y0 + 0.5 * ruler_h;
                let t1_top = y0 + ruler_h;
                let t2_top = t1_top + t1_h;
                let t3_top = t2_top + t2_h;

                for div_y in [t1_top, t2_top, t3_top] {
                    hp.line_segment(
                        [
                            egui::pos2(hdr_rect.left() + 3.0, div_y),
                            egui::pos2(hdr_rect.right(), div_y),
                        ],
                        egui::Stroke::new(0.8, theme.track_divider),
                    );
                }

                let zoom_pct = ((col_w / DEFAULT_SS_COL_WIDTH_PX) * 100.0).round() as i32;
                hp.text(
                    egui::pos2(hdr_rect.left() + 5.0, r0_mid),
                    egui::Align2::LEFT_CENTER,
                    format!("Seq {zoom_pct}%"),
                    egui::FontId::monospace(11.0),
                    theme.ruler_text,
                );
                let track3_label = if state.has_experimental_s2 {
                    "AF vs S2"
                } else {
                    "RCI-S² (shift-derived)"
                };
                for (mid_y, lbl) in [
                    (t1_top + 0.5 * t1_h, "3D SS"),
                    (t2_top + 0.5 * t2_h, "NMR SS"),
                    (t3_top + 0.5 * t3_h, track3_label),
                ] {
                    hp.text(
                        egui::pos2(hdr_rect.left() + 5.0, mid_y),
                        egui::Align2::LEFT_CENTER,
                        lbl,
                        egui::FontId::monospace(11.0),
                        theme.header_text,
                    );
                }
                if hdr_resp.double_clicked() {
                    state.reset_zoom();
                }
                hdr_resp.on_hover_text(
                    "Stacked 3-Track Secondary Structure & Concordance Ribbon (Ctrl+Scroll to zoom horizontally, double-click header to reset zoom)",
                );

                // 2. Horizontally Scrollable & Zoomable 3-Track Vector Canvas (`0..L`)
                let avail_w = ui.available_width().max(80.0);
                let content_w = ((seq_len as f32) * col_w).max(avail_w);

                let scroll_out = egui::ScrollArea::horizontal()
                    .id_salt("ss_ribbon_3track_scroll")
                    .max_height(total_h + 6.0)
                    .horizontal_scroll_offset(state.scroll_offset_px)
                    .show(ui, |ui| {
                        let (canvas_rect, canvas_resp) = ui.allocate_exact_size(
                            egui::vec2(content_w, total_h),
                            egui::Sense::click(),
                        );

                        // Handle Ctrl/Cmd + Scroll horizontal column zoom
                        if canvas_resp.hovered() {
                            let (ctrl_or_cmd, scroll_y) = ui.input(|i| {
                                (
                                    i.modifiers.ctrl || i.modifiers.command,
                                    i.smooth_scroll_delta.y,
                                )
                            });
                            if ctrl_or_cmd && scroll_y.abs() > 0.1 {
                                let factor = (0.003 * scroll_y).exp();
                                state.zoom_by(factor);
                            }
                        }

                        let cy0 = canvas_rect.top() + 2.0;
                        let r_ruler = egui::Rect::from_min_max(
                            egui::pos2(canvas_rect.left(), cy0),
                            egui::pos2(canvas_rect.right(), cy0 + ruler_h),
                        );
                        let r_t1 = egui::Rect::from_min_max(
                            egui::pos2(canvas_rect.left(), cy0 + ruler_h),
                            egui::pos2(canvas_rect.right(), cy0 + ruler_h + t1_h),
                        );
                        let r_t2 = egui::Rect::from_min_max(
                            egui::pos2(canvas_rect.left(), r_t1.bottom()),
                            egui::pos2(canvas_rect.right(), r_t1.bottom() + t2_h),
                        );
                        let r_t3 = egui::Rect::from_min_max(
                            egui::pos2(canvas_rect.left(), r_t2.bottom()),
                            egui::pos2(canvas_rect.right(), r_t2.bottom() + t3_h),
                        );
                        state.last_track_rects = [Some(r_t1), Some(r_t2), Some(r_t3)];

                        state.last_col_rects.reserve(seq_len);
                        for k in 0..seq_len {
                            let x_l = canvas_rect.left() + (k as f32) * col_w;
                            let x_r = x_l + col_w;
                            state.last_col_rects.push(egui::Rect::from_min_max(
                                egui::pos2(x_l, canvas_rect.top()),
                                egui::pos2(x_r, canvas_rect.bottom()),
                            ));
                        }

                        let hover_k = canvas_resp
                            .hover_pos()
                            .filter(|p| canvas_rect.contains(*p))
                            .and_then(|p| {
                                let idx = ((p.x - canvas_rect.left()) / col_w).floor() as isize;
                                (idx >= 0 && (idx as usize) < seq_len).then_some(idx as usize)
                            });

                        if canvas_resp.clicked() {
                            if let Some(pos) = canvas_resp.interact_pointer_pos() {
                                let idx = ((pos.x - canvas_rect.left()) / col_w).floor() as isize;
                                if idx >= 0 && (idx as usize) < seq_len {
                                    let k = idx as usize;
                                    state.selected_residue = Some(k);
                                    clicked_residue = Some(k);
                                }
                            }
                        }

                        let painter = ui.painter_at(canvas_rect);

                        // Background hover / selection column highlight & track dividers
                        if let Some(hk) = hover_k {
                            painter.rect_filled(state.last_col_rects[hk], 0.0, theme.hover_col_fill);
                        }
                        if let Some(sk) = state.selected_residue.filter(|&k| k < seq_len) {
                            painter.rect_filled(
                                state.last_col_rects[sk],
                                0.0,
                                theme.selected_col_fill,
                            );
                        }
                        for div_y in [r_t1.top(), r_t2.top(), r_t3.top()] {
                            painter.line_segment(
                                [
                                    egui::pos2(canvas_rect.left(), div_y),
                                    egui::pos2(canvas_rect.right(), div_y),
                                ],
                                egui::Stroke::new(0.8, theme.track_divider),
                            );
                        }

                        // Sequence ruler row (1-letter amino acid code + periodic residue ticks)
                        let ruler_mid_y = r_ruler.center().y;
                        let vis_clip = ui.clip_rect();
                        for (k, three) in input.sequence.iter().enumerate() {
                            let cx = canvas_rect.left() + ((k as f32) + 0.5) * col_w;
                            let sid = input.seq_ids.get(k).copied().unwrap_or((k + 1) as i32);
                            if col_w >= 14.0 {
                                let ch = three_to_one(three);
                                let lbl = if col_w >= 30.0 && (k == 0 || sid % 5 == 0) {
                                    format!("{ch}{sid}")
                                } else {
                                    ch.to_string()
                                };
                                let est_half_w = (lbl.chars().count() as f32) * 3.8 + 2.0;
                                if cx - est_half_w >= vis_clip.left()
                                    && cx + est_half_w <= vis_clip.right()
                                {
                                    painter.text(
                                        egui::pos2(cx, ruler_mid_y),
                                        egui::Align2::CENTER_CENTER,
                                        lbl,
                                        egui::FontId::monospace(11.0),
                                        theme.ruler_text,
                                    );
                                }
                            } else if k == 0 || sid % 10 == 0 {
                                let lbl = format!("{sid}");
                                let est_half_w = (lbl.len() as f32) * 3.8 + 2.0;
                                if cx - est_half_w >= vis_clip.left()
                                    && cx + est_half_w <= vis_clip.right()
                                {
                                    painter.text(
                                        egui::pos2(cx, ruler_mid_y),
                                        egui::Align2::CENTER_CENTER,
                                        lbl,
                                        egui::FontId::monospace(11.0),
                                        theme.ruler_text,
                                    );
                                }
                            }
                        }

                        // ------------------------------------------------------------------
                        // Track 1: 3D Coordinate Kabsch-Sander Secondary Structure (`3D SS`)
                        // ------------------------------------------------------------------
                        let t1_mid_y = r_t1.center().y;
                        let has_3d = input.has_3d_structure || !state.ss3d_track.is_empty();
                        if has_3d {
                            let ss_at = |idx: usize| -> SsState4 {
                                state
                                    .ss3d_track
                                    .get(idx)
                                    .copied()
                                    .unwrap_or(SsState4::Coil)
                            };

                            // Contiguous run decomposition across `0..seq_len`
                            let mut run_start = 0usize;
                            while run_start < seq_len {
                                let kind = ss_at(run_start);
                                let mut run_end = run_start;
                                while run_end + 1 < seq_len && ss_at(run_end + 1) == kind {
                                    run_end += 1;
                                }
                                let x0 = canvas_rect.left() + (run_start as f32) * col_w + 1.0;
                                let x1 = canvas_rect.left() + ((run_end + 1) as f32) * col_w - 1.0;
                                let span_w = (x1 - x0).max(2.0);
                                let run_len = run_end - run_start + 1;

                                // Subtle connector ticks at boundaries to adjacent segments
                                if run_start > 0 {
                                    let bx = canvas_rect.left() + (run_start as f32) * col_w;
                                    painter.line_segment(
                                        [
                                            egui::pos2(bx - 1.2, t1_mid_y),
                                            egui::pos2(bx + 1.2, t1_mid_y),
                                        ],
                                        egui::Stroke::new(1.2, theme.coil_baseline),
                                    );
                                }

                                match kind {
                                    SsState4::Helix => {
                                        // Smooth multi-lobe sinusoidal / shaded cylinder wave (#E05A47)
                                        let amp = (0.37 * t1_h).clamp(3.5, 6.2);
                                        let lobes = (run_len as f32 * 1.15).max(1.5);
                                        let n_samples = (run_len * 8).clamp(12, 640);
                                        let mut front_pts = Vec::with_capacity(n_samples + 1);
                                        let mut back_pts = Vec::with_capacity(n_samples + 1);
                                        for s in 0..=n_samples {
                                            let u = (s as f32) / (n_samples as f32);
                                            let x = x0 + u * span_w;
                                            let phase = u * lobes * std::f32::consts::TAU;
                                            // Taper amplitude gently at N/C helix caps so wave joins coil cleanly
                                            let env = (u * std::f32::consts::PI)
                                                .sin()
                                                .powf(0.28)
                                                .clamp(0.35, 1.0);
                                            let dy = amp * env * phase.sin();
                                            front_pts.push(egui::pos2(x, t1_mid_y - dy));
                                            back_pts.push(egui::pos2(x, t1_mid_y + 0.72 * dy));
                                        }
                                        painter.add(egui::Shape::line(
                                            back_pts,
                                            egui::Stroke::new(1.5, theme.helix_back_lobe),
                                        ));
                                        state.last_prim_counts.track1_helix_wave_points +=
                                            front_pts.len();
                                        painter.add(egui::Shape::line(
                                            front_pts,
                                            egui::Stroke::new(2.4, theme.helix_coral),
                                        ));
                                    }
                                    SsState4::Sheet => {
                                        // Directional horizontal bar + crisp C-terminal filled polygon arrowhead (#2A9D8F)
                                        let half_shaft = (0.22 * t1_h).clamp(2.2, 3.8);
                                        let half_head = (0.42 * t1_h).clamp(4.4, 6.8);
                                        let head_w = col_w
                                            .min(span_w * 0.65)
                                            .clamp(4.5, 14.0)
                                            .min(span_w);
                                        let shaft_end_x = (x1 - head_w).max(x0);
                                        if shaft_end_x > x0 + 0.5 {
                                            painter.rect_filled(
                                                egui::Rect::from_min_max(
                                                    egui::pos2(x0, t1_mid_y - half_shaft),
                                                    egui::pos2(
                                                        shaft_end_x + 0.6,
                                                        t1_mid_y + half_shaft,
                                                    ),
                                                ),
                                                1.0,
                                                theme.sheet_teal,
                                            );
                                        }
                                        let arrow_pts = vec![
                                            egui::pos2(shaft_end_x, t1_mid_y - half_head),
                                            egui::pos2(x1, t1_mid_y),
                                            egui::pos2(shaft_end_x, t1_mid_y + half_head),
                                        ];
                                        painter.add(egui::Shape::convex_polygon(
                                            arrow_pts,
                                            theme.sheet_teal,
                                            egui::Stroke::new(0.8, theme.sheet_teal),
                                        ));
                                        state.last_prim_counts.track1_sheet_arrows += 1;
                                    }
                                    SsState4::Ppii => {
                                        // Baseline + compact geometric diamond ticks (#7B61FF)
                                        painter.line_segment(
                                            [egui::pos2(x0, t1_mid_y), egui::pos2(x1, t1_mid_y)],
                                            egui::Stroke::new(1.1, theme.coil_baseline),
                                        );
                                        let rx = (0.28 * col_w).clamp(2.6, 4.8);
                                        let ry = (0.34 * t1_h).clamp(3.2, 5.4);
                                        for k in run_start..=run_end {
                                            let cx =
                                                canvas_rect.left() + ((k as f32) + 0.5) * col_w;
                                            let diamond = vec![
                                                egui::pos2(cx, t1_mid_y - ry),
                                                egui::pos2(cx + rx, t1_mid_y),
                                                egui::pos2(cx, t1_mid_y + ry),
                                                egui::pos2(cx - rx, t1_mid_y),
                                            ];
                                            painter.add(egui::Shape::convex_polygon(
                                                diamond,
                                                theme.ppii_violet,
                                                egui::Stroke::new(0.9, theme.ppii_violet),
                                            ));
                                            state.last_prim_counts.track1_ppii_diamonds += 1;
                                        }
                                    }
                                    SsState4::Coil => {
                                        painter.line_segment(
                                            [egui::pos2(x0, t1_mid_y), egui::pos2(x1, t1_mid_y)],
                                            egui::Stroke::new(1.5, theme.coil_baseline),
                                        );
                                        state.last_prim_counts.track1_coil_segments += 1;
                                    }
                                }
                                run_start = run_end + 1;
                            }
                        } else {
                            // Muted placeholder baseline when no 3D coordinate model is loaded yet
                            let mut x = canvas_rect.left() + 2.0;
                            while x < canvas_rect.right() - 2.0 {
                                let x_end = (x + 5.0).min(canvas_rect.right() - 2.0);
                                painter.line_segment(
                                    [egui::pos2(x, t1_mid_y), egui::pos2(x_end, t1_mid_y)],
                                    egui::Stroke::new(1.0, theme.nmr_unassigned_dash),
                                );
                                x += 9.0;
                            }
                        }

                        // ------------------------------------------------------------------
                        // Track 2: Live Solution NMR `CSI` / 4-State `SSP` Propensity (`NMR SS`)
                        // ------------------------------------------------------------------
                        let t2_mid_y = r_t2.center().y;
                        let max_bar_h = (0.44 * t2_h).max(4.0);
                        painter.line_segment(
                            [
                                egui::pos2(canvas_rect.left(), t2_mid_y),
                                egui::pos2(canvas_rect.right(), t2_mid_y),
                            ],
                            egui::Stroke::new(0.75, theme.track_divider),
                        );

                        let bar_pad_x = (col_w * 0.14).clamp(1.0, 3.0);
                        for k in 0..seq_len {
                            let x_l = canvas_rect.left() + (k as f32) * col_w + bar_pad_x;
                            let x_r =
                                (canvas_rect.left() + ((k + 1) as f32) * col_w - bar_pad_x).max(x_l + 1.5);
                            let res_prop = state
                                .cached_propensity
                                .as_ref()
                                .and_then(|rep| rep.residues.get(k));

                            match res_prop.and_then(|rp| rp.has_data.then_some(rp.ssp_score).flatten()) {
                                Some(ssp) if ssp.is_finite() => {
                                    let s = ssp.clamp(-1.0, 1.0) as f32;
                                    if s >= 0.0 {
                                        let h = (s * max_bar_h).max(1.4);
                                        painter.rect_filled(
                                            egui::Rect::from_min_max(
                                                egui::pos2(x_l, t2_mid_y - h),
                                                egui::pos2(x_r, t2_mid_y),
                                            ),
                                            1.0,
                                            theme.helix_coral,
                                        );
                                        state.last_prim_counts.track2_upward_helix_bars += 1;
                                    } else {
                                        let h = ((-s) * max_bar_h).max(1.4);
                                        painter.rect_filled(
                                            egui::Rect::from_min_max(
                                                egui::pos2(x_l, t2_mid_y),
                                                egui::pos2(x_r, t2_mid_y + h),
                                            ),
                                            1.0,
                                            theme.sheet_teal,
                                        );
                                        state.last_prim_counts.track2_downward_strand_bars += 1;
                                    }
                                }
                                _ => {
                                    // Subtle hollow dash for unassigned residues
                                    let dash_rect = egui::Rect::from_min_max(
                                        egui::pos2(x_l, t2_mid_y - 1.3),
                                        egui::pos2(x_r, t2_mid_y + 1.3),
                                    );
                                    painter.rect_stroke(
                                        dash_rect,
                                        0.8,
                                        egui::Stroke::new(0.9, theme.nmr_unassigned_dash),
                                        egui::StrokeKind::Inside,
                                    );
                                    state.last_prim_counts.track2_unassigned_dashes += 1;
                                }
                            }
                        }

                        // ------------------------------------------------------------------
                        // Track 3: `RCI-S^2` / `S^2` vs `pLDDT` Concordance Strip (`AF vs NMR`)
                        // ------------------------------------------------------------------
                        let blk_top = r_t3.top() + 1.5;
                        let blk_bot = r_t3.bottom() - 1.5;
                        let blk_pad_x = (col_w * 0.08).clamp(0.6, 2.0);

                        for k in 0..seq_len {
                            let x_l = canvas_rect.left() + (k as f32) * col_w + blk_pad_x;
                            let x_r =
                                (canvas_rect.left() + ((k + 1) as f32) * col_w - blk_pad_x).max(x_l + 1.5);
                            let blk = egui::Rect::from_min_max(
                                egui::pos2(x_l, blk_top),
                                egui::pos2(x_r, blk_bot),
                            );
                            let cls = state
                                .concordance_summary
                                .classes
                                .get(k)
                                .copied()
                                .unwrap_or(ConcordanceTrackClass::UnassignedNmr);
                            state.last_prim_counts.track3_concordance_blocks += 1;

                            match cls {
                                ConcordanceTrackClass::ConcordantOrdered => {
                                    painter.rect_filled(blk, 1.5, theme.concordant_ordered);
                                }
                                ConcordanceTrackClass::ConcordantDisordered => {
                                    painter.rect_filled(blk, 1.5, theme.concordant_disordered);
                                }
                                ConcordanceTrackClass::ConditionalFoldingOrCrystalArtifact => {
                                    painter.rect_filled(blk, 1.5, theme.conditional_folding);
                                    let hatches = paint_clipped_diagonal_hatch(
                                        &painter,
                                        blk,
                                        4.2,
                                        egui::Stroke::new(1.1, theme.conditional_hatch_stroke),
                                    );
                                    state.last_prim_counts.track3_hatch_strokes += hatches;
                                }
                                ConcordanceTrackClass::DynamicHingeOrAltState => {
                                    painter.rect_filled(blk, 1.5, theme.dynamic_hinge);
                                    painter.rect_stroke(
                                        blk,
                                        1.5,
                                        egui::Stroke::new(1.1, theme.dynamic_hinge_outer_border),
                                        egui::StrokeKind::Inside,
                                    );
                                    let inner = blk.shrink(1.8);
                                    if inner.width() > 1.2 && inner.height() > 1.2 {
                                        painter.rect_stroke(
                                            inner,
                                            1.0,
                                            egui::Stroke::new(
                                                1.0,
                                                theme.dynamic_hinge_inner_border,
                                            ),
                                            egui::StrokeKind::Inside,
                                        );
                                    }
                                    state.last_prim_counts.track3_double_borders += 1;
                                }
                                ConcordanceTrackClass::Indeterminate => {
                                    painter.rect_filled(
                                        blk,
                                        1.5,
                                        egui::Color32::from_rgb(148, 163, 184),
                                    );
                                }
                                ConcordanceTrackClass::UnassignedNmr => {
                                    painter.rect_filled(blk, 1.5, theme.unassigned_fill);
                                    painter.rect_stroke(
                                        blk,
                                        1.5,
                                        egui::Stroke::new(0.9, theme.unassigned_outline),
                                        egui::StrokeKind::Inside,
                                    );
                                }
                            }
                        }

                        // Selected residue crisp outline spanning all 3 stacked tracks
                        if let Some(sk) = state.selected_residue.filter(|&k| k < seq_len) {
                            painter.rect_stroke(
                                state.last_col_rects[sk].shrink(0.5),
                                1.5,
                                egui::Stroke::new(1.6, theme.selected_col_stroke),
                                egui::StrokeKind::Inside,
                            );
                        }

                        // Rich zero-emoji hover tooltip
                        if let Some(hk) = hover_k {
                            let three = &input.sequence[hk];
                            let one = three_to_one(three);
                            let sid = input.seq_ids.get(hk).copied().unwrap_or((hk + 1) as i32);
                            let ss3d_lbl = if has_3d {
                                state
                                    .ss3d_track
                                    .get(hk)
                                    .copied()
                                    .unwrap_or(SsState4::Coil)
                                    .label()
                            } else {
                                "No 3D model loaded"
                            };
                            let plddt_opt = state.plddt_track.get(hk).copied().flatten();
                            let prop_res = state
                                .cached_propensity
                                .as_ref()
                                .and_then(|rep| rep.residues.get(hk));
                            let s2_opt = state
                                .concordance_summary
                                .effective_s2
                                .get(hk)
                                .copied()
                                .flatten();
                            let cls = state
                                .concordance_summary
                                .classes
                                .get(hk)
                                .copied()
                                .unwrap_or(ConcordanceTrackClass::UnassignedNmr);

                            canvas_resp.on_hover_ui_at_pointer(|ui| {
                                ui.label(
                                    egui::RichText::new(format!("{three} {sid} ({one})")).strong(),
                                );
                                match plddt_opt {
                                    Some(pv) => {
                                        ui.monospace(format!(
                                            "3D SS     : {ss3d_lbl} (pLDDT {pv:.1})"
                                        ));
                                    }
                                    None => {
                                        ui.monospace(format!("3D SS     : {ss3d_lbl}"));
                                    }
                                }
                                match prop_res.filter(|r| r.has_data) {
                                    Some(rp) => {
                                        let ssp_s = rp
                                            .ssp_score
                                            .map(|v| format!("{v:+.2}"))
                                            .unwrap_or_else(|| "n/a".to_string());
                                        ui.monospace(format!(
                                            "NMR SS    : SSP {ssp_s} (H {:.0}% E {:.0}% P {:.0}% C {:.0}%)",
                                            rp.p_alpha * 100.0,
                                            rp.p_beta * 100.0,
                                            rp.p_ppii * 100.0,
                                            rp.p_coil * 100.0
                                        ));
                                    }
                                    None => {
                                        ui.monospace("NMR SS    : Unassigned");
                                    }
                                }
                                let rig_label = state.rigidity_source_label();
                                match s2_opt {
                                    Some(s2) => {
                                        ui.monospace(format!(
                                            "{rig_label}: {} (S^2 {s2:.2})",
                                            cls.label()
                                        ));
                                    }
                                    None => {
                                        ui.monospace(format!("{rig_label}: {}", cls.label()));
                                    }
                                }
                                ui.label(
                                    egui::RichText::new(
                                        "Click to sync 2D HSQC, 3D strips, covalent graph & 3D structure",
                                    )
                                    .italics()
                                    .weak(),
                                );
                            });
                        }
                    });
                state.scroll_offset_px = scroll_out.state.clamped_offset().x;
            });
        });

        state.last_widget_rect = Some(frame_resp.response.rect);

        SsRibbonOutput {
            clicked_residue,
            selected_residue: state.selected_residue,
        }
    }
}
