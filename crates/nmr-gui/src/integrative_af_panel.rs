//! Zero-emoji Integrative Structural & AlphaFold Studio panel (`T21.6b`, `nmr-gui`).
//!
//! Unifies Phase 21 (`docs/specs/21-alphafold-integrative-nmr.md` §1, §5, §6, §7.1, §7.2) in a
//! calm, canvas-first dockable/windowed `egui` studio compiling cleanly on both native desktop and
//! `wasm32-unknown-unknown`:
//! 1. **1-Click `.cif`/`.pdb` + `PAE` JSON Loader & Concordance Summary (`T21.1`, `T21.3`)**:
//!    Global Fowler–Sljoka–Williamson Spearman `rho_s`, `RMSD_rig`, 4-class structural discordance
//!    counts, `HomologFirewall` (`<= 25%` sequence identity, `TM < 0.50`) `SHA-256` audit badge,
//!    and 1-click synchronization with [`SsRibbonState`] and [`Mol3dViewerPanel`] (`AlphaFoldPlddt`
//!    and `AfNmrConcordance` color/putty modes).
//! 2. **Interactive `N x N` `PAE` Domain Heatmap (`T21.5`)**:
//!    Green-to-white (`0..31.75 A`) vector `PAE` matrix overlaid with normalized-Laplacian spectral
//!    rigid-domain bounding boxes (`D1..DD`), Tolman `theta_d / theta_1` Generalized Degree of Order
//!    (`GDO`) motion badges, `F8` single-medium Saupe cone-degeneracy candor notices, and 1-click
//!    `(i, j)` cell or domain-pair highlighting in [`Mol3dViewerPanel`].
//! 3. **Co-Folding Restraint Preview & Multi-Pose Complex Ranker (`T21.6a`)**:
//!    Schema-honest `AlphaFold 3` JSON (`dialect: "alphafold3"`, `version: 2` + sidecar restraints)
//!    and `Boltz-2` YAML (`version: 1`, `1`-based `pocket` & `contact` constraints with
//!    Shrake–Rupley `rSASA >= 0.20` surface pocket filtering vs core allostery), paired with a
//!    sortable multi-pose complex NMR re-scoring table and `InconclusiveOrAllViolated` decoy guard.
//! 4. **Bayesian Excited-State Hypothesis Explorer (`T21.6b`)**:
//!    `BIC` / Kass–Raftery `2 ln BF` bar chart across `LocalUnfolding`, `XxxProCisTrans`,
//!    `RotamerOrHisTautomerFlip`, `Alternate3DConformer`, and `Null`, paired with an interactive
//!    `Delta_omega_obs` vs `Delta_omega_calc` parity scatter plot linked to Dynamics Studio and 3D
//!    ribbon highlighting.
//!
//! # Provenance & Clean-Room Statement
//! Original clean-room human-computer interaction and vector-plotting implementation grounded in:
//! - Jumper, J. et al. (*Nature* **2021**, 596, 583–589) & Varadi, M. et al. (*Nucleic Acids Res.*
//!   **2022**, 50, D439–D444; **2024**, 52, D368–D375) — `pLDDT` and `0..31.75 A` `PAE` schemas.
//! - Fowler, N. J., Sljoka, A. & Williamson, M. P. (*Nat. Commun.* **2020**, 11, 5982; *Structure*
//!   **2022**, 30, 925–933) & Alderson, T. R. et al. (*Proc. Natl. Acad. Sci. U.S.A.* **2023**,
//!   120, e2304302120) — solution NMR rigidity concordance and conditionally folding regions.
//! - Ng, A. Y., Jordan, M. I. & Weiss, Y. (*Adv. Neural Inf. Process. Syst.* **2001**, 14, 849–856)
//!   & Tolman, J. R., Al-Hashimi, H. M., Kay, L. E. & Prestegard, J. H. (*J. Am. Chem. Soc.*
//!   **2001**, 123, 1416–1424) — spectral `PAE` domain partitioning and per-domain Saupe `GDO`.
//! - Abramson, J. et al. (*Nature* **2024**, 630, 493–500); Wohlwend, J. et al. (*bioRxiv* **2024**,
//!   DOI: `10.1101/2024.11.19.624167`); Passaro, S. et al. (*bioRxiv* **2025**, DOI:
//!   `10.1101/2025.06.14.659707`); Shrake, A. & Rupley, J. A. (*J. Mol. Biol.* **1973**, 79,
//!   351–371) — `AF3` JSON, `Boltz-2` YAML, and `rSASA` pocket filtering.
//! - Vallurupalli, P., Hansen, D. F. & Kay, L. E. (*Proc. Natl. Acad. Sci. U.S.A.* **2008**, 105,
//!   11766–11771); Skrynnikov, N. R., Dahlquist, F. W. & Kay, L. E. (*J. Am. Chem. Soc.* **2002**,
//!   124, 12352–12360); Kass, R. E. & Raftery, A. E. (*J. Am. Stat. Assoc.* **1995**, 90, 773–795)
//!   — signed `CEST` and unsigned `CPMG` excited-state `BIC` / Bayes-factor hypothesis testing.

use std::fmt::Write as _;

use eframe::egui;
use nmr_core::af_concordance::{
    evaluate_af_nmr_concordance, parse_af_structure_model_str, parse_modelfree_s2_tsv_str,
    parse_pae_json_str, AfConcordanceConfig, AfConcordanceReport, AfIngestOptions,
    AfNmrConcordanceClass, AfStructureModel, ConcordanceSummary, ExperimentalS2Entry, PaeMatrix,
};
use nmr_core::cofold_bridge::{
    export_af3_json, export_boltz2_yaml, Af3ExportConfig, Boltz2ExportConfig,
    CofoldRestraintBundle, PoseScoreReport, PoseVerdict,
};
use nmr_core::domain_dock::{
    cluster_pae_domains_for_model, evaluate_tolman_domain_gdo, DomainMotionRegime,
    PaeClusterConfig, PaeDomainPartition, TolmanGdoMotionReport,
};
use nmr_core::excited_state::{
    classify_excited_state, parse_exchange_observations_tsv, ExchangeSignMode, ExcitedStateConfig,
    ExcitedStateReport, ExcitedStateVerdict,
};
use nmr_core::nmrstar::Reference;
use nmr_core::rdc::{parse_rdc_tsv_str, RdcMediumSet};
use nmr_core::struct_shifts::StructureModel;

use crate::contour_painter::SyncCrosshair;
use crate::mol3d_overlay::{
    ConcordanceTrackClass, DistanceRestraint3d, QuantitativeTrack, RestraintAtomSelector,
    RestraintKind,
};
use crate::mol3d_viewer::Mol3dViewerPanel;
use crate::sidechain_widget::SequenceRibbonState;
use crate::ss_ribbon_widget::SsRibbonState;

/// Canonical upper clamp (`31.75 A`) of the AlphaFold Predicted Aligned Error (`PAE`) color scale.
pub const PAE_COLORBAR_MAX_ANGSTROM: f32 = 31.75;

/// Maximum rendered grid resolution per axis on the interactive `PAE` heatmap (`<= 160 x 160` cells).
pub const MAX_PAE_HEATMAP_BINS: usize = 160;

// ============================================================================
// 1. Theme & Canonical Green-to-White `0..31.75 A` PAE Colormap
// ============================================================================

/// Theme-aware palette (`Paper White` and `Dark Oscilloscope`) for the Integrative AF Studio.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IntegrativeStudioTheme {
    pub panel_bg: egui::Color32,
    pub card_bg: egui::Color32,
    pub plot_bg: egui::Color32,
    pub border: egui::Color32,
    pub grid_line: egui::Color32,
    pub text_primary: egui::Color32,
    pub text_muted: egui::Color32,
    pub accent_primary: egui::Color32,
    pub domain_box_primary: egui::Color32,
    pub domain_box_selected: egui::Color32,
    pub badge_ok_bg: egui::Color32,
    pub badge_ok_fg: egui::Color32,
    pub badge_warn_bg: egui::Color32,
    pub badge_warn_fg: egui::Color32,
    pub badge_alert_bg: egui::Color32,
    pub badge_alert_fg: egui::Color32,
    pub parity_diagonal: egui::Color32,
    pub parity_point_signed: egui::Color32,
    pub parity_point_unsigned: egui::Color32,
    pub parity_point_selected: egui::Color32,
}

impl IntegrativeStudioTheme {
    pub const fn for_mode(paper_white: bool) -> Self {
        if paper_white {
            Self {
                panel_bg: egui::Color32::from_rgb(248, 250, 252),
                card_bg: egui::Color32::from_rgb(255, 255, 255),
                plot_bg: egui::Color32::from_rgb(250, 252, 254),
                border: egui::Color32::from_rgb(203, 213, 225),
                grid_line: egui::Color32::from_rgb(226, 232, 240),
                text_primary: egui::Color32::from_rgb(15, 23, 42),
                text_muted: egui::Color32::from_rgb(71, 85, 105),
                accent_primary: egui::Color32::from_rgb(2, 132, 199),
                domain_box_primary: egui::Color32::from_rgb(15, 23, 42),
                domain_box_selected: egui::Color32::from_rgb(217, 119, 6),
                badge_ok_bg: egui::Color32::from_rgb(220, 252, 231),
                badge_ok_fg: egui::Color32::from_rgb(22, 101, 52),
                badge_warn_bg: egui::Color32::from_rgb(254, 243, 199),
                badge_warn_fg: egui::Color32::from_rgb(146, 64, 14),
                badge_alert_bg: egui::Color32::from_rgb(254, 226, 226),
                badge_alert_fg: egui::Color32::from_rgb(153, 27, 27),
                parity_diagonal: egui::Color32::from_rgb(100, 116, 139),
                parity_point_signed: egui::Color32::from_rgb(2, 132, 199),
                parity_point_unsigned: egui::Color32::from_rgb(124, 58, 237),
                parity_point_selected: egui::Color32::from_rgb(220, 38, 38),
            }
        } else {
            Self {
                panel_bg: egui::Color32::from_rgb(15, 20, 29),
                card_bg: egui::Color32::from_rgb(22, 29, 41),
                plot_bg: egui::Color32::from_rgb(12, 17, 25),
                border: egui::Color32::from_rgb(46, 58, 78),
                grid_line: egui::Color32::from_rgb(33, 43, 59),
                text_primary: egui::Color32::from_rgb(226, 232, 240),
                text_muted: egui::Color32::from_rgb(148, 163, 184),
                accent_primary: egui::Color32::from_rgb(56, 189, 248),
                domain_box_primary: egui::Color32::from_rgb(56, 189, 248),
                domain_box_selected: egui::Color32::from_rgb(251, 191, 36),
                badge_ok_bg: egui::Color32::from_rgb(20, 62, 42),
                badge_ok_fg: egui::Color32::from_rgb(134, 239, 172),
                badge_warn_bg: egui::Color32::from_rgb(69, 44, 15),
                badge_warn_fg: egui::Color32::from_rgb(253, 224, 71),
                badge_alert_bg: egui::Color32::from_rgb(76, 23, 29),
                badge_alert_fg: egui::Color32::from_rgb(252, 165, 165),
                parity_diagonal: egui::Color32::from_rgb(100, 116, 139),
                parity_point_signed: egui::Color32::from_rgb(56, 189, 248),
                parity_point_unsigned: egui::Color32::from_rgb(192, 132, 252),
                parity_point_selected: egui::Color32::from_rgb(248, 113, 113),
            }
        }
    }
}

/// Canonical AlphaFold green-to-white continuous `PAE` colormap across `0.0 ..= 31.75 A`
/// (`0 A` deep forest green `#004D26` -> `31.75 A` white `#FFFFFF`).
#[inline]
pub fn pae_green_to_white_color(pae_angstrom: f32) -> egui::Color32 {
    if !pae_angstrom.is_finite() {
        return egui::Color32::from_rgb(255, 255, 255);
    }
    let t = (pae_angstrom / PAE_COLORBAR_MAX_ANGSTROM).clamp(0.0, 1.0);
    const STOPS: [(f32, [f32; 3]); 5] = [
        (0.00, [0.0, 77.0, 38.0]),
        (0.25, [27.0, 120.0, 55.0]),
        (0.50, [127.0, 188.0, 65.0]),
        (0.75, [217.0, 240.0, 211.0]),
        (1.00, [255.0, 255.0, 255.0]),
    ];
    let mut lo = STOPS[0];
    let mut hi = STOPS[4];
    for w in STOPS.windows(2) {
        if t >= w[0].0 && t <= w[1].0 {
            lo = w[0];
            hi = w[1];
            break;
        }
    }
    let span = (hi.0 - lo.0).max(1e-6);
    let u = ((t - lo.0) / span).clamp(0.0, 1.0);
    let lerp = |a: f32, b: f32| -> u8 { (a + u * (b - a)).round().clamp(0.0, 255.0) as u8 };
    egui::Color32::from_rgb(
        lerp(lo.1[0], hi.1[0]),
        lerp(lo.1[1], hi.1[1]),
        lerp(lo.1[2], hi.1[2]),
    )
}

/// Backwards-compatible alias for the canonical `0..31.75 A` AlphaFold `PAE` colormap.
#[inline]
pub fn pae_viridis_like_color(pae_angstrom: f32) -> egui::Color32 {
    pae_green_to_white_color(pae_angstrom)
}

// ============================================================================
// 2. Tabs, View-Models & Outbound Studio Actions
// ============================================================================

/// Active sub-view inside [`IntegrativeAfPanel`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum IntegrativeStudioTab {
    #[default]
    ConcordanceSummary,
    PaeDomainHeatmap,
    CofoldPoseRanker,
    ExcitedStateExplorer,
}

impl IntegrativeStudioTab {
    pub const ALL: [Self; 4] = [
        Self::ConcordanceSummary,
        Self::PaeDomainHeatmap,
        Self::CofoldPoseRanker,
        Self::ExcitedStateExplorer,
    ];

    pub const fn label(self) -> &'static str {
        match self {
            Self::ConcordanceSummary => "AF Concordance",
            Self::PaeDomainHeatmap => "PAE & Domains",
            Self::CofoldPoseRanker => "Co-Fold & Poses",
            Self::ExcitedStateExplorer => "Excited States",
        }
    }
}

/// Co-folding restraint preview dialect (`T21.6a`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CofoldPreviewDialect {
    #[default]
    Boltz2Yaml,
    AlphaFold3Json,
}

/// Sort column for the multi-pose complex NMR re-scoring table (`T21.6a`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PoseSortColumn {
    #[default]
    TotalScoreDesc,
    CspScoreDesc,
    NoePreViolationAsc,
    RdcQFreeAsc,
}

/// Summary row for a single candidate complex pose in the Co-Fold Ranker (`T21.6a`).
#[derive(Clone, Debug, PartialEq)]
pub struct CofoldPoseRowView {
    pub pose_id: String,
    pub total_score: f64,
    pub csp_score: f64,
    pub csp_spearman_rho: Option<f64>,
    pub csp_roc_auc: Option<f64>,
    pub noe_pre_viol_energy: f64,
    pub max_viol_angstrom: f64,
    pub rdc_q_free: Option<f64>,
    pub pcs_q: Option<f64>,
    /// Representative receptor interface residue (`0`-based index) for 1-click 3D highlighting.
    pub highlight_seq_idx: Option<usize>,
}

/// Overall ensemble verdict for co-folded pose ranking (`T21.6a`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CofoldRankVerdictView {
    #[default]
    Unscored,
    TopPoseSelected,
    InconclusiveOrAllViolated,
}

impl CofoldRankVerdictView {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Unscored => "No poses scored",
            Self::TopPoseSelected => "Committed Top-1 NMR Pose",
            Self::InconclusiveOrAllViolated => "Abstained: InconclusiveOrAllViolated",
        }
    }
}

/// Candidate biophysical excited-state hypothesis bar (`T21.6b`).
#[derive(Clone, Debug, PartialEq)]
pub struct ExcitedHypothesisBarView {
    pub family_label: String,
    pub locus_detail: String,
    pub num_params_k: usize,
    pub log_likelihood: f64,
    pub bic: f64,
    /// Evidence `BIC(H_null) - BIC(H)` (`2 ln BF` vs null model; positive favors `H`).
    pub two_ln_bf_vs_null: f64,
    pub posterior_weight: f64,
    pub reduced_chi2: f64,
    pub representative_seq_idx: Option<usize>,
    pub representative_seq_id: Option<i32>,
}

/// Single observed vs predicted differential shift point (`Delta_omega_AB`) on the parity plot (`T21.6b`).
#[derive(Clone, Debug, PartialEq)]
pub struct ExcitedParityPointView {
    pub seq_idx: usize,
    pub seq_id: i32,
    pub res_name: String,
    pub atom_name: String,
    /// `true` for signed `CEST`/`R1rho`, `false` for unsigned `CPMG` `|Delta_omega_AB|`.
    pub is_signed: bool,
    pub obs_ppm: f64,
    pub calc_ppm: f64,
    pub sigma_tot_ppm: f64,
}

/// Overall Bayesian excited-state classification verdict (`T21.6b`).
#[derive(Clone, Debug, Default, PartialEq)]
pub enum ExcitedVerdictView {
    #[default]
    Unrun,
    Committed {
        winner_label: String,
        two_ln_bf_12: f64,
        two_ln_bf_10: f64,
        reduced_chi2: f64,
    },
    AmbiguousOrUnmodeledState {
        reason: String,
        two_ln_bf_12: f64,
        two_ln_bf_10: f64,
    },
}

/// Synchronization and navigation actions emitted by [`IntegrativeAfPanel::show`].
#[derive(Clone, Debug, PartialEq)]
pub enum IntegrativeAfAction {
    /// Select and synchronize a single polymer residue across 2D HSQC, 3D strips, sidechain, and 3D viewer.
    SelectResidue { seq_idx: usize, seq_id: i32 },
    /// Switch [`Mol3dViewerPanel`] quantitative ribbon color/putty mode (`AlphaFoldPlddt` or `AfNmrConcordance`).
    SetMol3dTrack {
        track: QuantitativeTrack,
        putty_enabled: bool,
    },
    /// Highlight an intra- or inter-domain pair `(domain_a, domain_b)` (`1`-based IDs, `0` = linker/tail)
    /// and anchor residues `(seq_idx_i, seq_idx_j)` in [`Mol3dViewerPanel`].
    HighlightDomainPair {
        domain_a: usize,
        domain_b: usize,
        seq_idx_i: usize,
        seq_idx_j: usize,
    },
    /// Synchronize loaded `.cif`/`.pdb` coordinates and concordance classes into [`SsRibbonState`] and [`Mol3dViewerPanel`].
    SyncStructureAndConcordance,
    /// Jump to an exchanging residue in `DynamicsStudioPanel` (`CEST` / `CPMG`) and highlight on 3D ribbon.
    JumpToDynamicsResidue {
        seq_idx: usize,
        seq_id: i32,
        prefer_cpmg_unsigned: bool,
    },
    /// Select a candidate complex pose row in the Co-Fold Ranker.
    SelectCofoldPose {
        pose_index: usize,
        pose_id: String,
        highlight_seq_idx: Option<usize>,
    },
}

/// Per-frame telemetry counters recorded by [`IntegrativeAfPanel::show`] for headless `egui` tests.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IntegrativeStudioTelemetry {
    pub pae_cells_drawn: usize,
    pub pae_domain_boxes_drawn: usize,
    pub gdo_badges_drawn: usize,
    pub bic_bars_drawn: usize,
    pub parity_points_drawn: usize,
    pub pae_texture_uploads: usize,
}

/// Alias for [`IntegrativeStudioTelemetry`].
pub type IntegrativeAfFrameTelemetry = IntegrativeStudioTelemetry;

// ============================================================================
// 3. Stateful Integrative Structural & AlphaFold Studio Controller
// ============================================================================

/// Stateful controller for the zero-emoji Integrative Structural & AlphaFold Studio (`nmr-gui`).
#[derive(Clone)]
pub struct IntegrativeAfPanel {
    pub active_tab: IntegrativeStudioTab,
    pub show_loader_drawer: bool,
    pub force_bfactor_plddt: bool,
    pub is_synthetic_demo: bool,

    // Raw text buffers for desktop + `wasm32` in-memory ingestion
    pub structure_input_buf: String,
    pub pae_json_buf: String,
    pub s2_tsv_buf: String,
    pub rdc_tsv_buf: String,
    pub status_message: String,
    pub last_error: Option<String>,

    // T21.1 + T21.5 evaluated models & reports
    pub af_model: Option<AfStructureModel>,
    pub pae_matrix: Option<PaeMatrix>,
    pub concordance_report: Option<AfConcordanceReport>,
    pub domain_partition: Option<PaeDomainPartition>,
    pub tolman_gdo_report: Option<TolmanGdoMotionReport>,
    pub loaded_rdc_media_count: usize,
    pub has_paramagnetic_pcs_or_pre: bool,

    // Interactive PAE selection & cached GPU/CPU texture state (`21.W3:P1-14`)
    pub selected_residue_idx: Option<usize>,
    pub selected_pae_cell: Option<(usize, usize)>,
    pub selected_domain_pair: Option<(usize, usize)>,
    pub last_pae_plot_rect: Option<egui::Rect>,
    pub pae_texture: Option<egui::TextureHandle>,
    pub pae_texture_key: Option<u64>,
    pub pae_texture_uploads: usize,

    // T21.6a Co-Folding Restraint Preview & Pose Ranker state
    pub preview_dialect: CofoldPreviewDialect,
    pub af3_json_preview: String,
    pub af3_sidecar_note: String,
    pub boltz2_yaml_preview: String,
    pub surface_pocket_seq_ids: Vec<i32>,
    pub buried_allosteric_seq_ids: Vec<i32>,
    pub pose_sort: PoseSortColumn,
    pub pose_verdict: CofoldRankVerdictView,
    pub pose_rows: Vec<CofoldPoseRowView>,
    pub selected_pose_idx: Option<usize>,

    // T21.6b Excited-State Hypothesis Explorer state
    pub exchange_tsv_buf: String,
    pub excited_verdict: ExcitedVerdictView,
    pub excited_n_obs: usize,
    pub excited_bars: Vec<ExcitedHypothesisBarView>,
    pub excited_parity_points: Vec<ExcitedParityPointView>,
    pub selected_parity_idx: Option<usize>,
    pub last_bic_chart_rect: Option<egui::Rect>,
    pub last_parity_plot_rect: Option<egui::Rect>,

    // Frame telemetry for deterministic headless tests
    pub last_telemetry: IntegrativeStudioTelemetry,
}

impl std::fmt::Debug for IntegrativeAfPanel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IntegrativeAfPanel")
            .field("active_tab", &self.active_tab)
            .field("is_synthetic_demo", &self.is_synthetic_demo)
            .field("status_message", &self.status_message)
            .field("pae_texture_loaded", &self.pae_texture.is_some())
            .field("pae_texture_key", &self.pae_texture_key)
            .field("pae_texture_uploads", &self.pae_texture_uploads)
            .field("last_telemetry", &self.last_telemetry)
            .finish()
    }
}

impl Default for IntegrativeAfPanel {
    fn default() -> Self {
        Self::new()
    }
}

impl IntegrativeAfPanel {
    pub fn new() -> Self {
        Self {
            active_tab: IntegrativeStudioTab::ConcordanceSummary,
            show_loader_drawer: false,
            force_bfactor_plddt: false,
            is_synthetic_demo: false,
            structure_input_buf: String::new(),
            pae_json_buf: String::new(),
            s2_tsv_buf: String::new(),
            rdc_tsv_buf: String::new(),
            status_message:
                "Load an AlphaFold .cif / .pdb + PAE JSON or click 'Load 2-Domain AF Demo' to start."
                    .to_string(),
            last_error: None,
            af_model: None,
            pae_matrix: None,
            concordance_report: None,
            domain_partition: None,
            tolman_gdo_report: None,
            loaded_rdc_media_count: 0,
            has_paramagnetic_pcs_or_pre: false,
            selected_residue_idx: None,
            selected_pae_cell: None,
            selected_domain_pair: None,
            last_pae_plot_rect: None,
            pae_texture: None,
            pae_texture_key: None,
            pae_texture_uploads: 0,
            preview_dialect: CofoldPreviewDialect::Boltz2Yaml,
            af3_json_preview: String::new(),
            af3_sidecar_note:
                "Public AlphaFold 3 JSON (dialect='alphafold3', version=2) encodes polymer/ligand chains and covalent bonds; experimental CSP pocket and NOE/PRE distance bounds are written to <prefix>.nmr_restraints.json for pose re-scoring and natively enforced in Boltz-2 YAML."
                    .to_string(),
            boltz2_yaml_preview: String::new(),
            surface_pocket_seq_ids: Vec::new(),
            buried_allosteric_seq_ids: Vec::new(),
            pose_sort: PoseSortColumn::TotalScoreDesc,
            pose_verdict: CofoldRankVerdictView::Unscored,
            pose_rows: Vec::new(),
            selected_pose_idx: None,
            exchange_tsv_buf: String::new(),
            excited_verdict: ExcitedVerdictView::Unrun,
            excited_n_obs: 0,
            excited_bars: Vec::new(),
            excited_parity_points: Vec::new(),
            selected_parity_idx: None,
            last_bic_chart_rect: None,
            last_parity_plot_rect: None,
            last_telemetry: IntegrativeStudioTelemetry::default(),
        }
    }

    /// Return the latest per-frame telemetry snapshot (`21.W3:P1-14`).
    pub fn last_telemetry(&self) -> &IntegrativeStudioTelemetry {
        &self.last_telemetry
    }

    /// Alias for [`Self::load_synthetic_two_domain_demo`].
    pub fn load_ubiquitin_demo(&mut self) -> Result<(Reference, String), String> {
        self.load_synthetic_two_domain_demo()
    }

    /// Parse `.cif`/`.pdb` coordinates (`structure_input_buf`), optional `PAE` JSON (`pae_json_buf`),
    /// optional `modelfree.tsv` $S^2$ (`s2_tsv_buf`), and optional `RDC` TSV (`rdc_tsv_buf`), then
    /// run `T21.1` concordance (`evaluate_af_nmr_concordance`) and `T21.5` spectral domain clustering
    /// (`cluster_pae_domains_for_model` + `evaluate_tolman_domain_gdo`).
    pub fn ingest_from_buffers(
        &mut self,
        nmr_shifts: Option<&Reference>,
    ) -> Result<String, String> {
        self.is_synthetic_demo = false;
        self.ingest_from_buffers_inner(nmr_shifts)
    }

    fn ingest_from_buffers_inner(
        &mut self,
        nmr_shifts: Option<&Reference>,
    ) -> Result<String, String> {
        let struct_trimmed = self.structure_input_buf.trim();
        if struct_trimmed.is_empty() {
            let err =
                "Structure buffer is empty; paste or load a .cif / .pdb model first.".to_string();
            self.last_error = Some(err.clone());
            return Err(err);
        }

        let ingest_opts = AfIngestOptions {
            target_chain_id: None,
            force_bfactor_plddt: self.force_bfactor_plddt,
        };
        let model =
            parse_af_structure_model_str(struct_trimmed, &ingest_opts).inspect_err(|e| {
                self.last_error = Some(e.clone());
            })?;

        let pae_opt = if !self.pae_json_buf.trim().is_empty() {
            let pm = parse_pae_json_str(self.pae_json_buf.trim()).inspect_err(|e| {
                self.last_error = Some(e.clone());
            })?;
            pm.validate_chain_length(model.residues.len())
                .inspect_err(|e| {
                    self.last_error = Some(e.clone());
                })?;
            Some(pm)
        } else {
            None
        };

        let s2_entries: Option<Vec<ExperimentalS2Entry>> = if !self.s2_tsv_buf.trim().is_empty() {
            let parsed = parse_modelfree_s2_tsv_str(self.s2_tsv_buf.trim()).inspect_err(|e| {
                self.last_error = Some(e.clone());
            })?;
            Some(parsed)
        } else {
            None
        };

        // Build fallback Reference from model sequence if external NMR shifts are not yet loaded
        let fallback_ref = Reference::new(model.sequence_3letter(), model.seq_ids());
        let eff_ref = nmr_shifts
            .filter(|r| !r.sequence.is_empty())
            .unwrap_or(&fallback_ref);

        let conc_cfg = AfConcordanceConfig::default();
        let conc_report = evaluate_af_nmr_concordance(
            &model,
            eff_ref,
            pae_opt.as_ref(),
            s2_entries.as_deref(),
            &conc_cfg,
        )
        .inspect_err(|e| {
            self.last_error = Some(e.clone());
        })?;

        let partition_opt = if let Some(ref pm) = pae_opt {
            Some(
                cluster_pae_domains_for_model(&model, pm, &PaeClusterConfig::default())
                    .inspect_err(|e| {
                        self.last_error = Some(e.clone());
                    })?,
            )
        } else {
            None
        };

        let rdc_media: Vec<RdcMediumSet> = if !self.rdc_tsv_buf.trim().is_empty() {
            parse_rdc_tsv_str(self.rdc_tsv_buf.trim()).inspect_err(|e| {
                self.last_error = Some(e.clone());
            })?
        } else {
            Vec::new()
        };
        self.loaded_rdc_media_count = rdc_media.len();

        self.tolman_gdo_report = match (&partition_opt, rdc_media.is_empty()) {
            (Some(part), false) => {
                let chain = model.to_pdb_chain();
                Some(evaluate_tolman_domain_gdo(
                    &chain, part, &rdc_media, 128, 42,
                ))
            }
            _ => None,
        };

        let n_res = model.residues.len();
        let n_dom = partition_opt.as_ref().map_or(0, |p| p.domains.len());
        let summary_msg = match &conc_report.summary {
            ConcordanceSummary::Evaluated {
                spearman_rho_s,
                rmsd_rigidity,
                ..
            } => {
                if self.is_synthetic_demo {
                    format!(
                        "Analyzed {n_res} residues (synthetic demo, rho_s = {spearman_rho_s:.2}, RMSD_rig = {rmsd_rigidity:.2} A, {n_dom} PAE domain(s))"
                    )
                } else {
                    format!(
                        "Analyzed {n_res} residues (rho_s = {spearman_rho_s:.2}, RMSD_rig = {rmsd_rigidity:.2} A, {n_dom} PAE domain(s))"
                    )
                }
            }
            ConcordanceSummary::InsufficientCoverage { n_eval, .. } => {
                if self.is_synthetic_demo {
                    format!(
                        "Loaded {n_res} residues (synthetic demo, {n_eval} NMR-evaluated; {n_dom} PAE domain(s))"
                    )
                } else {
                    format!(
                        "Loaded {n_res} residues ({n_eval} NMR-evaluated; {n_dom} PAE domain(s))"
                    )
                }
            }
        };

        self.af_model = Some(model);
        self.pae_matrix = pae_opt;
        self.pae_texture_key = None;
        self.concordance_report = Some(conc_report);
        self.domain_partition = partition_opt;
        self.last_error = None;
        self.status_message = summary_msg.clone();
        Ok(summary_msg)
    }

    /// Populate Tab 4 (`Excited States`) directly from an [`ExcitedStateReport`] (`T21.6b`).
    pub fn populate_from_excited_state_report(
        &mut self,
        report: &ExcitedStateReport,
        seq_ids: &[i32],
    ) {
        let resolve_idx = |sid: i32| -> usize {
            seq_ids
                .iter()
                .position(|&s| s == sid)
                .unwrap_or_else(|| (sid - 1).max(0) as usize)
        };
        self.excited_n_obs = report.n_observations;
        self.excited_verdict = match report.verdict {
            ExcitedStateVerdict::Committed(kind) => {
                let locus = report
                    .ranked_hypotheses
                    .first()
                    .map(|h| h.best_locus_summary.as_str())
                    .unwrap_or("");
                ExcitedVerdictView::Committed {
                    winner_label: format!("{kind:?} ({locus})"),
                    two_ln_bf_12: report.two_ln_bf_top1_vs_top2,
                    two_ln_bf_10: report.two_ln_bf_top1_vs_null,
                    reduced_chi2: report.top1_reduced_chi2,
                }
            }
            ExcitedStateVerdict::AmbiguousOrUnmodeledState => {
                ExcitedVerdictView::AmbiguousOrUnmodeledState {
                    reason: report.verdict_reason.clone(),
                    two_ln_bf_12: report.two_ln_bf_top1_vs_top2,
                    two_ln_bf_10: report.two_ln_bf_top1_vs_null,
                }
            }
        };

        self.excited_bars = report
            .ranked_hypotheses
            .iter()
            .map(|h| {
                let rep_sid = h.affected_seq_ids.first().copied();
                ExcitedHypothesisBarView {
                    family_label: format!("{:?}", h.kind),
                    locus_detail: h.best_locus_summary.clone(),
                    num_params_k: h.num_params_k,
                    log_likelihood: h.max_log_likelihood,
                    bic: h.bic,
                    two_ln_bf_vs_null: h.two_ln_bf_vs_null,
                    posterior_weight: h.posterior_weight,
                    reduced_chi2: h.reduced_chi2,
                    representative_seq_idx: rep_sid.map(resolve_idx),
                    representative_seq_id: rep_sid,
                }
            })
            .collect();

        self.excited_parity_points = report
            .ranked_hypotheses
            .first()
            .map(|top| {
                top.predictions
                    .iter()
                    .map(|p| ExcitedParityPointView {
                        seq_idx: resolve_idx(p.seq_id),
                        seq_id: p.seq_id,
                        res_name: p.res_name.clone(),
                        atom_name: p.atom.clone(),
                        is_signed: p.sign_mode == ExchangeSignMode::Signed,
                        obs_ppm: p.observed_dw_ppm,
                        calc_ppm: p.predicted_effective_dw_ppm,
                        sigma_tot_ppm: p.sigma_tot_ppm,
                    })
                    .collect()
            })
            .unwrap_or_default();
        self.selected_parity_idx = None;
    }

    /// Run `nmr_core::excited_state::classify_excited_state` in memory from `structure_input_buf`
    /// and `exchange_tsv_buf` (`desktop` + `wasm32`) and update Tab 4.
    pub fn evaluate_excited_state_from_buffers(
        &mut self,
        ground_shifts: Option<&Reference>,
    ) -> Result<ExcitedStateReport, String> {
        let ground_model = StructureModel::parse_str(&self.structure_input_buf, None, false)
            .inspect_err(|e| {
                self.last_error = Some(e.clone());
            })?;
        let obs = parse_exchange_observations_tsv(&self.exchange_tsv_buf).inspect_err(|e| {
            self.last_error = Some(e.clone());
        })?;
        let rep = classify_excited_state(
            ground_shifts,
            &ground_model,
            &obs,
            &[],
            &ExcitedStateConfig::default(),
        )
        .inspect_err(|e| {
            self.last_error = Some(e.clone());
        })?;
        let seq_ids: Vec<i32> = ground_model.residues.iter().map(|r| r.seq_id).collect();
        self.populate_from_excited_state_report(&rep, &seq_ids);
        self.last_error = None;
        self.status_message = format!(
            "Classified {} exchange shifts -> {}",
            rep.n_observations,
            rep.verdict.as_str()
        );
        Ok(rep)
    }

    /// Populate Tab 3 (`Co-Fold & Poses`) directly from a `T21.6a` [`CofoldRestraintBundle`] and
    /// [`PoseScoreReport`], rendering live `AlphaFold 3` JSON (`v2`) and `Boltz-2` YAML (`v1`)
    /// previews plus ranked complex poses.
    pub fn populate_from_cofold_report(
        &mut self,
        bundle: Option<&CofoldRestraintBundle>,
        report: &PoseScoreReport,
        seq_ids: &[i32],
    ) {
        let resolve_idx = |sid: i32| -> Option<usize> {
            seq_ids
                .iter()
                .position(|&s| s == sid)
                .or_else(|| (sid >= 1).then_some((sid - 1) as usize))
        };

        if let Some(b) = bundle {
            self.surface_pocket_seq_ids = b.pocket_contacts.iter().map(|p| p.seq_id).collect();
            self.buried_allosteric_seq_ids = b
                .allosteric_core_perturbations
                .iter()
                .map(|p| p.seq_id)
                .collect();
            self.af3_sidecar_note = b.af3_conditioning_note.clone();
            if let Ok(af3_txt) = export_af3_json(b, &Af3ExportConfig::default()) {
                self.af3_json_preview = af3_txt;
            }
            if let Ok(boltz_txt) = export_boltz2_yaml(b, &Boltz2ExportConfig::default()) {
                self.boltz2_yaml_preview = boltz_txt;
            }
        }

        self.pose_verdict = match report.verdict {
            PoseVerdict::RankedConclusive => CofoldRankVerdictView::TopPoseSelected,
            PoseVerdict::InconclusiveOrAllViolated => {
                CofoldRankVerdictView::InconclusiveOrAllViolated
            }
        };

        let default_hi = self
            .surface_pocket_seq_ids
            .first()
            .copied()
            .and_then(resolve_idx);
        self.pose_rows = report
            .poses
            .iter()
            .map(|p| {
                let top_csp_idx = p
                    .residue_csp_points
                    .iter()
                    .filter(|pt| pt.is_surface_2sigma_hit)
                    .max_by(|a, b| a.csp_obs_ppm.total_cmp(&b.csp_obs_ppm))
                    .and_then(|pt| resolve_idx(pt.seq_id))
                    .or(default_hi);
                CofoldPoseRowView {
                    pose_id: p.pose_id.clone(),
                    total_score: p.total_score,
                    csp_score: p.s_csp,
                    csp_spearman_rho: p.csp_spearman_rho,
                    csp_roc_auc: p.csp_contact_roc_auc,
                    noe_pre_viol_energy: p.noe_pre_violation_energy,
                    max_viol_angstrom: p.max_distance_violation_angstrom,
                    rdc_q_free: p.rdc_q_free,
                    pcs_q: p.pcs_q_factor,
                    highlight_seq_idx: top_csp_idx,
                }
            })
            .collect();
        self.selected_pose_idx = (!self.pose_rows.is_empty()).then_some(0);
    }

    /// Populate a deterministic 52-residue 2-domain AlphaFold `.cif` + `PAE` JSON + solution NMR
    /// concordance + Tolman `GDO` + Co-Fold pose + Excited-State demo dataset in memory (`wasm32`-safe).
    pub fn load_synthetic_two_domain_demo(&mut self) -> Result<(Reference, String), String> {
        let (cif_text, pae_json, demo_ref, rdc_tsv) = build_synthetic_two_domain_af_demo();
        self.structure_input_buf = cif_text;
        self.pae_json_buf = pae_json;
        self.s2_tsv_buf.clear();
        self.rdc_tsv_buf = rdc_tsv;
        self.is_synthetic_demo = true;
        let msg = self.ingest_from_buffers_inner(Some(&demo_ref))?;

        // Populate deterministic T21.6a Co-Fold preview & multi-pose ranking showcase
        self.surface_pocket_seq_ids = vec![8, 10, 12, 14, 16, 18];
        self.buried_allosteric_seq_ids = vec![6, 15];
        self.af3_json_preview = "{\n  \"name\": \"ferric_two_domain_complex\",\n  \"modelSeeds\": [1],\n  \"sequences\": [\n    { \"protein\": { \"id\": \"A\", \"sequence\": \"MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLED\" } },\n    { \"ligand\": { \"id\": \"L\", \"smiles\": \"c1ccc(cc1)C(=O)N\" } }\n  ],\n  \"bondedAtomPairs\": [],\n  \"dialect\": \"alphafold3\",\n  \"version\": 2\n}".to_string();
        self.boltz2_yaml_preview = "version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLED\n  - ligand:\n      id: L\n      smiles: \"c1ccc(cc1)C(=O)N\"\nconstraints:\n  - pocket:\n      binder: L\n      contacts:\n        - [A, 8]\n        - [A, 10]\n        - [A, 12]\n        - [A, 14]\n        - [A, 16]\n        - [A, 18]\n      max_distance: 6.0\n      force: false\n  - contact:\n      token1: [A, 12]\n      token2: [L, 1]\n      max_distance: 5.5\n      force: false\n".to_string();
        self.pose_verdict = CofoldRankVerdictView::TopPoseSelected;
        self.pose_rows = vec![
            CofoldPoseRowView {
                pose_id: "pose_01_native.cif".to_string(),
                total_score: 4.82,
                csp_score: 5.14,
                csp_spearman_rho: Some(0.89),
                csp_roc_auc: Some(0.96),
                noe_pre_viol_energy: 0.08,
                max_viol_angstrom: 0.21,
                rdc_q_free: Some(0.19),
                pcs_q: Some(0.11),
                highlight_seq_idx: Some(11),
            },
            CofoldPoseRowView {
                pose_id: "pose_02_tilted_4A.cif".to_string(),
                total_score: 1.15,
                csp_score: 2.40,
                csp_spearman_rho: Some(0.54),
                csp_roc_auc: Some(0.71),
                noe_pre_viol_energy: 0.86,
                max_viol_angstrom: 1.12,
                rdc_q_free: Some(0.31),
                pcs_q: Some(0.24),
                highlight_seq_idx: Some(13),
            },
            CofoldPoseRowView {
                pose_id: "pose_03_decoy_14A.cif".to_string(),
                total_score: -6.40,
                csp_score: 0.12,
                csp_spearman_rho: Some(-0.08),
                csp_roc_auc: Some(0.42),
                noe_pre_viol_energy: 5.90,
                max_viol_angstrom: 3.85,
                rdc_q_free: Some(0.58),
                pcs_q: Some(0.61),
                highlight_seq_idx: Some(39),
            },
        ];
        self.selected_pose_idx = Some(0);

        // Populate deterministic T21.6b Excited-State BIC & parity showcase
        self.excited_n_obs = 8;
        self.excited_verdict = ExcitedVerdictView::Committed {
            winner_label: "H2: XxxProCisTrans (Pro38 trans->cis)".to_string(),
            two_ln_bf_12: 18.4,
            two_ln_bf_10: 34.2,
            reduced_chi2: 0.94,
        };
        self.excited_bars = vec![
            ExcitedHypothesisBarView {
                family_label: "H2: XxxProCisTrans".to_string(),
                locus_detail: "Pro38 omega trans->cis (Delta_bg = 9.6 ppm)".to_string(),
                num_params_k: 1,
                log_likelihood: -6.10,
                bic: 14.28,
                two_ln_bf_vs_null: 34.20,
                posterior_weight: 0.9998,
                reduced_chi2: 0.94,
                representative_seq_idx: Some(37),
                representative_seq_id: Some(38),
            },
            ExcitedHypothesisBarView {
                family_label: "H1: LocalUnfolding".to_string(),
                locus_detail: "Window 36..40 -> ncIDP coil".to_string(),
                num_params_k: 2,
                log_likelihood: -14.26,
                bic: 32.68,
                two_ln_bf_vs_null: 15.80,
                posterior_weight: 0.0001,
                reduced_chi2: 3.12,
                representative_seq_idx: Some(36),
                representative_seq_id: Some(37),
            },
            ExcitedHypothesisBarView {
                family_label: "H3: RotamerOrHisFlip".to_string(),
                locus_detail: "Phe45 chi1 g- -> t ring-current".to_string(),
                num_params_k: 2,
                log_likelihood: -18.05,
                bic: 40.26,
                two_ln_bf_vs_null: 8.22,
                posterior_weight: 0.0001,
                reduced_chi2: 4.45,
                representative_seq_idx: Some(44),
                representative_seq_id: Some(45),
            },
            ExcitedHypothesisBarView {
                family_label: "H4: Alt3DConformer".to_string(),
                locus_detail: "alt_hinge_open.cif".to_string(),
                num_params_k: 1,
                log_likelihood: -20.40,
                bic: 42.88,
                two_ln_bf_vs_null: 5.60,
                posterior_weight: 0.0,
                reduced_chi2: 5.10,
                representative_seq_idx: Some(27),
                representative_seq_id: Some(28),
            },
            ExcitedHypothesisBarView {
                family_label: "H0: NullPerturbation".to_string(),
                locus_detail: "Zero structural shift change".to_string(),
                num_params_k: 0,
                log_likelihood: -24.24,
                bic: 48.48,
                two_ln_bf_vs_null: 0.0,
                posterior_weight: 0.0,
                reduced_chi2: 7.85,
                representative_seq_idx: None,
                representative_seq_id: None,
            },
        ];
        self.excited_parity_points = vec![
            ExcitedParityPointView {
                seq_idx: 37,
                seq_id: 38,
                res_name: "PRO".to_string(),
                atom_name: "CB".to_string(),
                is_signed: true,
                obs_ppm: 2.34,
                calc_ppm: 2.40,
                sigma_tot_ppm: 0.35,
            },
            ExcitedParityPointView {
                seq_idx: 37,
                seq_id: 38,
                res_name: "PRO".to_string(),
                atom_name: "CG".to_string(),
                is_signed: true,
                obs_ppm: -2.51,
                calc_ppm: -2.60,
                sigma_tot_ppm: 0.35,
            },
            ExcitedParityPointView {
                seq_idx: 36,
                seq_id: 37,
                res_name: "PRO".to_string(),
                atom_name: "CA".to_string(),
                is_signed: true,
                obs_ppm: -1.08,
                calc_ppm: -1.15,
                sigma_tot_ppm: 0.30,
            },
            ExcitedParityPointView {
                seq_idx: 36,
                seq_id: 37,
                res_name: "PRO".to_string(),
                atom_name: "C".to_string(),
                is_signed: true,
                obs_ppm: -1.29,
                calc_ppm: -1.35,
                sigma_tot_ppm: 0.32,
            },
            ExcitedParityPointView {
                seq_idx: 38,
                seq_id: 39,
                res_name: "ASP".to_string(),
                atom_name: "N".to_string(),
                is_signed: false,
                obs_ppm: 1.85,
                calc_ppm: 1.78,
                sigma_tot_ppm: 0.45,
            },
            ExcitedParityPointView {
                seq_idx: 38,
                seq_id: 39,
                res_name: "ASP".to_string(),
                atom_name: "H".to_string(),
                is_signed: false,
                obs_ppm: 0.42,
                calc_ppm: 0.39,
                sigma_tot_ppm: 0.12,
            },
        ];

        Ok((demo_ref, msg))
    }

    /// Synchronize the loaded `.cif`/`.pdb` model and `T21.1` concordance classes directly onto
    /// [`SsRibbonState`] and [`Mol3dViewerPanel`].
    pub fn sync_to_ribbon_and_mol3d(
        &self,
        ss_ribbon: &mut SsRibbonState,
        mol3d: &mut Mol3dViewerPanel,
        nmr_seq: &[String],
        nmr_seq_ids: &[i32],
    ) {
        if !self.structure_input_buf.trim().is_empty() && mol3d.ensemble.is_none() {
            let _ = mol3d.load_structure_text(&self.structure_input_buf, nmr_seq, nmr_seq_ids);
        }
        if let Some(ref rep) = self.concordance_report {
            let ss4: Vec<_> = rep.residues.iter().map(|r| r.state4_af).collect();
            let plddts: Vec<Option<f64>> = rep.residues.iter().map(|r| r.plddt).collect();
            let classes: Vec<AfNmrConcordanceClass> =
                rep.residues.iter().map(|r| r.concordance_class).collect();
            ss_ribbon.set_core_ss3d_slice(&ss4);
            ss_ribbon.set_plddt_slice(&plddts);
            ss_ribbon.set_core_concordance_override(&classes);
            if let ConcordanceSummary::Evaluated {
                spearman_rho_s,
                rmsd_rigidity,
                n_eval,
                ..
            } = rep.summary
            {
                ss_ribbon.concordance_summary.spearman_rho = Some(spearman_rho_s);
                ss_ribbon.concordance_summary.rigidity_rmsd = Some(rmsd_rigidity);
                ss_ribbon.concordance_summary.n_evaluated = n_eval;
            }
            let gui_classes: Vec<ConcordanceTrackClass> = classes
                .iter()
                .copied()
                .map(ConcordanceTrackClass::from)
                .collect();
            mol3d.track_data.set_plddt_slice(&plddts);
            mol3d.set_concordance_classes(&gui_classes);
            mol3d.flush_pending_updates();
        }
    }

    /// Render the Integrative Structural & AlphaFold Studio inside any `egui` container (`Window` or
    /// dock tab) and return any synchronization actions triggered during this frame.
    pub fn show(&mut self, ui: &mut egui::Ui, paper_white: bool) -> Vec<IntegrativeAfAction> {
        let theme = IntegrativeStudioTheme::for_mode(paper_white);
        self.last_telemetry = IntegrativeStudioTelemetry {
            pae_texture_uploads: self.pae_texture_uploads,
            ..Default::default()
        };
        let mut actions = Vec::new();

        egui::Frame::NONE
            .fill(theme.panel_bg)
            .inner_margin(egui::Margin::same(10))
            .show(ui, |ui| {
                self.draw_header_and_tabs(ui, &theme, &mut actions);
                ui.add_space(6.0);

                egui::CollapsingHeader::new("Advanced Parameters & Data Input")
                    .default_open(self.show_loader_drawer)
                    .show(ui, |ui| {
                        self.draw_loader_drawer(ui, &theme, &mut actions);
                    });
                ui.add_space(6.0);

                if let Some(ref err) = self.last_error {
                    egui::Frame::NONE
                        .fill(theme.badge_alert_bg)
                        .stroke(egui::Stroke::new(1.0, theme.badge_alert_fg))
                        .corner_radius(4.0)
                        .inner_margin(egui::Margin::symmetric(8, 5))
                        .show(ui, |ui| {
                            ui.label(
                                egui::RichText::new(err)
                                    .size(11.5)
                                    .color(theme.badge_alert_fg),
                            );
                        });
                    ui.add_space(6.0);
                }

                match self.active_tab {
                    IntegrativeStudioTab::ConcordanceSummary => {
                        self.draw_concordance_tab(ui, &theme, &mut actions);
                    }
                    IntegrativeStudioTab::PaeDomainHeatmap => {
                        self.draw_pae_heatmap_tab(ui, &theme, &mut actions);
                    }
                    IntegrativeStudioTab::CofoldPoseRanker => {
                        self.draw_cofold_tab(ui, &theme, &mut actions);
                    }
                    IntegrativeStudioTab::ExcitedStateExplorer => {
                        self.draw_excited_state_tab(ui, &theme, &mut actions);
                    }
                }
            });

        self.last_telemetry.pae_texture_uploads = self.pae_texture_uploads;
        actions
    }

    fn draw_header_and_tabs(
        &mut self,
        ui: &mut egui::Ui,
        theme: &IntegrativeStudioTheme,
        actions: &mut Vec<IntegrativeAfAction>,
    ) {
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing.x = 6.0;
            for tab in IntegrativeStudioTab::ALL {
                let selected = self.active_tab == tab;
                let btn = egui::Button::new(
                    egui::RichText::new(tab.label())
                        .size(12.0)
                        .strong()
                        .color(if selected {
                            theme.accent_primary
                        } else {
                            theme.text_primary
                        }),
                )
                .fill(if selected {
                    theme.card_bg
                } else {
                    theme.panel_bg
                })
                .stroke(egui::Stroke::new(
                    1.0,
                    if selected {
                        theme.accent_primary
                    } else {
                        theme.border
                    },
                ))
                .corner_radius(4.0);
                if ui.add(btn).clicked() {
                    self.active_tab = tab;
                }
            }

            ui.separator();
            if ui
                .button("Load 2-Domain AF Demo")
                .on_hover_text(
                    "Populate a 52-residue 2-domain AlphaFold .cif + PAE JSON + RDC + Co-Fold + Excited-State showcase",
                )
                .clicked()
                && self.load_synthetic_two_domain_demo().is_ok()
            {
                actions.push(IntegrativeAfAction::SyncStructureAndConcordance);
            }
        });

        ui.add_space(3.0);
        ui.label(
            egui::RichText::new(&self.status_message)
                .size(11.0)
                .color(theme.text_muted),
        );
    }

    fn draw_loader_drawer(
        &mut self,
        ui: &mut egui::Ui,
        theme: &IntegrativeStudioTheme,
        actions: &mut Vec<IntegrativeAfAction>,
    ) {
        egui::Frame::NONE
            .fill(theme.card_bg)
            .stroke(egui::Stroke::new(1.0, theme.border))
            .corner_radius(5.0)
            .inner_margin(egui::Margin::same(8))
            .show(ui, |ui| {
                ui.columns(2, |cols| {
                    cols[0].label(
                        egui::RichText::new("Model Coordinates (.cif / .pdb)")
                            .size(11.5)
                            .strong()
                            .color(theme.text_primary),
                    );
                    cols[0].add(
                        egui::TextEdit::multiline(&mut self.structure_input_buf)
                            .desired_rows(4)
                            .desired_width(f32::INFINITY)
                            .font(egui::TextStyle::Monospace),
                    );
                    cols[1].label(
                        egui::RichText::new("PAE Matrix JSON (AFDB / ColabFold / AF3)")
                            .size(11.5)
                            .strong()
                            .color(theme.text_primary),
                    );
                    cols[1].add(
                        egui::TextEdit::multiline(&mut self.pae_json_buf)
                            .desired_rows(4)
                            .desired_width(f32::INFINITY)
                            .font(egui::TextStyle::Monospace),
                    );
                });
                ui.add_space(4.0);
                ui.horizontal_wrapped(|ui| {
                    ui.checkbox(
                        &mut self.force_bfactor_plddt,
                        "Force B_iso/tempFactor as pLDDT on experimental headers",
                    );
                    if ui.button("Analyze & Cluster PAE").clicked()
                        && self.ingest_from_buffers(None).is_ok()
                    {
                        actions.push(IntegrativeAfAction::SyncStructureAndConcordance);
                    }
                    if ui
                        .button("Color 3D Ribbon by AlphaFold pLDDT")
                        .on_hover_text("Apply canonical 4-band AlphaFold pLDDT palette (#0053D6..#FF7D45)")
                        .clicked()
                    {
                        actions.push(IntegrativeAfAction::SetMol3dTrack {
                            track: QuantitativeTrack::AlphaFoldPlddt,
                            putty_enabled: true,
                        });
                    }
                    if ui
                        .button("Color 3D Putty by AF vs NMR Concordance")
                        .on_hover_text(
                            "Highlight Model-Ordered / Solution-Disordered and DynamicHingeOrAltState discrepancies on 3D putty",
                        )
                        .clicked()
                    {
                        actions.push(IntegrativeAfAction::SetMol3dTrack {
                            track: QuantitativeTrack::AfNmrConcordance,
                            putty_enabled: true,
                        });
                    }
                    if ui.button("Sync 3-Track SS Ribbon").clicked() {
                        actions.push(IntegrativeAfAction::SyncStructureAndConcordance);
                    }
                });
                ui.add_space(4.0);
                ui.horizontal_wrapped(|ui| {
                    ui.label(
                        egui::RichText::new("Sort Co-Fold Poses:")
                            .size(11.0)
                            .color(theme.text_muted),
                    );
                    ui.selectable_value(
                        &mut self.pose_sort,
                        PoseSortColumn::TotalScoreDesc,
                        "S_total",
                    );
                    ui.selectable_value(
                        &mut self.pose_sort,
                        PoseSortColumn::CspScoreDesc,
                        "S_CSP",
                    );
                    ui.selectable_value(
                        &mut self.pose_sort,
                        PoseSortColumn::NoePreViolationAsc,
                        "E_viol (NOE/PRE)",
                    );
                    ui.selectable_value(
                        &mut self.pose_sort,
                        PoseSortColumn::RdcQFreeAsc,
                        "RDC Q_free",
                    );
                });
            });
    }

    // ------------------------------------------------------------------------
    // Tab 1: AlphaFold vs Solution NMR Rigidity Concordance (`T21.1` & `T21.3`)
    // ------------------------------------------------------------------------

    fn draw_concordance_tab(
        &mut self,
        ui: &mut egui::Ui,
        theme: &IntegrativeStudioTheme,
        actions: &mut Vec<IntegrativeAfAction>,
    ) {
        let Some(ref rep) = self.concordance_report else {
            ui.label(
                egui::RichText::new(
                    "No AlphaFold / 3D structure model evaluated yet. Click 'Load 2-Domain AF Demo' or paste a .cif + PAE JSON above.",
                )
                .color(theme.text_muted),
            );
            return;
        };

        // Top metric cards
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing = egui::vec2(8.0, 6.0);
            match rep.summary {
                ConcordanceSummary::Evaluated {
                    n_eval,
                    spearman_rho_s,
                    c_corr_score,
                    rmsd_rigidity,
                    c_rmsd_score,
                } => {
                    draw_metric_card(
                        ui,
                        theme,
                        "Spearman rho_s",
                        &format!("{spearman_rho_s:+.3}"),
                        &format!("C_corr = {c_corr_score:.1} ({n_eval} eval)"),
                    );
                    draw_metric_card(
                        ui,
                        theme,
                        "Rigidity RMSD",
                        &format!("{rmsd_rigidity:.3}"),
                        &format!("C_rmsd = {c_rmsd_score:.1}"),
                    );
                }
                ConcordanceSummary::InsufficientCoverage { n_eval, .. } => {
                    draw_metric_card(
                        ui,
                        theme,
                        "Rigidity Concordance",
                        "Insufficient",
                        &format!("{n_eval} < 10 assigned non-terminal residues"),
                    );
                }
            }
            draw_metric_card(
                ui,
                theme,
                "Concordant Ordered / Disordered",
                &format!(
                    "{} / {}",
                    rep.n_concordant_ordered, rep.n_concordant_disordered
                ),
                "pLDDT & S^2_exp agree",
            );
            draw_metric_card(
                ui,
                theme,
                "Cond. Folding / Hinge Switch",
                &format!(
                    "{} / {}",
                    rep.n_conditional_folding, rep.n_dynamic_hinge_or_alt
                ),
                "MoRF/crystal vs solution alt-state",
            );
            let sha_prefix: String = rep.exclusion_list_sha256.chars().take(12).collect();
            draw_metric_card(
                ui,
                theme,
                "HomologFirewall (<= 25%)",
                &format!("max id {:.1}%", rep.max_retained_seq_identity * 100.0),
                &format!("SHA256 {sha_prefix}..."),
            );
        });

        ui.add_space(8.0);
        let mut clicked_res: Option<(usize, i32)> = None;
        egui::Frame::NONE
            .fill(theme.card_bg)
            .stroke(egui::Stroke::new(1.0, theme.border))
            .corner_radius(5.0)
            .inner_margin(egui::Margin::same(8))
            .show(ui, |ui| {
                ui.label(
                    egui::RichText::new("Per-Residue Rigidity & Secondary-Structure Concordance")
                        .size(12.0)
                        .strong()
                        .color(theme.text_primary),
                );
                ui.add_space(4.0);
                egui::ScrollArea::vertical()
                    .id_salt("integrative_af_concordance_table")
                    .max_height(240.0)
                    .show(ui, |ui| {
                        egui::Grid::new("integrative_af_conc_grid")
                            .striped(true)
                            .min_col_width(48.0)
                            .spacing([12.0, 4.0])
                            .show(ui, |ui| {
                                for hdr in [
                                    "Seq",
                                    "Res",
                                    "3D SS",
                                    "pLDDT",
                                    "S^2_exp",
                                    "NMR SSP",
                                    "dTheta (deg)",
                                    "Concordance Class",
                                ] {
                                    ui.label(
                                        egui::RichText::new(hdr)
                                            .size(11.0)
                                            .strong()
                                            .color(theme.text_muted),
                                    );
                                }
                                ui.end_row();

                                for r in &rep.residues {
                                    let is_sel = self.selected_residue_idx == Some(r.struct_index);
                                    let seq_lbl = format!("{}", r.seq_id);
                                    if ui.selectable_label(is_sel, seq_lbl).clicked() {
                                        clicked_res = Some((r.struct_index, r.seq_id));
                                    }
                                    ui.label(&r.res_name);
                                    ui.label(format!("{:?}", r.state4_af));
                                    ui.label(
                                        r.plddt
                                            .map(|v| format!("{v:.1}"))
                                            .unwrap_or_else(|| "-".to_string()),
                                    );
                                    ui.label(
                                        r.s2_exp
                                            .map(|v| format!("{v:.2}"))
                                            .unwrap_or_else(|| "-".to_string()),
                                    );
                                    ui.label(
                                        r.ssp_nmr
                                            .map(|v| format!("{v:+.2}"))
                                            .unwrap_or_else(|| "-".to_string()),
                                    );
                                    ui.label(
                                        r.delta_theta_deg
                                            .map(|v| format!("{v:.1}"))
                                            .unwrap_or_else(|| "-".to_string()),
                                    );
                                    let cls_color = concordance_class_color(r.concordance_class);
                                    ui.label(
                                        egui::RichText::new(r.concordance_class.as_str())
                                            .size(11.0)
                                            .strong()
                                            .color(cls_color),
                                    );
                                    ui.end_row();
                                }
                            });
                    });
            });

        if let Some((seq_idx, seq_id)) = clicked_res {
            self.selected_residue_idx = Some(seq_idx);
            actions.push(IntegrativeAfAction::SelectResidue { seq_idx, seq_id });
        }
    }

    // ------------------------------------------------------------------------
    // Tab 2: Interactive N x N PAE Domain Heatmap & Tolman GDO Badges (`T21.5`)
    // ------------------------------------------------------------------------

    fn draw_pae_heatmap_tab(
        &mut self,
        ui: &mut egui::Ui,
        theme: &IntegrativeStudioTheme,
        actions: &mut Vec<IntegrativeAfAction>,
    ) {
        let Some(ref pae) = self.pae_matrix else {
            ui.label(
                egui::RichText::new(
                    "No PAE JSON matrix loaded. Load a PAE matrix or click 'Load 2-Domain AF Demo' to inspect spectral rigid domains.",
                )
                .color(theme.text_muted),
            );
            return;
        };

        let n = pae.n_residues;
        if n == 0 {
            return;
        }

        // Top row: Tolman GDO badges & F8 single-medium dipolar candor notice
        if let Some(ref gdo_rep) = self.tolman_gdo_report {
            ui.horizontal_wrapped(|ui| {
                let (bg, fg, regime_txt) = match gdo_rep.regime {
                    DomainMotionRegime::RigidlyCoupledDomains => (
                        theme.badge_ok_bg,
                        theme.badge_ok_fg,
                        "Tolman GDO: RigidlyCoupledDomains",
                    ),
                    DomainMotionRegime::DecoupledInterDomainMotion => (
                        theme.badge_alert_bg,
                        theme.badge_alert_fg,
                        "Tolman GDO Warning: DecoupledInterDomainMotion",
                    ),
                    DomainMotionRegime::InsufficientDomainRdcs => (
                        theme.badge_warn_bg,
                        theme.badge_warn_fg,
                        "Tolman GDO: InsufficientDomainRdcs (< 8 work RDCs/domain)",
                    ),
                };
                draw_pill_badge(ui, bg, fg, regime_txt);
                self.last_telemetry.gdo_badges_drawn += 1;

                for fit in &gdo_rep.domain_fits {
                    let badge = format!(
                        "D{} ({}) theta={:.4}+/-{:.4} | theta_d/theta_1={:.2} | Q_free={:.3}",
                        fit.domain_id,
                        fit.medium_id,
                        fit.gdo_theta,
                        fit.gdo_sigma_mc,
                        fit.ratio_to_domain_1,
                        fit.q_free
                    );
                    draw_pill_badge(ui, theme.card_bg, theme.text_primary, &badge);
                    self.last_telemetry.gdo_badges_drawn += 1;
                }
            });
            if let Some(ref warn) = gdo_rep.warning_message {
                ui.label(
                    egui::RichText::new(warn)
                        .size(11.0)
                        .color(theme.badge_alert_fg),
                );
            }
            ui.add_space(4.0);
        }

        // F8 Single-Medium Dipolar Candor note whenever < 2 media without PCS/PRE
        if self.loaded_rdc_media_count < 2 && !self.has_paramagnetic_pcs_or_pre {
            egui::Frame::NONE
                .fill(theme.badge_warn_bg)
                .stroke(egui::Stroke::new(1.0, theme.badge_warn_fg))
                .corner_radius(4.0)
                .inner_margin(egui::Margin::symmetric(8, 4))
                .show(ui, |ui| {
                    ui.label(
                        egui::RichText::new(
                            "Dipolar Candor (F8): A single RDC alignment medium carries intrinsic 4-fold Saupe principal-axis pi-rotation cone degeneracy and zero translational sensitivity; unambiguous 3D multi-domain placement requires >= 2 non-coaxial RDC media or translational+orientational PCS / PRE / inter-domain NOE restraints (20% ByPeptidePlane RdcFree).",
                        )
                        .size(11.0)
                        .color(theme.badge_warn_fg),
                    );
                });
            ui.add_space(6.0);
        }

        // Quick 1-click domain-pair selector buttons
        if let Some(ref part) = self.domain_partition {
            if !part.domains.is_empty() {
                ui.horizontal_wrapped(|ui| {
                    ui.label(
                        egui::RichText::new("Rigid Domains:")
                            .size(11.5)
                            .strong()
                            .color(theme.text_muted),
                    );
                    for d_a in &part.domains {
                        for d_b in &part.domains {
                            if d_b.domain_id < d_a.domain_id {
                                continue;
                            }
                            let is_pair =
                                self.selected_domain_pair == Some((d_a.domain_id, d_b.domain_id));
                            let lbl = if d_a.domain_id == d_b.domain_id {
                                format!(
                                    "D{} ({} aa, med {:.1} A)",
                                    d_a.domain_id,
                                    d_a.residue_indices.len(),
                                    d_a.median_intra_pae_angstrom
                                )
                            } else {
                                format!("Highlight D{} <-> D{}", d_a.domain_id, d_b.domain_id)
                            };
                            if ui.selectable_label(is_pair, lbl).clicked() {
                                let i_anchor = d_a
                                    .residue_indices
                                    .get(d_a.residue_indices.len() / 2)
                                    .copied()
                                    .unwrap_or(0);
                                let j_anchor = d_b
                                    .residue_indices
                                    .get(d_b.residue_indices.len() / 2)
                                    .copied()
                                    .unwrap_or(i_anchor);
                                self.selected_domain_pair = Some((d_a.domain_id, d_b.domain_id));
                                self.selected_pae_cell = Some((i_anchor, j_anchor));
                                actions.push(IntegrativeAfAction::HighlightDomainPair {
                                    domain_a: d_a.domain_id,
                                    domain_b: d_b.domain_id,
                                    seq_idx_i: i_anchor,
                                    seq_idx_j: j_anchor,
                                });
                            }
                        }
                    }
                });
                ui.add_space(6.0);
            }
        }

        // Interactive N x N PAE Heatmap Canvas + 0..31.75 A Colorbar
        let avail_w = ui.available_width().max(240.0);
        let side = (avail_w - 90.0).clamp(180.0, 340.0);
        ui.horizontal(|ui| {
            let (plot_rect, resp) =
                ui.allocate_exact_size(egui::vec2(side, side), egui::Sense::click_and_drag());
            self.last_pae_plot_rect = Some(plot_rect);
            let painter = ui.painter_at(plot_rect);
            painter.rect_filled(plot_rect, 2.0, theme.plot_bg);

            let bins = n.clamp(1, MAX_PAE_HEATMAP_BINS);
            let domain_count = self.domain_partition.as_ref().map_or(0, |p| p.domains.len());
            let first_bits = pae.get(0, 0).unwrap_or(0.0).to_bits() as u64;
            let mid_bits = pae.get(n / 2, n / 2).unwrap_or(0.0).to_bits() as u64;
            let last_bits = pae
                .get(n.saturating_sub(1), n.saturating_sub(1))
                .unwrap_or(0.0)
                .to_bits() as u64;
            let off_bits = pae
                .get(0, n.saturating_sub(1))
                .unwrap_or(0.0)
                .to_bits() as u64;
            let mut key = 0xcbf2_9ce4_8422_2325_u64;
            for word in [
                n as u64,
                bins as u64,
                (n * n) as u64,
                first_bits,
                mid_bits,
                last_bits,
                off_bits,
                domain_count as u64,
            ] {
                key ^= word;
                key = key.wrapping_mul(0x0100_0000_01b3);
            }

            if self.pae_texture_key != Some(key) || self.pae_texture.is_none() {
                let mut pixels = Vec::with_capacity(bins * bins);
                for bi in 0..bins {
                    let r0 = (bi * n) / bins;
                    let r1 = (((bi + 1) * n) / bins).max(r0 + 1).min(n);
                    for bj in 0..bins {
                        let c0 = (bj * n) / bins;
                        let c1 = (((bj + 1) * n) / bins).max(c0 + 1).min(n);
                        let mut sum = 0.0f32;
                        let mut cnt = 0usize;
                        for rr in r0..r1 {
                            for cc in c0..c1 {
                                if let Some(v) = pae.get(rr, cc) {
                                    sum += v;
                                    cnt += 1;
                                }
                            }
                        }
                        let mean_pae = if cnt > 0 {
                            sum / (cnt as f32)
                        } else {
                            PAE_COLORBAR_MAX_ANGSTROM
                        };
                        pixels.push(pae_green_to_white_color(mean_pae));
                    }
                }
                let color_image = egui::ColorImage {
                    size: [bins, bins],
                    source_size: egui::vec2(bins as f32, bins as f32),
                    pixels,
                };
                let tex = ui.ctx().load_texture(
                    "pae_heatmap_tex",
                    color_image,
                    egui::TextureOptions::NEAREST,
                );
                self.pae_texture = Some(tex);
                self.pae_texture_key = Some(key);
                self.pae_texture_uploads += 1;
            }

            if let Some(ref tex) = self.pae_texture {
                painter.image(
                    tex.id(),
                    plot_rect,
                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                    egui::Color32::WHITE,
                );
            }
            self.last_telemetry.pae_cells_drawn = bins * bins;
            self.last_telemetry.pae_texture_uploads = self.pae_texture_uploads;

            // Overlay spectral rigid-domain bounding boxes (`T21.5`)
            if let Some(ref part) = self.domain_partition {
                let scale = plot_rect.width() / (n as f32);
                for dom in &part.domains {
                    let Some(&i_min) = dom.residue_indices.first() else {
                        continue;
                    };
                    let Some(&i_max) = dom.residue_indices.last() else {
                        continue;
                    };
                    let x0 = plot_rect.min.x + (i_min as f32) * scale;
                    let y0 = plot_rect.min.y + (i_min as f32) * scale;
                    let x1 = plot_rect.min.x + ((i_max + 1) as f32) * scale;
                    let y1 = plot_rect.min.y + ((i_max + 1) as f32) * scale;
                    let box_rect =
                        egui::Rect::from_min_max(egui::pos2(x0, y0), egui::pos2(x1, y1));
                    let is_sel = self
                        .selected_domain_pair
                        .is_some_and(|(a, b)| a == dom.domain_id || b == dom.domain_id);
                    let stroke_col = if is_sel {
                        theme.domain_box_selected
                    } else {
                        theme.domain_box_primary
                    };
                    painter.rect_stroke(
                        box_rect,
                        0.0,
                        egui::Stroke::new(if is_sel { 2.4 } else { 1.6 }, stroke_col),
                        egui::StrokeKind::Inside,
                    );
                    painter.text(
                        egui::pos2(x0 + 4.0, y0 + 3.0),
                        egui::Align2::LEFT_TOP,
                        format!("D{}", dom.domain_id),
                        egui::FontId::proportional(11.0),
                        egui::Color32::WHITE,
                    );
                    self.last_telemetry.pae_domain_boxes_drawn += 1;
                }
            }

            painter.rect_stroke(
                plot_rect,
                2.0,
                egui::Stroke::new(1.0, theme.border),
                egui::StrokeKind::Outside,
            );

            // Handle click & hover on PAE heatmap
            if let Some(pos) = resp.interact_pointer_pos().or_else(|| resp.hover_pos()) {
                if plot_rect.contains(pos) {
                    let j = (((pos.x - plot_rect.min.x) / plot_rect.width()) * (n as f32))
                        .floor()
                        .clamp(0.0, (n - 1) as f32) as usize;
                    let i = (((pos.y - plot_rect.min.y) / plot_rect.height()) * (n as f32))
                        .floor()
                        .clamp(0.0, (n - 1) as f32) as usize;
                    let raw_pae = pae.get(i, j).unwrap_or(0.0);
                    let sym_pae = pae.symmetrized(i, j).unwrap_or(raw_pae as f64);
                    let dom_i = self
                        .domain_partition
                        .as_ref()
                        .and_then(|p| p.domain_labels.get(i).copied())
                        .unwrap_or(0);
                    let dom_j = self
                        .domain_partition
                        .as_ref()
                        .and_then(|p| p.domain_labels.get(j).copied())
                        .unwrap_or(0);

                    if resp.clicked() {
                        self.selected_pae_cell = Some((i, j));
                        let pair = if dom_i <= dom_j {
                            (dom_i, dom_j)
                        } else {
                            (dom_j, dom_i)
                        };
                        self.selected_domain_pair = Some(pair);
                        actions.push(IntegrativeAfAction::HighlightDomainPair {
                            domain_a: dom_i,
                            domain_b: dom_j,
                            seq_idx_i: i,
                            seq_idx_j: j,
                        });
                    }

                    resp.on_hover_text(format!(
                        "Aligned res {} -> Scored res {}\nPAE(i, j) = {:.2} A | sym PAE = {:.2} A\nDomains: D{} <-> D{}",
                        i + 1,
                        j + 1,
                        raw_pae,
                        sym_pae,
                        dom_i,
                        dom_j
                    ));
                }
            }

            // Vertical 0..31.75 A Colorbar Legend
            ui.add_space(10.0);
            let (bar_rect, _) =
                ui.allocate_exact_size(egui::vec2(56.0, side), egui::Sense::hover());
            let bar_painter = ui.painter_at(bar_rect);
            let strip = egui::Rect::from_min_max(
                bar_rect.min,
                egui::pos2(bar_rect.min.x + 14.0, bar_rect.max.y),
            );
            let n_steps = 32usize;
            let step_h = strip.height() / (n_steps as f32);
            for s in 0..n_steps {
                let frac = (s as f32) / ((n_steps - 1) as f32);
                let val_a = frac * PAE_COLORBAR_MAX_ANGSTROM;
                let r = egui::Rect::from_min_max(
                    egui::pos2(strip.min.x, strip.min.y + (s as f32) * step_h),
                    egui::pos2(strip.max.x, strip.min.y + ((s + 1) as f32) * step_h),
                );
                bar_painter.rect_filled(r, 0.0, pae_green_to_white_color(val_a));
            }
            bar_painter.rect_stroke(
                strip,
                1.0,
                egui::Stroke::new(1.0, theme.border),
                egui::StrokeKind::Outside,
            );
            for (frac, lbl) in [(0.0f32, "0 A"), (0.5f32, "16 A"), (1.0f32, "31.8 A")] {
                let y = strip.min.y + frac * (strip.height() - 10.0);
                bar_painter.text(
                    egui::pos2(strip.max.x + 5.0, y),
                    egui::Align2::LEFT_TOP,
                    lbl,
                    egui::FontId::proportional(11.0),
                    theme.text_muted,
                );
            }
        });
    }

    // ------------------------------------------------------------------------
    // Tab 3: Co-Folding Restraint Preview & Multi-Pose Complex Ranker (`T21.6a`)
    // ------------------------------------------------------------------------

    fn draw_cofold_tab(
        &mut self,
        ui: &mut egui::Ui,
        theme: &IntegrativeStudioTheme,
        actions: &mut Vec<IntegrativeAfAction>,
    ) {
        ui.horizontal_wrapped(|ui| {
            ui.selectable_value(
                &mut self.preview_dialect,
                CofoldPreviewDialect::Boltz2Yaml,
                "Boltz-2 YAML (v1 pocket + contact)",
            );
            ui.selectable_value(
                &mut self.preview_dialect,
                CofoldPreviewDialect::AlphaFold3Json,
                "AlphaFold 3 JSON (v2 + sidecar)",
            );
            ui.separator();
            let surf_txt = format!(
                "Surface Pocket (rSASA >= 0.20): {} res",
                self.surface_pocket_seq_ids.len()
            );
            draw_pill_badge(ui, theme.badge_ok_bg, theme.badge_ok_fg, &surf_txt);
            let core_txt = format!(
                "Buried Allostery (rSASA < 0.20): {} res",
                self.buried_allosteric_seq_ids.len()
            );
            draw_pill_badge(ui, theme.badge_warn_bg, theme.badge_warn_fg, &core_txt);
        });

        ui.add_space(6.0);
        if self.preview_dialect == CofoldPreviewDialect::AlphaFold3Json {
            ui.label(
                egui::RichText::new(&self.af3_sidecar_note)
                    .size(11.0)
                    .color(theme.text_muted),
            );
            ui.add_space(3.0);
        }

        let preview_text = match self.preview_dialect {
            CofoldPreviewDialect::Boltz2Yaml => &self.boltz2_yaml_preview,
            CofoldPreviewDialect::AlphaFold3Json => &self.af3_json_preview,
        };
        if !preview_text.is_empty() {
            let mut ro_copy = preview_text.clone();
            egui::ScrollArea::vertical()
                .id_salt("integrative_cofold_schema_preview")
                .max_height(135.0)
                .show(ui, |ui| {
                    ui.add(
                        egui::TextEdit::multiline(&mut ro_copy)
                            .desired_rows(5)
                            .desired_width(f32::INFINITY)
                            .font(egui::TextStyle::Monospace),
                    );
                });
            ui.add_space(8.0);
        }

        // Pose Ranker Verdict Banner
        ui.horizontal_wrapped(|ui| {
            let (v_bg, v_fg) = match self.pose_verdict {
                CofoldRankVerdictView::TopPoseSelected => (theme.badge_ok_bg, theme.badge_ok_fg),
                CofoldRankVerdictView::InconclusiveOrAllViolated => {
                    (theme.badge_alert_bg, theme.badge_alert_fg)
                }
                CofoldRankVerdictView::Unscored => (theme.card_bg, theme.text_muted),
            };
            draw_pill_badge(ui, v_bg, v_fg, self.pose_verdict.label());
        });

        ui.add_space(4.0);
        let mut order: Vec<usize> = (0..self.pose_rows.len()).collect();
        order.sort_by(|&a, &b| {
            let ra = &self.pose_rows[a];
            let rb = &self.pose_rows[b];
            match self.pose_sort {
                PoseSortColumn::TotalScoreDesc => rb.total_score.total_cmp(&ra.total_score),
                PoseSortColumn::CspScoreDesc => rb.csp_score.total_cmp(&ra.csp_score),
                PoseSortColumn::NoePreViolationAsc => {
                    ra.noe_pre_viol_energy.total_cmp(&rb.noe_pre_viol_energy)
                }
                PoseSortColumn::RdcQFreeAsc => ra
                    .rdc_q_free
                    .unwrap_or(f64::INFINITY)
                    .total_cmp(&rb.rdc_q_free.unwrap_or(f64::INFINITY)),
            }
        });

        let mut clicked_pose: Option<(usize, String, Option<usize>)> = None;
        egui::Frame::NONE
            .fill(theme.card_bg)
            .stroke(egui::Stroke::new(1.0, theme.border))
            .corner_radius(5.0)
            .inner_margin(egui::Margin::same(8))
            .show(ui, |ui| {
                egui::Grid::new("integrative_cofold_pose_grid")
                    .striped(true)
                    .spacing([12.0, 4.0])
                    .show(ui, |ui| {
                        for hdr in [
                            "Rank",
                            "Pose",
                            "S_total",
                            "S_CSP",
                            "CSP AUC",
                            "E_viol",
                            "max Viol (A)",
                            "RDC Q_free",
                            "PCS Q",
                        ] {
                            ui.label(
                                egui::RichText::new(hdr)
                                    .size(11.0)
                                    .strong()
                                    .color(theme.text_muted),
                            );
                        }
                        ui.end_row();

                        for (disp_rank, &orig_idx) in order.iter().enumerate() {
                            let row = &self.pose_rows[orig_idx];
                            let is_sel = self.selected_pose_idx == Some(orig_idx);
                            ui.label(format!("#{}", disp_rank + 1));
                            if ui.selectable_label(is_sel, &row.pose_id).clicked() {
                                clicked_pose =
                                    Some((orig_idx, row.pose_id.clone(), row.highlight_seq_idx));
                            }
                            ui.label(format!("{:+.2}", row.total_score));
                            ui.label(format!("{:+.2}", row.csp_score));
                            ui.label(
                                row.csp_roc_auc
                                    .map(|v| format!("{v:.2}"))
                                    .unwrap_or_else(|| "-".to_string()),
                            );
                            ui.label(format!("{:.2}", row.noe_pre_viol_energy));
                            ui.label(format!("{:.2}", row.max_viol_angstrom));
                            ui.label(
                                row.rdc_q_free
                                    .map(|v| format!("{v:.3}"))
                                    .unwrap_or_else(|| "-".to_string()),
                            );
                            ui.label(
                                row.pcs_q
                                    .map(|v| format!("{v:.3}"))
                                    .unwrap_or_else(|| "-".to_string()),
                            );
                            ui.end_row();
                        }
                    });
            });

        if let Some((p_idx, p_id, hi_seq)) = clicked_pose {
            self.selected_pose_idx = Some(p_idx);
            actions.push(IntegrativeAfAction::SelectCofoldPose {
                pose_index: p_idx,
                pose_id: p_id,
                highlight_seq_idx: hi_seq,
            });
        }
    }

    // ------------------------------------------------------------------------
    // Tab 4: Bayesian Excited-State Hypothesis Explorer (`T21.6b`)
    // ------------------------------------------------------------------------

    fn draw_excited_state_tab(
        &mut self,
        ui: &mut egui::Ui,
        theme: &IntegrativeStudioTheme,
        actions: &mut Vec<IntegrativeAfAction>,
    ) {
        if self.excited_bars.is_empty() {
            ui.label(
                egui::RichText::new(
                    "No excited-state CEST / CPMG differential shifts classified yet. Click 'Load 2-Domain AF Demo' to explore BIC / Bayes-factor hypothesis ranking.",
                )
                .color(theme.text_muted),
            );
            return;
        }

        // Verdict banner
        ui.horizontal_wrapped(|ui| match &self.excited_verdict {
            ExcitedVerdictView::Committed {
                winner_label,
                two_ln_bf_12,
                two_ln_bf_10,
                reduced_chi2,
            } => {
                let mut badge = String::new();
                let _ = write!(
                    badge,
                    "Committed: {winner_label} | 2 ln BF(1,2) = {two_ln_bf_12:.1} | 2 ln BF(1,H0) = {two_ln_bf_10:.1} | chi^2_nu = {reduced_chi2:.2} (N = {})",
                    self.excited_n_obs
                );
                draw_pill_badge(ui, theme.badge_ok_bg, theme.badge_ok_fg, &badge);
            }
            ExcitedVerdictView::AmbiguousOrUnmodeledState {
                reason,
                two_ln_bf_12,
                two_ln_bf_10,
            } => {
                let mut badge = String::new();
                let _ = write!(
                    badge,
                    "Abstained (AmbiguousOrUnmodeledState): {reason} | 2 ln BF(1,2) = {two_ln_bf_12:.1} | 2 ln BF(1,H0) = {two_ln_bf_10:.1}"
                );
                draw_pill_badge(ui, theme.badge_warn_bg, theme.badge_warn_fg, &badge);
            }
            ExcitedVerdictView::Unrun => {}
        });
        ui.add_space(8.0);

        ui.columns(2, |cols| {
            // Left column: Kass-Raftery `2 ln BF` vs H0 & BIC Horizontal Bar Chart
            cols[0].label(
                egui::RichText::new("Hypothesis Evidence 2 ln BF vs H0 (Kass & Raftery 1995)")
                    .size(11.5)
                    .strong()
                    .color(theme.text_primary),
            );
            let bar_h = ((self.excited_bars.len() as f32) * 34.0 + 28.0).max(160.0);
            let bar_w = cols[0].available_width().max(200.0);
            let (chart_rect, chart_resp) =
                cols[0].allocate_exact_size(egui::vec2(bar_w, bar_h), egui::Sense::click());
            self.last_bic_chart_rect = Some(chart_rect);
            let cp = cols[0].painter_at(chart_rect);
            cp.rect_filled(chart_rect, 4.0, theme.plot_bg);

            let max_bf = self
                .excited_bars
                .iter()
                .map(|b| b.two_ln_bf_vs_null)
                .fold(12.0_f64, f64::max);
            let left_gutter = 132.0_f32.min(chart_rect.width() * 0.46);
            let plot_x0 = chart_rect.min.x + left_gutter;
            let plot_x1 = (chart_rect.max.x - 54.0).max(plot_x0 + 24.0);
            let span_x = plot_x1 - plot_x0;

            // Reference threshold lines at 2 ln BF = 6.0 and 10.0
            for (thresh, lbl) in [(6.0_f64, "6"), (10.0_f64, "10")] {
                let tx = plot_x0 + ((thresh / max_bf).clamp(0.0, 1.0) as f32) * span_x;
                cp.line_segment(
                    [
                        egui::pos2(tx, chart_rect.min.y + 6.0),
                        egui::pos2(tx, chart_rect.max.y - 16.0),
                    ],
                    egui::Stroke::new(1.0, theme.grid_line),
                );
                cp.text(
                    egui::pos2(tx, chart_rect.max.y - 14.0),
                    egui::Align2::CENTER_TOP,
                    lbl,
                    egui::FontId::proportional(11.0),
                    theme.text_muted,
                );
            }

            let mut clicked_hyp: Option<(usize, i32)> = None;
            for (idx, bar) in self.excited_bars.iter().enumerate() {
                let row_y = chart_rect.min.y + 10.0 + (idx as f32) * 32.0;
                cp.text(
                    egui::pos2(chart_rect.min.x + 6.0, row_y + 8.0),
                    egui::Align2::LEFT_CENTER,
                    &bar.family_label,
                    egui::FontId::proportional(11.0),
                    theme.text_primary,
                );
                let frac = (bar.two_ln_bf_vs_null.max(0.0) / max_bf).clamp(0.0, 1.0) as f32;
                let w = (frac * span_x).max(2.0);
                let rect = egui::Rect::from_min_max(
                    egui::pos2(plot_x0, row_y),
                    egui::pos2(plot_x0 + w, row_y + 17.0),
                );
                let fill = if idx == 0 {
                    theme.accent_primary
                } else {
                    theme.grid_line
                };
                cp.rect_filled(rect, 3.0, fill);
                cp.text(
                    egui::pos2(rect.max.x + 4.0, row_y + 8.5),
                    egui::Align2::LEFT_CENTER,
                    format!("{:.1}", bar.two_ln_bf_vs_null),
                    egui::FontId::proportional(11.0),
                    theme.text_muted,
                );
                self.last_telemetry.bic_bars_drawn += 1;

                if chart_resp.clicked() {
                    if let Some(pos) = chart_resp.interact_pointer_pos() {
                        if pos.y >= row_y && pos.y <= row_y + 22.0 {
                            if let (Some(si), Some(sid)) =
                                (bar.representative_seq_idx, bar.representative_seq_id)
                            {
                                clicked_hyp = Some((si, sid));
                            }
                        }
                    }
                }
            }
            cp.rect_stroke(
                chart_rect,
                4.0,
                egui::Stroke::new(1.0, theme.border),
                egui::StrokeKind::Outside,
            );
            if let Some((seq_idx, seq_id)) = clicked_hyp {
                actions.push(IntegrativeAfAction::JumpToDynamicsResidue {
                    seq_idx,
                    seq_id,
                    prefer_cpmg_unsigned: false,
                });
            }

            // Right column: Interactive Delta_omega_obs vs Delta_omega_calc Parity Scatter Plot
            cols[1].label(
                egui::RichText::new("Differential Shift Parity (Delta_omega_obs vs calc, ppm)")
                    .size(11.5)
                    .strong()
                    .color(theme.text_primary),
            );
            let p_side = cols[1].available_width().clamp(160.0, 230.0);
            let (par_rect, par_resp) =
                cols[1].allocate_exact_size(egui::vec2(p_side, bar_h), egui::Sense::click_and_drag());
            self.last_parity_plot_rect = Some(par_rect);
            let pp = cols[1].painter_at(par_rect);
            pp.rect_filled(par_rect, 4.0, theme.plot_bg);

            let pad = 22.0_f32;
            let inner = egui::Rect::from_min_max(
                egui::pos2(par_rect.min.x + pad, par_rect.min.y + 12.0),
                egui::pos2(par_rect.max.x - 12.0, par_rect.max.y - pad),
            );
            let (mut v_min, mut v_max) = (-3.0_f64, 3.0_f64);
            if !self.excited_parity_points.is_empty() {
                v_min = self
                    .excited_parity_points
                    .iter()
                    .flat_map(|p| [p.obs_ppm, p.calc_ppm])
                    .fold(0.0_f64, f64::min)
                    - 0.4;
                v_max = self
                    .excited_parity_points
                    .iter()
                    .flat_map(|p| [p.obs_ppm, p.calc_ppm])
                    .fold(0.0_f64, f64::max)
                    + 0.4;
                if (v_max - v_min).abs() < 0.5 {
                    v_min -= 0.5;
                    v_max += 0.5;
                }
            }
            let map_pt = |calc: f64, obs: f64| -> egui::Pos2 {
                let tx = ((calc - v_min) / (v_max - v_min)).clamp(0.0, 1.0) as f32;
                let ty = ((obs - v_min) / (v_max - v_min)).clamp(0.0, 1.0) as f32;
                egui::pos2(
                    inner.min.x + tx * inner.width(),
                    inner.max.y - ty * inner.height(),
                )
            };

            // y = x identity diagonal
            pp.line_segment(
                [map_pt(v_min, v_min), map_pt(v_max, v_max)],
                egui::Stroke::new(1.2, theme.parity_diagonal),
            );

            let pointer_pos = par_resp
                .interact_pointer_pos()
                .or_else(|| par_resp.hover_pos());
            let mut hovered_idx: Option<usize> = None;
            let mut best_d2 = 120.0_f32;

            for (idx, pt) in self.excited_parity_points.iter().enumerate() {
                let screen_pt = map_pt(pt.calc_ppm, pt.obs_ppm);
                if let Some(mp) = pointer_pos {
                    let d2 = mp.distance_sq(screen_pt);
                    if d2 < best_d2 {
                        best_d2 = d2;
                        hovered_idx = Some(idx);
                    }
                }
                let is_sel = self.selected_parity_idx == Some(idx);
                let col = if is_sel {
                    theme.parity_point_selected
                } else if pt.is_signed {
                    theme.parity_point_signed
                } else {
                    theme.parity_point_unsigned
                };
                pp.circle_filled(screen_pt, if is_sel { 5.2 } else { 4.0 }, col);
                self.last_telemetry.parity_points_drawn += 1;
            }

            pp.rect_stroke(
                par_rect,
                4.0,
                egui::Stroke::new(1.0, theme.border),
                egui::StrokeKind::Outside,
            );

            if let Some(h_idx) = hovered_idx {
                let pt = &self.excited_parity_points[h_idx];
                let mode_lbl = if pt.is_signed {
                    "Signed CEST/R1rho"
                } else {
                    "Unsigned CPMG |dw|"
                };
                if par_resp.clicked() {
                    self.selected_parity_idx = Some(h_idx);
                    self.selected_residue_idx = Some(pt.seq_idx);
                    actions.push(IntegrativeAfAction::JumpToDynamicsResidue {
                        seq_idx: pt.seq_idx,
                        seq_id: pt.seq_id,
                        prefer_cpmg_unsigned: !pt.is_signed,
                    });
                }
                par_resp.on_hover_text(format!(
                    "{}{} {} ({mode_lbl})\nobs = {:+.3} ppm | calc = {:+.3} ppm (+/- {:.2})\nClick to jump to Dynamics Studio & 3D ribbon",
                    pt.res_name, pt.seq_id, pt.atom_name, pt.obs_ppm, pt.calc_ppm, pt.sigma_tot_ppm
                ));
            }
        });
    }
}

// ============================================================================
// 4. Workbench Coupling Helper (`Mol3dViewerPanel` + `SsRibbonState` + Crosshair)
// ============================================================================

/// Apply an [`IntegrativeAfAction`] directly to the workbench's 3D viewer, 3-track SS ribbon,
/// 2D crosshair, and sequence ribbon (`wasm32`-clean, testable headlessly without `main.rs`).
#[allow(clippy::too_many_arguments)]
pub fn apply_integrative_af_action(
    action: &IntegrativeAfAction,
    panel: &IntegrativeAfPanel,
    ref_data: Option<&Reference>,
    nmr_seq: &[String],
    nmr_seq_ids: &[i32],
    crosshair: &mut SyncCrosshair,
    seq_ribbon: &mut SequenceRibbonState,
    ss_ribbon: &mut SsRibbonState,
    mol3d: &mut Mol3dViewerPanel,
) {
    match *action {
        IntegrativeAfAction::SelectResidue { seq_idx, .. } => {
            ss_ribbon.selected_residue = Some(seq_idx);
            crate::ss_ribbon_widget::apply_ss_ribbon_click_sync(
                seq_idx, ref_data, crosshair, seq_ribbon, mol3d,
            );
            mol3d.flush_pending_updates();
        }
        IntegrativeAfAction::SetMol3dTrack {
            track,
            putty_enabled,
        } => {
            panel.sync_to_ribbon_and_mol3d(ss_ribbon, mol3d, nmr_seq, nmr_seq_ids);
            mol3d.track_data.active_track = track;
            mol3d.track_data.putty_enabled = putty_enabled;
            mol3d.rebuild_for_test();
        }
        IntegrativeAfAction::HighlightDomainPair {
            domain_a,
            domain_b,
            seq_idx_i,
            seq_idx_j,
        } => {
            panel.sync_to_ribbon_and_mol3d(ss_ribbon, mol3d, nmr_seq, nmr_seq_ids);
            ss_ribbon.selected_residue = Some(seq_idx_i);
            seq_ribbon.selected_residue = seq_idx_i;
            mol3d.highlight_sequence_residue(Some(seq_idx_i), true);

            // Install an inter-residue / domain-pair visual guide vector in `mol3d` when `i != j`
            if let Some(ens) = mol3d.ensemble.as_ref() {
                let seq_id_i = ens
                    .primary_seq_ids
                    .get(seq_idx_i)
                    .copied()
                    .unwrap_or((seq_idx_i + 1) as i32);
                let seq_id_j = ens
                    .primary_seq_ids
                    .get(seq_idx_j)
                    .copied()
                    .unwrap_or((seq_idx_j + 1) as i32);
                if seq_id_i != seq_id_j {
                    let mut r = DistanceRestraint3d::pair(
                        90_001,
                        RestraintKind::Noe,
                        RestraintAtomSelector::new(None, seq_id_i, "CA"),
                        RestraintAtomSelector::new(None, seq_id_j, "CA"),
                        0.0,
                        100.0,
                    );
                    r.label = format!(
                        "PAE Domain Pair D{domain_a}:{} <-> D{domain_b}:{}",
                        seq_id_i, seq_id_j
                    );
                    mol3d.set_restraints(vec![r]);
                }
            }
            mol3d.flush_pending_updates();
        }
        IntegrativeAfAction::SyncStructureAndConcordance => {
            panel.sync_to_ribbon_and_mol3d(ss_ribbon, mol3d, nmr_seq, nmr_seq_ids);
        }
        IntegrativeAfAction::JumpToDynamicsResidue { seq_idx, .. } => {
            ss_ribbon.selected_residue = Some(seq_idx);
            crate::ss_ribbon_widget::apply_ss_ribbon_click_sync(
                seq_idx, ref_data, crosshair, seq_ribbon, mol3d,
            );
            mol3d.flush_pending_updates();
        }
        IntegrativeAfAction::SelectCofoldPose {
            highlight_seq_idx: Some(seq_idx),
            ..
        } => {
            ss_ribbon.selected_residue = Some(seq_idx);
            mol3d.highlight_sequence_residue(Some(seq_idx), true);
            mol3d.flush_pending_updates();
        }
        IntegrativeAfAction::SelectCofoldPose {
            highlight_seq_idx: None,
            ..
        } => {}
    }
}

// ============================================================================
// 5. Internal Drawing & Synthetic 2-Domain AlphaFold Fixture Builders
// ============================================================================

fn draw_metric_card(
    ui: &mut egui::Ui,
    theme: &IntegrativeStudioTheme,
    title: &str,
    value: &str,
    sub: &str,
) {
    egui::Frame::NONE
        .fill(theme.card_bg)
        .stroke(egui::Stroke::new(1.0, theme.border))
        .corner_radius(5.0)
        .inner_margin(egui::Margin::symmetric(10, 6))
        .show(ui, |ui| {
            ui.vertical(|ui| {
                ui.label(
                    egui::RichText::new(title)
                        .size(11.0)
                        .color(theme.text_muted),
                );
                ui.label(
                    egui::RichText::new(value)
                        .size(14.5)
                        .strong()
                        .color(theme.text_primary),
                );
                ui.label(egui::RichText::new(sub).size(11.0).color(theme.text_muted));
            });
        });
}

fn draw_pill_badge(ui: &mut egui::Ui, bg: egui::Color32, fg: egui::Color32, text: &str) {
    egui::Frame::NONE
        .fill(bg)
        .stroke(egui::Stroke::new(1.0, fg.gamma_multiply(0.45)))
        .corner_radius(4.0)
        .inner_margin(egui::Margin::symmetric(7, 3))
        .show(ui, |ui| {
            ui.label(egui::RichText::new(text).size(11.0).strong().color(fg));
        });
}

fn concordance_class_color(cls: AfNmrConcordanceClass) -> egui::Color32 {
    match cls {
        AfNmrConcordanceClass::ConcordantOrdered => egui::Color32::from_rgb(0x16, 0xA3, 0x4A),
        AfNmrConcordanceClass::ConcordantDisordered => egui::Color32::from_rgb(0x64, 0x74, 0x8B),
        AfNmrConcordanceClass::ConditionalFoldingOrCrystalArtifact => {
            egui::Color32::from_rgb(0xD9, 0x77, 0x06)
        }
        AfNmrConcordanceClass::DynamicHingeOrAltState => egui::Color32::from_rgb(0xDC, 0x26, 0x26),
        AfNmrConcordanceClass::UnassignedNmr => egui::Color32::from_rgb(148, 163, 184),
        #[allow(unreachable_patterns)]
        _ => egui::Color32::from_rgb(148, 163, 184),
    }
}

/// Build a self-contained 52-residue 2-domain AlphaFold ModelArchive `.cif` (`_ma_qa_metric_local`
/// `pLDDT`), `52 x 52` `PAE` JSON (`Domain 1 = 1..23`, `Linker = 24..29`, `Domain 2 = 30..52`),
/// matching assigned [`Reference`] shifts, and single-medium `N-H` RDC TSV.
fn build_synthetic_two_domain_af_demo() -> (String, String, Reference, String) {
    const SEQ1: &[u8; 52] = b"MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLED";
    let aa3_of = |c: u8| -> &'static str {
        match c {
            b'A' => "ALA",
            b'D' => "ASP",
            b'E' => "GLU",
            b'F' => "PHE",
            b'G' => "GLY",
            b'I' => "ILE",
            b'K' => "LYS",
            b'L' => "LEU",
            b'M' => "MET",
            b'N' => "ASN",
            b'P' => "PRO",
            b'Q' => "GLN",
            b'R' => "ARG",
            b'S' => "SER",
            b'T' => "THR",
            b'V' => "VAL",
            _ => "ALA",
        }
    };

    let n = SEQ1.len();
    let mut seq3 = Vec::with_capacity(n);
    let mut seq_ids = Vec::with_capacity(n);
    for (i, &b) in SEQ1.iter().enumerate() {
        seq3.push(aa3_of(b).to_string());
        seq_ids.push((i + 1) as i32);
    }

    let mut cif = String::with_capacity(18_000);
    cif.push_str("data_AF_DEMO_2DOM\n#\n");
    cif.push_str(
        "loop_\n_ma_qa_metric.id\n_ma_qa_metric.name\n_ma_qa_metric.mode\n_ma_qa_metric.type\n",
    );
    cif.push_str("1 pLDDT local pLDDT\n#\n");

    // Place helical-like backbone geometry per domain with intact C'-N peptide bonds
    cif.push_str("loop_\n_atom_site.group_PDB\n_atom_site.id\n_atom_site.type_symbol\n_atom_site.label_atom_id\n_atom_site.label_comp_id\n_atom_site.label_asym_id\n_atom_site.label_seq_id\n_atom_site.Cartn_x\n_atom_site.Cartn_y\n_atom_site.Cartn_z\n_atom_site.occupancy\n_atom_site.B_iso_or_equiv\n_atom_site.pdbx_PDB_model_num\n");

    let mut plddts = Vec::with_capacity(n);
    let mut atom_id = 1usize;
    let mut nh_rdc_tsv = String::from(
        "medium_id\tseq_id_1\tres_name_1\tatom_name_1\tseq_id_2\tres_name_2\tatom_name_2\tbond_type\td_obs_hz\tsigma_hz\n",
    );

    for (i, res) in seq3.iter().enumerate() {
        let in_d1 = i < 23;
        let in_d2 = i >= 29;
        let plddt = if in_d1 {
            92.4 - ((i % 4) as f64) * 1.2
        } else if in_d2 {
            89.6 - ((i % 5) as f64) * 1.1
        } else {
            43.5 + ((i % 3) as f64) * 2.5
        };
        plddts.push(plddt);

        let theta = (i as f64) * 1.7453292519943295; // ~100 deg alpha-helix rise
        let domain_shift_x = if in_d2 { 14.0 } else { 0.0 };
        let ca_x = domain_shift_x + 2.30 * theta.cos();
        let ca_y = 2.30 * theta.sin();
        let ca_z = (i as f64) * 1.50;
        let n_pos = [
            ca_x - 0.95 * theta.sin(),
            ca_y + 0.95 * theta.cos(),
            ca_z - 0.55,
        ];
        let ca_pos = [ca_x, ca_y, ca_z];
        let c_pos = [
            ca_x + 0.92 * theta.sin(),
            ca_y - 0.92 * theta.cos(),
            ca_z + 0.58,
        ];
        let o_pos = [c_pos[0] + 0.35, c_pos[1] - 0.25, c_pos[2] + 1.15];
        let cb_pos = [
            ca_x + 1.25 * theta.cos(),
            ca_y + 1.25 * theta.sin(),
            ca_z - 0.42,
        ];
        let h_pos = [
            n_pos[0] - 0.48 * theta.cos(),
            n_pos[1] - 0.48 * theta.sin(),
            n_pos[2] - 0.88,
        ];

        let sid = i + 1;
        let mut emit_atom = |aname: &str, elem: &str, p: [f64; 3]| {
            let _ = writeln!(
                cif,
                "ATOM {atom_id} {elem} {aname} {res} A {sid} {:.3} {:.3} {:.3} 1.00 {plddt:.2} 1",
                p[0], p[1], p[2]
            );
            atom_id += 1;
        };
        emit_atom("N", "N", n_pos);
        if res != "PRO" {
            emit_atom("H", "H", h_pos);
        }
        emit_atom("CA", "C", ca_pos);
        emit_atom("C", "C", c_pos);
        emit_atom("O", "O", o_pos);
        if res != "GLY" {
            emit_atom("CB", "C", cb_pos);
        }

        // Synthetic axial Saupe tensor RDCs (Da = 10 Hz, R = 0.2; Domain 2 scaled by 0.55 for decoupled motion showcase)
        if res != "PRO" && (in_d1 || in_d2) {
            let dx = h_pos[0] - n_pos[0];
            let dy = h_pos[1] - n_pos[1];
            let dz = h_pos[2] - n_pos[2];
            let r2 = (dx * dx + dy * dy + dz * dz).max(1e-9);
            let ux = dx / r2.sqrt();
            let uy = dy / r2.sqrt();
            let uz = dz / r2.sqrt();
            let scale = if in_d2 { 0.48 } else { 1.0 };
            let rdc_val = scale * 10.0 * ((3.0 * uz * uz - 1.0) + 0.30 * (ux * ux - uy * uy));
            let _ = writeln!(
                nh_rdc_tsv,
                "pf1\t{sid}\t{res}\tN\t{sid}\t{res}\tH\t1D_NH\t{rdc_val:.3}\t0.20"
            );
        }
    }
    cif.push_str("#\nloop_\n_ma_qa_metric_local.ordinal_id\n_ma_qa_metric_local.model_id\n_ma_qa_metric_local.label_asym_id\n_ma_qa_metric_local.label_seq_id\n_ma_qa_metric_local.label_comp_id\n_ma_qa_metric_local.metric_id\n_ma_qa_metric_local.metric_value\n");
    for (i, &pv) in plddts.iter().enumerate() {
        let _ = writeln!(cif, "{} 1 A {} {} 1 {pv:.2}", i + 1, i + 1, seq3[i]);
    }
    cif.push_str("#\n");

    // Build 52 x 52 AFDB v4 PAE JSON (`0..31.75 A`)
    let mut pae_json = String::with_capacity(24_000);
    pae_json.push_str("[{\"max_predicted_aligned_error\":31.75,\"predicted_aligned_error\":[");
    for i in 0..n {
        if i > 0 {
            pae_json.push(',');
        }
        pae_json.push('[');
        let d_i = if i < 23 {
            1
        } else if i >= 29 {
            2
        } else {
            0
        };
        for j in 0..n {
            if j > 0 {
                pae_json.push(',');
            }
            let d_j = if j < 23 {
                1
            } else if j >= 29 {
                2
            } else {
                0
            };
            let v = if i == j {
                0.50
            } else if d_i != 0 && d_i == d_j {
                (1.60 + (i.abs_diff(j) as f64) * 0.11).min(4.20)
            } else if d_i == 0 || d_j == 0 {
                24.50
            } else {
                21.80
            };
            let _ = write!(pae_json, "{v:.2}");
        }
        pae_json.push(']');
    }
    pae_json.push_str("]}]");

    // Build assigned NMR shifts (`Reference`): helical secondary shifts in D1 & D2, coil shifts in linker 24..29,
    // plus a planted conditional-folding segment at 19..22 (high pLDDT in AF, coil-like shifts in NMR).
    let mut demo_ref = Reference::new(seq3.clone(), seq_ids);
    for (i, res) in seq3.iter().enumerate() {
        let is_linker_or_morf = (19..29).contains(&i);
        let (ca_off, cb_off, c_off, h_off, n_off) = if is_linker_or_morf {
            (0.05, 0.02, 0.04, 0.0, 0.0)
        } else {
            (3.10, -0.45, 2.15, -0.25, -1.60)
        };
        if res != "PRO" {
            demo_ref.shifts.insert((i, "H".to_string()), 8.20 + h_off);
            demo_ref.shifts.insert((i, "N".to_string()), 120.0 + n_off);
        }
        demo_ref.shifts.insert((i, "CA".to_string()), 54.5 + ca_off);
        demo_ref.shifts.insert((i, "C".to_string()), 175.8 + c_off);
        if res != "GLY" {
            demo_ref.shifts.insert((i, "CB".to_string()), 31.0 + cb_off);
        }
    }

    (cif, pae_json, demo_ref, nh_rdc_tsv)
}
