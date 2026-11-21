//! Modular Unified Joint `HN + 13C` IDP Assignment Panel & Interactive `--cas` Slot Mapper
//! (`T23.3`, `v0.18.0`, native & `wasm32-unknown-unknown`).
//!
//! Provides a calm, zero-emoji `egui` panel and action-emitting state machine for configuring
//! and inspecting unified `HN + 2D CON + Proline-Bridge + N-N + CAS` intrinsically disordered
//! protein (IDP) backbone assignments:
//! 1. **Regime & Root/Calibration Controls**: Switch between `Auto (Joint)`, `IDP Joint`,
//!    `IDP 13C-Only`, and `Globular` regimes; toggle `--roots hnco`, `NH2` sidechain doublet
//!    filtering, field-scaled TROSY `1/2 1J_HN` (`+/-46 Hz / MHz`) cross-spectrum offset
//!    calibration, `F1(C')` NUS sidelobe pruning, `Gly CB` bleed-through guard, C-terminal
//!    `COO-` shift prior, and conservative 2-pass anchor-lock propagation.
//! 2. **Interactive `--cas` (`CACON` / `CANCO`) Slot Mapper**: Add, remove, enable/disable,
//!    and re-type amino-acid-selective 2D `(C'(i-1), N(i))` experiment slots (`Ala`, `Gly`,
//!    `Ser`, `Thr`, `Tavi`, `Other`) with explicit `Cacon (i-1)` vs `Canco (i)` coherence
//!    direction, `(13C', 15N)` axis-consistency indicator, loaded/peak-count badges, and
//!    reproducible CLI `--cas` flag preview.
//! 3. **Proline Bridges, C-Terminal `COO-` Anchor Hints & `HN <-> CON` Route Concordance**:
//!    Inspect resolved `X-Pro` bridges (with interactive anchor lock toggles), C-terminal
//!    carboxylate anchor hints (~179 ppm DSS), and per-residue `HN <-> CON` cross-route
//!    agreement (`CrossValidatedHnAndCon`, `HnOnly`, `ConProBridge`, `ConOnly`).
//!
//! # Provenance & Clean-Room Statement
//! Original clean-room human-computer interaction and state-machine architecture
//! (`docs/specs/06-gui-workbench.md`, `docs/specs/14-regime-solvers.md`), grounded strictly
//! in peer-reviewed NMR methodology:
//! - Bermel, W., Bertini, I., Felli, I. C., Piccioli, M. & Pierattelli, R. (*Prog. Nucl. Magn.
//!   Reson. Spectrosc.* **2006**, 48, 25-45; *J. Am. Chem. Soc.* **2006**, 128, 3918-3919;
//!   *J. Am. Chem. Soc.* **2009**, 131, 15339-15345) — protonless `13C`-detected `CON`, `CANCO`,
//!   `(H)N(CA)CON`, and `H(CC)CON` IDP assignment.
//! - Sahu, D., Bastidas, M. & Showalter, S. A. (*Anal. Biochem.* **2014**, 449, 17-25) &
//!   Bermel, W. et al. (*J. Biomol. NMR* **2012**, 53, 293-301) — amino-acid-selective
//!   `CAS-NMR` (`CACON` `i-1` / `CANCO` `i`) editing in IDPs.
//! - Pervushin, K., Riek, R., Wider, G. & Wuthrich, K. (*Proc. Natl. Acad. Sci. U.S.A.*
//!   **1997**, 94, 12366-12371) — TROSY `1/2 1J_HN` cross-spectrum offset calibration.
//! - Tamiola, K., Acar, B. & Mulder, F. A. A. (*J. Am. Chem. Soc.* **2010**, 132,
//!   18000-18003) & Platzer, G., Okon, M. & McIntosh, L. P. (*J. Biomol. NMR* **2014**, 60,
//!   109-129) — `ncIDP` coil shifts and C-terminal `COO-` pH-dependent shift priors.

use std::fmt::Write as _;

use eframe::egui;
use nmr_core::c13_detected::{parse_cas_residue_set, CasMode};

use crate::workbench_mode::{TYPE_BODY_PX, TYPE_CAPTION_PX, TYPE_HEADER_PX};

/// Default `1/2 1J_HN` TROSY shift magnitude in Hz (`92 Hz / 2 = 46 Hz`).
pub const TROSY_HALF_J_HN_HZ: f64 = 46.0;
/// `15N / 1H` gyromagnetic ratio (`gamma_N / gamma_H`) for converting `15N` Hz offsets to ppm.
pub const GAMMA_N15_OVER_H1: f64 = 0.101_329_118;
/// Nominal DSS carbonyl chemical shift (ppm) of a deprotonated C-terminal `COO-` group.
pub const C_TERM_COO_NOMINAL_PPM: f64 = 179.2;
/// Expected `13C'` chemical-shift window `(min_ppm, max_ppm)` for `2D CON` / `CAS` spectra.
pub const CAS_C_PRIME_RANGE_PPM: (f64, f64) = (165.0, 185.0);
/// Expected `15N` chemical-shift window `(min_ppm, max_ppm)` for `2D CON` / `CAS` (incl. Pro).
pub const CAS_N15_RANGE_PPM: (f64, f64) = (95.0, 148.0);

/// Assignment solver regime selection for the Unified Joint `HN + 13C` IDP workbench.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum IdpRegimeSelection {
    /// Auto-select `idp-joint` when both `HN` and `2D CON` / `13C` spectra are loaded.
    #[default]
    AutoJoint,
    /// Force unified `HN + 2D CON + Proline-Bridge + N-N + CAS` joint IDP solver.
    IdpJoint,
    /// Force protonless `13C`-detected `2D CON`-rooted IDP solver (`--regime idp-13c`).
    Idp13cOnly,
    /// Force standard amide-detected globular backbone solver (`--regime globular`).
    Globular,
}

impl IdpRegimeSelection {
    /// All selectable regimes in display order.
    pub const ALL: [Self; 4] = [
        Self::AutoJoint,
        Self::IdpJoint,
        Self::Idp13cOnly,
        Self::Globular,
    ];

    /// Human-readable regime label.
    pub const fn label(self) -> &'static str {
        match self {
            Self::AutoJoint => "Auto (Joint)",
            Self::IdpJoint => "IDP Joint (HN + 13C)",
            Self::Idp13cOnly => "IDP 13C-Only",
            Self::Globular => "Globular (HN)",
        }
    }

    /// Compact segment label for tight toolbars.
    pub const fn short_label(self) -> &'static str {
        match self {
            Self::AutoJoint => "Auto Joint",
            Self::IdpJoint => "idp-joint",
            Self::Idp13cOnly => "idp-13c",
            Self::Globular => "globular",
        }
    }

    /// Corresponding `--regime` CLI value.
    pub const fn cli_flag(self) -> &'static str {
        match self {
            Self::AutoJoint => "auto",
            Self::IdpJoint => "idp-joint",
            Self::Idp13cOnly => "idp-13c",
            Self::Globular => "globular",
        }
    }

    /// Concise scientific description shown in hover tooltips and summaries.
    pub const fn description(self) -> &'static str {
        match self {
            Self::AutoJoint => {
                "Auto-selects idp-joint when both HN and 2D CON/13C spectra are present."
            }
            Self::IdpJoint => {
                "Fuses HNCO-rooted HN spin systems with 2D CON, Proline bridges, N-N, and CAS."
            }
            Self::Idp13cOnly => {
                "Protonless 13C-detected CON-rooted IDP assignment through Proline residues."
            }
            Self::Globular => {
                "Standard amide-detected globular backbone assignment (docs/specs/03)."
            }
        }
    }

    /// True when protonless `2D CON` / `13C` channels participate in this regime.
    pub const fn uses_con_route(self) -> bool {
        matches!(self, Self::AutoJoint | Self::IdpJoint | Self::Idp13cOnly)
    }

    /// True when amide `HN` (`HSQC`/`HNCO`) channels participate in this regime.
    pub const fn uses_hn_route(self) -> bool {
        matches!(self, Self::AutoJoint | Self::IdpJoint | Self::Globular)
    }

    /// True when joint `HN <-> CON` node fusion is active.
    pub const fn uses_joint_fusion(self) -> bool {
        matches!(self, Self::AutoJoint | Self::IdpJoint)
    }
}

/// Amino-acid selectivity class for a `2D CAS` (`CACON`/`CANCO`) subspectrum slot.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum CasResidueType {
    Ala,
    Gly,
    #[default]
    Ser,
    Thr,
    Tavi,
    Other,
}

impl CasResidueType {
    /// All built-in CAS residue-type options in display order.
    pub const ALL: [Self; 6] = [
        Self::Ala,
        Self::Gly,
        Self::Ser,
        Self::Thr,
        Self::Tavi,
        Self::Other,
    ];

    /// Human-readable label shown in dropdowns and slot badges.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Ala => "Ala",
            Self::Gly => "Gly",
            Self::Ser => "Ser",
            Self::Thr => "Thr",
            Self::Tavi => "TAVI (T,A,V,I)",
            Self::Other => "Other",
        }
    }

    /// Compact badge tag for tables.
    pub const fn short_tag(self) -> &'static str {
        match self {
            Self::Ala => "ALA",
            Self::Gly => "GLY",
            Self::Ser => "SER",
            Self::Thr => "THR",
            Self::Tavi => "TAVI",
            Self::Other => "OTHER",
        }
    }

    /// CLI token used in `--cas <dir>:<token>=<path>`.
    pub fn cli_token(self, custom_residues: &str) -> String {
        match self {
            Self::Ala => "ala".to_string(),
            Self::Gly => "gly".to_string(),
            Self::Ser => "ser".to_string(),
            Self::Thr => "thr".to_string(),
            Self::Tavi => "tavi".to_string(),
            Self::Other => {
                let trimmed = custom_residues.trim();
                if trimmed.is_empty() {
                    "custom".to_string()
                } else {
                    trimmed.to_ascii_lowercase()
                }
            }
        }
    }

    /// Resolve to canonical 3-letter BMRB residue names via `nmr_core::c13_detected::parse_cas_residue_set`.
    pub fn resolve_residues(self, custom_residues: &str) -> Result<Vec<String>, String> {
        match self {
            Self::Ala => Ok(vec!["ALA".to_string()]),
            Self::Gly => Ok(vec!["GLY".to_string()]),
            Self::Ser => Ok(vec!["SER".to_string()]),
            Self::Thr => Ok(vec!["THR".to_string()]),
            Self::Tavi => Ok(vec![
                "THR".to_string(),
                "ALA".to_string(),
                "VAL".to_string(),
                "ILE".to_string(),
            ]),
            Self::Other => parse_cas_residue_set(custom_residues),
        }
    }
}

/// Coherence-transfer direction of a `2D CAS` `(C'(i-1), N(i))` experiment slot.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum CasDirection {
    /// `CAS-CACON` (`i-1`): selects the preceding residue `i - 1`.
    #[default]
    Cacon,
    /// `CAS-CANCO` (`i`): selects the current residue `i`.
    Canco,
}

impl CasDirection {
    /// Both directions in display order.
    pub const ALL: [Self; 2] = [Self::Cacon, Self::Canco];

    /// Human-readable label with residue index annotation.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Cacon => "CACON (i-1)",
            Self::Canco => "CANCO (i)",
        }
    }

    /// Compact CLI prefix matching `nmr_core::c13_detected::CasMode`.
    pub const fn cli_prefix(self) -> &'static str {
        match self {
            Self::Cacon => "cacon",
            Self::Canco => "canco",
        }
    }

    /// Convert to `nmr_core::c13_detected::CasMode`.
    pub const fn to_core_mode(self) -> CasMode {
        match self {
            Self::Cacon => CasMode::Cacon,
            Self::Canco => CasMode::Canco,
        }
    }

    /// Construct from `nmr_core::c13_detected::CasMode`.
    pub const fn from_core_mode(mode: CasMode) -> Self {
        match mode {
            CasMode::Cacon => Self::Cacon,
            CasMode::Canco => Self::Canco,
        }
    }
}

/// Axis-consistency verification status for a `2D CAS` `(C'(i-1), N(i))` spectrum slot.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum CasAxisConsistency {
    /// Axis metadata not yet verified against the `2D CON` reference frame.
    #[default]
    Unchecked,
    /// Verified `F2=13C'` (`165..185 ppm`) and `F1=15N` (`95..148 ppm`) matching `2D CON`.
    Consistent,
    /// Detected transposed `(15N, 13C')` axes; transpose required before matching roots.
    SwappedAxes,
    /// Nuclei are `(13C, 15N)` but ppm bounds fall outside expected `C'` / `15N` windows.
    OutOfRange,
    /// Non-`(13C, 15N)` nuclei detected (e.g., `1H`-detected spectrum misassigned to CAS).
    WrongNuclei,
}

impl CasAxisConsistency {
    /// Short pill text rendered in the CAS slot table.
    pub const fn badge_label(self) -> &'static str {
        match self {
            Self::Unchecked => "UNCHECKED",
            Self::Consistent => "AXES OK",
            Self::SwappedAxes => "SWAPPED F1/F2",
            Self::OutOfRange => "RANGE WARN",
            Self::WrongNuclei => "WRONG NUC",
        }
    }

    /// Detailed hover explanation.
    pub const fn description(self) -> &'static str {
        match self {
            Self::Unchecked => "Axis nuclei and ppm bounds have not been verified yet.",
            Self::Consistent => {
                "Verified F2=13C' (165-185 ppm) and F1=15N (95-148 ppm) aligned with 2D CON."
            }
            Self::SwappedAxes => {
                "Axes appear transposed (F2=15N, F1=13C'); transpose axes before matching."
            }
            Self::OutOfRange => {
                "Axis ppm sweep falls outside expected carbonyl (165-185) or 15N (95-148) bounds."
            }
            Self::WrongNuclei => {
                "Nuclei do not match a 2D (13C', 15N) CAS spectrum (check direct/indirect NUC)."
            }
        }
    }

    /// True if axes are strictly verified consistent.
    pub const fn is_consistent(self) -> bool {
        matches!(self, Self::Consistent)
    }

    /// True if the slot has a blocking axis mismatch.
    pub const fn is_error(self) -> bool {
        matches!(
            self,
            Self::SwappedAxes | Self::OutOfRange | Self::WrongNuclei
        )
    }

    /// Theme-aware badge foreground color.
    pub const fn color(self, paper_white: bool) -> egui::Color32 {
        match (self, paper_white) {
            (Self::Consistent, false) => egui::Color32::from_rgb(76, 214, 148),
            (Self::Consistent, true) => egui::Color32::from_rgb(20, 128, 76),
            (Self::Unchecked, false) => egui::Color32::from_rgb(154, 166, 184),
            (Self::Unchecked, true) => egui::Color32::from_rgb(94, 106, 122),
            (Self::SwappedAxes | Self::OutOfRange, false) => egui::Color32::from_rgb(248, 178, 68),
            (Self::SwappedAxes | Self::OutOfRange, true) => egui::Color32::from_rgb(176, 108, 12),
            (Self::WrongNuclei, false) => egui::Color32::from_rgb(242, 108, 114),
            (Self::WrongNuclei, true) => egui::Color32::from_rgb(184, 38, 46),
        }
    }

    /// Validate `(f2_nuc, f2_hi, f2_lo)` and `(f1_nuc, f1_hi, f1_lo)` against canonical
    /// `(13C', 15N)` `2D CON`/`CAS` axis bounds.
    pub fn validate_axes(
        f2_nuc: &str,
        f2_hi: f64,
        f2_lo: f64,
        f1_nuc: &str,
        f1_hi: f64,
        f1_lo: f64,
    ) -> Self {
        let is_c13 = |s: &str| {
            let u = s.trim().to_ascii_uppercase();
            u == "13C" || u == "C13" || u == "C" || u == "13C'" || u == "CO"
        };
        let is_n15 = |s: &str| {
            let u = s.trim().to_ascii_uppercase();
            u == "15N" || u == "N15" || u == "N"
        };
        if !f2_hi.is_finite() || !f2_lo.is_finite() || !f1_hi.is_finite() || !f1_lo.is_finite() {
            return Self::OutOfRange;
        }
        let f2_min = f2_hi.min(f2_lo);
        let f2_max = f2_hi.max(f2_lo);
        let f1_min = f1_hi.min(f1_lo);
        let f1_max = f1_hi.max(f1_lo);

        if is_c13(f2_nuc) && is_n15(f1_nuc) {
            let c_overlaps = f2_max >= CAS_C_PRIME_RANGE_PPM.0 && f2_min <= CAS_C_PRIME_RANGE_PPM.1;
            let n_overlaps = f1_max >= CAS_N15_RANGE_PPM.0 && f1_min <= CAS_N15_RANGE_PPM.1;
            if c_overlaps && n_overlaps {
                Self::Consistent
            } else {
                Self::OutOfRange
            }
        } else if is_n15(f2_nuc) && is_c13(f1_nuc) {
            Self::SwappedAxes
        } else {
            Self::WrongNuclei
        }
    }
}

/// One configured `--cas` amino-acid-selective 2D `(C'(i-1), N(i))` experiment slot.
#[derive(Clone, Debug, PartialEq)]
pub struct CasSlotEntry {
    /// Stable slot identifier within [`IdpJointPanelState`].
    pub id: usize,
    /// Selected amino-acid filter preset (`Ala`, `Gly`, `Ser`, `Thr`, `Tavi`, or `Other`).
    pub residue_type: CasResidueType,
    /// Custom residue selector token when `residue_type == CasResidueType::Other`.
    pub custom_residues: String,
    /// Coherence-transfer direction (`Cacon` `i-1` vs `Canco` `i`).
    pub direction: CasDirection,
    /// Verification badge for `(13C', 15N)` axis nuclei and ppm sweep consistency.
    pub axis_consistency: CasAxisConsistency,
    /// Display path or logical spectrum key (`wasm32`-safe plain string, never probed on disk).
    pub path: String,
    /// True when peaks or spectrum data are loaded into memory for this slot.
    pub loaded: bool,
    /// Number of picked 2D `(C', N)` peaks currently loaded in this slot.
    pub peak_count: usize,
    /// Whether this slot is active for the next assignment run.
    pub enabled: bool,
    /// Optional provenance or diagnostic note (e.g., Bruker `PULPROG` name).
    pub notes: String,
}

impl CasSlotEntry {
    /// Create a new unloaded, unchecked CAS slot entry.
    pub fn new(
        id: usize,
        residue_type: CasResidueType,
        direction: CasDirection,
        path: impl Into<String>,
    ) -> Self {
        Self {
            id,
            residue_type,
            custom_residues: String::new(),
            direction,
            axis_consistency: CasAxisConsistency::Unchecked,
            path: path.into(),
            loaded: false,
            peak_count: 0,
            enabled: true,
            notes: String::new(),
        }
    }

    /// Attach a custom residue selector string (used when `residue_type == CasResidueType::Other`).
    pub fn with_custom_residues(mut self, custom: impl Into<String>) -> Self {
        self.custom_residues = custom.into();
        self
    }

    /// Populate loaded peak status and verified axis consistency.
    pub fn with_loaded(
        mut self,
        loaded: bool,
        peak_count: usize,
        axis_consistency: CasAxisConsistency,
    ) -> Self {
        self.loaded = loaded;
        self.peak_count = peak_count;
        self.axis_consistency = axis_consistency;
        self
    }

    /// Attach an optional diagnostic note.
    pub fn with_notes(mut self, notes: impl Into<String>) -> Self {
        self.notes = notes.into();
        self
    }

    /// Resolve canonical 3-letter residue codes for this slot.
    pub fn resolved_residues(&self) -> Result<Vec<String>, String> {
        self.residue_type.resolve_residues(&self.custom_residues)
    }

    /// Short display summary of the residue filter and direction (e.g., `"SER (i-1)"`).
    pub fn display_filter_summary(&self) -> String {
        let dir = match self.direction {
            CasDirection::Cacon => "i-1",
            CasDirection::Canco => "i",
        };
        match self.residue_type {
            CasResidueType::Other => match self.resolved_residues() {
                Ok(list) => format!("{} ({dir})", list.join("+")),
                Err(_) => {
                    let raw = self.custom_residues.trim();
                    if raw.is_empty() {
                        format!("Other[empty] ({dir})")
                    } else {
                        format!("Other[{raw}?] ({dir})")
                    }
                }
            },
            other => format!("{} ({dir})", other.short_tag()),
        }
    }

    /// Format the exact repeatable `--cas` CLI argument for this slot.
    pub fn to_cli_arg(&self) -> String {
        let token = self.residue_type.cli_token(&self.custom_residues);
        let path_part = if self.path.trim().is_empty() {
            "<unassigned_path>"
        } else {
            self.path.trim()
        };
        format!("{}:{}={}", self.direction.cli_prefix(), token, path_part)
    }

    /// True when enabled, loaded, residue selector valid, and axis consistency is not an error.
    pub fn is_ready_for_solve(&self) -> bool {
        self.enabled
            && self.loaded
            && !self.axis_consistency.is_error()
            && self.resolved_residues().is_ok()
    }
}

/// Cross-route assignment provenance badge comparing amide `HN` and `13C`-detected `CON` routes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum JointRouteBadge {
    /// Assigned consistently by both amide `HN` (`HSQC`/`HNCO`) and protonless `2D CON` routes.
    CrossValidatedHnAndCon,
    /// Assigned via amide `HN` route only (missing or overlapped in `2D CON`).
    HnOnly,
    /// Proline imino `2D CON` bridge (`N ~ 132-143 ppm`) spanning `X(i-1)-Pro(i)` without an `HN`.
    ConProBridge,
    /// Non-Proline residue assigned via protonless `2D CON` only (e.g., exchange-broadened amide).
    ConOnly,
}

impl JointRouteBadge {
    /// All four route badges in display order.
    pub const ALL: [Self; 4] = [
        Self::CrossValidatedHnAndCon,
        Self::HnOnly,
        Self::ConProBridge,
        Self::ConOnly,
    ];

    /// Human-readable route badge label.
    pub const fn label(self) -> &'static str {
        match self {
            Self::CrossValidatedHnAndCon => "HN + CON Cross-Validated",
            Self::HnOnly => "HN Only",
            Self::ConProBridge => "CON Pro Bridge",
            Self::ConOnly => "CON Only",
        }
    }

    /// Compact pill code suitable for sequence ribbons, strip headers, and narrow table columns.
    pub const fn short_code(self) -> &'static str {
        match self {
            Self::CrossValidatedHnAndCon => "HN+CON",
            Self::HnOnly => "HN",
            Self::ConProBridge => "PRO-CON",
            Self::ConOnly => "CON",
        }
    }

    /// Concise scientific tooltip text.
    pub const fn description(self) -> &'static str {
        match self {
            Self::CrossValidatedHnAndCon => {
                "Both amide HN and 13C-detected 2D CON routes agree on this residue assignment."
            }
            Self::HnOnly => "Supported by amide HN spectra only (CON peak absent or overlapped).",
            Self::ConProBridge => {
                "Proline imino 2D CON bridge (15N 132-143 ppm) linking across an HN chain break."
            }
            Self::ConOnly => {
                "Supported by protonless 2D CON route only (exchange-broadened or missing HN)."
            }
        }
    }

    /// Theme-aware badge accent color.
    pub const fn color(self, paper_white: bool) -> egui::Color32 {
        match (self, paper_white) {
            (Self::CrossValidatedHnAndCon, false) => egui::Color32::from_rgb(76, 214, 148),
            (Self::CrossValidatedHnAndCon, true) => egui::Color32::from_rgb(20, 128, 76),
            (Self::HnOnly, false) => egui::Color32::from_rgb(98, 184, 255),
            (Self::HnOnly, true) => egui::Color32::from_rgb(28, 102, 184),
            (Self::ConProBridge, false) => egui::Color32::from_rgb(204, 138, 255),
            (Self::ConProBridge, true) => egui::Color32::from_rgb(122, 52, 182),
            (Self::ConOnly, false) => egui::Color32::from_rgb(248, 178, 68),
            (Self::ConOnly, true) => egui::Color32::from_rgb(176, 108, 12),
        }
    }
}

/// Resolved `X-Pro(-Y)` bridge spanning a Proline via `2D CON` imino nitrogen and `13C`/`CAS` links.
#[derive(Clone, Debug, PartialEq)]
pub struct ProlineBridgeEntry {
    /// 1-based sequence ID of the bridged Proline residue.
    pub pro_seq_id: i32,
    /// Preceding residue label (`i - 1`, e.g., `"S41"`).
    pub prev_label: String,
    /// Proline residue label (`i`, e.g., `"P42"`).
    pub pro_label: String,
    /// Optional following residue label (`i + 1`, e.g., `"G43"`).
    pub next_label: Option<String>,
    /// Proline imino `15N` chemical shift (ppm, typically `132-143 ppm`).
    pub con_n_ppm: f64,
    /// Preceding `C'(i-1)` chemical shift (ppm) at the `CON(C'_{i-1}, N_i)` root.
    pub con_c_prime_ppm: f64,
    /// Posterior bridge confidence in `[0.0, 1.0]`.
    pub confidence: f64,
    /// True when locked as a hard anchor for Pass-2 joint re-solving.
    pub locked: bool,
    /// Supporting experimental channels (e.g., `"CON"`, `"H(CC)CON"`, `"CAS-SER(i-1)"`).
    pub supporting_channels: Vec<String>,
}

impl ProlineBridgeEntry {
    /// Format the sequential chain motif (e.g., `"S41 - P42 - G43"`).
    pub fn chain_label(&self) -> String {
        match &self.next_label {
            Some(next) if !next.trim().is_empty() => {
                format!("{} - {} - {}", self.prev_label, self.pro_label, next)
            }
            _ => format!("{} - {}", self.prev_label, self.pro_label),
        }
    }
}

/// C-terminal `COO-` anchor hint (~179 ppm DSS, Platzer et al. 2014 / `ncIDP`).
#[derive(Clone, Debug, PartialEq)]
pub struct CTermAnchorHint {
    /// 1-based sequence ID of the C-terminal residue.
    pub seq_id: i32,
    /// Residue label (e.g., `"A112"`).
    pub residue_label: String,
    /// Observed terminal carbonyl `C'` chemical shift (ppm DSS).
    pub observed_c_prime_ppm: f64,
    /// Expected `ncIDP` / Platzer C-terminal `COO-` reference shift (ppm DSS).
    pub expected_coo_ppm: f64,
    /// Optional associated amide `15N` shift (ppm).
    pub n_ppm: Option<f64>,
    /// Anchor posterior confidence in `[0.0, 1.0]`.
    pub confidence: f64,
    /// True when a matching downfield `COO-` peak was found in `HN(CA)CO` / `CACO` / `CON`.
    pub detected: bool,
    /// Human-readable diagnostic note.
    pub note: String,
}

impl CTermAnchorHint {
    /// Signed offset `observed_c_prime_ppm - expected_coo_ppm` in ppm.
    pub fn delta_ppm(&self) -> f64 {
        self.observed_c_prime_ppm - self.expected_coo_ppm
    }
}

/// Per-residue `HN <-> CON` cross-route concordance summary row.
#[derive(Clone, Debug, PartialEq)]
pub struct ResidueRouteConcordance {
    /// 1-based residue sequence ID.
    pub seq_id: i32,
    /// 3-letter residue name (e.g., `"SER"`, `"PRO"`, `"ALA"`).
    pub res_name: String,
    /// Route provenance badge.
    pub badge: JointRouteBadge,
    /// Optional amide `HN`-only route confidence (`0..1`).
    pub hn_conf: Option<f64>,
    /// Optional protonless `CON`-only route confidence (`0..1`).
    pub con_conf: Option<f64>,
    /// Joint fused posterior confidence (`0..1`).
    pub joint_conf: f64,
    /// Optional `N_HN - N_CON` cross-route shift residual (ppm).
    pub delta_n_ppm: Option<f64>,
    /// Optional `C'_HNCO - C'_CON` cross-route shift residual (ppm).
    pub delta_c_prime_ppm: Option<f64>,
    /// True if residue is Proline.
    pub is_proline: bool,
    /// True if residue is the C-terminus.
    pub is_c_term: bool,
    /// Active CAS residue-filter tags corroborating this residue (e.g., `"CAS-SER(i-1)"`).
    pub cas_tags: Vec<String>,
    /// True when locked as a Pass-2 anchor.
    pub locked: bool,
}

/// Aggregated cross-route concordance statistics over [`IdpJointPanelState::concordance_rows`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ConcordanceSummaryStats {
    pub total: usize,
    pub cross_validated: usize,
    pub hn_only: usize,
    pub con_pro_bridge: usize,
    pub con_only: usize,
    pub high_conf_count: usize,
    pub mean_joint_conf: f64,
    pub max_abs_delta_n_ppm: f64,
    pub max_abs_delta_c_prime_ppm: f64,
}

/// Sub-tabs inside [`IdpJointPanelState`] keeping desktop (`1280x800` / `1440x900`) and docked
/// panels spacious, scannable, and overlap-free.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum IdpJointSubTab {
    #[default]
    RegimeAndCas,
    ProBridgesAndCTerm,
    RouteConcordance,
}

impl IdpJointSubTab {
    pub const ALL: [Self; 3] = [
        Self::RegimeAndCas,
        Self::ProBridgesAndCTerm,
        Self::RouteConcordance,
    ];

    pub const fn label(self) -> &'static str {
        match self {
            Self::RegimeAndCas => "Regime & CAS Slots",
            Self::ProBridgesAndCTerm => "Pro Bridges & C-Term",
            Self::RouteConcordance => "HN <-> CON Concordance",
        }
    }
}

/// Clean event enum emitted by [`IdpJointPanelState::show`] so `main.rs` can wire `v0.18.0`
/// solver execution, native/WASM CAS file dialogs, and sequence-ribbon / strip-tiler navigation.
#[derive(Clone, Debug, PartialEq)]
pub enum IdpJointAction {
    /// User selected a new solver regime (`AutoJoint`, `IdpJoint`, `Idp13cOnly`, `Globular`).
    RegimeChanged(IdpRegimeSelection),
    /// User toggled `--roots hnco`.
    ToggleRootsHnco(bool),
    /// User toggled `NH2` sidechain pair filtering.
    ToggleNh2Filtering(bool),
    /// User toggled TROSY `1/2 1J_HN` (`+/-46 Hz / MHz`) cross-spectrum calibration.
    ToggleTrosyCalibration(bool),
    /// User toggled `F1(C')` NUS sidelobe pruning.
    ToggleNusSidelobePruning(bool),
    /// User toggled `Gly CB` bleed-through guard.
    ToggleGlyCbGuard(bool),
    /// User toggled Platzer/`ncIDP` C-terminal `COO-` shift prior.
    ToggleCTermCooPrior(bool),
    /// User toggled conservative 2-pass anchor-lock propagation.
    ToggleTwoPassAnchorLock(bool),
    /// User adjusted spectrometer proton frequency (MHz) used for `1/2 1J_HN` calibration preview.
    SpectrometerMhzChanged(f64),
    /// User added a new `--cas` experiment slot.
    AddCasSlot(CasSlotEntry),
    /// User removed an existing `--cas` slot by its stable `id`.
    RemoveCasSlot(usize),
    /// User modified an existing `--cas` slot (residue type, custom selector, direction, enabled, or path).
    UpdateCasSlot(CasSlotEntry),
    /// User requested loading a spectrum/peak list into the given CAS slot `id`.
    RequestLoadCasSlot(usize),
    /// User clicked the primary Run Assignment button.
    RunJointAssignment,
    /// User clicked a residue row or Proline bridge to focus it in the sequence ribbon / strip tiler.
    InspectResidue { seq_id: i32, badge: JointRouteBadge },
    /// User toggled anchor lock on a resolved `X-Pro` bridge.
    ToggleProBridgeLock { pro_seq_id: i32, locked: bool },
    /// User clicked a C-terminal `COO-` anchor hint to inspect its carbonyl shift in the strip view.
    FocusCTermAnchor { seq_id: i32, c_prime_ppm: f64 },
    /// User loaded the built-in synthetic Pro/Ser-rich IDP demonstration dataset.
    LoadSyntheticDemo,
}

/// Theme palette for [`IdpJointPanelState`] matching FERRIC `Dark Oscilloscope` and `Paper White`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IdpJointPanelTheme {
    pub card_bg: egui::Color32,
    pub card_stroke: egui::Color32,
    pub text_primary: egui::Color32,
    pub text_muted: egui::Color32,
    pub accent: egui::Color32,
    pub ok: egui::Color32,
    pub warn: egui::Color32,
    pub err: egui::Color32,
    pub pill_bg_alpha: u8,
}

impl IdpJointPanelTheme {
    /// Construct theme palette for `Paper White` (`true`) or `Dark Oscilloscope` (`false`).
    pub const fn for_mode(paper_white: bool) -> Self {
        if paper_white {
            Self {
                card_bg: egui::Color32::from_rgb(246, 248, 252),
                card_stroke: egui::Color32::from_rgb(206, 214, 226),
                text_primary: egui::Color32::from_rgb(24, 30, 40),
                text_muted: egui::Color32::from_rgb(86, 98, 114),
                accent: egui::Color32::from_rgb(24, 104, 188),
                ok: egui::Color32::from_rgb(20, 128, 76),
                warn: egui::Color32::from_rgb(176, 108, 12),
                err: egui::Color32::from_rgb(184, 38, 46),
                pill_bg_alpha: 28,
            }
        } else {
            Self {
                card_bg: egui::Color32::from_rgb(26, 31, 40),
                card_stroke: egui::Color32::from_rgb(56, 66, 82),
                text_primary: egui::Color32::from_rgb(232, 238, 246),
                text_muted: egui::Color32::from_rgb(154, 166, 184),
                accent: egui::Color32::from_rgb(98, 184, 255),
                ok: egui::Color32::from_rgb(76, 214, 148),
                warn: egui::Color32::from_rgb(248, 178, 68),
                err: egui::Color32::from_rgb(242, 108, 114),
                pill_bg_alpha: 36,
            }
        }
    }
}

/// Complete state for the Unified Joint `HN + 13C` IDP Workbench & Interactive `--cas` Slot Mapper.
#[derive(Clone, Debug, PartialEq)]
pub struct IdpJointPanelState {
    pub regime: IdpRegimeSelection,
    pub roots_hnco: bool,
    pub filter_nh2_pairs: bool,
    pub trosy_half_j_calibration: bool,
    pub spectrometer_mhz: f64,
    pub f1_nus_sidelobe_pruning: bool,
    pub gly_cb_bleed_guard: bool,
    pub c_term_coo_prior: bool,
    pub two_pass_anchor_lock: bool,
    pub cas_slots: Vec<CasSlotEntry>,
    pub next_slot_id: usize,
    pub draft_residue_type: CasResidueType,
    pub draft_custom_residues: String,
    pub draft_direction: CasDirection,
    pub draft_path: String,
    pub proline_bridges: Vec<ProlineBridgeEntry>,
    pub c_term_hints: Vec<CTermAnchorHint>,
    pub concordance_rows: Vec<ResidueRouteConcordance>,
    pub active_tab: IdpJointSubTab,
    pub route_filter: Option<JointRouteBadge>,
    pub selected_residue: Option<i32>,
    pub status_message: String,
    pub solver_running: bool,
    pub synthetic_demo_loaded: bool,
}

impl Default for IdpJointPanelState {
    fn default() -> Self {
        Self::new()
    }
}

impl IdpJointPanelState {
    /// Create a fresh `IdpJointPanelState` with sensible `idp-joint` defaults.
    pub fn new() -> Self {
        Self {
            regime: IdpRegimeSelection::AutoJoint,
            roots_hnco: true,
            filter_nh2_pairs: true,
            trosy_half_j_calibration: true,
            spectrometer_mhz: 800.0,
            f1_nus_sidelobe_pruning: true,
            gly_cb_bleed_guard: true,
            c_term_coo_prior: true,
            two_pass_anchor_lock: true,
            cas_slots: Vec::new(),
            next_slot_id: 1,
            draft_residue_type: CasResidueType::Ser,
            draft_custom_residues: String::new(),
            draft_direction: CasDirection::Cacon,
            draft_path: String::new(),
            proline_bridges: Vec::new(),
            c_term_hints: Vec::new(),
            concordance_rows: Vec::new(),
            active_tab: IdpJointSubTab::RegimeAndCas,
            route_filter: None,
            selected_residue: None,
            status_message: "Ready: configure regime and CAS slots, or load synthetic demo."
                .to_string(),
            solver_running: false,
            synthetic_demo_loaded: false,
        }
    }

    /// Add a new `--cas` slot and return its stable slot `id`.
    pub fn add_cas_slot(
        &mut self,
        residue_type: CasResidueType,
        custom_residues: impl Into<String>,
        direction: CasDirection,
        path: impl Into<String>,
    ) -> usize {
        let id = self.next_slot_id;
        self.next_slot_id = self.next_slot_id.saturating_add(1);
        let entry = CasSlotEntry::new(id, residue_type, direction, path)
            .with_custom_residues(custom_residues);
        self.cas_slots.push(entry);
        id
    }

    /// Remove a `--cas` slot by its stable `id`. Returns `true` if a slot was removed.
    pub fn remove_cas_slot(&mut self, slot_id: usize) -> bool {
        let before = self.cas_slots.len();
        self.cas_slots.retain(|s| s.id != slot_id);
        self.cas_slots.len() != before
    }

    /// Update the residue type (and optional custom selector) of an existing `--cas` slot.
    pub fn set_cas_slot_type(
        &mut self,
        slot_id: usize,
        residue_type: CasResidueType,
        custom_residues: Option<String>,
    ) -> bool {
        if let Some(slot) = self.cas_slots.iter_mut().find(|s| s.id == slot_id) {
            slot.residue_type = residue_type;
            if let Some(custom) = custom_residues {
                slot.custom_residues = custom;
            }
            true
        } else {
            false
        }
    }

    /// Update the coherence direction (`Cacon` `i-1` vs `Canco` `i`) of an existing `--cas` slot.
    pub fn set_cas_slot_direction(&mut self, slot_id: usize, direction: CasDirection) -> bool {
        if let Some(slot) = self.cas_slots.iter_mut().find(|s| s.id == slot_id) {
            slot.direction = direction;
            true
        } else {
            false
        }
    }

    /// Update the loaded flag, peak count, and axis-consistency badge for a `--cas` slot.
    pub fn set_cas_slot_loaded(
        &mut self,
        slot_id: usize,
        loaded: bool,
        peak_count: usize,
        axis_consistency: CasAxisConsistency,
    ) -> bool {
        if let Some(slot) = self.cas_slots.iter_mut().find(|s| s.id == slot_id) {
            slot.loaded = loaded;
            slot.peak_count = peak_count;
            slot.axis_consistency = axis_consistency;
            true
        } else {
            false
        }
    }

    /// Count of enabled, loaded, valid `--cas` slots ready for the solver.
    pub fn ready_cas_slot_count(&self) -> usize {
        self.cas_slots
            .iter()
            .filter(|s| s.is_ready_for_solve())
            .count()
    }

    /// Field-scaled TROSY `1/2 1J_HN` (`46 Hz / MHz`) magnitude in ppm `(delta_1h_ppm, delta_15n_ppm)`.
    pub fn trosy_half_j_offsets_ppm(&self) -> (f64, f64) {
        let mhz = if self.spectrometer_mhz.is_finite() && self.spectrometer_mhz >= 100.0 {
            self.spectrometer_mhz
        } else {
            800.0
        };
        let dh = TROSY_HALF_J_HN_HZ / mhz;
        let dn = TROSY_HALF_J_HN_HZ / (mhz * GAMMA_N15_OVER_H1);
        (dh, dn)
    }

    /// Aggregate cross-route concordance statistics over `self.concordance_rows`.
    pub fn concordance_stats(&self) -> ConcordanceSummaryStats {
        let mut stats = ConcordanceSummaryStats {
            total: self.concordance_rows.len(),
            ..Default::default()
        };
        if self.concordance_rows.is_empty() {
            return stats;
        }
        let mut conf_sum = 0.0_f64;
        for row in &self.concordance_rows {
            match row.badge {
                JointRouteBadge::CrossValidatedHnAndCon => stats.cross_validated += 1,
                JointRouteBadge::HnOnly => stats.hn_only += 1,
                JointRouteBadge::ConProBridge => stats.con_pro_bridge += 1,
                JointRouteBadge::ConOnly => stats.con_only += 1,
            }
            if row.joint_conf >= 0.80 {
                stats.high_conf_count += 1;
            }
            conf_sum += row.joint_conf.clamp(0.0, 1.0);
            if let Some(dn) = row.delta_n_ppm {
                if dn.is_finite() {
                    stats.max_abs_delta_n_ppm = stats.max_abs_delta_n_ppm.max(dn.abs());
                }
            }
            if let Some(dc) = row.delta_c_prime_ppm {
                if dc.is_finite() {
                    stats.max_abs_delta_c_prime_ppm = stats.max_abs_delta_c_prime_ppm.max(dc.abs());
                }
            }
        }
        stats.mean_joint_conf = conf_sum / (self.concordance_rows.len() as f64);
        stats
    }

    /// Build a reproducible CLI flag preview mirroring the current GUI settings.
    pub fn cli_preview_command(&self) -> String {
        let mut out = format!("ferric assign --regime {}", self.regime.cli_flag());
        if self.roots_hnco && self.regime.uses_hn_route() {
            out.push_str(" --roots hnco");
        }
        if !self.filter_nh2_pairs {
            out.push_str(" --no-nh2-filter");
        }
        if self.trosy_half_j_calibration && self.regime.uses_hn_route() {
            out.push_str(" --trosy-calibrate");
        }
        for slot in &self.cas_slots {
            if slot.enabled {
                let _ = write!(out, " --cas {}", slot.to_cli_arg());
            }
        }
        out
    }

    /// Populate a clean-room synthetic Pro/Ser-rich IDP dataset preview (`wasm32`-clean, no disk I/O).
    pub fn load_synthetic_idp_demo(&mut self) {
        self.regime = IdpRegimeSelection::IdpJoint;
        self.roots_hnco = true;
        self.filter_nh2_pairs = true;
        self.trosy_half_j_calibration = true;
        self.spectrometer_mhz = 800.0;
        self.f1_nus_sidelobe_pruning = true;
        self.gly_cb_bleed_guard = true;
        self.c_term_coo_prior = true;
        self.two_pass_anchor_lock = true;
        self.cas_slots.clear();
        self.next_slot_id = 1;

        let s1 = self.add_cas_slot(
            CasResidueType::Ser,
            "",
            CasDirection::Cacon,
            "demo/cacon_ser.tsv",
        );
        self.set_cas_slot_loaded(s1, true, 14, CasAxisConsistency::Consistent);

        let s2 = self.add_cas_slot(
            CasResidueType::Ala,
            "",
            CasDirection::Cacon,
            "demo/cacon_ala.tsv",
        );
        self.set_cas_slot_loaded(s2, true, 11, CasAxisConsistency::Consistent);

        let s3 = self.add_cas_slot(
            CasResidueType::Gly,
            "",
            CasDirection::Cacon,
            "demo/cacon_gly.tsv",
        );
        self.set_cas_slot_loaded(s3, true, 9, CasAxisConsistency::Consistent);

        let s4 = self.add_cas_slot(
            CasResidueType::Tavi,
            "",
            CasDirection::Canco,
            "demo/canco_tavi.tsv",
        );
        self.set_cas_slot_loaded(s4, true, 18, CasAxisConsistency::Consistent);

        self.proline_bridges = vec![
            ProlineBridgeEntry {
                pro_seq_id: 14,
                prev_label: "S13".to_string(),
                pro_label: "P14".to_string(),
                next_label: Some("G15".to_string()),
                con_n_ppm: 136.42,
                con_c_prime_ppm: 173.18,
                confidence: 0.96,
                locked: true,
                supporting_channels: vec![
                    "CON".to_string(),
                    "H(CC)CON".to_string(),
                    "CAS-SER(i-1)".to_string(),
                ],
            },
            ProlineBridgeEntry {
                pro_seq_id: 28,
                prev_label: "A27".to_string(),
                pro_label: "P28".to_string(),
                next_label: Some("P29".to_string()),
                con_n_ppm: 137.85,
                con_c_prime_ppm: 175.44,
                confidence: 0.92,
                locked: true,
                supporting_channels: vec![
                    "CON".to_string(),
                    "CANCO".to_string(),
                    "CAS-ALA(i-1)".to_string(),
                ],
            },
            ProlineBridgeEntry {
                pro_seq_id: 29,
                prev_label: "P28".to_string(),
                pro_label: "P29".to_string(),
                next_label: Some("T30".to_string()),
                con_n_ppm: 135.91,
                con_c_prime_ppm: 174.92,
                confidence: 0.89,
                locked: false,
                supporting_channels: vec![
                    "CON".to_string(),
                    "(H)N(CA)CON".to_string(),
                    "CAS-TAVI(i)".to_string(),
                ],
            },
        ];

        self.c_term_hints = vec![CTermAnchorHint {
            seq_id: 64,
            residue_label: "A64 (C-term)".to_string(),
            observed_c_prime_ppm: 179.48,
            expected_coo_ppm: C_TERM_COO_NOMINAL_PPM,
            n_ppm: Some(128.64),
            confidence: 0.97,
            detected: true,
            note: "Downfield COO- C' resonance confirmed in HN(CA)CO and CACO (pH 6.8)."
                .to_string(),
        }];

        self.concordance_rows = vec![
            ResidueRouteConcordance {
                seq_id: 12,
                res_name: "ALA".to_string(),
                badge: JointRouteBadge::CrossValidatedHnAndCon,
                hn_conf: Some(0.94),
                con_conf: Some(0.93),
                joint_conf: 0.98,
                delta_n_ppm: Some(0.04),
                delta_c_prime_ppm: Some(-0.01),
                is_proline: false,
                is_c_term: false,
                cas_tags: vec!["CAS-ALA(i-1)".to_string()],
                locked: true,
            },
            ResidueRouteConcordance {
                seq_id: 13,
                res_name: "SER".to_string(),
                badge: JointRouteBadge::CrossValidatedHnAndCon,
                hn_conf: Some(0.91),
                con_conf: Some(0.95),
                joint_conf: 0.97,
                delta_n_ppm: Some(-0.03),
                delta_c_prime_ppm: Some(0.02),
                is_proline: false,
                is_c_term: false,
                cas_tags: vec!["CAS-SER(i-1)".to_string()],
                locked: true,
            },
            ResidueRouteConcordance {
                seq_id: 14,
                res_name: "PRO".to_string(),
                badge: JointRouteBadge::ConProBridge,
                hn_conf: None,
                con_conf: Some(0.96),
                joint_conf: 0.96,
                delta_n_ppm: None,
                delta_c_prime_ppm: None,
                is_proline: true,
                is_c_term: false,
                cas_tags: vec!["CAS-SER(i-1)".to_string()],
                locked: true,
            },
            ResidueRouteConcordance {
                seq_id: 15,
                res_name: "GLY".to_string(),
                badge: JointRouteBadge::CrossValidatedHnAndCon,
                hn_conf: Some(0.92),
                con_conf: Some(0.90),
                joint_conf: 0.96,
                delta_n_ppm: Some(0.05),
                delta_c_prime_ppm: Some(0.01),
                is_proline: false,
                is_c_term: false,
                cas_tags: vec!["CAS-GLY(i-1)".to_string()],
                locked: false,
            },
            ResidueRouteConcordance {
                seq_id: 16,
                res_name: "SER".to_string(),
                badge: JointRouteBadge::ConOnly,
                hn_conf: None,
                con_conf: Some(0.86),
                joint_conf: 0.86,
                delta_n_ppm: None,
                delta_c_prime_ppm: None,
                is_proline: false,
                is_c_term: false,
                cas_tags: vec!["CAS-GLY(i-1)".to_string()],
                locked: false,
            },
            ResidueRouteConcordance {
                seq_id: 27,
                res_name: "ALA".to_string(),
                badge: JointRouteBadge::CrossValidatedHnAndCon,
                hn_conf: Some(0.90),
                con_conf: Some(0.92),
                joint_conf: 0.96,
                delta_n_ppm: Some(0.02),
                delta_c_prime_ppm: Some(-0.02),
                is_proline: false,
                is_c_term: false,
                cas_tags: vec!["CAS-ALA(i-1)".to_string()],
                locked: true,
            },
            ResidueRouteConcordance {
                seq_id: 28,
                res_name: "PRO".to_string(),
                badge: JointRouteBadge::ConProBridge,
                hn_conf: None,
                con_conf: Some(0.92),
                joint_conf: 0.92,
                delta_n_ppm: None,
                delta_c_prime_ppm: None,
                is_proline: true,
                is_c_term: false,
                cas_tags: vec!["CAS-ALA(i-1)".to_string()],
                locked: true,
            },
            ResidueRouteConcordance {
                seq_id: 29,
                res_name: "PRO".to_string(),
                badge: JointRouteBadge::ConProBridge,
                hn_conf: None,
                con_conf: Some(0.89),
                joint_conf: 0.89,
                delta_n_ppm: None,
                delta_c_prime_ppm: None,
                is_proline: true,
                is_c_term: false,
                cas_tags: Vec::new(),
                locked: false,
            },
            ResidueRouteConcordance {
                seq_id: 30,
                res_name: "THR".to_string(),
                badge: JointRouteBadge::CrossValidatedHnAndCon,
                hn_conf: Some(0.88),
                con_conf: Some(0.91),
                joint_conf: 0.95,
                delta_n_ppm: Some(-0.06),
                delta_c_prime_ppm: Some(0.03),
                is_proline: false,
                is_c_term: false,
                cas_tags: vec!["CAS-TAVI(i)".to_string()],
                locked: false,
            },
            ResidueRouteConcordance {
                seq_id: 31,
                res_name: "GLU".to_string(),
                badge: JointRouteBadge::HnOnly,
                hn_conf: Some(0.83),
                con_conf: None,
                joint_conf: 0.83,
                delta_n_ppm: None,
                delta_c_prime_ppm: None,
                is_proline: false,
                is_c_term: false,
                cas_tags: Vec::new(),
                locked: false,
            },
            ResidueRouteConcordance {
                seq_id: 63,
                res_name: "VAL".to_string(),
                badge: JointRouteBadge::CrossValidatedHnAndCon,
                hn_conf: Some(0.95),
                con_conf: Some(0.94),
                joint_conf: 0.98,
                delta_n_ppm: Some(0.01),
                delta_c_prime_ppm: Some(0.00),
                is_proline: false,
                is_c_term: false,
                cas_tags: vec!["CAS-TAVI(i)".to_string()],
                locked: true,
            },
            ResidueRouteConcordance {
                seq_id: 64,
                res_name: "ALA".to_string(),
                badge: JointRouteBadge::CrossValidatedHnAndCon,
                hn_conf: Some(0.96),
                con_conf: Some(0.97),
                joint_conf: 0.99,
                delta_n_ppm: Some(-0.02),
                delta_c_prime_ppm: Some(0.01),
                is_proline: false,
                is_c_term: true,
                cas_tags: vec!["CAS-ALA(i-1)".to_string()],
                locked: true,
            },
        ];

        self.synthetic_demo_loaded = true;
        self.status_message =
            "Synthetic Demo loaded: 4 CAS slots, 3 X-Pro bridges, 1 C-term COO- anchor, 12 residues."
                .to_string();
    }

    /// Render the Unified Joint `HN + 13C` IDP workbench panel inside `ui` and return emitted actions.
    pub fn show(&mut self, ui: &mut egui::Ui, paper_white: bool) -> Vec<IdpJointAction> {
        let theme = IdpJointPanelTheme::for_mode(paper_white);
        let mut actions = Vec::new();

        ui.push_id("ferric_idp_joint_panel", |ui| {
            self.render_header(ui, theme, paper_white, &mut actions);
            ui.add_space(6.0);
            self.render_tab_bar(ui, theme);
            ui.add_space(6.0);

            egui::ScrollArea::vertical()
                .id_salt("idp_joint_panel_scroll")
                .auto_shrink([false, false])
                .show(ui, |ui| match self.active_tab {
                    IdpJointSubTab::RegimeAndCas => {
                        self.render_regime_and_toggles(ui, theme, &mut actions);
                        ui.add_space(8.0);
                        self.render_cas_slot_mapper(ui, theme, paper_white, &mut actions);
                    }
                    IdpJointSubTab::ProBridgesAndCTerm => {
                        self.render_proline_bridges_section(ui, theme, paper_white, &mut actions);
                        ui.add_space(8.0);
                        self.render_c_term_hints_section(ui, theme, &mut actions);
                    }
                    IdpJointSubTab::RouteConcordance => {
                        self.render_concordance_section(ui, theme, paper_white, &mut actions);
                    }
                });
        });

        actions
    }

    /// Render as a standalone dockable/floating `egui::Window` (useful for desktop & `wasm32`).
    pub fn show_window(
        &mut self,
        ctx: &egui::Context,
        open: &mut bool,
        paper_white: bool,
    ) -> Vec<IdpJointAction> {
        let mut actions = Vec::new();
        egui::Window::new("IDP Joint HN + 13C Assignment")
            .open(open)
            .default_size(egui::vec2(960.0, 680.0))
            .min_width(640.0)
            .min_height(460.0)
            .resizable(true)
            .show(ctx, |ui| {
                actions = self.show(ui, paper_white);
            });
        actions
    }

    fn render_header(
        &mut self,
        ui: &mut egui::Ui,
        theme: IdpJointPanelTheme,
        paper_white: bool,
        actions: &mut Vec<IdpJointAction>,
    ) {
        let stats = self.concordance_stats();
        let ready_cas = self.ready_cas_slot_count();

        egui::Frame::new()
            .fill(theme.card_bg)
            .stroke(egui::Stroke::new(1.0, theme.card_stroke))
            .corner_radius(6.0)
            .inner_margin(egui::Margin::symmetric(10, 8))
            .show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.spacing_mut().item_spacing = egui::vec2(8.0, 4.0);
                    ui.label(
                        egui::RichText::new("Unified Joint HN + 13C IDP Workbench")
                            .size(TYPE_HEADER_PX)
                            .strong()
                            .color(theme.text_primary),
                    );
                    if self.synthetic_demo_loaded {
                        draw_status_pill(ui, "SYNTHETIC DEMO", theme.warn, theme.pill_bg_alpha);
                    }
                    draw_status_pill(
                        ui,
                        self.regime.short_label(),
                        theme.accent,
                        theme.pill_bg_alpha,
                    );
                    draw_status_pill(
                        ui,
                        &format!("CAS {ready_cas}/{}", self.cas_slots.len()),
                        if ready_cas > 0 {
                            theme.ok
                        } else {
                            theme.text_muted
                        },
                        theme.pill_bg_alpha,
                    );
                    draw_status_pill(
                        ui,
                        &format!("X-Pro {}", self.proline_bridges.len()),
                        JointRouteBadge::ConProBridge.color(paper_white),
                        theme.pill_bg_alpha,
                    );
                    if stats.total > 0 {
                        draw_status_pill(
                            ui,
                            &format!("HN+CON {}/{}", stats.cross_validated, stats.total),
                            JointRouteBadge::CrossValidatedHnAndCon.color(paper_white),
                            theme.pill_bg_alpha,
                        );
                    }
                });

                ui.add_space(4.0);
                ui.horizontal_wrapped(|ui| {
                    ui.spacing_mut().item_spacing = egui::vec2(8.0, 4.0);
                    let run_label = if self.solver_running {
                        "Solving..."
                    } else {
                        "Run IDP Assignment"
                    };
                    if ui
                        .add_enabled(
                            !self.solver_running,
                            egui::Button::new(
                                egui::RichText::new(run_label).size(TYPE_BODY_PX).strong(),
                            )
                            .min_size(egui::vec2(148.0, 26.0)),
                        )
                        .clicked()
                    {
                        actions.push(IdpJointAction::RunJointAssignment);
                    }

                    if ui
                        .add(
                            egui::Button::new(
                                egui::RichText::new("Load Synthetic Demo").size(TYPE_BODY_PX),
                            )
                            .min_size(egui::vec2(144.0, 26.0)),
                        )
                        .on_hover_text(
                            "Populate a clean-room synthetic Pro/Ser-rich IDP dataset preview.",
                        )
                        .clicked()
                    {
                        self.load_synthetic_idp_demo();
                        actions.push(IdpJointAction::LoadSyntheticDemo);
                    }

                    ui.label(
                        egui::RichText::new(&self.status_message)
                            .size(TYPE_CAPTION_PX)
                            .color(theme.text_muted),
                    );
                });
            });
    }

    fn render_tab_bar(&mut self, ui: &mut egui::Ui, theme: IdpJointPanelTheme) {
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing = egui::vec2(6.0, 4.0);
            for tab in IdpJointSubTab::ALL {
                let selected = self.active_tab == tab;
                let mut text = egui::RichText::new(tab.label()).size(TYPE_BODY_PX);
                if selected {
                    text = text.strong().color(theme.accent);
                }
                if ui
                    .add(
                        egui::Button::selectable(selected, text)
                            .min_size(egui::vec2(0.0, 26.0))
                            .corner_radius(4.0),
                    )
                    .clicked()
                {
                    self.active_tab = tab;
                }
            }
        });
    }

    fn render_regime_and_toggles(
        &mut self,
        ui: &mut egui::Ui,
        theme: IdpJointPanelTheme,
        actions: &mut Vec<IdpJointAction>,
    ) {
        egui::Frame::new()
            .fill(theme.card_bg)
            .stroke(egui::Stroke::new(1.0, theme.card_stroke))
            .corner_radius(6.0)
            .inner_margin(egui::Margin::symmetric(10, 8))
            .show(ui, |ui| {
                ui.label(
                    egui::RichText::new("1. Assignment Regime & Channel Calibration")
                        .size(TYPE_BODY_PX)
                        .strong()
                        .color(theme.text_primary),
                );
                ui.add_space(4.0);

                ui.horizontal_wrapped(|ui| {
                    ui.spacing_mut().item_spacing = egui::vec2(6.0, 4.0);
                    ui.label(
                        egui::RichText::new("Regime:")
                            .size(TYPE_BODY_PX)
                            .color(theme.text_muted),
                    );
                    for regime in IdpRegimeSelection::ALL {
                        let selected = self.regime == regime;
                        let mut text = egui::RichText::new(regime.label()).size(TYPE_BODY_PX);
                        if selected {
                            text = text.strong();
                        }
                        if ui
                            .add(
                                egui::Button::selectable(selected, text)
                                    .min_size(egui::vec2(0.0, 24.0))
                                    .corner_radius(4.0),
                            )
                            .on_hover_text(regime.description())
                            .clicked()
                            && !selected
                        {
                            self.regime = regime;
                            actions.push(IdpJointAction::RegimeChanged(regime));
                        }
                    }
                });

                ui.add_space(4.0);
                ui.label(
                    egui::RichText::new(self.regime.description())
                        .size(TYPE_CAPTION_PX)
                        .color(theme.text_muted),
                );

                ui.add_space(6.0);
                ui.separator();
                ui.add_space(4.0);

                // Primary HN/CON root & TROSY calibration toggles
                ui.horizontal_wrapped(|ui| {
                    ui.spacing_mut().item_spacing = egui::vec2(14.0, 6.0);
                    let hn_enabled = self.regime.uses_hn_route();
                    if ui
                        .add_enabled(
                            hn_enabled,
                            egui::Checkbox::new(
                                &mut self.roots_hnco,
                                egui::RichText::new("HNCO-supported roots (--roots hnco)")
                                    .size(TYPE_BODY_PX),
                            ),
                        )
                        .on_hover_text(
                            "Anchor amide roots on (H_i, N_i, C'_{i-1}) peaks from HNCO.",
                        )
                        .changed()
                    {
                        actions.push(IdpJointAction::ToggleRootsHnco(self.roots_hnco));
                    }

                    if ui
                        .add_enabled(
                            hn_enabled,
                            egui::Checkbox::new(
                                &mut self.filter_nh2_pairs,
                                egui::RichText::new("Filter Asn/Gln NH2 doublet pairs")
                                    .size(TYPE_BODY_PX),
                            ),
                        )
                        .on_hover_text(
                            "Suppress matched sidechain NH2 proton doublets before root fusion.",
                        )
                        .changed()
                    {
                        actions.push(IdpJointAction::ToggleNh2Filtering(self.filter_nh2_pairs));
                    }

                    if ui
                        .add_enabled(
                            hn_enabled,
                            egui::Checkbox::new(
                                &mut self.trosy_half_j_calibration,
                                egui::RichText::new("TROSY 1/2 1J_HN auto-calibration")
                                    .size(TYPE_BODY_PX),
                            ),
                        )
                        .on_hover_text(
                            "Auto-calibrate +/-46 Hz / MHz (dH, dN) offset between TROSY 3Ds and decoupled HSQC/HNCO.",
                        )
                        .changed()
                    {
                        actions.push(IdpJointAction::ToggleTrosyCalibration(
                            self.trosy_half_j_calibration,
                        ));
                    }
                });

                ui.add_space(4.0);
                // Field-scaled TROSY offset readout & spectrometer frequency selector
                ui.horizontal_wrapped(|ui| {
                    ui.spacing_mut().item_spacing = egui::vec2(10.0, 4.0);
                    ui.label(
                        egui::RichText::new("Field (1H MHz):")
                            .size(TYPE_CAPTION_PX)
                            .color(theme.text_muted),
                    );
                    let mut mhz = self.spectrometer_mhz;
                    if ui
                        .add(
                            egui::DragValue::new(&mut mhz)
                                .range(400.0..=1200.0)
                                .speed(10.0)
                                .suffix(" MHz")
                                .max_decimals(1),
                        )
                        .changed()
                    {
                        self.spectrometer_mhz = mhz;
                        actions.push(IdpJointAction::SpectrometerMhzChanged(mhz));
                    }
                    let (dh, dn) = self.trosy_half_j_offsets_ppm();
                    let trosy_text = if self.trosy_half_j_calibration {
                        format!(
                            "1/2 1J_HN (+/-46 Hz): d1H = +/-{dh:.4} ppm, d15N = +/-{dn:.3} ppm"
                        )
                    } else {
                        "TROSY offset calibration disabled (0.000 ppm shift)".to_string()
                    };
                    ui.label(
                        egui::RichText::new(trosy_text)
                            .size(TYPE_CAPTION_PX)
                            .color(theme.text_muted),
                    );
                });

                ui.add_space(6.0);
                // Secondary IDP strip hygiene & 2-pass anchor-lock toggles
                ui.horizontal_wrapped(|ui| {
                    ui.spacing_mut().item_spacing = egui::vec2(14.0, 6.0);
                    if ui
                        .checkbox(
                            &mut self.f1_nus_sidelobe_pruning,
                            egui::RichText::new("F1(C') NUS sidelobe pruning")
                                .size(TYPE_CAPTION_PX),
                        )
                        .changed()
                    {
                        actions.push(IdpJointAction::ToggleNusSidelobePruning(
                            self.f1_nus_sidelobe_pruning,
                        ));
                    }

                    if ui
                        .checkbox(
                            &mut self.gly_cb_bleed_guard,
                            egui::RichText::new("Gly CB bleed-through guard").size(TYPE_CAPTION_PX),
                        )
                        .changed()
                    {
                        actions.push(IdpJointAction::ToggleGlyCbGuard(self.gly_cb_bleed_guard));
                    }

                    if ui
                        .checkbox(
                            &mut self.c_term_coo_prior,
                            egui::RichText::new("C-term COO- (~179 ppm) prior")
                                .size(TYPE_CAPTION_PX),
                        )
                        .changed()
                    {
                        actions.push(IdpJointAction::ToggleCTermCooPrior(self.c_term_coo_prior));
                    }

                    if ui
                        .checkbox(
                            &mut self.two_pass_anchor_lock,
                            egui::RichText::new("2-pass anchor-lock propagation")
                                .size(TYPE_CAPTION_PX),
                        )
                        .changed()
                    {
                        actions.push(IdpJointAction::ToggleTwoPassAnchorLock(
                            self.two_pass_anchor_lock,
                        ));
                    }
                });

                ui.add_space(4.0);
                let cli_cmd = compact_text(&self.cli_preview_command(), 116);
                ui.label(
                    egui::RichText::new(format!("CLI equivalent: {cli_cmd}"))
                        .size(TYPE_CAPTION_PX)
                        .monospace()
                        .color(theme.text_muted),
                );
            });
    }

    fn render_cas_slot_mapper(
        &mut self,
        ui: &mut egui::Ui,
        theme: IdpJointPanelTheme,
        paper_white: bool,
        actions: &mut Vec<IdpJointAction>,
    ) {
        egui::Frame::new()
            .fill(theme.card_bg)
            .stroke(egui::Stroke::new(1.0, theme.card_stroke))
            .corner_radius(6.0)
            .inner_margin(egui::Margin::symmetric(10, 8))
            .show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.spacing_mut().item_spacing = egui::vec2(8.0, 4.0);
                    ui.label(
                        egui::RichText::new("2. Interactive --cas (CACON / CANCO) Slot Mapper")
                            .size(TYPE_BODY_PX)
                            .strong()
                            .color(theme.text_primary),
                    );
                    ui.label(
                        egui::RichText::new(
                            "Map 2D amino-acid-selective (C'(i-1), N(i)) subspectra.",
                        )
                        .size(TYPE_CAPTION_PX)
                        .color(theme.text_muted),
                    );
                });

                ui.add_space(6.0);
                // Add-slot bar
                ui.horizontal_wrapped(|ui| {
                    ui.spacing_mut().item_spacing = egui::vec2(8.0, 4.0);
                    ui.label(
                        egui::RichText::new("AA:")
                            .size(TYPE_CAPTION_PX)
                            .color(theme.text_muted),
                    );
                    egui::ComboBox::from_id_salt("cas_draft_res_type")
                        .selected_text(
                            egui::RichText::new(self.draft_residue_type.label())
                                .size(TYPE_BODY_PX),
                        )
                        .width(126.0)
                        .show_ui(ui, |ui| {
                            for rt in CasResidueType::ALL {
                                ui.selectable_value(
                                    &mut self.draft_residue_type,
                                    rt,
                                    egui::RichText::new(rt.label()).size(TYPE_BODY_PX),
                                );
                            }
                        });

                    if self.draft_residue_type == CasResidueType::Other {
                        ui.add(
                            egui::TextEdit::singleline(&mut self.draft_custom_residues)
                                .hint_text("e.g. VAL,ILE or ASX")
                                .desired_width(120.0)
                                .font(egui::FontId::proportional(TYPE_BODY_PX)),
                        );
                    }

                    ui.label(
                        egui::RichText::new("Dir:")
                            .size(TYPE_CAPTION_PX)
                            .color(theme.text_muted),
                    );
                    egui::ComboBox::from_id_salt("cas_draft_direction")
                        .selected_text(
                            egui::RichText::new(self.draft_direction.label()).size(TYPE_BODY_PX),
                        )
                        .width(114.0)
                        .show_ui(ui, |ui| {
                            for dir in CasDirection::ALL {
                                ui.selectable_value(
                                    &mut self.draft_direction,
                                    dir,
                                    egui::RichText::new(dir.label()).size(TYPE_BODY_PX),
                                );
                            }
                        });

                    ui.add(
                        egui::TextEdit::singleline(&mut self.draft_path)
                            .hint_text("spectrum path or key (e.g. cacon_ser.tsv)")
                            .desired_width(220.0)
                            .font(egui::FontId::proportional(TYPE_BODY_PX)),
                    );

                    let draft_valid = self
                        .draft_residue_type
                        .resolve_residues(&self.draft_custom_residues)
                        .is_ok();
                    if ui
                        .add_enabled(
                            draft_valid,
                            egui::Button::new(
                                egui::RichText::new("+ Add CAS Slot")
                                    .size(TYPE_BODY_PX)
                                    .strong(),
                            )
                            .min_size(egui::vec2(118.0, 24.0)),
                        )
                        .clicked()
                    {
                        let path = if self.draft_path.trim().is_empty() {
                            format!(
                                "{}_{}.tsv",
                                self.draft_direction.cli_prefix(),
                                self.draft_residue_type
                                    .cli_token(&self.draft_custom_residues)
                            )
                        } else {
                            self.draft_path.trim().to_string()
                        };
                        let id = self.add_cas_slot(
                            self.draft_residue_type,
                            self.draft_custom_residues.trim().to_string(),
                            self.draft_direction,
                            path,
                        );
                        if let Some(created) = self.cas_slots.iter().find(|s| s.id == id).cloned() {
                            actions.push(IdpJointAction::AddCasSlot(created));
                        }
                        self.draft_path.clear();
                    }
                });

                if self.draft_residue_type == CasResidueType::Other {
                    if let Err(err) = self
                        .draft_residue_type
                        .resolve_residues(&self.draft_custom_residues)
                    {
                        ui.add_space(2.0);
                        ui.label(
                            egui::RichText::new(format!("Custom selector: {err}"))
                                .size(TYPE_CAPTION_PX)
                                .color(theme.err),
                        );
                    }
                }

                ui.add_space(8.0);
                if self.cas_slots.is_empty() {
                    ui.label(
                        egui::RichText::new(
                            "No CAS slots configured. Add Ala, Gly, Ser, Thr, TAVI, or custom slots above.",
                        )
                        .size(TYPE_CAPTION_PX)
                        .color(theme.text_muted),
                    );
                    return;
                }

                let mut to_remove: Option<usize> = None;
                let mut updated_slots: Vec<CasSlotEntry> = Vec::new();
                let mut load_requests: Vec<usize> = Vec::new();

                for slot in &mut self.cas_slots {
                    let slot_id = slot.id;
                    let mut changed = false;

                    egui::Frame::new()
                        .fill(if paper_white {
                            egui::Color32::from_rgb(255, 255, 255)
                        } else {
                            egui::Color32::from_rgb(20, 24, 32)
                        })
                        .stroke(egui::Stroke::new(1.0, theme.card_stroke))
                        .corner_radius(5.0)
                        .inner_margin(egui::Margin::symmetric(8, 6))
                        .show(ui, |ui| {
                            ui.push_id(("cas_slot_row", slot_id), |ui| {
                                ui.horizontal_wrapped(|ui| {
                                    ui.spacing_mut().item_spacing = egui::vec2(8.0, 4.0);
                                    if ui.checkbox(&mut slot.enabled, "").changed() {
                                        changed = true;
                                    }

                                    ui.label(
                                        egui::RichText::new(format!("#{slot_id}"))
                                            .size(TYPE_CAPTION_PX)
                                            .monospace()
                                            .color(theme.text_muted),
                                    );

                                    let prev_rt = slot.residue_type;
                                    egui::ComboBox::from_id_salt(("cas_row_rt", slot_id))
                                        .selected_text(
                                            egui::RichText::new(slot.residue_type.label())
                                                .size(TYPE_BODY_PX),
                                        )
                                        .width(120.0)
                                        .show_ui(ui, |ui| {
                                            for rt in CasResidueType::ALL {
                                                ui.selectable_value(
                                                    &mut slot.residue_type,
                                                    rt,
                                                    egui::RichText::new(rt.label())
                                                        .size(TYPE_BODY_PX),
                                                );
                                            }
                                        });
                                    if slot.residue_type != prev_rt {
                                        changed = true;
                                    }

                                    if slot.residue_type == CasResidueType::Other
                                        && ui
                                            .add(
                                                egui::TextEdit::singleline(
                                                    &mut slot.custom_residues,
                                                )
                                                .hint_text("AA codes")
                                                .desired_width(96.0)
                                                .font(egui::FontId::proportional(TYPE_BODY_PX)),
                                            )
                                            .changed()
                                    {
                                        changed = true;
                                    }

                                    let prev_dir = slot.direction;
                                    egui::ComboBox::from_id_salt(("cas_row_dir", slot_id))
                                        .selected_text(
                                            egui::RichText::new(slot.direction.label())
                                                .size(TYPE_BODY_PX),
                                        )
                                        .width(112.0)
                                        .show_ui(ui, |ui| {
                                            for dir in CasDirection::ALL {
                                                ui.selectable_value(
                                                    &mut slot.direction,
                                                    dir,
                                                    egui::RichText::new(dir.label())
                                                        .size(TYPE_BODY_PX),
                                                );
                                            }
                                        });
                                    if slot.direction != prev_dir {
                                        changed = true;
                                    }

                                    draw_status_pill(
                                        ui,
                                        slot.axis_consistency.badge_label(),
                                        slot.axis_consistency.color(paper_white),
                                        theme.pill_bg_alpha,
                                    )
                                    .on_hover_text(slot.axis_consistency.description());

                                    let load_badge = if slot.loaded {
                                        format!("LOADED ({} pks)", slot.peak_count)
                                    } else {
                                        "NOT LOADED".to_string()
                                    };
                                    draw_status_pill(
                                        ui,
                                        &load_badge,
                                        if slot.loaded {
                                            theme.ok
                                        } else {
                                            theme.text_muted
                                        },
                                        theme.pill_bg_alpha,
                                    );

                                    if ui
                                        .add(
                                            egui::Button::new(
                                                egui::RichText::new("Load...").size(TYPE_CAPTION_PX),
                                            )
                                            .min_size(egui::vec2(62.0, 22.0)),
                                        )
                                        .clicked()
                                    {
                                        load_requests.push(slot_id);
                                    }

                                    if ui
                                        .add(
                                            egui::Button::new(
                                                egui::RichText::new("Remove")
                                                    .size(TYPE_CAPTION_PX)
                                                    .color(theme.err),
                                            )
                                            .min_size(egui::vec2(62.0, 22.0)),
                                        )
                                        .clicked()
                                    {
                                        to_remove = Some(slot_id);
                                    }
                                });

                                ui.add_space(3.0);
                                ui.horizontal_wrapped(|ui| {
                                    ui.spacing_mut().item_spacing = egui::vec2(10.0, 2.0);
                                    let display_path = if slot.path.trim().is_empty() {
                                        "(no path bound)".to_string()
                                    } else {
                                        compact_text(&slot.path, 68)
                                    };
                                    ui.label(
                                        egui::RichText::new(format!("Path: {display_path}"))
                                            .size(TYPE_CAPTION_PX)
                                            .monospace()
                                            .color(theme.text_muted),
                                    )
                                    .on_hover_text(&slot.path);

                                    ui.label(
                                        egui::RichText::new(format!(
                                            "Filter: {}",
                                            slot.display_filter_summary()
                                        ))
                                        .size(TYPE_CAPTION_PX)
                                        .color(theme.text_primary),
                                    );

                                    if let Err(err) = slot.resolved_residues() {
                                        ui.label(
                                            egui::RichText::new(format!("Invalid: {err}"))
                                                .size(TYPE_CAPTION_PX)
                                                .color(theme.err),
                                        );
                                    }
                                });
                            });
                        });
                    ui.add_space(4.0);

                    if changed {
                        updated_slots.push(slot.clone());
                    }
                }

                for slot in updated_slots {
                    actions.push(IdpJointAction::UpdateCasSlot(slot));
                }
                for slot_id in load_requests {
                    actions.push(IdpJointAction::RequestLoadCasSlot(slot_id));
                }
                if let Some(slot_id) = to_remove {
                    if self.remove_cas_slot(slot_id) {
                        actions.push(IdpJointAction::RemoveCasSlot(slot_id));
                    }
                }
            });
    }

    fn render_proline_bridges_section(
        &mut self,
        ui: &mut egui::Ui,
        theme: IdpJointPanelTheme,
        paper_white: bool,
        actions: &mut Vec<IdpJointAction>,
    ) {
        egui::Frame::new()
            .fill(theme.card_bg)
            .stroke(egui::Stroke::new(1.0, theme.card_stroke))
            .corner_radius(6.0)
            .inner_margin(egui::Margin::symmetric(10, 8))
            .show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.spacing_mut().item_spacing = egui::vec2(8.0, 4.0);
                    ui.label(
                        egui::RichText::new("Resolved X-Pro Bridges (2D CON Imino 15N 132-143 ppm)")
                            .size(TYPE_BODY_PX)
                            .strong()
                            .color(theme.text_primary),
                    );
                    draw_route_badge_pill(ui, JointRouteBadge::ConProBridge, paper_white);
                });

                ui.add_space(6.0);
                if self.proline_bridges.is_empty() {
                    ui.label(
                        egui::RichText::new(
                            "No resolved X-Pro bridges yet. Run idp-joint or load the synthetic demo.",
                        )
                        .size(TYPE_CAPTION_PX)
                        .color(theme.text_muted),
                    );
                    return;
                }

                egui::Grid::new("idp_proline_bridges_grid")
                    .num_columns(6)
                    .spacing([14.0, 6.0])
                    .striped(true)
                    .show(ui, |ui| {
                        for header in [
                            "Bridge",
                            "Pro 15N (ppm)",
                            "Prev C' (ppm)",
                            "Confidence",
                            "Channels",
                            "Actions",
                        ] {
                            ui.label(
                                egui::RichText::new(header)
                                    .size(TYPE_CAPTION_PX)
                                    .strong()
                                    .color(theme.text_muted),
                            );
                        }
                        ui.end_row();

                        for bridge in &mut self.proline_bridges {
                            ui.label(
                                egui::RichText::new(bridge.chain_label())
                                    .size(TYPE_BODY_PX)
                                    .strong()
                                    .color(theme.text_primary),
                            );
                            ui.label(
                                egui::RichText::new(format!("{:.2}", bridge.con_n_ppm))
                                    .size(TYPE_BODY_PX)
                                    .monospace(),
                            );
                            ui.label(
                                egui::RichText::new(format!("{:.2}", bridge.con_c_prime_ppm))
                                    .size(TYPE_BODY_PX)
                                    .monospace(),
                            );
                            let conf_color = if bridge.confidence >= 0.80 {
                                theme.ok
                            } else {
                                theme.warn
                            };
                            ui.label(
                                egui::RichText::new(format!("{:.0}%", bridge.confidence * 100.0))
                                    .size(TYPE_BODY_PX)
                                    .strong()
                                    .color(conf_color),
                            );
                            let channels = if bridge.supporting_channels.is_empty() {
                                "-".to_string()
                            } else {
                                bridge.supporting_channels.join(", ")
                            };
                            ui.label(
                                egui::RichText::new(channels)
                                    .size(TYPE_CAPTION_PX)
                                    .color(theme.text_muted),
                            );
                            ui.horizontal(|ui| {
                                ui.spacing_mut().item_spacing.x = 6.0;
                                let lock_txt = if bridge.locked { "Locked" } else { "Lock" };
                                if ui
                                    .add(egui::Button::selectable(
                                        bridge.locked,
                                        egui::RichText::new(lock_txt).size(TYPE_CAPTION_PX),
                                    ))
                                    .clicked()
                                {
                                    bridge.locked = !bridge.locked;
                                    actions.push(IdpJointAction::ToggleProBridgeLock {
                                        pro_seq_id: bridge.pro_seq_id,
                                        locked: bridge.locked,
                                    });
                                }
                                if ui
                                    .add(egui::Button::new(
                                        egui::RichText::new("Inspect").size(TYPE_CAPTION_PX),
                                    ))
                                    .clicked()
                                {
                                    self.selected_residue = Some(bridge.pro_seq_id);
                                    actions.push(IdpJointAction::InspectResidue {
                                        seq_id: bridge.pro_seq_id,
                                        badge: JointRouteBadge::ConProBridge,
                                    });
                                }
                            });
                            ui.end_row();
                        }
                    });
            });
    }

    fn render_c_term_hints_section(
        &mut self,
        ui: &mut egui::Ui,
        theme: IdpJointPanelTheme,
        actions: &mut Vec<IdpJointAction>,
    ) {
        egui::Frame::new()
            .fill(theme.card_bg)
            .stroke(egui::Stroke::new(1.0, theme.card_stroke))
            .corner_radius(6.0)
            .inner_margin(egui::Margin::symmetric(10, 8))
            .show(ui, |ui| {
                ui.label(
                    egui::RichText::new(
                        "C-Terminal COO- Anchor Hints (Platzer / ncIDP ~179 ppm DSS)",
                    )
                    .size(TYPE_BODY_PX)
                    .strong()
                    .color(theme.text_primary),
                );
                ui.add_space(6.0);

                if self.c_term_hints.is_empty() {
                    ui.label(
                        egui::RichText::new("No C-terminal COO- anchor hints available yet.")
                            .size(TYPE_CAPTION_PX)
                            .color(theme.text_muted),
                    );
                    return;
                }

                for hint in &self.c_term_hints {
                    ui.horizontal_wrapped(|ui| {
                        ui.spacing_mut().item_spacing = egui::vec2(10.0, 4.0);
                        draw_status_pill(
                            ui,
                            if hint.detected { "DETECTED" } else { "ABSENT" },
                            if hint.detected { theme.ok } else { theme.warn },
                            theme.pill_bg_alpha,
                        );
                        ui.label(
                            egui::RichText::new(&hint.residue_label)
                                .size(TYPE_BODY_PX)
                                .strong()
                                .color(theme.text_primary),
                        );
                        ui.label(
                            egui::RichText::new(format!(
                                "C' = {:.2} ppm (expected {:.2}, delta {:+.2} ppm)",
                                hint.observed_c_prime_ppm,
                                hint.expected_coo_ppm,
                                hint.delta_ppm()
                            ))
                            .size(TYPE_BODY_PX)
                            .monospace(),
                        );
                        if let Some(n_ppm) = hint.n_ppm {
                            ui.label(
                                egui::RichText::new(format!("15N = {n_ppm:.2} ppm"))
                                    .size(TYPE_CAPTION_PX)
                                    .monospace()
                                    .color(theme.text_muted),
                            );
                        }
                        ui.label(
                            egui::RichText::new(format!("conf {:.0}%", hint.confidence * 100.0))
                                .size(TYPE_CAPTION_PX)
                                .strong()
                                .color(if hint.confidence >= 0.80 {
                                    theme.ok
                                } else {
                                    theme.warn
                                }),
                        );
                        if ui
                            .add(egui::Button::new(
                                egui::RichText::new("Focus C'").size(TYPE_CAPTION_PX),
                            ))
                            .clicked()
                        {
                            self.selected_residue = Some(hint.seq_id);
                            actions.push(IdpJointAction::FocusCTermAnchor {
                                seq_id: hint.seq_id,
                                c_prime_ppm: hint.observed_c_prime_ppm,
                            });
                        }
                    });
                    if !hint.note.trim().is_empty() {
                        ui.label(
                            egui::RichText::new(&hint.note)
                                .size(TYPE_CAPTION_PX)
                                .color(theme.text_muted),
                        );
                    }
                    ui.add_space(4.0);
                }
            });
    }

    fn render_concordance_section(
        &mut self,
        ui: &mut egui::Ui,
        theme: IdpJointPanelTheme,
        paper_white: bool,
        actions: &mut Vec<IdpJointAction>,
    ) {
        let stats = self.concordance_stats();

        egui::Frame::new()
            .fill(theme.card_bg)
            .stroke(egui::Stroke::new(1.0, theme.card_stroke))
            .corner_radius(6.0)
            .inner_margin(egui::Margin::symmetric(10, 8))
            .show(ui, |ui| {
                ui.label(
                    egui::RichText::new("HN <-> CON Cross-Route Concordance Summary")
                        .size(TYPE_BODY_PX)
                        .strong()
                        .color(theme.text_primary),
                );
                ui.add_space(6.0);

                // Route filter pills + counts
                ui.horizontal_wrapped(|ui| {
                    ui.spacing_mut().item_spacing = egui::vec2(6.0, 4.0);
                    let all_sel = self.route_filter.is_none();
                    if ui
                        .add(egui::Button::selectable(
                            all_sel,
                            egui::RichText::new(format!("All ({})", stats.total))
                                .size(TYPE_CAPTION_PX),
                        ))
                        .clicked()
                    {
                        self.route_filter = None;
                    }
                    for badge in JointRouteBadge::ALL {
                        let count = match badge {
                            JointRouteBadge::CrossValidatedHnAndCon => stats.cross_validated,
                            JointRouteBadge::HnOnly => stats.hn_only,
                            JointRouteBadge::ConProBridge => stats.con_pro_bridge,
                            JointRouteBadge::ConOnly => stats.con_only,
                        };
                        let selected = self.route_filter == Some(badge);
                        let label = format!("{} ({count})", badge.short_code());
                        if ui
                            .add(egui::Button::selectable(
                                selected,
                                egui::RichText::new(label)
                                    .size(TYPE_CAPTION_PX)
                                    .color(badge.color(paper_white)),
                            ))
                            .on_hover_text(badge.description())
                            .clicked()
                        {
                            self.route_filter = if selected { None } else { Some(badge) };
                        }
                    }
                });

                if stats.total > 0 {
                    ui.add_space(4.0);
                    ui.horizontal_wrapped(|ui| {
                        ui.spacing_mut().item_spacing = egui::vec2(14.0, 4.0);
                        ui.label(
                            egui::RichText::new(format!(
                                "High-conf (>=80%): {}/{}",
                                stats.high_conf_count, stats.total
                            ))
                            .size(TYPE_CAPTION_PX)
                            .color(theme.ok),
                        );
                        ui.label(
                            egui::RichText::new(format!(
                                "Mean joint conf: {:.1}%",
                                stats.mean_joint_conf * 100.0
                            ))
                            .size(TYPE_CAPTION_PX)
                            .color(theme.text_primary),
                        );
                        ui.label(
                            egui::RichText::new(format!(
                                "Max |d15N|: {:.3} ppm",
                                stats.max_abs_delta_n_ppm
                            ))
                            .size(TYPE_CAPTION_PX)
                            .color(theme.text_muted),
                        );
                        ui.label(
                            egui::RichText::new(format!(
                                "Max |d13C'|: {:.3} ppm",
                                stats.max_abs_delta_c_prime_ppm
                            ))
                            .size(TYPE_CAPTION_PX)
                            .color(theme.text_muted),
                        );
                    });
                }

                ui.add_space(8.0);
                if self.concordance_rows.is_empty() {
                    ui.label(
                        egui::RichText::new(
                            "No residue concordance rows yet. Run idp-joint or load the synthetic demo.",
                        )
                        .size(TYPE_CAPTION_PX)
                        .color(theme.text_muted),
                    );
                    return;
                }

                let mut inspect_req: Option<(i32, JointRouteBadge)> = None;

                egui::Grid::new("idp_concordance_grid")
                    .num_columns(8)
                    .spacing([12.0, 5.0])
                    .striped(true)
                    .show(ui, |ui| {
                        for header in [
                            "Residue",
                            "Route",
                            "HN Conf",
                            "CON Conf",
                            "Joint",
                            "dN / dC' (ppm)",
                            "CAS / Anchor",
                            "View",
                        ] {
                            ui.label(
                                egui::RichText::new(header)
                                    .size(TYPE_CAPTION_PX)
                                    .strong()
                                    .color(theme.text_muted),
                            );
                        }
                        ui.end_row();

                        for row in &self.concordance_rows {
                            if let Some(filter) = self.route_filter {
                                if row.badge != filter {
                                    continue;
                                }
                            }
                            let is_selected = self.selected_residue == Some(row.seq_id);
                            let mut res_txt =
                                egui::RichText::new(format!("{}{}", row.res_name, row.seq_id))
                                    .size(TYPE_BODY_PX)
                                    .monospace();
                            if is_selected {
                                res_txt = res_txt.strong().color(theme.accent);
                            }
                            ui.label(res_txt);

                            draw_route_badge_pill(ui, row.badge, paper_white);

                            let hn_str = row
                                .hn_conf
                                .map(|c| format!("{:.0}%", c * 100.0))
                                .unwrap_or_else(|| "-".to_string());
                            ui.label(egui::RichText::new(hn_str).size(TYPE_CAPTION_PX).monospace());

                            let con_str = row
                                .con_conf
                                .map(|c| format!("{:.0}%", c * 100.0))
                                .unwrap_or_else(|| "-".to_string());
                            ui.label(
                                egui::RichText::new(con_str)
                                    .size(TYPE_CAPTION_PX)
                                    .monospace(),
                            );

                            let joint_color = if row.joint_conf >= 0.80 {
                                theme.ok
                            } else {
                                theme.warn
                            };
                            ui.label(
                                egui::RichText::new(format!("{:.0}%", row.joint_conf * 100.0))
                                    .size(TYPE_BODY_PX)
                                    .strong()
                                    .color(joint_color),
                            );

                            let delta_str = match (row.delta_n_ppm, row.delta_c_prime_ppm) {
                                (Some(dn), Some(dc)) => format!("{dn:+.2} / {dc:+.2}"),
                                (Some(dn), None) => format!("{dn:+.2} / -"),
                                (None, Some(dc)) => format!("- / {dc:+.2}"),
                                (None, None) => "-".to_string(),
                            };
                            ui.label(
                                egui::RichText::new(delta_str)
                                    .size(TYPE_CAPTION_PX)
                                    .monospace()
                                    .color(theme.text_muted),
                            );

                            let mut tags: Vec<&str> =
                                row.cas_tags.iter().map(String::as_str).collect();
                            if row.is_c_term {
                                tags.push("COO-");
                            }
                            if row.locked {
                                tags.push("LOCK");
                            }
                            let tag_str = if tags.is_empty() {
                                "-".to_string()
                            } else {
                                tags.join(", ")
                            };
                            ui.label(
                                egui::RichText::new(tag_str)
                                    .size(TYPE_CAPTION_PX)
                                    .color(theme.text_muted),
                            );

                            if ui
                                .add(egui::Button::new(
                                    egui::RichText::new("Strip").size(TYPE_CAPTION_PX),
                                ))
                                .clicked()
                            {
                                inspect_req = Some((row.seq_id, row.badge));
                            }
                            ui.end_row();
                        }
                    });

                if let Some((seq_id, badge)) = inspect_req {
                    self.selected_residue = Some(seq_id);
                    actions.push(IdpJointAction::InspectResidue { seq_id, badge });
                }
            });
    }
}

/// Reusable zero-emoji pill widget for [`JointRouteBadge`] (for sequence ribbon, strip headers,
/// and concordance tables).
pub fn draw_route_badge_pill(
    ui: &mut egui::Ui,
    badge: JointRouteBadge,
    paper_white: bool,
) -> egui::Response {
    let color = badge.color(paper_white);
    let alpha = if paper_white { 28 } else { 36 };
    draw_status_pill(ui, badge.short_code(), color, alpha).on_hover_text(badge.description())
}

fn draw_status_pill(
    ui: &mut egui::Ui,
    text: &str,
    accent: egui::Color32,
    bg_alpha: u8,
) -> egui::Response {
    let bg = egui::Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), bg_alpha);
    egui::Frame::new()
        .fill(bg)
        .stroke(egui::Stroke::new(1.0, accent))
        .corner_radius(4.0)
        .inner_margin(egui::Margin::symmetric(6, 2))
        .show(ui, |ui| {
            ui.label(
                egui::RichText::new(text)
                    .size(TYPE_CAPTION_PX)
                    .strong()
                    .color(accent),
            )
        })
        .response
}

fn compact_text(input: &str, max_chars: usize) -> String {
    let count = input.chars().count();
    if count <= max_chars || max_chars < 6 {
        return input.to_string();
    }
    let keep = max_chars.saturating_sub(3);
    let prefix: String = input.chars().take(keep).collect();
    format!("{prefix}...")
}
