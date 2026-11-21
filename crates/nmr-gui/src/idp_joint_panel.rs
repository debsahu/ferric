//! Modular Unified Joint `HN + 13C` IDP Assignment Panel & Interactive `--cas` Slot Mapper
//! (`T23.3`, `v0.18.0`, native & `wasm32-unknown-unknown`).
//!
//! Directly consumes canonical `nmr_core::joint_idp_solver` types, constants, and solver results
//! (zero duplicated route enums or TROSY math) to provide a calm, zero-emoji `egui` panel and
//! action-emitting state machine for unified `HN + 2D CON + Proline-Bridge + N-N + CAS`
//! intrinsically disordered protein (IDP) backbone assignments:
//! 1. **Regime & Channel Calibration Controls**: Switch between `Auto (Joint)`, `IDP Joint`,
//!    `IDP 13C-Only`, and `Globular` regimes; toggle `--roots hnco`, `NH2` sidechain doublet
//!    filtering, field-scaled TROSY `46 Hz (1/2 1J_NH)` cross-spectrum offset calibration,
//!    `F1(C')` NUS sidelobe pruning, `Gly CB` bleed-through guard, C-terminal `COO-` shift
//!    prior, and conservative 2-pass anchor-lock propagation; preview a strictly runnable
//!    `ferric assign` CLI command matching `crates/nmr-cli/src/main.rs`.
//! 2. **Interactive `--cas` (`CACON` / `CANCO`) Slot Mapper**: Add (bounded by [`MAX_CAS_SLOTS`]),
//!    remove, enable/disable, edit paths, and re-type amino-acid-selective 2D `(C'(i-1), N(i))`
//!    experiment slots (`Ala`, `Gly`, `Ser`, `Thr`, `Tavi`, `Other`) with explicit `Cacon (i-1)`
//!    vs `Canco (i)` direction, live `(13C', 15N)` axis-consistency validation, loaded/peak-count
//!    badges, and direct conversion to [`CasSpectrum`] for [`JointIdpDataset`].
//! 3. **Proline Bridges, C-Terminal `COO-` Anchor Hints & `HN <-> CON` Route Concordance**:
//!    Populated directly from real [`JointIdpAssignResult`] via [`IdpJointPanelState::sync_from_joint_result`],
//!    inspecting resolved `X-Pro` bridges (`131.5-144.5 ppm` DSS), C-terminal `COO-` anchors
//!    (`~179.0 ppm` DSS), QC verdicts/reasons, and per-residue [`JointRouteSupport`] concordance.
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
//!   **1997**, 94, 12366-12371) — TROSY `46 Hz (1/2 1J_NH)` cross-spectrum offset calibration.
//! - Tamiola, K., Acar, B. & Mulder, F. A. A. (*J. Am. Chem. Soc.* **2010**, 132,
//!   18000-18003) & Platzer, G., Okon, M. & McIntosh, L. P. (*J. Biomol. NMR* **2014**, 60,
//!   109-129) — `ncIDP` coil shifts and C-terminal `COO-` pH-dependent shift priors.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use eframe::egui;
use nmr_core::c13_detected::{
    parse_cas_residue_set, C13DetectedExp, C13PickedDataset, C13Strip3DPeak, CasMode, CasPeak,
    CasSpectrum, Con2DPeak,
};
use nmr_core::idp_solver::IdpShiftOffsets;
use nmr_core::joint_idp_solver::predict_joint_ncidp_shifts;
pub use nmr_core::joint_idp_solver::{
    assign_joint_idp_cpu, joint_idp_to_reference, trosy_expected_shift_ppm, CTermAnchorSummary,
    JointIdpAssignResult, JointIdpConfig, JointIdpDataset, JointIdpResult, JointRouteSupport,
    ProlineBridgeSummary, TrosyOffsetCalibration, CANONICAL_TROSY_FALLBACK_MHZ,
    CTERM_COO_DSS_CENTER_PPM, PROLINE_N15_MAX_PPM, PROLINE_N15_MIN_PPM, TROSY_HALF_J_HN_HZ,
};
use nmr_core::peaks::Peak;
use nmr_core::qc::DatasetQcVerdict;
use nmr_core::spinsystems::HncacbPolarity;

use crate::workbench_mode::{TYPE_BODY_PX, TYPE_CAPTION_PX, TYPE_HEADER_PX};

/// Expected `13C'` chemical-shift window `(min_ppm, max_ppm)` for `2D CON` / `CAS` spectra.
pub const CAS_C_PRIME_RANGE_PPM: (f64, f64) = (165.0, 185.0);
/// Expected `15N` chemical-shift window `(min_ppm, max_ppm)` for `2D CON` / `CAS` (incl. Pro).
pub const CAS_N15_RANGE_PPM: (f64, f64) = (95.0, 148.0);
/// Hard upper bound on interactive `--cas` slots to keep UI/solver work bounded (`P2-8`).
pub const MAX_CAS_SLOTS: usize = 16;

/// All four canonical [`JointRouteSupport`] categories in display order.
pub const ALL_JOINT_ROUTE_SUPPORTS: [JointRouteSupport; 4] = [
    JointRouteSupport::CrossValidatedHnAndCon,
    JointRouteSupport::HnRoute,
    JointRouteSupport::ConProBridge,
    JointRouteSupport::ConRouteOnly,
];

/// Human-readable label for a canonical [`JointRouteSupport`] variant.
pub const fn joint_route_label(route: JointRouteSupport) -> &'static str {
    match route {
        JointRouteSupport::CrossValidatedHnAndCon => "HN + CON Cross-Validated",
        JointRouteSupport::HnRoute => "HN Only",
        JointRouteSupport::ConProBridge => "CON Pro Bridge",
        JointRouteSupport::ConRouteOnly => "CON Only",
    }
}

/// Compact pill code for sequence ribbons, strip headers, and concordance tables.
pub const fn joint_route_short_code(route: JointRouteSupport) -> &'static str {
    match route {
        JointRouteSupport::CrossValidatedHnAndCon => "XVal",
        JointRouteSupport::HnRoute => "HN",
        JointRouteSupport::ConProBridge => "[Pro]",
        JointRouteSupport::ConRouteOnly => "CON",
    }
}

/// Scientific hover description for a canonical [`JointRouteSupport`] variant.
pub const fn joint_route_description(route: JointRouteSupport) -> &'static str {
    match route {
        JointRouteSupport::CrossValidatedHnAndCon => {
            "Both amide HN and 13C-detected 2D CON routes corroborate this residue assignment."
        }
        JointRouteSupport::HnRoute => {
            "Supported by amide HN triple-resonance spectra only (CON peak absent or overlapped)."
        }
        JointRouteSupport::ConProBridge => {
            "Proline imino 2D CON bridge (15N 131.5-144.5 ppm DSS) spanning an HN chain break."
        }
        JointRouteSupport::ConRouteOnly => {
            "Supported by protonless 2D CON route only (exchange-broadened or missing amide HN)."
        }
    }
}

/// Theme-aware accent color for a canonical [`JointRouteSupport`] variant.
pub const fn joint_route_color(route: JointRouteSupport, paper_white: bool) -> egui::Color32 {
    match (route, paper_white) {
        (JointRouteSupport::CrossValidatedHnAndCon, false) => egui::Color32::from_rgb(76, 214, 148),
        (JointRouteSupport::CrossValidatedHnAndCon, true) => egui::Color32::from_rgb(20, 128, 76),
        (JointRouteSupport::HnRoute, false) => egui::Color32::from_rgb(98, 184, 255),
        (JointRouteSupport::HnRoute, true) => egui::Color32::from_rgb(28, 102, 184),
        (JointRouteSupport::ConProBridge, false) => egui::Color32::from_rgb(204, 138, 255),
        (JointRouteSupport::ConProBridge, true) => egui::Color32::from_rgb(122, 52, 182),
        (JointRouteSupport::ConRouteOnly, false) => egui::Color32::from_rgb(248, 178, 68),
        (JointRouteSupport::ConRouteOnly, true) => egui::Color32::from_rgb(176, 108, 12),
    }
}

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

    /// Corresponding valid `--regime` CLI value (`None` for `AutoJoint`, which omits `--regime`
    /// so `ferric assign` auto-selects when both HN and `--con` inputs are supplied).
    pub const fn cli_regime_arg(self) -> Option<&'static str> {
        match self {
            Self::AutoJoint => None,
            Self::IdpJoint => Some("idp-joint"),
            Self::Idp13cOnly => Some("idp-13c"),
            Self::Globular => Some("globular"),
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

    /// Map a classifier tag or canonical 3-letter residue list to `(CasResidueType, custom_str)`.
    pub fn from_filter_tag(tag: &str, residue_types: &[String]) -> (Self, String) {
        let up = tag.trim().to_ascii_uppercase();
        match up.as_str() {
            "ALA" => (Self::Ala, String::new()),
            "GLY" => (Self::Gly, String::new()),
            "SER" => (Self::Ser, String::new()),
            "THR" => (Self::Thr, String::new()),
            "TAVI" => (Self::Tavi, String::new()),
            _ => {
                if residue_types.len() == 1 {
                    match residue_types[0].as_str() {
                        "ALA" => return (Self::Ala, String::new()),
                        "GLY" => return (Self::Gly, String::new()),
                        "SER" => return (Self::Ser, String::new()),
                        "THR" => return (Self::Thr, String::new()),
                        _ => {}
                    }
                }
                let custom = if !residue_types.is_empty() {
                    residue_types.join("+")
                } else {
                    tag.trim().to_string()
                };
                (Self::Other, custom)
            }
        }
    }

    /// Valid CLI token used in `--cas <dir>:<token>=<path>`, returning `None` when `Other` is empty
    /// or fails `parse_cas_residue_set` validation.
    pub fn cli_token(self, custom_residues: &str) -> Option<String> {
        match self {
            Self::Ala => Some("ala".to_string()),
            Self::Gly => Some("gly".to_string()),
            Self::Ser => Some("ser".to_string()),
            Self::Thr => Some("thr".to_string()),
            Self::Tavi => Some("tavi".to_string()),
            Self::Other => {
                let parsed = parse_cas_residue_set(custom_residues).ok()?;
                if parsed.is_empty() {
                    None
                } else {
                    Some(parsed.join("+").to_ascii_lowercase())
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
            Self::Unchecked => {
                "Axis nuclei and ppm bounds have not been verified yet; load spectrum or peak list to validate."
            }
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

    /// True only when axes have been explicitly validated consistent.
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

    /// Validate observed `(c_prime_ppm, n_ppm)` peak coordinates when loading a peak TSV without
    /// binary spectrum axis headers.
    pub fn validate_peaks(peaks: &[CasPeak]) -> Self {
        if peaks.is_empty() {
            return Self::Unchecked;
        }
        let mut c_min = f64::INFINITY;
        let mut c_max = f64::NEG_INFINITY;
        let mut n_min = f64::INFINITY;
        let mut n_max = f64::NEG_INFINITY;
        for p in peaks {
            if !p.c_prime_ppm.is_finite() || !p.n_ppm.is_finite() {
                return Self::OutOfRange;
            }
            c_min = c_min.min(p.c_prime_ppm);
            c_max = c_max.max(p.c_prime_ppm);
            n_min = n_min.min(p.n_ppm);
            n_max = n_max.max(p.n_ppm);
        }
        // Detect transposed peak columns (15N in C' column and 13C' in N column).
        let c_looks_like_n =
            c_min >= CAS_N15_RANGE_PPM.0 - 5.0 && c_max <= CAS_N15_RANGE_PPM.1 + 5.0;
        let n_looks_like_c =
            n_min >= CAS_C_PRIME_RANGE_PPM.0 - 5.0 && n_max <= CAS_C_PRIME_RANGE_PPM.1 + 5.0;
        if c_looks_like_n && n_looks_like_c {
            return Self::SwappedAxes;
        }
        let c_ok = c_min >= CAS_C_PRIME_RANGE_PPM.0 - 5.0 && c_max <= CAS_C_PRIME_RANGE_PPM.1 + 5.0;
        let n_ok = n_min >= CAS_N15_RANGE_PPM.0 - 5.0 && n_max <= CAS_N15_RANGE_PPM.1 + 5.0;
        if c_ok && n_ok {
            Self::Consistent
        } else {
            Self::OutOfRange
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
    /// Loaded 2D `(C'(i-1), N(i))` peaks consumed directly by [`assign_joint_idp_cpu`].
    pub peaks: Vec<CasPeak>,
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
            peaks: Vec::new(),
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

    /// Populate loaded peaks and run live axis validation.
    pub fn with_peaks_validated(
        mut self,
        peaks: Vec<CasPeak>,
        axis_override: Option<(&str, f64, f64, &str, f64, f64)>,
    ) -> Self {
        self.peak_count = peaks.len();
        self.loaded = !peaks.is_empty();
        self.axis_consistency = match axis_override {
            Some((f2_nuc, f2_hi, f2_lo, f1_nuc, f1_hi, f1_lo)) => {
                CasAxisConsistency::validate_axes(f2_nuc, f2_hi, f2_lo, f1_nuc, f1_hi, f1_lo)
            }
            None => CasAxisConsistency::validate_peaks(&peaks),
        };
        self.peaks = peaks;
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

    /// Format a strictly valid, runnable repeatable `--cas` CLI argument for this slot, or `None`
    /// if the slot is not ready, has an empty path, or has an invalid/empty `Other` selector (`P1-7`).
    pub fn to_cli_arg(&self) -> Option<String> {
        if !self.is_ready_for_solve() {
            return None;
        }
        let path_part = self.path.trim();
        if path_part.is_empty() {
            return None;
        }
        let token = self.residue_type.cli_token(&self.custom_residues)?;
        Some(format!(
            "{}:{}={}",
            self.direction.cli_prefix(),
            token,
            path_part
        ))
    }

    /// True ONLY when enabled, loaded with >= 1 peak, residue selector is valid, and axis
    /// consistency has been explicitly verified as [`CasAxisConsistency::Consistent`] (`P2-7`).
    pub fn is_ready_for_solve(&self) -> bool {
        self.enabled
            && self.loaded
            && self.peak_count > 0
            && self.axis_consistency.is_consistent()
            && self.resolved_residues().is_ok()
    }

    /// Build a [`CasSpectrum`] ready for [`JointIdpDataset::cas_spectra`] when valid and loaded.
    pub fn to_cas_spectrum(&self) -> Option<CasSpectrum> {
        if !self.is_ready_for_solve() || self.peaks.is_empty() {
            return None;
        }
        let residues = self.resolved_residues().ok()?;
        let tag = self
            .residue_type
            .cli_token(&self.custom_residues)
            .unwrap_or_else(|| "cas".to_string());
        Some(CasSpectrum {
            name: format!("{}_{}", self.direction.cli_prefix(), tag),
            mode: self.direction.to_core_mode(),
            residues,
            peaks: self.peaks.clone(),
        })
    }
}

/// Resolved `X-Pro(-Y)` bridge spanning a Proline via `2D CON` imino nitrogen and `13C`/`CAS` links.
#[derive(Clone, Debug, PartialEq)]
pub struct ProlineBridgeEntry {
    /// 0-based sequence index of the bridged Proline residue.
    pub pro_seq_idx: usize,
    /// 1-based sequence ID of the bridged Proline residue.
    pub pro_seq_id: i32,
    /// Preceding residue label (`i - 1`, e.g., `"SER13"`).
    pub prev_label: String,
    /// Proline residue label (`i`, e.g., `"PRO14"`).
    pub pro_label: String,
    /// Optional following residue label (`i + 1`, e.g., `"GLY15"`).
    pub next_label: Option<String>,
    /// Proline imino `15N` chemical shift (ppm, canonical `131.5-144.5 ppm` DSS).
    pub con_n_ppm: Option<f64>,
    /// Preceding `C'(i-1)` chemical shift (ppm) at the `CON(C'_{i-1}, N_i)` root.
    pub con_c_prime_ppm: Option<f64>,
    /// Posterior bridge confidence in `[0.0, 1.0]`.
    pub confidence: f64,
    /// True when the Proline residue was bridged by a `2D CON` node.
    pub bridged: bool,
    /// True when locked as a hard anchor for Pass-2 joint re-solving.
    pub locked: bool,
    /// Supporting experimental channels (e.g., `"CON"`, `"CANCO"`, `"CAS"`).
    pub supporting_channels: Vec<String>,
}

impl ProlineBridgeEntry {
    /// Format the sequential chain motif (e.g., `"SER13 - PRO14 - GLY15"`).
    pub fn chain_label(&self) -> String {
        match &self.next_label {
            Some(next) if !next.trim().is_empty() => {
                format!("{} - {} - {}", self.prev_label, self.pro_label, next)
            }
            _ => format!("{} - {}", self.prev_label, self.pro_label),
        }
    }

    /// Construct directly from a core [`ProlineBridgeSummary`].
    pub fn from_core_summary(
        summary: &ProlineBridgeSummary,
        seq_ids: &[i32],
        locked: bool,
        supporting_channels: Vec<String>,
    ) -> Self {
        let pro_id = seq_ids
            .get(summary.pro_seq_idx)
            .copied()
            .unwrap_or((summary.pro_seq_idx + 1) as i32);
        let prev_id = summary
            .pro_seq_idx
            .checked_sub(1)
            .and_then(|i| seq_ids.get(i).copied())
            .unwrap_or(pro_id - 1);
        let next_label = summary.next_residue.as_ref().map(|r| {
            let next_id = seq_ids
                .get(summary.pro_seq_idx + 1)
                .copied()
                .unwrap_or(pro_id + 1);
            format!("{r}{next_id}")
        });
        Self {
            pro_seq_idx: summary.pro_seq_idx,
            pro_seq_id: pro_id,
            prev_label: format!("{}{}", summary.prev_residue, prev_id),
            pro_label: format!("PRO{pro_id}"),
            next_label,
            con_n_ppm: summary.n_pro_ppm,
            con_c_prime_ppm: summary.c_prev_ppm,
            confidence: summary.confidence,
            bridged: summary.bridged,
            locked,
            supporting_channels,
        }
    }
}

/// C-terminal `COO-` anchor hint (`~179.0 ppm` DSS, Platzer et al. 2014 / `ncIDP`).
#[derive(Clone, Debug, PartialEq)]
pub struct CTermAnchorHint {
    /// 0-based sequence index of the C-terminal candidate residue.
    pub seq_idx: usize,
    /// 1-based sequence ID of the C-terminal candidate residue.
    pub seq_id: i32,
    /// Residue label (e.g., `"ALA22 (C-term)"`).
    pub residue_label: String,
    /// Observed terminal carbonyl `C'` chemical shift (ppm).
    pub observed_c_prime_ppm: f64,
    /// Offset-corrected DSS carbonyl shift (ppm).
    pub observed_c_prime_dss_ppm: f64,
    /// Expected `ncIDP` / Platzer C-terminal `COO-` reference shift (`CTERM_COO_DSS_CENTER_PPM`).
    pub expected_coo_ppm: f64,
    /// Optional associated amide `15N` shift (ppm).
    pub n_ppm: Option<f64>,
    /// Anchor posterior confidence in `[0.0, 1.0]`.
    pub confidence: f64,
    /// True when the solver assigned this candidate node to the C-terminal residue.
    pub assigned_to_cterm: bool,
    /// Human-readable diagnostic note.
    pub note: String,
}

impl CTermAnchorHint {
    /// Signed offset `observed_c_prime_dss_ppm - expected_coo_ppm` in ppm.
    pub fn delta_ppm(&self) -> f64 {
        self.observed_c_prime_dss_ppm - self.expected_coo_ppm
    }

    /// Construct directly from a core [`CTermAnchorSummary`].
    pub fn from_core_summary(
        summary: &CTermAnchorSummary,
        seq_ids: &[i32],
        n_ppm: Option<f64>,
    ) -> Self {
        let seq_id = seq_ids
            .get(summary.seq_idx)
            .copied()
            .unwrap_or((summary.seq_idx + 1) as i32);
        let suffix = if summary.assigned_to_cterm {
            " (C-term)"
        } else {
            ""
        };
        let note = if summary.assigned_to_cterm {
            format!(
                "Node #{} assigned to C-terminus (DSS C' = {:.2} ppm vs {:.1} ppm COO- center).",
                summary.node_idx, summary.c_curr_dss_ppm, CTERM_COO_DSS_CENTER_PPM
            )
        } else {
            format!(
                "Node #{} exhibits downfield C' ({:.2} ppm DSS); inspect as C-term candidate.",
                summary.node_idx, summary.c_curr_dss_ppm
            )
        };
        Self {
            seq_idx: summary.seq_idx,
            seq_id,
            residue_label: format!("{}{seq_id}{suffix}", summary.residue),
            observed_c_prime_ppm: summary.c_curr_ppm,
            observed_c_prime_dss_ppm: summary.c_curr_dss_ppm,
            expected_coo_ppm: CTERM_COO_DSS_CENTER_PPM,
            n_ppm,
            confidence: summary.confidence,
            assigned_to_cterm: summary.assigned_to_cterm,
            note,
        }
    }
}

/// Per-residue `HN <-> CON` cross-route concordance summary row backed by [`JointRouteSupport`].
#[derive(Clone, Debug, PartialEq)]
pub struct ResidueRouteConcordance {
    /// 0-based residue sequence index.
    pub seq_idx: usize,
    /// 1-based residue sequence ID.
    pub seq_id: i32,
    /// 3-letter residue name (e.g., `"SER"`, `"PRO"`, `"ALA"`).
    pub res_name: String,
    /// Canonical core route provenance tag.
    pub route: JointRouteSupport,
    /// Raw pre-calibration posterior (`0..1`).
    pub raw_posterior: f64,
    /// Calibrated joint posterior confidence (`0..1`).
    pub joint_conf: f64,
    /// Optional `N_HN - N_CON` cross-route shift residual (ppm).
    pub delta_n_ppm: Option<f64>,
    /// True if residue is Proline.
    pub is_proline: bool,
    /// True if residue exhibits a C-terminal `COO-` anchor.
    pub is_c_term: bool,
    /// True when locked as a Pass-2 anchor.
    pub locked: bool,
    /// Assigned joint node index.
    pub node_idx: Option<usize>,
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

/// Clean event enum emitted by [`IdpJointPanelState::show`]; every variant is consumed live by
/// `NmrWorkbenchApp` in `crates/nmr-gui/src/main.rs` (`180.gui:P1-5`).
#[derive(Clone, Debug, PartialEq)]
pub enum IdpJointAction {
    /// User selected a new solver regime (`AutoJoint`, `IdpJoint`, `Idp13cOnly`, `Globular`).
    RegimeChanged(IdpRegimeSelection),
    /// User toggled `--roots hnco`.
    ToggleRootsHnco(bool),
    /// User toggled `NH2` sidechain pair filtering.
    ToggleNh2Filtering(bool),
    /// User toggled TROSY `46 Hz (1/2 1J_NH)` cross-spectrum calibration.
    ToggleTrosyCalibration(bool),
    /// User toggled `F1(C')` NUS sidelobe pruning.
    ToggleNusSidelobePruning(bool),
    /// User toggled `Gly CB` bleed-through guard.
    ToggleGlyCbGuard(bool),
    /// User toggled Platzer/`ncIDP` C-terminal `COO-` shift prior.
    ToggleCTermCooPrior(bool),
    /// User toggled conservative 2-pass anchor-lock propagation.
    ToggleTwoPassAnchorLock(bool),
    /// User adjusted spectrometer proton frequency (MHz) used for `46 Hz (1/2 1J_NH)` scaling.
    SpectrometerMhzChanged(f64),
    /// User added a new `--cas` experiment slot.
    AddCasSlot(CasSlotEntry),
    /// User removed an existing `--cas` slot by its stable `id`.
    RemoveCasSlot(usize),
    /// User modified an existing `--cas` slot (residue type, custom selector, direction, enabled, or path).
    UpdateCasSlot(CasSlotEntry),
    /// User requested loading a spectrum/peak list into the given CAS slot `id`.
    RequestLoadCasSlot(usize),
    /// User clicked the primary Run IDP Assignment button.
    RunJointAssignment,
    /// User clicked a residue row or Proline bridge to focus it in the sequence ribbon / strip tiler.
    InspectResidue {
        seq_id: i32,
        route: JointRouteSupport,
    },
    /// User toggled anchor lock on a resolved `X-Pro` bridge.
    ToggleProBridgeLock { pro_seq_id: i32, locked: bool },
    /// User clicked a C-terminal `COO-` anchor hint to inspect its carbonyl shift in the strip view.
    FocusCTermAnchor { seq_id: i32, c_prime_ppm: f64 },
    /// User loaded and solved the built-in clean-room synthetic Pro/Ser-rich IDP demonstration.
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

/// Optional input file path hints mirrored in [`IdpJointPanelState::cli_preview_command`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct JointCliPathHints {
    pub seq_path: String,
    pub out_prefix: String,
    pub peaks_path: String,
    pub con_path: String,
    pub cccon_path: String,
    pub hncacon_path: String,
    pub hncaconh_path: String,
}

/// Complete state for the Unified Joint `HN + 13C` IDP Workbench & Interactive `--cas` Slot Mapper.
#[derive(Clone, Debug, PartialEq)]
pub struct IdpJointPanelState {
    pub open: bool,
    pub regime: IdpRegimeSelection,
    pub roots_hnco: bool,
    pub filter_nh2_pairs: bool,
    pub trosy_half_j_calibration: bool,
    pub spectrometer_mhz: f64,
    pub temperature_k: f64,
    pub ph: f64,
    pub hncacb_polarity: HncacbPolarity,
    pub f1_nus_sidelobe_pruning: bool,
    pub gly_cb_bleed_guard: bool,
    pub c_term_coo_prior: bool,
    pub two_pass_anchor_lock: bool,
    pub cli_paths: JointCliPathHints,
    pub cas_slots: Vec<CasSlotEntry>,
    pub next_slot_id: usize,
    pub draft_residue_type: CasResidueType,
    pub draft_custom_residues: String,
    pub draft_direction: CasDirection,
    pub draft_path: String,
    pub proline_bridges: Vec<ProlineBridgeEntry>,
    pub c_term_hints: Vec<CTermAnchorHint>,
    pub concordance_rows: Vec<ResidueRouteConcordance>,
    pub last_qc_verdict: Option<DatasetQcVerdict>,
    pub last_qc_reasons: Vec<String>,
    pub last_trosy_calibration: Option<TrosyOffsetCalibration>,
    pub last_fusion_rms_ppm: Option<(f64, f64)>,
    pub last_fitted_offsets: Option<IdpShiftOffsets>,
    pub last_n_nodes: usize,
    pub active_tab: IdpJointSubTab,
    pub route_filter: Option<JointRouteSupport>,
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
    /// Create a fresh `IdpJointPanelState` with canonical `idp-joint` defaults.
    pub fn new() -> Self {
        Self {
            open: false,
            regime: IdpRegimeSelection::AutoJoint,
            roots_hnco: true,
            filter_nh2_pairs: true,
            trosy_half_j_calibration: true,
            spectrometer_mhz: CANONICAL_TROSY_FALLBACK_MHZ,
            temperature_k: 298.15,
            ph: 6.5,
            hncacb_polarity: HncacbPolarity::Auto,
            f1_nus_sidelobe_pruning: true,
            gly_cb_bleed_guard: true,
            c_term_coo_prior: true,
            two_pass_anchor_lock: true,
            cli_paths: JointCliPathHints::default(),
            cas_slots: Vec::new(),
            next_slot_id: 1,
            draft_residue_type: CasResidueType::Ser,
            draft_custom_residues: String::new(),
            draft_direction: CasDirection::Cacon,
            draft_path: String::new(),
            proline_bridges: Vec::new(),
            c_term_hints: Vec::new(),
            concordance_rows: Vec::new(),
            last_qc_verdict: None,
            last_qc_reasons: Vec::new(),
            last_trosy_calibration: None,
            last_fusion_rms_ppm: None,
            last_fitted_offsets: None,
            last_n_nodes: 0,
            active_tab: IdpJointSubTab::RegimeAndCas,
            route_filter: None,
            selected_residue: None,
            status_message: "Ready: configure regime and CAS slots, or run idp-joint.".to_string(),
            solver_running: false,
            synthetic_demo_loaded: false,
        }
    }

    /// Add a new `--cas` slot bounded by [`MAX_CAS_SLOTS`] (`P2-8`), returning its stable `id` or
    /// an explicit error when the cap is reached.
    pub fn add_cas_slot(
        &mut self,
        residue_type: CasResidueType,
        custom_residues: impl Into<String>,
        direction: CasDirection,
        path: impl Into<String>,
    ) -> Result<usize, String> {
        if self.cas_slots.len() >= MAX_CAS_SLOTS {
            let msg = format!(
                "Cannot add CAS slot: limit of {MAX_CAS_SLOTS} slots reached (remove an existing slot first)."
            );
            self.status_message = msg.clone();
            return Err(msg);
        }
        let id = self.next_slot_id;
        self.next_slot_id = self.next_slot_id.saturating_add(1);
        let entry = CasSlotEntry::new(id, residue_type, direction, path)
            .with_custom_residues(custom_residues);
        self.cas_slots.push(entry);
        Ok(id)
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

    /// Update the path of an existing `--cas` slot and reset its loaded / axis-consistency state
    /// to [`CasAxisConsistency::Unchecked`] until re-validated (`P2-7`).
    pub fn set_cas_slot_path(&mut self, slot_id: usize, new_path: impl Into<String>) -> bool {
        if let Some(slot) = self.cas_slots.iter_mut().find(|s| s.id == slot_id) {
            slot.path = new_path.into();
            slot.loaded = false;
            slot.peak_count = 0;
            slot.peaks.clear();
            slot.axis_consistency = CasAxisConsistency::Unchecked;
            true
        } else {
            false
        }
    }

    /// Validate `(f2, f1)` axis nuclei and ppm bounds live for an existing `--cas` slot (`P2-7`).
    #[allow(clippy::too_many_arguments)]
    pub fn set_cas_slot_axis_window(
        &mut self,
        slot_id: usize,
        f2_nuc: &str,
        f2_hi: f64,
        f2_lo: f64,
        f1_nuc: &str,
        f1_hi: f64,
        f1_lo: f64,
    ) -> Option<CasAxisConsistency> {
        let slot = self.cas_slots.iter_mut().find(|s| s.id == slot_id)?;
        let status = CasAxisConsistency::validate_axes(f2_nuc, f2_hi, f2_lo, f1_nuc, f1_hi, f1_lo);
        slot.axis_consistency = status;
        Some(status)
    }

    /// Load 2D CAS peaks into an existing slot and run live axis validation (`P2-7`).
    pub fn load_cas_slot_from_peaks(
        &mut self,
        slot_id: usize,
        peaks: Vec<CasPeak>,
        axis_override: Option<(&str, f64, f64, &str, f64, f64)>,
    ) -> Option<CasAxisConsistency> {
        let slot = self.cas_slots.iter_mut().find(|s| s.id == slot_id)?;
        let status = match axis_override {
            Some((f2_nuc, f2_hi, f2_lo, f1_nuc, f1_hi, f1_lo)) => {
                CasAxisConsistency::validate_axes(f2_nuc, f2_hi, f2_lo, f1_nuc, f1_hi, f1_lo)
            }
            None => CasAxisConsistency::validate_peaks(&peaks),
        };
        slot.peak_count = peaks.len();
        slot.loaded = !peaks.is_empty();
        slot.peaks = peaks;
        slot.axis_consistency = status;
        Some(status)
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

    /// Replace CAS slots from an ingest execution plan while respecting [`MAX_CAS_SLOTS`].
    pub fn sync_cas_slots_from_ingest(&mut self, incoming: Vec<CasSlotEntry>) -> usize {
        self.cas_slots.clear();
        self.next_slot_id = 1;
        let mut added = 0usize;
        for mut slot in incoming {
            if self.cas_slots.len() >= MAX_CAS_SLOTS {
                break;
            }
            slot.id = self.next_slot_id;
            self.next_slot_id = self.next_slot_id.saturating_add(1);
            self.cas_slots.push(slot);
            added += 1;
        }
        added
    }

    /// Collect all enabled, loaded, axis-consistent CAS slots as [`CasSpectrum`] objects for the solver.
    pub fn build_ready_cas_spectra(&self) -> Vec<CasSpectrum> {
        self.cas_slots
            .iter()
            .filter_map(CasSlotEntry::to_cas_spectrum)
            .collect()
    }

    /// Count of enabled, loaded, axis-consistent `--cas` slots ready for the solver.
    pub fn ready_cas_slot_count(&self) -> usize {
        self.cas_slots
            .iter()
            .filter(|s| s.is_ready_for_solve())
            .count()
    }

    /// Canonical field-scaled TROSY `46 Hz (1/2 1J_NH)` signed shift `(d1H_ppm, d15N_ppm)` directly
    /// delegated to [`nmr_core::joint_idp_solver::trosy_expected_shift_ppm`] (`180.gui:P1-6`).
    pub fn trosy_half_j_offsets_ppm(&self) -> (f64, f64) {
        trosy_expected_shift_ppm(self.spectrometer_mhz)
    }

    /// Build a [`JointIdpConfig`] mirroring the panel's current toggles and conditions.
    pub fn build_joint_config(&self, seed: u64) -> JointIdpConfig {
        #[allow(unused_mut)]
        let mut cfg = JointIdpConfig {
            temperature_k: if self.temperature_k.is_finite() {
                self.temperature_k.clamp(250.0, 360.0)
            } else {
                298.15
            },
            ph: if self.ph.is_finite() {
                self.ph.clamp(1.0, 12.0)
            } else {
                6.5
            },
            proton_mhz: if self.spectrometer_mhz.is_finite() && self.spectrometer_mhz > 50.0 {
                self.spectrometer_mhz
            } else {
                CANONICAL_TROSY_FALLBACK_MHZ
            },
            hncacb_polarity: self.hncacb_polarity,
            auto_trosy_calibration: self.trosy_half_j_calibration,
            enable_nus_sidelobe_pruning: self.f1_nus_sidelobe_pruning,
            enable_gly_cb_guard: self.gly_cb_bleed_guard,
            enable_cterm_coo_prior: self.c_term_coo_prior,
            enable_two_pass_anchor_lock: self.two_pass_anchor_lock,
            include_hsqc_without_hnco: !self.roots_hnco,
            seed,
            ..JointIdpConfig::default()
        };
        #[cfg(target_arch = "wasm32")]
        {
            cfg.replicas = cfg.replicas.min(12);
            cfg.generations = cfg.generations.min(2);
            cfg.steps_per_system = cfg.steps_per_system.min(450);
        }
        cfg
    }

    /// Populate Proline bridges, C-terminal `COO-` anchor hints, residue route concordance, and QC
    /// diagnostics directly from a real [`JointIdpAssignResult`] (`180.gui:P1-6`).
    pub fn sync_from_joint_result(&mut self, result: &JointIdpAssignResult, seq_ids: &[i32]) {
        self.last_qc_verdict = Some(result.qc_verdict);
        self.last_qc_reasons = result.qc_reasons.clone();
        self.last_trosy_calibration = Some(result.trosy_calibration.clone());
        self.last_fusion_rms_ppm = Some(result.fusion_rms_ppm);
        self.last_fitted_offsets = Some(result.fitted_offsets);
        self.last_n_nodes = result.nodes.len();

        let locked_nodes: BTreeMap<usize, bool> = result
            .pass1_locked_anchors
            .iter()
            .map(|&(_nid, k)| (k, true))
            .collect();

        // 1. Proline bridges from core `proline_bridge_summaries()`
        self.proline_bridges = result
            .proline_bridge_summaries()
            .iter()
            .map(|summary| {
                let mut channels = Vec::new();
                if let Some(nid) = summary.pro_node_idx {
                    if let Some(node) = result.nodes.get(nid) {
                        if node.con_matched {
                            channels.push("CON".to_string());
                        }
                        if !node.ca_curr.is_empty() || !node.ca_prev.is_empty() {
                            channels.push("CANCO/CA".to_string());
                        }
                        if !node.cb_prev.is_empty() || !node.sidechain_c_prev.is_empty() {
                            channels.push("H(CC)CON".to_string());
                        }
                        if !node.nn_prev_cands.is_empty() {
                            channels.push("(H)N(CA)CON".to_string());
                        }
                    }
                }
                let locked = locked_nodes
                    .get(&summary.pro_seq_idx)
                    .copied()
                    .unwrap_or(false);
                ProlineBridgeEntry::from_core_summary(summary, seq_ids, locked, channels)
            })
            .collect();

        // 2. C-terminal `COO-` anchors from core `cterm_anchor_summaries()`
        self.c_term_hints = result
            .cterm_anchor_summaries()
            .iter()
            .map(|summary| {
                let n_ppm = result.nodes.get(summary.node_idx).map(|n| n.n_curr);
                CTermAnchorHint::from_core_summary(summary, seq_ids, n_ppm)
            })
            .collect();

        // 3. Per-residue `HN <-> CON` cross-route concordance directly from `residue_assignments`
        self.concordance_rows = result
            .residue_assignments
            .iter()
            .filter_map(|ra| {
                let route = ra.route_support?;
                let seq_id = seq_ids
                    .get(ra.seq_idx)
                    .copied()
                    .unwrap_or((ra.seq_idx + 1) as i32);
                let delta_n_ppm = ra.node_idx.and_then(|nid| {
                    let node = result.nodes.get(nid)?;
                    if node.hn_matched && node.con_matched {
                        node.hn_n_curr.map(|hn_n| hn_n - node.n_curr)
                    } else {
                        None
                    }
                });
                Some(ResidueRouteConcordance {
                    seq_idx: ra.seq_idx,
                    seq_id,
                    res_name: ra.residue.clone(),
                    route,
                    raw_posterior: ra.raw_posterior,
                    joint_conf: ra.confidence,
                    delta_n_ppm,
                    is_proline: ra.is_proline,
                    is_c_term: ra.is_cterm_coo,
                    locked: ra.anchor_locked,
                    node_idx: ra.node_idx,
                })
            })
            .collect();

        let (pro_ok, pro_tot) = result.proline_bridge_recovery;
        self.status_message = format!(
            "Joint IDP solved: {} nodes, {} assigned ({} XVal, {} Pro/{pro_tot}, {} CON-only, {} HN-only), QC={:?}",
            result.nodes.len(),
            self.concordance_rows.len(),
            result.n_cross_validated_hn_con,
            pro_ok,
            result.n_con_route_only,
            result.n_hn_route_only,
            result.qc_verdict
        );
    }

    /// Build a `0`-based residue index -> [`JointRouteSupport`] lookup map for the sequence ribbon
    /// and strip tiler.
    pub fn route_map_by_res_idx(&self) -> BTreeMap<usize, JointRouteSupport> {
        self.concordance_rows
            .iter()
            .map(|row| (row.seq_idx, row.route))
            .collect()
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
            match row.route {
                JointRouteSupport::CrossValidatedHnAndCon => stats.cross_validated += 1,
                JointRouteSupport::HnRoute => stats.hn_only += 1,
                JointRouteSupport::ConProBridge => stats.con_pro_bridge += 1,
                JointRouteSupport::ConRouteOnly => stats.con_only += 1,
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
        }
        stats.mean_joint_conf = conf_sum / (self.concordance_rows.len() as f64);
        stats
    }

    /// Build a strictly valid, runnable `ferric assign` CLI command mirroring current GUI state
    /// (`180.gui:P1-7`). Emits ONLY real flags parsed by `crates/nmr-cli/src/main.rs::cmd_assign`,
    /// skips unloaded / unchecked / invalid / empty-path CAS slots, and never emits placeholder
    /// paths like `<unassigned_path>` or invalid tokens like `"custom"`.
    pub fn cli_preview_command(&self) -> String {
        let mut out = String::from("ferric assign");
        if let Some(regime_arg) = self.regime.cli_regime_arg() {
            let _ = write!(out, " --regime {regime_arg}");
        }
        let seq_arg = if self.cli_paths.seq_path.trim().is_empty() {
            "sequence.fasta"
        } else {
            self.cli_paths.seq_path.trim()
        };
        let out_arg = if self.cli_paths.out_prefix.trim().is_empty() {
            "idp_joint_out"
        } else {
            self.cli_paths.out_prefix.trim()
        };
        let _ = write!(out, " --seq {seq_arg} --out {out_arg} --engine cpu");

        if !self.cli_paths.peaks_path.trim().is_empty() && self.regime.uses_hn_route() {
            let _ = write!(out, " --peaks {}", self.cli_paths.peaks_path.trim());
        }
        if !self.cli_paths.con_path.trim().is_empty() && self.regime.uses_con_route() {
            let _ = write!(out, " --con {}", self.cli_paths.con_path.trim());
        }
        if !self.cli_paths.cccon_path.trim().is_empty() && self.regime.uses_con_route() {
            let _ = write!(out, " --cccon {}", self.cli_paths.cccon_path.trim());
        }
        if !self.cli_paths.hncacon_path.trim().is_empty() && self.regime.uses_con_route() {
            let _ = write!(out, " --hncacon {}", self.cli_paths.hncacon_path.trim());
        }
        if !self.cli_paths.hncaconh_path.trim().is_empty() && self.regime.uses_hn_route() {
            let _ = write!(out, " --hncaconh {}", self.cli_paths.hncaconh_path.trim());
        }
        if self.roots_hnco && self.regime.uses_hn_route() {
            out.push_str(" --roots hnco");
        }
        if self.hncacb_polarity != HncacbPolarity::Auto && self.regime.uses_hn_route() {
            let _ = write!(out, " --hncacb-polarity {}", self.hncacb_polarity.as_str());
        }
        if self.regime.uses_con_route() {
            let t_k = if self.temperature_k.is_finite() {
                self.temperature_k.clamp(250.0, 360.0)
            } else {
                298.15
            };
            let ph = if self.ph.is_finite() {
                self.ph.clamp(1.0, 12.0)
            } else {
                6.5
            };
            let _ = write!(out, " --temp-k {t_k:.2} --ph {ph:.2}");
        }
        for slot in &self.cas_slots {
            if let Some(cas_arg) = slot.to_cli_arg() {
                let _ = write!(out, " --cas {cas_arg}");
            }
        }
        out
    }

    /// Synthesize a clean-room 22-residue Pro/Ser-rich IDP dataset, run the REAL
    /// [`assign_joint_idp_cpu`] solver (fast, `wasm32`-clean, zero hardcoded fake results), and
    /// populate this panel via [`Self::sync_from_joint_result`] (`180.gui:P1-6`).
    pub fn load_synthetic_idp_demo(
        &mut self,
    ) -> Result<(Vec<String>, Vec<i32>, JointIdpDataset, JointIdpAssignResult), String> {
        self.regime = IdpRegimeSelection::IdpJoint;
        self.roots_hnco = true;
        self.filter_nh2_pairs = true;
        self.trosy_half_j_calibration = true;
        self.spectrometer_mhz = CANONICAL_TROSY_FALLBACK_MHZ;
        self.temperature_k = 298.15;
        self.ph = 6.5;
        self.f1_nus_sidelobe_pruning = true;
        self.gly_cb_bleed_guard = true;
        self.c_term_coo_prior = true;
        self.two_pass_anchor_lock = true;

        let seq: Vec<String> = [
            "MET", "ALA", "SER", "PRO", "GLY", "SER", "THR", "VAL", "ASP", "GLU", "ALA", "PRO",
            "PRO", "THR", "LEU", "LYS", "ASN", "GLY", "SER", "ILE", "GLN", "ALA",
        ]
        .iter()
        .map(|s| (*s).to_string())
        .collect();
        let seq_ids: Vec<i32> = (1..=(seq.len() as i32)).collect();
        let n_res = seq.len();

        let mut cfg = self.build_joint_config(42);
        cfg.replicas = 12;
        cfg.generations = 2;
        cfg.steps_per_system = 300;

        // Compute clean-room ncIDP coil shifts with pre-Pro corrections and C-term COO- shift.
        let preds = predict_joint_ncidp_shifts(&seq, &cfg);
        let mut h_sh = vec![8.20_f64; n_res];
        let mut n_sh = vec![120.0_f64; n_res];
        let mut c_sh = vec![174.0_f64; n_res];
        let mut ca_sh = vec![55.0_f64; n_res];
        let mut cb_sh: Vec<Option<f64>> = vec![None; n_res];
        for (k, sh) in preds.iter().enumerate() {
            // Deterministic slight site dispersion so peaks are distinct without any RNG dependency.
            let spread = ((k as f64) * 1.618_033_988_7).sin();
            h_sh[k] = 8.15 + 0.35 * spread;
            n_sh[k] = sh.n_ppm + 0.45 * spread;
            c_sh[k] = if k + 1 == n_res {
                CTERM_COO_DSS_CENTER_PPM + 0.35
            } else {
                sh.c_ppm + 0.25 * spread
            };
            ca_sh[k] = sh.ca_ppm + 0.30 * spread;
            cb_sh[k] = sh.cb_ppm.map(|v| v + 0.28 * spread);
        }

        let mut hn_peaks = Vec::new();
        let mut con_roots = Vec::new();
        let mut strip_peaks = Vec::new();
        let mut ser_peaks = Vec::new();
        let mut ala_peaks = Vec::new();
        let mut gly_peaks = Vec::new();

        for k in 1..n_res {
            let is_pro = seq[k] == "PRO";
            // Simulate one exchange-broadened amide (k == 5, SER6) seen only in CON,
            // and one CON-missing amide (k == 14, LEU15) seen only in HN so all 4 route badges appear.
            let has_hn = !is_pro && k != 5;
            let has_con = k != 14;

            if has_hn {
                let h = h_sh[k];
                let n = n_sh[k];
                let mk_peak = |exp: &str, c: Option<f64>, sign: i32, amp: f64| Peak {
                    exp: exp.to_string(),
                    h,
                    n,
                    c,
                    sign,
                    height: Some(amp),
                    volume: Some(amp),
                    h2: None,
                };
                hn_peaks.push(mk_peak("HSQC", None, 1, 1.0));
                hn_peaks.push(mk_peak("HNCO", Some(c_sh[k - 1]), 1, 1.0));
                hn_peaks.push(mk_peak("HNCACO", Some(c_sh[k]), 1, 0.9));
                hn_peaks.push(mk_peak("HNCACB", Some(ca_sh[k]), 1, 1.0));
                if let Some(cb) = cb_sh[k] {
                    hn_peaks.push(mk_peak("HNCACB", Some(cb), -1, 0.85));
                }
                hn_peaks.push(mk_peak("CBCACONH", Some(ca_sh[k - 1]), 1, 0.95));
                if let Some(cb_prev) = cb_sh[k - 1] {
                    hn_peaks.push(mk_peak("CBCACONH", Some(cb_prev), -1, 0.80));
                }
            }

            if has_con {
                con_roots.push(Con2DPeak {
                    c_prev_ppm: c_sh[k - 1],
                    n_curr_ppm: n_sh[k],
                    height: 1.0,
                    volume: 1.0,
                    is_proline_imino_region: n_sh[k] >= PROLINE_N15_MIN_PPM,
                    deconvolved_from_cluster: false,
                });
                strip_peaks.push(C13Strip3DPeak {
                    exp: C13DetectedExp::Canco3D,
                    c_prime_ppm: c_sh[k - 1],
                    n_ppm: n_sh[k],
                    c_indirect_ppm: ca_sh[k],
                    sign: 1,
                    height: 1.0,
                    volume: 1.0,
                    anchor_root_idx: None,
                });
                strip_peaks.push(C13Strip3DPeak {
                    exp: C13DetectedExp::Canco3D,
                    c_prime_ppm: c_sh[k - 1],
                    n_ppm: n_sh[k],
                    c_indirect_ppm: ca_sh[k - 1],
                    sign: 1,
                    height: 0.65,
                    volume: 0.65,
                    anchor_root_idx: None,
                });
                strip_peaks.push(C13Strip3DPeak {
                    exp: C13DetectedExp::Cbcaco3D,
                    c_prime_ppm: c_sh[k - 1],
                    n_ppm: n_sh[k],
                    c_indirect_ppm: ca_sh[k - 1],
                    sign: 1,
                    height: 0.95,
                    volume: 0.95,
                    anchor_root_idx: None,
                });
                if let Some(cb_prev) = cb_sh[k - 1] {
                    strip_peaks.push(C13Strip3DPeak {
                        exp: C13DetectedExp::Cbcaco3D,
                        c_prime_ppm: c_sh[k - 1],
                        n_ppm: n_sh[k],
                        c_indirect_ppm: cb_prev,
                        sign: -1,
                        height: 0.85,
                        volume: 0.85,
                        anchor_root_idx: None,
                    });
                }
                if seq[k - 1] == "SER" {
                    ser_peaks.push(CasPeak {
                        c_prime_ppm: c_sh[k - 1],
                        n_ppm: n_sh[k],
                        height: 1.0,
                        volume: 1.0,
                    });
                }
                if seq[k - 1] == "ALA" {
                    ala_peaks.push(CasPeak {
                        c_prime_ppm: c_sh[k - 1],
                        n_ppm: n_sh[k],
                        height: 1.0,
                        volume: 1.0,
                    });
                }
                if seq[k - 1] == "GLY" {
                    gly_peaks.push(CasPeak {
                        c_prime_ppm: c_sh[k - 1],
                        n_ppm: n_sh[k],
                        height: 1.0,
                        volume: 1.0,
                    });
                }
            }
        }

        // Populate CAS slots through live validation (`P2-7`).
        self.cas_slots.clear();
        self.next_slot_id = 1;
        if let Ok(id_ser) = self.add_cas_slot(
            CasResidueType::Ser,
            "",
            CasDirection::Cacon,
            "demo/cacon_ser.tsv",
        ) {
            let _ = self.load_cas_slot_from_peaks(id_ser, ser_peaks, None);
        }
        if let Ok(id_ala) = self.add_cas_slot(
            CasResidueType::Ala,
            "",
            CasDirection::Cacon,
            "demo/cacon_ala.tsv",
        ) {
            let _ = self.load_cas_slot_from_peaks(id_ala, ala_peaks, None);
        }
        if let Ok(id_gly) = self.add_cas_slot(
            CasResidueType::Gly,
            "",
            CasDirection::Cacon,
            "demo/cacon_gly.tsv",
        ) {
            let _ = self.load_cas_slot_from_peaks(id_gly, gly_peaks, None);
        }

        let ds = JointIdpDataset {
            hn_peaks,
            c13_dataset: C13PickedDataset {
                con_roots,
                caco_peaks: Vec::new(),
                strip_peaks,
            },
            cas_spectra: self.build_ready_cas_spectra(),
            hncacon_nn_edges: Vec::new(),
            hncaconh_peaks: Vec::new(),
            reciprocal_nn_pairs: Vec::new(),
            proton_mhz: self.spectrometer_mhz,
            user_locks: Vec::new(),
        };

        let res = assign_joint_idp_cpu(&ds, &seq, &cfg)?;
        self.sync_from_joint_result(&res, &seq_ids);
        self.synthetic_demo_loaded = true;
        self.status_message = format!(
            "Synthetic Demo solved (real assign_joint_idp_cpu): {} residues, {} X-Pro bridges, QC={:?}",
            self.concordance_rows.len(),
            self.proline_bridges.iter().filter(|b| b.bridged).count(),
            res.qc_verdict
        );
        Ok((seq, seq_ids, ds, res))
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
        let bridged_pro = self.proline_bridges.iter().filter(|b| b.bridged).count();

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
                        &format!("X-Pro {bridged_pro}/{}", self.proline_bridges.len()),
                        joint_route_color(JointRouteSupport::ConProBridge, paper_white),
                        theme.pill_bg_alpha,
                    );
                    if stats.total > 0 {
                        draw_status_pill(
                            ui,
                            &format!("XVal {}/{}", stats.cross_validated, stats.total),
                            joint_route_color(
                                JointRouteSupport::CrossValidatedHnAndCon,
                                paper_white,
                            ),
                            theme.pill_bg_alpha,
                        );
                    }
                    if let Some(qc) = self.last_qc_verdict {
                        let (qc_lbl, qc_col) = match qc {
                            DatasetQcVerdict::Pass => ("QC PASS", theme.ok),
                            DatasetQcVerdict::Warn => ("QC WARN", theme.warn),
                            DatasetQcVerdict::Abstain => ("QC ABSTAIN", theme.err),
                        };
                        draw_status_pill(ui, qc_lbl, qc_col, theme.pill_bg_alpha);
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
                            "Run real assign_joint_idp_cpu on a clean-room synthetic Pro/Ser IDP.",
                        )
                        .clicked()
                    {
                        let _ = self.load_synthetic_idp_demo();
                        actions.push(IdpJointAction::LoadSyntheticDemo);
                    }

                    ui.label(
                        egui::RichText::new(&self.status_message)
                            .size(TYPE_CAPTION_PX)
                            .color(theme.text_muted),
                    );
                });

                if !self.last_qc_reasons.is_empty() {
                    ui.add_space(3.0);
                    let summary = compact_text(&self.last_qc_reasons.join("; "), 130);
                    ui.label(
                        egui::RichText::new(format!("QC Notes: {summary}"))
                            .size(TYPE_CAPTION_PX)
                            .color(theme.warn),
                    );
                }
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
                                egui::RichText::new("TROSY 1/2 1J_NH auto-calibration")
                                    .size(TYPE_BODY_PX),
                            ),
                        )
                        .on_hover_text(
                            "Auto-calibrate 46 Hz (1/2 1J_NH), field-scaled to ppm, between TROSY 3Ds and decoupled HSQC/HNCO.",
                        )
                        .changed()
                    {
                        actions.push(IdpJointAction::ToggleTrosyCalibration(
                            self.trosy_half_j_calibration,
                        ));
                    }
                });

                ui.add_space(4.0);
                // Field-scaled TROSY offset readout via canonical `trosy_expected_shift_ppm`
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
                            "46 Hz (1/2 1J_NH), field-scaled to ppm: d1H = {dh:+.4} ppm, d15N = {dn:+.3} ppm"
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
                            egui::RichText::new(format!(
                                "C-term COO- (~{CTERM_COO_DSS_CENTER_PPM:.0} ppm) prior"
                            ))
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
                )
                .on_hover_text(self.cli_preview_command());
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
                        egui::RichText::new(format!(
                            "Map 2D AA-selective (C'(i-1), N(i)) subspectra ({}/{MAX_CAS_SLOTS}).",
                            self.cas_slots.len()
                        ))
                        .size(TYPE_CAPTION_PX)
                        .color(theme.text_muted),
                    );
                });

                ui.add_space(6.0);
                // Add-slot bar
                let at_cap = self.cas_slots.len() >= MAX_CAS_SLOTS;
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

                    let draft_valid = !at_cap
                        && self
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
                        let token_part = self
                            .draft_residue_type
                            .cli_token(&self.draft_custom_residues)
                            .unwrap_or_else(|| "cas".to_string());
                        let path = if self.draft_path.trim().is_empty() {
                            format!("{}_{}.tsv", self.draft_direction.cli_prefix(), token_part)
                        } else {
                            self.draft_path.trim().to_string()
                        };
                        if let Ok(id) = self.add_cas_slot(
                            self.draft_residue_type,
                            self.draft_custom_residues.trim().to_string(),
                            self.draft_direction,
                            path,
                        ) {
                            if let Some(created) =
                                self.cas_slots.iter().find(|s| s.id == id).cloned()
                            {
                                actions.push(IdpJointAction::AddCasSlot(created));
                            }
                            self.draft_path.clear();
                        }
                    }
                });

                if at_cap {
                    ui.add_space(2.0);
                    ui.label(
                        egui::RichText::new(format!(
                            "Maximum of {MAX_CAS_SLOTS} CAS slots reached."
                        ))
                        .size(TYPE_CAPTION_PX)
                        .color(theme.warn),
                    );
                }

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
                        egui::RichText::new(format!(
                            "Resolved X-Pro Bridges (2D CON Imino 15N {PROLINE_N15_MIN_PPM:.1}-{PROLINE_N15_MAX_PPM:.1} ppm DSS)"
                        ))
                        .size(TYPE_BODY_PX)
                        .strong()
                        .color(theme.text_primary),
                    );
                    draw_route_support_pill(ui, JointRouteSupport::ConProBridge, paper_white);
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
                            let n_str = bridge
                                .con_n_ppm
                                .map(|v| format!("{v:.2}"))
                                .unwrap_or_else(|| "-".to_string());
                            ui.label(egui::RichText::new(n_str).size(TYPE_BODY_PX).monospace());
                            let c_str = bridge
                                .con_c_prime_ppm
                                .map(|v| format!("{v:.2}"))
                                .unwrap_or_else(|| "-".to_string());
                            ui.label(egui::RichText::new(c_str).size(TYPE_BODY_PX).monospace());
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
                                        route: JointRouteSupport::ConProBridge,
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
                    egui::RichText::new(format!(
                        "C-Terminal COO- Anchor Hints (Platzer / ncIDP ~{CTERM_COO_DSS_CENTER_PPM:.1} ppm DSS)"
                    ))
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
                            if hint.assigned_to_cterm {
                                "ASSIGNED C-TERM"
                            } else {
                                "CANDIDATE"
                            },
                            if hint.assigned_to_cterm {
                                theme.ok
                            } else {
                                theme.warn
                            },
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
                                "C' = {:.2} ppm (DSS {:.2}, delta {:+.2} ppm)",
                                hint.observed_c_prime_ppm,
                                hint.observed_c_prime_dss_ppm,
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
                    for route in ALL_JOINT_ROUTE_SUPPORTS {
                        let count = match route {
                            JointRouteSupport::CrossValidatedHnAndCon => stats.cross_validated,
                            JointRouteSupport::HnRoute => stats.hn_only,
                            JointRouteSupport::ConProBridge => stats.con_pro_bridge,
                            JointRouteSupport::ConRouteOnly => stats.con_only,
                        };
                        let selected = self.route_filter == Some(route);
                        let label = format!("{} ({count})", joint_route_short_code(route));
                        if ui
                            .add(egui::Button::selectable(
                                selected,
                                egui::RichText::new(label)
                                    .size(TYPE_CAPTION_PX)
                                    .color(joint_route_color(route, paper_white)),
                            ))
                            .on_hover_text(joint_route_description(route))
                            .clicked()
                        {
                            self.route_filter = if selected { None } else { Some(route) };
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
                        if let Some((rms_c, rms_n)) = self.last_fusion_rms_ppm {
                            ui.label(
                                egui::RichText::new(format!(
                                    "Fusion RMS: C' {rms_c:.3} / N {rms_n:.3} ppm"
                                ))
                                .size(TYPE_CAPTION_PX)
                                .color(theme.text_muted),
                            );
                        }
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

                let mut inspect_req: Option<(i32, JointRouteSupport)> = None;

                egui::Grid::new("idp_concordance_grid")
                    .num_columns(7)
                    .spacing([12.0, 5.0])
                    .striped(true)
                    .show(ui, |ui| {
                        for header in [
                            "Residue",
                            "Route",
                            "Raw Post.",
                            "Calib. Conf",
                            "dN (ppm)",
                            "Flags",
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
                                if row.route != filter {
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

                            draw_route_support_pill(ui, row.route, paper_white);

                            ui.label(
                                egui::RichText::new(format!("{:.0}%", row.raw_posterior * 100.0))
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

                            let delta_str = row
                                .delta_n_ppm
                                .map(|dn| format!("{dn:+.2}"))
                                .unwrap_or_else(|| "-".to_string());
                            ui.label(
                                egui::RichText::new(delta_str)
                                    .size(TYPE_CAPTION_PX)
                                    .monospace()
                                    .color(theme.text_muted),
                            );

                            let mut tags: Vec<&str> = Vec::new();
                            if row.is_proline {
                                tags.push("PRO");
                            }
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
                                inspect_req = Some((row.seq_id, row.route));
                            }
                            ui.end_row();
                        }
                    });

                if let Some((seq_id, route)) = inspect_req {
                    self.selected_residue = Some(seq_id);
                    actions.push(IdpJointAction::InspectResidue { seq_id, route });
                }
            });
    }
}

/// Reusable zero-emoji pill widget for [`JointRouteSupport`] (used in concordance tables,
/// sequence ribbons, and strip headers).
pub fn draw_route_support_pill(
    ui: &mut egui::Ui,
    route: JointRouteSupport,
    paper_white: bool,
) -> egui::Response {
    let color = joint_route_color(route, paper_white);
    let alpha = if paper_white { 28 } else { 36 };
    draw_status_pill(ui, joint_route_short_code(route), color, alpha)
        .on_hover_text(joint_route_description(route))
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
