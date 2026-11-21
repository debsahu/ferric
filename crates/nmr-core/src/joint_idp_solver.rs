//! Unified Joint `HN + 2D CON + Proline-Bridge + N–N + CAS` IDP Backbone Resonance Assignment
//! Solver (`Phase 23`, `T23.2`).
//!
//! Provenance: Clean-room original implementation derived strictly from peer-reviewed biomolecular
//! NMR literature and public BMRB/IUPAC standards (zero proprietary, academic-licensed, or
//! GPL/AGPL source code or residue libraries consulted):
//! - Bermel, W., Bertini, I., Felli, I. C., Lee, Y.-M., Luchinat, C. & Pierattelli, R. (*J. Am.
//!   Chem. Soc.* 2006, 128, 3918–3919 — protonless `2D CON` direct `13C'` detection in IDPs).
//! - Bermel, W., Bertini, I., Felli, I. C. & Pierattelli, R. (*J. Am. Chem. Soc.* 2009, 131,
//!   15339–15345 — `CANCO`, `(H)N(CA)CON`, `H(CC)CON`, `CBCACON`, and continuous through-Proline
//!   IDP backbone assignment).
//! - Sahu, D., Bastidas, M. & Showalter, S. A. (*Anal. Biochem.* 2014, 449, 17–25; *J. Biomol.
//!   NMR* 2014, 59, 63–74 — `CAS-NMR` amino-acid-selective `(C'_{i-1}, N_i)` `CACON` and `CANCO`
//!   subspectra for disordered proteins).
//! - Pervushin, K., Riek, R., Wider, G. & Wüthrich, K. (*Proc. Natl. Acad. Sci. USA* 1997, 94,
//!   12366–12371 — TROSY `46 Hz (1/2 1J_NH)`, field-scaled to ppm relative to decoupled spectra).
//! - Tamiola, K., Acar, B. & Mulder, F. A. A. (*J. Am. Chem. Soc.* 2010, 132, 18000–18003 — `ncIDP`
//!   neighbor-corrected random-coil chemical shifts and pre-Proline shift offsets).
//! - Platzer, G., Okon, M. & McIntosh, L. P. (*J. Biomol. NMR* 2014, 60, 109–129 — pH-dependent
//!   random-coil and C-terminal `COO-` (`~179 ppm` DSS) chemical shifts).
//! - Kuhn, H. W. (*Nav. Res. Logist. Q.* 1955, 2, 83–97 — exact polynomial-time Hungarian
//!   maximum-cardinality minimum-cost bipartite matching for `HN ↔ CON` root fusion).

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::f64::consts::TAU;
#[cfg(not(target_arch = "wasm32"))]
use std::thread;

use serde::{Deserialize, Serialize};

use crate::c13_detected::{
    cas_llr_adjustment, n15_obs_mhz, C13DetectedExp, C13PickedDataset, CasMode, CasSpectrum,
    Con2DPeak, CAS_LOG_LIKELIHOOD_NATS, CAS_MATCH_TOL_C_PRIME_PPM, CAS_MATCH_TOL_N15_PPM,
};
use crate::idp_solver::{
    check_idp_matrix_bounds, evaluate_idp_auto_reference_safety, fit_1d_unlabelled_marginal_offset,
    predict_ncidp_shifts, register_and_build_con_dipeptide_systems,
    select_hncacb_intra_and_backfill_prev, sequence_complexity_metrics, ConDipeptideSystem,
    IdpAutoRefSafetyReport, IdpShiftOffsets, IdpSolverConfig, NcIdpResiduePrediction,
    IDP_BG_CA_MUS, IDP_BG_CB_MUS, IDP_BG_C_MUS, IDP_BG_N_MUS, IDP_BG_SC_MUS, MAX_IDP_CANDS,
    MAX_IDP_CB_CANDS, MAX_IDP_SC_CANDS,
};
use crate::nmrstar::Reference;
use crate::peaks::Peak;
use crate::priors::one_to_three;
use crate::qc::DatasetQcVerdict;
use crate::rng::Rng;
use crate::solver::{AssignResult, Matrix2D, FORBID};
use crate::spinsystems::{normalize_hncacb_and_cbcaconh_polarity, HncacbPolarity, PolarityReport};

// ============================================================================
// Canonical Physical Constants (Shared by Core, CLI & GUI)
// ============================================================================

/// Canonical half scalar coupling `46 Hz (1/2 1J_NH)`, field-scaled to ppm
/// (Pervushin et al., *PNAS* 1997; `1J_NH ≈ -92 Hz`), shifting TROSY resonances relative to
/// decoupled `1H–15N` amide peaks by `(-46.0 / nu_H, +46.0 / nu_N)` ppm.
pub const TROSY_HALF_J_HN_HZ: f64 = 46.0;

/// Canonical fallback `1H` Larmor frequency (`800.0 MHz`) used by [`trosy_expected_shift_ppm`]
/// when spectrometer frequency is unspecified (`<= 50.0 MHz` or non-finite).
pub const CANONICAL_TROSY_FALLBACK_MHZ: f64 = 800.0;

/// Minimum matched distinct amide roots required for TROSY offset detection (`>= 5`).
pub const MIN_TROSY_MATCHED_ROOTS: usize = 5;

/// Canonical lower `15N` DSS boundary (`131.5 ppm`) for Proline tertiary imino roots in `2D CON`.
pub const PROLINE_N15_MIN_PPM: f64 = 131.5;

/// Canonical upper `15N` DSS boundary (`144.5 ppm`) for Proline tertiary imino roots in `2D CON`.
pub const PROLINE_N15_MAX_PPM: f64 = 144.5;

/// Backwards-compatible alias for [`PROLINE_N15_MIN_PPM`].
pub const PROLINE_N_DSS_MIN_PPM: f64 = PROLINE_N15_MIN_PPM;

/// Backwards-compatible alias for [`PROLINE_N15_MAX_PPM`].
pub const PROLINE_N_DSS_MAX_PPM: f64 = PROLINE_N15_MAX_PPM;

/// Lower `13CA` DSS boundary (`41.0 ppm`) of the Glycine `CA` band for `Gly CB` bleed-through guard.
pub const GLY_CA_DSS_MIN_PPM: f64 = 41.0;

/// Upper `13CA` DSS boundary (`46.5 ppm`) of the Glycine `CA` band for `Gly CB` bleed-through guard.
pub const GLY_CA_DSS_MAX_PPM: f64 = 46.5;

/// Nominal deprotonated C-terminal `COO-` carbonyl `13C'` chemical shift (`~179.0 ppm` DSS;
/// Platzer, Okon & McIntosh, *J. Biomol. NMR* 2014).
pub const CTERM_COO_DSS_CENTER_PPM: f64 = 179.0;

/// Lower DSS `13C'` threshold (`177.6 ppm`) for flagging a candidate C-terminal `COO-` carbonyl.
pub const CTERM_COO_DSS_MIN_PPM: f64 = 177.6;

/// Upper DSS `13C'` threshold (`181.8 ppm`) for flagging a candidate C-terminal `COO-` carbonyl.
pub const CTERM_COO_DSS_MAX_PPM: f64 = 181.8;

/// Canonical pre-Proline carbonyl `13C'` shift correction (`-1.90 ppm`; Wishart 1995 / Tamiola 2010).
pub const PRE_PRO_C_PRIME_CORRECTION_PPM: f64 = -1.90;

/// Canonical pre-Proline `13CA` shift correction (`-1.80 ppm`; Wishart 1995 / Tamiola 2010).
pub const PRE_PRO_CA_CORRECTION_PPM: f64 = -1.80;

/// Canonical pre-Proline `15N` shift correction (`+1.65 ppm`; Schwarzinger 2001 / Tamiola 2010).
pub const PRE_PRO_N_CORRECTION_PPM: f64 = 1.65;

/// Canonical pre-Proline `13CB` shift correction (`-0.35 ppm`; Tamiola 2010).
pub const PRE_PRO_CB_CORRECTION_PPM: f64 = -0.35;

/// Maximum per-channel candidate shifts packed per `JointIdpNode`.
pub const MAX_JOINT_CANDS: usize = MAX_IDP_CANDS;

/// Maximum `13CB` candidates packed per `JointIdpNode`.
pub const MAX_JOINT_CB_CANDS: usize = MAX_IDP_CB_CANDS;

/// Maximum aliphatic sidechain carbons packed per `JointIdpNode`.
pub const MAX_JOINT_SC_CANDS: usize = MAX_IDP_SC_CANDS;

/// Half of `chi^2_{1, 0.95}` (`1.9207294`) for Wilks look-elsewhere hypothesis gating.
const WILKS_HALF_CHI2_1_95: f64 = 1.920_729_410_347_062;

/// Dedicated monotone PAV calibration knots `(raw_posterior, calibrated_confidence)` empirically
/// fitted via `fit_idp_calibration_knots_pav` (`5,737` assignments across `8` stress regimes) on
/// two disjoint synthetic IDP sequences (`cal_seq_a` 115 aa, `cal_seq_b` 130 aa; seeds `1001..1003`)
/// strictly separate from all held-out evaluation sequences and seeds (`180.core:P1-1`).
pub const JOINT_IDP_CALIBRATION_KNOTS: &[(f64, f64)] = &[
    (0.00, 0.000),
    (0.10, 0.184),
    (0.25, 0.296),
    (0.40, 0.442),
    (0.55, 0.686),
    (0.70, 0.847),
    (0.82, 0.889),
    (0.90, 0.906),
    (0.96, 0.912),
    (1.00, 0.914),
];

/// Compute the canonical TROSY `46 Hz (1/2 1J_NH)`, field-scaled to ppm shift `(dH_ppm, dN_ppm)`
/// relative to a decoupled `1H–15N` amide spectrum (`-46 / nu_H` in `1H`, `+46 / nu_N` in `15N`).
///
/// Falls back canonically to `800.0 MHz` whenever `obs_mhz_1h` is non-finite or `<= 50.0 MHz`.
#[inline]
pub fn trosy_expected_shift_ppm(obs_mhz_1h: f64) -> (f64, f64) {
    let mhz = if obs_mhz_1h.is_finite() && obs_mhz_1h > 50.0 {
        obs_mhz_1h
    } else {
        CANONICAL_TROSY_FALLBACK_MHZ
    };
    let dh = -TROSY_HALF_J_HN_HZ / mhz;
    let dn = TROSY_HALF_J_HN_HZ / n15_obs_mhz(mhz);
    (dh, dn)
}

/// Canonical pre-Proline random-coil chemical-shift offsets `(dN, dC', dCA, dCB)` in ppm
/// exerted on residue `k` when `seq[k + 1] == "PRO"` (`P2-14`).
#[inline]
pub fn pre_pro_shift_corrections(curr_aa: &str) -> (f64, f64, f64, f64) {
    match curr_aa {
        "GLY" => (1.45, -1.50, -1.05, 0.00),
        "PRO" => (0.90, -1.80, -1.80, 0.12),
        _ => (
            PRE_PRO_N_CORRECTION_PPM,
            PRE_PRO_C_PRIME_CORRECTION_PPM,
            PRE_PRO_CA_CORRECTION_PPM,
            PRE_PRO_CB_CORRECTION_PPM,
        ),
    }
}

// ============================================================================
// Core Public Data Structures (Canonical Core/CLI/GUI Contract)
// ============================================================================

/// Classification of a unified peptide-bond / residue node keyed on `(C'_{i-1}, N_i)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum JointNodeKind {
    /// Amide root `(H_i, N_i, C'_{i-1})` supported by `HNCO`/`HSQC`, fused with a matching
    /// `2D CON(C'_{i-1}, N_i)` root when present (or `HN`-only when `con_matched == false`).
    AmideFused,
    /// Proline-only `2D CON(C'_{i-1}, N_i)` bridge (`N_i ≈ 131.5–144.5 ppm` DSS, no `1H^N`), directly
    /// linking preceding `HN` `C'_{i-1}` across `Pro_i` into following `HN` `C'_i` and `(CA/CB)_i`.
    ProlineConBridge,
    /// Exchange-broadened non-Proline `2D CON(C'_{i-1}, N_i)` root missing from `1H-15N` spectra.
    ConOnlyAmide,
}

/// Experimental route corroboration tag emitted for every assigned residue.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum JointRouteSupport {
    /// Assigned residue supported by BOTH `1H`-detected (`HN`) and `13C`-detected (`CON`) routes.
    CrossValidatedHnAndCon,
    /// Assigned residue supported by `1H`-detected (`HN`) triple-resonance experiments only.
    HnRoute,
    /// Proline residue bridged directly via the `2D CON` tertiary imino (`131.5–144.5 ppm`) pathway.
    ConProBridge,
    /// Non-Proline residue recovered via `2D CON` / `13C`-detected channels only (e.g., fast amide exchange).
    ConRouteOnly,
}

/// Summary of one `X–Pro(–Y)` Proline bridge for CLI & GUI inspection.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProlineBridgeSummary {
    pub pro_seq_idx: usize,
    pub prev_residue: String,
    pub next_residue: Option<String>,
    pub pro_node_idx: Option<usize>,
    pub c_prev_ppm: Option<f64>,
    pub n_pro_ppm: Option<f64>,
    pub confidence: f64,
    pub bridged: bool,
}

/// Summary of a C-terminal `COO-` (`~179 ppm` DSS) carbonyl anchor candidate for CLI & GUI inspection.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CTermAnchorSummary {
    pub seq_idx: usize,
    pub residue: String,
    pub node_idx: usize,
    pub c_curr_ppm: f64,
    pub c_curr_dss_ppm: f64,
    pub confidence: f64,
    pub assigned_to_cterm: bool,
}

/// Directional `(H)N(CA)CON` sequential `N(i-1)` cross-peak anchored on a `CON(i)` root `(C'_{i-1}, N_i)`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DirectionalNnPeak {
    /// `13C'(i-1)` coordinate in ppm.
    pub c_prev_ppm: f64,
    /// Detected `15N(i)` coordinate in ppm.
    pub n_curr_ppm: f64,
    /// Preceding-residue `15N(i-1)` partner coordinate in ppm (including `Pro` `131.5–144.5 ppm`!).
    pub n_prev_ppm: f64,
    /// Signal-to-noise ratio (or relative peak intensity weight).
    pub snr: f64,
}

/// Raw `BT_HNCACONH` / `HN(CA)CONH` `3D` `(1H_i, 15N_i, 15N_{i±1})` cross-peak before TROSY alignment.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HncaconhPeak {
    /// Detected amide `1H(i)` shift in ppm (may carry `46 Hz (1/2 1J_NH)` TROSY offset).
    pub h_ppm: f64,
    /// Detected amide `15N(i)` shift in ppm (may carry `46 Hz (1/2 1J_NH)` TROSY offset).
    pub n_curr_ppm: f64,
    /// Sequential neighbor `15N(i±1)` partner shift in ppm.
    pub n_partner_ppm: f64,
    /// Peak height / SNR.
    pub height: f64,
}

/// Result of field-scaled TROSY `46 Hz (1/2 1J_NH)` cross-spectrum `(dH, dN)` calibration.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TrosyOffsetCalibration {
    /// Fitted `(H_trosy - H_decoupled)` shift in ppm (`0.0` if decoupled / already aligned).
    pub dh_ppm: f64,
    /// Fitted `(N_trosy - N_decoupled)` shift in ppm (`0.0` if decoupled / already aligned).
    pub dn_ppm: f64,
    /// Expected physical magnitude `46.0 Hz / nu_H` in ppm.
    pub expected_half_j_h_ppm: f64,
    /// Expected physical magnitude `46.0 Hz / nu_N` in ppm.
    pub expected_half_j_n_ppm: f64,
    /// True if a statistically significant `46 Hz (1/2 1J_NH)` TROSY component shift was detected.
    pub trosy_detected: bool,
    /// Number of distinct amide roots matched after alignment.
    pub matched_roots: usize,
    /// Background-subtracted coincidence log-likelihood gain over `(0.0, 0.0)`.
    pub score_gain: f64,
}

/// Unified peptide-bond / residue node keyed on `(C'_{i-1}, N_i)` combining `HN` and `13C` channels.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct JointIdpNode {
    /// 0-based node identifier (`0..S`).
    pub id: usize,
    /// Node category (`AmideFused`, `ProlineConBridge`, or `ConOnlyAmide`).
    pub kind: JointNodeKind,
    /// True if corroborated by `1H`-detected `HNCO`/`HSQC` amide peaks.
    pub hn_matched: bool,
    /// True if corroborated by a `2D CON(C'_{i-1}, N_i)` peak.
    pub con_matched: bool,
    /// Amide proton `1H(i)` shift in ppm (`None` for `ProlineConBridge` and `ConOnlyAmide`).
    pub h_curr: Option<f64>,
    /// Unified `15N(i)` shift in ppm (in `CON`/reference frame when fused).
    pub n_curr: f64,
    /// Original `15N(i)` shift in the `HN` spectrum frame (retained for `HN` strip matching).
    pub hn_n_curr: Option<f64>,
    /// Unified `13C'(i-1)` shift in ppm (`Some` for all `CON` and `HNCO`-fused roots).
    pub c_prev: Option<f64>,
    /// Relative root intensity.
    pub root_height: f64,
    /// Pooled intraresidue `13CA(i)` candidates (`HNCACB`/`HNCA`/`CANCO`/`NCACO`).
    pub ca_curr: Vec<f64>,
    /// Pooled intraresidue `13CB(i)` candidates (`HNCACB`).
    pub cb_curr: Vec<f64>,
    /// Pooled preceding `13CA(i-1)` candidates (`CBCACONH`/`HNCOCA`/`HNCACB`/`CBCACON`/`H(CC)CON`/`CANCO`).
    pub ca_prev: Vec<f64>,
    /// Pooled preceding `13CB(i-1)` candidates (`CBCACONH`/`HNCACB`/`CBCACON`/`H(CC)CON`).
    pub cb_prev: Vec<f64>,
    /// Pooled intraresidue `13C'(i)` candidates (`HN(CA)CO`/`NCACO`/`CACO`), linking into `t.c_prev`.
    pub c_curr: Vec<f64>,
    /// Preceding `(i-1)` aliphatic sidechain carbons (`H(CC)CON` / `CCC(CO)N`: `CG*, CD*, CE*`).
    pub sidechain_c_prev: Vec<f64>,
    /// Directional `(H)N(CA)CON` sequential `N(i-1)` partner candidates `(n_prev_ppm, snr)`.
    pub nn_prev_cands: Vec<(f64, f64)>,
    /// Reciprocal `BT_HNCACONH` neighbor node indices `(partner_node_idx, weight)`.
    /// Note: reciprocal `BT_HNCACONH` edges are **symmetric** (`s <-> t`), disambiguated directionally
    /// by `ΔC'`, `ΔCA`, `ΔCB`, and directional `(H)N(CA)CON`.
    pub recip_nn_partners: Vec<(usize, f64)>,
    /// Opposite-phase constant-time `2D CACO` Glycine indicator for `i - 1`.
    pub prev_gly_opposite_phase: bool,
    /// True when preceding-residue `CB`-sensitive experiments (`CBCACONH`/`CBCACON`/`H(CC)CON`) were active.
    pub has_prev_cb_exp: bool,
    /// True when intraresidue `CB`-sensitive experiments (`HNCACB`) were active on this node.
    pub has_curr_cb_exp: bool,
    /// True if `Gly CB` bleed-through guard suppressed a suspect intraresidue `CB(i)` peak.
    pub gly_cb_curr_suppressed: bool,
    /// True if `Gly CB` bleed-through guard suppressed a suspect preceding `CB(i-1)` peak.
    pub gly_cb_prev_suppressed: bool,
    /// Count of `F1(C')` NUS sidelobes pruned from `HN(CA)CO` / `HNCO` on this node.
    pub f1_sidelobes_pruned: usize,
    /// Diagnostics / registration warnings attached to this node.
    pub registration_warnings: Vec<String>,
}

impl JointIdpNode {
    /// True if at least one 3D / multi-channel carbon or `N–N` evidence list is populated.
    #[inline]
    pub fn has_3d_support(&self) -> bool {
        !self.ca_curr.is_empty()
            || !self.cb_curr.is_empty()
            || !self.ca_prev.is_empty()
            || !self.cb_prev.is_empty()
            || !self.c_curr.is_empty()
            || !self.sidechain_c_prev.is_empty()
            || !self.nn_prev_cands.is_empty()
            || !self.recip_nn_partners.is_empty()
    }

    /// Experimental route support tag for this node.
    #[inline]
    pub fn route_support(&self) -> JointRouteSupport {
        match self.kind {
            JointNodeKind::ProlineConBridge => JointRouteSupport::ConProBridge,
            JointNodeKind::AmideFused => {
                if self.hn_matched && self.con_matched {
                    JointRouteSupport::CrossValidatedHnAndCon
                } else if self.hn_matched {
                    JointRouteSupport::HnRoute
                } else {
                    JointRouteSupport::ConRouteOnly
                }
            }
            JointNodeKind::ConOnlyAmide => JointRouteSupport::ConRouteOnly,
        }
    }
}

/// Complete multi-experiment input bundle for the unified joint `HN + 13C` IDP solver.
#[derive(Clone, Debug, Default)]
pub struct JointIdpDataset {
    /// `1H`-detected amide triple-resonance peaks (`HSQC`, `HNCO`, `HNCACO`, `HNCACB`, `CBCACONH`/`HNCOCACB`, `HNCA`, `HNCOCA`).
    pub hn_peaks: Vec<Peak>,
    /// `13C`-detected peak bundle (`2D CON`, `2D CACO`, `3D CANCO`, `CBCACON`, `H(CC)CON`/`CCCON`, `NCACO`).
    pub c13_dataset: C13PickedDataset,
    /// Amino-acid-selective `(C'_{i-1}, N_i)` `2D CAS` subspectra (`cacon` on `i-1`, `canco` on `i`).
    pub cas_spectra: Vec<CasSpectrum>,
    /// Directional `(H)N(CA)CON` sequential `N(i-1)` partner peaks.
    pub hncacon_nn_edges: Vec<DirectionalNnPeak>,
    /// Raw `BT_HNCACONH` / `HN(CA)CONH` peaks (auto-calibrated for TROSY `46 Hz (1/2 1J_NH)`).
    pub hncaconh_peaks: Vec<HncaconhPeak>,
    /// Optional pre-extracted reciprocal `(N_a, N_b)` amide adjacency pairs in ppm.
    pub reciprocal_nn_pairs: Vec<(f64, f64)>,
    /// Spectrometer `1H` Larmor frequency in `MHz` (fallback `800.0 MHz`; used for field-scaled TROSY `46 Hz`).
    pub proton_mhz: f64,
    /// Optional user-locked `(node_idx, seq_idx)` hard constraints.
    pub user_locks: Vec<(usize, usize)>,
}

/// Configuration for the unified joint `HN + 13C` IDP backbone solver.
#[derive(Clone, Debug)]
pub struct JointIdpConfig {
    /// Sample temperature in Kelvin (`default = 298.15`).
    pub temperature_k: f64,
    /// Sample pH (`default = 6.5`).
    pub ph: f64,
    /// Fallback spectrometer `1H` frequency in `MHz` when `dataset.proton_mhz <= 50.0` (`default = 800.0`).
    pub proton_mhz: f64,
    /// `HNCACB`/`CBCACONH` polarity mode (`default = HncacbPolarity::Auto`).
    pub hncacb_polarity: HncacbPolarity,
    /// Automatically estimate unlabelled `(dN, dC', dCA, dCB)` reference offsets (`default = true`).
    pub auto_reference_offset: bool,
    /// Explicit offsets used when `auto_reference_offset == false`.
    pub manual_offsets: IdpShiftOffsets,
    /// Enable field-scaled TROSY `46 Hz (1/2 1J_NH)` `(dH, dN)` auto-calibration on `BT_HNCACONH` (`default = true`).
    pub auto_trosy_calibration: bool,
    /// Enable `F1(C')` NUS sidelobe pruning on `HN(CA)CO` / `HNCO` strips (`default = true`).
    pub enable_nus_sidelobe_pruning: bool,
    /// Enable `Gly CB` bleed-through guard for `CA ∈ [41.0, 46.5] ppm` DSS (`default = true`).
    pub enable_gly_cb_guard: bool,
    /// Enable Platzer/ncIDP C-terminal `COO-` (`~179 ppm` DSS) carbonyl prior (`default = true`).
    pub enable_cterm_coo_prior: bool,
    /// Enable conservative 2-pass anchor-lock propagation (`default = true`).
    pub enable_two_pass_anchor_lock: bool,
    /// Allow `HSQC` roots without an `HNCO` peak when corroborated by `HNCACB`/`HNCA` (`default = true`).
    pub include_hsqc_without_hnco: bool,
    /// Core `ncIDP` random-coil standard deviations (ppm).
    pub sigma_ca_rc: f64,
    pub sigma_cb_rc: f64,
    pub sigma_c_rc: f64,
    pub sigma_n_rc: f64,
    pub sigma_pro_n_rc: f64,
    /// Sequential matching tolerances (ppm) in `P(s -> t)`.
    pub sigma_seq_ca: f64,
    pub sigma_seq_cb: f64,
    pub sigma_seq_c: f64,
    /// Bidirectional candidate-multiplicity ambiguity penalty weight (`default = 0.35`).
    pub w_mult: f64,
    /// Per-atom unary LLR floor (`default = -5.5`).
    pub llr_floor: f64,
    /// Per-channel sequential pair LLR floor (`default = -3.8`).
    pub pair_floor: f64,
    /// Elliptical `CAS` `(C', N)` match tolerances (ppm).
    pub cas_c_tol_ppm: f64,
    pub cas_n_tol_ppm: f64,
    /// Maximum per-spectrum `CAS` LLR scale (nats, routed through `c13_detected::cas_llr_adjustment`, `default = 3.0`).
    pub cas_max_llr: f64,
    /// Baseline breakthrough / false-positive rate floor (`default = 0.10`).
    pub cas_base_fp: f64,
    /// Number of SA trajectories per pass (`default = 32`).
    pub replicas: usize,
    /// Number of evolutionary islands (`default = 4`).
    pub islands: usize,
    /// Number of evolutionary crossover generations per pass (`default = 3`).
    pub generations: usize,
    /// MCMC steps per node per generation (`default = 1_800`).
    pub steps_per_system: usize,
    /// Initial SA temperature (`default = 16.0`).
    pub t_start: f64,
    /// Final SA temperature (`default = 0.08`).
    pub t_end: f64,
    /// Minimum Pass-1 ensemble consensus required to auto-lock an anchor (`default = 0.90`).
    pub anchor_min_consensus: f64,
    /// Minimum Pass-1 local Boltzmann probability required to auto-lock an anchor (`default = 0.88`).
    pub anchor_min_local_prob: f64,
    /// Minimum sequential edge score `P(s, t)` (nats) for auto-locking a run/bridge (`default = 1.20`).
    pub anchor_min_pair_score: f64,
    /// High-confidence threshold (`default = 0.80`).
    pub hc_threshold: f64,
    /// Deterministic RNG seed (`default = 42`).
    pub seed: u64,
}

impl Default for JointIdpConfig {
    fn default() -> Self {
        Self {
            temperature_k: 298.15,
            ph: 6.5,
            proton_mhz: CANONICAL_TROSY_FALLBACK_MHZ,
            hncacb_polarity: HncacbPolarity::Auto,
            auto_reference_offset: true,
            manual_offsets: IdpShiftOffsets::default(),
            auto_trosy_calibration: true,
            enable_nus_sidelobe_pruning: true,
            enable_gly_cb_guard: true,
            enable_cterm_coo_prior: true,
            enable_two_pass_anchor_lock: true,
            include_hsqc_without_hnco: true,
            sigma_ca_rc: 0.45,
            sigma_cb_rc: 0.55,
            sigma_c_rc: 0.40,
            sigma_n_rc: 0.85,
            sigma_pro_n_rc: 1.15,
            sigma_seq_ca: 0.085,
            sigma_seq_cb: 0.115,
            sigma_seq_c: 0.055,
            w_mult: 0.35,
            llr_floor: -5.5,
            pair_floor: -3.8,
            cas_c_tol_ppm: CAS_MATCH_TOL_C_PRIME_PPM,
            cas_n_tol_ppm: CAS_MATCH_TOL_N15_PPM,
            cas_max_llr: CAS_LOG_LIKELIHOOD_NATS,
            cas_base_fp: 0.10,
            replicas: 32,
            islands: 4,
            generations: 3,
            steps_per_system: 1_800,
            t_start: 16.0,
            t_end: 0.08,
            anchor_min_consensus: 0.90,
            anchor_min_local_prob: 0.88,
            anchor_min_pair_score: 1.20,
            hc_threshold: 0.80,
            seed: 42,
        }
    }
}

impl JointIdpConfig {
    fn as_idp_solver_config(&self) -> IdpSolverConfig {
        IdpSolverConfig {
            temperature_k: self.temperature_k,
            ph: self.ph,
            auto_reference_offset: self.auto_reference_offset,
            manual_offsets: self.manual_offsets,
            sigma_ca_rc: self.sigma_ca_rc,
            sigma_cb_rc: self.sigma_cb_rc,
            sigma_c_rc: self.sigma_c_rc,
            sigma_n_rc: self.sigma_n_rc,
            sigma_pro_n_rc: self.sigma_pro_n_rc,
            sigma_seq_ca: self.sigma_seq_ca,
            sigma_seq_c: self.sigma_seq_c,
            w_mult: self.w_mult,
            llr_floor: self.llr_floor,
            pair_floor: self.pair_floor,
            replicas: self.replicas,
            islands: self.islands,
            generations: self.generations,
            steps_per_system: self.steps_per_system,
            t_start: self.t_start,
            t_end: self.t_end,
            hc_threshold: self.hc_threshold,
            seed: self.seed,
            ..IdpSolverConfig::default()
        }
    }
}

/// Output of the unified node-fusion and hygiene preprocessing stage.
#[derive(Clone, Debug, PartialEq)]
pub struct JointNodeBuildReport {
    pub nodes: Vec<JointIdpNode>,
    pub fitted_offsets: IdpShiftOffsets,
    pub auto_ref_report: IdpAutoRefSafetyReport,
    pub trosy_calibration: TrosyOffsetCalibration,
    pub polarity_report: Option<PolarityReport>,
    pub hnco_to_con_drift_ppm: (f64, f64),
    pub fusion_rms_c_ppm: f64,
    pub fusion_rms_n_ppm: f64,
    pub hn_unmatched_frac: f64,
    pub n_fused_hn_con: usize,
    pub n_hn_only: usize,
    pub n_proline_bridges: usize,
    pub n_con_only_amide: usize,
    pub n_sidelobes_pruned: usize,
    pub n_gly_cb_guarded: usize,
    pub n_reciprocal_nn_edges: usize,
    pub n_directional_nn_edges: usize,
    pub c_term_coo_candidates: Vec<usize>,
    pub warnings: Vec<String>,
}

/// Per-residue assignment summary emitted by the unified joint IDP solver.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct JointResidueAssignment {
    pub seq_idx: usize,
    pub residue: String,
    pub node_idx: Option<usize>,
    pub confidence: f64,
    pub raw_posterior: f64,
    pub route_support: Option<JointRouteSupport>,
    pub anchor_locked: bool,
    pub is_proline: bool,
    pub is_cterm_coo: bool,
}

/// Complete output of the unified joint `HN + 2D CON + Proline-Bridge + N–N + CAS` IDP solver.
#[derive(Clone, Debug, PartialEq)]
pub struct JointIdpAssignResult {
    /// Unified peptide-bond / residue nodes (`0..S`).
    pub nodes: Vec<JointIdpNode>,
    /// Assigned sequence index `k in 1..L` per node (`-1` if unassigned).
    pub position: Vec<i32>,
    /// Calibrated posterior confidence in `[0.0, 1.0]` per node.
    pub confidence: Vec<f64>,
    /// Pre-calibration combined ensemble + local Boltzmann posterior in `[0.0, 1.0]` per node.
    pub raw_posterior: Vec<f64>,
    /// Explicit route-corroboration tag per node (`None` if unassigned).
    pub route_support: Vec<Option<JointRouteSupport>>,
    /// Per-residue summary table across `sequence[0..L]`.
    pub residue_assignments: Vec<JointResidueAssignment>,
    /// Final total objective score (`U + P`).
    pub score: f64,
    /// Pass-1 total objective score before anchor-lock propagation.
    pub pass1_score: f64,
    /// Objective improvement achieved by Pass-2 anchor-lock propagation (`>= 0.0`).
    pub pass2_score_gain: f64,
    /// Per-replica objective scores from the final pass.
    pub run_scores: Vec<f64>,
    /// Anchors `(node_idx, seq_idx)` locked between Pass 1 and Pass 2.
    pub pass1_locked_anchors: Vec<(usize, usize)>,
    /// Unlabelled global `(dN, dC', dCA, dCB)` reference offsets applied.
    pub fitted_offsets: IdpShiftOffsets,
    /// Plausibility & TMS-referencing audit report.
    pub auto_ref_report: IdpAutoRefSafetyReport,
    /// Field-scaled TROSY `46 Hz (1/2 1J_NH)` calibration summary for `BT_HNCACONH`.
    pub trosy_calibration: TrosyOffsetCalibration,
    /// Automatic `HNCACB`/`CBCACONH` polarity report (if `HN` peaks supplied).
    pub polarity_report: Option<PolarityReport>,
    /// Node indices exhibiting a C-terminal `COO-` `~179 ppm` DSS carbonyl signature.
    pub c_term_coo_candidates: Vec<usize>,
    /// Count of assigned nodes corroborated by BOTH `HN` and `CON`.
    pub n_cross_validated_hn_con: usize,
    /// Count of assigned nodes supported by `HN` route only.
    pub n_hn_route_only: usize,
    /// Count of assigned Prolines bridged via `2D CON`.
    pub n_con_pro_bridges: usize,
    /// Count of assigned exchange-broadened non-Proline nodes recovered via `CON` only.
    pub n_con_route_only: usize,
    /// `(assigned_prolines, total_assignable_prolines_k_ge_1)`.
    pub proline_bridge_recovery: (usize, usize),
    /// Total `F1(C')` NUS sidelobes pruned during preprocessing.
    pub n_sidelobes_pruned: usize,
    /// Total suspect `Gly CB` bleed-through candidates suppressed.
    pub n_gly_cb_guarded: usize,
    /// Empirical sequential `CA` RMS (ppm) across assigned neighbors.
    pub emp_rms_seq_ca: f64,
    /// Empirical sequential `C'` RMS (ppm) across assigned neighbors.
    pub emp_rms_seq_c: f64,
    /// Empirical `HN ↔ CON` fusion RMS `(C', N)` in ppm.
    pub fusion_rms_ppm: (f64, f64),
    /// Unlabelled dataset QC verdict.
    pub qc_verdict: DatasetQcVerdict,
    /// Human-readable QC diagnostics and warnings.
    pub qc_reasons: Vec<String>,
}

/// Canonical type alias exported for GUI & CLI consumers (`180.core:P1-4`).
pub type JointIdpResult = JointIdpAssignResult;

impl JointIdpAssignResult {
    /// Convert into `AssignResult` for interoperability with `evaluate` and `ProteinReliabilityCurve`.
    pub fn as_assign_result(&self) -> AssignResult {
        AssignResult {
            position: self.position.clone(),
            confidence: self.confidence.clone(),
            score: self.score,
            run_scores: self.run_scores.clone(),
        }
    }

    /// Build structured Proline-bridge summaries across every assignable `PRO` residue (`k >= 1`).
    pub fn proline_bridge_summaries(&self) -> Vec<ProlineBridgeSummary> {
        let len = self.residue_assignments.len();
        let mut out = Vec::new();
        for k in 1..len {
            let ra = &self.residue_assignments[k];
            if !ra.is_proline {
                continue;
            }
            let prev_res = self.residue_assignments[k - 1].residue.clone();
            let next_res = (k + 1 < len).then(|| self.residue_assignments[k + 1].residue.clone());
            let (c_prev, n_pro) = ra
                .node_idx
                .and_then(|nid| self.nodes.get(nid))
                .map(|n| (n.c_prev, Some(n.n_curr)))
                .unwrap_or((None, None));
            out.push(ProlineBridgeSummary {
                pro_seq_idx: k,
                prev_residue: prev_res,
                next_residue: next_res,
                pro_node_idx: ra.node_idx,
                c_prev_ppm: c_prev,
                n_pro_ppm: n_pro,
                confidence: ra.confidence,
                bridged: ra.node_idx.is_some(),
            });
        }
        out
    }

    /// Build structured C-terminal `COO-` (`~179 ppm` DSS) anchor summaries for GUI & CLI inspection.
    pub fn cterm_anchor_summaries(&self) -> Vec<CTermAnchorSummary> {
        let last_k = self.residue_assignments.len().saturating_sub(1);
        let mut out = Vec::new();
        for &nid in &self.c_term_coo_candidates {
            let Some(node) = self.nodes.get(nid) else {
                continue;
            };
            let c_obs = node
                .c_curr
                .iter()
                .copied()
                .find(|&c| {
                    let cd = c - self.fitted_offsets.d_c;
                    (CTERM_COO_DSS_MIN_PPM..=CTERM_COO_DSS_MAX_PPM).contains(&cd)
                })
                .or_else(|| node.c_curr.first().copied())
                .unwrap_or(CTERM_COO_DSS_CENTER_PPM);
            let c_dss = c_obs - self.fitted_offsets.d_c;
            let pos_i = self.position.get(nid).copied().unwrap_or(-1);
            let assigned_to_cterm = pos_i >= 0 && (pos_i as usize) == last_k;
            let seq_idx = if pos_i >= 0 && (pos_i as usize) < self.residue_assignments.len() {
                pos_i as usize
            } else {
                last_k
            };
            let residue = self
                .residue_assignments
                .get(seq_idx)
                .map(|r| r.residue.clone())
                .unwrap_or_else(|| "UNK".to_string());
            let conf = self.confidence.get(nid).copied().unwrap_or(0.0);
            out.push(CTermAnchorSummary {
                seq_idx,
                residue,
                node_idx: nid,
                c_curr_ppm: c_obs,
                c_curr_dss_ppm: c_dss,
                confidence: conf,
                assigned_to_cterm,
            });
        }
        out
    }
}

// ============================================================================
// Sequence Normalization & Enhanced `ncIDP` + Pre-Pro + Platzer C-Term COO-
// ============================================================================

fn normalize_seq_3(sequence: &[String]) -> Vec<String> {
    sequence
        .iter()
        .map(|s| {
            let t = s.trim();
            if t.len() == 1 {
                one_to_three(t.chars().next().unwrap_or('X')).to_string()
            } else {
                t.to_ascii_uppercase()
            }
        })
        .collect()
}

/// Longest contiguous homopolymer run length in `sequence` (for repeat QC gating).
fn max_homopolymer_run(seq: &[String]) -> usize {
    let mut best = 0usize;
    let mut cur = 0usize;
    let mut prev: Option<&str> = None;
    for r in seq {
        if prev == Some(r.as_str()) {
            cur += 1;
        } else {
            cur = 1;
            prev = Some(r.as_str());
        }
        if cur > best {
            best = cur;
        }
    }
    best
}

/// Predict temperature-, pH-, neighbor-, all-4-nucleus pre-Proline- (`C' -1.90`, `CA -1.80`,
/// `N +1.65`, `CB -0.35 ppm`), and Platzer C-terminal `COO-`-corrected `ncIDP` backbone shifts.
pub fn predict_joint_ncidp_shifts(
    sequence: &[String],
    cfg: &JointIdpConfig,
) -> Vec<NcIdpResiduePrediction> {
    let seq = normalize_seq_3(sequence);
    let idp_cfg = cfg.as_idp_solver_config();
    let mut preds = predict_ncidp_shifts(&seq, &idp_cfg);

    // Harmonize all 4 canonical pre-Proline shifts (`P2-14`: `C' -1.90`, `CA -1.80`, `N +1.65`,
    // `CB -0.35 ppm`) explicitly against `predict_ncidp_shifts`'s baseline (`-1.95 C'`, `-1.95 CA`,
    // `+1.65 N`, `-0.35 CB`) for non-Gly/non-Pro residues preceding `PRO`.
    for k in 0..seq.len().saturating_sub(1) {
        if seq[k + 1] == "PRO" && seq[k] != "GLY" && seq[k] != "PRO" {
            let (dn_t, dc_t, dca_t, dcb_t) = pre_pro_shift_corrections(&seq[k]);
            preds[k].n_ppm += dn_t - 1.65;
            preds[k].c_ppm += dc_t - (-1.95);
            preds[k].ca_ppm += dca_t - (-1.95);
            if let Some(cb) = &mut preds[k].cb_ppm {
                *cb += dcb_t - (-0.35);
            }
        }
    }

    if cfg.enable_cterm_coo_prior && !preds.is_empty() {
        let last = preds.len() - 1;
        let ph_c = cfg.ph.clamp(1.0, 11.0);
        let f_deprot = 1.0 - 1.0 / (1.0 + 10.0_f64.powf(ph_c - 3.55));
        let target_coo = (preds[last].c_ppm + 0.85 * f_deprot).clamp(174.5, 180.8);
        if f_deprot > 0.5 && target_coo < 178.5 {
            preds[last].c_ppm = 0.5 * (target_coo + CTERM_COO_DSS_CENTER_PPM);
        } else {
            preds[last].c_ppm = target_coo;
        }
        preds[last].sigma_c = preds[last].sigma_c.max(0.65);
    }
    preds
}

// ============================================================================
// Crowded-HSQC-Robust Field-Scaled TROSY `46 Hz (1/2 1J_NH)` Calibration (P2-5)
// ============================================================================

/// Auto-calibrate a field-scaled TROSY `46 Hz (1/2 1J_NH)` cross-spectrum `(ΔH, ΔN)` offset
/// between a TROSY 3D (`BT_HNCACONH`) and decoupled `HSQC`/`HNCO` amide roots.
///
/// Evaluates the decoupled null `(0, 0)` against all four physical `46 Hz (1/2 1J_NH)` multiplet
/// quadrants using 1-to-1 matched-pair coincidence scoring with crowded-HSQC background subtraction
/// (averaging non-physical decoy shifts so dense `~0.22 ppm` IDP HSQCs do not inflate the null),
/// requiring `>= 5` (`MIN_TROSY_MATCHED_ROOTS`) distinct matched roots and a Wilks excess margin
/// over `(0, 0)`, followed by bounded local median refinement (`±0.028 ppm H`, `±0.28 ppm N`).
pub fn calibrate_trosy_hn_offset(
    trosy_roots: &[(f64, f64)],
    decoupled_roots: &[(f64, f64)],
    proton_mhz: f64,
) -> TrosyOffsetCalibration {
    let (exp_dh_signed, exp_dn_signed) = trosy_expected_shift_ppm(proton_mhz);
    let expected_h = exp_dh_signed.abs();
    let expected_n = exp_dn_signed.abs();

    let clean_t: Vec<(f64, f64)> = trosy_roots
        .iter()
        .copied()
        .filter(|&(h, n)| h.is_finite() && n.is_finite())
        .collect();
    let clean_d: Vec<(f64, f64)> = decoupled_roots
        .iter()
        .copied()
        .filter(|&(h, n)| h.is_finite() && n.is_finite())
        .collect();

    if clean_t.len() < MIN_TROSY_MATCHED_ROOTS || clean_d.len() < MIN_TROSY_MATCHED_ROOTS {
        return TrosyOffsetCalibration {
            dh_ppm: 0.0,
            dn_ppm: 0.0,
            expected_half_j_h_ppm: expected_h,
            expected_half_j_n_ppm: expected_n,
            trosy_detected: false,
            matched_roots: 0,
            score_gain: 0.0,
        };
    }

    // Tight 1-to-1 coincidence bandwidth (`0.018 ppm 1H`, `0.18 ppm 15N`) so crowded IDP HSQCs
    // with `~0.22 ppm` 1H dispersion separate cleanly from shifted TROSY peaks.
    let bw_h = 0.018f64;
    let bw_n = 0.18f64;

    let eval_1to1_coinc = |dh: f64, dn: f64| -> (f64, usize) {
        let mut edges: Vec<(f64, usize, usize)> = Vec::new();
        for (ti, &(th, tn)) in clean_t.iter().enumerate() {
            let hc = th - dh;
            let nc = tn - dn;
            for (di, &(rh, rn)) in clean_d.iter().enumerate() {
                let d2 = ((hc - rh) / bw_h).powi(2) + ((nc - rn) / bw_n).powi(2);
                if d2 <= 4.0 {
                    edges.push((d2, ti, di));
                }
            }
        }
        edges.sort_by(|a, b| {
            a.0.total_cmp(&b.0)
                .then_with(|| a.1.cmp(&b.1))
                .then_with(|| a.2.cmp(&b.2))
        });
        let mut used_t = vec![false; clean_t.len()];
        let mut used_d = vec![false; clean_d.len()];
        let mut score = 0.0f64;
        let mut matched = 0usize;
        for (d2, ti, di) in edges {
            if used_t[ti] || used_d[di] {
                continue;
            }
            used_t[ti] = true;
            used_d[di] = true;
            matched += 1;
            score += 2.6 * (-0.5 * d2).exp();
        }
        (score, matched)
    };

    // Estimate empirical crowded-HSQC random-coincidence baseline from 6 non-physical decoy shifts
    let decoy_shifts = [
        (0.32, 3.2),
        (-0.32, -3.2),
        (0.32, -3.2),
        (-0.32, 3.2),
        (0.26, 0.0),
        (0.0, 3.8),
    ];
    let mut bg_score_sum = 0.0f64;
    let mut bg_match_sum = 0.0f64;
    for &(ddh, ddn) in &decoy_shifts {
        let (sc, m) = eval_1to1_coinc(ddh, ddn);
        bg_score_sum += sc;
        bg_match_sum += m as f64;
    }
    let bg_score = bg_score_sum / (decoy_shifts.len() as f64);
    let bg_matches = bg_match_sum / (decoy_shifts.len() as f64);

    let (null_raw, null_matched) = eval_1to1_coinc(0.0, 0.0);
    let null_excess = (null_raw - bg_score).max(0.0);

    // Physical `46 Hz (1/2 1J_NH)` quadrants (`(-H, +N)` is canonical TROSY)
    let candidates = [
        (-expected_h, expected_n),
        (expected_h, -expected_n),
        (-expected_h, -expected_n),
        (expected_h, expected_n),
    ];

    let mut best_cand = (0.0, 0.0);
    let mut best_excess = null_excess;
    let mut best_matched = null_matched;
    for &(ch, cn) in &candidates {
        let (sc, m) = eval_1to1_coinc(ch, cn);
        let ex = (sc - bg_score).max(0.0);
        if ex > best_excess {
            best_excess = ex;
            best_cand = (ch, cn);
            best_matched = m;
        }
    }

    // Local bounded median refinement (`±0.028 ppm H`, `±0.28 ppm N`) around the winning quadrant
    if best_cand != (0.0, 0.0) {
        let mut dhs = Vec::new();
        let mut dns = Vec::new();
        for &(th, tn) in &clean_t {
            let hc = th - best_cand.0;
            let nc = tn - best_cand.1;
            let mut best: Option<(f64, f64, f64)> = None;
            for &(rh, rn) in &clean_d {
                let eh = hc - rh;
                let en = nc - rn;
                if eh.abs() <= 0.032 && en.abs() <= 0.32 {
                    let d2 = (eh / bw_h).powi(2) + (en / bw_n).powi(2);
                    if best.is_none_or(|b| d2 < b.0) {
                        best = Some((d2, th - rh, tn - rn));
                    }
                }
            }
            if let Some((_, dhi, dni)) = best {
                dhs.push(dhi);
                dns.push(dni);
            }
        }
        if dhs.len() >= MIN_TROSY_MATCHED_ROOTS {
            dhs.sort_by(|a, b| a.total_cmp(b));
            dns.sort_by(|a, b| a.total_cmp(b));
            let med_h = dhs[dhs.len() / 2].clamp(best_cand.0 - 0.028, best_cand.0 + 0.028);
            let med_n = dns[dns.len() / 2].clamp(best_cand.1 - 0.28, best_cand.1 + 0.28);
            let (ref_sc, ref_m) = eval_1to1_coinc(med_h, med_n);
            let ref_ex = (ref_sc - bg_score).max(0.0);
            if ref_ex >= best_excess {
                best_excess = ref_ex;
                best_cand = (med_h, med_n);
                best_matched = ref_m;
            }
        }
    }

    let gain = (best_excess - null_excess).max(0.0);
    let detected = best_cand != (0.0, 0.0)
        && best_matched >= MIN_TROSY_MATCHED_ROOTS
        && (best_matched as f64) >= bg_matches + 3.0
        && best_matched >= null_matched + 3
        && gain > (4.0_f64.ln() + WILKS_HALF_CHI2_1_95);

    if detected {
        TrosyOffsetCalibration {
            dh_ppm: best_cand.0,
            dn_ppm: best_cand.1,
            expected_half_j_h_ppm: expected_h,
            expected_half_j_n_ppm: expected_n,
            trosy_detected: true,
            matched_roots: best_matched,
            score_gain: gain,
        }
    } else {
        TrosyOffsetCalibration {
            dh_ppm: 0.0,
            dn_ppm: 0.0,
            expected_half_j_h_ppm: expected_h,
            expected_half_j_n_ppm: expected_n,
            trosy_detected: false,
            matched_roots: null_matched,
            score_gain: 0.0,
        }
    }
}

// ============================================================================
// Built-In IDP Strip Hygiene (`F1(C')` NUS Sidelobes & `Gly CB` Bleed-Through)
// ============================================================================

/// Prune `F1(13C')` NUS reconstruction sidelobes from a single amide carbonyl strip
/// (`HN(CA)CO` or `HNCO`), given `(c_prime_ppm, abs_height)` pairs (`P2-13`).
///
/// - Drops out-of-window (`< 163` or `> 186 ppm`) artifacts and deduplicates sub-peaks (`<= 0.06 ppm`).
/// - **Preserves genuine 2-peak `(C'_i, C'_{i-1})` doublets** even when the weaker `C'_{i-1}`
///   peak is `~0.10–0.18x` of `C'_i` (never prunes a 2-peak strip above the `0.04x` thermal floor).
/// - When `>= 3` distinct in-window peaks are present, first removes symmetric flanking NUS
///   sidelobe pairs `(c0 - Δ, c0 + Δ)` (`|Δ_L - Δ_R| <= 0.25 ppm`, both `< 0.28 * h0`), then
///   drops excess weak sidelobes (`< 0.19 * kh` within `±2.2 ppm` or `< 0.14 * max_h`) while
///   always preserving the top 2 resolved peaks (up to `max_keep`).
pub fn prune_f1_c_prime_nus_sidelobes(
    peaks: &[(f64, f64)],
    max_keep: usize,
) -> (Vec<(f64, f64)>, usize) {
    let cap = max_keep.max(1);
    let mut sorted: Vec<(f64, f64)> = peaks
        .iter()
        .copied()
        .filter(|&(c, h)| c.is_finite() && h.is_finite() && (163.0..=186.0).contains(&c))
        .map(|(c, h)| (c, h.abs().max(1e-6)))
        .collect();
    let mut pruned = peaks.len().saturating_sub(sorted.len());
    if sorted.is_empty() {
        return (Vec::new(), pruned);
    }

    sorted.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.total_cmp(&b.0)));

    // Deduplicate sub-voxel split peaks (`<= 0.06 ppm`)
    let mut dedup: Vec<(f64, f64)> = Vec::with_capacity(sorted.len());
    for (c, h) in sorted {
        if dedup.iter().any(|&(kc, _)| (kc - c).abs() <= 0.06) {
            pruned += 1;
        } else {
            dedup.push((c, h));
        }
    }
    if dedup.is_empty() {
        return (Vec::new(), pruned);
    }

    let max_h = dedup[0].1;

    // Preserve genuine `<= 2`-peak `(C'_i, C'_{i-1})` doublets (including weak `0.10..0.18x` `C'_{i-1}`)
    if dedup.len() <= 2 {
        let mut kept = Vec::with_capacity(cap.min(dedup.len()));
        for (idx, (c, h)) in dedup.into_iter().enumerate() {
            if idx < cap && (idx == 0 || h >= 0.04 * max_h) {
                kept.push((c, h));
            } else {
                pruned += 1;
            }
        }
        return (kept, pruned);
    }

    // `>= 3` peaks: first identify & flag symmetric flanking NUS sidelobe pairs around `(c0, h0)`
    let (c0, h0) = dedup[0];
    let mut is_sym_sidelobe = vec![false; dedup.len()];
    for i in 1..dedup.len() {
        if is_sym_sidelobe[i] || dedup[i].1 >= 0.28 * h0 {
            continue;
        }
        let di = dedup[i].0 - c0;
        if di.abs() < 0.15 || di.abs() > 3.50 {
            continue;
        }
        for j in (i + 1)..dedup.len() {
            if is_sym_sidelobe[j] || dedup[j].1 >= 0.28 * h0 {
                continue;
            }
            let dj = dedup[j].0 - c0;
            // Opposite sides of `c0` with symmetric lag `| |di| - |dj| | <= 0.25 ppm`
            if di * dj < 0.0 && (di.abs() - dj.abs()).abs() <= 0.25 {
                is_sym_sidelobe[i] = true;
                is_sym_sidelobe[j] = true;
                break;
            }
        }
    }

    let mut remaining: Vec<(f64, f64)> = Vec::with_capacity(dedup.len());
    for (idx, pk) in dedup.into_iter().enumerate() {
        if is_sym_sidelobe[idx] {
            pruned += 1;
        } else {
            remaining.push(pk);
        }
    }

    // If pruning symmetric sidelobes already left `<= 2` peaks, keep the doublet intact!
    if remaining.len() <= 2 {
        let mut kept = Vec::with_capacity(cap.min(remaining.len()));
        for (idx, (c, h)) in remaining.into_iter().enumerate() {
            if idx < cap && (idx == 0 || h >= 0.04 * max_h) {
                kept.push((c, h));
            } else {
                pruned += 1;
            }
        }
        return (kept, pruned);
    }

    // Otherwise (`>= 3` remaining), keep top 2 peaks and reject weak asymmetric excess sidelobes
    let mut kept: Vec<(f64, f64)> = Vec::with_capacity(cap);
    for (idx, (c, h)) in remaining.into_iter().enumerate() {
        if kept.len() >= cap {
            pruned += 1;
            continue;
        }
        if idx < 2 && (idx == 0 || h >= 0.04 * max_h) {
            kept.push((c, h));
            continue;
        }
        let is_local_sidelobe = kept
            .iter()
            .any(|&(kc, kh)| (kc - c).abs() <= 2.20 && h < 0.19 * kh);
        let is_global_floor = h < 0.14 * max_h;
        if is_local_sidelobe || is_global_floor {
            pruned += 1;
        } else {
            kept.push((c, h));
        }
    }

    (kept, pruned)
}

/// Apply the DSS-frame `Gly CB` bleed-through guard across all `JointIdpNode`s.
///
/// Whenever a node's primary `CA` candidate falls in the Glycine `CA` band `41.0..=46.5 ppm` DSS
/// (after subtracting `offsets.d_ca`), removes suspect bleed-through / phase-ringing `CB` peaks
/// sitting within `±1.25 ppm` of that `Gly CA` or inside `41.0..=46.5 ppm` DSS.
pub fn apply_gly_cb_bleedthrough_guard(
    nodes: &mut [JointIdpNode],
    offsets: &IdpShiftOffsets,
) -> usize {
    let mut total_guarded = 0usize;
    let is_gly_ca_dss = |ca_obs: f64| -> bool {
        let ca_dss = ca_obs - offsets.d_ca;
        (GLY_CA_DSS_MIN_PPM..=GLY_CA_DSS_MAX_PPM).contains(&ca_dss)
    };

    for node in nodes.iter_mut() {
        if let Some(&ca0) = node.ca_curr.first() {
            if is_gly_ca_dss(ca0) && !node.cb_curr.is_empty() {
                let ca0_dss = ca0 - offsets.d_ca;
                let before = node.cb_curr.len();
                node.cb_curr.retain(|&cb| {
                    let cb_dss = cb - offsets.d_cb;
                    let near_gly_ca = (cb_dss - ca0_dss).abs() <= 1.25;
                    let inside_gly_band =
                        (GLY_CA_DSS_MIN_PPM..=GLY_CA_DSS_MAX_PPM).contains(&cb_dss);
                    !(near_gly_ca || inside_gly_band)
                });
                let removed = before.saturating_sub(node.cb_curr.len());
                if removed > 0 {
                    node.gly_cb_curr_suppressed = true;
                    total_guarded += removed;
                }
            }
        }
        if let Some(&ca0) = node.ca_prev.first() {
            if is_gly_ca_dss(ca0) && !node.cb_prev.is_empty() {
                let ca0_dss = ca0 - offsets.d_ca;
                let before = node.cb_prev.len();
                node.cb_prev.retain(|&cb| {
                    let cb_dss = cb - offsets.d_cb;
                    let near_gly_ca = (cb_dss - ca0_dss).abs() <= 1.25;
                    let inside_gly_band =
                        (GLY_CA_DSS_MIN_PPM..=GLY_CA_DSS_MAX_PPM).contains(&cb_dss);
                    !(near_gly_ca || inside_gly_band)
                });
                let removed = before.saturating_sub(node.cb_prev.len());
                if removed > 0 {
                    node.gly_cb_prev_suppressed = true;
                    total_guarded += removed;
                }
            }
        }
    }
    total_guarded
}

// ============================================================================
// Overlap & Breakthrough-Calibrated `CAS` Likelihood (Shared with `c13_detected`)
// ============================================================================

/// Precomputed overlap- and breakthrough-calibrated match table for one `CAS` subspectrum,
/// delegating core LLR calibration to [`cas_llr_adjustment`] (`P2-3`) scaled by per-root
/// spectral-overlap fanout and peak-intensity confidence.
#[derive(Clone, Debug)]
pub struct CalibratedCasTable {
    pub spectrum: CasSpectrum,
    pub mode: CasMode,
    pub residues: Vec<String>,
    /// Per-node overlap-damped hit strength in `[0.0, 1.0]` (`0.0` = no hit).
    pub hit_weight: Vec<f64>,
    /// Calibrated effective true-positive detection probability `p_tp`.
    pub p_tp_eff: f64,
    /// Calibrated effective breakthrough / false-positive probability `p_fp`.
    pub p_fp_eff: f64,
    /// Number of nodes matched by at least one peak in this `CAS` spectrum.
    pub n_nodes_hit: usize,
    /// Number of distinct CAS peaks that matched at least one node.
    pub n_matched_peaks: usize,
    /// Count of matching residues in `sequence[1..L]` (or `0..L-1` for `Cacon`).
    pub seq_type_count: usize,
    /// Maximum LLR scale (nats) forwarded to `cas_llr_adjustment`.
    pub max_llr: f64,
}

impl CalibratedCasTable {
    /// Evaluate calibrated Bayesian log-likelihood ratio (nats) for placing `node_idx` at a
    /// sequence position whose target residue (`k - 1` for `Cacon`, `k` for `Canco`) is `target_res`.
    #[inline]
    pub fn llr_for(&self, node_idx: usize, target_res: &str) -> f64 {
        let w = self.hit_weight.get(node_idx).copied().unwrap_or(0.0);
        // Delegate base breakthrough-calibrated LLR to `c13_detected::cas_llr_adjustment` (`P2-3`),
        // passing `n_matched_peaks` so local spectral overlap (`w`) and global breakthrough
        // (`n_matched_peaks > seq_type_count`) are cleanly decoupled without double-penalizing!
        if w > 0.0 {
            w * cas_llr_adjustment(
                &self.spectrum,
                true,
                target_res,
                self.max_llr,
                self.seq_type_count,
                self.n_matched_peaks,
            )
        } else {
            cas_llr_adjustment(
                &self.spectrum,
                false,
                target_res,
                self.max_llr,
                self.seq_type_count,
                self.n_matched_peaks,
            )
        }
    }
}

/// Build overlap-fanout and breakthrough-calibrated `CAS` hit tables against unified `JointIdpNode`s.
pub fn build_calibrated_cas_tables(
    cas_spectra: &[CasSpectrum],
    nodes: &[JointIdpNode],
    sequence: &[String],
    cfg: &JointIdpConfig,
) -> Vec<CalibratedCasTable> {
    let seq = normalize_seq_3(sequence);
    let k_len = seq.len();
    let s_len = nodes.len();
    if cas_spectra.is_empty() || s_len == 0 || k_len < 2 {
        return Vec::new();
    }

    let ct = cfg.cas_c_tol_ppm.max(1e-4);
    let nt = cfg.cas_n_tol_ppm.max(1e-4);
    let mut tables = Vec::with_capacity(cas_spectra.len());

    for cas in cas_spectra {
        let clean_peaks: Vec<(f64, f64, f64)> = cas
            .peaks
            .iter()
            .filter(|p| p.c_prime_ppm.is_finite() && p.n_ppm.is_finite())
            .map(|p| (p.c_prime_ppm, p.n_ppm, p.height.abs().max(1.0)))
            .collect();

        let seq_type_count = match cas.mode {
            CasMode::Cacon => (1..k_len)
                .filter(|&k| cas.matches_residue(&seq[k - 1]))
                .count(),
            CasMode::Canco => (1..k_len).filter(|&k| cas.matches_residue(&seq[k])).count(),
        };

        if clean_peaks.is_empty() {
            tables.push(CalibratedCasTable {
                spectrum: cas.clone(),
                mode: cas.mode,
                residues: cas.residues.clone(),
                hit_weight: vec![0.0; s_len],
                p_tp_eff: 0.0,
                p_fp_eff: cfg.cas_base_fp.clamp(0.10, 0.52),
                n_nodes_hit: 0,
                n_matched_peaks: 0,
                seq_type_count,
                max_llr: cfg.cas_max_llr.max(0.5),
            });
            continue;
        }

        let eval_matches = |dc: f64, dn: f64| -> (Vec<f64>, usize) {
            let mut weights = vec![0.0f64; s_len];
            let mut matched_pks = 0usize;
            let mut heights: Vec<f64> = clean_peaks.iter().map(|x| x.2).collect();
            heights.sort_by(|a, b| a.total_cmp(b));
            let med_h = heights[heights.len() / 2].max(1e-6);

            for &(cp_raw, np_raw, h) in &clean_peaks {
                let cp = cp_raw - dc;
                let np = np_raw - dn;
                let mut hits_for_pk: Vec<(usize, f64)> = Vec::new();
                for (s_idx, node) in nodes.iter().enumerate() {
                    let Some(rc) = node.c_prev else {
                        continue;
                    };
                    let d2 = ((cp - rc) / ct).powi(2) + ((np - node.n_curr) / nt).powi(2);
                    if d2 <= 1.0 {
                        hits_for_pk.push((s_idx, d2));
                    }
                }
                if !hits_for_pk.is_empty() {
                    matched_pks += 1;
                    let fanout = hits_for_pk.len() as f64;
                    let w_overlap = 1.0 / (1.0 + 0.45 * (fanout - 1.0));
                    let rel_h = (h / med_h).clamp(0.05, 3.0);
                    let w_int = (0.55 + 0.45 * (rel_h / 0.60).clamp(0.0, 1.0)).clamp(0.55, 1.0);
                    for (s_idx, d2) in hits_for_pk {
                        let w_dist = (-0.25 * d2).exp();
                        let w = (w_overlap * w_int * w_dist).clamp(0.0, 1.0);
                        if w > weights[s_idx] {
                            weights[s_idx] = w;
                        }
                    }
                }
            }
            (weights, matched_pks)
        };

        let (mut hit_weight, mut matched_pks) = eval_matches(0.0, 0.0);
        if matched_pks * 2 < clean_peaks.len() && clean_peaks.len() >= 5 && s_len >= 5 {
            let mut dcs = Vec::new();
            let mut dns = Vec::new();
            for &(cp, np, _) in &clean_peaks {
                let mut best: Option<(f64, f64, f64)> = None;
                for node in nodes {
                    let Some(rc) = node.c_prev else {
                        continue;
                    };
                    let dc = cp - rc;
                    let dn = np - node.n_curr;
                    if dc.abs() <= 0.16 && dn.abs() <= 0.48 {
                        let d2 = (dc / ct).powi(2) + (dn / nt).powi(2);
                        if best.is_none_or(|b| d2 < b.0) {
                            best = Some((d2, dc, dn));
                        }
                    }
                }
                if let Some((_, dc, dn)) = best {
                    dcs.push(dc);
                    dns.push(dn);
                }
            }
            if dcs.len() >= 5 {
                dcs.sort_by(|a, b| a.total_cmp(b));
                dns.sort_by(|a, b| a.total_cmp(b));
                let med_dc = dcs[dcs.len() / 2];
                let med_dn = dns[dns.len() / 2];
                if med_dc.abs() <= 0.14 && med_dn.abs() <= 0.42 {
                    let (shifted_w, shifted_pks) = eval_matches(med_dc, med_dn);
                    if shifted_pks >= matched_pks + 2 {
                        hit_weight = shifted_w;
                        matched_pks = shifted_pks;
                    }
                }
            }
        }

        let n_nodes_hit = hit_weight.iter().filter(|&&w| w > 0.0).count();
        let excess_fp = if seq_type_count > 0 && matched_pks > seq_type_count {
            (((matched_pks - seq_type_count) as f64) / (matched_pks as f64)) * 0.45
        } else {
            0.0
        };
        let p_fp_eff = (0.12 + excess_fp).clamp(0.10, 0.52);
        let p_tp_eff = 0.82f64;

        tables.push(CalibratedCasTable {
            spectrum: cas.clone(),
            mode: cas.mode,
            residues: cas.residues.clone(),
            hit_weight,
            p_tp_eff,
            p_fp_eff,
            n_nodes_hit,
            n_matched_peaks: matched_pks,
            seq_type_count,
            max_llr: cfg.cas_max_llr.max(0.5),
        });
    }

    tables
}

// ============================================================================
// Unlabelled Reference Offset Estimation (Delegated to `idp_solver`, P2-3)
// ============================================================================

/// Strictly unlabelled, permutation-invariant global `(dN, dC', dCA, dCB)` offset estimator over
/// unified `JointIdpNode`s using shared [`fit_1d_unlabelled_marginal_offset`] and
/// [`evaluate_idp_auto_reference_safety`] from `idp_solver` (`P2-3`, TMS-safe windows).
pub fn estimate_joint_reference_offsets(
    nodes: &[JointIdpNode],
    sequence: &[String],
    cfg: &JointIdpConfig,
) -> (IdpShiftOffsets, IdpAutoRefSafetyReport) {
    if nodes.len() < 5 || sequence.len() < 2 {
        return (
            IdpShiftOffsets::default(),
            IdpAutoRefSafetyReport::default(),
        );
    }
    let preds = predict_joint_ncidp_shifts(sequence, cfg);

    let mut obs_n: Vec<f64> = nodes
        .iter()
        .map(|s| s.n_curr)
        .filter(|v| v.is_finite() && (98.0..=148.0).contains(v))
        .collect();
    obs_n.sort_by(|a, b| a.total_cmp(b));

    let mut obs_c: Vec<f64> = nodes
        .iter()
        .filter_map(|s| s.c_prev)
        .filter(|v| v.is_finite() && (163.0..=185.0).contains(v))
        .collect();
    obs_c.sort_by(|a, b| a.total_cmp(b));

    let mut obs_ca: Vec<f64> = Vec::with_capacity(nodes.len() * 2);
    let mut obs_cb: Vec<f64> = Vec::with_capacity(nodes.len() * 2);
    for n in nodes {
        let mut ca_s: Vec<f64> = n
            .ca_curr
            .iter()
            .chain(n.ca_prev.iter())
            .copied()
            .filter(|v| v.is_finite() && (37.0..=71.0).contains(v))
            .collect();
        ca_s.sort_by(|a, b| a.total_cmp(b));
        ca_s.dedup_by(|a, b| (*a - *b).abs() < 0.04);
        obs_ca.extend(ca_s);

        let mut cb_s: Vec<f64> = n
            .cb_prev
            .iter()
            .chain(n.cb_curr.iter())
            .copied()
            .filter(|v| v.is_finite() && (10.0..=76.0).contains(v))
            .collect();
        cb_s.sort_by(|a, b| a.total_cmp(b));
        cb_s.dedup_by(|a, b| (*a - *b).abs() < 0.06);
        obs_cb.extend(cb_s.into_iter().take(MAX_JOINT_CB_CANDS));
    }
    obs_ca.sort_by(|a, b| a.total_cmp(b));
    obs_cb.sort_by(|a, b| a.total_cmp(b));

    let has_con_pro = nodes.iter().any(|n| n.con_matched && n.n_curr >= 131.0);
    let mut mod_n: Vec<(f64, f64)> = preds[1..]
        .iter()
        .filter(|p| has_con_pro || p.residue != "PRO")
        .map(|p| (p.n_ppm, p.sigma_n))
        .collect();
    mod_n.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut mod_c: Vec<(f64, f64)> = preds[..preds.len() - 1]
        .iter()
        .map(|p| (p.c_ppm, p.sigma_c))
        .collect();
    mod_c.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut mod_ca: Vec<(f64, f64)> = preds.iter().map(|p| (p.ca_ppm, p.sigma_ca)).collect();
    mod_ca.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut mod_cb: Vec<(f64, f64)> = preds
        .iter()
        .filter_map(|p| p.cb_ppm.map(|cb| (cb, p.sigma_cb)))
        .collect();
    mod_cb.sort_by(|a, b| a.0.total_cmp(&b.0));

    let raw = IdpShiftOffsets {
        d_n: fit_1d_unlabelled_marginal_offset(&obs_n, &mod_n, 38.0, 4.0, 0.02),
        d_c: fit_1d_unlabelled_marginal_offset(&obs_c, &mod_c, 16.0, 4.0, 0.01),
        d_ca: fit_1d_unlabelled_marginal_offset(&obs_ca, &mod_ca, 30.0, 4.0, 0.01),
        d_cb: fit_1d_unlabelled_marginal_offset(&obs_cb, &mod_cb, 60.0, 4.0, 0.015),
    };

    let proxy_systems: Vec<ConDipeptideSystem> = nodes
        .iter()
        .map(|n| {
            let mut s =
                ConDipeptideSystem::new(n.id, n.c_prev.unwrap_or(175.0), n.n_curr, n.root_height);
            s.ca_curr = n.ca_curr.clone();
            s.cb_curr = n.cb_curr.clone();
            s.ca_prev = n.ca_prev.clone();
            s.cb_prev = n.cb_prev.clone();
            s.c_curr = n.c_curr.clone();
            s
        })
        .collect();
    let safety = evaluate_idp_auto_reference_safety(&proxy_systems, &raw);
    (safety.safe_offsets, safety)
}

// ============================================================================
// Connected-Component Kuhn–Munkres (Hungarian) 1-to-1 `HN ↔ CON` Fusion (P2-1)
// ============================================================================

/// Exact Kuhn–Munkres (Hungarian) minimum-cost maximum-cardinality bipartite matching on an
/// `n_rows x n_cols` cost matrix where `f64::INFINITY` denotes forbidden edges (`P2-1`).
///
/// Returns matched `(row, col)` pairs sorted by `row`. Uses a clean-room Jonker–Volgenant /
/// Kuhn–Munkres shortest-augmenting-path formulation with finite dummy cost `BIG_COST` for
/// unassigned vertices so every real edge `< BIG_COST` strictly increases matching cardinality.
fn hungarian_min_cost_max_card(cost: &[Vec<f64>]) -> Vec<(usize, usize)> {
    let n_rows = cost.len();
    if n_rows == 0 {
        return Vec::new();
    }
    let n_cols = cost[0].len();
    if n_cols == 0 {
        return Vec::new();
    }

    // Pad to square `n x n` (`n = n_rows + n_cols` dummy slots so every real vertex can either
    // match a valid partner with cost `c_ij << UNMATCHED_COST` or stay unmatched cleanly).
    const UNMATCHED_COST: f64 = 1.0e5;
    const FORBID_EDGE: f64 = 1.0e9;

    // Transpose if `n_rows > n_cols` so `m <= n` isn't needed with square padding of size `n_rows + n_cols`
    // Actually for small connected components (typically `<= 16`), a `dim = n_rows + n_cols` matrix
    // handles rectangular maximum-cardinality minimum-cost matching in one exact pass!
    let dim = n_rows + n_cols;
    let mut a = vec![vec![FORBID_EDGE; dim + 1]; dim + 1];
    for r in 0..n_rows {
        for c in 0..n_cols {
            let v = cost[r][c];
            if v.is_finite() && v < UNMATCHED_COST * 0.5 {
                a[r + 1][c + 1] = v;
            }
        }
        // Dummy unmatched column for row `r`
        a[r + 1][n_cols + r + 1] = UNMATCHED_COST;
    }
    for c in 0..n_cols {
        // Dummy unmatched row for col `c`
        a[n_rows + c + 1][c + 1] = UNMATCHED_COST;
        for r in 0..n_rows {
            a[n_rows + c + 1][n_cols + r + 1] = 0.0;
        }
    }

    let mut u = vec![0.0f64; dim + 1];
    let mut v = vec![0.0f64; dim + 1];
    let mut p = vec![0usize; dim + 1];
    let mut way = vec![0usize; dim + 1];

    for i in 1..=dim {
        p[0] = i;
        let mut j0 = 0usize;
        let mut minv = vec![f64::INFINITY; dim + 1];
        let mut used = vec![false; dim + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = f64::INFINITY;
            let mut j1 = 0usize;
            for j in 1..=dim {
                if !used[j] {
                    let cur = a[i0][j] - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for j in 0..=dim {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }

    let mut matched = Vec::new();
    for j in 1..=n_cols {
        let i = p[j];
        if i >= 1 && i <= n_rows {
            let c_ij = cost[i - 1][j - 1];
            if c_ij.is_finite() && c_ij < UNMATCHED_COST * 0.5 {
                matched.push((i - 1, j - 1));
            }
        }
    }
    matched.sort_by_key(|&(r, _)| r);
    matched
}

/// Partition bipartite candidate edges `(hn_idx, con_idx, cost)` into connected components and solve
/// each component via exact Hungarian 1-to-1 optimal matching (`P2-1`).
fn match_hn_con_bipartite_hungarian(
    n_hn: usize,
    n_con: usize,
    edges: &[(usize, usize, f64)],
) -> Vec<(usize, usize, f64)> {
    if n_hn == 0 || n_con == 0 || edges.is_empty() {
        return Vec::new();
    }
    let mut adj_h: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n_hn];
    let mut adj_c: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n_con];
    for &(h, c, w) in edges {
        adj_h[h].push((c, w));
        adj_c[c].push((h, w));
    }

    let mut vis_h = vec![false; n_hn];
    let mut vis_c = vec![false; n_con];
    let mut result = Vec::new();

    for start_h in 0..n_hn {
        if vis_h[start_h] || adj_h[start_h].is_empty() {
            continue;
        }
        let mut comp_h = Vec::new();
        let mut comp_c = Vec::new();
        let mut q: VecDeque<(bool, usize)> = VecDeque::new();
        vis_h[start_h] = true;
        q.push_back((true, start_h));

        while let Some((is_h, u)) = q.pop_front() {
            if is_h {
                comp_h.push(u);
                for &(c, _) in &adj_h[u] {
                    if !vis_c[c] {
                        vis_c[c] = true;
                        q.push_back((false, c));
                    }
                }
            } else {
                comp_c.push(u);
                for &(h, _) in &adj_c[u] {
                    if !vis_h[h] {
                        vis_h[h] = true;
                        q.push_back((true, h));
                    }
                }
            }
        }
        comp_h.sort_unstable();
        comp_c.sort_unstable();

        // Fast path for isolated 1-to-1 component
        if comp_h.len() == 1 && comp_c.len() == 1 {
            let h = comp_h[0];
            let c = comp_c[0];
            if let Some(&(_, w)) = adj_h[h].iter().find(|&&(cc, _)| cc == c) {
                result.push((h, c, w));
            }
            continue;
        }

        // Guard against pathological giant connected component (> 64 vertices): fallback to sorted greedy
        if comp_h.len() + comp_c.len() > 64 {
            let mut local_edges: Vec<(f64, usize, usize)> = Vec::new();
            for &h in &comp_h {
                for &(c, w) in &adj_h[h] {
                    local_edges.push((w, h, c));
                }
            }
            local_edges.sort_by(|a, b| a.0.total_cmp(&b.0));
            let mut used_h = BTreeSet::new();
            let mut used_c = BTreeSet::new();
            for (w, h, c) in local_edges {
                if used_h.insert(h) && used_c.insert(c) {
                    result.push((h, c, w));
                }
            }
            continue;
        }

        let mut c_map: BTreeMap<usize, usize> = BTreeMap::new();
        for (idx, &c) in comp_c.iter().enumerate() {
            c_map.insert(c, idx);
        }
        let mut cost = vec![vec![f64::INFINITY; comp_c.len()]; comp_h.len()];
        for (ri, &h) in comp_h.iter().enumerate() {
            for &(c, w) in &adj_h[h] {
                if let Some(&ci) = c_map.get(&c) {
                    if w < cost[ri][ci] {
                        cost[ri][ci] = w;
                    }
                }
            }
        }
        for (ri, ci) in hungarian_min_cost_max_card(&cost) {
            result.push((comp_h[ri], comp_c[ci], cost[ri][ci]));
        }
    }

    result.sort_by_key(|&(h, c, _)| (h, c));
    result
}

// ============================================================================
// Unified Node Fusion Builder (`HN + 2D CON + Proline-Bridge + N–N`)
// ============================================================================

/// Canonicalize `1H`-detected experiment labels, accepting all `CBCACONH` / `HNCOCACB` /
/// `CBCA(CO)NH` / `HN(CO)CACB` aliases (`P2-4`).
fn canonical_hn_exp(raw: &str) -> &'static str {
    let norm: String = raw
        .trim()
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .map(|c| c.to_ascii_uppercase())
        .collect();
    match norm.as_str() {
        "HSQC" | "15NHSQC" | "N15HSQC" | "BESTHSQC" | "TROSY" | "TROSYHSQC" | "SOFAST"
        | "SOFASTHMQC" | "NH" => "HSQC",
        "HNCO" | "BESTHNCO" => "HNCO",
        "HNCACO" | "BESTHNCACO" => "HNCACO",
        "HNCACB" | "BESTHNCACB" | "CBCA" => "HNCACB",
        "CBCACONH" | "HNCOCACB" | "BESTCBCACONH" | "BESTHNCOCACB" => "CBCACONH",
        "HNCA" | "BESTHNCA" => "HNCA",
        "HNCOCA" | "BESTHNCOCA" => "HNCOCA",
        _ => "",
    }
}

#[inline]
fn dedup_push_f64(dst: &mut Vec<f64>, val: f64, tol: f64, max_len: usize) {
    if !val.is_finite() || dst.len() >= max_len {
        return;
    }
    if !dst.iter().any(|&x| (x - val).abs() <= tol) {
        dst.push(val);
    }
}

/// Pool two candidate lists (`primary` and `secondary`), placing peaks corroborated by BOTH routes
/// first, followed by unique primary and secondary peaks.
fn pool_corroborated_channels(
    primary: &[f64],
    secondary: &[f64],
    match_tol: f64,
    dedup_tol: f64,
    max_len: usize,
) -> Vec<f64> {
    let mut out = Vec::with_capacity(max_len);
    for &p in primary {
        if let Some(&q) = secondary
            .iter()
            .min_by(|&&a, &&b| (a - p).abs().total_cmp(&(b - p).abs()))
        {
            if (p - q).abs() <= match_tol {
                dedup_push_f64(&mut out, 0.5 * (p + q), dedup_tol, max_len);
            }
        }
    }
    for &p in primary {
        dedup_push_f64(&mut out, p, dedup_tol, max_len);
    }
    for &q in secondary {
        dedup_push_f64(&mut out, q, dedup_tol, max_len);
    }
    out
}

#[derive(Clone, Debug)]
struct HnRootDraft {
    h: f64,
    n: f64,
    c_prev: Option<f64>,
    root_height: f64,
    ca_curr: Vec<f64>,
    cb_curr: Vec<f64>,
    ca_prev: Vec<f64>,
    cb_prev: Vec<f64>,
    c_curr: Vec<f64>,
    has_prev_cb_exp: bool,
    has_curr_cb_exp: bool,
    f1_sidelobes_pruned: usize,
}

fn sanitize_c13_dataset(raw: &C13PickedDataset) -> C13PickedDataset {
    let con_roots: Vec<Con2DPeak> = raw
        .con_roots
        .iter()
        .filter(|p| {
            p.c_prev_ppm.is_finite()
                && p.n_curr_ppm.is_finite()
                && p.height.is_finite()
                && (162.0..=186.0).contains(&p.c_prev_ppm)
                && (98.0..=148.0).contains(&p.n_curr_ppm)
        })
        .cloned()
        .collect();

    let caco_peaks = raw
        .caco_peaks
        .iter()
        .filter(|p| {
            p.c_curr_ppm.is_finite()
                && p.ca_curr_ppm.is_finite()
                && p.height.is_finite()
                && (162.0..=186.0).contains(&p.c_curr_ppm)
                && (36.0..=72.0).contains(&p.ca_curr_ppm)
        })
        .cloned()
        .collect();

    // Auto-anchor unanchored `CANCO`/`CBCACON`/`CCCON` 3D peaks onto nearest `CON` root
    // using noise-tolerant `(0.14 C', 0.50 N)` windows so noisy strips still attach without leaking truth!
    let mut strip_peaks = Vec::with_capacity(raw.strip_peaks.len());
    for p in &raw.strip_peaks {
        if !(p.c_prime_ppm.is_finite()
            && p.n_ppm.is_finite()
            && p.c_indirect_ppm.is_finite()
            && p.height.is_finite())
        {
            continue;
        }
        let mut q = p.clone();
        if q.anchor_root_idx.is_some_and(|idx| idx >= con_roots.len()) {
            q.anchor_root_idx = None;
        }
        if q.anchor_root_idx.is_none()
            && matches!(
                q.exp,
                C13DetectedExp::Canco3D | C13DetectedExp::Cbcaco3D | C13DetectedExp::Cccon3D
            )
            && !con_roots.is_empty()
        {
            let mut best: Option<(usize, f64)> = None;
            for (r_idx, r) in con_roots.iter().enumerate() {
                let d2 = ((q.c_prime_ppm - r.c_prev_ppm) / 0.14).powi(2)
                    + ((q.n_ppm - r.n_curr_ppm) / 0.50).powi(2);
                if d2 <= 1.0 && best.is_none_or(|b| d2 < b.1) {
                    best = Some((r_idx, d2));
                }
            }
            q.anchor_root_idx = best.map(|b| b.0);
        }
        strip_peaks.push(q);
    }

    C13PickedDataset {
        con_roots,
        caco_peaks,
        strip_peaks,
    }
}

/// Estimate bounded `(dC', dN)` carrier drift (`HNCO - CON`) between `HNCO` roots and `2D CON` roots.
fn estimate_hnco_to_con_drift(
    hn_roots: &[HnRootDraft],
    con_systems: &[ConDipeptideSystem],
) -> (f64, f64) {
    let hn_cn: Vec<(f64, f64)> = hn_roots
        .iter()
        .filter_map(|r| r.c_prev.map(|c| (c, r.n)))
        .collect();
    if hn_cn.len() < 4 || con_systems.len() < 4 {
        return (0.0, 0.0);
    }

    let count_matches = |dc: f64, dn: f64| -> usize {
        hn_cn
            .iter()
            .filter(|&&(hc, hn)| {
                con_systems.iter().any(|cs| {
                    ((hc - dc - cs.c_prev) / 0.12).powi(2) + ((hn - dn - cs.n_curr) / 0.42).powi(2)
                        <= 1.0
                })
            })
            .count()
    };

    let direct = count_matches(0.0, 0.0);
    let (max_dc, max_dn) = if direct * 2 >= hn_cn.len() {
        (0.22, 0.55)
    } else {
        (3.50, 1.50)
    };

    let mut best_dc = 0.0f64;
    let mut best_dn = 0.0f64;
    let mut best_votes = direct;

    let mut cand_pairs: Vec<(f64, f64)> = Vec::new();
    for &(hc, hn) in &hn_cn {
        for cs in con_systems {
            let dc = hc - cs.c_prev;
            let dn = hn - cs.n_curr;
            if dc.abs() <= max_dc && dn.abs() <= max_dn {
                cand_pairs.push((dc, dn));
            }
        }
    }
    if cand_pairs.len() < 4 {
        return (0.0, 0.0);
    }

    let mut bins: BTreeMap<(i32, i32), Vec<(f64, f64)>> = BTreeMap::new();
    for &(dc, dn) in &cand_pairs {
        let bc = (dc / 0.08).round() as i32;
        let bn = (dn / 0.30).round() as i32;
        bins.entry((bc, bn)).or_default().push((dc, dn));
    }
    let mut top_bins: Vec<((i32, i32), usize)> = bins.iter().map(|(&k, v)| (k, v.len())).collect();
    top_bins.sort_by_key(|b| std::cmp::Reverse(b.1));

    for ((bc, bn), _) in top_bins.into_iter().take(6) {
        let mut dcs = Vec::new();
        let mut dns = Vec::new();
        for o_c in -1..=1 {
            for o_n in -1..=1 {
                if let Some(list) = bins.get(&(bc + o_c, bn + o_n)) {
                    for &(dc, dn) in list {
                        dcs.push(dc);
                        dns.push(dn);
                    }
                }
            }
        }
        if dcs.len() < 4 {
            continue;
        }
        dcs.sort_by(|a, b| a.total_cmp(b));
        dns.sort_by(|a, b| a.total_cmp(b));
        let mdc = dcs[dcs.len() / 2];
        let mdn = dns[dns.len() / 2];
        let votes = count_matches(mdc, mdn);
        if votes > best_votes + 2
            || (votes == best_votes && mdc.abs() + mdn.abs() < best_dc.abs() + best_dn.abs())
        {
            best_votes = votes;
            best_dc = mdc;
            best_dn = mdn;
        }
    }

    if best_votes >= direct + 2 && (best_dc.abs() >= 0.015 || best_dn.abs() >= 0.05) {
        (best_dc, best_dn)
    } else {
        (0.0, 0.0)
    }
}

/// Build unified `JointIdpNode`s by fusing `HNCO`-supported amide roots with `2D CON` via exact
/// connected-component Hungarian 1-to-1 matching (`P2-1`), preserving Proline `CON` bridges and
/// exchange-broadened `CON`-only amides, pooling all multi-experiment channels, calibrating TROSY
/// `46 Hz (1/2 1J_NH)` `(dH, dN)` (`P2-5`) and global TMS/DSS reference offsets (`P2-3`), and
/// applying `F1(C')` NUS sidelobe pruning (`P2-13`) and `Gly CB` bleed-through guards.
pub fn build_joint_idp_nodes(
    dataset: &JointIdpDataset,
    sequence: &[String],
    cfg: &JointIdpConfig,
) -> JointNodeBuildReport {
    let mut warnings: Vec<String> = Vec::new();
    let proton_mhz = if dataset.proton_mhz.is_finite() && dataset.proton_mhz > 50.0 {
        dataset.proton_mhz
    } else if cfg.proton_mhz.is_finite() && cfg.proton_mhz > 50.0 {
        cfg.proton_mhz
    } else {
        CANONICAL_TROSY_FALLBACK_MHZ
    };

    // 1. Sanitize `HN` peaks, normalize experiment aliases (`P2-4`), and normalize polarity (`RD-1`)
    let clean_hn_raw: Vec<Peak> = dataset
        .hn_peaks
        .iter()
        .filter_map(|p| {
            if !(p.h.is_finite()
                && p.n.is_finite()
                && p.c.is_none_or(f64::is_finite)
                && p.height.is_none_or(f64::is_finite)
                && p.volume.is_none_or(f64::is_finite)
                && (4.5..=12.5).contains(&p.h)
                && (95.0..=148.0).contains(&p.n))
            {
                return None;
            }
            let canon = canonical_hn_exp(&p.exp);
            if canon.is_empty() {
                return None;
            }
            let mut q = p.clone();
            q.exp = canon.to_string();
            Some(q)
        })
        .collect();

    let (clean_hn, polarity_report) = if clean_hn_raw.is_empty() {
        (Vec::new(), None)
    } else {
        let (norm, rep) =
            normalize_hncacb_and_cbcaconh_polarity(&clean_hn_raw, cfg.hncacb_polarity);
        (norm, Some(rep))
    };

    let mut hsqc_pks: Vec<&Peak> = Vec::new();
    let mut hnco_pks: Vec<&Peak> = Vec::new();
    let mut hncaco_pks: Vec<&Peak> = Vec::new();
    let mut hncacb_pks: Vec<&Peak> = Vec::new();
    let mut cbcaconh_pks: Vec<&Peak> = Vec::new();
    let mut hnca_pks: Vec<&Peak> = Vec::new();
    let mut hncoca_pks: Vec<&Peak> = Vec::new();

    for p in &clean_hn {
        match p.exp.as_str() {
            "HSQC" => hsqc_pks.push(p),
            "HNCO" if p.c.is_some() => hnco_pks.push(p),
            "HNCACO" if p.c.is_some() => hncaco_pks.push(p),
            "HNCACB" if p.c.is_some() => hncacb_pks.push(p),
            "CBCACONH" if p.c.is_some() => cbcaconh_pks.push(p),
            "HNCA" if p.c.is_some() => hnca_pks.push(p),
            "HNCOCA" if p.c.is_some() => hncoca_pks.push(p),
            _ => {}
        }
    }

    let has_prev_cb_hn_exp = !cbcaconh_pks.is_empty();
    let has_curr_cb_hn_exp = !hncacb_pks.is_empty();
    let cbcaconh_has_both_signs =
        cbcaconh_pks.iter().any(|p| p.sign > 0) && cbcaconh_pks.iter().any(|p| p.sign < 0);

    // 2. Construct `HN` amide roots (`HNCO`-preferred + optional `HSQC` fallback with 3D support)
    let mut hn_drafts: Vec<HnRootDraft> = Vec::new();
    let mut sorted_hnco = hnco_pks.clone();
    sorted_hnco.sort_by(|a, b| {
        b.height
            .map(f64::abs)
            .unwrap_or(1.0)
            .total_cmp(&a.height.map(f64::abs).unwrap_or(1.0))
            .then_with(|| a.n.total_cmp(&b.n))
            .then_with(|| a.h.total_cmp(&b.h))
    });

    for hp in sorted_hnco {
        let c_val = hp.c.unwrap_or(0.0);
        if !(162.0..=186.0).contains(&c_val) {
            continue;
        }
        let dup = hn_drafts.iter().any(|r| {
            (r.h - hp.h).abs() <= 0.022
                && (r.n - hp.n).abs() <= 0.22
                && r.c_prev.is_some_and(|rc| (rc - c_val).abs() <= 0.08)
        });
        if dup {
            continue;
        }
        hn_drafts.push(HnRootDraft {
            h: hp.h,
            n: hp.n,
            c_prev: Some(c_val),
            root_height: hp.height.map(f64::abs).unwrap_or(1.0e5).max(1.0),
            ca_curr: Vec::new(),
            cb_curr: Vec::new(),
            ca_prev: Vec::new(),
            cb_prev: Vec::new(),
            c_curr: Vec::new(),
            has_prev_cb_exp: has_prev_cb_hn_exp,
            has_curr_cb_exp: has_curr_cb_hn_exp,
            f1_sidelobes_pruned: 0,
        });
    }

    if cfg.include_hsqc_without_hnco && !hsqc_pks.is_empty() {
        for sp in &hsqc_pks {
            let covered_by_hnco = hn_drafts
                .iter()
                .any(|r| ((r.h - sp.h) / 0.045).hypot((r.n - sp.n) / 0.45) <= 1.0);
            if covered_by_hnco {
                continue;
            }
            let has_3d = hncacb_pks
                .iter()
                .chain(hnca_pks.iter())
                .chain(cbcaconh_pks.iter())
                .any(|p| ((p.h - sp.h) / 0.055).hypot((p.n - sp.n) / 0.50) <= 1.0);
            if has_3d || (hnco_pks.is_empty() && hncacb_pks.is_empty() && hnca_pks.is_empty()) {
                hn_drafts.push(HnRootDraft {
                    h: sp.h,
                    n: sp.n,
                    c_prev: None,
                    root_height: sp.height.map(f64::abs).unwrap_or(1.0e5).max(1.0),
                    ca_curr: Vec::new(),
                    cb_curr: Vec::new(),
                    ca_prev: Vec::new(),
                    cb_prev: Vec::new(),
                    c_curr: Vec::new(),
                    has_prev_cb_exp: has_prev_cb_hn_exp,
                    has_curr_cb_exp: has_curr_cb_hn_exp,
                    f1_sidelobes_pruned: 0,
                });
            }
        }
    }

    // 3. Populate 3D `HN` channels per amide root with noise-adaptive `(tol_h, tol_n)`
    let mut total_sidelobes_pruned = 0usize;
    let tol_h = 0.055f64;
    let tol_n = 0.48f64;

    for draft in &mut hn_drafts {
        let strip_collect = |pks: &[&Peak]| -> Vec<(f64, i32, f64)> {
            let mut hits = Vec::new();
            for p in pks {
                let d = ((p.h - draft.h) / tol_h).hypot((p.n - draft.n) / tol_n);
                if d <= 1.0 {
                    if let Some(c) = p.c {
                        let w = p.height.map(f64::abs).unwrap_or(1.0e4).max(1.0)
                            * (-0.45 * d * d).exp();
                        hits.push((c, p.sign, w));
                    }
                }
            }
            hits.sort_by(|a, b| b.2.total_cmp(&a.2).then_with(|| a.0.total_cmp(&b.0)));
            hits
        };

        // (a) `HN(CA)CO` with doublet-preserving `F1(C')` NUS sidelobe pruning (`P2-13`)
        let raw_hncaco: Vec<(f64, f64)> = strip_collect(&hncaco_pks)
            .into_iter()
            .map(|(c, _, w)| (c, w))
            .collect();
        let hncaco_clean = if cfg.enable_nus_sidelobe_pruning {
            let (kept, pruned) = prune_f1_c_prime_nus_sidelobes(&raw_hncaco, 3);
            draft.f1_sidelobes_pruned += pruned;
            total_sidelobes_pruned += pruned;
            kept
        } else {
            raw_hncaco
        };
        let mut distinct_c_curr = Vec::new();
        let mut fallback_c_curr = Vec::new();
        for (c_val, _) in hncaco_clean {
            if draft.c_prev.is_some_and(|cp| (cp - c_val).abs() <= 0.12) {
                dedup_push_f64(&mut fallback_c_curr, c_val, 0.06, 2);
            } else {
                dedup_push_f64(&mut distinct_c_curr, c_val, 0.06, MAX_JOINT_CANDS);
            }
        }
        if !distinct_c_curr.is_empty() {
            draft.c_curr = distinct_c_curr;
        } else {
            draft.c_curr = fallback_c_curr;
        }

        // (b) `HNCACB`, `CBCACONH` (`HNCOCACB` alias), `HNCOCA`, `HNCA`
        //     Uses shared `idp_solver::select_hncacb_intra_and_backfill_prev` (`P2-3`).
        let hncacb_strip = strip_collect(&hncacb_pks);
        let cbcaconh_strip = strip_collect(&cbcaconh_pks);
        let hncoca_strip = strip_collect(&hncoca_pks);
        let hnca_strip = strip_collect(&hnca_pks);

        let hncacb_pos: Vec<(f64, f64)> = hncacb_strip
            .iter()
            .filter(|&&(c, s, _)| s >= 0 && (36.0..=72.0).contains(&c))
            .map(|&(c, _, w)| (c, w))
            .collect();
        let hncacb_neg: Vec<(f64, f64)> = hncacb_strip
            .iter()
            .filter(|&&(c, s, _)| s < 0 && (9.0..=76.0).contains(&c))
            .map(|&(c, _, w)| (c, w))
            .collect();

        for &(c, _, _) in &hncoca_strip {
            if (36.0..=72.0).contains(&c) {
                dedup_push_f64(&mut draft.ca_prev, c, 0.12, MAX_JOINT_CANDS);
            }
        }

        let mut hncocacb_all_c: Vec<f64> = Vec::with_capacity(cbcaconh_strip.len());
        for &(c, sgn, _) in &cbcaconh_strip {
            hncocacb_all_c.push(c);
            if cbcaconh_has_both_signs {
                if sgn >= 0 && (36.0..=72.0).contains(&c) {
                    dedup_push_f64(&mut draft.ca_prev, c, 0.12, MAX_JOINT_CANDS);
                } else if sgn < 0 && (9.0..=76.0).contains(&c) {
                    dedup_push_f64(&mut draft.cb_prev, c, 0.14, MAX_JOINT_CB_CANDS);
                }
            } else {
                let matches_pos = hncacb_pos.iter().any(|&(v, _)| (v - c).abs() <= 0.28);
                let matches_neg = hncacb_neg.iter().any(|&(v, _)| (v - c).abs() <= 0.30);
                match (matches_pos, matches_neg) {
                    (true, false) => dedup_push_f64(&mut draft.ca_prev, c, 0.12, MAX_JOINT_CANDS),
                    (false, true) => {
                        dedup_push_f64(&mut draft.cb_prev, c, 0.14, MAX_JOINT_CB_CANDS)
                    }
                    (true, true) => {
                        if (38.0..=68.0).contains(&c) {
                            dedup_push_f64(&mut draft.ca_prev, c, 0.12, MAX_JOINT_CANDS);
                        }
                        if (9.0..=76.0).contains(&c) {
                            dedup_push_f64(&mut draft.cb_prev, c, 0.14, MAX_JOINT_CB_CANDS);
                        }
                    }
                    (false, false) => {
                        if (39.0..=65.5).contains(&c) && draft.ca_prev.is_empty() {
                            dedup_push_f64(&mut draft.ca_prev, c, 0.12, MAX_JOINT_CANDS);
                        } else if (9.0..=76.0).contains(&c) {
                            dedup_push_f64(&mut draft.cb_prev, c, 0.14, MAX_JOINT_CB_CANDS);
                        }
                    }
                }
            }
        }

        let mut all_ca_obs: Vec<(f64, f64)> = hncacb_pos.clone();
        for &(c, _, w) in &hnca_strip {
            if (36.0..=72.0).contains(&c) {
                all_ca_obs.push((c, w));
            }
        }
        all_ca_obs.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.total_cmp(&b.0)));

        let ca_intra = select_hncacb_intra_and_backfill_prev(
            &all_ca_obs,
            &mut draft.ca_prev,
            &hncocacb_all_c,
            0.20,
        );
        for v in ca_intra {
            dedup_push_f64(&mut draft.ca_curr, v, 0.10, MAX_JOINT_CANDS);
        }

        let cb_intra = select_hncacb_intra_and_backfill_prev(
            &hncacb_neg,
            &mut draft.cb_prev,
            &hncocacb_all_c,
            0.24,
        );
        for v in cb_intra {
            dedup_push_f64(&mut draft.cb_curr, v, 0.12, MAX_JOINT_CB_CANDS);
        }
    }

    // 4. Sanitize & assemble `13C`-detected `CON` dipeptide systems
    let clean_c13 = sanitize_c13_dataset(&dataset.c13_dataset);
    let mut con_systems = register_and_build_con_dipeptide_systems(&clean_c13);

    for p in &clean_c13.strip_peaks {
        if let Some(r_idx) = p.anchor_root_idx {
            if r_idx < con_systems.len() {
                let sys = &mut con_systems[r_idx];
                match p.exp {
                    C13DetectedExp::Cccon3D => {
                        if (7.0..=76.0).contains(&p.c_indirect_ppm)
                            && !sys
                                .ca_prev
                                .iter()
                                .any(|&v| (v - p.c_indirect_ppm).abs() <= 0.20)
                            && !sys
                                .cb_prev
                                .iter()
                                .any(|&v| (v - p.c_indirect_ppm).abs() <= 0.22)
                        {
                            dedup_push_f64(
                                &mut sys.sidechain_c_prev,
                                p.c_indirect_ppm,
                                0.18,
                                MAX_JOINT_SC_CANDS,
                            );
                        }
                    }
                    C13DetectedExp::Cbcaco3D
                        if p.sign < 0 && (9.5..=76.0).contains(&p.c_indirect_ppm) =>
                    {
                        dedup_push_f64(
                            &mut sys.cb_prev,
                            p.c_indirect_ppm,
                            0.14,
                            MAX_JOINT_CB_CANDS,
                        );
                    }
                    _ => {}
                }
            }
        }
    }

    let has_c13_prev_cb = clean_c13
        .strip_peaks
        .iter()
        .any(|p| matches!(p.exp, C13DetectedExp::Cbcaco3D | C13DetectedExp::Cccon3D));

    // 5. Noise-scaled Connected-Component Hungarian 1-to-1 `HN ↔ CON` Fusion (`P2-1`)
    let (sh_c, sh_n) = estimate_hnco_to_con_drift(&hn_drafts, &con_systems);
    if sh_c.abs() > 0.02 || sh_n.abs() > 0.08 {
        warnings.push(format!(
            "Aligned HNCO-to-CON carrier offset (dC'={sh_c:+.3} ppm, dN={sh_n:+.3} ppm) before node fusion"
        ));
    }

    // Estimate empirical cross-spectrum `(C', N)` noise scatter via MAD of nearest-neighbor
    // residuals so noisy spectra scale the Hungarian matching gate instead of fragmenting nodes!
    let mut nn_res_c = Vec::new();
    let mut nn_res_n = Vec::new();
    for hd in &hn_drafts {
        let Some(hc_raw) = hd.c_prev else {
            continue;
        };
        let hc = hc_raw - sh_c;
        let hn = hd.n - sh_n;
        let mut best: Option<(f64, f64, f64)> = None;
        for cs in &con_systems {
            let dc = (hc - cs.c_prev).abs();
            let dn = (hn - cs.n_curr).abs();
            if dc <= 0.28 && dn <= 0.85 {
                let d2 = (dc / 0.08).powi(2) + (dn / 0.28).powi(2);
                if best.is_none_or(|b| d2 < b.0) {
                    best = Some((d2, dc, dn));
                }
            }
        }
        if let Some((_, dc, dn)) = best {
            nn_res_c.push(dc);
            nn_res_n.push(dn);
        }
    }
    let (sig_c_est, sig_n_est) = if nn_res_c.len() >= 8 {
        nn_res_c.sort_by(|a, b| a.total_cmp(b));
        nn_res_n.sort_by(|a, b| a.total_cmp(b));
        let mad_c = nn_res_c[nn_res_c.len() / 2];
        let mad_n = nn_res_n[nn_res_n.len() / 2];
        (
            (1.4826 * mad_c).clamp(0.025, 0.12),
            (1.4826 * mad_n).clamp(0.08, 0.35),
        )
    } else {
        (0.035, 0.12)
    };
    let gate_c = (3.4 * sig_c_est).clamp(0.10, 0.28);
    let gate_n = (3.4 * sig_n_est).clamp(0.36, 0.85);

    let mut bipartite_edges: Vec<(usize, usize, f64)> = Vec::new();
    for (hi, hd) in hn_drafts.iter().enumerate() {
        let Some(hc_raw) = hd.c_prev else {
            continue;
        };
        let hc = hc_raw - sh_c;
        let hn = hd.n - sh_n;
        for (ci, cs) in con_systems.iter().enumerate() {
            let dc = (hc - cs.c_prev).abs();
            let dn = (hn - cs.n_curr).abs();
            let d2_gate = (dc / gate_c).powi(2) + (dn / gate_n).powi(2);
            if d2_gate <= 1.0 {
                // Cost scaled by empirical `(sig_c_est, sig_n_est)` for optimal 1-to-1 assignment
                let cost = (dc / sig_c_est).powi(2) + (dn / sig_n_est).powi(2);
                bipartite_edges.push((hi, ci, cost));
            }
        }
    }

    let matched_pairs =
        match_hn_con_bipartite_hungarian(hn_drafts.len(), con_systems.len(), &bipartite_edges);

    let mut hn_used = vec![false; hn_drafts.len()];
    let mut con_used = vec![false; con_systems.len()];
    let mut nodes: Vec<JointIdpNode> = Vec::with_capacity(hn_drafts.len() + con_systems.len());
    let mut con_idx_to_node: Vec<Option<usize>> = vec![None; con_systems.len()];
    let mut sq_fus_c = 0.0f64;
    let mut sq_fus_n = 0.0f64;

    // (a) Fused `HN + CON` amide nodes (exact 1-to-1 Hungarian matches)
    for &(hi, ci, _) in &matched_pairs {
        hn_used[hi] = true;
        con_used[ci] = true;
        let hd = &hn_drafts[hi];
        let cs = &con_systems[ci];

        if let Some(hc_raw) = hd.c_prev {
            let dc = (hc_raw - sh_c) - cs.c_prev;
            let dn = (hd.n - sh_n) - cs.n_curr;
            sq_fus_c += dc * dc;
            sq_fus_n += dn * dn;
        }

        let c_shift = if sh_c.abs() > 0.50 { sh_c } else { 0.0 };
        let hn_ca_curr: Vec<f64> = hd.ca_curr.iter().map(|&v| v - c_shift).collect();
        let hn_cb_curr: Vec<f64> = hd.cb_curr.iter().map(|&v| v - c_shift).collect();
        let hn_ca_prev: Vec<f64> = hd.ca_prev.iter().map(|&v| v - c_shift).collect();
        let hn_cb_prev: Vec<f64> = hd.cb_prev.iter().map(|&v| v - c_shift).collect();
        let hn_c_curr: Vec<f64> = hd.c_curr.iter().map(|&v| v - sh_c).collect();

        let ca_prev =
            pool_corroborated_channels(&hn_ca_prev, &cs.ca_prev, 0.24, 0.10, MAX_JOINT_CANDS);
        let cb_prev =
            pool_corroborated_channels(&hn_cb_prev, &cs.cb_prev, 0.30, 0.12, MAX_JOINT_CB_CANDS);
        let mut ca_curr =
            pool_corroborated_channels(&hn_ca_curr, &cs.ca_curr, 0.24, 0.10, MAX_JOINT_CANDS);
        let mut cb_curr =
            pool_corroborated_channels(&hn_cb_curr, &cs.cb_curr, 0.30, 0.12, MAX_JOINT_CB_CANDS);
        let c_curr =
            pool_corroborated_channels(&hn_c_curr, &cs.c_curr, 0.14, 0.06, MAX_JOINT_CANDS);

        if !cs.ca_prev.is_empty() && ca_curr.len() > 1 {
            let filt: Vec<f64> = ca_curr
                .iter()
                .copied()
                .filter(|&v| !cs.ca_prev.iter().any(|&p| (p - v).abs() <= 0.16))
                .collect();
            if !filt.is_empty() {
                ca_curr = filt;
            }
        }
        if !cs.cb_prev.is_empty() && cb_curr.len() > 1 {
            let filt: Vec<f64> = cb_curr
                .iter()
                .copied()
                .filter(|&v| !cs.cb_prev.iter().any(|&p| (p - v).abs() <= 0.20))
                .collect();
            if !filt.is_empty() {
                cb_curr = filt;
            }
        }

        let node_id = nodes.len();
        con_idx_to_node[ci] = Some(node_id);
        nodes.push(JointIdpNode {
            id: node_id,
            kind: JointNodeKind::AmideFused,
            hn_matched: true,
            con_matched: true,
            h_curr: Some(hd.h),
            n_curr: cs.n_curr,
            hn_n_curr: Some(hd.n),
            c_prev: Some(cs.c_prev),
            root_height: cs.root_height.max(hd.root_height),
            ca_curr,
            cb_curr,
            ca_prev,
            cb_prev,
            c_curr,
            sidechain_c_prev: cs.sidechain_c_prev.clone(),
            nn_prev_cands: cs.nn_prev_cands.clone(),
            recip_nn_partners: Vec::new(),
            prev_gly_opposite_phase: cs.prev_gly_opposite_phase,
            has_prev_cb_exp: hd.has_prev_cb_exp || cs.has_cbcacon_evidence || has_c13_prev_cb,
            has_curr_cb_exp: hd.has_curr_cb_exp,
            gly_cb_curr_suppressed: false,
            gly_cb_prev_suppressed: false,
            f1_sidelobes_pruned: hd.f1_sidelobes_pruned,
            registration_warnings: cs.registration_warnings.clone(),
        });
    }

    let n_matched_fus = matched_pairs.len();
    let fusion_rms_c_ppm = if n_matched_fus > 0 {
        (sq_fus_c / (n_matched_fus as f64)).sqrt()
    } else {
        0.0
    };
    let fusion_rms_n_ppm = if n_matched_fus > 0 {
        (sq_fus_n / (n_matched_fus as f64)).sqrt()
    } else {
        0.0
    };
    let n_hn_with_cprev = hn_drafts.iter().filter(|d| d.c_prev.is_some()).count();
    let hn_unmatched_frac = if n_hn_with_cprev > 0 && !con_systems.is_empty() {
        (n_hn_with_cprev.saturating_sub(n_matched_fus) as f64) / (n_hn_with_cprev as f64)
    } else {
        0.0
    };

    // (b) Unmatched `HN` amide roots (`HN`-only or `CON`-dropped)
    let c_shift_unmatched = if sh_c.abs() > 0.50 { sh_c } else { 0.0 };
    for (hi, hd) in hn_drafts.iter().enumerate() {
        if hn_used[hi] {
            continue;
        }
        let node_id = nodes.len();
        nodes.push(JointIdpNode {
            id: node_id,
            kind: JointNodeKind::AmideFused,
            hn_matched: true,
            con_matched: false,
            h_curr: Some(hd.h),
            n_curr: hd.n - sh_n,
            hn_n_curr: Some(hd.n),
            c_prev: hd.c_prev.map(|c| c - sh_c),
            root_height: hd.root_height,
            ca_curr: hd.ca_curr.iter().map(|&v| v - c_shift_unmatched).collect(),
            cb_curr: hd.cb_curr.iter().map(|&v| v - c_shift_unmatched).collect(),
            ca_prev: hd.ca_prev.iter().map(|&v| v - c_shift_unmatched).collect(),
            cb_prev: hd.cb_prev.iter().map(|&v| v - c_shift_unmatched).collect(),
            c_curr: hd.c_curr.iter().map(|&v| v - sh_c).collect(),
            sidechain_c_prev: Vec::new(),
            nn_prev_cands: Vec::new(),
            recip_nn_partners: Vec::new(),
            prev_gly_opposite_phase: false,
            has_prev_cb_exp: hd.has_prev_cb_exp,
            has_curr_cb_exp: hd.has_curr_cb_exp,
            gly_cb_curr_suppressed: false,
            gly_cb_prev_suppressed: false,
            f1_sidelobes_pruned: hd.f1_sidelobes_pruned,
            registration_warnings: Vec::new(),
        });
    }

    // (c) Unmatched `2D CON` roots -> provisionally `ProlineConBridge` vs `ConOnlyAmide`
    for (ci, cs) in con_systems.iter().enumerate() {
        if con_used[ci] {
            continue;
        }
        let node_id = nodes.len();
        con_idx_to_node[ci] = Some(node_id);
        let provisional_kind = if cs.n_curr >= PROLINE_N15_MIN_PPM {
            JointNodeKind::ProlineConBridge
        } else {
            JointNodeKind::ConOnlyAmide
        };
        nodes.push(JointIdpNode {
            id: node_id,
            kind: provisional_kind,
            hn_matched: false,
            con_matched: true,
            h_curr: None,
            n_curr: cs.n_curr,
            hn_n_curr: None,
            c_prev: Some(cs.c_prev),
            root_height: cs.root_height,
            ca_curr: cs.ca_curr.clone(),
            cb_curr: cs.cb_curr.clone(),
            ca_prev: cs.ca_prev.clone(),
            cb_prev: cs.cb_prev.clone(),
            c_curr: cs.c_curr.clone(),
            sidechain_c_prev: cs.sidechain_c_prev.clone(),
            nn_prev_cands: cs.nn_prev_cands.clone(),
            recip_nn_partners: Vec::new(),
            prev_gly_opposite_phase: cs.prev_gly_opposite_phase,
            has_prev_cb_exp: cs.has_cbcacon_evidence || has_c13_prev_cb,
            has_curr_cb_exp: false,
            gly_cb_curr_suppressed: false,
            gly_cb_prev_suppressed: false,
            f1_sidelobes_pruned: 0,
            registration_warnings: cs.registration_warnings.clone(),
        });
    }

    // 6. Attach directional `(H)N(CA)CON` sequential `N(i-1)` edges onto matching nodes
    let mut n_directional_nn_edges = 0usize;
    for edge in &dataset.hncacon_nn_edges {
        if !(edge.c_prev_ppm.is_finite()
            && edge.n_curr_ppm.is_finite()
            && edge.n_prev_ppm.is_finite()
            && edge.snr.is_finite())
        {
            continue;
        }
        let mut best_node: Option<(usize, f64)> = None;
        for (s_idx, node) in nodes.iter().enumerate() {
            let Some(cp) = node.c_prev else {
                continue;
            };
            let d2 = ((edge.c_prev_ppm - cp) / 0.14).powi(2)
                + ((edge.n_curr_ppm - node.n_curr) / 0.48).powi(2);
            if d2 <= 1.0 && best_node.is_none_or(|b| d2 < b.1) {
                best_node = Some((s_idx, d2));
            }
        }
        if let Some((s_idx, _)) = best_node {
            let snr = edge.snr.abs().max(2.0);
            if !nodes[s_idx]
                .nn_prev_cands
                .iter()
                .any(|&(np, _)| (np - edge.n_prev_ppm).abs() <= 0.25)
            {
                nodes[s_idx].nn_prev_cands.push((edge.n_prev_ppm, snr));
                n_directional_nn_edges += 1;
            }
        }
    }
    for node in &mut nodes {
        node.nn_prev_cands
            .sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.total_cmp(&b.0)));
        node.nn_prev_cands.truncate(4);
    }

    // 7. Field-scaled TROSY `46 Hz (1/2 1J_NH)` calibration & reciprocal `BT_HNCACONH` `N–N` edges
    //    Retains partner-checked near-diagonal `0.0..=0.35 ppm` links with smooth down-weight (`P2-2`)!
    let amide_hn_roots: Vec<(usize, f64, f64)> = nodes
        .iter()
        .filter_map(|n| Some((n.id, n.h_curr?, n.hn_n_curr.unwrap_or(n.n_curr))))
        .collect();
    let decoupled_pairs: Vec<(f64, f64)> = amide_hn_roots.iter().map(|&(_, h, n)| (h, n)).collect();

    let clean_hncaconh: Vec<&HncaconhPeak> = dataset
        .hncaconh_peaks
        .iter()
        .filter(|p| {
            p.h_ppm.is_finite()
                && p.n_curr_ppm.is_finite()
                && p.n_partner_ppm.is_finite()
                && p.height.is_finite()
        })
        .collect();

    let mut trosy_unique_roots: Vec<(f64, f64)> = Vec::new();
    for p in &clean_hncaconh {
        if !trosy_unique_roots
            .iter()
            .any(|&(h, n)| (h - p.h_ppm).abs() <= 0.025 && (n - p.n_curr_ppm).abs() <= 0.25)
        {
            trosy_unique_roots.push((p.h_ppm, p.n_curr_ppm));
        }
    }

    let trosy_calibration = if cfg.auto_trosy_calibration && !trosy_unique_roots.is_empty() {
        calibrate_trosy_hn_offset(&trosy_unique_roots, &decoupled_pairs, proton_mhz)
    } else {
        let (eh, en) = trosy_expected_shift_ppm(proton_mhz);
        TrosyOffsetCalibration {
            expected_half_j_h_ppm: eh.abs(),
            expected_half_j_n_ppm: en.abs(),
            ..TrosyOffsetCalibration::default()
        }
    };

    let partner_dn = if trosy_calibration.trosy_detected {
        let count_partner_hits = |shift_n: f64| -> usize {
            clean_hncaconh
                .iter()
                .filter(|p| {
                    amide_hn_roots
                        .iter()
                        .any(|&(_, _, rn)| ((p.n_partner_ppm - shift_n) - rn).abs() <= 0.30)
                })
                .count()
        };
        let hits_raw = count_partner_hits(0.0);
        let hits_shifted = count_partner_hits(trosy_calibration.dn_ppm);
        if hits_shifted > hits_raw + 2 {
            trosy_calibration.dn_ppm
        } else {
            0.0
        }
    } else {
        0.0
    };

    // Extract directed `(node_s, partner_n, diag_weight)` observations without hard-excluding
    // near-diagonal Ser-Ser / Gln-Gln `0.0..=0.35 ppm` links (`P2-2`)!
    let mut directed_hncaconh: Vec<(usize, f64, f64)> = Vec::new();
    for p in &clean_hncaconh {
        let h_corr = p.h_ppm - trosy_calibration.dh_ppm;
        let n_corr = p.n_curr_ppm - trosy_calibration.dn_ppm;
        let n_part = p.n_partner_ppm - partner_dn;

        let mut best_root: Option<(usize, f64)> = None;
        for &(nid, rh, rn) in &amide_hn_roots {
            let d2 = ((h_corr - rh) / 0.05).powi(2) + ((n_corr - rn) / 0.45).powi(2);
            if d2 <= 1.0 && best_root.is_none_or(|b| d2 < b.1) {
                best_root = Some((nid, d2));
            }
        }
        let Some((nid, _)) = best_root else {
            continue;
        };

        let dn_diag = (n_part - n_corr).abs();
        // Smooth diagonal down-weighting for near-diagonal `0.0..=0.35 ppm` links (`P2-2`),
        // gated strictly on matching a DISTINCT amide root `t != nid` below!
        let w_diag = if dn_diag < 0.35 {
            0.45 + 0.55 * (dn_diag / 0.35).clamp(0.0, 1.0)
        } else {
            1.0
        };
        directed_hncaconh.push((nid, n_part, w_diag));
    }

    let mut recip_edge_weights: BTreeMap<(usize, usize), f64> = BTreeMap::new();
    for &(s_idx, n_part_s, ws) in &directed_hncaconh {
        let ns_hn = nodes[s_idx].hn_n_curr.unwrap_or(nodes[s_idx].n_curr);
        for &(t_idx, n_part_t, wt) in &directed_hncaconh {
            if s_idx == t_idx {
                continue;
            }
            let nt_hn = nodes[t_idx].hn_n_curr.unwrap_or(nodes[t_idx].n_curr);
            if (n_part_s - nt_hn).abs() <= 0.32 && (n_part_t - ns_hn).abs() <= 0.32 {
                let (a, b) = if s_idx < t_idx {
                    (s_idx, t_idx)
                } else {
                    (t_idx, s_idx)
                };
                let w = ws.min(wt);
                let entry = recip_edge_weights.entry((a, b)).or_insert(0.0);
                if w > *entry {
                    *entry = w;
                }
            }
        }
    }

    for &(na, nb) in &dataset.reciprocal_nn_pairs {
        if !(na.is_finite() && nb.is_finite()) {
            continue;
        }
        let dn_diag = (na - nb).abs();
        let w_diag = if dn_diag < 0.35 {
            0.45 + 0.55 * (dn_diag / 0.35).clamp(0.0, 1.0)
        } else {
            1.0
        };
        let matching_a: Vec<usize> = nodes
            .iter()
            .filter(|n| (n.n_curr - na).abs() <= 0.25)
            .map(|n| n.id)
            .collect();
        let matching_b: Vec<usize> = nodes
            .iter()
            .filter(|n| (n.n_curr - nb).abs() <= 0.25)
            .map(|n| n.id)
            .collect();
        if matching_a.len() <= 2 && matching_b.len() <= 2 {
            for &sa in &matching_a {
                for &sb in &matching_b {
                    if sa != sb {
                        let (a, b) = if sa < sb { (sa, sb) } else { (sb, sa) };
                        let entry = recip_edge_weights.entry((a, b)).or_insert(0.0);
                        if w_diag > *entry {
                            *entry = w_diag;
                        }
                    }
                }
            }
        }
    }

    for (&(a, b), &w) in &recip_edge_weights {
        nodes[a].recip_nn_partners.push((b, w));
        nodes[b].recip_nn_partners.push((a, w));
    }
    let n_reciprocal_nn_edges = recip_edge_weights.len();

    // 8. Unlabelled global reference offset estimation (DSS/TMS, shared with `idp_solver`)
    let (fitted_offsets, auto_ref_report) = if cfg.auto_reference_offset {
        estimate_joint_reference_offsets(&nodes, sequence, cfg)
    } else {
        (cfg.manual_offsets, IdpAutoRefSafetyReport::default())
    };
    warnings.extend(auto_ref_report.warnings.clone());
    if let Some(ref tms) = auto_ref_report.tms_reference_hint {
        warnings.push(tms.clone());
    }

    // 9. Finalize `ProlineConBridge` vs `ConOnlyAmide` classification in DSS frame (`131.5..=144.5 ppm`)
    for node in &mut nodes {
        if !node.hn_matched && node.con_matched {
            let n_dss = node.n_curr - fitted_offsets.d_n;
            node.kind = if (PROLINE_N15_MIN_PPM..=PROLINE_N15_MAX_PPM).contains(&n_dss) {
                JointNodeKind::ProlineConBridge
            } else {
                JointNodeKind::ConOnlyAmide
            };
        }
    }

    // 10. Apply DSS-frame `Gly CB` bleed-through guard (`CA ∈ [41.0, 46.5] ppm` DSS)
    let n_gly_cb_guarded = if cfg.enable_gly_cb_guard {
        apply_gly_cb_bleedthrough_guard(&mut nodes, &fitted_offsets)
    } else {
        0
    };

    // 11. Detect candidate C-terminal `COO-` (`~179 ppm` DSS) nodes
    let mut c_term_coo_candidates = Vec::new();
    if cfg.enable_cterm_coo_prior {
        for node in &nodes {
            let has_coo = node.c_curr.iter().any(|&c| {
                let c_dss = c - fitted_offsets.d_c;
                (CTERM_COO_DSS_MIN_PPM..=CTERM_COO_DSS_MAX_PPM).contains(&c_dss)
            });
            if has_coo {
                c_term_coo_candidates.push(node.id);
            }
        }
    }

    let n_fused_hn_con = nodes
        .iter()
        .filter(|n| n.kind == JointNodeKind::AmideFused && n.hn_matched && n.con_matched)
        .count();
    let n_hn_only = nodes
        .iter()
        .filter(|n| n.kind == JointNodeKind::AmideFused && n.hn_matched && !n.con_matched)
        .count();
    let n_proline_bridges = nodes
        .iter()
        .filter(|n| n.kind == JointNodeKind::ProlineConBridge)
        .count();
    let n_con_only_amide = nodes
        .iter()
        .filter(|n| n.kind == JointNodeKind::ConOnlyAmide)
        .count();

    JointNodeBuildReport {
        nodes,
        fitted_offsets,
        auto_ref_report,
        trosy_calibration,
        polarity_report,
        hnco_to_con_drift_ppm: (sh_c, sh_n),
        fusion_rms_c_ppm,
        fusion_rms_n_ppm,
        hn_unmatched_frac,
        n_fused_hn_con,
        n_hn_only,
        n_proline_bridges,
        n_con_only_amide,
        n_sidelobes_pruned: total_sidelobes_pruned,
        n_gly_cb_guarded,
        n_reciprocal_nn_edges,
        n_directional_nn_edges,
        c_term_coo_candidates,
        warnings,
    }
}

// ============================================================================
// Joint Unary Matrix `U(s, k)` & Joint Pairwise Matrix `P(s -> t)`
// ============================================================================

#[inline]
fn logpdf_2comp(x: f64, mu: f64, sigma: f64) -> f64 {
    let s1 = sigma.max(0.05);
    let s2 = (s1 * 3.8).max(0.25);
    let inv_s2pi = 1.0 / TAU.sqrt();
    let z1 = (x - mu) / s1;
    let z2 = (x - mu) / s2;
    let p1 = 0.78 * inv_s2pi / s1 * (-0.5 * z1 * z1).exp();
    let p2 = 0.22 * inv_s2pi / s2 * (-0.5 * z2 * z2).exp();
    (p1 + p2).max(1e-30).ln()
}

#[inline]
fn bg_logpdf(x: f64, bg_mus: &[f64], bg_sigma: f64) -> f64 {
    let s = bg_sigma.max(0.1);
    let inv_n = 1.0 / (bg_mus.len().max(1) as f64);
    let inv_s2pi = 1.0 / TAU.sqrt();
    let mut sum = 0.0f64;
    for &mu in bg_mus {
        let z = (x - mu) / s;
        if z.abs() < 10.0 {
            sum += inv_n * inv_s2pi / s * (-0.5 * z * z).exp();
        }
    }
    sum.max(1e-30).ln()
}

#[inline]
fn best_cand_llr(
    cands: &[f64],
    mu: f64,
    sigma: f64,
    bg_mus: &[f64],
    bg_sigma: f64,
    llr_floor: f64,
) -> f64 {
    if cands.is_empty() {
        return 0.0;
    }
    let mut best = f64::NEG_INFINITY;
    for (idx, &v) in cands.iter().take(MAX_JOINT_CANDS).enumerate() {
        let llr = logpdf_2comp(v, mu, sigma) - bg_logpdf(v, bg_mus, bg_sigma) - 0.12 * (idx as f64);
        if llr > best {
            best = llr;
        }
    }
    best.max(llr_floor)
}

/// Fallible joint unary scoring matrix `U` `(S x L)` enforcing [`MAX_IDP_MATRIX_CELLS`].
pub fn try_joint_idp_unary_matrix(
    nodes: &[JointIdpNode],
    sequence: &[String],
    cas_tables: &[CalibratedCasTable],
    cfg: &JointIdpConfig,
    offsets: &IdpShiftOffsets,
    user_locks: &[(usize, usize)],
) -> Result<Matrix2D, String> {
    check_idp_matrix_bounds(nodes.len(), sequence.len())?;
    Ok(joint_idp_unary_matrix(
        nodes, sequence, cas_tables, cfg, offsets, user_locks,
    ))
}

/// Compute unified joint IDP unary scoring matrix `U` `(S x L)`.
pub fn joint_idp_unary_matrix(
    nodes: &[JointIdpNode],
    sequence: &[String],
    cas_tables: &[CalibratedCasTable],
    cfg: &JointIdpConfig,
    offsets: &IdpShiftOffsets,
    user_locks: &[(usize, usize)],
) -> Matrix2D {
    let s_len = nodes.len();
    let seq = normalize_seq_3(sequence);
    let k_len = seq.len();
    if check_idp_matrix_bounds(s_len, k_len).is_err() {
        return Matrix2D::filled(0, 0, FORBID);
    }
    let mut u = Matrix2D::filled(s_len, k_len, FORBID);
    if s_len == 0 || k_len < 2 {
        return u;
    }

    let preds = predict_joint_ncidp_shifts(&seq, cfg);
    let dn = offsets.d_n;
    let dc = offsets.d_c;
    let dca = offsets.d_ca;
    let dcb = offsets.d_cb;
    let llr_floor = cfg.llr_floor;

    let n_with_3d = nodes.iter().filter(|n| n.has_3d_support()).count();
    let multi_3d_dataset = n_with_3d * 2 >= s_len;

    let bg_n: Vec<f64> = IDP_BG_N_MUS.iter().map(|&m| (m as f64) + dn).collect();
    let bg_c: Vec<f64> = IDP_BG_C_MUS.iter().map(|&m| (m as f64) + dc).collect();
    let bg_ca: Vec<f64> = IDP_BG_CA_MUS.iter().map(|&m| (m as f64) + dca).collect();
    let bg_cb: Vec<f64> = IDP_BG_CB_MUS.iter().map(|&m| (m as f64) + dcb).collect();
    let bg_sc: Vec<f64> = IDP_BG_SC_MUS.iter().map(|&m| (m as f64) + dcb).collect();

    for (s_idx, node) in nodes.iter().enumerate() {
        let n_dss = node.n_curr - dn;
        let has_3d = node.has_3d_support();

        let sys_assign_bonus = match node.kind {
            JointNodeKind::ProlineConBridge => {
                if has_3d {
                    2.85
                } else {
                    1.45
                }
            }
            JointNodeKind::AmideFused => {
                if node.con_matched && has_3d {
                    3.05
                } else if has_3d || node.con_matched {
                    2.45
                } else if multi_3d_dataset {
                    -6.0
                } else {
                    0.5
                }
            }
            JointNodeKind::ConOnlyAmide => {
                if has_3d {
                    2.65
                } else if multi_3d_dataset {
                    -6.2
                } else {
                    0.5
                }
            }
        };

        let has_cterm_coo_peak = cfg.enable_cterm_coo_prior
            && node.c_curr.iter().any(|&c| {
                let c_dss = c - dc;
                (CTERM_COO_DSS_MIN_PPM..=CTERM_COO_DSS_MAX_PPM).contains(&c_dss)
            });

        for k in 1..k_len {
            let cur_aa = seq[k].as_str();
            let is_pro = cur_aa == "PRO";

            match node.kind {
                JointNodeKind::AmideFused => {
                    if is_pro {
                        continue;
                    }
                }
                JointNodeKind::ProlineConBridge => {
                    if !is_pro && n_dss >= 132.5 {
                        continue;
                    }
                }
                JointNodeKind::ConOnlyAmide => {
                    if is_pro && n_dss < 131.0 {
                        continue;
                    }
                }
            }

            let cur_p = &preds[k];
            let prv_p = &preds[k - 1];
            let mut total = sys_assign_bonus;

            if node.kind == JointNodeKind::ProlineConBridge {
                if is_pro {
                    total += 1.15;
                } else {
                    total -= 4.80;
                }
            }

            // 1. Root `15N(k)` & `13C'(k-1)`
            total += (logpdf_2comp(node.n_curr, cur_p.n_ppm + dn, cur_p.sigma_n)
                - bg_logpdf(node.n_curr, &bg_n, 1.4))
            .max(llr_floor);

            if let Some(c_prev) = node.c_prev {
                total += (logpdf_2comp(c_prev, prv_p.c_ppm + dc, prv_p.sigma_c)
                    - bg_logpdf(c_prev, &bg_c, 0.7))
                .max(llr_floor);
            }

            // 2. `CA(k)` & `CA(k-1)`
            total += best_cand_llr(
                &node.ca_curr,
                cur_p.ca_ppm + dca,
                cur_p.sigma_ca,
                &bg_ca,
                1.2,
                llr_floor,
            );
            total += best_cand_llr(
                &node.ca_prev,
                prv_p.ca_ppm + dca,
                prv_p.sigma_ca,
                &bg_ca,
                1.2,
                llr_floor,
            );

            // 3. `CB(k-1)` and `Gly(k-1)` signatures
            if let Some(mu_cb_prev) = prv_p.cb_ppm {
                if !node.cb_prev.is_empty() {
                    total += best_cand_llr(
                        &node.cb_prev,
                        mu_cb_prev + dcb,
                        prv_p.sigma_cb,
                        &bg_cb,
                        2.5,
                        llr_floor,
                    );
                } else if node.has_prev_cb_exp && !node.gly_cb_prev_suppressed {
                    total -= 1.35;
                }
                if node.prev_gly_opposite_phase {
                    total -= 4.50;
                }
            } else {
                if !node.cb_prev.is_empty() {
                    let ca_in_gly_band = node.ca_prev.first().is_some_and(|&ca| {
                        let ca_dss = ca - dca;
                        (GLY_CA_DSS_MIN_PPM..=GLY_CA_DSS_MAX_PPM).contains(&ca_dss)
                    });
                    total -= if ca_in_gly_band { 1.60 } else { 4.20 };
                } else if node.has_prev_cb_exp && !node.ca_prev.is_empty() {
                    total += 2.15;
                }
                if node.prev_gly_opposite_phase {
                    total += 3.60;
                }
            }

            // 4. `CB(k)` and `Gly(k)` signatures
            if let Some(mu_cb_cur) = cur_p.cb_ppm {
                if !node.cb_curr.is_empty() {
                    total += best_cand_llr(
                        &node.cb_curr,
                        mu_cb_cur + dcb,
                        cur_p.sigma_cb,
                        &bg_cb,
                        2.5,
                        llr_floor,
                    );
                } else if node.has_curr_cb_exp && !node.gly_cb_curr_suppressed {
                    total -= 1.35;
                }
            } else if !node.cb_curr.is_empty() {
                let ca_in_gly_band = node.ca_curr.first().is_some_and(|&ca| {
                    let ca_dss = ca - dca;
                    (GLY_CA_DSS_MIN_PPM..=GLY_CA_DSS_MAX_PPM).contains(&ca_dss)
                });
                total -= if ca_in_gly_band { 1.60 } else { 4.20 };
            } else if node.has_curr_cb_exp && !node.ca_curr.is_empty() {
                total += 2.15;
            }

            // 5. Intraresidue `C'(k)` (`HN(CA)CO` / `NCACO` / `CACO`) + Platzer C-term `COO-` prior
            if !node.c_curr.is_empty() {
                total += 0.90
                    * best_cand_llr(
                        &node.c_curr,
                        cur_p.c_ppm + dc,
                        cur_p.sigma_c,
                        &bg_c,
                        0.7,
                        llr_floor,
                    );
                if cfg.enable_cterm_coo_prior {
                    let is_c_term = k + 1 == k_len;
                    if is_c_term && has_cterm_coo_peak {
                        let best_coo_z = node
                            .c_curr
                            .iter()
                            .map(|&c| ((c - dc) - cur_p.c_ppm).abs() / 0.85)
                            .fold(f64::INFINITY, f64::min);
                        if best_coo_z <= 2.5 {
                            total += 2.40 * (-0.25 * best_coo_z * best_coo_z).exp();
                        }
                    } else if !is_c_term && has_cterm_coo_peak {
                        let extreme_coo = node.c_curr.iter().all(|&c| (c - dc) >= 178.4);
                        if extreme_coo {
                            total -= 2.20;
                        }
                    }
                }
            }

            // 6. Aliphatic sidechain `H(CC)CON` / `CCC(CO)N` peaks of `k - 1`
            if !node.sidechain_c_prev.is_empty() {
                if prv_p.sidechain_c.is_empty() {
                    total -= 1.10 * (node.sidechain_c_prev.len().min(3) as f64);
                } else {
                    let exp_sc: Vec<f64> =
                        prv_p.sidechain_c.iter().take(4).map(|&m| m + dcb).collect();
                    let mut sc_sum = 0.0f64;
                    for &v in node.sidechain_c_prev.iter().take(MAX_JOINT_SC_CANDS) {
                        let mut best_comp = f64::NEG_INFINITY;
                        for &mu_s in &exp_sc {
                            let lp = logpdf_2comp(v, mu_s, 0.75);
                            if lp > best_comp {
                                best_comp = lp;
                            }
                        }
                        let llr = (best_comp - bg_logpdf(v, &bg_sc, 1.8)).max(-4.0);
                        sc_sum += 0.55 * llr;
                    }
                    total += sc_sum;
                }
            }

            // 7. Overlap- and breakthrough-calibrated `CAS` (`cacon` on `k-1`, `canco` on `k`)
            for tab in cas_tables {
                let target_res = match tab.mode {
                    CasMode::Cacon => seq[k - 1].as_str(),
                    CasMode::Canco => cur_aa,
                };
                total += tab.llr_for(s_idx, target_res);
            }

            u.set(s_idx, k, if total.is_finite() { total } else { llr_floor });
        }
    }

    for &(s_lock, k_lock) in user_locks {
        if s_lock < s_len && k_lock >= 1 && k_lock < k_len && u.get(s_lock, k_lock) > FORBID * 0.5 {
            for k in 0..k_len {
                if k != k_lock {
                    u.set(s_lock, k, FORBID);
                }
            }
            for s in 0..s_len {
                if s != s_lock {
                    u.set(s, k_lock, FORBID);
                }
            }
        }
    }

    u
}

#[inline]
fn channel_pair_llr(
    a_cands: &[f64],
    b_cands: &[f64],
    sigma: f64,
    range_ppm: f64,
    pair_floor: f64,
) -> f64 {
    let na = a_cands.len().min(MAX_JOINT_CANDS);
    let nb = b_cands.len().min(MAX_JOINT_CANDS);
    if na == 0 || nb == 0 {
        return 0.0;
    }
    let inv_sqrt_2pi = 1.0 / TAU.sqrt();
    let s = sigma.max(0.02);
    let bg_log = (1.0 / range_ppm.max(2.0)).ln();
    let norm_log = (inv_sqrt_2pi / s).ln();
    let mut best = f64::NEG_INFINITY;
    for (ia, &va) in a_cands[..na].iter().enumerate() {
        for (ib, &vb) in b_cands[..nb].iter().enumerate() {
            let z = (va - vb) / s;
            let lp = norm_log - 0.5 * z * z - 0.14 * ((ia + ib) as f64);
            let llr = (lp - bg_log).max(pair_floor);
            if llr > best {
                best = llr;
            }
        }
    }
    let mult_gate = (best / 1.5).clamp(0.0, 1.0);
    best - 0.18 * ((na * nb) as f64).ln() * mult_gate
}

#[inline]
fn pair_compat_weight(raw: f64) -> f64 {
    let u = ((raw - 2.10) / 2.40).clamp(0.0, 1.0);
    u * u * (3.0 - 2.0 * u)
}

/// Fallible unified pairwise sequential matrix `P` `(S x S)` enforcing [`MAX_IDP_MATRIX_CELLS`].
pub fn try_joint_idp_pair_matrix(
    nodes: &[JointIdpNode],
    cfg: &JointIdpConfig,
) -> Result<Matrix2D, String> {
    check_idp_matrix_bounds(nodes.len(), 2)?;
    Ok(joint_idp_pair_matrix(nodes, cfg))
}

/// Compute unified pairwise sequential transition matrix `P(s -> t)` `(S x S)` combining:
/// 1. **`ΔC'` across ALL residues including Prolines** (`s.c_curr` matching `t.c_prev`),
/// 2. **`ΔCA` and `ΔCB`** (`s.ca_curr <-> t.ca_prev`, `s.cb_curr <-> t.cb_prev`),
/// 3. **Directional `(H)N(CA)CON` `N–N` edges** (`t.nn_prev_cands` matching `s.n_curr`, including
///    near-diagonal `0.0..=0.35 ppm` links with smooth down-weight),
/// 4. **Symmetric reciprocal `BT_HNCACONH` `N–N` edges** (post-TROSY calibration, `s <-> t`
///    symmetric bonus disambiguated directionally by `ΔC'`, `ΔCA`, `ΔCB`, and `(H)N(CA)CON`),
/// 5. **Smooth bidirectional candidate-multiplicity ambiguity damping** (`w_mult`).
pub fn joint_idp_pair_matrix(nodes: &[JointIdpNode], cfg: &JointIdpConfig) -> Matrix2D {
    let s_len = nodes.len();
    if check_idp_matrix_bounds(s_len, 2).is_err() {
        return Matrix2D::filled(0, 0, 0.0);
    }
    let mut p = Matrix2D::filled(s_len, s_len, 0.0);
    if s_len == 0 {
        return p;
    }

    let sig_ca = cfg.sigma_seq_ca.max(0.02);
    let sig_cb = cfg.sigma_seq_cb.max(0.03);
    let sig_c = cfg.sigma_seq_c.max(0.02);
    let p_floor = cfg.pair_floor;
    let w_mult = cfg.w_mult.max(0.0);

    let mut raw_mat = vec![0.0f64; s_len * s_len];
    for s in 0..s_len {
        let ns = &nodes[s];
        for t in 0..s_len {
            if s == t {
                continue;
            }
            let nt = &nodes[t];

            let sc_ca = channel_pair_llr(&ns.ca_curr, &nt.ca_prev, sig_ca, 28.0, p_floor);
            let sc_cb = channel_pair_llr(&ns.cb_curr, &nt.cb_prev, sig_cb, 56.0, p_floor);
            let sc_c = if let Some(c_prev_t) = nt.c_prev {
                channel_pair_llr(&ns.c_curr, &[c_prev_t], sig_c, 13.0, p_floor)
            } else {
                0.0
            };

            // Directional `(H)N(CA)CON` sequential `N(i-1)` channel (including smooth near-diagonal
            // `0.0..=0.35 ppm` down-weighting consistent with `idp_solver.rs` v0.17.3 R2).
            let sc_nn_dir = if !nt.nn_prev_cands.is_empty() {
                let mut best_nn: Option<f64> = None;
                for &(cand_n, snr) in &nt.nn_prev_cands {
                    let dn = (cand_n - ns.n_curr).abs();
                    if dn < 0.30 {
                        let sep_diag = (cand_n - nt.n_curr).abs();
                        let diag_w = if sep_diag < 0.35 {
                            0.45 + 0.55 * (sep_diag / 0.35).clamp(0.0, 1.0)
                        } else {
                            1.0
                        };
                        let bonus =
                            diag_w * (2.1 + (snr.clamp(3.0, 12.0) / 4.0) * (1.0 - dn / 0.38));
                        if best_nn.is_none_or(|b| bonus > b) {
                            best_nn = Some(bonus);
                        }
                    }
                }
                best_nn.unwrap_or(-0.55)
            } else {
                0.0
            };

            // Symmetric reciprocal `BT_HNCACONH` `N–N` adjacency bonus (`P(s,t)` and `P(t,s)`)
            let sc_nn_recip = ns
                .recip_nn_partners
                .iter()
                .find(|&&(partner, _)| partner == t)
                .map(|&(_, w)| 2.10 * w.clamp(0.35, 1.5))
                .unwrap_or(0.0);

            let total = sc_ca + sc_cb + sc_c + sc_nn_dir + sc_nn_recip;
            raw_mat[s * s_len + t] = if total.is_finite() { total } else { 0.0 };
        }
    }

    let mut n_out = vec![1.0f64; s_len];
    let mut n_in = vec![1.0f64; s_len];
    if w_mult > 0.0 {
        for s in 0..s_len {
            let mut sum_out = 0.0f64;
            let mut sum_in = 0.0f64;
            for u in 0..s_len {
                if u != s {
                    sum_out += pair_compat_weight(raw_mat[s * s_len + u]);
                    sum_in += pair_compat_weight(raw_mat[u * s_len + s]);
                }
            }
            n_out[s] = sum_out.max(1.0);
            n_in[s] = sum_in.max(1.0);
        }
    }

    for s in 0..s_len {
        for t in 0..s_len {
            if s == t {
                p.set(s, t, FORBID);
                continue;
            }
            let raw_st = raw_mat[s * s_len + t];
            let final_st = if raw_st > 0.0 && w_mult > 0.0 {
                let gate_mult = (raw_st / 2.0).clamp(0.0, 1.0);
                raw_st - gate_mult * w_mult * 0.5 * (n_out[s].ln() + n_in[t].ln())
            } else {
                raw_st
            };
            p.set(s, t, if final_st.is_finite() { final_st } else { 0.0 });
        }
    }

    p
}

// ============================================================================
// Multi-Stage Island SA Ensemble + Conservative 2-Pass Anchor-Lock Propagation
// ============================================================================

#[derive(Clone, Debug)]
struct JointReplicaState {
    at: Vec<i32>,
    pos: Vec<i32>,
    score: f64,
}

#[inline]
fn pair_score_opt(a: i32, b: i32, p_mat: &Matrix2D) -> f64 {
    if a >= 0 && b >= 0 {
        p_mat.get(a as usize, b as usize)
    } else {
        0.0
    }
}

fn evaluate_joint_score(at: &[i32], u_mat: &Matrix2D, p_mat: &Matrix2D) -> f64 {
    let mut total = 0.0f64;
    let k_len = at.len();
    for k in 0..k_len {
        let s = at[k];
        if s >= 0 {
            total += u_mat.get(s as usize, k);
            if k + 1 < k_len && at[k + 1] >= 0 {
                total += p_mat.get(s as usize, at[k + 1] as usize);
            }
        }
    }
    total
}

fn build_valid_slots(u_mat: &Matrix2D, locked_slot: &[bool]) -> Vec<Vec<usize>> {
    let s_len = u_mat.rows;
    let k_len = u_mat.cols;
    let mut valid = vec![Vec::new(); s_len];
    for (s, row) in valid.iter_mut().enumerate() {
        for (k, &is_locked) in locked_slot.iter().enumerate().take(k_len) {
            if !is_locked && u_mat.get(s, k) > FORBID * 0.5 {
                row.push(k);
            }
        }
    }
    valid
}

fn build_top_pair_neighbors(p_mat: &Matrix2D, top_k: usize) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let s_len = p_mat.rows;
    let mut next_nbrs = vec![Vec::new(); s_len];
    let mut prev_nbrs = vec![Vec::new(); s_len];
    for s in 0..s_len {
        let mut outs: Vec<(usize, f64)> = (0..s_len)
            .filter(|&t| t != s)
            .map(|t| (t, p_mat.get(s, t)))
            .filter(|&(_, v)| v > 0.4)
            .collect();
        outs.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        next_nbrs[s] = outs.into_iter().take(top_k).map(|x| x.0).collect();

        let mut ins: Vec<(usize, f64)> = (0..s_len)
            .filter(|&t| t != s)
            .map(|t| (t, p_mat.get(t, s)))
            .filter(|&(_, v)| v > 0.4)
            .collect();
        ins.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        prev_nbrs[s] = ins.into_iter().take(top_k).map(|x| x.0).collect();
    }
    (next_nbrs, prev_nbrs)
}

fn delta_single_move(
    s: usize,
    k_new: usize,
    at: &[i32],
    pos: &[i32],
    u_mat: &Matrix2D,
    p_mat: &Matrix2D,
) -> f64 {
    let k_len = at.len();
    let k_old_i = pos[s];
    if k_old_i == k_new as i32 {
        return 0.0;
    }
    let t_i = at[k_new];
    if t_i >= 0 && k_old_i >= 0 {
        let k_old = k_old_i as usize;
        let t = t_i as usize;
        if u_mat.get(t, k_old) <= FORBID * 0.5 || u_mat.get(s, k_new) <= FORBID * 0.5 {
            return FORBID;
        }
        let du = (u_mat.get(s, k_new) + u_mat.get(t, k_old))
            - (u_mat.get(s, k_old) + u_mat.get(t, k_new));
        let (lo, hi) = if k_old < k_new {
            (k_old, k_new)
        } else {
            (k_new, k_old)
        };
        let mut old_p = 0.0;
        let mut new_p = 0.0;
        if hi == lo + 1 {
            let a = at[lo];
            let b = at[hi];
            if lo > 0 {
                old_p += pair_score_opt(at[lo - 1], a, p_mat);
                new_p += pair_score_opt(at[lo - 1], b, p_mat);
            }
            old_p += pair_score_opt(a, b, p_mat);
            new_p += pair_score_opt(b, a, p_mat);
            if hi + 1 < k_len {
                old_p += pair_score_opt(b, at[hi + 1], p_mat);
                new_p += pair_score_opt(a, at[hi + 1], p_mat);
            }
        } else {
            if k_old > 0 {
                old_p += pair_score_opt(at[k_old - 1], s as i32, p_mat);
                new_p += pair_score_opt(at[k_old - 1], t as i32, p_mat);
            }
            if k_old + 1 < k_len {
                old_p += pair_score_opt(s as i32, at[k_old + 1], p_mat);
                new_p += pair_score_opt(t as i32, at[k_old + 1], p_mat);
            }
            if k_new > 0 {
                old_p += pair_score_opt(at[k_new - 1], t as i32, p_mat);
                new_p += pair_score_opt(at[k_new - 1], s as i32, p_mat);
            }
            if k_new + 1 < k_len {
                old_p += pair_score_opt(t as i32, at[k_new + 1], p_mat);
                new_p += pair_score_opt(s as i32, at[k_new + 1], p_mat);
            }
        }
        du + (new_p - old_p)
    } else if t_i < 0 && k_old_i >= 0 {
        let k_old = k_old_i as usize;
        if u_mat.get(s, k_new) <= FORBID * 0.5 {
            return FORBID;
        }
        let du = u_mat.get(s, k_new) - u_mat.get(s, k_old);
        let mut old_p = 0.0;
        let mut new_p = 0.0;
        if k_old > 0 {
            old_p += pair_score_opt(at[k_old - 1], s as i32, p_mat);
        }
        if k_old + 1 < k_len {
            old_p += pair_score_opt(s as i32, at[k_old + 1], p_mat);
        }
        if k_new > 0 && k_new - 1 != k_old {
            new_p += pair_score_opt(at[k_new - 1], s as i32, p_mat);
        }
        if k_new + 1 < k_len && k_new + 1 != k_old {
            new_p += pair_score_opt(s as i32, at[k_new + 1], p_mat);
        }
        du + (new_p - old_p)
    } else if t_i >= 0 && k_old_i < 0 {
        let t = t_i as usize;
        if u_mat.get(s, k_new) <= FORBID * 0.5 {
            return FORBID;
        }
        let du = u_mat.get(s, k_new) - u_mat.get(t, k_new);
        let mut old_p = 0.0;
        let mut new_p = 0.0;
        if k_new > 0 {
            old_p += pair_score_opt(at[k_new - 1], t as i32, p_mat);
            new_p += pair_score_opt(at[k_new - 1], s as i32, p_mat);
        }
        if k_new + 1 < k_len {
            old_p += pair_score_opt(t as i32, at[k_new + 1], p_mat);
            new_p += pair_score_opt(s as i32, at[k_new + 1], p_mat);
        }
        du + (new_p - old_p)
    } else {
        if u_mat.get(s, k_new) <= FORBID * 0.5 {
            return FORBID;
        }
        let mut new_p = 0.0;
        if k_new > 0 {
            new_p += pair_score_opt(at[k_new - 1], s as i32, p_mat);
        }
        if k_new + 1 < k_len {
            new_p += pair_score_opt(s as i32, at[k_new + 1], p_mat);
        }
        u_mat.get(s, k_new) + new_p
    }
}

fn apply_single_move(s: usize, k_new: usize, at: &mut [i32], pos: &mut [i32]) {
    let k_old_i = pos[s];
    let t_i = at[k_new];
    if k_old_i >= 0 {
        at[k_old_i as usize] = t_i;
    }
    if t_i >= 0 {
        pos[t_i as usize] = k_old_i;
    }
    at[k_new] = s as i32;
    pos[s] = k_new as i32;
}

fn delta_unassign(s: usize, at: &[i32], pos: &[i32], u_mat: &Matrix2D, p_mat: &Matrix2D) -> f64 {
    let k_old_i = pos[s];
    if k_old_i < 0 {
        return 0.0;
    }
    let k = k_old_i as usize;
    let mut loss = u_mat.get(s, k);
    if k > 0 {
        loss += pair_score_opt(at[k - 1], s as i32, p_mat);
    }
    if k + 1 < at.len() {
        loss += pair_score_opt(s as i32, at[k + 1], p_mat);
    }
    -loss
}

/// Evaluate exact local objective contribution of two disjoint blocks `[k1..k1+blen]` and
/// `[k2..k2+blen]` (including flanking sequential links).
fn eval_two_blocks_local(
    at: &[i32],
    u_mat: &Matrix2D,
    p_mat: &Matrix2D,
    k1: usize,
    k2: usize,
    blen: usize,
) -> f64 {
    let k_len = at.len();
    let (lo, hi) = if k1 < k2 { (k1, k2) } else { (k2, k1) };
    let mut sc = 0.0;
    if lo > 0 {
        sc += pair_score_opt(at[lo - 1], at[lo], p_mat);
    }
    for o in 0..blen {
        let s = at[lo + o];
        if s >= 0 {
            sc += u_mat.get(s as usize, lo + o);
        }
        if o + 1 < blen {
            sc += pair_score_opt(at[lo + o], at[lo + o + 1], p_mat);
        }
    }
    if lo + blen == hi {
        sc += pair_score_opt(at[lo + blen - 1], at[hi], p_mat);
    } else {
        sc += pair_score_opt(at[lo + blen - 1], at[lo + blen], p_mat);
        sc += pair_score_opt(at[hi - 1], at[hi], p_mat);
    }
    for o in 0..blen {
        let s = at[hi + o];
        if s >= 0 {
            sc += u_mat.get(s as usize, hi + o);
        }
        if o + 1 < blen {
            sc += pair_score_opt(at[hi + o], at[hi + o + 1], p_mat);
        }
    }
    if hi + blen < k_len {
        sc += pair_score_opt(at[hi + blen - 1], at[hi + blen], p_mat);
    }
    sc
}

fn polish_joint_state(
    state: &mut JointReplicaState,
    u_mat: &Matrix2D,
    p_mat: &Matrix2D,
    valid_slots: &[Vec<usize>],
    locked_node: &[bool],
    locked_slot: &[bool],
) {
    let s_len = u_mat.rows;
    let k_len = u_mat.cols;
    for _pass in 0..8 {
        let mut improved = false;
        for s in 0..s_len {
            if locked_node[s] || valid_slots[s].is_empty() {
                continue;
            }
            let mut best_k: Option<usize> = None;
            let mut best_d = 1e-6f64;
            for &k in &valid_slots[s] {
                if locked_slot[k] {
                    continue;
                }
                if state.at[k] >= 0 && locked_node[state.at[k] as usize] {
                    continue;
                }
                let d = delta_single_move(s, k, &state.at, &state.pos, u_mat, p_mat);
                if d > best_d {
                    best_d = d;
                    best_k = Some(k);
                }
            }
            if let Some(k) = best_k {
                apply_single_move(s, k, &mut state.at, &mut state.pos);
                state.score += best_d;
                improved = true;
            } else if state.pos[s] >= 0 {
                let d_un = delta_unassign(s, &state.at, &state.pos, u_mat, p_mat);
                if d_un > 1e-6 {
                    let k_old = state.pos[s] as usize;
                    state.at[k_old] = -1;
                    state.pos[s] = -1;
                    state.score += d_un;
                    improved = true;
                }
            }
        }
        for blen in 2..=4 {
            if k_len <= blen + 1 {
                continue;
            }
            let mut k1 = 1usize;
            while k1 + blen <= k_len {
                if (k1..k1 + blen).any(|k| locked_slot[k])
                    || (k1..k1 + blen).all(|k| state.at[k] < 0)
                {
                    k1 += 1;
                    continue;
                }
                let mut best_k2: Option<usize> = None;
                let mut best_gain = 1e-6f64;
                for k2 in (k1 + blen)..=(k_len - blen) {
                    if (k2..k2 + blen).any(|k| locked_slot[k]) {
                        continue;
                    }
                    let ok = (0..blen).all(|o| {
                        let a = state.at[k1 + o];
                        let b = state.at[k2 + o];
                        (a < 0 || u_mat.get(a as usize, k2 + o) > FORBID * 0.5)
                            && (b < 0 || u_mat.get(b as usize, k1 + o) > FORBID * 0.5)
                    });
                    if !ok {
                        continue;
                    }
                    let before = eval_two_blocks_local(&state.at, u_mat, p_mat, k1, k2, blen);
                    for o in 0..blen {
                        state.at.swap(k1 + o, k2 + o);
                    }
                    let after = eval_two_blocks_local(&state.at, u_mat, p_mat, k1, k2, blen);
                    for o in 0..blen {
                        state.at.swap(k1 + o, k2 + o);
                    }
                    let gain = after - before;
                    if gain > best_gain {
                        best_gain = gain;
                        best_k2 = Some(k2);
                    }
                }
                if let Some(k2) = best_k2 {
                    for o in 0..blen {
                        let a = state.at[k1 + o];
                        let b = state.at[k2 + o];
                        state.at[k1 + o] = b;
                        state.at[k2 + o] = a;
                        if a >= 0 {
                            state.pos[a as usize] = (k2 + o) as i32;
                        }
                        if b >= 0 {
                            state.pos[b as usize] = (k1 + o) as i32;
                        }
                    }
                    state.score += best_gain;
                    improved = true;
                }
                k1 += 1;
            }
        }
        if !improved {
            break;
        }
    }
    state.score = evaluate_joint_score(&state.at, u_mat, p_mat);
}

fn initialize_joint_replica(
    u_mat: &Matrix2D,
    p_mat: &Matrix2D,
    valid_slots: &[Vec<usize>],
    locked_anchors: &[(usize, usize)],
    rng: &mut Rng,
) -> JointReplicaState {
    let s_len = u_mat.rows;
    let k_len = u_mat.cols;
    let mut at = vec![-1i32; k_len];
    let mut pos = vec![-1i32; s_len];

    for &(s, k) in locked_anchors {
        if s < s_len && k < k_len && at[k] < 0 && pos[s] < 0 && u_mat.get(s, k) > FORBID * 0.5 {
            at[k] = s as i32;
            pos[s] = k as i32;
        }
    }

    let mut order: Vec<usize> = (0..s_len).filter(|&s| pos[s] < 0).collect();
    rng.shuffle(&mut order);
    for s in order {
        let mut best_k: Option<usize> = None;
        let mut best_u = -1.5f64;
        for &k in &valid_slots[s] {
            if at[k] < 0 {
                let u_val = u_mat.get(s, k) + rng.normal(0.0, 1.2);
                if u_val > best_u {
                    best_u = u_val;
                    best_k = Some(k);
                }
            }
        }
        if let Some(k) = best_k {
            at[k] = s as i32;
            pos[s] = k as i32;
        }
    }

    let score = evaluate_joint_score(&at, u_mat, p_mat);
    JointReplicaState { at, pos, score }
}

fn crossover_joint_states(
    parent_a: &JointReplicaState,
    parent_b: &JointReplicaState,
    u_mat: &Matrix2D,
    p_mat: &Matrix2D,
    locked_anchors: &[(usize, usize)],
    rng: &mut Rng,
) -> JointReplicaState {
    let s_len = u_mat.rows;
    let k_len = u_mat.cols;
    let mut at = vec![-1i32; k_len];
    let mut pos = vec![-1i32; s_len];

    for &(s, k) in locked_anchors {
        if s < s_len && k < k_len && at[k] < 0 && pos[s] < 0 {
            at[k] = s as i32;
            pos[s] = k as i32;
        }
    }

    for (k, slot) in at.iter_mut().enumerate().take(k_len).skip(1) {
        if *slot < 0 && parent_a.at[k] >= 0 && parent_a.at[k] == parent_b.at[k] {
            let s = parent_a.at[k] as usize;
            if pos[s] < 0 && u_mat.get(s, k) > FORBID * 0.5 {
                *slot = s as i32;
                pos[s] = k as i32;
            }
        }
    }

    let cut = if k_len > 3 {
        rng.range_usize(1, k_len - 1)
    } else {
        1
    };
    let take_a_first = rng.f64() < 0.5;
    for (k, slot) in at.iter_mut().enumerate().take(k_len).skip(1) {
        if *slot >= 0 {
            continue;
        }
        let primary = if (k < cut) == take_a_first {
            parent_a.at[k]
        } else {
            parent_b.at[k]
        };
        let secondary = if (k < cut) == take_a_first {
            parent_b.at[k]
        } else {
            parent_a.at[k]
        };
        for cand in [primary, secondary] {
            if cand >= 0 {
                let s = cand as usize;
                if pos[s] < 0 && u_mat.get(s, k) > FORBID * 0.5 {
                    *slot = cand;
                    pos[s] = k as i32;
                    break;
                }
            }
        }
    }

    let score = evaluate_joint_score(&at, u_mat, p_mat);
    JointReplicaState { at, pos, score }
}

#[allow(clippy::too_many_arguments)]
fn run_sa_replica(
    mut state: JointReplicaState,
    u_mat: &Matrix2D,
    p_mat: &Matrix2D,
    valid_slots: &[Vec<usize>],
    next_nbrs: &[Vec<usize>],
    prev_nbrs: &[Vec<usize>],
    locked_node: &[bool],
    locked_slot: &[bool],
    steps: usize,
    t_start: f64,
    t_end: f64,
    rng: &mut Rng,
) -> JointReplicaState {
    let s_len = u_mat.rows;
    let k_len = u_mat.cols;
    let unlocked_nodes: Vec<usize> = (0..s_len)
        .filter(|&s| !locked_node[s] && !valid_slots[s].is_empty())
        .collect();
    if unlocked_nodes.is_empty() || k_len < 2 || steps == 0 {
        return state;
    }

    let mut best = state.clone();
    let ratio = (t_end / t_start.max(1e-6)).max(1e-6);

    for step in 0..steps {
        let frac = (step as f64) / (steps as f64);
        let temp = (t_start * ratio.powf(frac)).max(1e-4);
        let phase = rng.usize(16);

        if phase < 9 {
            let s = unlocked_nodes[rng.usize(unlocked_nodes.len())];
            let slots = &valid_slots[s];
            let k_new = slots[rng.usize(slots.len())];
            if locked_slot[k_new] || (state.at[k_new] >= 0 && locked_node[state.at[k_new] as usize])
            {
                continue;
            }
            let d = delta_single_move(s, k_new, &state.at, &state.pos, u_mat, p_mat);
            if d > FORBID * 0.5 && (d >= 0.0 || rng.f64() < (d / temp).exp()) {
                apply_single_move(s, k_new, &mut state.at, &mut state.pos);
                state.score += d;
            }
        } else if phase == 9 {
            let s = unlocked_nodes[rng.usize(unlocked_nodes.len())];
            if state.pos[s] >= 0 {
                let d = delta_unassign(s, &state.at, &state.pos, u_mat, p_mat);
                if d >= 0.0 || rng.f64() < (d / temp).exp() {
                    let k_old = state.pos[s] as usize;
                    state.at[k_old] = -1;
                    state.pos[s] = -1;
                    state.score += d;
                }
            }
        } else if phase < 14 {
            let s = unlocked_nodes[rng.usize(unlocked_nodes.len())];
            let ks_i = state.pos[s];
            if ks_i < 0 {
                continue;
            }
            let ks = ks_i as usize;
            let forward = rng.f64() < 0.5;
            if forward && ks + 1 < k_len && !locked_slot[ks + 1] && !next_nbrs[s].is_empty() {
                let t = next_nbrs[s][rng.usize(next_nbrs[s].len())];
                if !locked_node[t]
                    && (state.at[ks + 1] < 0 || !locked_node[state.at[ks + 1] as usize])
                {
                    let d = delta_single_move(t, ks + 1, &state.at, &state.pos, u_mat, p_mat);
                    if d > FORBID * 0.5 && (d >= 0.0 || rng.f64() < (d / temp).exp()) {
                        apply_single_move(t, ks + 1, &mut state.at, &mut state.pos);
                        state.score += d;
                    }
                }
            } else if !forward && ks >= 2 && !locked_slot[ks - 1] && !prev_nbrs[s].is_empty() {
                let t = prev_nbrs[s][rng.usize(prev_nbrs[s].len())];
                if !locked_node[t]
                    && (state.at[ks - 1] < 0 || !locked_node[state.at[ks - 1] as usize])
                {
                    let d = delta_single_move(t, ks - 1, &state.at, &state.pos, u_mat, p_mat);
                    if d > FORBID * 0.5 && (d >= 0.0 || rng.f64() < (d / temp).exp()) {
                        apply_single_move(t, ks - 1, &mut state.at, &mut state.pos);
                        state.score += d;
                    }
                }
            }
        } else if k_len >= 6 {
            let blen = rng.range_usize(2, 6.min(k_len / 2 + 1));
            if k_len <= 2 * blen + 1 {
                continue;
            }
            let k1 = rng.range_usize(1, k_len - blen + 1);
            let k2 = rng.range_usize(1, k_len - blen + 1);
            if k1.abs_diff(k2) < blen {
                continue;
            }
            if (0..blen).any(|o| locked_slot[k1 + o] || locked_slot[k2 + o]) {
                continue;
            }
            let feasible = (0..blen).all(|o| {
                let a = state.at[k1 + o];
                let b = state.at[k2 + o];
                (a < 0 || u_mat.get(a as usize, k2 + o) > FORBID * 0.5)
                    && (b < 0 || u_mat.get(b as usize, k1 + o) > FORBID * 0.5)
            });
            if !feasible {
                continue;
            }
            let before = eval_two_blocks_local(&state.at, u_mat, p_mat, k1, k2, blen);
            for o in 0..blen {
                state.at.swap(k1 + o, k2 + o);
            }
            let after = eval_two_blocks_local(&state.at, u_mat, p_mat, k1, k2, blen);
            let d = after - before;
            if d >= 0.0 || rng.f64() < (d / temp).exp() {
                for o in 0..blen {
                    let a = state.at[k1 + o];
                    let b = state.at[k2 + o];
                    if a >= 0 {
                        state.pos[a as usize] = (k1 + o) as i32;
                    }
                    if b >= 0 {
                        state.pos[b as usize] = (k2 + o) as i32;
                    }
                }
                state.score += d;
            } else {
                for o in 0..blen {
                    state.at.swap(k1 + o, k2 + o);
                }
            }
        }

        if state.score > best.score {
            best = state.clone();
        }
    }

    polish_joint_state(
        &mut best,
        u_mat,
        p_mat,
        valid_slots,
        locked_node,
        locked_slot,
    );
    best
}

fn run_island_ensemble(
    u_mat: &Matrix2D,
    p_mat: &Matrix2D,
    locked_anchors: &[(usize, usize)],
    cfg: &JointIdpConfig,
    pass_seed_offset: u64,
) -> (JointReplicaState, Vec<JointReplicaState>) {
    let s_len = u_mat.rows;
    let k_len = u_mat.cols;
    let mut locked_node = vec![false; s_len];
    let mut locked_slot = vec![false; k_len];
    for &(s, k) in locked_anchors {
        if s < s_len && k < k_len {
            locked_node[s] = true;
            locked_slot[k] = true;
        }
    }

    let valid_slots = build_valid_slots(u_mat, &locked_slot);
    let (next_nbrs, prev_nbrs) = build_top_pair_neighbors(p_mat, 6);

    let n_replicas = cfg.replicas.max(4);
    let n_islands = cfg.islands.max(1).min(n_replicas);
    let n_gens = cfg.generations.max(1);
    let scale_n = s_len.max(k_len).max(1);
    let steps_per_gen = (cfg.steps_per_system.max(200) * scale_n) / n_gens;

    let mut replicas: Vec<JointReplicaState> = (0..n_replicas)
        .map(|r| {
            let mut rng = Rng::new(
                cfg.seed
                    .wrapping_add(pass_seed_offset)
                    .wrapping_add((r as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
            );
            initialize_joint_replica(u_mat, p_mat, &valid_slots, locked_anchors, &mut rng)
        })
        .collect();

    let mut elite_pool: Vec<JointReplicaState> = Vec::new();

    for gen in 0..n_gens {
        let gen_t_start = if gen == 0 {
            cfg.t_start
        } else {
            (cfg.t_start * 0.45_f64.powi(gen as i32)).max(1.8)
        };
        let gen_seed_base = cfg
            .seed
            .wrapping_add(pass_seed_offset)
            .wrapping_add(((gen + 1) as u64).wrapping_mul(0x517C_C1B7_2722_0A95));

        #[cfg(target_arch = "wasm32")]
        let next_replicas: Vec<JointReplicaState> = replicas
            .into_iter()
            .enumerate()
            .map(|(r_idx, st)| {
                let mut rng = Rng::new(gen_seed_base.wrapping_add((r_idx as u64) * 10_007));
                run_sa_replica(
                    st,
                    u_mat,
                    p_mat,
                    &valid_slots,
                    &next_nbrs,
                    &prev_nbrs,
                    &locked_node,
                    &locked_slot,
                    steps_per_gen,
                    gen_t_start,
                    cfg.t_end,
                    &mut rng,
                )
            })
            .collect();

        #[cfg(not(target_arch = "wasm32"))]
        let next_replicas: Vec<JointReplicaState> = {
            let n_threads = thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
                .clamp(1, n_replicas);
            let chunk = n_replicas.div_ceil(n_threads);
            let mut indexed: Vec<(usize, JointReplicaState)> =
                replicas.into_iter().enumerate().collect();
            let batches: Vec<Vec<(usize, JointReplicaState)>> = thread::scope(|scope| {
                let mut handles = Vec::new();
                while !indexed.is_empty() {
                    let take = chunk.min(indexed.len());
                    let batch: Vec<(usize, JointReplicaState)> = indexed.drain(..take).collect();
                    let u_ref = u_mat;
                    let p_ref = p_mat;
                    let vs_ref = &valid_slots;
                    let nn_ref = &next_nbrs;
                    let pn_ref = &prev_nbrs;
                    let ln_ref = &locked_node;
                    let ls_ref = &locked_slot;
                    handles.push(scope.spawn(move || {
                        batch
                            .into_iter()
                            .map(|(r_idx, st)| {
                                let mut rng =
                                    Rng::new(gen_seed_base.wrapping_add((r_idx as u64) * 10_007));
                                let out = run_sa_replica(
                                    st,
                                    u_ref,
                                    p_ref,
                                    vs_ref,
                                    nn_ref,
                                    pn_ref,
                                    ln_ref,
                                    ls_ref,
                                    steps_per_gen,
                                    gen_t_start,
                                    cfg.t_end,
                                    &mut rng,
                                );
                                (r_idx, out)
                            })
                            .collect::<Vec<_>>()
                    }));
                }
                handles
                    .into_iter()
                    .map(|h| h.join().unwrap_or_default())
                    .collect()
            });
            let mut flat: Vec<(usize, JointReplicaState)> = batches.into_iter().flatten().collect();
            flat.sort_by_key(|(idx, _)| *idx);
            flat.into_iter().map(|(_, st)| st).collect()
        };

        replicas = next_replicas;
        elite_pool.extend(replicas.iter().cloned());

        if gen + 1 < n_gens {
            let island_size = n_replicas.div_ceil(n_islands);
            let mut evolved = Vec::with_capacity(n_replicas);
            for isl in 0..n_islands {
                let lo = isl * island_size;
                let hi = (lo + island_size).min(n_replicas);
                if lo >= hi {
                    continue;
                }
                let mut group: Vec<&JointReplicaState> = replicas[lo..hi].iter().collect();
                group.sort_by(|a, b| b.score.total_cmp(&a.score));
                let best_a = group[0];
                let best_b = group[1.min(group.len() - 1)];
                for local_i in 0..group.len() {
                    if local_i == 0 {
                        evolved.push(best_a.clone());
                    } else if local_i == 1 && group.len() >= 3 {
                        evolved.push(best_b.clone());
                    } else {
                        let mut rng = Rng::new(
                            gen_seed_base.wrapping_add(((isl * 131 + local_i * 37) as u64) + 777),
                        );
                        let partner = group[rng.usize(group.len().min(4))];
                        evolved.push(crossover_joint_states(
                            best_a,
                            partner,
                            u_mat,
                            p_mat,
                            locked_anchors,
                            &mut rng,
                        ));
                    }
                }
            }
            replicas = evolved;
        }
    }

    elite_pool.sort_by(|a, b| b.score.total_cmp(&a.score));
    let best = elite_pool
        .first()
        .cloned()
        .unwrap_or_else(|| JointReplicaState {
            at: vec![-1; k_len],
            pos: vec![-1; s_len],
            score: 0.0,
        });
    (best, replicas)
}

// ============================================================================
// Honest Adaptive-Temperature Posterior & Unlabelled QC Calibration (`P1-1`, `P1-2`)
// ============================================================================

#[derive(Clone, Debug, Default)]
struct PosteriorDiagnostics {
    p_ens: Vec<f64>,
    p_local: Vec<f64>,
    raw_post: Vec<f64>,
    emp_rms_seq_ca: f64,
    emp_rms_seq_c: f64,
    mean_cands_per_ch: f64,
    ambig_placement_frac: f64,
    mean_ens_agreement: f64,
    connectivity_frac: f64,
    support_3d_frac: f64,
    node_ratio: f64,
}

/// Compute honest per-node ensemble consensus `p_ens`, full local-swap + multi-way cycle +
/// `2..=4`-residue motif-swap Boltzmann specificity `p_local` with adaptive `tau_local`, and
/// unlabelled data-quality diagnostics (`180.core:P1-1`).
///
/// Strictly NO Pass-2 lock floor: locked and unlocked placements are evaluated identically
/// against the full competitor landscape.
#[allow(clippy::too_many_arguments)]
fn compute_posteriors_honest(
    best: &JointReplicaState,
    replicas: &[JointReplicaState],
    nodes: &[JointIdpNode],
    seq: &[String],
    u_mat: &Matrix2D,
    p_mat: &Matrix2D,
    fusion_rms_c_ppm: f64,
    fusion_rms_n_ppm: f64,
    hn_unmatched_frac: f64,
) -> PosteriorDiagnostics {
    let s_len = u_mat.rows;
    let k_len = u_mat.cols;
    let mut diag = PosteriorDiagnostics {
        p_ens: vec![0.0; s_len],
        p_local: vec![0.0; s_len],
        raw_post: vec![0.0; s_len],
        ..PosteriorDiagnostics::default()
    };
    if s_len == 0 || k_len < 2 || replicas.is_empty() {
        return diag;
    }

    let n_assignable = k_len.saturating_sub(1).max(1);
    diag.node_ratio = (s_len as f64) / (n_assignable as f64);

    // 1. Empirical unlabelled matching-noise & multiplicity diagnostics
    let mut ch_cand_sum = 0usize;
    let mut ch_nonempty = 0usize;
    let mut n_with_3d = 0usize;
    for nd in nodes {
        if nd.has_3d_support() {
            n_with_3d += 1;
        }
        for ch in [
            &nd.ca_curr,
            &nd.ca_prev,
            &nd.cb_prev,
            &nd.cb_curr,
            &nd.c_curr,
        ] {
            if !ch.is_empty() {
                ch_nonempty += 1;
                ch_cand_sum += ch.len().min(MAX_JOINT_CANDS);
            }
        }
    }
    diag.support_3d_frac = (n_with_3d as f64) / (s_len as f64);
    diag.mean_cands_per_ch = if ch_nonempty > 0 {
        (ch_cand_sum as f64) / (ch_nonempty as f64)
    } else {
        1.0
    };

    let mut sq_ca_diffs = Vec::new();
    let mut sq_c_diffs = Vec::new();
    let mut n_assigned = 0usize;
    let mut n_connected = 0usize;
    for k in 1..k_len {
        let sa_i = best.at[k];
        if sa_i < 0 {
            continue;
        }
        n_assigned += 1;
        let sa = sa_i as usize;
        let prev_ok = k > 1 && best.at[k - 1] >= 0 && p_mat.get(best.at[k - 1] as usize, sa) >= 1.0;
        let next_ok =
            k + 1 < k_len && best.at[k + 1] >= 0 && p_mat.get(sa, best.at[k + 1] as usize) >= 1.0;
        if prev_ok || next_ok {
            n_connected += 1;
        }
        if k + 1 < k_len && best.at[k + 1] >= 0 {
            let sb = best.at[k + 1] as usize;
            let na = &nodes[sa];
            let nb = &nodes[sb];
            if !na.ca_curr.is_empty() && !nb.ca_prev.is_empty() {
                let mut best_df = f64::INFINITY;
                for &va in na.ca_curr.iter().take(MAX_JOINT_CANDS) {
                    for &vb in nb.ca_prev.iter().take(MAX_JOINT_CANDS) {
                        best_df = best_df.min((va - vb).abs());
                    }
                }
                if best_df <= 0.40 {
                    sq_ca_diffs.push(best_df * best_df);
                }
            }
            if let Some(cp_b) = nb.c_prev {
                if !na.c_curr.is_empty() {
                    let mut best_dc = f64::INFINITY;
                    for &va in na.c_curr.iter().take(MAX_JOINT_CANDS) {
                        best_dc = best_dc.min((va - cp_b).abs());
                    }
                    if best_dc <= 0.30 {
                        sq_c_diffs.push(best_dc * best_dc);
                    }
                }
            }
        }
    }
    diag.connectivity_frac = if n_assigned > 0 {
        (n_connected as f64) / (n_assigned as f64)
    } else {
        0.0
    };
    diag.emp_rms_seq_ca = if sq_ca_diffs.len() >= 5 {
        (sq_ca_diffs.iter().sum::<f64>() / (sq_ca_diffs.len() as f64)).sqrt()
    } else {
        0.0
    };
    diag.emp_rms_seq_c = if sq_c_diffs.len() >= 5 {
        (sq_c_diffs.iter().sum::<f64>() / (sq_c_diffs.len() as f64)).sqrt()
    } else {
        0.0
    };

    // 2. Boltzmann-weighted replica consensus `p_ens`
    let max_sc = replicas
        .iter()
        .map(|r| r.score)
        .fold(f64::NEG_INFINITY, f64::max);
    let tau_ens = 6.5f64;
    let rep_w: Vec<f64> = replicas
        .iter()
        .map(|r| ((r.score - max_sc) / tau_ens).clamp(-35.0, 0.0).exp())
        .collect();
    let sum_w: f64 = rep_w.iter().sum::<f64>().max(1e-12);

    let mut ens_sum = 0.0f64;
    for s in 0..s_len {
        let k_best = best.pos[s];
        if k_best < 0 {
            continue;
        }
        let mut agree = 0.0f64;
        for (r, &w) in replicas.iter().zip(rep_w.iter()) {
            if r.pos[s] == k_best {
                agree += w;
            }
        }
        let pe = (agree / sum_w).clamp(0.0, 1.0);
        diag.p_ens[s] = pe;
        ens_sum += pe;
    }
    diag.mean_ens_agreement = if n_assigned > 0 {
        ens_sum / (n_assigned as f64)
    } else {
        0.0
    };

    // 3. Evaluate single/2-swap, multi-way cycle, and `2..=4`-residue motif-swap competitors
    let mut work_at = best.at.clone();
    let mut top_single: Vec<Vec<f64>> = vec![Vec::new(); s_len];
    let mut top_cycle: Vec<Vec<f64>> = vec![Vec::new(); s_len];
    let mut best_motif: Vec<f64> = vec![f64::NEG_INFINITY; s_len];
    let mut d_unassign_vec: Vec<f64> = vec![f64::NEG_INFINITY; s_len];
    let mut n_ambig = 0usize;

    for s in 0..s_len {
        let k_i = best.pos[s];
        if k_i < 0 {
            continue;
        }
        let k_star = k_i as usize;
        let u_star = u_mat.get(s, k_star);
        let pair_prev = if k_star > 1 && best.at[k_star - 1] >= 0 {
            p_mat.get(best.at[k_star - 1] as usize, s).max(0.0)
        } else {
            0.0
        };
        let pair_next = if k_star + 1 < k_len && best.at[k_star + 1] >= 0 {
            p_mat.get(s, best.at[k_star + 1] as usize).max(0.0)
        } else {
            0.0
        };
        let walk_anchor = 0.72 * (pair_prev + pair_next) + 1.65;

        let mut s_deltas = Vec::with_capacity(k_len);
        let mut c_deltas = Vec::with_capacity(k_len);
        for k_alt in 1..k_len {
            if k_alt == k_star || u_mat.get(s, k_alt) <= FORBID * 0.5 {
                continue;
            }
            c_deltas.push((u_mat.get(s, k_alt) - u_star) - walk_anchor);
            let d = delta_single_move(s, k_alt, &best.at, &best.pos, u_mat, p_mat);
            if d > FORBID * 0.5 {
                s_deltas.push(d);
            }
        }
        // Also consider replacing `s` at `k_star` with any currently unassigned node `u` (decoy competition!)
        for u_cand in 0..s_len {
            if u_cand == s || best.pos[u_cand] >= 0 || u_mat.get(u_cand, k_star) <= FORBID * 0.5 {
                continue;
            }
            let d_rep = delta_single_move(u_cand, k_star, &best.at, &best.pos, u_mat, p_mat);
            if d_rep > FORBID * 0.5 {
                s_deltas.push(d_rep);
            }
        }

        s_deltas.sort_by(|a, b| b.total_cmp(a));
        c_deltas.sort_by(|a, b| b.total_cmp(a));
        top_single[s].extend(s_deltas.into_iter().take(8));
        top_cycle[s].extend(c_deltas.into_iter().take(3));
        d_unassign_vec[s] = delta_unassign(s, &best.at, &best.pos, u_mat, p_mat);

        // Short motif-swap competitors (`blen = 2..=4`) covering `k_star`
        let mut best_blk = f64::NEG_INFINITY;
        for blen in 2..=4usize.min(k_len.saturating_sub(1)) {
            let win_lo = k_star.saturating_sub(blen - 1).max(1);
            let win_hi = k_star.min(k_len - blen);
            if win_lo > win_hi {
                continue;
            }
            for k_win in win_lo..=win_hi {
                for k_alt in 1..=(k_len - blen) {
                    if k_alt.abs_diff(k_win) < blen {
                        continue;
                    }
                    let feasible = (0..blen).all(|o| {
                        let a = work_at[k_win + o];
                        let b = work_at[k_alt + o];
                        (a < 0 || u_mat.get(a as usize, k_alt + o) > FORBID * 0.5)
                            && (b < 0 || u_mat.get(b as usize, k_win + o) > FORBID * 0.5)
                    });
                    if !feasible {
                        continue;
                    }
                    let before = eval_two_blocks_local(&work_at, u_mat, p_mat, k_win, k_alt, blen);
                    for o in 0..blen {
                        work_at.swap(k_win + o, k_alt + o);
                    }
                    let after = eval_two_blocks_local(&work_at, u_mat, p_mat, k_win, k_alt, blen);
                    for o in 0..blen {
                        work_at.swap(k_win + o, k_alt + o);
                    }
                    let delta = after - before;
                    if delta > best_blk {
                        best_blk = delta;
                    }
                }
            }
        }
        best_motif[s] = best_blk;

        let strongest_alt = top_single[s]
            .first()
            .copied()
            .unwrap_or(f64::NEG_INFINITY)
            .max(best_blk)
            .max(d_unassign_vec[s]);
        if strongest_alt >= -2.0 {
            n_ambig += 1;
        }
    }
    diag.ambig_placement_frac = if n_assigned > 0 {
        (n_ambig as f64) / (n_assigned as f64)
    } else {
        1.0
    };

    // 4. Adaptive temperature `tau_local_eff` scaled by empirical RMS, node inflation,
    //    missing channels, fusion scatter, ambiguity, and sequence repeat factor!
    let (h_bits, max_aa, max_dipep, top3_dipep) = sequence_complexity_metrics(seq);
    let max_homo = max_homopolymer_run(seq);
    let seq_extreme_repeat =
        k_len >= 16 && (h_bits < 1.65 || max_aa >= 0.55 || max_dipep >= 0.35 || max_homo >= 6);
    let seq_warn_repeat =
        k_len >= 16 && (h_bits < 2.60 || max_aa >= 0.35 || top3_dipep >= 0.35 || max_homo >= 4);

    let rms_ca_scale = if diag.emp_rms_seq_ca > 0.042 {
        (diag.emp_rms_seq_ca / 0.040).clamp(1.0, 2.6)
    } else {
        1.0
    };
    let rms_c_scale = if diag.emp_rms_seq_c > 0.028 {
        (diag.emp_rms_seq_c / 0.026).clamp(1.0, 2.4)
    } else {
        1.0
    };
    let rms_scale = rms_ca_scale.max(rms_c_scale);

    // Decoy/node inflation factor: `n_nodes / n_assignable > 1.15` widens uncertainty!
    let inflation_scale = if diag.node_ratio > 1.15 {
        (1.0 + 1.35 * (diag.node_ratio - 1.15)).clamp(1.0, 2.8)
    } else {
        1.0
    };

    // Missing channels / sparse connectivity / fusion scatter
    let conn_deficit = (0.88 - diag.connectivity_frac).max(0.0);
    let sup_deficit = (0.85 - diag.support_3d_frac).max(0.0);
    let channel_scale = (1.0 + 2.8 * conn_deficit + 1.5 * sup_deficit).clamp(1.0, 2.2);

    let fus_n_excess = (fusion_rms_n_ppm - 0.08).max(0.0);
    let fus_c_excess = (fusion_rms_c_ppm - 0.028).max(0.0);
    let unm_excess = (hn_unmatched_frac - 0.16).max(0.0);
    let fusion_scale =
        (1.0 + 5.5 * fus_n_excess + 14.0 * fus_c_excess + 1.6 * unm_excess).clamp(1.0, 2.4);

    let repeat_scale = if seq_extreme_repeat {
        2.60
    } else if seq_warn_repeat {
        1.95
    } else {
        1.0
    };
    let ambig_scale = (1.0 + 1.45 * (diag.ambig_placement_frac - 0.15).max(0.0)).clamp(1.0, 2.0);

    let tau_local_eff = (2.05
        * rms_scale
        * inflation_scale
        * channel_scale
        * fusion_scale
        * repeat_scale
        * ambig_scale)
        .clamp(2.05, 16.0);

    for s in 0..s_len {
        if best.pos[s] < 0 {
            continue;
        }
        let mut alt_sum = (d_unassign_vec[s] / tau_local_eff).clamp(-35.0, 18.0).exp();
        for &d in &top_single[s] {
            alt_sum += (d / tau_local_eff).clamp(-35.0, 18.0).exp();
        }
        for &cd in &top_cycle[s] {
            alt_sum += (cd / (1.15 * tau_local_eff)).clamp(-35.0, 18.0).exp();
        }
        if best_motif[s] > -5.0 {
            alt_sum += (best_motif[s] / (1.15 * tau_local_eff))
                .clamp(-35.0, 18.0)
                .exp();
        }
        let p_loc = (1.0 / (1.0 + alt_sum)).clamp(0.0, 1.0);
        diag.p_local[s] = p_loc;
        // Honest combination: `p_ens` alone can never inflate a low-specificity placement!
        let combined = 0.72 * p_loc + 0.28 * (p_loc * diag.p_ens[s]).sqrt();
        diag.raw_post[s] = combined.clamp(0.0, 1.0);
    }

    diag
}

/// Map pre-calibration raw posterior through dedicated monotone [`JOINT_IDP_CALIBRATION_KNOTS`].
#[inline]
pub fn calibrate_joint_posterior(raw: f64) -> f64 {
    if !raw.is_finite() {
        return 0.0;
    }
    let x = raw.clamp(0.0, 1.0);
    let knots = JOINT_IDP_CALIBRATION_KNOTS;
    if knots.is_empty() {
        return x;
    }
    if x <= knots[0].0 {
        return knots[0].1.clamp(0.0, 1.0);
    }
    for w in knots.windows(2) {
        let (x0, y0) = w[0];
        let (x1, y1) = w[1];
        if x <= x1 {
            let t = if (x1 - x0).abs() > 1e-12 {
                (x - x0) / (x1 - x0)
            } else {
                0.0
            };
            return (y0 + t * (y1 - y0)).clamp(0.0, 1.0);
        }
    }
    knots.last().map_or(x, |k| k.1).clamp(0.0, 1.0)
}

/// Select conservative Pass-1 unequivocal anchors to lock before Pass-2 re-annealing:
/// 1. Unequivocal `>= 3`-residue contiguous runs (`p_ens >= 0.90`, `p_local >= 0.88`, `P >= 1.2`),
/// 2. Unequivocal `X–Pro` (or `X–Pro–Y`) bridges where `Pro` is a `ProlineConBridge`,
/// 3. Unequivocal C-terminal `COO-` anchor at `k == L - 1`.
#[allow(clippy::too_many_arguments)]
fn select_conservative_pass1_anchors(
    best: &JointReplicaState,
    p_ens: &[f64],
    p_local: &[f64],
    nodes: &[JointIdpNode],
    seq: &[String],
    u_mat: &Matrix2D,
    p_mat: &Matrix2D,
    c_term_coo_candidates: &[usize],
    cfg: &JointIdpConfig,
) -> Vec<(usize, usize)> {
    let k_len = seq.len();
    let mut locked_positions = vec![false; k_len];

    let node_strong = |s: usize, k: usize| -> bool {
        p_ens[s] >= cfg.anchor_min_consensus
            && p_local[s] >= cfg.anchor_min_local_prob
            && u_mat.get(s, k) >= -1.0
    };

    // 1. Unequivocal `>= 3`-residue contiguous runs
    let mut k = 1usize;
    while k < k_len {
        let s0 = best.at[k];
        if s0 < 0 || !node_strong(s0 as usize, k) {
            k += 1;
            continue;
        }
        let mut end = k;
        while end + 1 < k_len {
            let sa = best.at[end];
            let sb = best.at[end + 1];
            if sb >= 0
                && node_strong(sb as usize, end + 1)
                && p_mat.get(sa as usize, sb as usize) >= cfg.anchor_min_pair_score
            {
                end += 1;
            } else {
                break;
            }
        }
        if end - k + 1 >= 3 {
            for slot in locked_positions.iter_mut().take(end + 1).skip(k) {
                *slot = true;
            }
        }
        k = end + 1;
    }

    // 2. Unequivocal `X–Pro` (and `X–Pro–Y`) bridges
    for kp in 2..k_len {
        if seq[kp] != "PRO" {
            continue;
        }
        let sp_i = best.at[kp];
        let sx_i = best.at[kp - 1];
        if sp_i < 0 || sx_i < 0 {
            continue;
        }
        let sp = sp_i as usize;
        let sx = sx_i as usize;
        if nodes[sp].kind == JointNodeKind::ProlineConBridge
            && node_strong(sp, kp)
            && node_strong(sx, kp - 1)
            && p_mat.get(sx, sp) >= cfg.anchor_min_pair_score
        {
            locked_positions[kp - 1] = true;
            locked_positions[kp] = true;
            if kp + 1 < k_len {
                let sy_i = best.at[kp + 1];
                if sy_i >= 0
                    && node_strong(sy_i as usize, kp + 1)
                    && p_mat.get(sp, sy_i as usize) >= cfg.anchor_min_pair_score
                {
                    locked_positions[kp + 1] = true;
                }
            }
        }
    }

    // 3. Unequivocal C-terminal `COO-` anchor
    if cfg.enable_cterm_coo_prior && k_len >= 3 {
        let last = k_len - 1;
        let s_last = best.at[last];
        let s_prev = best.at[last - 1];
        if s_last >= 0
            && s_prev >= 0
            && c_term_coo_candidates.contains(&(s_last as usize))
            && p_ens[s_last as usize] >= 0.92
            && p_local[s_last as usize] >= 0.90
            && p_mat.get(s_prev as usize, s_last as usize) >= cfg.anchor_min_pair_score
        {
            locked_positions[last] = true;
            if node_strong(s_prev as usize, last - 1) {
                locked_positions[last - 1] = true;
            }
        }
    }

    let mut anchors = Vec::new();
    for (pos_k, &is_locked) in locked_positions.iter().enumerate() {
        if is_locked && best.at[pos_k] >= 0 {
            anchors.push((best.at[pos_k] as usize, pos_k));
        }
    }
    anchors
}

// ============================================================================
// NMR-STAR / Reference Conversion & Top-Level Entrypoints
// ============================================================================

/// Convert a [`JointIdpAssignResult`] into a standard [`Reference`] chemical-shift table and
/// per-residue calibrated confidence map for NMR-STAR, TSV, NEF, and HTML quality reporting.
pub fn joint_idp_to_reference(
    result: &JointIdpAssignResult,
    sequence: &[String],
    seq_ids: &[i32],
) -> (Reference, BTreeMap<usize, f64>) {
    let seq = normalize_seq_3(sequence);
    let ids: Vec<i32> = if seq_ids.len() == seq.len() {
        seq_ids.to_vec()
    } else {
        (1..=seq.len() as i32).collect()
    };
    let mut shifts: BTreeMap<(usize, String), f64> = BTreeMap::new();
    let mut conf: BTreeMap<usize, f64> = BTreeMap::new();

    let mut record_conf = |k: usize, c: f64| {
        let e = conf.entry(k).or_insert(0.0);
        if c > *e {
            *e = c;
        }
    };

    for (s_idx, &pos_i) in result.position.iter().enumerate() {
        if pos_i < 1 || (pos_i as usize) >= seq.len() {
            continue;
        }
        let k = pos_i as usize;
        let node = &result.nodes[s_idx];
        let c_val = result.confidence.get(s_idx).copied().unwrap_or(0.0);

        if let Some(h) = node.h_curr {
            if seq[k] != "PRO" {
                shifts.insert((k, "H".to_string()), h);
            }
        }
        shifts.insert((k, "N".to_string()), node.n_curr);
        if let Some(&ca) = node.ca_curr.first() {
            shifts.insert((k, "CA".to_string()), ca);
        }
        if let Some(&cb) = node.cb_curr.first() {
            if seq[k] != "GLY" {
                shifts.insert((k, "CB".to_string()), cb);
            }
        }
        if let Some(&c_cur) = node.c_curr.first() {
            shifts.insert((k, "C".to_string()), c_cur);
        }
        record_conf(k, c_val);

        // Preceding residue `k - 1` shifts observed on this peptide-bond node
        if let Some(c_prev) = node.c_prev {
            shifts.entry((k - 1, "C".to_string())).or_insert(c_prev);
        }
        if let Some(&ca_p) = node.ca_prev.first() {
            shifts.entry((k - 1, "CA".to_string())).or_insert(ca_p);
        }
        if let Some(&cb_p) = node.cb_prev.first() {
            if seq[k - 1] != "GLY" {
                shifts.entry((k - 1, "CB".to_string())).or_insert(cb_p);
            }
        }
        record_conf(k - 1, (c_val * 0.95).clamp(0.0, 1.0));
    }

    (
        Reference {
            sequence: seq,
            seq_ids: ids,
            shifts,
            ambiguity_codes: BTreeMap::new(),
        },
        conf,
    )
}

/// CPU entrypoint alias (`assign_joint_idp_cpu`) matching `assign_idp_13c_cpu` naming convention.
pub fn assign_joint_idp_cpu(
    dataset: &JointIdpDataset,
    sequence: &[String],
    cfg: &JointIdpConfig,
) -> Result<JointIdpAssignResult, String> {
    assign_joint_idp(dataset, sequence, cfg)
}

/// Run the unified joint `HN + 2D CON + Proline-Bridge + N–N + CAS` IDP backbone solver.
///
/// Never panics and never produces `NaN`/`Inf` scores or confidences on empty, degenerate,
/// or hostile inputs; returns `Err` only if `S * L` or `S * S` exceeds `MAX_IDP_MATRIX_CELLS`.
pub fn assign_joint_idp(
    dataset: &JointIdpDataset,
    sequence: &[String],
    cfg: &JointIdpConfig,
) -> Result<JointIdpAssignResult, String> {
    let seq = normalize_seq_3(sequence);
    let k_len = seq.len();

    let build = build_joint_idp_nodes(dataset, &seq, cfg);
    let nodes = build.nodes;
    let s_len = nodes.len();

    check_idp_matrix_bounds(s_len, k_len)?;

    let total_pro_k_ge_1 = if k_len >= 2 {
        seq[1..].iter().filter(|r| r.as_str() == "PRO").count()
    } else {
        0
    };

    if s_len == 0 || k_len < 2 {
        let mut qc_reasons = build.warnings;
        if k_len < 2 {
            qc_reasons.push(
                "Sequence has fewer than 2 residues; no peptide bonds assignable".to_string(),
            );
        }
        if s_len == 0 {
            qc_reasons
                .push("No valid HN or 2D CON peptide-bond roots found in dataset".to_string());
        }
        let residue_assignments = seq
            .iter()
            .enumerate()
            .map(|(idx, res)| JointResidueAssignment {
                seq_idx: idx,
                residue: res.clone(),
                node_idx: None,
                confidence: 0.0,
                raw_posterior: 0.0,
                route_support: None,
                anchor_locked: false,
                is_proline: res == "PRO",
                is_cterm_coo: false,
            })
            .collect();
        return Ok(JointIdpAssignResult {
            nodes,
            position: vec![-1; s_len],
            confidence: vec![0.0; s_len],
            raw_posterior: vec![0.0; s_len],
            route_support: vec![None; s_len],
            residue_assignments,
            score: 0.0,
            pass1_score: 0.0,
            pass2_score_gain: 0.0,
            run_scores: Vec::new(),
            pass1_locked_anchors: Vec::new(),
            fitted_offsets: build.fitted_offsets,
            auto_ref_report: build.auto_ref_report,
            trosy_calibration: build.trosy_calibration,
            polarity_report: build.polarity_report,
            c_term_coo_candidates: build.c_term_coo_candidates,
            n_cross_validated_hn_con: 0,
            n_hn_route_only: 0,
            n_con_pro_bridges: 0,
            n_con_route_only: 0,
            proline_bridge_recovery: (0, total_pro_k_ge_1),
            n_sidelobes_pruned: build.n_sidelobes_pruned,
            n_gly_cb_guarded: build.n_gly_cb_guarded,
            emp_rms_seq_ca: 0.0,
            emp_rms_seq_c: 0.0,
            fusion_rms_ppm: (build.fusion_rms_c_ppm, build.fusion_rms_n_ppm),
            qc_verdict: DatasetQcVerdict::Abstain,
            qc_reasons,
        });
    }

    let cas_tables = build_calibrated_cas_tables(&dataset.cas_spectra, &nodes, &seq, cfg);
    let u_mat = try_joint_idp_unary_matrix(
        &nodes,
        &seq,
        &cas_tables,
        cfg,
        &build.fitted_offsets,
        &dataset.user_locks,
    )?;
    let p_mat = try_joint_idp_pair_matrix(&nodes, cfg)?;

    let valid_user_locks: Vec<(usize, usize)> = dataset
        .user_locks
        .iter()
        .copied()
        .filter(|&(s, k)| s < s_len && k >= 1 && k < k_len && u_mat.get(s, k) > FORBID * 0.5)
        .collect();

    // Pass 1: Multi-seed island SA ensemble
    let (pass1_best, pass1_replicas) =
        run_island_ensemble(&u_mat, &p_mat, &valid_user_locks, cfg, 0);
    let pass1_score = pass1_best.score;

    let p1_diag = compute_posteriors_honest(
        &pass1_best,
        &pass1_replicas,
        &nodes,
        &seq,
        &u_mat,
        &p_mat,
        build.fusion_rms_c_ppm,
        build.fusion_rms_n_ppm,
        build.hn_unmatched_frac,
    );

    // Conservative Pass-1 -> Pass-2 anchor-lock propagation
    let mut pass1_locked_anchors = if cfg.enable_two_pass_anchor_lock {
        select_conservative_pass1_anchors(
            &pass1_best,
            &p1_diag.p_ens,
            &p1_diag.p_local,
            &nodes,
            &seq,
            &u_mat,
            &p_mat,
            &build.c_term_coo_candidates,
            cfg,
        )
    } else {
        Vec::new()
    };

    let mut combined_locks = valid_user_locks.clone();
    for &(s, k) in &pass1_locked_anchors {
        if !combined_locks.iter().any(|&(us, uk)| us == s || uk == k) {
            combined_locks.push((s, k));
        }
    }

    let n_assignable_slots = k_len.saturating_sub(1);
    let run_pass2 = cfg.enable_two_pass_anchor_lock
        && !pass1_locked_anchors.is_empty()
        && combined_locks.len() < s_len.min(n_assignable_slots);

    let (final_best, final_replicas, final_diag, pass2_score_gain) = if run_pass2 {
        let (p2_best, p2_replicas) =
            run_island_ensemble(&u_mat, &p_mat, &combined_locks, cfg, 0xA5A5_5A5A_1234_5678);
        if p2_best.score >= pass1_best.score - 1e-6 {
            // Evaluate honest posteriors WITHOUT any circular Pass-2 lock floor (`180.core:P1-1`)
            let p2_diag = compute_posteriors_honest(
                &p2_best,
                &p2_replicas,
                &nodes,
                &seq,
                &u_mat,
                &p_mat,
                build.fusion_rms_c_ppm,
                build.fusion_rms_n_ppm,
                build.hn_unmatched_frac,
            );
            let gain = (p2_best.score - pass1_score).max(0.0);
            (p2_best, p2_replicas, p2_diag, gain)
        } else {
            pass1_locked_anchors.clear();
            (pass1_best, pass1_replicas, p1_diag, 0.0)
        }
    } else {
        (pass1_best, pass1_replicas, p1_diag, 0.0)
    };

    // Comprehensive unlabelled Dataset QC Verdict (`180.core:P1-2`):
    // Ports Spec 14 §2.4 low-complexity/repeat guards + decoy/node inflation + fusion RMS +
    // empirical sequential matching RMS + ensemble agreement/ambiguity gates!
    let (h_aa_bits, max_aa_frac, max_dipep_frac, top3_dipep_frac) =
        sequence_complexity_metrics(&seq);
    let max_homo_run = max_homopolymer_run(&seq);

    let mut qc_reasons = build.warnings;
    let mut severe_abstain = false;
    let mut severe_degeneracy_warn = false;
    let mut compound_noise_warn = false;
    let mut mild_warn = false;

    // (a) 3D support & sequential connectivity gates
    if final_diag.support_3d_frac < 0.25 {
        qc_reasons.push(format!(
            "Catastrophic 3D strip support loss ({:.1}% < 25.0% of roots supported)",
            final_diag.support_3d_frac * 100.0
        ));
        severe_abstain = true;
    } else if s_len >= 6 && final_diag.connectivity_frac < 0.15 {
        qc_reasons.push(format!(
            "Sequential connectivity collapsed ({:.1}% < 15.0% linked)",
            final_diag.connectivity_frac * 100.0
        ));
        severe_abstain = true;
    } else if final_diag.support_3d_frac < 0.55
        || (s_len >= 6 && final_diag.connectivity_frac < 0.45)
    {
        qc_reasons.push(format!(
            "Partial 3D support ({:.1}%) or sparse sequential connectivity ({:.1}%)",
            final_diag.support_3d_frac * 100.0,
            final_diag.connectivity_frac * 100.0
        ));
        mild_warn = true;
    }

    // (b) Root collapse & decoy/duplicate inflation gates
    let root_ratio = final_diag.node_ratio;
    if k_len >= 20 && root_ratio < 0.35 {
        qc_reasons.push(format!(
            "Severe root collapse ({s_len}/{n_assignable_slots} = {:.1}% < 35.0% expected peptide bonds)",
            root_ratio * 100.0
        ));
        severe_abstain = true;
    } else if k_len >= 20 && root_ratio < 0.70 {
        qc_reasons.push(format!(
            "Root deficit ({s_len}/{n_assignable_slots} = {:.1}% < 70.0% expected peptide bonds)",
            root_ratio * 100.0
        ));
        severe_degeneracy_warn = true;
    } else if k_len >= 20 && root_ratio >= 1.35 {
        qc_reasons.push(format!(
            "Decoy or duplicate root inflation ({s_len}/{n_assignable_slots} = {:.2}x expected peptide bonds >= 1.35x)",
            root_ratio
        ));
        severe_degeneracy_warn = true;
    }

    // (c) Primary-sequence low-complexity & repeat gate (Spec 14 §2.4 via shared `idp_solver`)
    if k_len >= 16
        && (h_aa_bits < 1.65 || max_aa_frac >= 0.55 || max_dipep_frac >= 0.35 || max_homo_run >= 6)
    {
        qc_reasons.push(format!(
            "Extreme low-complexity / homopolymeric repeat sequence (H={h_aa_bits:.2} bits, max_AA={:.1}%, max_dipep={:.1}%, max_run={max_homo_run})",
            max_aa_frac * 100.0,
            max_dipep_frac * 100.0
        ));
        severe_abstain = true;
    } else if k_len >= 16
        && (h_aa_bits < 2.60 || max_aa_frac >= 0.35 || top3_dipep_frac >= 0.35 || max_homo_run >= 4)
    {
        qc_reasons.push(format!(
            "Low-complexity / repetitive IDP sequence (H={h_aa_bits:.2} bits, max_AA={:.1}%, top3_dipep={:.1}%, max_run={max_homo_run})",
            max_aa_frac * 100.0,
            top3_dipep_frac * 100.0
        ));
        severe_degeneracy_warn = true;
    }

    // (d) Matching noise, HN<->CON fusion scatter, ensemble disagreement & placement ambiguity gates
    let noise_high = final_diag.emp_rms_seq_ca >= 0.068
        || final_diag.emp_rms_seq_c >= 0.042
        || build.fusion_rms_n_ppm >= 0.14
        || build.fusion_rms_c_ppm >= 0.045;
    let ambig_high = final_diag.ambig_placement_frac >= 0.22
        || final_diag.mean_ens_agreement < 0.78
        || final_diag.mean_cands_per_ch >= 1.65;

    if s_len >= 8 && (noise_high || ambig_high) {
        qc_reasons.push(format!(
            "Elevated matching noise or assignment ambiguity (seq_CA_RMS={:.3}, seq_C_RMS={:.3}, fus_N_RMS={:.3}, fus_C_RMS={:.3} ppm, ambig={:.1}%, ens_agree={:.1}%)",
            final_diag.emp_rms_seq_ca,
            final_diag.emp_rms_seq_c,
            build.fusion_rms_n_ppm,
            build.fusion_rms_c_ppm,
            final_diag.ambig_placement_frac * 100.0,
            final_diag.mean_ens_agreement * 100.0
        ));
        if noise_high && (ambig_high || root_ratio >= 1.35) {
            compound_noise_warn = true;
        } else {
            mild_warn = true;
        }
    }

    // Severe compound degradation -> Abstain
    if s_len >= 12
        && ((final_diag.ambig_placement_frac >= 0.55 && final_diag.mean_ens_agreement < 0.65)
            || (root_ratio >= 2.00 && (noise_high || final_diag.ambig_placement_frac >= 0.30))
            || (final_diag.emp_rms_seq_ca >= 0.105 && final_diag.ambig_placement_frac >= 0.35))
    {
        qc_reasons.push(
            "Compound degradation (severe noise, decoy inflation, and low ensemble margin); abstaining from high-confidence calls"
                .to_string(),
        );
        severe_abstain = true;
    }

    if !build.auto_ref_report.refused_nuclei.is_empty() {
        mild_warn = true;
    }

    let qc_verdict = if severe_abstain {
        DatasetQcVerdict::Abstain
    } else if severe_degeneracy_warn || compound_noise_warn || mild_warn {
        DatasetQcVerdict::Warn
    } else {
        DatasetQcVerdict::Pass
    };

    // Map honest raw posteriors through dedicated `JOINT_IDP_CALIBRATION_KNOTS` + QC scale/cap
    let confidence: Vec<f64> = final_diag
        .raw_post
        .iter()
        .enumerate()
        .map(|(s, &raw)| {
            if final_best.pos[s] < 0 {
                0.0
            } else {
                let mut cal = calibrate_joint_posterior(raw);
                match qc_verdict {
                    DatasetQcVerdict::Abstain => {
                        cal = (cal * 0.45).min((cfg.hc_threshold - 0.35).clamp(0.10, 0.45));
                    }
                    DatasetQcVerdict::Warn => {
                        if compound_noise_warn {
                            cal = (cal * 0.78).min(0.79);
                        } else if severe_degeneracy_warn {
                            cal = (cal * 0.84).min(0.86);
                        } else {
                            cal = (cal * 0.93).min(0.95);
                        }
                    }
                    DatasetQcVerdict::Pass => {}
                }
                cal.clamp(0.0, 1.0)
            }
        })
        .collect();

    // Explicit `JointRouteSupport` tagging & route concordance tallies
    let mut route_support: Vec<Option<JointRouteSupport>> = vec![None; s_len];
    let mut n_cross_validated_hn_con = 0usize;
    let mut n_hn_route_only = 0usize;
    let mut n_con_pro_bridges = 0usize;
    let mut n_con_route_only = 0usize;

    for s in 0..s_len {
        if final_best.pos[s] >= 0 {
            let tag = nodes[s].route_support();
            route_support[s] = Some(tag);
            match tag {
                JointRouteSupport::CrossValidatedHnAndCon => n_cross_validated_hn_con += 1,
                JointRouteSupport::HnRoute => n_hn_route_only += 1,
                JointRouteSupport::ConProBridge => n_con_pro_bridges += 1,
                JointRouteSupport::ConRouteOnly => n_con_route_only += 1,
            }
        }
    }

    let mut assigned_prolines = 0usize;
    let locked_k_set: BTreeSet<usize> = combined_locks.iter().map(|&(_, k)| k).collect();
    let mut residue_assignments = Vec::with_capacity(k_len);
    for (k, res) in seq.iter().enumerate() {
        let node_opt = (final_best.at[k] >= 0).then_some(final_best.at[k] as usize);
        let is_pro = res == "PRO";
        if k >= 1 && is_pro && node_opt.is_some() {
            assigned_prolines += 1;
        }
        let (conf, raw_p, r_sup, is_coo) = if let Some(s) = node_opt {
            (
                confidence[s],
                final_diag.raw_post[s],
                route_support[s],
                k + 1 == k_len && build.c_term_coo_candidates.contains(&s),
            )
        } else {
            (0.0, 0.0, None, false)
        };
        residue_assignments.push(JointResidueAssignment {
            seq_idx: k,
            residue: res.clone(),
            node_idx: node_opt,
            confidence: conf,
            raw_posterior: raw_p,
            route_support: r_sup,
            anchor_locked: locked_k_set.contains(&k),
            is_proline: is_pro,
            is_cterm_coo: is_coo,
        });
    }

    let run_scores: Vec<f64> = final_replicas
        .iter()
        .map(|r| if r.score.is_finite() { r.score } else { 0.0 })
        .collect();
    let final_score = if final_best.score.is_finite() {
        final_best.score
    } else {
        0.0
    };

    Ok(JointIdpAssignResult {
        nodes,
        position: final_best.pos,
        confidence,
        raw_posterior: final_diag.raw_post,
        route_support,
        residue_assignments,
        score: final_score,
        pass1_score: if pass1_score.is_finite() {
            pass1_score
        } else {
            0.0
        },
        pass2_score_gain: if pass2_score_gain.is_finite() {
            pass2_score_gain
        } else {
            0.0
        },
        run_scores,
        pass1_locked_anchors,
        fitted_offsets: build.fitted_offsets,
        auto_ref_report: build.auto_ref_report,
        trosy_calibration: build.trosy_calibration,
        polarity_report: build.polarity_report,
        c_term_coo_candidates: build.c_term_coo_candidates,
        n_cross_validated_hn_con,
        n_hn_route_only,
        n_con_pro_bridges,
        n_con_route_only,
        proline_bridge_recovery: (assigned_prolines, total_pro_k_ge_1),
        n_sidelobes_pruned: build.n_sidelobes_pruned,
        n_gly_cb_guarded: build.n_gly_cb_guarded,
        emp_rms_seq_ca: final_diag.emp_rms_seq_ca,
        emp_rms_seq_c: final_diag.emp_rms_seq_c,
        fusion_rms_ppm: (build.fusion_rms_c_ppm, build.fusion_rms_n_ppm),
        qc_verdict,
        qc_reasons,
    })
}
